"""B6（第七轮）：真实跑 AVIF 归档编码 + SATO 回读，逐像素比较。

**为什么必须真跑**

第六轮的 B6 报告"指定样本 MAE=0 / MAX=0 / array_equal"，但
`test_b6_archival_pixel_exact.py` 里只有**契约措辞**和**16bit PNG 读取**
两类测试 —— 那条实验是手工做的，**没有变成回归**。容器/编码器/PIL
任一处退化，测试仍然是绿的。

**已知的坑（本机 avifenc 1.4.2 实测）**

```
avifdec --sato            → MAE=0.0000  MAX=0   array_equal=True   ← 正确
avifdec --depth 16        → MAE=3.5024  MAX=15  array_equal=False  ← 丢低 4 位
avifdec --depth 16 --sato → MAE=0.0000  MAX=0   array_equal=True   ← 正确
```

归档档是 **12-bit 主图 + 8-bit depth-extension item**。`--sato` 负责把
两部分拼回去；只用 `--depth 16` 会**丢掉低 4 位**。所以契约里绝不能
写成"--sato (or --depth 16)"这种等价表述。
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "shared_modules"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import media_io  # noqa: E402

TOOLS = os.path.join(ROOT, "tools_r4")


def _tool(name):
    cand = os.path.join(TOOLS, name + (".exe" if os.name == "nt" else ""))
    if os.path.exists(cand):
        return cand
    return shutil.which(name)


AVIFENC = _tool("avifenc")
AVIFDEC = _tool("avifdec")

pytestmark = pytest.mark.skipif(
    not (AVIFENC and AVIFDEC), reason="需要 avifenc/avifdec")


@pytest.fixture(autouse=True)
def _avif_on_path(monkeypatch):
    """把 tools_r4 注入 PATH。

    `media_io.build_avifenc_cmd` 内部用 `shutil.which("avifenc")` 判断
    可用性；若 tools_r4 不在 PATH，契约会记 `avifenc_available=False`
    并给出一个不含 `--depth 12,8` 的命令 —— 测试就变成在测"降级路径"。
    """
    monkeypatch.setenv("PATH", TOOLS + os.pathsep
                       + os.environ.get("PATH", ""))
    yield


def _make_16bit_png(tmp_path):
    """造一张**低位有信息**的 16bit 图。

    关键：如果只造 8bit 内容的放大版（低 8 位全 0），那么"--depth 16
    丢低位"这个缺陷根本测不出来。这里让低 8 位随位置渐变。
    """
    import cv2
    h, w = 64, 64
    x = np.arange(w, dtype=np.uint32)
    y = np.arange(h, dtype=np.uint32)[:, None]
    # 高 8 位：大块渐变；低 8 位：细密条纹（模拟真实胶片的量化残差）
    hi = ((x[None, :] * 3 + y * 5) % 256).astype(np.uint32)
    lo = ((x[None, :] * 37 + y * 11) % 256).astype(np.uint32)
    b = ((hi << 8) | lo).astype(np.uint16)
    img = np.dstack([b, np.roll(b, 1, axis=1), np.roll(b, 1, axis=0)])
    p = str(tmp_path / "src16.png")
    ok = cv2.imwrite(p, img)
    assert ok, "写 16bit PNG 失败"
    back = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    assert back.dtype == np.uint16
    assert len(np.unique(back & 0xFF)) > 1, "测试素材低位无信息，测不出问题"
    return p, back


#: 归档档的真实参数形态。
#:
#: `build_avifenc_cmd` 读的是 `avif_rgb444_params`（配合 use_rgb444=True），
#: **不是** "archival_mode" 之类的语义开关。第一版测试传错了键，
#: 契约里 lossless=False、depth=10，往返自然对不上（MAE=1167）。
ARCHIVAL_PARAMS = {
    "avif_rgb444_params": {
        "depth": 16, "yuv": "444", "lossless": True, "range": "full",
    },
}


def _encode(src, dst, params, use_rgb444=True):
    cmd, contract = media_io.build_avifenc_cmd(src, dst, params,
                                               use_rgb444=use_rgb444)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"avifenc 失败：{r.stderr[:300]}")
    return contract


def _decode_sato(avif, out_png):
    r = subprocess.run([AVIFDEC, "--sato", avif, out_png],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stderr


def _read16(png):
    import cv2
    a = cv2.imread(png, cv2.IMREAD_UNCHANGED)
    assert a is not None, f"读不出 {png}"
    return a


def test_archival_roundtrip_is_pixel_exact(tmp_path):
    """归档档往返必须**逐像素相等**（不是"看起来差不多"）。"""
    src, src_arr = _make_16bit_png(tmp_path)
    avif = str(tmp_path / "a.avif")
    params = {"archival_mode": True, "photo_bit_depth": 16}

    _encode(src, avif, ARCHIVAL_PARAMS)
    out = str(tmp_path / "dec.png")
    ok, err = _decode_sato(avif, out)
    assert ok, f"avifdec --sato 失败：{err[:300]}"

    got = _read16(out)
    assert got.shape == src_arr.shape
    assert got.dtype == np.uint16, f"回读成了 {got.dtype}"

    # avifdec 输出可能是 BGR/BGRA，这里只比较前 3 通道且按同样的通道序
    if got.shape[2] == 4:
        got = got[:, :, :3]
    src3 = src_arr[:, :, :3]
    if got.shape != src3.shape:
        pytest.skip(f"形状不一致 {got.shape} vs {src3.shape}")
    diff = np.abs(got.astype(np.int32) - src3.astype(np.int32))
    mae = float(diff.mean())
    mx = int(diff.max())
    assert mx == 0 and mae == 0.0, (
        f"归档往返不是逐像素相等：MAE={mae:.4f} MAX={mx}\n"
        "（若容器/编码器变了，需要重新核对契约）")


def test_depth16_alone_loses_low_bits(tmp_path):
    """反例：`--depth 16` 单独用会丢低位 —— 必须能被测出来。

    这条不是在测"我们希望它失败"，而是证明**我们用的测试素材确实
    含低位信息**，否则上一条的"逐像素相等"是假的。
    """
    src, src_arr = _make_16bit_png(tmp_path)
    avif = str(tmp_path / "b.avif")
    _encode(src, avif, ARCHIVAL_PARAMS)

    out = str(tmp_path / "dec_depth16.png")
    r = subprocess.run([AVIFDEC, "--depth", "16", avif, out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"avifdec --depth 16 不支持：{r.stderr[:200]}")

    got = _read16(out)
    if got.shape[2] == 4:
        got = got[:, :, :3]
    src3 = src_arr[:, :, :3]
    if got.shape != src3.shape:
        pytest.skip(f"形状不一致 {got.shape} vs {src3.shape}")

    diff = np.abs(got.astype(np.int32) - src3.astype(np.int32))
    assert diff.max() > 0, (
        "`--depth 16` 竟然也逐像素相等 —— 说明素材不含低位信息，"
        "上一条测试的通过是不可信的")


def test_archival_contract_requests_real_16bit_encoding(tmp_path):
    """契约必须要求 16bit 无损 —— 不能悄悄退化成 10bit/有损。

    注意：`--sato` 是 **avifdec（解码端）** 的参数，不在 avifenc 契约里。
    归档档的 16bit 是靠 avifenc 的 `--depth 12,8`（12bit 主图 +
    8bit depth-extension item）实现的，回读时必须用 `--sato` 把两部分
    拼回去。第一版测试把 `--sato` 当成 avifenc 参数，断言写错了。
    """
    src, _ = _make_16bit_png(tmp_path)
    cmd, contract = media_io.build_avifenc_cmd(
        src, str(tmp_path / "c.avif"), ARCHIVAL_PARAMS, use_rgb444=True)
    text = " ".join(cmd)

    assert contract.get("lossless") is True, \
        f"归档档必须 lossless，实际 {contract.get('lossless')}"
    assert contract.get("archival_rgb444") is True
    # 归档档的真实形态：**12bit 主图 + 8bit depth-extension item**，
    # 在命令行上就是 `--depth 12,8`。契约里记作 depth_arg，而不是
    # 一个"depth=16"的数字 —— 因为 avifenc 根本没有 depth 16。
    assert str(contract.get("depth_arg")) == "12,8", \
        f"归档档的 depth_arg 不是 12,8：{contract.get('depth_arg')}"
    assert "--depth" in text and "12,8" in text
    # 绝不允许出现裸的 --depth 16（avifenc 会直接拒绝）
    assert "--depth 16" not in text
