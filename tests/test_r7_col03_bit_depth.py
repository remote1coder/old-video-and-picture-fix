"""COL-03（第七轮）：16bit 帧不得在 AI 路径被静默降成 8bit。

**旧实现**

```python
img = Image.open(fp).convert('RGB')      # rgb48 PNG -> uint8
output, _ = upsampler.enhance(np.array(img))
Image.fromarray(output).save(...)
```

stage3 抽帧用 `-pix_fmt rgb48`、回编码用 `-yuv420p16le` —— 整条链本来
是为 16bit 设计的，但 PIL 这一步把输入**静默降成 8bit**（实测
`PIL mode: RGB`、`dtype: uint8`），低位信息直接丢掉。

**修法**：`cv2.imread(fp, cv2.IMREAD_UNCHANGED)` 保留 uint16，且 cv2
本身就是 BGR（与 RealESRGANer 的官方输入约定一致，不需要换通道）。

**踩到的坑（很关键）**：`RealESRGANer.enhance()` **已经**在内部把输出
还原到输入量级并转好类型：

```python
if max_range == 65535:                      # 16bit 输入
    output = (output_img * 65535.0).round().astype(np.uint16)
else:                                       # 8bit 输入
    output = (output_img * 255.0).round().astype(np.uint8)
```

第一版修复里我又乘了一次 65535 → 数值爆掉 → clip 后整帧只剩黑白两值
（实测"唯一值 1 个、全 65535"）。现在直接写回返回数组。
"""

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VSTAGES = os.path.join(ROOT, "video_restoration_pipeline", "stages")
if VSTAGES not in sys.path:
    sys.path.insert(0, VSTAGES)

import stage3_ai_upscale as s3  # noqa: E402


def _rgb48_frame(tmp_path, size="64x64"):
    out = str(tmp_path / "frame_00000000.png")
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc2=s={size}:d=1:r=1", "-frames:v", "1",
         "-pix_fmt", "rgb48", "-y", out],
        capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"ffmpeg 合成失败：{r.stderr[:200]}")
    return out


def test_cv2_read_preserves_16bit(tmp_path):
    """cv2.IMREAD_UNCHANGED 必须给出 uint16，且低位确有信息。"""
    p = _rgb48_frame(tmp_path)
    a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    assert a.dtype == np.uint16, f"cv2 读成了 {a.dtype}"
    assert len(np.unique(a & 0xFF)) > 1, "低位全 0 —— 这不是真的 16bit 源"


def test_pil_would_lose_precision(tmp_path):
    """反证：PIL 的同一条路确实会降成 uint8（说明为什么必须换 cv2）。"""
    from PIL import Image
    p = _rgb48_frame(tmp_path)
    arr = np.array(Image.open(p).convert("RGB"))
    assert arr.dtype == np.uint8
    assert arr.max() <= 255


def test_uint16_roundtrip_is_lossless(tmp_path):
    """纯读写往返必须逐像素一致（证明写回管道本身不丢精度）。"""
    p = _rgb48_frame(tmp_path)
    a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    dst = str(tmp_path / "rt.png")
    cv2.imwrite(dst, a)
    b = cv2.imread(dst, cv2.IMREAD_UNCHANGED)
    assert np.array_equal(a, b)


def test_ai_path_keeps_uint16_and_real_precision(tmp_path):
    """真实跑一次超分：输出必须是 uint16 且精度远高于 8bit。"""
    src_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    src_dir.mkdir()
    _rgb48_frame(src_dir)

    r = s3.run_realesrgan_batch([(str(src_dir), str(out_dir))],
                                "realesr-general-x4v3", 2, gpu_id=0,
                                logger=None)
    if r is not True:
        pytest.skip(f"AI 子进程未成功（{r}）—— 记为环境未覆盖，不算通过")

    produced = list(out_dir.glob("*.png"))
    assert produced, "没有产出"
    b = cv2.imread(str(produced[0]), cv2.IMREAD_UNCHANGED)
    assert b.dtype == np.uint16, f"输出被降成 {b.dtype}"
    n_unique = len(np.unique(b[..., 1]))
    assert n_unique > 256, (
        f"输出只有 {n_unique} 个唯一值 —— 实际仍是 8bit 精度，"
        "16bit 容器只是摆设")


def test_no_pil_read_in_ai_loop():
    """静态守卫：AI 循环里不许再用 PIL 读帧（会静默降位深）。"""
    src = open(s3.__file__, encoding="utf-8").read()
    assert "cv2.imread(fp, cv2.IMREAD_UNCHANGED)" in src
    # PIL 只允许出现在 cv2 读失败的回退分支里
    assert "Image.open(fp).convert('RGB'))[:, :, ::-1]" in src, \
        "PIL 回退分支应显式转 BGR（否则通道错）"
    assert "np.array(Image.open(fp).convert('RGB'))\\n" not in src, \
        "仍有裸 PIL 读取（未转 BGR）"


def test_enhance_output_is_not_rescaled_again():
    """静态守卫：不能再对 enhance 的返回值乘 65535/255。

    `enhance()` 内部已经 `(output_img * max_range).round().astype(...)`。
    再乘一次会让数值爆掉（实测整帧只剩两个值）。
    """
    src = open(s3.__file__, encoding="utf-8").read()
    assert "* (65535.0 if _is16 else 255.0)" not in src, \
        "又对 enhance 输出做了二次量级换算"
    assert "cv2.imwrite(os.path.join(output_dir, os.path.basename(fp)), output)" in src
