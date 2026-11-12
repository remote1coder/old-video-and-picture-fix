# -*- coding: utf-8 -*-
"""R23-B09 的最后一块：`444 工作 + 420 交付` 必须**真的**降了色度。

plan 层（`resolve()` 给出 `working=444 / delivery=yuv420p10le / conversion`）
在 `test_r23_b05_b08_contracts.py` 里验过了。这里补真实成品：

1. 计划说交付是 420 → ffprobe 读成品也必须报 420（头不能骗人）；
2. 色度平面尺寸必须真的减半（420 的"真身"）；
3. **高频色度必须被抹平** —— 只改头不改数据这种情况，前两条都能通过。

第 3 条是核心。做法：造 1 像素宽的色度条纹（Cb 沿 X 在 64/192 间交替），
444 交付后 std≈4096（16-bit 码值），420 交付后 std≈3.5 —— 条纹被平均掉了。
这个数量级差（1000 倍）不可能靠改头伪造。
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_PROJ, "shared_modules"),
           os.path.join(_PROJ, "video_restoration_pipeline"),
           os.path.join(_PROJ, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from work_format import resolve as wf_resolve   # noqa: E402

W, H, FRAMES = 64, 48, 5

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="需要 ffmpeg/ffprobe")


def _mid_raw():
    """Y 平场 + Cb/Cr 沿 X 在 64/192（8-bit 刻度）交替 —— 1 像素宽高频色度。"""
    y = np.full((H, W), 128, dtype=np.uint16) << 6
    x = np.arange(W)
    cb = np.where(x % 2 == 0, 64, 192).astype(np.uint16)[None, :].repeat(H, 0) << 6
    cr = np.where(x % 2 == 0, 192, 64).astype(np.uint16)[None, :].repeat(H, 0) << 6
    return np.concatenate([y.ravel(), cb.ravel(), cr.ravel()]).astype(
        "<u2").tobytes() * FRAMES


@pytest.fixture(scope="module")
def mid(tmp_path_factory):
    """444p16le 无损中间件（ffv1），等价于 stage1..6 的产物。"""
    d = tmp_path_factory.mktemp("dlfmt")
    raw = d / "mid.raw"
    raw.write_bytes(_mid_raw())
    path = str(d / "mid.mkv")
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt",
         "yuv444p16le", "-s", "%dx%d" % (W, H), "-framerate", "5",
         "-i", str(raw), "-c:v", "ffv1", path],
        capture_output=True, timeout=180)
    assert r.returncode == 0, (r.stderr or b"")[-400:]
    return path


def _encode(mid, out, pix_fmt):
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", mid, "-c:v", "ffv1",
         "-pix_fmt", pix_fmt, out], capture_output=True, timeout=180)
    assert r.returncode == 0, (r.stderr or b"")[-400:]
    return out


def _probe_pix_fmt(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", path],
        capture_output=True, text=True, errors="replace", timeout=60)
    assert r.returncode == 0, (r.stderr or "")[-300:]
    return r.stdout.strip()


def _planes(path, pix_fmt):
    """解第一帧，按平面切开。只接受本项目会产出的 420/444 两种。"""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
        capture_output=True, timeout=180)
    assert r.returncode == 0, (r.stderr or b"")[-400:]
    a = np.frombuffer(r.stdout, dtype="<u2").astype(np.int32)
    y = a[:W * H]
    if len(a) == W * H * 3:
        cw, ch = W, H
    else:
        cw, ch = W // 2, H // 2
    cb = a[W * H:W * H + cw * ch]
    cr = a[W * H + cw * ch:W * H + 2 * cw * ch]
    assert cb.size == cw * ch and cr.size == cw * ch, (
        "解码长度与声明的色度布局不符：%d 个样本 / 期望 chroma %dx%d"
        % (len(a), cw, ch))
    return y, cb.reshape(ch, cw), cr.reshape(ch, cw)


def test_444_work_420_delivery_is_an_allowed_plan():
    """工作 444 + 交付 420 是**合法**组合，但它必须显式写出转换。"""
    plan = wf_resolve({"chroma_mode": "444", "pix_fmt": "yuv420p10le",
                       "encoder": "libsvtav1"})
    assert plan["ok"] is True, plan.get("blockers")
    assert plan["working"]["chroma"] == "444"
    assert plan["delivery"]["pix_fmt"] == "yuv420p10le"
    assert plan["delivery"]["chroma"] == "420"
    conv = plan["conversion"]
    assert conv and conv["from"] == "444" and conv["to"] == "420"
    assert conv["stage"] == "stage7_final"
    # 编码器边界拿到的就是交付格式（stage7 用的就是这个键）
    assert plan["boundaries"]["stage7_final"] == "yuv420p10le"


def test_delivered_file_reports_the_planned_pix_fmt(mid, tmp_path):
    """计划说 420，成品 ffprobe 就必须报 420。"""
    out = _encode(mid, str(tmp_path / "deliver420.mkv"), "yuv420p10le")
    assert _probe_pix_fmt(out) == "yuv420p10le"
    # 对照：444 交付
    out4 = _encode(mid, str(tmp_path / "deliver444.mkv"), "yuv444p16le")
    assert _probe_pix_fmt(out4) == "yuv444p16le"


def test_420_delivery_really_halves_the_chroma_planes(mid, tmp_path):
    """色度平面的**样本数**必须减半（420 的真身）。"""
    _, cb420, _ = _planes(_encode(mid, str(tmp_path / "a.mkv"),
                                  "yuv420p10le"), "yuv420p10le")
    _, cb444, _ = _planes(_encode(mid, str(tmp_path / "b.mkv"),
                                  "yuv444p16le"), "yuv444p16le")
    assert cb444.shape == (H, W)
    assert cb420.shape == (H // 2, W // 2)
    assert cb420.size == cb444.size // 4


def test_420_delivery_flattens_high_frequency_chroma(mid, tmp_path):
    """核心判据：只改头不改数据过不了这一关。

    1 像素宽的色度条纹在 444 里完整保留（std 4096 个 16-bit 码值），
    在 420 里被相邻平均掉（std 只剩个位数）。
    """
    _, cb444, _ = _planes(_encode(mid, str(tmp_path / "c.mkv"),
                                  "yuv444p16le"), "yuv444p16le")
    _, cb420, _ = _planes(_encode(mid, str(tmp_path / "d.mkv"),
                                  "yuv420p10le"), "yuv420p10le")

    # 444：条纹原样在（64<<6=4096 / 192<<6=12288 交替）
    assert cb444.std() > 4000, cb444[0, :8].tolist()
    assert cb444.max() - cb444.min() > 8000
    # 420：条纹被抹平，只剩 ±3 个码值的舍入残留
    assert cb420.std() < 20, (
        "420 交付后色度仍保留高频（std=%.1f）—— 说明只是改了 pix_fmt 标记，"
        "数据没真的降采样" % cb420.std())
    assert cb420.max() - cb420.min() < 64

    # 1000 倍的量级差：不可能是"改头"造成的
    assert cb444.std() / max(cb420.std(), 1e-9) > 100


def test_444_delivery_keeps_the_chroma_it_was_given(mid, tmp_path):
    """正例：选 444 交付时，色度不能被顺手降掉（防止修成"一律 420"）。"""
    plan = wf_resolve({"chroma_mode": "444", "pix_fmt": "yuv444p16le",
                       "encoder": "ffv1"})
    assert plan["ok"] is True, plan.get("blockers")
    assert plan["conversion"] is None, "444 工作 + 444 交付不该有转换"
    out = _encode(mid, str(tmp_path / "keep444.mkv"), "yuv444p16le")
    _, cb, cr = _planes(out, "yuv444p16le")
    assert cb.std() > 4000 and cr.std() > 4000
    # Y 本来就平，全程不该被色度影响
    y, _, _ = _planes(out, "yuv444p16le")
    assert y.std() == 0
