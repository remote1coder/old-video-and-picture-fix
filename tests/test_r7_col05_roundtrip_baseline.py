"""COL-05（第七轮）：先建立"无效果往返"基准，再谈模型/滤镜的影响。

任务书 §4.3 的顺序要求：

> **先跑无效果往返。** 源解码→当前中间格式→最终解码，关闭AI、锐化、
> eq、人脸、颗粒，显式约定所有转换。若这条路径已偏，先修转换；
> 不要用模型调参抵消。

这条测试只做**编解码往返**，不涉及任何 stage 逻辑：把已知色块喂给
ffmpeg，用 stage7 实际使用的编码参数编一遍，再解回来比较。

任何超出实测噪声的偏差，都说明"编码链路本身"有问题；此时再去讨论
AI 模型或滤镜的色偏是**没有意义的**。
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "shared_modules") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "shared_modules"))

import media_io as M  # noqa: E402


#: 已知色块：纯红/纯绿/纯蓝/中灰/近黑/近白，用来同时发现通道颠倒、
#: 范围错配（tv vs pc）和截断。
_PATCHES = [
    ("red",   (220, 20, 20)),
    ("green", (20, 200, 20)),
    ("blue",  (20, 20, 220)),
    ("gray",  (128, 128, 128)),
    ("near_black", (16, 16, 16)),
    ("near_white", (235, 235, 235)),
]


def _make_patch_video(tmp_path, colorspace="bt709", pix_fmt="yuv420p10le"):
    """用 ffmpeg 生成 6 段色块拼接的短片（每段 8 帧）。"""
    inputs = []
    for name, rgb in _PATCHES:
        inputs += ["-f", "lavfi", "-i",
                   f"color=c=0x{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                   f":s=64x64:d=0.32:r=25"]
    out = str(tmp_path / "patches.mkv")
    cmd = ["ffmpeg", "-v", "error"] + inputs + [
        "-filter_complex",
        "".join(f"[{i}:v]" for i in range(len(_PATCHES)))
        + f"concat=n={len(_PATCHES)}:v=1:a=0[v]",
        "-map", "[v]",
        "-c:v", "ffv1", "-pix_fmt", pix_fmt,
        "-colorspace", colorspace,
        "-color_primaries", colorspace,
        "-color_trc", "bt709",
        "-color_range", "tv",
        "-y", out]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"ffmpeg 合成色块失败：{r.stderr[:300]}")
    return out


def _extract_frame(path, idx, tmp_path, tag):
    p = str(tmp_path / f"{tag}_{idx}.png")
    subprocess.run(["ffmpeg", "-v", "error", "-i", path,
                    "-vf", f"select=eq(n\\,{idx})", "-frames:v", "1",
                    "-pix_fmt", "rgb48", "-y", p], capture_output=True)
    return p


def _mean_rgb(png_path):
    """读取 RGB48 PNG 并返回平均 RGB（0-255 标度）。

    用 cv2.IMREAD_UNCHANGED：它保留 uint16，且给出的是 BGR —— 这里
    显式反转成 RGB，避免"测出来的通道顺序"和"实际的通道顺序"混为一谈。
    """
    import cv2
    a = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    assert a is not None, f"读不出 {png_path}"
    if a.ndim == 2:                       # 灰度
        a = np.stack([a] * 3, axis=-1)
    if a.shape[2] == 4:
        a = a[:, :, :3]
    a = a[:, :, ::-1].astype(np.float64)   # BGR -> RGB
    if a.dtype != np.uint8:
        a = a / 257.0                      # 16bit -> 8bit 标度
    return a.reshape(-1, 3).mean(axis=0)


def test_codec_roundtrip_preserves_colour(tmp_path):
    """stage7 用的编码参数往返后，色块应保持原来的颜色。"""
    src = _make_patch_video(tmp_path)
    # 走 stage7 同款编码：ffv1 + yuv420p10le（不转颜色）
    out = str(tmp_path / "roundtrip.mkv")
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", src, "-c:v", "ffv1",
         "-pix_fmt", "yuv420p10le", "-an", "-y", out],
        capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"往返编码失败：{r.stderr[:200]}")

    frames_per = 8
    worst = 0.0
    details = []
    for i, (name, rgb) in enumerate(_PATCHES):
        idx = i * frames_per + 2
        a = _mean_rgb(_extract_frame(src, idx, tmp_path, f"src{i}"))
        b = _mean_rgb(_extract_frame(out, idx, tmp_path, f"out{i}"))
        delta = float(np.max(np.abs(a - b)))
        details.append(f"{name}: src={a.round(1).tolist()} out={b.round(1).tolist()} d={delta:.1f}")
        worst = max(worst, delta)

    assert worst <= 6.0, (
        "无效果往返就已经偏移超过 6/255 —— 先修编码/转换，"
        "不要用模型或滤镜去抵消。\n" + "\n".join(details))


def test_roundtrip_does_not_swap_channels(tmp_path):
    """往返后 R 通道仍然最亮在红色块上（防通道颠倒）。"""
    src = _make_patch_video(tmp_path)
    out = str(tmp_path / "rt2.mkv")
    subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-c:v", "ffv1",
                    "-pix_fmt", "yuv420p10le", "-an", "-y", out],
                   capture_output=True)
    f = _mean_rgb(_extract_frame(out, 2, tmp_path, "red"))
    assert f[0] > f[1] + 40 and f[0] > f[2] + 40, \
        f"红色块的 R 通道不再主导（{f.round(1).tolist()}）—— 通道可能被换过"


def test_roundtrip_keeps_near_black_and_white(tmp_path):
    """近黑/近白不能被范围错配压掉（tv/pc 混用的典型症状）。"""
    src = _make_patch_video(tmp_path)
    out = str(tmp_path / "rt3.mkv")
    subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-c:v", "ffv1",
                    "-pix_fmt", "yuv420p10le", "-an", "-y", out],
                   capture_output=True)
    nb = _mean_rgb(_extract_frame(out, 4 * 8 + 2, tmp_path, "nb"))
    nw = _mean_rgb(_extract_frame(out, 5 * 8 + 2, tmp_path, "nw"))
    assert nb.max() < 40, f"近黑被抬亮到 {nb.round(1).tolist()}（范围错配）"
    assert nw.min() > 200, f"近白被压暗到 {nw.round(1).tolist()}（范围错配）"
