# -*- coding: utf-8 -*-
"""抽帧基准必须精确；`-ss` seek 抽帧**不进判据链路**。

## 背景

`verify_frame_alignment` 判的是"**是不是整体错位了 1 帧**"。
抽帧工具本身若有 ±1 帧误差，等于把要检测的误差引入测量。

## 三种取帧方式的实测矩阵（240 帧素材，与"顺序解码"逐像素对拍）

| 方式 | ffv1 | h264 | 1fps 素材 |
|---|---|---|---|
| 单段 `-ss t -i f` | 差 32764 ✗ | 差 201 ✗ | — |
| 两段 `-ss t-2 -i f -ss 2` | 差 0 ✓ | 差 0 ✓ | **抽到 k+1 ✗** |
| `-ss` 放在 `-i` 之后（精确）| 对，但等于全片解码 | 同左 | 同左 |
| **一次 select 全片解码（批量）** | **差 0 ✓** | **差 0 ✓** | **差 0 ✓** |

两段式看似能用，但第二段是**相对位移**，落点取决于第一段落在哪个关键帧
—— 结果随帧率/GOP 结构漂移。`test_r7_artifact_verify.py::test_alignment_passes_on_reencode`
就抓到过它在 1fps 素材上抽到 k+1（对应=0.9972 而邻居 k+1=1.0，正好反了）。

所以 `_SEEK_EXTRACT_ENABLED = False`：宁可慢，不用不可靠的快。
`_extract_frames_seek` 保留供人工诊断（长片上游件 4 帧 6.5 秒，确实快）。
"""
import os
import subprocess
import tempfile

import numpy as np
import pytest

import conftest  # noqa: F401

import artifact_verify as AV


def _make_video(tmp_path, codec="h264", frames=48, rate=10):
    out = tmp_path / f"v_{codec}_{rate}.mkv"
    venc = (["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"]
            if codec == "h264"
            else ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p"])
    cmd = (["ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-f", "lavfi", "-i",
            f"testsrc2=size=160x120:rate={rate}:duration={frames/rate}"]
           + venc + [str(out)])
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except Exception:
        pytest.skip("cannot synthesize test video")
    return str(out)


def _maxdiff(a, b):
    import cv2
    x = cv2.imread(a, cv2.IMREAD_UNCHANGED)
    y = cv2.imread(b, cv2.IMREAD_UNCHANGED)
    if x is None or y is None or x.shape != y.shape:
        return None
    return int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max())


def test_seek_is_off_by_default():
    """守卫：seek 抽帧不许悄悄回到判据链路里。"""
    assert AV._SEEK_EXTRACT_ENABLED is False, (
        "seek 被打开了 —— 它在 1fps 素材上会抽到 k+1，"
        "而这条判据判的就是'是否错位 1 帧'")


@pytest.mark.parametrize("codec", ["h264", "ffv1"])
@pytest.mark.parametrize("rate", [1, 10, 25])
def test_batch_extraction_matches_single_frame_decode(tmp_path, codec, rate):
    """批量抽帧必须与"单独解到第 k 帧"逐像素一致 —— 含首末帧。"""
    p = _make_video(tmp_path, codec, frames=max(12, rate * 4), rate=rate)
    n = AV._frames_or_count(p)
    if not n or n < 4:
        pytest.skip("cannot count frames")
    idxs = sorted({0, n // 3, n // 2, n - 1})
    with tempfile.TemporaryDirectory() as td:
        batch = AV._extract_frames_batch(p, idxs, td, "b")
        assert batch, "批量抽帧返回空"
        for k in idxs:
            assert k in batch, f"帧 {k} 缺失"
            single = os.path.join(td, f"one_{k}.png")
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", p,
                 "-vf", f"select=eq(n\\,{k}),scale=320:-2",
                 "-frames:v", "1", "-y", single],
                capture_output=True, timeout=180)
            d = _maxdiff(batch[k], single)
            assert d == 0, (
                f"{codec}@{rate}fps 第 {k} 帧：批量抽帧与单独解码差 {d}（应为 0）")


def test_seek_returns_empty_when_it_cannot_be_trusted(tmp_path):
    """seek 拿不到时间基准时必须返回空，而不是瞎抽。"""
    p = tmp_path / "not_a_video.txt"
    p.write_text("hello", encoding="utf-8")
    with tempfile.TemporaryDirectory() as td:
        assert AV._extract_frames_seek(str(p), [0, 1], td, "x") == {}
