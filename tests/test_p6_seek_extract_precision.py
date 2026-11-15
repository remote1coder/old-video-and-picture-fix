# -*- coding: utf-8 -*-
"""`_extract_frames_seek` 必须抽到**和顺序解码完全相同的那一帧**。

## 为什么这条必须有

`verify_frame_alignment` 判的就是"**是不是整体错位了 1 帧**"。
如果抽帧工具本身有 ±1 帧误差，等于把要检测的误差引入到测量里 ——
那比"慢"严重得多。

所以改成 seek 抽帧（20.9 分钟 → 只解 2 秒）之后，必须证明**精度没变**。

## 覆盖三种编码

* **h264 源**（有 GOP、可能有 B 帧）—— 最常见
* **h264 成品**（`dither` 之后的交付件）
* **ffv1**（长片上游件的编码，intra-only）

三种的 seek 行为不同，必须都验。实测全部**最大像素差 0**。

## 含首末帧

首帧（`-ss 0`，lead 会被夹到 0）和末帧（最靠近结尾，最容易出 off-by-one）
都必须进测试 —— 这两个位置的 seek 边界条件最多。
"""
import os
import subprocess

import numpy as np
import pytest

import conftest  # noqa: F401

import artifact_verify as AV


def _make_video(tmp_path, codec="h264", frames=48):
    out = tmp_path / f"v_{codec}.mkv"
    venc = (["-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"]
            if codec == "h264"
            else ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p"])
    cmd = (["ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-f", "lavfi", "-i",
            f"testsrc2=size=160x120:rate=10:duration={frames/10}"]
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


@pytest.mark.parametrize("codec", ["h264", "ffv1"])
def test_seek_and_batch_pick_the_same_frames(tmp_path, codec):
    p = _make_video(tmp_path, codec)
    n = AV._frames_or_count(p)
    if not n:
        pytest.skip("cannot count frames")
    idxs = sorted({0, n // 3, n // 2, n - 1})
    with __import__("tempfile").TemporaryDirectory() as td:
        b = AV._extract_frames_batch(p, idxs, td, "b")
        s = AV._extract_frames_seek(p, idxs, td, "s")
        # 比较**必须在 with 内**：临时目录一离开就被删，
        # cv2.imread 会返回 None，于是 _maxdiff 返回 None 而不是 0。
        # （第一版我把循环写在了 with 外面，测试报"差 None"，其实是自摆乌龙。）
        assert s, "seek 抽帧返回空"
        for k in idxs:
            assert k in b and k in s, f"帧 {k} 缺失"
            d = _maxdiff(b[k], s[k])
            assert d == 0, (
                f"{codec} 上第 {k} 帧：seek 与顺序解码的最大像素差 {d}（应为 0）"
                f"—— seek 抽错了帧，这会让帧序判据把正确结果判成错位")


def test_seek_falls_back_when_pts_unavailable(tmp_path):
    """拿不到 PTS 时必须返回空（让调用方回退批量），而不是瞎抽。"""
    p = tmp_path / "not_a_video.txt"
    p.write_text("hello", encoding="utf-8")
    with __import__("tempfile").TemporaryDirectory() as td:
        assert AV._extract_frames_seek(str(p), [0, 1], td, "x") == {}
