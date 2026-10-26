# -*- coding: utf-8 -*-
"""STR-04：VFR 素材必须回退 PNG 路径，不能走流式压成 CFR。

流式路径用 rawvideo 传帧，不携带逐帧 PTS，编码端只能固定
`-framerate` + `-r`，VFR 素材的间隔会被压成 CFR —— 帧数不变但时序已变。
在支持逐帧 PTS 之前，`_is_vfr()` 判定为真时 stage3 应显式回退。

本测试同时锁住**不误伤**：数据不足/无法判定时返回 False（按"未知"处理，
交给调用方决定），真正的 CFR 素材不能被判成 VFR。
"""
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage3_ai_upscale as s3  # noqa: E402


def _uniform(n=100, dt=0.04):
    return {i: i * dt for i in range(n)}


def test_uniform_intervals_are_cfr():
    assert s3._is_vfr(_uniform()) is False


def test_varying_intervals_are_vfr():
    # 一半 40ms、一半 80ms —— 明显变帧率
    ts, t = {}, 0.0
    for i in range(100):
        ts[i] = t
        t += 0.04 if i % 2 == 0 else 0.08
    assert s3._is_vfr(ts) is True


def test_tiny_jitter_stays_cfr():
    """时间基/浮点抖动不应被当成 VFR（默认容差 1%）。"""
    ts, t = {}, 0.0
    for i in range(100):
        ts[i] = t
        t += 0.04 + (1e-6 if i % 3 == 0 else 0.0)
    assert s3._is_vfr(ts) is False


def test_insufficient_data_is_not_vfr():
    """数据不足时按未知处理（False），不冒充判定。"""
    assert s3._is_vfr({}) is False
    assert s3._is_vfr({0: 0.0}) is False
    assert s3._is_vfr({0: 0.0, 1: 0.04}) is False
    assert s3._is_vfr(None) is False


def test_unusable_values_are_not_vfr():
    assert s3._is_vfr({0: None, 1: None, 2: None}) is False
    assert s3._is_vfr({0: "x", 1: "y", 2: "z"}) is False


def test_real_cfr_clip_is_not_flagged():
    """真实 CFR 素材不能被判成 VFR（防误伤）。

    就地用 ffmpeg 合成一段 25fps 短视频；ffmpeg 不可用时跳过。
    """
    import shutil
    import tempfile
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe not available")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "cfr.mkv")
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=64x48:rate=25:d=2",
         "-c:v", "ffv1", "-an", p],
        capture_output=True, timeout=120)
    if r.returncode != 0:
        pytest.skip("cannot synthesize a clip here")
    try:
        pts = s3._query_all_pts(p)
        assert pts, "应能取到 PTS"
        assert s3._is_vfr(pts) is False, "CFR 素材不应被判为 VFR"
    finally:
        for f in os.listdir(d):
            try:
                os.unlink(os.path.join(d, f))
            except OSError:
                pass
        os.rmdir(d)
