# -*- coding: utf-8 -*-
"""stage3 数帧逻辑的回归测试。

背景（2026-10-22 实测）：
    `_count_video_frames` 原来只用 `ffprobe -count_frames`，超时写死 60 秒。
    `-count_frames` 要**逐帧解码**，耗时随文件大小线性增长：
        38 MB 的 1080p AV1  → 19.4 s
    stage1 产出的 1.6 GB 16-bit 无损 mkv 需要十几分钟 → 必然超时；
    随后 `duration × fps` 兜底又因为 vspipe 产出的 mkv `format=duration`
    是 N/A 而失败 → 返回 0。

    调用方当时的处理是静默 `total_frames = 30000`，于是对 599 帧的素材
    切出 **118 个窗口（115 个空）**，白烧约 25 分钟，且日志里的
    "Processing frames 29952-30000" 与实际帧数完全脱节。

现在：
    1. 先读 nb_frames 元数据（瞬间、权威）；
    2. 再用 `-count_packets`（只解复用，快 60~70 倍）—— 命中即返回；
    3. 才轮到 `-count_frames`（超时按文件大小放宽）；
    4. 再退 duration×fps；
    5. 全失败返回 **0**，由调用方显式报错（不再猜 30000）。
"""
import os
import subprocess
import sys
import time

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "shared_modules"))
sys.path.insert(0, os.path.join(_ROOT, "video_restoration_pipeline", "stages"))
sys.path.insert(0, os.path.join(_ROOT, "video_restoration_pipeline"))


def _load_stage3():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stage3_under_test",
        os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                     "stage3_ai_upscale.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def s3():
    return _load_stage3()


def _make_video(path, frames=30, size="128x96"):
    """用 ffmpeg 生成一段定长测试视频（无音频）。"""
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           "-f", "lavfi", "-i", f"testsrc=size={size}:rate=10:duration={frames/10}",
           "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", "-y", path]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        pytest.skip("ffmpeg 不可用，跳过：" + r.stderr[-200:])


def test_counts_frames_of_real_mkv(s3, tmp_path):
    """基本正确性：数出来的帧数要与实际生成的一致。"""
    p = str(tmp_path / "n30.mkv")
    _make_video(p, frames=30)
    assert s3._count_video_frames(p) == 30


def test_fast_path_is_really_fast(s3, tmp_path):
    """必须走快速路径（解复用），不能退化成逐帧解码。

    这里用时长做代理指标：30 帧的小文件两种方式都快，所以真正的判据是
    **不能调用 -count_frames**。用 monkeypatch 记录被调用的 ffprobe 参数。
    """
    p = str(tmp_path / "n30.mkv")
    _make_video(p, frames=30)

    calls = []
    real_run = subprocess.run

    def _spy(cmd, *a, **kw):
        calls.append(list(cmd))
        return real_run(cmd, *a, **kw)

    s3.subprocess.run = _spy
    try:
        n = s3._count_video_frames(p)
    finally:
        s3.subprocess.run = real_run

    assert n == 30
    joined = [" ".join(c) for c in calls]
    assert any("-count_packets" in x for x in joined), \
        "没有使用 -count_packets 快速路径"
    assert not any("-count_frames" in x for x in joined), (
        "退化成逐帧解码了（-count_frames）；这正是 1.6 GB 中间产物超时的原因")


def test_missing_file_returns_zero_not_raise(s3, tmp_path):
    """不存在的文件返回 0（由调用方显式报错），不抛异常。"""
    assert s3._count_video_frames(str(tmp_path / "nope.mkv")) == 0


def test_corrupt_file_returns_zero(s3, tmp_path):
    """损坏文件返回 0，不抛异常。"""
    p = tmp_path / "bad.mkv"
    p.write_bytes(b"not a video" * 100)
    assert s3._count_video_frames(str(p)) == 0


def test_no_hardcoded_30000_fallback():
    """调用方不得再有 `total_frames = 30000` 这类猜测。

    回归：静默猜 30000 会让 599 帧的素材切出 118 个窗口（115 个空）。
    用 AST 而不是文本匹配 —— 注释/docstring 里出现这个字符串是正常的
    （就是在解释这个 bug），只有**真正的赋值**才算违规。
    """
    import ast
    path = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                        "stage3_ai_upscale.py")
    tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if not (isinstance(tgt, ast.Name) and tgt.id == "total_frames"):
                continue
            v = node.value
            if isinstance(v, ast.Constant) and v.value == 30000:
                offenders.append(node.lineno)
    assert not offenders, (
        f"stage3_ai_upscale.py 第 {offenders} 行仍有 `total_frames = 30000` "
        f"写死兜底 —— 会让短视频切出上百个空窗口")
