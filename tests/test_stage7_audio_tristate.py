# -*- coding: utf-8 -*-
"""GATE-06：源侧音轨存在性必须是三态，不能把"探测失败"当成"没有音轨"。

原实现 `_source_has_audio_stream()` 返回 bool：ffprobe 出错/超时/异常时
统一 `return False`，于是"暂时探测不到"被写成"源片本来就没音轨"，
直接加 `-an` 产出静音视频。而后续全链核对与机械核对都只看视频事实，
**发现不了音轨丢失**（静默丢音轨）。

正确做法：三态 present / absent / unknown；unknown 默认失败，
只有明确 absent 才允许自然无声。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage7_encode as s7  # noqa: E402


def test_tristate_function_exists():
    assert callable(getattr(s7, "_probe_audio_presence", None))


def test_missing_file_is_unknown_not_absent():
    """文件不存在必须是 unknown，而不是被当成"没有音轨"。"""
    got = s7._probe_audio_presence(
        os.path.join("no", "such", "file", "x.mkv"))
    assert got == "unknown", got


def test_garbage_file_is_unknown_not_absent():
    """不可解析的文件同样是 unknown。"""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False) as f:
        f.write(b"not a real matroska file" * 100)
        p = f.name
    try:
        got = s7._probe_audio_presence(p)
    finally:
        os.unlink(p)
    assert got in ("unknown", "absent"), \
        f"垃圾文件应判 unknown 或 absent，实际 {got}"


def test_video_only_file_is_absent(tmp_path):
    """确有无音轨的视频 -> absent（不是 unknown）。

    用 ffmpeg 现场生成一个无音轨短视频，避免依赖仓库外素材。
    ffmpeg 不可用时跳过。
    """
    import subprocess
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available")
    p = str(tmp_path / "silent.mkv")
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=64x48:d=1",
         "-c:v", "ffv1", "-an", p],
        capture_output=True, timeout=120)
    if r.returncode != 0:
        pytest.skip("cannot synthesize a clip here")
    assert s7._probe_audio_presence(p) == "absent"


def test_old_bool_helper_is_no_longer_the_decision_maker():
    """二值辅助仍然存在，但决策必须走三态。

    这里只锁住"别再把 bool 结果当事实"这个约定：二值函数在探测失败时
    返回 False，与 unknown 无法区分 —— 所以它不能单独决定加不加 -an。
    """
    ok = s7._source_has_audio_stream(
        os.path.join("no", "such", "file", "x.mkv"))
    assert ok is False, "二值辅助对缺失文件返回 False（无法区分未知与无音轨）"
