# -*- coding: utf-8 -*-
"""音频三级：规格 / 范围 / 内容（R26-A08）。

复核原话："音频检查现在比流规格，这是相对旧版只比有无的改进，
**但没有比较内容、样本、起止与同步**，不足以满足上轮的音频等价验收要求。"

三个必须能抓的反例：
  ① 相同规格、内容不同（旧实现放过）；
  ② 整体偏移；
  ③ 尾部截断。
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "probes"))
sys.path.insert(0, str(ROOT / "shared_modules"))

import strict_compare as sc  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="ffmpeg not available")


def _mk(path, *, freq=440, dur=2.0, offset=0.0, extra_in=None):
    """造一个带音轨的小片。"""
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "lavfi", "-i", f"testsrc2=size=64x48:rate=10:duration={dur}",
           "-f", "lavfi", "-i",
           f"sine=frequency={freq}:sample_rate=44100:duration={dur}"]
    if extra_in:
        cmd += extra_in
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-shortest", str(path)]
    r = subprocess.run(cmd, capture_output=True, timeout=180)
    assert r.returncode == 0, r.stderr[:300]
    return str(path)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("aud")
    a = _mk(d / "a.mp4", freq=440)
    same_copy = d / "a_copy.mp4"
    shutil.copyfile(a, same_copy)
    # ① 同规格（44.1k/2ch? sine 默认 mono → 两边一致）但内容不同
    diff = _mk(d / "diff.mp4", freq=880)
    # ③ 尾部截断（同起点，时长更短）
    short = _mk(d / "short.mp4", freq=440, dur=1.0)
    return {"a": a, "copy": str(same_copy), "diff": diff, "short": short}


def test_same_file_all_three_levels_pass(media):
    r = sc.check_audio_three_levels(media["a"], media["copy"],
                                    require_audio="content")
    assert r["ok"] is True, r["problems"]
    assert r["content"] == "pcm-identical", r
    assert len(r["a"]) == 1 and r["a"][0]["sample_rate"] == "44100"


def test_same_spec_different_content_is_caught(media):
    """① 同规格不同内容：spec 级放过，content 级必须拒。"""
    spec = sc.check_audio_three_levels(media["a"], media["diff"],
                                       require_audio="spec")
    assert spec["ok"] is True, "规格确实相同（这正是旧实现唯一比的东西）"

    full = sc.check_audio_three_levels(media["a"], media["diff"],
                                       require_audio="content")
    assert full["ok"] is False, "同规格不同内容必须被 content 级抓到"
    assert full["content"] in ("differs", "unreadable"), full
    joined = " | ".join(full["problems"])
    assert "audio content differs" in joined, joined


def test_tail_truncation_is_caught_by_range(media):
    """③ 尾部截断：range 级必须拒（时长/起止不同）。"""
    r = sc.check_audio_three_levels(media["a"], media["short"],
                                    require_audio="range")
    assert r["ok"] is False, r
    joined = " | ".join(r["problems"])
    assert "duration" in joined, joined


def test_unreadable_is_not_identical(tmp_path, media):
    """读不到音轨 ≠ 一致。"""
    silent = tmp_path / "novideo.mp4"
    silent.write_bytes(b"not a media file")
    r = sc.check_audio_three_levels(media["a"], str(silent),
                                    require_audio="spec")
    assert r["ok"] is False
    assert any("could not be read" in x for x in r["problems"]), r


def test_level_normalisation():
    assert sc._audio_level_of(True) == "spec"
    assert sc._audio_level_of("content") == "content"
    assert sc._audio_level_of("RANGE") == "range"
    assert sc._audio_level_of(False) == ""
    assert sc._audio_level_of("bogus") == "spec"
