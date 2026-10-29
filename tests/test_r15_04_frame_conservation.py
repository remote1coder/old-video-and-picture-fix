# -*- coding: utf-8 -*-
"""R15-04：中间产物不许被"重定时"悄悄改掉帧数。

**真机证据**

| 场景 | 输入帧 | 输出帧 |
|---|---|---|
| 真实 VFR 素材（stage4 无 passthrough） | 458 | **381**（-16.8%） |
| 受控复现（合成 VFR） | 120 | **102**（-15%） |
| 加 `-fps_mode passthrough` | 120 | 120 |

**根本原因**：`stage3` 的 VFR 路径写出的 MKV 声明 `r_frame_rate=25/1`
（帧数对、标签错），`stage4` 的默认 `-vsync cfr` 按 25fps 重定时。

本文件验证 `shared_modules/frame_conservation.py` 的判定，并用真实
ffmpeg 走一遍"坏命令 vs 好命令"。
"""
import os
import shutil
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import frame_conservation as fc  # noqa: E402


def _ffmpeg():
    return shutil.which("ffmpeg") is not None


def _run(cmd, timeout=120):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, shell=False)


def _make_vfr(path, tmp):
    """造一个"帧数对但声明 25/1"的 VFR MKV（复现 stage3 的写法）。"""
    base = str(tmp / "base.mp4")
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
              "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=2",
              "-c:v", "libx264", "-pix_fmt", "yuv420p", base])
    assert r.returncode == 0, r.stderr[-300:]

    frames = tmp / "frames"
    frames.mkdir(exist_ok=True)
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", base,
              "-vsync", "0", str(frames / "f_%08d.png")])
    assert r.returncode == 0, r.stderr[-300:]
    names = sorted(os.listdir(str(frames)))

    lines = []
    for n in names:
        lines.append("file '%s'" % (frames / n).as_posix())
        lines.append("duration 0.033333")
    concat = str(tmp / "concat.txt")
    with open(concat, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # stage3 的写法：concat + passthrough
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
              "-f", "concat", "-safe", "0", "-i", concat,
              "-fps_mode", "passthrough", "-c:v", "ffv1",
              "-pix_fmt", "yuv420p16le", "-an", str(path)])
    assert r.returncode == 0, r.stderr[-300:]
    return str(path), len(names)


@pytest.mark.skipif(not _ffmpeg(), reason="ffmpeg 不在 PATH 上")
def test_bad_command_drops_frames_good_command_preserves(tmp_path):
    """对照：同一个输入，坏命令丢帧、好命令守恒；判定函数要能分辨。"""
    vfr, n_in = _make_vfr(tmp_path / "vfr.mkv", tmp_path)
    assert n_in >= 50

    bad = str(tmp_path / "bad.mkv")
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", vfr,
              "-filter_complex", "[0:v]null[vout]", "-map", "[vout]",
              "-c:v", "ffv1", "-pix_fmt", "yuv420p16le", "-an", bad])
    assert r.returncode == 0, r.stderr[-300:]

    good = str(tmp_path / "good.mkv")
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", vfr,
              "-filter_complex", "[0:v]null[vout]", "-map", "[vout]",
              "-fps_mode", "passthrough",
              "-c:v", "ffv1", "-pix_fmt", "yuv420p16le", "-an", good])
    assert r.returncode == 0, r.stderr[-300:]

    n_bad = fc._count_frames(bad)
    n_good = fc._count_frames(good)
    assert n_bad is not None and n_good is not None
    assert n_bad < n_in, f"坏命令竟然没丢帧（{n_in} -> {n_bad}）—— 素材不够 VFR"
    assert n_good == n_in, f"好命令丢帧了（{n_in} -> {n_good}）"

    ok_bad, rep_bad = fc.check_conservation(vfr, bad, "sim-broken")
    assert ok_bad is False, rep_bad
    assert any("frame count changed" in x or "r_frame_rate changed" in x
               for x in rep_bad["reasons"]), rep_bad["reasons"]

    ok_good, rep_good = fc.check_conservation(vfr, good, "sim-good")
    assert ok_good is True, rep_good


# ---------------------------------------------------------------- 判定细节

def test_missing_files_are_rejected(tmp_path):
    ok, rep = fc.check_conservation(str(tmp_path / "nope.mkv"),
                                   str(tmp_path / "also.mp4"), "x")
    assert ok is False
    assert any("missing" in r for r in rep["reasons"])


def test_rate_parsing_handles_na_and_zero():
    assert fc._rate("30000/1001") == pytest.approx(29.970, abs=1e-3)
    assert fc._rate("25") == 25.0
    assert fc._rate("N/A") is None
    assert fc._rate("0/0") is None
    assert fc._rate("") is None
    assert fc._rate(None) is None


def test_describe_is_metadata_only(tmp_path):
    """`describe()` 不许做全解码（长片会不可接受）。

    只查 **describe 函数体**（模块文档里解释"为什么不用 count_frames"
    是正常的，不能因此误判）。
    """
    src = open(os.path.join(_ROOT, "shared_modules",
                            "frame_conservation.py"),
               encoding="utf-8").read()
    body = src.split("def describe(", 1)[1].split("def check_conservation", 1)[0]
    assert "-count_frames" not in body, \
        "describe() 路径上不能出现 -count_frames"
    assert "count_frames" not in body, \
        "describe() 不得调用精确计数"


def test_large_files_skip_exact_counting(tmp_path, monkeypatch):
    """超过阈值时只做元数据检查，并在报告里注明（不假装做过）。"""
    a = tmp_path / "a.mkv"
    b = tmp_path / "b.mkv"
    for p in (a, b):
        p.write_bytes(b"x" * 1024)

    monkeypatch.setattr(fc, "describe",
                        lambda p: {"path": p, "r_frame_rate": 30.0,
                                   "avg_frame_rate": 30.0,
                                   "duration": 10.0, "bytes": 10 ** 9})
    ok, rep = fc.check_conservation(str(a), str(b), "big",
                                    exact_max_bytes=1024)
    assert ok is True
    assert rep["exact"] is False
    assert "exact counting skipped" in rep.get("note", "")
