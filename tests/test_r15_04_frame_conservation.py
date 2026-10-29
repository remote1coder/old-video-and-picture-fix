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


# ---------------------------------------------------------------- 帧率标签告警

def test_fps_label_mismatch_warns_but_does_not_fail(tmp_path, monkeypatch):
    """中间产物帧率标签与意图不符 → **告警**，不判失败。

    VFR 中间产物常被 ffmpeg 标成 25/1（时间戳仍正确）。下游只要用了
    `-fps_mode passthrough` 就不会丢帧，所以这里只提示、不阻断。
    """
    a = tmp_path / "a.mkv"
    b = tmp_path / "b.mkv"
    for p in (a, b):
        p.write_bytes(b"x" * 512)

    monkeypatch.setattr(
        fc, "describe",
        lambda p: {"path": p, "r_frame_rate": 25.0, "avg_frame_rate": 25.0,
                   "duration": 10.0, "bytes": len(b"x" * 512)})

    class L:
        def __init__(self):
            self.events = []

        def info(self, e, m="", **k):
            self.events.append(("info", e))

        def warning(self, e, m="", **k):
            self.events.append(("warn", e))

        def error(self, e, m="", **k):
            self.events.append(("error", e))

    log = L()
    ok, rep = fc.check_conservation(str(a), str(b), "stage4", log,
                                    expected_fps=30.146,
                                    exact_max_bytes=10)
    assert ok is True, "标签不符只告警，不应判失败"
    assert rep.get("fps_label_mismatch", {}).get("labeled") == 25.0
    assert ("warn", "FRAME_FPS_LABEL_MISMATCH") in log.events


def test_fps_label_match_does_not_warn(tmp_path, monkeypatch):
    a = tmp_path / "a.mkv"
    b = tmp_path / "b.mkv"
    for p in (a, b):
        p.write_bytes(b"x" * 512)

    monkeypatch.setattr(
        fc, "describe",
        lambda p: {"path": p, "r_frame_rate": 30000 / 1001,
                   "avg_frame_rate": 30000 / 1001,
                   "duration": 10.0, "bytes": 512})

    ok, rep = fc.check_conservation(str(a), str(b), "stage4",
                                    expected_fps=30000 / 1001,
                                    exact_max_bytes=10)
    assert ok is True
    assert "fps_label_mismatch" not in rep


# ---------------------------------------------------------------- VFR 保护

@pytest.mark.skipif(not _ffmpeg(), reason="ffmpeg 不在 PATH 上")
def test_vfr_intermediate_preserves_variable_timing(tmp_path):
    """VFR 中间产物的**可变时序必须保留**。

    这条是**防回归**用的：stage3 的 VFR 写出（concat + 逐帧 duration +
    passthrough）声明的 `r_frame_rate` 是 `25/1`，看着像 bug，其实 VFR 流
    本来就没有单一帧率 —— 那只是个占位值。

    我一度想"修"它：在输入侧加 `-r 30`。实测证明那会**摧毁** VFR：

    | 写法 | 真 VFR（期望 8.0s）产出时长 |
    |---|---|
    | 现状（无 -r） | **7.92s**（可变时序保留） |
    | 输入侧 -r 30 | **4.00s**（被压平成 CFR） |

    所以现状是对的。这条测试把"不能加 -r 去压平"这件事锁住。
    """
    frames = tmp_path / "frames"
    frames.mkdir()
    base = str(tmp_path / "base.mp4")
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
              "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30:duration=1",
              "-c:v", "libx264", "-pix_fmt", "yuv420p", base])
    assert r.returncode == 0, r.stderr[-300:]
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", base,
              "-vsync", "0", str(frames / "f_%08d.png")])
    assert r.returncode == 0, r.stderr[-300:]
    names = sorted(os.listdir(str(frames)))
    assert len(names) >= 20

    # 前一半 0.0333，后一半 0.1 —— 明显变帧率
    half = len(names) // 2
    lines = []
    for i, n in enumerate(names):
        lines.append("file '%s'" % (frames / n).as_posix())
        lines.append("duration %.6f" % (0.033333 if i < half else 0.1))
    concat = str(tmp_path / "concat.txt")
    with open(concat, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    expected = half * 0.033333 + (len(names) - half) * 0.1

    # stage3 的写法（**不带** -r）
    out = str(tmp_path / "out.mkv")
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
              "-f", "concat", "-safe", "0", "-i", concat,
              "-fps_mode", "passthrough", "-c:v", "ffv1",
              "-pix_fmt", "yuv420p16le", "-an", out])
    assert r.returncode == 0, r.stderr[-300:]

    got = fc._duration(out)
    assert got is not None
    # 关键：时长必须接近**可变**时序算出来的值，而不是被压成某个固定帧率
    assert abs(got - expected) / expected < 0.05, \
        f"VFR 时序被压平了：期望 {expected:.2f}s，实测 {got:.2f}s"

    # 反证：加了 -r 就会被压平（确认这条测试真的能抓到问题）
    flat = str(tmp_path / "flat.mkv")
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
              "-r", "30", "-f", "concat", "-safe", "0", "-i", concat,
              "-fps_mode", "passthrough", "-c:v", "ffv1",
              "-pix_fmt", "yuv420p16le", "-an", flat])
    got_flat = fc._duration(flat) if r.returncode == 0 else None
    assert got_flat is not None
    assert abs(got_flat - expected) / expected > 0.05, \
        "-r 竟然没压平 VFR（那这条测试就失去意义了）"
