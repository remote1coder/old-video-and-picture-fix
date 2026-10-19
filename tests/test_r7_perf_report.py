"""性能重算（第七轮）：用实测墙钟 + 实际帧数，而不是拍脑袋。

第六轮的"5.6h / 18h"内部矛盾：仅按它自述的 14420 帧、AI 约 6 秒/帧，
AI 单项就是 `14420×6/3600 ≈ 24.03 小时`。任务书要求撤回这组数字，
「以实际输出帧数与 stage 墙钟时间重估」。

本模块只做**测量与换算**，不做系统级采样（CPU/GPU/显存/IO 都不测），
也不把线性外推说成置信区间。
"""

import json
import os
import subprocess
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "shared_modules") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "shared_modules"))

import perf_report as pr  # noqa: E402


def _write_log(tmp_path, rows):
    p = tmp_path / "job_job_001.log"
    with open(p, "w", encoding="utf-8") as f:
        for ts, ev, msg in rows:
            f.write(json.dumps({"timestamp": ts, "event": ev,
                                "message": msg}, ensure_ascii=False) + "\n")
    return str(p)


def test_parse_stage_timings_basic(tmp_path):
    log = _write_log(tmp_path, [
        ("2026-10-19T10:00:00+08:00", "STAGE_START", "Stage 0: a.py"),
        ("2026-10-19T10:00:01+08:00", "STAGE_SKIP", "Stage 0 skipped"),
        ("2026-10-19T10:00:01+08:00", "STAGE_START", "Stage 1: b.py"),
        ("2026-10-19T10:00:11+08:00", "STAGE_DONE", "Stage 1 complete"),
    ])
    t = pr.parse_stage_timings(log)
    by = {r["stage"]: r for r in t}
    assert by[0]["status"] == "skipped"
    assert by[1]["status"] == "done"
    assert by[1]["wall_s"] == pytest.approx(10.0, abs=0.01)


def test_parse_picks_last_run(tmp_path):
    """多次运行累积的日志，要取**最后一次**（不是把两次拼起来）。"""
    log = _write_log(tmp_path, [
        ("2026-10-19T10:00:00+08:00", "STAGE_START", "Stage 0: a.py"),
        ("2026-10-19T10:00:05+08:00", "STAGE_DONE", "Stage 0 complete"),
        ("2026-10-19T10:10:00+08:00", "STAGE_START", "Stage 0: a.py"),
        ("2026-10-19T10:10:02+08:00", "STAGE_DONE", "Stage 0 complete"),
    ])
    t = pr.parse_stage_timings(log)
    assert len(t) == 1
    assert t[0]["wall_s"] == pytest.approx(2.0, abs=0.01)


def test_running_stage_has_no_wall_time(tmp_path):
    log = _write_log(tmp_path, [
        ("2026-10-19T10:00:00+08:00", "STAGE_START", "Stage 3: c.py"),
    ])
    t = pr.parse_stage_timings(log)
    assert t[0]["status"] == "running"
    assert t[0]["wall_s"] is None


def test_probe_frame_count_counts_real_frames(tmp_path):
    """数**实际帧数**，不是时长 × 帧率。"""
    p = str(tmp_path / "c.mp4")
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=s=64x64:d=1:r=25", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-y", p], capture_output=True)
    if r.returncode != 0:
        pytest.skip("ffmpeg 不可用")
    n = pr.probe_frame_count(p)
    assert n == 25, f"实际帧数应为 25，得到 {n}"


def test_report_computes_fps_from_real_frames(tmp_path):
    p = str(tmp_path / "c.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=s=64x64:d=1:r=25", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-y", p], capture_output=True)
    timings = [{"stage": 1, "status": "done", "wall_s": 5.0}]
    rep = pr.build_perf_report(timings, {1: p})
    row = rep["rows"][0]
    assert row["frames"] == 25
    assert row["fps"] == pytest.approx(5.0, abs=0.01)


def test_missing_artifact_is_reported_not_guessed():
    timings = [{"stage": 2, "status": "done", "wall_s": 3.0}]
    rep = pr.build_perf_report(timings, {})
    row = rep["rows"][0]
    assert row["frames"] is None and row["fps"] is None
    assert "no artifact" in row["note"]


def test_extrapolate_is_linear_and_labelled():
    rep = {"rows": [{"stage": 1, "status": "done", "wall_s": 10.0,
                     "frames": 100, "fps": 10.0, "artifact": "x",
                     "note": ""}]}
    e = pr.extrapolate(rep, 36000)
    assert e[0]["hours"] == pytest.approx(1.0, abs=0.001)   # 36000/10/3600


def test_render_warns_it_is_not_a_confidence_interval():
    rep = {"rows": [{"stage": 1, "status": "done", "wall_s": 10.0,
                     "frames": 100, "fps": 10.0, "artifact": "x",
                     "note": ""}],
           "total_wall_s": 10.0}
    text = pr.render_perf_report(rep, target_frames=1000)
    assert "不是置信区间" in text
    assert "下界估计" in text


def test_source_does_not_claim_to_measure_gpu():
    """诚实性：源码里不能声称测了 CPU/GPU/显存（它没有）。"""
    src = open(os.path.join(ROOT, "shared_modules", "perf_report.py"),
               encoding="utf-8").read()
    assert "不测 CPU/GPU 占用" in src
    assert "不把\"输入帧数\"和\"输出帧数\"混为一谈" in src
