# -*- coding: utf-8 -*-
"""R21-C05：时间比较漏掉中段漂移 / 最终时长用错量。

## 审核给的两个精确反例

**① 中段累计漂移**

> `expected[i] = 0.04*i`，`actual[i] = expected[i] + 0.001*min(i, 2000-i)`，
> i=0…2000。当前端点和相邻差都通过，中段却偏 1 秒。

根因：只比"原点 / 末端跨度 / 相邻间隔"。每步只差 1 ms（< 2 ms 容差），
末端又刚好回到 0，所以三条判据全过。**残差必须逐帧累加起来看。**

**② 最终时长把 `out_last` 当完整覆盖**

> 三帧 0/0.04/0.08、末帧 0.04 的覆盖是 0.12；当前得到 0.08，并报告
> measured=true。传入正无穷也会被接纳。

根因：`out_last = 末帧 PTS − 首帧 PTS`，**不含末帧自己的时长**；
且没有拒绝非有限值。

## 本文件锁住的

- 中段漂移必须被拒，且报出最坏帧号与幅度
- 合法样本（完全相同 / 有合同的整体平移）仍通过
- 最终测量：优先视频 packet 跨度（含末帧时长）；容器时长要标注"可能含音轨";
  `out_last` 只在最后兜底且明确标注"缺末帧时长"；Inf 一律不算测到
"""
import math
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import time_mapping as tm  # noqa: E402
import stage7_encode as s7  # noqa: E402


class _Log:
    def __init__(self):
        self.events = []

    def _rec(self, ev, msg="", **kw):
        self.events.append((ev, msg, kw))

    info = warning = error = critical = _rec


# ------------------------------------------------ ① 中段累计漂移

def _drift_case(n=2001, step=0.04, amp=0.001):
    exp = [step * i for i in range(n)]
    act = [e + amp * min(i, n - 1 - i) for i, e in enumerate(exp)]
    return act, exp


def test_middle_drift_is_rejected():
    """审核原例：端点对齐、每步 1ms，中段偏 1 秒。"""
    act, exp = _drift_case()
    ok, problems, m = tm.check_against_expected(act, exp, time_base=0.001)
    assert ok is False, "中段偏 1 秒却判通过"
    assert any("drift" in p for p in problems), problems
    assert m["max_residual_index"] == 1000
    assert abs(m["max_residual_s"] - 1.0) < 1e-9


def test_drift_of_a_single_frame_beyond_tolerance_is_rejected():
    act, exp = _drift_case(n=50, step=0.04, amp=0.0)
    act[20] += 0.005          # 单帧偏 5ms > 2ms
    ok, problems, _ = tm.check_against_expected(act, exp, time_base=0.001)
    assert ok is False


def test_identical_timeline_still_passes():
    exp = [0.04 * i for i in range(200)]
    ok, problems, m = tm.check_against_expected(list(exp), exp, time_base=0.001)
    assert ok is True, problems
    assert m["max_residual_s"] == 0.0


def test_a_contractual_origin_shift_is_not_drift():
    """整体平移不是漂移：给了 allow_origin_shift 就必须通过。"""
    exp = [0.04 * i for i in range(200)]
    act = [x + 12.5 for x in exp]
    ok, problems, m = tm.check_against_expected(
        act, exp, time_base=0.001, allow_origin_shift=True)
    assert ok is True, problems
    assert abs(m["max_residual_s"]) < 1e-9


def test_non_finite_actual_is_rejected_before_anything_else():
    exp = [0.04 * i for i in range(10)]
    act = list(exp)
    act[3] = float("inf")
    ok, problems, _ = tm.check_against_expected(act, exp, time_base=0.001)
    assert ok is False
    assert any("non-finite" in p for p in problems), problems


# ------------------------------------------------ ② 最终测量的量

def _patch_probes(monkeypatch, stream=None, fmt=None, packet=None, frames=20):
    monkeypatch.setattr(
        s7, "_probe_container_duration",
        lambda path, stream_type="v", prefer="any": (
            (stream, "stream-duration") if prefer == "stream" and stream
            else ((fmt, "container-duration") if prefer == "format" and fmt
                  else (0.0, ""))))
    monkeypatch.setattr(
        s7, "_probe_final_timeline",
        lambda path, **kw: (packet or {"ok": False, "span": 0.0,
                                       "frames": 0, "varies": None,
                                       "problem": "no packets"}))
    # R22-A04：`measured` 现在还要求 frames>0（零帧不算测到），
    # 所以桩必须给出一个合理帧数 —— 这些用例测的是取值顺序，不是帧数。
    monkeypatch.setattr(s7, "_count_frames_quietly", lambda *a, **k: frames)


def test_retime_span_without_tail_is_flagged_incomplete(monkeypatch):
    """3 帧 0/0.04/0.08 末帧 0.04 → 真覆盖 0.12；只有 out_last 时不得假装 0.08。"""
    _patch_probes(monkeypatch)
    info = s7.measure_final_candidate(
        "x.mkv", _Log(),
        retime_info={"action": "retimed", "out_last": 0.08, "frames": 3})
    assert info["measured"] is True
    assert info["duration"] == pytest.approx(0.08)
    assert info["coverage_complete"] is False
    assert any("excludes the last frame" in p for p in info["problems"]), info


def test_retime_span_plus_tail_gives_the_true_coverage(monkeypatch):
    _patch_probes(monkeypatch)
    info = s7.measure_final_candidate(
        "x.mkv", _Log(),
        retime_info={"action": "retimed", "out_last": 0.08, "frames": 3,
                     "last_frame_duration": 0.04})
    assert info["duration"] == pytest.approx(0.12)
    assert info["coverage_complete"] is True


def test_infinite_retime_span_is_not_accepted(monkeypatch):
    _patch_probes(monkeypatch)
    info = s7.measure_final_candidate(
        "x.mkv", _Log(),
        retime_info={"action": "retimed", "out_last": float("inf"),
                     "frames": 3})
    assert info["measured"] is False
    assert info["duration"] == 0.0


def test_video_packet_span_wins_when_the_timeline_changed(monkeypatch):
    """重定时过的代，必须用**含末帧时长**的视频跨度，而不是容器总时长。"""
    _patch_probes(
        monkeypatch, stream=0.0, fmt=9.99,
        packet={"ok": True, "span": 0.12, "frames": 3, "varies": False,
                "problem": "", "coverage": "observed",
                "last_frame_duration": 0.04})
    info = s7.measure_final_candidate(
        "x.mkv", _Log(),
        retime_info={"action": "retimed", "out_last": 0.08, "frames": 3})
    assert info["basis"] == "video-packet-span"
    assert info["duration"] == pytest.approx(0.12)
    assert info["coverage_complete"] is True


def test_container_duration_is_labelled_when_nothing_else_works(monkeypatch):
    _patch_probes(monkeypatch, stream=0.0, fmt=12.0)
    info = s7.measure_final_candidate("x.mkv", _Log())
    assert info["measured"] is True
    assert info["basis"] == "container-duration"
    assert any("audio tail" in p for p in info["problems"]), info


def test_video_stream_duration_is_preferred_and_complete(monkeypatch):
    _patch_probes(monkeypatch, stream=10.0, fmt=11.0)
    info = s7.measure_final_candidate("x.mkv", _Log())
    assert info["basis"] == "stream-duration"
    assert info["duration"] == pytest.approx(10.0)
    assert info["coverage_complete"] is True
