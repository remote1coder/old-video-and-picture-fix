# -*- coding: utf-8 -*-
"""R22-A04：最终时间门必须要求**完整的视频证据**。

## 审核给的三个反例

1. **零视频帧 + 容器 12 秒**
   > 受控 probe 返回"视频时长取不到、无视频 packet、容器时长 12 秒、帧数 0"
   > 时，测量结果为 `coverage_complete=False`、`frames=0`、`measured=True`，
   > 执行原 final_timeline 判定 AST，得到 required PASS。

2. **0/0.04/0.08 三帧、duration 全部 N/A**
   > `_probe_final_timeline` 仍返回 `ok=True, span=0.08, problem=""`。
   > 代码以"end 是否大于 first"判断尾时长是否存在；多帧即使所有 duration
   > 都没有，末 PTS 也大于首 PTS，因而漏报。

3. **`coverage_complete` 没有被判据消费**
   > 算出来了，但门禁只看 `measured`。

## 本文件锁住什么

- `measured` = 时长 > 0 **且** 帧数 > 0（零帧不是"测到了"）
- `_probe_final_timeline` 的 `coverage` 三态：observed / derived / unknown
- `final_timeline` 的 PASS 需要 `measured and coverage_complete`
- 合法样本仍然通过：单帧带 duration、非零共同原点、多帧完整覆盖
"""
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage7_encode as s7  # noqa: E402


class _Log:
    def __init__(self):
        self.events = []

    def _rec(self, e, m="", **k):
        self.events.append(e)

    info = warning = error = critical = _rec


def _patch(monkeypatch, *, stream=0.0, fmt=0.0, packet=None, frames=0):
    monkeypatch.setattr(
        s7, "_probe_container_duration",
        lambda path, stream_type="v", prefer="any": (
            (stream, "stream-duration") if prefer == "stream" and stream
            else ((fmt, "container-duration") if prefer == "format" and fmt
                  else (0.0, ""))))
    monkeypatch.setattr(s7, "_probe_final_timeline",
                        lambda path, **kw: (packet or {
                            "ok": False, "span": 0.0, "frames": 0,
                            "varies": None, "problem": "no packet timestamps",
                            "coverage": "unknown"}))
    monkeypatch.setattr(s7, "_count_frames_quietly", lambda *a, **k: frames)


# ------------------------------------------------ 反例 1：零帧 + 容器时长

def test_zero_video_frames_is_not_a_measurement(monkeypatch):
    _patch(monkeypatch, fmt=12.0, frames=0)
    info = s7.measure_final_candidate("x.mkv", _Log())
    assert info["duration"] == pytest.approx(12.0)
    assert info["frames"] == 0
    assert info["measured"] is False, "零帧被判成了'测到了'"
    assert info["coverage_complete"] is False
    assert any("no video frames" in p for p in info["problems"]), info["problems"]


def test_container_duration_alone_is_not_a_complete_coverage(monkeypatch):
    """容器时长只能诊断，不能当视频覆盖（即使帧数正常）。"""
    _patch(monkeypatch, fmt=12.0, frames=20)
    info = s7.measure_final_candidate("x.mkv", _Log())
    assert info["measured"] is True
    assert info["coverage_complete"] is False
    assert any("audio tail" in p for p in info["problems"]), info["problems"]


# ------------------------------------------------ 反例 2：末帧无 duration

def test_last_frame_without_duration_is_derived_not_observed(monkeypatch):
    """0/0.04/0.08、duration 全 N/A：只能算 derived，且必须报"少一帧"。"""
    _patch(monkeypatch, frames=3, packet={
        "ok": True, "span": 0.08, "frames": 3, "varies": False,
        "problem": "the last packet has no duration; the span excludes the "
                   "final frame",
        "coverage": "derived", "last_frame_duration": 0.0})
    info = s7.measure_final_candidate("x.mkv", _Log())
    assert info["measured"] is True
    assert info["coverage_complete"] is False, "缺末帧时长却标成完整覆盖"
    assert info.get("coverage") == "derived"
    assert any("excludes the final frame" in p for p in info["problems"])


def test_single_frame_without_duration_is_unknown(monkeypatch):
    _patch(monkeypatch, frames=1, packet={
        "ok": False, "span": 0.0, "frames": 1, "varies": None,
        "problem": "the last packet has no duration",
        "coverage": "unknown", "last_frame_duration": 0.0})
    info = s7.measure_final_candidate("x.mkv", _Log())
    assert info["coverage_complete"] is False
    assert info.get("coverage") in (None, "unknown")


# ------------------------------------------------ 合法样本

def test_single_frame_with_duration_is_observed(monkeypatch):
    """单帧**带** duration 是合法的一代（不是所有单帧都该被拒）。"""
    _patch(monkeypatch, frames=1, packet={
        "ok": True, "span": 0.04, "frames": 1, "varies": False,
        "problem": "", "coverage": "observed",
        "last_frame_duration": 0.04})
    info = s7.measure_final_candidate("x.mkv", _Log())
    assert info["measured"] is True
    assert info["coverage_complete"] is True


def test_full_coverage_marks_both_fields(monkeypatch):
    _patch(monkeypatch, frames=20, packet={
        "ok": True, "span": 0.667, "frames": 20, "varies": False,
        "problem": "", "coverage": "observed", "last_frame_duration": 0.033})
    info = s7.measure_final_candidate("x.mkv", _Log())
    assert info["measured"] is True
    assert info["coverage_complete"] is True
    assert info["basis"] == "video-packet-span"


# ------------------------------------------------ _probe_final_timeline 自身

def _fake_packets(monkeypatch, rows):
    """把 ffprobe 的 csv 输出喂给真实的 `_probe_final_timeline`。"""
    class R:
        returncode = 0
        stdout = "\n".join(f"{p},{d}" for p, d in rows) + "\n"
        stderr = ""

    monkeypatch.setattr(s7.subprocess, "run", lambda *a, **k: R())


def test_probe_timeline_marks_missing_tail_duration(monkeypatch):
    """审核原例：PTS 递增但 duration 全 N/A —— 旧实现返回 ok=True 且 problem 空。"""
    _fake_packets(monkeypatch, [(0.0, "N/A"), (0.04, "N/A"), (0.08, "N/A")])
    out = s7._probe_final_timeline("x.mkv")
    assert out["ok"] is True                 # 时间信息可用
    assert out["coverage"] == "derived"      # 但覆盖不完整
    assert "excludes the final frame" in out["problem"]
    assert out["last_frame_duration"] == 0.0


def test_probe_timeline_marks_observed_when_tail_has_duration(monkeypatch):
    _fake_packets(monkeypatch, [(0.0, 0.04), (0.04, 0.04), (0.08, 0.04)])
    out = s7._probe_final_timeline("x.mkv")
    assert out["coverage"] == "observed"
    assert out["span"] == pytest.approx(0.12)   # 末帧 pts + 它自己的时长


def test_probe_timeline_single_frame_without_duration_is_unknown(monkeypatch):
    _fake_packets(monkeypatch, [(0.0, "N/A")])
    out = s7._probe_final_timeline("x.mkv")
    assert out["coverage"] == "unknown"
    assert out["ok"] is False               # 连一帧多长都不知道


# ------------------------------------------------ 判据接线

def test_final_timeline_gate_consumes_coverage():
    """门禁必须消费 `coverage_complete`，不能只看 measured。

    R23-B04 把判定收敛到 `stage7.final_timeline_status()`，所以这里改成
    **调用那个判据**（行为断言），不再 grep 源码字面量 —— 源码搬一次家
    就失效的断言，保护的不是行为。
    """
    import stage7_encode as _s7
    good = {"measured": True, "coverage_complete": True, "duration": 0.4,
            "basis": "video-packet-span"}
    assert _s7.final_timeline_status(good)[0] == "PASS"
    status, why = _s7.final_timeline_status(
        dict(good, coverage_complete=False))
    assert status == "UNVERIFIED"
    assert "coverage" in why.lower(), why

    # 门禁本身必须调用这个判据（否则上面测的是"另一个函数"）
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage7_encode.py"),
               encoding="utf-8", errors="replace").read()
    assert "final_timeline_status(_final_tl)" in src, \
        "final_timeline 的判定没有走单一来源"
