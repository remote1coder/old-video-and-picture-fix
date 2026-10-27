# -*- coding: utf-8 -*-
"""GATE-04：相对音画差必须**逐分量**相减，不能先压成标量再相减。

原实现 `_measure_av_drift()` 把 (start, end, span) 三个分量用
`max(..., key=abs)` 压成一个带符号标量，调用方再对两个标量相减：

    _signed = _signed - _ref_drift

这会**抵消新增的错位**。报告给出的反例（秒，无 priming）：

    源   视频[0,10]  音频[0,10.5]  → 分量 (0, 0.5, 0.5) → 压成 0.5
    成品 视频[0,10]  音频[0.5,10.5] → 分量 (0.5, 0.5, 0) → 压成 0.5
    两个标量相减 = 0 → 判"同步"

但音频开始时间比源**新增了 500ms 延迟**，是真实回归。

修法：先逐分量相减，再取最大的那个差值（本测试锁住这一点）。
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


# 报告反例的三分量
SRC_VEC = (0.0, 0.5, 0.5)    # 源：a_start-v_start, a_end-v_end, a_span-v_span
OUT_VEC = (0.5, 0.5, 0.0)    # 成品


def test_scalar_subtraction_misses_the_regression():
    """旧算法（两个标量相减）必然漏掉这个 500ms 新增延迟。"""
    old_signed = (max(OUT_VEC, key=abs)
                  - max(SRC_VEC, key=abs))
    assert abs(old_signed) * 1000 == 0, \
        "旧算法应得出 0ms（这正是它被漏检的原因）"


def test_per_component_subtraction_catches_it():
    """新算法（逐分量相减后取最大）必须抓到 500ms。"""
    deltas = [o - r for o, r in zip(OUT_VEC, SRC_VEC)]
    worst = max(deltas, key=abs)
    assert abs(worst) * 1000 == pytest.approx(500.0), deltas
    # 方向：音频滞后（正）
    assert worst > 0


def test_vec_helper_exists_and_is_a_triple():
    assert callable(getattr(s7, "_measure_av_drift_vec", None))
    got = s7._measure_av_drift_vec(
        os.path.join("no", "such", "file.mkv"))
    assert got is None, "测不出必须返回 None，而不是抛异常或返回标量"


def test_scalar_and_vec_agree_when_measurable(monkeypatch):
    """能测出来时，标量版应等于向量版里绝对值最大的那一项。"""
    vals = {"v_start": 0.0, "a_start": 0.25,
            "v_span": 10.0, "a_span": 10.25}

    def fake_start(path, kind):
        return vals["v_start" if kind == "v" else "a_start"]

    def fake_dur(path, kind):
        return vals["v_span" if kind == "v" else "a_span"]

    monkeypatch.setattr(s7, "_probe_audio_presence", lambda p: "present")
    monkeypatch.setattr(s7, "_probe_stream_start", fake_start)
    monkeypatch.setattr(s7, "_get_stream_duration", fake_dur)
    monkeypatch.setattr(s7, "_probe_encoder_priming", lambda p, k: 0.0)

    scalar = s7._measure_av_drift("whatever.mkv")
    vec = s7._measure_av_drift_vec("whatever.mkv")
    assert vec is not None and len(vec) == 3
    assert scalar == pytest.approx(max(vec, key=abs))


# ------------------- R12-08：调用**真实**同步函数的方向阈值组合 -------------------

class _Log:
    """最小 logger 替身，记录 (event, message)。"""

    def __init__(self):
        self.events = []

    def info(self, ev, msg="", **kw):
        self.events.append(("info", ev, msg))

    def warning(self, ev, msg="", **kw):
        self.events.append(("warning", ev, msg))

    def error(self, ev, msg="", **kw):
        self.events.append(("error", ev, msg))

    def debug(self, *a, **k):
        pass


def _patch_probes(monkeypatch, video, audio, ref_vec=(0.0, 0.0, 0.0)):
    """给真实 `_validate_av_sync` 打桩：video/audio = (raw_start, span)。"""
    intervals = {"v": video, "a": audio}

    monkeypatch.setattr(s7, "_probe_audio_presence", lambda p: "present",
                        raising=False)
    monkeypatch.setattr(s7, "_probe_stream_start",
                        lambda p, st: intervals[st][0], raising=False)
    monkeypatch.setattr(s7, "_get_stream_duration",
                        lambda p, st: intervals[st][1], raising=False)
    monkeypatch.setattr(s7, "_probe_encoder_priming",
                        lambda p, st: 0.0, raising=False)
    monkeypatch.setattr(s7, "_measure_av_drift_vec",
                        lambda p: ref_vec, raising=False)


def test_r12_08_small_lead_over_fail_line_is_caught(monkeypatch):
    """报告反例：起点 -250ms（超 200ms 失败线）、跨度 +300ms（未超 400ms）。

    旧实现取"绝对值最大"的 +300ms → 套滞后 400ms 线 → 只警告、返回 0。
    逐分量判定后，起点 -250ms 单独命中超前失败线 → 必须 86。
    """
    # 源：视频/音频都是 [0, 10] → ref_vec 全 0
    # 成品：视频 [0,10]，音频 [-0.25, 10.05]（跨度 10.30）
    _patch_probes(monkeypatch, video=(0.0, 10.0), audio=(-0.25, 10.30))
    log = _Log()
    rc = s7._validate_av_sync("dummy.mkv", log, {},
                              reference_path="src.mkv")
    assert rc == 86, f"起点超前 250ms 超过 200ms 失败线，应 86；实际 {rc}"
    assert any(ev == "STAGE7_SYNC_FAIL" for _lv, ev, _m in log.events), \
        log.events


def test_r12_08_large_lag_under_fail_line_still_passes(monkeypatch):
    """对照：同一组里 +300ms 滞后确实未超 400ms —— 单独看它不该失败。

    这条保证上面的用例不是因为"任何非零都失败"而通过的。
    """
    # 只有跨度 +300ms（滞后），起点/终点都对齐
    _patch_probes(monkeypatch, video=(0.0, 10.0), audio=(0.0, 10.30))
    log = _Log()
    rc = s7._validate_av_sync("dummy.mkv", log, {},
                              reference_path="src.mkv")
    assert rc == 0, f"仅滞后 300ms 未超 400ms 失败线，不应失败；实际 {rc}"


def test_r12_08_direction_thresholds_are_asymmetric(monkeypatch):
    """同一幅度：超前要严（200ms）、滞后可宽（400ms）。

    取 250ms：超前应失败、滞后应只告警 —— 直接验证"按分量自身符号取阈值"。
    """
    # 仅整体位移 250ms、跨度不变（避免混入跨度项）
    #   音频提前 250ms → 起点/终点都是 -250ms（超前）→ 超 200ms 线 → 86
    _patch_probes(monkeypatch, video=(0.0, 10.0), audio=(-0.25, 10.0))
    assert s7._validate_av_sync("dummy.mkv", _Log(), {},
                                reference_path="src.mkv") == 86

    #   音频滞后 250ms → 起点/终点都是 +250ms（滞后）→ 未超 400ms 线 → 只告警
    _patch_probes(monkeypatch, video=(0.0, 10.0), audio=(0.25, 10.0))
    assert s7._validate_av_sync("dummy.mkv", _Log(), {},
                                reference_path="src.mkv") == 0
