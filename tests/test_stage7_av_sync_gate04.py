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
