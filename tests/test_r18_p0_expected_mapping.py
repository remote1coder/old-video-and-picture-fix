# -*- coding: utf-8 -*-
"""第十八轮 P0-C：预期输出映射（`shared_modules/time_mapping.py`）。

审核要求把"时间映射"做成跨阶段数据，至少包含：
帧身份/顺序、精确时间基、PTS、duration、末帧时长、起始偏移、变换来源、
预期帧数。

本模块实现其中**逐帧 PTS 与时长**这一层，并给出五种变换各自的期望时间轴。
目的是让判据从"像不像源"变成"像不像**你要的那个东西**"。
"""
import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "shared_modules") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "shared_modules"))

import time_mapping as tm  # noqa: E402


SRC = [i / 30.0 for i in range(30)]      # 30 帧 @ 29.97-ish，1s 左右
SRC_1S = [i * (1 / 25.0) for i in range(25)]   # 25 帧 @ 25fps = 恰好 1s


# ------------------------------------------------ 五种变换

def test_identity_keeps_the_source_timeline():
    out = tm.expected_pts(SRC, tm.Transform("identity"))
    assert out == pytest.approx(SRC)


def test_speed_two_halves_the_duration_but_keeps_frame_count():
    """T12 的一半：合法 2× 变速必须被表达出来，而不是被改回源速度。"""
    out = tm.expected_pts(SRC_1S, tm.Transform("speed", 2.0))
    assert len(out) == len(SRC_1S), "变速不改变帧数"
    assert out[0] == pytest.approx(SRC_1S[0])
    assert out[-1] - out[0] == pytest.approx(0.48, abs=1e-6), \
        "1s 的素材 2× 后跨度应约 0.48s（末帧不重复计时）"


def test_speed_half_doubles_the_duration():
    out = tm.expected_pts(SRC_1S, tm.Transform("speed", 0.5))
    assert out[-1] - out[0] == pytest.approx(1.92, abs=1e-6)


def test_fps_reinterpret_keeps_count_and_respaces():
    """人工帧率重解释：同帧数、换时间基。"""
    out = tm.expected_pts(SRC_1S, tm.Transform("fps_reinterpret", 50.0))
    assert len(out) == len(SRC_1S)
    assert out[1] - out[0] == pytest.approx(0.02, abs=1e-9)


def test_double_doubles_frames_and_halves_intervals():
    """倍帧（QTGMC 50i→50p）：帧数翻倍、间隔减半。"""
    src = [0.0, 0.04, 0.08]
    out = tm.expected_pts(src, tm.Transform("double"))
    assert len(out) == 5, out
    assert out[1] - out[0] == pytest.approx(0.02, abs=1e-9)


def test_decimate_drops_one_frame_per_cycle():
    """抽帧（IVTC cycle=5 去 1 帧）。"""
    src = [i * 0.04 for i in range(10)]
    out = tm.expected_pts(src, tm.Transform("decimate", 5))
    assert len(out) == 8, out
    assert 0.16 not in out, "cycle=5 时每 5 帧的第 5 帧应被丢掉"


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        tm.expected_pts(SRC, tm.Transform("warp", 1.0))


def test_single_frame_source_raises():
    with pytest.raises(ValueError):
        tm.expected_pts([0.0], tm.Transform("identity"))


# ------------------------------------------------ 从 params 推断

@pytest.mark.parametrize("params,kind,value", [
    ({}, "identity", 1.0),
    ({"expected_speed": 1.0}, "identity", 1.0),
    ({"expected_speed": 2.0}, "speed", 2.0),
    ({"fps_interpret": "25"}, "fps_reinterpret", 25.0),
    ({"fps_interpret": "30000/1001"}, "fps_reinterpret", 30000 / 1001),
    ({"fps_interpret": "25", "expected_speed": 2.0},
     "fps_reinterpret", 25.0),      # 重解释优先
])
def test_transform_from_params(params, kind, value):
    t = tm.transform_from_params(params)
    assert t.kind == kind
    assert t.value == pytest.approx(value, rel=1e-9)


# ------------------------------------------------ 对照校验

def test_check_rejects_frame_count_mismatch():
    ok, problems, _ = tm.check_against_expected([0.0, 0.04], [0.0, 0.04, 0.08])
    assert ok is False and any("frame count" in p for p in problems)


def test_check_accepts_quantisation_level_error():
    """仅 v2 量化级偏差（1ms）必须通过。"""
    exp = tm.expected_pts(SRC_1S, tm.Transform("identity"))
    actual = [p + 0.001 for p in exp]
    ok, problems, metrics = tm.check_against_expected(actual, exp)
    assert ok is True, problems
    assert metrics["max_interval_error"] <= tm.INTERVAL_TOLERANCE_S


def test_check_blocks_middle_interval_error_even_if_last_matches():
    """T13：末点相同、中间错 40ms —— 必须失败。"""
    exp = tm.expected_pts(SRC_1S, tm.Transform("identity"))
    bad = list(exp)
    bad[10:] = [p + 0.04 for p in bad[10:]]
    bad[-1] = exp[-1]
    ok, problems, metrics = tm.check_against_expected(bad, exp)
    assert ok is False, "中间间隔错 40ms 不能被末帧掩盖"
    assert metrics["max_interval_error"] > tm.INTERVAL_TOLERANCE_S


def test_check_accepts_a_correctly_speed_changed_candidate():
    """2× 变速的候选：与**变速后的期望**比 → 通过（不是与源比）。"""
    t = tm.Transform("speed", 2.0)
    exp = tm.expected_pts(SRC_1S, t)
    ok, problems, _ = tm.check_against_expected(exp, exp)
    assert ok is True, problems


def test_check_rejects_a_candidate_still_at_source_speed():
    """T12 的另一半：把源速度的结果当 2× 交付 → 必须被识别出来。"""
    t = tm.Transform("speed", 2.0)
    exp = tm.expected_pts(SRC_1S, t)
    ok, problems, _ = tm.check_against_expected(SRC_1S, exp)
    assert ok is False
    assert any("duration" in p for p in problems), problems


# ------------------------------------------------ 映射描述

def test_describe_mapping_has_the_fields_the_audit_asked_for():
    d = tm.describe_mapping(SRC_1S, tm.Transform("speed", 2.0))
    for key in ("transform", "source_frames", "expected_frames",
                "first_pts", "last_pts", "last_duration_s",
                "median_interval_s"):
        assert key in d, f"缺少 {key}"
    assert d["source_frames"] == 25
    assert d["expected_frames"] == 25
    assert d["last_duration_s"] is not None
