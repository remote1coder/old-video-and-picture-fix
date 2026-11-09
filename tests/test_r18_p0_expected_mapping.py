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
    """倍帧（QTGMC 50i→50p）：帧数翻倍、间隔减半。

    **2026-11-09 修正**：这里原来断言 ``len(out) == 5``，把 2N−1 的缺陷
    固化成了规格（审核 R18-A05 原文："新增测试直接断言 3→5，固化了缺陷"）。

    正确的倍帧是每帧两个场，N 帧得 **2N** 个时刻；
    末帧的第二场由前一间隔推断（`last_duration_basis` 会记录来源）。
    3 帧 @0.04s → 0, .02, .04, .06, .08, .10（6 个）。
    """
    src = [0.0, 0.04, 0.08]
    out = tm.expected_pts(src, tm.Transform("double"))
    assert len(out) == 6, out
    assert out[1] - out[0] == pytest.approx(0.02, abs=1e-9)
    assert out[-1] == pytest.approx(0.10, abs=1e-9)


def test_decimate_drops_one_frame_per_cycle():
    """抽帧（IVTC cycle=5 去 1 帧）。"""
    src = [i * 0.04 for i in range(10)]
    out = tm.expected_pts(src, tm.Transform("decimate", 5))
    assert len(out) == 8, out
    assert 0.16 not in out, "cycle=5 时每 5 帧的第 5 帧应被丢掉"


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        tm.expected_pts(SRC, tm.Transform("warp", 1.0))


def test_single_frame_is_rejected_only_when_the_transform_needs_an_interval():
    """单帧不再一律拒绝（审核 R18-A05 修正）。

    原断言是"单帧一律 raise"，等于把"不足两帧就什么都不支持"写成了规格。
    审核要求：单帧按**明确 duration** 处理，缺 duration 才记录未知。

    - `identity`：合法，返回原时间轴；
    - `double`：需要显式 `last_duration`，缺了就拒绝并说明缺什么。
    """
    assert tm.expected_pts([0.0], tm.Transform("identity")) == [0.0]
    with pytest.raises(ValueError) as ei:
        tm.expected_pts([0.0], tm.Transform("double"))
    assert "duration" in str(ei.value).lower()
    # 给了 duration 就应当能做
    assert len(tm.expected_pts([0.0], tm.Transform("double"),
                               last_duration=0.04)) == 2


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
    """末点相同、中间间隔错 —— 必须失败。

    **2026-11-09 修正**：原构造是
    ``bad[10:] = [p + 0.04 ...]; bad[-1] = exp[-1]``，这在**等间隔**源上
    必然造成非单调（总和必须相等，补回来就会出现 ≤0 的间隔）。
    新的严格单调检查会先拒绝它 —— 那本身是正确行为，但会让这个用例
    测不到"间隔判据"。

    这里换成**单调且末点相同**的构造，单独验证间隔判据：
    前两个间隔拉长、后两个压短，总和不变。
    """
    exp = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20]
    actual = [0.0, 0.06, 0.12, 0.16, 0.18, 0.20]

    ok, problems, metrics = tm.check_against_expected(
        actual, exp, time_base=0.001)
    assert ok is False, f"中间间隔错 20ms 必须失败，实际 {problems}"
    assert metrics["max_interval_error"] > metrics["interval_tolerance_s"], metrics


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
