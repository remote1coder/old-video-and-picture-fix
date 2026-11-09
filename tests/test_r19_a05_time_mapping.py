# -*- coding: utf-8 -*-
"""第十九轮 A5：时间映射的数学与合同（对应 R18-A05）。

审核指出的四个问题：

1. **`double` 只产生 2N−1 个 PTS**。3 帧得到 5 帧，最后一个场没有表达。
   更糟的是：上一轮的新增测试**直接断言 3→5**，把缺陷固化成规格。
2. `decimate` 固定每周期删最后一帧，不代表实际 `VDecimate` 的内容选择。
3. 比较函数**未验证有限数值与严格单调** —— 受控测试里 NaN、轻微倒序、
   整段偏移 10 秒都能通过。
4. 容差是固定的 20ms/50ms，与时间基、片长、帧率无关；
   于是"三帧错误变速"这类结构错误被容差包庇。

本文件用**目标签名**写测试（`expected_pts(..., last_duration=None)`、
`check_against_expected(..., time_base=None)`），在实现改好之前应当失败。
"""
import math
import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import time_mapping as tm  # noqa: E402


SRC_3 = [0.0, 0.04, 0.08]          # 3 帧 @ 25fps
SRC_25 = [i * 0.04 for i in range(25)]


# ------------------------------------------------ double 必须是 2N

def test_double_produces_two_times_per_input_frame():
    """**核心缺陷**：3 帧倍帧必须得到 **6** 个时刻，不是 5 个。

    审核原文：
    > `double` 对 N 帧只产生 2N−1 个 PTS：3 帧得到 5 帧，最后一个场没有表达。
    > 新增测试直接断言 3→5，固化了缺陷。

    物理含义：每帧的两场各占半个源间隔，末帧的第二场同样占半个间隔，
    所以时刻是 0, .02, .04, .06, .08, .10 —— 共 6 个。
    """
    out = tm.expected_pts(SRC_3, tm.Transform("double"))
    assert len(out) == 6, f"3 帧倍帧应得 6 个时刻，实际 {len(out)}: {out}"


def test_double_last_field_uses_previous_interval():
    """末帧的第二场时刻由**前一间隔**推断，不是凭空猜。

    审核："用末帧 duration 表达第二场，不允许 2N−1。"
    """
    out = tm.expected_pts(SRC_3, tm.Transform("double"))
    assert out[-1] == pytest.approx(0.10, abs=1e-9)
    # 全程间隔必须均匀（这是"倍帧"的定义，不是自由插值）
    iv = [b - a for a, b in zip(out, out[1:])]
    assert max(iv) - min(iv) < 1e-9, iv


def test_double_records_inferred_last_duration():
    """推断出的末帧 duration 必须可追溯，不能悄悄用。"""
    d = tm.describe_mapping(SRC_3, tm.Transform("double"))
    assert d["expected_frames"] == 6
    assert d.get("last_duration_basis") in (
        "inferred_from_previous_interval", "given"), d.get("last_duration_basis")


# ------------------------------------------------ 单帧输入

def test_single_frame_identity_is_allowed():
    """单帧 + 保时：合法，不该因为"不足两帧"就拒绝。"""
    out = tm.expected_pts([0.0], tm.Transform("identity"))
    assert out == [0.0]


def test_single_frame_double_needs_an_explicit_duration():
    """单帧倍帧：给了 duration 才做，否则**明确拒绝并说明缺什么**。"""
    with pytest.raises(ValueError) as ei:
        tm.expected_pts([0.0], tm.Transform("double"))
    assert "duration" in str(ei.value).lower()

    out = tm.expected_pts([0.0], tm.Transform("double"), last_duration=0.04)
    assert len(out) == 2, out
    assert out[1] == pytest.approx(0.02, abs=1e-9)


def test_single_frame_speed_is_allowed():
    assert tm.expected_pts([0.0], tm.Transform("speed", 2.0)) == [0.0]


# ------------------------------------------------ 非法数值必须拒绝

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_check_rejects_non_finite_values(bad):
    """NaN / Inf 绝不能被判为"通过"。

    修复前：`nan > tolerance` 是 False，于是 NaN 一路静默通过。
    """
    exp = [0.0, 0.04, 0.08]
    actual = [0.0, bad, 0.08]
    ok, problems, _ = tm.check_against_expected(actual, exp)
    assert ok is False, "NaN/Inf 必须拒绝"
    assert any("finite" in p.lower() or "nan" in p.lower()
               or "inf" in p.lower() for p in problems), problems


def test_check_rejects_non_monotonic_actual():
    """轻微倒序必须拒绝 —— 时间轴不是单调的就不是时间轴。"""
    exp = [0.0, 0.04, 0.08, 0.12]
    actual = [0.0, 0.04, 0.039, 0.12]
    ok, problems, _ = tm.check_against_expected(actual, exp)
    assert ok is False
    assert any("monoton" in p.lower() or "order" in p.lower()
               for p in problems), problems


def test_check_rejects_duplicate_pts():
    exp = [0.0, 0.04, 0.08, 0.12]
    actual = [0.0, 0.04, 0.04, 0.12]
    ok, problems, _ = tm.check_against_expected(actual, exp)
    assert ok is False


# ------------------------------------------------ 原点：默认必须一致

def test_check_rejects_whole_timeline_offset_by_default():
    """整段偏移 10 秒，默认必须拒绝。

    审核：独立比较时移除原点是**可以合理的**，但必须有显式的共同音画原点
    合同；不能默默移除后宣称同步正确。
    """
    exp = [0.0, 0.04, 0.08]
    actual = [p + 10.0 for p in exp]
    ok, problems, _ = tm.check_against_expected(actual, exp)
    assert ok is False, "默认不得忽略整段偏移"
    assert any("origin" in p.lower() for p in problems), problems


def test_check_allows_offset_only_when_explicitly_permitted():
    """显式允许时可以通过，但必须在 metrics 里留痕。"""
    exp = [0.0, 0.04, 0.08]
    actual = [p + 10.0 for p in exp]
    ok, problems, metrics = tm.check_against_expected(
        actual, exp, allow_origin_shift=True)
    assert ok is True, problems
    assert metrics.get("origin_shift_s") == pytest.approx(10.0, abs=1e-6)


# ------------------------------------------------ 容差必须从时间基推导

def test_tolerance_derives_from_time_base():
    """给了容器时间基，容差必须由它推导，而不是固定 20ms。"""
    exp = [i / 30.0 for i in range(30)]         # 33.333ms 间隔
    # 仅 1ms 量化抖动
    actual = [p + 0.001 for p in exp]
    ok, _p, metrics = tm.check_against_expected(
        actual, exp, time_base=0.001)
    assert ok is True
    assert metrics.get("interval_tolerance_s") == pytest.approx(
        2 * 0.001, abs=1e-9), metrics


def test_tolerance_from_time_base_rejects_structural_error():
    """三帧素材上的错误变速：固定 20ms 会放过，按时间基推导必须拒绝。"""
    exp = [0.0, 0.04, 0.08]
    # 中间一帧错 15ms —— 小于旧容差 20ms，但远大于 2×1ms
    actual = [0.0, 0.055, 0.08]
    ok, problems, metrics = tm.check_against_expected(
        actual, exp, time_base=0.001)
    assert ok is False, f"15ms 结构错误必须拒绝，metrics={metrics}"
    assert metrics["max_interval_error"] > metrics["interval_tolerance_s"]


def test_tolerance_default_is_documented_not_silent():
    """没给时间基时仍可用，但必须标明容差来源是默认值。"""
    exp = [0.0, 0.04, 0.08]
    ok, _p, metrics = tm.check_against_expected(exp, exp)
    assert ok is True
    assert metrics.get("interval_tolerance_source") == "default_fixed"


# ------------------------------------------------ decimate 的模型标签

def test_decimate_is_labelled_as_idealised_model():
    """`decimate` 只是理想模型（固定删每周期最后一帧），必须标出来。

    审核："不能固定删每组最后一帧冒充 VDecimate"。
    真实 VDecimate 的选择由内容决定，这里不做假设，但必须让下游看得见
    这是一个**模型**而不是实测。
    """
    d = tm.describe_mapping([i * 0.04 for i in range(10)],
                            tm.Transform("decimate", 5))
    assert d["model"] == "idealised"


def test_identity_and_speed_are_exact_models():
    d = tm.describe_mapping(SRC_3, tm.Transform("identity"))
    assert d["model"] == "exact"
