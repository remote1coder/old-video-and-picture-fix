# -*- coding: utf-8 -*-
"""R20-B05：时间容差不得随片长膨胀，尾时长须守住（审核 R19-B05）。

审核给的四个受控反例（旧代码全部 `ok=true`）：

| 输入 | 旧结果 | 问题 |
|---|---|---|
| 6001 帧、40ms 间隔、平移 10 秒 | ok=true | 原点误差被 `(N−1)×2×tb` = 12 秒容差放过 |
| 同样 6001 帧，每帧 41ms（末端慢 6 秒） | ok=true | 末端累计误差同样被放过 |
| 3 帧 double + 显式 `last_duration=.2` | 末场 .10 | 应 .18：显式尾时长优先于"前一间隔推断" |
| 显式容差传 NaN | 通过 | 参数本身未作有限/非负校验 |

**核心原则**：容器时间戳是量化的，但**原点误差与全片速度误差不会被量化放大成数秒**。
逐点容差只能是"一个量化步长"的量级，不能乘以帧数。
"""
import math
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "shared_modules") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "shared_modules"))

from time_mapping import (  # noqa: E402
    check_against_expected, expected_pts, Transform,
)

N = 6001
EXPECTED = [i * 0.040 for i in range(N)]
TB = 0.001          # 1ms 时间基


# ------------------------------------------------ 反例 1：全片平移

def test_whole_clip_shift_of_10s_is_rejected():
    """6001 帧平移 10 秒 —— 旧代码 12 秒容差放过。"""
    actual = [x + 10.0 for x in EXPECTED]
    ok, probs, m = check_against_expected(actual, EXPECTED, time_base=TB)
    assert ok is False, f"平移 10 秒被判通过；容差={m.get('last_tolerance_s')}"
    assert any("origin" in p for p in probs), probs


def test_origin_shift_allowed_only_when_declared():
    """显式声明 allow_origin_shift=True 才允许扣原点（审核：不能自动扣除）。"""
    actual = [x + 10.0 for x in EXPECTED]
    ok, probs, _ = check_against_expected(
        actual, EXPECTED, time_base=TB, allow_origin_shift=True)
    # 扣掉原点后逐点仍一致 → 通过
    assert ok is True, probs


# ------------------------------------------------ 反例 2：累计漂移

def test_accumulated_drift_6s_is_rejected():
    """每帧 41ms（末端慢 6 秒）—— 旧代码同样放过。"""
    actual = [i * 0.041 for i in range(N)]
    ok, probs, m = check_against_expected(actual, EXPECTED, time_base=TB)
    assert ok is False, f"累计漂移 6 秒被判通过；容差={m.get('last_tolerance_s')}"
    assert any("duration" in p for p in probs), probs


def test_tolerance_does_not_scale_with_frame_count():
    """容差本身不能随帧数增长（这是根因）。"""
    _, _, m_short = check_against_expected(
        EXPECTED[:10], EXPECTED[:10], time_base=TB)
    _, _, m_long = check_against_expected(EXPECTED, EXPECTED, time_base=TB)
    assert m_short["interval_tolerance_s"] == m_long["interval_tolerance_s"]
    assert m_short["last_tolerance_s"] == m_long["last_tolerance_s"], (
        f"跨度容差随帧数增长：10 帧 {m_short['last_tolerance_s']} vs "
        f"6001 帧 {m_long['last_tolerance_s']}")


# ------------------------------------------------ 反例 3：NaN / 非法容差

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.001, "abc"])
def test_invalid_tolerance_is_rejected(bad):
    """显式容差必须是有限非负数；否则一切都能通过。"""
    ok, probs, _ = check_against_expected(
        EXPECTED[:100], EXPECTED[:100], time_base=TB, interval_tolerance=bad)
    assert ok is False, f"非法容差 {bad!r} 未被拒绝"
    assert any("finite" in p or "not a number" in p for p in probs), probs


# ------------------------------------------------ 反例 4：double 尾时长

def test_double_uses_explicit_last_duration():
    """3 帧 + last_duration=.2 → 末场 .18（不是按前一间隔的 .10）。"""
    t = Transform(kind="double", value=2.0)
    out = expected_pts([0.0, 0.04, 0.08], t, last_duration=0.2)
    assert len(out) == 6, out
    assert abs(out[-1] - 0.18) < 1e-9, f"末场 {out[-1]}，应为 0.18"


def test_double_falls_back_to_interval_inference():
    """不给 last_duration 时仍按前一间隔推断（并应由 describe_mapping 标 estimated）。"""
    t = Transform(kind="double", value=2.0)
    out = expected_pts([0.0, 0.04, 0.08], t)
    assert abs(out[-1] - 0.10) < 1e-9, out


def test_double_rejects_bad_last_duration():
    t = Transform(kind="double", value=2.0)
    for bad in (float("nan"), 0.0, -0.1):
        with pytest.raises(ValueError):
            expected_pts([0.0, 0.04, 0.08], t, last_duration=bad)


# ------------------------------------------------ 正例不能误杀

def test_exact_match_passes():
    ok, probs, _ = check_against_expected(EXPECTED, EXPECTED, time_base=TB)
    assert ok is True, probs


def test_30fps_ms_quantisation_passes():
    """30fps 的 33.333ms 在 1ms 时间基上被舍成 33/34 交替 —— 合法量化，必须通过。"""
    # 期望：精确有理数 1/30；实际：量化到整数毫秒后再转回秒
    exp = [i / 30.0 for i in range(300)]
    act = [round(i * 1000 / 30) / 1000.0 for i in range(300)]
    ok, probs, _ = check_against_expected(act, exp, time_base=TB)
    assert ok is True, probs


def test_single_frame_ms_jitter_passes():
    """单帧在 1ms 时间基下的合法抖动不能误杀。"""
    ok, probs, _ = check_against_expected([0.001], [0.0], time_base=TB)
    assert ok is True, probs
