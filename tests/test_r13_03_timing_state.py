# -*- coding: utf-8 -*-
"""R13-03：日志说"估算"，媒体却可能写 verified。

**缺陷**

`_timing_basis` 原来只赋值、**从不参与任何判断**；`_timing_state` 的默认值
就是 `"verified"`。于是：

- packet 测量失败 → 退回帧数比例估算 → 估算出正数 → 状态仍是 `verified`
  → `timing_state=verified` 被写进成品元数据；
- 上游声明的变速碰巧等于**估算**比值时，代码会拿这个"吻合"当独立验证，
  直接按它去改音频 atempo。

而帧数比例只在"帧间隔恒定"时成立：同帧数变速、倍帧不变时长都会让它给错值。

**修法**：决策逻辑抽成 `_decide_timing()` 纯函数，`timing_basis` 参与判断——
非实测（`frame-ratio` / `unknown`）永远不能得到 `verified`，也不能作为
改音频的依据。
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

MEASURED = "packet-span"
ESTIMATED = "frame-ratio (estimated)"
UNKNOWN = "unknown"


class _Log:
    def __init__(self):
        self.events = []

    def _r(self, ev):
        self.events.append(ev)

    def info(self, ev, msg="", **k):
        self._r(ev)

    def warning(self, ev, msg="", **k):
        self._r(ev)

    def error(self, ev, msg="", **k):
        self._r(ev)

    def has(self, ev):
        return ev in self.events


# ------------------------------------------------ 第 5 项：时长关系

def test_unchanged_duration_with_measurement_is_verified():
    """实测且时长不变（250 帧 10s → 仍 10s）→ verified。"""
    ratio, state = s7._decide_timing(10.0, 10.0, MEASURED, None, None)
    assert (ratio, state) == (1.0, "verified")


def test_declared_speed_with_measurement_applies_atempo():
    """实测且声明 2 倍速、实测比值也对得上 → adjusted 且 tempo=2。"""
    ratio, state = s7._decide_timing(10.0, 5.0, MEASURED, 2.0, None)
    assert state == "adjusted", state
    assert ratio == pytest.approx(2.0)


def test_duration_change_without_declaration_is_unexplained():
    """时长变了却没人声明 → unexplained，不能偷偷拉音频。"""
    ratio, state = s7._decide_timing(10.0, 7.0, MEASURED, None, None)
    assert state == "unexplained"
    assert ratio == 1.0


def test_declared_but_mismatched_is_unexplained():
    """声明 2 倍但实测 1.2 倍 → 声明不可信，不改音频。"""
    ratio, state = s7._decide_timing(10.0, 8.33, MEASURED, 2.0, None)
    assert state == "unexplained"
    assert ratio == 1.0


# ------------------------------------------------ 第 6 项：估算不得 verified

def test_estimated_basis_never_becomes_verified():
    """**核心**：估算（帧比例）即使时长完全吻合，也只能是 unverified。"""
    ratio, state = s7._decide_timing(10.0, 10.0, ESTIMATED, None, None)
    assert state == "unverified", \
        f"估算不能标 verified，实际 {state}"
    assert ratio == 1.0


def test_unknown_basis_never_becomes_verified():
    """packet 完全不可读、连估算都没有 → unverified。"""
    ratio, state = s7._decide_timing(10.0, 10.0, UNKNOWN, None, None)
    assert state == "unverified"


def test_missing_duration_is_unverified():
    """时长量不出 → unverified（原实现就是这个分支，锁住不要退化）。"""
    assert s7._decide_timing(0.0, 10.0, MEASURED, None, None) == (1.0, "unverified")
    assert s7._decide_timing(10.0, 0.0, MEASURED, None, None) == (1.0, "unverified")


# ------------------------------------------------ 第 7 项：声明吻合不等于验证

def test_declared_matching_estimate_is_not_independent_evidence():
    """声明值碰巧等于估算比值 → 仍 unverified，且**不许**据此改音频。

    这是审核点名的场景：估算 2.0、声明也是 2.0，看起来"对上了"，
    但估算本身来自"帧数比例"，不是时间轴测量。
    """
    log = _Log()
    ratio, state = s7._decide_timing(10.0, 5.0, ESTIMATED, 2.0, None, log)
    assert state == "unverified", state
    assert ratio == 1.0, "不允许用估算吻合去改 atempo"
    assert log.has("STAGE7_TIMING_DECLARED_BUT_UNMEASURED"), log.events


def test_declared_matching_estimate_via_tempo_key():
    """同一个保护也要覆盖 expected_tempo 入口。"""
    ratio, state = s7._decide_timing(10.0, 5.0, ESTIMATED, None, 2.0)
    assert (ratio, state) == (1.0, "unverified")


# ------------------------------------------------ 第 8 项：末帧/偏移不混淆

def test_nonzero_source_offset_is_not_a_failure():
    """源片自带偏移但时长一致 → 不应被判 unexplained。

    （这里只验证时长维度；音画偏移由 _validate_av_sync 的逐分量判据负责。）
    """
    ratio, state = s7._decide_timing(10.0, 10.0, MEASURED, None, None)
    assert state == "verified"
    assert ratio == 1.0


def test_basis_is_written_to_output_metadata():
    """timing_basis 必须随成品落地，便于事后判断结论依据。"""
    src = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                       "stage7_encode.py")
    text = open(src, encoding="utf-8", errors="replace").read()
    assert 'f"timing_basis={_timing_basis}"' in text, \
        "元数据里要能看出这个结论是实测还是估算"
