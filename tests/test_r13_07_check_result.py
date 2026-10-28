# -*- coding: utf-8 -*-
"""R13-07：未知、跳过、强制与通过必须分开。

审核指出的三处混淆：

1. 帧序检查返回 ``None``（方法无判别力）时，前置路径仍可能记 ``passed``；
2. 同卷改名**跳过**后置核对，却构造 ``ok=True, checked=0`` 并记
   "帧序已验证" —— 等于凭空造了一次检查；
3. 机械门禁止强制的硬错误，在全链/同步门仍可能从另一 force 入口绕过。

本文件验证 `check_result` 的四态语义与发布决策；
具体门接入的断言在各门自己的测试里。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check_result as cr  # noqa: E402


# ---------------------------------------------------------------- 四态语义

def test_four_statuses_are_distinct():
    assert cr.PASS != cr.FAIL != cr.UNVERIFIED != cr.NOT_APPLICABLE
    for s in (cr.PASS, cr.FAIL, cr.UNVERIFIED, cr.NOT_APPLICABLE):
        r = cr.CheckResult("x", s)
        assert r.status == s


def test_unknown_status_is_rejected():
    """写错状态必须立刻报错，不能静默当成某个值。"""
    with pytest.raises(ValueError):
        cr.CheckResult("x", "UNKNOWN")
    with pytest.raises(ValueError):
        cr.CheckResult("x", "ok")       # 旧代码里的 ok=True 语义不能混进来


def test_not_applicable_is_not_pass():
    """**核心**：同卷改名跳过 ≠ 验过。

    旧实现构造 `{"ok": True, "skipped": "same-volume rename", "checked": 0}`
    并记 "Frame order verified"，等于把"没做"写成"做了且通过"。
    """
    r = cr.CheckResult("frame_order", cr.NOT_APPLICABLE,
                       reason="same-volume rename reuses the candidate bytes")
    assert r.is_pass is False
    assert r.status == cr.NOT_APPLICABLE


def test_unverified_is_not_pass():
    """方法无判别力（帧序返回 None）→ UNVERIFIED，不是 PASS。"""
    r = cr.CheckResult("frame_order", cr.UNVERIFIED,
                       reason="static frames give no discriminative power")
    assert r.is_pass is False


# ---------------------------------------------------------------- 强制语义

def test_hard_error_is_never_forceable():
    """输入不可消费 / 候选不可解码 / 摘要不可核对 → 强制也不放行。"""
    r = cr.CheckResult("mechanical", cr.FAIL, reason="source unreadable",
                       hard_error=True)
    assert r.is_forceable is False

    d = cr.publish_decision([r], force_enabled=True)
    assert d["publish"] is False
    assert "mechanical" in d["blocked_by"]


def test_quality_fail_is_forceable_but_stays_fail():
    """质量 FAIL 可被强制，但结论**仍然是 FAIL**，不许改写。"""
    r = cr.CheckResult("mechanical", cr.FAIL, reason="duration +3.7%")
    assert r.is_forceable is True
    assert r.effective_status() == cr.FAIL, "强制不得把 FAIL 改成 PASS"

    d = cr.publish_decision([r], force_enabled=True)
    assert d["publish"] is True
    assert d["verified"] is False
    assert d["forced"] is True


def test_quality_fail_without_force_blocks():
    r = cr.CheckResult("sync", cr.FAIL, reason="audio leads 250ms")
    d = cr.publish_decision([r], force_enabled=False)
    assert d["publish"] is False


def test_unverified_publishes_but_is_not_verified():
    """方法无判别力 → 可以出片，但**结论是"未验证"**。

    依据审核："静态画面导致帧序方法无判别力，并不等于帧序错乱；
    记录方法限制。" 所以不阻断，也绝不算通过。
    """
    r = cr.CheckResult("frame_order", cr.UNVERIFIED, reason="no verdict")
    d = cr.publish_decision([r], force_enabled=False)
    assert d["publish"] is True, "方法局限不该判整个作业失败"
    assert d["verified"] is False, "没能给出结论就不是验证通过"


# ---------------------------------------------------------------- 汇总

def test_all_pass_is_verified():
    rs = [cr.CheckResult("mechanical", cr.PASS),
          cr.CheckResult("sync", cr.PASS),
          cr.CheckResult("frame_order", cr.PASS)]
    d = cr.publish_decision(rs, force_enabled=False)
    assert d["publish"] is True
    assert d["verified"] is True


def test_all_not_applicable_is_not_verified():
    """全都没做 → 可以发布（不需要检查），但**不能说验证过了**。"""
    rs = [cr.CheckResult("frame_order", cr.NOT_APPLICABLE, reason="skip")]
    d = cr.publish_decision(rs, force_enabled=False)
    assert d["publish"] is True
    assert d["verified"] is False, "没做任何检查不能算 verified"


def test_mixed_pass_and_not_applicable_is_verified():
    rs = [cr.CheckResult("mechanical", cr.PASS),
          cr.CheckResult("frame_order", cr.NOT_APPLICABLE, reason="same vol")]
    d = cr.publish_decision(rs, force_enabled=False)
    assert d["publish"] is True
    assert d["verified"] is True


def test_hard_error_beats_force_even_with_other_failures():
    """硬错误优先：同组里有可强制的质量失败也不能放行。"""
    rs = [cr.CheckResult("mechanical", cr.FAIL, reason="dur"),
          cr.CheckResult("sync", cr.FAIL, reason="unreadable", hard_error=True)]
    d = cr.publish_decision(rs, force_enabled=True)
    assert d["publish"] is False
    assert "sync" in d["blocked_by"]


def test_result_serialises_force_and_digest():
    """落盘时强制标记与对象摘要都要带上，事后能复核。"""
    r = cr.CheckResult("mechanical", cr.FAIL, reason="dur",
                       subject_digest="a" * 64, forced=True,
                       force_key="allow_verify_fail_publish")
    d = r.as_dict()
    assert d["subject_digest"] == "a" * 64
    assert d["forced"] is True
    assert d["force_key"] == "allow_verify_fail_publish"
    assert d["status"] == cr.FAIL
