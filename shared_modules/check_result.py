# -*- coding: utf-8 -*-
"""统一的检查结果（CheckResult）与发布决策。

**R13-07：未知、跳过、强制与通过必须分开。**

当前代码把它们混在一起，具体有三处（审核逐条指出）：

1. **帧序检查返回 None**（方法无判别力）时，调用方仍可能记成 `passed`；
2. **同卷改名跳过后置核对**时，构造出 `ok=True, checked=0` 并记"帧序已验证"
   —— 那等于**凭空造了一次检查**；
3. **机械门禁止强制的硬错误**，在全链/同步门仍可能从另一个 force 入口绕过。

本模块只做两件事：给检查一个**明确的四态**，以及一处**可测试的发布决策**。
它不是工作流框架，也不替代各检查器自己的判断。

四态语义
--------
================  ====================================================
``PASS``          检查执行了，结论是通过
``FAIL``          检查执行了，结论是不通过
``UNVERIFIED``    检查**没能给出结论**（输入不可用、方法无判别力、
                  检查器抛异常）。**不是通过。**
``NOT_APPLICABLE``本代不需要做这项检查（例如同卷改名、功能显式关闭）
================  ====================================================

``NOT_APPLICABLE`` 与 ``PASS`` 的区别是审计的核心要求：
"同卷 rename 只复用候选件的检查证据，不构造新 PASS/checked=0 冒充新检查"。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

PASS = "PASS"
FAIL = "FAIL"
UNVERIFIED = "UNVERIFIED"
NOT_APPLICABLE = "NOT_APPLICABLE"

ALL_STATUSES = (PASS, FAIL, UNVERIFIED, NOT_APPLICABLE)


class CheckResult:
    """一项检查的结论 + 它凭什么得出这个结论。

    Attributes:
        name: 检查名（``sync`` / ``frame_order`` / ``mechanical`` …）。
        status: 四态之一。
        reason: 人可读的原因；``UNVERIFIED`` / ``FAIL`` 必须写。
        method: 用什么方法得出的（``artifact_verify.verify_output`` 等）。
        subject_digest: 被检查对象的 SHA-256（绑身份，避免"核 B 写 A"）。
        forced: 是否被用户强制放行。
        force_key: 用的是哪个强制开关。
        hard_error: 是否为"硬错误"——输入不可消费、候选无法解码、
            身份摘要无法核对、事务准备失败。**这类永远不可强制。**
        evidence: 结构化证据（采样点、实测值等），便于事后复核。
    """

    __slots__ = ("name", "status", "reason", "method", "subject_digest",
                 "forced", "force_key", "hard_error", "evidence")

    def __init__(self, name: str, status: str, reason: str = "",
                 method: str = "", subject_digest: Optional[str] = None,
                 forced: bool = False, force_key: Optional[str] = None,
                 hard_error: bool = False,
                 evidence: Optional[Dict[str, Any]] = None):
        if status not in ALL_STATUSES:
            raise ValueError(f"unknown check status: {status!r}")
        self.name = name
        self.status = status
        self.reason = reason
        self.method = method
        self.subject_digest = subject_digest
        self.forced = bool(forced)
        self.force_key = force_key
        self.hard_error = bool(hard_error)
        self.evidence = evidence or {}

    # -- 语义 --------------------------------------------------------------
    @property
    def is_pass(self) -> bool:
        """只有 PASS 才是"验过了并通过"。

        ``NOT_APPLICABLE`` 不是 PASS —— 它表示"这次没做检查"。
        """
        return self.status == PASS

    @property
    def is_forceable(self) -> bool:
        """能否被用户的强制开关放行。

        硬错误（输入不可消费/候选不可解码/摘要不可核对/事务准备失败）
        永远不可强制——用户不能靠一个开关让不存在的能力成功。
        """
        return (self.status == FAIL) and not self.hard_error

    def effective_status(self) -> str:
        """强制后的**记录用**状态：仍然是 FAIL，只是加了 forced 标记。

        强制不放宽事实：报告里必须留着 FAIL 和用户的选择，
        不能因为"用户同意了"就把结论改成 PASS。
        """
        return self.status

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "status": self.status,
                             "reason": self.reason, "method": self.method}
        if self.subject_digest:
            d["subject_digest"] = self.subject_digest
        if self.forced:
            d["forced"] = True
        if self.force_key:
            d["force_key"] = self.force_key
        if self.hard_error:
            d["hard_error"] = True
        if self.evidence:
            d["evidence"] = self.evidence
        return d

    def __repr__(self) -> str:      # pragma: no cover - 调试用
        return (f"CheckResult({self.name!r}, {self.status}"
                f"{', forced' if self.forced else ''}"
                f"{', hard' if self.hard_error else ''})")


def publish_decision(results: Iterable[CheckResult],
                     force_enabled: bool = False) -> Dict[str, Any]:
    """由一组检查结果给出**唯一**的发布决定。

    规则（对应审核 §4.4）：

    - 有 ``FAIL`` 且 ``hard_error=True`` → 拒绝，强制也拒绝；
    - 有 ``FAIL`` 且允许强制 → 允许发布，但 ``verified=False``；
    - 有 ``FAIL`` 且不允许强制 → 拒绝；
    - 有 ``UNVERIFIED`` → 不能当作通过；需要显式强制才发布，
      且 ``verified=False``；
    - 全是 ``PASS`` / ``NOT_APPLICABLE`` → 发布，``verified`` 仅当
      至少有一项 PASS（全 NOT_APPLICABLE 时是"未验证"而非"已验证"）。

    Args:
        results: 各检查的结论。
        force_enabled: 用户是否打开了强制开关（``allow_*_fail_publish``）。

    Returns:
        ``{"publish": bool, "verified": bool, "blocked_by": [names],
        "forced": bool, "reasons": {name: reason}}``
    """
    rs: List[CheckResult] = list(results)
    blocked = [r.name for r in rs if r.status == FAIL and r.hard_error]
    fails = [r for r in rs if r.status == FAIL and not r.hard_error]
    unver = [r for r in rs if r.status == UNVERIFIED]
    passes = [r for r in rs if r.status == PASS]

    reasons = {r.name: r.reason for r in rs if r.reason}
    decision = {
        "publish": True,
        "verified": bool(passes) and not fails and not unver and not blocked,
        "blocked_by": blocked,
        "forced": False,
        "reasons": reasons,
    }

    if blocked:
        decision["publish"] = False
        return decision

    # ``UNVERIFIED`` 表示"这次没能给出结论"（方法无判别力、输入不可用等）。
    # 它**不是通过**，但也不等于缺陷：静态画面让帧序方法失效，不该因此
    # 判整个作业失败。所以它只让 verified=False，不阻断发布。
    # （是否允许输出"未验证"的产物属于 profile 策略；默认允许并如实标注。）
    if fails:
        if not force_enabled:
            decision["publish"] = False
            decision["blocked_by"] = [r.name for r in fails]
            return decision
        decision["forced"] = True
        decision["verified"] = False
    if unver:
        decision["verified"] = False
    return decision
