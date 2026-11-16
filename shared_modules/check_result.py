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

**``status`` 与 ``severity`` 是两件事（R14-04）**

``status`` 回答"结论是什么"，``severity`` 回答"这件事有多严重"。
把它们塞进一个字段就会出现本轮审核复现的矛盾：同步在 WARN 档
（120ms）**返回 0 允许出片**，但结果被记成 ``FAIL``，
汇总得到 ``publish=false`` —— 同一次运行两个相反的结论。

所以：

- ``severity="warning"`` + ``status=PASS`` → 允许出片，报告保留警告；
- ``severity="error"`` + ``status=FAIL`` → 阻断（除非该项有对应授权）；
- ``severity="info"`` → 正常。

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
        required: 是否为**必需**检查（R21-C04）。必需检查"没能给出结论"
            （``UNVERIFIED``）时不得进入正常发布，除非用户显式强制；
            可选的质量提示仍然只降 ``verified``，不阻断。
    """

    __slots__ = ("name", "status", "reason", "method", "subject_digest",
                 "forced", "force_key", "hard_error", "evidence", "severity",
                 "required")

    def __init__(self, name: str, status: str, reason: str = "",
                 method: str = "", subject_digest: Optional[str] = None,
                 forced: bool = False, force_key: Optional[str] = None,
                 hard_error: bool = False,
                 evidence: Optional[Dict[str, Any]] = None,
                 severity: str = "info", required: bool = False):
        if status not in ALL_STATUSES:
            raise ValueError(f"unknown check status: {status!r}")
        if severity not in ("info", "warning", "error"):
            raise ValueError(f"unknown severity: {severity!r}")
        self.severity = severity
        self.name = name
        self.status = status
        self.reason = reason
        self.method = method
        self.subject_digest = subject_digest
        self.forced = bool(forced)
        self.force_key = force_key
        self.hard_error = bool(hard_error)
        self.evidence = evidence or {}
        self.required = bool(required)

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

        R21-C04：**必需**项的 ``UNVERIFIED`` 也允许强制 —— 它表示"测不出来"，
        用户可能明知如此仍要出片（例如探针在某类素材上不可用）。
        强制后状态仍是 UNVERIFIED，报告里留痕，``verified`` 保持 False。
        非必需项不需要强制：它们本来就不阻断。
        """
        if self.hard_error:
            return False
        if self.status == FAIL:
            return True
        return self.status == UNVERIFIED and self.required

    def effective_status(self) -> str:
        """强制后的**记录用**状态：仍然是 FAIL，只是加了 forced 标记。

        强制不放宽事实：报告里必须留着 FAIL 和用户的选择，
        不能因为"用户同意了"就把结论改成 PASS。
        """
        return self.status

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "status": self.status,
                             "reason": self.reason, "method": self.method,
                             "severity": self.severity}
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
        if self.required:
            d["required"] = True
        return d

    def __repr__(self) -> str:      # pragma: no cover - 调试用
        return (f"CheckResult({self.name!r}, {self.status}"
                f"{', forced' if self.forced else ''}"
                f"{', hard' if self.hard_error else ''})")


def publish_decision(results: Iterable[CheckResult],
                     force_enabled: bool = False,
                     enabled_forces: Optional[Iterable[str]] = None
                     ) -> Dict[str, Any]:
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
        force_enabled: 粗粒度强制开关（保留给旧调用方）。
        enabled_forces: 用户**实际打开**的开关名集合（如
            ``{"allow_sync_fail_publish"}``）。给出时按检查项自己的
            ``force_key`` 匹配 —— sync 的授权不能替代 verify 的授权。
            为 None 时退回 ``force_enabled``。

    Returns:
        ``{"publish": bool, "verified": bool, "blocked_by": [names],
        "forced": bool, "reasons": {name: reason}}``
    """
    rs: List[CheckResult] = list(results)
    blocked = [r.name for r in rs if r.status == FAIL and r.hard_error]
    fails = [r for r in rs if r.status == FAIL and not r.hard_error]
    unver = [r for r in rs if r.status == UNVERIFIED]
    passes = [r for r in rs if r.status == PASS]
    warn = [r.name for r in rs if r.severity == "warning"]

    def _force_ok(r: CheckResult) -> bool:
        """这一项**自己**的授权是否打开。

        R14-04：sync 的 `allow_sync_fail_publish` 不能替代 verify 的
        `allow_verify_fail_publish`。没给 `enabled_forces` 时才退回粗粒度开关。
        """
        if not r.is_forceable:
            return False
        if enabled_forces is None:
            return bool(force_enabled)
        key = r.force_key
        if not key:
            return bool(force_enabled)
        return key in set(enabled_forces)

    reasons = {r.name: r.reason for r in rs if r.reason}
    # ---- R26-A01 + 指令 §1.1：`verified` 只要求**必需项** ----
    #
    # 原来是 `not fails and not unver` —— 任何一项 UNVERIFIED（**包括可选**）
    # 都会让 verified=False。后果：静态画面让帧序方法失效（合法的
    # "方法无判别力"），整代就永远不可能 verified，于是要么永远走诊断，
    # 要么有人去放宽发布门。
    #
    # 指令原话：`verified=True` 表示**本次计划的必需证据全部满足**、
    # 没有阻断性错误；它**不要求每一种可选启发式都得出肯定结论**。
    # 所以这里只看 required 的项。
    #
    # `severity == "warning"` 的项同样不参与 verified 计算：
    # 生产者自己把它标成警告，消费方不能把它升级成硬门（R26-A02）。
    def _hard(r: "CheckResult") -> bool:
        """这一项是不是**硬门**。

        R26-A02：判据用 `severity`，不用 `required`。
        `required` 的默认值是 `False`，所以"没声明"和"明确声明可选"
        长得一样 —— 拿它判会把未声明的 FAIL 放过（审核的
        `test_quality_fail_without_force_blocks` 就是这样）。

        `severity="warning"` 是生产者**明确**声明"这不是硬门"；
        其余（`info` / `error` / 未声明）一律按硬门 —— **fail-closed**。
        """
        # `required=True` 一律是硬门，哪怕生产者把它标成 warning
        # （`timeline_mapping` 就是这样：它必需，但"未验证"不是错误级）。
        if getattr(r, "required", False) is True:
            return True
        # 未声明必需性时 fail-closed：只有**明确**标成 warning 的才放过。
        return str(getattr(r, "severity", "") or "").lower() != "warning"

    _req_fail = [r for r in fails if _hard(r)]
    _req_unver = [r for r in unver if _hard(r)]

    decision = {
        "publish": True,
        "verified": bool(passes) and not _req_fail and not _req_unver
                    and not blocked,
        "blocked_by": blocked,
        "forced": False,
        "reasons": reasons,
        # 警告不阻断，但必须跟着结论走，不能只在日志里
        "warnings": warn,
        "forced_checks": [],
        # 审计用：哪些项让 verified 落下来（可选 UNVERIFIED 不算）
        "unverified_required": [r.name for r in _req_unver],
        "unverified_optional": [r.name for r in unver if not _hard(r)],
    }

    if blocked:
        decision["publish"] = False
        return decision

    # ``UNVERIFIED`` 表示"这次没能给出结论"（方法无判别力、输入不可用等）。
    # 它**不是通过**，但也不等于缺陷：静态画面让帧序方法失效，不该因此
    # 判整个作业失败。所以**可选的** UNVERIFIED 只记状态，不阻断、也不降
    # verified（R26-A01）；**必需的** UNVERIFIED 阻断（R21-C04）。
    if fails:
        # R26-A02：`severity="warning"` 的 FAIL 是生产者自己标的警告，
        # 不是硬门。原来它照样 `publish=False`，与 producer 的意图相反。
        _hard_fails = [r for r in fails if _hard(r)]
        _warn_fails = [r for r in fails if not _hard(r)]
        if _warn_fails:
            decision["warnings"] = (list(decision.get("warnings") or [])
                                    + [r.name for r in _warn_fails])
        if _hard_fails:
            not_forced = [r for r in _hard_fails if not _force_ok(r)]
            if not_forced:
                decision["publish"] = False
                decision["blocked_by"] = [r.name for r in not_forced]
                return decision
            decision["forced"] = True
            decision["forced_checks"] = [r.name for r in _hard_fails]
            decision["verified"] = False
    # ---- R21-C04：**必需**检查的 UNVERIFIED 必须阻断 ----
    # 审核反例：只给 `final_timeline=UNVERIFIED` → publish=true。
    # "测不出来"于是等于"没问题"。必需性由检查项自己声明：
    #   * 完整性/时间等必需门：UNVERIFIED 不可发布（可被 force_key 强制，
    #     强制后 verified=False 且记录在 forced_checks 里）；
    #   * 可选的主观质量/低置信启发式：只记状态，不阻断也不降 verified。
    _unver_required = [r for r in unver if _hard(r)]
    if _unver_required:
        _not_forced = [r for r in _unver_required if not _force_ok(r)]
        if _not_forced:
            decision["publish"] = False
            decision["blocked_by"] = [r.name for r in _not_forced]
            decision["verified"] = False
            return decision
        decision["forced"] = True
        decision["forced_checks"] = ([r.name for r in _unver_required]
                                     + decision["forced_checks"])
        decision["verified"] = False
    # R26-A01：**不再**因为存在可选 UNVERIFIED 就把 verified 置 False。
    # verified 的定义是"必需证据全部满足"，不是"所有启发式都给了肯定答案"。
    return decision
