# -*- coding: utf-8 -*-
"""发布/检查点回执对象（审核 R19-A01 / R20-S2）。

设计原则（执行指令 §3.2）：

> **三个小对象，而不是万能 FrozenReceipt。**

| 对象 | 职责 | 关键字段 |
|---|---|---|
| `RunIdentity` | 定义**本次实际要做什么** | 输入内容摘要、有效计划摘要、实现/模型/工具指纹 |
| `CheckpointReceipt` | 证明**中间工作可复用** | 身份、生产 attempt、覆盖的逻辑阶段、精确产物与摘要、时间/颜色合同引用 |
| `GenerationReceipt` | 证明**最终哪一整代已提交** | schema、代/attempt、RunIdentity、必需正式对象清单/摘要、marker 摘要、持久提交状态 |

为什么拆三个：

- 一个"万能回执"会把"要做什么""做到哪""最后提交了哪一代"混成一个 blob，
  任何一处变化都只能整份失效 —— 检查点复用就退化成"要么全信要么全扔"。
- 拆开后 `CheckpointReceipt` 可以带 `attempt_id`，
  但 `resume` 允许跨 attempt 复用**已验证过**的旧检查点；
  而 `GenerationReceipt` 只认「这一代是否已持久提交」。

**帧 PTS 等大数组不进这些对象**，放带摘要的合同文件里，回执只引用它
（执行指令明确要求）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: 回执 schema 版本。**改字段语义必须升版本**，
#: 旧 schema 缺必要字段按 `legacy/unproven` 处理（不复用、不冒充已验证）。
RECEIPT_SCHEMA = 2


def _canonical(obj: Any) -> bytes:
    """稳定序列化：键排序、无多余空格、UTF-8。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def sha256_of(obj: Any) -> str:
    """对象（可 JSON 序列化）的 SHA-256 十六进制摘要。"""
    return hashlib.sha256(_canonical(obj)).hexdigest()


def file_digest(path: str, chunk: int = 1 << 20) -> Optional[str]:
    """文件的 SHA-256（读不到返回 None，**不吞异常**）。"""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


@dataclass
class RunIdentity:
    """本次**实际要做什么**。

    审核要求它是"冻结"的：必须在
    「合并参数 → 类型校验 → requested/effective 模式 → 冻结计划」
    **之后**才生成（B06 修的顺序问题正是为此）。
    """

    #: 输入内容摘要（不是路径 —— 路径相同内容可能变）
    input_digest: str = ""
    #: 有效计划摘要（合并+规范化+模式落定后的参数）
    plan_digest: str = ""
    #: 实现指纹：代码版本 / 模型文件名+大小 / 外部工具版本
    impl_fingerprint: str = ""
    #: 便于排查的可读标签（不参与摘要）
    labels: Dict[str, str] = field(default_factory=dict)

    schema: int = RECEIPT_SCHEMA

    def digest(self) -> str:
        """本身份的摘要（供回执引用）。`labels` 不参与 —— 它们只是注释。"""
        return sha256_of({
            "schema": self.schema,
            "input_digest": self.input_digest,
            "plan_digest": self.plan_digest,
            "impl_fingerprint": self.impl_fingerprint,
        })

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # R23-B01：生产写出的回执必须能被 `load_receipt()` 读回来。
        # 原来 to_dict 不写 `kind`，而 load_receipt **要求** kind ——
        # 于是"生产者→序列化→消费者"这条往返在正式代码里从未成立
        # （测试里是手工 `dict(obj.to_dict(), kind=...)` 补上的）。
        d["kind"] = "RunIdentity"
        d["digest"] = self.digest()
        return d


@dataclass
class CheckpointReceipt:
    """**中间工作可复用**的证明。

    与 `GenerationReceipt` 的区别：

    - 检查点可以**跨 attempt 复用**（resume 合法），但必须显式声明 `reused_from`；
    - 检查点**不代表最终发布**，它只覆盖到某个逻辑阶段。
    """

    #: 依赖的 RunIdentity 摘要
    run_identity: str = ""
    #: 生产这次检查点的 attempt
    attempt_id: str = ""
    #: 覆盖的逻辑阶段（如 ``["stage3", "stage4"]``）
    covered_stages: List[str] = field(default_factory=list)
    #: 精确产物 → 摘要（**内容摘要，不是文件大小**）
    artifacts: Dict[str, str] = field(default_factory=dict)
    #: 时间/颜色合同的**引用**（大数组不进这里）
    contracts: Dict[str, str] = field(default_factory=dict)
    #: 边界检查结论（帧数/时长/尺寸/颜色）
    boundary_checks: Dict[str, Any] = field(default_factory=dict)
    #: 若本次是复用旧检查点，写明来源（否则为空）
    reused_from: str = ""

    schema: int = RECEIPT_SCHEMA

    def digest(self) -> str:
        return sha256_of({
            "schema": self.schema,
            "run_identity": self.run_identity,
            "covered_stages": sorted(self.covered_stages),
            "artifacts": self.artifacts,
            "contracts": self.contracts,
            "boundary_checks": self.boundary_checks,
        })

    def is_usable_for(self, run_identity: str) -> Tuple[bool, str]:
        """能否给当前的 RunIdentity 复用。

        Returns:
            ``(ok, reason)``。身份不匹配、无产物、或产物摘要缺失都不可用。
        """
        if not run_identity:
            return False, "run identity is empty"
        if self.run_identity != run_identity:
            return False, "checkpoint belongs to a different run identity"
        if not self.covered_stages:
            return False, "checkpoint covers no stages"
        if not self.artifacts:
            return False, "checkpoint lists no artifacts"
        for p, d in self.artifacts.items():
            if not d:
                return False, f"artifact {p!r} has no digest"
        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # R23-B01：生产写出的回执必须能被 `load_receipt()` 读回来。
        # 原来 to_dict 不写 `kind`，而 load_receipt **要求** kind ——
        # 于是"生产者→序列化→消费者"这条往返在正式代码里从未成立
        # （测试里是手工 `dict(obj.to_dict(), kind=...)` 补上的）。
        d["kind"] = "CheckpointReceipt"
        d["digest"] = self.digest()
        return d


@dataclass
class GenerationReceipt:
    """**最终哪一整代已提交**的证明。

    审核的关键要求（§3.2 第 7 条）：

    > `terminal_pending=True` 或 COMMITTED 写失败，即使媒体已落地，
    > 也不能 JOB_DONE。接口显式序列化终态持久化情况。

    所以 `terminal_persisted` 不是"注释"，是**判据**：
    为 False 时 is_complete() 一律为 False。
    """

    #: 代号（每次发布递增；同一目标的重试沿用同一个 generation）
    generation: int = 0
    attempt_id: str = ""
    #: 依赖的 RunIdentity 摘要
    run_identity: str = ""
    #: 这一代的**必需**正式对象（由可信计划决定，不信报告自报）
    required_objects: List[str] = field(default_factory=list)
    #: 正式对象 → 摘要
    object_digests: Dict[str, str] = field(default_factory=dict)
    #: marker 自身内容的摘要（marker 也列入同一事务的必需对象）
    marker_digest: str = ""
    #: 终态是否已**持久化**（COMMITTED 写成功）。False → 不能算完成。
    terminal_persisted: bool = False
    #: journal 终态（COMMITTED / ROLLED_BACK / ...）
    journal_phase: str = ""
    #: 检查结论摘要（CheckResult 等的汇总）
    check_digest: str = ""

    schema: int = RECEIPT_SCHEMA

    def digest(self) -> str:
        return sha256_of({
            "schema": self.schema,
            "generation": self.generation,
            "attempt_id": self.attempt_id,
            "run_identity": self.run_identity,
            "required_objects": sorted(self.required_objects),
            "object_digests": self.object_digests,
            "marker_digest": self.marker_digest,
            "terminal_persisted": self.terminal_persisted,
            "journal_phase": self.journal_phase,
            "check_digest": self.check_digest,
        })

    def is_complete(self) -> Tuple[bool, str]:
        """这一代是否**可以说"完成了"**。

        R21-C02：审核指出旧实现"使用字段真值判断，未严格要求 COMMITTED、
        非空 attempt、真实布尔类型；字符串 `"false"` 可以被当成真"。
        这里逐条收紧：

        - `terminal_persisted` 必须是**真布尔 True**（`"false"` 之类一律不算）；
        - `journal_phase` 必须是 `COMMITTED` —— 停在 PREPARED/COMMITTING
          的一代，即使媒体已在正式路径，也不是"已完成"；
        - `attempt_id` 不能为空（空 attempt 说明回执不是一次具体执行写的）；
        - 每个必需对象都要有摘要，marker 摘要与 run identity 都非空。

        Returns:
            ``(ok, reason)``。
        """
        if self.terminal_persisted is not True:
            return False, "terminal state was not persisted (terminal_pending)"
        if str(self.journal_phase or "") != "COMMITTED":
            return False, (f"journal phase is {self.journal_phase!r}, "
                           f"not COMMITTED")
        if not str(self.attempt_id or "").strip():
            return False, "attempt id is empty; the receipt is not bound to an execution"
        if not self.required_objects:
            return False, "no required objects declared"
        missing = [p for p in self.required_objects
                   if not self.object_digests.get(p)]
        if missing:
            return False, f"required object(s) lack a digest: {missing[:3]}"
        if not self.marker_digest:
            return False, "marker digest is missing"
        if not self.run_identity:
            return False, "run identity is missing"
        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # R23-B01：生产写出的回执必须能被 `load_receipt()` 读回来。
        # 原来 to_dict 不写 `kind`，而 load_receipt **要求** kind ——
        # 于是"生产者→序列化→消费者"这条往返在正式代码里从未成立
        # （测试里是手工 `dict(obj.to_dict(), kind=...)` 补上的）。
        d["kind"] = "GenerationReceipt"
        d["digest"] = self.digest()
        return d


def load_receipt(raw: Dict[str, Any]) -> Optional[Any]:
    """从 dict 反序列化（按 schema 判别）。

    旧 schema 或缺必要字段 → 返回 None（调用方按 ``legacy/unproven`` 处理，
    **不得**自动复用）。
    """
    if not isinstance(raw, dict):
        return None
    schema = raw.get("schema")
    if schema != RECEIPT_SCHEMA:
        return None
    kind = raw.get("kind")
    try:
        if kind == "RunIdentity":
            return RunIdentity(
                input_digest=raw["input_digest"],
                plan_digest=raw["plan_digest"],
                impl_fingerprint=raw["impl_fingerprint"],
                labels=dict(raw.get("labels") or {}),
            )
        if kind == "CheckpointReceipt":
            return CheckpointReceipt(
                run_identity=raw["run_identity"],
                attempt_id=raw["attempt_id"],
                covered_stages=list(raw.get("covered_stages") or []),
                artifacts=dict(raw.get("artifacts") or {}),
                contracts=dict(raw.get("contracts") or {}),
                boundary_checks=dict(raw.get("boundary_checks") or {}),
                reused_from=raw.get("reused_from") or "",
            )
        if kind == "GenerationReceipt":
            # R21-C02：`terminal_persisted` 只认**真布尔**。
            # 旧实现 `bool(raw.get(...))` 会把字符串 "false" 读成 True，
            # 于是"没持久化"的一代被当成完成。
            _tp = raw.get("terminal_persisted")
            if not isinstance(_tp, bool):
                return None
            _obj = GenerationReceipt(
                generation=int(raw["generation"]),
                attempt_id=str(raw["attempt_id"] or ""),
                run_identity=str(raw["run_identity"] or ""),
                required_objects=list(raw.get("required_objects") or []),
                object_digests=dict(raw.get("object_digests") or {}),
                marker_digest=raw.get("marker_digest") or "",
                terminal_persisted=_tp,
                journal_phase=raw.get("journal_phase") or "",
                check_digest=raw.get("check_digest") or "",
            )
            return _verify_declared_digest(raw, _obj)
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _verify_declared_digest(raw: Dict[str, Any], obj: Any) -> Optional[Any]:
    """反序列化后核对"声明摘要"是否与内容一致（R21-C02）。

    回执里带 `digest` 字段却没被验过，等于给了一份可以自称的证明。
    缺摘要（旧对象）按原样返回；有摘要但对不上 → 视为不可信（None）。

    注意：`digest` 只在 `to_dict()` 里出现，`digest()` 的计算输入不含它，
    所以这里没有自引用问题。
    """
    declared = raw.get("digest")
    if declared is None:
        return obj
    try:
        actual = obj.digest()
    except Exception:
        return None
    if str(declared) != str(actual):
        return None
    return obj
