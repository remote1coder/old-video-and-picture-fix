# -*- coding: utf-8 -*-
"""最小生产发布器：媒体与报告作为**一代**一起提交，失败可恢复。

**要解决什么**

`stage7` 原来自己拼发布流程：先 `replace` 媒体，再转正 verify / fullchain
报告。报告失败只置一个标记然后 `return 82` —— 磁盘上留下
**B 媒体 + A 报告**：媒体是新的、报告描述的是上一代。旧媒体既没有备份，
也没有这次提交的持久恢复记录。进程若在两次 `replace` 之间消失，
同样没有恢复依据。

**本模块提供什么（且仅此）**

1. ``Artifact`` —— 本代的一个必需对象（逻辑名 + 候选件 + 最终路径）；
2. ``publish()`` —— 一个事务：
   - 校验决策允许发布、候选件可读且身份冻结；
   - 取**覆盖完整目标集合**的锁（顺序固定，避免死锁）；
   - 锁内先处理未完成事务，再备份旧代并写 ``PREPARED``；
   - ``COMMITTING`` → 逐个 ``replace`` → 核对 → ``COMMITTED``；
   - 任一步失败：按备份恢复**完整旧代**；恢复不了就写
     ``RECOVERY_REQUIRED`` 并如实返回；
3. ``recover()`` —— 按 journal 收拾上次未完成的事务。

**不做什么**

- 不做业务判断（检查结果由调用方算好并通过 ``decision`` 传进来）；
- 不提供跨文件系统多文件原子性（`os.replace` 只保证单次替换）；
- 不对"任意外部读者"承诺整组快照 —— 只对遵守本协议的读者成立。

借用了 ``publish_journal`` 的持久化与三态判定，但**没有**搬它的已知缺口
（空摘要当作可恢复、缺摘要判通过等）。
"""
from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional

from publish_journal import (
    PHASE_COMMITTED,
    PHASE_COMMITTING,
    PHASE_PREPARED,
    PHASE_RECOVERY_REQUIRED,
    PHASE_ROLLED_BACK,
    file_digest,
    file_state,
    journal_state,
    read_journal,
    write_journal,
)
from publish_lock import TargetLockTimeout, acquire_target_locks

#: 事务结束后保留 journal：它是审计记录，也是下次启动识别未完成事务的依据。
BACKUP_SUFFIX = ".pubbak"


class Artifact:
    """本代的一个必需对象。

    Attributes:
        name: 逻辑名（``media`` / ``verify`` / ``fullchain`` …），进报告用。
        staging: **attempt 私有**的候选件，必须已经写好并关闭。
        final: 用户可见的最终路径。
        required: 缺失是否阻断提交。
    """

    __slots__ = ("name", "staging", "final", "required")

    def __init__(self, name: str, staging: str, final: str,
                 required: bool = True):
        self.name = name
        self.staging = staging
        self.final = final
        self.required = required

    def __repr__(self) -> str:      # pragma: no cover - 调试用
        return f"Artifact({self.name!r}, {self.staging!r} -> {self.final!r})"


class PublishResult:
    """一次发布的结果。``ok`` 为真才表示**整代**落地。"""

    __slots__ = ("ok", "phase", "failed_at", "restored", "errors",
                 "journal_path", "attempt_id", "published", "recovery",
                 "cleanup_pending", "terminal_pending",
                 "generation_receipt")

    def __init__(self, attempt_id: str, journal_path: str):
        self.attempt_id = attempt_id
        self.journal_path = journal_path
        self.ok = False
        self.phase: Optional[str] = None
        self.failed_at: Optional[str] = None
        self.restored = False          # 失败后是否恢复了完整旧代
        self.errors: List[str] = []
        self.published: List[str] = []
        self.recovery: Optional[Dict[str, Any]] = None
        self.cleanup_pending: List[str] = []
        # R16-01：终态（COMMITTED / ROLLED_BACK）没能持久化。为 True 时
        # **必须保留备份**，否则下一次无从恢复/确认。
        self.terminal_pending: bool = False
        #: R20-A01：成功发布后产出的**整代回执**（GenerationReceipt）。
        #: `ok=True` 且 `terminal_pending=False` 时才有意义；
        #: 调用方应把它原子写到**本 attempt 私有**的 result JSON，
        #: 而不是让 main 去 glob/猜最新文件。
        self.generation_receipt: Optional[Any] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok, "phase": self.phase, "failed_at": self.failed_at,
            "restored": self.restored, "errors": self.errors,
            "published": self.published,
            "attempt_id": self.attempt_id,
            "journal_path": self.journal_path,
            "cleanup_pending": self.cleanup_pending,
            "recovery": self.recovery,
            # 审核 R19-A01：
            # > `publisher.py:458–474` 在 COMMITTED 写失败时设置
            # > terminal_pending，但仍返回 ok=True；as_dict 未包含
            # > terminal_pending。媒体已经晋升与任务已经可靠完成是两种状态。
            "terminal_pending": self.terminal_pending,
            "generation_receipt": (
                self.generation_receipt.to_dict()
                if self.generation_receipt is not None else None),
        }

    def __repr__(self) -> str:      # pragma: no cover - 调试用
        return (f"PublishResult(ok={self.ok}, phase={self.phase}, "
                f"failed_at={self.failed_at}, restored={self.restored})")


def _backup_path(backup_dir: str, final: str, attempt_id: str) -> str:
    """备份放在 ``backup_dir`` 里，用**目标 basename** 区分。

    放回目标同级会让输出目录里堆 `.pubbak` 文件，也容易和"下一代的
    同名目标"混淆。
    """
    base = os.path.basename(final)
    return os.path.join(backup_dir, f"{base}{BACKUP_SUFFIX}-{attempt_id}")


def _restore_tmp_path(final: str, attempt_id: str) -> str:
    """恢复时的目标侧私有暂存。

    **不要**直接往正式文件复制：复制中断会留下半截文件，
    把一个"未知状态"变成"更糟的未知状态"。
    """
    return f"{final}.restore-{attempt_id}.partial"


def _stat_identity(path: str):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (getattr(st, "st_dev", 0), getattr(st, "st_ino", 0),
            st.st_size, st.st_mtime_ns)


def _promote_into_place(staging: str, final: str, expected_digest: str,
                        attempt_id: str, logger=None) -> None:
    """把候选件落到最终路径，**跨卷也安全**。

    同卷：`os.replace` 只是改名，字节必然一致。
    跨卷：先复制到**目标卷的私有暂存**，核对摘要后再 `replace` ——
    绝不直接往正式文件复制（复制中断会留下截断的成品，把上一代一起毁掉）。
    """
    try:
        os.replace(staging, final)
        return
    except OSError as exc:
        if getattr(exc, "errno", None) not in (18, 1):     # EXDEV / EPERM
            # 只有"跨卷"这一类才走复制路径；其它错误原样抛
            if getattr(exc, "winerror", None) != 17:       # ERROR_NOT_SAME_DEVICE
                raise
    xvol = f"{final}.xvol-{attempt_id}.partial"
    try:
        shutil.copyfile(staging, xvol)
        _st, _dig = file_state(xvol)
        if _st != "PRESENT" or _dig != expected_digest:
            raise OSError(
                "cross-volume copy does not match the prepared digest "
                f"(expected={expected_digest} got={_dig})")
        os.replace(xvol, final)
    except OSError:
        try:
            if os.path.exists(xvol):
                os.remove(xvol)
        except OSError:
            pass
        raise
    finally:
        pass
    try:
        if os.path.exists(staging):
            os.remove(staging)
    except OSError:
        if logger:
            logger.warning("PUBLISH_STAGING_CLEANUP_FAIL",
                           "Cross-volume publish succeeded but the staging "
                           "file could not be removed", staging=staging)


def _decision_allows(decision: Optional[Dict[str, Any]]) -> Optional[str]:
    """返回拒绝原因；None 表示允许。"""
    if decision is None:
        return None                      # 调用方没提供 → 不额外拦
    if not isinstance(decision, dict):
        return "publish decision is not a mapping"
    if decision.get("error"):
        return f"publish decision could not be built: {decision['error']}"
    if decision.get("publish") is False:
        return ("publish decision does not allow publication "
                f"(blocked_by={decision.get('blocked_by')})")
    return None


def publish(artifacts: List[Artifact], journal_path: str, backup_dir: str,
            attempt_id: str, decision: Optional[Dict[str, Any]] = None,
            logger=None, lock_timeout: float = 30.0,
            keep_backups: bool = False,
            allowed_output_roots: Optional[List[str]] = None,
            run_identity: Optional[str] = None,
            required_objects: Optional[List[str]] = None,
            marker_digest: str = ""
            ) -> PublishResult:
    """把一组候选件作为一个事务提交到各自的最终路径。

    Args:
        artifacts: 本代必需对象。``staging`` 必须已写完并关闭。
        journal_path: 持久事务记录的路径。
        backup_dir: 旧代备份目录（同卷）。
        attempt_id: 本次尝试标识，进入报告与备份文件名。
        decision: ``check_result.publish_decision()`` 的结论。``publish``
            为 False 或含 ``error`` 时直接拒绝，**一个字节都不动**。
        logger: 可选。
        lock_timeout: 取锁期限。
        keep_backups: 成功后是否保留备份（默认清理）。

    Returns:
        :class:`PublishResult`。
    """
    res = PublishResult(attempt_id, journal_path)

    # ---- 1. 决策必须允许 ----
    _deny = _decision_allows(decision)
    if _deny:
        res.failed_at = "decision"
        res.errors.append(_deny)
        if logger:
            logger.error("PUBLISH_DECISION_DENIED",
                         "Refusing to publish: " + _deny,
                         attempt_id=attempt_id)
        return res

    # ---- 2. 候选件必须存在、可读、且最终路径不重复 ----
    finals: List[str] = []
    plan_rows: List[Dict[str, Any]] = []
    for art in artifacts:
        if art.required and not os.path.isfile(art.staging):
            res.failed_at = "prepare"
            res.errors.append(f"{art.name}: staging file is missing: "
                              f"{art.staging}")
            continue
        _st, _dig = file_state(art.staging)
        if _st != "PRESENT" or not _dig:
            res.failed_at = "prepare"
            res.errors.append(
                f"{art.name}: staging file is not readable ({_st}): "
                f"{art.staging}")
            continue
        _id = _stat_identity(art.staging)
        plan_rows.append({"artifact": art, "digest": _dig,
                          "identity": _id, "staging_state": _st})
        if art.final in finals:
            res.failed_at = "prepare"
            res.errors.append(
                f"{art.name}: duplicate final path {art.final}")
        finals.append(art.final)
    if res.errors:
        if logger:
            for _e in res.errors:
                logger.error("PUBLISH_PREPARE_FAIL", _e,
                             attempt_id=attempt_id)
        return res

    targets = sorted(set(finals))
    old_snapshot: Dict[str, Dict[str, Any]] = {}
    try:
        with acquire_target_locks(targets, timeout=lock_timeout,
                                  logger=logger):
            # ---- 3. 先处理上一次未完成的事务 ----
            # R20-B03：本次只锁了 `targets`。如果旧 journal 记录的是**另一组**
            # 目标，`_recover_locked` 就会在"只锁了本次集合"的临界区里去动
            # 未被锁住的对象 —— 正是审核说的"持 A 的锁操作 B"。
            # 先读一次旧 journal，确认它的 targets **被本次锁覆盖**；
            # 覆盖不了就拒绝，要求先显式 recover。
            _prior_doc = None
            if os.path.isfile(journal_path):
                _prior_doc = read_journal(journal_path)
            if _prior_doc is not None:
                from publish_journal import _norm_path as _np2
                _locked = {_np2(t) for t in targets} - {None}
                _prior_t = {_np2(t) for t in (_prior_doc.get("targets") or [])}
                _prior_t.discard(None)
                _uncovered = sorted(_prior_t - _locked)
                if _uncovered:
                    res.failed_at = "recover_prior"
                    res.errors.append(
                        "a previous transaction touches targets outside this "
                        f"publish's lock set ({len(_uncovered)} path(s)); "
                        "run recover() explicitly before publishing")
                    if logger:
                        logger.error(
                            "PUBLISH_PRIOR_OUTSIDE_LOCK",
                            res.errors[-1], attempt_id=attempt_id,
                            uncovered=_uncovered[:5])
                    return res

            _prior = _recover_locked(journal_path, backup_dir, attempt_id,
                                     logger=logger,
                                     allowed_roots=allowed_output_roots,
                                     expected_doc=_prior_doc)
            if _prior is not None and not _prior.get("ok", True):
                res.failed_at = "recover_prior"
                res.recovery = _prior
                res.errors.append(
                    "a previous unpublished transaction could not be "
                    "resolved; refusing to stack another one on top")
                if logger:
                    logger.error("PUBLISH_PRIOR_UNRESOLVED",
                                 res.errors[-1], attempt_id=attempt_id,
                                 recovery=_prior)
                return res

            # ---- 4. 备份旧代 ----
            try:
                os.makedirs(backup_dir, exist_ok=True)
            except OSError as exc:
                res.failed_at = "backup"
                res.errors.append(f"cannot create backup dir: {exc}")
                return res

            for row in plan_rows:
                art = row["artifact"]
                _fstate, _fdig = file_state(art.final)
                if _fstate == "ERROR":
                    res.failed_at = "backup"
                    res.errors.append(
                        f"{art.name}: existing target exists but is "
                        f"unreadable; refusing to touch it: {art.final}")
                    break
                if _fstate == "ABSENT":
                    old_snapshot[art.final] = {"existed": False,
                                               "old_digest": None,
                                               "backup": None,
                                               "backup_digest": None}
                    continue
                bk = _backup_path(backup_dir, art.final, attempt_id)
                try:
                    shutil.copyfile(art.final, bk)
                except OSError as exc:
                    res.failed_at = "backup"
                    res.errors.append(
                        f"{art.name}: cannot back up existing target "
                        f"({exc}); nothing has been replaced")
                    break
                _bkst, _bkdig = file_state(bk)
                if _bkst != "PRESENT" or not _bkdig or _bkdig != _fdig:
                    res.failed_at = "backup"
                    res.errors.append(
                        f"{art.name}: backup digest mismatch or unreadable; "
                        "refusing to prepare the commit")
                    break
                old_snapshot[art.final] = {
                    "existed": True, "old_digest": _fdig,
                    "backup": bk, "backup_digest": _bkdig}
            if res.errors:
                # 备份阶段失败：还没动任何正式文件，清掉这次的备份即可
                _cleanup_backups(old_snapshot, logger)
                res.failed_at = "backup"
                return res

            # ---- 5. PREPARED ----
            doc = {
                "phase": PHASE_PREPARED,
                "attempt_id": attempt_id,
                "targets": targets,
                "records": {},
                "started_at": _now_str(),
            }
            for row in plan_rows:
                art = row["artifact"]
                doc["records"][art.final] = {
                    "artifact": art.name,
                    "staging": art.staging,
                    "new_digest": row["digest"],
                    "existed": old_snapshot[art.final]["existed"],
                    "old_digest": old_snapshot[art.final]["old_digest"],
                    "backup": old_snapshot[art.final]["backup"],
                    "backup_digest":
                        old_snapshot[art.final]["backup_digest"],
                }
            if not write_journal(journal_path, doc):
                res.failed_at = "journal_prepared"
                res.errors.append("could not persist PREPARED")
                _cleanup_backups(old_snapshot, logger)
                return res
            res.phase = PHASE_PREPARED

            # ---- 6. 候选身份复核（锁外算的摘要，锁内确认没被换过）----
            for row in plan_rows:
                art = row["artifact"]
                if _stat_identity(art.staging) != row["identity"]:
                    res.failed_at = "identity"
                    res.errors.append(
                        f"{art.name}: staging file changed after it was "
                        "verified; refusing to publish")
                    break
            if res.errors:
                write_journal(journal_path,
                              dict(doc, phase=PHASE_RECOVERY_REQUIRED,
                                   reason="staging identity changed"))
                _cleanup_backups(old_snapshot, logger)
                return res

            # ---- 7. COMMITTING ----
            if not write_journal(journal_path, dict(doc, phase=PHASE_COMMITTING)):
                res.failed_at = "journal_committing"
                res.errors.append("could not persist COMMITTING")
                _cleanup_backups(old_snapshot, logger)
                return res
            res.phase = PHASE_COMMITTING

            for row in plan_rows:
                art = row["artifact"]
                try:
                    _promote_into_place(art.staging, art.final, row["digest"],
                                        attempt_id, logger)
                except OSError as exc:
                    res.failed_at = f"replace:{art.name}"
                    res.errors.append(
                        f"{art.name}: os.replace failed: {exc}")
                    break
                _chk_st, _chk_dig = file_state(art.final)
                if _chk_st != "PRESENT" or _chk_dig != row["digest"]:
                    res.failed_at = f"verify:{art.name}"
                    res.errors.append(
                        f"{art.name}: published file does not match the "
                        "prepared digest; treating the generation as broken")
                    break
                res.published.append(art.final)

            if res.errors:
                # ---- 8. 失败 → 尽量恢复完整旧代 ----
                rec = _restore_locked(old_snapshot, attempt_id, logger)
                res.recovery = rec
                res.restored = bool(rec.get("fully_restored"))
                write_journal(
                    journal_path,
                    dict(doc, phase=PHASE_RECOVERY_REQUIRED,
                         reason="; ".join(res.errors), rollback=rec))
                if res.restored:
                    # R16-01：回滚**成功**必须落成 ROLLED_BACK 终态。
                    # 原来这里保留 RECOVERY_REQUIRED 并删备份：磁盘上是完整旧代，
                    # journal 却说"仍需恢复"，下一次 recover 要求备份 → 失败 →
                    # 下一次 publish 也被自己的未解决记录挡住（审核已复现）。
                    if write_journal(
                            journal_path,
                            dict(doc, phase=PHASE_ROLLED_BACK,
                                 reason="; ".join(res.errors), rollback=rec)):
                        # 终态已持久化 → 备份不再需要
                        _cleanup_backups(old_snapshot, logger)
                        res.phase = PHASE_ROLLED_BACK
                        if logger:
                            logger.error(
                                "PUBLISH_ROLLED_BACK",
                                "Publish failed but the previous generation was "
                                "fully restored and recorded",
                                attempt_id=attempt_id,
                                failed_at=res.failed_at, errors=res.errors)
                    else:
                        # 终态没写成：下一次仍要能按磁盘事实确认已回滚，
                        # 所以**保留备份**，不谎称完成。
                        res.cleanup_pending.append("journal_rolled_back")
                        # 终态没落盘 —— 与 COMMITTED 失败同样处理：不能删备份。
                        res.terminal_pending = True
                        if logger:
                            logger.error(
                                "PUBLISH_ROLLBACK_TERMINAL_FAIL",
                                "The previous generation was restored but ROLLED_BACK "
                                "could not be persisted; keeping backups so the next "
                                "run can confirm it",
                                attempt_id=attempt_id, failed_at=res.failed_at)
                else:
                    if logger:
                        logger.error(
                            "PUBLISH_RECOVERY_REQUIRED",
                            "Publish failed AND the previous generation could "
                            "not be fully restored; backups and journal are "
                            "kept for manual recovery",
                            attempt_id=attempt_id, failed_at=res.failed_at,
                            rollback=rec, errors=res.errors)
                return res

            # ---- 9. COMMITTED ----
            if not write_journal(journal_path, dict(doc, phase=PHASE_COMMITTED)):
                # 媒体已经全部换好；journal 写不进去说明"完成记录"缺失，
                # 但**不能**因此谎报失败并把已成立的提交撤销。
                res.cleanup_pending.append("journal_committed")
                # R16-01：终态没持久化 → **保留备份**。
                # 原来这里照样清备份，磁盘 journal 停在 COMMITTING，
                # 下次 recover 因缺备份无法恢复，publish 也被永久挡住。
                res.terminal_pending = True
                if logger:
                    logger.error(
                        "PUBLISH_JOURNAL_COMMITTED_FAIL",
                        "All artifacts were replaced but COMMITTED could not "
                        "be persisted; generation is live, journal is stale",
                        attempt_id=attempt_id)
            res.phase = PHASE_COMMITTED
            res.ok = True

            # ---- R20-A01：产出整代回执 ----
            # 关键点（审核 §3.2 第 7 条）：
            #   `terminal_pending=True` 时**不能**声称完成 ——
            #   媒体已晋升 ≠ 任务已可靠完成。
            try:
                from receipts import GenerationReceipt as _GR
                _digs = {row["artifact"].final: row["digest"]
                         for row in plan_rows}
                res.generation_receipt = _GR(
                    generation=int(doc.get("run_generation") or 1),
                    attempt_id=attempt_id,
                    run_identity=(run_identity or ""),
                    required_objects=(list(required_objects)
                                      if required_objects
                                      else list(_digs.keys())),
                    object_digests=_digs,
                    # marker 由**调用方**准备并列入同一事务的必需对象；
                    # publish 只记录它的摘要（不自己生成，避免循环定义）
                    marker_digest=(marker_digest or ""),
                    terminal_persisted=not res.terminal_pending,
                    journal_phase=PHASE_COMMITTED,
                    check_digest="",
                )
            except Exception as _rex:        # pragma: no cover
                if logger:
                    logger.error("PUBLISH_RECEIPT_FAIL",
                                 "publish succeeded but the generation receipt "
                                 "could not be built", error=str(_rex))
    except TargetLockTimeout as exc:
        res.failed_at = "lock"
        res.errors.append(f"could not acquire the target locks: {exc}")
        if logger:
            logger.error("PUBLISH_LOCK_TIMEOUT", res.errors[-1],
                         attempt_id=attempt_id, targets=targets)
        return res
    except Exception as exc:                 # pragma: no cover - 兜底
        res.failed_at = "unexpected"
        res.errors.append(f"{type(exc).__name__}: {exc}")
        if logger:
            logger.error("PUBLISH_UNEXPECTED", res.errors[-1],
                         attempt_id=attempt_id)
        return res

    # ---- 10. 清理（与提交成功分开记录）----
    if res.ok and not keep_backups and not getattr(res, "terminal_pending", False):
        # R16-01：终态（COMMITTED / ROLLED_BACK）没写进去时，备份是下一次恢复的唯一依据，不能删。
        for fin, snap in old_snapshot.items():
            bk = snap.get("backup")
            if not bk:
                continue
            try:
                os.remove(bk)
            except OSError as exc:
                res.cleanup_pending.append(f"{bk}: {exc}")
    res.cleanup_pending = res.cleanup_pending or []
    return res


def _now_str() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _cleanup_backups(snapshot: Dict[str, Dict[str, Any]], logger,
                     backup_dir: Optional[str] = None,
                     attempt_id: Optional[str] = None,
                     backup_suffix: str = BACKUP_SUFFIX) -> None:
    """删除本事务的备份文件。

    审核 R18-A02：journal 里的 ``backup`` 是**磁盘数据，不是可信指令**。
    受控探针把它指向一个无关文件，旧实现就直接 ``os.remove`` 了它。

    所以：**恢复路径必须**传 ``backup_dir`` + ``attempt_id``，此时删除前校验

    - 备份确实落在 ``backup_dir`` 内（解析链接/junction 后判断）；
    - basename 等于 ``{target_basename}{suffix}-{attempt_id}``
      （与 :func:`_backup_path` 的命名规则一致）。

    任一条不满足 → 只记 ``PUBLISH_BACKUP_CLEANUP_REFUSED``，**不删**。
    """
    strict = bool(backup_dir) and bool(attempt_id)
    for _fin, snap in (snapshot or {}).items():
        bk = snap.get("backup")
        if not bk:
            continue
        if strict:
            expect = (os.path.basename(str(_fin)) + str(backup_suffix)
                      + "-" + str(attempt_id))
            try:
                from publish_journal import _is_within as _within
                in_dir = _within(bk, backup_dir)
            except Exception:          # pragma: no cover - 极端情况保守拒绝
                in_dir = False
            if not in_dir or os.path.basename(bk) != expect:
                if logger:
                    logger.warning(
                        "PUBLISH_BACKUP_CLEANUP_REFUSED",
                        "Refusing to delete a backup that does not provably "
                        "belong to this transaction",
                        backup=bk, expected_name=expect, target=_fin)
                continue
        try:
            os.remove(bk)
        except OSError:
            if logger:
                logger.warning("PUBLISH_BACKUP_CLEANUP_FAIL",
                               "Could not remove a backup file", backup=bk)


GENERATION_ALL_OLD = "all_old"
GENERATION_ALL_NEW = "all_new"
GENERATION_MIXED = "mixed"
GENERATION_UNKNOWN = "unknown"
GENERATION_IDENTICAL = "identical"
#: 全组「新旧字节完全相同」：既满足旧代也满足新代。
#: 这种情形不能凭"先匹配 new"决定 —— 那会把"已回滚到旧代、
#: 但新旧内容恰好相同"误判成新代。终态由事务状态（journal
#: phase）决定。


def classify_generation(records: Optional[Dict[str, Dict[str, Any]]]) -> str:
    """按**磁盘事实**判定这一组目标现在处于哪一代。

    R16-01：恢复不能只看 journal 的 phase 字段，也不能盲目要求备份存在。
    磁盘上的完整事实只有四种；只有先分清是哪一种，才知道该回滚、该补记
    提交、还是必须保留现场。

    Args:
        records: journal 的 ``records``（final -> row），每行含
            ``existed`` / ``old_digest`` / ``new_digest``。

    Returns:
        ``all_old``  每个目标都等于旧代（本就不存在的确实不存在）
        ``all_new``  每个目标都等于新代
        ``mixed``    两者混杂 —— 半代，必须处理
        ``unknown``  有目标读不出摘要或摘要缺失，无法判定
    """
    if not records:
        return GENERATION_UNKNOWN

    n_old = n_new = n_ident = 0
    for fin, row in records.items():
        st, dig = file_state(fin)
        if st == "ERROR":
            # 目录 / 不可访问 ≠ 不存在（审核明确要求区分）
            return GENERATION_UNKNOWN
        want_old = bool(row.get("existed"))
        old_d = row.get("old_digest")
        new_d = row.get("new_digest")
        if st == "ABSENT":
            # 目标不存在：只有当"旧代本来就不存在"才算符合旧代
            if not want_old:
                n_old += 1
            else:
                return GENERATION_MIXED if n_new else GENERATION_UNKNOWN
            continue
        # PRESENT
        if dig is None or (old_d is None and new_d is None):
            return GENERATION_UNKNOWN
        # ---- 中性对象：新旧摘要相同 ----
        # 此时"等于旧代"和"等于新代"是同一件事。原实现先匹配 new，于是
        # "媒体已回滚到旧代、但新旧字节恰好一样"被记成 n_new —— 整组随后
        # 被判 mixed，尽管磁盘其实完整等于旧代（审核探针复现）。
        if old_d is not None and new_d is not None and old_d == new_d:
            if dig != old_d:
                return GENERATION_UNKNOWN   # 与两者都不符
            n_ident += 1
            continue
        if new_d and dig == new_d:
            n_new += 1
        elif old_d is not None and dig == old_d:
            n_old += 1
        else:
            # 既不等于旧代也不等于新代 —— 半覆盖/外来内容
            return GENERATION_UNKNOWN

    # 真正的"混杂"必须同时存在**明确的**旧代与**明确的**新代；
    # 中性对象（n_ident）不参与这个判断 —— 它对两种终态都成立。
    if n_old and n_new:
        return GENERATION_MIXED
    if n_old:
        return GENERATION_ALL_OLD
    if n_new:
        return GENERATION_ALL_NEW
    if n_ident:
        return GENERATION_IDENTICAL
    return GENERATION_UNKNOWN


def _restore_locked(snapshot: Dict[str, Dict[str, Any]], attempt_id: str,
                    logger=None) -> Dict[str, Any]:
    """把目标恢复到旧代。**先预检全组，再动手。**

    预检：每个需要恢复的目标都必须有可读且与 ``old_digest`` 一致的备份。
    任一不满足 → 一个字节都不动，返回 ``fully_restored=False``。
    """
    restored: List[str] = []
    removed: List[str] = []
    failed: List[Dict[str, Any]] = []
    impossible: List[str] = []

    # --- 预检（只读）---
    for fin, snap in snapshot.items():
        # 审核 R19-B02：`existed` 只认**真布尔**。
        # `not snap.get("existed")` 会把 None（未知）也当成"原来不存在"，
        # 随后在执行阶段直接删目标。未知必须进 impossible，交给人工。
        _ex = snap.get("existed")
        if _ex is None:
            impossible.append(fin)
            continue
        if _ex is False:
            # 旧代本来不存在 → 恢复动作是"撤下本次新建的文件"
            continue
        bk = snap.get("backup")
        old_dig = snap.get("old_digest")
        if not bk or not old_dig:
            impossible.append(fin)
            continue
        st, dig = file_state(bk)
        if st != "PRESENT" or not dig or dig != old_dig:
            impossible.append(fin)

    if impossible:
        if logger:
            logger.error(
                "PUBLISH_RESTORE_PRECHECK_FAILED",
                "Cannot restore: some backups are missing or do not match the "
                "recorded old digest. Nothing was modified.",
                impossible=impossible)
        return {"restored": [], "removed": [], "failed": [],
                "impossible": impossible, "fully_restored": False,
                "precheck_failed": True}

    # --- 执行 ---
    for fin, snap in snapshot.items():
        try:
            _ex2 = snap.get("existed")
            if _ex2 is None:
                # 未知身份：**保留现场**，不删不改
                failed.append({"target": fin,
                               "reason": "existed is unknown; kept in place"})
                continue
            if _ex2 is False:
                # 本次新建的 → 撤下
                if os.path.exists(fin):
                    os.remove(fin)
                    removed.append(fin)
                continue
            # **本就没被替换成功的目标**：内容仍是旧代 → 不需要动它。
            # 原实现会无条件再 replace 一次，若那次 replace 也失败，
            # 就把"其实完好"的旧代误判成"恢复失败"。
            _cur_st, _cur_dig = file_state(fin)
            if _cur_st == "PRESENT" and _cur_dig == snap.get("old_digest"):
                restored.append(fin)
                continue
            tmp = _restore_tmp_path(fin, attempt_id)
            shutil.copyfile(snap["backup"], tmp)
            _st, _dig = file_state(tmp)
            if _st != "PRESENT" or _dig != snap["old_digest"]:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                failed.append({"path": fin,
                               "error": "restored copy digest mismatch"})
                continue
            os.replace(tmp, fin)
            _fst, _fdig = file_state(fin)
            if _fst == "PRESENT" and _fdig == snap["old_digest"]:
                restored.append(fin)
            else:
                failed.append({"path": fin,
                               "error": "post-restore digest mismatch"})
        except OSError as exc:
            failed.append({"path": fin, "error": str(exc)})

    ok = not failed and not impossible
    return {"restored": restored, "removed": removed, "failed": failed,
            "impossible": impossible, "fully_restored": ok}


def recover(journal_path: str, backup_dir: str, attempt_id: str,
            logger=None,
            allowed_output_roots: Optional[List[str]] = None
            ) -> Optional[Dict[str, Any]]:
    """按 journal 收拾上次未完成的事务（供启动/重试入口调用）。

    Returns:
        ``None`` 表示没有需要处理的事务；否则是恢复报告。
    """
    # ---- "存在但不是普通文件" ≠ "没有事务" ----
    # 原来用 `not os.path.isfile(...)` 一并吞掉：journal 路径上是**目录**
    # （或被替换成设备/链接）时返回 None，调用方会当成"无事务"直接继续发布，
    # 把一份未知状态当成干净起点。审核要求：目录/不可访问是**未知或错误**。
    if not os.path.exists(journal_path):
        return None                     # 真的没有事务
    if not os.path.isfile(journal_path):
        return {"ok": False,
                "reason": "journal path exists but is not a regular file",
                "journal_path": journal_path}
    # R16-01 / B07：**不能**因为 phase==COMMITTED 就提前返回 ——
    # "journal 说已提交但磁盘上有一份报告是坏的"也要被查到，
    # 复核全组摘要的逻辑在 `_recover_locked` 里，必须让它跑到。
    #
    # ---- R20-B03：只读预检 → 取锁 → 锁内重读比对 ----
    # 旧实现是"先读 journal，直接拿它的 targets 取锁，锁内才校验"：
    #   * 锁实现会 **建目录 + 开锁文件**，所以"只是先取锁"已有磁盘副作用；
    #   * 首次读到 A、锁内重读到 B 时，结果是**锁了 A 却检查 B**。
    # 现在拆成三步，集合变了就释放锁重来（有界 3 次），绝不持 A 的锁操作 B。
    last_reason = None
    for _try in range(3):
        # ---- ① 只读：读 + 结构 + 归属（不建任何文件）----
        doc0 = read_journal(journal_path)
        if doc0 is None:
            return {"ok": False, "reason": "journal unreadable",
                    "journal_path": journal_path}
        _pre = _precheck_doc(doc0, backup_dir, allowed_output_roots,
                             journal_path, logger)
        if _pre is not None:
            return _pre
        fp0 = _txn_fingerprint(doc0)
        targets0 = _norm_target_set(doc0)
        if not targets0:
            return {"ok": False, "journal_path": journal_path,
                    "reason": "journal declares no usable targets"}

        # ---- ② 锁可信目标集合 ----
        try:
            with acquire_target_locks(targets0, timeout=30, logger=logger):
                # ---- ③ 锁内重读并比对事务指纹 ----
                doc1 = read_journal(journal_path)
                if doc1 is None:
                    return {"ok": False, "journal_path": journal_path,
                            "reason": "journal disappeared while taking "
                                      "the lock"}
                if _txn_fingerprint(doc1) != fp0:
                    last_reason = ("journal changed between the read-only "
                                   "precheck and acquiring the lock "
                                   "(attempt/targets/records differ)")
                    if logger:
                        try:
                            logger.warning(
                                "PUBLISH_JOURNAL_CHANGED_UNDER_LOCK",
                                "The journal was rewritten between the "
                                "read-only precheck and acquiring the lock; "
                                "releasing and retrying so we never operate on "
                                "a target set we did not lock.",
                                journal_path=journal_path, attempt=_try + 1)
                        except BaseException:
                            pass
                    continue            # 退出 with → 下一轮重新取锁
                return _recover_locked(journal_path, backup_dir, attempt_id,
                                       logger=logger,
                                       allowed_roots=allowed_output_roots,
                                       expected_doc=doc1)
        except TargetLockTimeout as exc:
            return {"ok": False, "reason": f"lock timeout: {exc}",
                    "journal_path": journal_path}

    return {"ok": False, "journal_path": journal_path,
            "reason": ("journal kept changing under the lock after 3 "
                       f"attempts: {last_reason}")}


def _norm_target_set(doc: Dict[str, Any]) -> List[str]:
    """规范化 journal 的 targets（排序去重），用于"集合是否一致"的比较。"""
    from publish_journal import _norm_path as _np
    out = set()
    for t in (doc.get("targets") or []):
        n = _np(t)
        if n is not None:
            out.add(n)
    return sorted(out)


def _txn_fingerprint(doc: Dict[str, Any]) -> str:
    """事务指纹：attempt + 目标集合 + records 键集合。

    审核 R19-B03 的反例是"首次 journal 指向 A，取锁后重读指向 B"。
    只比 `attempt_id` 不够 —— 同一个 attempt 也可能被改成不同的 targets。
    把三者一起摘要是最省事且够用的判据。
    """
    import hashlib
    import json as _json
    payload = {
        "attempt": str(doc.get("attempt_id") or ""),
        "phase": str(doc.get("phase") or ""),
        "targets": _norm_target_set(doc),
        "record_keys": sorted(
            x for x in ((doc.get("records") or {}).keys()) if isinstance(x, str)),
    }
    blob = _json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _precheck_doc(doc: Dict[str, Any], backup_dir: str,
                  allowed_roots: Optional[List[str]],
                  journal_path: str, logger=None
                  ) -> Optional[Dict[str, Any]]:
    """**只读**预检：结构 + 归属。不创建、不修改任何文件。

    审核 R19-B03：
    > 先只读校验授权与结构；锁住可信目标集合；锁内重读后再比对事务身份、
    > 对象集合及快照。集合变化时有界重试或拒绝，不能持 A 的锁操作 B。

    Returns:
        ``None`` 表示通过；否则是错误报告 dict（调用方直接返回）。
    """
    # ---- 结构 ----
    # 校验器自身出错也必须失败，不能"problems=[] 后放行"（R18-A02）。
    try:
        from publish_journal import validate_journal_structure as _vjs
        _struct_problems = _vjs(doc)
    except Exception as _vexc:
        if logger:
            logger.error(
                "PUBLISH_JOURNAL_VALIDATOR_ERROR",
                "The journal structure validator itself failed; refusing to "
                "act on any record", journal_path=journal_path,
                error=f"{type(_vexc).__name__}: {_vexc}")
        return {"ok": False, "journal_path": journal_path,
                "reason": ("journal structure validator failed: "
                           f"{type(_vexc).__name__}: {_vexc}")}
    if _struct_problems:
        if logger:
            logger.error(
                "PUBLISH_JOURNAL_STRUCTURE_INVALID",
                "The journal's structure is inconsistent; refusing to act on "
                "any of its records.",
                journal_path=journal_path, problems=_struct_problems[:5])
        return {"ok": False, "journal_path": journal_path,
                "reason": ("journal structure is inconsistent: "
                           + "; ".join(_struct_problems[:3]))}

    # ---- 归属 ----
    try:
        from publish_journal import validate_journal_ownership as _vjo
        _roots = list(allowed_roots or [])
        if not _roots:
            _roots = [os.path.dirname(os.path.abspath(backup_dir))]
            if logger:
                try:
                    logger.warning(
                        "PUBLISH_ALLOWED_ROOTS_INFERRED",
                        "allowed_output_roots was not supplied; falling back to "
                        "dirname(backup_dir). This breaks layouts where the "
                        "backup dir and the final outputs live under different "
                        "parents. Pass allowed_output_roots explicitly.",
                        inferred_root=_roots[0], backup_dir=backup_dir)
                except BaseException:
                    pass
        _own_problems = _vjo(doc, backup_dir, _roots, BACKUP_SUFFIX)
    except Exception as _oexc:
        if logger:
            logger.error(
                "PUBLISH_JOURNAL_OWNERSHIP_VALIDATOR_ERROR",
                "The journal ownership validator itself failed; refusing to "
                "act on any record", journal_path=journal_path,
                error=f"{type(_oexc).__name__}: {_oexc}")
        return {"ok": False, "journal_path": journal_path,
                "reason": ("journal ownership validator failed: "
                           f"{type(_oexc).__name__}: {_oexc}")}
    if _own_problems:
        if logger:
            logger.error(
                "PUBLISH_JOURNAL_OWNERSHIP_INVALID",
                "Refusing to touch any path: the journal names something "
                "that does not belong to this transaction",
                journal_path=journal_path, problems=_own_problems[:5])
        return {"ok": False, "journal_path": journal_path,
                "reason": ("journal ownership is not provable: "
                           + "; ".join(_own_problems[:3]))}
    return None


def _recover_locked(journal_path: str, backup_dir: str, attempt_id: str,
                    logger=None,
                    allowed_roots: Optional[List[str]] = None,
                    expected_doc: Optional[Dict[str, Any]] = None
                    ) -> Optional[Dict[str, Any]]:
    """锁内版本的恢复（``publish`` 与 ``recover`` 共用）。

    **R16-01**：恢复不再"看到未解决就要求备份存在"，而是先按**磁盘事实**
    判定这一组处于哪一代，再决定动作：

    - ``all_old``  磁盘已是完整旧代 → 只补记 ROLLED_BACK 终态，不动文件
    - ``all_new``  新代已完整落地但没记录 → 按协议补记 COMMITTED
    - ``mixed`` / ``unknown`` → 用备份回滚；回滚成功写 ROLLED_BACK

    同样不能只凭 ``phase`` 字段认可完整：COMMITTED / ROLLED_BACK 都要复核
    全组摘要（审核 B07）。
    """
    # ---- "存在但不是普通文件" ≠ "没有事务" ----
    # 原来用 `not os.path.isfile(...)` 一并吞掉：journal 路径上是**目录**
    # （或被替换成设备/链接）时返回 None，调用方会当成"无事务"直接继续发布，
    # 把一份未知状态当成干净起点。审核要求：目录/不可访问是**未知或错误**。
    if not os.path.exists(journal_path):
        return None                     # 真的没有事务
    if not os.path.isfile(journal_path):
        return {"ok": False,
                "reason": "journal path exists but is not a regular file",
                "journal_path": journal_path}
    # R20-B03：调用方（`recover`）可能已经在**同一把锁内**读过并比对过指纹。
    # 给了 `expected_doc` 就直接用它，避免再读一次时 journal 又被换掉 ——
    # 那样会出现"校验的是 A、操作的是 B"的同一个漏洞。
    doc = expected_doc if expected_doc is not None else read_journal(journal_path)
    if doc is None:
        return {"ok": False, "reason": "journal unreadable"}

    # ---- R17-02：先校验**结构**，再做任何基于 records 的动作 ----
    # 恢复是按 records 迭代的。如果 records 里有**未列入 targets** 的对象，
    # 恢复就会对"目标集合之外"的路径执行删除/回滚计划 ——
    # 等于让 journal 自行扩张它的作用域（审核受控探针复现）。
    # 结构不成立时按"未知"处理：一个路径都不碰。
    # ---- 校验器自身出错也必须失败，不能"problems=[] 后放行" ----
    # 审核 R18-A02：原来 `except Exception: _struct_problems = []`
    # 会把"校验器没跑成"变成"没有问题"，是典型的失败后放行。
    # ---- 只读预检：结构 + 归属（R20-B03 抽成独立函数）----
    # 抽出来的目的：`recover` 必须在**取锁之前**就拒绝坏 journal ——
    # 锁实现会建目录、开锁文件，那已经是磁盘副作用了。
    _pre = _precheck_doc(doc, backup_dir, allowed_roots, journal_path, logger)
    if _pre is not None:
        return _pre

    records = doc.get("records") or {}
    phase = doc.get("phase")

    # ---- 已是终态：复核全组摘要，不能只看字段 ----
    if phase == PHASE_COMMITTED:
        gen = classify_generation(records)
        # IDENTICAL：新旧字节相同，COMMITTED 与 ROLLED_BACK 都成立 ——
        # 事务已声明提交，按提交收尾。
        if gen in (GENERATION_ALL_NEW, GENERATION_IDENTICAL):
            return {"ok": True, "action": "already_committed",
                    "generation": gen}
        return {"ok": False, "generation": gen,
                "reason": (f"journal says COMMITTED but the group on disk is "
                           f"{gen}; not destroying the unresolved record")}
    if phase == PHASE_ROLLED_BACK:
        gen = classify_generation(records)
        if gen in (GENERATION_ALL_OLD, GENERATION_IDENTICAL):
            return {"ok": True, "action": "already_rolled_back",
                    "generation": gen}
        return {"ok": False, "generation": gen,
                "reason": (f"journal says ROLLED_BACK but the group on disk is "
                           f"{gen}")}

    if not records:
        return {"ok": False, "reason": "journal has no records"}

    snapshot: Dict[str, Dict[str, Any]] = {}
    for fin, rec in records.items():
        snapshot[fin] = {
            # 审核 R19-B02：**不能** bool(rec.get(...)) —— 字段缺失会被
            # 读成"原来不存在"，随后走进删除分支。校验器已保证这里是真 bool；
            # 这里再兜一层：非 bool 直接当未知，交给调用方拒绝。
            "existed": rec.get("existed") if isinstance(
                rec.get("existed"), bool) else None,
            "old_digest": rec.get("old_digest"),
            "backup": rec.get("backup"),
            "backup_digest": rec.get("backup_digest"),
        }

    gen = classify_generation(records)

    # ---- 磁盘已是完整旧代：只补记终态（不动任何文件）----
    # IDENTICAL 也走这里：两种终态在磁盘上等价，而"补记回滚"**不动任何文件**，
    # 是唯一不会产生副作用的选择。
    if gen in (GENERATION_ALL_OLD, GENERATION_IDENTICAL):
        if write_journal(journal_path,
                         dict(doc, phase=PHASE_ROLLED_BACK, generation=gen,
                              resolved=True)):
            _cleanup_backups(snapshot, logger, backup_dir,
                             doc.get("attempt_id"))
            if logger:
                logger.warning(
                    "PUBLISH_RECOVERY_CONFIRMED_ROLLED_BACK",
                    "The group on disk is already the complete previous "
                    "generation; recorded the terminal state",
                    journal_path=journal_path, generation=gen)
            return {"ok": True, "action": "confirmed_rolled_back",
                    "generation": gen, "fully_restored": True}
        return {"ok": False, "generation": gen, "reason":
                "rolled back but the terminal state could not be persisted; "
                "backups are kept"}

    # ---- 新代已完整落地但未记录：按协议补记提交 ----
    if gen == GENERATION_ALL_NEW:
        if write_journal(journal_path,
                         dict(doc, phase=PHASE_COMMITTED, generation=gen,
                              reason="recorded by recovery: the group matches "
                                     "the prepared new digests")):
            _cleanup_backups(snapshot, logger, backup_dir,
                             doc.get("attempt_id"))
            if logger:
                logger.warning(
                    "PUBLISH_RECOVERY_RECORDED_COMMITTED",
                    "All targets already hold the new generation; recorded "
                    "COMMITTED so the next run is not blocked",
                    journal_path=journal_path, generation=gen)
            return {"ok": True, "action": "recorded_committed",
                    "generation": gen}
        return {"ok": False, "generation": gen, "reason":
                "the new generation is live but COMMITTED could not be "
                "persisted; backups are kept"}

    # ---- 混合 / 未知：用备份回滚 ----
    rec_report = _restore_locked(snapshot, attempt_id, logger)
    if rec_report.get("fully_restored"):
        if write_journal(journal_path,
                         dict(doc, phase=PHASE_ROLLED_BACK,
                              rollback=rec_report, resolved=True)):
            _cleanup_backups(snapshot, logger, backup_dir,
                             doc.get("attempt_id"))
            if logger:
                logger.warning(
                    "PUBLISH_RECOVERED",
                    "An unfinished transaction was rolled back to the previous "
                    "generation and recorded",
                    journal_path=journal_path, rollback=rec_report)
            return {"ok": True, "action": "rolled_back", "generation": gen,
                    **rec_report}
        if logger:
            logger.error(
                "PUBLISH_RECOVERY_TERMINAL_FAIL",
                "Rolled back but ROLLED_BACK could not be persisted; keeping "
                "backups", journal_path=journal_path)
        return {"ok": False, "action": "rolled_back_unrecorded",
                "generation": gen, **rec_report}

    write_journal(journal_path,
                  dict(doc, phase=PHASE_RECOVERY_REQUIRED,
                       rollback=rec_report))
    if logger:
        logger.error(
            "PUBLISH_RECOVER_INCOMPLETE",
            "Cannot roll back to a complete previous generation; backups and "
            "journal are kept", journal_path=journal_path,
            generation=gen, rollback=rec_report)
    return {"ok": False, "action": "incomplete", "generation": gen,
            **rec_report}