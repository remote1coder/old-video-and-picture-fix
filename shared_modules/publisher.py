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
                 "cleanup_pending")

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

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok, "phase": self.phase, "failed_at": self.failed_at,
            "restored": self.restored, "errors": self.errors,
            "published": self.published,
            "attempt_id": self.attempt_id,
            "journal_path": self.journal_path,
            "cleanup_pending": self.cleanup_pending,
            "recovery": self.recovery,
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
            keep_backups: bool = False) -> PublishResult:
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
            _prior = _recover_locked(journal_path, backup_dir, attempt_id,
                                     logger)
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
                    _cleanup_backups(old_snapshot, logger)
                    if logger:
                        logger.error(
                            "PUBLISH_ROLLED_BACK",
                            "Publish failed but the previous generation was "
                            "fully restored", attempt_id=attempt_id,
                            failed_at=res.failed_at, errors=res.errors)
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
                if logger:
                    logger.error(
                        "PUBLISH_JOURNAL_COMMITTED_FAIL",
                        "All artifacts were replaced but COMMITTED could not "
                        "be persisted; generation is live, journal is stale",
                        attempt_id=attempt_id)
            res.phase = PHASE_COMMITTED
            res.ok = True
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
    if res.ok and not keep_backups:
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


def _cleanup_backups(snapshot: Dict[str, Dict[str, Any]], logger) -> None:
    for _fin, snap in (snapshot or {}).items():
        bk = snap.get("backup")
        if not bk:
            continue
        try:
            os.remove(bk)
        except OSError:
            if logger:
                logger.warning("PUBLISH_BACKUP_CLEANUP_FAIL",
                               "Could not remove a backup file", backup=bk)


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
        if not snap.get("existed"):
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
            if not snap.get("existed"):
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
            logger=None) -> Optional[Dict[str, Any]]:
    """按 journal 收拾上次未完成的事务（供启动/重试入口调用）。

    Returns:
        ``None`` 表示没有需要处理的事务；否则是恢复报告。
    """
    if not os.path.isfile(journal_path):
        return None
    doc = read_journal(journal_path)
    if doc is None:
        return {"ok": False, "reason": "journal unreadable",
                "journal_path": journal_path}
    if doc.get("phase") == PHASE_COMMITTED:
        return {"ok": True, "action": "already_committed",
                "journal_path": journal_path}
    targets = doc.get("targets") or []
    try:
        with acquire_target_locks(targets, timeout=30, logger=logger):
            return _recover_locked(journal_path, backup_dir, attempt_id,
                                   logger)
    except TargetLockTimeout as exc:
        return {"ok": False, "reason": f"lock timeout: {exc}",
                "journal_path": journal_path}


def _recover_locked(journal_path: str, backup_dir: str, attempt_id: str,
                    logger=None) -> Optional[Dict[str, Any]]:
    """锁内版本的恢复（``publish`` 与 ``recover`` 共用）。"""
    if not os.path.isfile(journal_path):
        return None
    doc = read_journal(journal_path)
    if doc is None:
        return {"ok": False, "reason": "journal unreadable"}
    phase = doc.get("phase")
    if phase == PHASE_COMMITTED:
        return {"ok": True, "action": "already_committed"}

    records = doc.get("records") or {}
    snapshot: Dict[str, Dict[str, Any]] = {}
    for fin, rec in records.items():
        snapshot[fin] = {
            "existed": bool(rec.get("existed")),
            "old_digest": rec.get("old_digest"),
            "backup": rec.get("backup"),
            "backup_digest": rec.get("backup_digest"),
        }
    if not snapshot:
        return {"ok": False, "reason": "journal has no records"}

    rec_report = _restore_locked(snapshot, attempt_id, logger)
    if rec_report.get("fully_restored"):
        write_journal(journal_path,
                      dict(doc, phase=PHASE_RECOVERY_REQUIRED,
                           resolved=True, rollback=rec_report))
        if logger:
            logger.warning("PUBLISH_RECOVERED",
                           "An unfinished transaction was rolled back to the "
                           "previous generation",
                           journal_path=journal_path, rollback=rec_report)
        return {"ok": True, "action": "rolled_back", **rec_report}

    write_journal(journal_path,
                  dict(doc, phase=PHASE_RECOVERY_REQUIRED,
                       rollback=rec_report))
    if logger:
        logger.error("PUBLISH_RECOVER_INCOMPLETE",
                     "Cannot roll back to a complete previous generation; "
                     "backups and journal are kept",
                     journal_path=journal_path, rollback=rec_report)
    return {"ok": False, "action": "incomplete", **rec_report}
