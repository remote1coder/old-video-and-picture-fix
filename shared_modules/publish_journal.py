"""PUB-04 配套（第七轮）：发布事务的持久 journal。

**为什么需要它**

提交前项目只有两样东西可以判断"上次发生了什么"：

1. 内存里的 `snapshot`（进程一死就没了）；
2. 磁盘上的 `.pubbak` 备份文件（**没有归属信息**）。

于是硬终止后剩下一堆 `.pubbak`，谁也说不清：
* 它是哪次事务留下的？
* 事务走到哪一步了（备份完？替换到一半？已提交？）；
* 这个备份还完整吗？属于哪一代？

靠"发现备份 + 摘要不符"就自动回滚是**猜测**——残留备份可能来自一次
早已成功发布的旧尝试。

**本模块提供什么**

* 一次性把"原目标是否存在 / 旧摘要 / 备份路径及其摘要 / 新暂存及其
  摘要 / attempt / schema / 目标集合 / 阶段"写进一个 journal 文件；
* 写入是**崩溃安全**的：同目录 tmp → fsync → `os.replace` → 再 fsync 目录；
* 阶段推进按 `PREPARED → COMMITTING → COMMITTED`，失败转
  `RECOVERY_REQUIRED`；
* 读取时**校验路径归属**：journal 里出现的路径必须落在事务授权的
  根目录内，且不得经 `..`、绝对路径越界、junction/符号链接逃逸；
* 恢复**不做猜测**：只把已知状态报告给调用方，由调用方决定；
  未知/损坏/外部修改一律保留现场。

**明确不承诺**

* 不是断电耐久性证明。`fsync` 降低了丢数据概率，但不同文件系统与
  平台（尤其 Windows 的写穿透）行为不同；本项目未做断电实测。
* 不是多文件整组原子。它只让"上次做到哪一步"可判断，不改变
  "多个文件逐个替换"的事实。
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from typing import (Any, Dict, Iterable, List, Optional, Sequence, Tuple)

__all__ = [
    "JOURNAL_SCHEMA",
    "PHASE_PREPARED",
    "PHASE_COMMITTING",
    "PHASE_COMMITTED",
    "PHASE_RECOVERY_REQUIRED",
    "JournalPathError",
    "file_digest",
    "journal_path_for",
    "write_journal",
    "read_journal",
    "validate_journal_paths",
    "classify_recovery",
]

JOURNAL_SCHEMA = "publish-journal/1"

PHASE_PREPARED = "PREPARED"
PHASE_COMMITTING = "COMMITTING"
PHASE_COMMITTED = "COMMITTED"
PHASE_RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
# R16-01：完整回滚**成功且已持久化**的终态。
# 原来回滚成功后仍写 RECOVERY_REQUIRED：磁盘上明明是完整旧代，journal 却说
# "仍需恢复"，于是下一次 publish 被自己的未解决记录挡住（审核已复现）。
PHASE_ROLLED_BACK = "ROLLED_BACK"

_VALID_PHASES = {PHASE_PREPARED, PHASE_COMMITTING, PHASE_COMMITTED,
                 PHASE_RECOVERY_REQUIRED}


class JournalPathError(ValueError):
    """journal 里出现了越界或无法解析的路径。"""


def file_digest(path: str, chunk: int = 1 << 20) -> Optional[str]:
    """文件的 SHA-256；文件不存在返回 None。

    用内容摘要而不是"大小 + mtime"：等长替换、mtime 诱饵都能骗过后者。
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except (FileNotFoundError, IsADirectoryError, PermissionError, OSError):
        return None


def file_state(path: str, chunk: int = 1 << 20) -> Tuple[str, Optional[str]]:
    """把"文件在不在"分成 **PRESENT / ABSENT / ERROR** 三态。

    R13-06：`file_digest()` 对"不存在""权限不足""I/O 错误""是目录"
    **一律返回 None**，调用方无法区分：

    - "旧代本来就没有这个文件"（ABSENT，正常）
    - "文件就在那儿但读不出来"（ERROR，状态未知，必须保留现场）

    混为一谈会让回滚在"读不出摘要"时跳过校验并报"已恢复"。

    Returns:
        ``(state, digest)``：
        ``("PRESENT", sha256)`` 存在且可读；
        ``("ABSENT", None)``    确实不存在；
        ``("ERROR", None)``     存在但不可用（权限 / I/O / 非普通文件）。
    """
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return "ABSENT", None
    except OSError:
        return "ERROR", None
    if not stat.S_ISREG(st.st_mode):
        return "ERROR", None          # 目录/设备/管道：不是可发布产物
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                h.update(b)
        return "PRESENT", h.hexdigest()
    except OSError:
        return "ERROR", None


def journal_path_for(target: str) -> str:
    """journal 与目标同目录：`<name>.journal.json`。

    与锁文件同理——放 temp_dir 会导致不同 job 看不到彼此的事务。
    """
    abs_target = os.path.abspath(target)
    cand = abs_target + ".journal.json"
    if len(cand) < 240:
        return cand
    d = os.path.dirname(abs_target)
    h = hashlib.sha1(abs_target.encode("utf-8")).hexdigest()[:16]
    return os.path.join(d, f".publish-journal-{h}.json")


def _fsync_dir(path: str) -> None:
    """尽力刷新目录项（POSIX 有效；Windows 上是 no-op）。"""
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_journal(journal_path: str, payload: Dict[str, Any]) -> bool:
    """原子写入 journal（tmp → fsync → replace → fsync dir）。

    失败返回 False，**不抛异常** —— 调用方可据此决定"没有 journal
    就不许动目标"。
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(journal_path)) or ".",
                    exist_ok=True)
    except OSError:
        return False

    doc = dict(payload)
    doc.setdefault("schema", JOURNAL_SCHEMA)
    doc.setdefault("updated_at", _now())
    # R16-01：**必须先丢掉 payload 里带来的旧 digest**。
    # `read_journal()` 为了不改动语义会把 digest 放回返回的对象里，
    # 于是"读出来 → 改 phase → 再写回"这种更新会把**旧 digest 卷进哈希**：
    # 写出来的文件自己都读不回来（`read_journal` 校验失败返回 None）。
    # 这正是 recover 补记终态后立即不可读的原因。
    doc.pop("digest", None)
    doc["digest"] = hashlib.sha256(
        json.dumps(doc, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    data = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                      indent=2).encode("utf-8")

    tmp = f"{journal_path}.tmp-{os.getpid()}-{int(time.time() * 1000)}"
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, journal_path)
        _fsync_dir(os.path.dirname(os.path.abspath(journal_path)) or ".")
        return True
    except OSError:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        return False


def journal_state(journal_path: str) -> str:
    """journal 的三态：``"missing"`` / ``"valid"`` / ``"invalid"``。

    PHOTO-03：`read_journal()` 对"不存在""读取失败""JSON 损坏"
    "摘要不符"一律返回 None，调用方于是把**损坏的 journal** 当成
    "没有旧事务"直接开始下一次发布 —— 这违反了"状态未知则保留现场"。
    本函数把两者区分开：调用方对 ``invalid`` 应阻断并要求人工处理。
    """
    if not journal_path or not os.path.exists(journal_path):
        return "missing"
    return "valid" if read_journal(journal_path) is not None else "invalid"


def read_journal(journal_path: str) -> Optional[Dict[str, Any]]:
    """读 journal；不存在/损坏/摘要不符都返回 None（不猜测修复）。"""
    try:
        with open(journal_path, "rb") as fh:
            doc = json.loads(fh.read().decode("utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    if doc.get("schema") != JOURNAL_SCHEMA:
        return None
    recorded = doc.pop("digest", None)
    calc = hashlib.sha256(
        json.dumps(doc, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    doc["digest"] = recorded
    if recorded != calc:
        # 被外部改过或写坏了 —— 保留现场，不当作可信指令
        return None
    return doc


def _resolve_existing_prefix(path: str) -> str:
    """解析路径中**存在的最近祖先**的 realpath，再把不存在的尾巴拼回去。

    为什么不能直接 `os.path.realpath(path)`：Windows 上对**不存在的路径**，
    realpath 会原样返回、**不解析父级符号链接/junction**。实测：

    ```
    realpath(".../inner/escape/evil.avif")   # evil.avif 不存在
      -> "...\\inner\\escape\\evil.avif"       # escape 这个 junction 没被解析
    ```

    而 `escape` 实际指向范围外的目录。于是"越界校验"被绕过。
    journal 里出现的目标/备份/暂存路径**多数本来就不存在**（还要创建），
    所以必须走这条路。
    """
    p = os.path.abspath(path)
    tail: List[str] = []
    while True:
        try:
            exists = os.path.exists(p) or os.path.islink(p)
        except OSError:
            exists = False
        if exists:
            resolved = os.path.realpath(p)
            return os.path.join(resolved, *reversed(tail)) if tail else resolved
        parent, name = os.path.split(p)
        if parent == p:          # 到根了还没有存在的祖先
            return os.path.realpath(path)
        tail.append(name)
        p = parent


def _is_within(child: str, parent: str) -> bool:
    """child 是否在 parent 目录内（解析符号链接/junction 后判断）。"""
    try:
        c = _resolve_existing_prefix(child)
        p = _resolve_existing_prefix(parent)
    except OSError:
        return False
    if os.path.normcase(c) == os.path.normcase(p):
        return True
    return os.path.normcase(c).startswith(os.path.normcase(p) + os.sep)


#: 已知的 phase 取值。未知 phase 必须拒绝 ——
#: 只检查"是字符串"会让 `phase="WHATEVER"` 一路通过。
KNOWN_PHASES = (
    PHASE_PREPARED,
    PHASE_COMMITTING,
    PHASE_COMMITTED,
    PHASE_RECOVERY_REQUIRED,
    PHASE_ROLLED_BACK,
)


def _norm_path(path: Any) -> Optional[str]:
    """规范化成可比较的键；失败返回 None（不抛）。"""
    try:
        return os.path.normcase(os.path.abspath(str(path)))
    except (OSError, ValueError, TypeError):
        return None


def validate_journal_structure(doc: Dict[str, Any]) -> List[str]:
    """校验 journal 的**结构完整性**（与路径越界、归属分开）。

    审核 R18-A02 指出的缺口：

    - 原实现只要求 ``records 的键 ⊆ targets``，于是
      ``targets=[媒体, 必需报告]``、``records={媒体}`` **能通过** ——
      遗漏的报告根本没被检查。
    - **非 dict 的 record 值**没被拒绝（后面 ``rec.get(...)`` 会直接崩）。
    - targets 里的**重复路径**没被拒绝。
    - ``phase`` 只检查"是字符串"，未知值也能通过。
    - ``attempt_id`` 完全没检查。

    现在要求（缺一不可）：

    1. ``attempt_id`` 必须是非空字符串；
    2. ``phase`` 必须是 :data:`KNOWN_PHASES` 之一；
    3. ``targets`` 必须是非空字符串列表，规范化后**无重复**；
    4. ``records`` 必须是非空 dict，键为字符串、**值为 dict**，
       规范化后**无重复**；
    5. **``targets`` 与 ``records`` 规范化后必须一一对应**
       （缺项、多项都报）。

    Returns:
        违规说明列表；空列表表示通过。
    """
    problems: List[str] = []
    if not isinstance(doc, dict):
        return ["journal document is not an object"]

    # ---- attempt_id ----
    attempt = doc.get("attempt_id")
    if not isinstance(attempt, str) or not attempt.strip():
        problems.append("attempt_id is missing or empty")

    # ---- phase ----
    phase = doc.get("phase")
    if phase is None:
        problems.append("phase is missing")
    elif not isinstance(phase, str):
        problems.append(f"phase is not a string: {phase!r}")
    elif phase not in KNOWN_PHASES:
        problems.append(
            f"phase is not a known value: {phase!r} "
            f"(known: {list(KNOWN_PHASES)})")

    # ---- targets ----
    targets = doc.get("targets")
    tkeys: List[str] = []
    if not isinstance(targets, (list, tuple)) or not targets:
        problems.append("targets is missing or empty")
        targets = []
    else:
        for t in targets:
            if not isinstance(t, str) or not t.strip():
                problems.append(f"targets contains a non-string entry: {t!r}")
                continue
            n = _norm_path(t)
            if n is None:
                problems.append(f"targets path cannot be normalised: {t!r}")
                continue
            tkeys.append(n)
    if len(set(tkeys)) != len(tkeys):
        problems.append("targets contains duplicate paths after normalisation")

    # ---- records ----
    records = doc.get("records")
    rkeys: List[str] = []
    if not isinstance(records, dict) or not records:
        problems.append("records is missing or empty")
    else:
        for key, val in records.items():
            if not isinstance(key, str):
                problems.append(f"records has a non-string key: {key!r}")
                continue
            n = _norm_path(key)
            if n is None:
                problems.append(f"records key cannot be normalised: {key!r}")
                continue
            rkeys.append(n)
            if not isinstance(val, dict):
                problems.append(
                    f"records[{key!r}] is not an object "
                    f"(got {type(val).__name__})")
        if len(set(rkeys)) != len(rkeys):
            problems.append(
                "records contains duplicate paths after normalisation")

    # ---- 完整对应（双向）----
    if tkeys and rkeys:
        tset, rset = set(tkeys), set(rkeys)
        missing = sorted(tset - rset)
        extra = sorted(rset - tset)
        if missing:
            problems.append(
                f"{len(missing)} target(s) have no record — journal is "
                f"incomplete: {missing[:3]}")
        if extra:
            problems.append(
                f"{len(extra)} record(s) are not declared in targets: "
                f"{extra[:3]}")

    return problems


def validate_journal_ownership(doc: Dict[str, Any],
                               backup_dir: str,
                               allowed_roots: Sequence[str],
                               backup_suffix: str = ".pubbak"
                               ) -> List[str]:
    """校验 journal 里提到的路径**确实属于本事务**。

    为什么需要单独一层
    ------------------
    journal 是**磁盘上的数据，不是可信指令**。受控探针把 ``backup`` 指向
    一个无关文件，恢复收尾的 ``_cleanup_backups`` 就直接
    ``os.remove`` 了它 —— 审核 R18-A02 明确记录了这个后果。

    ``validate_journal_paths`` 只回答"在不在允许根内"，
    回答不了"这个备份是不是**这一次尝试**造的"。本函数补上：

    1. 每个 target 必须落在 ``allowed_roots`` 内；
    2. 每个非空的 ``backup`` 必须落在 ``backup_dir`` 内；
    3. ``backup`` 的 basename 必须**等于** ``{target_basename}{suffix}-{attempt_id}``
       （与 ``publisher._backup_path()`` 的命名规则一致）——
       journal 自摘要只能证明内容没被改，**不能授予访问权限**。

    Returns:
        违规说明列表；空列表表示通过。
    """
    problems: List[str] = []
    if not isinstance(doc, dict):
        return ["journal document is not an object"]

    roots = [r for r in (allowed_roots or []) if isinstance(r, str) and r]
    if not roots:
        return ["no allowed_roots given; refusing to trust any journal path"]

    bdir = _norm_path(backup_dir) if backup_dir else None
    if not bdir:
        problems.append("backup_dir is missing; cannot prove backup ownership")

    attempt = doc.get("attempt_id")
    if not isinstance(attempt, str) or not attempt.strip():
        # 结构校验会报；这里直接返回，避免用空 attempt 去比对文件名
        return problems + ["attempt_id is missing; cannot prove ownership"]

    records = doc.get("records")
    if not isinstance(records, dict):
        return problems + ["records is not an object"]

    for target, rec in records.items():
        if not isinstance(target, str) or not isinstance(rec, dict):
            continue                      # 结构校验已经报过

        if not any(_is_within(target, r) for r in roots):
            problems.append(
                f"target is outside every allowed root: {target}")

        bk = rec.get("backup")
        if not bk:
            continue                      # 没有备份（新建文件），跳过
        if not isinstance(bk, str):
            problems.append(f"backup is not a string: {bk!r}")
            continue
        if bdir and not _is_within(bk, bdir):
            problems.append(f"backup is outside backup_dir: {bk}")
            continue
        expect = (os.path.basename(target) + str(backup_suffix)
                  + "-" + str(attempt))
        actual = os.path.basename(bk)
        if actual != expect:
            problems.append(
                f"backup does not belong to this transaction "
                f"(expected name {expect!r}, got {actual!r}): {bk}")

    return problems


def validate_journal_paths(doc: Dict[str, Any],
                           allowed_roots: Sequence[str]) -> List[str]:
    """校验 journal 里所有路径都在 `allowed_roots` 内。

    journal 是**磁盘上的数据，不是可信指令**。恢复/清理前必须确认：
    目标、备份、暂存路径都落在事务授权的根目录内，且不因 `..`、
    绝对路径、junction 或符号链接越界。

    返回违规说明列表（空列表 = 通过）。
    """
    problems: List[str] = []
    if not allowed_roots:
        return ["no allowed_roots given; refusing to trust any journal path"]

    fields: List[str] = []
    for key in ("targets", "backups", "staging"):
        vals = doc.get(key)
        if isinstance(vals, str):
            fields.append(vals)
        elif isinstance(vals, Iterable):
            fields.extend([v for v in vals if isinstance(v, str)])
    if isinstance(doc.get("target"), str):
        fields.append(doc["target"])

    # PHOTO-04：生产 journal 把 backup / staging 写在
    # `records[target]` 里（stage6 写入、恢复/清理也从那里取），
    # 只校验顶层键等于**没校验真正会被操作的路径**。
    # 攻击者/误写只要把 records 里的 backup 指到授权根之外，
    # 恢复时就会去读写那个路径。
    _recs = doc.get("records")
    if isinstance(_recs, dict):
        for _t, _r in _recs.items():
            if isinstance(_t, str):
                fields.append(_t)
            if not isinstance(_r, dict):
                continue
            for _k in ("backup", "staging", "path", "target"):
                _v = _r.get(_k)
                if isinstance(_v, str):
                    fields.append(_v)

    for p in fields:
        if os.path.isabs(p) or p.startswith("..") or f"..{os.sep}" in p:
            if not any(_is_within(p, r) for r in allowed_roots):
                problems.append(f"path escapes allowed roots: {p}")
                continue
        if not any(_is_within(p, r) for r in allowed_roots):
            problems.append(f"path outside allowed roots: {p}")
    return problems


def classify_recovery(doc: Dict[str, Any]) -> Dict[str, Any]:
    """把 journal + 磁盘实际状态整理成"可判断的恢复报告"。

    **不做任何修改**，只回答："按记录，现在是哪种情况"。

    判定原则（用**内容摘要**识别新旧代，而不是相信阶段字段）：

    ```
    对每个目标：
      current_digest == new_digest  → 已是新代（替换成功）
      current_digest == old_digest  → 仍是旧代
      其它（含文件不存在、摘要不同） → UNKNOWN（保留现场，不猜）
    ```
    """
    targets = doc.get("targets") or []
    records = doc.get("records") or {}
    report: Dict[str, Any] = {
        "phase": doc.get("phase"),
        "attempt_id": doc.get("attempt_id"),
        "targets": {},
        "all_old": True,
        "all_new": True,
        "has_unknown": False,
    }

    for t in targets:
        rec = records.get(t) or {}
        # R13-06：用三态而不是"摘要为 None 就当作不存在"。
        # 文件存在但读不出来（权限 / I/O / 是目录）属于**状态未知**，
        # 必须走 UNKNOWN 保留现场，不能冒充 ABSENT。
        _fstate, cur = file_state(t)
        old = rec.get("old_digest")
        new = rec.get("new_digest")
        if _fstate == "ERROR":
            state = "UNKNOWN"
            report["has_unknown"] = True
            report["all_old"] = False
            report["all_new"] = False
        elif cur is not None and new is not None and cur == new:
            state = "NEW"
            report["all_old"] = False
        elif cur is None and rec.get("existed") is False:
            # R13-05：**"旧代本来就不存在"只满足旧代，不满足新代。**
            #
            # 原实现这个分支两个标志都不动，于是"首次发布只替换了 AVIF、
            # 其余目标本来就该被新建但还没建"被算成 all_new=True →
            # 推荐 ALREADY_COMMITTED → journal 被清掉，半成品当成完整代。
            #
            # 新代要求这个目标**被创建**（new_digest 非空），现在它不在，
            # 所以 NEW 侧不成立。将来若要支持"删除目标"，应写显式操作类型，
            # 不能靠 None 猜。
            state = "ABSENT_OK"      # 对**旧代**是正确的"不存在"
            report["all_new"] = False
        elif cur is not None and old is not None and cur == old:
            state = "OLD"
            report["all_new"] = False
        else:
            state = "UNKNOWN"
            report["has_unknown"] = True
            report["all_old"] = False
            report["all_new"] = False
        report["targets"][t] = {
            "state": state,
            "file_state": _fstate,
            "current_digest": cur,
            "old_digest": old,
            "new_digest": new,
            "backup": rec.get("backup"),
            "backup_digest": rec.get("backup_digest"),
            "staging": rec.get("staging"),
        }

    if report["has_unknown"]:
        report["recommendation"] = (
            "KEEP_AND_ASK: at least one target does not match any recorded "
            "generation. Do not auto-restore; inspect the files.")
    elif report["all_new"]:
        report["recommendation"] = (
            "ALREADY_COMMITTED: every target already matches the new "
            "generation; nothing to roll back.")
    elif report["all_old"]:
        report["recommendation"] = (
            "ROLLBACK_SAFE: every target still matches the old generation; "
            "a restore would be a no-op or can proceed from backups.")
    else:
        report["recommendation"] = (
            "MIXED: some targets are new and some are old; a restore to the "
            "old generation is safe ONLY if every old_digest is known and "
            "every backup verifies.")
    return report


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
