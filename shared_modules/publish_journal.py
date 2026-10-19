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
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

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
    data = json.dumps(doc, ensure_ascii=False, sort_keys=True,
                      indent=2).encode("utf-8")
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
        cur = file_digest(t)
        old = rec.get("old_digest")
        new = rec.get("new_digest")
        if cur is not None and new is not None and cur == new:
            state = "NEW"
            report["all_old"] = False
        elif cur is None and rec.get("existed") is False:
            state = "ABSENT_OK"      # 本来就该不存在
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
