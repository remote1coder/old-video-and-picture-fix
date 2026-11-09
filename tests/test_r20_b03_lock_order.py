# -*- coding: utf-8 -*-
"""R20-B03：锁前只读预检；锁内重读必须比对事务指纹（审核 R19-B03）。

审核的两个反例：

1. **锁前后 journal 从 A 换成 B** —— 旧实现先用 A 的 targets 取锁，
   锁内重读拿到 B，于是「只锁 A，却检查 B」并返回 `already_committed`。
2. **"只是先取锁"也已有磁盘副作用** —— 锁实现会建目录、开锁文件，
   所以结构/归属校验必须**在取锁之前**做完。

本文件锁住：

- 坏 journal 在取锁前就被拒，且**没有创建任何锁目录**；
- journal 在"预检后、取锁前"被换掉时，`recover` 不会用旧集合去操作新内容；
- `publish` 遇到"旧 journal 的目标不被本次锁覆盖"时明确拒绝。
"""
import json
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "shared_modules") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "shared_modules"))

import publish_lock  # noqa: E402
import publisher as pub  # noqa: E402
from publish_journal import file_digest, write_journal  # noqa: E402


class _Cap:
    def __init__(self):
        self.rows = []

    def _add(self, level, event, msg="", **kw):
        self.rows.append({"level": level, "event": event, "msg": msg, **kw})

    def info(self, e, m="", **k): self._add("info", e, m, **k)
    def warning(self, e, m="", **k): self._add("warning", e, m, **k)
    def error(self, e, m="", **k): self._add("error", e, m, **k)
    def critical(self, e, m="", **k): self._add("critical", e, m, **k)
    def debug(self, e, m="", **k): self._add("debug", e, m, **k)

    def events(self, level=None):
        return [r["event"] for r in self.rows
                if level is None or r["level"] == level]


def _rec(target, existed=False, old=None, new=None, attempt="att-A",
         backup_dir=None):
    """构造一条 record。

    ``backup_dir`` 必须与后面 `recover/publish` 传的一致 —— 否则归属校验会
    先拒（backup 必须在 backup_dir 内），根本走不到取锁。
    """
    d = {
        "artifact": "media",
        "staging": str(target) + ".staging",
        "new_digest": new or "d" * 64,
        "existed": bool(existed),
    }
    if existed:
        d["old_digest"] = old or "e" * 64
        d["backup"] = pub._backup_path(str(backup_dir), target, attempt)
        d["backup_digest"] = old or "e" * 64
    return d


def _journal(tmp_path, name="j.json"):
    return str(tmp_path / "out" / name)


# ------------------------------------------------ ① 锁前拒绝，零副作用

def test_bad_journal_is_rejected_before_any_lock_side_effect(tmp_path):
    """结构坏的 journal 必须在**取锁前**被拒，且不留下锁目录。"""
    jp = _journal(tmp_path)
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    t = str(tmp_path / "m.mkv")
    # 空 record → 结构非法
    write_journal(jp, {"phase": "PREPARED", "attempt_id": "a",
                       "targets": [t], "records": {t: {}}})

    lock_root_before = os.path.isdir(str(tmp_path / ".publish_locks"))
    rep = pub.recover(jp, str(tmp_path / "bak"), "att-B")
    assert rep is not None and rep.get("ok") is False
    assert "structure is inconsistent" in rep.get("reason", ""), rep

    # 关键：没有因为"先取锁"而建出锁目录
    if not lock_root_before:
        assert not os.path.isdir(str(tmp_path / ".publish_locks"))


def test_ownership_failure_also_happens_before_lock(tmp_path):
    """归属失败（target 在允许根外）也要在取锁前拒。"""
    jp = _journal(tmp_path)
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    outside = str(tmp_path / "elsewhere" / "m.mkv")
    write_journal(jp, {"phase": "PREPARED", "attempt_id": "a",
                       "targets": [outside],
                       "records": {outside: _rec(outside)}})

    rep = pub.recover(jp, str(tmp_path / "bak"), "att-B",
                      allowed_output_roots=[str(tmp_path / "out")])
    assert rep is not None and rep.get("ok") is False
    assert "ownership" in rep.get("reason", "").lower(), rep


# ------------------------------------------------ ② 锁前后换组

def test_journal_swapped_between_precheck_and_lock_is_detected(tmp_path, monkeypatch):
    """预检读到 A、取锁后变成 B —— 必须发现并**不会**用 A 的锁操作 B。"""
    out = tmp_path / "out"
    out.mkdir()
    a = str(out / "a.mkv")
    b = str(out / "b.mkv")
    for p in (a, b):
        open(p, "wb").write(b"OLD-" + p.encode())

    jp = _journal(tmp_path)
    # A 版：COMMITTED + all_new（会返回 already_committed）
    da = file_digest(a)
    write_journal(jp, {"phase": "COMMITTED", "attempt_id": "att-A",
                       "targets": [a],
                       "records": {a: _rec(a, existed=True, old=da, new=da,
                                               backup_dir=tmp_path / "bak")}})

    real_acquire = pub.acquire_target_locks
    state = {"swapped": False}
    taken = []          # 记录每次实际取锁的目标集合

    def _swap_then_lock(targets, **kw):
        taken.append(sorted(targets))
        # 在真正取锁**之前**把 journal 换成 B 版
        if not state["swapped"]:
            state["swapped"] = True
            write_journal(jp, {"phase": "COMMITTED", "attempt_id": "att-B",
                               "targets": [b],
                               "records": {b: _rec(b, existed=True,
                                                   old=file_digest(b),
                                                   new=file_digest(b),
                                                   attempt="att-B",
                                                   backup_dir=tmp_path / "bak")}})
        return real_acquire(targets, **kw)

    monkeypatch.setattr(pub, "acquire_target_locks", _swap_then_lock)
    log = _Cap()
    rep = pub.recover(jp, str(tmp_path / "bak"), "att-C", logger=log,
                      allowed_output_roots=[str(out)])

    assert state["swapped"] is True, "测试前提未成立：journal 没被换"

    # ---- 核心断言：**锁的目标集合与最终操作的目标集合必须一致** ----
    # 审核要的不是"必须失败"，而是"不能持 A 的锁去操作 B"。
    # 本实现选择"有界重试"：第一轮发现变了 → 释放 → 第二轮用 B 的锁重新做。
    assert len(taken) >= 2, f"应当重试（实际取锁 {taken}）"
    # Windows 下路径大小写可能被规范化，用 normcase 比
    _nc = lambda xs: [os.path.normcase(x) for x in xs]
    assert _nc(taken[0]) == _nc([a]), f"第一轮应锁 A，实际 {taken[0]}"
    assert _nc(taken[-1]) == _nc([b]), (
        f"最后一轮应锁 B（而不是沿用 A），实际 {taken[-1]}")
    # 第一轮的"变了"必须留下痕迹
    assert "PUBLISH_JOURNAL_CHANGED_UNDER_LOCK" in log.events("warning"), log.events()
    # 第二轮一致 → 允许成功；但结果必须是 **B** 的，不是 A 的
    assert rep is not None
    if rep.get("ok"):
        assert rep.get("action") == "already_committed", rep


# ------------------------------------------------ ③ publish 的锁覆盖

def test_publish_refuses_when_prior_journal_outside_lock(tmp_path):
    """旧 journal 指向本次 targets 之外 → publish 必须拒绝，不能持 A 锁动 B。"""
    out = tmp_path / "out"
    tmp_out = tmp_path / "temp"
    out.mkdir()
    tmp_out.mkdir()
    mine = str(out / "mine.mkv")
    other = str(out / "other.mkv")
    for p in (mine, other):
        open(p, "wb").write(b"OLD")

    jp = _journal(tmp_path)
    # 旧事务：目标是 other（不在本次 targets 里）
    write_journal(jp, {"phase": "PREPARED", "attempt_id": "old",
                       "targets": [other],
                       "records": {other: _rec(other, existed=True,
                                               old=file_digest(other),
                                               backup_dir=tmp_out / "publish_backup")}})

    staging = tmp_path / "s.bin"
    staging.write_bytes(b"NEW-CONTENT")
    arts = [pub.Artifact("media", str(staging), mine)]
    log = _Cap()
    res = pub.publish(arts, jp, str(tmp_out / "publish_backup"), "att-new",
                      logger=log,
                      allowed_output_roots=[str(out)])

    assert res.ok is False, res
    assert res.failed_at == "recover_prior", res.failed_at
    assert "PUBLISH_PRIOR_OUTSIDE_LOCK" in log.events("error"), log.events()
    # 本次目标未被创建
    assert not os.path.exists(mine) or open(mine, "rb").read() == b"OLD"


def test_txn_fingerprint_changes_with_targets():
    """指纹必须对 targets 变化敏感（只比 attempt_id 不够）。"""
    d1 = {"attempt_id": "x", "phase": "PREPARED",
          "targets": ["/a"], "records": {"/a": {}}}
    d2 = dict(d1, targets=["/b"], records={"/b": {}})
    d3 = dict(d1)
    assert pub._txn_fingerprint(d1) == pub._txn_fingerprint(d3)
    assert pub._txn_fingerprint(d1) != pub._txn_fingerprint(d2)


def test_fingerprint_is_order_insensitive_for_targets():
    """targets 顺序不同但集合相同 → 指纹相同（否则会误判"变了"）。"""
    d1 = {"attempt_id": "x", "phase": "PREPARED",
          "targets": ["/a", "/b"], "records": {"/a": {}, "/b": {}}}
    d2 = {"attempt_id": "x", "phase": "PREPARED",
          "targets": ["/b", "/a"], "records": {"/b": {}, "/a": {}}}
    assert pub._txn_fingerprint(d1) == pub._txn_fingerprint(d2)
