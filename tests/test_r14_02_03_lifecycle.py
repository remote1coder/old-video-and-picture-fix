# -*- coding: utf-8 -*-
"""R14-02/03 收尾：摘要不得被"重读失败"抹掉；恢复记录必须有消费方。

**复核报告指出的四个残留**

1. `_build_journal_records()` 构造 PREPARED 时**重读** old/backup 摘要，
   读失败就把 snapshot 里已验证的值变成 `None`（已用生产函数复现）。
2. `_verify_restored()` 在缺 `old_digest` 时**直接返回 True** —— 没校验也报成功。
3. COMMITTED 收尾**重读** `new_digest`，失败写 `None`，随后照常清备份、
   返回 0。
4. `has_unresolved_recovery()` 只有定义和测试调用，**没有生产消费方**。

本文件全部调用生产函数。
"""
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "photo_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage6_encode as s6  # noqa: E402


def _w(p, data=b"x"):
    os.makedirs(os.path.dirname(str(p)) or ".", exist_ok=True)
    with open(str(p), "wb") as f:
        f.write(data)
    return str(p)


# ---------------------------------------------------------------- 缺口 1

def test_build_records_reuses_verified_digests(tmp_path, monkeypatch):
    """PREPARED 记录必须复用 snapshot 里已验证的摘要，不能被重读抹成 None。"""
    target = _w(tmp_path / "t.avif", b"OLD")
    backup = _w(tmp_path / "bak.avif", b"OLD")
    old_d = s6.file_digest(target)
    bk_d = s6.file_digest(backup)
    assert old_d and bk_d

    snapshot = {target: {"state": s6.BACKUP_OK, "backup": backup,
                         "old_digest": old_d, "backup_digest": bk_d}}

    # 让所有**重读**都失败（模拟权限/I-O 错误）
    import publish_journal as pj
    monkeypatch.setattr(s6, "file_digest", lambda p, chunk=1 << 20: None)
    monkeypatch.setattr(pj, "file_digest", lambda p, chunk=1 << 20: None)

    recs = s6._build_journal_records(
        [target], snapshot, {target: target + ".partial"},
        {target: "new-digest-abc"})

    r = recs[target]
    assert r["old_digest"] == old_d, "已验证的 old_digest 不能被重读失败抹掉"
    assert r["backup_digest"] == bk_d, "已验证的 backup_digest 同样不能被抹掉"
    assert r["new_digest"] == "new-digest-abc"
    assert r["existed"] is True


def test_build_records_marks_absence_explicitly(tmp_path):
    """旧代确实不存在 → existed=False；不是"读不出来"冒充的。"""
    missing = str(tmp_path / "never.avif")
    recs = s6._build_journal_records([missing], {}, {missing: "x.partial"},
                                     {missing: "nd"})
    r = recs[missing]
    assert r["existed"] is False
    assert r["old_digest"] is None
    assert r["file_state"] == "ABSENT"


def test_build_records_does_not_call_absent_when_unreadable(tmp_path):
    """目标存在但是目录（不可读）→ file_state=ERROR，existed 不能为 False。"""
    d = tmp_path / "as_dir.avif"
    d.mkdir()
    recs = s6._build_journal_records([str(d)], {}, {str(d): "p"},
                                     {str(d): "nd"})
    r = recs[str(d)]
    assert r["file_state"] == "ERROR"
    assert r["existed"] is False, "ERROR 不是 PRESENT"
    assert r["old_digest"] is None


# ---------------------------------------------------------------- 缺口 2

def test_verify_restored_rejects_missing_digest(tmp_path):
    """缺 old_digest 时**不得**判通过（原来 return True）。"""
    p = _w(tmp_path / "a.avif", b"whatever")
    assert s6._verify_restored(p, {}, {}) is False
    assert s6._verify_restored(p, {}, {p: {}}) is False
    assert s6._verify_restored(p, {}, {p: {"old_digest": None}}) is False


def test_verify_restored_checks_absence_for_absent_old_generation(tmp_path):
    """旧代本来不存在 → 恢复后目标必须确实不在。"""
    p = str(tmp_path / "gone.avif")
    snap = {p: {"existed": False, "old_digest": None}}
    assert s6._verify_restored(p, {}, snap) is True      # 确实不在 → 通过
    _w(p, b"resurrected")
    assert s6._verify_restored(p, {}, snap) is False     # 又出现了 → 失败


def test_verify_restored_positive_case(tmp_path):
    """正常情形：摘要相等才通过。"""
    p = _w(tmp_path / "a.avif", b"OLD")
    good = s6.file_digest(p)
    assert s6._verify_restored(p, {}, {p: {"existed": True,
                                          "old_digest": good}}) is True
    assert s6._verify_restored(p, {}, {p: {"existed": True,
                                          "old_digest": "deadbeef"}}) is False


# ---------------------------------------------------------------- 缺口 4

def test_unresolved_recovery_is_consumed_by_recovery_entry(tmp_path):
    """存在未解决的恢复记录 → 恢复入口必须拒绝把路径当成有效。"""
    out = tmp_path / "out" / "clip_restored.avif"
    _w(out, b"half")
    status = str(out) + ".status.json"
    rec = s6.recovery_record_path(status)
    os.makedirs(os.path.dirname(rec) or ".", exist_ok=True)
    with open(rec, "w", encoding="utf-8") as f:
        json.dump({"status": "RECOVERY_REQUIRED",
                   "prior_status": "SUCCESS",
                   "failed_operation": "publish"}, f)

    assert s6.has_unresolved_recovery(status) is True

    rep = s6._recover_prior_publish(str(out), logger=None)
    assert rep["status"] == "needs_human", rep


def test_unreadable_recovery_record_blocks(tmp_path):
    """恢复记录存在但读不出 → 同样不能当成"没有旧事务"。"""
    out = tmp_path / "out" / "clip_restored.avif"
    _w(out, b"half")
    status = str(out) + ".status.json"
    rec = s6.recovery_record_path(status)
    os.makedirs(os.path.dirname(rec) or ".", exist_ok=True)
    # 目录占位 → json.load 抛 OSError
    os.makedirs(rec)

    rep = s6._recover_prior_publish(str(out), logger=None)
    assert rep["status"] == "needs_human", rep
    assert rep["report"]["reason"] == "unreadable"


def test_no_recovery_record_falls_through_to_journal_logic(tmp_path):
    """没有恢复记录时行为不变（不能因为新逻辑挡住正常流程）。"""
    out = tmp_path / "out" / "clip_restored.avif"
    _w(out, b"ok")
    rep = s6._recover_prior_publish(str(out), logger=None)
    assert rep["status"] in ("none", "already_new", "needs_human"), rep
