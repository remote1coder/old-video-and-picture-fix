# -*- coding: utf-8 -*-
"""第十九轮 A2：journal 严格校验与恢复所有权（对应 R18-A02）。

审核指出的四个缺口：

1. **只要求 `records ⊆ targets`**，未要求完整对应 ——
   `targets=[媒体, 必需报告]`、`records={媒体}` 能通过校验，
   恢复返回 `already_committed/all_new`，**遗漏的报告没被检查**。
2. **非 dict 的 record 值未被拒绝**。
3. 恢复**没有验证 backup 属于允许根和本事务** ——
   受控探针把 backup 指向无关文件，恢复收尾就请求删除它
   （审核明确：探针只记录意图，没实际删除任何文件）。
4. **校验器发生异常时把 problems 设为空**，形成"失败后放行"。

本文件按"结构 / 归属 / 恢复"三层写，全部调用生产函数。
"""
import hashlib
import json
import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import publish_journal as pj  # noqa: E402
import publisher as pub  # noqa: E402


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _doc(targets, records, attempt="att-A", phase=pj.PHASE_PREPARED):
    return {"attempt_id": attempt, "phase": phase,
            "targets": list(targets), "records": dict(records)}


# ================================================ 结构层

def test_structure_rejects_a_target_with_no_record(tmp_path):
    """**核心缺口**：targets 有两项、records 只有一项 → 必须报"缺记录"。

    审核：`targets=[媒体,必需报告]`、`records={媒体}` 可通过校验，
    遗漏报告没有被检查。
    """
    a = str(tmp_path / "a.mkv")
    b = str(tmp_path / "a.verify.json")
    problems = pj.validate_journal_structure(
        _doc([a, b], {a: {"existed": True}}))
    assert problems, "缺一项记录必须被报出来"
    assert any(("missing" in p.lower()) or ("not recorded" in p.lower())
               or ("incomplete" in p.lower()) for p in problems), problems


def test_structure_rejects_non_dict_record_value(tmp_path):
    """非 dict 的 record 值必须拒绝 —— 后面 `rec.get(...)` 会直接崩。"""
    a = str(tmp_path / "a.mkv")
    problems = pj.validate_journal_structure(
        _doc([a], {a: "not-a-dict"}))
    assert problems, "非 dict 的 record 值必须被拒绝"


def test_structure_rejects_duplicate_targets(tmp_path):
    """重复路径必须拒绝 —— 否则同一个文件会被登记两次、语义不明。"""
    a = str(tmp_path / "a.mkv")
    problems = pj.validate_journal_structure(
        _doc([a, a], {a: {"existed": True}}))
    assert problems, "重复 targets 必须被拒绝"


def test_structure_rejects_unknown_phase(tmp_path):
    """未知 phase 必须拒绝（不能只检查"是字符串"）。"""
    a = str(tmp_path / "a.mkv")
    problems = pj.validate_journal_structure(
        _doc([a], {a: {"existed": True}}, phase="WEIRD_PHASE"))
    assert problems, "未知 phase 必须被拒绝"


def test_structure_requires_attempt_id(tmp_path):
    """缺 attempt_id 必须拒绝 —— 没有它就无法说"这是哪一次尝试"。"""
    a = str(tmp_path / "a.mkv")
    doc = {"phase": pj.PHASE_PREPARED, "targets": [a],
           "records": {a: {"existed": True}}}
    problems = pj.validate_journal_structure(doc)
    assert problems, "缺 attempt_id 必须被拒绝"


def test_structure_rejects_zero_targets_span(tmp_path):
    """records 与 targets 完全无交集 → 拒绝（不是"子集"就够了）。"""
    a = str(tmp_path / "a.mkv")
    b = str(tmp_path / "b.mkv")
    problems = pj.validate_journal_structure(
        _doc([a], {b: {"existed": True}}))
    assert problems


def test_structure_accepts_a_fully_consistent_doc(tmp_path):
    """完整一致时必须**不报**（防止修过头把正常 journal 全拒了）。"""
    a = str(tmp_path / "a.mkv")
    b = str(tmp_path / "a.verify.json")
    problems = pj.validate_journal_structure(
        _doc([a, b], {a: {"existed": True}, b: {"existed": False}}))
    assert problems == [], problems


# ================================================ 归属层

def test_ownership_rejects_backup_outside_backup_dir(tmp_path):
    """backup 指向 backup_dir 之外 → 必须拒绝。

    审核：受控探针把 backup 指向无关文件，恢复收尾就请求删除它。
    """
    bdir = tmp_path / "bak"
    bdir.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    a = out / "a.mkv"
    a.write_bytes(b"NEW")
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"MUST-NOT-BE-DELETED")

    doc = _doc([str(a)], {str(a): {"existed": True, "backup": str(victim)}})
    problems = pj.validate_journal_ownership(
        doc, str(bdir), [str(out)], ".pubbak")
    assert problems, "backup 在 backup_dir 之外必须被拒绝"


def test_ownership_rejects_backup_for_another_attempt(tmp_path):
    """backup 的 attempt 与 journal 不符 → 拒绝（不是本事务的备份）。"""
    bdir = tmp_path / "bak"
    bdir.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    a = out / "a.mkv"
    a.write_bytes(b"NEW")
    bk = bdir / "a.mkv.pubbak-att-OTHER"
    bk.write_bytes(b"OLD")

    doc = _doc([str(a)], {str(a): {"existed": True, "backup": str(bk)}},
               attempt="att-A")
    problems = pj.validate_journal_ownership(
        doc, str(bdir), [str(out)], ".pubbak")
    assert problems, "别的 attempt 的备份不能被本事务清理"


def test_ownership_rejects_target_outside_allowed_roots(tmp_path):
    """target 在允许根之外 → 拒绝。"""
    bdir = tmp_path / "bak"
    bdir.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    a = outside / "a.mkv"
    a.write_bytes(b"X")

    doc = _doc([str(a)], {str(a): {"existed": True}})
    problems = pj.validate_journal_ownership(
        doc, str(bdir), [str(out)], ".pubbak")
    assert problems, "target 越界必须被拒绝"


def test_ownership_accepts_a_canonical_backup(tmp_path):
    """规范路径必须放行（防止修过头）。"""
    bdir = tmp_path / "bak"
    bdir.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    a = out / "a.mkv"
    a.write_bytes(b"NEW")
    bk = bdir / "a.mkv.pubbak-att-A"
    bk.write_bytes(b"OLD")

    doc = _doc([str(a)], {str(a): {"existed": True, "backup": str(bk)}},
               attempt="att-A")
    problems = pj.validate_journal_ownership(
        doc, str(bdir), [str(out)], ".pubbak")
    assert problems == [], problems


# ================================================ 恢复层

def test_recover_refuses_when_the_validator_raises(tmp_path, monkeypatch):
    """**核心缺口**：校验器抛异常时必须失败，不能"problems=[] 后放行"。"""
    bdir = tmp_path / "bak"
    bdir.mkdir()
    a = tmp_path / "a.mkv"
    a.write_bytes(b"NEW")
    jp = str(tmp_path / "journal.json")
    assert pj.write_journal(jp, _doc([str(a)], {str(a): {"existed": True}}))

    def boom(_doc):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(pj, "validate_journal_structure", boom, raising=True)
    res = pub.recover(jp, str(bdir), "att-X")
    assert res is not None and res.get("ok") is False, res
    assert "validat" in str(res.get("reason", "")).lower() or \
        "structure" in str(res.get("reason", "")).lower(), res


def test_recover_does_not_delete_an_unrelated_file(tmp_path):
    """backup 指向无关文件 → 恢复**不得删除它**。

    构造一个"磁盘已是完整旧代"的场景（会走 confirmed_rolled_back →
    _cleanup_backups）。旧实现会直接 os.remove(journal 里写的 backup)。
    """
    bdir = tmp_path / "bak"
    bdir.mkdir()
    a = tmp_path / "a.mkv"
    a.write_bytes(b"OLD-GENERATION")
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"MUST-NOT-BE-DELETED")

    doc = _doc([str(a)], {str(a): {"existed": True,
                                   "old_digest": _sha(str(a)),
                                   "backup": str(victim)}})
    jp = str(tmp_path / "journal.json")
    assert pj.write_journal(jp, doc)

    res = pub.recover(jp, str(bdir), "att-Y")
    assert victim.exists(), "无关文件被删掉了"
    assert victim.read_bytes() == b"MUST-NOT-BE-DELETED"
    assert res is not None and res.get("ok") is False, res


def test_recover_accepts_a_canonical_journal(tmp_path):
    """规范的 all_old journal：确认回滚、清理本事务备份、返回 ok。"""
    bdir = tmp_path / "bak"
    bdir.mkdir()
    a = tmp_path / "a.mkv"
    a.write_bytes(b"OLD")
    bk = bdir / "a.mkv.pubbak-att-C"
    bk.write_bytes(b"OLD")

    doc = _doc([str(a)], {str(a): {"existed": True,
                                   "old_digest": _sha(str(a)),
                                   "backup": str(bk)}}, attempt="att-C")
    jp = str(tmp_path / "journal.json")
    assert pj.write_journal(jp, doc)

    res = pub.recover(jp, str(bdir), "att-C")
    assert res is not None and res.get("ok") is True, res
    assert res.get("action") == "confirmed_rolled_back", res
    assert not bk.exists(), "本事务的备份应被清理"
