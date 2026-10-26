# -*- coding: utf-8 -*-
"""PHOTO-03 / 04 / 05：照片发布 journal 的三处缺陷。

- **PHOTO-03**：`read_journal()` 对"不存在/损坏/摘要不符"一律返回 None，
  调用方把**损坏的 journal** 当成"没有旧事务"继续发布。
  → 新增 `journal_state()` 三态；`invalid` 必须阻断（needs_human）。
- **PHOTO-04**：路径校验只看顶层 `targets`/`backups`/`staging`，
  而生产 journal 把备份写在 `records[target].backup` —— 真正会被操作的
  路径从未被校验。→ 校验必须覆盖 records 内的嵌套路径。
- **PHOTO-05**：PREPARED 时只记 `old_digest`/`backup_digest`，
  `new_digest` 要等全部替换完才写。于是"第一个目标替换后硬终止"时，
  已替换文件既非 OLD 也非 NEW → 判 UNKNOWN，无法自动回滚。
  → PREPARED 时就记录预期新代摘要。
"""
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import publish_journal as pj  # noqa: E402


# ---------------------------------------------------------------- PHOTO-03

def test_journal_state_missing(tmp_path):
    assert pj.journal_state(str(tmp_path / "nope.json")) == "missing"
    assert pj.journal_state("") == "missing"


def test_journal_state_invalid_on_corruption(tmp_path):
    p = tmp_path / "j.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert pj.journal_state(str(p)) == "invalid"


def test_journal_state_invalid_on_digest_mismatch(tmp_path):
    """journal 被外部改过（digest 对不上）必须判 invalid，不是 missing。"""
    p = tmp_path / "j.json"
    doc = {"schema": pj.JOURNAL_SCHEMA, "phase": pj.PHASE_PREPARED,
           "targets": [], "records": {}, "digest": "deadbeef"}
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert pj.journal_state(str(p)) == "invalid"


def test_journal_state_valid_roundtrip(tmp_path):
    p = str(tmp_path / "j.json")
    assert pj.write_journal(p, {"schema": pj.JOURNAL_SCHEMA,
                                "phase": pj.PHASE_PREPARED,
                                "targets": [], "records": {}})
    assert pj.journal_state(p) == "valid"


# ---------------------------------------------------------------- PHOTO-04

def test_nested_backup_outside_root_is_rejected():
    """records 里的 backup 指向授权根之外必须被拦下。"""
    allowed = [os.path.abspath(os.sep + "allowed_root")]
    doc = {
        "targets": [os.path.join(allowed[0], "a.avif")],
        "records": {
            os.path.join(allowed[0], "a.avif"): {
                "backup": os.path.abspath(os.sep + "outside" + os.sep
                                          + "sneaky.bak"),
                "old_digest": "x",
            }
        },
    }
    problems = pj.validate_journal_paths(doc, allowed)
    assert problems, "嵌套 backup 越界必须报问题"


def test_nested_staging_outside_root_is_rejected():
    allowed = [os.path.abspath(os.sep + "allowed_root")]
    doc = {
        "targets": [os.path.join(allowed[0], "a.avif")],
        "records": {
            os.path.join(allowed[0], "a.avif"): {
                "staging": os.path.abspath(os.sep + "outside" + os.sep
                                           + "s.tmp"),
            }
        },
    }
    assert pj.validate_journal_paths(doc, allowed)


def test_nested_paths_inside_root_are_accepted():
    root = os.path.abspath(os.sep + "allowed_root")
    doc = {
        "targets": [os.path.join(root, "a.avif")],
        "records": {
            os.path.join(root, "a.avif"): {
                "backup": os.path.join(root, "a.avif.bak"),
                "staging": os.path.join(root, "a.avif.tmp"),
            }
        },
    }
    assert pj.validate_journal_paths(doc, [root]) == []


# ---------------------------------------------------------------- PHOTO-05

def test_classify_recognises_new_when_new_digest_recorded(tmp_path):
    """只要 PREPARED 记了 new_digest，半提交后的文件就能被识别为 NEW。"""
    t = tmp_path / "out.avif"
    t.write_bytes(b"NEWGENERATION" * 20)
    new_digest = pj.file_digest(t)
    doc = {
        "targets": [str(t)],
        "records": {str(t): {
            "existed": True,
            "old_digest": "oldoldold",
            "new_digest": new_digest,
            "backup": str(tmp_path / "b.bak"),
            "backup_digest": "bkbk",
        }},
    }
    rep = pj.classify_recovery(doc)
    assert rep["targets"][str(t)]["state"] == "NEW", rep


def test_classify_without_new_digest_is_unknown(tmp_path):
    """反例：没有 new_digest 时同一份文件只能判 UNKNOWN（原缺陷）。"""
    t = tmp_path / "out.avif"
    t.write_bytes(b"NEWGENERATION" * 20)
    doc = {
        "targets": [str(t)],
        "records": {str(t): {
            "existed": True,
            "old_digest": "oldoldold",
            "backup": str(tmp_path / "b.bak"),
            "backup_digest": "bkbk",
        }},
    }
    rep = pj.classify_recovery(doc)
    assert rep["targets"][str(t)]["state"] == "UNKNOWN", rep


def test_stage6_records_new_digest_at_prepare():
    """源码护栏：PREPARED 的 records 必须含 new_digest（不能又退回只记旧代）。"""
    src = os.path.join(_ROOT, "photo_restoration_pipeline", "stages",
                       "stage6_encode.py")
    text = open(src, encoding="utf-8", errors="replace").read()
    assert '"new_digest": file_digest(_stg)' in text, \
        "PREPARED 阶段应记录预期新代摘要（取自 staging）"
