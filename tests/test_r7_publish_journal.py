"""持久 journal（第七轮）：让"上次做到哪一步"可判断，而不是靠猜测。

覆盖四类要求：
1. 崩溃安全写入（tmp → fsync → replace）；
2. 损坏/被外部修改的 journal **不被信任**（不当作指令）；
3. 路径归属校验：越界、`..`、绝对路径、符号链接都拒绝；
4. 恢复判定用**内容摘要**识别新旧代，不迷信阶段字段。
"""

import json
import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "shared_modules"))

import publish_journal as pj  # noqa: E402


def _base_doc(tmp_path, targets, phase=pj.PHASE_COMMITTING):
    return {
        "schema": pj.JOURNAL_SCHEMA,
        "attempt_id": "att-1",
        "phase": phase,
        "targets": [str(t) for t in targets],
        "records": {},
        "roots": [str(tmp_path)],
    }


def test_write_and_read_roundtrip(tmp_path):
    t = tmp_path / "a.avif"
    t.write_bytes(b"x")
    jp = pj.journal_path_for(str(t))
    doc = _base_doc(tmp_path, [t])
    assert pj.write_journal(jp, doc) is True

    got = pj.read_journal(jp)
    assert got is not None
    assert got["attempt_id"] == "att-1"
    assert got["schema"] == pj.JOURNAL_SCHEMA


def test_journal_lives_next_to_target(tmp_path):
    t = tmp_path / "sub" / "a.avif"
    t.parent.mkdir()
    jp = pj.journal_path_for(str(t))
    assert os.path.dirname(jp) == str(t.parent)


def test_tampered_journal_is_rejected(tmp_path):
    """被外部改过的 journal 不能当可信指令。"""
    t = tmp_path / "a.avif"
    t.write_bytes(b"x")
    jp = pj.journal_path_for(str(t))
    pj.write_journal(jp, _base_doc(tmp_path, [t]))

    raw = json.loads(open(jp, encoding="utf-8").read())
    raw["phase"] = pj.PHASE_COMMITTED      # 篡改
    open(jp, "w", encoding="utf-8").write(json.dumps(raw))

    assert pj.read_journal(jp) is None, "篡改后的 journal 仍被信任"


def test_corrupt_journal_returns_none(tmp_path):
    t = tmp_path / "a.avif"
    jp = pj.journal_path_for(str(t))
    open(jp, "w", encoding="utf-8").write("{not json")
    assert pj.read_journal(jp) is None


def test_path_escape_is_rejected(tmp_path):
    """`..` 越界必须被拒绝。"""
    outside = tmp_path.parent / "outside.avif"
    doc = {
        "targets": [str(tmp_path / "ok.avif"), str(outside)],
        "backups": [str(tmp_path / "b.pubbak")],
    }
    problems = pj.validate_journal_paths(doc, [str(tmp_path)])
    assert problems, "越界路径没被拒绝"
    assert any("outside" in p or "escapes" in p for p in problems)


def test_relative_parent_traversal_rejected(tmp_path):
    doc = {"targets": ["../../etc/passwd"]}
    problems = pj.validate_journal_paths(doc, [str(tmp_path)])
    assert problems


def test_no_roots_means_refuse(tmp_path):
    """没给授权根目录时必须拒绝，而不是放行。"""
    doc = {"targets": [str(tmp_path / "a.avif")]}
    problems = pj.validate_journal_paths(doc, [])
    assert problems


def test_symlink_escape_rejected(tmp_path):
    """符号链接/junction 逃逸必须被拒绝。

    注意：本机（Windows + 受限权限）实测 `os.symlink` 会**静默退化成
    mkdir**（`os.path.islink` 为 False、`os.readlink` 报 WinError 4390
    "not a reparse point"）。所以先确认"真的是重解析点"，否则 skip，
    避免用一个假链接得出假结论。
    """
    inner = tmp_path / "inner"
    inner.mkdir()
    outside = tmp_path.parent / "real_outside"
    outside.mkdir(exist_ok=True)
    link = inner / "escape"
    try:
        os.symlink(str(outside), str(link), target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        pytest.skip("当前环境不支持创建符号链接")

    # 确认确实是链接（防止退化成 mkdir 造成假测试）
    try:
        os.readlink(str(link))
    except OSError:
        pytest.skip("symlink 退化成普通目录（本机权限限制），无法验证逃逸")

    doc = {"targets": [str(link / "evil.avif")]}
    problems = pj.validate_journal_paths(doc, [str(tmp_path)])
    assert problems, "通过符号链接越界的路径没被拒绝"


def test_classify_all_new(tmp_path):
    t = tmp_path / "a.avif"
    t.write_bytes(b"NEW")
    doc = _base_doc(tmp_path, [t])
    doc["records"][str(t)] = {"old_digest": pj.file_digest(t) and "oldold",
                              "new_digest": pj.file_digest(t),
                              "existed": True}
    rep = pj.classify_recovery(doc)
    assert rep["all_new"] is True
    assert "ALREADY_COMMITTED" in rep["recommendation"]


def test_classify_all_old(tmp_path):
    t = tmp_path / "a.avif"
    t.write_bytes(b"OLD")
    d = pj.file_digest(t)
    doc = _base_doc(tmp_path, [t])
    doc["records"][str(t)] = {"old_digest": d, "new_digest": "different",
                              "existed": True}
    rep = pj.classify_recovery(doc)
    assert rep["all_old"] is True
    assert "ROLLBACK_SAFE" in rep["recommendation"]


def test_classify_mixed_and_unknown(tmp_path):
    a = tmp_path / "a.avif"
    b = tmp_path / "b.avif"
    a.write_bytes(b"NEW")
    b.write_bytes(b"OLD")
    da, db = pj.file_digest(a), pj.file_digest(b)
    doc = _base_doc(tmp_path, [a, b])
    doc["records"][str(a)] = {"old_digest": "x", "new_digest": da,
                              "existed": True}
    doc["records"][str(b)] = {"old_digest": db, "new_digest": "y",
                              "existed": True}
    rep = pj.classify_recovery(doc)
    assert rep["all_new"] is False and rep["all_old"] is False
    assert "MIXED" in rep["recommendation"]

    # 外部改动（两个摘要都不匹配）→ UNKNOWN，保留现场
    b.write_bytes(b"EXTERNAL-EDIT")
    rep2 = pj.classify_recovery(doc)
    assert rep2["has_unknown"] is True
    assert "KEEP_AND_ASK" in rep2["recommendation"]


def test_equal_length_content_swap_is_detected(tmp_path):
    """等长内容替换必须能被摘要识别（大小相同不能当身份）。"""
    t = tmp_path / "a.avif"
    t.write_bytes(b"AAAA")
    d1 = pj.file_digest(t)
    t.write_bytes(b"BBBB")
    d2 = pj.file_digest(t)
    assert d1 != d2, "等长替换没被识别 —— 不能用大小当身份"


# ---------------------------------------------------------------------------
# 接入守卫：journal 必须真的接在提交路径上
# ---------------------------------------------------------------------------

def test_stage6_writes_journal_before_touching_targets():
    """照片 stage6 必须在改目标之前写 PREPARED，成功后写 COMMITTED。"""
    import os as _os
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    src = open(_os.path.join(root, "photo_restoration_pipeline", "stages",
                             "stage6_encode.py"), encoding="utf-8").read()
    assert "STAGE6_JOURNAL_PREPARED" in src, "没有在提交前持久化 journal"
    assert "phase=PHASE_PREPARED" in src
    assert "phase=PHASE_COMMITTED" in src
    assert "phase=PHASE_RECOVERY_REQUIRED" in src
    # journal 写失败时必须拒绝提交
    assert "STAGE6_JOURNAL_WRITE_FAIL" in src
    # PREPARED 必须出现在第一次 os.replace 之前
    i_prep = src.index("STAGE6_JOURNAL_PREPARED")
    i_replace = src.index("os.replace(tmp_avif, work_output_path)")
    assert i_prep < i_replace, "journal 写在了改动目标之后"


def test_stage6_journal_records_old_digest_and_backup(tmp_path):
    """记录里必须含旧摘要与备份摘要（恢复判断依赖它们）。

    直接调生产函数 `_build_journal_records()`，用真实文件，
    不再靠搜源码字符串。
    """
    import sys as _sys
    _root = tmp_path
    tgt = _root / "out.avif"
    tgt.write_bytes(b"OLD-GEN")
    bk = _root / "0_out.avif"
    bk.write_bytes(b"OLD-GEN")
    stg = _root / "out.avif.partial"
    stg.write_bytes(b"NEW-GEN")

    for cand in (os.path.join(os.path.dirname(os.path.dirname(
                     os.path.abspath(__file__))),
                     "photo_restoration_pipeline", "stages"),):
        if cand not in _sys.path:
            _sys.path.insert(0, cand)
    import stage6_encode as s6

    recs = s6._build_journal_records(
        [str(tgt)],
        {str(tgt): {"state": s6.BACKUP_OK, "backup": str(bk)}},
        {str(tgt): str(stg)},
        {str(tgt): s6.file_digest(str(stg))})

    r = recs[str(tgt)]
    assert r["old_digest"] == s6.file_digest(str(tgt))
    assert r["backup_digest"] == s6.file_digest(str(bk))
    assert r["new_digest"] == s6.file_digest(str(stg))
