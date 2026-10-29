# -*- coding: utf-8 -*-
"""R14-02 / R14-03：照片的取消诊断与备份摘要。

**R14-02 缺陷**

`stage6` 把**正式** `status_path` 交给取消守卫（`cancel_guard` 在清理时
写 `{"status": "FAILED"}`）。于是"编码中取消"——一次**还没发布**的尝试
——会让旧 AVIF/TIFF 继续存在、旧 `SUCCESS` 却变成 `FAILED`。
`delete_output=False` 只保护媒体，不保护状态。

**R14-03 缺陷**

`_snapshot_for_rollback()` 复制备份后无条件写 `state=BACKUP_OK`，
而 `file_digest()` 对权限/I/O 问题返回 `None`。于是"备份存在但无法核对"
被当成可恢复；回滚时又因为 `expected=None` 跳过校验并报"已恢复"。

本文件用**真实调用**验证，不做字符串扫描。
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

import cancel_guard as cg  # noqa: E402
import stage6_encode as s6  # noqa: E402


# ---------------------------------------------------------------- R14-02

def test_cancel_writes_attempt_not_official_status(tmp_path):
    """守卫只写 attempt 诊断；正式 status 一个字节都不变。"""
    official = tmp_path / "clip_restored.avif.status.json"
    official.write_text(json.dumps({"status": "SUCCESS", "published": True}),
                        encoding="utf-8")
    before = official.read_bytes()

    out = tmp_path / "clip_restored.avif"
    out.write_bytes(b"OLD-AVIF")
    attempt = str(official) + ".attempt-1234-5678.json"

    g = cg.CancelGuard()
    g.watch(str(out), logger=None, stage=6, delete_output=False,
            status_path=attempt)
    result = g.cleanup_and_record(signum=None)

    assert official.read_bytes() == before, \
        "取消不得改写上一代的正式状态"
    rec = json.loads(official.read_text(encoding="utf-8"))
    assert rec["status"] == "SUCCESS"

    assert os.path.exists(attempt), "取消必须留下 attempt 级记录"
    att = json.loads(open(attempt, encoding="utf-8").read())
    assert att["status"] == "FAILED"
    assert att["stage"] == 6
    assert result is not None


def test_cancel_with_delete_output_still_protects_status(tmp_path):
    """即使允许删媒体，正式状态也不能被取消记录覆盖。"""
    official = tmp_path / "o.avif.status.json"
    official.write_text('{"status": "SUCCESS"}', encoding="utf-8")
    before = official.read_bytes()
    out = tmp_path / "o.avif"
    out.write_bytes(b"data")
    attempt = str(official) + ".attempt-9-9.json"

    g = cg.CancelGuard()
    g.watch(str(out), logger=None, stage=6, delete_output=True,
            status_path=attempt)
    g.cleanup_and_record(signum=None)

    assert official.read_bytes() == before
    assert js_payload(official)["status"] == "SUCCESS"


def js_payload(p):
    return json.loads(open(str(p), encoding="utf-8").read())


def test_stage6_registers_attempt_status_path():
    """生产代码必须把 attempt 路径交给守卫，而不是正式 status。

    这条**只是**调用形态护栏；真实行为由上面两条用例证明。
    放在一起是为了防止有人把 `status_path=status_path` 写回去。
    """
    src = os.path.join(_ROOT, "photo_restoration_pipeline", "stages",
                       "stage6_encode.py")
    text = open(src, encoding="utf-8", errors="replace").read()
    assert "status_path=attempt_status_path" in text, \
        "取消守卫必须用 attempt 级状态路径"
    assert "status_path=status_path)" not in text, \
        "不得把正式 status 交给取消守卫"


# ---------------------------------------------------------------- R14-03

def test_backup_digest_unreadable_is_failed(tmp_path, monkeypatch):
    """复制成功但摘要读不出 → BACKUP_FAILED，提交必须停。"""
    target = tmp_path / "old.avif"
    target.write_bytes(b"OLD")
    bdir = tmp_path / "bak"
    bdir.mkdir()

    calls = {"n": 0}

    def _fake_digest(path, chunk=1 << 20):
        calls["n"] += 1
        return None                      # 模拟权限/I/O 读不出

    monkeypatch.setattr(s6, "file_digest", _fake_digest)
    snap = s6._snapshot_for_rollback([str(target)], str(bdir))

    entry = snap[str(target)]
    assert entry["state"] == s6.BACKUP_FAILED, entry
    assert s6._snapshot_has_failure(snap) is True
    assert target.read_bytes() == b"OLD", "预检失败不得改动旧文件"


def test_backup_digest_mismatch_is_failed(tmp_path, monkeypatch):
    """备份可读但摘要与原文件不符 → BACKUP_FAILED。"""
    target = tmp_path / "old.avif"
    target.write_bytes(b"OLD")
    bdir = tmp_path / "bak"
    bdir.mkdir()

    seq = iter(["aaa", "bbb"])           # old != backup

    def _fake_digest(path, chunk=1 << 20):
        return next(seq)

    monkeypatch.setattr(s6, "file_digest", _fake_digest)
    snap = s6._snapshot_for_rollback([str(target)], str(bdir))
    assert snap[str(target)]["state"] == s6.BACKUP_FAILED
    assert s6._snapshot_has_failure(snap) is True


def test_backup_happy_path_records_both_digests(tmp_path):
    """正常路径：两个摘要都存在且相等，才 BACKUP_OK。"""
    target = tmp_path / "old.avif"
    target.write_bytes(b"OLD-CONTENT")
    bdir = tmp_path / "bak"
    bdir.mkdir()

    snap = s6._snapshot_for_rollback([str(target)], str(bdir))
    entry = snap[str(target)]
    assert entry["state"] == s6.BACKUP_OK
    assert entry["old_digest"] and entry["backup_digest"]
    assert entry["old_digest"] == entry["backup_digest"]
    assert s6._snapshot_has_failure(snap) is False


def test_absent_target_stays_absent(tmp_path):
    """本来就不存在 → ABSENT（回滚时撤下新建），不是 FAILED 也不是 OK。"""
    missing = tmp_path / "never.avif"
    bdir = tmp_path / "bak"
    bdir.mkdir()
    snap = s6._snapshot_for_rollback([str(missing)], str(bdir))
    assert snap[str(missing)]["state"] == s6.BACKUP_ABSENT
    assert s6._snapshot_has_failure(snap) is False


# ---------------------------------------------------------------- 摘要贯穿

def test_file_state_distinguishes_error_from_absent(tmp_path):
    """ERROR（存在但读不出）不能变成 ABSENT —— 这是贯穿的前提。"""
    from publish_journal import file_state
    d = tmp_path / "adir"
    d.mkdir()
    assert file_state(str(d))[0] == "ERROR"
    assert file_state(str(tmp_path / "nope"))[0] == "ABSENT"
