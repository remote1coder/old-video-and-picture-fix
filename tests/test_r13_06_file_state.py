# -*- coding: utf-8 -*-
"""R13-06：不可读与不存在被合并成一个 None。

**缺陷**

`file_digest()` 对 FileNotFoundError / PermissionError / IsADirectoryError /
OSError **一律返回 None**。于是调用方无法区分：

- "旧代本来就没有这个文件"（ABSENT，正常，可以新建）
- "文件就在那儿但读不出来"（ERROR，状态未知，**必须保留现场**）

后果：
1. `_snapshot_for_rollback()` 复制成功后即使摘要读不到，仍可能记 BACKUP_OK；
2. 回滚只在"记录摘要非空"时校验 —— 读不出就**跳过检查**并报"已恢复"；
3. `classify_recovery()` 把"读不出来"当成 ABSENT_OK，可能被算成完整旧代。

**修法**：`file_state()` 返回 PRESENT / ABSENT / ERROR 三态；
分类器与回滚预检都按三态处理，ERROR 一律视为"状态未知"。
"""
import hashlib
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "photo_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import publish_journal as pj  # noqa: E402
import stage6_encode as s6  # noqa: E402


# ------------------------------------------------ file_state 三态本身

def test_file_state_present_absent_error(tmp_path):
    """PRESENT / ABSENT / ERROR 必须可区分。"""
    f = tmp_path / "ok.bin"
    f.write_bytes(b"abc")
    st, digest = pj.file_state(str(f))
    assert st == "PRESENT"
    assert digest == hashlib.sha256(b"abc").hexdigest()

    assert pj.file_state(str(tmp_path / "nope.bin"))[0] == "ABSENT"

    # 目录：存在但不是普通文件 → ERROR（不是 ABSENT）
    d = tmp_path / "adir"
    d.mkdir()
    assert pj.file_state(str(d))[0] == "ERROR"


def test_file_digest_still_returns_none_for_all_cases(tmp_path):
    """旧接口保持兼容（调用点很多），三态由 file_state 提供。"""
    d = tmp_path / "adir"
    d.mkdir()
    assert pj.file_digest(str(d)) is None
    assert pj.file_digest(str(tmp_path / "nope")) is None


# ------------------------------------------------ 分类器不得把 ERROR 当 ABSENT

def test_classify_recovery_unreadable_is_unknown_not_absent(tmp_path):
    """目标是目录（存在但不可读）→ UNKNOWN，不能算 ABSENT_OK。

    旧实现 `cur is None and existed is False` → ABSENT_OK，
    于是一个"读不出来"的目标会让整组被判成完整旧代。
    """
    d = tmp_path / "slot.avif"
    d.mkdir()                      # 占位成目录，模拟"存在但读不出"
    path = str(d)
    doc = {
        "phase": "PREPARED", "attempt_id": "t", "targets": [path],
        "records": {path: {"existed": False, "old_digest": None,
                           "new_digest": hashlib.sha256(b"X").hexdigest(),
                           "backup": None, "backup_digest": None,
                           "staging": None}},
    }
    rep = pj.classify_recovery(doc)
    assert rep["targets"][path]["state"] == "UNKNOWN", rep["targets"][path]
    assert rep["targets"][path]["file_state"] == "ERROR"
    assert rep["has_unknown"] is True
    assert rep["all_old"] is False, "读不出来 ≠ 完整旧代"
    assert rep["recommendation"].startswith("KEEP_AND_ASK")


def test_classify_recovery_absent_still_absent_ok(tmp_path):
    """真正不存在时才走 ABSENT_OK（对照，防止过度收紧）。"""
    path = str(tmp_path / "missing.avif")
    doc = {
        "phase": "PREPARED", "attempt_id": "t", "targets": [path],
        "records": {path: {"existed": False, "old_digest": None,
                           "new_digest": hashlib.sha256(b"X").hexdigest(),
                           "backup": None, "backup_digest": None,
                           "staging": None}},
    }
    rep = pj.classify_recovery(doc)
    assert rep["targets"][path]["state"] == "ABSENT_OK"
    assert rep["targets"][path]["file_state"] == "ABSENT"


# ------------------------------------------------ 回滚预检拒绝不可读目标

def test_rollback_refuses_unreadable_target(tmp_path):
    """当前目标读不出来 → 预检失败，**不许**改动它。"""
    d = tmp_path / "slot.avif"
    d.mkdir()
    bk = tmp_path / "0_slot.avif"
    bk.write_bytes(b"OLD")
    old = hashlib.sha256(b"OLD").hexdigest()

    snapshot = {
        str(d): {"state": s6.BACKUP_OK, "backup": str(bk),
                 "old_digest": old, "backup_digest": old},
    }
    rb = s6._rollback_publish(snapshot, None)
    assert rb["precheck_failed"] is True, rb
    assert rb["fully_restored"] is False
    assert d.is_dir(), "不可读的目标不能被替换掉"
