# -*- coding: utf-8 -*-
"""R12-02：恢复必须"先验完整套备份，再改动任何文件"。

**审核报告指出的三条缺口**

1. 某一备份校验失败后，仍调用 `_rollback_publish()` 去恢复**其他**目标 ——
   目标被改掉、好备份被消耗，然后才返回 needs_human；
   日志里"leaving everything untouched"不成立。
2. 普通异常/取消走内存 snapshot，而 `_rollback_publish()` 自身不核摘要 ——
   坏备份能覆盖仍然完好的目标，却报 `fully_restored=True`。
3. 回滚消耗备份后仍留 PREPARED journal，下次启动即使所有目标都是完整
   旧代，也要求那些**已被消耗**的备份 → 卡进 needs_human。

**修法**

- 阶段一：全组预检（只读）。任一目标不可满足 → `precheck_failed=True`
  并**直接返回，一个字节都不动**。
- 已是旧代（`old_digest` 匹配）→ 跳过，不需要备份 → 解决缺口 3。
- 备份必须同时满足 `backup_digest` 与 `old_digest` 才可用于恢复。
- 阶段二：执行；恢复后用 `old_digest` 复核。

本测试全部使用**真实文件**，直接调生产函数 `_rollback_publish()`。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "photo_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage6_encode as s6  # noqa: E402


def _w(path, data: bytes) -> str:
    with open(path, "wb") as f:
        f.write(data)
    return str(path)


def _d(path):
    return s6.file_digest(str(path))


def test_bad_backup_aborts_before_touching_anything(tmp_path):
    """缺口 1：一份坏备份 → **其他目标与好备份都不许被动**。"""
    a = tmp_path / "a.avif"
    b = tmp_path / "b.avif"
    _w(a, b"NEW-A")          # 已被改坏（当前内容不是旧代）
    _w(b, b"NEW-B")
    bk_a = tmp_path / "0_a.avif"
    bk_b = tmp_path / "1_b.avif"
    _w(bk_a, b"OLD-A")       # A 的好备份
    _w(bk_b, b"TAMPERED")    # B 的坏备份

    snapshot = {
        str(a): {"state": s6.BACKUP_OK, "backup": str(bk_a),
                 "old_digest": s6.file_digest(str(bk_a)),
                 "backup_digest": s6.file_digest(str(bk_a))},
        str(b): {"state": s6.BACKUP_OK, "backup": str(bk_b),
                 "old_digest": "0000deadbeef",       # 与坏备份不符
                 "backup_digest": "1111cafebabe"},
    }
    a_before, bk_a_before = a.read_bytes(), bk_a.read_bytes()

    rb = s6._rollback_publish(snapshot, None)

    assert rb["precheck_failed"] is True, rb
    assert rb["fully_restored"] is False
    assert a.read_bytes() == a_before, "A 不能被改 (缺口 1)"
    assert bk_a.read_bytes() == bk_a_before, "A 的好备份不能被消耗"


def test_already_old_with_consumed_backup_is_noop_success(tmp_path):
    """缺口 3：目标已是完整旧代、备份已被消耗 → 应无操作确认成功。"""
    a = tmp_path / "a.avif"
    _w(a, b"OLD-A")
    snapshot = {
        str(a): {"state": s6.BACKUP_OK,
                 "backup": str(tmp_path / "gone" / "0_a.avif"),   # 不存在
                 "old_digest": _d(a),
                 "backup_digest": _d(a)},
    }
    rb = s6._rollback_publish(snapshot, None)
    assert rb["fully_restored"] is True, rb
    assert rb["skipped"] == [str(a)]
    assert a.read_bytes() == b"OLD-A"


def test_bad_backup_never_reports_fully_restored(tmp_path):
    """缺口 2：坏备份绝不能覆盖好目标，也不能报 fully_restored=True。"""
    a = tmp_path / "a.avif"
    _w(a, b"STILL-GOOD-OLD")     # 目标其实还是完好的旧代
    bk = tmp_path / "0_a.avif"
    _w(bk, b"BROKEN")
    snapshot = {
        str(a): {"state": s6.BACKUP_OK, "backup": str(bk),
                 # 这里故意不给 old_digest 匹配（模拟摘要不可信的场景）
                 "old_digest": "aaaabbbbcccc",
                 "backup_digest": "ddddeeeeffff"},
    }
    rb = s6._rollback_publish(snapshot, None)
    assert rb["fully_restored"] is False, rb
    assert a.read_bytes() == b"STILL-GOOD-OLD", "好目标不能被坏备份覆盖"


def test_good_backup_restores_and_verifies(tmp_path):
    """正常路径：好备份能把目标恢复回旧代，并核对摘要。"""
    a = tmp_path / "a.avif"
    _w(a, b"NEW-A")
    bk = tmp_path / "0_a.avif"
    _w(bk, b"OLD-A")
    snapshot = {
        str(a): {"state": s6.BACKUP_OK, "backup": str(bk),
                 "old_digest": s6.file_digest(str(bk)),
                 "backup_digest": s6.file_digest(str(bk))},
    }
    rb = s6._rollback_publish(snapshot, None)
    assert rb["fully_restored"] is True, rb
    assert rb["restored"] == [str(a)]
    assert a.read_bytes() == b"OLD-A"
    # 备份**不**被消耗：整组确认完成前保持可重试
    assert bk.exists(), "copyfile 恢复不应消耗备份"


def test_absent_target_is_removed(tmp_path):
    """本次新建的目标，回滚时删除。"""
    a = tmp_path / "a.avif"
    _w(a, b"NEW-A")
    snapshot = {str(a): {"state": s6.BACKUP_ABSENT, "backup": None,
                         "old_digest": None, "backup_digest": None}}
    rb = s6._rollback_publish(snapshot, None)
    assert rb["fully_restored"] is True
    assert not a.exists()


def test_missing_backup_aborts_without_modifying(tmp_path):
    """声明有备份但文件不在 → 预检失败，不动任何文件。"""
    a = tmp_path / "a.avif"
    _w(a, b"NEW-A")
    snapshot = {str(a): {"state": s6.BACKUP_OK,
                         "backup": str(tmp_path / "nope.bak"),
                         "old_digest": "x", "backup_digest": "y"}}
    rb = s6._rollback_publish(snapshot, None)
    assert rb["precheck_failed"] is True
    assert a.read_bytes() == b"NEW-A"
