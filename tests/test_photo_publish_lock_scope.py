# -*- coding: utf-8 -*-
"""PHOTO-01：发布锁必须在恢复 / 备份 / journal 写入之前获取。

原实现把锁放在"改任何目标之前"（提交段前），但：
  - `_recover_prior_publish()`（L1372）会**恢复/清理文件**
  - 备份（L1456 附近）与 journal 写入（L1507 附近）也在锁之前
两个任务写同一输出时，后来的任务可以在前一个持锁期间恢复/清理它的事务，
或用陈旧快照覆盖 journal —— 后续提交段互斥保护不了**已经发生的**改动。

设计依据（联网核对）：崩溃恢复应作为"每次启动的标准前导"在
**单一 mutation lock 之内**执行；锁按确定顺序获取，用 try/finally 释放。

本测试是**结构护栏**：这段逻辑内联在 main() 里且有 4 处提前 return，
不便直接跑并发；这里锁住"顺序对了"和"每处 return 都释放"。
"""
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "photo_restoration_pipeline", "stages",
                    "stage6_encode.py")


def _text():
    return open(_SRC, encoding="utf-8", errors="replace").read()


def test_lock_acquired_before_recovery():
    """acquire_target_locks 必须出现在 _recover_prior_publish 之前。"""
    t = _text()
    # 注意：`acquire_target_locks(` 也出现在 import 行，必须找调用处
    i_lock = t.find("_lock_cm = acquire_target_locks(")
    i_rec = t.find("_prior = _recover_prior_publish(")
    assert i_lock != -1 and i_rec != -1
    # _recover_prior_publish 还有函数定义处，取**调用**处
    # 注意：`_recover_prior_publish(work_output_path` 也匹配**函数定义**，
    # 所以用 `_prior = ` 前缀锁定真正的调用点。
    assert i_rec != -1, "找不到恢复调用点"
    assert i_lock < i_rec, "锁必须在恢复之前获取"


def test_lock_acquired_before_backup_and_journal():
    t = _text()
    i_lock = t.find("_lock_cm = acquire_target_locks(")
    i_bk = t.find('"publish_backup"')
    i_jp = t.find("write_journal(_journal_path")
    assert i_bk != -1 and i_jp != -1
    assert i_lock < i_bk, "锁必须在备份失败分支之前获取"
    assert i_lock < i_jp, "锁必须在 journal 写入之前获取"


def test_every_early_return_in_guarded_region_releases():
    """锁获取点之后、提交段 try 之前，每个 return 前都要有 _rel_lock()。"""
    t = _text()
    i_lock = t.find('_lock_holder["cm"] = _lock_cm')
    i_rel_def = t.find("def _rel_lock(")
    assert i_lock != -1 and i_rel_def != -1
    # 守卫区间：锁获取之后 ~ 提交段（用 STAGE6_JOURNAL_PREPARED 作右界）
    i_right = t.find("STAGE6_JOURNAL_PREPARED")
    assert i_right != -1
    seg = t[i_lock:i_right]
    returns = [m.start() for m in re.finditer(r"^\s+return \d+\s*$", seg, re.M)]
    assert returns, "守卫区间内应有提前 return（否则测试失去意义）"
    for pos in returns:
        window = seg[max(0, pos - 260):pos]
        # 锁超时分支里锁**没拿到**，不该也不需要释放
        if "publish_lock" in window:
            continue
        assert "_rel_lock()" in window, \
            "提前 return 前没有释放锁：" + seg[max(0, pos - 80):pos + 20]


def test_release_is_idempotent_by_construction():
    """_rel_lock 必须能重复调用（置空后再调用是空操作）。"""
    t = _text()
    i = t.find("def _rel_lock(")
    body = t[i:i + 400]
    assert "_lock_holder" in body
    assert '.get("cm")' in body, "应通过 holder 判空，避免重复释放"
    assert '_lock_holder["cm"] = None' in body, "释放后必须置空"


def test_commit_stage_uses_same_release_helper():
    """提交段结束也走 _rel_lock（不能直接 __exit__ 造成双重释放）。"""
    t = _text()
    assert "_lock_cm.__exit__(None, None, None)" not in t, \
        "提交段应改用可重入的 _rel_lock()"
