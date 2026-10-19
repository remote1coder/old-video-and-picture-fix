"""PUB-04（第七轮）：目标集合锁必须按"单路径"互斥，不能按集合摘要。

任务书反例：`{AVIF, TIFF}` 与 `{同一 AVIF}` 必须互斥。
若用"整个集合的摘要"作锁，两者摘要不同 → 各自拿到不同的锁 →
同时写同一个 AVIF，互斥失效。

做法：每个目标路径一把独立锁，按字典序固定顺序获取。
"""

import os
import sys
import threading
import time

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "shared_modules"))

import publish_lock as pl  # noqa: E402


def test_normalize_dedups_and_sorts():
    a, b = "b.avif", "a.avif"
    got = pl.normalize_targets([a, b, a, None, ""])
    assert len(got) == 2
    assert got == sorted(got), "必须排序（固定顺序防死锁）"


def test_lock_path_is_next_to_target(tmp_path):
    """锁文件必须与目标同目录，否则跨 job（temp_dir 不同）不互斥。"""
    t = tmp_path / "out" / "x.avif"
    t.parent.mkdir()
    lp = pl.lock_path_for(str(t))
    assert os.path.dirname(lp) == str(t.parent)


def test_overlapping_sets_block_each_other(tmp_path):
    """核心：{A,B} 与 {A} 必须互斥。"""
    a = tmp_path / "out.avif"
    b = tmp_path / "out.tif"
    a.write_bytes(b"x")
    b.write_bytes(b"y")

    inside = threading.Event()
    release = threading.Event()
    blocked_done = threading.Event()
    order = []

    def holder():
        with pl.acquire_target_locks([str(a), str(b)], timeout=5):
            inside.set()
            release.wait(5)

    def contender():
        inside.wait(5)
        try:
            with pl.acquire_target_locks([str(a)], timeout=0.5):
                order.append("got")
        except pl.TargetLockTimeout:
            order.append("blocked")
        blocked_done.set()

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=contender)
    t1.start(); t2.start()
    blocked_done.wait(10)
    release.set()
    t1.join(5); t2.join(5)

    assert order == ["blocked"], (
        f"重叠目标集合没有互斥（实际 {order}）—— "
        "这正是'集合摘要作锁'会犯的错")


def test_disjoint_sets_do_not_block(tmp_path):
    """不相交的目标不应互相阻塞（不能退化成全局锁）。"""
    a = tmp_path / "a.avif"
    b = tmp_path / "b.avif"
    a.write_bytes(b"x"); b.write_bytes(b"y")

    ok = []
    with pl.acquire_target_locks([str(a)], timeout=2):
        with pl.acquire_target_locks([str(b)], timeout=0.5):
            ok.append("both")
    assert ok == ["both"]


def test_same_thread_reacquire_times_out(tmp_path):
    """同一进程重复获取同一目标（未释放）应超时，而不是静默通过。"""
    a = tmp_path / "a.avif"
    a.write_bytes(b"x")
    with pl.acquire_target_locks([str(a)], timeout=2):
        with pytest.raises(pl.TargetLockTimeout):
            with pl.acquire_target_locks([str(a)], timeout=0.3):
                pass


def test_locks_released_after_exception(tmp_path):
    """异常退出时锁必须释放，否则后续运行永久阻塞。"""
    a = tmp_path / "a.avif"
    a.write_bytes(b"x")
    with pytest.raises(ValueError):
        with pl.acquire_target_locks([str(a)], timeout=2):
            raise ValueError("simulated failure")

    with pl.acquire_target_locks([str(a)], timeout=1):
        pass  # 能进来就说明锁已释放


def test_all_or_nothing(tmp_path):
    """拿不到第二个锁时，第一个锁也要释放（不全持有）。"""
    a = tmp_path / "a.avif"
    b = tmp_path / "b.avif"
    a.write_bytes(b"x"); b.write_bytes(b"y")

    # 先独占 b
    with pl.acquire_target_locks([str(b)], timeout=2):
        with pytest.raises(pl.TargetLockTimeout):
            # 尝试同时拿 a+b → b 拿不到 → 必须整体失败
            with pl.acquire_target_locks([str(a), str(b)], timeout=0.3):
                pass
        # a 不能仍被持有
        with pl.acquire_target_locks([str(a)], timeout=1):
            pass
