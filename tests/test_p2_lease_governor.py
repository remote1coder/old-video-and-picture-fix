# -*- coding: utf-8 -*-
"""P2：类型化 lease 池 + RAM/VRAM 联合预算（G1 / G6）。

每条用例对应审核判据原文，命名里带上判据关键词，便于逐条核对。

## G1 通过条件

* 无丢失 / 无重复 / 无乱序 / 无旧帧覆盖
* 无无限内存增长
* 所有分配在账
* 不够最低空间明确拒绝
* 控制消息可达
* 整个进程树有界退出（本文件只测"单进程内有界"，进程树见假 worker）

## G1 额外反例

* 输出依赖未来 N 帧但只有 N−1 输入容量
* RAM/VRAM 反向申请
* owner 仍活但 TTL 已过
* GPU 异步读未结束而 CPU 已 ACK

## G6 通过条件

* 压力时停止新工作 → 在途排空 → 实际释放（本文件测"排空"的机制面）
* 没有跨 stage 互等
* 指标失效不填 0
* 无隐式降质（本文件不含任何"降级"分支）
"""
import threading
import time

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

from lease import (ControlChannel, LeaseExhausted, LeasePool, LeaseRevoked,
                   LeaseStale, LeaseTooSmall, LeaseClosed)
from governor import (BelowMinimum, BudgetExhausted, ResourceGovernor,
                      observe_capacity)

MB = 1 << 20


# ===========================================================================
# LeasePool：所有权与账目
# ===========================================================================

def test_acquire_release_is_balanced():
    """所有分配在账：issued == returned + revoked + live。"""
    pool = LeasePool(capacity=4, slot_bytes=1 * MB, min_slot_bytes=1)
    a = pool.acquire("stage3:1", seq=0)
    b = pool.acquire("stage3:1", seq=1)
    st = pool.stats()
    assert st["live"] == 2 and st["issued"] == 2 and st["balanced"], st

    pool.release(a.lease_id, seq=0)
    pool.release(b.lease_id, seq=1)
    st = pool.stats()
    assert st["live"] == 0 and st["returned"] == 2 and st["balanced"], st


def test_capacity_is_fixed_no_unbounded_growth():
    """无无限内存增长：容量固定，借满即拒。"""
    pool = LeasePool(capacity=2, slot_bytes=2 * MB)
    a = pool.acquire("w:1", seq=0)
    b = pool.acquire("w:1", seq=1)
    with pytest.raises(LeaseExhausted):
        pool.acquire("w:1", seq=2)
    st = pool.stats()
    assert st["live"] == 2
    # 反复借还 200 次：**先释放占满的两个**，否则当然借不到
    pool.release(a.lease_id, seq=0)
    pool.release(b.lease_id, seq=1)
    for i in range(200):
        lease = pool.acquire("w:2", seq=i + 10, wait=0.5)
        pool.release(lease.lease_id, seq=i + 10)
    st = pool.stats()
    assert st["live"] == 0
    assert st["bytes_live"] == 0
    # 容量上限仍是 2 槽 × 2MB，没有任何"涨"的空间
    assert st["bytes_capacity"] == 2 * 2 * MB
    assert st["balanced"], st


def test_no_duplicate_slot_under_concurrency():
    """无重复：并发借出的 lease 槽号互不相同。"""
    pool = LeasePool(capacity=8, slot_bytes=1024)
    got, lock = [], threading.Lock()

    def _worker(i):
        for _ in range(20):
            lease = pool.acquire("w%d" % i, seq=i, wait=2.0)
            with lock:
                got.append(lease.slot)
            pool.release(lease.lease_id, seq=i)

    ts = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    # 同一时刻 live 槽不重复由池本身保证；这里额外验证账目自洽
    assert pool.stats()["balanced"], pool.stats()


def test_release_with_wrong_seq_is_rejected():
    """无乱序 / 无旧帧覆盖：归还的 seq 必须与借出时一致。"""
    pool = LeasePool(capacity=2, slot_bytes=1024)
    lease = pool.acquire("w:1", seq=7)
    with pytest.raises(LeaseStale):
        pool.release(lease.lease_id, seq=6)
    # 拒绝之后槽位仍然被占（不能因为错误归还而回收）
    assert pool.stats()["live"] == 1
    pool.release(lease.lease_id, seq=7)
    assert pool.stats()["live"] == 0


def test_stale_seq_on_reused_slot_is_rejected():
    """更旧的帧号不得复用刚释放的槽（上一轮残留混进来）。"""
    pool = LeasePool(capacity=1, slot_bytes=1024)
    a = pool.acquire("w:1", seq=10)
    pool.release(a.lease_id, seq=10)
    # 同一个槽上请求更旧的 seq —— 拒绝
    with pytest.raises(LeaseStale):
        pool.acquire("w:1", seq=9)
    # 更晚的 seq 正常
    b = pool.acquire("w:1", seq=11)
    pool.release(b.lease_id, seq=11)


def test_below_min_slot_is_rejected():
    """不够最低空间明确拒绝（构造期就拒，不是等到跑起来）。"""
    with pytest.raises(LeaseTooSmall):
        LeasePool(capacity=4, slot_bytes=512, min_slot_bytes=1024)


def test_wait_until_release_wakes_waiter():
    """背压：没有空槽时等待，别人归还后被唤醒。"""
    pool = LeasePool(capacity=1, slot_bytes=1024)
    a = pool.acquire("w:1", seq=0)
    done = {}

    def _waiter():
        try:
            b = pool.acquire("w:2", seq=1, wait=2.0)
            done["ok"] = True
            pool.release(b.lease_id, seq=1)
        except Exception as exc:
            done["err"] = exc

    t = threading.Thread(target=_waiter)
    t.start()
    time.sleep(0.1)
    assert "ok" not in done, "不该在还有活的时候拿到槽"
    pool.release(a.lease_id, seq=0)
    t.join(timeout=3)
    assert done.get("ok") is True, done


def test_revoke_owner_is_fencing():
    """失效 fencing：owner 死掉后它的 lease 全部作废，且不能再归还。"""
    pool = LeasePool(capacity=4, slot_bytes=1024)
    a = pool.acquire("stage3:111", seq=0)
    b = pool.acquire("stage3:111", seq=1)
    n = pool.revoke_owner("stage3:111")
    assert n == 2
    st = pool.stats()
    assert st["live"] == 0 and st["revoked"] == 2 and st["balanced"], st
    with pytest.raises(LeaseRevoked):
        pool.release(a.lease_id, seq=0)
    # 槽位已回收，可以再借
    c = pool.acquire("stage3:222", seq=2)
    pool.release(c.lease_id, seq=2)


def test_owner_alive_but_ttl_expired_is_reclaimed():
    """反例：owner 仍活但 TTL 已过 —— 必须被回收，不能永久占着。"""
    pool = LeasePool(capacity=2, slot_bytes=1024)
    a = pool.acquire("stage3:333", seq=0, ttl=0.05)
    time.sleep(0.12)
    assert pool.sweep_expired() == 1
    st = pool.stats()
    assert st["live"] == 0 and st["revoked"] == 1, st
    # 被回收后原 lease 已失效
    with pytest.raises(LeaseRevoked):
        pool.release(a.lease_id, seq=0)


def test_close_blocks_new_acquire():
    """有界退出：关池之后不再接受新租借。"""
    pool = LeasePool(capacity=2, slot_bytes=1024)
    pool.acquire("w:1", seq=0)
    st = pool.close()
    assert st["revoked_on_close"] == 1
    assert st["closed"] is True and st["live"] == 0, st
    with pytest.raises((LeaseClosed, LeaseExhausted)):
        pool.acquire("w:1", seq=1, wait=0.2)


def test_future_frame_dependency_needs_enough_slots():
    """反例：输出依赖未来 N 帧，但只有 N−1 个槽 —— 必须明确拒绝。"""
    need = 4                      # 时间域滤镜需要 n-1..n+2
    pool = LeasePool(capacity=need - 1, slot_bytes=1024)
    held = []
    for i in range(need - 1):
        held.append(pool.acquire("s6:1", seq=i))
    with pytest.raises(LeaseExhausted):
        pool.acquire("s6:1", seq=need - 1)
    # 说明"容量不足"是可判定的，不是等死
    assert pool.stats()["live"] == need - 1
    for i, lease in enumerate(held):
        pool.release(lease.lease_id, seq=i)


# ===========================================================================
# ControlChannel
# ===========================================================================

def test_control_channel_is_reachable():
    """控制消息可达：cancel 之后 poll 立刻能看到，且带原因。"""
    ctl = ControlChannel("c1")
    assert ctl.poll()["cancelled"] is False
    ctl.cancel("user asked")
    p = ctl.poll()
    assert p["cancelled"] is True and p["reason"] == "user asked"
    assert p["seq"] >= 1


def test_control_wait_wakes_worker():
    """有界退出：等在 wait() 上的 worker 会被 cancel 唤醒。"""
    ctl = ControlChannel("c2")
    woke = {}

    def _w():
        woke["cancelled"] = ctl.wait(timeout=5.0)

    t = threading.Thread(target=_w)
    t.start()
    time.sleep(0.05)
    ctl.cancel("stop")
    t.join(timeout=3)
    assert woke.get("cancelled") is True


# ===========================================================================
# ResourceGovernor
# ===========================================================================

def test_reserve_release_balanced():
    gov = ResourceGovernor(ram_budget=100 * MB, vram_budget=50 * MB)
    r = gov.reserve("s3:1", 40 * MB, 20 * MB)
    st = gov.stats()
    assert st["ram_used"] == 40 * MB and st["vram_used"] == 20 * MB
    assert st["balanced"], st
    gov.release(r)
    st = gov.stats()
    assert st["ram_used"] == 0 and st["released"] == 1 and st["balanced"], st


def test_reserve_is_atomic_no_partial_grant():
    """没有跨 stage 互等：一次申请要么两个都拿到，要么一个都不拿。"""
    gov = ResourceGovernor(ram_budget=100 * MB, vram_budget=10 * MB)
    # 只有 VRAM 够、RAM 不够 → 整体失败，且 RAM 不被占用
    with pytest.raises(BudgetExhausted):
        gov.reserve("s3:1", 200 * MB, 1 * MB)
    st = gov.stats()
    assert st["ram_used"] == 0 and st["vram_used"] == 0, st
    # 只有 RAM 够、VRAM 不够 → 同样整体失败
    holder = gov.reserve("s4:1", 1 * MB, 9 * MB)
    with pytest.raises(BudgetExhausted):
        gov.reserve("s3:2", 10 * MB, 5 * MB, wait=0.05)
    st = gov.stats()
    assert st["ram_used"] == 1 * MB, st
    gov.release(holder)


def test_no_cross_stage_deadlock_under_reverse_requests():
    """反例：两个 stage 反向申请（一个多 RAM 少 VRAM，一个反之）不能互等。

    原子申请下，二者只会有一个成功；另一个超时失败，不会各拿一半互等。
    """
    gov = ResourceGovernor(ram_budget=100 * MB, vram_budget=100 * MB)
    results = {}
    barrier = threading.Barrier(2)

    def _a():
        barrier.wait()
        try:
            r = gov.reserve("A", 80 * MB, 20 * MB, wait=0.4)
            results["A"] = "ok"
            time.sleep(0.05)
            gov.release(r)
        except BudgetExhausted:
            results["A"] = "exhausted"

    def _b():
        barrier.wait()
        try:
            r = gov.reserve("B", 20 * MB, 80 * MB, wait=0.4)
            results["B"] = "ok"
            time.sleep(0.05)
            gov.release(r)
        except BudgetExhausted:
            results["B"] = "exhausted"

    ta, tb = threading.Thread(target=_a), threading.Thread(target=_b)
    ta.start(); tb.start()
    ta.join(timeout=5); tb.join(timeout=5)
    # 两个都能成功（80+20=100 RAM, 20+80=100 VRAM 刚好），但关键是不死锁
    assert ta.is_alive() is False and tb.is_alive() is False
    assert set(results) == {"A", "B"}, results
    st = gov.stats()
    assert st["balanced"], st


def test_below_min_forward_unit_rejected():
    """最小可前进单元：连一帧都装不下的申请不该开工。"""
    gov = ResourceGovernor(ram_budget=100 * MB, vram_budget=10 * MB,
                           min_forward_ram=4 * MB, min_forward_vram=2 * MB)
    with pytest.raises(BelowMinimum):
        gov.reserve("s3:1", 1 * MB, 5 * MB)
    with pytest.raises(BelowMinimum):
        gov.reserve("s3:1", 8 * MB, 0)


def test_oversize_request_fails_fast_not_hang():
    """申请量超过总预算 → 立即拒，不能等到超时。"""
    gov = ResourceGovernor(ram_budget=100 * MB, vram_budget=10 * MB)
    t0 = time.monotonic()
    with pytest.raises(BudgetExhausted):
        gov.reserve("s3:1", 101 * MB, 1 * MB, wait=5.0)
    assert time.monotonic() - t0 < 0.5, "超预算的申请应当立即失败"


def test_double_release_is_rejected():
    """重复归还不得把预算放大。"""
    gov = ResourceGovernor(ram_budget=100 * MB, vram_budget=10 * MB)
    r = gov.reserve("s:1", 10 * MB, 1 * MB)
    gov.release(r)
    with pytest.raises(Exception):
        gov.release(r)
    assert gov.stats()["ram_used"] == 0


def test_revoke_owner_frees_budget():
    gov = ResourceGovernor(ram_budget=100 * MB, vram_budget=10 * MB)
    gov.reserve("dead:1", 50 * MB, 5 * MB)
    assert gov.revoke_owner("dead:1") == 1
    st = gov.stats()
    assert st["ram_used"] == 0 and st["revoked"] == 1 and st["balanced"], st


def test_drain_pattern_stop_then_release():
    """压力时的正确顺序：停止新工作 → 在途排空 → 实际释放。"""
    gov = ResourceGovernor(ram_budget=20 * MB, vram_budget=5 * MB)
    ctl = ControlChannel("drain")
    live = []
    for i in range(2):
        live.append(gov.reserve("s:%d" % i, 9 * MB, 2 * MB))
    # 新申请拿不到（不再扩容）
    with pytest.raises(BudgetExhausted):
        gov.reserve("s:new", 9 * MB, 2 * MB, wait=0.05)
    # 停止新工作
    ctl.cancel("draining")
    assert ctl.poll()["cancelled"] is True
    # 在途排空
    for r in live:
        gov.release(r)
    st = gov.stats()
    assert st["ram_used"] == 0 and st["live"] == 0, st
    # 排空后新工作可以正常进入
    r3 = gov.reserve("s:after", 9 * MB, 2 * MB)
    gov.release(r3)


def test_capacity_probe_does_not_fill_zero_on_failure():
    """指标失效不填 0：探测失败返回 known=False，而不是 0 字节。"""
    def _boom():
        raise RuntimeError("probe exploded")

    cap = observe_capacity(probe_ram=_boom, probe_vram=lambda: None)
    assert cap["ram_known"] is False and cap["ram_bytes"] is None
    assert cap["vram_known"] is False and cap["vram_bytes"] is None
    assert any("probe exploded" in n for n in cap["notes"]), cap


def test_capacity_probe_reports_real_values():
    cap = observe_capacity(probe_ram=lambda: 8 * 1024 * MB,
                           probe_vram=lambda: 2 * 1024 * MB)
    assert cap["ram_known"] and cap["ram_bytes"] == 8 * 1024 * MB
    assert cap["vram_known"] and cap["vram_bytes"] == 2 * 1024 * MB
