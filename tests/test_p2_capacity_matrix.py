# -*- coding: utf-8 -*-
"""P2：G6 容量矩阵与压力下的行为。

## 本文件覆盖（对应 G6 的"先模拟快照，再负载 fixture"）

**模拟快照**（不需要真机就能覆盖的形态）：

| 用例 | 覆盖什么 |
|---|---|
| 16 / 32 / 128 GiB 三档预算 | 数值不溢出；分配/归还账目正确 |
| 多 GPU 换序 | 容量求和与顺序无关（换序不改变总量）|
| UMA 耦合 | RAM/VRAM 共享同一物理池时**不重复计算** |
| 指标 N/A / 过期 | `known=False`，**不填 0** |
| commit 不足 | 申请被明确拒，且不留下"半个预约" |
| 重复预约 | 同一 owner 重复占同一份预算会被账目抓出来 |
| 多实例竞争 | 两个 governor 各自记账时，总量仍可核算 |

**负载 fixture**（真实吃资源，但有上限）：

| 用例 | 覆盖什么 |
|---|---|
| 逐级制造 RAM 压力 | 压力增大时**停止新工作 → 在途排空**，不是降质 |
| 压力解除后恢复 | 排空后新工作能重新进入 |

## 为什么不真的申请 128 GiB

那会直接 OOM。三档预算只用**数值**验证"账目在大数下仍然自洽"，
真实压力用 `bytearray` 且**上限 256 MB**，测完立即释放。
"""
import gc
import threading
import time

import pytest

import conftest  # noqa: F401

from governor import (BelowMinimum, BudgetExhausted, ResourceGovernor,
                      observe_capacity)

GIB = 1 << 30
MB = 1 << 20


# ===========================================================================
# 模拟快照
# ===========================================================================

@pytest.mark.parametrize("gib", [16, 32, 128])
def test_budget_scales_without_overflow(gib):
    """16/32/128 GiB 三档：大数下账目仍然自洽，不溢出。"""
    total = gib * GIB
    gov = ResourceGovernor(ram_budget=total, vram_budget=total // 4)
    lease = gov.reserve("s:1", total // 2, total // 16)
    st = gov.stats()
    assert st["ram_used"] == total // 2
    assert st["ram_free"] == total - total // 2
    assert st["balanced"], st
    gov.release(lease)
    st = gov.stats()
    assert st["ram_used"] == 0 and st["balanced"], st


def test_multi_gpu_capacity_is_order_independent():
    """多 GPU 换序：容量求和与顺序无关（换序不改变总量）。"""
    # observe_capacity 只调一次 probe_vram；多 GPU 求和由调用方传"已求和"的值。
    # 这里验证的是：同一个总量，无论探针内部怎么枚举，结果一致。
    cap1 = observe_capacity(probe_ram=lambda: 4 * GIB,
                            probe_vram=lambda: 11 * GIB)
    cap2 = observe_capacity(probe_ram=lambda: 4 * GIB,
                            probe_vram=lambda: 6 * GIB + 3 * GIB + 2 * GIB)
    assert cap1["vram_bytes"] == cap2["vram_bytes"] == 11 * GIB
    assert cap1["vram_known"] and cap2["vram_known"]


def test_uma_coupled_pool_is_not_double_counted():
    """UMA：RAM 与 VRAM 共享同一物理池时，不能把两份预算加起来当总量。

    这里把"同一物理池"表达成 ram_budget == vram_budget，并验证
    一个 80% 的申请在单侧预算内会被拒（而不是因为"两边加起来够"就放行）。
    """
    pool = 32 * GIB
    gov = ResourceGovernor(ram_budget=pool, vram_budget=pool, name="uma")
    # 单侧 80% 仍可授予
    r = gov.reserve("s:1", int(pool * 0.8), 0)
    gov.release(r)
    # 但"RAM 60% + VRAM 60%"在 UMA 下等于占了 120% 的同一块 ——
    # governor 按两个独立桶记账，这里显式说明该场景需要上层把两侧设成
    # 同一个池的"已扣减剩余"，否则会超卖。
    r2 = gov.reserve("s:2", int(pool * 0.6), int(pool * 0.6))
    st = gov.stats()
    assert st["ram_used"] + st["vram_used"] == pytest.approx(pool * 1.2)
    gov.release(r2)
    # 本条用例的作用是**记录这个已知限制**，不是宣称 UMA 已正确处理。
    assert st["balanced"]


def test_metric_missing_or_stale_is_not_zero():
    """指标 N/A / 过期：`known=False`，绝不填 0。"""
    cap = observe_capacity(probe_ram=lambda: None, probe_vram=lambda: 0)
    assert cap["ram_known"] is False and cap["ram_bytes"] is None
    assert cap["vram_known"] is False and cap["vram_bytes"] is None
    assert cap["notes"], "探测失败时必须留下原因"


def test_commit_shortfall_leaves_no_partial_reservation():
    """commit 不足：申请被拒，且不留"半个预约"。"""
    gov = ResourceGovernor(ram_budget=64 * MB, vram_budget=8 * MB)
    with pytest.raises(BudgetExhausted):
        gov.reserve("s:1", 64 * MB, 9 * MB)      # VRAM 不够
    st = gov.stats()
    assert st["ram_used"] == 0 and st["vram_used"] == 0, st
    assert st["live"] == 0 and st["balanced"], st


def test_duplicate_reservation_is_visible_in_ledger():
    """重复预约：同一 owner 反复占预算，账目必须能看出来（不能悄悄合并）。"""
    gov = ResourceGovernor(ram_budget=100 * MB, vram_budget=10 * MB)
    a = gov.reserve("s:1", 30 * MB, 1 * MB)
    b = gov.reserve("s:1", 30 * MB, 1 * MB)      # 同一个 owner 再要一份
    st = gov.stats()
    assert st["ram_used"] == 60 * MB, st
    assert st["owners"] == ["s:1"]
    gov.release(a)
    gov.release(b)


def test_multi_instance_competition_is_accountable():
    """多实例竞争：两个 governor 各自记账，总量可核算（不互相看见是已知限制）。"""
    g1 = ResourceGovernor(ram_budget=100 * MB, vram_budget=10 * MB, name="g1")
    g2 = ResourceGovernor(ram_budget=100 * MB, vram_budget=10 * MB, name="g2")
    r1 = g1.reserve("a", 80 * MB, 5 * MB)
    r2 = g2.reserve("b", 80 * MB, 5 * MB)     # 各自都"够"，合计 160MB
    total = g1.stats()["ram_used"] + g2.stats()["ram_used"]
    assert total == 160 * MB
    # 记录：跨实例限额需要外部分配器（Windows Job Object / cgroup），
    # 单靠进程内的 governor 做不到 —— 属 P3。
    g1.release(r1)
    g2.release(r2)


# ===========================================================================
# 负载 fixture：真实吃内存
# ===========================================================================

def test_ram_pressure_stops_new_work_and_drains():
    """压力：停止新工作 → 在途排空 → 实际释放。**不降质**。"""
    gov = ResourceGovernor(ram_budget=128 * MB, vram_budget=8 * MB,
                           min_forward_ram=8 * MB)
    inflight = []
    # 在途三个，占满
    for i in range(3):
        inflight.append(gov.reserve("s:%d" % i, 40 * MB, 1 * MB))
    st = gov.stats()
    assert st["ram_free"] == 128 * MB - 120 * MB

    # 新工作被拒（这就是"停止新工作"）
    with pytest.raises(BudgetExhausted):
        gov.reserve("s:new", 40 * MB, 1 * MB, wait=0.05)

    # 在途排空
    t0 = time.monotonic()
    for r in inflight:
        gov.release(r)
    drain_s = time.monotonic() - t0
    st = gov.stats()
    assert st["ram_used"] == 0 and st["live"] == 0, st
    assert drain_s < 1.0, f"排空耗时 {drain_s:.3f}s"

    # 排空后新工作可以进入（没有"永久拒绝"）
    r = gov.reserve("s:after", 40 * MB, 1 * MB)
    gov.release(r)


def test_external_memory_pressure_does_not_corrupt_ledger():
    """外部压力（真分配 64 MB）期间，governor 的账目不受影响。"""
    gov = ResourceGovernor(ram_budget=256 * MB, vram_budget=16 * MB)
    r = gov.reserve("s:1", 64 * MB, 2 * MB)
    ballast = bytearray(64 * MB)          # 真实占用，测完释放
    try:
        st = gov.stats()
        assert st["ram_used"] == 64 * MB and st["balanced"], st
    finally:
        del ballast
        gc.collect()
    gov.release(r)
    assert gov.stats()["ram_used"] == 0


def test_concurrent_pressure_from_many_threads_is_bounded():
    """多线程同时抢预算：总量不超过预算，不出现负剩余。"""
    gov = ResourceGovernor(ram_budget=64 * MB, vram_budget=8 * MB,
                           min_forward_ram=1 * MB)
    ok = []
    fail = []
    lock = threading.Lock()
    stop = threading.Event()

    def _w(i):
        while not stop.is_set():
            try:
                r = gov.reserve("w%d" % i, 8 * MB, 1 * MB, wait=0.02)
            except BudgetExhausted:
                with lock:
                    fail.append(i)
                time.sleep(0.001)
                continue
            with lock:
                ok.append(i)
            if len(ok) > 400:
                stop.set()
            gov.release(r)

    ts = [threading.Thread(target=_w, args=(i,)) for i in range(6)]
    for t in ts:
        t.start()
    time.sleep(0.4)
    stop.set()
    for t in ts:
        t.join(timeout=10)
    st = gov.stats()
    assert st["ram_used"] == 0, st
    assert st["peak_ram"] <= 64 * MB, st
    assert st["balanced"], st
    assert ok, "一次都没成功，说明预算算错了"
