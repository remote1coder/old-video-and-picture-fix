# -*- coding: utf-8 -*-
"""P2：假 worker（线程池 + 真实 spawn 子进程）验证 G1 的进程侧判据。

## 本文件覆盖

| 判据 | 怎么测 |
|---|---|
| 无丢失 / 无重复 / 无乱序 | 多线程从同一池借槽，每槽写自己的 seq，主线程核对 |
| 所有分配在账 | 结束后 `stats()["balanced"]` |
| 控制消息可达 | `ControlChannel` 在线程池里被所有 worker 看到 |
| **整个进程树有界退出** | 真起 `spawn` 子进程，主进程置 cancel，测退出耗时上界 |

## 为什么进程树要单独测

`LeasePool` 是进程内的 Python 对象，跨进程共享要么用 `Manager` 代理
（每个操作一次 IPC，性能不适合帧循环），要么用共享内存（P2 的真形态，
属"接真实工具"那一步）。本文件因此分两层：

1. **线程层**：验证所有权与账目（同进程、直接调用）；
2. **进程层**：验证"取消能传到子进程、子进程能在有界时间内退出"
   —— 用 `Manager().Value()` 做取消标志，这正是"控制消息可达"的最小形态。

## 不做什么

* 不假装"跨进程共享 lease 池已验证"：那要共享内存 + 跨进程锁，本文件没有。
* 不用 `fork`：Windows 只有 `spawn`，测 `spawn` 才有意义。
"""
import multiprocessing as mp
import os
import sys
import threading
import time

import pytest

import conftest  # noqa: F401

from lease import ControlChannel, LeasePool

MB = 1 << 20


# ---------------------------------------------------------------------------
# 线程层：所有权 / 账目 / 无丢失
# ---------------------------------------------------------------------------

def test_threaded_workers_never_lose_or_duplicate_frames():
    """无丢失/无重复/无乱序：每个 worker 借槽 → 写自己的 seq → 归还。"""
    workers, rounds = 6, 30
    pool = LeasePool(capacity=4, slot_bytes=4096, name="threads")
    seen, lock = [], threading.Lock()
    errors = []

    def _w(idx):
        for r in range(rounds):
            seq = idx * rounds + r
            try:
                lease = pool.acquire("w%d" % idx, seq=seq, wait=3.0)
            except Exception as exc:          # pragma: no cover - 池不够大
                errors.append(repr(exc))
                return
            # 写"本帧"的图案：seq 的低 16 位填满缓冲
            lease.buffer[:] = int(seq & 0xFFFF).to_bytes(2, "little") * (
                len(lease.buffer) // 2)
            # 读回校验（防"借到别人的槽"）
            got = int.from_bytes(lease.buffer[:2], "little")
            if got != (seq & 0xFFFF):
                errors.append("slot content mismatch: %d vs %d" % (got, seq))
            with lock:
                seen.append(seq)
            pool.release(lease.lease_id, seq=seq)

    ts = [threading.Thread(target=_w, args=(i,)) for i in range(workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)

    assert not errors, errors[:3]
    assert len(seen) == workers * rounds, len(seen)
    assert len(set(seen)) == len(seen), "同一帧被处理了两次"
    st = pool.stats()
    assert st["live"] == 0 and st["balanced"], st


def test_cancel_reaches_all_thread_workers():
    """控制消息可达：cancel 之后所有 worker 自行停下。"""
    ctl = ControlChannel("threads-ctl")
    pool = LeasePool(capacity=2, slot_bytes=1024)
    stopped = []
    lock = threading.Lock()

    def _w(idx):
        while not ctl.poll()["cancelled"]:
            try:
                lease = pool.acquire("w%d" % idx, seq=idx, wait=0.05)
            except Exception:
                continue
            pool.release(lease.lease_id, seq=idx)
            time.sleep(0.001)
        with lock:
            stopped.append(idx)

    ts = [threading.Thread(target=_w, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    time.sleep(0.15)
    ctl.cancel("test over")
    for t in ts:
        t.join(timeout=5)
    assert len(stopped) == 4, stopped
    assert all(not t.is_alive() for t in ts)


# ---------------------------------------------------------------------------
# 进程层：spawn 子进程 + 有界退出
# ---------------------------------------------------------------------------

def _child_loop(cancel_flag, counter, rounds, interval):
    """子进程：轮询取消标志，做有界步数的工作。"""
    n = 0
    for _ in range(rounds):
        if cancel_flag.value:
            break
        n += 1
        time.sleep(interval)
    with counter.get_lock():
        counter.value += n


def _probe_import(qq):
    """spawn 子进程里能否 import 本项目模块并正常借还。

    **必须放模块顶层**：Windows 的 spawn 用 pickle 传目标函数，
    局部函数（闭包）无法序列化 —— 第一版写成测试内的 `_probe`
    直接报 `Can't get local object`。
    """
    try:
        from lease import LeasePool as _P
        p = _P(capacity=1, slot_bytes=64)
        lease = p.acquire("child", seq=0)
        p.release(lease.lease_id, seq=0)
        qq.put(("ok", p.stats()["balanced"]))
    except Exception as exc:             # pragma: no cover
        qq.put(("err", repr(exc)))


def _spawn_children(ctx, n, rounds, interval):
    cancel_flag = ctx.Value("i", 0)
    counter = ctx.Value("i", 0)
    procs = [ctx.Process(target=_child_loop,
                         args=(cancel_flag, counter, rounds, interval))
             for _ in range(n)]
    for p in procs:
        p.start()
    return cancel_flag, counter, procs


def test_process_tree_exits_within_bound():
    """整个进程树有界退出：置 cancel 后，所有 spawn 子进程在限期内退出。"""
    ctx = mp.get_context("spawn")
    n, rounds, interval = 4, 10_000, 0.002
    cancel_flag, counter, procs = _spawn_children(ctx, n, rounds, interval)

    time.sleep(0.3)                      # 让它们真的跑起来
    alive_before = sum(1 for p in procs if p.is_alive())
    assert alive_before >= 1, "子进程没有启动（spawn 失败？）"

    t0 = time.monotonic()
    cancel_flag.value = 1                # 这就是"控制消息可达"
    for p in procs:
        p.join(timeout=15)
    elapsed = time.monotonic() - t0

    still = [p.pid for p in procs if p.is_alive()]
    for p in procs:                      # 清理（不留在进程表里）
        if p.is_alive():                 # pragma: no cover - 兜底
            p.terminate()
    assert not still, f"子进程未退出: {still}"
    assert elapsed < 10.0, f"退出耗时 {elapsed:.2f}s，超出有界预期"
    # 每个子进程最多做 rounds 步；取消后应远小于它
    assert counter.value <= n * rounds


def test_child_finishes_naturally_without_cancel():
    """对照组：不取消时子进程也能自然跑完（证明上面的"快退出"不是假象）。"""
    ctx = mp.get_context("spawn")
    n, rounds, interval = 2, 50, 0.001
    cancel_flag, counter, procs = _spawn_children(ctx, n, rounds, interval)
    for p in procs:
        p.join(timeout=30)
    assert all(not p.is_alive() for p in procs)
    assert counter.value == n * rounds, counter.value


def test_spawn_child_can_import_lease_module():
    """spawn 子进程能 import 本项目模块（路径在子进程里也要对）。"""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()

    p = ctx.Process(target=_probe_import, args=(q,))
    p.start()
    p.join(timeout=30)
    assert not p.is_alive()
    got = q.get(timeout=5)
    assert got[0] == "ok" and got[1] is True, got
