# -*- coding: utf-8 -*-
"""类型化帧/图 **lease 池**（P2 / G1）。

## 这一层解决什么

现有链路用"路径 + 约定"在 stage 之间传帧：谁能读、读到的是不是别人的、
上一轮的残留算不算数，全靠调用方自觉。管道要把它变成**显式的租借**：

* 每个缓冲区有 **slot**，借出时登记 `owner` / `seq` / `ttl`；
* 归还时必须带**同一个 seq**，否则拒绝（防"旧帧覆盖新帧"）；
* owner 失联（超时 / 进程死）由 `revoke_owner()` **强制回收**（fencing）；
* 池容量固定，**没有空槽就明确拒绝或等待**，绝不悄悄扩容。

## 判据（对应 G1 的通过条件）

| 判据 | 实现方式 |
|---|---|
| 无丢失 | 每次 `acquire` 都记账；`stats()` 的 `issued == returned + revoked + live` |
| 无重复 | slot 被占用时不可能再发出去（同一 slot 只有一个 lease）|
| 无乱序 | `release(lease_id, seq)` 的 `seq` 必须与借出时一致 |
| 无旧帧覆盖 | `seq` 单调校验：归还后写回的 `seq` 必须 ≥ 该 slot 的 `last_seq` |
| 无无限增长 | `capacity` 固定；`slot_bytes` 固定；`stats()["bytes_live"]` 有上界 |
| 所有分配在账 | `issued_bytes` / `returned_bytes` / `live_bytes` 三者自洽 |
| 不够最低空间明确拒绝 | `acquire()` 在 `slot_bytes < min_slot_bytes` 时抛 `LeaseTooSmall` |
| 控制消息可达 | 见 `ControlChannel`（同文件下方）|
| 有界退出 | `close()` 之后所有 `acquire` 立即失败，不再新增 |

## 不做的事

* **不分配真正的共享内存**：本轮先用 `bytearray` 占位（假 worker 阶段）。
  真共享内存在 Windows 上要走 `CreateFileMapping`，属于"接真实工具"那一步。
  但**容量账目与所有权语义现在就是真的**，换底层不动上层。
* **不猜 slot 大小**：调用方必须给出 `slot_bytes`，小于 `min_slot_bytes` 直接拒。
"""
from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class LeaseError(Exception):
    """lease 层的基类异常。"""


class LeaseTooSmall(LeaseError):
    """请求的槽容量低于最小可前进单元。"""


class LeaseExhausted(LeaseError):
    """池已满且等待超时。"""


class LeaseStale(LeaseError):
    """归还的 seq 比该 slot 已见过的更旧 —— 旧帧覆盖。"""


class LeaseRevoked(LeaseError):
    """这个 lease 已被 fencing 回收，不能再使用。"""


class LeaseClosed(LeaseError):
    """池已关闭。"""


@dataclass
class FrameLease:
    """一次借出。

    Attributes:
        lease_id: 唯一 id（归还时的凭据）。
        slot: 占用的槽号。
        seq: 帧序号（**必须与归还时一致**，且不得小于该 slot 的 last_seq）。
        owner: 借出方标识（stage/worker 名 + pid）。
        bytes: 该槽的字节容量。
        acquired_at: 借出时刻（monotonic）。
        ttl: 存活秒数；`sweep_expired()` 会回收超时的。
        buffer: 占位缓冲区（本轮是 bytearray）。
        state: ``live`` / ``returned`` / ``revoked``。
    """

    lease_id: str
    slot: int
    seq: int
    owner: str
    bytes: int
    acquired_at: float
    ttl: float
    buffer: bytearray
    state: str = "live"
    meta: Dict[str, Any] = field(default_factory=dict)
    #: R26-A06：**槽位代际**。同一槽被复用后这个值会变，旧 lease 因此
    #: 能被识别为 stale。
    #:
    #: **局限（必须写清）**：Python 的裸 `bytearray` 引用**无法被撤销**。
    #: 旧 lease 手里的 `buffer` 仍指向同一块内存，写它就能污染新帧 ——
    #: 代际号只能让**遵守 API 的调用方**发现"我已经过期了"，
    #: 不能阻止故意/疏忽的裸写。真正的隔离要靠"离开作用域不保留引用"
    #: 或跨进程共享槽位描述符（见指令 §1.4）。
    generation: int = 0

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.acquired_at) > self.ttl


class LeasePool:
    """固定容量的类型化 lease 池。**线程安全**（假 worker 是多线程/多进程）。"""

    def __init__(self, capacity: int, slot_bytes: int,
                 *, min_slot_bytes: int = 1, name: str = "pool"):
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive")
        if int(slot_bytes) < int(min_slot_bytes):
            raise LeaseTooSmall(
                f"slot_bytes={slot_bytes} is below the minimum forward unit "
                f"({min_slot_bytes}); refusing to hand out a slot that cannot "
                f"carry one full frame")
        self.capacity = int(capacity)
        self.slot_bytes = int(slot_bytes)
        self.min_slot_bytes = int(min_slot_bytes)
        self.name = str(name)
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._closed = False
        self._slots: List[Optional[FrameLease]] = [None] * self.capacity
        # R25-08 / PA05：**预分配**槽位缓冲区。
        #
        # 原来 `_issue()` 每次 `bytearray(self.slot_bytes)` 新建一块 ——
        # 借还四次，如果消费者继续持有旧 buffer，内存里**真的有 4 块**，
        # 而账目 `live=0, bytes_live=0, balanced=True` 一点异常都看不出来。
        # 复核就是这么打穿的：容量 1×1024 的池保留了 4096 字节。
        #
        # 预分配后，物理上限就是 `capacity * slot_bytes`，与账目一致：
        # 归还即回收，下一次借到的是**同一块**（内容会被覆盖，
        # 这正是"有界槽位"的语义 —— 消费者不能再访问已归还的槽）。
        self._buffers: List[bytearray] = [
            bytearray(self.slot_bytes) for _ in range(self.capacity)]
        #: 每个 slot 见过的最大 seq（防旧帧覆盖）
        self._last_seq: List[int] = [-1] * self.capacity
        #: R26-A06：每个 slot 的代际（复用一次 +1）
        self._gen: List[int] = [0] * self.capacity
        #: 被隔离的 slot（超时/撤销后不立即复用）
        self._quarantined: set = set()
        self._live: Dict[str, FrameLease] = {}
        # 账目
        self.issued = 0
        self.returned = 0
        self.revoked = 0
        self.rejected = 0
        self.issued_bytes = 0
        self.returned_bytes = 0

    # ------------------------------------------------------------------ 借还
    def acquire(self, owner: str, seq: int, *, ttl: float = 30.0,
                wait: float = 0.0, meta: Optional[Dict[str, Any]] = None
                ) -> FrameLease:
        """借一个槽。

        Args:
            owner: 借出方标识（建议 ``"stage3:12345"``）。
            seq: 帧序号。
            ttl: 存活秒数，超时会被 `sweep_expired()` 回收。
            wait: 没有空槽时最多等多少秒；0 表示立即失败。
            meta: 附加信息（形状/stride/dtype 等，供审计）。

        Raises:
            LeaseClosed: 池已关闭。
            LeaseExhausted: 等不到空槽。
            LeaseStale: seq 比该 slot 见过的更旧（调用方拿错帧了）。
        """
        deadline = time.monotonic() + max(0.0, float(wait))
        with self._cond:
            while True:
                if self._closed:
                    raise LeaseClosed(f"pool {self.name!r} is closed")
                slot = self._find_free_slot()
                if slot is not None:
                    return self._issue(slot, owner, int(seq), float(ttl), meta)
                if time.monotonic() >= deadline:
                    self.rejected += 1
                    raise LeaseExhausted(
                        f"pool {self.name!r} has no free slot "
                        f"(capacity={self.capacity}, live={len(self._live)})")
                self._cond.wait(min(0.05, max(0.001, deadline - time.monotonic())))

    def _find_free_slot(self) -> Optional[int]:
        for i, cur in enumerate(self._slots):
            # R26-A06：隔离中的槽**不再分配** —— 那是"无法确认旧 owner
            # 已停止写"的槽，交给新帧等于把污染合法化。
            if cur is None and i not in self._quarantined:
                return i
        return None

    def is_current(self, lease: "FrameLease") -> bool:
        """这个 lease 还是不是它那个槽的**当前**持有者（R26-A06）。

        判据：`state == "live"` 且 `generation` 与该槽的当前代际相同。
        槽被复用一次，代际就 +1，旧 lease 立即变成 stale。

        **局限**：它只保护"遵守 API 的调用方"。Python 的裸 `bytearray`
        引用无法被撤销 —— 旧 lease 手里的 `buffer` 仍指向同一块内存。
        真正的隔离要靠跨进程共享槽位描述符或"离开作用域不保留引用"。
        """
        if lease is None:
            return False
        if str(getattr(lease, "state", "")) != "live":
            return False
        _slot = int(getattr(lease, "slot", -1))
        if _slot < 0 or _slot >= self.capacity:
            return False
        return int(getattr(lease, "generation", -1)) == int(self._gen[_slot])

    def quarantine(self, lease_id: str, reason: str = "") -> bool:
        """把该 lease 的槽**隔离**：撤销并**不再分配**（R26-A06）。

        用于"超时/所有者进程可能还在写"的场景 —— 指令 §1.3 的原话是
        "先 fence 旧 owner，不能直接复用；无法确认则 quarantine 并中止
        该执行组"。`release()` 是正常归还（槽可立即复用），
        `quarantine()` 是异常路径（槽退役）。

        Returns:
            是否找到并隔离了该 lease。
        """
        with self._cond:
            lease = self._live.get(str(lease_id))
            if lease is None:
                return False
            lease.state = "revoked"
            self._quarantined.add(lease.slot)
            self._slots[lease.slot] = None
            del self._live[lease.lease_id]
            self.revoked += 1
            self._cond.notify_all()
            return True

    def quarantined_slots(self) -> List[int]:
        """当前被隔离的槽（审计用）。"""
        return sorted(self._quarantined)

    def _issue(self, slot: int, owner: str, seq: int, ttl: float,
               meta: Optional[Dict[str, Any]]) -> FrameLease:
        if seq < self._last_seq[slot]:
            # 同一 slot 上出现更旧的帧号 → 说明调用方把上一轮的帧拿来了
            self.rejected += 1
            raise LeaseStale(
                f"slot {slot} last saw seq={self._last_seq[slot]}, now asked "
                f"for seq={seq}; refusing to let an older frame reuse it")
        lease = FrameLease(
            lease_id=uuid.uuid4().hex[:16], slot=slot, seq=seq, owner=str(owner),
            bytes=self.slot_bytes, acquired_at=time.monotonic(), ttl=ttl,
            buffer=self._buffers[slot], meta=dict(meta or {}),
            generation=self._gen[slot])
        # R26-A06：**先推进代际，再登记新 lease** —— 这样从这一刻起
        # 任何更早的 lease 都能被 `is_current()` 判成过期。
        self._gen[slot] += 1
        lease.generation = self._gen[slot]
        self._slots[slot] = lease
        self._last_seq[slot] = seq
        self._live[lease.lease_id] = lease
        self.issued += 1
        self.issued_bytes += self.slot_bytes
        return lease

    def release(self, lease_id: str, seq: int) -> None:
        """归还。

        Args:
            seq: **必须与借出时一致** —— 不一致说明调用方搞错了帧。

        Raises:
            LeaseRevoked: 该 lease 已被回收。
            LeaseStale: seq 不符。
        """
        with self._cond:
            lease = self._live.get(str(lease_id))
            if lease is None:
                # 已回收或从未存在
                raise LeaseRevoked(f"lease {lease_id!r} is not live")
            if int(seq) != int(lease.seq):
                self.rejected += 1
                raise LeaseStale(
                    f"lease {lease_id!r} carries seq={lease.seq}, release "
                    f"asked with seq={seq}; refusing to recycle a mismatched "
                    f"frame")
            lease.state = "returned"
            self._slots[lease.slot] = None
            del self._live[lease.lease_id]
            self.returned += 1
            self.returned_bytes += lease.bytes
            self._cond.notify_all()

    # ------------------------------------------------------------------ 回收
    def revoke_owner(self, owner: str) -> int:
        """把某个 owner 的所有 lease **强制回收**（fencing）。

        owner 进程死了、或它的 TTL 过了，缓冲区不能永远占着 ——
        但**不能**把它的数据当成有效结果。这里只回收槽位，
        调用方据返回值知道"这些 lease 被作废了"。

        Returns:
            被回收的 lease 数。
        """
        with self._cond:
            n = 0
            for lease in list(self._live.values()):
                if lease.owner == str(owner):
                    lease.state = "revoked"
                    self._slots[lease.slot] = None
                    del self._live[lease.lease_id]
                    self.revoked += 1
                    n += 1
            if n:
                self._cond.notify_all()
            return n

    def sweep_expired(self, now: Optional[float] = None) -> int:
        """回收**超时**的 lease。返回回收数。"""
        t = float(now if now is not None else time.monotonic())
        with self._cond:
            n = 0
            for lease in list(self._live.values()):
                if (t - lease.acquired_at) > lease.ttl:
                    lease.state = "revoked"
                    self._slots[lease.slot] = None
                    del self._live[lease.lease_id]
                    self.revoked += 1
                    n += 1
            if n:
                self._cond.notify_all()
            return n

    # ------------------------------------------------------------------ 观测
    def stats(self) -> Dict[str, Any]:
        """账目快照。**必须自洽**：issued == returned + revoked + live。"""
        with self._lock:
            live = len(self._live)
            return {
                "name": self.name,
                "capacity": self.capacity,
                "slot_bytes": self.slot_bytes,
                "min_slot_bytes": self.min_slot_bytes,
                "live": live,
                "issued": self.issued,
                "returned": self.returned,
                "revoked": self.revoked,
                "rejected": self.rejected,
                "bytes_live": live * self.slot_bytes,
                "bytes_capacity": self.capacity * self.slot_bytes,
                "issued_bytes": self.issued_bytes,
                "returned_bytes": self.returned_bytes,
                "closed": self._closed,
                "balanced": self.issued == (self.returned + self.revoked + live),
                "owners": sorted({ls.owner for ls in self._live.values()}),
            }

    def close(self, *, revoke_live: bool = True) -> Dict[str, Any]:
        """关池。之后 `acquire` 立即失败；`revoke_live` 时回收所有在借的。"""
        with self._cond:
            self._closed = True
            n = 0
            if revoke_live:
                for lease in list(self._live.values()):
                    lease.state = "revoked"
                    self._slots[lease.slot] = None
                    del self._live[lease.lease_id]
                    self.revoked += 1
                    n += 1
            self._cond.notify_all()
        return {"revoked_on_close": n, **self.stats()}


# ---------------------------------------------------------------------------
# 控制通道（cancel / pause / status）
# ---------------------------------------------------------------------------

class ControlChannel:
    """一个极小的**可轮询**控制通道。

    为什么不用信号：Windows 没有 POSIX 信号，且本项目的"取消"是
    **协作式**的（stage 主动查）。所以给一个原子标志 + 单调序号：

    * `cancel()` 置位并递增 seq；
    * `poll()` 返回 (cancelled, seq)，调用方在循环里查；
    * `wait(timeout)` 阻塞等待，用于"在途排空"。

    判据（G1）："控制消息可达，整个进程树有界退出" —— 只要每个
    worker 在每次取帧/放帧时 poll 一次，就能在**一个有界步数**内退出。
    """

    def __init__(self, name: str = "ctl"):
        self.name = str(name)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._cancelled = False
        self._seq = 0
        self._reason = ""

    def cancel(self, reason: str = "") -> int:
        with self._cond:
            self._cancelled = True
            self._seq += 1
            self._reason = str(reason or "")
            self._cond.notify_all()
            return self._seq

    def poll(self) -> Dict[str, Any]:
        with self._lock:
            return {"cancelled": self._cancelled, "seq": self._seq,
                    "reason": self._reason}

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def wait(self, timeout: float = 1.0) -> bool:
        """等取消或超时。返回是否已取消。"""
        with self._cond:
            if not self._cancelled:
                self._cond.wait(timeout)
            return self._cancelled
