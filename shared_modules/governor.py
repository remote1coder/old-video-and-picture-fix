# -*- coding: utf-8 -*-
"""RAM / VRAM **联合**资源预算（P2 / G6）。

## 为什么必须"联合"

分开申请（先要 RAM，再要 VRAM）会死锁：

```
A: 拿到 RAM，等 VRAM      B: 拿到 VRAM，等 RAM      → 互等
```

`reserve()` 因此是**一次原子的**：要么两个都拿到，要么一个都不拿。
这也是"没有跨 stage 互等"这条判据的实现方式 —— 不是靠超时兜底，
而是从根上不给"半拿"的机会。

## 不做什么

* **不硬编码整机 80%**：预算由政策层给定（`ram_budget` / `vram_budget`），
  这里只负责"在给定预算内分配"。整机目标是上层的事，写死在这里
  就等于把"能调"变成"不能调"。
* **不因资源不足改质量**：拿不到就等/拒/停，绝不降倍率、换模型、降位深。
  这一点由 B05 的隐式降质守卫兜底，这里不做任何"降级"分支。
* **不猜指标**：`observe()` 允许探测失败，失败时返回 `known=False`
  而不是填 0。填 0 会让"指标失效"看起来像"资源充足"。

## 与 lease 的关系

* `LeasePool` 管**槽位**（帧缓冲区的所有权）；
* `ResourceGovernor` 管**字节预算**（整个执行组能占多少 RAM/VRAM）。

两者独立：一个 stage 可以先向 governor 预约预算，再向 pool 借槽。
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


class GovernorError(Exception):
    """预算层基类异常。"""


class BelowMinimum(GovernorError):
    """申请量低于最小可前进单元 —— 这种申请本身就不该开工。"""


class BudgetExhausted(GovernorError):
    """预算不足且等待超时。"""


@dataclass
class Reservation:
    """一次联合预约。"""

    res_id: str
    owner: str
    ram: int
    vram: int
    granted_at: float
    state: str = "live"        # live / released / revoked
    meta: Dict[str, Any] = field(default_factory=dict)


class ResourceGovernor:
    """RAM / VRAM 联合预算。

    线程安全；也适用于"同一进程内的多个 worker 线程"这种最简单的假场景。
    """

    def __init__(self, ram_budget: int, vram_budget: int,
                 *, min_forward_ram: int = 1, min_forward_vram: int = 0,
                 name: str = "gov"):
        """Args:
            ram_budget: 本执行组允许占用的 RAM 字节上限（由政策层给）。
            vram_budget: 同上，VRAM。
            min_forward_ram: 最小可前进单元（RAM）。低于它的申请直接拒。
            min_forward_vram: 同上（VRAM）。**0 表示不要求**（纯 CPU 阶段）。
        """
        if int(ram_budget) <= 0:
            raise ValueError("ram_budget must be positive")
        if int(vram_budget) < 0:
            raise ValueError("vram_budget must be non-negative")
        self.ram_budget = int(ram_budget)
        self.vram_budget = int(vram_budget)
        self.min_forward_ram = int(min_forward_ram)
        self.min_forward_vram = int(min_forward_vram)
        self.name = str(name)

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._live: Dict[str, Reservation] = {}
        self._ram_used = 0
        self._vram_used = 0
        self._closed = False

        # 观测
        self.granted = 0
        self.released = 0
        self.revoked = 0
        self.rejected = 0
        self.peak_ram = 0
        self.peak_vram = 0
        #: 超限累计（等待/拒绝的次数与总时长）
        self.over_budget_events = 0
        self.over_budget_seconds = 0.0

    # ------------------------------------------------------------------ 预约
    def reserve(self, owner: str, ram: int, vram: int, *,
                wait: float = 0.0, meta: Optional[Dict[str, Any]] = None
                ) -> Reservation:
        """**原子**申请 RAM+VRAM。要么两个都拿到，要么一个都不拿。

        Raises:
            BelowMinimum: 申请量低于最小可前进单元（不该开工）。
            BudgetExhausted: 预算不足且等待超时。
            GovernorError: 池已关闭。
        """
        ram = int(ram)
        vram = int(vram)
        if ram < 0 or vram < 0:
            raise BelowMinimum(f"negative request (ram={ram}, vram={vram})")
        if ram < self.min_forward_ram:
            self.rejected += 1
            raise BelowMinimum(
                f"ram request {ram} is below the minimum forward unit "
                f"({self.min_forward_ram}); a stage that cannot hold one "
                f"frame must not start")
        if self.min_forward_vram and vram < self.min_forward_vram:
            self.rejected += 1
            raise BelowMinimum(
                f"vram request {vram} is below the minimum forward unit "
                f"({self.min_forward_vram})")
        if ram > self.ram_budget:
            # 单次申请就超过总预算 → 永远等不到，直接拒
            self.rejected += 1
            raise BudgetExhausted(
                f"ram request {ram} exceeds the whole budget "
                f"({self.ram_budget}); waiting would never succeed")
        if vram > self.vram_budget:
            self.rejected += 1
            raise BudgetExhausted(
                f"vram request {vram} exceeds the whole budget "
                f"({self.vram_budget})")

        deadline = time.monotonic() + max(0.0, float(wait))
        _waited = 0.0
        with self._cond:
            while True:
                if self._closed:
                    raise GovernorError(f"governor {self.name!r} is closed")
                if (self._ram_used + ram <= self.ram_budget
                        and self._vram_used + vram <= self.vram_budget):
                    return self._grant(owner, ram, vram, meta, _waited)
                if time.monotonic() >= deadline:
                    self.rejected += 1
                    self.over_budget_events += 1
                    raise BudgetExhausted(
                        f"not enough budget for ram={ram}, vram={vram} "
                        f"(free ram={self.ram_budget - self._ram_used}, "
                        f"free vram={self.vram_budget - self._vram_used})")
                _t0 = time.monotonic()
                self._cond.wait(min(0.05, max(0.001,
                                             deadline - time.monotonic())))
                _waited += time.monotonic() - _t0
                self.over_budget_seconds += time.monotonic() - _t0

    def _grant(self, owner: str, ram: int, vram: int,
               meta: Optional[Dict[str, Any]], waited: float) -> Reservation:
        res = Reservation(res_id=uuid.uuid4().hex[:16], owner=str(owner),
                          ram=ram, vram=vram,
                          granted_at=time.monotonic(),
                          meta=dict(meta or {}))
        res.meta.setdefault("waited_s", round(waited, 3))
        self._live[res.res_id] = res
        self._ram_used += ram
        self._vram_used += vram
        self.granted += 1
        self.peak_ram = max(self.peak_ram, self._ram_used)
        self.peak_vram = max(self.peak_vram, self._vram_used)
        return res

    def release(self, res: Reservation) -> None:
        """归还预算。重复归还会被拒（防"归还两次把预算放大"）。"""
        with self._cond:
            cur = self._live.get(getattr(res, "res_id", ""))
            if cur is None:
                raise GovernorError(
                    f"reservation {getattr(res, 'res_id', '?')!r} is not live; "
                    f"double release would inflate the budget")
            self._ram_used -= cur.ram
            self._vram_used -= cur.vram
            cur.state = "released"
            del self._live[cur.res_id]
            self.released += 1
            self._cond.notify_all()

    def revoke_owner(self, owner: str) -> int:
        """把某个 owner 的预约全部收回（fencing，用于 owner 死掉）。"""
        with self._cond:
            n = 0
            for res in list(self._live.values()):
                if res.owner == str(owner):
                    self._ram_used -= res.ram
                    self._vram_used -= res.vram
                    res.state = "revoked"
                    del self._live[res.res_id]
                    self.revoked += 1
                    n += 1
            if n:
                self._cond.notify_all()
            return n

    # ------------------------------------------------------------------ 观测
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            live = len(self._live)
            return {
                "name": self.name,
                "ram_budget": self.ram_budget,
                "vram_budget": self.vram_budget,
                "live": live,
                "ram_used": self._ram_used,
                "vram_used": self._vram_used,
                "ram_free": self.ram_budget - self._ram_used,
                "vram_free": self.vram_budget - self._vram_used,
                "granted": self.granted,
                "released": self.released,
                "revoked": self.revoked,
                "rejected": self.rejected,
                "peak_ram": self.peak_ram,
                "peak_vram": self.peak_vram,
                "over_budget_events": self.over_budget_events,
                "over_budget_seconds": round(self.over_budget_seconds, 3),
                "closed": self._closed,
                "balanced": self.granted == (self.released + self.revoked
                                             + live),
                "owners": sorted({r.owner for r in self._live.values()}),
            }

    def close(self, *, revoke_live: bool = True) -> Dict[str, Any]:
        with self._cond:
            self._closed = True
            n = 0
            if revoke_live:
                for res in list(self._live.values()):
                    self._ram_used -= res.ram
                    self._vram_used -= res.vram
                    res.state = "revoked"
                    del self._live[res.res_id]
                    self.revoked += 1
                    n += 1
            self._cond.notify_all()
        return {"revoked_on_close": n, **self.stats()}


# ---------------------------------------------------------------------------
# 能力探测（只读监测模式）
# ---------------------------------------------------------------------------

def observe_capacity(probe_ram: Optional[Callable[[], Optional[int]]] = None,
                     probe_vram: Optional[Callable[[], Optional[int]]] = None
                     ) -> Dict[str, Any]:
    """**只读**探测可用容量。探测失败返回 ``known=False``，**不填 0**。

    审核 G6："指标 N/A/过期"必须和"资源充足"区分开。填 0 会让
    "读不到"看起来像"没有余量"，或者反过来让调用方误以为"无穷"。

    Returns:
        ``{ram_bytes, vram_bytes, ram_known, vram_known, notes}``
    """
    out: Dict[str, Any] = {"ram_bytes": None, "vram_bytes": None,
                           "ram_known": False, "vram_known": False,
                           "notes": []}
    if probe_ram is not None:
        try:
            v = probe_ram()
            if isinstance(v, int) and v > 0:
                out["ram_bytes"] = int(v)
                out["ram_known"] = True
            else:
                out["notes"].append(f"ram probe returned {v!r}")
        except Exception as exc:
            out["notes"].append(f"ram probe failed: {exc}")
    else:
        # Windows 上不引入 psutil 的兜底：读 TotalPhysicalMemory 的可用部分
        try:
            import ctypes
            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                out["ram_bytes"] = int(ms.ullAvailPhys)
                out["ram_known"] = True
        except Exception as exc:
            out["notes"].append(f"GlobalMemoryStatusEx failed: {exc}")

    if probe_vram is not None:
        try:
            v = probe_vram()
            if isinstance(v, int) and v > 0:
                out["vram_bytes"] = int(v)
                out["vram_known"] = True
            else:
                out["notes"].append(f"vram probe returned {v!r}")
        except Exception as exc:
            out["notes"].append(f"vram probe failed: {exc}")
    else:
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, timeout=10)
            if r.returncode == 0:
                vals = [int(x) for x in (r.stdout or b"").decode().split()
                        if x.strip().isdigit()]
                if vals:
                    out["vram_bytes"] = int(sum(vals)) * 1024 * 1024
                    out["vram_known"] = True
            else:
                out["notes"].append("nvidia-smi returned non-zero")
        except Exception as exc:
            out["notes"].append(f"nvidia-smi failed: {exc}")
    return out
