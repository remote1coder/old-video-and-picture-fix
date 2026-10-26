# -*- coding: utf-8 -*-
"""帧传输方式选择：PNG 文件中转 vs 共享内存零拷贝。

背景
----
stage3 的 AI 子进程原本对每帧做 `cv2.imread(png)` → `enhance()` →
`cv2.imwrite(png)`。实测大尺寸 16bit 帧时这处 PNG 编解码是 CPU 瓶颈
（GPU 利用率仅 16%、功耗 104W/300W，而 CPU 打满）。

改用 `multiprocessing.shared_memory` 可零拷贝传帧，实测基准
（tutorialpedia.org）：400MB 数组 pickle 2.4s → 共享内存 0.1s（24×），
峰值 CPU 85% → 12%（7× 降低）。

但共享内存要**按可用内存自适应缩减**，否则直接 OOM。本模块负责决策。

设计依据（均已联网核实）
------------------------
1. `budget = available × fraction`，"leave 50% for everything else"
   —— github.com/pproenca/dot-skills, mem-bound-the-working-set.md
2. 循环内**动态重估**：`min(chunk, available // 4)`
   —— engineering.zooz.com, Efficient Memory Management
3. **绝对红线**：严禁接近 100% 占用；保留绝对下限
   —— php.cn 内存分配实践 + 用户明确要求"至少留 1 GB 以上空闲"
4. 用后验证峰值是否守住预算 —— mem-bound-the-working-set.md

Windows 注意事项（已核实）
-------------------------
`SharedMemory.unlink()` 在 Windows 上是 CPython 已知 bug（bpo-40882，
PR 20684 未合），调用无效；但 Windows 会在**持有该内存的全部进程退出后
自动回收**，所以泄漏被限制在进程生命周期内。仍需 atexit 兜底。
"""
from __future__ import annotations

import atexit
import os
from typing import Any, Dict, Optional

# ---- 默认值（可被 config 覆盖；设非正数即退回默认）----------------------
DEFAULT_TRANSPORT = "auto"          # auto | shared_memory | png
DEFAULT_BUFFER_FRAMES = 256         # 期望的窗口帧数
DEFAULT_MEMORY_FRACTION = 0.5       # 最多占用「可用内存」的比例
DEFAULT_MIN_BUFFER_FRAMES = 8       # 低于此值回退 PNG
DEFAULT_MIN_FREE_BYTES = 1024 ** 3  # 绝对底线：至少留 1 GB 空闲

_VALID_TRANSPORTS = ("auto", "shared_memory", "png")


def available_bytes() -> Optional[int]:
    """当前可用物理内存（字节）。psutil 优先，失败退回 ctypes/Windows API。"""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    # 零依赖回退：Windows GlobalMemoryStatusEx
    try:
        import ctypes

        class _MSX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = _MSX()
        m.dwLength = ctypes.sizeof(_MSX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return int(m.ullAvailPhys)
    except Exception:
        pass
    return None


def _cfg(params: Optional[dict], key: str, default: float,
         lo: float = 0.0) -> float:
    """读配置；非正数/非法值一律退回 default（不允许用 0 关掉判定）。"""
    try:
        v = float((params or {}).get(key, default))
    except (TypeError, ValueError):
        return default
    return v if v > lo else default


def plan_transport(frame_bytes: int,
                   requested_frames: int = DEFAULT_BUFFER_FRAMES,
                   params: Optional[dict] = None,
                   out_scale: int = 1) -> Dict[str, Any]:
    """决定这一轮用共享内存还是 PNG，以及缓冲区放多少帧。

    Args:
        frame_bytes: 单帧字节数（如 RGB48 的 w*h*3*2）。必须 > 0。
        requested_frames: 期望的窗口帧数（来自 config / CLI）。
        params: 合并后的参数字典（含 ai_frame_* 键）。
        out_scale: 超分倍数（相对输入的边长倍数）。

            STR-06：调用方实际会创建**两份** ring —— 输入 ring 与输出 ring。
            输出帧面积是输入的 out_scale² 倍，所以每帧真实占用是
            frame_bytes × (1 + out_scale²)：2× 时 5 倍、4× 时 17 倍。
            旧实现只按 frame_bytes 算预算，于是"输入 ring 落在预算内"
            的表象下，两份 ring 合计可能远超预留内存底线。

    Returns:
        dict，至少含：
            transport      "shared_memory" | "png"
            buffer_frames  实际可用帧数（png 模式为 requested_frames）
            reason         人读得懂的判定理由（进日志）
            budget_bytes   本次内存预算（png 模式为 None）
            available_bytes当时探测到的可用内存
    """
    mode = str((params or {}).get("ai_frame_transport",
                                  DEFAULT_TRANSPORT)).strip().lower()
    if mode not in _VALID_TRANSPORTS:
        mode = DEFAULT_TRANSPORT

    # 2026-10-25 修：`requested_frames` 参数此前被忽略，只从 params 读，
    # 导致调用方传进来的窗口帧数无效（实测 ring 预算永远等于
    # ai_frame_buffer_frames=256，窗口一大就永远"超预算"）。
    # 现在：显式传入的优先，未传（None）才回退 params。
    if requested_frames is None:
        req = int(_cfg(params, "ai_frame_buffer_frames",
                       DEFAULT_BUFFER_FRAMES, lo=0))
    else:
        req = int(requested_frames)
    req = max(1, req)
    fraction = _cfg(params, "ai_frame_memory_fraction",
                    DEFAULT_MEMORY_FRACTION)
    min_frames = int(_cfg(params, "ai_frame_min_buffer_frames",
                          DEFAULT_MIN_BUFFER_FRAMES, lo=0))
    min_frames = max(1, min_frames)
    min_free = int(_cfg(params, "ai_frame_min_free_bytes",
                        DEFAULT_MIN_FREE_BYTES, lo=0))

    if frame_bytes <= 0:
        return {"transport": "png", "buffer_frames": req,
                "reason": "frame_bytes<=0，无法估算，退回 PNG",
                "budget_bytes": None, "available_bytes": None}

    # ---- 手动强制：用户说了算，但仍做一次安全下限检查 ----
    if mode == "png":
        return {"transport": "png", "buffer_frames": req,
                "reason": "ai_frame_transport=png（手动指定）",
                "budget_bytes": None, "available_bytes": None}
    if mode == "shared_memory":
        forced = True
    else:
        forced = False

    avail = available_bytes()
    if avail is None:
        return {"transport": "png", "buffer_frames": req,
                "reason": "探测不到可用内存，退回 PNG（保守）",
                "budget_bytes": None, "available_bytes": None}

    # ---- 预算：available × fraction，且不得侵入绝对底线 ----
    budget = int(avail * fraction)
    floor_budget = avail - min_free
    if floor_budget < budget:
        budget = max(0, floor_budget)

    # STR-06：按「输入 ring + 输出 ring」合计预算。
    # out_scale<=1 时退化为只有输入 ring（保持旧行为，便于单测对照）。
    try:
        _os = int(out_scale)
    except (TypeError, ValueError):
        _os = 1
    if _os < 1:
        _os = 1
    per_frame = frame_bytes * (1 + _os * _os) if _os > 1 else frame_bytes
    fits = budget // per_frame if per_frame else 0
    effective = min(req, fits)

    if effective < min_frames:
        return {
            "transport": "png", "buffer_frames": req,
            "reason": (f"可用内存不足：avail={avail/2**30:.2f}GB × "
                       f"{fraction} − {min_free/2**30:.0f}GB 底线 = "
                       f"{budget/2**30:.2f}GB，每帧 {frame_bytes/2**20:.1f}MB "
                       f"只放得下 {fits} 帧 < 下限 {min_frames}，退回 PNG"),
            "budget_bytes": budget, "available_bytes": avail,
        }

    if forced and effective < req:
        # 用户强制共享内存：尊重，但要留痕说明窗口被压小了
        return {
            "transport": "shared_memory", "buffer_frames": effective,
            "reason": (f"ai_frame_transport=shared_memory（手动指定），"
                       f"但内存只放得下 {effective}/{req} 帧，已按内存缩减"),
            "budget_bytes": budget, "available_bytes": avail,
        }

    return {
        "transport": "shared_memory", "buffer_frames": effective,
        "reason": (f"avail={avail/2**30:.2f}GB × {fraction}，每帧 "
                   f"{frame_bytes/2**20:.1f}MB → 缓冲区 {effective} 帧 "
                   f"（请求 {req}）"),
        "budget_bytes": budget, "available_bytes": avail,
    }


class FrameRing:
    """一块共享内存 + 环形帧槽（极简实现，够 stage3 窗口用）。

    只在本进程创建/销毁；AI 子进程通过 `attach()` 按名字挂上来。
    """

    def __init__(self, name: str, frame_bytes: int, slots: int):
        from multiprocessing import shared_memory
        self._shm_mod = shared_memory
        self.name = name
        self.frame_bytes = int(frame_bytes)
        self.slots = int(slots)
        self._shm = shared_memory.SharedMemory(
            create=True, size=self.frame_bytes * self.slots, name=name)
        atexit.register(self._safe_close)

    def _safe_close(self) -> None:
        try:
            self._shm.close()
        except Exception:
            pass

    def slot(self, i: int):
        """第 i 个槽的 numpy view（零拷贝）。"""
        import numpy as np
        off = (i % self.slots) * self.frame_bytes
        return np.ndarray((self.frame_bytes,), dtype=np.uint8,
                          buffer=self._shm.buf[off:off + self.frame_bytes])

    def close(self) -> None:
        self._safe_close()
        # Windows 上 unlink() 是 no-op（bpo-40882），由 OS 在进程退出后回收
        try:
            self._shm.unlink()
        except Exception:
            pass

    @staticmethod
    def attach(name: str, frame_bytes: int, slots: int) -> "FrameRing":
        from multiprocessing import shared_memory
        ring = FrameRing.__new__(FrameRing)
        ring._shm_mod = shared_memory
        ring.name = name
        ring.frame_bytes = int(frame_bytes)
        ring.slots = int(slots)
        ring._shm = shared_memory.SharedMemory(name=name)
        atexit.register(ring._safe_close)
        return ring
