# -*- coding: utf-8 -*-
"""把 CPU/内存/显存占用压到配置上限以内（默认 80%）。

## 为什么要有这一层

长片跑批会把整机吃满：`vspipe` 默认按物理核数开线程、`libsvtav1` 默认
占满逻辑核、ffmpeg 解码/滤镜也各自开线程。多者叠加的结果是"每个都以为
自己有整台机器"，实际并发达标 100% 以上 —— 轻则拖慢同一台机器上的其它
工作，重则把内存/显存推到临界点，让一次 40 分钟的跑批在中途崩掉。

用户的要求很明确：**任何一项都不超过 80%，慢一点完全可以接受。**

## 这一层怎么落地（以及它的边界）

能直接限的：

* **CPU 线程**：`vspipe -t N`、`ffmpeg -threads N`、SVTAV1 `lp=N`、
  x265 `pools=N`。这是真正起作用的那一项 —— CPU 打满是"崩溃"和
  "其它程序卡死"的主要来源。
* **RAM / VRAM**：给 `governor.ResourceGovernor` 一个
  `max_fraction` 预算，它的原子联合预约就不会越过分界线。

**不能**直接限的（必须如实说）：

* 单个 ffmpeg 进程内部的峰值内存 —— 那要靠 cgroup / Job Object，
  本模块不碰全机配置，只用"限制并发 + 限制线程"来间接压住。
* 显存的瞬时峰值 —— 同理，靠 `ResourceGovernor` 的预约而不是硬隔离。

所以本模块的承诺是"**线程与预算都在 80% 以内**"，不是
"物理上绝不可能超过 80%"。这两句话不能混。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "max_fraction": 0.80,
    "min_threads": 1,
    "reserve_cores": 0,      # 额外让出的核数（给系统/桌面用）
}

_KEY = "_resource_limits"


def _as_bool(v, d=False):
    if isinstance(v, bool):
        return v
    if v is None:
        return d
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return d


def _as_float(v, d):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def resolve(params: Any = None, config: Any = None) -> Dict[str, Any]:
    """解析出本机的线程/预算上限（幂等，可在任何 stage 里再算一次）。"""
    src: Dict[str, Any] = {}
    if isinstance(config, dict) and isinstance(config.get("resource_limits"),
                                               dict):
        src.update(config["resource_limits"])
    if isinstance(params, dict):
        p = params.get(_KEY)
        if isinstance(p, dict):
            src.update(p)
        pc = params.get("resource_limits")
        if isinstance(pc, dict):
            src.update(pc)

    frac = _as_float(src.get("max_fraction"), DEFAULTS["max_fraction"])
    # 夹紧到 (0, 1]：0.0 会让所有进程单线程甚至 0 线程，>1 等于没限。
    frac = min(1.0, max(0.05, frac))
    cpu = os.cpu_count() or 4
    reserve = max(0, int(_as_float(src.get("reserve_cores"), 0)))
    threads = int(cpu * frac) - reserve
    threads = max(int(_as_float(src.get("min_threads"), 1)), threads)
    threads = max(1, min(cpu, threads))
    return {
        "enabled": _as_bool(src.get("enabled"), DEFAULTS["enabled"]),
        "max_fraction": frac,
        "cpu_count": cpu,
        "threads": threads,
        "reserve_cores": reserve,
    }


def get(params: Any = None) -> Dict[str, Any]:
    """取**已经算好**的那份；没有就现算一份（单跑 stage 时也能用）。"""
    if isinstance(params, dict) and isinstance(params.get(_KEY), dict):
        return params[_KEY]
    return resolve(params, None)


def threads(params: Any = None) -> int:
    p = get(params)
    return int(p["threads"]) if p.get("enabled") else (os.cpu_count() or 4)


def num_threads(params: Any = None) -> int:
    """本进程应该用的 VapourSynth 线程数（写进 vpy 的 `core.num_threads`）。

    ## 为什么不用 `vspipe` 的命令行参数

    `vspipe -t` 是 **`--timecodes`**（写时间码文件），不是线程数 ——
    实测 R72 的 `vspipe --help`：`-t, --timecodes FILE`。
    用错它会把一个文件名当成线程数，行为完全跑偏。
    而 `-r/--requests` 是"并发帧请求数"，也不是 CPU 上限。

    真正管住 VS 并行度的是脚本里的 `core.num_threads`，而且它
    **必须出现在任何 filter 创建之前**（与 `determinism_preamble` 同一约束）。
    """
    p = get(params)
    if not p.get("enabled"):
        return os.cpu_count() or 4
    return int(p["threads"])


def ffmpeg_threads(params: Any = None) -> List[str]:
    """`ffmpeg -threads N`（**输出侧**选项，放在 `-i` 之后）。"""
    p = get(params)
    if not p.get("enabled"):
        return []
    return ["-threads", str(int(p["threads"]))]


def svt_logical_processors(params: Any = None) -> str:
    """SVTAV1 的 `lp=N`（写进 `-svtav1-params`）。"""
    p = get(params)
    if not p.get("enabled"):
        return ""
    return f"lp={int(p['threads'])}"


def x265_pools(params: Any = None) -> str:
    p = get(params)
    if not p.get("enabled"):
        return ""
    return f"pools={int(p['threads'])}"


def describe(plan: Dict[str, Any]) -> str:
    if not plan.get("enabled"):
        return "resource limits: DISABLED (processes may use every core)"
    return (f"resource limits: max_fraction={plan['max_fraction']:.2f} "
            f"cpu={plan['cpu_count']} -> threads={plan['threads']} "
            f"(reserve={plan['reserve_cores']})")
