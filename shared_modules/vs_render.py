# -*- coding: utf-8 -*-
"""VapourSynth 渲染的确定性控制。

## 为什么需要这个模块

本项目长期假设"同输入 + 同参数 → 同输出"。2026-11-06 实测发现**这个假设
对 stage1 不成立**：同一个 vpy、同一个源，连跑两次的结果**不同**。

### 实测数据（60 帧 1080p，完整 stage1 链路）

```
逐帧对比两次渲染：
    完全相同 30 帧 / 有差异 30 帧
    汇总 PSNR  y:48.71 u:51.53 v:39.63 average:45.48 min:42.07
```

### 逐段定位（累加滤镜，每档跑两次）

| 变体 | 差异帧 | PSNR min | 结论 |
|---|---|---|---|
| source + resize | 0/60 | inf | 确定 |
| + TemporalDegrain | 10/60 | 131.67 | **源头**（0.005 LSB） |
| + Deblock | 8/60 | 144.10 | 未放大 |
| + Stab（Depan） | **33/60** | **41.12** | **放大 400 倍** |
| + 颜色矩阵（完整） | 27/60 | 42.07 | 1.8 LSB |

**TemporalDegrain 是源头，`haf.Stab` 是放大器。**
Depan 的运动估计对输入极敏感：0.005 LSB 的扰动足以让某个块的 SAD 排序翻转，
运动向量差 1 像素，补偿后整块位移 —— 误差被放大到 1.8 LSB。

### 根因：VapourSynth 帧级并行 + mvtools

vapoursynth-mvtools 文档写明：

> All: **Free multithreading, courtesy of VapourSynth.**
> Analyse: No "temporal" parameter, as it's sort of incompatible with
> multithreading.

运动估计会从"相邻块已确定的运动向量"取 predictor（VapourSynth R81 博客
描述 Analyse 生成 4 个 predictor）。当多个候选项 SAD 相同（平局）时，
选哪个取决于**评价顺序**；帧级并行改变顺序 → 运动向量不同。

### 线程数扫描（只有单线程才确定）

| num_threads | 耗时 | 差异帧 |
|---|---|---|
| 默认（全核） | 4.95 s | 35/60 |
| 8 | 5.72 s | 27/60 |
| 4 | 7.39 s | 27/60 |
| 2 | 12.25 s | 13/60 |
| **1** | **23.47 s** | **0/60** |

差异帧数随线程数下降但**不归零**——只要有并行就有竞态。
`OMP_NUM_THREADS=1` **无效**（mvtools 的并行来自 VapourSynth，不是 OpenMP）。

## 所以这个模块做什么

提供 `determinism_preamble(params)`：当 `vs.deterministic` 为真时，
返回一行 `core.num_threads = 1`，**必须插在 `core = vs.core` 之后、
任何 filter 创建之前**才有意义。

> 教训：第一版我把这句写在脚本末尾（`set_output()` 前），那时 source 和
> resize 早就建好了，设置不生效 —— 于是得出"单线程仍非确定"的**假结论**。
> 位置错了，结论就错了。

## 代价与取舍

`num_threads=1` 让 stage1 慢 **4.78 倍**（整片从 1 小时变 4.8 小时）。
所以：

- **默认关闭**（`vs.deterministic: false`）—— 生产要速度
- **验收 / 复现时打开** —— 所有"逐帧等价"判据都必须在确定性模式下做，
  否则测到的是噪声本底（Y 48 / U 51 / V 39 级别），不是真实差异
"""
import os
from typing import Any, Dict, List

#: 配置键（params 里）。由 main 从 `config["vs"]["deterministic"]` 注入。
PARAM_KEY = "vs_deterministic"

#: 实测的多线程本底噪声（供判据引用，避免各处重复写死数字）。
NONDETERMINISM_NOISE = {
    "sample": "60 帧 1080p，完整 stage1 链路（TemporalDegrain + Deblock + Stab）",
    "differing_frames": "30/60",
    "psnr_y": 48.71,
    "psnr_u": 51.53,
    "psnr_v": 39.63,
    "psnr_min": 42.07,
    "note": ("任何低于此量级的'差异'都不能作为'两条路径不等价'的证据 —— "
             "那是本底噪声。要判定真实差异必须先开 vs.deterministic。"),
}


def wants_deterministic(params: Any) -> bool:
    """本次运行是否要求确定性渲染。

    接受多种写法（True / "true" / 1 / "yes" / "on"），其余一律视为假。
    任何异常都返回 False —— 这个开关**不能**因为坏配置而把生产变慢 4.78 倍。
    """
    try:
        v = (params or {}).get(PARAM_KEY)
    except Exception:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return False


def determinism_preamble(params: Any) -> List[str]:
    """返回要插入 vpy 的代码行（未请求确定性时返回空列表）。

    调用方必须把结果放在 `core = vs.core` 之后、**任何 filter 创建之前**。
    """
    if not wants_deterministic(params):
        return []
    return [
        "",
        "# ---- 确定性渲染（vs.deterministic = true）----",
        "# VapourSynth 的帧级并行会让 mvtools 产生非确定输出：",
        "# 实测同一素材连跑两次，60 帧里 35 帧不同，",
        "# 逐帧 PSNR Y 48.4 / U 51.2 / V 39.3、min 42.07 dB。",
        "# 线程数扫描证实**只有 num_threads=1 完全确定**",
        "# （默认 4.95s/35 帧 → 1 线程 23.47s/0 帧，慢 4.78 倍）。",
        "# 打开它是用时间换可复现：验收对照、问题复现必须开。",
        "# 注意：这行**必须**出现在任何 filter 创建之前才生效。",
        "core.num_threads = 1",
        "",
    ]


def thread_preamble(params: Any) -> List[str]:
    """按资源上限给出 `core.num_threads = N`（未启用/不需要时返回空）。

    与 `determinism_preamble` 同一插入位置：`core = vs.core` 之后、
    **任何 filter 创建之前**。两者同时存在时以确定性为准
    （`num_threads = 1` 已经低于任何上限，不需要再写一次）。
    """
    if wants_deterministic(params):
        return []
    try:
        from resource_limits import get as _rl_get
        plan = _rl_get(params)
    except Exception:                                  # pragma: no cover
        return []
    if not plan.get("enabled"):
        return []
    n = int(plan.get("threads") or 0)
    if n <= 0 or n >= int(plan.get("cpu_count") or 0):
        return []
    return [
        "",
        f"# ---- 资源限幅：VS 线程数不超过整机 {plan.get('max_fraction')} ----",
        f"core.num_threads = {n}",
        "",
    ]


def render_cost_note() -> str:
    """一句可直接写进日志的代价说明。

    R23-B14：不写死倍数。第十八轮的 4.78× 是"一台机器 + 60 帧 1080p"
    的单点观测，随素材/机器变化，不能当通用成本；日志里只说明**方向**。
    """
    return ("vs.deterministic=true 会让 VapourSynth 单线程渲染，明显更慢"
            "（具体倍数取决于素材与机器），换取逐像素可复现。")
