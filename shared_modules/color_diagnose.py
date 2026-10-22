#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""色彩偏色诊断（**只读，一个像素都不改**）。

为什么需要这个模块
------------------
用户素材是「数码相机翻拍老照片/文档」，最需要判断的是：
画面里的"黄"到底是 **染料退化（可修）** 还是 **物体本色（内容，不可动）**。

分不清这两者时，任何"自动白平衡"都可能把红色婚礼照、黄昏风景、米色卡纸
强行拉成中性 —— 那就是**改变内容**，与"只改画面质量不改内容"的意图相反。

判据（**双条件，必须同时满足才建议校正**）
------------------------------------------
1. **中性参考偏色**：画面里"本该是中性的"像素（够亮 + 低饱和）的 R/B
   偏离 1.0 超过 ``neutral_rb_tolerance``（默认 0.10）。
2. **蓝通道量程塌缩**：B 通道 (p99−p1) 占满量程的比例低于
   ``blue_range_min``（默认 0.65）。

依据（联网核实的实测数据，EachMoment 实验室，184 张 1972–1981 年
chromogenic 打印件，600dpi/16bit 扫描）：

    Red 通道（黄染料代理）  剩余动态范围 96%
    Green 通道（品红代理）              71%
    Blue 通道（青染料代理）             38%，暗部低至 8%

青染料最先褪、它负责吸收红光；青没了，黄+品红占主导，画面偏橙红。
**所以"黄"的本质是"缺蓝"** —— 正确的修法是逐通道拉伸，而不是整体
调色温（后者会把蓝天和肤色一起搞坏）。

为什么必须是双条件
------------------
只看"中性参考偏色"会把**暖色场景**误判成褪色：
    红色婚礼照（红嫁衣+红喜字）通道均值 R=0.610/G=0.420/B=0.405，
    全局 mean_diff 达 0.205，但画面里根本没有偏黄的中性参考
    —— 它的红来自物体本身。
加上"蓝通道量程塌缩"这一条，就能把这类画面排除掉：
真正的褪色件蓝通道只用掉四成量程，而正常彩色照片三通道量程相当。

输出
----
``color_cast_report.json``  逐张完整指标
``color_cast_summary.txt``  人读摘要
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

#: 中性参考候选：亮度取前百分之几
NEUTRAL_TOP_PCT = 10.0
#: 中性参考候选：该像素的 (max-min) 饱和度上限（0..1）
NEUTRAL_MAX_SAT = 0.12
#: 中性参考像素占比低于这个值时，退回"最亮 1%"
NEUTRAL_MIN_SHARE = 0.005
#: R/B 偏离 1.0 超过多少算"中性参考偏色"
NEUTRAL_RB_TOLERANCE = 0.10
#: 蓝通道量程占比低于多少算"塌缩"
BLUE_RANGE_MIN = 0.65

_VERDICT_OK = "ok"                # 无需校正
_VERDICT_CONTENT_COLOR = "content_color"  # 偏色来自物体本色 → 不要动
_VERDICT_FADED = "faded"          # 双条件成立 → 建议逐通道拉伸


def _to_float_rgb(arr: np.ndarray) -> np.ndarray:
    """把任意位深的图像数组统一成 float32 RGB，范围 0..1。

    - uint8  → /255
    - uint16 → /65535
    - 已有 float 且最大值 <= 1.0 → 视为已归一化
    - 2 通道（灰度）→ 复制成 3 通道
    - 4 通道 → 丢弃 alpha
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=2)
    if a.ndim == 3 and a.shape[2] >= 3:
        a = a[:, :, :3]
    if a.dtype == np.uint8:
        f = a.astype(np.float32) / 255.0
    elif a.dtype == np.uint16:
        f = a.astype(np.float32) / 65535.0
    else:
        f = a.astype(np.float32)
        mx = float(f.max()) if f.size else 0.0
        if mx > 1.5:  # 仍是 8/16-bit 量级但 dtype 不是整数
            f = f / (65535.0 if mx > 300 else 255.0)
    return np.clip(f, 0.0, 1.0)


def _read_rgb_float(path: str, max_edge: int = 1200) -> Optional[np.ndarray]:
    """读图并缩到 max_edge 以内（诊断只需统计量，不需要全分辨率）。"""
    try:
        import sys
        from pathlib import Path
        _here = Path(__file__).resolve().parent
        if str(_here) not in sys.path:
            sys.path.insert(0, str(_here))
        import media_io  # noqa: E402

        arr, _meta = media_io.read_image_array(path)
        f = _to_float_rgb(arr)
    except Exception:
        # 退回 PIL（不依赖 media_io 时仍可工作）
        try:
            from PIL import Image
            with Image.open(path) as im:
                f = _to_float_rgb(np.array(im.convert("RGB")))
        except Exception:
            return None

    h, w = f.shape[:2]
    if max(h, w) > max_edge:
        step = int(np.ceil(max(h, w) / max_edge))
        f = f[::step, ::step, :]
    return f


def analyze_array(f: np.ndarray) -> Dict[str, Any]:
    """对一个 float32 RGB 数组算诊断指标。**纯读取，不修改入参。**"""
    r, g, b = f[:, :, 0], f[:, :, 1], f[:, :, 2]

    means = [float(r.mean()), float(g.mean()), float(b.mean())]
    ranges = []
    for ch in (r, g, b):
        p1, p99 = np.percentile(ch, [1, 99])
        ranges.append(float(p99 - p1))  # 0..1 满量程的比例

    # ---- 中性参考：够亮 + 低饱和 ----
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = mx - mn

    thr = float(np.percentile(lum, 100.0 - NEUTRAL_TOP_PCT))
    mask = (lum >= thr) & (sat <= NEUTRAL_MAX_SAT)
    share = float(mask.mean())
    fallback = False
    if share < NEUTRAL_MIN_SHARE:
        # 画面里没有明显的"低饱和亮部"（比如满屏红色婚礼照）
        thr2 = float(np.percentile(lum, 99.0))
        mask = lum >= thr2
        share = float(mask.mean())
        fallback = True

    if mask.any():
        nr, ng, nb = (float(r[mask].mean()), float(g[mask].mean()),
                      float(b[mask].mean()))
    else:  # 理论上不会发生
        nr, ng, nb = means
    neutral_rb = nr / max(nb, 1e-6)
    neutral_dev = abs(neutral_rb - 1.0)

    # ---- 判据 ----
    neutral_off = neutral_dev > NEUTRAL_RB_TOLERANCE
    blue_deficit = ranges[2] < BLUE_RANGE_MIN

    if neutral_off and blue_deficit:
        verdict = _VERDICT_FADED
        recommend = True
        reasons = [
            f"中性参考 R/B={neutral_rb:.3f} 偏离中性超过 "
            f"{NEUTRAL_RB_TOLERANCE:.2f}",
            f"蓝通道量程仅占 {ranges[2]*100:.0f}%（< "
            f"{BLUE_RANGE_MIN*100:.0f}%），符合青染料褪色的特征",
        ]
    elif neutral_off and not blue_deficit:
        verdict = _VERDICT_CONTENT_COLOR
        recommend = False
        reasons = [
            f"中性参考 R/B={neutral_rb:.3f} 有偏移，但蓝通道量程正常"
            f"（{ranges[2]*100:.0f}%）—— 偏色来自物体本色/场景色温，"
            f"校正会改变画面内容",
        ]
    else:
        verdict = _VERDICT_OK
        recommend = False
        reasons = [f"中性参考 R/B={neutral_rb:.3f} 基本中性，无需校正"]

    # 建议的逐通道拉伸量（L1：把每个通道 p1..p99 拉回 0..1）
    stretch = [round(1.0 / max(x, 1e-3), 3) for x in ranges]

    return {
        "channel_mean_rgb": [round(x, 4) for x in means],
        "channel_range_pct_rgb": [round(x * 100, 1) for x in ranges],
        "neutral_ref_rgb": [round(nr, 4), round(ng, 4), round(nb, 4)],
        "neutral_ref_rb_ratio": round(neutral_rb, 4),
        "neutral_ref_pixel_share_pct": round(share * 100, 2),
        "neutral_ref_fallback_brightest": fallback,
        "blue_range_deficit": bool(blue_deficit),
        "suggested_per_channel_stretch": stretch,
        "verdict": verdict,
        "recommend_correction": bool(recommend),
        "reasons": reasons,
    }


def diagnose_file(path: str, params: Optional[dict] = None,
                  logger=None) -> Dict[str, Any]:
    """诊断单张。返回 dict；读不出来时 verdict='unreadable'。"""
    f = _read_rgb_float(path)
    if f is None:
        return {
            "input_path": str(path),
            "verdict": "unreadable",
            "recommend_correction": False,
            "reasons": ["无法读取该图像"],
        }
    res = analyze_array(f)
    res["input_path"] = str(path)
    res["analyzed_shape"] = list(f.shape)
    return res


def _iter_photo_files(inputs: Iterable[str]) -> List[str]:
    """展开文件/目录为图片路径列表（只认常见图片扩展名）。"""
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    out: List[str] = []
    for p in inputs:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if os.path.splitext(name)[1].lower() in exts:
                    out.append(os.path.join(p, name))
        elif os.path.isfile(p):
            out.append(p)
    return out


def diagnose_paths(inputs: Iterable[str],
                   out_dir: str,
                   params: Optional[dict] = None,
                   logger=None) -> Dict[str, Any]:
    """批量诊断并写报告。**不写任何图片**，只写 json/txt。"""
    files = _iter_photo_files(inputs)
    results: List[Dict[str, Any]] = []
    for i, p in enumerate(files, 1):
        r = diagnose_file(p, params=params, logger=logger)
        results.append(r)
        if logger and (i % 25 == 0 or i == len(files)):
            logger.info("COLOR_DIAGNOSE_PROGRESS",
                        f"{i}/{len(files)}",
                        recommend=sum(1 for x in results
                                      if x.get("recommend_correction")))

    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, "color_cast_report.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump({
            "generated_by": "color_diagnose",
            "note": "只读诊断，未修改任何像素",
            "criteria": {
                "neutral_rb_tolerance": NEUTRAL_RB_TOLERANCE,
                "blue_range_min": BLUE_RANGE_MIN,
                "neutral_top_pct": NEUTRAL_TOP_PCT,
                "neutral_max_sat": NEUTRAL_MAX_SAT,
            },
            "summary": _summarize(results),
            "images": results,
        }, f, ensure_ascii=False, indent=2)

    tp = os.path.join(out_dir, "color_cast_summary.txt")
    with open(tp, "w", encoding="utf-8") as f:
        f.write(_render_text(results))
    return {"report_json": jp, "summary_txt": tp,
            "summary": _summarize(results), "results": results}


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    by_verdict: Dict[str, int] = {}
    for r in results:
        v = r.get("verdict", "unknown")
        by_verdict[v] = by_verdict.get(v, 0) + 1
    rec = [r for r in results if r.get("recommend_correction")]
    return {
        "total": total,
        "by_verdict": by_verdict,
        "recommend_count": len(rec),
        "recommend_files": [r["input_path"] for r in rec],
    }


def _render_text(results: List[Dict[str, Any]]) -> str:
    s = _summarize(results)
    lines = []
    lines.append("=" * 78)
    lines.append("色彩偏色诊断报告（只读，未修改任何像素）")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"总计 {s['total']} 张")
    for v, n in sorted(s["by_verdict"].items()):
        label = {
            "ok": "无需校正（中性参考正常）",
            "content_color": "偏色来自物体本色/场景色温 → 不建议校正",
            "faded": "疑似染料褪色 → 建议逐通道拉伸",
            "unreadable": "无法读取",
        }.get(v, v)
        lines.append(f"  {v:<14} {n:>5} 张   {label}")
    lines.append("")
    lines.append("判据（双条件必须同时满足）：")
    lines.append(f"  1) 中性参考 R/B 偏离 1.0 > {NEUTRAL_RB_TOLERANCE:.2f}")
    lines.append(f"  2) 蓝通道量程占比 < {BLUE_RANGE_MIN*100:.0f}%"
                 "（青染料褪色的特征）")
    lines.append("")
    lines.append(f"建议校正: {s['recommend_count']} 张")
    lines.append("")
    lines.append("-" * 78)
    lines.append("%-46s %8s %8s %-14s" % ("文件", "中性R/B", "蓝量程%", "判定"))
    lines.append("-" * 78)
    for r in results:
        lines.append("%-46s %8.3f %8.1f %-14s" % (
            os.path.basename(r.get("input_path", "?")),
            r.get("neutral_ref_rb_ratio", 0.0),
            r.get("channel_range_pct_rgb", [0, 0, 0])[2],
            r.get("verdict", "?"),
        ))
    lines.append("")
    return "\n".join(lines)
