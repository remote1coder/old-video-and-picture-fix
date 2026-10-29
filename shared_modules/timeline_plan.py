# -*- coding: utf-8 -*-
"""最小 TimelinePlan：把"时间轴怎么理解"跨阶段传下去（R14-05）。

**要解决什么**

审核指出：`_fps_source` 只在 stage3 内部用，**没有跨阶段传递**；
stage7 测出"生成物可测"后可能把局部 timing 标成 verified ——
而"生成物能测量"**不等于**"未知的源时间轴已被保留"。

所以至少要分开三个**不同**的结论：

| 结论 | 谁判定 | 含义 |
|---|---|---|
| ① 源时间轴是否已验证 | stage3 | PTS 覆盖完整、CFR/VFR 判定有依据 |
| ② 用户请求的变换是否执行 | stage3 + stage7 | 声明的变速/重解释有没有真的落地 |
| ③ 输出时间轴是否测得 | stage7 | 成品自身的 packet 跨度量得出来 |

三者不能互相替代。**特别是**：③ 成立**不能**让 ① 变成已验证。

**本模块提供**

- ``build_plan()`` —— stage3 侧构造；
- ``claim_source_verified()`` —— 唯一允许宣称"源时序已验证"的判定；
- ``write_plan()`` / ``read_plan()`` —— job 级侧车读写。

不做什么：不解析 PTS、不探测 ffprobe、不做编码决策。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

SCHEMA = "timeline-plan/1"

#: 源时间轴被认为"已验证"的状态集合
VERIFIED_STATES = ("cfr", "vfr")


def build_plan(job_id: str, source_timeline: Optional[str],
               fps_value: Optional[str] = None,
               fps_source: Optional[str] = None,
               pts_usable: int = 0, expected_frames: int = 0,
               assumptions: Optional[list] = None) -> Dict[str, Any]:
    """stage3 侧的构造。

    Args:
        source_timeline: ``cfr`` / ``vfr`` / ``unknown``。
        fps_value: 精确有理数字符串（``30000/1001``）。
        fps_source: ``probe`` / ``pts-derived`` / ``user-assumed`` /
            ``manifest-assumed``。**必须如实**：manifest 注入的值不是用户选择。
        pts_usable / expected_frames: 覆盖度证据。
    """
    tl = (source_timeline or "unknown").strip().lower()
    src_verified = tl in VERIFIED_STATES
    _manual = fps_source in ("user-assumed", "manifest-assumed")
    _assumptions = list(assumptions or [])
    if _manual and not _assumptions:
        # 人工/清单来源**自动**留痕，不依赖调用方记得传
        _assumptions.append(
            f"fps came from a spec ({fps_source}), not from measurement")
    if tl not in VERIFIED_STATES and not _assumptions:
        _assumptions.append("source timeline could not be established")
    return {
        "schema": SCHEMA,
        "job_id": job_id,
        "source_timeline": tl,
        "source_verified": src_verified,
        "fps": ({"value": str(fps_value), "source": fps_source}
                if fps_value else None),
        "pts_coverage": {
            "usable": int(pts_usable or 0),
            "expected": int(expected_frames or 0),
            "complete": src_verified,
        },
        "manual_interpretation": _manual,
        "assumptions": _assumptions,
        # 以下两项由 stage7 填
        "output_measured": False,
        "transform_applied_verified": None,
    }


def claim_source_verified(plan: Optional[Dict[str, Any]],
                          timing_state: Optional[str]) -> bool:
    """唯一允许宣称"**源**时序已验证"的判定。

    只有"上游确实验证过源时间轴"**且**"本次时序判定为 verified"才成立。
    成品能不能测（③）与此无关 —— 这正是原实现的混淆点。

    Args:
        plan: stage3 写下的计划；``None`` 表示没有上游计划。
        timing_state: stage7 本次的时序判定（``verified`` / ``adjusted`` /
            ``unverified`` / ``unexplained``）。

    Returns:
        bool。缺任何一侧证据都返回 ``False``。
    """
    if not isinstance(plan, dict):
        return False
    if plan.get("source_verified") is not True:
        return False
    return timing_state == "verified"


def write_plan(temp_dir: str, plan: Dict[str, Any]) -> Optional[str]:
    """写到 ``{temp_dir}/{job_id}_timeline_plan.json``；失败返回 None。"""
    try:
        path = os.path.join(str(temp_dir),
                            f"{plan.get('job_id', 'job')}_timeline_plan.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        return path
    except (OSError, TypeError):
        return None


def read_plan(temp_dir: str, job_id: str) -> Optional[Dict[str, Any]]:
    """读回计划；不存在/读不出/结构不对都返回 ``None``（不猜）。"""
    try:
        path = os.path.join(str(temp_dir), f"{job_id}_timeline_plan.json")
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            return None
        return doc
    except (OSError, ValueError, TypeError):
        return None
