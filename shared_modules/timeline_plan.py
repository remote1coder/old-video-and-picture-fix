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
        # stage3 能确定的：用户是否要求把时间轴按某个帧率重解释。
        # **注意**：这里不再写 `transform_applied_verified: None` ——
        # 一个"声明了却永远是 None"的字段比没有更糟（读的人会以为它有意义）。
        # 变换是否真的落地由 stage7 用 `build_transform_verdict()` 裁定。
        "transform_request": {
            "fps_reinterpreted": _manual,
            "fps_source": fps_source,
            "requested_by_user": (fps_source == "user-assumed"),
        },
        # stage7 填
        "output_measured": False,
    }


def build_transform_verdict(declared_speed=None, declared_tempo=None,
                            original_duration=None, processed_duration=None,
                            timing_state: Optional[str] = None,
                            tempo_applied: float = 1.0,
                            tolerance: float = 0.01) -> Dict[str, Any]:
    """裁定"用户请求的变换是否真的执行了"（结论 ②）。

    `source_verified`（① 源时间轴可信）与 `output_measured`（③ 输出测得）
    都不能替代它：源可信、输出可测，**不代表**用户要求的变速/重解释落地了；
    反过来，变换执行正确也不代表源时间轴被验证过。

    Args:
        declared_speed / declared_tempo: 上游声明的变速意图（None = 没要求）。
        original_duration / processed_duration: 用于算实测比值。
        timing_state: ``_decide_timing()`` 的状态
            （``verified`` / ``adjusted`` / ``unverified`` / ``unexplained``）。
        tempo_applied: 实际施加的 atempo（1.0 = 没改音频）。
        tolerance: 相对容差。

    Returns:
        dict，``status`` ∈ ``NOT_APPLICABLE`` / ``PASS`` / ``FAIL`` / ``UNVERIFIED``。
        字段永远齐全，不会出现 None 占位。
    """
    _declared = None
    _kind = []
    for _v, _name in ((declared_tempo, "declared_tempo"),
                      (declared_speed, "declared_speed")):
        if _v is None:
            continue
        try:
            _f = float(_v)
        except (TypeError, ValueError):
            continue
        if _f <= 0:
            continue
        _kind.append(_name)
        if _declared is None:
            _declared = _f

    out: Dict[str, Any] = {
        "requested": _declared is not None,
        "kind": "+".join(_kind) if _kind else "none",
        "expected_ratio": _declared,
        "observed_ratio": None,
        "tempo_applied": float(tempo_applied or 1.0),
        "status": "NOT_APPLICABLE",
        "reason": "",
    }

    if _declared is None:
        out["reason"] = "no speed/tempo change was requested"
        return out

    # 有请求：先看能不能算出实测比值
    try:
        _orig = float(original_duration or 0)
        _proc = float(processed_duration or 0)
    except (TypeError, ValueError):
        _orig = _proc = 0.0
    if _orig > 0 and _proc > 0:
        out["observed_ratio"] = round(_orig / _proc, 6)

    if timing_state in ("unverified", "unexplained"):
        out["status"] = "UNVERIFIED"
        out["reason"] = (
            f"the transform was requested ({out['kind']}={_declared}), but the "
            f"timing could not be verified (timing_state={timing_state}); the "
            "requested change is NOT confirmed to have been applied")
        return out

    if out["observed_ratio"] is None:
        out["status"] = "UNVERIFIED"
        out["reason"] = ("durations unavailable; cannot confirm the requested "
                         f"{out['kind']}={_declared} was applied")
        return out

    _rel = abs(out["observed_ratio"] - _declared) / max(_declared, 1e-9)
    if _rel <= tolerance:
        out["status"] = "PASS"
        out["reason"] = (f"{out['kind']}={_declared} matches the measured "
                         f"ratio {out['observed_ratio']}")
    else:
        out["status"] = "FAIL"
        out["reason"] = (f"{out['kind']}={_declared} but the measured ratio is "
                         f"{out['observed_ratio']} "
                         f"({_rel * 100:.1f}% off, tolerance "
                         f"{tolerance * 100:.1f}%)")
    return out


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
