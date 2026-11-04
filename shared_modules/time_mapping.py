# -*- coding: utf-8 -*-
"""预期输出映射：把"用户要求的变换"算成**逐帧期望时间轴**。

## 为什么需要它

审核 R17-03 的核心结论：

> 进入 VFR 重封装后，代码按原素材 PTS 回填，未按用户重解释、变速、
> IVTC/QTGMC 选择输出映射。受控探针确认：合法 2 倍速度的同帧数结果被改回
> 源 PTS；合法倍帧结果则因帧数不同被拒绝。

也就是说，原来的判据只有两种极端：**要么按源时间轴原样套**，**要么拒绝**。
中间那一大片"用户确实要求了变换"的情形没有表达方式。

本模块补上这块：给定源时间轴 + 一个**明确的变换描述**，算出期望的逐帧 PTS，
再用它去核对候选。判据从"像不像源"变成"像不像**你要的那个东西**"。

## 支持哪些变换

| kind | 含义 | 期望时间轴 |
|---|---|---|
| ``identity`` | 保时保帧（默认） | 与源逐帧一致 |
| ``speed`` | 整体变速（音频同步由上层负责） | 每帧 PTS 除以 ``value`` |
| ``fps_reinterpret`` | 人工帧率重解释（同帧数、换时间基） | 从首帧起按 ``1/value`` 重新排 |
| ``decimate`` | 抽帧（如 IVTC cycle=5 去 1 帧） | 每 ``value`` 帧丢最后一帧 |
| ``double`` | 倍帧（如 QTGMC 50i→50p） | 每帧复制一份，间隔减半 |

> 这是**最小可用**集合：够覆盖审核点名的四种场景（保时、变速、重解释、倍帧/抽帧）。
> 复合变换（例如"先倍帧再变速"）目前需要调用方自己组合 —— 组合语义（谁先谁后、
> 音频跟谁）还没定，先不假装支持。

## 与"源 PTS 回填"的关系

`identity` 的期望序列就是源 PTS，因此原来那种"按源回填"的行为仍然可达 ——
但它现在只是**五种变换中的一种**，不再是对所有情况的默认答案。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 末帧时间容差（秒）。mkvmerge v2 默认 timestamp scale 为 1ms，
#: 正常误差 1–2ms；50ms 只用于抓"整体对不上"这类粗错。
LAST_TOLERANCE_S = 0.05

#: 逐帧间隔容差（秒）。与 `vfr_timecodes.INTERVAL_TOLERANCE` 同口径。
INTERVAL_TOLERANCE_S = 0.02


@dataclass(frozen=True)
class Transform:
    """一次时间变换的**明确**描述。

    Args:
        kind: ``identity`` / ``speed`` / ``fps_reinterpret`` / ``decimate`` /
            ``double`` 之一。
        value: 含义随 kind 变化（speed 倍数、目标 fps、抽帧周期）。
            对 ``identity`` 与 ``double`` 无意义。
    """

    kind: str = "identity"
    value: float = 1.0

    def normalized(self) -> "Transform":
        k = (self.kind or "identity").strip().lower()
        if k not in ("identity", "speed", "fps_reinterpret", "decimate",
                     "double"):
            raise ValueError(f"unknown transform kind: {self.kind!r}")
        v = float(self.value if self.value is not None else 1.0)
        if k in ("speed", "fps_reinterpret", "decimate") and v <= 0:
            raise ValueError(f"{k} requires a positive value, got {v!r}")
        return Transform(k, v)


def transform_from_params(params: Optional[Dict[str, Any]]) -> Transform:
    """从 params 推断变换。**只认显式声明**，不猜。

    优先级：``fps_interpret`` > ``expected_speed`` > ``identity``。
    （``decimate`` / ``double`` 由调用方按 stage1 的实际处理显式给出，
    因为仅凭 params 无法区分"倍帧"和"抽帧"是否真的发生了。）
    """
    p = params or {}

    _fp = p.get("fps_interpret")
    if _fp not in (None, ""):
        try:
            fps = float(str(_fp).split("/")[0]) / (
                float(str(_fp).split("/")[1]) if "/" in str(_fp) else 1.0)
            if fps > 0:
                return Transform("fps_reinterpret", fps)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    _sp = p.get("expected_speed")
    if _sp not in (None, ""):
        try:
            sp = float(_sp)
            if sp > 0 and abs(sp - 1.0) > 1e-9:
                return Transform("speed", sp)
        except (TypeError, ValueError):
            pass

    return Transform("identity", 1.0)


def expected_pts(source_pts: Sequence[float],
                 transform: Transform) -> List[float]:
    """按变换算出**期望的**逐帧 PTS（秒）。

    Raises:
        ValueError: 变换描述非法，或源时间轴不足两帧（无法推断间隔）。
    """
    t = transform.normalized()
    src = [float(x) for x in source_pts]
    if len(src) < 2:
        raise ValueError("need at least two source timestamps")
    base = src[0]

    if t.kind == "identity":
        return list(src)

    if t.kind == "speed":
        # 保持帧数，整体时间轴按倍数压缩/拉伸（音频由上层 atempo 对齐）
        return [base + (p - base) / t.value for p in src]

    if t.kind == "fps_reinterpret":
        step = 1.0 / t.value
        return [base + i * step for i in range(len(src))]

    if t.kind == "double":
        # 每帧复制一份：原间隔减半，帧数翻倍
        out: List[float] = []
        for i, p in enumerate(src):
            out.append(p)
            if i + 1 < len(src):
                out.append(base + (p - base) + (src[i + 1] - p) / 2.0)
        return out

    if t.kind == "decimate":
        cycle = int(round(t.value))
        if cycle < 2:
            raise ValueError("decimate cycle must be >= 2")
        return [p for i, p in enumerate(src) if i % cycle != cycle - 1]

    raise ValueError(f"unsupported transform: {t.kind!r}")  # pragma: no cover


def describe_mapping(source_pts: Sequence[float],
                     transform: Transform) -> Dict[str, Any]:
    """生成可直接写进报告/身份摘要的映射描述。"""
    t = transform.normalized()
    exp = expected_pts(source_pts, t)
    intervals = [b - a for a, b in zip(exp, exp[1:])]
    return {
        "transform": {"kind": t.kind, "value": t.value},
        "source_frames": len(source_pts),
        "expected_frames": len(exp),
        "first_pts": round(exp[0], 6) if exp else None,
        "last_pts": round(exp[-1], 6) if exp else None,
        "last_duration_s": round(intervals[-1], 6) if intervals else None,
        "median_interval_s": (round(sorted(intervals)[len(intervals) // 2], 6)
                              if intervals else None),
    }


def check_against_expected(actual_pts: Sequence[float],
                           expected: Sequence[float],
                           interval_tolerance: float = INTERVAL_TOLERANCE_S,
                           last_tolerance: float = LAST_TOLERANCE_S
                           ) -> Tuple[bool, List[str], Dict[str, Any]]:
    """把候选实测时间轴与**期望**时间轴对照。

    检查项（缺一不可）：
    1. 帧数一致；
    2. 逐帧间隔的最大偏差 ≤ ``interval_tolerance``（**不能用末帧掩盖中间错**）；
    3. 末帧相对首帧的跨度偏差 ≤ ``last_tolerance``。

    Returns:
        ``(ok, problems, metrics)``
    """
    problems: List[str] = []
    metrics: Dict[str, Any] = {
        "actual_frames": len(actual_pts),
        "expected_frames": len(expected),
    }
    if not expected:
        return False, ["expected mapping is empty"], metrics
    if len(actual_pts) != len(expected):
        problems.append(
            f"frame count differs: actual={len(actual_pts)} "
            f"expected={len(expected)}")
        return False, problems, metrics

    a = [float(x) for x in actual_pts]
    e = [float(x) for x in expected]
    a_span = a[-1] - a[0]
    e_span = e[-1] - e[0]
    span_err = abs(a_span - e_span)
    metrics.update(actual_last=round(a[-1], 6), expected_last=round(e[-1], 6),
                   duration_error=round(span_err, 6))

    if span_err > last_tolerance:
        problems.append(
            f"duration off by {span_err:.3f}s (tolerance {last_tolerance:.3f}s)")

    a_int = [b - x for x, b in zip(a, a[1:])]
    e_int = [b - x for x, b in zip(e, e[1:])]
    max_err = max((abs(x - y) for x, y in zip(a_int, e_int)), default=0.0)
    metrics["max_interval_error"] = round(max_err, 6)
    if max_err > interval_tolerance:
        problems.append(
            f"per-frame intervals off by {max_err:.3f}s "
            f"(tolerance {interval_tolerance:.3f}s)")

    return (not problems), problems, metrics
