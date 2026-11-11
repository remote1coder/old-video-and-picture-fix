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

import math

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
                 transform: Transform,
                 last_duration: Optional[float] = None) -> List[float]:
    """按变换算出**期望的**逐帧 PTS（秒）。

    Args:
        source_pts: 源逐帧 PTS（秒，严格递增）。
        transform: 变换描述。
        last_duration: 源**末帧的持续时间**（秒）。只有"源不足两帧、
            无法从相邻帧推断间隔"时才需要；不给就拒绝，不能凭空猜时刻。

    Raises:
        ValueError: 变换非法、源为空、时间戳非有限、或缺少必需的 duration。

    Note:
        `decimate` 是**理想模型**（假设每 cycle 丢最后一帧），
        不代表真实 `VDecimate` 的内容选择；`describe_mapping()` 会把它
        标成 ``model="idealised"``，下游不得据此断言等价。
    """
    t = transform.normalized()
    src = [float(x) for x in source_pts]
    if not src:
        raise ValueError("source timeline is empty")
    for i, p in enumerate(src):
        if not math.isfinite(p):
            # NaN 会让后续所有比较恒为 False（表现为"通过"），必须在入口挡住
            raise ValueError(f"source timestamp #{i} is not finite: {p!r}")
    base = src[0]

    if t.kind == "identity":
        # 保时：单帧也合法，不需要间隔
        return list(src)

    if t.kind == "speed":
        return [base + (p - base) / t.value for p in src]

    if t.kind == "fps_reinterpret":
        step = 1.0 / t.value
        return [base + i * step for i in range(len(src))]

    if t.kind == "double":
        # ---- 倍帧：每个输入帧产生**两个**输出时刻（不是 2N−1）----
        # 审核 R18-A05：
        #   > `double` 对 N 帧只产生 2N−1 个 PTS：3 帧得到 5 帧，
        #   > 最后一个场没有表达。新增测试直接断言 3→5，固化了缺陷。
        # 物理含义：每帧的两场各占半个源间隔；末帧的第二场同样占半个间隔，
        # 所以 3 帧 @0.04 → 0, .02, .04, .06, .08, .10（6 个）。
        if len(src) >= 2:
            # 显式 last_duration 的校验（与单帧分支同一口径）
            _ld = None
            if last_duration is not None:
                _ld = float(last_duration)
                if not math.isfinite(_ld) or _ld <= 0:
                    raise ValueError(
                        "last_duration must be a positive finite number, got "
                        f"{last_duration!r}")

            out: List[float] = []
            for i, p in enumerate(src):
                out.append(p)
                if i + 1 < len(src):
                    half = (src[i + 1] - p) / 2.0
                elif _ld is not None:
                    # 审核 R19-B05：**显式源 last_duration 优先于推断**。
                    # 反例：3 帧 [0,.04,.08] + last_duration=.2 →
                    # 末场应为 .08+.1=.18，而不是按前一间隔得到的 .10。
                    half = _ld / 2.0
                else:
                    # 末帧：用**前一间隔**推断第二场。对 CFR 源是准确的；
                    # 对 VFR 源这是一次推断，会写进 describe_mapping 的
                    # last_duration_basis，报告能看出它的来源。
                    half = (p - src[i - 1]) / 2.0
                if not math.isfinite(half) or half <= 0:
                    raise ValueError(
                        "source timestamps must be strictly increasing to "
                        f"derive a half-interval (at index {i})")
                out.append(p + half)
            return out

        # 单帧：必须有显式末帧 duration
        if last_duration is None:
            raise ValueError(
                "doubling a single-frame clip requires an explicit "
                "last_duration (frame duration in seconds); refusing to "
                "invent a timestamp")
        d = float(last_duration)
        if not math.isfinite(d) or d <= 0:
            raise ValueError(
                "last_duration must be a positive finite number, got "
                f"{last_duration!r}")
        return [base, base + d / 2.0]

    if t.kind == "decimate":
        cycle = int(round(t.value))
        if cycle < 2:
            raise ValueError("decimate cycle must be >= 2")
        return [p for i, p in enumerate(src) if i % cycle != cycle - 1]

    raise ValueError(f"unsupported transform: {t.kind!r}")  # pragma: no cover


def describe_mapping(source_pts: Sequence[float],
                     transform: Transform,
                     last_duration: Optional[float] = None) -> Dict[str, Any]:
    """生成可直接写进报告/身份摘要的映射描述。

    ``model`` 字段区分：

    - ``exact``：`identity` / `speed` / `fps_reinterpret` / `double` ——
      给定源时间轴就能算出确定的期望值。
    - ``idealised``：`decimate` —— 这里假设"每 cycle 丢最后一帧"，
      而真实 `VDecimate` 的选择由**内容**决定（哪帧更像重复场）。
      只能做粗检，**不能用来断言等价**（审核 R18-A05）。

    ``last_duration_basis`` 记录 `double` 末帧第二场的时刻是怎么来的：
    ``inferred_from_previous_interval``（推断）或 ``given``（调用方显式给）。
    """
    t = transform.normalized()
    exp = expected_pts(source_pts, t, last_duration=last_duration)
    intervals = [b - a for a, b in zip(exp, exp[1:])]

    if t.kind == "double":
        if len(source_pts) < 2:
            basis = "given" if last_duration is not None else "unknown"
        else:
            basis = "inferred_from_previous_interval"
    else:
        basis = None

    return {
        "transform": {"kind": t.kind, "value": t.value},
        "model": "idealised" if t.kind == "decimate" else "exact",
        "last_duration_basis": basis,
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
                           interval_tolerance: Optional[float] = None,
                           last_tolerance: Optional[float] = None,
                           time_base: Optional[float] = None,
                           allow_origin_shift: bool = False
                           ) -> Tuple[bool, List[str], Dict[str, Any]]:
    """把候选实测时间轴与**期望**时间轴对照。

    检查项（缺一不可）：

    1. 两侧都是**有限**数值 —— NaN 会让 ``err > tol`` 恒为 False，静默通过；
    2. 两侧都**严格递增** —— 不单调的时间轴不是时间轴；
    3. 帧数一致；
    4. 起始原点一致（除非显式 ``allow_origin_shift=True``）；
    5. 逐帧间隔最大偏差 ≤ 间隔容差；
    6. 末帧相对首帧的跨度偏差 ≤ 跨度容差。

    容差来源（审核 R18-A05："按容器量化、采样时基与明确规则计算"）：

    - 给 ``time_base``（容器时间基，秒）：
      ``interval_tolerance = 2 × time_base``；
      ``last_tolerance = (N−1) × interval_tolerance``（跨度是逐帧误差的累积）。
    - 没给：退回固定默认值，并在 metrics 标 ``default_fixed``，
      让报告能看出这不是从时间基推出来的。

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

    # ---- 1) 有限性 ----
    for name, seq in (("actual", a), ("expected", e)):
        bad = [i for i, v in enumerate(seq) if not math.isfinite(v)]
        if bad:
            problems.append(
                f"{name} has non-finite timestamps at index {bad[:5]} "
                f"(count={len(bad)})")
    if problems:
        return False, problems, metrics

    # ---- 2) 严格递增 ----
    for name, seq in (("actual", a), ("expected", e)):
        for i in range(1, len(seq)):
            if seq[i] <= seq[i - 1]:
                problems.append(
                    f"{name} timestamps are not strictly increasing at index "
                    f"{i} ({seq[i - 1]!r} -> {seq[i]!r}); a non-monotonic in "
                    f"time order cannot be accepted")
                break
    if problems:
        return False, problems, metrics

    # ---- 3) 容差来源 ----
    if interval_tolerance is None:
        if time_base is not None:
            tb = abs(float(time_base))
            if not math.isfinite(tb) or tb <= 0:
                return False, [f"time_base must be positive and finite: "
                               f"{time_base!r}"], metrics
            interval_tolerance = 2.0 * tb
            tol_source = "derived_from_time_base"
        else:
            interval_tolerance = INTERVAL_TOLERANCE_S
            tol_source = "default_fixed"
    else:
        tol_source = "explicit"

    # 审核 R19-B05：显式容差本身必须有限且非负，否则一切都能通过
    for _name, _v in (("interval_tolerance", interval_tolerance),
                      ("last_tolerance", last_tolerance)):
        if _v is not None:
            try:
                _f = float(_v)
            except (TypeError, ValueError):
                return False, [f"{_name} is not a number: {_v!r}"], metrics
            if not math.isfinite(_f) or _f < 0:
                return False, [f"{_name} must be finite and >= 0: {_v!r}"], metrics
    metrics["interval_tolerance_s"] = interval_tolerance
    metrics["interval_tolerance_source"] = tol_source

    if last_tolerance is None:
        # 审核 R19-B05：**不能**用 (N−1)×interval 让容差随片长膨胀。
        # 末帧/跨度的误差不是一个"每帧误差累加"的量 —— 若每帧都准，跨度自然准。
        # 给 2 倍逐点容差，覆盖末帧 duration 与量化边界的不确定性。
        last_tolerance = 2.0 * interval_tolerance
    metrics["last_tolerance_s"] = last_tolerance

    # ---- 4) 原点 ----
    shift = a[0] - e[0]
    metrics["origin_shift_s"] = round(shift, 9)
    if abs(shift) > interval_tolerance and not allow_origin_shift:
        problems.append(
            f"timeline origin differs by {shift:.6f}s; pass "
            f"allow_origin_shift=True only when an explicit shared A/V "
            f"origin contract exists")

    # ---- 5) 跨度与逐帧间隔（用相对量，抵消原点）----
    a_span = a[-1] - a[0]
    e_span = e[-1] - e[0]
    span_err = abs(a_span - e_span)
    metrics.update(actual_last=round(a[-1], 6),
                   expected_last=round(e[-1], 6),
                   duration_error=round(span_err, 6))
    if span_err > last_tolerance:
        problems.append(
            f"duration off by {span_err:.6f}s "
            f"(tolerance {last_tolerance:.6f}s)")

    a_int = [b - x for x, b in zip(a, a[1:])]
    e_int = [b - x for x, b in zip(e, e[1:])]
    max_err = max((abs(x - y) for x, y in zip(a_int, e_int)), default=0.0)
    metrics["max_interval_error"] = round(max_err, 6)
    if max_err > interval_tolerance:
        problems.append(
            f"per-frame intervals off by {max_err:.6f}s "
            f"(tolerance {interval_tolerance:.6f}s)")

    # ---- 6) 逐帧**累计残差**（R21-C05）----
    # 审核的精确反例：expected[i] = 0.04i，
    # actual[i] = expected[i] + 0.001·min(i, 2000−i)，i = 0…2000。
    # 每一步只差 1 ms（小于 2 ms 间隔容差）、端点完全一致（跨度误差 0），
    # 但中段整整偏了 1 秒 —— 旧判据全部通过。
    #
    # 为什么必须单独查：**相邻差**只能发现"某一步跳了"，
    # "每一步都少一点点"只有把残差**累加**才看得见。
    # 残差用相对原点计算，所以合法的整体平移不会误报。
    # 容差沿用逐点容差（不随帧数放大）—— 逐帧 PTS 若真是一对一复制，
    # 舍入误差不会累积。
    resid = [(x - a[0]) - (y - e[0]) for x, y in zip(a, e)]
    worst_i = max(range(len(resid)), key=lambda i: abs(resid[i]))
    metrics.update(max_residual_s=round(resid[worst_i], 6),
                   max_residual_index=worst_i,
                   residual_tolerance_s=interval_tolerance)
    if abs(resid[worst_i]) > interval_tolerance:
        problems.append(
            f"per-frame timestamps drift by {resid[worst_i]:.6f}s at frame "
            f"{worst_i} (tolerance {interval_tolerance:.6f}s); the middle of "
            f"the timeline does not match even though the endpoints do")

    return (not problems), problems, metrics
