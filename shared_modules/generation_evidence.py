# -*- coding: utf-8 -*-
"""生成回执要绑定的**证据块**：颜色成组描述 + 物理边界判定（审核 R19-A03）。

## 为什么需要这个模块

`stage3` 的融合回执（G34：stage3 里顺手把 stage4 也做了）只记了
``out_pix_fmt`` 一个字符串。审核指出两个缺口：

1. **颜色是一组属性**（matrix / range / primaries / transfer）。单看
   ``pix_fmt`` 既不能证明"没被改色"，也不能证明"改对了色" ——
   4:2:0 配 bt601 与配 bt709 是两种完全不同的画面。
2. **物理边界没有判定结论**。经典路线由 ``stage4_postprocess.py`` 自己调
   ``frame_conservation.check_conservation()`` 做帧数/时长守恒自查；
   融合路线把 stage4 吞进 stage3 之后，这一步**从来没跑过** ——
   等于"优化路径绕过了唯一的边界门"。

本模块把这两项抽成共享代码，两条路线用同一份实现，避免再次漂移
（与 ``fused_receipt.py`` 是同一个理由）。

## 判定语义

- ``color.complete=False`` 表示**有字段未知**，不等于"颜色错了"，
  但足以让回执标成"证据不全"，消费方必须把它当"未证明"。
- ``bounds.ok=False`` 表示**已知边界被破坏**，回执不得被采信。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

#: 颜色描述里必须成组出现的字段（缺一项就是"证据不全"）
COLOR_FIELDS = ("pix_fmt", "range", "matrix", "primaries", "transfer")


def color_group(path: str) -> Dict[str, Any]:
    """探测产物**实际**的颜色表示，成组返回（不做"猜一个标准"的塌缩）。

    Returns:
        ``{pix_fmt, range, matrix, primaries, transfer, probed, complete,
        unknown_fields, error}``。``probed=False`` 表示 ffprobe 失败，
        此时所有字段都不可信。
    """
    out: Dict[str, Any] = {k: None for k in COLOR_FIELDS}
    out.update({"probed": False, "complete": False,
                "unknown_fields": list(COLOR_FIELDS), "error": ""})
    try:
        from media_io import probe_color
    except Exception as exc:                      # pragma: no cover - 部署异常
        out["error"] = f"media_io.probe_color unavailable: {exc}"
        return out
    info = probe_color(str(path)) or {}
    out["probed"] = bool(info.get("probed"))
    for k in COLOR_FIELDS:
        out[k] = info.get(k)
    if not out["probed"]:
        out["error"] = str(info.get("error") or "probe failed")
        return out
    unknown = [k for k in COLOR_FIELDS if not out.get(k)]
    out["unknown_fields"] = unknown
    out["complete"] = not unknown
    return out


def bounds_report(src: str, out: str, stage: str, params: Optional[Dict[str, Any]] = None,
                  logger=None) -> Dict[str, Any]:
    """帧数/时长守恒的**判定结论**（与经典 stage4 调用的是同一份实现）。

    Returns:
        ``{ok, performed, report, reason}``。``performed=False`` 表示核对
        没能执行（模块缺失等）—— 这是"未验证"，不是"通过"。
    """
    p = params or {}
    res: Dict[str, Any] = {"ok": False, "performed": False,
                           "report": None, "reason": ""}
    try:
        from frame_conservation import check_conservation
        from rate_spec import parse_rate_or_none as _prn
    except Exception as exc:                      # pragma: no cover - 部署异常
        res["reason"] = f"helpers unavailable: {exc}"
        if logger:
            logger.warning("G34_BOUNDS_SKIP",
                           "frame_conservation/rate_spec missing; the boundary "
                           "check was NOT performed", error=str(exc))
        return res

    want = _prn(p.get("original_fps") or p.get("fps"))
    want_dur = None
    try:
        _wd = p.get("duration")
        want_dur = float(_wd) if _wd else None
    except (TypeError, ValueError):
        want_dur = None

    try:
        ok, rep = check_conservation(
            str(src), str(out), stage, logger,
            expected_fps=(want.value if want else None),
            expected_duration=want_dur)
    except Exception as exc:                      # pragma: no cover - 探测异常
        res["reason"] = f"check_conservation raised: {exc}"
        if logger:
            logger.error("G34_BOUNDS_ERROR",
                         "The boundary check raised; treating the output as "
                         "UNVERIFIED, not as passing.", error=str(exc))
        return res
    res.update(ok=bool(ok), performed=True, report=rep)
    if not ok:
        res["reason"] = "; ".join((rep or {}).get("reasons") or [])[:400]
    return res


def _changed_tags(src_color: Dict[str, Any],
                  out_color: Dict[str, Any]) -> list:
    """源与产物**都有值但不同**的颜色字段（R21-C06）。

    审核："known→另一 known 值，需要与实际转换吻合"。融合路径里 stage3 与
    stage4 都不做色彩空间转换（只做 16 位 4:2:0 的像素格式转换），所以
    matrix/range/primaries/transfer 的任何变化都意味着"标签被改写"或
    "确实做了转换但没声明" —— 两种都必须报出来，不能静默。
    """
    changed = []
    for k in ("range", "matrix", "primaries", "transfer"):
        a, b = src_color.get(k), out_color.get(k)
        if a and b and str(a) != str(b):
            changed.append({"field": k, "source": a, "output": b})
    return changed


def _lost_tags(src_color: Dict[str, Any],
               out_color: Dict[str, Any]) -> list:
    """源**有**标签、产物**丢了**的字段列表。

    这才是真正的问题。反例（2026-11-10 N20 实测）：合成素材本身没有
    matrix/primaries/transfer 标签（`STAGE1_COLOR_RANGE_UNKNOWN`），
    融合产物同样没有 —— 这是**正确**的。若要求"字段必须齐全"，
    就等于逼实现去**编造**一个矩阵，正好违反 COL-04
    "unknown 不等于 known，绝不填默认值"。

    所以判据是**相对**的：只在"源说得出、产物说不出"时才算缺陷。
    """
    lost = []
    for k in ("pix_fmt", "range", "matrix", "primaries", "transfer"):
        if src_color.get(k) and not out_color.get(k):
            lost.append(k)
    return lost


def collect(src: str, out: str, stage: str,
            params: Optional[Dict[str, Any]] = None,
            logger=None) -> Dict[str, Any]:
    """一次拿齐两类证据，直接塞进回执。

    Returns:
        ``{"color": {...}, "color_source": {...}, "color_lost_tags": [...],
        "bounds": {...}}``。
    """
    oc = color_group(out)
    sc = color_group(src)
    return {"color": oc,
            "color_source": sc,
            "color_lost_tags": _lost_tags(sc, oc),
            # R21-C06：源与产物都有值但不同 → 必须报出来
            "color_changed_tags": _changed_tags(sc, oc),
            "bounds": bounds_report(src, out, stage, params, logger)}


def receipt_evidence_problems(rcpt: Dict[str, Any]) -> list:
    """回执里的证据块是否足以支撑"这一次转换是对的"。

    返回问题列表（空 = 可以采信）。消费方（``fused_receipt``）用它把
    "证据不全"和"证据说不对"分开处理，但两者都**不能**当通过。

    颜色判据是**相对**的（见 `_lost_tags`）：产物字段不全本身不是问题，
    "源有、产物没有"才是。
    """
    problems = []
    color = (rcpt or {}).get("color")
    if not isinstance(color, dict):
        problems.append("no color evidence (legacy receipt)")
    else:
        if not color.get("probed"):
            problems.append("color was never probed")
        lost = (rcpt or {}).get("color_lost_tags")
        if isinstance(lost, list) and lost:
            problems.append("colour tags lost between input and output: "
                            + ",".join(lost))
        changed = (rcpt or {}).get("color_changed_tags")
        if isinstance(changed, list) and changed:
            problems.append(
                "colour tags changed between input and output without a "
                "declared conversion: "
                + ",".join(f"{c.get('field')}={c.get('source')}->{c.get('output')}"
                           for c in changed if isinstance(c, dict)))
    bounds = (rcpt or {}).get("bounds")
    if not isinstance(bounds, dict):
        problems.append("no bounds evidence (legacy receipt)")
    elif not bounds.get("performed"):
        problems.append("bounds check was not performed: "
                        + str(bounds.get("reason") or "unknown reason")[:200])
    elif not bounds.get("ok"):
        problems.append("bounds check failed: "
                        + str(bounds.get("reason") or "")[:200])
    return problems
