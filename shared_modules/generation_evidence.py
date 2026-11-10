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


def collect(src: str, out: str, stage: str,
            params: Optional[Dict[str, Any]] = None,
            logger=None) -> Dict[str, Any]:
    """一次拿齐两类证据，直接塞进回执。

    Returns:
        ``{"color": {...}, "bounds": {...}}``。
    """
    return {"color": color_group(out),
            "bounds": bounds_report(src, out, stage, params, logger)}


def receipt_evidence_problems(rcpt: Dict[str, Any]) -> list:
    """回执里的证据块是否足以支撑"这一次转换是对的"。

    返回问题列表（空 = 可以采信）。消费方（``fused_receipt``）用它把
    "证据不全"和"证据说不对"分开处理，但两者都**不能**当通过。
    """
    problems = []
    color = (rcpt or {}).get("color")
    if not isinstance(color, dict):
        problems.append("no color evidence (legacy receipt)")
    else:
        if not color.get("probed"):
            problems.append("color was never probed")
        elif not color.get("complete"):
            problems.append("color fields unknown: "
                            + ",".join(color.get("unknown_fields") or []))
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
