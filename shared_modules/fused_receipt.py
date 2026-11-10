# -*- coding: utf-8 -*-
"""G34 融合回执的**共享验证器**（审核 R19-A03）。

## 为什么要有这个模块

`main._fused_step34()` 与 `stage5_ai_face.py` 都要回答同一个问题：

> "这份 `.fused_stage34.json` 能不能证明**这一次**的 stage4 已经跑过？"

它们各自实现过一遍（R20 两轮），而审核明确要求
「main/stage5 通过**共享验证器**消费」。两份实现的危险不是重复本身，
而是**它们会漂移**——一处收紧了、另一处还松着，就变成"堵了一个入口、
另一个入口照旧"。

## 判定顺序

1. 回执是 dict 且带 `attempt_id`、与期望一致（不认旧回执）
2. 带 `input_digest`（= `compute_param_hash(源, params)`），
   且与**现算**的一致（源内容或参数变了 → 过期）
3. `output` 非空、存在、且 `output_size` 与记录一致（产物被换过 → 过期）

任何一步不满足都返回 ``(False, reason)``，调用方**走经典路线**而不是报错
——融合是优化，不是必须；用它换来的误判才是真损失。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple


def _current_digest(src_path: str, params: Dict[str, Any]) -> Optional[str]:
    """现算"源 + 参数"的身份摘要；算不出返回 None。"""
    if not src_path:
        return None
    try:
        from checkpoint import compute_param_hash
        return compute_param_hash(str(src_path), params or {})
    except Exception:
        return None


def verify_fused_receipt(rcpt_path: str, params: Dict[str, Any],
                         expected_attempt: str = "",
                         src_path: str = "") -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """校验一份融合回执是否可用于**本次**执行。

    Args:
        rcpt_path: 回执文件路径。
        params: 本次有效参数（用于现算 input_digest）。
        expected_attempt: 本次 attempt；为空则**不校验** attempt
            （保留给"手工检查"这类无 attempt 场景，生产必须传）。
        src_path: 源素材路径。为空时按
            ``params["input_path"]`` → ``params["_input_path"]`` →
            ``rcpt["input_path"]`` 三级兜底。

    Returns:
        ``(ok, reason, rcpt)``。``ok`` 为真时 ``rcpt`` 是已解析的回执 dict。
    """
    try:
        with open(rcpt_path, "r", encoding="utf-8") as fh:
            rcpt = json.load(fh) or {}
    except (OSError, ValueError) as exc:
        return False, f"receipt unreadable: {exc}", None
    if not isinstance(rcpt, dict):
        return False, "receipt is not an object", None

    # ---- 1) attempt ----
    if expected_attempt:
        got = str(rcpt.get("attempt_id") or "")
        if not got:
            return False, "receipt has no attempt_id (legacy)", rcpt
        if got != str(expected_attempt):
            return False, (f"receipt belongs to attempt {got!r}, "
                           f"expected {expected_attempt!r}"), rcpt

    # ---- 2) input_digest ----
    rec_dig = str(rcpt.get("input_digest") or "")
    if not rec_dig:
        return False, "receipt has no input_digest (legacy)", rcpt
    src = (src_path or params.get("input_path")
           or params.get("_input_path") or rcpt.get("input_path") or "")
    cur_dig = _current_digest(str(src), params)
    if cur_dig is None:
        return False, "cannot compute the current input digest", rcpt
    if rec_dig != cur_dig:
        return False, (f"source/params changed since the receipt "
                       f"(recorded {rec_dig[:16]}, current {str(cur_dig)[:16]})"), rcpt

    # ---- 3) 产物仍在且未被替换 ----
    out = rcpt.get("output")
    if not out or not os.path.exists(str(out)):
        return False, "receipt output is gone", rcpt
    rec_size = rcpt.get("output_size")
    if rec_size:
        try:
            if os.path.getsize(str(out)) != int(rec_size):
                return False, "receipt output size changed", rcpt
        except (OSError, TypeError, ValueError):
            return False, "receipt output size unreadable", rcpt

    return True, "", rcpt


def find_and_verify(temp_dir: str, job_id: str, stem: str,
                    params: Dict[str, Any],
                    expected_attempt: str = "",
                    src_path: str = ""
                    ) -> Tuple[Optional[Dict[str, Any]], str, str]:
    """在 ``temp_dir`` 里找候选回执，返回**第一个通过校验**的。

    遍历顺序按路径排序（只为确定性，**不**代表"越新越可信"——
    可信与否完全由 `verify_fused_receipt` 判定）。

    Returns:
        ``(rcpt, path, reason)``。找不到/全不通过时 ``rcpt`` 为 None，
        ``reason`` 汇总每个候选被拒的原因（截断到前 5 个）。
    """
    import glob
    pattern = os.path.join(str(temp_dir),
                           f"{job_id}_{stem}_step4_postprocess*"
                           f".fused_stage34.json")
    hits = sorted(glob.glob(pattern))
    if not hits:
        return None, "", "no fused receipt"
    reasons = []
    for h in hits:
        ok, why, rcpt = verify_fused_receipt(
            h, params, expected_attempt=expected_attempt, src_path=src_path)
        if ok:
            return rcpt, h, ""
        reasons.append(f"{os.path.basename(h)}: {why}")
    return None, "", "; ".join(reasons[:5])
