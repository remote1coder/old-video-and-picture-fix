# -*- coding: utf-8 -*-
"""必需检查集合：**由有效计划派生**，不由报告自报（B02 / R25）。

## 为什么需要这个模块

R24-A01 的核心问题：`_verify_report_problems` 只看"报告里有没有这个名字"，
合格与否却听报告自己的 `required` 字段。审核用它打出 8 个假通过：

| 反例 | 现象 |
|---|---|
| C01 | verify 填未知 schema → 完成验证 True |
| C02 | verify 删掉 attempt → True |
| C03 | `run_identity={}` → True |
| C04 | `final_timeline=FAIL` 但报告写 `required=False` → True |
| C05 | `mechanical=ERROR` 但 `required=False` → True |
| C06 | 计划要 fullchain，报告自称 `NOT_APPLICABLE` → True |
| C07 | fullchain 写外来 attempt → True |
| C08 | fullchain 缺 checks → True |
| C09 | fullchain 未知 schema → True |

一句话：**报告能给自己降标准**。这比"少了一个 if"严重得多 —— 它让
"完成"这件事失去意义。

## 本模块的分工

```
计划（planned + params）
   ↓  derive()
RequiredChecks（本次发布必须满足什么、允许哪些状态）
   ↓  evaluate(report, required)
问题列表（报告是否满足计划，而不是报告自己说了算）
```

* **生产者**（stage7）按同一份计划**执行**检查；
* **消费者**（main）按同一份计划**核验**报告；
* 报告里的 `required` 字段**不再参与验收**（只在审计时看，
  与计划不符时单独报"报告自称的必需性与计划不符"，但判定以计划为准）。

## 三类完成状态

审核 A02 要求区分，且**都要传播到普通结束/resume/manifest/日志**：

- ``VERIFIED``：全部必需检查满足、整代正式发布、可按身份复用。
- ``UNVERIFIED``：检查事实仍 FAIL/UNVERIFIED，有明确授权 → 独立诊断状态，
  **不覆盖旧正式代，不计入普通 DONE**。
- ``FAILED``：失败/取消/待恢复，不作为已完成产物复用。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA = "required-checks/1"

#: 完成结论（三态）。
VERIFIED = "verified"
UNVERIFIED = "unverified"
FAILED = "failed"

#: 检查结论的合法取值（认不出的状态一律按问题处理）。
KNOWN_STATUSES = frozenset({
    "PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE", "SKIP", "ERROR", "WARN",
})

#: 算"合格"的状态。
_PASSING = frozenset({"PASS", "NOT_APPLICABLE"})


def _has_audio(params: Optional[Dict[str, Any]]) -> bool:
    """计划里是否声明了音轨处理。"""
    p = params or {}
    for k in ("audio_streams", "has_audio", "multichannel_preserve",
              "audio_passthrough", "audio_sync_tolerance_ms"):
        if p.get(k):
            return True
    return False


def _wants_transform(params: Optional[Dict[str, Any]]) -> bool:
    """计划里是否请求了时间变换（speed/atempo/tempo）。"""
    p = params or {}
    for k in ("target_speed", "speed", "atempo", "tempo",
              "expected_speed", "playback_speed"):
        v = p.get(k)
        if v in (None, "", 0, 0.0, "1x", "1.0"):
            continue
        return True
    return False


def derive(planned: Optional[Dict[str, Any]] = None,
           params: Optional[Dict[str, Any]] = None,
           config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """由**有效计划**推导本次发布的必需检查集合。

    Args:
        planned: `_planned_generation_roles()` 的结果（`required`/`optional`）。
        params: 生效参数（读 `full_chain_verify`、音频/变换相关开关）。
        config: 配置（保留，供后续扩展）。

    Returns:
        ``{schema, checks: {name: {required, allowed, force_key, scope,
        not_applicable_allowed}}, digest}``。

    设计取舍（按审核 §B02）：
    * `mechanical` / `final_timeline` **总是必需** —— 它们是"发布前做过
      机械核对与时间测量"的底线，不能由报告关掉。
    * `frame_order` **不是**无条件必需 —— 实测（N22）它常是 UNVERIFIED
      （采样点不足），生产允许发布。但它的**允许状态**包含 UNVERIFIED，
      所以"报告把它写成 FAIL"仍会被拒。
    * `fullchain` 的必要性来自 `planned.required`；计划要求时
      `NOT_APPLICABLE` **不允许**（C06）。
    """
    pl = planned or {}
    p = params or {}
    required_roles = dict(pl.get("required") or {})
    opt_roles = dict(pl.get("optional") or {})

    checks: Dict[str, Dict[str, Any]] = {}

    # ---- 底线两项：不能由报告关掉 ----
    checks["mechanical"] = {
        "required": True,
        "allowed": ["PASS"],
        "force_key": "allow_verify_fail_publish",
        "scope": "media + planned reports are readable and consistent",
        "not_applicable_allowed": False,
    }
    checks["final_timeline"] = {
        "required": True,
        # 时间测量本身可以在"没有时间轴概念"的产物上不适用（如单张图），
        # 但视频发布必须有结论。
        "allowed": ["PASS", "NOT_APPLICABLE"],
        "force_key": "allow_verify_fail_publish",
        "scope": "final timeline measured after the last mux",
        "not_applicable_allowed": True,
    }

    # ---- 帧序：允许 UNVERIFIED，但 FAIL 不合格 ----
    checks["frame_order"] = {
        "required": False,
        "allowed": ["PASS", "NOT_APPLICABLE", "UNVERIFIED"],
        "force_key": "",
        "scope": "frame order alignment (UNVERIFIED is legal when the "
                 "sample has too few discriminating points)",
        "not_applicable_allowed": True,
    }

    # ---- 全链：按计划 ----
    if "fullchain" in required_roles:
        checks["fullchain"] = {
            "required": True,
            # C06：计划要求全链时，报告不能说"这次不适用"
            "allowed": ["PASS"],
            "force_key": "",
            "scope": "full-chain comparison was required by the plan",
            "not_applicable_allowed": False,
        }
    elif "fullchain" in opt_roles:
        checks["fullchain"] = {
            "required": False,
            "allowed": ["PASS", "NOT_APPLICABLE", "UNVERIFIED"],
            "force_key": "",
            "scope": "full-chain comparison is optional for this plan",
            "not_applicable_allowed": True,
        }
    else:
        checks["fullchain"] = {
            "required": False,
            "allowed": ["PASS", "NOT_APPLICABLE", "UNVERIFIED"],
            "force_key": "",
            "scope": "full-chain comparison disabled by the plan",
            "not_applicable_allowed": True,
        }

    # ---- 颜色：有颜色证据就要合格；不适用可以 ----
    checks["color"] = {
        "required": False,
        "allowed": ["PASS", "NOT_APPLICABLE", "UNVERIFIED"],
        "force_key": "allow_verify_fail_publish",
        "scope": "colour tags relation between source and output",
        "not_applicable_allowed": True,
    }

    # ---- 同步：只有计划里有音轨才要求 ----
    if _has_audio(p):
        checks["sync"] = {
            "required": True,
            "allowed": ["PASS", "NOT_APPLICABLE"],
            "force_key": "allow_sync_fail_publish",
            "scope": "audio/video synchronisation",
            "not_applicable_allowed": True,
        }

    # ---- 变换：只有请求了变换才要求 ----
    if _wants_transform(p):
        checks["transform"] = {
            "required": True,
            "allowed": ["PASS", "UNVERIFIED"],
            "force_key": "allow_verify_fail_publish",
            "scope": "requested time transform was actually applied",
            "not_applicable_allowed": False,
        }

    doc: Dict[str, Any] = {"schema": SCHEMA, "checks": checks}
    doc["digest"] = _digest(doc)
    return doc


def _digest(doc: Dict[str, Any]) -> str:
    body = {k: v for k, v in doc.items() if k != "digest"}
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def evaluate_checks(report_checks: Any,
                    required: Dict[str, Any]) -> List[str]:
    """按计划核验报告里的检查结论。

    Args:
        report_checks: 报告里的 `checks`（形如 ``{name: {status, ...}}``）。
        required: `derive()` 的结果。

    Returns:
        问题列表。空表示"报告满足计划"。

    关键：**报告里的 `required` 字段不参与判定**。若它与计划不一致，
    单独报一条"报告自称的必需性与计划不符"（这是审计信息，
    不是放行理由）。
    """
    probs: List[str] = []
    if not isinstance(report_checks, dict) or not report_checks:
        return ["the report carries no check results at all; it does not "
                "show that anything was checked"]

    spec = dict((required or {}).get("checks") or {})

    seen: Dict[str, str] = {}
    for name, row in report_checks.items():
        if not isinstance(row, dict):
            probs.append(f"check {name!r} is not an object")
            continue
        st = str(row.get("status") or "").upper()
        seen[str(name)] = st
        if st not in KNOWN_STATUSES:
            probs.append(f"check {name!r} has an unknown status {st!r}")

    for name, rule in spec.items():
        st = seen.get(name)
        allowed = {str(x).upper() for x in (rule.get("allowed") or [])}
        if st is None:
            if rule.get("required"):
                probs.append(
                    f"the plan requires the {name!r} check but the report "
                    f"does not contain it")
            continue
        if st == "NOT_APPLICABLE" and not rule.get("not_applicable_allowed"):
            probs.append(
                f"check {name!r} claims NOT_APPLICABLE but the plan does not "
                f"allow it to be inapplicable")
            continue
        if st not in allowed:
            probs.append(
                f"check {name!r} is {st!r}, which the plan does not accept "
                f"(allowed: {sorted(allowed)})")
            continue
        # ---- 报告自称的必需性与计划不符时，只看"放松"方向 ----
        # 报告说 required=True 而计划说 False：报告**更严**，无害，不报。
        # 报告说 required=False 而计划说 True：报告在给自己**降标准**，
        # 这正是 R24-A01 的核心漏洞，必须报出来。
        if rule.get("required") and row.get("required") is False:
            probs.append(
                f"check {name!r} declares required=False but the plan "
                f"requires it; a report may not relax its own acceptance "
                f"(the plan wins)")

    # 报告里出现计划没定义的检查项：不判错（可能新增），但状态已在上文校验
    return probs


def decide(required: Dict[str, Any],
           check_problems: Iterable[str],
           publish_decision: Any,
           forced_keys: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """把"检查是否满足"+"发布决定"折成**三态结论**。

    Args:
        required: `derive()` 的结果。
        check_problems: `evaluate_checks()` 的问题。
        publish_decision: 报告里的 `publish_decision`（dict）。
        forced_keys: 用户实际打开的强制开关名集合。

    Returns:
        ``{verdict, reasons, verified_fact, forced}``。

    * 有任何检查问题 → ``FAILED``（不因 forced 而变成 VERIFIED；
      forced 只允许它成为 **UNVERIFIED 诊断导出**，由上层决定是否落盘）。
    * `publish_decision.verified is False` → ``UNVERIFIED``（A02/C10：
      不能进普通 DONE）。
    * 否则 ``VERIFIED``。
    """
    probs = [str(x) for x in (check_problems or [])]
    dec = publish_decision if isinstance(publish_decision, dict) else {}
    forced = set(str(x) for x in (forced_keys or []))

    if probs:
        return {"verdict": FAILED, "reasons": probs,
                "verified_fact": False, "forced": bool(forced)}
    if not dec:
        return {"verdict": FAILED,
                "reasons": ["the report carries no publish decision; the "
                            "conclusion it reached cannot be checked"],
                "verified_fact": False, "forced": bool(forced)}
    if dec.get("publish") is not True:
        return {"verdict": FAILED,
                "reasons": [f"the report's own publish decision is "
                            f"{dec.get('publish')!r}, not True"],
                "verified_fact": False, "forced": bool(forced)}
    if dec.get("verified") is False:
        # A02：强制导出不是"已验证完成"。
        return {"verdict": UNVERIFIED,
                "reasons": ["the report says the published media is NOT "
                            "verified (forced export)"],
                "verified_fact": False,
                "forced": bool(dec.get("forced")) or bool(forced)}
    return {"verdict": VERIFIED, "reasons": [], "verified_fact": True,
            "forced": bool(dec.get("forced")) or bool(forced)}
