# -*- coding: utf-8 -*-
"""P5：**显式中间存储**策略（物化边界的唯一决策点）。

## 这一层要回答什么

现在中间产物一律是"无损 FFV1 / zlib TIFF，放在 temp 里，跑完按 keep_steps
清理"—— 这些规则**散落在各个 stage 里**，没有一处能回答：

* 这批活允许把中间产物写成有损的吗？（默认**不允许**）
* 允许的话，用哪个 codec、什么质量、几次代际？
* 这些缓存下一次还能不能复用？
* 跑之前我能不能看到"这次会有哪些有损转换"？

P5 要求的正是这个：把物化边界的**配置与判定**收到一处，并让"有损"必须
**显式授权**。G10 的通过条件是两半：

1. **没有显式许可时**，模拟 OOM/低磁盘/编码失败，不得启用有损或物化；
2. **显式许可后**，codec / 采样 / 位深 / 质量 / 代数必须与**运行预览**一致；
   同计划可复用、改质量失效、损坏缓存拒绝。

## 三条硬规则（写进代码而不是注释）

1. `mode: lossy` 但 `lossy_allowed` 不为真 → **拒绝**（不是"降级成无损"，
   也不是"警告后照做"）。理由：有损一旦发生就不可逆，静默降级会让人以为
   自己拿到的是无损链的结论。
2. 有损计划的缓存**不得**被无损计划复用，反之亦然。判据是产物侧车里记的
   `storage` 块（mode / lossy_generations / codec / pix_fmt），不是文件名。
3. `max_lossy_generations` 是**硬上限**：链路里连续有损转换的代数超过它就
   拒绝。默认 0（无损）。

## 边界（不要越界声称）

本模块只落地"策略解析 + 守卫 + 预览 + 侧车身份"。**没有**把任何 stage 的
写出路径改成有损 —— 那需要逐 stage 的 codec 接线与独立验收。
所以现在可以说"无许可不得启用有损"，**不能**说"有损备选已实现"。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

#: 已知的无损载体（中间产物的默认值）。
LOSSLESS_CODECS = {
    "video": frozenset({"ffv1", "huffyuv", "utvideo", "rawvideo"}),
    "image": frozenset({"tiff-zlib", "png", "tiff-deflate"}),
}

#: 已知的有损载体（**只登记，不代表已接线**）。
LOSSY_CODECS = {
    "video": frozenset({"libx264", "libx265", "libsvtav1", "mjpeg"}),
    "image": frozenset({"jpeg", "avif"}),
}

MODES = frozenset({"lossless", "lossy"})

#: 策略默认值 —— **与改动前的行为完全一致**（G10：旧配置仍走旧行为）。
DEFAULTS: Dict[str, Any] = {
    "mode": "lossless",
    "lossy_allowed": False,
    "codec": "",                 # 空 = 跟随 stage 自己的默认（FFV1 / TIFF-zlib）
    "quality": None,             # 有损才需要（crf / q）
    "pix_fmt": "",               # 空 = 跟随 work_format 的边界
    "max_lossy_generations": 0,
    "cleanup": {"keep_steps": 2, "purge_stale": True},
    "reuse_identity": ["plan", "source", "params", "impl"],
}

#: 允许出现在 `pix_fmt` 里的写法（宽松校验：拒绝明显不合法的字符串就行，
#: 真正的合法性由 media_io / work_format 在各自边界上判定）。
_PIX_FMT_HINTS = ("yuv420p", "yuv422p", "yuv444p", "rgb24", "rgb48",
                  "gray", "gbrp")


def _as_bool(value, default: bool = False) -> bool:
    """严格布尔：字符串 "false" 不能被当成真（R19-B06 的老毛病）。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return default


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve(params: Any = None, config: Any = None
            ) -> Tuple[Dict[str, Any], List[str]]:
    """解析出**本次运行**的存储策略，并给出拒绝理由。

    Args:
        params: job/manifest 侧覆盖（优先级最高）。
        config: 全局配置的 `intermediate_storage` 段。

    Returns:
        ``(policy, problems)``。``problems`` 非空表示**不能开跑** ——
        调用方必须拒绝并解释，不能自己挑一条路继续。
    """
    src: Dict[str, Any] = {}
    if isinstance(config, dict):
        c = config.get("intermediate_storage")
        if isinstance(c, dict):
            src.update(c)
    if isinstance(params, dict):
        p = params.get("intermediate_storage")
        if isinstance(p, dict):
            src.update(p)

    policy: Dict[str, Any] = {
        "mode": str(src.get("mode", DEFAULTS["mode"])).strip().lower(),
        "lossy_allowed": _as_bool(src.get("lossy_allowed"),
                                  DEFAULTS["lossy_allowed"]),
        "codec": str(src.get("codec", DEFAULTS["codec"]) or "").strip(),
        "quality": src.get("quality", DEFAULTS["quality"]),
        "pix_fmt": str(src.get("pix_fmt", DEFAULTS["pix_fmt"]) or "").strip(),
        "max_lossy_generations": _as_int(
            src.get("max_lossy_generations"),
            DEFAULTS["max_lossy_generations"]),
        "cleanup": dict(DEFAULTS["cleanup"]),
        "reuse_identity": list(DEFAULTS["reuse_identity"]),
    }
    _cu = src.get("cleanup")
    if isinstance(_cu, dict):
        policy["cleanup"].update(
            {k: _cu[k] for k in ("keep_steps", "purge_stale") if k in _cu})
    policy["cleanup"]["keep_steps"] = _as_int(
        policy["cleanup"].get("keep_steps"), 2)
    policy["cleanup"]["purge_stale"] = _as_bool(
        policy["cleanup"].get("purge_stale"), True)
    _ri = src.get("reuse_identity")
    if isinstance(_ri, (list, tuple)) and _ri:
        policy["reuse_identity"] = [str(x) for x in _ri]

    problems: List[str] = []
    if policy["mode"] not in MODES:
        problems.append(
            f"intermediate_storage.mode={policy['mode']!r} is not one of "
            f"{sorted(MODES)}")
    # ---- 规则 1：有损必须**显式**授权 ----
    if policy["mode"] == "lossy" and not policy["lossy_allowed"]:
        problems.append(
            "intermediate_storage.mode=lossy requires lossy_allowed: true. "
            "Lossy intermediates are irreversible and must never be enabled "
            "implicitly (by an OOM, a low-disk condition, or a fallback).")
    # ---- 规则 3：代数上限 ----
    if policy["mode"] == "lossless":
        if policy["max_lossy_generations"] != 0:
            problems.append(
                "max_lossy_generations must be 0 in lossless mode "
                f"(got {policy['max_lossy_generations']})")
        if policy["codec"] and policy["codec"].lower() not in \
                (LOSSLESS_CODECS["video"] | LOSSLESS_CODECS["image"]):
            problems.append(
                f"codec {policy['codec']!r} is not a known lossless carrier; "
                "refusing rather than silently switching to a lossy encoder")
    else:
        if not policy["codec"]:
            problems.append("lossy mode requires an explicit codec")
        elif policy["codec"].lower() not in \
                (LOSSY_CODECS["video"] | LOSSY_CODECS["image"]):
            problems.append(
                f"codec {policy['codec']!r} is not a known lossy carrier")
        if policy["max_lossy_generations"] < 1:
            problems.append(
                "lossy mode requires max_lossy_generations >= 1")
        if policy["quality"] in (None, ""):
            problems.append(
                "lossy mode requires an explicit quality (crf/q); "
                "an encoder default is not a decision this plan can verify")
    if policy["pix_fmt"] and not any(
            h in policy["pix_fmt"] for h in _PIX_FMT_HINTS):
        problems.append(
            f"intermediate_storage.pix_fmt={policy['pix_fmt']!r} does not look "
            "like a pixel format this project can carry")
    return policy, problems


def lossy_generations(policy: Dict[str, Any]) -> int:
    """本次计划允许/期待的**有损代数**（无损恒为 0）。"""
    if str((policy or {}).get("mode")) != "lossy":
        return 0
    return max(0, _as_int((policy or {}).get("max_lossy_generations"), 1))


def preview_lines(policy: Dict[str, Any]) -> List[str]:
    """运行预览：**每一次有损转换都要列出来**。

    G10 原文：「选定有损必须在运行预览中列明每次转换」。所以这里不是
    打印一句"使用有损中间件"，而是逐边界列出 —— 让人看得见"哪一步会掉
    精度、掉多少、掉几次"。
    """
    policy = policy or {}
    mode = str(policy.get("mode") or "lossless")
    if mode != "lossy":
        return ["intermediate storage: lossless (FFV1 / TIFF-zlib; unchanged)"]
    codec = policy.get("codec") or "<unset>"
    q = policy.get("quality")
    pf = policy.get("pix_fmt") or "follow-work-format"
    n = lossy_generations(policy)
    out = [f"intermediate storage: LOSSY ({codec}, quality={q}, "
           f"pix_fmt={pf}, max_generations={n})"]
    for i in range(1, n + 1):
        out.append(
            f"  lossy conversion #{i}: an intermediate boundary will be "
            f"written with {codec} at quality {q} — pixels are permanently "
            f"altered from this generation on")
    out.append("  consequence: acceptance results obtained on a lossless "
               "chain CANNOT be reused for this run (and vice versa)")
    return out


def storage_block(policy: Dict[str, Any]) -> Dict[str, Any]:
    """写进中间产物身份侧车的 `storage` 块。

    它是"这份产物是在什么存储策略下生成的"的唯一凭据；复用判定靠它，
    **不靠文件名**（文件名会被后续拼接/覆盖，也不带质量与代数）。
    """
    policy = policy or {}
    return {
        "mode": str(policy.get("mode") or "lossless"),
        "codec": str(policy.get("codec") or ""),
        "quality": policy.get("quality"),
        "pix_fmt": str(policy.get("pix_fmt") or ""),
        "lossy_generations": lossy_generations(policy),
    }


def cache_mismatch(recorded: Any, policy: Dict[str, Any]) -> str:
    """这份缓存能不能用在**当前**策略下。返回空串表示可以。

    三个维度任一不同都算不可复用：
      * `mode`（有损 vs 无损的验收结论不可互换）；
      * `codec` / `quality`（改质量必须失效）；
      * `lossy_generations`（代数不同意味着累积损失不同）。
    """
    want = storage_block(policy)
    got = recorded if isinstance(recorded, dict) else None
    # `mode: unknown` 是"产物侧车没拿到策略"（例如 stage 被单独调用），
    # 语义等同**没有记录**——不能因此把一份合法的历史产物判成不可复用。
    if isinstance(got, dict) and str(got.get("mode") or "") not in MODES:
        got = None
    if got is None:
        # 旧侧车没有 storage 块 —— 只能是"历史的无损产物"。
        # 当前计划也是无损 → 可以复用（旧行为）；否则拒绝。
        if want["mode"] == "lossless":
            return ""
        return ("artifact carries no storage record (produced before "
                "intermediate storage was tracked); it cannot be reused by a "
                "lossy plan")
    if str(got.get("mode") or "") != want["mode"]:
        return (f"artifact was produced with storage mode "
                f"{got.get('mode')!r}, the current plan is {want['mode']!r}")
    if want["mode"] == "lossy":
        if str(got.get("codec") or "") != want["codec"]:
            return (f"artifact codec {got.get('codec')!r} != plan codec "
                    f"{want['codec']!r}")
        if str(got.get("quality")) != str(want["quality"]):
            return (f"artifact quality {got.get('quality')!r} != plan quality "
                    f"{want['quality']!r} (changing quality must invalidate "
                    "the cache)")
        if _as_int(got.get("lossy_generations"), 0) != want["lossy_generations"]:
            return (f"artifact lossy_generations "
                    f"{got.get('lossy_generations')!r} != plan "
                    f"{want['lossy_generations']}")
    return ""


def describe(policy: Dict[str, Any]) -> str:
    """一行摘要（日志用）。"""
    policy = policy or {}
    return ("mode=%s codec=%s quality=%s pix_fmt=%s max_lossy_generations=%s "
            "keep_steps=%s" % (
                policy.get("mode"),
                policy.get("codec") or "-",
                policy.get("quality"),
                policy.get("pix_fmt") or "-",
                policy.get("max_lossy_generations"),
                (policy.get("cleanup") or {}).get("keep_steps")))
