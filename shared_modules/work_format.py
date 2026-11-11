# -*- coding: utf-8 -*-
"""工作表示（色度采样 / 位深）选择与能力前置核验（R22 §5.2）。

## 它解决什么问题

审核 R22 §5.2：

> 保留默认兼容路径，新增或贯通明确的工作表示选择，区分模型 RGB、处理中间
> 表示、最终输出采样。建议内部语义为 `auto / preserve-or-444 / 420` 等清楚
> 枚举……不要把 auto/on/off 强套给位深枚举；每项处理功能继续保留其 auto/on/off。
>
> 若某插件只接受 420，说明该边界是否可转换、代价和回退，给可行经典路径；
> **不能静默转换后仍称全程 444 保真**。

也就是说：项目原来全链写死 4:2:0，「4:4:4 不可用」被当成设计结论记录，
而它其实是**能力缺口**。本模块把"用哪种工作表示"变成一个显式选择，
并且——这一点最重要——**在动手之前**核验这条链能否真的做到。

## 四个模式

| 模式 | 含义 | 默认 |
|---|---|---|
| `auto` | 走兼容路径（420），但**给出建议**与理由，不改行为 | ✅ |
| `preserve` | 跟随源素材的色度采样（源 444 就 444，源 420 就 420）| |
| `444` | 全链要求 4:4:4；能力不足则**明确失败**（除非用户接受降级）| |
| `420` | 显式走 4:2:0（与历史行为一致）| |

`auto` 之所以不自动切 444：切换会改变全部中间产物的格式与体积
（实测 444 中间件体积约为 420 的 1.5 倍），而"默认不变"是本项目
反复强调的兼容底线。建议写在 `recommendation` 里，由用户决定。

## 三个表示层（审核要求区分）

| 层 | 字段 | 说明 |
|---|---|---|
| 模型 RGB | `model_rgb` | Real-ESRGAN 等模型内部是 RGB；进出边界都是 rgb48le |
| 处理中间表示 | `boundaries["stageN_out"]` | 各 stage 落盘/传递的 YUV 格式 |
| 最终输出采样 | `boundaries["stage7_final"]` | 交给编码器的格式 |

## 能力核验是**实测**的，不是查表

`ffmpeg_encoder_formats()` 直接读 `ffmpeg -h encoder=<name>` 的
"Supported pixel formats"。本机实测（2026-11-12）：

    libsvtav1  → yuv420p yuv420p10le            ← **不支持 444**
    libx265    → … yuv444p10le …
    libaom-av1 → … yuv444p10le …

所以「选 444 但没换编码器」是一个**必须拦下**的组合：libsvtav1 拿到
yuv444p10le 输入会直接失败（或在旧的静默转换写法里悄悄降成 420，
然后被当成"全程 444"）。这正是审核点名的那类问题。

## 参与身份

`param_keys()` 列出必须进计划身份的键。它们都是普通键（不带 `_` 前缀），
因此会自动进入 `_identity_semantics()`；把它们列在这里是为了让
`checkpoint` / `stage_deps` 显式引用，而不是靠"恰好没被过滤"。

切换模式 → 计划摘要变 → 旧产物失效。**不允许**复用旧 420 再换标签。
"""
from __future__ import annotations

import re
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

SCHEMA = "work-format/1"

#: 工作表示模式。`auto` 是默认（与历史行为一致）。
MODES: Tuple[str, ...] = ("auto", "preserve", "444", "420")

#: 只有这几个键影响工作表示；全部是普通键（自动进身份摘要）。
_KEYS = ("chroma_mode", "chroma_mode_accept_degraded")

#: 4:4:4 与 4:2:0 各自的边界格式表。
#: 位深沿用项目现有策略：处理中间件 16-bit，成品前降到 10-bit。
_BOUNDARIES = {
    "444": {
        "stage1_render": "yuv444p16le",
        "stage2_step2": "yuv444p16le",
        "stage3_step3": "yuv444p16le",
        "stage3_fused_in": "yuv444p16le",
        "stage4_mask": "yuv444p",
        "stage4_out": "yuv444p16le",
        "stage5_out": "yuv444p10le",
        "stage6_out": "yuv444p10le",
        "stage7_work": "yuv444p16",
        "stage7_final": "yuv444p10le",
    },
    "420": {
        "stage1_render": "yuv420p16le",
        "stage2_step2": "yuv420p16le",
        "stage3_step3": "yuv420p16le",
        "stage3_fused_in": "yuv420p16le",
        "stage4_mask": "yuv420p",
        "stage4_out": "yuv420p16le",
        "stage5_out": "yuv420p10le",
        "stage6_out": "yuv420p10le",
        "stage7_work": "yuv444p16",   # 抖动在 444 空间做，最后才降采样
        "stage7_final": "yuv420p10le",
    },
}

#: `_CHROMA_RE`：从像素格式名里读色度采样。只认本项目真实出现的写法。
_CHROMA_PATTERNS = (
    (re.compile(r"444"), "444"),
    (re.compile(r"422"), "422"),
    (re.compile(r"420"), "420"),
    (re.compile(r"^gray|^gbrp|^rgb|^bgr|^ya"), "rgb-or-gray"),
)


def chroma_of_pix_fmt(pix_fmt: str) -> str:
    """像素格式 → ``"444"`` / ``"422"`` / ``"420"`` / ``"rgb-or-gray"`` / ``"unknown"``。

    未知就返回 ``"unknown"``，**不要**猜成 420 —— "读不出来"和"是 420"
    是两件事，猜了就等于把一个未知当成已知。
    """
    s = str(pix_fmt or "").strip().lower()
    if not s:
        return "unknown"
    for rx, name in _CHROMA_PATTERNS:
        if rx.search(s):
            return name
    return "unknown"


def _ffmpeg() -> str:
    import shutil
    return shutil.which("ffmpeg") or "ffmpeg"


#: 位深后缀 → 可替换的位深集合（只列本项目用到的）。
_DEPTH_RX = re.compile(r"(yuv(?:420|422|444)p|gbrp|gray|ya8)(8|10|12|16)(le|be)?$",
                       re.IGNORECASE)


def set_bit_depth(pix_fmt: str, depth: int) -> str:
    """改像素格式的位深，别的都不动（``yuv420p16le`` → ``yuv420p10le``）。

    为什么需要它：边界格式表按"处理中间件 16-bit / 成品 10-bit"写死，
    但项目允许 `vs_bit_depth: 10` 让中间件也走 10-bit。用字符串
    ``replace("16le","10le")`` 去做会在 `rgb48le`、`yuv444p16be` 这类
    名字上出岔子 —— 这里用正则精确匹配"平面格式 + 位深 + 可选字节序"。

    8-bit 要去掉数字：ffmpeg 的格式名是 ``yuv444p``，不是 ``yuv444p8``。

    认不出的格式**原样返回**（不猜），调用方自己决定是否报错。
    """
    s = str(pix_fmt or "")
    d = int(depth)
    m = _DEPTH_RX.match(s)
    if not m:
        return s
    if d == 8:
        return f"{m.group(1)}"
    return f"{m.group(1)}{d}{m.group(3) or ''}"


def bit_depth_of(pix_fmt: str) -> Optional[int]:
    """从像素格式名读位深；``rgb48le`` = 16，``yuv444p`` = 8，认不出返回 None。"""
    s = str(pix_fmt or "").strip().lower()
    if not s:
        return None
    if s.startswith("rgb48") or s.startswith("rgba64") or s.startswith("bgr48"):
        return 16
    if s.startswith("rgb24") or s.startswith("bgr24"):
        return 8
    m = _DEPTH_RX.match(s)
    if m:
        return int(m.group(2))
    # 8-bit 平面格式没有数字后缀
    if re.match(r"^(yuv(?:420|422|444)p|gray|ya8)$", s):
        return 8
    return None


def ffmpeg_encoder_formats(encoder: str, timeout: int = 20) -> Set[str]:
    """读 ``ffmpeg -h encoder=<name>`` 的 "Supported pixel formats"。

    Returns:
        set[str]；读不到（编码器不存在 / 输出格式变了）返回**空集** ——
        空集表示"没能证明支持"，调用方必须把它当 blocker，而不是当"都能"。
    """
    enc = str(encoder or "").strip()
    if not enc:
        return set()
    try:
        r = subprocess.run([_ffmpeg(), "-hide_banner", "-h", f"encoder={enc}"],
                           capture_output=True, text=True, errors="replace",
                           timeout=timeout, shell=False)
    except Exception:
        return set()
    if r.returncode != 0:
        return set()
    for ln in (r.stdout or "").splitlines():
        if "Supported pixel formats" in ln:
            _, _, rest = ln.partition(":")
            return {t for t in rest.split() if t}
    return set()


_VS_RX = re.compile(r"^yuv(420|422|444)p(8|10|12|16)", re.IGNORECASE)


def vs_format_of(pix_fmt: str) -> str:
    """ffmpeg 像素格式 → VapourSynth 格式名（``yuv444p16le`` → ``vs.YUV444P16``）。

    认不出返回空串 —— 调用方据此决定"保持原样"还是"报错"，
    不要在这里猜一个默认值。
    """
    m = _VS_RX.match(str(pix_fmt or ""))
    if not m:
        return ""
    return f"vs.YUV{m.group(1)}P{m.group(2)}"


def _norm_mode(value) -> str:
    s = str(value or "").strip().lower()
    return s if s in MODES else "auto"


def _probe_source_pix_fmt(path: str) -> str:
    if not path:
        return ""
    import shutil
    probe = shutil.which("ffprobe") or "ffprobe"
    try:
        r = subprocess.run(
            [probe, "-hide_banner", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=pix_fmt",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, errors="replace",
            timeout=20, shell=False)
    except Exception:
        return ""
    return (r.stdout or "").strip() if r.returncode == 0 else ""


def resolve(params: Dict[str, Any], *, source_path: str = "",
            source_pix_fmt: str = "", logger=None) -> Dict[str, Any]:
    """把模式解析成**一张边界格式表 + 一份能力结论**。

    Args:
        params: 任务参数（读 `chroma_mode` / `chroma_mode_accept_degraded`
            / `video_encoder` / `encoder` / `vs_bit_depth`）。
        source_path: 源素材路径；给了就现场探测它的 pix_fmt。
        source_pix_fmt: 已知源 pix_fmt（优先于探测）。
        logger: 可选日志器。

    Returns:
        计划 dict（见模块 docstring）。``ok=False`` 时调用方必须**明确失败**，
        除非 ``accepted_degraded`` 为真（用户显式接受降级）。
    """
    p = params or {}
    mode = _norm_mode(p.get("chroma_mode"))
    accept = bool(p.get("chroma_mode_accept_degraded"))
    src_fmt = str(source_pix_fmt or "").strip()
    if not src_fmt and source_path:
        src_fmt = _probe_source_pix_fmt(source_path)
    src_chroma = chroma_of_pix_fmt(src_fmt)

    blockers: List[Dict[str, Any]] = []
    warnings: List[str] = []
    recommendation, reason = "420", ""
    resolved = "420"
    declared = str(p.get("pix_fmt") or "").strip()
    declared_chroma = chroma_of_pix_fmt(declared)

    if mode == "420":
        resolved, reason = "420", "explicitly requested 4:2:0"
    elif mode == "444":
        resolved, reason = "444", "explicitly requested 4:4:4"
    elif mode == "preserve":
        if src_chroma == "444":
            resolved = "444"
            reason = "preserve: the source is 4:4:4"
        elif src_chroma == "420":
            resolved = "420"
            reason = "preserve: the source is 4:2:0"
        else:
            resolved = "420"
            reason = ("preserve was requested but the source chroma could not "
                      "be read; falling back to 4:2:0 is NOT equivalent")
            blockers.append({
                "boundary": "source",
                "kind": "source_chroma_unknown",
                "detail": (f"source pix_fmt={src_fmt!r} → chroma="
                           f"{src_chroma!r}; 'preserve' cannot be honored"),
                "cost": "a silent 4:2:0 fallback would destroy real chroma "
                        "detail, so this must fail instead",
            })
    else:  # auto
        if declared_chroma == "444":
            # 用户已经把成品格式写成 444。**不能**只在最后一步换标签 ——
            # 那正是审核说的"静默转换后仍称 444 保真"。整链跟着走。
            resolved = "444"
            reason = (f"auto: pix_fmt={declared!r} asks for 4:4:4, so the whole "
                      f"chain follows it (relabeling only the last stage would "
                      f"not bring chroma detail back)")
        elif src_chroma == "444":
            recommendation = "444"
            reason = ("auto: the source carries 4:4:4 chroma, so preserving it "
                      "would keep chroma detail; the default stays 4:2:0 for "
                      "compatibility — set chroma_mode: preserve to keep it")
        elif src_chroma == "unknown":
            reason = ("auto: source chroma unknown; staying on the compatible "
                      "4:2:0 path")
        else:
            reason = (f"auto: source chroma is {src_chroma}; 4:2:0 loses nothing "
                      f"that the source actually has")

    # 显式模式与 `pix_fmt` 冲突 → 必须报出来，不能悄悄改一边
    if mode != "auto" and declared_chroma in ("420", "444") \
            and declared_chroma != resolved:
        blockers.append({
            "boundary": "stage7_final",
            "kind": "pix_fmt_conflict",
            "detail": (f"chroma_mode={mode} resolves to {resolved}, but "
                       f"pix_fmt={declared!r} is {declared_chroma}"),
            "cost": ("one of the two would have to be ignored; silently "
                     "ignoring either produces a run whose label and content "
                     "disagree"),
        })

    boundaries = dict(_BOUNDARIES[resolved])

    # ---- 能力前置核验：编码器必须真的接受这个格式 ----
    enc = str(p.get("video_encoder") or p.get("encoder") or "").strip()
    final_fmt = boundaries["stage7_final"]
    if resolved == "444":
        fmts = ffmpeg_encoder_formats(enc) if enc else set()
        if not enc:
            blockers.append({
                "boundary": "stage7_final",
                "kind": "encoder_unknown",
                "detail": "no encoder configured; cannot verify 4:4:4 support",
                "cost": "the encoder may silently pick another format",
            })
        elif final_fmt not in fmts:
            blockers.append({
                "boundary": "stage7_final",
                "kind": "encoder_unsupported",
                "detail": (f"{enc} does not list {final_fmt} in its supported "
                           f"pixel formats ({', '.join(sorted(fmts)) or 'none read'})"),
                "cost": ("the encoder would either fail or silently subsample "
                         "to 4:2:0 while the run still claims 4:4:4"),
                "suggest": [e for e in ("libx265", "libaom-av1")
                            if final_fmt in ffmpeg_encoder_formats(e)],
            })
    # 未声明位深时只提示，不拦（项目默认 16 处理 / 10 成品）
    if str(p.get("chroma_mode") or "") and mode != "auto" and not src_fmt:
        warnings.append("source pix_fmt was not probed; the resolution cannot "
                        "cite what the source actually carried")

    ok = not blockers or accept
    plan = {
        "schema": SCHEMA,
        "mode": mode,
        "resolved": resolved,
        "reason": reason,
        "recommendation": recommendation,
        "accepted_degraded": accept,
        "source": {"pix_fmt": src_fmt, "chroma": src_chroma,
                   "probed": bool(src_fmt)},
        "model_rgb": {"pix_fmt": "rgb48le", "bit_depth": 16,
                      "note": "models take RGB; this layer is not subsampled"},
        "boundaries": boundaries,
        "blockers": blockers,
        "warnings": warnings,
        "encoder": enc,
        "ok": ok,
    }
    if logger:
        if blockers and not accept:
            logger.error("WORK_FORMAT_UNSUPPORTED",
                         "The requested working representation cannot be "
                         "honored by this chain", mode=mode,
                         resolved=resolved, blockers=blockers)
        elif blockers and accept:
            logger.warning("WORK_FORMAT_DEGRADED",
                           "Proceeding after the user accepted a degraded "
                           "working representation", mode=mode,
                           resolved=resolved, blockers=blockers)
        else:
            logger.info("WORK_FORMAT", "Working representation resolved",
                        mode=mode, resolved=resolved, reason=reason)
    return plan


def plan_for(params: Dict[str, Any]) -> Dict[str, Any]:
    """从 params 里取**已冻结**的工作表示计划（没有就现算）。

    为什么优先取冻结值：stage 运行期会往 params 里注入内部键；
    如果每个 stage 各自现算，同一份计划可能出现两种结论。
    main 在计划阶段把结果放进 ``params["_work_format"]`` 后即冻结。
    """
    cached = (params or {}).get("_work_format")
    if isinstance(cached, dict) and cached.get("schema") == SCHEMA:
        return cached
    return resolve(params or {})


def boundary_pix_fmt(params: Dict[str, Any], name: str, *,
                     default: str = "") -> str:
    """取某个边界的像素格式；未知边界名返回 ``default``。

    这是各 stage 的唯一取值入口 —— 它们不该自己拼 ``yuv420p...`` 字符串，
    否则"选了 444 但某个 stage 还写死 420"这种分叉会再次发生。
    """
    plan = plan_for(params)
    b = plan.get("boundaries") or {}
    return str(b.get(name) or default)


def param_keys() -> Tuple[str, ...]:
    """必须进计划身份的键（切换表示 → 旧产物失效）。"""
    return _KEYS


def describe(plan: Dict[str, Any]) -> str:
    """一行摘要（日志/报告用）。"""
    return (f"chroma_mode={plan.get('mode')} → {plan.get('resolved')} "
            f"({plan.get('reason')})")
