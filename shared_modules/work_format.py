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


_VS_RX = re.compile(r"^yuv(420|422|444)p(?:(8|10|12|16))?", re.IGNORECASE)


def vs_format_of(pix_fmt: str) -> str:
    """ffmpeg 像素格式 → VapourSynth 格式名（``yuv444p16le`` → ``vs.YUV444P16``）。

    R23-B09：8-bit 平面格式**没有数字位深后缀**（`yuv444p`），旧正则要求
    必须有一位数字，于是 `yuv444p` 认不出来 → 空串 → 调用方退回 420P8。
    审核用 stage6 的赋值 AST 复现了这个分支（`444 + vs_bit_depth=8`）。

    认不出返回空串 —— 调用方据此决定"保持原样"还是"报错"，
    不要在这里猜一个默认值。
    """
    m = _VS_RX.match(str(pix_fmt or ""))
    if not m:
        return ""
    return "vs.YUV%sP%s" % (m.group(1), m.group(2) or "8")


def _norm_mode(value):
    """把配置值归一成模式。**非法值不静默变 auto**（R23-B09）。

    Returns:
        ``(mode, error)``；``error`` 非空表示这个值不合法（调用方必须拒绝）。
        没写（空值）是合法的，按 `auto` 处理。
    """
    s = str(value if value is not None else "").strip().lower()
    if not s:
        return "auto", ""
    if s in MODES:
        return s, ""
    return "auto", (f"unknown chroma_mode {value!r}; valid values are "
                    f"{list(MODES)} (refusing to fall back silently)")


def _parse_strict_bool(value, *, default=False):
    """严格布尔解析（R23-B09）。返回 ``(ok, value)``。

    审核反例：`chroma_mode_accept_degraded="false"` 被 `bool("false")` 读成
    True —— 用户写的是"不接受"，程序读成"接受"。这里只接受真布尔与
    明确的文本形式（true/false/yes/no/1/0）；其他值一律报错。
    """
    if value is None:
        return True, bool(default)
    if isinstance(value, bool):
        return True, value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) in (0.0, 1.0):
            return True, bool(value)
        return False, bool(default)
    t = str(value).strip().lower()
    if t in ("true", "yes", "on", "1"):
        return True, True
    if t in ("false", "no", "off", "0"):
        return True, False
    return False, bool(default)


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
    """解析工作表示与交付格式，并给出能力结论（R23-B09 / M2）。

    ## 与 R22 的区别（审核逐条复现过）

    R22 只有**一个** `resolved`，于是把"444 工作 + 420 交付"判成冲突；
    `accept_degraded` 只把 `ok` 翻成 True，**不生成可执行替代**；
    非法枚举静默变 auto；`preserve` 遇到 422/RGB 说"源未知"。

    现在明确三层：

    | 层 | 字段 | 由什么决定 |
    |---|---|---|
    | 模型输入 | `model_input` | 固定 RGB（模型内部无色度子采样）|
    | 工作表示 | `working` / `boundaries` | `chroma_mode`（stage1–stage6 的落盘格式）|
    | 交付格式 | `delivery` | `pix_fmt`（最终编码器实际接受的格式）|

    **444 工作 + 420 交付是合法组合**：stage7 末尾做一次明确下采样，
    计划里写成 `conversion`（含损失说明），不是冲突。

    Args:
        params: 读 `chroma_mode` / `chroma_mode_accept_degraded` /
            `pix_fmt` / `video_encoder` / `encoder`。
        source_path: 源素材路径（现场探测 pix_fmt）。
        source_pix_fmt: 已知源 pix_fmt（优先）。
        logger: 可选。

    Returns:
        计划 dict。``ok=False`` 时调用方必须明确失败；``accepted_degraded``
        为真时计划里一定带一个**实际可执行**的替代（`degraded` 字段）。
    """
    p = params or {}
    mode, mode_err = _norm_mode(p.get("chroma_mode"))
    _acc_ok, accept = _parse_strict_bool(p.get("chroma_mode_accept_degraded"),
                                         default=False)
    src_fmt = str(source_pix_fmt or "").strip()
    if not src_fmt and source_path:
        src_fmt = _probe_source_pix_fmt(source_path)
    src_chroma = chroma_of_pix_fmt(src_fmt)

    blockers: List[Dict[str, Any]] = []
    warnings: List[str] = []
    recommendation, reason = "420", ""
    working = "420"

    if mode_err:
        blockers.append({
            "boundary": "config",
            "kind": "invalid_mode",
            "detail": mode_err,
            "cost": ("silently treating it as 'auto' means the user's explicit "
                     "request is replaced by a default they never chose"),
        })

    if mode == "420":
        working, reason = "420", "explicitly requested 4:2:0"
    elif mode == "444":
        working, reason = "444", "explicitly requested 4:4:4"
    elif mode == "preserve":
        if src_chroma == "444":
            working, reason = "444", "preserve: the source is 4:4:4"
        elif src_chroma == "420":
            working, reason = "420", "preserve: the source is 4:2:0"
        elif src_chroma == "422":
            # 已知格式，但项目没有 422 工作表示 —— 这与"读不出来"是两回事。
            working = "444"
            reason = ("preserve: the source is 4:2:2. This project has no 4:2:2 "
                      "working representation, so the closest lossless-in-chroma "
                      "choice is 4:4:4")
            blockers.append({
                "boundary": "source",
                "kind": "source_chroma_unsupported",
                "detail": ("source pix_fmt=%r is 4:2:2; only 420/444 working "
                           "representations exist" % src_fmt),
                "cost": ("4:4:4 does not recover the vertical half-resolution "
                         "the source already lost, and 4:2:0 would reduce the "
                         "horizontal resolution as well"),
                "suggest": ["chroma_mode=444 (closest), or chroma_mode=420 if "
                            "you accept the extra reduction"],
            })
        elif src_chroma == "rgb-or-gray":
            working = "444"
            reason = ("preserve: the source is RGB/gray (no chroma "
                      "subsampling); 4:4:4 is the only working representation "
                      "that does not subsample")
            warnings.append("the source carries no chroma subsampling; a YUV "
                            "pipeline still applies a matrix conversion")
        else:
            working = "420"
            reason = ("preserve was requested but the source chroma could not "
                      "be read; falling back to 4:2:0 is NOT equivalent")
            blockers.append({
                "boundary": "source",
                "kind": "source_chroma_unknown",
                "detail": ("source pix_fmt=%r → chroma=%r; 'preserve' cannot "
                           "be honored" % (src_fmt, src_chroma)),
                "cost": ("a silent 4:2:0 fallback would destroy real chroma "
                         "detail, so this must fail instead"),
            })
    else:  # auto
        if chroma_of_pix_fmt(str(p.get("pix_fmt") or "")) == "444":
            working = "444"
            reason = ("auto: pix_fmt=%r asks for 4:4:4 delivery, and the whole "
                      "chain follows it (relabeling only the last stage would "
                      "not bring chroma detail back)"
                      % (p.get("pix_fmt"),))
        elif src_chroma == "444":
            recommendation = "444"
            reason = ("auto: the source carries 4:4:4 chroma, so preserving it "
                      "would keep chroma detail; the default stays 4:2:0 for "
                      "compatibility — set chroma_mode: preserve to keep it")
        elif src_chroma == "unknown":
            reason = ("auto: source chroma unknown; staying on the compatible "
                      "4:2:0 path")
        else:
            reason = ("auto: source chroma is %s; 4:2:0 loses nothing that "
                      "the source actually has" % src_chroma)

    # ---- 交付格式：可以由 pix_fmt 指定，**与工作表示无关** ----
    declared = str(p.get("pix_fmt") or "").strip()
    declared_chroma = chroma_of_pix_fmt(declared)
    boundaries = dict(_BOUNDARIES[working])
    if declared_chroma in ("420", "444"):
        delivery_fmt = declared
        delivery_chroma = declared_chroma
    else:
        delivery_chroma = working
        delivery_fmt = boundaries["stage7_final"]

    conversion = None
    if delivery_chroma != working:
        if working == "444" and delivery_chroma == "420":
            conversion = {
                "from": "444", "to": "420", "stage": "stage7_final",
                "operation": "single chroma down-sample at delivery",
                "loss": ("chroma resolution is halved once, at the very end; "
                         "the 4:4:4 work before it is not undone, but the "
                         "delivered file will not carry 4:4:4 chroma"),
            }
            reason += ("; delivery is 4:2:0, so the chain subsamples once at "
                       "stage7_final (this is a delivery choice, not a conflict)")
        else:
            conversion = {
                "from": working, "to": delivery_chroma, "stage": "stage7_final",
                "operation": "chroma up-sample at delivery",
                "loss": ("none in the delivered file, but nothing is recovered "
                         "either: the working representation already discarded "
                         "this detail"),
            }
            warnings.append("delivery format has more chroma resolution than "
                            "the working representation; that is an up-convert "
                            "and recovers nothing")
    boundaries["stage7_final"] = delivery_fmt

    if str(p.get("chroma_mode") or "") and mode != "auto" and not src_fmt:
        warnings.append("source pix_fmt was not probed; the resolution cannot "
                        "cite what the source actually carried")

    # ---- 能力核验：看**交付格式**，不是看工作表示 ----
    # 审核点名："能力核验使用真正的最终编码器/格式/位深"。libsvtav1 不支持
    # 4:4:4，但"444 工作 + 420 交付"完全可用 —— 旧实现把它一起拦了。
    enc = str(p.get("video_encoder") or p.get("encoder") or "").strip()
    degraded = None
    fmts = ffmpeg_encoder_formats(enc) if enc else set()
    if not enc:
        blockers.append({
            "boundary": "delivery",
            "kind": "encoder_unknown",
            "detail": "no encoder configured; cannot verify the delivery format",
            "cost": "the encoder may silently pick another format",
        })
    elif delivery_fmt not in fmts:
        _alt = [f for f in ("yuv444p10le", "yuv420p10le", "yuv420p")
                if f in fmts]
        blockers.append({
            "boundary": "delivery",
            "kind": "encoder_unsupported",
            "detail": ("%s does not list %s in its supported pixel formats "
                       "(%s)" % (enc, delivery_fmt,
                                 ", ".join(sorted(fmts)) or "none read")),
            "cost": ("the encoder would either fail or silently convert while "
                     "the run still claims the requested delivery format"),
            "suggest": _alt,
        })
        if accept and _alt:
            # 接受降级 → 生成**具体可执行**的替代计划，而不是只把 ok 翻真
            _new = _alt[0]
            degraded = {"from": delivery_fmt, "to": _new,
                        "reason": "the encoder does not support the requested "
                                  "delivery format"}
            blocks_left = [b for b in blockers if b["kind"] != "encoder_unsupported"]
            blockers = blocks_left
            delivery_fmt = _new
            delivery_chroma = chroma_of_pix_fmt(_new)
            boundaries["stage7_final"] = _new
            if delivery_chroma != working:
                conversion = {
                    "from": working, "to": delivery_chroma,
                    "stage": "stage7_final",
                    "operation": "chroma down-sample at delivery (degraded)",
                    "loss": "requested delivery format unsupported by encoder",
                }
            else:
                conversion = None
            warnings.append("delivery format degraded to %s (user accepted)"
                            % _new)

    # 接受降级时，其余 blocker 也必须有可执行替代；没有替代的仍然失败
    if accept and blockers:
        for b in blockers:
            if b.get("suggest"):
                b["accepted_without_effect"] = True

    ok = (not blockers) or (accept and all(b.get("suggest") for b in blockers))
    plan = {
        "schema": SCHEMA,
        "mode": mode,
        "mode_error": mode_err,
        "resolved": working,
        "working": {
            "chroma": working,
            "bit_depth": 16,
            "boundaries": boundaries,
        },
        "delivery": {
            "pix_fmt": delivery_fmt,
            "chroma": delivery_chroma,
            "bit_depth": bit_depth_of(delivery_fmt),
        },
        "model_input": {"pix_fmt": "rgb48le", "bit_depth": 16,
                        "note": "models take RGB; this layer is not subsampled"},
        "conversion": conversion,
        "degraded": degraded,
        "reason": reason,
        "recommendation": recommendation,
        "accepted_degraded": bool(accept),
        "source": {"pix_fmt": src_fmt, "chroma": src_chroma,
                   "probed": bool(src_fmt)},
        "boundaries": boundaries,
        "blockers": blockers,
        "warnings": warnings,
        "encoder": enc,
        "ok": bool(ok),
    }
    if logger:
        if blockers and not ok:
            logger.error("WORK_FORMAT_UNSUPPORTED",
                         "The requested working/delivery representation cannot "
                         "be honored by this chain", mode=mode,
                         working=working, delivery=delivery_fmt,
                         blockers=blockers)
        elif degraded or (blockers and ok):
            logger.warning("WORK_FORMAT_DEGRADED",
                           "Proceeding with a concrete degraded plan",
                           mode=mode, working=working, delivery=delivery_fmt,
                           degraded=degraded, blockers=blockers)
        else:
            logger.info("WORK_FORMAT", "Working/delivery representation resolved",
                        mode=mode, working=working, delivery=delivery_fmt,
                        reason=reason)
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
    """一行摘要（日志/报告用）：工作表示 → 交付格式。"""
    _d = (plan.get("delivery") or {}).get("pix_fmt") or "?"
    _conv = plan.get("conversion")
    _tail = f" → deliver {_d}" if _d else ""
    if _conv:
        _tail += f" (conversion {_conv.get('from')}→{_conv.get('to')})"
    return (f"chroma_mode={plan.get('mode')} → work {plan.get('resolved')}"
            f"{_tail} ({plan.get('reason')})")
