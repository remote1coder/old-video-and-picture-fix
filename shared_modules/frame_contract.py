# -*- coding: utf-8 -*-
"""帧边界合同（R22 §5.1）。

## 为什么需要它

审核要求：把"AI 推理完成 → 经典物化 / 融合后处理"这个分叉点做成**共同生产函数**，
两条模式消费**同一份固定帧源**和**同一份处理计划**。要判断"两路输出是否只差
在融合本身"，先得能证明"两边吃进去的是同一串帧"。

所以需要一个**边界合同**：在分叉点记下"这一串帧到底是什么"。

## 字段（对应审核 §5.1 的"边界合同至少包含"）

| 字段 | 含义 |
|---|---|
| `frames` | 帧数 |
| `width` / `height` | 输出帧尺寸 |
| `pix_fmt` / `bit_depth` / `channel_order` | 像素格式三件套 |
| `frames_digest` | **有序**帧内容身份（帧序算进摘要）|
| `frames_digest_proven` | 摘要是否真的算过；流式管道拿不到时为 False |
| `timeline` | 时间合同引用（TimelinePlan 路径 + 摘要）|
| `color` | 颜色合同（与 `generation_evidence.color_group` 同口径）|
| `ai` | 实现 / 模型 / 工具指纹 |

## 一条硬规则

**`frames_digest` 只有在真算过时才是证据。** 流式 raw 管道里帧不进磁盘、
也不经过本模块，所以那条路径记 `frames_digest_proven=False` 并注明原因 ——
绝不写一个"看起来像摘要"的值。

审核原话："raw管道携带不了的PTS和frame props用同身份的旁证记录，
不能仅给一个FPS就丢掉VFR信息。" 同理，摘要拿不到就写"未证明"，
不拿 FIFO/顺序号之类的替代物冒充内容身份。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA = "frame-boundary/1"

#: 允许的通道顺序（只写实际用到的两种）。
CHANNEL_ORDERS = ("rgb", "bgr")

#: 像素格式 → (位深, 通道顺序)。只列**需要显式指定通道顺序**的格式；
#: 其余按命名规则推导（见 `pix_fmt_info`）。
_PIX_FMT_INFO = {
    "rgb48le": (16, "rgb"),
    "gbrp16le": (16, "rgb"),
    "yuv420p16le": (16, None),
    "yuv420p10le": (10, None),
    "yuv420p": (8, None),
}

#: 位深后缀（`yuv444p16le` / `gbrp10le` / `gray16le` 里的 16/10/16）。
#: R24-06 / R23-A05：格式表原来只有 420 那几项，`yuv444p16le` 落进
#: "未知格式" → `bit_depth=None` → 反而被新规则当成非法。444 现在是
#: 合法的工作表示，必须能识别。
_DEPTH_RX = re.compile(r"(?:p|gray|ya|rgb)(8|9|10|12|14|16)(?:le|be)?$")

#: 没有位深后缀 = 8bit（`yuv420p` / `yuv444p` / `gbrp` / `gray` / `ya8`）。
_PLAIN8_RX = re.compile(r"^(?:yuv\d{3}p|gbrp|gray|ya8)$")


def pix_fmt_info(pix_fmt: str) -> Tuple[Optional[int], Optional[str]]:
    """像素格式 → ``(bit_depth, channel_order)``；未知格式返回 ``(None, None)``。

    未知不等于错误 —— 但要如实写 None，让消费者知道"这一项没被识别"。

    R24-06 / R23-A05：优先查显式表（需要指定通道顺序的那几个），
    其余按**命名规则**推位深。原来只查表，于是 `yuv444p16le` 被判未知、
    `bit_depth=None`，反而被新规则当成非法格式拒掉 —— 444 明明是现在
    合法的工作表示。按规则推也避免了"以后每加一种格式就得改表"。
    """
    s = str(pix_fmt or "").strip().lower()
    if not s:
        return (None, None)
    exact = _PIX_FMT_INFO.get(s)
    if exact is not None:
        return exact
    order = "rgb" if s.startswith(("gbr", "rgb")) else None
    m = _DEPTH_RX.search(s)
    if m:
        return (int(m.group(1)), order)
    if _PLAIN8_RX.match(s):
        return (8, order)
    return (None, None)


def digest_ordered_frames(paths: Iterable[str], chunk: int = 1 << 20
                          ) -> Optional[str]:
    """**有序**帧内容摘要：把每帧的字节按顺序喂进同一个 SHA-256。

    为什么帧序要算进去：同一组帧换个顺序，时长/帧数/尺寸全都一样，
    只有内容顺序不同 —— 那正是"融合与经典错位"最典型的形态。

    Args:
        paths: 按**播放顺序**排好的帧文件路径。

    Returns:
        十六进制摘要；任一帧读不出来时返回 ``None``（**不**跳过、不补零）。
    """
    h = hashlib.sha256()
    n = 0
    for p in paths:
        try:
            with open(p, "rb") as f:
                while True:
                    b = f.read(chunk)
                    if not b:
                        break
                    h.update(b)
        except OSError:
            return None
        n += 1
    if n == 0:
        return None
    # 把帧数也绑进去：否则"两帧 A+B"和"一帧 AB"会撞同一个摘要
    h.update(b"|frames=" + str(n).encode("ascii"))
    return h.hexdigest()


def build_contract(*, frames: int, width: int, height: int, pix_fmt: str,
                   timeline: Optional[Dict[str, Any]] = None,
                   color: Optional[Dict[str, Any]] = None,
                   ai: Optional[Dict[str, Any]] = None,
                   frames_digest: Optional[str] = None,
                   frames_digest_proven: Optional[bool] = None,
                   frames_digest_domain: str = "",
                   pix_fmt_proven: Optional[bool] = None,
                   execution: Optional[Dict[str, Any]] = None,
                   source: str = "") -> Dict[str, Any]:
    """构造帧边界合同。

    Args:
        frames: 帧数。
        width / height: **输出**帧尺寸。
        pix_fmt: 输出像素格式。
        timeline: 时间合同（通常是 TimelinePlan 的路径 + 摘要）。
        color: 颜色合同（`generation_evidence.color_group()` 的产物）。
        ai: 实现/模型/工具指纹（`model` / `backend` / `scale` / `impl`）。
        frames_digest: 有序帧内容身份；没算就传 None。
        frames_digest_proven: 显式标注摘要是否可作证据；None 时按
            `frames_digest` 是否为空推断。
        frames_digest_domain: 摘要是对**哪种字节表示**算的
            （``"raw-rgb48le"`` / ``"png-bytes"`` …）。两份摘要只有在
            域相同时才可比 —— 不同表示算出的摘要本来就该不同。
        source: ``"model"`` / ``"fixed-frames"`` / ``"png-path"`` 等，
            说明这一串帧是怎么来的。

    Returns:
        合同 dict（可直接 json 落盘）。
    """
    bit_depth, order = pix_fmt_info(pix_fmt)
    if frames_digest_proven is None:
        frames_digest_proven = bool(frames_digest)
    if pix_fmt_proven is None:
        pix_fmt_proven = bool(pix_fmt)
    # ---- R24-06：摘要域不能留空 ----
    # 审核要求"摘要域必填"，而 `compare_contracts` 现在把"摘要相同但域
    # 未声明"判为不可信（那是**对的**：两个不同表示上算出同一个摘要文本，
    # 恰恰最危险）。所以生产者必须给出域。
    #
    # 没显式给时按**像素格式**推：同一格式的两份合同可比，不同格式的两份
    # 会被判"域不同、不可比" —— 正是我们要的语义。不写 "unspecified"
    # 之类的占位符，那样等于把"没声明"伪装成"已声明"。
    _domain = str(frames_digest_domain or "").strip()
    if not _domain:
        _domain = ("pix:" + str(pix_fmt)) if str(pix_fmt or "") else ""
    return {
        "schema": SCHEMA,
        "frames": int(frames),
        "width": int(width),
        "height": int(height),
        "pix_fmt": str(pix_fmt),
        "bit_depth": bit_depth,
        "channel_order": order,
        "frames_digest": frames_digest or None,
        "frames_digest_proven": bool(frames_digest_proven),
        "frames_digest_domain": _domain,
        # R23-B08：格式是**探测到的**还是**请求值回填**的，必须分开。
        # 审核：模拟 probe 失败时，生产写入器把请求的尺寸/格式填进去，
        # 虽然写了 gaps，`contract_problems` 仍然返回空 → 合同"看起来完整"。
        "pix_fmt_proven": bool(pix_fmt_proven),
        # 这一串帧是谁产的：模型 / 冻结帧源 / PNG 中转
        "source": str(source),
        # R23-B08：`ai` 只放**模型/实现身份**；"这条链怎么跑的"（经典/融合）
        # 放 `execution`。两者混在一起会让"经典 vs 融合"的合法差异被当成
        # 模型身份差异 —— 反之也会让真正的模型替换躲在执行差异后面。
        "execution": dict(execution or {}),
        "timeline": dict(timeline or {}),
        "color": dict(color or {}),
        "ai": dict(ai or {}),
    }


def contract_problems(contract: Dict[str, Any]) -> List[str]:
    """合同自身是否完整（空列表 = 可以作为边界证据）。

    注意：这里只检查**字段齐不齐**，不检查"两条链路是否一致" ——
    后者是调用方拿两份合同比出来的（见 `compare_contracts`）。
    """
    p: List[str] = []
    if not isinstance(contract, dict):
        return ["contract is not an object"]
    if str(contract.get("schema") or "") != SCHEMA:
        p.append(f"schema is {contract.get('schema')!r}, expected {SCHEMA!r}")
    for k in ("frames", "width", "height"):
        try:
            if int(contract.get(k) or 0) <= 0:
                p.append(f"{k} must be positive, got {contract.get(k)!r}")
        except (TypeError, ValueError):
            p.append(f"{k} is not an integer: {contract.get(k)!r}")
    if not str(contract.get("pix_fmt") or ""):
        p.append("pix_fmt is empty")
    elif contract.get("bit_depth") is None:
        p.append(f"pix_fmt {contract.get('pix_fmt')!r} is not a known format; "
                 f"bit_depth is unknown")
    if contract.get("pix_fmt_proven") is False:
        p.append("the output pixel format was not probed; the recorded value "
                 "is the requested one, not a measured fact")
    if not contract.get("frames_digest_proven"):
        p.append("frames_digest is not proven (this boundary cannot be used "
                 "as evidence that both paths consumed the same frames)")
    elif not str(contract.get("frames_digest") or ""):
        # R23-B08：`proven=True` 但摘要为空 —— 旧实现返回空问题列表。
        p.append("frames_digest is marked proven but no digest is recorded")
    _tl = contract.get("timeline")
    if not isinstance(_tl, dict) or not _tl:
        p.append("no timeline reference (this boundary cannot prove when the "
                 "frames are supposed to be shown)")
    elif not any(str(_tl.get(k) or "") for k in ("digest", "content_digest")):
        # ---- R24-06 / R23-A05：路径不是内容身份 ----
        # 审核反例：只给 `timeline.plan_path`，问题列表为空。
        # 一个路径说明"计划文件在哪"，不说明"这份计划是哪一份" ——
        # 文件被改写、被同名替换都看不出来。要求内容摘要。
        p.append("timeline reference carries no content identity "
                 "(a path, fps or frame count is not a content digest)")
    if not isinstance(contract.get("color"), dict) or not contract.get("color"):
        p.append("no colour contract recorded for this boundary")
    if not isinstance(contract.get("ai"), dict) or not contract.get("ai"):
        p.append("no implementation/model fingerprint recorded")
    for gap in (contract.get("gaps") or []):
        p.append(f"recorded gap: {gap}")
    return p


def compare_contracts(a: Dict[str, Any], b: Dict[str, Any],
                      *, tol_frames: int = 0) -> List[str]:
    """比较两份合同，返回差异列表（空 = 同一串帧、同一套格式）。

    只比**影响像素**的字段：帧数、尺寸、像素格式/位深/通道顺序、帧摘要。
    颜色/时间/AI 指纹单独由调用方处理 —— 它们"可以合理地不同"
    （例如两条链路的时间合同来源不同），不应在这里一律判错。
    """
    p: List[str] = []
    for k in ("frames", "width", "height", "pix_fmt", "bit_depth",
              "channel_order"):
        va, vb = a.get(k), b.get(k)
        if k == "frames":
            try:
                if abs(int(va or 0) - int(vb or 0)) > int(tol_frames):
                    p.append(f"frames differs: {va} vs {vb}")
            except (TypeError, ValueError):
                p.append(f"frames unparsable: {va!r} vs {vb!r}")
            continue
        if va != vb:
            p.append(f"{k} differs: {va!r} vs {vb!r}")
    # ---- R23-B08：声明层（时间/颜色/模型）也要比，且**分开报** ----
    # 审核反例：同摘要、域/时间/颜色/模型都不同，`compare_contracts` 仍为空。
    # 它们不属于"像素差异"，但也绝不能"完全不比较"。
    for _k, _label in (("timeline", "timeline reference"),
                       ("color", "colour contract"),
                       ("ai", "implementation/model fingerprint")):
        _va, _vb = a.get(_k), b.get(_k)
        if _va != _vb:
            if not _va or not _vb:
                p.append(f"{_label} is missing on one side")
            else:
                p.append(f"{_label} differs (declaration layer, not pixels)")

    da, db = a.get("frames_digest"), b.get("frames_digest")
    dom_a = str(a.get("frames_digest_domain") or "")
    dom_b = str(b.get("frames_digest_domain") or "")
    if not a.get("frames_digest_proven") or not b.get("frames_digest_proven"):
        p.append("frames_digest is unproven on at least one side; "
                 "cannot claim the two paths consumed the same frames")
    elif dom_a and dom_b and dom_a != dom_b:
        # ---- R24-06 / R23-A05：域不同就**不可比**，与摘要是否相同无关 ----
        # 审核反例：两份合同 `frames_digest` 文本相同、域分别是 raw RGB 与
        # PNG，旧实现只在"摘要不同"时才比域 → 空问题列表。
        # 但"两个不同表示上算出同一个摘要文本"恰恰是最危险的情况：
        # 看着一致，实际量的不是同一个东西。
        p.append(f"frames_digest domains differ ({dom_a} vs {dom_b}); "
                 f"summaries over different representations are not "
                 f"comparable")
    elif da != db:
        p.append(f"frames_digest differs: {str(da)[:16]} vs "
                 f"{str(db)[:16]} (domain={dom_a or 'unspecified'})")
    elif not dom_a or not dom_b:
        # 摘要相同、但至少一侧没写域 → 无法确认两边量的是同一个东西。
        # "摘要域必填"是审核的明确要求；缺它时相等不算证据。
        p.append("frames_digest matches but the digest domain is unspecified "
                 "on at least one side; equality over unknown domains cannot "
                 "be trusted")
    return p


def write_contract(path: str, contract: Dict[str, Any]) -> bool:
    """原子写合同侧车；失败返回 False（不抛）。"""
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".",
                    exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(contract, f, ensure_ascii=False, sort_keys=True,
                      indent=2)
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


def read_contract(path: str) -> Optional[Dict[str, Any]]:
    """读合同侧车；读不出来或结构不对返回 None。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    return doc
