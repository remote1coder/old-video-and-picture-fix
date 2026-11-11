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
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA = "frame-boundary/1"

#: 允许的通道顺序（只写实际用到的两种）。
CHANNEL_ORDERS = ("rgb", "bgr")

#: 像素格式 → (位深, 通道顺序)。只列本项目实际产出的格式。
_PIX_FMT_INFO = {
    "rgb48le": (16, "rgb"),
    "gbrp16le": (16, "rgb"),
    "yuv420p16le": (16, None),
    "yuv420p10le": (10, None),
    "yuv420p": (8, None),
}


def pix_fmt_info(pix_fmt: str) -> Tuple[Optional[int], Optional[str]]:
    """像素格式 → ``(bit_depth, channel_order)``；未知格式返回 ``(None, None)``。

    未知不等于错误 —— 但要如实写 None，让消费者知道"这一项没被识别"。
    """
    return _PIX_FMT_INFO.get(str(pix_fmt or ""), (None, None))


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
        "frames_digest_domain": str(frames_digest_domain or ""),
        # 这一串帧是谁产的：模型 / 冻结帧源 / PNG 中转
        "source": str(source),
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
    if not contract.get("frames_digest_proven"):
        p.append("frames_digest is not proven (this boundary cannot be used "
                 "as evidence that both paths consumed the same frames)")
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
    da, db = a.get("frames_digest"), b.get("frames_digest")
    if not a.get("frames_digest_proven") or not b.get("frames_digest_proven"):
        p.append("frames_digest is unproven on at least one side; "
                 "cannot claim the two paths consumed the same frames")
    elif da != db:
        # 域不同（raw bytes vs PNG 字节）时摘要本来就该不同 —— 那是
        # "没法这样比"，不是"两边帧不一样"。分开报，别混成一条。
        dom_a = str(a.get("frames_digest_domain") or "")
        dom_b = str(b.get("frames_digest_domain") or "")
        if dom_a and dom_b and dom_a != dom_b:
            p.append(f"frames_digest domains differ ({dom_a} vs {dom_b}); "
                     f"summaries over different representations are not "
                     f"comparable")
        else:
            p.append(f"frames_digest differs: {str(da)[:16]} vs "
                     f"{str(db)[:16]} (domain={dom_a or 'unspecified'})")
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
