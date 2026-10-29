# -*- coding: utf-8 -*-
"""帧数守恒：中间产物不许被"重定时"悄悄改掉帧数（R15-04）。

**为什么需要它**

`stage4` 原来没有 `-fps_mode passthrough`。`stage3` 的 VFR 路径写出的 MKV
把 `r_frame_rate` 标成 `25/1`（帧数对、标签错），于是 `stage4` 的默认
`-vsync cfr` 按 25fps 重新定时，**丢掉放不下的帧**：

| 场景 | 输入帧 | 输出帧 |
|---|---|---|
| 真机（VFR 素材） | 458 | **381**（-16.8%） |
| 受控复现（合成 VFR） | 120 | **102**（-15%） |

加了 `-fps_mode passthrough` 后两者都恢复到 1:1。但**同样的错可以再犯**
（别处少一个 flag 就够了），所以要有一道守门。

**成本取舍**

FFV1 MKV 的 `nb_frames` 是 `N/A`，精确计数只能 `-count_frames` 全解码 ——
对长片太贵。所以分两层：

1. **元数据快查（总是做，几乎免费）**：比较输入/输出的
   `r_frame_rate` 与时长。帧率被改写 = 极可能发生了重定时，直接判失败。
2. **精确计数（仅在文件较小时做）**：短片全解码数帧，逐帧确认。

长片只做第 1 层；stage7 的全链门会用 `count_frames` 做最终核对，
两层合起来既有早失败的定位能力，也不给长片加几分钟解码。
"""
from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional, Tuple

#: 精确计数只对这个大小以下的文件做（默认 512 MB）
DEFAULT_EXACT_MAX_BYTES = 512 * 1024 * 1024

#: 时长允许的相对偏差（重定时通常伴随明显变化）
DEFAULT_DURATION_TOLERANCE = 0.02


def _probe(path: str, entries: str) -> Optional[str]:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", entries, "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, errors="replace",
            timeout=60, shell=False)
        if r.returncode != 0:
            return None
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _rate(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    t = text.strip()
    if t in ("", "N/A", "0/0"):
        return None
    try:
        if "/" in t:
            a, b = t.split("/", 1)
            num, den = int(a), int(b)
            return (num / den) if den else None
        v = float(t)
        return v if v > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def _duration(path: str) -> Optional[float]:
    t = _probe(path, "format=duration")
    try:
        v = float(t) if t else 0.0
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _count_frames(path: str) -> Optional[int]:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, errors="replace",
            timeout=3600, shell=False)
        if r.returncode != 0:
            return None
        return int(r.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def describe(path: str) -> Dict[str, Any]:
    """取一份便宜的元数据快照（不解码）。"""
    return {
        "path": path,
        "r_frame_rate": _rate(_probe(path, "stream=r_frame_rate")),
        "avg_frame_rate": _rate(_probe(path, "stream=avg_frame_rate")),
        "duration": _duration(path),
        "bytes": os.path.getsize(path) if os.path.isfile(path) else None,
    }


def check_conservation(src: str, out: str, stage: str,
                       logger=None,
                       exact_max_bytes: int = DEFAULT_EXACT_MAX_BYTES,
                       duration_tolerance: float = DEFAULT_DURATION_TOLERANCE,
                       expected_fps: Optional[float] = None,
                       ) -> Tuple[bool, Dict[str, Any]]:
    """核对 ``src`` → ``out`` 有没有被重定时/丢帧。

    Args:
        src: 本阶段的输入（中间产物）。
        out: 本阶段写出的中间产物。
        stage: 阶段名，用于日志。
        exact_max_bytes: ``out`` 小于此值时才做精确帧计数。
        expected_fps: 本次应当产出的帧率。给出时，若 ``out`` 声明的
            ``r_frame_rate`` 与它明显不符，记一条**告警**并写进报告 ——
            VFR 中间产物常被 ffmpeg 标成 25/1（时间戳正确、标签错），
            下游若漏掉 `-fps_mode passthrough` 就会按这个错标签重定时丢帧。

    Returns:
        ``(ok, report)``。``ok=False`` 表示**这一阶段的输出不可信**，
        调用方应当失败而不是继续。
    """
    rep: Dict[str, Any] = {"stage": stage, "src": None, "out": None,
                           "reasons": [], "exact": False, "ok": True}

    if not (os.path.isfile(src) and os.path.isfile(out)):
        rep["ok"] = False
        rep["reasons"].append("missing input or output file")
        return False, rep

    a = describe(src)
    b = describe(out)
    rep["src"] = a
    rep["out"] = b

    # ---- 第 1 层：帧率被改写 = 极可能重定时 ----
    if a["r_frame_rate"] and b["r_frame_rate"]:
        diff = abs(a["r_frame_rate"] - b["r_frame_rate"])
        if diff / max(a["r_frame_rate"], 1e-9) > 0.001:
            rep["reasons"].append(
                f"r_frame_rate changed {a['r_frame_rate']:.4f} -> "
                f"{b['r_frame_rate']:.4f} (re-timed; frames may have been "
                f"dropped)")
            rep["ok"] = False

    # ---- 时长必须基本不变（重定时会保持时长但丢帧，这里只抓"被拉伸"）----
    if a["duration"] and b["duration"]:
        rel = abs(a["duration"] - b["duration"]) / a["duration"]
        if rel > duration_tolerance:
            rep["reasons"].append(
                f"duration changed {a['duration']:.3f}s -> "
                f"{b['duration']:.3f}s ({rel * 100:.1f}%)")
            rep["ok"] = False

    # ---- 第 2 层：小文件精确计数 ----
    if b["bytes"] is not None and b["bytes"] <= exact_max_bytes:
        n_in = _count_frames(src)
        n_out = _count_frames(out)
        rep["exact"] = True
        rep["frames_in"] = n_in
        rep["frames_out"] = n_out
        if n_in is not None and n_out is not None and n_in != n_out:
            rep["reasons"].append(
                f"frame count changed {n_in} -> {n_out} "
                f"({(n_out - n_in) / n_in * 100:+.1f}%)")
            rep["ok"] = False
    else:
        rep["exact"] = False
        rep["note"] = ("exact counting skipped (file larger than "
                       f"{exact_max_bytes} bytes); metadata check only")

    # ---- 附加：中间产物的帧率标签是否与本次意图一致（只告警）----
    if expected_fps and b["r_frame_rate"]:
        _rel = abs(b["r_frame_rate"] - expected_fps) / max(expected_fps, 1e-9)
        if _rel > 0.001:
            rep["fps_label_mismatch"] = {
                "expected": expected_fps,
                "labeled": b["r_frame_rate"],
            }
            if logger is not None:
                logger.warning(
                    "FRAME_FPS_LABEL_MISMATCH",
                    "This intermediate declares a different frame rate than "
                    "this run intended. For a VFR track that is expected "
                    "(there is no single frame rate; ffmpeg commonly writes "
                    "25/1 as a placeholder), so this is informational, NOT a "
                    "failure. Two rules follow from it: (1) any downstream "
                    "ffmpeg call MUST use `-fps_mode passthrough`, otherwise "
                    "it re-times and drops frames; (2) do NOT 'fix' the label "
                    "by adding `-r` to the encoder — measured, that flattens "
                    "VFR (an 8.0s variable-timeline clip came out 4.0s).",
                    stage=stage, expected=round(expected_fps, 4),
                    labeled=round(b["r_frame_rate"], 4))

    if logger is not None:
        if rep["ok"]:
            logger.info(
                "FRAME_CONSERVATION_OK",
                "Intermediate preserved its frame count and timing",
                stage=stage, exact=rep["exact"],
                frames=(f"{rep.get('frames_in')} -> {rep.get('frames_out')}"
                        if rep["exact"] else None))
        else:
            logger.error(
                "FRAME_CONSERVATION_BROKEN",
                "This stage changed the frame count/timing of the "
                "intermediate. Refusing to treat the output as valid.",
                stage=stage, reasons=rep["reasons"],
                src_r_fps=(a or {}).get("r_frame_rate"),
                out_r_fps=(b or {}).get("r_frame_rate"),
                frames=(f"{rep.get('frames_in')} -> {rep.get('frames_out')}"
                        if rep["exact"] else None))
    return rep["ok"], rep
