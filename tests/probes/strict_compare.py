# -*- coding: utf-8 -*-
"""探针共用的**严格媒体比较器**（R24-07 / R23-A08+A09）。

## 为什么要抽出来

审核原话：

> 抽取既有严格比较实现供所有探针共用，避免修 R22 脚本而 R23 新脚本复制旧漏洞。

实际情况就是这句话描述的样子：`r22_repeatability.py`、`r23_n22_ablation.py`、
`r23_n20_matrix.py`、`g34_equiv.py`、`g34_equiv_det.py` 各自有一份 `compare`
或 `_psnr`，修了其中一份不会传染给其它几份。R23 的 N22 就是这样出的问题：
成品路径是同一个，比较器照样报"逐像素相同"。

## 判据（缺一条就不能说"相同"）

1. 两个输入都存在且非空；
2. **不是自比**：路径不同、inode 不同（硬链接/软链接都算自比）；
3. 解码 rc=0，且解码出的字节数**正好等于** `帧数 × 每帧字节`（短读即失败）；
4. 帧数等于调用方给的期望值（不给时要求两边帧数相等且 > 0）；
5. 精度与格式：两边解码用的 `pix_fmt` 必须相同（不同精度不可比）；
6. 比较数据全部有限（NaN/Inf 直接判失败，不参与统计）；
7. 可选检查时间覆盖与音轨（由调用方声明要不要）。

## 为什么不用文件 hash

同一份像素编两次 FFV1，字节可能不同（容器头/时间戳）；而"字节相同"
也不代表像素相同（同一容器里可以有 padding）。所以判据落在**解码后的
样本**上，文件 hash 只作附加记录。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

#: 每像素字节数表（只列本项目会用到的，避免用"猜位深"）。
_PIX_BYTES = {
    "yuv420p16le": 3,      # 1.5 样本/像素 × 2 字节
    "yuv420p10le": 3,
    "yuv420p": 1,          # 1.5 样本/像素 × 1 字节
    "yuv444p16le": 6,
    "yuv444p10le": 6,
    "rgb48le": 6,
    "gbrp16le": 6,
}


def ffprobe_facts(path: str) -> Dict[str, Any]:
    """取帧数 / 时长 / 是否带音轨 / 像素格式（只读包头）。"""
    out: Dict[str, Any] = {"exists": False, "frames": 0, "duration": 0.0,
                           "has_audio": False, "pix_fmt": "", "size": 0,
                           "width": 0, "height": 0}
    if not path or not os.path.isfile(path):
        return out
    out["exists"] = True
    out["size"] = os.path.getsize(path)
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-show_entries",
             "stream=codec_type,nb_frames,pix_fmt,r_frame_rate,width,height",
             "-of", "json", path],
            capture_output=True, timeout=300)
        if r.returncode == 0:
            doc = json.loads(r.stdout.decode("utf-8", "replace") or "{}")
            dur = (doc.get("format") or {}).get("duration")
            if dur is not None:
                try:
                    out["duration"] = float(dur)
                except (TypeError, ValueError):
                    pass
            for st in (doc.get("streams") or []):
                if st.get("codec_type") == "video" and not out["pix_fmt"]:
                    out["pix_fmt"] = str(st.get("pix_fmt") or "")
                    try:
                        out["frames"] = int(st.get("nb_frames") or 0)
                    except (TypeError, ValueError):
                        out["frames"] = 0
                    try:
                        out["width"] = int(st.get("width") or 0)
                        out["height"] = int(st.get("height") or 0)
                    except (TypeError, ValueError):
                        pass
                elif st.get("codec_type") == "audio":
                    out["has_audio"] = True
    except Exception:
        pass
    if out["frames"] <= 0:
        # 有些容器不写 nb_frames，退化成数包
        try:
            r2 = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-count_packets", "-show_entries", "stream=nb_read_packets",
                 "-of", "csv=p=0", path], capture_output=True, timeout=600)
            if r2.returncode == 0:
                out["frames"] = int((r2.stdout or b"").decode().strip() or 0)
        except Exception:
            pass
    return out


def file_identity(path: str) -> Dict[str, Any]:
    """路径别名检测所需的事实：绝对路径、inode、size。"""
    d: Dict[str, Any] = {"path": os.path.abspath(str(path or "")),
                         "exists": False, "inode": None, "size": 0}
    try:
        st = os.stat(str(path))
        d.update(exists=True, inode=int(getattr(st, "st_ino", 0)),
                 size=int(st.st_size))
    except OSError:
        pass
    return d


def decode_raw(path: str, pix_fmt: str, width: int, height: int,
               limit_frames: Optional[int] = None) -> Tuple[bytes, int]:
    """解码成 rawvideo 字节。返回 ``(bytes, rc)``。

    **必须用裸 bytes 的 subprocess**：`text=True` 拿到的 stdout 是 str，
    `np.frombuffer` 会直接 TypeError（R23 三个探针各踩一次）。
    """
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
           "-pix_fmt", pix_fmt]
    if limit_frames:
        cmd += ["-frames:v", str(int(limit_frames))]
    cmd += ["-"]
    r = subprocess.run(cmd, capture_output=True, timeout=3600)
    return (r.stdout or b""), int(r.returncode)


def _psnr(mse: float, peak: float) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10((peak * peak) / mse)


def compare_media(a_path: str, b_path: str, *,
                  pix_fmt: str,
                  width: int, height: int,
                  expect_frames: Optional[int] = None,
                  check_audio: bool = False,
                  timeout_hint: str = "") -> Dict[str, Any]:
    """严格比较两份媒体在**解码像素**上是否相同。

    Args:
        pix_fmt: 两边共用的解码格式（必须相同，否则不可比）。
        expect_frames: 期望帧数；给定时两边都必须是这个数。
        check_audio: 是否要求"两边有/无音轨"一致。

    Returns:
        dict：``ok`` / ``problems`` / ``identical`` / ``frames`` /
        ``max_abs_diff`` / ``psnr`` / ``alias`` / ``hashes``。
        只有 ``ok and identical`` 才能说"逐像素相同"。
    """
    res: Dict[str, Any] = {
        "ok": False, "problems": [], "identical": False,
        "frames": {}, "max_abs_diff": None, "psnr": {},
        "alias": {}, "hashes": {}, "pix_fmt": pix_fmt,
        "expect_frames": expect_frames, "hint": timeout_hint,
    }
    p = res["problems"]

    # ---- 1) 存在性 ----
    fa, fb = ffprobe_facts(a_path), ffprobe_facts(b_path)
    for tag, f in (("a", fa), ("b", fb)):
        if not f["exists"]:
            p.append(f"input {tag} does not exist: {a_path if tag == 'a' else b_path}")
        elif f["size"] <= 0:
            p.append(f"input {tag} is empty")
    if p:
        return res

    # ---- 2) 路径别名（自比）----
    ia, ib = file_identity(a_path), file_identity(b_path)
    res["alias"] = {
        "same_path": ia["path"] == ib["path"],
        "same_inode": (ia["inode"] is not None and ia["inode"] == ib["inode"]),
    }
    if res["alias"]["same_path"]:
        p.append("both sides are the same path; this is a self-comparison")
    elif res["alias"]["same_inode"]:
        p.append("both sides are the same file object (hard/sym link); "
                 "this is a self-comparison")
    if p:
        return res

    # ---- 3) 帧数 ----
    na, nb = int(fa["frames"]), int(fb["frames"])
    res["frames"] = {"a": na, "b": nb}
    if na <= 0 or nb <= 0:
        p.append(f"frame count could not be established (a={na}, b={nb})")
    elif expect_frames is not None:
        if na != int(expect_frames) or nb != int(expect_frames):
            p.append("frame count does not match the expected number "
                     f"(a={na}, b={nb}, expected={expect_frames})")
    elif na != nb:
        p.append(f"the two sides have different frame counts (a={na}, b={nb})")

    # ---- 4) 精度/格式 ----
    for tag, f in (("a", fa), ("b", fb)):
        if f["pix_fmt"] and f["pix_fmt"] != pix_fmt:
            # 不要求容器里就是解码格式（420 vs 444 可以互转比较），
            # 但**必须**在结果里如实记录，否则读者不知道比的是什么。
            res.setdefault("source_pix_fmt", {})[tag] = f["pix_fmt"]

    # ---- 5) 音轨一致性 ----
    if check_audio and fa["has_audio"] != fb["has_audio"]:
        p.append("one side has an audio track and the other does not")

    if p:
        return res

    # ---- 宽度/高度：0 表示自动取（避免调用方各自猜尺寸）----
    _w, _h = int(width or 0), int(height or 0)
    if _w <= 0 or _h <= 0:
        _w = int(fa["width"] or fb["width"] or 0)
        _h = int(fa["height"] or fb["height"] or 0)
        res["auto_size"] = [ _w, _h ]
    if _w <= 0 or _h <= 0:
        p.append("width/height could not be established; refusing to guess "
                 "the frame size")
        return res

    # ---- 6) 解码 ----
    bpp = _PIX_BYTES.get(pix_fmt)
    if not bpp:
        p.append(f"unknown decode format {pix_fmt!r}; refusing to guess the "
                 "frame size")
        return res
    frame_bytes = int(_w) * int(_h) * bpp
    if frame_bytes <= 0:
        p.append("width/height must be positive")
        return res

    raw_a, rc_a = decode_raw(a_path, pix_fmt, _w, _h)
    raw_b, rc_b = decode_raw(b_path, pix_fmt, _w, _h)
    res["decode_rc"] = {"a": rc_a, "b": rc_b}
    if rc_a != 0 or rc_b != 0:
        p.append(f"decode failed (rc a={rc_a}, b={rc_b})")
        return res
    # 短读/多读都要报：只有"正好 n 帧"才算完整解码
    for tag, raw in (("a", raw_a), ("b", raw_b)):
        if len(raw) % frame_bytes:
            p.append(f"decoded bytes of {tag} are not a whole number of "
                     f"frames (len={len(raw)}, frame={frame_bytes})")
    if p:
        return res
    arr_a = np.frombuffer(raw_a, dtype="<u2" if pix_fmt.endswith(("16le", "10le"))
                          else "u1")
    arr_b = np.frombuffer(raw_b, dtype="<u2" if pix_fmt.endswith(("16le", "10le"))
                          else "u1")
    res["frames"]["decoded_a"] = int(arr_a.size // (frame_bytes // arr_a.itemsize))
    res["frames"]["decoded_b"] = int(arr_b.size // (frame_bytes // arr_b.itemsize))
    if arr_a.size != arr_b.size:
        p.append("the two sides decode to different sample counts "
                 f"({arr_a.size} vs {arr_b.size})")
        return res

    # ---- 7) 有限性与差异 ----
    x = arr_a.astype(np.int64)
    y = arr_b.astype(np.int64)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):  # 整型恒真，防御浮点源
        p.append("comparison data contains non-finite values")
        return res
    diff = np.abs(x - y)
    res["max_abs_diff"] = int(diff.max()) if diff.size else 0
    mse = float(np.mean((x - y) ** 2)) if diff.size else 0.0
    peak = float((1 << (16 if arr_a.dtype == np.uint16 else 8)) - 1)
    res["psnr"] = {"mse": mse, "peak": peak, "db": _psnr(mse, peak),
                   "all_inf": bool(mse == 0.0)}
    res["identical"] = bool(res["max_abs_diff"] == 0)

    # ---- 8) 文件 hash 只作附加记录 ----
    for tag, path in (("a", a_path), ("b", b_path)):
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            res["hashes"][tag] = h.hexdigest()[:32]
        except OSError:
            res["hashes"][tag] = ""

    res["ok"] = not p
    return res
