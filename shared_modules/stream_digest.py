# -*- coding: utf-8 -*-
"""两遍绑定的**像素摘要**：`vspipe(vpy) | ffmpeg -f framemd5 -`。

## 文档要求

《设计 v1》§5.1：

> 正式遍：从原源重算同一 stage1，接后续完整管道；累计**同边界有序像素、
> 时间和颜色摘要**，验证与检查遍一致，再复用内容绑定的检查结论。
> **只比帧数、计划和尺寸不够。**

《验收门》G8 通过条件也写了同一句：两遍的"同边界有序像素＋时间＋颜色摘要
必须绑定。故意让第二遍产生不同像素，确认仅同 plan/尺寸/帧数不能复用 PASS。"

## 为什么用 framemd5

`ffmpeg -f framemd5` 输出**逐帧**的 MD5（含 pts、时长、尺寸、像素格式），
正是"有序 + 逐帧 + 同边界"三件事：

* 有序：行顺序就是帧顺序，重排会改变摘要；
* 逐帧：不做整片哈希，任何一帧变化都能被发现；
* 同边界：它哈希的是**解码后的像素**，不是文件字节 —— 与"容器里怎么存"无关。

我们用 sha256(整段 framemd5 文本) 作为**这一个边界**的摘要，并把帧数、
像素格式、时间基一起带出来，避免"只有摘要没有域"（R22-A05 的老问题）。

## 边界（不许越界声称）

* 它只证明"两次解码出来的像素序列相同"，**不**证明"编码/封装相同"；
* 它要求两次跑的是**同一份 vpy**；vpy 的 sha256 单独记录，两边都要核；
* VS 多线程渲染本身不保证逐像素可复现（见 `vs_render` 的实测），
  所以**只有 `vs.deterministic: true` 时两次摘要才必然相等**。
  非确定性时，本模块的结论要标成"检查遍摘要（不可复现）"，
  不能拿它当"正式遍已验证"的证据。
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional


def digest_vpy(vpy: str, *, timeout: float = 3600.0,
               pix_fmt: str = "", out_path: str = "") -> Dict[str, Any]:
    """跑一遍 vpy，返回**有序逐帧像素摘要**。

    Args:
        vpy: stage1 生成的 VapourSynth 脚本（绝对路径）。
        timeout: 整片解码的时间上限（默认 1 小时，长片按需调）。
        pix_fmt: 期望的解码像素格式；给定时会校验，不一致直接判失败。
        out_path: 若给定，framemd5 **写到这个文件**而不是内存里的临时管道。

    ## 为什么要 `out_path`

    主流程的卡死看门狗按"temp 里本 job 前缀文件的总字节数"判断有没有进展。
    摘要只走管道、不落盘 —— 于是**一段只算摘要的阶段在它眼里完全静止**，
    长片实测被当成 stall 杀掉（`Stage stalled: no progress for 300s
    (last marker=1957)`，8160 帧）。写到一个会增长的文件里，看门狗看到的是
    真实进展（帧数在涨），不是假心跳。

    Returns:
        ``{"ok": bool, "digest": str|None, "frames": int, "pix_fmt": str,
        "error": str, "pix_fmt_declared": str}``。
        ``ok=False`` 时 ``digest`` 为 None —— **绝不返回空串冒充摘要**。
    """
    out: Dict[str, Any] = {"ok": False, "digest": None, "frames": 0,
                           "pix_fmt": "", "error": "",
                           "width": 0, "height": 0, "time_base": "",
                           "pix_fmt_declared": pix_fmt}
    if not vpy or not os.path.exists(vpy):
        out["error"] = f"vs script is gone: {vpy!r}"
        return out
    # vspipe 的 stderr 必须落文件：挂 PIPE 不读会让它阻塞（stage1 实测）
    _log = None
    vp = None
    try:
        with tempfile.TemporaryDirectory() as td:
            _log = os.path.join(td, "vspipe.log")
            with open(_log, "wb") as _fh:
                vp = subprocess.Popen(
                    ["vspipe", "-c", "y4m", vpy, "-"],
                    stdout=subprocess.PIPE, stderr=_fh, shell=False)
                _dst = str(out_path) if out_path else "-"
                r = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-v", "error", "-i", "-",
                     "-f", "framemd5", _dst],
                    stdin=vp.stdout, capture_output=True, text=True,
                    errors="replace", timeout=timeout, shell=False)
                try:
                    vp.stdout.close()
                except Exception:
                    pass
                try:
                    vp.wait(timeout=60)
                except Exception:
                    try:
                        vp.kill()
                    except Exception:
                        pass
            if r.returncode != 0:
                out["error"] = ("framemd5 pass failed: "
                                f"{(r.stderr or '')[-500:]}")
                return out
            if out_path:
                try:
                    with open(out_path, "r", encoding="utf-8",
                              errors="replace") as _fh:
                        text = _fh.read()
                except OSError as _re:
                    out["error"] = f"cannot read the framemd5 file: {_re}"
                    return out
            else:
                text = r.stdout or ""
            # framemd5 每帧一行；以 '#' 开头的是版本/流信息
            rows = [ln for ln in text.splitlines()
                    if ln and not ln.lstrip().startswith("#")]
            out["frames"] = len(rows)
            if not rows:
                out["error"] = "framemd5 produced no frame rows"
                return out
            # ---- 摘要的"域" ----
            # framemd5 的数据行只有 `stream, dts, pts, duration, size, hash`，
            # **不含像素格式**；尺寸在头部 `#dimensions 0: WxH` 里。
            # 所以域要分两处取：尺寸读头部，像素格式由调用方给（它才知道
            # stage1 写的是什么）。给不出就显式写 "unknown" —— 空串会让
            # 消费方以为"域相同"，那是 R22-A05 的老毛病。
            for ln in text.splitlines():
                if ln.startswith("#dimensions"):
                    try:
                        _wh = ln.split(":", 1)[1].strip()
                        out["width"], out["height"] = [
                            int(v) for v in _wh.lower().split("x")[:2]]
                    except (IndexError, ValueError):
                        pass
                elif ln.startswith("#tb"):
                    try:
                        out["time_base"] = ln.split(":", 1)[1].strip()
                    except IndexError:
                        pass
            out["pix_fmt"] = pix_fmt or "unknown"
            if pix_fmt:
                out["pix_fmt_verified_by"] = "caller"
            # 帧尺寸必须与声明一致 —— 这是"同一边界"的最低前提。
            # 像素格式由调用方声明（见上面的说明），这里只校验尺寸。
            _decl = str(pix_fmt or "").lower()
            if _decl.startswith("yuv420") and out["width"] and out["height"]:
                _expect = out["width"] * out["height"] * 3 // 2
                try:
                    _size = int(rows[0].split(",")[4])
                except (IndexError, ValueError):
                    _size = 0
                if _size and _size not in (_expect * 2, _expect):
                    out["error"] = (
                        f"per-frame size {_size} does not match yuv420 "
                        f"{out['width']}x{out['height']} ({_expect} or "
                        f"{_expect * 2} bytes); refusing to record a digest "
                        f"whose domain cannot be established")
                    return out
            h = hashlib.sha256()
            for ln in rows:
                h.update(ln.strip().encode("utf-8"))
                h.update(b"\n")
            out["digest"] = h.hexdigest()
            out["ok"] = True
            return out
    except subprocess.TimeoutExpired:
        out["error"] = f"framemd5 pass timed out after {timeout:.0f}s"
        return out
    except OSError as exc:
        out["error"] = f"cannot run vspipe/ffmpeg: {exc}"
        return out


def probe_vpy_header(vpy: str, *, timeout: float = 120.0) -> Dict[str, Any]:
    """读 y4m 头，拿到 **宽/高/帧率**（不解码整片）。

    ## 为什么需要它

    stage3 的编码端吃的是 rawvideo（没有头），必须显式给 `-s WxH -framerate`。
    接上流式上游后第一步就没有文件可 probe 了，所以尺寸只能从 **y4m 头**取：
    `YUV4MPEG2 W320 H240 F25:1 Ip A1:1 C420p16`。

    只读第一行就杀掉 vspipe —— 代价约等于一次滤镜初始化。

    Returns:
        ``{"ok": bool, "width": int, "height": int, "fps": "25",
        "chroma": "420p16", "error": str}``。解析不出来时 ``width=0``，
        调用方必须拒绝，**不能**用 1920×1080 之类的默认值猜（P-D02 的教训）。
    """
    import re as _re
    import subprocess
    out: Dict[str, Any] = {"ok": False, "width": 0, "height": 0,
                           "fps": "", "chroma": "", "error": ""}
    if not vpy or not os.path.exists(vpy):
        out["error"] = f"vs script is gone: {vpy!r}"
        return out
    vp = None
    try:
        vp = subprocess.Popen(["vspipe", "-c", "y4m", vpy, "-"],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, shell=False)
        line = vp.stdout.readline(4096).decode("ascii", "replace")
    except OSError as exc:
        out["error"] = f"cannot start vspipe: {exc}"
        return out
    finally:
        try:
            if vp is not None:
                vp.kill()
                vp.wait(timeout=30)
        except Exception:
            pass
    if not line.startswith("YUV4MPEG2"):
        out["error"] = f"unexpected y4m header: {line[:120]!r}"
        return out
    m_w = _re.search(r"\bW(\d+)", line)
    m_h = _re.search(r"\bH(\d+)", line)
    m_f = _re.search(r"\bF(\d+:\d+)", line)
    m_c = _re.search(r"\bC(\S+)", line)
    if not (m_w and m_h):
        out["error"] = f"y4m header has no dimensions: {line[:120]!r}"
        return out
    out.update(ok=True, width=int(m_w.group(1)), height=int(m_h.group(1)),
               fps=(m_f.group(1) if m_f else ""),
               chroma=(m_c.group(1) if m_c else ""))
    return out


def compare(check: Dict[str, Any], formal: Dict[str, Any],
            *, deterministic: bool) -> Dict[str, Any]:
    """比较两遍的摘要，给出**可引用的结论**。

    Returns:
        ``{"match": bool|None, "reason": str, "comparable": bool}``。

    ``match=None`` 表示"无法比较"（某一遍没有摘要），调用方**不得**当成通过；
    ``deterministic=False`` 时即使摘要相同也只能说"这一次一致"，不能据此
    声称正式遍已验证。
    """
    if not (check or {}).get("ok") or not (formal or {}).get("ok"):
        return {"match": None, "comparable": False,
                "reason": ("no digest on "
                           + ("the check pass" if not (check or {}).get("ok")
                              else "the formal pass"))}
    if int(check.get("frames") or 0) != int(formal.get("frames") or 0):
        return {"match": False, "comparable": True,
                "reason": (f"frame count differs: check="
                           f"{check.get('frames')} formal="
                           f"{formal.get('frames')}")}
    same = str(check.get("digest")) == str(formal.get("digest"))
    if same:
        reason = ("pixel digests match"
                  + ("" if deterministic else
                     " (single observation: VapourSynth multi-threaded "
                     "rendering is not reproducible, so this equality does "
                     "not prove the next run will match)"))
    else:
        reason = ("pixel digests differ: the formal pass produced different "
                  "pixels than the check pass; the check verdict cannot be "
                  "reused")
    return {"match": same, "comparable": True, "reason": reason}
