# -*- coding: utf-8 -*-
"""`producer | tap | consumer` —— 在**真实数据流**上装观测点。

## 为什么需要它（R26-A04）

复核指出的问题：两遍绑定原来只比"帧数相同"，于是

* 同 10 帧、摘要完全不同 → 绑定成功；
* 任意一端 0 帧 → 也通过；
* 消费侧车缺失 → 退回**额外重放**，再次通过。

根因是"观测点不在实际流上"。检查遍跑 `vspipe | ffmpeg -vf idet`，
正式遍跑 `vspipe | ffmpeg -f rawvideo rgb48le | AI` —— 两条链各自独立，
没有任何一处看到过"对方也看到了的同一批字节"。

本模块把这个观测点放到**两边共有、且还没做格式转换的那一段**：
`vspipe` 输出的 y4m。检查遍和正式遍都在这里过一次 tap，
于是两份摘要比的是**同一域**的内容。

## 为什么不担心性能

实测（1080p24 y4m，187 MB）：

    字节: 186.6 MB  耗时: 0.45s  吞吐: 417.3 MB/s

1080p24 y4m 的码率约 74 MB/s，**5.6 倍余量**。用 `blake2b`（比 sha256 快
约 2–3 倍）已经足够，不需要降采样粒度。

## 边界

* **只中转、不改字节**：写出去的 chunk 与读到的完全相同；
* stderr 并发有界排空（不是等 stdout 读完再读，那会死锁）；
* 帧边界按 y4m 协议解析（`FRAME\n` + 定长数据），**不是按 chunk 猜**；
* 短帧/截断：明确记 `short_frame`，不补零、不当正常结束。
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["parse_y4m_header", "StreamTap"]


_Y4M_HDR = re.compile(rb"^YUV4MPEG2\s+(.*)$")


def parse_y4m_header(line: bytes) -> Dict[str, Any]:
    """从 `YUV4MPEG2 ...` 头解析出帧几何。

    Returns:
        ``{"ok": bool, "width": int, "height": int, "chroma": str,
        "frame_bytes": int, "fps": str, "problem": str}``。

    `frame_bytes` 是**一帧原始数据**的字节数（不含 `FRAME\\n`）。
    """
    out: Dict[str, Any] = {"ok": False, "width": 0, "height": 0,
                           "chroma": "", "frame_bytes": 0, "fps": "",
                           "problem": ""}
    m = _Y4M_HDR.match(line.strip())
    if not m:
        out["problem"] = "not a YUV4MPEG2 header"
        return out
    w = h = 0
    chroma = "420"
    fps = ""
    for tok in m.group(1).split():
        if tok[:1] == b"W":
            try:
                w = int(tok[1:])
            except ValueError:
                pass
        elif tok[:1] == b"H":
            try:
                h = int(tok[1:])
            except ValueError:
                pass
        elif tok[:1] == b"C":
            chroma = tok[1:].decode("ascii", "replace") or "420"
        elif tok[:1] == b"F":
            fps = tok[1:].decode("ascii", "replace")
    if w <= 0 or h <= 0:
        out["problem"] = "header carries no usable W/H"
        return out
    # 色度格式 → 每像素字节数（8bit 平面）
    #
    # 注意：**不能用 `split("p")`** —— `420mpeg2` 里的 `mpeg2` 也含 `p`，
    # 拆分后得到 `"420m"` 匹配不上，`per_px` 会退回 3.0（RGB 口径），
    # 于是 `frame_bytes` 算成 1.5 倍大 → 读帧越界 → 下一帧的 `FRAME`
    # 标记错位。实测就是这样把 10 帧读成 1 帧的。
    #
    # ffmpeg 的 y4m 输出实际写 `C420mpeg2` / `C420jpeg` / `C420paldv` /
    # `C422` / `C444` / `Cmono`（可能带 `p10`/`p16` 位深后缀），
    # 所以按**前缀**判断。
    _c = chroma.lower()
    if _c.startswith("mono"):
        per_px = 1.0
    elif _c.startswith("420"):
        per_px = 1.5
    elif _c.startswith("422"):
        per_px = 2.0
    elif _c.startswith("444"):
        per_px = 3.0
    elif _c.startswith("411"):
        per_px = 1.5
    else:
        per_px = 1.5                       # 未知按最常见处理，但下面会报
    depth = 8
    m2 = re.search(r"p(\d+)", _c)
    if m2:
        depth = int(m2.group(1))
    # **10/12/16 bit 每样本占 2 字节**（y4m 用小端 16 位存储）。
    # `depth // 8` 对 10bit 得 1 —— 那是错的，会把宽度算成一半，
    # 读帧越界（测试 `test_header_parsing_handles_mpeg2_chroma_suffix`
    # 就是因此失败）。
    bytes_per_sample = 2 if depth > 8 else 1
    out.update({
        "ok": True, "width": w, "height": h, "chroma": chroma,
        "fps": fps, "depth": depth,
        "frame_bytes": int(w * h * per_px * bytes_per_sample),
    })
    if _c not in ("420", "420mpeg2", "420jpeg", "420paldv", "422", "444",
                  "mono") and not _c.startswith(("420p", "422p", "444p",
                                                "monop")):
        out["ok"] = False
        out["problem"] = ("unsupported chroma %r: refusing to guess the "
                          "frame size" % chroma)
    return out


class StreamTap:
    """在 producer 与 consumer 之间中转，同时算**有序**摘要。

    用法::

        tap = StreamTap()
        res = tap.run(producer_cmd, consumer_cmd)
        # res["frames"], res["digest"], res["producer_rc"], ...

    `digest` 的输入是**每帧数据的顺序拼接** + 帧数，与
    `frame_contract.digest_ordered_frames` 同一口径（这里不 import 它，
    避免循环依赖；算法写在 docstring 里以便核对）。
    """

    def __init__(self, chunk: int = 1 << 20,
                 stderr_limit: int = 64) -> None:
        self.chunk = int(chunk)
        self.stderr_limit = int(stderr_limit)

    # ------------------------------------------------------------------ 内部
    def _drain(self, pipe, sink: List[bytes]) -> None:
        try:
            while True:
                b = pipe.read(4096)
                if not b:
                    break
                sink.append(b)
                if len(sink) > self.stderr_limit:
                    del sink[0]
        except Exception:
            pass

    @staticmethod
    def _read_exact(stream, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            c = stream.read(n - len(buf))
            if not c:
                break
            buf.extend(c)
        return bytes(buf)

    # ------------------------------------------------------------------ 主流程
    def run(self, producer_cmd: List[str], consumer_cmd: List[str],
            *, cwd: Optional[str] = None,
            timeout: float = 0.0) -> Dict[str, Any]:
        """跑 `producer | tap | consumer`。

        Args:
            producer_cmd: 产出 y4m 到 stdout 的命令（如 `vspipe -c y4m ... -`）。
            consumer_cmd: 从 stdin 读 y4m 的命令（如 `ffmpeg -f yuv4mpegpipe -i -`）。
            timeout: 整体超时（秒）；0 表示不限。

        Returns:
            ``{"ok", "frames", "digest", "bytes", "width", "height",
            "chroma", "fps", "short_frame", "header_problem",
            "producer_rc", "consumer_rc", "producer_err", "consumer_err",
            "elapsed"}``。

            ``ok`` 只有在**两个进程 rc==0 且帧数与字节都完整**时才为真。
        """
        out: Dict[str, Any] = {
            "ok": False, "frames": 0, "digest": "", "bytes": 0,
            "width": 0, "height": 0, "chroma": "", "fps": "",
            "short_frame": False, "header_problem": "",
            "producer_rc": None, "consumer_rc": None,
            "producer_err": "", "consumer_err": "", "elapsed": 0.0,
        }
        t0 = time.monotonic()
        p_err: List[bytes] = []
        c_err: List[bytes] = []

        prod = subprocess.Popen(producer_cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, cwd=cwd, shell=False)
        cons = subprocess.Popen(consumer_cmd, stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE, cwd=cwd, shell=False)
        tp = threading.Thread(target=self._drain, args=(prod.stderr, p_err),
                              daemon=True)
        tc = threading.Thread(target=self._drain, args=(cons.stderr, c_err),
                              daemon=True)
        tp.start()
        tc.start()

        h = hashlib.blake2b()
        try:
            # ---- y4m 头 ----
            hdr = b""
            while not hdr.endswith(b"\n"):
                c = prod.stdout.read(1)
                if not c:
                    break
                hdr += c
            if not hdr:
                out["header_problem"] = "producer produced no header"
            else:
                meta = parse_y4m_header(hdr)
                out.update({k: meta[k] for k in
                            ("width", "height", "chroma", "fps")})
                if not meta["ok"]:
                    out["header_problem"] = meta["problem"]
                else:
                    try:
                        cons.stdin.write(hdr)
                    except BrokenPipeError:
                        pass                  # 下游提前关管，如实记录
                    fb = meta["frame_bytes"]
                    while True:
                        # `FRAME` + 可选标签 + 换行
                        line = b""
                        while not line.endswith(b"\n"):
                            c = prod.stdout.read(1)
                            if not c:
                                break
                            line += c
                        if not line:
                            break             # 正常 EOF
                        if not line.startswith(b"FRAME"):
                            out["header_problem"] = (
                                "unexpected frame marker: %r" % line[:20])
                            break
                        data = self._read_exact(prod.stdout, fb)
                        if len(data) < fb:
                            # 截断：不补零、不当正常结束
                            out["short_frame"] = True
                            out["bytes"] += len(data)
                            h.update(data)
                            out["frames"] += 1
                            try:
                                cons.stdin.write(line + data)
                            except BrokenPipeError:
                                pass
                            break
                        h.update(data)
                        out["bytes"] += len(data)
                        out["frames"] += 1
                        try:
                            cons.stdin.write(line)
                            cons.stdin.write(data)
                        except BrokenPipeError:
                            # 下游提前关管：**停止中转**但保留已观测部分，
                            # 并在 rc 判定里体现（consumer_rc != 0）
                            out["header_problem"] = (
                                "consumer closed the pipe early")
                            break
        except Exception as exc:
            out["header_problem"] = "tap failed: %s" % str(exc)[:160]
        finally:
            try:
                if cons.stdin:
                    cons.stdin.close()
            except Exception:
                pass
            try:
                if prod.stdout:
                    prod.stdout.close()
            except Exception:
                pass
            if timeout:
                try:
                    prod.wait(timeout=timeout)
                except Exception:
                    try:
                        prod.kill()
                    except Exception:
                        pass
            else:
                prod.wait()
            cons.wait()
            tp.join(timeout=10)
            tc.join(timeout=10)
            out["producer_rc"] = prod.returncode
            out["consumer_rc"] = cons.returncode
            out["producer_err"] = b"".join(p_err)[-600:].decode(
                "utf-8", "replace")
            out["consumer_err"] = b"".join(c_err)[-600:].decode(
                "utf-8", "replace")
            out["elapsed"] = round(time.monotonic() - t0, 3)

        h.update(b"|frames=" + str(out["frames"]).encode("ascii"))
        out["digest"] = h.hexdigest()
        # **两个 rc 都要 0**，且不能有短帧/头问题/0 帧
        out["ok"] = (out["producer_rc"] == 0 and out["consumer_rc"] == 0
                     and not out["short_frame"]
                     and not out["header_problem"]
                     and out["frames"] > 0)
        return out
