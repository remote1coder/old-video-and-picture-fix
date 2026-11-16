# -*- coding: utf-8 -*-
"""探针共用的**严格媒体比较器**（B01 / R25）。

## 这一版修的是 R24 的真缺陷

R24 的版本被审核用受控解码观测打出 5 个错：

| 反例 | R24 行为 | 正确行为 |
|---|---|---|
| M01 双方解码 0 字节、rc=0 | `ok=True, identical=True` | 拒绝（空不等于相同）|
| M02 声明 5 帧、双方只解出 1 帧 | 通过 | 拒绝（短读）|
| M03 双方解出 6 帧、声明 5 帧 | 通过 | 拒绝（多帧）|
| M04 rgb48 `[255,0,0]` vs `[256,0,0]` | `max_abs_diff=255` | 应为 **1** |
| M05 10bit 用 65535 当峰值 | `peak=65535` | 应为 **1023** |
| M06 合法 2×2 `yuv420p`（6 字节/帧）| 按 4 字节/帧**错拒** | 通过 |
| M07 2×2 与 1×4 同字节数 | 判相同 | 拒绝（异尺寸不可比）|

根因四条：

1. **dtype 靠后缀猜**：`pix_fmt.endswith(("16le","10le"))` 对 `rgb48le` 不成立
   → 按 `u1` 读，一个 16-bit 样本被拆成两个字节 → 差 1 变成差 255。
2. **不看解码帧数**：只有"字节数是整帧倍数"就放行，`decoded_a` 算出来却
   从不参与判定 → 0 帧、1 帧、6 帧都能过。
3. **平面字节数按"每像素字节"表**：`yuv420p` 写 1 字节/像素，实际是
   1.5 字节/像素（2×2 帧 = 6 字节，被判成 4）。
4. **尺寸只信调用方**：两边 ffprobe 报了不同宽高也照样比。

## 本版的判据（缺一条就不能说"相同"）

1. 输入存在、非空；**不是同一文件对象**（含卷/设备身份，不只比 inode）。
2. 两边流规格一致：宽高、像素格式、位深。
3. 解码 rc=0；**解码出的帧数**两边相等，且等于 `expect_frames`（若给出）。
4. 每帧字节数按**平面几何**精确计算（420 的色度平面向上取整）。
5. 逐帧**流式**比较，不把整片捕获进内存。
6. 有效峰值按**位深**（10bit → 1023），不按 uint16 容器。
7. 时间/音频按调用方声明成为**必需项**（不是"有/无音轨"就算比过）。
8. 结果写 JSON 时，无穷 PSNR 用显式标记，不输出 `Infinity`。

## 关于 stride

`-f rawvideo` 输出是**紧凑布局**（无行填充），所以按平面几何算出的字节数
就是每帧字节数。若将来改用带 stride 的读法，必须在 `decode_raw` 里显式
处理 —— 本模块不做隐式补齐。

来源：FFmpeg 像素格式定义（`pixfmt_8h`）与 torchcodec 的 RawFrame 文档
（"Odd sizes round up, so the chroma of a 4:2:0 frame 481 samples wide is
241 wide"）一致；`count_frames` 与 `count_packets` 是不同观测（ffprobe 文档），
本模块只用解码后的实际字节数判定，不拿包数冒充显示帧数。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: 单次读取块大小（流式）。
CHUNK = 1 << 20

#: **只给受控反例用的注入点**：`decode_raw(path, pix_fmt, w, h) -> (bytes, rc)`。
#: 默认 None（走流式）。注入不改变判据，只换取字节的方式 ——
#: 审核 B01 的 M01–M07 需要"受控解码观测"，而生产路径必须流式。
decode_raw = None


# ---------------------------------------------------------------------------
# 像素格式规格
# ---------------------------------------------------------------------------

#: 每个格式的平面划分：``(名字, 宽除数, 高除数, 每样本字节)``。
#: 除数 2 表示该方向减半，**并且按向上取整**（4:2:0 的奇数尺寸）。
_PLANES: Dict[str, Tuple[Tuple[str, int, int, int], ...]] = {
    # 8-bit
    "yuv420p": (("y", 1, 1, 1), ("u", 2, 2, 1), ("v", 2, 2, 1)),
    "yuv422p": (("y", 1, 1, 1), ("u", 2, 1, 1), ("v", 2, 1, 1)),
    "yuv444p": (("y", 1, 1, 1), ("u", 1, 1, 1), ("v", 1, 1, 1)),
    "gray": (("y", 1, 1, 1),),
    # 16-bit 容器（10/12/16 都用 2 字节存）
    "yuv420p10le": (("y", 1, 1, 2), ("u", 2, 2, 2), ("v", 2, 2, 2)),
    "yuv420p12le": (("y", 1, 1, 2), ("u", 2, 2, 2), ("v", 2, 2, 2)),
    "yuv420p16le": (("y", 1, 1, 2), ("u", 2, 2, 2), ("v", 2, 2, 2)),
    "yuv422p10le": (("y", 1, 1, 2), ("u", 2, 1, 2), ("v", 2, 1, 2)),
    "yuv422p16le": (("y", 1, 1, 2), ("u", 2, 1, 2), ("v", 2, 1, 2)),
    "yuv444p10le": (("y", 1, 1, 2), ("u", 1, 1, 2), ("v", 1, 1, 2)),
    "yuv444p16le": (("y", 1, 1, 2), ("u", 1, 1, 2), ("v", 1, 1, 2)),
    "gray10le": (("y", 1, 1, 2),),
    "gray16le": (("y", 1, 1, 2),),
    # planar RGB（平面序 G,B,R —— 比较是对称的，序不影响 max/RMSE）
    "gbrp": (("g", 1, 1, 1), ("b", 1, 1, 1), ("r", 1, 1, 1)),
    "gbrp10le": (("g", 1, 1, 2), ("b", 1, 1, 2), ("r", 1, 1, 2)),
    "gbrp16le": (("g", 1, 1, 2), ("b", 1, 1, 2), ("r", 1, 1, 2)),
    # packed RGB
    "rgb24": (("rgb", 1, 1, 3),),
    "bgr24": (("bgr", 1, 1, 3),),
    "rgb48le": (("rgb", 1, 1, 6),),
}

#: 有效位深（不是容器位数）。`rgb48le` 每通道 16 bit，总共 6 字节。
_BIT_DEPTH: Dict[str, int] = {
    "yuv420p": 8, "yuv422p": 8, "yuv444p": 8, "gray": 8,
    "gbrp": 8, "rgb24": 8, "bgr24": 8,
    "yuv420p10le": 10, "yuv422p10le": 10, "yuv444p10le": 10, "gray10le": 10,
    "gbrp10le": 10,
    "yuv420p12le": 12,
    "yuv420p16le": 16, "yuv422p16le": 16, "yuv444p16le": 16,
    "gray16le": 16, "gbrp16le": 16, "rgb48le": 16,
}

#: 通道顺序（"作为像素"看时的排列）。只对 RGB 类有意义。
_CHANNEL_ORDER: Dict[str, str] = {
    "rgb24": "rgb", "rgb48le": "rgb",
    "bgr24": "bgr",
    "gbrp": "gbr", "gbrp10le": "gbr", "gbrp16le": "gbr",
}


def pix_spec(pix_fmt: str) -> Optional[Dict[str, Any]]:
    """像素格式 → ``{planes, bit_depth, peak, dtype}``。

    不认识的格式返回 ``None`` —— **不猜**。调用方必须把它当成"无法比较"，
    而不是退回某个默认位深。
    """
    key = str(pix_fmt or "").strip().lower()
    planes = _PLANES.get(key)
    depth = _BIT_DEPTH.get(key)
    if not planes or depth is None:
        return None
    return {
        "pix_fmt": key,
        "planes": planes,
        "bit_depth": int(depth),
        # 有效峰值：10bit → 1023，16bit → 65535，8bit → 255。
        # R24 的错：10bit 用了 65535（容器位宽），PSNR 虚高 36 dB。
        "peak": float((1 << int(depth)) - 1),
        "channel_order": _CHANNEL_ORDER.get(key, ""),
        "dtype": "<u2" if depth > 8 else "u1",
    }


def frame_bytes(pix_fmt: str, width: int, height: int) -> Optional[int]:
    """一帧的**紧凑**字节数。4:2:0/4:2:2 的色度平面按向上取整。

    M06 的错就在这：`yuv420p` 2×2 是 4 + 1 + 1 = **6** 字节，
    R24 按"1 字节/像素"算成 4，把合法帧判成"不是整帧"。
    """
    spec = pix_spec(pix_fmt)
    if not spec or width <= 0 or height <= 0:
        return None
    total = 0
    for _name, wd, hd, bps in spec["planes"]:
        pw = (int(width) + wd - 1) // wd        # ceil(width / wd)
        ph = (int(height) + hd - 1) // hd       # ceil(height / hd)
        total += pw * ph * bps
    return total


def plane_geometry(pix_fmt: str, width: int,
                   height: int) -> Optional[List[Dict[str, Any]]]:
    """逐平面尺寸（供诊断与逐通道统计用）。"""
    spec = pix_spec(pix_fmt)
    if not spec or width <= 0 or height <= 0:
        return None
    out: List[Dict[str, Any]] = []
    for name, wd, hd, bps in spec["planes"]:
        pw = (int(width) + wd - 1) // wd
        ph = (int(height) + hd - 1) // hd
        out.append({"name": name, "width": pw, "height": ph,
                    "bytes_per_sample": bps, "samples": pw * ph})
    return out


# ---------------------------------------------------------------------------
# 探测
# ---------------------------------------------------------------------------

def ffprobe_facts(path: str, stream_index: int = 0) -> Dict[str, Any]:
    """取**指定视频流**的帧数 / 时长 / 尺寸 / 像素格式 / 音频流信息。

    R24 的问题之一是"没明确选流"。这里固定按 `-select_streams v:<idx>`
    取第一条（或指定）视频流，音频信息单独列。
    """
    out: Dict[str, Any] = {
        "exists": False, "size": 0, "frames": 0, "duration": 0.0,
        "pix_fmt": "", "width": 0, "height": 0,
        "audio_streams": 0, "audio_layouts": [], "stream_index": int(stream_index),
        "start_time": None, "probe_ok": False, "probe_error": "",
        # 包数是**另一种**观测，单独记，绝不拿来当显示帧数
        "packets": 0, "counted_frames": 0,
    }
    p = str(path or "")
    if not p or not os.path.isfile(p):
        return out
    out["exists"] = True
    out["size"] = os.path.getsize(p)

    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", f"v:{stream_index}",
             "-show_entries",
             "stream=index,width,height,pix_fmt,nb_frames,start_time,"
             "codec_type:format=duration",
             "-of", "json", p],
            capture_output=True, timeout=600)
        if r.returncode != 0:
            out["probe_error"] = (r.stderr or b"")[-200:].decode("utf-8", "replace")
            return out
        doc = json.loads((r.stdout or b"").decode("utf-8", "replace") or "{}")
    except Exception as exc:
        out["probe_error"] = f"{type(exc).__name__}: {exc}"[:200]
        return out

    out["probe_ok"] = True
    _dur = (doc.get("format") or {}).get("duration")
    if _dur is not None:
        try:
            out["duration"] = float(_dur)
        except (TypeError, ValueError):
            pass
    for st in (doc.get("streams") or []):
        if st.get("codec_type") != "video":
            continue
        out["pix_fmt"] = str(st.get("pix_fmt") or "")
        try:
            out["frames"] = int(st.get("nb_frames") or 0)
            out["width"] = int(st.get("width") or 0)
            out["height"] = int(st.get("height") or 0)
        except (TypeError, ValueError):
            pass
        if st.get("start_time") is not None:
            try:
                out["start_time"] = float(st.get("start_time"))
            except (TypeError, ValueError):
                pass
        break

    # 音频：**逐流**记录数量/声道布局，不用"有没有"代替
    try:
        ra = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index,channels,channel_layout,"
             "sample_rate,codec_name",
             "-of", "json", p],
            capture_output=True, timeout=600)
        if ra.returncode == 0:
            adoc = json.loads((ra.stdout or b"").decode("utf-8", "replace") or "{}")
            astreams = adoc.get("streams") or []
            out["audio_streams"] = len(astreams)
            for s in astreams:
                out["audio_layouts"].append({
                    "channels": s.get("channels"),
                    "layout": s.get("channel_layout"),
                    "sample_rate": s.get("sample_rate"),
                    "codec": s.get("codec_name"),
                })
    except Exception:
        pass

    # `nb_frames` 在无索引容器里常为 N/A；此时用**解码帧计数**，不是包数。
    if out["frames"] <= 0:
        out["counted_frames"] = _count_frames(p, stream_index)
        out["frames"] = out["counted_frames"]
    out["packets"] = _count_packets(p, stream_index)
    return out


def _count_frames(path: str, stream_index: int = 0) -> int:
    """用 `-count_frames` 数**解码后的帧**（ffprobe 文档：与包数是不同观测）。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", f"v:{stream_index}",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "csv=p=0", path],
            capture_output=True, timeout=3600)
        if r.returncode == 0:
            return int((r.stdout or b"").decode().strip() or 0)
    except Exception:
        pass
    return 0


def _count_packets(path: str, stream_index: int = 0) -> int:
    """包数（**只作诊断**，不参与"帧数"判定）。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", f"v:{stream_index}",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", path],
            capture_output=True, timeout=3600)
        if r.returncode == 0:
            return int((r.stdout or b"").decode().strip() or 0)
    except Exception:
        pass
    return 0


def file_identity(path: str) -> Dict[str, Any]:
    """文件对象身份（含**卷/设备**，不只 inode 整数）。

    审核 R25 B01 要求："别名判断包含卷/设备身份，不能只比 inode 整数。"
    Windows 上 `st_ino` 是 file index、`st_dev` 是卷序列；Linux 上分别是
    inode 与 st_dev。两者一起才能判"同一个文件对象"。
    """
    d: Dict[str, Any] = {"path": os.path.abspath(str(path or "")),
                         "exists": False, "inode": None, "device": None,
                         "size": 0, "mtime_ns": None}
    try:
        st = os.stat(str(path))
        d.update(exists=True,
                 inode=int(getattr(st, "st_ino", 0) or 0),
                 device=int(getattr(st, "st_dev", 0) or 0),
                 size=int(st.st_size),
                 mtime_ns=int(getattr(st, "st_mtime_ns", 0) or 0))
    except OSError:
        pass
    return d


def _same_object(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[bool, str]:
    """判两个路径是否指向**同一个文件对象**。"""
    if a["path"] == b["path"]:
        return True, "both sides are the same path"
    if (a["inode"] and b["inode"]
            and a["inode"] == b["inode"]
            and a["device"] and b["device"]
            and a["device"] == b["device"]):
        return True, ("both sides are the same file object "
                      f"(dev={a['device']}, inode={a['inode']})")
    return False, ""


# ---------------------------------------------------------------------------
# 解码（流式）
# ---------------------------------------------------------------------------

def _decode_cmd(path: str, pix_fmt: str, stream_index: int = 0) -> List[str]:
    # `-fps_mode passthrough` 是**必需**的，不是可选项。
    #
    # ffmpeg 输出 rawvideo 时默认 `-fps_mode cfr`：它会按**容器时长**补齐
    # 帧率，末尾复制最后一帧凑满。实测（2026-11-14，24 帧 10bit mkv，
    # 带一条 AAC 音轨）：
    #
    #     nb_read_frames=24   packets=24   **解码出 25 帧**
    #
    # 那第 25 帧是解码器凭空补的副本。它有两个害处：
    #   ① 让"解码帧数 == 期望帧数"这条判据把正确结果判成失败；
    #   ② 更糟的是它**会掩盖末帧差异** —— 两边都补同一个副本，
    #      比较结果照样"相同"，而真正的末帧可能一边少了一个。
    # 比较器绝不能比较自己造出来的帧。
    return ["ffmpeg", "-v", "error", "-nostdin",
            "-i", str(path),
            "-map", f"0:v:{stream_index}",
            "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]


def _read_exact(stream, n: int) -> bytes:
    """从管道读满 n 字节；EOF 时返回已读到的部分（可能 < n）。"""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def iter_frames(path: str, pix_fmt: str, width: int, height: int,
                *, stream_index: int = 0,
                timeout_hint: str = "") -> Tuple[Any, Dict[str, Any]]:
    """逐帧产出原始字节（**不把整片读进内存**）。

    Returns:
        ``(iterator, info)``。iterator 结束后可读 `info["frames"]` 得到
        实际帧数、`info["short_read"]` 判断尾字节。
    """
    fb = frame_bytes(pix_fmt, width, height)
    if not fb:
        raise ValueError(f"unknown or unusable pix_fmt {pix_fmt!r}")
    info: Dict[str, Any] = {"rc": None, "bytes_per_frame": fb,
                            "frames": 0, "short_read": False,
                            "stderr_tail": "", "timeout_hint": timeout_hint}
    proc = subprocess.Popen(_decode_cmd(path, pix_fmt, stream_index),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    # R25-03：**stderr 必须并发排空**。
    #
    # 原来只在 `finally` 里 `proc.stderr.read()` —— 而那时 stdout 已经读完了。
    # 只要 ffmpeg 往 stderr 写满管道缓冲（默认 64 KiB，多流/多警告的片
    # 几秒就到），它就阻塞在写上；父进程又在等 stdout 的新数据 →
    # 双方互等，`wait()` 永不返回。Python 官方文档明确提示这个双管道陷阱。
    #
    # 做法：起一个守护线程持续读 stderr，内存只保留**有界环形尾部**
    # （诊断够用，不会因整片日志撑爆内存）。
    _err_tail: List[bytes] = []
    _err_lock = threading.Lock()

    def _drain_stderr() -> None:
        try:
            while True:
                _b = proc.stderr.read(4096)
                if not _b:
                    break
                with _err_lock:
                    _err_tail.append(_b)
                    if len(_err_tail) > 64:          # ≈ 256 KiB 上限
                        del _err_tail[0]
        except Exception:
            pass

    _err_thread = threading.Thread(target=_drain_stderr, daemon=True)
    _err_thread.start()

    def _gen():
        try:
            while True:
                buf = _read_exact(proc.stdout, fb)
                if not buf:
                    break
                info["frames"] += 1
                if len(buf) < fb:
                    # 尾字节不是整帧 → 明确拒绝，不补零、不丢
                    info["short_read"] = True
                    yield buf
                    break
                yield buf
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            # 排空线程自己会读到 EOF；这里只等它收尾（有界等待）
            try:
                _err_thread.join(timeout=10)
            except Exception:
                pass
            try:
                proc.stderr.close()
            except Exception:
                pass
            info["rc"] = proc.wait()
            with _err_lock:
                _err = b"".join(_err_tail)
            info["stderr_tail"] = _err[-400:].decode("utf-8", "replace")

    return _gen(), info


# ---------------------------------------------------------------------------
# 比较
# ---------------------------------------------------------------------------

def _finite_or_none(x: Any) -> Optional[float]:
    """JSON 友好：非有限值一律 None（不输出 Infinity/NaN）。"""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _normalize_facts(d: Any) -> Dict[str, Any]:
    """把（可能是受控 stub 的）探测结果补齐成完整结构。

    受控反例只提供它关心的几个键（`exists/size/frames/duration/pix_fmt/
    width/height`），其余键缺失是正常的。**缺键不等于字段错**：这里只
    给安全默认，不掩盖"探不到"（`probe_ok` 仍按 exists 推断）。
    """
    out: Dict[str, Any] = dict(d) if isinstance(d, dict) else {}
    out.setdefault("probe_ok", bool(out.get("exists")))
    out.setdefault("probe_error", "")
    for k, v in (("size", 0), ("frames", 0), ("duration", 0.0),
                 ("pix_fmt", ""), ("width", 0), ("height", 0),
                 ("audio_streams", 0), ("packets", 0),
                 ("counted_frames", 0), ("stream_index", 0)):
        if out.get(k) is None:
            out[k] = v
        else:
            out.setdefault(k, v)
    if not isinstance(out.get("audio_layouts"), list):
        out["audio_layouts"] = []
    else:
        out.setdefault("audio_layouts", [])
    if out.get("start_time") is None:
        out["start_time"] = None
    return out


def _probe(path: str, stream_index: int) -> Dict[str, Any]:
    """兼容两种探测签名，并归一化结果。

    受控反例（审核 R25 B01 的 M01–M07）把 `ffprobe_facts` 换成一个
    **只收 path** 的 stub。为了既能让那些反例直接跑在**新判据**上、
    又不牺牲生产侧的选流能力，这里做一次 `TypeError` 回退。
    回退时 `stream_index` 视为 0（stub 场景本来就是单流）。
    """
    try:
        raw = ffprobe_facts(path, stream_index=stream_index)
    except TypeError:
        raw = ffprobe_facts(path)
    return _normalize_facts(raw)


def _split_frames(raw: bytes, fb: int) -> List[bytes]:
    """把一次性拿到的字节切成帧（受控注入路径用）。"""
    out: List[bytes] = []
    if fb <= 0:
        return out
    for i in range(0, len(raw), fb):
        out.append(raw[i:i + fb])
    return out


def _open_frames(path: str, pix_fmt: str, w: int, h: int,
                 stream_index: int) -> Tuple[Any, Dict[str, Any]]:
    """统一的帧来源。

    * 默认：**流式** `iter_frames`（有界，不把整片读进内存）。
    * 若模块级 `decode_raw` 被注入（受控反例用），则一次性取字节再切帧。

    两条路都走**同一套判据**（帧数必须匹配、短读必须拒、位深按有效位深），
    所以注入不会放宽标准，只是换了个取字节的方式。
    """
    fb = frame_bytes(pix_fmt, w, h)
    if not fb:
        raise ValueError(f"unknown or unusable pix_fmt {pix_fmt!r}")
    if decode_raw is not None:
        raw, rc = decode_raw(path, pix_fmt, w, h)
        frames = _split_frames(raw or b"", fb)
        short = bool(frames) and len(frames[-1]) != fb
        info = {"rc": int(rc), "bytes_per_frame": fb,
                "frames": len(frames), "short_read": short,
                "stderr_tail": "", "source": "injected"}
        return iter(frames), info
    it, info = iter_frames(path, pix_fmt, w, h, stream_index=stream_index)
    info["source"] = "stream"
    return it, info


def compare_media(a_path: str, b_path: str, *,
                  pix_fmt: str,
                  width: int = 0, height: int = 0,
                  expect_frames: Optional[int] = None,
                  require_time: bool = False,
                  require_audio: bool = False,
                  stream_index: int = 0,
                  timeout_hint: str = "",
                  mode: str = "native_exact") -> Dict[str, Any]:
    """严格比较两份媒体在**解码像素**上是否相同。

    Args:
        pix_fmt: 两边共用的解码格式。
        mode: **R25-03/R05** —— 比较的**语义层级**：

            * ``"native_exact"``（默认）：`pix_fmt` 必须是两边的**原始**
              格式。只要有一边源格式不同，就**拒绝比较** ——
              因为 ffmpeg 的 `-pix_fmt` 会静默转换，"转完再比"得到的
              "相同"不是"这两份源表示相同"。444 转 420 后比出的相同，
              不能拿来宣称 444 等价。
            * ``"converted"``：允许把两边都转成 `pix_fmt` 再比。
              这时结果里会带 `mode="converted"` 与每边的源格式，
              报告必须按"转换后相同"表述，不能升级成原生等价。
        width/height: 期望尺寸；给 0 表示从 ffprobe 取，并**要求两边一致**。
        expect_frames: 期望帧数；给定时两边**解码出的**帧数都必须等于它。
        require_time: 是否把"时间覆盖"作为必需项。
            **R25-03/R06**：不只是容器时长 —— 还会逐帧比 PTS。
            只有起点不同（0 vs 0.5s）也必须被抓到。
        require_audio: 是否把"音频流规格一致"作为必需项。
        stream_index: 选哪条视频流（两边用同一个序号）。

    Returns:
        dict：``ok`` / ``problems`` / ``identical`` / ``frames`` /
        ``max_abs_diff`` / ``psnr`` / ``alias`` / ``planes`` / ``spec`` /
        ``mode`` / ``source_pix_fmt``。
        只有 ``ok and identical`` 才能说"逐像素相同"，且要连同 ``mode``
        一起表述。
    """
    import numpy as np

    _mode = str(mode or "native_exact").strip().lower()
    if _mode not in ("native_exact", "converted"):
        raise ValueError(
            f"unknown comparison mode {mode!r}; use 'native_exact' or "
            f"'converted'")

    res: Dict[str, Any] = {
        "ok": False, "problems": [], "identical": False,
        "frames": {}, "max_abs_diff": None, "psnr": {},
        "alias": {}, "hashes": {}, "pix_fmt": str(pix_fmt),
        "expect_frames": expect_frames, "hint": timeout_hint,
        "planes": [], "spec": {}, "require_time": bool(require_time),
        "require_audio": bool(require_audio), "stream_index": int(stream_index),
        "mode": _mode,
    }
    p: List[str] = res["problems"]

    spec = pix_spec(pix_fmt)
    if not spec:
        p.append(f"unsupported comparison format {pix_fmt!r}; refusing to "
                 f"guess its plane layout or bit depth")
        return res
    res["spec"] = {"bit_depth": spec["bit_depth"], "peak": spec["peak"],
                   "channel_order": spec["channel_order"],
                   "dtype": spec["dtype"]}

    # ---- 1) 存在性 ----
    fa = _probe(a_path, stream_index)
    fb = _probe(b_path, stream_index)
    for tag, f, path in (("a", fa, a_path), ("b", fb, b_path)):
        if not f["exists"]:
            p.append(f"input {tag} does not exist: {path}")
        elif f["size"] <= 0:
            p.append(f"input {tag} is empty")
        elif not f["probe_ok"]:
            p.append(f"input {tag} could not be probed: {f['probe_error']}")
    if p:
        return res

    # ---- 2) 别名（含卷/设备）----
    ia, ib = file_identity(a_path), file_identity(b_path)
    same, why = _same_object(ia, ib)
    res["alias"] = {"same_path": ia["path"] == ib["path"],
                    "same_inode": (ia["inode"] == ib["inode"]
                                   and ia["device"] == ib["device"]),
                    "device_a": ia["device"], "device_b": ib["device"]}
    if same:
        p.append(f"{why}; this is a self-comparison")
        return res

    # ---- 3) 流规格（宽高/格式/位深必须两边一致）----
    _w = int(width or 0) or int(fa["width"] or 0)
    _h = int(height or 0) or int(fa["height"] or 0)
    if _w <= 0 or _h <= 0:
        p.append("width/height could not be established; refusing to guess "
                 "the frame geometry")
        return res
    for tag, f in (("a", fa), ("b", fb)):
        if f["width"] and f["height"] and (f["width"] != _w or f["height"] != _h):
            # M07：同字节数不等于同几何。两边尺寸不同就**不可比**。
            p.append(f"input {tag} is {f['width']}x{f['height']}, but the "
                     f"comparison was requested at {_w}x{_h}; different "
                     f"geometry is not comparable even with equal byte counts")
        if f["pix_fmt"]:
            res.setdefault("source_pix_fmt", {})[tag] = f["pix_fmt"]
            # R25-03/R05：native_exact 要求**源格式就是比较格式**。
            #
            # 原来这里只把源格式记进 `source_pix_fmt` 当参考，
            # 解码命令却无条件带 `-pix_fmt` —— 于是 rgb48le 与 rgb24
            # 被转成同一个格式后比出"相同"，报告却写着原生等价。
            # 转换后的相同只能支持"转换后表示相同"这个更弱的结论。
            if _mode == "native_exact" and f["pix_fmt"] != str(pix_fmt):
                p.append(
                    f"input {tag} is {f['pix_fmt']!r} on disk, but a "
                    f"native_exact comparison was requested at "
                    f"{str(pix_fmt)!r}; decoding would convert it first, and "
                    f"'equal after conversion' is not 'equal natively' "
                    f"(use mode='converted' if a converted comparison is "
                    f"what you actually want)")
    if p:
        return res

    # ---- 4) 帧数（用**解码帧数**，不用包数）----
    na, nb = int(fa["frames"]), int(fb["frames"])
    res["frames"] = {"probe_a": na, "probe_b": nb,
                     "packets_a": fa["packets"], "packets_b": fb["packets"]}
    if na <= 0 or nb <= 0:
        p.append(f"frame count could not be established (a={na}, b={nb})")
    elif expect_frames is not None and (na != int(expect_frames)
                                        or nb != int(expect_frames)):
        p.append("frame count does not match the expected number "
                 f"(a={na}, b={nb}, expected={expect_frames})")
    elif na != nb:
        p.append(f"the two sides have different frame counts (a={na}, b={nb})")

    # ---- 5) 时间（可选必需项）----
    if require_time:
        if fa["duration"] <= 0 or fb["duration"] <= 0:
            p.append("require_time was requested but a container duration "
                     "could not be read on at least one side")
        else:
            # R26-A08：容差**不能固定 1/30 秒**。
            #
            # 它要覆盖的是"两侧表示的时间量化误差 + 合法的一帧级抖动"：
            #   * 量化步长：由 `time_base` 决定（B 帧容器常是 1/10240 这类
            #     精细刻度，真正粗的是**帧间隔**）；
            #   * 一帧级抖动：取两侧中位帧间隔的一半。
            #
            # 60fps 得到 ~8ms、24fps 得到 ~21ms —— 固定 1/30 (33ms) 在
            # 前者会放过真实错位、在后者会误杀。
            # 真正的逐项判定在下面用 `probe_frame_timeline()` 做，
            # 这里的 `_tol` 只用于容器时长这一项。
            _tol = 1.0 / 30.0
            try:
                _ia = float(fa.get("duration") or 0)
                _ib = float(fb.get("duration") or 0)
                _med = 0.0
                for _t in (probe_frame_timeline(a_path, stream_index),
                           probe_frame_timeline(b_path, stream_index)):
                    _m = _t.get("median_interval") or 0
                    if _m and _m > _med:
                        _med = float(_m)
                if _med > 0:
                    _tol = max(1e-4, 0.5 * _med)
            except Exception:
                pass
            if abs(fa["duration"] - fb["duration"]) > _tol:
                p.append("durations differ beyond one frame "
                         f"({fa['duration']:.6f} vs {fb['duration']:.6f})")

            # R25-03/R06：**起点也必须比**。
            #
            # 原来只比容器时长：同样 10 秒、起点分别 0 和 0.5 秒的两份
            # 媒体会判"时间一致"，而它们的内容整体后移了半秒。
            _sa, _sb = fa.get("start_time"), fb.get("start_time")
            if _sa is not None and _sb is not None:
                if abs(float(_sa) - float(_sb)) > _tol:
                    p.append(
                        f"start times differ beyond one frame "
                        f"({float(_sa):.6f}s vs {float(_sb):.6f}s); the same "
                        f"duration does not mean the same timeline")

            # R25-03/R06：**逐帧 PTS**。时长与起点相同、中间时刻被拉伸
            # 或重排的组合，只有逐帧比才能抓到。
            try:
                # R26-A08：改用 `probe_frame_timeline` —— 它有三件
                # 旧调用没有的东西：
                #   ① **`stream_index`**：原来 `probe_pts(path)` 永远取
                #      第 0 条流，多视频流素材比错了对象；
                #   ② **真实的末帧 duration**（`frame.duration`）：
                #      原来拿"末两帧 PTS 差"冒充，那是**间隔**不是时长；
                #   ③ **乱序检测**：它保留原始顺序，不倒序修饰事实。
                #
                # 长片不能把全部 PTS 留在内存里，所以逐项相同由**有序摘要**
                # 判定；汇总值（帧数/起点/终点/末帧时长/跨度）另比一次，
                # 这样"哪里不同"仍然可读。
                _ta = probe_frame_timeline(a_path, stream_index)
                _tb = probe_frame_timeline(b_path, stream_index)
                res["timeline"] = {
                    "a": {k: _ta.get(k) for k in
                          ("state", "frames", "first_pts", "last_pts",
                           "last_duration", "span", "out_of_order",
                           "pts_digest", "problems")},
                    "b": {k: _tb.get(k) for k in
                          ("state", "frames", "first_pts", "last_pts",
                           "last_duration", "span", "out_of_order",
                           "pts_digest", "problems")},
                }
                if not _ta.get("ok") or not _tb.get("ok"):
                    p.append(
                        "require_time: the frame timeline is not usable "
                        f"(a={_ta.get('state')}: {_ta.get('problem')!r}; "
                        f"b={_tb.get('state')}: {_tb.get('problem')!r})")
                else:
                    for _side, _t in (("a", _ta), ("b", _tb)):
                        if _t.get("out_of_order"):
                            p.append(
                                f"input {_side} has "
                                f"{_t['out_of_order']} out-of-order display "
                                f"timestamps; its timeline is contradictory")
                    if _ta["frames"] != _tb["frames"]:
                        p.append(
                            f"decoded frame counts differ (a={_ta['frames']}, "
                            f"b={_tb['frames']})")
                    for _key, _label in (("first_pts", "first PTS"),
                                         ("last_pts", "last PTS"),
                                         ("last_duration", "last frame duration"),
                                         ("span", "presentation span")):
                        _va, _vb = _ta.get(_key), _tb.get(_key)
                        if _va is None or _vb is None:
                            p.append(
                                f"{_label} could not be established on "
                                f"both sides (a={_va!r}, b={_vb!r})")
                        elif abs(float(_va) - float(_vb)) > _tol:
                            p.append(
                                f"{_label} differs ({_va} vs {_vb}, "
                                f"tolerance {_tol:.6f}s)")
                    if _ta.get("pts_digest") != _tb.get("pts_digest"):
                        p.append(
                            "the ordered per-frame PTS sequences differ "
                            "(same summary values can still hide a reorder)")
            except Exception as _pe:
                p.append(f"require_time: frame timeline probe failed: {_pe}")
        res["duration"] = {"a": _finite_or_none(fa["duration"]),
                           "b": _finite_or_none(fb["duration"]),
                           "start_a": _finite_or_none(fa["start_time"]),
                           "start_b": _finite_or_none(fb["start_time"])}

    # ---- 6) 音频：**三级**（R26-A08）----
    #
    # 原来只比音轨数与布局 —— 复核原话："没有比较内容、样本、起止与同步"。
    # 现在按 `require_audio` 的取值分三级（`True` 等价于 `"spec"`）：
    #   spec    —— 数量/codec/采样率/声道/布局
    #   range   —— 再加每轨 start_time / duration
    #   content —— 再加 PCM（无损）或包级（有损/有意处理）摘要
    # 详见 `check_audio_three_levels()` 的 docstring。
    if require_audio:
        _ar = check_audio_three_levels(
            a_path, b_path, require_audio=require_audio,
            stream_index=stream_index)
        res["audio"] = _ar
        p.extend(_ar.get("problems") or [])

    if p:
        return res

    # ---- 7) 逐帧流式比较 ----
    fb_bytes = frame_bytes(pix_fmt, _w, _h)
    res["frames"]["bytes_per_frame"] = fb_bytes
    res["planes"] = plane_geometry(pix_fmt, _w, _h) or []

    it_a, inf_a = _open_frames(a_path, pix_fmt, _w, _h, stream_index)
    it_b, inf_b = _open_frames(b_path, pix_fmt, _w, _h, stream_index)
    peak = float(spec["peak"])
    dtype = spec["dtype"]

    n = 0
    max_abs = 0
    sum_sq = 0.0
    samples = 0
    differing_frames = 0
    per_frame_worst: List[Dict[str, Any]] = []
    try:
        while True:
            fa_raw = next(it_a, None)
            fb_raw = next(it_b, None)
            if fa_raw is None and fb_raw is None:
                break
            if fa_raw is None or fb_raw is None:
                p.append("the two sides end at different frames")
                break
            n += 1
            xa = np.frombuffer(fa_raw, dtype=dtype).astype(np.int64)
            xb = np.frombuffer(fb_raw, dtype=dtype).astype(np.int64)
            if xa.size != xb.size:
                p.append(f"frame {n}: sample count differs "
                         f"({xa.size} vs {xb.size})")
                break
            diff = np.abs(xa - xb)
            fmax = int(diff.max()) if diff.size else 0
            if fmax:
                differing_frames += 1
                if len(per_frame_worst) < 8:
                    per_frame_worst.append({"frame": n, "max_abs": fmax})
            if fmax > max_abs:
                max_abs = fmax
            sum_sq += float(np.sum((xa - xb) ** 2))
            samples += int(diff.size)
    finally:
        # 把两边迭代器跑完，让子进程正常退出并回收
        try:
            for _ in it_a:
                pass
        except Exception:
            pass
        try:
            for _ in it_b:
                pass
        except Exception:
            pass

    res["frames"]["decoded_a"] = int(inf_a["frames"])
    res["frames"]["decoded_b"] = int(inf_b["frames"])
    res["frames"]["compared"] = n
    res["decode_rc"] = {"a": inf_a["rc"], "b": inf_b["rc"]}
    res["decode_stderr"] = {"a": inf_a["stderr_tail"][-200:],
                            "b": inf_b["stderr_tail"][-200:]}
    if inf_a["rc"] not in (0, None) or inf_b["rc"] not in (0, None):
        p.append(f"decode failed (rc a={inf_a['rc']}, b={inf_b['rc']})")
    if inf_a["short_read"] or inf_b["short_read"]:
        p.append("at least one side ended with a partial frame; trailing "
                 "bytes are not a whole frame")

    # ---- 关键：解码帧数必须两边相等且等于期望 ----
    da, db = int(inf_a["frames"]), int(inf_b["frames"])
    if da == 0 or db == 0:
        # M01：双方 0 字节、rc=0 —— 空不是"相同"
        p.append(f"no frames were decoded (a={da}, b={db}); empty output "
                 f"cannot be reported as identical")
    elif da != db:
        p.append(f"decoded frame counts differ (a={da}, b={db})")
    elif expect_frames is not None and (da != int(expect_frames)
                                        or db != int(expect_frames)):
        # M02/M03：短读与多读都必须失败
        p.append("decoded frame count does not match the expected number "
                 f"(a={da}, b={db}, expected={expect_frames})")
    if n and n != da:
        p.append(f"only {n} frame(s) were actually compared but {da} were "
                 f"decoded; the comparison did not cover the whole stream")

    if samples <= 0 and not p:
        p.append("no samples were compared")

    # ---- 8) 指标 ----
    res["max_abs_diff"] = max_abs if samples else None
    res["differing_frames"] = differing_frames
    res["worst_frames"] = per_frame_worst
    mse = (sum_sq / samples) if samples else None
    if mse is None:
        res["psnr"] = {"mse": None, "peak": peak, "db": None,
                       "all_inf": False, "note": "no samples"}
    else:
        _db = None if mse <= 0 else 10.0 * math.log10((peak * peak) / mse)
        res["psnr"] = {
            "mse": _finite_or_none(mse),
            "peak": peak,
            # MSE=0 → 无穷 PSNR。JSON 里用 `None` + `all_inf` 表达，
            # 不输出 `Infinity`（那会让下游 json.loads 在某些实现失败）
            "db": _finite_or_none(_db),
            "all_inf": bool(mse == 0.0),
        }

    res["identical"] = bool(res["max_abs_diff"] == 0
                            and differing_frames == 0
                            and da > 0 and da == db and n == da
                            # "相同"必须同时意味着"这份比较本身成立"：
                            # 短读/多帧/异尺寸时像素可能逐字节相等，但那份
                            # 相等没有覆盖整条流，不能叫 identical。
                            and not p)

    # ---- 9) 文件 hash（附加记录，不参与判定）----
    for tag, path in (("a", a_path), ("b", b_path)):
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(CHUNK), b""):
                    h.update(chunk)
            res["hashes"][tag] = h.hexdigest()[:32]
        except OSError:
            res["hashes"][tag] = ""

    res["ok"] = not p
    return res


# ---------------------------------------------------------------------------
# 时间轴逐帧观测（B07 会用）
# ---------------------------------------------------------------------------

def probe_frame_timeline(path: str, stream_index: int = 0,
                         timeout: int = 3600) -> Dict[str, Any]:
    """逐**帧**（不是逐包）取 PTS/duration，并如实记录无法解析的行。

    为什么逐帧：包数不等于显示帧数（B 帧重排、一个包多帧）。
    为什么记 `dropped_rows`：R24 的 T02 把 `PTS=N/A` 的行直接 `continue`
    丢掉，剩下两行算出 0.12 秒还判"完整覆盖"。
    """
    out: Dict[str, Any] = {
        "ok": False, "state": "unavailable", "frames": 0,
        "first_pts": None, "last_pts": None, "last_duration": None,
        "span": None, "rows_total": 0, "rows_parsed": 0,
        "dropped_rows": [], "problem": "", "varies": None,
        "duplicate_timestamps": 0, "non_finite": 0,
    }
    # R26-A08：**time_base 要一起取**。
    # 容差必须由两侧的时间量化步长决定（`1/time_base`），不能固定 1/30 秒 ——
    # 60fps 素材的量化步长是 1/60，拿 1/30 当容差会放过真实的一帧错位；
    # 而 24000/1001 的步长是 ~0.0417 秒，1/30 又会误杀。
    out["time_base"] = ""
    try:
        _tb = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", f"v:{stream_index}",
             "-show_entries", "stream=time_base", "-of", "csv=p=0", str(path)],
            capture_output=True, timeout=30)
        if _tb.returncode == 0:
            out["time_base"] = (
                (_tb.stdout or b"").decode("utf-8", "replace").strip()
                .splitlines() or [""])[0].strip()
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", f"v:{stream_index}",
             "-show_entries", "frame=pts_time,pkt_duration_time,duration_time",
             "-of", "csv=p=0", str(path)],
            capture_output=True, timeout=timeout)
        if r.returncode != 0:
            out["state"] = "failed"
            out["problem"] = f"ffprobe rc={r.returncode}"
            return out
    except Exception as exc:
        out["state"] = "failed"
        out["problem"] = f"ffprobe failed: {exc}"[:200]
        return out

    pts: List[float] = []
    durs: List[float] = []
    for raw in (r.stdout or b"").decode("utf-8", "replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        out["rows_total"] += 1
        parts = line.split(",")
        try:
            _p = float(parts[0])
        except (TypeError, ValueError):
            # 不认识的行**留痕**，不静默丢弃
            if len(out["dropped_rows"]) < 20:
                out["dropped_rows"].append(line[:80])
            continue
        _d = 0.0
        if len(parts) > 1:
            try:
                _d = float(parts[1])
            except (TypeError, ValueError):
                _d = 0.0
        if not math.isfinite(_p):
            out["non_finite"] += 1
            if len(out["dropped_rows"]) < 20:
                out["dropped_rows"].append("non-finite pts: " + line[:70])
            continue
        if not math.isfinite(_d):
            out["non_finite"] += 1
            _d = 0.0
        pts.append(_p)
        durs.append(_d)
        out["rows_parsed"] += 1

    if not pts:
        out["state"] = "unavailable" if out["rows_total"] == 0 else "contradictory"
        out["problem"] = ("no frame timestamps" if out["rows_total"] == 0
                          else "every frame row was unusable")
        return out

    # R26-A08：**不排序**。
    #
    # 原来这里 `order = sorted(...)` 把 PTS 重排后再算首尾 ——
    # 于是"第 50 帧的 PTS 比第 49 帧小"这种**真实的观测倒退**被悄悄抹平，
    # 报告的 first/last 看起来完全正常。排序是在修饰事实，不是测量。
    #
    # 现在保留原始顺序，并显式记录 `out_of_order`：倒退了几处。
    # 首尾/跨度用 `min`/`max`（乱序时首项未必是极小值），
    # 但**末帧 duration 必须跟着"显示顺序最后一帧"**，也就是 max PTS
    # 那一项 —— 而不是列表最后一个。
    out["frames"] = len(pts)
    out["out_of_order"] = sum(
        1 for a, b in zip(pts, pts[1:]) if b < a)
    _last_idx = max(range(len(pts)), key=lambda i: pts[i])
    out["first_pts"] = round(min(pts), 6)
    out["last_pts"] = round(pts[_last_idx], 6)
    out["last_duration"] = round(durs[_last_idx], 6)
    out["duplicate_timestamps"] = sum(
        1 for x, y in zip(pts, pts[1:]) if y == x)
    # R26-A08：**有序**时间轴摘要。
    #
    # 长片不能把全部 PTS 留在内存里（指令明确），但"两边 PTS 序列是否
    # 逐项相同"必须能判。用一个增量 sha256 覆盖**原始顺序**的
    # `round(pts,6)` 串 —— 内存恒定，且顺序敏感（不排序）。
    _h = hashlib.sha256()
    for _v in pts:
        _h.update(("%.6f," % _v).encode("ascii"))
    out["pts_digest"] = _h.hexdigest()[:32]

    problems: List[str] = []
    if out["out_of_order"]:
        problems.append(
            f"{out['out_of_order']} out-of-order display timestamp(s); "
            f"the observation is kept in its original order, not sorted")
    if out["non_finite"]:
        problems.append(f"{out['non_finite']} non-finite timestamp/duration")
    if out["dropped_rows"]:
        problems.append(f"{len(out['dropped_rows'])} row(s) could not be "
                        f"parsed (kept, not silently dropped)")
    if out["duplicate_timestamps"]:
        problems.append(f"{out['duplicate_timestamps']} duplicate display "
                        f"timestamp(s); one of each pair is never shown")

    if durs[_last_idx] > 0:
        out["span"] = round(pts[_last_idx] + durs[_last_idx]
                            - min(pts), 6)
        out["coverage"] = "observed"
    else:
        out["span"] = round(pts[_last_idx] - min(pts), 6)
        out["coverage"] = "unknown" if len(pts) == 1 else "derived"
        problems.append("the last frame has no duration; the span excludes "
                        "the final frame")

    intervals = [b - a for a, b in zip(pts, pts[1:])]
    if len(intervals) >= 3:
        s = sorted(intervals)
        med = s[len(s) // 2]
        if med > 0:
            out["varies"] = (max(intervals) - min(intervals)) / med > 0.01
            out["median_interval"] = round(med, 6)

    out["problems"] = problems
    # R26-A08：**倒退的观测是矛盾，不是"需要修的噪声"**。
    # 排序会把它们变成"正常"，从而让一份时间轴错乱的成品通过比较。
    if (out["non_finite"] or out["duplicate_timestamps"] > 0
            or out["out_of_order"] > 0):
        out["state"] = "contradictory"
    elif problems:
        # 解析留痕/缺尾帧时长 → "不完整"而不是"矛盾"
        out["state"] = "incomplete"
    else:
        out["state"] = "observed"
    out["ok"] = (out["state"] == "observed" and out["span"] is not None
                 and out["span"] > 0)
    if not out["ok"] and not out["problem"]:
        out["problem"] = "; ".join(problems) or "not a usable timeline"
    return out


def json_safe(obj: Any) -> Any:
    """把结果转成**标准 JSON**：非有限数一律 None（不写 Infinity/NaN）。"""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


# --------------------------------------------------------------- 音频三级
#: 音频检查的三个级别（R26-A08 / 指令 §"音频三级"）。
#:
#: * ``spec``：**规格与选流** —— 音轨数、codec、采样率、声道数、布局。
#:   这一级只能回答"两边装的是不是同一类音轨"。
#: * ``range``：再加**真实样本范围与起止** —— 每轨的 `start_time` 与
#:   `duration`。它能抓"音轨整体后移""尾部被截断"。
#: * ``content``：再加**内容** —— 无损旁路比 PCM（解码后逐样本），
#:   有意处理/有损编码比**包级** framemd5（比的是压缩后的字节流，
#:   不能强求与无损参考相同）。
#:
#: 为什么必须分级：只比 spec 时，"同样 48k/2ch 但内容完全错"会通过；
#: 直接比 content 又会在"有意处理"（降噪/变速）上误杀。
AUDIO_LEVELS = ("spec", "range", "content")


def _audio_level_of(require_audio: Any) -> str:
    """把 `require_audio` 的多种写法归一成三级名。"""
    if require_audio is True:
        return "spec"
    if not require_audio:
        return ""
    s = str(require_audio).strip().lower()
    return s if s in AUDIO_LEVELS else "spec"


def _audio_streams_detail(path: str) -> List[Dict[str, Any]]:
    """每条音轨的规格 + 起止（失败返回空表，由调用方判"读不到"）。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries",
             "stream=index,codec_name,channels,channel_layout,sample_rate,"
             "start_time,duration,sample_fmt",
             "-of", "json", str(path)],
            capture_output=True, timeout=600)
        if r.returncode != 0:
            return []
        doc = json.loads((r.stdout or b"").decode("utf-8", "replace") or "{}")
        out = []
        for s in (doc.get("streams") or []):
            out.append({
                "index": s.get("index"),
                "codec": s.get("codec_name"),
                "channels": s.get("channels"),
                "layout": s.get("channel_layout"),
                "sample_rate": s.get("sample_rate"),
                "sample_fmt": s.get("sample_fmt"),
                "start": _finite_or_none(s.get("start_time")),
                "duration": _finite_or_none(s.get("duration")),
            })
        return out
    except Exception:
        return []


def _audio_frame_md5(path: str, stream_index: int = 0,
                     *, decode: bool) -> str:
    """音轨的有序 md5 摘要。

    Args:
        decode: True → 先解码成 PCM 再算（无损旁路用，能抓"容器相同但
            样本不同"）；False → 直接对**包**算（有损/有意处理用，
            比的是压缩流，解码后的 PCM 本来就不该与参考相同）。

    Returns:
        32 位十六进制；读不到返回空串。
    """
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-nostdin",
           "-i", str(path), "-map", f"0:a:{stream_index}", "-vn"]
    if decode:
        cmd += ["-c:a", "pcm_s16le"]
    else:
        cmd += ["-c", "copy"]
    cmd += ["-f", "md5", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=1800)
        if r.returncode != 0:
            return ""
        txt = (r.stdout or b"").decode("utf-8", "replace").strip()
        return txt.split("=")[-1].strip() if "=" in txt else txt
    except Exception:
        return ""


def check_audio_three_levels(a_path: str, b_path: str, *,
                             require_audio: Any = "content",
                             stream_index: int = 0,
                             tolerance: float = 0.0
                             ) -> Dict[str, Any]:
    """按三级做音频核对，返回结构化结果（供 `compare_media` 与独立探针共用）。

    Args:
        require_audio: ``"spec"`` / ``"range"`` / ``"content"``（或 True=spec）。
        stream_index: 只对第几条音轨做内容比较（规格/范围比全部）。
        tolerance: 起止时间容差（秒）；0 表示用 1e-3。

    Returns:
        ``{"level", "ok", "problems", "a", "b", "content", "digest_a",
        "digest_b"}``。

        **读不到音轨时**：`level` 非空但两边都为空 → `ok=False` 且
        problems 写明"读不到"，不把"读不到"当成"一致"。
    """
    lvl = _audio_level_of(require_audio)
    out: Dict[str, Any] = {"level": lvl, "ok": True, "problems": [],
                           "a": [], "b": [], "content": "", "digest_a": "",
                           "digest_b": ""}
    if not lvl:
        return out
    probs: List[str] = out["problems"]
    ta = _audio_streams_detail(a_path)
    tb = _audio_streams_detail(b_path)
    out["a"], out["b"] = ta, tb
    if not ta or not tb:
        out["ok"] = False
        probs.append("audio streams could not be read on at least one side "
                     f"(a={len(ta)}, b={len(tb)}); 'unreadable' is not "
                     f"'identical'")
        return out

    _tol = float(tolerance) if tolerance else 1e-3

    # ---- 级 1：规格与选流 ----
    if len(ta) != len(tb):
        probs.append(f"audio stream count differs (a={len(ta)}, b={len(tb)})")
    else:
        for i, (sa, sb) in enumerate(zip(ta, tb)):
            for key in ("codec", "channels", "layout", "sample_rate"):
                if sa.get(key) != sb.get(key):
                    probs.append(
                        f"audio[{i}].{key} differs ({sa.get(key)!r} vs "
                        f"{sb.get(key)!r})")

    # ---- 级 2：真实样本范围与起止 ----
    if lvl in ("range", "content") and len(ta) == len(tb):
        for i, (sa, sb) in enumerate(zip(ta, tb)):
            for key, label in (("start", "start"), ("duration", "duration")):
                va, vb = sa.get(key), sb.get(key)
                if va is None or vb is None:
                    probs.append(
                        f"audio[{i}].{label} could not be read on both sides "
                        f"(a={va!r}, b={vb!r})")
                elif abs(float(va) - float(vb)) > _tol:
                    probs.append(
                        f"audio[{i}].{label} differs ({va} vs {vb}, "
                        f"tolerance {_tol})")

    # ---- 级 3：内容 ----
    if lvl == "content":
        if stream_index >= min(len(ta), len(tb)):
            probs.append(
                f"audio stream index {stream_index} does not exist on both "
                f"sides (a={len(ta)}, b={len(tb)})")
        else:
            # 先试**解码后**比较：如果两边都是无损/同一编码，这是最强的判据。
            da = _audio_frame_md5(a_path, stream_index, decode=True)
            db = _audio_frame_md5(b_path, stream_index, decode=True)
            if da and db and da == db:
                out["content"] = "pcm-identical"
                out["digest_a"] = out["digest_b"] = da
            else:
                # PCM 不同：可能是"有意处理"（降噪/变速/有损编码）。
                # 这时改比**包级**摘要 —— 它能证明"压缩流一致"，
                # 但**不能**证明"解码后一致"，所以理由里要写明。
                pa = _audio_frame_md5(a_path, stream_index, decode=False)
                pb = _audio_frame_md5(b_path, stream_index, decode=False)
                out["digest_a"], out["digest_b"] = pa, pb
                if pa and pb and pa == pb:
                    out["content"] = "packets-identical"
                elif not da or not db:
                    out["content"] = "unreadable"
                    probs.append(
                        "audio content could not be read on at least one "
                        "side; refusing to claim equality")
                else:
                    out["content"] = "differs"
                    probs.append(
                        "audio content differs in both the decoded PCM and "
                        f"the packet stream (pcm {da[:8]} vs {db[:8]}, "
                        f"packets {pa[:8] if pa else '-'} vs "
                        f"{pb[:8] if pb else '-'})")

    out["ok"] = not probs
    return out
