# -*- coding: utf-8 -*-
"""G'MIC 的**类型化二进制端点**：stdin(PNG) → 滤镜 → stdout(PNG)。

## 为什么需要它（R26-A11 事实更正）

上一轮把"G'MIC CLI 只吃文件"当成了照片链不能全链流式的理由。复核纠正：
G'MIC 官方支持 `-.ext` 类型化 stdin / `-o -.ext` stdout（带指定位深）。
本机 4.0.6 实测（11×7 uint16 灰度）：

    输入 113 字节 -> rc=0, 输出 113 字节, 魔数正确, 逐值相等 max_abs=0
    坏输入      -> rc=-1, stdout 0 字节

所以"照片链必须在 S1/S5 各落一份 TIFF"这个结论不成立：
**只在这一个端点物化是可行的，而且可以完全不落盘**。

## 三重校验（缺一个就会被"看起来成功"骗过）

1. **rc == 0** —— 坏输入必须靠 rc 判，不能因为 stdout 有字节就继续；
   （复核记录的另一个版本在坏输入时 stdout 仍有 1036 字节帮助文本。）
2. **魔数**是 `\\x89PNG\\r\\n\\x1a\\n` —— 防止把日志/帮助当图像。
3. **能解码**，且尺寸/通道/位深与调用方声明的一致 ——
   防止"滤镜静默改了分辨率/位深"。

## 精度

PNG16 适合**整型像素**端点。浮点工作域不能未经批准量化成 PNG16；
需要浮点时用 G'MIC 的原生表示（`.cimg`）并单独验证精度 ——
本模块只提供整型 PNG 通道，并在返回值里写清 `bit_depth`，
让调用方自己判断"这一跳有没有降精度"。
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import Any, Dict, List, Optional

__all__ = ["find_gmic", "gmic_filter_png"]

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: 候选版本，**新的在前**（4.0.6 是本机实测过的那个）。
_GMIC_CANDIDATES = (
    "tools_r4/gmic/gmic-4.0.6-cli-win64/gmic.exe",
    "tools_r4/gmic/gmic-4.0.5-cli-win64/gmic.exe",
)


def find_gmic(explicit: str = "") -> str:
    """找可用的 gmic 可执行文件。返回空串表示找不到。"""
    if explicit:
        return explicit if os.path.isfile(explicit) else ""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)                    # shared_modules/ 的上级
    for rel in _GMIC_CANDIDATES:
        p = os.path.join(root, rel.replace("/", os.sep))
        if os.path.isfile(p):
            return p
    # PATH 里
    import shutil
    w = shutil.which("gmic")
    return w or ""


def gmic_filter_png(data: bytes, args: Optional[List[str]] = None, *,
                    gmic: str = "", timeout: float = 300.0,
                    expect_width: int = 0, expect_height: int = 0,
                    expect_channels: int = 0,
                    expect_bit_depth: int = 0) -> Dict[str, Any]:
    """把 PNG 字节喂给 G'MIC，取回 PNG 字节。

    Args:
        data: 输入 PNG（**必须**是整型位深；浮点请用原生表示）。
        args: 滤镜链（如 ``["-blur", "2"]``）。空表示 identity。
        gmic: 显式可执行路径；空则自动探测。
        timeout: 秒。
        expect_*: 声明期望的几何/位深。给 0 表示不核（只记录实际值）。

    Returns:
        ``{"ok", "png", "width", "height", "channels", "bit_depth",
        "error", "stderr_tail", "elapsed", "rc"}``。

        ``ok`` 只有在 **rc==0 且魔数正确且能解码且几何/位深符合声明** 时为真。
    """
    out: Dict[str, Any] = {
        "ok": False, "png": b"", "width": 0, "height": 0, "channels": 0,
        "bit_depth": 0, "error": "", "stderr_tail": "", "elapsed": 0.0,
        "rc": None,
    }
    exe = find_gmic(gmic)
    if not exe:
        out["error"] = ("gmic executable not found; looked in "
                        "tools_r4/gmic/*/gmic.exe and PATH")
        return out
    if not isinstance(data, (bytes, bytearray)) or not data:
        out["error"] = "empty or non-bytes input"
        return out

    cmd = [exe, "-.png"] + [str(a) for a in (args or [])] + ["-o", "-.png"]
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, input=bytes(data), capture_output=True,
                           timeout=timeout, shell=False)
    except subprocess.TimeoutExpired:
        out["error"] = "gmic timed out after %.0fs" % timeout
        out["elapsed"] = round(time.monotonic() - t0, 3)
        return out
    except OSError as exc:
        out["error"] = "cannot run gmic: %s" % exc
        out["elapsed"] = round(time.monotonic() - t0, 3)
        return out
    out["elapsed"] = round(time.monotonic() - t0, 3)
    out["rc"] = p.returncode
    out["stderr_tail"] = (p.stderr or b"")[-600:].decode("utf-8", "replace")

    # ---- ① rc ----
    if p.returncode != 0:
        # **不能**看 stdout 有没有字节：坏输入时某些版本会吐帮助文本。
        out["error"] = ("gmic exited %s (stdout had %d bytes, ignored)"
                        % (p.returncode, len(p.stdout or b"")))
        return out

    blob = p.stdout or b""
    # ---- ② 魔数 ----
    if blob[:8] != _PNG_MAGIC:
        out["error"] = ("rc=0 but stdout is not a PNG (starts with %r); "
                        "refusing to treat it as an image"
                        % (blob[:16], ))
        return out

    # ---- ③ 能解码 + 几何/位深 ----
    try:
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from media_io import read_png_bytes
        arr = read_png_bytes(blob)
    except Exception as exc:
        out["error"] = "gmic returned a PNG that cannot be decoded: %s" % exc
        return out

    h, w = int(arr.shape[0]), int(arr.shape[1])
    channels = 1 if arr.ndim == 2 else int(arr.shape[2])
    # 位深看 dtype（read_png_bytes 由 IHDR 决定，不经 PIL）
    bit_depth = 16 if arr.dtype.name.startswith("uint16") else 8
    out.update({"png": blob, "width": w, "height": h,
                "channels": channels, "bit_depth": bit_depth})

    problems = []
    if expect_width and w != int(expect_width):
        problems.append("width %d != declared %d" % (w, expect_width))
    if expect_height and h != int(expect_height):
        problems.append("height %d != declared %d" % (h, expect_height))
    if expect_channels and channels != int(expect_channels):
        problems.append("channels %d != declared %d"
                        % (channels, expect_channels))
    if expect_bit_depth and bit_depth != int(expect_bit_depth):
        problems.append("bit depth %d != declared %d"
                        % (bit_depth, expect_bit_depth))
    if problems:
        out["error"] = ("gmic changed the representation: "
                        + "; ".join(problems))
        return out

    out["ok"] = True
    return out
