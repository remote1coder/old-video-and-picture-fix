# -*- coding: utf-8 -*-
"""照片链的**常驻模型 worker**：加载一次，反复处理。

## 为什么单独一个进程

指令 §1.4："保留不同 Python 运行时的常驻模型 worker；控制面统一不等于
所有 torch/VS/G'MIC 必须塞进同一解释器。"

事实：`RealESRGANer` / `GFPGANer` 只能在带 torch 的解释器里活着
（本项目是 `.venv_r4`，torch 2.11+cu128）。主流程用的是另一个解释器，
硬塞会把两条链的依赖绑死。

更要紧的是**不要每张图重新加载模型**：`RealESRGAN_x4plus` 加载约 2–3 秒，
一次跑批几十张就是几分钟的纯等待。

## 协议

stdin / stdout 各一行一个 JSON（**UTF-8，不留日志到 stdout**）：

请求::

    {"op": "upscale", "png_b64": "...", "model": "RealESRGAN_x4plus",
     "scale": 2, "half": true}
    {"op": "face", "png_b64": "...", "backend": "GFPGAN",
     "strength": 0.5, "upscale": 2}
    {"op": "ping"}
    {"op": "shutdown"}

响应::

    {"ok": true, "png_b64": "...", "width": 0, "height": 0,
     "model": "...", "elapsed": 0.0}
    {"ok": false, "error": "..."}

**所有日志走 stderr** —— stdout 只放协议行，混一行日志就会让调用方解析失败。

## 直接运行

本文件用 `--self-test` 可以只验证"模型能不能加载"，不处理任何图::

    .venv_r4/Scripts/python.exe shared_modules/photo_model_worker.py --self-test
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from typing import Any, Dict, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

#: 模型名 → (arch 构造, 原生倍数)
_MODEL_SPECS = {
    "realesr-general-x4v3": ("srvgg", 4),
    "realesr-general-wdn-x4v3": ("srvgg", 4),
    "RealESRGAN_x2plus": ("rrdb", 2),
    "RealESRGAN_x4plus": ("rrdb", 4),
}
_DEFAULT_MODEL_FILE = {
    "realesr-general-x4v3": "realesr-general-x4v3.pth",
    "realesr-general-wdn-x4v3": "realesr-general-wdn-x4v3.pth",
    "RealESRGAN_x2plus": "RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus": "RealESRGAN_x4plus.pth",
}


def _log(msg: str) -> None:
    """日志**只走 stderr** —— stdout 是协议通道。"""
    sys.stderr.write("[photo-worker] %s\n" % msg)
    sys.stderr.flush()


def _project_models_dir() -> str:
    return os.path.join(os.path.dirname(_HERE), "models")


# ------------------------------------------------------------------ 模型加载
_CACHE: Dict[str, Any] = {}


def _build_arch(kind: str):
    if kind == "srvgg":
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_conv=32, upscale=4, act_type="prelu")
    from basicsr.archs.rrdbnet_arch import RRDBNet
    return None                      # 由调用方按倍数构造


def load_upsampler(model: str, *, half: bool = True,
                   gpu_id: int = 0, model_dir: str = "") -> Any:
    """加载（或复用）超分模型。**同名模型只加载一次**。"""
    key = "up:%s:%s:%s" % (model, half, gpu_id)
    if key in _CACHE:
        return _CACHE[key]
    if model not in _MODEL_SPECS:
        raise ValueError("unknown upscale model %r (known: %s)"
                         % (model, sorted(_MODEL_SPECS)))
    kind, native = _MODEL_SPECS[model]
    md = model_dir or _project_models_dir()
    path = os.path.join(md, _DEFAULT_MODEL_FILE[model])
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    from realesrgan import RealESRGANer
    if kind == "srvgg":
        arch = _build_arch("srvgg")
    else:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                       num_block=23, num_grow_ch=32, scale=native)
    # **`scale=` 必须是模型原生倍数**，不是目标倍数 —— 传目标会让
    # `enhance()` 输出原生尺寸（面积差 4 倍），帧数/尺寸全错。
    up = RealESRGANer(scale=native, model_path=path, model=arch,
                      half=bool(half), gpu_id=int(gpu_id))
    _CACHE[key] = up
    _log("loaded upsampler %s (native x%d) from %s" % (model, native, path))
    return up


def load_gfpgan(*, upscale: int = 2, gpu_id: int = 0,
                model_dir: str = "") -> Any:
    """加载（或复用）GFPGAN。"""
    key = "gfpgan:%s:%s" % (upscale, gpu_id)
    if key in _CACHE:
        return _CACHE[key]
    md = model_dir or _project_models_dir()
    # **worker 与身份模块必须用同一个文件名**（GFPGANv1.4.pth）。
    path = os.path.join(md, "GFPGANv1.4.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    from gfpgan import GFPGANer
    g = GFPGANer(model_path=path, upscale=int(upscale), arch="clean",
                 channel_multiplier=2, bg_upsampler=None, device=None)
    _CACHE[key] = g
    _log("loaded GFPGAN from %s (upscale=%d)" % (path, upscale))
    return g


# ------------------------------------------------------------------ 两个操作
def op_upscale(req: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    import cv2
    from media_io import read_png_bytes, encode_png

    png = base64.b64decode(req["png_b64"])
    arr = read_png_bytes(png)                    # uint16 RGB
    if arr.dtype != np.uint16:
        raise ValueError("expects RGB16 PNG, got %s" % arr.dtype)
    model = str(req.get("model") or "realesr-general-x4v3")
    target = int(req.get("scale") or 2)
    up = load_upsampler(model, half=bool(req.get("half", True)),
                        gpu_id=int(req.get("gpu_id", 0)),
                        model_dir=str(req.get("model_dir") or ""))

    # RealESRGANer 吃 BGR、8bit 或 float。16bit 用 rgb→bgr + 归一化到 0..1
    bgr = cv2.cvtColor(arr[:, :, ::-1].copy(), cv2.COLOR_RGB2BGR)
    t0 = time.monotonic()
    out_bgr, _ = up.enhance(bgr, outscale=target)
    dt = round(time.monotonic() - t0, 3)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    # enhance() 返回 uint8 或 float；统一回到 uint16 域
    if out_rgb.dtype == np.uint8:
        out16 = (out_rgb.astype(np.uint16) * 257)
    else:
        out16 = np.clip(np.rint(out_rgb * 65535.0), 0, 65535).astype(np.uint16)
    return {"ok": True, "png_b64": base64.b64encode(
                encode_png(out16)).decode("ascii"),
            "width": int(out16.shape[1]), "height": int(out16.shape[0]),
            "model": model, "scale": target, "elapsed": dt}


def op_face(req: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    import cv2
    from media_io import read_png_bytes, encode_png

    png = base64.b64decode(req["png_b64"])
    arr = read_png_bytes(png)
    if arr.dtype != np.uint16:
        raise ValueError("expects RGB16 PNG, got %s" % arr.dtype)
    g = load_gfpgan(upscale=int(req.get("upscale") or 2),
                    gpu_id=int(req.get("gpu_id", 0)),
                    model_dir=str(req.get("model_dir") or ""))
    # GFPGANer 吃 BGR uint8（模型输入本来就是 8bit）。
    # **只在这一跳降位深**；ROI 之外的高精度背景由 S4 的混合负责保住。
    bgr8 = cv2.cvtColor(
        (arr[:, :, ::-1] // 257).astype(np.uint8), cv2.COLOR_RGB2BGR)
    t0 = time.monotonic()
    _cropped, _restored, restored_img = g.enhance(
        bgr8, has_aligned=False,
        only_center_face=bool(req.get("only_center_face", False)),
        paste_back=True)
    dt = round(time.monotonic() - t0, 3)
    if restored_img is None:
        return {"ok": False,
                "error": "GFPGAN returned no image (no face detected?)",
                "no_face": True}
    out_rgb = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
    out16 = out_rgb.astype(np.uint16) * 257
    return {"ok": True, "png_b64": base64.b64encode(
                encode_png(out16)).decode("ascii"),
            "width": int(out16.shape[1]), "height": int(out16.shape[0]),
            "backend": "GFPGAN", "elapsed": dt}


_OPS = {"upscale": op_upscale, "face": op_face}


# ------------------------------------------------------------------ 主循环
def serve(stdin=None, stdout=None) -> int:
    sin = stdin or sys.stdin
    sout = stdout or sys.stdout
    _log("ready (pid=%d)" % os.getpid())
    for line in sin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            sout.write(json.dumps({"ok": False,
                                   "error": "bad json: %s" % exc}) + "\n")
            sout.flush()
            continue
        op = str(req.get("op") or "")
        if op == "shutdown":
            sout.write(json.dumps({"ok": True, "bye": True}) + "\n")
            sout.flush()
            return 0
        if op == "ping":
            sout.write(json.dumps({"ok": True, "loaded": sorted(_CACHE)}) + "\n")
            sout.flush()
            continue
        fn = _OPS.get(op)
        if fn is None:
            sout.write(json.dumps({"ok": False,
                                   "error": "unknown op %r" % op}) + "\n")
            sout.flush()
            continue
        try:
            resp = fn(req)
        except Exception as exc:
            resp = {"ok": False,
                    "error": "%s: %s" % (type(exc).__name__, exc)}
        sout.write(json.dumps(resp) + "\n")
        sout.flush()
    return 0


def _self_test(model: str = "realesr-general-x4v3") -> int:
    """只验证模型能加载，不处理任何图。"""
    try:
        up = load_upsampler(model, half=False)
        _ = up
        g = load_gfpgan(upscale=2)
        _ = g
        print(json.dumps({"ok": True, "loaded": sorted(_CACHE)}))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False,
                          "error": "%s: %s" % (type(exc).__name__, exc)}))
        return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model", default="realesr-general-x4v3")
    a = ap.parse_args()
    if a.self_test:
        return _self_test(a.model)
    return serve()


if __name__ == "__main__":
    # ==== **必须最先做**：把真正的 stdout 留成协议通道，fd 1 指向 stderr ====
    #
    # 踩过：`realesrgan` 的库代码会往 stdout 打印
    # `\tInput is a 16-bit image` —— 调用方按 JSON 行解析时立刻
    # `JSONDecodeError: Expecting value: line 1 column 2`。
    #
    # 做法与 `stream_ai_template` 一致：dup 一份原始 fd 1 当数据通道，
    # 再把 fd 1 重定向到 fd 2（stderr）。这样任何第三方库的 print
    # 都进 stderr，协议行不受影响。
    import os as _os
    _REAL_OUT = _os.fdopen(_os.dup(1), "w", buffering=1, encoding="utf-8",
                           newline="\n")
    _os.dup2(2, 1)
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # 现在是 stderr 的副本
    except Exception:
        pass
    if "--self-test" in sys.argv:
        # self-test 用 stderr 输出 JSON 也行；但为了可读，仍走真 stdout
        import json as _j
        try:
            _rc = _self_test()
        except Exception as _e:
            _REAL_OUT.write(_j.dumps({"ok": False, "error": str(_e)}) + "\n")
            _rc = 1
        _REAL_OUT.flush()
        sys.exit(_rc)
    sys.exit(serve(stdout=_REAL_OUT))
