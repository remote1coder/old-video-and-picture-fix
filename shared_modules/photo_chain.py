# -*- coding: utf-8 -*-
"""照片链的**内存编排**：把已具备内存入口的 stage 串起来。

## 现状（诚实版）

| stage | 内存入口 | 说明 |
|---|---|---|
| S1 preprocess | ✅ | 三条 G'MIC 链走 `gmic_stdio` |
| S2 ai_pre | ✅ | 通道校正 |
| S3 ai_upscale | ❌ | **需要模型常驻 worker**（超分模型要在进程里活着）|
| S4 ai_face | ❌ | 同上（GFPGAN/CodeFormer）|
| S5 postprocess | ✅ | 复用 `_apply_selective_processing` |
| S6 encode | — | 最终候选本来就要落盘，不需要内存入口 |

所以本模块**只编排 S1→S2→S5**。遇到请求里包含 S3/S4 时
**明确报"能力不足"**，不静默跳过 —— 静默跳过等于"交付了一张没超分、
没修脸的照片还说过全链"。

## 为什么用 `spec_from_file_location` 加载

两条链有同名模块（`stage1_preprocess` / `stage3_ai_upscale` / …），
`sys.modules` 按名字缓存。这里给每个模块一个带链前缀的唯一名
（`_photo_s1` 等），既不污染 `sys.modules` 的公共名字，也不动 `sys.path`。
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["MEMORY_STAGES", "STAGE_SCRIPTS", "capabilities",
           "run_inproc_chain"]

#: 具备内存入口的 stage（键是计划里的名字）。
#:
#: `s3` / `s4` 在**给了 `client=`** 时也可用 —— 它们是模型阶段，
#: 必须跑在带 torch 的常驻 worker 里（`photo_model_client`）。
MEMORY_STAGES = frozenset({"s1", "s2", "s3", "s4", "s5"})
#: 必须依赖模型 worker 的 stage。
MODEL_STAGES = frozenset({"s3", "s4"})

#: 全部 6 个 stage 的脚本名（用于"请求了但没有内存入口"的判断）。
STAGE_SCRIPTS = {
    "s1": "stage1_preprocess",
    "s2": "stage2_ai_pre",
    "s3": "stage3_ai_upscale",
    "s4": "stage4_ai_face",
    "s5": "stage5_postprocess",
    "s6": "stage6_encode",
}

_DEFAULT_ORDER: Tuple[str, ...] = ("s1", "s2", "s3", "s4", "s5")

_CACHE: Dict[str, Any] = {}


def _photo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "photo_restoration_pipeline")


def _load(stage: str):
    """按**绝对路径**加载照片链的某个 stage 模块（唯一模块名，不污染）。"""
    if stage in _CACHE:
        return _CACHE[stage]
    name = STAGE_SCRIPTS[stage]
    path = os.path.join(_photo_root(), "stages", name + ".py")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(
        "_photo_%s" % stage, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHE[stage] = mod
    return mod


def capabilities(*, with_model_client: bool = False) -> Dict[str, Any]:
    """这张表就是"能不能跑内存链"的唯一依据。

    `s3`/`s4` 只有在**给了模型 worker client** 时才算可用 ——
    没给时它们仍需回退到 file 模式。
    """
    avail = set(MEMORY_STAGES)
    if not with_model_client:
        avail -= MODEL_STAGES
    return {
        "schema": "photo-inproc-capabilities/1",
        "memory_stages": sorted(avail),
        "model_stages": sorted(MODEL_STAGES),
        "needs_model_client": sorted(MODEL_STAGES) if not with_model_client
                              else [],
        "file_only_stages": sorted(set(STAGE_SCRIPTS) - avail - {"s6"}),
        "note": ("S3/S4 run through a resident model worker "
                 "(photo_model_worker); S6 always writes the final "
                 "candidate by design"),
    }


def run_inproc_chain(data: bytes, params: Dict[str, Any], *,
                     stages: Optional[Sequence[str]] = None,
                     client=None,
                     logger=None) -> Dict[str, Any]:
    """按 `stages` 顺序在内存里跑照片链。

    Args:
        data: 入口 PNG（已规范化的 RGB16；EXIF/ICC/方向由调用方处理）。
        params: 生效参数（各 stage 自己会挑它认识的键）。
        stages: 默认 ``("s1","s2","s3","s4","s5")``。**包含 S3/S4 时**
            会直接返回"能力不足"，而不是跳过它们继续跑。

    Returns:
        ``{"ok", "png", "stages_run", "unsupported", "warnings",
        "error", "trace"}``。
    """
    out: Dict[str, Any] = {"ok": False, "png": b"", "stages_run": [],
                           "unsupported": [], "warnings": [], "error": "",
                           "trace": []}
    _stages = tuple(stages) if stages else _DEFAULT_ORDER

    # ---- 先判能力，再动手 ----
    #
    # `s3`/`s4` 是模型阶段：**没有 client 就不能跑**。
    # 拒绝必须发生在动手之前 —— 静默跳过等于交付一张没超分/没修脸的
    # 照片还说过全链。
    _need = []
    for s in _stages:
        if s == "s6":
            continue
        if s not in MEMORY_STAGES:
            _need.append("%s (no in-memory entry)" % s)
        elif s in MODEL_STAGES and client is None:
            _need.append("%s (needs a model worker client)" % s)
    if _need:
        out["unsupported"] = [s.split(" ")[0] for s in _need]
        out["error"] = (
            "this chain needs %s. Refusing to silently skip them and call "
            "the result a full chain." % ("; ".join(_need),))
        return out

    cur = data
    for st in _stages:
        if st == "s6":
            # 编码不在内存链里（最终候选本来就要落盘）
            continue
        # s3/s4 走 worker，不需要在本进程加载它们的脚本
        mod = None
        if st not in MODEL_STAGES:
            try:
                mod = _load(st)
            except Exception as exc:
                out["error"] = "cannot load %s: %s" % (st, exc)
                return out
        try:
            if st == "s1":
                r = mod.preprocess_png_bytes(cur, params, logger=logger)
            elif st == "s2":
                # 退化信息由**计划**给（stage1 的 sidecar 或 classifier）
                r = mod.correct_channels_bytes(
                    cur, params,
                    degraded_channels=params.get("degraded_channels"),
                    logger=logger)
            elif st == "s3":
                # 模型阶段：**不在本进程里加载模型**，走常驻 worker
                r = client.upscale(
                    cur,
                    model=str(params.get("upscale_model")
                              or "realesr-general-x4v3"),
                    scale=int(params.get("upscale_scale") or 2),
                    half=bool(params.get("ai_half", True)))
            elif st == "s4":
                # 记录输入几何：GFPGAN 的 `upscale=` 是**为了让模型拿到
                # 足够分辨率**而做的放大，不是"我们要把照片放大"。
                # 不把它缩回去，S3 的 2× 会再叠一层变成 4×（实测 64×48
                # 走完 S3+S4 变成 256×192）。S4 的语义是**只修脸、不改几何**。
                import numpy as _np2
                from media_io import read_png_bytes as _rp2, encode_png as _enc2
                _before = _rp2(cur)
                _h0, _w0 = int(_before.shape[0]), int(_before.shape[1])
                r = client.face(cur,
                                upscale=int(params.get("face_upscale") or 2),
                                only_center_face=bool(
                                    params.get("only_center_face", False)))
                if not r.get("ok") and r.get("no_face"):
                    # **无脸是合法不适用**（与 S4 的 `face_boxes == []`
                    # 分支同一语义）：交付当前图并继续，不判整链失败。
                    # 但要留痕 —— 让人知道这一张没做人脸修复。
                    r = {"ok": True, "png": cur, "no_face": True,
                         "warnings": ["no face detected; S4 delivered the "
                                      "image unchanged"]}
                elif r.get("ok") and (int(r.get("height") or 0) != _h0
                                      or int(r.get("width") or 0) != _w0):
                    import cv2 as _cv2
                    _a = _rp2(r["png"])
                    _a = _cv2.resize(_a, (_w0, _h0),
                                     interpolation=_cv2.INTER_AREA)
                    r = dict(r, png=_enc2(_a), width=_w0, height=_h0,
                             resized_from="%dx%d" % (int(r.get("width") or 0),
                                                     int(r.get("height") or 0)),
                             warnings=list(r.get("warnings") or []) + [
                                 "GFPGAN upscale is a side effect; resized "
                                 "back to the S4 input geometry"])
            elif st == "s5":
                r = mod.postprocess_bytes(cur, params, logger=logger)
            else:                                    # pragma: no cover
                out["error"] = "no dispatcher for %s" % st
                return out
        except Exception as exc:
            out["error"] = ("%s raised %s: %s"
                            % (st, type(exc).__name__, exc))
            out["trace"].append({"stage": st, "error": out["error"]})
            return out
        out["trace"].append({
            "stage": st, "ok": bool(r.get("ok")),
            "error": r.get("error") or "",
            "warnings": list(r.get("warnings") or []),
        })
        if not r.get("ok"):
            out["error"] = "%s failed: %s" % (st, r.get("error"))
            return out
        out["warnings"].extend(r.get("warnings") or [])
        out["stages_run"].append(st)
        cur = r["png"]

    out.update({"ok": True, "png": cur})
    return out
