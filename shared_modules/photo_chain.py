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
MEMORY_STAGES = frozenset({"s1", "s2", "s5"})

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


def capabilities() -> Dict[str, Any]:
    """这张表就是"能不能跑内存链"的唯一依据。"""
    return {
        "schema": "photo-inproc-capabilities/1",
        "memory_stages": sorted(MEMORY_STAGES),
        "file_only_stages": sorted(set(STAGE_SCRIPTS) - MEMORY_STAGES
                                  - {"s6"}),
        "note": ("S3/S4 need a resident model worker; S6 always writes the "
                 "final candidate by design"),
    }


def run_inproc_chain(data: bytes, params: Dict[str, Any], *,
                     stages: Optional[Sequence[str]] = None,
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
    _need = [s for s in _stages if s not in MEMORY_STAGES and s != "s6"]
    if _need:
        out["unsupported"] = _need
        out["error"] = (
            "this chain requests %s, which have no in-memory entry yet "
            "(they need a resident model worker). Refusing to silently skip "
            "them and call the result a full chain." % (", ".join(_need),))
        return out

    cur = data
    for st in _stages:
        if st == "s6":
            # 编码不在内存链里（最终候选本来就要落盘）
            continue
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
