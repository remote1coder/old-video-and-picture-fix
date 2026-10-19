"""ID 第 8 点（第七轮）：逐 stage 依赖与最小重算。

**为什么需要"逐 stage"而不是一个全局摘要**

现有的 `params_identity` 是**全局**的：任何语义参数变化都会让所有
产物失效。保守，但代价太大：

* 把 CRF 从 24 改成 18 → 只需重跑 stage7（编码），却会让整条 AI 链
  重来（小时级）；
* 只改 `log_level` → 什么都不该重算；
* 换一个人脸模型 → 传统降噪（stage2）的产物完全没受影响。

任务书 §2.3：

> 阶段身份按依赖划分。换编码 CRF 应主要使编码及其验证失效，不必重跑
> 超分；只改日志等级不应重算；未启用的人脸模型更新不应使传统降噪
> 失效。同名权重换内容则必须使使用它的阶段和下游失效。

**本模块做什么**

1. 定义"哪个参数影响哪个 stage"（`param_affects_stage`）；
2. 计算某个 stage 的**局部身份**（只取影响它的参数）；
3. 结合侧车里记录的上次局部身份，判断该 stage 能否复用；
4. 把结论汇总成一份**人能读的复用计划**（任务书 §2.2 要求）。

**保守原则**：认不出的参数按"影响所有 stage"处理 —— 宁可多跑一次，
不能因为漏判而复用了身份不符的产物。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "STAGE_NAMES",
    "GLOBAL_PARAMS",
    "param_affects_stage",
    "stage_scope_keys",
    "stage_identity",
    "build_reuse_plan",
    "render_reuse_plan",
]

STAGE_NAMES: Dict[int, str] = {
    0: "audio_extract",
    1: "preprocess",
    2: "traditional_check",
    3: "ai_upscale",
    4: "postprocess",
    5: "ai_face",
    6: "grain_synthesis",
    7: "encode",
}

#: 明确只影响单个 stage 的参数（精确键）。
#: 前缀匹配见 `_STAGE_PREFIXES`。
_STAGE_KEYS: Dict[int, Set[str]] = {
    0: {"audio_extract_codec", "audio_extract_enabled"},
    1: {"target_fps", "target_resolution", "vs_bit_depth",
        "frame_rate_mode", "progressive_threshold"},
    2: {"deblock", "deblock_alt", "psf_handling", "psf_handling_alt",
        "traditional_strength", "degrain_enabled"},
    3: {"ai_enabled", "ai_force_allow_banned_upscale",
        "upscale_model", "upscale_scale", "model_dir", "max_scale",
        "lanczos_target", "ban_intermediate_pixels",
        "ban_original_input_pixels", "force_allow_banned_upscale",
        "gpu_memory_fallback", "gpu_priority", "force_traditional_first"},
    4: {"unsharp_enabled", "unsharp_strength", "gblur_enabled",
        "gblur_sigma", "eq_enabled", "eq_contrast", "eq_saturation",
        "eq_brightness", "sharpen_enabled"},
    5: {"ai_enabled", "face_restore", "face_detect_threshold",
        "face_detect_threshold_lowres", "face_detect_fallback",
        "face_detect_lowres_max_dimension", "static_segment_duration",
        "gfpgan_model", "codeformer_fidelity"},
    6: {"grain_strength", "grain_factory3", "grain_enabled",
        "grain_seed"},
    7: {"crf", "preset", "encoder", "pix_fmt", "color_range",
        "svt_tile_cols", "svt_tile_rows", "preset_max_quality",
        "audio_bitrate", "audio_codec", "audio_channels",
        "audio_noise_reduce", "audio_normalize"},
}

#: 前缀 → stage。顺序无关（会全部匹配）。
_STAGE_PREFIXES: Dict[int, Tuple[str, ...]] = {
    0: ("audio_extract",),
    1: ("deinterlace", "ivtc", "qtgmc", "vfm", "vdecimate", "stabilize",
        "shake_"),
    2: ("traditional_", "deblock", "psf_", "degrain", "srestore"),
    3: ("ai_upscale", "upscale_", "ai_model", "ai_max", "ai_gpu"),
    4: ("unsharp", "gblur", "eq_", "sharpen", "contrast", "saturation"),
    5: ("ai_face", "face_", "gfpgan", "codeformer", "ai_static"),
    6: ("grain", "ai_grain"),
    7: ("crf", "preset", "encoder", "pix_fmt", "svt_", "color_range",
        "audio_", "bitrate"),
}

#: 影响**所有** stage 的参数（处理位深、像素格式等贯穿全链）。
GLOBAL_PARAMS: Set[str] = {
    "vs_bit_depth", "pix_fmt", "color_range", "platform",
    "no_wsl_redirect",
}

#: 明确**不影响产物内容**的参数（只改这些不该重算任何 stage）。
#: 注意：`output_dir` / `temp_dir` 已在 `_identity_semantics` 里剔除，
#: 这里列出的是仍会进 params 但不影响媒体的键。
NON_SEMANTIC_PARAMS: Set[str] = {
    "log_level", "log_dir", "log_max_lines", "log_max_line_bytes",
    "log_backup_count", "manifest_format", "cleanup_intermediates",
    "keep_steps", "retry_count", "publish_lock_timeout",
    "output_dir", "temp_dir", "output_path", "input_path",
    "duration", "job_id", "run_id", "attempt_id",
}


def _norm_key(key: str) -> str:
    return str(key).strip().lower()


def _claimants(key: str) -> Set[int]:
    """哪些 stage 明确"认领"了这个参数。"""
    k = _norm_key(key)
    out: Set[int] = set()
    for s, keys in _STAGE_KEYS.items():
        if k in {_norm_key(x) for x in keys}:
            out.add(s)
    for s, prefs in _STAGE_PREFIXES.items():
        if any(k.startswith(p) for p in prefs):
            out.add(s)
    return out


def param_affects_stage(key: str, stage: int) -> bool:
    """这个参数的变化是否会让该 stage 的产物失效？

    * 非语义参数（日志、目录）→ False（任何 stage 都不受影响）
    * 全局参数（位深、像素格式）→ True（所有 stage）
    * **被某些 stage 认领** → 只对认领者 True，其他全 False
    * **无人认领（认不出的键）→ True**（保守：宁可多跑，不可误复用）

    **为什么必须先"收集认领者"再判断**（这是第七轮真跑抓到的缺陷）：

    旧实现是"如果被**别的** stage 认领 → 返回 False"。它隐含假设
    "一个参数只属于一个 stage"。但 `ai_enabled` 同时影响 stage3 和
    stage5 —— 它没有出现在任何 stage 的表里，于是落到"无人认领 →
    保守 True"分支，**连 stage1 都被判失效**。

    实测后果：改 CRF 重跑时，日志显示
    `stage 1 (preprocess): params-changed keys=ai_enabled,...`
    —— 本该只重跑 stage7，却从 stage1 全链重来。
    """
    k = _norm_key(key)
    if k in NON_SEMANTIC_PARAMS:
        return False
    if k in GLOBAL_PARAMS:
        return True
    who = _claimants(k)
    if who:
        return stage in who
    return True


def stage_scope_keys(params: Dict[str, Any], stage: int) -> List[str]:
    """列出该 stage 身份包含哪些参数键（排序后，便于比较）。"""
    return sorted(k for k in (params or {}) if param_affects_stage(k, stage))


def stage_identity(params: Dict[str, Any], stage: int) -> str:
    """某个 stage 的**局部**参数身份。

    只包含影响它的参数；与全局 `params_identity` 并存，
    用于判断"这个 stage 能否复用"。
    """
    subset = {k: (params or {})[k] for k in stage_scope_keys(params, stage)}
    payload = json.dumps(subset, sort_keys=True, ensure_ascii=False,
                         default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_reuse_plan(
    params: Dict[str, Any],
    recorded_by_stage: Dict[int, Optional[Dict[str, Any]]],
    *,
    result_artifact: Optional[str] = None,
    ai_stages: Sequence[int] = (3, 5),
) -> Dict[str, Any]:
    """生成复用计划（任务书 §2.2 要求的那六项）。

    参数
    ----
    params
        本次实际执行的参数（应含 CLI 覆盖）。
    recorded_by_stage
        `{stage: sidecar_dict_or_None}` —— 上次该 stage 留下的身份记录；
        None 表示没有可用记录（身份缺失）。
    result_artifact
        最终成品路径（若存在且有 done 标记，可作为"已全部完成"的信号）。

    返回
    ----
    ```
    {
      "reusable": [{"stage":3,"path":...,"reason":"identity-match"}, ...],
      "invalidated": [{"stage":4,"reason":"params-changed",
                       "changed_keys":[...]}, ...],
      "stale_files": [path, ...],          # 不可复用但**保留在磁盘**
      "restart_from": 4,                   # 从哪一 stage 开始重算
      "will_run_ai": True,                 # 是否触发昂贵 AI 阶段
      "notes": [...]
    }
    ```
    """
    reusable: List[Dict[str, Any]] = []
    invalidated: List[Dict[str, Any]] = []
    stale: List[str] = []
    notes: List[str] = []
    first_invalid: Optional[int] = None

    for stage in sorted(STAGE_NAMES):
        rec = recorded_by_stage.get(stage)
        if rec is None:
            # 没有记录：stage 0 是从源头开始，属正常；其余记为身份缺失
            if stage == 0:
                continue
            invalidated.append({"stage": stage, "reason": "identity-missing",
                                "changed_keys": []})
            first_invalid = first_invalid if first_invalid is not None else stage
            continue

        prev_id = rec.get("stage_identity")
        if not prev_id:
            # 旧侧车（第七轮之前）没有局部身份 → 按身份缺失处理
            invalidated.append({"stage": stage,
                                "reason": "identity-missing",
                                "changed_keys": []})
            first_invalid = first_invalid if first_invalid is not None else stage
            stale.append(rec.get("artifact") or rec.get("path") or "")
            continue

        cur_id = stage_identity(params, stage)
        if cur_id == prev_id:
            reusable.append({"stage": stage,
                             "path": rec.get("artifact") or rec.get("path"),
                             "reason": "identity-match"})
        else:
            # 找出到底哪些键变了（便于人看，也避免"不知道为什么重算"）
            prev_scope = rec.get("stage_scope") or {}
            changed = [k for k in stage_scope_keys(params, stage)
                       if prev_scope.get(k) != params.get(k)]
            invalidated.append({"stage": stage,
                                "reason": "params-changed",
                                "changed_keys": changed})
            first_invalid = first_invalid if first_invalid is not None else stage
            stale.append(rec.get("artifact") or rec.get("path") or "")

    # 失效会向下游传播：一旦某个 stage 失效，后面的都不能复用
    if first_invalid is not None:
        propagated = {d["stage"] for d in invalidated}
        reusable = [r for r in reusable
                    if r["stage"] < first_invalid
                    and r["stage"] not in propagated]
        for r in list(reusable):
            pass
        # 下游全部标为依赖变化（即使它自己的局部身份没变）
        for stage in sorted(STAGE_NAMES):
            if stage > first_invalid and stage not in {
                    d["stage"] for d in invalidated}:
                invalidated.append({"stage": stage,
                                    "reason": "upstream-invalidated",
                                    "changed_keys": []})

    restart_from = first_invalid if first_invalid is not None else None
    will_run_ai = bool(restart_from is not None
                       and any(s >= restart_from for s in ai_stages))

    if result_artifact:
        notes.append(
            f"final artifact considered at: {result_artifact} "
            "(its own completion marker decides reuse, not this plan)")

    return {
        "reusable": sorted(reusable, key=lambda r: r["stage"]),
        "invalidated": sorted(invalidated, key=lambda r: r["stage"]),
        "stale_files": [p for p in stale if p],
        "restart_from": restart_from,
        "will_run_ai": will_run_ai,
        "notes": notes,
    }


def render_reuse_plan(plan: Dict[str, Any]) -> str:
    """把计划渲染成人能直接读的几行（日志 / 控制台）。"""
    lines: List[str] = []
    reusable = plan.get("reusable") or []
    invalid = plan.get("invalidated") or []
    if reusable:
        lines.append("可复用:")
        for r in reusable:
            lines.append(f"  - stage {r['stage']} ({STAGE_NAMES.get(r['stage'])}): "
                         f"{r.get('path') or '(no path)'} [{r['reason']}]")
    if invalid:
        lines.append("需重算:")
        for r in invalid:
            extra = ""
            if r.get("changed_keys"):
                extra = " keys=" + ",".join(r["changed_keys"][:8])
            lines.append(f"  - stage {r['stage']} ({STAGE_NAMES.get(r['stage'])}): "
                         f"{r['reason']}{extra}")
    if plan.get("stale_files"):
        lines.append("旧文件保留（不删除、不洗白）:")
        for p in plan["stale_files"][:10]:
            lines.append(f"  - {p}")
    rf = plan.get("restart_from")
    lines.append(f"从 stage {rf if rf is not None else '(无需重算)'} 开始")
    lines.append(f"是否触发 AI 阶段: {'是' if plan.get('will_run_ai') else '否'}")
    for n in plan.get("notes") or []:
        lines.append(n)
    return "\n".join(lines)
