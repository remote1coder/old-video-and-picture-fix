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
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "STAGE_NAMES",
    "PHOTO_STAGE_NAMES",
    "photo_param_affects_stage",
    "photo_stage_identity",
    "photo_stage_scope_keys",
    "GLOBAL_PARAMS",
    "param_affects_stage",
    "stage_scope_keys",
    "stage_identity",
    "degraded_note_path",
    "read_degraded_note",
    "degraded_note_is_current",
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
    1: {"target_resolution", "vs_bit_depth",
        "frame_rate_mode", "progressive_threshold",
        # R18-A04：确定性渲染只改变 stage1（TemporalDegrain / Stab）
        # 与 stage6（GrainFactory3）的输出，不该让其他阶段失效。
        "vs_deterministic"},
    2: {"deblock", "traditional_strength", "degrain_enabled"},
    3: {"ai_enabled", "ai_force_allow_banned_upscale",
        # R18-A04：执行组模式只改变 stage3+4 的**执行方式**（是否融合），
        # 不影响其他阶段。改它不该让 stage0/1/2 重算。
        "chain_mode",
        "upscale_model", "upscale_scale", "model_dir", "max_scale",
        "lanczos_target", "ban_intermediate_pixels",
        "ban_original_input_pixels", "force_allow_banned_upscale",
        "gpu_memory_fallback", "gpu_priority", "force_traditional_first"},
    4: {"unsharp_enabled", "unsharp_strength", "gblur_enabled",
        "chain_mode",
        "gblur_sigma", "eq_enabled", "eq_contrast", "eq_saturation",
        "eq_brightness", "sharpen_enabled"},
    5: {"ai_enabled", "face_restore", "face_detect_threshold",
        "face_detect_threshold_lowres", "face_detect_fallback",
        "face_detect_lowres_max_dimension", "static_segment_duration",
        "gfpgan_model", "codeformer_fidelity"},
    6: {"grain_strength", "grain_factory3", "grain_enabled",
        "grain_seed", "vs_deterministic"},
    7: {"crf", "preset", "encoder", "video_encoder", "pix_fmt",
        "color_range",
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
    7: ("crf", "preset", "encoder", "video_encoder", "pix_fmt", "svt_",
        "color_range",
        "audio_", "bitrate"),
}

#: **声明了但没有任何读取点**的键（R23-13 静态扫描 + 逐条人工核对）。
#:
#: 它们出现在 config.yaml 里、可能还在文档中，但代码**从不读取**。
#: 处理方式：
#:   * 从 `CORE_QUALITY_PARAMS` 与 `_STAGE_KEYS` 中移除 —— 否则改它们只会
#:     产生**无谓的失效**（重算一遍，输出一模一样）；
#:   * 保留在这里，让 `tests/probes/param_capability_matrix.py` 与
#:     `tests/test_r23_param_capability.py` 能区分"已知未实现"与"新出现的死键"。
#:
#: 要"实现"其中任何一个时，把它从这里删掉、加进 `_STAGE_KEYS` 与
#: `CORE_QUALITY_PARAMS`，并补一条**行为**测试（不是 grep）。
DECLARED_NOT_IMPLEMENTED = frozenset({
    "target_fps",            # 时间轴改帧率：stage1 只认 fps_interpret/original_fps
    "psf_handling",          # PSF 处理策略：stage1 用 psf_use_input_type1 决定
    "psf_handling_alt",
    "deblock_alt",           # 备选去块：只实现了 deblock（DeblockPP7 分支未接）
    "qtgmc_input_type_rule", # 规则文本；实际由 qtgmc_input_type 直接决定
    "qtgmc_tff_required",
})

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
    # ---- R23-05：main 在每个 stage 前**临时注入**、跑完就 pop 的位置键 ----
    # `upstream_artifact` 是"上游产物在哪个路径"，不是"选了什么"。
    # 它进 stage_scope 会让**同一个 stage 在"有/无 pin"两次运行里算出不同身份**
    # —— 实测（R23-05 编排计数）：同配置重跑时 stage2 报 `params-changed` 但
    # `changed_keys` 为空，于是 2→7 全部重算。
    # 上游**内容**由侧车里的 `upstream_digest` 绑定，`main` 在复用前会比对
    # （ID-08），所以把它排除出参数身份不会丢证明。
    "upstream_artifact",
    # ---- R23-13：下面这些**会被读**，但不改变媒体像素/音频 ----------
    # 它们控制"怎么跑"和"门禁多严"，不是"跑出什么"。
    # 放进这里而不是 CORE_QUALITY_PARAMS：改它们不该让中间产物重算
    # （跑一次几十分钟，只为换个超时值，纯属浪费）。
    "stage_inactivity_timeout",     # 看门狗：多久没进展算卡死
    "stage_hard_timeout_hours",     # 硬超时
    "ai_frame_transport",           # 帧传输方式（共享内存/PNG）
    "ai_stream_raw",                # 零落盘流式
    "ai_frame_buffer_frames",
    "ai_frame_memory_fraction",
    "ai_frame_min_buffer_frames",
    "ai_frame_min_free_bytes",
    "color_drift_threshold",        # 颜色漂移门禁阈值（只影响是否拒绝）
    "color_drift_enforce",
    "full_chain_verify",            # 是否要求 fullchain 报告（对象集合，不是媒体）
    "full_chain_verify_mode",
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
    # ---- 内部/回执键不是内容语义 ----
    # 审核 R18-A04：视频链**缺少**这条跳过规则，而照片链
    # （`photo_param_affects_stage`）本来就有。后果：
    # `_run_identity` / `_shake_probe` / `_stabilize_effective` 这些纯内部键
    # 落到下面的"无人认领 → 保守 True"分支，**污染每一个 stage 的身份**。
    # 于是仅改 CRF 也会让 stage1 身份变化 → 全链重算。
    if k.startswith("_"):
        return False
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


def degraded_note_path(artifact: str) -> str:
    """降级侧车路径（stage6 写的就是这个名字）。"""
    return str(artifact) + ".degraded.json"


def read_degraded_note(artifact: str) -> Optional[Dict[str, Any]]:
    """读取降级侧车；不存在 / 读不出来返回 None。

    R23-08：侧车此前**只写不读** —— 检查点不知道"这份中间产物是降级货"，
    最终报告也不提。于是"换了插件、强度减半"这件事只留在会轮转的日志里。
    """
    import json
    p = degraded_note_path(artifact)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def degraded_note_is_current(artifact: str) -> Tuple[bool, str]:
    """侧车是否**描述当前这份字节**。

    Returns:
        ``(ok, reason)``。侧车缺失 → ``(False, "no sidecar")``；
        有侧车但摘要不符 → ``(False, "sidecar belongs to other bytes")``。
        "缺失"与"过时"在调用方那里含义不同，所以这里都要区分。
    """
    note = read_degraded_note(artifact)
    if note is None:
        return False, "no sidecar"
    want = str(note.get("output_sha256") or "")
    if not want:
        return False, "sidecar records no content digest"
    try:
        import hashlib
        h = hashlib.sha256()
        with open(artifact, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
    except OSError:
        return False, "artifact is unreadable"
    if got != want:
        return False, "sidecar belongs to other bytes"
    return True, "ok"


def build_reuse_plan(
    params: Dict[str, Any],
    recorded_by_stage: Dict[int, Optional[Dict[str, Any]]],
    *,
    result_artifact: Optional[str] = None,
    ai_stages: Sequence[int] = (3, 5),
    applicable_stages: Optional[Sequence[int]] = None,
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

    _applicable = (set(int(x) for x in applicable_stages)
                   if applicable_stages is not None else None)

    for stage in sorted(STAGE_NAMES):
        # ---- R23-05：**计划里不适用**的阶段不参与判定 ----
        # 审核要求的"计划子集"就体现在这里：`face_restore` 关掉时 stage5
        # 根本不跑，它没有侧车不是"身份缺失"，更不该让 6/7 跟着失效。
        # 实测（R23-05 编排计数）：不加这一条，同配置重跑仍会重跑 stage7。
        if _applicable is not None and stage not in _applicable:
            continue
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
            _art = rec.get("artifact") or rec.get("path") or ""
            # ---- R23-08：降级侧车必须被**消费** ----
            # 侧车存在但描述的是别的字节 → 这份产物在降级记录之后被换过，
            # 不能复用（否则"已降级"的说明会挂到一份没降级的产物上）。
            _note = read_degraded_note(_art) if _art else None
            if _note is not None:
                _ok, _why = degraded_note_is_current(_art)
                if not _ok:
                    invalidated.append({
                        "stage": stage,
                        "reason": "stale-degraded-sidecar",
                        "changed_keys": [],
                        "detail": _why,
                    })
                    first_invalid = (first_invalid
                                     if first_invalid is not None else stage)
                    stale.append(_art)
                    continue
            _entry = {"stage": stage, "path": _art,
                      "reason": "identity-match"}
            if _note is not None:
                # 复用的是**降级产物** —— 这件事必须跟着复用结论走，
                # 否则下游会以为拿到的是完整 GrainFactory3 的颗粒。
                _entry["degraded"] = True
                _entry["degraded_plugin"] = str(
                    _note.get("plugin_used") or "unknown")
                notes.append(f"stage{stage}: reusing a DEGRADED artifact "
                             f"({_entry['degraded_plugin']})")
            reusable.append(_entry)
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
    # FIX-2026-10-21（16 分钟真实素材实跑发现）：
    # `will_run_ai` 原来**只看** restart_from 与 ai_stages 的位置关系，
    # 完全不看 AI 到底有没有被启用、有没有被尺寸规则禁掉。于是：
    #   * 全默认（config `ai.enabled: false`）→ 报 will_run_ai=true；
    #   * 1920x1080 素材（`ai_upscale_banned: true`，判据 height>=1080
    #     or width>=1920）→ 同样报 true。
    # 而 main.py 的 stage 调度用的是
    # `ai_enabled and not (ai_banned and not force_allow)` —— 两者**直接矛盾**。
    # 复用计划是给用户预估"这次要跑多久"的，谎报"要跑几小时 AI"比不报更糟。
    ai_enabled = bool(params.get("ai_enabled", False))
    ai_banned = bool(params.get("ai_upscale_banned", False))
    force_allow = bool(params.get("force_allow_banned_upscale", False))
    face_restore = str(params.get("face_restore") or "")
    ai_skip_reason: Optional[str] = None
    if not ai_enabled:
        ai_skip_reason = "ai.enabled=false (pass --ai-enabled to turn it on)"
    elif ai_banned and not force_allow:
        ai_skip_reason = ("ai_upscale_banned (input >=1080p) and "
                          "force_allow_banned_upscale=false")
    # stage5 还需要人脸模型名非空（与 main.py 的 gate 一致）
    effective_ai_stages = [s for s in ai_stages if s != 5 or face_restore]
    will_run_ai = bool(ai_skip_reason is None and restart_from is not None
                       and any(s >= restart_from for s in effective_ai_stages))

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
        # None = 本次会跑 AI；否则说明为什么不跑（含 --ai-enabled 提示）
        "ai_skip_reason": ai_skip_reason,
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
    if not plan.get("will_run_ai") and plan.get("ai_skip_reason"):
        lines.append(f"  原因: {plan['ai_skip_reason']}")
    for n in plan.get("notes") or []:
        lines.append(n)
    return "\n".join(lines)


# ===========================================================================
# 照片链（第七轮）：同一套局部身份 schema，但**不做 plan/裁剪**
# ===========================================================================
#
# 为什么照片链不做"最小重算"：
#
#   照片是**单张**处理，整条链是秒级；而视频是数万帧、小时级。
#   "改 CRF 只重跑编码"在视频上能省几小时，在照片上只省几秒 ——
#   引入 plan/裁剪的复杂度不划算。
#
# 但**身份记录必须一致**：两条链的侧车都要有 `stage_identity` +
# `stage_scope`，否则跨链追溯（"这张图是用哪套参数产出的"）就断了。

#: 照片链的 stage 名（与 `photo_restoration_pipeline/stages/` 对应）
PHOTO_STAGE_NAMES: Dict[int, str] = {
    1: "preprocess",
    2: "ai_pre_and_traditional",
    3: "ai_upscale",
    4: "ai_face",
    5: "postprocess",
    6: "encode",
}

#: 照片链的参数作用域（精确键）
_PHOTO_STAGE_KEYS: Dict[int, Set[str]] = {
    1: {"deband_threshold", "scratch_remove", "scratch_remove_strength",
        "denoise_strength", "descreen", "photo_rotate", "photo_crop"},
    2: {"deblock_strength", "traditional_strength", "psf_handling",
        "ai_pre_enabled", "pre_denoise"},
    3: {"ai_enabled", "upscale_model", "upscale_scale", "model_dir",
        "max_scale", "lanczos_target", "ban_intermediate_pixels",
        "ban_original_input_pixels", "force_allow_banned_upscale",
        "force_traditional_first", "gpu_memory_fallback", "gpu_priority"},
    4: {"ai_enabled", "face_restore", "face_detect_threshold",
        "face_detect_threshold_lowres", "face_detect_fallback",
        "face_detect_lowres_max_dimension", "gfpgan_model",
        "codeformer_fidelity"},
    5: {"unsharp_enabled", "unsharp_strength", "gblur_enabled", "gblur_sigma",
        "eq_enabled", "eq_contrast", "eq_saturation", "eq_brightness",
        "sharpen_enabled", "grain_synthesis", "grain_strength"},
    6: {"crf", "encoder", "pix_fmt", "photo_bit_depth", "archival_mode",
        "avif_params", "avif_rgb444_params", "jpeg_quality", "tiff_compression",
        "icc_profile", "output_format"},
}

#: 前缀 -> stage。**更长的前缀优先**（见 `_photo_claimants`）。
_PHOTO_STAGE_PREFIXES: Dict[int, Tuple[str, ...]] = {
    1: ("deband", "scratch", "denoise", "descreen", "photo_"),
    2: ("deblock", "traditional_", "psf_", "ai_pre", "pre_denoise"),
    # `ai_` 兜底归到这里（AI 超分）；`ai_face*` 有更长的前缀，会归 stage4
    3: ("upscale", "ai_model", "ai_max", "ai_gpu", "ai_"),
    4: ("ai_face", "face_", "gfpgan", "codeformer"),
    5: ("unsharp", "gblur", "eq_", "sharpen", "grain_"),
    6: ("crf", "encoder", "pix_fmt", "avif", "archival", "jpeg_",
        "tiff_", "icc_", "output_format", "photo_bit_depth"),
}


def _photo_claimants(key: str) -> Set[int]:
    """哪些 stage 认领这个键。

    **精确键优先于前缀，长前缀优先于短前缀**。

    实测踩到的坑：加了 `ai_` 前缀后，`ai_face_threshold` 会同时命中
    stage4 的 `ai_face` 和 stage3 的 `ai_`，结果它同时影响两个 stage ——
    于是改人脸阈值也会让 AI 超分 stage 失效。按最长前缀优先即可：
    `ai_face`（7 字符）胜过 `ai_`（3 字符）。
    """
    k = _norm_key(key)
    exact: Set[int] = set()
    for s, keys in _PHOTO_STAGE_KEYS.items():
        if k in {_norm_key(x) for x in keys}:
            exact.add(s)
    if exact:
        return exact

    best_len = 0
    best: Set[int] = set()
    for s, prefs in _PHOTO_STAGE_PREFIXES.items():
        for pref in prefs:
            if k.startswith(pref):
                if len(pref) > best_len:
                    best_len, best = len(pref), {s}
                elif len(pref) == best_len:
                    best.add(s)
    return best


def photo_param_affects_stage(key: str, stage: int) -> bool:
    """照片链版：这个参数是否影响该 stage。语义与视频链一致。"""
    k = _norm_key(key)
    # CLI 内部参数（`_stage` / `_mode` / `_quick` ...）不是质量参数。
    # 实测：`_stage` 混进了 stage1 的 scope，让每次换 --stage 都改变身份。
    if k.startswith("_"):
        return False
    if k in NON_SEMANTIC_PARAMS:
        return False
    if k in GLOBAL_PARAMS:
        return True
    who = _photo_claimants(k)
    if who:
        return stage in who
    return True


def photo_stage_scope_keys(params: Dict[str, Any], stage: int) -> List[str]:
    return sorted(k for k in (params or {})
                  if photo_param_affects_stage(k, stage))


def photo_stage_identity(params: Dict[str, Any], stage: int) -> str:
    """照片链某个 stage 的局部参数身份。"""
    subset = {k: (params or {})[k] for k in photo_stage_scope_keys(params, stage)}
    payload = json.dumps(subset, sort_keys=True, ensure_ascii=False,
                         default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
