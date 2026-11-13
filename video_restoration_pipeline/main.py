#!/usr/bin/env python3
"""
Video Restoration Pipeline - Main Controller
Orchestrates video restoration stages with manifest-driven batch processing.

v7.32 Compliance:
- Parallel worker pool for concurrent job processing
- Disk space estimation before processing
- Preview mode support
- Config path propagated to child processes
- Dynamic timeout based on video duration
"""

import os
import re
import sys
import json
import time
import uuid
import hashlib
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Add shared_modules to path
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from manifest_loader import (  # noqa: E402
    load_manifest, filter_jobs, save_manifest, update_job_status,
    save_manifest_job_result, acquire_run_ownership, update_manifest_with_writer,
)
from checkpoint import (cleanup_old_intermediates, cleanup_all_intermediates,  # noqa: E402
                         check_wsl_space)


def _current_run_param_hashes(last_produced: Dict[int, Any]) -> set:
    """本轮运行**实际产出过**的 param_hash 集合 —— 孤儿回收的判据基准。

    中间产物命名是 `{job}_{input}_step{N}_{op}_p{hash}.mkv`，其中 hash 由各
    stage 以**自己的输入**（上一步产物）为身份基准算出。所以同一轮里不同
    step 的 hash **天然不同**，判定"是否本轮产物"必须用集合，拿单个值去
    比会把上一步的合法产物误判成孤儿。
    """
    hashes: set = set()
    for paths in (last_produced or {}).values():
        for p in paths or []:
            m = re.search(r"_p([0-9a-f]+)\.", os.path.basename(str(p)))
            if m:
                hashes.add(m.group(1))
    return hashes


def parse_args():
    parser = argparse.ArgumentParser(description="Video Restoration Pipeline")
    parser.add_argument("--input", "-i", nargs="+", help="Input directories")
    parser.add_argument("--output", "-o", help="Output directory")
    # R14-06：授权写根。给出时，任何落在根外的写路径都会在**首次落盘前**
    # 被拦下并说明是哪一项、来自哪个来源。可重复给出多个根。
    parser.add_argument(
        "--allow-write-root", action="append", default=None,
        metavar="DIR",
        help="Authorised write root (repeatable). Any write path outside "
             "these roots aborts the job before anything is written.")
    # R14-06：显式"重新定位输出"。给定时，媒体与**全部**侧车（verify/
    # fullchain/status/contract/journal）都改到这个目录，只保留原文件名。
    # temp/log 不受影响（它们有独立语义，且可能被有意分开配置）。
    parser.add_argument(
        "--relocate-output", default=None, metavar="DIR",
        help="Relocate the final media AND all its sidecars into DIR "
             "(keeps basenames). temp/log are unaffected.")
    parser.add_argument("--config", "-c", default=str(SCRIPT_DIR / "config.yaml"),
                        help="Config file path")
    parser.add_argument("--manifest", "-m", help="Use existing manifest")
    parser.add_argument("--output-manifest", help="Output manifest path for --scan")
    parser.add_argument("--preview-only", action="store_true",
                        help="Preview mode (generate preview frames/clips without full processing)")
    # FIX-M1: --preview alias for --preview-only per CLAUDE.md §0 (lists both
    # --preview and --preview-only as shared flags). v7.32 §3.3 mode C uses
    # --preview-only; --preview is an alias so users can use either name.
    # --mode preview remains the canonical entry.
    parser.add_argument("--preview", action="store_true",
                        help="Alias for --preview-only (preview mode, no execution)")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Keep intermediate files")
    parser.add_argument("--chain", default=None,
                        choices=["none", "stage34", "stage67", "both"],
                        help="execution-group mode; only changes HOW stages run, "
                             "never which features are available")
    parser.add_argument("--no-wsl-redirect", action="store_true",
                        help="Disable WSL path redirect")
    parser.add_argument("--stage", type=int, help="Run single stage only")
    parser.add_argument("--job-id", help="Process specific job ID only")
    parser.add_argument("--gpu", default="auto",
                        help="GPU backend (auto/cuda/rocm/mps/cpu)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    # V-P-001~004 fix: mode parameters
    parser.add_argument("--mode", choices=["scan", "edit", "batch", "quick", "interactive", "preview"],
                        help="Execution mode (v7.3 §3.3)")
    parser.add_argument("--scan", action="store_true",
                        help="Step 1: Scan input dirs and generate manifest")
    parser.add_argument("--edit-manifest", action="store_true",
                        help="Step 2: Interactive manifest review")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing manifest (skip DONE)")
    # ISSUE-V-MAIN-001 / BUG-047: AI CLI overrides (highest priority per v7.3 §3.10.12)
    parser.add_argument("--ai-enabled", action="store_true",
                        help="Enable AI upscale (overrides config.ai.enabled)")
    # FIX-2026-10-22：补上与照片链对齐的 CLI 开关。
    #
    # **注意措辞**：视频链并非"没有放行入口"——能力一直存在，有三条路：
    #   1. `config.yaml` 的 `ai.force_allow_banned_upscale: false`（第 261 行）
    #   2. manifest 的 `user_overrides: {force_allow_banned_upscale: true}`
    #      （merge_params 第 6 层应用，实测 params 里能拿到 True）
    #   3. stage3 自己被 ban 时的报错也提示"Use force_allow_banned_upscale
    #      to override"
    #
    # 缺的只是**命令行**入口：照片链一直有 `--force-allow-banned-upscale`，
    # 视频链没有，而对单个素材临时放行时改 config / 编辑 manifest 都很别扭
    # （config 是全局的，manifest 要手改 YAML）。这里补的是易用性缺口。
    parser.add_argument(
        "--fps-interpret", metavar="RATE", default=None,
        help="Explicitly reinterpret the timeline at this frame rate "
             "(e.g. 30000/1001). This is a USER ASSUMPTION, not a "
             "probe result; it overrides whatever ffprobe reports.")
    parser.add_argument("--force-allow-banned-upscale", action="store_true",
                        help="Override the AI-upscale ban for this run "
                             "(e.g. 1080p source). Equivalent to config "
                             "ai.force_allow_banned_upscale=true or manifest "
                             "user_overrides. Produces >1080p output; "
                             "use deliberately.")
    parser.add_argument("--ai-upscale-factor", type=int, choices=[2, 4], default=None,
                        help="AI upscale factor 2x or 4x (overrides config.ai.upscale_scale)")
    parser.add_argument("--face-restore", choices=["GFPGAN", "CodeFormer", ""], default=None,
                        help="Face restore model (overrides config.ai.face_restore)")
    parser.add_argument("--ai-model", choices=["realesr-general-x4v3",
                                                "RealESRGAN_x2plus", "RealESRGAN_x4plus",
                                                "DAT_x2", "DAT_x4",
                                                "DAT_2_x2", "DAT_2_x4"], default=None,
                        help="AI upscale model (overrides config.ai.upscale_model)")
    # ISSUE-V-MGEN-003: incremental manifest scan
    parser.add_argument("--incremental", action="store_true",
                        help="Incremental scan: preserve existing manifest user_overrides/status")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and return config dict."""
    import yaml  # type: ignore[import-untyped]
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def merge_params(job: Dict[str, Any], config: Dict[str, Any],
                 cli_overrides: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """Merge parameters with priority:
    CLI > user_overrides > recommended_params > audio_override > paths > global+audio+ai > defaults
    Uses deep merge for nested dicts.
    Records PARAM_OVERRIDE events per v7.3 §3.10.12.

    FIX-P5-1: Corrected priority order per v7.32 §3.2 / CLAUDE.md §3:
      global → paths[type] → audio_override[type] → recommended_params → user_overrides → CLI
      ai_/audio_ prefix keys are set conditionally (only if not already set by a higher layer).
    """
    params: Dict[str, Any] = {}
    overrides_log = []

    def _log_override(key: str, old_val: Any, new_val: Any, source: str):
        if old_val != new_val:
            overrides_log.append({"key": key, "old": old_val, "new": new_val, "source": source})

    # FIX-W1/W8: cross-domain shared keys. `denoise`/`denoise_params` are reused
    # by BOTH video (stage1 reads bare `denoise`) and audio (`audio_denoise`).
    # The bare<->audio_ prefix sync in _set_param must NOT fire for these,
    # otherwise setting the video denoise (from paths[]) transiently overwrites
    # the audio denoise (audio_denoise) and vice versa. v7.32 §3.2 reuses the
    # key name across domains (known design flaw); this guard decouples them.
    _CROSS_DOMAIN_KEYS = {"denoise", "denoise_params"}

    def _set_param(k: str, v: Any, source: str, is_dict_merge: bool = False):
        """Set param and sync ai_/audio_ prefixed variants."""
        old = params.get(k)
        if is_dict_merge and isinstance(v, dict) and isinstance(params.get(k), dict):
            params[k] = _deep_merge(params.get(k, {}), v)
        else:
            params[k] = v
        _log_override(k, old, v, source)
        # Sync prefixed variants so both forms reflect the override.
        # ORIGINAL: synced audio_ prefix unconditionally, polluting audio_denoise
        # when paths[] set the video denoise. FIX-W1/W8: skip cross-domain keys
        # whose bare/prefixed forms refer to DIFFERENT params (video vs audio
        # denoise) — syncing them would corrupt the other domain's value.
        if k not in _CROSS_DOMAIN_KEYS:
            for prefix in ("ai_", "audio_"):
                pk = f"{prefix}{k}"
                if pk in params and pk != k:
                    old_p = params[pk]
                    params[pk] = v
                    if old_p != v:
                        _log_override(pk, old_p, v, source)
        # FIX-F1: reverse sync - when k itself is a prefixed form (e.g. ai_enabled,
        # audio_denoise) and the corresponding bare key (enabled, denoise) already
        # exists in params, sync the bare key too. ORIGINAL only synced bare->prefix,
        # so CLI overrides setting `ai_enabled=True` left `enabled=False` (stale),
        # and stages reading the bare key saw the CLI override as no-op. Cross-domain
        # keys are exempt for the same reason as the forward sync (video vs audio
        # denoise refer to different params).
        if k not in _CROSS_DOMAIN_KEYS:
            for prefix in ("ai_", "audio_"):
                if k.startswith(prefix) and len(k) > len(prefix):
                    bare = k[len(prefix):]
                    if bare in params and bare != k:
                        old_b = params[bare]
                        params[bare] = v
                        if old_b != v:
                            _log_override(bare, old_b, v, source)

    # 1. Start with global
    base = config.get("global", {})
    params = _deep_merge(params, base)

    # 2. Merge audio/ai top-level sections at global priority level
    #    Conditional assignment: only set if key doesn't already exist
    #    ORIGINAL: lines 164-174 unconditionally set ai_/audio_ prefixed keys,
    #    overwriting higher-priority values. Now conditional.
    audio_config = config.get("audio", {})
    for k, v in audio_config.items():
        # FIX-W1/W8: `denoise`/`denoise_params` are cross-domain (video stage1
        # reads bare `denoise`). Route audio's denoise to the audio_ namespace
        # ONLY, so the global audio value cannot masquerade as the video denoise
        # method. Other audio keys (hum_freq, arnndn_model, ...) stay bare
        # because stage7 reads them bare and they do not conflict with video.
        if k in ("denoise", "denoise_params"):
            if f"audio_{k}" not in params:
                params[f"audio_{k}"] = v
            continue
        if k not in params:
            params[k] = v
        if f"audio_{k}" not in params:
            params[f"audio_{k}"] = v

    ai_config = config.get("ai", {})
    for k, v in ai_config.items():
        if k not in params:
            params[k] = v
        if f"ai_{k}" not in params:
            params[f"ai_{k}"] = v

    # 3. Override with path-specific params
    detected_class = job.get("detected_class", "")
    path_key = detected_class.lower()
    # FIX-NEW-V-10 / R12: user_overrides 里的 `path` 应能切换分型参数组，
    # 否则 `path: dv_digital` 只能写进 params["path"] 而 paths[path_key]
    # 仍按 detected_class 取——实测三型矩阵三例参数完全相同。
    # user_overrides 在第 6 步才应用，这里**预读**它（优先级最高，
    # 只影响 paths/audio_override 的查表键，不改变其他层的应用顺序）。
    _user_ov_pre = job.get("user_overrides", {}) or {}
    _path_override = _user_ov_pre.get("path")
    if isinstance(_path_override, str) and _path_override:
        path_key = _path_override.lower()
        _log_override("path", detected_class, path_key, "user_overrides")
    paths = config.get("paths", {})
    if path_key in paths:
        for k, v in paths[path_key].items():
            _set_param(k, v, "paths")

    # 4. Override with audio_override (CORRECT POSITION: after paths, before recommended_params)
    #    ORIGINAL: audio_override was applied after user_overrides (line 147-154),
    #    causing it to override user_overrides. Now correctly between paths and recommended_params.
    # FIX-NEW-V-10 / R12: 合并两类来源——
    #   a) config["audio_override"][path_key]（分型默认）
    #   b) user_overrides["audio_override"]（**扁平** dict，直接给
    #      hum_freq/hum_dual 等；此前只认 (a)，实测传 (b) 整段被忽略，
    #      三例 audio_override 全落回 config 默认值）
    audio_override = config.get("audio_override", {})
    _merged_audio: Dict[str, Any] = {}
    if path_key in audio_override:
        _merged_audio.update(audio_override[path_key] or {})
    _user_audio = _user_ov_pre.get("audio_override")
    if isinstance(_user_audio, dict):
        for _k in _user_audio:
            _log_override(_k, _merged_audio.get(_k), _user_audio[_k],
                          "user_overrides")
        _merged_audio.update(_user_audio)
    for k, v in _merged_audio.items():
        # FIX-W1/W8: route cross-domain `denoise`/`denoise_params` into the
        # audio_ namespace so audio_override (higher priority than paths)
        # cannot clobber the video `denoise` set by paths[]. stage1 reads
        # bare `denoise` for video; audio reads `audio_denoise`.
        if k in ("denoise", "denoise_params"):
            _set_param(f"audio_{k}", v, "audio_override")
        else:
            _set_param(k, v, "audio_override")

    # 5. Override with job recommended_params
    rec_params = job.get("recommended_params", {})
    if rec_params:
        for k, v in rec_params.items():
            _set_param(k, v, "recommended_params", is_dict_merge=True)

    # 6. Override with user_overrides
    user_ov = job.get("user_overrides", {})
    if user_ov:
        for k, v in user_ov.items():
            _set_param(k, v, "user_overrides", is_dict_merge=True)

    # 7. Override with CLI (highest priority)
    if cli_overrides:
        for k, v in cli_overrides.items():
            _set_param(k, v, "cli")

    # grain_factory3 as nested dict
    if "grain_factory3" in config:
        if "grain_factory3" not in params:
            params["grain_factory3"] = config["grain_factory3"]
        else:
            params["grain_factory3"] = _deep_merge(
                config["grain_factory3"], params["grain_factory3"]
            )

    # Inject job classification fields (critical for stage scripts)
    params["detected_class"] = job.get("detected_class", "ANALOG_TAPE")
    params["field_order"] = job.get("field_order", "unknown")
    params["needs_ivtc"] = job.get("needs_ivtc", False)
    params["is_psf"] = job.get("is_psf", False)
    # FIX-W2: propagate psf_tff (original TFF/BFF for PSF material) so stage1's
    # SeparateFields+Weave uses the correct field order per v7.32 §3.10.1.
    params["psf_tff"] = job.get("psf_tff", False)
    params["original_width"] = job.get("original_width", 0)
    params["original_height"] = job.get("original_height", 0)
    params["ai_upscale_banned"] = job.get("ai_upscale_banned", False)
    params["job_id"] = job.get("job_id", "")
    params["input_path"] = job.get("input_path", "")
    params["field_segments"] = job.get("field_segments", [])
    params["duration"] = job.get("duration", 0)
    # FIX-P6-6: Inject original_fps from classify output for PSF frame rate verification
    # per v7.32 §3.10.1
    if "original_fps" not in params:
        params["original_fps"] = job.get("fps", 0)

    # V-P-006 fix: log PARAM_OVERRIDE events
    if logger and overrides_log:
        for entry in overrides_log:
            logger.info("PARAM_OVERRIDE",
                       f"{entry['key']}: {entry['old']} -> {entry['new']} (source={entry['source']})")

    # FIX-R5-07: job 级落盘目录也必须进 params。
    # 原来这里只注入分类字段，temp_dir/log_dir 默默沿用 config global 的
    # 相对 "./temp"/"./logs"，按**进程 cwd** 解析——manifest 里 FIX-L9
    # 记的 job 级 temp_dir 只有 loader 用，执行链根本不用。后果：
    #   1) 扫描后换 cwd 执行，中间产物/日志散落到新 cwd 下（实测 1.1GB
    #      落进仓库 temp/，而 .gitignore 只有 /temp1/ 没有 /temp/）；
    #   2) 换 config 执行时，loader 按 manifest 的 temp_dir 找产物、
    #      stage 按 config 的 temp_dir 写产物，两边对不上 -> resume 失效。
    # output_path 在 process_job 里已从 job 注入（main.py:732），这里补齐
    # 另外两个；job 字段优先于 config global，与 FIX-L9 的设计意图一致；
    # 但执行期 CLI 若显式给了同名键（未来加 --temp-dir 之类），CLI 优先。
    if job.get("temp_dir") and "temp_dir" not in cli_overrides:
        params["temp_dir"] = job["temp_dir"]
    if job.get("log_path") and "log_dir" not in cli_overrides:
        params["log_dir"] = str(Path(job["log_path"]).parent)

    return params


#: 硬超时的"最终保险"默认值（小时）。
#:
#: 职责划分（2026-10-21 起的设计，见 config.yaml 与 run_stage 注释）：
#:   * **看门狗**（`stage_inactivity_timeout`）负责判"卡死"——证据是
#:     temp 目录在阈值内一个字节没长（STAGE_STALLED，rc=87）；
#:   * **硬超时**只作最终保险，防"一直在写但写了三天"这类真失控。
DEFAULT_HARD_TIMEOUT_HOURS = 48.0


def _calculate_timeout(video_duration: float, has_ai: bool = False,
                       width: int = 0, height: int = 0,
                       hard_hours: float = DEFAULT_HARD_TIMEOUT_HOURS) -> int:
    """返回 stage 的**最终保险**超时（秒）—— 不是工作估算。

    FIX-2026-10-22：修一个"注释与实现相反"的缺陷。

    2026-10-21 已经把"判死"机制换成**看门狗**（temp 目录在
    `stage_inactivity_timeout` 内没长 → 判卡死，有证据）。config 与
    run_stage 的注释都明确写着：

        看门狗负责"卡死"（有证据），48 小时 timeout 只作最终保险。
        慢机器 / 复杂画面只要还在推进就不会被杀。

    但实现是 `return min(estimate, 3600*48)` —— **估算值继续当判死线**，
    48h 只是它的上限。于是估算准不准直接决定健康任务会不会被杀：

      * 20s / 1920x1080 / 开 AI：`int(20/60) == 0` → 分辨率缩放与 AI 开销
        **全被乘零**，只剩 base **600s**；
      * 962s / 1920x1080：给 3.8h，而实测 stage7 单独就要 3h（仅 27% 余量）。

    实测后果（2026-10-22，1080p 隔行素材强制开 AI）：stage1 的 QTGMC
    反交错处理 651 帧，**正常推进、正常写盘**，跑到第 600 秒被硬超时杀掉
    —— 正是设计要避免的"慢的被杀"。更糟的是这次杀进程又撞上
    cancel_guard 的 `subprocess` NameError（已修），报错连"超时"都看不出来。

    现在按注释本意实现：**估算只作参考与告警，硬超时恒取保险值**。

    为什么不用 `max(estimate, hard)`：那样估算一旦超过 48h（例如 1 小时
    4K + AI 的估算达 84h）就重新变成判死线，等于把刚修掉的毛病又装回去。
    改成：估算超过保险值时往 stderr 打一条告警，提示调高
    `stage_hard_timeout_hours` —— 让"要不要放宽"由人决定，而不是被一个
    猜出来的数字静默决定。
    """
    base = 600  # 10 min setup
    per_minute_480p = 120  # 2 min per video minute at 640x480
    # 像素比（以 640x480 为基准），限制在 [1, 12] 避免异常值
    px = float((width or 640) * (height or 480))
    scale = min(12.0, max(1.0, px / (640.0 * 480.0)))
    per_minute = per_minute_480p * scale
    if has_ai:
        per_minute += 300 * scale  # +5 min per video minute, also scaled
    # 估算值（参考用）。短视频按"不满一分钟计一分钟"，否则会被截断成 0，
    # 让参考值失真到看不出分辨率/AI 的影响。
    _minutes = max(1.0, float(video_duration) / 60.0)
    estimate = base + int(_minutes * per_minute)
    hard = int(max(0.1, float(hard_hours)) * 3600)
    if estimate > hard:
        # 不作判死，只提示：让操作者知道"校准估算已经超过保险值"。
        sys.stderr.write(
            "WARN: stage 估算耗时 %ds (%.1fh) 已超过硬超时保险值 %ds (%.1fh)。"
            "硬超时不会按估算判死；若该素材确实需要更久，请调高 config 的 "
            "stage_hard_timeout_hours。\n"
            % (estimate, estimate / 3600.0, hard, hard / 3600.0))
    return hard


def _write_stage_params_file(params: Dict[str, Any],
                             params_json: str) -> Optional[str]:
    """把超长参数写进**计划内目录**的临时文件，返回路径。

    为什么不能裸用 ``mkstemp()``
    ---------------------------
    不传 ``dir`` 时它会落到系统临时目录（Windows 上是 C 盘用户 temp）。
    那既不在本轮的授权写根内（审核 R17-06："mkstemp 未指定计划内目录"），
    也违反"项目内优先、别塞 C 盘"的约定。

    目录选择顺序：``params["temp_dir"]`` → 项目内 ``temp/``。
    绝不退回系统临时目录 —— 宁可报错，也不写到计划外。
    """
    import tempfile as _tempfile

    _pf_dir = str(params.get("temp_dir") or "")
    if not _pf_dir:
        _pf_dir = str(SCRIPT_DIR / "temp")
    try:
        os.makedirs(_pf_dir, exist_ok=True)
    except OSError:
        pass
    fd, _path = _tempfile.mkstemp(suffix=".json", prefix="vparams_",
                                  dir=_pf_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(params_json)
    return _path


def _resolve_log_dir(log_dir: Any) -> str:
    """把 bootstrap 日志目录解析成**确定的绝对路径**。

    原实现直接用配置值（默认 ``"./logs"``）→ 相对**当前工作目录**：
    同一个 run 从不同 cwd 启动会写到不同地方，甚至写到项目之外。
    审核 R17-06：bootstrap 日志可以有自己的授权位置，但不能默认为不受约束。

    相对路径锚到**项目根**（与"从项目根启动"时的既有落点一致，不改变布局）；
    显式配置的绝对路径原样尊重 —— 那是用户的选择。
    """
    _raw = str(log_dir if log_dir not in (None, "") else "./logs")
    if os.path.isabs(_raw):
        return _raw
    return str((SCRIPT_DIR.parent / _raw).resolve())


def run_stage(script_name: str, job: Dict[str, Any], params: Dict[str, Any],
              config_path: str) -> Tuple[int, str]:
    """Run a stage script and return (exit_code, stderr)."""
    script_path = SCRIPT_DIR / "stages" / script_name
    if not script_path.exists():
        return 80, f"Script not found: {script_path}"  # Dependency missing

    # Build command
    cmd = [sys.executable, str(script_path)]

    # Add standard args
    cmd.extend(["--input", job["input_path"]])
    cmd.extend(["--job-id", job["job_id"]])

    # ISSUE-V-MAIN-003: avoid command-line length limits by using params file when long
    import json
    params_json = json.dumps(params, default=str)
    params_file = None
    # ---- R23-LONG-01：命令行长度必须按**转义后**的总长判断 ----
    #
    # Windows 的 CreateProcess 上限是 **32767 个字符**。这里有两个坑，旧的
    # `len(params_json) > 30000` 两个都踩：
    #
    # 坑 1：`len(params_json)` **不是**命令行里的实际占用。params 是 JSON，
    #   里面全是 `"` 和 Windows 路径的 `\`；`subprocess.list2cmdline()` 会把
    #   它们再转义一遍。**实测增幅 1.13–1.30 倍**（内容相关：纯数字 1.16、
    #   Windows 路径 1.13、路径+中文+引号 1.30）。
    #   再叠加 head/tail 与安全余量（约 1.5 千字符），`len(params_json)` 到
    #   **2.5 万以上就可能越界**；而旧判据要到 3 万才切文件 ——
    #   中间那 5000 字符是"看着安全、实际必炸"的区间。
    # 坑 2：失败形态极难查。Popen 抛
    #   `[WinError 206] The filename or extension is too long`，被下面那个
    #   `except Exception: return 82` 吞成 "Stage 1 failed: 82"；失败发生在
    #   10 毫秒内、temp 一个字节都没写 —— 只会让人去怀疑素材或滤镜。
    #
    # 实测（2026-11-12，16 分钟 1080p 真实素材，962.295s/28840 帧）：stage1 必中。
    # params 里现在有 `_work_format` 计划（boundaries 表）、`_frozen_param_digest`
    # 等结构化字段，比旧版大得多。
    #
    # 现在按 list2cmdline 的真实结果判断；写文件失败时**明确报错**，
    # 不再回落 inline（回落只会二次触发 206，再被吞成看不懂的 82）。
    _CMDLINE_LIMIT = 32767      # Windows CreateProcess 硬上限（字符）
    _SAFETY_MARGIN = 1024       # 环境块 / CreateProcess 自身开销
    _head_len = len(subprocess.list2cmdline(cmd))
    # 后面固定还会追加 --config，可能还有 --no-wsl-redirect / --cleanup-intermediates
    _tail_len = (len(subprocess.list2cmdline(["--config", str(config_path)]))
                 + 64)
    _quoted_params = subprocess.list2cmdline([params_json])
    _need = _head_len + len(_quoted_params) + _tail_len + _SAFETY_MARGIN
    if _need > _CMDLINE_LIMIT:
        try:
            params_file = _write_stage_params_file(params, params_json)
            cmd.extend(["--params-file", params_file])
        except Exception as exc:
            return 80, (
                "cannot pass stage parameters: the inlined arguments would need "
                "%d characters (Windows limit %d; params %d chars -> %d after "
                "escaping) and the params file could not be written: %s"
                % (_need, _CMDLINE_LIMIT, len(params_json),
                   len(_quoted_params), exc))
    else:
        cmd.extend(["--params", params_json])

    # Add config path (FIXED: use passed config_path, not hardcoded)
    cmd.extend(["--config", config_path])

    # Add flags
    # R4（关键修复）：只有下面这两个 stage 会真正做 WSL 路径重定向决策
    # （它们内部调用 wsl_redirect_path），其余 stage **并没有定义
    # --no-wsl-redirect**。原来无条件给所有 stage 传该参数，
    # 于是 stage2 起全被 argparse 判为未知参数 -> sys.exit(2)，
    # 表现为 "Stage 2 failed: 2"，Windows（无 WSL）下视频链根本跑不完。
    _STAGES_WITH_WSL_REDIRECT = {"stage0_audio_extract.py",
                                 "stage1_preprocess.py"}
    if (params.get("no_wsl_redirect", False)
            and script_name in _STAGES_WITH_WSL_REDIRECT):
        cmd.append("--no-wsl-redirect")
    if params.get("cleanup_intermediates", True):
        cmd.append("--cleanup-intermediates")

    # Dynamic timeout
    video_duration = params.get("duration", 0) or _get_video_duration(job["input_path"])
    has_ai = params.get("ai_enabled", False)
    # FIX-2026-10-21：把分辨率传进去 —— 旧调用只传时长，1080p 与 480p
    # 拿一样超时（实测 962s/1080p 的 stage7 需要 3 小时，旧公式只给 42 分钟）。
    # FIX-2026-10-22：硬超时改为**最终保险**（默认 48h，走
    # `stage_hard_timeout_hours`），判死交给下面的看门狗。
    # FIX-2026-10-24：`stage_hard_timeout_hours` 设 **0 或负数 = 完全禁用
    # 硬超时**（有的素材确实要跑非常久，48 小时也不够）。
    try:
        _hard_hours = float(params.get("stage_hard_timeout_hours",
                                       DEFAULT_HARD_TIMEOUT_HOURS))
    except (TypeError, ValueError):
        _hard_hours = DEFAULT_HARD_TIMEOUT_HOURS

    if _hard_hours <= 0:
        # 硬超时停用。此时唯一防线是下面的看门狗；若看门狗也关了，
        # 就是真的不限时（用户明确要求，需自行承担风险）。
        timeout = None
        logger.warning("STAGE_HARD_TIMEOUT_DISABLED",
                       "stage_hard_timeout_hours<=0: the hard timeout is OFF; "
                       "only the inactivity watchdog (if enabled) can stop a "
                       "stalled stage", job_id=job.get("job_id", ""))
    else:
        timeout = _calculate_timeout(
            video_duration, has_ai,
            width=int(job.get("original_width")
                      or params.get("original_width") or 0),
            height=int(job.get("original_height")
                       or params.get("original_height") or 0),
            hard_hours=_hard_hours)

    # Run under inactivity watchdog (FIX-2026-10-21)
    #
    # 为什么不再用固定的 subprocess.run(timeout=...)：判断"卡死"的正确证据是
    # **产出不再增长**，而不是"总耗时超过某个猜出来的数"。画面复杂度、GPU
    # 降频、并发负载都会让总耗时成倍波动 —— 16 分钟 1080p 纯传统链的 stage7
    # 实测就要 3 小时，而开了 AI 之后"4 小时上限"必然不够。
    #
    # 新语义：
    #   * `stage_inactivity_timeout`（默认 600s）内 temp 目录一个字节没长 →
    #     判卡死（rc=87，带 STAGE_STALLED 日志）。这是**有证据**的判定。
    #   * 总 timeout 只作最终保险（默认 48h），防"一直在写但写了三天"。
    #   * 慢机器 / 复杂画面只要还在推进就不会被杀。
    #
    # 延迟导入：main.py 在模块级并不把 shared_modules 放进 sys.path，
    # 只有 run_stage 执行时子进程的 PYTHONPATH 才包含它。
    from cancel_guard import run_managed as _cg_run_managed
    from cancel_guard import StageStalled as _CgStageStalled
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SCRIPT_DIR.parent)

        temp_dir = (params.get("temp_dir")
                    or params.get("temp_dir_original") or "")
        out_path = params.get("output_path") or ""
        out_dir = os.path.dirname(out_path) if out_path else ""
        # 注意用 **stem（不含扩展名）**：stage7 的暂存文件是
        # `{stem}.attempt-{pid}-{ts}.mkv`，而成品是 `{stem}.avif`——
        # 若用 basename（含扩展名）做前缀，两者匹配不上，attempt 文件会被
        # 排除在进度标记之外 → 明明在写却被判停滞（2026-10-21 实测踩到）。
        out_stem = Path(out_path).stem if out_path else ""
        try:
            inactivity = float(params.get("stage_inactivity_timeout", 600.0))
        except (TypeError, ValueError):
            inactivity = 600.0

        def _progress_marker() -> int:
            """进度标记 = temp 中间产物 + 本 job 的成品/暂存。

            FIX-2026-10-21（16 分钟长片实测）：第一版只统计 temp_dir，结果
            **stage7 必然被误判停滞** —— 因为它把成品写到 `output/`，不在
            temp 里：`{job}_restored.attempt-{pid}-{ts}.mkv`。
            实测 10 分钟（正好等于默认 600s）后被 watchdog 杀掉，rc=87，
            而日志里连 STAGE_STALLED 都没有（当时传了 logger=None）。

            所以必须同时看：
            * temp_dir 下所有中间产物（stage1/2/6 都写这里）；
            * output 目录里**以本 job 成品名为前缀**的文件
              （`.attempt-*` 暂存 / `.partial` / 成品本身）。
              只取前缀匹配，避免同目录其他 job 的文件掩盖停滞。
            """
            total = 0
            try:
                for _root, _dirs, _files in os.walk(temp_dir):
                    for _fn in _files:
                        try:
                            total += os.path.getsize(os.path.join(_root, _fn))
                        except OSError:
                            pass
            except OSError:
                pass
            if out_dir and out_stem:
                try:
                    for _fn in os.listdir(out_dir):
                        if not _fn.startswith(out_stem):
                            continue
                        try:
                            total += os.path.getsize(
                                os.path.join(out_dir, _fn))
                        except OSError:
                            pass
                except OSError:
                    pass
            return total

        _watch_on = bool(temp_dir or out_dir) and inactivity > 0
        r = _cg_run_managed(
            cmd, timeout=timeout, logger=None,
            progress_watch=_progress_marker if _watch_on else None,
            inactivity_timeout=inactivity if _watch_on else None,
            capture_output=True, text=True, errors="replace", env=env,
            shell=False)
        stderr = r.stderr if r.stderr else ""
        return r.returncode, stderr
    except _CgStageStalled as e:
        return 87, (f"Stage stalled: no progress for "
                    f"{e.inactivity_timeout:.0f}s "
                    f"(last marker={e.last_progress})")
    except subprocess.TimeoutExpired:
        return 87, f"Stage exceeded the final safety limit ({timeout}s)"
    except OSError as e:
        # R23-LONG-01：命令行越界在 Python 侧是 OSError(WinError 206)，
        # 必须单独说清楚。否则它落进下面的 `except Exception` 变成
        # "Stage 1 failed: 82"，运维只会去怀疑素材/滤镜，查不到真因。
        if getattr(e, "winerror", None) == 206:
            return 80, (
                "command line too long for CreateProcess (WinError 206): the "
                "stage arguments exceed the Windows 32767-character limit "
                "(%d characters of params were inlined)"
                % len(params_json))
        return 82, str(e)
    except Exception as e:
        return 82, str(e)  # IO error
    finally:
        # Clean up temporary params file
        if params_file and os.path.exists(params_file):
            try:
                os.unlink(params_file)
            except Exception:
                pass


def _get_video_duration(video_path: str) -> float:
    """Get video duration via ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False
        )
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return 0.0


def _verify_stage2_output(job_id: str, input_path: str,
                          temp_dir: str, job_logger) -> Tuple[bool, str]:
    """FIX-W2: validate stage2 traditional-restoration output.

    Checks the three §3.10.8 requirements: file exists, non-empty, and
    ffprobe-decodeable. Extracted from the inline exit-0 check so the exit-3
    (skip) path re-uses the same validations.

    FIX-W2(pattern): ORIGINAL used Path(input_path).name (WITH extension) in
    the glob pattern, but stage2 writes `{job_id}_{stem}_step2_traditional*.mkv`
    using Path.stem (NO extension) — see stage2_traditional_check.py:146 and
    stage3_ai_upscale.py:605. The old pattern never matched, so the gate
    always falsely reported "missing". Use .stem to align with stage2/stage3.

    Returns (True, path) on success or (False, reason) on failure.
    """
    import glob
    input_stem = Path(input_path).stem
    pattern = f"{temp_dir}/{job_id}_{input_stem}_step2*"
    # R4（关键修复）：与 stage2 内那个 glob 完全同类的问题 ——
    # 通配 `_step2*` 同样命中 sidecar（`...step2_traditional.mkv.identity.json`），
    # 而 sidecar 写在 mkv 之后、mtime 更晚，于是 candidates[0] 选中的是
    # 那个 .json，ffprobe 自然解不出 -> "Stage 2 output not decodeable"。
    # R3-01：黑名单元信息不全（挡不住未来的新侧车类型），统一收编到
    # checkpoint.select_media_candidates 的媒体白名单 + mtime 新→旧 排序。
    from checkpoint import select_media_candidates
    candidates = select_media_candidates(glob.glob(pattern))
    if not candidates:
        return False, "Stage 2 output file missing"
    step2_output = str(candidates[0])
    if os.path.getsize(step2_output) == 0:
        return False, "Stage 2 output file is empty"
    try:
        from checkpoint import verify_ffprobe_decodeable
        if not verify_ffprobe_decodeable(step2_output):
            return False, "Stage 2 output not decodeable by ffprobe"
    except Exception as e:
        return False, f"Stage 2 output verification failed: {e}"
    return True, step2_output


def _safe_input_stem(input_path: str) -> str:
    """FIX-V03: artifact identity contract (mirrors checkpoint.build_intermediate_name).

    ORIGINAL resume scans used ``Path(input_path).name`` (WITH extension) while
    every stage writes ``{job_id}_{Path(input).stem}_stepN*`` (see
    stage2_traditional_check.py:146, stage3_ai_upscale.py:1206,
    stage4_postprocess.py:138). The patterns never matched, so valid artifacts
    were reported missing. Generator and reader must use the same contract.
    """
    return Path(input_path).stem.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _snapshot_stage_files(temp_dir: str, job_id: str, input_path: str,
                          stage_num: int) -> Dict[str, Tuple[int, int]]:
    """拍下某个 step 的现有产物指纹（mtime_ns, size），用于区分"本次新增"。

    ID-05（第七轮）配套：登记身份时必须知道"哪些是这个 stage 刚写的"，
    不能把整个 glob 结果当成自己的产物。
    """
    out: Dict[str, Tuple[int, int]] = {}
    try:
        for p in find_stage_artifacts(temp_dir, job_id, input_path, stage_num):
            try:
                st = os.stat(p)
                out[p] = (st.st_mtime_ns, st.st_size)
            except OSError:
                pass
    except Exception:
        pass
    return out


def find_stage_artifacts(temp_dir: str, job_id: str, input_path: str,
                         stage_num: int) -> List[str]:
    """Return existing **media** artifacts of (job_id, stage_num), newest first.

    R3-01：按后缀白名单过滤侧车。第三轮新增的 `<媒体>.identity.json` 与媒体
    同名前缀、同目录，若不过滤会被下游当成视频输入（B10：真实 stage7 rc=84）。
    """
    import glob
    from checkpoint import select_media_candidates
    pattern = str(Path(temp_dir) / f"{job_id}_{_safe_input_stem(input_path)}_step{stage_num}*")
    candidates = glob.glob(pattern)
    return [str(p) for p in select_media_candidates(candidates)]


# --- RV-09: 产物身份（resume 不能再"只看名字 + 能解码"）------------------
#: 实现/恢复契约版本。改动产物语义时必须升版本，否则旧产物的身份无法区分。
#: R2-07：输入摘要从"首/尾采样"改成"全内容流式哈希"，身份语义变了，
#: 旧 marker 一律判为不属于当前实现（安全方向：重算，而不是错配）。
VIDEO_IMPL_VERSION = "video-pipeline/3.0-r2-07-full-content-identity"


def _input_identity_status(recorded: Any, current: Dict[str, Any]) -> str:
    """Compare a recorded input identity with the current one.

    Returns one of:
      * ``"match"``    —— 两边都拿到了摘要且相等；
      * ``"changed"``  —— 两边都拿到了摘要但不相等（源内容变了，禁止复用）；
      * ``"unproven"`` —— 任一侧拿不到摘要（缺失/读取失败/旧格式）。
        R2-07 要求这种情形**拒绝复用**，不能落回"文件名能对上"的猜测。
    """
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return "unproven"
    old_digest = recorded.get("digest")
    new_digest = current.get("digest")
    if not old_digest or not new_digest:
        return "unproven"
    return "match" if str(old_digest) == str(new_digest) else "changed"


def _chunked_sha256(path: str, chunk: int = 1 << 20) -> Optional[str]:
    """Content fingerprint: SHA-256 over the **whole** file (streamed).

    mtime 不能作为身份（复制/重写都会变），只用 size + 内容。

    R2-07 / A10：旧实现只取首 1MiB，并且**仅当 size > 2MiB** 才补读尾 1MiB。
    后果有两个盲区：
      * 1–2MiB 的文件：第二个 1MiB 之后的任何改动都看不见
        —— 实测 1.5MiB 文件在 1200KiB 处改 1024 字节，旧摘要**不变**；
      * >2MiB 的文件：只采首尾，中部改动同样看不见。
    采样指纹不能声称"内容未变"，这里改成流式全内容哈希。
    代价：O(文件大小)；SHA-256 吞吐约数百 MB/s，对素材库可接受。
    若将来确实需要性能折中，必须是显式开关 + 明确可靠性边界，不能默认采样。
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        h.update(str(os.path.getsize(path)).encode("ascii"))
        return h.hexdigest()
    except Exception:
        return None


#: R2-07：身份算法模式。写进 marker，便于将来区分"全量哈希"与任何显式折中模式。
IDENTITY_DIGEST_MODE = "full-stream-sha256"


def _plain_sha256(path: str, chunk: int = 1 << 20) -> Optional[str]:
    """**纯内容** SHA-256 —— 与 `publish_journal.file_digest` 同口径。

    R16-03 修正：`_chunked_sha256()` 在内容之后**又拼了 size 字符串**，
    与项目里其余摘要（publisher / journal / stage7 的 verify 报告 /
    artifact_verify）**不是同一套算法**。拿它去比报告里的 `media_sha256`
    必然不等 —— 实测同一份成品：纯内容 `3f1fbf3c…`、内容+size
    `bb52754f…`，于是**每个视频 job 的完成标记都被误拒**。

    用途区分：
      * 跨模块比对（对报告 / 对 journal）→ 本函数
      * 需要"内容 + 大小"的输入身份 → 继续用 `_chunked_sha256()`
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def build_input_identity(input_path: str) -> Dict[str, Any]:
    """RV-09 / R2-07: identity of the *source* material an artifact was made from."""
    ident: Dict[str, Any] = {"path": str(input_path), "size": None,
                             "digest": None, "mode": IDENTITY_DIGEST_MODE}
    try:
        st = os.stat(input_path)
        ident["size"] = st.st_size
    except Exception:
        pass
    ident["digest"] = _chunked_sha256(input_path)
    return ident


# ID-02（第七轮）：纯位置字段 —— 它们不影响媒体内容，不能进身份摘要。
# 把 temp_dir / 日志目录之类算进身份，会导致"只改了输出目录就触发全量重算"，
# 以及"同一份结果换个目录就认不出来"。
#
# R22-A03：**处理选择不能靠"从身份里删字段"来稳定**。
# 审核复现：`ivtc=VIVTC` 与 `ivtc=sRestore` 曾经给出**同一个摘要** ——
# 因为 R21 把这个键当成"运行期注入"排除了。可它同时承载用户的实际选择，
# 排除它等于宣称"两种反交错算法等价"。
#
# 正解是把三份数据分开（见 `_plan_snapshot`）：
#   ① 不可变用户请求（job.user_overrides / CLI）
#   ② 解析后有效计划 —— 在 stage 开始**之前**快照，参与身份
#   ③ 执行观测/降级记录 —— stage 跑完才注入，**不参与**身份
# 所以 `ivtc` 等键回到身份里；真正该排除的只有纯位置字段与
# `upstream_artifact`（上游精确产物路径，位置性质）。
_IDENTITY_POSITIONAL_KEYS = frozenset({
    "temp_dir", "log_dir", "output_dir", "output_path",
    "input_path", "log_path", "manifest_path", "params_file",
    "_effective_params", "job_id", "run_id", "attempt_id",
    "upstream_artifact",      # 上游精确产物路径（位置，不是选择）
})

#: 只在"执行之后"才被写入的键 —— 它们属于观测记录，不属于计划。
#: 注意：**不要**把它们塞进 `_IDENTITY_POSITIONAL_KEYS`，
#: 那是把"选择"和"位置"混为一谈的老问题。
_POST_EXECUTION_KEYS = frozenset({
    "ivtc_auto_fallback",     # 自动回退发生过的标记
    "ai_upscale_banned",      # 分类器结论（执行前由 classify 给出，随 job 进计划）
})


def _identity_semantics(params: Dict[str, Any]) -> Dict[str, Any]:
    """从**计划快照**里挑出影响媒体语义的部分，用于身份摘要。

    任务书 §2.3：『按语义规范化的有效参数；不要把日志路径、输出目录等
    纯位置字段当作媒体语义，导致无谓重算』。

    R22-A03：如果 `params["_plan_snapshot"]` 存在，就用它 —— 那是 stage
    开始**之前**冻结的视图，运行期的任何注入都不会改变它。
    """
    src = params or {}
    snap = src.get("_plan_snapshot")
    if isinstance(snap, dict):
        # 快照已经过滤过一遍；这里只防御性剔除位置键
        return {k: v for k, v in snap.items()
                if k not in _IDENTITY_POSITIONAL_KEYS}

    out: Dict[str, Any] = {}
    for k, v in src.items():
        if k in _IDENTITY_POSITIONAL_KEYS:
            continue
        # 没有快照时（resume 新进程 / 老调用）额外剔除"纯观测"键：
        # 它们只可能在执行后出现，新进程重算时本来就没有，不剔会永远对不上。
        if k in _POST_EXECUTION_KEYS:
            continue
        if k.startswith("_"):          # 内部临时键（_shake_probe 等）
            continue
        out[k] = v
    return out


#: 允许的执行组模式。未知值**不得静默归零**（审核 R19-B06）。
_KNOWN_CHAIN_MODES = frozenset({"none", "stage34"})


def _strict_bool(value, default: bool = False) -> bool:
    """把配置值解析成**严格**布尔。

    审核 R19-B06：
    > `bool('false')` 转为 True …… 同样未关闭。

    YAML 正常写 ``false`` 会得到 Python ``False``，但 CLI/env 传进来的
    ``"false"`` 是字符串 —— 直接 ``bool()`` 会变成 ``True``，
    等于"用户想关、系统却打开"。

    Args:
        value: 原始值（bool / str / int / None）。
        default: 无法识别时的返回值。

    Returns:
        bool
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off", ""):
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def build_params_identity(config: Dict[str, Any], job: Dict[str, Any],
                          logger=None,
                          params: Optional[Dict[str, Any]] = None) -> str:
    """RV-09: digest of the *effective* parameters of this job.

    **ID-02（第七轮）关键修复**：旧实现固定用
    `merge_params(job, config, {}, logger)` —— **空 CLI 覆盖**。
    而真正执行 stage 的 params 是带 CLI 覆盖的（`--ai-model`、
    `--ai-upscale-factor`、`--face-restore` …）。于是：

    ```
    实际执行：merge_params(job, config, {"ai_upscale_model": "x4plus"})
    身份摘要：merge_params(job, config, {})          # 忽略了 CLI
    ```

    用户用 CLI 换了模型，摘要却没变 → 旧产物被当作"同一身份"复用 →
    **记录与实际执行不符**。身份必须来自"实际执行的那一份参数"。

    参数来源优先级：
    1. 显式传入的 `params`（调用方手上就是执行用的那份）；
    2. `job["_effective_params"]`（merge 之后挂上去的同一对象引用）；
    3. 兜底：重新按无 CLI 覆盖合并（**仅为兼容旧调用**，会给
       `STAGE_IDENTITY_FALLBACK` 告警，便于发现遗漏的调用点）。
    """
    effective = params
    if effective is None:
        effective = job.get("_effective_params")
    if effective is None:
        try:
            effective = merge_params(job, config, {}, logger)
        except Exception:
            effective = dict(job or {})
        if logger:
            try:
                logger.warning(
                    "STAGE_IDENTITY_FALLBACK",
                    "build_params_identity() had no effective params at hand "
                    "and fell back to merging WITHOUT CLI overrides; the "
                    "resulting identity may not match what actually ran",
                    job_id=job.get("job_id", ""))
            except Exception:
                pass

    payload = json.dumps(_identity_semantics(effective), sort_keys=True,
                         ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


#: 模型文件的候选扩展名（只用于"给了名字没给路径"时定位）。
_MODEL_EXTS = (".pth", ".onnx", ".bin", ".pt", ".safetensors")


def _resolve_model_path(name_or_path: Any,
                        params: Optional[Dict[str, Any]] = None) -> str:
    """把 `upscale_model` / `face_restore` 这类值解析成**实际文件路径**。

    两个来源都可能：
      * 用户直接给了绝对/相对路径（审核 A04 的复现脚本就是这么做的）；
      * 用户给的是模型名（`realesr-general-x4v3`），实际文件在
        `params["model_dir"]` 或项目 `models/` 下，扩展名见 `_MODEL_EXTS`。

    解析不出来返回空串 —— **不抛异常**：模型名对不上文件是常见情况
    （走内置默认或下载缓存），那不该让身份计算失败。
    """
    raw = str(name_or_path or "").strip()
    if not raw:
        return ""
    if os.path.isfile(raw):
        return os.path.abspath(raw)

    roots: List[str] = []
    _md = str((params or {}).get("model_dir") or "").strip()
    if _md:
        roots.append(_md)
    roots.append(str(SCRIPT_DIR.parent / "models"))

    base = os.path.basename(raw)
    cands = [base]
    if not os.path.splitext(base)[1]:
        cands = [base + ext for ext in _MODEL_EXTS]
    for root in roots:
        for c in cands:
            p = os.path.join(root, c)
            if os.path.isfile(p):
                return os.path.abspath(p)
    return ""


def _model_file_digest(path: str) -> str:
    """模型文件的**内容**摘要。

    为什么需要它：`_impl_fingerprint` 原来只记模型**名字**。审核 R23-A04
    用独立反例证明——把同一路径的权重从 `WEIGHT-A` 换成 `WEIGHT-B`，
    RunIdentity **完全相同**。也就是说"换了权重"在身份里不可见，
    旧中间产物会被错误复用。

    **故意不做 `(size, mtime)` 缓存**。审核允许缓存，但要求"保守失效"：
    同一秒内替换成**同样大小**的另一份权重（审核的复现脚本就是
    `WEIGHT-A` → `WEIGHT-B`，字节数相同）会让 `(size, mtime)` 完全一致，
    缓存命中 → 正是要检出的场景被漏掉。而身份计算是低频操作
    （每个 job 1–3 次），一次全文件 sha256 对 140 MB 权重约 0.3 秒，
    相比整链几小时可以忽略 —— 用确定性换这点 I/O 是划算的。
    """
    p = str(path or "").strip()
    if not p:
        return ""
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()[:32]


def _impl_fingerprint(params: Optional[Dict[str, Any]] = None) -> str:
    """实现 / 模型 / 工具指纹（R22-A02）。

    审核指出：main 只给 `_run_identity.input/params`，stage7 却去取
    `.digest`，实际传空串。回执类严格要求身份，生产调用却没按它需要的
    形状提供 —— 这里补上"实现"这一维。

    只吃**稳定的**东西：实现版本、Python 主次版本、模型名 + **模型字节**、
    人脸后端。时间戳、路径、随机数一律不进（否则每次运行身份都不同）。

    R24-04 / R23-A04：原来只记模型名，同路径换权重身份不变。现在把模型
    文件的字节摘要并进来（解析不到文件时退回"只有名字"，行为同旧版）。
    """
    parts = [str(VIDEO_IMPL_VERSION), "py" + sys.version.split()[0]]
    if isinstance(params, dict):
        _um = str(params.get("upscale_model") or "")
        _fr = str(params.get("face_restore") or "")
        parts.append(_um)
        parts.append(_fr)
        # 模型**字节**身份（不是路径，不是 mtime）—— 内容变了身份才变
        for _label, _val in (("upscale", _um), ("face", _fr)):
            _fp = _resolve_model_path(_val, params)
            _d = _model_file_digest(_fp) if _fp else ""
            parts.append(f"{_label}-bytes:{_d or '-'}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


def build_run_identity(config: Dict[str, Any], job: Dict[str, Any],
                       logger=None,
                       params: Optional[Dict[str, Any]] = None
                       ) -> Dict[str, Any]:
    """构造**可重算**的 RunIdentity（R22-A02）。

    审核：`main` 生成的是 ``{input, params}``，没有 ``digest``；
    stage7 两处 publish 读 ``.get("digest")`` → 恒为空串；回执因此拿不到
    有效身份。这里给出固定四字段并从中导出摘要：

    - ``schema``
    - ``input_digest``（内容身份）
    - ``plan_digest``（规范化有效计划身份）
    - ``impl_fingerprint``（实现/模型/工具）

    摘要只由这四项算出（`_recompute_run_identity_digest`），
    人读用的 ``input`` 副本**不参与** —— 否则加一个字段就换身份。

    Returns:
        dict；``digest`` 一定非空。
    """
    _in = build_input_identity(job.get("input_path", ""))
    ri: Dict[str, Any] = {
        "schema": "run-identity/1",
        "input_digest": str((_in or {}).get("digest") or ""),
        "plan_digest": build_params_identity(config, job, logger,
                                             params=params),
        "impl_fingerprint": _impl_fingerprint(params),
        "input": _in,
    }
    ri["digest"] = _recompute_run_identity_digest(ri)
    return ri


def completion_marker_path(final_path: str) -> str:
    return str(final_path) + ".done.json"


def write_completion_marker(final_path: str, job: Dict[str, Any],
                            config: Dict[str, Any], logger=None,
                            params: Optional[Dict[str, Any]] = None) -> bool:
    """验证本代是否可判"完成"（R22-A01 起**只验证，不再写标记**）。

    名字没改，是因为调用点和既有测试都用它；但**行为**变了。

    旧实现在 `verify_published_marker` 成功后就 `return True`，于是：

    * 私有 result / attempt / terminal_pending 全部被跳过；
    * journal 终态根本没查；
    * 没有 marker 时还会"写一份新的"，等于事后补证据。

    审核用真实小文件复现：PREPARED 的 journal + 错 attempt + 不存在的
    result，这个函数仍返回 True 并让 `finish_job_completion` 记 JOB_DONE；
    而同一个对象上 `verify_generation_objects` 明确报
    "publish journal is not COMMITTED"。同一个合同两次判断不一致，
    这条路径本身就是缺陷。

    现在只做一件事：调 `verify_generation_completion()`（唯一入口）。
    标记只能由 stage7 在发布事务里生成；main 不再写、也不再覆盖。

    Returns:
        True 仅当：私有 result 一致 + marker payload 完整 + 整代磁盘事实成立。
    """
    _expect = str((params or {}).get("attempt_id") or "")
    ok, why = verify_generation_completion(
        final_path, job, config, params, logger, expect_attempt=_expect)
    if ok:
        if logger:
            logger.info(
                "COMPLETION_VERIFIED",
                "This generation is proven complete: the attempt-private "
                "result, the published marker payload and the on-disk "
                "objects all agree.",
                output_path=str(final_path))
        return True
    if logger:
        logger.error(
            "COMPLETION_UNPROVEN",
            "Refusing to treat this artifact as complete.",
            output_path=str(final_path), reason=why)
    return False


def required_generation_reports(final_path: str) -> List[str]:
    """一代的**必需报告**：少一份就不是完整代（R21-C02）。

    stage7 每次都会发布这两个对象（`Artifact("verify")` /
    `Artifact("fullchain")`），所以"媒体在、报告缺"必然意味着这一代被换过
    或者发布被打断 —— 不能当完成。
    """
    return [str(final_path) + ".verify.json",
            str(final_path) + ".fullchain.json"]


#: payload manifest 的三态（R22-A01）。旧格式**不能**当作"通过"。
MANIFEST_OK = "ok"
MANIFEST_LEGACY = "legacy"
MANIFEST_BAD = "bad"

#: 允许出现在 payload manifest 里的对象角色。
_MANIFEST_ROLES = ("media", "verify", "fullchain", "marker")


def check_payload_manifest(marker: Dict[str, Any], final_path: str,
                           logger=None, planned: Optional[Dict[str, Any]] = None
                           ) -> Tuple[str, str, list]:
    """核对 marker 引用的 payload manifest（R23-B02：集合由**计划**决定）。

    R22 的版本修掉了两个洞（旧格式直接放行、media 路径不看），但仍是
    **被检查者自报集合**：审核请求 `full_chain_verify=True`，把 fullchain
    从清单里删掉再删文件，验证照样通过 —— "清单没列它"就自动变成合法。

    现在必需集合来自 `_planned_generation_roles()`：

    - 每个**计划必需**的角色都必须在清单里，且路径等于规范路径；
    - 清单里的角色必须属于"计划必需 ∪ 计划可选"（多写陌生角色 = BAD）；
    - 可选角色（如关闭 fullchain 时）允许缺失；给了就必须自洽；
    - 每个条目的摘要必须与磁盘一致；media 摘要必须等于 `marker.output_sha256`。

    Returns:
        ``(state, reason, objects)``。
    """
    pl = planned if isinstance(planned, dict) else \
        _planned_generation_roles(str(final_path))
    req = {k: str(v) for k, v in (pl.get("required") or {}).items()}
    opt = {k: str(v) for k, v in (pl.get("optional") or {}).items()}
    # marker 自身不可能出现在自己的 payload 里，由 marker_digest 绑定
    req_in_payload = {k: v for k, v in req.items() if k != "marker"}

    pm = marker.get("payload_manifest")
    if not isinstance(pm, dict):
        return (MANIFEST_LEGACY,
                "marker has no payload manifest (legacy format); it cannot "
                "prove which generation these bytes belong to", [])

    recorded = str(marker.get("payload_manifest_digest") or "")
    if not recorded:
        return MANIFEST_BAD, "marker has a payload manifest but no digest", []
    text = json.dumps(pm, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    if _plain_sha256_text(text) != recorded:
        return (MANIFEST_BAD,
                "payload manifest does not match the digest in the marker", [])

    objs = pm.get("objects")
    if not isinstance(objs, list) or not objs:
        return MANIFEST_BAD, "payload manifest lists no objects", []

    allowed = set(req) | set(opt)
    seen = {}
    out = []
    for row in objs:
        if not isinstance(row, dict):
            return MANIFEST_BAD, "payload manifest has a malformed entry", []
        name = str(row.get("name") or "")
        path = str(row.get("path") or "")
        want = str(row.get("sha256") or "")
        if not name or not path or not want:
            return MANIFEST_BAD, f"manifest entry {name!r} is incomplete", []
        if name not in _MANIFEST_ROLES:
            return MANIFEST_BAD, f"manifest entry has unknown role {name!r}", []
        if name not in allowed:
            return (MANIFEST_BAD,
                    f"manifest lists role {name!r}, which this plan does not "
                    f"publish (allowed: {sorted(allowed)})", [])
        if name in seen:
            return MANIFEST_BAD, f"manifest lists role {name!r} twice", []
        _canon = req.get(name) or opt.get(name) or ""
        if _canon and os.path.abspath(path) != os.path.abspath(_canon):
            if name == "media":
                # 媒体这一条保留原来的措辞：它是最常被复现的绕过路径
                # （拿别处正确的文件冒充当前成品）。
                return (MANIFEST_BAD,
                        f"manifest media path is not the artifact under test "
                        f"({path} != {_canon})", [])
            return (MANIFEST_BAD,
                    f"manifest role {name!r} points at {path!r}, but this "
                    f"generation's canonical path is {_canon!r}", [])
        seen[name] = (path, want)
        got = _plain_sha256(path) if os.path.exists(path) else None
        if got is None:
            return MANIFEST_BAD, f"object {name!r} is missing: {path}", []
        if got != want:
            return (MANIFEST_BAD,
                    f"object {name!r} changed since it was published "
                    f"(recorded {want[:16]}, now {got[:16]})", [])
        out.append((name, path, want))

    # ---- 计划必需的角色一个都不能少 ----
    missing = sorted(set(req_in_payload) - set(seen))
    if missing:
        return (MANIFEST_BAD,
                "payload manifest is missing planned object(s): "
                f"{missing}", [])

    if "media" not in seen:
        return MANIFEST_BAD, "manifest has no media entry", []
    media_path, media_dig = seen["media"]
    if os.path.abspath(media_path) != os.path.abspath(str(final_path)):
        return (MANIFEST_BAD,
                f"manifest media path is not the artifact under test "
                f"({media_path} != {final_path})", [])
    if str(marker.get("output_sha256") or "") != media_dig:
        return MANIFEST_BAD, "marker output_sha256 does not match the media", []

    # payload 记录的 attempt 必须与 marker 顶层一致（同一代）
    _pm_att = str(pm.get("attempt_id") or "")
    _mk_att = str(marker.get("attempt_id") or "")
    if _pm_att and _mk_att and _pm_att != _mk_att:
        return (MANIFEST_BAD,
                f"payload manifest belongs to attempt {_pm_att!r}, marker "
                f"says {_mk_att!r}", [])

    return MANIFEST_OK, "payload manifest matches the published bytes", out


class _LockUnavailable(RuntimeError):
    """完成验证的锁依赖不可用（R23-B02）。

    旧实现 `return nullcontext()`：拿不到锁也照常跑完、照常返回成功，
    只在日志里写一条 error。审核用受控导入失败复现过 —— 这等于
    "以为持锁、实际没持"，而完成验证的全部意义就是锁内核对。
    """


class _FailingLock:
    """一个进入即失败的上下文：把"锁不可用"变成明确失败。"""

    def __init__(self, reason: str):
        self.reason = str(reason or "unknown")

    def __enter__(self):
        raise _LockUnavailable(self.reason)

    def __exit__(self, *exc):
        return False


def _acq_final(final_path: str, logger=None):
    """目标锁上下文（R23-B02：锁不可用 → **验证失败**，不再降级为成功）。

    与 R22 的区别只有一点，但是关键的一点：
    R22 拿不到锁时返回 `nullcontext()`，验证照跑、成功照给；
    现在返回一个进入即抛 `_LockUnavailable` 的上下文，由调用方转成
    `published_unverified`（不是 DONE）。
    """
    try:
        from publish_lock import acquire_target_locks as _acq
    except Exception as exc:                # pragma: no cover - 部署异常
        if logger:
            logger.error(
                "COMPLETION_LOCK_UNAVAILABLE",
                "The completion lock dependency is unavailable; this "
                "generation cannot be verified (lock-free success is not "
                "accepted).", output_path=str(final_path), error=str(exc))
        return _FailingLock(str(exc))
    try:
        return _acq([str(final_path)], timeout=120, logger=logger)
    except Exception as exc:                # pragma: no cover - 部署异常
        if logger:
            logger.error(
                "COMPLETION_LOCK_UNAVAILABLE",
                "Could not acquire the target lock for completion "
                "verification; the generation stays unverified.",
                output_path=str(final_path), error=str(exc))
        return _FailingLock(str(exc))


def _real_bool(value):
    """只接受**真布尔**（R23-B01）。``"false"`` / ``1`` / ``"true"`` 都不算。"""
    return value if isinstance(value, bool) else None


def _planned_generation_roles(final_path: str, config=None, params=None):
    """由**可信计划**推出这一代的必需/可选对象角色与规范路径（R23-B02）。

    审核复现：请求 `full_chain_verify=True`，把 fullchain 从 marker 清单里
    删掉再删文件，仍然 `JOB_DONE`。根因是必需集合取自**被检查者自报**的
    marker。这里改成从计划推：

    - `media` / `verify` / `marker` 永远必需；
    - `fullchain` 由 `full_chain_verify` 决定（缺省与 stage7 一致：开启）；
    - 关闭时它进 `optional`：**允许不存在**；存在就必须自洽（不能半坏）。

    Returns:
        ``{"required": {role: path}, "optional": {role: path},
           "full_chain_verify": bool}``。
    """
    p = str(final_path)
    full = _strict_bool((params or {}).get("full_chain_verify"), True)
    req = {
        "media": p,
        "verify": p + ".verify.json",
        "marker": completion_marker_path(p),
    }
    opt = {}
    if full:
        req["fullchain"] = p + ".fullchain.json"
    else:
        opt["fullchain"] = p + ".fullchain.json"
    return {"required": req, "optional": opt, "full_chain_verify": full}


def _resolve_execution_publication(
        data: Dict[str, Any], expect_attempt: str
) -> Tuple[bool, str, str]:
    """从私有结果里解出 **（执行编号 → 最终发布编号）**（R24-01 / R23-A01）。

    为什么需要这个函数
    ------------------
    `stage7` 在 WSL 重定向场景会做**两次发布**：
      1. 中转代 → 重定向路径，attempt = `<执行> -wsl` 的**前身**；
      2. 最终代 → 用户原路径，attempt = `<执行>` + `-wsl`。
    私有结果顶层记的是**执行**编号（`_attempt_id`），内层
    `generation_receipt` / `publish_result` / 完成 marker 记的是**最终发布**
    编号。旧实现要求两者完全相同，结果是把**正常回拷路径整个拒掉**：

        False: generation receipt belongs to attempt 'att1-wsl', not 'att1'

    审核用生产 publisher 复现了这条（`round23_completion_results.json` 的
    `parent_execution_child_publication`）。

    为什么不直接用后缀判断
    ----------------------
    审核明确要求：不能用 `endswith('-wsl')` 放行任意对象。那种判据对
    `att9-wsl` 也成立 —— 别的执行的对象搬过来照样过，等于没绑定。
    所以这里要求结果**显式声明**关系（`execution` 段），并且：

      * `direct`   —— 发布编号必须**等于**执行编号；
      * `copyback` —— 发布编号必须**恰好**是 `执行编号 + 受控后缀`，
                      且 `derived_from` 必须等于本次执行编号。

    两者都失败即拒。旧结果（没有 `execution` 段）按 direct 处理：
    发布编号 = 执行编号，与改动前行为一致（不放松）。

    Returns:
        ``(ok, reason, publication_attempt)``。
    """
    ex = data.get("execution")
    if not isinstance(ex, dict):
        # 旧结果没有声明。它只能是"直接发布"，绝不推断成回拷。
        return True, "legacy result: direct publication", str(expect_attempt)
    try:
        from receipts import (COPYBACK_ATTEMPT_SUFFIX as _CB,
                              EXECUTION_PUBLICATION_SCHEMA as _EP)
    except Exception:                        # pragma: no cover - 部署异常
        _CB = "-wsl"
        _EP = "execution-publication/1"

    if str(ex.get("schema") or "") != _EP:
        return (False,
                f"execution/publication schema is {ex.get('schema')!r}, "
                f"expected {_EP!r}", "")
    att = str(ex.get("attempt_id") or "")
    pub = str(ex.get("publication_attempt") or "")
    rel = str(ex.get("relation") or "")
    if not att or not pub:
        return False, ("the result declares an execution/publication "
                       "relation but carries no attempt ids"), ""
    if att != str(expect_attempt):
        return (False,
                f"result was produced by execution {att!r}, not the current "
                f"attempt {expect_attempt!r}", "")

    if rel == "direct":
        if pub != att:
            return (False,
                    "a direct publication must carry the execution's own "
                    f"attempt, got publication={pub!r} for execution {att!r}",
                    "")
        return True, "direct publication", pub

    if rel == "copyback":
        src = str(ex.get("derived_from") or "")
        if src != att:
            return (False,
                    f"the copyback publication claims to derive from {src!r}, "
                    f"not from this execution {att!r}", "")
        if pub != att + _CB:
            return (False,
                    f"copyback publication {pub!r} is not the controlled "
                    f"derivation of execution {att!r} "
                    f"(allowed: execution + {_CB!r})", "")
        return True, "copyback publication", pub

    return False, f"unknown execution/publication relation {rel!r}", ""


def load_private_result(final_path: str, expect_attempt: str,
                        params: Optional[Dict[str, Any]], logger=None,
                        current_identity: Optional[Dict[str, Any]] = None,
                        planned: Optional[Dict[str, Any]] = None
                        ) -> Tuple[bool, str, Dict[str, Any]]:
    """读取并严格核对 **attempt 私有发布结果**（R23-B01）。

    审核 R22-B01 用生产 marker/publisher + 自产小文件复现了八类"错误通过"。
    原来的实现有三个结构性问题：

    1. **类型用真值判断**：`ok="false"`（字符串）为真、`schema=True` 等于 1、
       `terminal_pending=0` 为假 —— 三个坏字段一起也照样 DONE。
    2. **RunIdentity 只与自身比**：`digest` 只要能从自己的四个字段重算出来
       就算通过，从不与**当前**任务/计划/输入比。于是"另一份输入的计划"
       也能冒充这一代。
    3. **`generation_receipt` / `publish_result` 根本没进判据**：result 里
       有没有这两份、内容对不对，都不影响结论。

    现在逐条改成**关系**：

    - 类型：`schema` 必须是真 int 1、`ok` 必须是真 True、
      `terminal_pending` 必须是真 False/0/None；
    - 身份：四个字段必须与**当前** RunIdentity 完全一致（不是自洽即可）；
    - 回执：必须存在、可反序列化、`is_complete()` 通过、attempt 一致、
      `run_identity` 等于当前身份摘要、`journal_phase=COMMITTED`；
    - 发布结果：必须存在、`ok is True`、`phase=COMMITTED`、
      `attempt_id` 一致、`journal_path` 等于规范路径、`published` 覆盖必需对象。

    Args:
        current_identity: 当前计划的身份（`build_run_identity` 的结果）。
            不给时按 "无法核对" 处理 —— 拒绝，而不是退回自洽检查。
        planned: `_planned_generation_roles()` 的结果；用于核对 published
            是否覆盖计划必需对象。

    Returns:
        ``(ok, reason, data)``；``data`` 仍返回原始 dict（诊断用），
        失败时调用方不得使用它做任何判断。
    """
    rp = str((params or {}).get("_stage7_result_path") or "")
    if not rp:
        return False, "no attempt-private result path was provided", {}
    if not os.path.exists(rp):
        return False, f"attempt-private result is missing: {rp}", {}
    try:
        with open(rp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        return False, f"attempt-private result is unreadable: {exc}", {}
    if not isinstance(data, dict):
        return False, "attempt-private result is not an object", {}

    # ---- 1) 严格类型 ----
    _schema = data.get("schema")
    if isinstance(_schema, bool) or not isinstance(_schema, int) or _schema != 1:
        return (False,
                f"result schema is {_schema!r}, expected the integer 1", {})
    got_att = str(data.get("attempt_id") or "")
    if got_att != str(expect_attempt):
        return (False,
                f"result belongs to attempt {got_att!r}, not {expect_attempt!r}",
                {})

    # ---- 1b) 执行编号 → 最终发布编号（R24-01 / R23-A01）----
    # 回拷时两者不同：顶层是执行，内层 receipt/publish_result/marker 是
    # 最终发布。下面所有"内层对象属于哪次尝试"的判据都用 publication。
    _rel_ok, _rel_why, _pub_att = _resolve_execution_publication(
        data, str(expect_attempt))
    if not _rel_ok:
        return False, _rel_why, {}

    fin = str(data.get("final_output_path") or "")
    if os.path.abspath(fin) != os.path.abspath(str(final_path)):
        return (False,
                f"result describes {fin!r}, not the artifact under test "
                f"{final_path!r}", {})
    if data.get("ok") is not True:
        return (False,
                f"result reports a failed publish (ok={data.get('ok')!r}; "
                f"only the boolean True counts)", {})
    _tp = data.get("terminal_pending")
    if not (_tp is False or _tp is None or
            (isinstance(_tp, int) and not isinstance(_tp, bool) and _tp == 0)):
        return (False,
                f"result says the terminal state was not persisted "
                f"(terminal_pending={_tp!r})", {})

    # ---- 2) RunIdentity：必须与**当前**身份一致，不是自洽即可 ----
    ri = data.get("run_identity")
    if not isinstance(ri, dict):
        return False, "result carries no run_identity object", {}
    want = str(ri.get("digest") or "")
    if not want:
        return False, "run_identity has no digest", {}
    _re = _recompute_run_identity_digest(ri)
    if _re != want:
        return (False,
                f"run_identity digest cannot be reproduced "
                f"(recorded {want[:16]}, recomputed {str(_re)[:16]})", {})
    if not isinstance(current_identity, dict) or not current_identity:
        return (False,
                "no current run identity was supplied; the result's identity "
                "cannot be bound to this plan", {})
    _cur_digest = str(current_identity.get("digest") or "")
    if want != _cur_digest:
        _fields = [(k, str(ri.get(k) or "")[:16],
                    str(current_identity.get(k) or "")[:16])
                   for k in ("schema", "input_digest", "plan_digest",
                             "impl_fingerprint")
                   if str(ri.get(k) or "") != str(current_identity.get(k) or "")]
        return (False,
                "run_identity does not match the current plan "
                f"(recorded {want[:16]}, current {_cur_digest[:16]}; "
                f"differing fields: {[f[0] for f in _fields] or 'digest only'})",
                {})

    # ---- 3) GenerationReceipt：必须存在并自证 ----
    _gr_raw = data.get("generation_receipt")
    if not isinstance(_gr_raw, dict):
        return False, ("result carries no generation receipt; the published "
                       "generation is not proven"), {}
    try:
        from receipts import load_receipt as _load_receipt
    except Exception as exc:                # pragma: no cover - 部署异常
        return False, f"cannot load the receipt module: {exc}", {}
    _gr = _load_receipt(dict(_gr_raw, kind="GenerationReceipt"))
    if _gr is None:
        return False, ("generation receipt cannot be deserialized (schema or "
                       "declared digest is not trustworthy)"), {}
    _gr_ok, _gr_why = _gr.is_complete()
    if not _gr_ok:
        return False, f"generation receipt is incomplete: {_gr_why}", {}
    if str(_gr.attempt_id) != str(_pub_att):
        return (False,
                f"generation receipt belongs to attempt {_gr.attempt_id!r}, "
                f"not the final publication {_pub_att!r} "
                f"(execution {expect_attempt!r}, relation: {_rel_why})", {})
    if str(_gr.run_identity) != _cur_digest:
        return (False,
                "generation receipt was produced for a different run identity "
                f"({str(_gr.run_identity)[:16]} != {_cur_digest[:16]})", {})

    # ---- 4) PublishResult：必须存在且与本次执行一致 ----
    _pr = data.get("publish_result")
    if not isinstance(_pr, dict):
        return False, "result carries no structured publish result", {}
    if _pr.get("ok") is not True:
        return (False,
                f"publish result does not report success (ok={_pr.get('ok')!r})",
                {})
    _ph = str(_pr.get("phase") or "")
    if _ph != "COMMITTED":
        return False, f"publish result phase is {_ph!r}, not COMMITTED", {}
    if str(_pr.get("attempt_id") or "") != str(_pub_att):
        return (False,
                f"publish result belongs to attempt "
                f"{_pr.get('attempt_id')!r}, not the final publication "
                f"{_pub_att!r}", {})
    _jp = str(_pr.get("journal_path") or "")
    _want_jp = _journal_path_for(str(final_path))
    if os.path.abspath(_jp) != os.path.abspath(_want_jp):
        return (False,
                f"publish result points at journal {_jp!r}, expected "
                f"{_want_jp!r}", {})
    _pub = {os.path.abspath(str(x)) for x in (_pr.get("published") or [])}
    _req = {os.path.abspath(str(x))
            for r, x in (planned or {}).get("required", {}).items()
            if r != "marker"}
    if not _pub:
        return False, "publish result lists no published objects", {}
    if _req:
        _miss = sorted(_req - _pub)
        if _miss:
            return (False,
                    "publish result does not cover required object(s): "
                    f"{[os.path.basename(x) for x in _miss[:3]]}", {})

    return True, "private result is consistent with this attempt", data


def _journal_path_for(final_path: str) -> str:
    """这一代的 journal 规范路径（publisher 就是写在这里）。"""
    try:
        from publish_journal import journal_path_for as _jpf
        return str(_jpf(str(final_path)))
    except Exception:                       # pragma: no cover
        return str(final_path) + ".journal.json"


def _recompute_run_identity_digest(ri: Dict[str, Any]) -> str:
    """从 RunIdentity 的**固定四字段**重算摘要（R22-A02）。

    只吃 schema / input_digest / plan_digest / impl_fingerprint ——
    不把整个 dict 塞进去，否则加一个便于人读的字段就会改变身份。
    """
    core = {
        "schema": str(ri.get("schema") or ""),
        "input_digest": str(ri.get("input_digest") or ""),
        "plan_digest": str(ri.get("plan_digest") or ""),
        "impl_fingerprint": str(ri.get("impl_fingerprint") or ""),
    }
    return _plain_sha256_text(
        json.dumps(core, sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")))


def verify_generation_completion(final_path: str, job: Dict[str, Any],
                                 config: Dict[str, Any],
                                 params: Optional[Dict[str, Any]] = None,
                                 logger=None,
                                 expect_attempt: Optional[str] = None
                                 ) -> Tuple[bool, str]:
    """**唯一**的整代完成证明入口（R23-B01/B02 重写）。

    审核 R22 的判词是准的：

    > 几份各自自洽的记录，没有被验证成"当前请求的同一次执行、同一份最终产物"。

    原来的实现按"字段齐不齐"验证；现在按**关系**验证：

    ```
    计划必需角色  == marker.payload.objects  == journal.targets ⊆ receipt.required
    result.attempt == marker.attempt == journal.attempt == 本次 attempt（正常）
    receipt.run_identity == 当前计划的 RunIdentity 摘要
    receipt.marker_digest == 当前 marker 文件的摘要
    每个必需角色的磁盘摘要 == marker 记录 == receipt 记录
    ```

    执行顺序固定、无提前返回 True：

    1. 目标是锁集合内（**锁不可用 = 未验证**，不是成功）
    2. 解析**计划**：必需/可选对象由 `_planned_generation_roles()` 决定
    3. 正常收尾 → 读 attempt 私有结果（含 receipt / publish_result / 身份）
    4. marker 身份 + payload 三态（集合、路径、摘要、attempt 一致）
    5. journal 终态：COMMITTED + resolved 真布尔 + targets 覆盖计划必需对象
    6. 回执对象摘要 vs 磁盘
    7. 整代磁盘事实

    Args:
        expect_attempt: 正常收尾传本次 attempt；resume 传 None。resume 允许
            跨 attempt，但仍要求"同一份完整代"（result 存在时它的 attempt
            必须等于 marker/journal 的 attempt）。

    Returns:
        ``(ok, reason)``。
    """
    p = str(final_path)
    if expect_attempt is not None and not str(expect_attempt).strip():
        return False, "completion requires a non-empty attempt id"

    planned = _planned_generation_roles(p, config, params)
    try:
        with _acq_final(p, logger):
            return _verify_generation_locked(p, job, config, params, logger,
                                             expect_attempt, planned)
    except _LockUnavailable as exc:
        # 锁不可用 = **未验证**。不能返回同一个成功（R23-B02）。
        if logger:
            logger.error(
                "COMPLETION_UNVERIFIED_NO_LOCK",
                "The target lock is unavailable, so this generation cannot be "
                "verified. It stays unverified rather than becoming a success.",
                output_path=p, reason=str(exc))
        return False, ("completion lock is unavailable; the generation cannot "
                       f"be verified: {exc}")


def _verify_generation_locked(p: str, job: Dict[str, Any],
                              config: Dict[str, Any],
                              params: Optional[Dict[str, Any]], logger,
                              expect_attempt: Optional[str],
                              planned: Dict[str, Any]) -> Tuple[bool, str]:
    """锁内执行的实际核对（拆出来只是为了在 `with` 里保持一段直线代码）。"""
    # ---- 当前计划身份（唯一来源：main 冻结的那份；退化时才现算）----
    _cur = (params or {}).get("_run_identity")
    if not isinstance(_cur, dict) or not str(_cur.get("digest") or ""):
        _cur = build_run_identity(config, job, logger, params=params)

    # ---- 1) 私有 result（正常收尾必须；resume 只在存在时作一致性核对）----
    _res: Dict[str, Any] = {}
    if expect_attempt is not None:
        ok, why, _res = load_private_result(
            p, str(expect_attempt), params, logger,
            current_identity=_cur, planned=planned)
        if not ok:
            return False, why
    else:
        _rp = str((params or {}).get("_stage7_result_path") or "")
        if _rp and os.path.exists(_rp):
            _res = _load_result_for_consistency(_rp, p)
            if _res is None:
                return (False,
                        "a private result exists but cannot be read; resume "
                        "cannot prove which generation it describes")

    # ---- 1b) 执行编号 → 最终发布编号（R24-01 / R23-A01）----
    # marker / journal / receipt 记的都是**最终发布**编号；回拷时它等于
    # 执行编号 + `-wsl`。下面所有"这一代属于哪次尝试"的判据都用 publication，
    # 否则正常的 WSL 回拷会被自己拒掉（审核实测 A01）。
    # resume 没有当前执行编号时，以私有结果里的执行编号为基准。
    _pub_att = str(expect_attempt or "")
    if _res:
        _base_att = str(expect_attempt or _res.get("attempt_id") or "")
        if _base_att:
            _rel_ok, _rel_why, _pub_att = _resolve_execution_publication(
                _res, _base_att)
            if not _rel_ok:
                return False, _rel_why

    # ---- 2) marker 存在且身份字段对得上 ----
    mp = completion_marker_path(p)
    if not os.path.exists(mp):
        return False, ("no published completion marker; this generation "
                       "cannot be proven (re-run rather than trusting it)")
    try:
        with open(mp, "r", encoding="utf-8") as f:
            mdata = json.load(f) or {}
    except (OSError, ValueError) as exc:
        return False, f"completion marker is unreadable: {exc}"
    if not isinstance(mdata, dict):
        return False, "completion marker is not an object"

    _st = _input_identity_status(mdata.get("input_identity") or {},
                                 build_input_identity(job.get("input_path", "")))
    if _st == "changed":
        return False, "input content changed since this generation"
    if _st == "unproven":
        return False, "input identity cannot be verified"
    if str(mdata.get("job_id") or "") != str(job.get("job_id") or ""):
        return (False,
                f"marker belongs to job {mdata.get('job_id')!r}, "
                f"not {job.get('job_id')!r}")
    if str(mdata.get("impl") or "") != VIDEO_IMPL_VERSION:
        return (False,
                f"marker was produced by {mdata.get('impl')!r}, "
                f"current is {VIDEO_IMPL_VERSION!r}")

    _rec_params = str(mdata.get("params_identity") or "")
    if not _rec_params:
        return False, "marker has no params identity"
    _cur_params = str((params or {}).get("_frozen_params_identity") or "")
    if not _cur_params:
        _cur_params = build_params_identity(config, job, logger, params=params)
    if _rec_params != _cur_params:
        _rec_view = mdata.get("params_semantics") or {}
        _cur_view = (_identity_semantics(params)
                     if isinstance(params, dict) else {})
        if _rec_view and _cur_view and logger:
            _diff = [(k, _rec_view.get(k), _cur_view.get(k))
                     for k in sorted(set(_rec_view) | set(_cur_view))
                     if _rec_view.get(k) != _cur_view.get(k)]
            if _diff:
                logger.error(
                    "COMPLETION_PARAMS_DIFF",
                    "The marker records a different effective-parameter set; "
                    "showing the first differences", differences=_diff[:6])
        return False, "effective parameters changed since this generation"

    # ---- 3) payload：集合由计划决定 ----
    state, why, objects = check_payload_manifest(mdata, p, logger,
                                                 planned=planned)
    if state == MANIFEST_LEGACY:
        return False, f"legacy marker cannot prove this generation: {why}"
    if state != MANIFEST_OK:
        return False, why

    # ---- 3b) marker 的 attempt：必须能绑定到一次具体执行 ----
    # 放在 payload 检查**之后**：旧格式 marker（没有 payload/attempt）应当
    # 明确报 "legacy"，而不是被这一条抢先说成"没有 attempt"—— 诊断信息
    # 要指向真正的缺口。
    _mk_att = str(mdata.get("attempt_id") or "")
    if not _mk_att:
        return False, ("marker carries no attempt id; it cannot be bound to "
                       "an execution")
    if expect_attempt is not None and _mk_att != _pub_att:
        return (False,
                f"marker belongs to attempt {_mk_att!r}, not the final "
                f"publication {_pub_att!r} of the current attempt "
                f"{expect_attempt!r}")

    # ---- 4) journal：终态 + 覆盖 + 同一 attempt ----
    jp = _journal_path_for(p)
    if not os.path.exists(jp):
        return False, ("no publish journal for this generation; its terminal "
                       "state cannot be proven (legacy/unproven)")
    try:
        from publish_journal import read_journal as _rj
        _doc = _rj(jp)
    except Exception as exc:                # pragma: no cover - 部署异常
        return False, f"cannot verify the publish journal: {exc}"
    if not isinstance(_doc, dict):
        return False, "publish journal is unreadable"
    _phase = str(_doc.get("phase") or "")
    if _phase != "COMMITTED":
        return False, f"publish journal is not COMMITTED (phase={_phase!r})"
    # `resolved` 只出现在**恢复**写的终态里；正常 COMMITTED 由 phase 表达。
    # 所以：缺席 = 正常；出现就必须是真布尔 True（`"false"` 字符串不算）。
    if "resolved" in _doc and _doc.get("resolved") is not True:
        return (False,
                "publish journal is unresolved "
                f"(resolved={_doc.get('resolved')!r}; only boolean True counts)")
    if str(_doc.get("attempt_id") or "") != _mk_att:
        return (False,
                f"journal belongs to attempt {_doc.get('attempt_id')!r}, "
                f"marker says {_mk_att!r}")
    if job is not None:
        _want_j = str(job.get("job_id") or "")
        _got_j = str(_doc.get("job_id") or "")
        if _want_j and _got_j and _want_j != _got_j:
            return (False,
                    f"journal belongs to job {_got_j!r}, not {_want_j!r}")
    _targets = [str(x) for x in (_doc.get("targets") or [])]
    if not _targets:
        return False, ("publish journal records no targets; it cannot prove "
                       "which objects were committed")
    _t_set = {os.path.abspath(x) for x in _targets}
    _need = {os.path.abspath(str(v))
             for k, v in (planned.get("required") or {}).items() if k != "marker"}
    _miss = sorted(_need - _t_set)
    if _miss:
        return (False,
                "publish journal does not cover planned object(s): "
                f"{[os.path.basename(x) for x in _miss[:3]]}")

    # ---- 5) resume：result 若存在，必须描述**同一次发布** ----
    # 比的是**发布编号**：回拷结果里 `_res["attempt_id"]` 是执行编号，
    # 而 marker 记的是最终发布编号（执行 + `-wsl`）；直接比会把正常的
    # 回拷代判成"两份不同代"。用 `_pub_att` 比才是同一件事。
    if expect_attempt is None and _res:
        if _pub_att and _pub_att != _mk_att:
            return (False,
                    f"the private result points at publication {_pub_att!r} "
                    f"but the published marker is attempt {_mk_att!r}")

    # ---- 6) 回执：marker 摘要 + 每个必需对象的摘要 ----
    _gr_raw = _res.get("generation_receipt") if _res else None
    if isinstance(_gr_raw, dict):
        try:
            from receipts import load_receipt as _lr
            _gr = _lr(dict(_gr_raw, kind="GenerationReceipt"))
        except Exception:
            _gr = None
        if _gr is None:
            return False, "generation receipt is not trustworthy"
        _mk_dig = _plain_sha256(mp)
        if str(_gr.marker_digest) != str(_mk_dig):
            return (False,
                    "generation receipt was written for a different marker "
                    f"({str(_gr.marker_digest)[:16]} != {str(_mk_dig)[:16]})")
        for name, path, digest in objects:
            _rec = (_gr.object_digests or {}).get(path)
            if _rec is None:
                _rec = (_gr.object_digests or {}).get(os.path.abspath(path))
            if _rec is None:
                return (False,
                        f"generation receipt does not bind object {name!r}")
            if str(_rec) != str(digest):
                return (False,
                        f"generation receipt digest for {name!r} does not "
                        f"match the marker/disk")

    # ---- 7) 整代磁盘事实 ----
    return verify_generation_objects(p, job, logger,
                                     expected_objects=objects,
                                     marker=mdata,
                                     config=config, params=params,
                                     planned=planned)


def _load_result_for_consistency(result_path: str, final_path: str
                                 ) -> Optional[Dict[str, Any]]:
    """resume 用：只读 result 的 attempt/最终路径，不做"本次执行"要求。"""
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    fin = str(data.get("final_output_path") or "")
    if fin and os.path.abspath(fin) != os.path.abspath(str(final_path)):
        return None
    return data


#: verify 报告里**每次发布都必须有**的检查项（R24-02 / R23-A02）。
#:
#: 这三项回答的是"这一代到底做没做机械核对、帧顺序核对、最终时间轴测量"。
#: 缺任何一项，报告就没有证明它声称的事 —— 而旧实现只看"报告非空 +
#: media_sha256 对得上"，于是一份写着 `required=True/status=FAIL` 的报告
#: 照样能让完成验证返回 True。
#:
#: 名字必须与 stage7 的 `_checks[...]` 赋值点一致（见
#: `stage7_encode.py` 的 `_checks["mechanical"] / ["frame_order"] /
#: ["final_timeline"]`）。
_ALWAYS_REQUIRED_CHECKS = ("mechanical", "frame_order", "final_timeline")

#: 只有计划要求时才必需的检查项：检查名 → `_planned_generation_roles()`
#: 里的角色名。角色在 `required` 里才要求报告必须有这一项。
_PLAN_REQUIRED_CHECKS = {
    "fullchain": "fullchain",
}

#: 检查结论里**算合格**的状态。`SKIP` 不算 —— 跳过不等于通过；
#: `UNVERIFIED` 也不算（它是"没验到"，不是"验过了"）。
_CHECK_PASSING = frozenset({"PASS", "NOT_APPLICABLE"})

#: 检查结论里**认识**的状态。认不出的状态按问题上报，不静默忽略。
_CHECK_KNOWN = frozenset({"PASS", "FAIL", "UNVERIFIED", "NOT_APPLICABLE",
                          "SKIP", "ERROR", "WARN"})


def _verify_report_problems(doc: Dict[str, Any], *,
                            media_digest: str = "",
                            identity_digest: str = "",
                            publication_attempt: str = "",
                            planned: Optional[Dict[str, Any]] = None
                            ) -> List[str]:
    """`<media>.verify.json` 的**语义**问题（R24-02）。

    为什么不能只验字节
    ------------------
    审核 R23-A02 用生产 publisher 造了一整代：摘要完全正确，但报告里写着
    错误 schema、外来 run/attempt、`required=True/status=FAIL`、
    `publish=False`，fullchain 也 `ok=False`。正式完成入口仍返回

        True: media + planned reports + COMMITTED journal (3 objects)

    也就是说，最后一道验证**没有验证它声称要验证的结果**。摘要只能证明
    "这份报告属于这个文件"，不能证明"报告说这一代合格"。

    Returns:
        问题列表；空列表才表示报告的语义也成立。
    """
    probs: List[str] = []
    if not isinstance(doc, dict) or not doc:
        return ["verify report is empty"]

    # ---- 关联身份：报告必须属于**这次执行/这次计划** ----
    _att = str(doc.get("attempt_id") or "")
    if publication_attempt and _att and _att != publication_attempt:
        probs.append(
            f"verify report belongs to attempt {_att!r}, not the final "
            f"publication {publication_attempt!r}")
    _ri = doc.get("run_identity")
    if isinstance(_ri, dict):
        _rid = str(_ri.get("digest") or "")
        if identity_digest and _rid and _rid != identity_digest:
            probs.append("verify report was written for a different run "
                         "identity than the current plan")
    elif identity_digest:
        probs.append("verify report carries no run identity; it cannot be "
                     "bound to the plan that produced this media")

    # ---- 媒体身份 ----
    _md = str(doc.get("media_sha256") or "")
    if media_digest and _md and _md != media_digest:
        probs.append("verify report describes different media bytes")

    # ---- 检查项：集合、状态、必需性 ----
    raw_checks = doc.get("checks")
    if not isinstance(raw_checks, dict) or not raw_checks:
        probs.append("verify report carries no check results; it does not "
                     "show that anything was actually checked")
        raw_checks = {}

    seen: Dict[str, str] = {}
    for name, row in raw_checks.items():
        if not isinstance(row, dict):
            probs.append(f"check {name!r} is not an object")
            continue
        st = str(row.get("status") or "").upper()
        seen[str(name)] = st
        if st not in _CHECK_KNOWN:
            probs.append(f"check {name!r} has an unknown status {st!r}")
        # 报告自己声明 required 的项，必须真的合格
        if row.get("required") is True and st not in _CHECK_PASSING:
            probs.append(f"check {name!r} is required but reports {st!r}")

    _roles = (planned or {}).get("required") or {}
    must = list(_ALWAYS_REQUIRED_CHECKS)
    for chk, role in _PLAN_REQUIRED_CHECKS.items():
        if role in _roles:
            must.append(chk)
    for name in must:
        if name not in seen:
            probs.append(f"required check {name!r} is missing from the report")
        elif seen[name] not in _CHECK_PASSING:
            probs.append(f"required check {name!r} did not pass "
                         f"(status={seen[name]!r})")

    # ---- 报告自己的发布决定 ----
    dec = doc.get("publish_decision")
    if not isinstance(dec, dict):
        probs.append("verify report carries no publish decision; the "
                     "conclusion it reached cannot be checked")
    else:
        if dec.get("publish") is not True:
            probs.append("the report's own publish decision is "
                         f"{dec.get('publish')!r}, not True")
        # `verified=False` 不在这里拦：它是"用户显式强制发布"的合法结果
        # （`allow_verify_fail_publish`）。但它必须被上层看到 —— 由
        # `_verify_generation_locked` 记进 verdict，而不是包装成"已验证"。
    return probs


def _fullchain_report_problems(doc: Dict[str, Any], *,
                               publication_attempt: str = ""
                               ) -> List[str]:
    """`<media>.fullchain.json` 的语义问题（R24-02）。

    两种合法形态：

    * `status="NOT_APPLICABLE"` —— 用户关掉了全链检查，报告**明确**说这一代
      没有全链证据。允许（与"计划里 fullchain 可选"一致），但不算通过。
    * 真实检查结果 —— `ok` 必须是 True，且每一项 `checks[*].ok` 不能是 False。

    审核反例给的是 `{'ok': False, 'run_identity': 'foreign', ...}`，
    旧实现照样放行。这里 `ok is not True` 直接报问题。
    """
    probs: List[str] = []
    if not isinstance(doc, dict) or not doc:
        return ["fullchain report is empty"]

    if str(doc.get("status") or "").upper() == "NOT_APPLICABLE":
        _att = str(doc.get("attempt_id") or "")
        if publication_attempt and _att and _att != publication_attempt:
            probs.append(
                f"fullchain report belongs to attempt {_att!r}, not the final "
                f"publication {publication_attempt!r}")
        return probs

    if doc.get("hard_error"):
        probs.append(f"fullchain report records a hard error "
                     f"({doc.get('hard_error')!r})")
    if doc.get("ok") is not True:
        probs.append("fullchain report says ok="
                     f"{doc.get('ok')!r}, not True")
    checks = doc.get("checks")
    if isinstance(checks, dict):
        for name, row in checks.items():
            if isinstance(row, dict) and row.get("ok") is False:
                probs.append(f"fullchain check {name!r} failed")
    elif checks is not None:
        probs.append("fullchain report's checks is not an object")
    return probs


def verify_generation_objects(final_path: str, job: Optional[Dict[str, Any]] = None,
                              logger=None, expected_objects=None,
                              marker: Optional[Dict[str, Any]] = None,
                              config=None, params=None,
                              planned: Optional[Dict[str, Any]] = None
                              ) -> Tuple[bool, str]:
    """核对**磁盘上这一代**是否完整（R23-B02：集合由计划决定）。

    与 R22 的区别：

    - 兜底集合不再写死"两份报告"，而是由 `_planned_generation_roles()`
      给出（`full_chain_verify` 关闭时 fullchain 进可选）；
    - `verify` 报告的 `media_sha256` **必须存在**（空 `{}` 不再放过）；
    - journal 的 `resolved` 必须是真布尔 True（字符串 `"false"` 之前能过）；
    - journal 的 `targets` **必须非空**并覆盖计划必需对象（空列表之前被
      `if _targets:` 跳过，等于"没记录也算过"）。

    Returns:
        ``(ok, reason)``。
    """
    p = str(final_path)
    if not os.path.exists(p):
        return False, "media is missing"
    try:
        if os.path.getsize(p) == 0:
            return False, "media is empty"
    except OSError as exc:
        return False, f"media is unreadable: {exc}"

    pl = planned if isinstance(planned, dict) else \
        _planned_generation_roles(p, config, params)
    _req_roles = dict(pl.get("required") or {})
    _opt_roles = dict(pl.get("optional") or {})

    # ---- 期望集合 ----
    if expected_objects:
        objs = [(str(n), str(pp), str(d)) for (n, pp, d) in expected_objects]
    else:
        media_digest = _plain_sha256(p)
        if not media_digest:
            return False, "media is unreadable"
        objs = [("media", p, media_digest)]
        for _role, rp in _req_roles.items():
            if _role in ("media", "marker"):
                continue
            d = _plain_sha256(str(rp))
            if d is None:
                return False, ("required report is missing: "
                               f"{os.path.basename(str(rp))}")
            objs.append((_role, str(rp), d))

    media = [o for o in objs if o[0] == "media"]
    if len(media) != 1:
        return False, "the expected object set has no unique media entry"
    media_digest = media[0][2]

    for name, path, _d in objs:
        if not os.path.exists(path):
            return False, f"object {name!r} is missing: {path}"

    # ---- 计划必需的角色一个都不能少 ----
    _have = {o[0] for o in objs}
    _miss_roles = sorted(set(_req_roles) - {"marker"} - _have)
    if _miss_roles:
        return (False,
                f"the verified object set lacks planned role(s): {_miss_roles}")

    # ---- verify 报告必须描述**当前**媒体，且必须给出摘要 ----
    verify_rows = [o for o in objs if o[0] == "verify"]
    if not verify_rows:
        return False, "the required object set has no verify report"
    try:
        with open(verify_rows[0][1], "r", encoding="utf-8") as f:
            _v = json.load(f) or {}
    except (OSError, ValueError) as exc:
        return False, f"verify report is unreadable: {exc}"
    if not isinstance(_v, dict) or not _v:
        return False, ("verify report is empty; it does not describe any "
                       "measurement of this media")
    recorded = _v.get("media_sha256")
    if not recorded:
        return False, ("verify report does not record media_sha256; it cannot "
                       "prove it describes this media")
    if media_digest and str(recorded) != str(media_digest):
        return False, ("verify report describes different media bytes "
                       f"(recorded {str(recorded)[:16]}, "
                       f"actual {str(media_digest)[:16]})")

    # ---- R24-02：报告的**语义**也要成立 ----
    # 上面只证明"这份报告属于这个文件"；这里证明"报告说这一代合格"。
    # 审核反例：摘要正确、内容写着 required=True/status=FAIL、
    # publish=False，旧实现仍返回 True。
    _ident_digest = ""
    try:
        _ident_digest = str(((params or {}).get("_run_identity") or {})
                            .get("digest") or "")
    except AttributeError:
        _ident_digest = ""
    _pub_att = ""
    if isinstance(marker, dict):
        _pub_att = str(marker.get("attempt_id") or "")
    if not _pub_att:
        _pub_att = str((params or {}).get("attempt_id") or "")

    _sem = _verify_report_problems(
        _v, media_digest=str(media_digest or ""),
        identity_digest=_ident_digest,
        publication_attempt=_pub_att, planned=pl)
    if _sem:
        return (False,
                "the verify report does not support completion: "
                + "; ".join(_sem[:4]))

    # ---- R24-02：全链报告的内容结论 ----
    _fc_rows = [o for o in objs if o[0] == "fullchain"]
    if _fc_rows:
        try:
            with open(_fc_rows[0][1], "r", encoding="utf-8") as f:
                _fc_doc = json.load(f) or {}
        except (OSError, ValueError) as exc:
            return False, f"fullchain report is unreadable: {exc}"
        _fcp = _fullchain_report_problems(_fc_doc,
                                          publication_attempt=_pub_att)
        if _fcp:
            return (False,
                    "the fullchain report does not support completion: "
                    + "; ".join(_fcp[:4]))

    # ---- journal 终态 ----
    try:
        from publish_journal import read_journal as _rj
        jp = _journal_path_for(p)
        if not os.path.exists(jp):
            return False, ("no publish journal for this generation; its "
                           "terminal state cannot be proven (legacy/unproven)")
        _doc = _rj(jp)
        if not isinstance(_doc, dict):
            return False, "publish journal is unreadable"
        phase = str(_doc.get("phase") or "")
        if phase != "COMMITTED":
            return False, f"publish journal is not COMMITTED (phase={phase!r})"
        # 同上：恢复路径才写 `resolved`；正常 COMMITTED 由 phase 表达。
        if "resolved" in _doc and _doc.get("resolved") is not True:
            return (False,
                    "publish journal is unresolved "
                    f"(resolved={_doc.get('resolved')!r})")
        if job is not None:
            _want = str(job.get("job_id") or "")
            _got = str(_doc.get("job_id") or "")
            if _want and _got and _want != _got:
                return False, (f"journal belongs to job {_got!r}, "
                               f"not {_want!r}")
        # marker 与 journal 必须描述同一次执行
        if isinstance(marker, dict):
            _mk_att = str(marker.get("attempt_id") or "")
            _j_att = str(_doc.get("attempt_id") or "")
            if _mk_att and _j_att and _mk_att != _j_att:
                return (False,
                        f"journal belongs to attempt {_j_att!r}, marker says "
                        f"{_mk_att!r}")
        # ---- journal 必须**记录并覆盖**计划必需对象 ----
        _targets = [str(x) for x in (_doc.get("targets") or [])]
        if not _targets:
            return False, ("publish journal records no targets; it cannot "
                           "prove which objects were committed")
        _t_set = {os.path.abspath(x) for x in _targets}
        _need = {os.path.abspath(x[1]) for x in objs}
        _missing = sorted(_need - _t_set)
        if _missing:
            return False, ("publish journal does not cover object(s): "
                           f"{[os.path.basename(x) for x in _missing[:3]]}")
    except Exception as exc:                # pragma: no cover - 部署异常
        return False, f"cannot verify the publish journal: {exc}"

    return True, ("media + planned reports + COMMITTED journal "
                  f"({len(objs)} objects)")


def verify_payload_manifest(marker: Dict[str, Any], logger=None
                            ) -> Tuple[bool, str]:
    """`check_payload_manifest` 的**布尔包装**（R22-A01 起语义收紧）。

    保留这个函数是为了既有的两态调用点；但行为变了：
    旧格式（无 payload）现在返回 **False**，不再是"没有可核对的内容 = 通过"。
    审核用"旧格式 marker + 同大小替换媒体"复现过这条绕过。

    Returns:
        ``(ok, reason)``。需要区分 legacy / bad 的调用方请直接用
        `check_payload_manifest()`。
    """
    final_path = str(marker.get("output_path") or "")
    state, why, _objs = check_payload_manifest(marker, final_path, logger)
    return state == MANIFEST_OK, why


def _plain_sha256_text(text: str) -> str:
    """UTF-8 文本的 sha256（与 `_sha256_text` 同义，单独命名避免歧义）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_published_marker(final_path: str, job: Dict[str, Any],
                            config: Dict[str, Any],
                            params: Optional[Dict[str, Any]] = None,
                            logger=None) -> Tuple[bool, str]:
    """验证"已发布的这一代"（R22-A01：与 resume 走同一条核心路径）。

    历史上这个函数只核对 marker 的身份字段与 payload，**不看** journal
    终态和整代对象 —— 于是它成了绕过完整检查的快捷入口。现在它只是
    `verify_generation_completion(expect_attempt=None)` 的别名：

    * 不要求本 attempt 的私有 result（这是它与正常收尾的唯一区别）
    * marker payload、整代磁盘事实、journal COMMITTED 全部照查

    Returns:
        ``(ok, reason)``。
    """
    return verify_generation_completion(
        final_path, job, config, params, logger, expect_attempt=None)


def resume_artifact_matches(final_path: str, job: Dict[str, Any],
                            config: Dict[str, Any], logger=None,
                            params: Optional[Dict[str, Any]] = None
                            ) -> Tuple[bool, str]:
    """RV-09 / R22-A01：resume 是否可以把这份产物当"已完成"。

    与正常收尾**共用同一个核心验证器**（`verify_generation_completion`），
    唯一区别是允许跨 attempt（`expect_attempt=None`，不要求本 attempt 的
    私有 result）。这正是审核要求的"两者不要混成同一宽松条件"：
    核心（marker payload + 整代磁盘事实 + journal 终态 + 参数身份）相同，
    只有"要不要当前 attempt 的结果"这一项不同。

    Args:
        params: 有效参数。**应当传**（resume 也要用当前 CLI 覆盖重算参数
            身份）；不传时退回 `build_params_identity` 的兜底合并，
            可能与发布时不一致 —— 那时会拒绝复用，这是保守方向。
    """
    return verify_generation_completion(
        final_path, job, config, params, logger, expect_attempt=None)


def find_final_output(job: Dict[str, Any], config: Dict[str, Any],
                      args, plan=None) -> List[str]:
    """FIX-V03: locate the stage7 final output.

    Stage7 writes to ``output_dir`` (stage7_encode.py:517-521), NOT to temp, so
    a completed job may have no ``step7`` file in temp_dir at all. The old
    resume scan only looked in temp_dir and therefore never saw finished jobs.

    RV-09: the plain ``{stem}_restored.*`` name is NOT job-specific, so two jobs
    whose inputs share a basename resolve to the same file. Candidates are now
    ordered explicit-output → job-scoped name → plain name, and callers must
    still verify identity via :func:`resume_artifact_matches`.
    """
    import glob
    candidates: List[str] = []
    output_dir = ""
    if isinstance(args, dict):
        output_dir = args.get("output") or ""
        _relocate = args.get("relocate_output")
    else:
        output_dir = getattr(args, "output", "") or ""
        _relocate = getattr(args, "relocate_output", None)
    if not output_dir:
        output_dir = config.get("global", {}).get("output_dir", "./output")

    # ---- R15-02：**权威目标优先** ----
    # 原来这里只看 job.output_path，完全不理会 relocate 的结果 ——
    # 于是执行写到 B、完成标记却写到 A（给旧媒体写本次完成记录）。
    # 现在先按与执行方**同一套**逻辑算出权威路径并优先使用它。
    try:
        from run_plan import resolve_final_media_path as _rfmp
    except ImportError:                      # pragma: no cover - 部署异常
        _rfmp = None
    _auth = None
    if plan is not None:
        _auth = plan.get("output_path") if hasattr(plan, "get") else None
    elif _rfmp is not None:
        try:
            _auth, _ = _rfmp(job, output_dir=output_dir,
                             relocate_output=_relocate)
        except Exception:
            _auth = None
    if _auth and os.path.exists(_auth):
        candidates.append(_auth)

    explicit = job.get("output_path") or ""
    if explicit and os.path.exists(explicit) and explicit not in candidates:
        candidates.append(explicit)
    stem = _safe_input_stem(job.get("input_path", ""))
    job_id = str(job.get("job_id", "") or "")
    # RV-09: 同名不同 job 的候选：{stem}_{job_id}_restored.* 由
    # _ensure_unique_final_outputs 在检测到冲突时产生。
    # R3-01（关键修复）：`_restored.*` 通配同时命中 completion marker
    # `<...>_restored.mkv.done.json`（写在 mkv 之后、mtime 更晚），裸 sort
    # 会把 .json 选成"最终产物"，resume/验收拿到的是 JSON 不是视频。
    # 统一走 checkpoint 媒体白名单过滤。
    from checkpoint import select_media_candidates
    job_scoped: List[str] = []
    if job_id:
        job_scoped = [str(p) for p in select_media_candidates(
            glob.glob(str(Path(output_dir) / f"{stem}_{job_id}_restored.*")))]
    plain = [str(p) for p in select_media_candidates(
        glob.glob(str(Path(output_dir) / f"{stem}_restored.*")))]
    candidates.extend(job_scoped + plain)
    return candidates


def _ensure_unique_final_outputs(jobs: List[Dict[str, Any]],
                                 config: Dict[str, Any], args,
                                 logger=None) -> None:
    """RV-09: two jobs with the same basename must not share one output path.

    The default name is ``{output_dir}/{stem}_restored.mkv``; two jobs from
    different input directories with the same filename therefore collide and
    one silently overwrites the other. When a collision is detected the later
    job gets a job-scoped name (single jobs are unaffected, so existing output
    names are preserved).
    """
    seen: Dict[str, str] = {}
    for job in jobs:
        if job.get("output_path"):
            continue
        stem = _safe_input_stem(job.get("input_path", ""))
        output_dir = ""
        if isinstance(args, dict):
            output_dir = args.get("output") or ""
        else:
            output_dir = getattr(args, "output", "") or ""
        if not output_dir:
            output_dir = config.get("global", {}).get("output_dir", "./output")
        default_path = str(Path(output_dir) / f"{stem}_restored.mkv")
        if default_path in seen:
            job_id = str(job.get("job_id", "") or "job")
            unique = str(Path(output_dir) / f"{stem}_{job_id}_restored.mkv")
            job["output_path"] = unique
            if logger:
                logger.warning(
                    "FINAL_OUTPUT_COLLISION",
                    f"Job {job_id} would write to the same final path as "
                    f"{seen[default_path]}; using a job-scoped name instead",
                    collided=default_path, used=unique)
        else:
            seen[default_path] = str(job.get("job_id", ""))


class _BufferedLog:
    """预检阶段的日志缓冲：只进内存 + stderr，**不落盘**。

    R15-03：`process_job` 原来在预检之前就 `logger.info("JOB_START", ...)`，
    而 StructuredLogger 会创建日志目录 —— 一个路径越界的 job 仍然会在
    授权根之外造出目录。这里把"计划阶段"的所有日志先接住，
    等路径校验通过、真正建好 job logger 之后再回放进去。
    """

    def __init__(self, stream=None):
        self.records = []
        self._stream = stream

    def _add(self, level, event, msg="", **kw):
        self.records.append((level, event, msg, kw))
        if self._stream is not None:
            try:
                self._stream.write(
                    "[preflight %s] %s: %s\n" % (level, event, msg))
            except Exception:
                pass

    def info(self, event, msg="", **kw):
        self._add("info", event, msg, **kw)

    def warning(self, event, msg="", **kw):
        self._add("warning", event, msg, **kw)

    def error(self, event, msg="", **kw):
        self._add("error", event, msg, **kw)

    def debug(self, event, msg="", **kw):
        self._add("debug", event, msg, **kw)

    def flush_into(self, target):
        """把缓冲的记录回放到真实 logger（顺序不变）。"""
        if target is None:
            return
        for level, event, msg, kw in self.records:
            fn = getattr(target, level, None) or getattr(target, "info", None)
            try:
                fn(event, msg, **kw)
            except Exception:
                pass


def _build_cli_overrides(args) -> Dict[str, Any]:
    """从 CLI/args 构造 ``merge_params`` 的覆盖字典。

    **R16-04**：批次预检、串行执行、并行 worker 必须消费**同一份**覆盖，
    否则三处会各自算出不同的有效路径（审核已确认批次传了空覆盖、
    worker 又漏了 allow_write_root/relocate_output）。
    """
    if not isinstance(args, dict):
        args = vars(args)
    ov: Dict[str, Any] = {}
    _output = args.get("output")
    _no_wsl = args.get("no_wsl_redirect", False)
    _keep = args.get("keep_intermediates", False)
    _gpu = args.get("gpu")
    _ai_enabled = args.get("ai_enabled")
    _ai_scale = args.get("ai_upscale_factor")
    _face_restore = args.get("face_restore")
    _ai_model = args.get("ai_model")
    if _output:
        ov["output_dir"] = _output
    if _no_wsl:
        ov["no_wsl_redirect"] = True
    if _keep:
        ov["cleanup_intermediates"] = False
    if _gpu:
        ov["gpu_backend"] = _gpu
    # ISSUE-V-MAIN-001 / BUG-047: AI CLI 覆盖必须进 params
    # FIX-P5-1: 用 ai_ 前缀，与 merge_params 的 ai_config 展开一致
    if _ai_enabled is True:
        ov["ai_enabled"] = True
    if _ai_scale is not None:
        ov["ai_upscale_scale"] = _ai_scale
    if _face_restore is not None:
        ov["ai_face_restore"] = _face_restore
    if _ai_model is not None:
        ov["ai_upscale_model"] = _ai_model
    # FIX-2026-10-22：强制放行 AI 封禁（1080p 素材）。必须在 merge_params
    # 之前进覆盖，否则 stage 列表里根本没有 stage3 → AI 静默不执行。
    # R16-05：用户明确要求重解释时间轴（与"探测到的帧率"分开）
    _fpsint = args.get("fps_interpret")
    if _fpsint:
        ov["fps_interpret"] = _fpsint
        ov["_fps_from_user"] = True
    if args.get("force_allow_banned_upscale"):
        ov["force_allow_banned_upscale"] = True
    return ov


def _plan_write_roots(args, config: Dict[str, Any]) -> Optional[List[str]]:
    """授权写根：CLI 优先，其次 config。**批次与单 job 必须是同一集合**。

    返回 ``None`` 表示"未限制"（调用方不得把它当成"任意可写"的许可去放宽）。
    """
    _raw = (args.get("allow_write_root") if isinstance(args, dict)
            else getattr(args, "allow_write_root", None))
    if not _raw:
        _raw = config.get("allowed_write_roots")
    if not _raw:
        return None
    if isinstance(_raw, str):
        return [_raw]
    return [str(r) for r in _raw if r]


def _resolve_plan_for_job(job: Dict[str, Any], config: Dict[str, Any],
                          args, logger=None):
    """一次算出 job 的**完整有效计划**（merge_params + resolve_run_plan）。

    返回 ``(params, plan, module_available)``。``plan`` 可能为 ``None``
    （模块缺失）。调用方**不得**在 ``plan.ok`` 为 False 时开始落盘。
    """
    _cli = _build_cli_overrides(args)
    params = merge_params(job, config, _cli, logger)
    params["output_path"] = job.get("output_path", "")
    try:
        from run_plan import resolve_run_plan as _rp
    except ImportError:
        return params, None, False
    _roots = _plan_write_roots(args, config)
    _reloc = (args.get("relocate_output") if isinstance(args, dict)
              else getattr(args, "relocate_output", None))
    plan = _rp(job, params, write_roots=_roots, relocate_output=_reloc)
    return params, plan, True


def _chain_mode_on(params) -> bool:
    """当前是否启用了含 stage34 的融合执行组。"""
    try:
        return str((params or {}).get("chain_mode") or "none").lower() in (
            "stage34", "both")
    except Exception:
        return False


def _fused_step34(params, job, logger=None):
    """找 stage3 留下的 G34 融合回执，返回被融合产出的 step4 路径或 None。

    判定逻辑**全部**在 `shared_modules/fused_receipt.py` —— 与 `stage5`
    共用同一个验证器（审核 R19-A03：main/stage5 通过共享验证器消费）。

    历史：这里曾自己实现过一遍（attempt → input_digest → output/size），
    stage5 也实现过一遍。两份会漂移，一处收紧另一处照旧就等于没修。
    """
    try:
        temp_dir = (params or {}).get("temp_dir") or "."
        stem = Path(job.get("input_path", "")).stem
        job_id = job.get("job_id", "")

        from fused_receipt import find_and_verify as _fav
        _rcpt, _path, _why = _fav(
            str(temp_dir), str(job_id), stem, params or {},
            expected_attempt=str((params or {}).get("attempt_id") or ""),
            src_path=str(job.get("input_path") or ""))
        if _rcpt is None:
            if logger:
                # 保留细分事件名：统一成一个 `_REJECTED` 会丢掉"为什么拒"的
                # 可检索性（值班时要能 grep 出"是产物没了还是尺寸变了"）。
                _w = str(_why or "")
                if "output is gone" in _w:
                    _ev = "STAGE4_FUSED_RECEIPT_STALE"
                elif "size changed" in _w:
                    _ev = "STAGE4_FUSED_RECEIPT_SIZE_MISMATCH"
                elif ("source changed" in _w or "quality params changed" in _w
                      or "frozen" in _w):
                    _ev = "STAGE4_FUSED_RECEIPT_INPUT_CHANGED"
                elif ("no input_digest" in _w or "no attempt_id" in _w
                      or "source-only input_digest" in _w):
                    _ev = "STAGE4_FUSED_RECEIPT_LEGACY"
                elif ("no color evidence" in _w or "color was never probed" in _w
                      or "colour tags lost" in _w
                      or "bounds check" in _w):
                    # R20-A03：回执在、身份也对，但**证据块不全**。
                    # 这和"过期"是两件事：过期是"这不是本次的"，
                    # 未证明是"是本次的，但没人证明它对"。
                    _ev = "STAGE4_FUSED_RECEIPT_UNPROVEN"
                else:
                    _ev = "STAGE4_FUSED_RECEIPT_REJECTED"
                logger.warning(
                    _ev,
                    "No usable G34 receipt for this attempt/source; running "
                    "stage4 the classic way instead.", reason=_w)
            return None
        return _rcpt.get("output")
    except Exception as exc:              # pragma: no cover
        if logger:
            logger.warning("STAGE4_FUSED_RECEIPT_ERROR", str(exc))
        return None


def process_job(job: Dict[str, Any], config: Dict[str, Any],
                args, logger) -> Tuple[int, str]:
    """Process a single job through all stages. Returns (exit_code, stderr)."""
    job_id = job["job_id"]
    input_path = job["input_path"]

    # R15-03：计划阶段**零落盘**。所有日志先进缓冲（同时打 stderr），
    # 路径校验通过、job logger 建好之后再回放 —— 这样"越界路径的 job
    # 在授权根外造出日志目录"就不可能发生。
    _pre_log = _BufferedLog(sys.stderr)
    _pre_log.info("JOB_START", f"Starting job {job_id}",
                  job_id=job_id, input_path=input_path)

    # BUG-042 fix: normalize args to dict early so all downstream code uses dict API
    if not isinstance(args, dict):
        args = vars(args)

    # Build CLI overrides —— **必须与批次预检、并行 worker 同一份**
    # （R16-04；内联构造曾导致三处算出不同的有效路径）
    cli_overrides = _build_cli_overrides(args)

    # Merge params
    params = merge_params(job, config, cli_overrides, _pre_log)
    params["output_path"] = job.get("output_path", "")

    # ---- R20-A03：给本次执行一个**可核对的 attempt 标识** ----
    # 审核：融合回执"按 job+stem glob、字典序最后一个"决定 stage4 是否完成，
    # 于是上一轮残留的回执会让本轮误判。修法是让每个回执带 attempt，
    # 消费方只认**本 attempt** 的那一份。
    # 这里生成一次，随 params 传给所有 stage（`attempt_id` 在
    # `_IDENTITY_POSITIONAL_KEYS` 里，不参与内容身份摘要）。
    if not params.get("attempt_id"):
        import uuid as _uuid
        params["attempt_id"] = _uuid.uuid4().hex[:16]

    # ---- R20-A01：给 stage7 一个 **attempt 私有** 的结果路径 ----
    # 审核 §3.2 第 1 步："main 冻结身份并给 stage7 明确的 attempt 私有结果路径。
    # stage7 通过参数传递接收，不用 glob 猜。"
    # main 从这条**确定路径**读回执，而不是从共享文件名/最新 mtime 反推。
    try:
        _td = str(params.get("temp_dir") or ".")
        _stem0 = Path(str(job.get("input_path") or "in")).stem
        params["_stage7_result_path"] = str(
            Path(_td) / f"{job.get('job_id','job')}_{_stem0}"
                        f".attempt-{params['attempt_id']}.result.json")
    except Exception:                      # pragma: no cover
        params["_stage7_result_path"] = ""

    # ---- R14-06：有效路径预检，必须先于**首次落盘** ----
    # 原来 `--output` 只换 output_dir，而最终媒体仍取 job.output_path，
    # 于是"换了输出目录"其实没换 —— 2026-10-28 那次误覆盖就是这样发生的。
    # 这里把全部读写路径物化一次，预检通过后写回 params，
    # 后续 stage 只消费这一份（不在 stage 内重新决定位置）。
    try:
        from run_plan import resolve_run_plan as _resolve_run_plan
    except ImportError as _rpe:             # pragma: no cover - 部署异常
        _resolve_run_plan = None
        # R15-03：用户已经要求限制写根时，模块缺失**必须失败** ——
        # 否则"以为受限"的运行会悄悄回到无限制行为。
        if args.get("allow_write_root") or config.get("allowed_write_roots"):
            sys.stderr.write(
                "[run-plan] FATAL: write roots were requested but "
                "shared_modules/run_plan.py could not be imported "
                f"({_rpe}). Refusing to run without path validation.\n")
            _pre_log.error(
                "RUNPLAN_MODULE_MISSING",
                "Write roots were requested but the run-plan module is "
                "unavailable; refusing to run without path validation.",
                error=str(_rpe))
            return 82, "run-plan module missing while write roots requested"

    if _resolve_run_plan is not None:
        _roots = _plan_write_roots(args, config)
        _plan = _resolve_run_plan(
            job, params, write_roots=_roots,
            relocate_output=args.get("relocate_output"))
        for _w in _plan.warnings:
            _pre_log.warning("RUNPLAN_WARN", _w, job_id=job_id)
        if not _plan.ok:
            for _e in _plan.errors:
                _pre_log.error("RUNPLAN_BLOCKED", _e, job_id=job_id)
            _pre_log.error(
                "RUNPLAN_ABORT",
                "Refusing to start: the effective run plan has blocking "
                "path problems. Nothing has been written for this job. "
                "Pass --allow-write-root to declare the authorised roots, "
                "or --relocate-output to move the outputs explicitly.",
                job_id=job_id, errors=_plan.errors)
            return 82, "run-plan preflight failed"
        # 写回唯一计划：媒体、temp、log 都以它为准
        if _plan.get("output_path"):
            params["output_path"] = _plan.get("output_path")
        if _plan.get("temp_dir"):
            params["temp_dir"] = _plan.get("temp_dir")
        if _plan.get("log_dir"):
            params["log_dir"] = _plan.get("log_dir")
        params["_run_plan"] = _plan.as_dict()
        # R15-02：完成标记必须消费**同一个**计划对象，不能靠重新读正式路径
        params["_run_plan_obj"] = _plan
        _pre_log.info(
            "RUNPLAN_OK", "effective run plan resolved",
            job_id=job_id,
            output=params.get("output_path"),
            output_source=_plan.source_of("output_path"),
            temp_source=_plan.source_of("temp_dir"),
            log_source=_plan.source_of("log_dir"),
            write_roots=_plan.write_roots or None)
    # ID-02（第七轮）：把**实际执行用的这一份**参数挂到 job 上，
    # 让 build_params_identity() 拿到同一份（含 CLI 覆盖），
    # 而不是退回"空 CLI 覆盖"的另一次合并。
    job["_effective_params"] = params

    # FIX-W19: if the startup dependency check found sRestore unavailable, force
    # ivtc_auto_fallback=False so the VFM _Combed failure path does not request
    # a sRestore fallback that will fail (plugin missing). Per v7.32 §3.6.
    if config.get("_srestore_unavailable"):
        params["ivtc_auto_fallback"] = False

    # FIX-P2-6/§3.10.10: Combined WSL2 redirect + disk space pre-check with fallback chain
    # Replaces separate WSL redirect block and _check_disk_space().
    # Per v7.32 §3.10.10: native partition → /mnt/ fallback → FAILED
    if not check_wsl_space(params, logger=logger, job=job):
        return 87, "Insufficient disk space"

    # ---- R15-03：重定向之后必须**再核一次** ----
    # `checkpoint.check_wsl_space()` 会改写 params["temp_dir"] /
    # params["output_dir"]（WSL 重定向、空间不足回退）。校验完再换位置，
    # 等于校验没做。这里对**实际生效**的那份路径重新校验；
    # 越界就停，不让"验证过又偷偷换目录"发生。
    if _resolve_run_plan is not None and _roots:
        try:
            from run_plan import path_key as _pkey
        except ImportError:                 # pragma: no cover
            _pkey = None
        # R16-04：把**重定向后的实际路径**明确传进去。
        # 否则 resolver 仍按 job.temp_dir 优先 → 批准一个实际不使用的路径，
        # 于是"校验过旧路径、实际写新路径"（审核已静态确认）。
        _redir_ov = {}
        if params.get("temp_dir"):
            _redir_ov["temp_dir"] = str(params["temp_dir"])
        if params.get("output_path"):
            _redir_ov["output_path"] = str(params["output_path"])
        _plan2 = _resolve_run_plan(job, params, write_roots=_roots,
                                   relocate_output=args.get("relocate_output"),
                                   redirect_overrides=_redir_ov)
        if not _plan2.ok:
            for _e in _plan2.errors:
                logger.error("RUNPLAN_BLOCKED_AFTER_REDIRECT", _e,
                             job_id=job_id)
            logger.error(
                "RUNPLAN_ABORT_AFTER_REDIRECT",
                "WSL/disk fallback moved a write path outside the authorised "
                "roots; refusing to continue.", job_id=job_id)
            return 82, "run-plan invalid after redirect"
        if _pkey is not None:
            _old_tmp = _pkey(_plan.get("temp_dir"))
            _new_tmp = _pkey(_plan2.get("temp_dir"))
            if _old_tmp != _new_tmp:
                logger.warning(
                    "RUNPLAN_TEMP_REDIRECTED",
                    "The WSL/disk fallback redirected temp_dir; the new "
                    "location was re-validated against the authorised roots",
                    job_id=job_id,
                    before=_plan.get("temp_dir"),
                    after=_plan2.get("temp_dir"))
        params["_run_plan_obj"] = _plan2
        params["_run_plan"] = _plan2.as_dict()

    log_dir = params.get("log_dir", "./logs")
    job_logger = get_logger(log_dir=log_dir, job_id=job_id)
    # R15-03：路径已校验、logger 已建好 —— 现在才把计划阶段的记录回放。
    # 顺序不变（JOB_START 仍是最早一条），但此前一个字都没写盘。
    _pre_log.flush_into(logger)
    _pre_log.flush_into(job_logger)

    # ---- 执行模式必须在构建复用计划**之前**冻结 ----
    # 审核 R18-A04：原来 `build_reuse_plan()` 在前，
    # 注入 chain_mode / vs_deterministic / _run_identity 在后 ——
    # 于是"计划时用的 params"与"stage 实际执行的 params"不是同一份，
    # 局部 stage 身份对不上，下次比较必然不同 → 每个 stage 都被判失效。
    #
    # 现在先确定执行模式（含能力回退原因），再算身份与复用计划。
    # 注意：`_run_identity` 是**内部回执**，不是内容语义 ——
    # `param_affects_stage()` 会跳过 `_` 前缀（R18-A04 一并修复），
    # 所以它不会污染任何 stage 的身份。


    # ---- chain.mode：默认 none，行为与改动前一致 ----
    # 它只决定**怎么执行**（经典两阶段 vs 融合执行组），不影响功能开关。
    try:
        if not params.get("chain_mode"):
            _cfg_chain = config.get("chain") or {}
            _raw_mode = str(
                (args.get("chain") if isinstance(args, dict)
                 else getattr(args, "chain", None))
                or _cfg_chain.get("mode") or "none").strip().lower()
            if _raw_mode not in _KNOWN_CHAIN_MODES:
                job_logger.warning(
                    "CHAIN_MODE_UNKNOWN",
                    "Unknown chain mode; falling back to none. "
                    "See allowed values.",
                    requested=_raw_mode, allowed=sorted(_KNOWN_CHAIN_MODES))
                _raw_mode = "none"
            params["chain_mode"] = _raw_mode
        # ---- vs.deterministic：VapourSynth 渲染确定性（默认关）----
        # 打开后 stage1/stage6 会在 vpy 里写 `core.num_threads = 1`，
        # 输出逐像素可复现。
        #
        # R23-B14：**不要**把"4.78 倍"当通用成本写进普通日志。
        # 那个数字是第十八轮在**一台机器、一段 60 帧 1080p 样本**上的观测
        # （4.95 s → 23.47 s），不是对所有素材/机器成立的上界，更不该出现在
        # 每一次正常跑批的日志里冒充"已知代价"。历史值留在注释里备查。
        if "vs_deterministic" not in params:
            _cfg_vs = config.get("vs") or {}
            if not isinstance(_cfg_vs, dict):
                _cfg_vs = {}
            # 审核 R19-B06：`bool("false")` 为 True，会把字符串 false 读成开。
            params["vs_deterministic"] = _strict_bool(
                _cfg_vs.get("deterministic"), default=False)
            if params["vs_deterministic"]:
                job_logger.warning(
                    "VS_DETERMINISTIC_ON",
                    "VapourSynth rendering is forced to a single thread so "
                    "outputs are pixel-reproducible. Single-thread rendering "
                    "is materially slower and the slowdown depends on the "
                    "clip and the machine; use it for acceptance runs, not "
                    "bulk production.")

        job_logger.info("CHAIN_MODE", "Effective execution-group mode",
                        chain_mode=params.get("chain_mode"))
    except Exception as _cm_exc:          # pragma: no cover
        job_logger.warning("CHAIN_MODE_FAIL", str(_cm_exc))
        params["chain_mode"] = "none"
        params.setdefault("vs_deterministic", False)

    # ---- R22 §5.2：工作表示（色度/位深）必须在**动手之前**解析并核验 ----
    # 审核要求"手动选择通过前置能力核验后执行"、"无法满足的要求应明确
    # 失败/降级，用户决定是否接受"。所以这一步放在计划冻结之前：
    #   * 解析结果冻结进 `params["_work_format"]`（下划线开头，不进身份）；
    #   * 有 blocker 且用户没显式接受降级 → 直接停，不跑任何 stage。
    try:
        from work_format import resolve as _wf_resolve, describe as _wf_desc
        _wf_plan = _wf_resolve(
            params, source_path=str(job.get("input_path") or ""),
            logger=job_logger)
        params["_work_format"] = _wf_plan
        # ---- R24-04 / R23-A04：把**解析结论**并入参与身份的语义视图 ----
        # `_work_format` 以下划线开头，会被 `_identity_semantics` 过滤掉；
        # 而"同一份用户请求解析出不同表示"（编码器能力变了、源色度读法不同）
        # 必须让旧产物失效 —— 所以另写一份**不带下划线**的最小结论。
        #
        # 只放**决定性**字段。boundaries 表、reason 文本这些会随代码措辞变化，
        # 放进去每次升级都会让所有旧产物失效 —— 那是噪声，不是语义。
        _deliv = (_wf_plan.get("delivery") or {})
        _conv = _wf_plan.get("conversion")
        _degr = _wf_plan.get("degraded")
        params["work_format_resolved"] = {
            "schema": 1,
            "working_chroma": str(_wf_plan.get("resolved") or ""),
            "delivery_pix_fmt": str(_deliv.get("pix_fmt") or ""),
            "delivery_chroma": str(_deliv.get("chroma") or ""),
            "conversion": (f"{_conv.get('from')}->{_conv.get('to')}"
                           if isinstance(_conv, dict) else ""),
            "degraded": (f"{_degr.get('from')}->{_degr.get('to')}"
                         if isinstance(_degr, dict) else ""),
            "encoder": str(_wf_plan.get("encoder") or ""),
        }
        # `compute_param_hash` 的白名单是**扁平键**且对 dict 值递归过滤
        # （`checkpoint._extract_quality`），所以上面那份可读结构进不了
        # 阶段身份。再压一个摘要键 —— 阶段身份用这个，人读用上面那份。
        try:
            params["work_format_digest"] = hashlib.sha256(
                json.dumps(params["work_format_resolved"], sort_keys=True,
                           ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        except (TypeError, ValueError):
            params["work_format_digest"] = ""
        job_logger.info("WORK_FORMAT_PLAN", _wf_desc(_wf_plan),
                        mode=_wf_plan.get("mode"),
                        working=_wf_plan.get("resolved"),
                        delivery=(_wf_plan.get("delivery") or {}).get("pix_fmt"),
                        recommendation=_wf_plan.get("recommendation"),
                        conversion=_wf_plan.get("conversion"),
                        degraded=_wf_plan.get("degraded"),
                        boundaries=_wf_plan.get("boundaries"))
        # R23-B09：判据收敛到计划自己的 `ok`（它已经要求"接受降级时必须
        # 给出可执行替代"）。`blockers and not accepted` 会让"接受了但没
        # 替代"悄悄跑起来。
        if not _wf_plan.get("ok"):
            job_logger.error(
                "WORK_FORMAT_REFUSED",
                "The requested working representation cannot be produced by "
                "this chain. Nothing was executed. Either change "
                "`chroma_mode` / `pix_fmt` / `video_encoder`, or set "
                "`chroma_mode_accept_degraded: true` to accept the stated "
                "degradation.", blockers=_wf_plan.get("blockers"))
            return 86, ("work format unsupported: "
                        f"{_wf_plan.get('blockers')}")
    except Exception as _wfe:
        # 解析本身失败不等于"能力不足"——不能因此静默跑默认路径
        job_logger.error("WORK_FORMAT_RESOLVE_FAIL",
                         "Cannot resolve the working representation; refusing "
                         "to run a generation whose chroma/bit depth is "
                         "undetermined", error=str(_wfe))
        return 86, f"work format resolve failed: {_wfe}"

    # ---- R17-01：把"本次 run 的身份"随 params 传给 stage ----
    # 审核要求：marker 不能只比对"当前 verify 与当前媒体彼此一致" —— 只要
    # B 把媒体和报告一起换掉，两者仍然自洽，A 无法察觉这份报告已不是自己那代。
    # 做法：main 把自己算出的**输入身份 + 参数身份**塞进 params，stage7 原样
    # 回写进 verify.json。B 换代时会覆盖成 B 的身份，A 一比就知道不是自己的。
    # 成本为零（不额外读素材），且两边用的是同一份数值，不会各算各的。
    # 注意：`_` 前缀会被 `_identity_semantics()` 与 `param_affects_stage()` 跳过
    # （R18-A04），因此不会改变 params 身份摘要，也不影响 CORE_QUALITY_PARAMS。
    #
    # ---- ①②之后：执行模式已冻结，**这时**才算全局身份 ----
    # 审核 R19-B06：
    #   > `_run_identity` 在约 1577 行构造，chain_mode 在约 1590 行、
    #   > vs_deterministic 在 1603 行才注入。模式冻结 → 内部身份 → 局部身份/计划
    #   > 还没有完全实现。
    # 顺序必须是：chain_mode / vs_deterministic 落定 → 再算 _run_identity，
    # 否则身份摘要里记录的是"模式未知"的 params，与实际执行模式不符。
    try:
        # R22-A02：必须给出**可重算**的 RunIdentity（含非空 digest）。
        # 旧写法只有 {input, params}，stage7 取 `.digest` 恒为空串，
        # 回执里的 run_identity 一直是空的。
        params["_run_identity"] = build_run_identity(
            config, job, job_logger, params=params)
        if not params["_run_identity"].get("digest"):
            raise ValueError("run identity digest is empty")
    except Exception as _rid_exc:         # pragma: no cover
        job_logger.warning("RUN_IDENTITY_FAIL", str(_rid_exc))

    # ---- R21-C06：质量参数摘要必须在**最终有效计划**落定之后才算 ----
    # 审核 R20-C06："`main._frozen_param_digest` 当前早于 resolve_run_plan 和
    # chain_mode 落定。只保留一个最终有效计划冻结点，冻结后不再偷偷更改参与
    # 身份的参数；冻结失败不能变成空字符串然后跳过校验。"
    #
    # 原来的位置在 resolve_run_plan（约 1880 行）和 chain_mode（约 2010 行）
    # **之前** —— 那时 `_run_plan` 还没解析、`chain_mode` 还是用户原话，
    # 冻结下来的摘要与实际执行的参数不是同一份。
    # 现在与 `_run_identity` 同一处：计划已解析、模式已冻结。
    try:
        from checkpoint import compute_param_hash as _cph_frozen
        _fz_digest = _cph_frozen(str(job.get("input_path") or ""), params)
        if not _fz_digest:
            raise ValueError("compute_param_hash returned an empty digest")
        params["_frozen_param_digest"] = _fz_digest
        job_logger.info("FROZEN_PARAM_DIGEST",
                        "Froze the quality-param digest after the effective "
                        "plan and execution mode were settled",
                        digest=str(_fz_digest)[:16],
                        chain_mode=params.get("chain_mode"))
    except Exception as _fz:
        # 冻结失败**不能**退化成空串然后跳过校验 —— 那样融合回执会
        # "看起来有摘要、实际不比"。身份都算不出来，这一代不该继续。
        job_logger.error(
            "FROZEN_PARAM_DIGEST_FAIL",
            "Cannot freeze the quality-param digest; refusing to run a "
            "generation whose identity cannot be established", error=str(_fz))
        return 82, f"cannot freeze the param digest: {_fz}"


    # ---- R21-C02 / R21-C06：marker 的身份字段在**这里**冻结 ----
    # 审核 §4.1 第 2–3 步：stage7 负责"媒体 + 必需报告"的不可变清单，
    # marker 候选由该清单生成并随同一事务发布。`impl` / `input_identity` /
    # `params_identity` 只有 main 算得出来（stage7 拿不到 job/config），
    # 所以先算好作为**种子**下传。
    #
    # 位置很关键：必须在 `resolve_run_plan` 与 `chain_mode` **落定之后**。
    # 实测（N20）：原来它在两者之前，marker 里的语义视图没有 `chain_mode`，
    # 验证时却有 → 摘要对不上 → 整批 job 报
    # `COMPLETION_MARKER_PARAMS_DIFF: chain_mode null -> "none"` 并以 82 结束。
    try:
        # R22-A03：先把**计划**抄成不可变快照，再算身份。
        # 此后 stage 往 params 里注入的任何东西（`ivtc` 的自动回退结论、
        # 上游产物路径……）都不会再改变这一代的身份 —— 因为身份读快照。
        # 反过来，用户在 overrides/CLI 里指定的 `ivtc` 在快照里，
        # 所以 VIVTC 与 sRestore 会被正确区分开（审核 A03 的复现点）。
        params["_plan_snapshot"] = _identity_semantics(params)
        _frozen_identity = build_params_identity(
            config, job, job_logger, params=params)
        params["_frozen_params_identity"] = _frozen_identity
        params["_marker_seed"] = {
            "impl": VIDEO_IMPL_VERSION,
            "job_id": job.get("job_id", ""),
            "input_path": job.get("input_path", ""),
            "input_identity": build_input_identity(job.get("input_path", "")),
            # 同一个值不能两端各算一次（R20 的 input_digest 教训）
            "params_identity": _frozen_identity,
            # 语义视图一起存进 marker：身份不匹配时能直接 diff
            "params_semantics": _identity_semantics(params),
        }
    except Exception as _mse:              # pragma: no cover - 身份算不出
        params["_marker_seed"] = {}
        params["_frozen_params_identity"] = ""
        _pre_log.warning("MARKER_SEED_FAIL",
                         "Cannot freeze the marker identity; stage7 will "
                         "not publish a marker for this generation",
                         error=str(_mse))


    # Determine stage order
    stages = [
        (0, "stage0_audio_extract.py"),
        (1, "stage1_preprocess.py"),
        (2, "stage2_traditional_check.py"),
    ]

    ai_enabled = params.get("ai_enabled", False)
    ai_banned = job.get("ai_upscale_banned", False)
    force_allow = params.get("force_allow_banned_upscale", False)

    # V-P-007 fix: force_traditional_first gate
    # AI stages are appended conditionally; main loop will gate them
    if ai_enabled and not (ai_banned and not force_allow):
        stages.append((3, "stage3_ai_upscale.py"))
        stages.append((4, "stage4_postprocess.py"))

    face_restore = params.get("face_restore", "")
    # FIX-P1-12: stage5 (face restore) must run AFTER stage3/4 (AI upscale).
    # When ai_upscale_banned=True and force_allow=False, stage3/4 are skipped
    # (line above), so stage5 would have no input → fail. Gate stage5 with the
    # same banned condition. (force_allow=True keeps stage5 as normal.)
    if ai_enabled and face_restore and not (ai_banned and not force_allow):
        stages.append((5, "stage5_ai_face.py"))

    stages.append((6, "stage6_grain_synthesis.py"))
    stages.append((7, "stage7_encode.py"))

    _stage = args.get("stage")
    if _stage is not None:
        stage_map = {s[0]: s[1] for s in stages}
        if _stage in stage_map:
            stages = [(_stage, stage_map[_stage])]
        else:
            return 81, "Invalid stage number"
    else:
        # ID 第 8 点（第七轮）：跑之前先把**复用计划**算出来并记账。
        #
        # 旧行为只有一个粗粒度的 `resume_from_stage`（由 FAILED 降级逻辑
        # 决定），从来没有回答过"哪些阶段能复用、为什么不能、会触发几次
        # 昂贵的 AI"。现在按 stage 的**局部身份**逐级判定：
        # 换 CRF 只让 stage7 失效、换日志级别不失效、换人脸模型不影响
        # 传统降噪；上游失效则下游一律失效（传播）。
        _reuse_plan = None
        try:
            from stage_deps import (build_reuse_plan,
                                    render_reuse_plan)
            _recorded: Dict[int, Optional[Dict[str, Any]]] = {}
            _temp = params.get("temp_dir", "./temp")
            for _sn in range(8):
                _cands = find_stage_artifacts(_temp, job_id, input_path, _sn)
                _rec = None
                for _c in _cands:
                    _side = str(_c) + ".identity.json"
                    if os.path.exists(_side):
                        try:
                            with open(_side, "r", encoding="utf-8") as _f:
                                _d = json.load(_f)
                            if int(_d.get("stage", -1)) == _sn:
                                _rec = _d
                                _rec.setdefault("artifact", _c)
                                break
                        except Exception:
                            continue
                _recorded[_sn] = _rec
            # ---- R23-05：**计划里不适用**的阶段不参与复用判定 ----
            # stage3 由 `ai_enabled` 决定、stage5 由 `face_restore` 决定；
            # 它们没跑过就没有侧车，那不是"身份缺失"。
            _applicable = [0, 1, 2, 4, 6, 7]
            if _strict_bool(params.get("ai_enabled"), False):
                _applicable.append(3)
            if str(params.get("face_restore") or "").strip():
                _applicable.append(5)
            _reuse_plan = build_reuse_plan(
                params, _recorded, applicable_stages=sorted(_applicable))
            # ---- R23-08：降级侧车必须被**这一代**记住 ----
            # 复用计划已经读过侧车；这里把"复用的是降级产物"落到 params，
            # stage7 会写进 verify 侧车（必需对象），完成验证与读报告的人
            # 都能看到 —— 而不是只留在会轮转的日志里。
            try:
                _deg = [{"stage": r["stage"], "path": r.get("path"),
                         "plugin": r.get("degraded_plugin")}
                        for r in (_reuse_plan.get("reusable") or [])
                        if r.get("degraded")]
                if _deg:
                    params["_degraded_inputs"] = _deg
                    job_logger.warning(
                        "STAGE_REUSE_DEGRADED",
                        "This run reuses a degraded intermediate artifact; "
                        "the degradation will be recorded in this "
                        "generation's verify sidecar", degraded=_deg)
            except Exception:
                pass
            job_logger.info(
                "STAGE_REUSE_PLAN",
                "Computed a per-stage reuse plan before running",
                restart_from=_reuse_plan.get("restart_from"),
                will_run_ai=_reuse_plan.get("will_run_ai"),
                ai_skip_reason=_reuse_plan.get("ai_skip_reason"),
                reusable=[r["stage"] for r in _reuse_plan.get("reusable", [])],
                invalidated=[{"stage": d["stage"],
                              "reason": d["reason"],
                              "keys": d.get("changed_keys", [])[:5]}
                             for d in _reuse_plan.get("invalidated", [])],
                stale_files=len(_reuse_plan.get("stale_files", [])))
            job_logger.info("STAGE_REUSE_PLAN_TEXT",
                            "\n" + render_reuse_plan(_reuse_plan))
        except Exception as _rp_exc:
            job_logger.warning("STAGE_REUSE_PLAN_FAIL", str(_rp_exc))

        # BUG-044 fix: resume from last completed stage + 1 when available
        resume_from = job.get("resume_from_stage")
        if resume_from is not None:
            stages = [(n, s) for n, s in stages if n >= resume_from]
            job_logger.info("RESUME_STAGE", f"Resuming job from stage {resume_from}")
            # Clear one-shot resume marker so re-runs don't keep skipping
            job.pop("resume_from_stage", None)
        elif _reuse_plan and _reuse_plan.get("restart_from") is not None \
                and not job.get("force_full_run"):
            # 计划说前 N 个 stage 可复用 → 从第一个失效的 stage 开始，
            # 而不是从头重跑（这正是"最小重算"的实际价值）。
            _rf = int(_reuse_plan["restart_from"])
            if _rf > 0:
                stages = [(n, s) for n, s in stages if n >= _rf]
                job_logger.info(
                    "STAGE_MIN_RERUN",
                    f"Reusing stages before {_rf} per the per-stage identity "
                    f"plan; starting at stage {_rf}",
                    restart_from=_rf)
        elif params.get("cleanup_intermediates", True):
            # FIX-NEW-V-07 / R6-10: 全新（非 resume）运行必须先清空本 job 的
            # 全部陈旧中间产物。keep_steps 是"保留最近 N 个阶段"，而
            # cleanup_old_intermediates 在 current_step=None 时按 step 号降序
            # 取前 keep_steps 个保留——上一次运行残留的高 step 产物（step3/
            # step4）会永久霸占保留位，把本轮刚生成的 step1 当"最旧"删掉，
            # 导致下游 stage 报 NO_INPUT（实测 GFPGAN 案例 rc=86）。
            try:
                _purged = cleanup_all_intermediates(
                    params.get("temp_dir", "./temp"), job_id)
                job_logger.info("STALE_INTERMEDIATES_PURGED",
                                "Purged stale intermediates for fresh run",
                                temp_dir=params.get("temp_dir", "./temp"),
                                purged=_purged)
            except Exception as e:  # 清理失败不应阻断运行
                job_logger.warning("STALE_INTERMEDIATES_PURGE_FAIL",
                                   f"Could not purge stale intermediates: {e}")

    # Run stages
    traditional_stage_completed = False
    # ID-04：本 job 内"每个 stage 本次真正产出的路径"，供下游精确消费
    _last_produced: Dict[int, List[str]] = {}


    for stage_num, script_name in stages:
        # ---- G34：stage3 已经把 stage4 一起做完了吗 ----
        # 融合模式下 stage3 直接产出 step4 并留下回执；这里必须**跳过**
        # stage4，否则它会去找不存在的 step3 → NO_INPUT，把一次成功的融合
        # 判成失败。回执不成立（缺失/尺寸不符/产物不在）时照常跑 stage4。
        if stage_num == 4 and _chain_mode_on(params):
            _fused = _fused_step34(params, job, job_logger)
            if _fused:
                job_logger.info(
                    "STAGE4_SKIPPED_FUSED",
                    "stage4 already ran inside the G34 execution group; "
                    "skipping the standalone stage to avoid a NO_INPUT failure",
                    output=_fused, chain_mode=params.get("chain_mode"))
                _last_produced[4] = [_fused]
                continue
        # ID-04（第七轮）：把**上一个 stage 本次真正产出**的精确路径传下去，
        # 让下游不再 glob 重选。
        #
        # 为什么必须显式传：旧流程里 main 这边校验过候选，但 stage 内部
        # 又自己 glob 一次（按 mtime 取最新），两次选择完全可能不是同一个
        # 文件 —— "验证 A、消费 B"。现在改为：编排器选定并传递，
        # stage 只消费被点名的那一份。
        #
        # 找不到本次产出（如 resume 从中间 stage 开始、或上游被跳过）时
        # 不下发 key，stage 回落到自己的 glob 逻辑（兼容独立 CLI 运行）。
        _pinned_upstream = None
        for _prev in range(stage_num - 1, -1, -1):
            _cands = _last_produced.get(_prev) or []
            _alive = [_c for _c in _cands if os.path.exists(_c)]
            if _alive:
                _pinned_upstream = _alive[0]
                break
            if _cands:
                # 本次确实产出过，但**已经被 cleanup 删除**（keep_steps 只保留
                # 最近 N 个 stage）。实测：stage5/6 被跳过时，stage4 的产物
                # 在 stage6 结束时被清理，于是 stage7 拿到一个不存在的 pin →
                # STAGE7_PINNED_INPUT_MISSING → rc=82。
                # 这里跳过已消失的候选，继续往更早的 stage 找。
                job_logger.warning(
                    "STAGE_PIN_CANDIDATE_GONE",
                    f"Stage {_prev} artifact produced by this run is no longer "
                    "on disk (removed by intermediate cleanup); trying an "
                    "earlier stage instead",
                    stage=stage_num, gone=_cands[:3])
        if _pinned_upstream:
            params["upstream_artifact"] = _pinned_upstream
            job_logger.info("STAGE_UPSTREAM_PINNED",
                            f"Stage {stage_num} will consume the artifact "
                            f"produced by this run: {_pinned_upstream}",
                            stage=stage_num, upstream=_pinned_upstream)
        else:
            params.pop("upstream_artifact", None)

        job_logger.info("STAGE_START", f"Stage {stage_num}: {script_name}")
        # ID-05（第七轮）：stage 执行**前**拍一次产物指纹，执行后只登记
        # 真正变了的那些。旧实现直接遍历 find_stage_artifacts() 的整个
        # glob 结果并逐条写入当前身份 —— temp 里任何**陈旧**的同 step
        # 文件（上次运行留下、或别的参数生成的）都会被"洗白"成当前
        # 身份，下次 resume 就把它当成合法产物复用。
        _pre_stage_files: Dict[str, Tuple[int, int]] = (
            _snapshot_stage_files(params.get("temp_dir", "./temp"),
                                  job_id, input_path, stage_num)
            if stage_num != 7 else {})
        exit_code, stderr = run_stage(script_name, job, params, args.get("config"))
        if exit_code == 0:
            job_logger.info("STAGE_DONE", f"Stage {stage_num} complete")
            # R2-07 补完：给本 stage 刚产出的中间产物登记身份（源 + 参数 + 实现）。
            # stage7 的最终产物另有 completion marker，不重复登记。
            if stage_num != 7:
                try:
                    _td = params.get("temp_dir", "./temp")
                    _post_stage_files = _snapshot_stage_files(
                        _td, job_id, input_path, stage_num)
                    _produced = [
                        _p for _p, _sig in _post_stage_files.items()
                        if _pre_stage_files.get(_p) != _sig]
                    _stale = [p for p in _post_stage_files
                              if p not in _produced]
                    if _stale:
                        # 不删、不洗白，只如实说明"这些不是本次产出"
                        job_logger.warning(
                            "STAGE_STALE_ARTIFACTS_PRESENT",
                            f"{len(_stale)} artifact(s) for stage {stage_num} "
                            "were not produced by this run and are left "
                            "WITHOUT the current identity (kept on disk)",
                            stale=_stale[:10])
                    for _art in _produced:
                        record_stage_artifact_identity(
                            _art, job, config, stage_num, input_path,
                            job_logger,
                            # ID-08：把本次实际消费的上游绑进侧车
                            upstream_artifact=params.get("upstream_artifact"),
                            # ID 第 8 点：局部身份需要"实际执行的参数"
                            params=params)
                    # ID-04：记下本次产出，供下一个 stage 作为精确上游
                    if _produced:
                        _last_produced[stage_num] = _produced
                        # 优先用"带当前 param_hash"的那一份（同一 stage 可能
                        # 有多个变体，最新写入的排后面，这里按 mtime 取最后）
                        _last_produced[stage_num] = sorted(
                            _produced,
                            key=lambda _p: os.stat(_p).st_mtime_ns
                            if os.path.exists(_p) else 0)[::-1]
                except Exception as e:
                    job_logger.warning(
                        "STAGE_IDENTITY_SKIP",
                        f"Could not register artifact identity: {e}")
            # FIX-P3-6: Per v7.32 §3.8: Auto-generate forced-upscale comparison
            # frames after stage3 AI upscale completes successfully
            if stage_num == 3 and force_allow:
                try:
                    from preview import auto_generate_comparison_if_needed
                    _temp_dir = params.get("temp_dir", "./temp")
                    _input_name = Path(input_path).name
                    import glob as _glob
                    _step3_pattern = f"{_temp_dir}/{job_id}_{_input_name}_step3*"
                    # R3-01：`_step3*` 通配同样命中 sidecar（mtime 晚于媒体），
                    # 裸 sort 会把 .json 当成 stage3 产物送给 preview 抽帧。
                    from checkpoint import select_media_candidates as _smc
                    _step3_cands = _smc(_glob.glob(_step3_pattern))
                    if _step3_cands:
                        _step3_output = str(_step3_cands[0])
                        _comp_frames = auto_generate_comparison_if_needed(
                            original_path=input_path,
                            processed_path=_step3_output,
                            output_dir=params.get("output_dir", "./output"),
                            config=config,
                            num_pairs=3
                        )
                        if _comp_frames:
                            job_logger.info(
                                "PREVIEW_FORCED_UPSCALE",
                                f"Generated {len(_comp_frames)} "
                                f"comparison frames",
                                frames=_comp_frames)
                except Exception as e:
                    job_logger.warning(
                        "PREVIEW_FORCED_UPSCALE_FAIL",
                        f"Comparison frame generation failed: {e}")
            if stage_num == 2:
                # FIX-W2: validate stage2 output via shared helper (exists +
                # non-empty + ffprobe-decodeable per v7.32 §3.10.8).
                temp_dir = params.get("temp_dir", "./temp")
                ok, msg = _verify_stage2_output(job_id, input_path, temp_dir, job_logger)
                if not ok:
                    job_logger.error("TRADITIONAL_GATE_BLOCK", msg)
                    return 86, msg
                traditional_stage_completed = True
                job_logger.info("TRADITIONAL_GATE",
                               "Traditional restoration completed, proceeding to AI stages")
        elif exit_code == 3:
            job_logger.info("STAGE_SKIP", f"Stage {stage_num} skipped")
            if stage_num == 2:
                # FIX-W2: exit-3 (skip) path must still re-validate the stage2
                # intermediate per v7.32 §3.10.8. stage3 requires a
                # step2_traditional file; if stage2 skipped without producing
                # one, block here with a clear message (86) rather than letting
                # stage3 fail opaquely.
                temp_dir = params.get("temp_dir", "./temp")
                ok, msg = _verify_stage2_output(job_id, input_path, temp_dir, job_logger)
                if not ok:
                    job_logger.error("TRADITIONAL_GATE_BLOCK", msg)
                    return 86, msg
                traditional_stage_completed = True
                job_logger.info("TRADITIONAL_GATE",
                               "Traditional restoration skipped (already done), proceeding to AI stages")
        else:
            # FIX-2026-10-21：**必须把 stderr 记进来**。原来只记 exit_code，
            # 于是 "Stage 7 failed: 87" 完全看不出是停滞、最终保险还是别的 ——
            # 16 分钟长片实测就因此多花了一轮排查（87 还被超时和磁盘不足复用）。
            job_logger.error(
                "STAGE_FAILED",
                f"Stage {stage_num} failed: {exit_code}"
                + (f" | {stderr[-500:]}" if stderr else ""),
                stderr=(stderr[-2000:] if stderr else None))
            # FIX-P6-7: VFM _Combed validation failed → auto-fallback to sRestore
            # per v7.32 §3.10.13
            if stage_num == 1 and exit_code == 86:
                combed_fallback = params.get("ivtc_auto_fallback", True)
                # FIX-W12: Distinguish IVTC-combed 86 from deint-quality 86.
                # stage1 returns 86 for BOTH: (a) VFM _Combed pre-check failure
                # (needs_ivtc + VIVTC path) and (b) post-render deinterlace
                # quality <95% (field_segments path). The IVTC sRestore path is
                # taken only when ivtc=="VIVTC" AND needs_ivtc AND combed_fallback;
                # otherwise a 86 with field_segments present is a deint-quality
                # failure -> clear field_segments and re-run conservative.
                if (combed_fallback and params.get("ivtc", "VIVTC") == "VIVTC"
                        and params.get("needs_ivtc", False)):
                    job_logger.warning("IVTC_AUTO_FALLBACK",
                                      "VFM _Combed check failed, switching to sRestore",
                                      job_id=job_id)
                    # Update params for sRestore path
                    params["ivtc"] = "sRestore"
                    # Re-run stage1 with sRestore
                    exit_code2, stderr2 = run_stage(
                        script_name, job, params, args.get("config"))
                    if exit_code2 in (0, 3):
                        job_logger.info("IVTC_SRESTORE_DONE",
                                       "sRestore IVTC fallback complete")
                        # Continue with remaining stages
                        if exit_code2 == 0:
                            job_logger.info("STAGE_DONE",
                                           f"Stage {stage_num} complete (sRestore)")
                        # FIX-W9: the `continue` below skips the loop-bottom
                        # cleanup_old_intermediates call, leaving stale VFM
                        # step1 intermediates on disk. Run cleanup here so the
                        # pre-sRestore VFM intermediates are removed before
                        # proceeding. (continue is still needed to avoid
                        # falling through to `return exit_code, stderr` which
                        # still holds the original exit_code=86.)
                        if params.get("cleanup_intermediates", True):
                            temp_dir = params.get("temp_dir", "./temp")
                            keep_steps = params.get("keep_steps", 2)
                            cleanup_old_intermediates(
                                temp_dir, job_id, keep_steps,
                                current_run_hashes=_current_run_param_hashes(
                                    _last_produced))
                        continue
                    else:
                        job_logger.error("IVTC_SRESTORE_FAIL",
                                        "sRestore fallback also failed")
                        return exit_code2, stderr2
                elif params.get("field_segments"):
                    # FIX-W12: deinterlace quality <95% on segmented field order.
                    # Clear field_segments and re-run stage1 with global
                    # conservative (non-segmented) processing per v7.32 §3.10.13.
                    job_logger.warning("DEINT_QUALITY_FALLBACK",
                                      "Deinterlace quality <95% on field_segments; "
                                      "re-running stage1 with global conservative "
                                      "processing (field_segments cleared)",
                                      job_id=job_id)
                    params["field_segments"] = []
                    if params.get("cleanup_intermediates", True):
                        temp_dir = params.get("temp_dir", "./temp")
                        keep_steps = params.get("keep_steps", 2)
                        cleanup_old_intermediates(
                            temp_dir, job_id, keep_steps,
                            current_run_hashes=_current_run_param_hashes(
                                _last_produced))
                    exit_code2, stderr2 = run_stage(
                        script_name, job, params, args.get("config"))
                    if exit_code2 in (0, 3):
                        if exit_code2 == 0:
                            job_logger.info("STAGE_DONE",
                                           f"Stage {stage_num} complete (conservative)")
                        continue
                    else:
                        job_logger.error("DEINT_QUALITY_FALLBACK_FAIL",
                                        "Conservative re-run also failed")
                        return exit_code2, stderr2
            # V-P-007/V-P-028 fix: force_traditional_first gate
            force_traditional_first = params.get("force_traditional_first", True)
            if force_traditional_first and stage_num == 2:
                job_logger.error("TRADITIONAL_GATE_BLOCK",
                                "Traditional restoration failed, blocking AI stages per force_traditional_first")
                return 86, stderr
            return exit_code, stderr

        # FIX-P5-7: 删除不可达的 traditional_stage_completed gate 冗余代码
        # 注：traditional_stage_completed 在 exit_code==0 或 ==3 时已设为 True；
        # 若 exit_code 为其他值，已在上方 else 分支提前 return。
        # 因此以下 gate 逻辑已在前面的 if/else 中完整覆盖，此处无需重复检查。

        # Cleanup old intermediates after successful stage
        #
        # ID-04 关联修复（第七轮）：**本 stage 没有产出时不要清理**。
        #
        # 实测（2 秒 1080i，AI 全开）：stage5（无人脸）与 stage6（颗粒关闭）
        # 都被 STAGE_SKIP 跳过、不产生任何文件。但清理照旧执行
        # `keep_steps=2, current_step=6` → 保留集 = {step5, step6} = 空集
        # → **把 step4 删掉**。而 step7 真正需要的上游正是 step4，
        # 于是 stage7 报 PINNED_INPUT_MISSING / NO_INPUT → rc=82 →
        # 整链从 stage0 重来。
        #
        # 语义上：跳过 = 这一代没有新产物 = 没有"更老的一代"需要腾空间，
        # 清理的必要性不存在，风险却实实在在。
        _produced_this_stage = bool(_last_produced.get(stage_num))
        if params.get("cleanup_intermediates", True) and (
                _produced_this_stage or stage_num == 7):
            temp_dir = params.get("temp_dir", "./temp")
            keep_steps = params.get("keep_steps", 2)
            # FIX-NEW-1: Removed spurious input_name arg — function signature is
            # cleanup_old_intermediates(temp_dir, job_id, keep_steps, current_step)
            # and matches files by job_id prefix; input_name is not a parameter.
            # FIX-NEW-V-07 / R6-10: 传 current_step，让判定走
            # "step < current_step 且不在保留集" 的显式语义，而不是
            # current_step=None 时"按 step 号降序取前 keep_steps 个"。
            cleanup_old_intermediates(
                temp_dir, job_id, keep_steps, current_step=stage_num,
                current_run_hashes=_current_run_param_hashes(_last_produced))

    # ---- R21-C02：**先证明完成，再说 JOB_DONE** ----
    # 审核：main 先记 JOB_DONE，再调用 write_completion_marker，
    # 且**没有检查它的 False 返回值**。受控调用确认：即使标记写入明确失败，
    # 仍返回 (0, "")。于是"这一代没被证明"和"任务成功"同时成立。
    # 顺序现在反过来，而且抽成可测函数（见 `finish_job_completion`）。
    return finish_job_completion(job, config, args, params, logger, job_logger)


def finish_job_completion(job: Dict[str, Any], config: Dict[str, Any], args,
                          params: Dict[str, Any], logger, job_logger
                          ) -> Tuple[int, str]:
    """任务收尾：**先证明完成，再报 JOB_DONE**（R21-C02）。

    审核要求（§3.3）："`JOB_DONE` 必须放在整代验证成功之后。
    `write_completion_marker` 返回 False、抛异常、终态 pending、
    私有结果缺失/错误 attempt，都应显式传播失败或未完成状态。"

    Returns:
        ``(exit_code, detail)``。只有**全部**证据成立才返回 ``(0, "")``；
        其余一律 ``82``，并留下 `JOB_COMPLETION_UNPROVEN`。
    """
    # `--stage N`（单步调试/补跑）**不宣称完成**：这一趟根本没跑完整链，
    # 用"没有成品"去否掉它是错的。只有默认全链或显式 `--stage 7` 才要求
    # 完整代的证据。
    #
    # 注意：**worker 里的 `args` 是 dict**（见 `_build_worker_tasks`），
    # 单进程模式才是 Namespace。只写 getattr 会在并行入口永远拿到 None。
    _stage = (args.get("stage") if isinstance(args, dict)
              else getattr(args, "stage", None))
    if _stage not in (None, 7):
        try:
            logger.info("JOB_PARTIAL_STAGE_RUN",
                        "Single-stage run; this invocation does not claim a "
                        "completed generation", stage=_stage)
        except Exception:
            pass
        return 0, ""

    _marker_ok = False
    _marker_path = ""
    _marker_why = ""
    try:
        finals = find_final_output(job, config, args,
                                   plan=params.get("_run_plan_obj"))
    except Exception as e:
        finals = []
        _marker_why = f"find_final_output raised: {e}"
    for cand in finals:
        if os.path.exists(cand) and os.path.getsize(cand) > 0:
            _marker_path = cand
            try:
                _marker_ok = bool(write_completion_marker(
                    cand, job, config, job_logger, params=params))
            except Exception as e:      # 写标记抛异常也算失败，不能算完成
                _marker_ok = False
                _marker_why = f"write_completion_marker raised: {e}"
            break
    else:
        _marker_why = _marker_why or "no finished output candidate was found"

    if not _marker_ok:
        job_logger.error(
            "JOB_COMPLETION_UNPROVEN",
            "The job produced output but its completion could not be proven "
            "(marker/identity/generation). This is NOT a successful job.",
            output=_marker_path, reason=_marker_why)
        return 82, (_marker_why or "completion could not be proven")

    logger.info("JOB_DONE", f"Job {job.get('job_id', '')} complete")
    return 0, ""


def _worker_wrapper(task: Dict[str, Any]) -> str:
    """Top-level worker for ProcessPoolExecutor (must be picklable in spawn mode).

    Args:
        task: dict with keys: job, config_path, log_dir, manifest_path, args
    """
    job = task["job"]
    config_path = task["config_path"]
    log_dir = task["log_dir"]
    manifest_path = task["manifest_path"]
    args = task["args"]

    jcfg = load_config(config_path)
    # RV-01: level 现在被 logger 真正实现（不再是非法实参）
    jlogger = get_logger(log_dir=log_dir,
                         level=jcfg.get("global", {}).get("log_level", "INFO"))
    jmanifest = load_manifest(manifest_path)
    # FIX-V08: worker 只报告自己的 job 结果（锁内读-合并-写），不再整份回写快照
    # RV-10: 必须把结构化结果 **返回给父进程**，否则父进程无从聚合失败
    # （ORIGINAL 只返回 job_id，父进程末尾无条件打印 Done、退出码恒为 0）。
    return _run_single_job(job, jcfg, args, jlogger, jmanifest, manifest_path,
                           attempt_group=task.get("manifest_attempt_group"),
                           owner=task.get("manifest_owner"),
                           generation=task.get("manifest_generation"))


# --- RV-10: 结构化任务结果（父进程据此聚合，不再"无条件成功"）--------------
#: 正常完成
JOB_DONE = "DONE"
#: 处理失败
JOB_FAILED = "FAILED"
#: 被取消/中断（退出码 88）
JOB_CANCELLED = "cancelled"
#: 结果没能落到 manifest（保存失败）——不能当成成功
JOB_PERSIST_FAILED = "persist-failed"
#: manifest 写入/IO 失败的退出码（AGENTS.md §4.4：82 = I/O error）
EXIT_PERSIST_FAILED = 82
#: 被信号中断（与 shared_modules/cancel_guard.EXIT_INTERRUPTED 一致）
EXIT_INTERRUPTED = 88


def _result(job_id: str, status: str, exit_code: int,
            error: str = "") -> Dict[str, Any]:
    return {"job_id": job_id, "status": status, "exit_code": int(exit_code),
            "error": error}


def aggregate_results(results: List[Dict[str, Any]], logger=None) -> int:
    """RV-10: fold per-job results into a single exit code.

    Returns 0 only when every job reported DONE. Otherwise the most severe
    observed exit code is returned (project range 80–113), so a failed batch
    can never leave the CLI with rc=0.
    """
    if not results:
        return 0
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1
    failures = [r for r in results if r.get("status") != JOB_DONE]
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    if logger is not None:
        (logger.error if failures else logger.info)(
            "BATCH_SUMMARY",
            f"{len(results) - len(failures)}/{len(results)} jobs complete "
            f"({summary})",
            counts=counts)
    if not failures:
        return 0
    codes = [int(r.get("exit_code") or 0) for r in failures]
    # 取最严重（数值最大）的失败码；未知时落到 86（校验/处理失败）
    worst = max(codes) if codes else 0
    return worst if worst > 0 else 86


def _artifact_digest(artifact: str) -> Optional[str]:
    """R3-06 / B08: 产物内容摘要（流式全内容 SHA-256）。

    只比对 size 时，等长替换（FIRST -> OTHER）判定不出来。
    """
    try:
        if not os.path.isfile(artifact):
            return None
        h = hashlib.sha256()
        with open(artifact, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def intermediate_identity_path(artifact: str) -> str:
    return str(artifact) + ".identity.json"


def _stage_identity_for(params: Dict[str, Any], stage_num: int) -> Optional[str]:
    """ID 第 8 点：局部参数身份（失败不影响登记）。"""
    try:
        from stage_deps import stage_identity
        return stage_identity(params, stage_num)
    except Exception:
        return None


def _stage_scope_snapshot(params: Dict[str, Any],
                          stage_num: int) -> Dict[str, Any]:
    """把影响本 stage 的参数值一起存下来，便于解释"为什么重算"。"""
    try:
        from stage_deps import stage_scope_keys
        return {k: params.get(k) for k in stage_scope_keys(params, stage_num)}
    except Exception:
        return {}


def _read_color_description(artifact: str) -> Dict[str, Any]:
    """COL-04：探测并返回该产物的实际颜色描述（失败时 probed=False）。"""
    try:
        from media_io import probe_color
        return probe_color(artifact)
    except Exception as exc:
        return {"probed": False,
                "error": f"{type(exc).__name__}: {exc}"}


def _read_exec_fingerprint(artifact: str) -> Optional[Dict[str, Any]]:
    """读 stage 自己写的 `.exec.json`；没有就返回 None（不猜）。"""
    try:
        from exec_fingerprint import read_exec_sidecar
        return read_exec_sidecar(artifact)
    except Exception:
        return None


def record_stage_artifact_identity(artifact: str, job: Dict[str, Any],
                                   config: Dict[str, Any], stage_num: int,
                                   input_path: str, logger=None,
                                   upstream_artifact: Optional[str] = None,
                                   params: Optional[Dict[str, Any]] = None
                                   ) -> bool:
    """R2-07 补完：给**每个中间产物**写身份侧车。

    之前 resume 只能"按文件名找到 + 能解码"就复用（A17 复现的缺陷）。
    现在每个 stage 成功后由调度器登记：源身份 + 有效参数身份 + 实现版本 +
    job_id + 产物自身大小。校验发生在 resume 时（见
    :func:`intermediate_identity_matches`）。
    """
    payload = {
        "impl": VIDEO_IMPL_VERSION,
        "job_id": job.get("job_id", ""),
        "stage": stage_num,
        "artifact": str(artifact),
        "artifact_size": (os.path.getsize(artifact)
                          if os.path.exists(artifact) else None),
        # R3-06 / B08：只记 size 挡不住**等长替换**（实测 5 字节 FIRST→OTHER
        # 仍判定 identity-match）。身份必须绑定**内容摘要**。
        "artifact_digest": _artifact_digest(artifact),
        "source_identity": build_input_identity(input_path),
        "params_identity": build_params_identity(config, job, logger),
        # ID-08（第七轮）：把**上游产物**也绑进来。
        # 旧侧车只记 source_identity（原始素材）与 params_identity，
        # 完全没有"这份中间产物是由哪一份上游产物做出来的"——
        # 于是上游换了一版（同参数、同输入、不同中间结果）也察觉不到。
        "upstream_artifact": (str(upstream_artifact)
                              if upstream_artifact else None),
        "upstream_digest": (_artifact_digest(upstream_artifact)
                            if upstream_artifact else None),
        # ID-07（第七轮）：并入"实际执行"记录（后端/实际模型/权重指纹/工具版本）。
        # 配置里写的是请求；这里记的是真的跑了什么。
        "exec": _read_exec_fingerprint(artifact),
        # ID 第 8 点（第七轮）：本 stage 的**局部**参数身份。
        # 只含影响本 stage 的参数 —— 换 CRF 只让 stage7 失效，
        # 换日志级别不失效，换人脸模型不影响传统降噪。
        "stage_identity": _stage_identity_for(params, stage_num),
        "stage_scope": _stage_scope_snapshot(params, stage_num),
        # COL-04（第七轮）：记录这一代**实际**的颜色表示。字段分开保留
        # （pix_fmt/range/matrix/primaries/transfer），不塌缩成一个
        # "standard"；探测失败时 probed=False，字段一律 None，
        # 绝不用默认值冒充"已知"。
        "color": _read_color_description(artifact),
        "recorded_at": datetime.now().astimezone().isoformat(),
    }
    try:
        path = intermediate_identity_path(artifact)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True, indent=2)
        os.replace(tmp, path)
        return True
    except Exception as e:
        if logger:
            logger.warning("STAGE_IDENTITY_WRITE_FAIL",
                           f"Could not record identity for {artifact}: {e}")
        return False


def intermediate_identity_matches(artifact: str, job: Dict[str, Any],
                                  config: Dict[str, Any],
                                  logger=None,
                                  expected_stage: Optional[int] = None,
                                  expected_upstream: Optional[str] = None
                                  ) -> Tuple[bool, str]:
    """R2-07 补完：这个中间产物**是不是**本次 job / 本次输入 / 本次参数产出的。

    "能解码"只证明文件没坏，证明不了它属于本次任务。
    返回 ``(ok, reason)``；侧车缺失或读不出来一律按 **unproven** 拒绝
    （安全方向：重算，不是错配）。
    """
    path = intermediate_identity_path(artifact)
    if not os.path.exists(path):
        return False, ("intermediate has no identity sidecar (produced by an "
                       "older implementation or an interrupted run)")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"unreadable identity sidecar: {e}"
    if str(data.get("impl", "")) != VIDEO_IMPL_VERSION:
        return False, (f"produced by implementation {data.get('impl')!r}, "
                       f"current is {VIDEO_IMPL_VERSION!r}")
    if str(data.get("job_id", "")) != str(job.get("job_id", "")):
        return False, f"belongs to job {data.get('job_id')!r}, not this one"
    recorded_src = data.get("source_identity") or {}
    current_src = build_input_identity(job.get("input_path", ""))
    status = _input_identity_status(recorded_src, current_src)
    if status == "changed":
        return False, "source content changed since this intermediate was made"
    if status == "unproven":
        return False, "source identity cannot be verified for this intermediate"
    recorded_params = data.get("params_identity")
    if not str(recorded_params or "").strip():
        # ID-01（第七轮）第二处：旧写法
        #     if recorded_params and recorded_params != ...
        # 空值短路放行，等于"没有身份也当有效中间产物复用"。
        # 最终产物与中间产物**两条路径都要拒绝**（此前只修了一条会漏）。
        # 注意：拒绝复用 ≠ 删除文件；旧文件保留，只是不进严格恢复。
        return False, ("intermediate has no params identity; cannot prove it "
                       "matches the current parameters (file kept)")
    if recorded_params != build_params_identity(config, job, logger):
        return False, "effective parameters changed since it was produced"
    # ID-08（第七轮）：侧车记了 stage，却从来没比对过 —— 于是
    # "把 step3 的文件塞进 step5 的位置"这种错配发现不了。
    if expected_stage is not None:
        recorded_stage = data.get("stage")
        if recorded_stage is None:
            return False, ("identity sidecar does not record which stage "
                           "produced this artifact")
        if int(recorded_stage) != int(expected_stage):
            return False, (f"artifact was produced by stage {recorded_stage}, "
                           f"expected stage {expected_stage}")

    # ID-08：上游摘要链。上游换了一版（同输入同参数、不同中间结果）时，
    # 只看 source/params 是察觉不到的。
    if expected_upstream:
        rec_up = data.get("upstream_digest")
        cur_up = _artifact_digest(expected_upstream)
        if not rec_up or not cur_up:
            return False, ("upstream digest missing on one side; cannot prove "
                           "this intermediate came from the current upstream")
        if str(rec_up) != str(cur_up):
            return False, ("upstream artifact changed since this intermediate "
                           "was produced")

    # R3-06 / B08：内容摘要必须一致（等长替换也要能发现）；
    # 任一侧拿不到摘要 → unproven，拒绝复用。
    recorded_digest = data.get("artifact_digest")
    current_digest = _artifact_digest(artifact)
    if not recorded_digest or not current_digest:
        return False, ("artifact content digest missing on one side; refusing "
                       "to reuse an unverified intermediate")
    if str(recorded_digest) != str(current_digest):
        return False, "artifact content changed since it was recorded"
    size = data.get("artifact_size")
    if size is not None and os.path.exists(artifact) and \
            int(size) != os.path.getsize(artifact):
        return False, "artifact size differs from what was recorded"
    return True, "identity-match"


def _apply_startup_job_decision(manifest_path: str, job_id: str,
                                decision: Dict[str, Any],
                                expect_statuses: Optional[list],
                                logger=None) -> bool:
    """R2-04 / A09: 启动期对**单个 job** 做锁内局部更新。

    旧代码在启动期对整份 manifest 做 `save_manifest(manifest, ...)`，
    而 `manifest` 是本次进程开头读入的**旧快照**：另一个运行在此期间写好的
    DONE 会被整表覆盖回 PENDING，新 generation 也被抹掉（A09 实测）。

    这里改成：
      1. 锁内**重新读取**当前磁盘内容；
      2. 只改这一个 job 的指定字段；
      3. 若磁盘上该 job 已经不在预期状态（例如已经是 DONE），
         **不写**。状态只能前进，不能靠旧快照倒退。

    Returns:
        True 表示写入成功；False 表示跳过或失败。
    """
    def mutator(doc):
        for j in doc.get("jobs", []):
            if str(j.get("job_id", "")) != str(job_id):
                continue
            current = str(j.get("status", ""))
            if expect_statuses and current not in expect_statuses:
                if logger:
                    logger.info("STARTUP_DECISION_SKIPPED",
                                f"Job {job_id} is now {current!r} on disk; "
                                "not applying a decision made from a stale "
                                "snapshot",
                                job_id=job_id, on_disk_status=current,
                                expected=list(expect_statuses))
                return False
            # 状态倒退保护：绝不能把 DONE/FAILED 的完成态退回 PENDING
            new_status = str(decision.get("status", current))
            if current in ("DONE",) and new_status in ("PENDING", "FAILED"):
                if logger:
                    logger.info("STARTUP_DECISION_SKIPPED",
                                f"Refusing to move job {job_id} from DONE "
                                f"back to {new_status}", job_id=job_id)
                return False
            j.update(decision)
            return True
        return False

    try:
        # R3-04 / B01：**不能再把 save_manifest 当 writer 传进去**。
        # update_manifest_with_writer 已经在锁内，而 save_manifest 内部又会去
        # 获取同一把（非重入的）文件锁 —— 于是自己等自己超时，返回 False、
        # 磁盘保持 PENDING、还记一条 STARTUP_DECISION_FAILED（实测 B01）。
        # 第三轮传 save_manifest 是为了"所有写回经过同一入口"，但这在持锁
        # 上下文中是错的。锁内必须用**不再加锁**的写入函数（默认即如此）。
        return update_manifest_with_writer(manifest_path, mutator)
    except Exception as e:
        if logger:
            logger.warning("STARTUP_DECISION_FAILED",
                           f"Locked startup update failed for {job_id}: {e}",
                           job_id=job_id)
        return False


def _persist_job_result(manifest_path: str, job_id: str, status: str,
                        error_message: str = "", attempt: Optional[int] = None,
                        attempt_group: Optional[str] = None,
                        owner: Optional[str] = None,
                        generation: Optional[int] = None,
                        logger=None) -> Tuple[bool, str]:
    """R2-06 / A08: 持久化边界统一处理。

    `save_manifest_job_result` 的真实契约是"成功返回 True，失败返回 False 或
    **抛异常**"（比如 `Path.replace` 被拒 → RuntimeError/PermissionError）。
    旧代码只判断了返回值，异常直接穿透 worker 边界，调用方拿不到结构化的
    `persist-failed`，更拿不到 82 —— 一次磁盘拒绝就变成了未分类崩溃。

    这里把锁超时、写入、替换异常都收口，保留原始原因，统一返回
    ``(ok, reason)``。磁盘彻底不可写时**不谎称**失败状态已落盘：
    返回失败并把原因交给调用方写日志/标准错误。
    """
    try:
        saved = save_manifest_job_result(
            manifest_path, job_id, status, error_message=error_message,
            attempt=attempt, attempt_group=attempt_group,
            owner=owner, generation=generation)
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        if logger:
            logger.error("JOB_RESULT_PERSIST_EXCEPTION",
                         f"Persisting job {job_id} raised {reason}",
                         job_id=job_id, status=status)
        return False, reason
    if not saved:
        return False, "save_manifest_job_result returned False"
    return True, ""


def _run_single_job(job: Dict[str, Any], config: Dict[str, Any],
                    args, logger, manifest: Dict[str, Any], manifest_path: str,
                    attempt_group: Optional[str] = None,
                    owner: Optional[str] = None,
                    generation: Optional[int] = None) -> Dict[str, Any]:
    """Run a single job with retry logic and manifest update.

    FIX-V08: manifest persistence goes through save_manifest_job_result
    (locked read-merge-write of one job). ORIGINAL saved the worker's whole
    stale snapshot, so a concurrent worker's completion was lost.

    RV-10: returns a **structured result**. ORIGINAL returned nothing, so the
    parent could not tell success from failure and always exited 0 — while in
    memory the job was already marked FAILED. A result that cannot be
    persisted is reported as ``persist-failed`` (never as success).
    """
    job_id = job["job_id"]
    retry_count = config.get("global", {}).get("retry_count", 2)
    success = False
    final_attempt = 0
    last_error = ""
    exit_code = 0
    for attempt in range(retry_count + 1):
        exit_code, stderr = process_job(job, config, args, logger)
        if exit_code in (0, 3):
            success = True
            final_attempt = attempt
            break
        else:
            from checkpoint import is_retryable_error
            last_error = f"exit={exit_code}: {stderr}"[:2000]
            if is_retryable_error(exit_code, stderr) and attempt < retry_count:
                wait = 5 if attempt == 0 else 15
                logger.warning("JOB_RETRY", f"Retrying job {job_id} in {wait}s")
                time.sleep(wait)
            else:
                final_attempt = attempt
                break
    status = JOB_DONE if success else JOB_FAILED
    if not success and exit_code == EXIT_INTERRUPTED:
        status = JOB_CANCELLED
    # NEW-V-06: manifest 的持久状态词表只有 PENDING/DONE/FAILED/SKIP
    # （manifest_loader.VALID_STATUSES，update_job_status 对越界值抛
    # ValueError）。取消**不**写成新状态：写成 FAILED，下次运行才会重试
    # 它——取消不是内容性失败，不该把 job 永久晾着。但"被取消过"必须留在
    # 持久审计里，所以 error_message 带 `cancelled:` 前缀；内存态 status
    # 仍是 JOB_CANCELLED（_result / BATCH_SUMMARY / 退出码 88 可见）。
    persist_status = "DONE" if success else "FAILED"
    audit_error = last_error
    if status == JOB_CANCELLED:
        audit_error = f"cancelled: {last_error}" if last_error else "cancelled"
    # 内存快照同步（供本次进程内后续逻辑读取）；落盘由下面的锁定单任务更新完成
    update_job_status(manifest, job_id, persist_status,
                      error_message=audit_error)

    saved, save_reason = _persist_job_result(
        manifest_path, job_id, persist_status,
        error_message=audit_error, attempt=final_attempt,
        attempt_group=attempt_group, owner=owner, generation=generation,
        logger=logger)
    if not saved:
        # 结果没落盘 = 这次运行不能被算作完成：重跑时必须重新处理
        logger.error("JOB_RESULT_PERSIST_FAILED",
                     f"Job {job_id} result could not be written to the "
                     "manifest; reporting it as NOT complete",
                     job_id=job_id, status=status, reason=save_reason)
        return _result(job_id, JOB_PERSIST_FAILED, EXIT_PERSIST_FAILED,
                       (last_error or save_reason or "manifest save failed")
                       + " | " + save_reason)
    return _result(job_id, status, 0 if success else int(exit_code), last_error)


def main():
    args = parse_args()
    config = load_config(args.config)
    gcfg = config.get("global", {})
    log_dir = gcfg.get("log_dir", "./logs")
    # ---- bootstrap 日志必须有**确定且受控**的落点 ----
    # 原实现直接用配置里的相对路径（默认 "./logs"）→ 相对**当前工作目录**：
    # 同一个 run 从不同 cwd 启动会写到不同地方，甚至写到项目之外。
    # 审核 R17-06：bootstrap 日志可以有自己的授权位置，但不能默认为不受约束。
    # 这里把相对路径锚到**项目根**（与"从项目根启动"时的既有落点一致，
    # 不改变现有布局）；显式配置的绝对路径原样尊重 —— 那是用户的选择。
    log_dir = _resolve_log_dir(log_dir)
    logger = get_logger(log_dir=log_dir, level=gcfg.get("log_level", "INFO"))
    logger.info("PIPELINE_START", "Video restoration pipeline starting")

    # R2-07 / A11：CLI 覆盖必须在**任何身份/恢复判定之前**并入 config。
    # 旧代码把 --ai-enabled 只交给 process_job 内部的 cli_overrides，
    # 而 build_params_identity() 走的是 merge_params(job, config, {})，
    # 于是"用 AI 关闭跑出来的旧产物"在加了 --ai-enabled 之后仍然被判为可复用
    # （实测：执行函数调用次数为 0，直接标 DONE）。
    if getattr(args, "ai_enabled", False):
        ai_cfg = config.get("ai")
        ai_cfg = dict(ai_cfg) if isinstance(ai_cfg, dict) else {}
        if ai_cfg.get("enabled") is not True:
            logger.info("CLI_CONFIG_OVERRIDE",
                        "CLI --ai-enabled merged into config before identity "
                        "checks", key="ai.enabled", old=ai_cfg.get("enabled"),
                        new=True)
        ai_cfg["enabled"] = True
        config["ai"] = ai_cfg

    # Dependency check
    try:
        from dependency_check import run_all_checks
        ok, issues, dep_warnings = run_all_checks(config)
        if dep_warnings:
            for w in dep_warnings:
                logger.warn("DEP_CHECK", w)
        # FIX-W19: Per v7.32 §3.6, sRestore is an OPTIONAL dependency - its
        # absence must NOT block startup but MUST disable the IVTC auto-fallback
        # path (which depends on sRestore). detect_hum_frequency-style: scan the
        # non-blocking warnings for sRestore unavailability and set a config
        # flag that process_job reads to force ivtc_auto_fallback=False. ORIGINAL
        # made sRestore non-blocking (good) but never propagated the availability
        # to the fallback gate, so a job could still request sRestore fallback and
        # fail opaquely when the plugin was missing.
        srestore_unavailable = any(
            "srestore unavailable" in str(w).lower()
            or "srestore returned none" in str(w).lower()
            for w in (dep_warnings or [])
        )
        if srestore_unavailable:
            config["_srestore_unavailable"] = True
            logger.warning("IVTC_AUTO_FALLBACK_DISABLED",
                          "sRestore optional dependency unavailable - "
                          "IVTC auto-fallback disabled per v7.32 §3.6")
        if not ok:
            for issue in issues:
                logger.error("DEP_CHECK", issue)
            sys.exit(80)
    except Exception as e:
        logger.warning("DEP_CHECK", f"Check skipped: {e}")

    # V-P-001~004 fix: mode-aware manifest handling
    mode = args.mode
    if args.scan or (mode == "scan"):
        if not args.input:
            print("Error: --scan requires --input directories")
            sys.exit(81)
        # FIX-P5-3: Use --output-manifest when provided per v7.32 §3.3
        manifest_path = (getattr(args, 'output_manifest', None)
                         or args.manifest
                         or str(SCRIPT_DIR / "batch_manifest.yaml"))
        from manifest_generate import generate_manifest
        # FIX-P5-4: Pass --incremental flag to generate_manifest per v7.32 §3.3
        incremental = getattr(args, 'incremental', False)
        # FIX-R5-07: 把 --output 传给扫描器，manifest 的 output_path 才会
        # 落在用户指定的目录（绝对路径）。原来扫描忽略 --output，manifest
        # 记的是 config 的相对 "./output"，执行时按 cwd 解析 -> 落错地方
        # 或 stage7 直接 exit 84（目录不存在）。
        generate_manifest(args.input, manifest_path, config, "video",
                          incremental=incremental,
                          cli_output_dir=getattr(args, 'output', None))
        logger.info("SCAN_DONE", f"Manifest saved to {manifest_path}")
        print(f"Scan complete. Manifest saved to {manifest_path}")
        print("Next: Review with --edit-manifest, then execute with --mode batch")
        return
    elif args.edit_manifest or (mode == "edit"):
        manifest_path = args.manifest or str(SCRIPT_DIR / "batch_manifest.yaml")
        if not Path(manifest_path).exists():
            print(f"Error: Manifest not found: {manifest_path}")
            sys.exit(81)
        from manifest_editor import edit_manifest_interactive
        edit_manifest_interactive(manifest_path)
        logger.info("EDIT_DONE", f"Manifest edited: {manifest_path}")
        print(f"Manifest editing complete: {manifest_path}")
        print("Next: Execute with --mode batch")
        return
    elif mode == "quick":
        # V-P-001~004 fix: quick mode (scan + execute, no persistent manifest)
        if not args.input:
            print("Error: --mode quick requires --input directories")
            sys.exit(81)
        manifest_path = str(SCRIPT_DIR / "batch_manifest_quick.yaml")
        from manifest_generate import generate_manifest
        generate_manifest(args.input, manifest_path, config, "video",
                          cli_output_dir=getattr(args, 'output', None))
        # FIX-W7: wrap execution in try/finally so the temporary quick manifest
        # is removed even if a job crashes mid-batch. ORIGINAL only unlinked on
        # the happy path, leaving batch_manifest_quick.yaml behind on crash.
        # RV-10: 串行路径也必须收集结构化结果并聚合退出码
        results: List[Dict[str, Any]] = []
        try:
            manifest = load_manifest(manifest_path)
            jobs = filter_jobs(manifest, ["PENDING"])
            logger.info("QUICK_START", f"Quick mode: {len(jobs)} jobs")
            print(f"Quick mode: scanning done, executing {len(jobs)} jobs...")
            for job in jobs:
                results.append(_run_single_job(job, config, args, logger,
                                               manifest, manifest_path))
        finally:
            try:
                Path(manifest_path).unlink(missing_ok=True)
            except Exception:
                pass
        rc = aggregate_results(results, logger)
        if rc != 0:
            for r in results:
                if r["status"] != JOB_DONE:
                    print(f"[{r['job_id']}] {r['status']} "
                          f"(exit {r['exit_code']})", file=sys.stderr)
            logger.info("PIPELINE_DONE",
                        f"Video pipeline finished with failures (quick mode, rc={rc})")
            print("Finished with failures.", file=sys.stderr)
            sys.exit(rc)
        logger.info("PIPELINE_DONE", "Video pipeline complete (quick mode)")
        print("Done.")
        return
    elif mode == "interactive":
        # V-P-001~004 fix: interactive mode (single file, step-by-step)
        if not args.input or len(args.input) != 1:
            print("Error: --mode interactive requires exactly one --input file")
            sys.exit(81)
        input_file = args.input[0]
        if not Path(input_file).is_file():
            print(f"Error: Input file not found: {input_file}")
            sys.exit(81)
        # Create a single-job manifest
        manifest_path = str(SCRIPT_DIR / "batch_manifest_interactive.yaml")
        from manifest_generate import generate_manifest
        generate_manifest([str(Path(input_file).parent)], manifest_path, config, "video")
        manifest = load_manifest(manifest_path)
        jobs = [j for j in manifest.get("jobs", []) if j.get("input_path") == input_file]
        if not jobs:
            print(f"Error: File not found in scanned manifest: {input_file}")
            sys.exit(81)
        job = jobs[0]
        print(f"\nInteractive mode: {input_file}")
        print(f"  Detected class: {job.get('detected_class', 'unknown')}")
        print(f"  Field order: {job.get('field_order', 'unknown')}")
        print(f"  Recommended params: {job.get('recommended_params', {})}")
        # FIX-W7: low-confidence classification requires explicit manual
        # confirmation per v7.32 §3.9 ("<80% 时交互模式强制人工确认").
        _conf = job.get("confidence", 0.0)
        if job.get("needs_confirmation", False) or (isinstance(_conf, (int, float)) and _conf < 0.80):
            logger.info("LOW_CONFIDENCE_CONFIRM",
                       f"Confidence {_conf:.2f} < 0.80, requires manual confirmation",
                       detected=job.get("detected_class", "unknown"))
            try:
                lc_resp = input(
                    f"低置信度({_conf:.2f}),检测为{job.get('detected_class', 'unknown')}。"
                    "继续处理? [y/N]: ").strip().lower()
            except EOFError:
                logger.warning("INTERACTIVE_SKIP",
                              "Non-interactive env, skipping low-confidence file")
                print("Skipped (non-interactive).")
                return
            if lc_resp not in ("y", "yes"):
                logger.info("USER_SKIP", "User declined low-confidence file")
                print("Skipped.")
                return
        confirm = input("Proceed with recommended parameters? [Y/n/edit]: ").strip().lower()
        if confirm in ("n", "no", "skip"):
            print("Skipped.")
            return
        if confirm in ("e", "edit"):
            # Allow simple key=value overrides
            print("Enter parameter overrides (key=value), empty line to finish:")
            while True:
                line = input("> ").strip()
                if not line:
                    break
                if "=" in line:
                    k, v = line.split("=", 1)
                    job.setdefault("user_overrides", {})[k.strip()] = v.strip()
                    print(f"  Set {k.strip()} = {v.strip()}")
        # FIX-P5-7: AI upscale ban double confirmation per v7.32 §3.10.2
        # In interactive mode, if file is AI-banned and AI is enabled, require explicit override
        job_params = merge_params(job, config, {}, logger)
        if job.get("ai_upscale_banned") and job_params.get("ai_enabled"):
            print(f"\n⚠️  WARNING: This file is flagged as AI-upscale-banned "
                  f"(detected_class={job.get('detected_class')}).")
            print(f"  AI super-resolution may produce artifacts on already-high-resolution content.")
            confirm_ban = input("Force AI upscale anyway? [y/N]: ").strip().lower()
            if confirm_ban not in ("y", "yes"):
                print("AI upscale disabled for this file.")
                job.setdefault("user_overrides", {})["ai_enabled"] = False
        # Run single job
        # RV-10: 单任务模式同样要按结果决定退出码
        _res = _run_single_job(job, config, args, logger, manifest, manifest_path)
        try:
            Path(manifest_path).unlink(missing_ok=True)
        except Exception:
            pass
        rc = aggregate_results([_res], logger)
        if rc != 0:
            print(f"[{job['job_id']}] {_res['status']} "
                  f"(exit {_res['exit_code']})", file=sys.stderr)
            logger.info("PIPELINE_DONE",
                        f"Video pipeline finished with failures "
                        f"(interactive mode, rc={rc})")
            print("Finished with failures.", file=sys.stderr)
            sys.exit(rc)
        logger.info("PIPELINE_DONE", "Video pipeline complete (interactive mode)")
        print("Done.")
        return

    # Generate or load manifest
    if args.manifest:
        manifest_path = args.manifest
    elif args.input:
        manifest_path = str(SCRIPT_DIR / "batch_manifest.yaml")
        from manifest_generate import generate_manifest
        # FIX-P5-4: Pass --incremental flag to generate_manifest
        incremental = getattr(args, 'incremental', False)
        generate_manifest(args.input, manifest_path, config, "video",
                          incremental=incremental)
    else:
        print("Error: Provide --input or --manifest")
        sys.exit(81)

    manifest = load_manifest(manifest_path)

    # V-P-010 / BUG-044 fix: FAILED job downgrade logic with stage-aware retry
    # Verify intermediate file integrity for FAILED jobs and resume from last_completed_stage+1
    try:
        from checkpoint import verify_ffprobe_decodeable
        temp_dir = config.get("global", {}).get("temp_dir", "./temp")
        for job in manifest.get("jobs", []):
            if job.get("status") != "FAILED":
                continue
            job_id = job.get("job_id", "")
            input_path = job.get("input_path", "")
            # RV-09: 素材换了就绝不复用旧 intermediate —— 中间产物的名字里
            # 只有 job_id/stem/stage/param_hash，无法反映"输入内容是否变了"。
            current_ident = build_input_identity(input_path)
            recorded_ident = job.get("input_identity")
            input_changed = bool(
                isinstance(recorded_ident, dict) and
                recorded_ident.get("digest") and current_ident.get("digest") and
                recorded_ident["digest"] != current_ident["digest"])
            if input_changed:
                logger.info("RESUME_INPUT_CHANGED",
                            f"Job {job_id}: input content changed since the "
                            "intermediates were produced; recomputing from "
                            "scratch", input=input_path)
            job["input_identity"] = current_ident
            last_completed = -1
            if not input_changed:
                # FIX-P5-6: Scan step0..step7 to find highest valid intermediate,
                # check all candidates
                # FIX-V03: use the shared artifact-identity helper (stem, not name)
                for stage_num in range(7, -1, -1):
                    candidates = find_stage_artifacts(temp_dir, job_id, input_path,
                                                      stage_num)
                    for cand in candidates:
                        # R3-06：FAILED 恢复入口同样必须过逐产物身份门 ——
                        # 只查"能解码"会把别的输入/参数产出的文件当成自己的成果。
                        ok_id, why = intermediate_identity_matches(
                            cand, job, config, logger,
                            expected_stage=stage_num)
                        if not ok_id:
                            logger.info(
                                "RESUME_ARTIFACT_IDENTITY_REJECTED",
                                f"Job {job_id}: {cand} cannot be bound to "
                                f"this run ({why})", candidate=cand)
                            continue
                        if verify_ffprobe_decodeable(cand):
                            last_completed = stage_num
                            break
                    if last_completed >= 0:
                        break
            if last_completed >= 0:
                logger.info("FAILED_DOWNGRADE",
                           f"Job {job_id} has valid step{last_completed}, downgrading to PENDING "
                           f"(will resume from stage {last_completed + 1})")
                # R2-04：锁内局部更新（不用开头的旧快照整表覆盖）
                _apply_startup_job_decision(
                    manifest_path, job_id,
                    {"status": "PENDING",
                     "last_completed_stage": last_completed,
                     "resume_from_stage": last_completed + 1,
                     "input_identity": current_ident},
                    expect_statuses=["FAILED"], logger=logger)
            else:
                # ID-06（第七轮）：**身份不符 ≠ 文件损坏**。
                #
                # 旧实现（FIX-P5-2）在这里把该 job 的所有中间产物一律
                # `os.remove()` 并记成 "corrupt intermediate"。但走到这个
                # 分支的原因通常是：
                #   * 参数变了（改了 CRF/模型/滤镜）→ 产物本身完好；
                #   * 输入内容变了 → 产物完好，只是不属于这次输入；
                #   * 侧车缺身份（旧格式）→ 产物完好，只是无法证明归属。
                # 把它们当"损坏"删掉，等于**用户改了参数就静默销毁上一版
                # 结果**，且日志谎称是损坏。
                #
                # 新行为：
                #   * 只有 ffprobe 真的解不开的，才按损坏删除（记录在案）；
                #   * 身份不符/身份缺失的全部**保留**，标记 legacy/unproven，
                #     让用户看得见、可导出，但不进严格恢复。
                _truly_corrupt: List[str] = []
                _kept_unproven: List[str] = []
                for _stage_num in range(8):
                    for fpath in find_stage_artifacts(
                            temp_dir, job_id, input_path, _stage_num):
                        if verify_ffprobe_decodeable(fpath):
                            _kept_unproven.append(fpath)
                            continue
                        _truly_corrupt.append(fpath)
                        try:
                            os.remove(fpath)
                            logger.info("FAILED_CLEANUP",
                                        "Removed genuinely undecodable "
                                        "intermediate", path=fpath)
                        except Exception as e:
                            logger.warning("FAILED_CLEANUP_FAIL",
                                           f"Could not remove {fpath}: {e}",
                                           path=fpath)
                if _kept_unproven:
                    logger.warning(
                        "RESUME_ARTIFACTS_KEPT_UNPROVEN",
                        f"Job {job_id}: kept {len(_kept_unproven)} intermediate "
                        "file(s) that could not be bound to this run but are "
                        "still decodable. They are NOT deleted and NOT treated "
                        "as verified; they will be recomputed as needed.",
                        kept=_kept_unproven[:10])
                logger.info("FAILED_DOWNGRADE",
                           f"Job {job_id} has no reusable intermediates "
                           f"({len(_truly_corrupt)} removed as corrupt, "
                           f"{len(_kept_unproven)} kept as unproven); "
                           "downgraded to PENDING")
                _apply_startup_job_decision(
                    manifest_path, job_id,
                    {"status": "PENDING",
                     "last_completed_stage": -1,
                     "resume_from_stage": 0,
                     "input_identity": current_ident},
                    expect_statuses=["FAILED"], logger=logger)
    except Exception as e:
        logger.warning("FAILED_CHECK", f"FAILED job check skipped: {e}")

    # BUG-043 / V-P-004 fix: resume mode must check step7 first; only DONE if final output valid
    target_statuses = ["PENDING", "FAILED"]
    if args.resume:
        try:
            from checkpoint import verify_ffprobe_decodeable
            temp_dir = config.get("global", {}).get("temp_dir", "./temp")
            for job in manifest.get("jobs", []):
                if job.get("status") != "PENDING":
                    continue
                job_id = job.get("job_id", "")
                input_path = job.get("input_path", "")
                # Only mark DONE if step7 output exists and is valid
                step7_done = False
                # FIX-V03: temp 内的 step7 中间产物 + output_dir 的最终产物都要看
                candidates = find_stage_artifacts(temp_dir, job_id, input_path, 7)
                candidates += find_final_output(job, config, args)
                # RV-09: 产物必须"能解码"**且**属于当前 job（输入内容 + 有效参数
                # + 实现版本）。只查名字和可解码性会把别的 job / 别的参数 / 别的
                # 素材的产物当成自己的成果。
                for cand in candidates:
                    if not verify_ffprobe_decodeable(cand):
                        continue
                    ok, reason = resume_artifact_matches(cand, job, config,
                                                         logger)
                    if not ok:
                        logger.info("RESUME_REJECTED",
                                    f"Job {job_id}: {cand} is not a valid "
                                    f"result for the current input/params "
                                    f"({reason}); recomputing",
                                    candidate=cand)
                        continue
                    step7_done = True
                    logger.info("RESUME_DONE",
                                f"Job {job_id} has a verified final output "
                                f"{cand}, marking DONE", candidate=cand)
                    # R2-04：锁内把这一个 job 置为 DONE（不用旧快照整表覆盖）
                    _apply_startup_job_decision(
                        manifest_path, job_id,
                        {"status": "DONE",
                         "input_identity": build_input_identity(input_path)},
                        expect_statuses=["PENDING", "FAILED"], logger=logger)
                    break
                if not step7_done:
                    # R2-07 / A17：部分恢复**先过源身份门**。
                    # 旧代码只做"中间文件存在 + 能解码"，素材换了照样从 stage7 接着跑。
                    # 中间产物名字里只有 job_id/stem/stage，无法证明它属于当前输入，
                    # 因此必须以 job 记录的 input_identity 为准；记录缺失或读不出来
                    # 时按 "unproven" 拒绝复用（安全方向：重算）。
                    current_ident = build_input_identity(input_path)
                    ident_status = _input_identity_status(
                        job.get("input_identity"), current_ident)
                    if ident_status == "changed":
                        logger.info("RESUME_INPUT_CHANGED",
                                    f"Job {job_id}: input content changed; "
                                    "old intermediates are not reused",
                                    input=input_path)
                    elif ident_status == "unproven":
                        logger.info("RESUME_IDENTITY_UNPROVEN",
                                    f"Job {job_id}: cannot prove the existing "
                                    "intermediates were made from this input; "
                                    "recomputing from scratch",
                                    input=input_path)
                    job["input_identity"] = current_ident
                    # Determine highest valid completed stage for partial resume
                    last_completed = -1
                    if ident_status == "match":
                        for stage_num in range(6, -1, -1):
                            candidates = find_stage_artifacts(temp_dir, job_id,
                                                              input_path, stage_num)
                            for cand in candidates:
                                # R2-07 补完：解码成功只说明文件没坏，
                                # 还要证明它属于**本次**输入/参数/实现。
                                ok_id, why = intermediate_identity_matches(
                                    cand, job, config, logger)
                                if not ok_id:
                                    logger.info(
                                        "RESUME_ARTIFACT_IDENTITY_REJECTED",
                                        f"Job {job_id}: {cand} cannot be "
                                        f"bound to this run ({why})",
                                        candidate=cand)
                                    continue
                                if verify_ffprobe_decodeable(cand):
                                    last_completed = stage_num
                                    break
                            if last_completed >= 0:
                                break
                    if last_completed >= 0:
                        logger.info("RESUME_PARTIAL",
                                   f"Job {job_id} resuming from stage {last_completed + 1}")
                        # R2-04：锁内逐 job 局部更新
                        _apply_startup_job_decision(
                            manifest_path, job_id,
                            {"last_completed_stage": last_completed,
                             "resume_from_stage": last_completed + 1,
                             "input_identity": current_ident},
                            expect_statuses=["PENDING"], logger=logger)
                    else:
                        _apply_startup_job_decision(
                            manifest_path, job_id,
                            {"last_completed_stage": -1,
                             "resume_from_stage": 0,
                             "input_identity": current_ident},
                            expect_statuses=["PENDING"], logger=logger)
        except Exception as e:
            logger.warning("RESUME_CHECK", f"Resume check skipped: {e}")

    # R3-04 / B02：启动期的降级与 resume 决议可能已经把某些任务改成 DONE，
    # 而 `manifest` 是本次进程开头读入的**旧快照**。若直接拿它筛选，
    # 磁盘上已经完成的任务会被再执行一次（实测：磁盘已 DONE，
    # process_job 仍被调用 1 次，传入的 job 还是 PENDING）。
    # 执行列表必须来自**最新**的磁盘状态。
    try:
        manifest = load_manifest(manifest_path)
    except Exception as e:
        logger.warning("MANIFEST_REREAD_FAILED",
                       f"Could not re-read the manifest before scheduling: {e}; "
                       "falling back to the startup snapshot")
    jobs = filter_jobs(manifest, target_statuses)
    if args.job_id:
        jobs = [j for j in jobs if j.get("job_id") == args.job_id]
    else:
        # FIX-P5-5: Sort by priority per v7.32 §3.3: priority=1 first, then others
        jobs.sort(key=lambda j: (0 if j.get("priority") == 1 else 1))

    # RV-09: 同名不同 job 不能共用一个最终输出路径；同时把当前输入身份写进
    # job，供下一次运行判断"素材是否换过"。
    _ensure_unique_final_outputs(jobs, config, args, logger)
    for _j in jobs:
        if not _j.get("input_identity"):
            _j["input_identity"] = build_input_identity(_j.get("input_path", ""))

    logger.info("BATCH_START", f"Processing {len(jobs)} video jobs")
    print(f"Processing {len(jobs)} video jobs...")

    # V-P-009 fix: enhanced preview mode with motion segments
    # Also handle mode == "preview"
    # FIX-M1: --preview is an alias for --preview-only (CLAUDE.md §0).
    if args.preview_only or args.preview or (mode == "preview"):
        from preview import extract_preview_frames, generate_preview_clip, generate_comparison_frames
        for job in jobs[:3]:
            try:
                preview_dir = Path(gcfg.get("temp_dir", "./temp")) / "preview"
                preview_dir.mkdir(parents=True, exist_ok=True)
                # Static frames
                frames = extract_preview_frames(job["input_path"], str(preview_dir), num_frames=3)
                logger.info("PREVIEW", f"Preview frames for {job['job_id']}: {frames}")
                # Motion segments (0.5s WebP animation)
                for seg_idx, start_pct in enumerate([0.3, 0.6]):
                    try:
                        dur = job.get("duration", 0)
                        start_sec = dur * start_pct if dur > 0 else start_pct * 30
                        webp_path = str(preview_dir / f"{job['job_id']}_motion_{seg_idx}.webp")
                        if generate_preview_clip(job["input_path"], webp_path,
                                                 start_sec=start_sec, duration_sec=0.5):
                            logger.info("PREVIEW_MOTION",
                                       f"Motion preview for {job['job_id']}: {webp_path}")
                    except Exception as e:
                        logger.warning("PREVIEW_MOTION_FAIL",
                                      f"Motion preview failed for {job['job_id']}: {e}")

                # ISSUE-V-MAIN-002: generate AI before/after comparison frames in preview
                job_params = merge_params(job, config, {}, logger)
                if job_params.get("ai_enabled") or job_params.get("force_allow_banned_upscale"):
                    try:
                        sim_path = str(preview_dir / f"{job['job_id']}_sim_upscale.mkv")
                        # Lightweight Lanczos simulation for quick preview
                        # Real AI comparison is generated when force_allow_banned_upscale=True
                        sim_cmd = [
                            "ffmpeg", "-hide_banner", "-v", "error", "-y",
                            "-i", job["input_path"],
                            "-vf",
                            "scale=iw*2:ih*2:flags=lanczos,unsharp=luma_msize_x=3:luma_msize_y=3:luma_amount=0.5",
                            "-t", "2", "-c:v", "libx264", "-crf", "28",
                            "-pix_fmt", "yuv420p", sim_path,
                        ]
                        sim_r = subprocess.run(sim_cmd, capture_output=True, timeout=60, shell=False)
                        if sim_r.returncode == 0 and os.path.exists(sim_path):
                            comp_frames = generate_comparison_frames(
                                job["input_path"], sim_path, str(preview_dir), num_pairs=2
                            )
                            logger.info("PREVIEW_COMPARE",
                                       f"Comparison frames for {job['job_id']}: {comp_frames}")
                            # If force-allowed, additionally attempt real AI preview on 1s clip
                            if job_params.get("force_allow_banned_upscale"):
                                logger.info("PREVIEW_FORCE",
                                           f"force_allow_banned_upscale active for {job['job_id']}; "
                                           f"run full pipeline to obtain true AI comparison")
                    except Exception as e:
                        logger.warning("PREVIEW_COMPARE_FAIL",
                                      f"Comparison preview failed for {job['job_id']}: {e}")
            except Exception as e:
                logger.warning("PREVIEW_FAIL", f"Preview failed for {job['job_id']}: {e}")
        print("Preview mode complete. Inspect frames and re-run without --preview.")
        return

    # RV-08: 本 run 先领取归属（generation 单调递增），迟到旧 run 的结果
    # 会被 manifest_loader 拒绝；attempt_group 只是 UUID，没有时间顺序。
    # 必须在串行/并行两条路径**之前**完成，否则串行分支拿不到凭据。
    run_owner = f"video-run-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    _claimed_ids = [j.get("job_id") for j in jobs]
    if not _claimed_ids:
        # R3-05：本次没有任务 —— 既不能领取别人的任务，也不必递增代际。
        logger.info("RUN_CLAIM_SKIPPED",
                    "No jobs selected for this run; not claiming anything "
                    "and not bumping the global generation")
        print("No jobs to process.")
        return
    # ---- 领取（= 改写 manifest）之前必须先过路径校验 ----
    # 审核 R17-06：'manifest ownership 写入早于完整计划检查'。领取会**改写
    # manifest 文件本身**；如果它落在授权写根之外，就变成"先写再验"——
    # 事后报错也已经写过磁盘了。这里在写之前先判一次。

    # ---- R16-04：dispatch 前**一次性**检查所有 job 的计划 ----
    # 这里必须用与 job 执行**完全相同**的合并规则和计划解析：
    #   * 同一份 CLI 覆盖（原来传空 dict → 算出的输出可能与实际不同）
    #   * 同一份授权根（CLI 优先、其次 config）
    #   * 同一份 relocate_output（漏传会让"重定位后撞名"在预检里看不见）
    # 三者任一不同，批次检查就只是在检查另一套计划。
    if len(jobs) > 1:
        _plans = []
        _plan_fail = []
        _mem_log = _BufferedLog(sys.stderr)
        for _j in jobs:
            try:
                _jparams, _p, _avail = _resolve_plan_for_job(
                    _j, config, args, _mem_log)
                if not _avail:
                    logger.error(
                        "BATCH_PLAN_MODULE_MISSING",
                        "shared_modules/run_plan.py is unavailable; the batch "
                        "collision check was NOT performed")
                    _plans = None
                    break
                _plans.append(_p)
                if not _p.ok:
                    _plan_fail.append((_j.get("job_id", ""), _p.errors))
            except Exception as _pe:
                _plan_fail.append((_j.get("job_id", ""), [str(_pe)]))

        if _plans is not None:
            if _plan_fail:
                for _jid, _errs in _plan_fail:
                    logger.error("BATCH_PLAN_INVALID",
                                 "A job's effective path plan is invalid; "
                                 "refusing to start the batch.",
                                 job_id=_jid, errors=_errs)
                    # R16-04：光说"计划无效"没用 —— 必须把**哪条路径**越界
                    # 直接写到 stderr，否则用户只能去翻日志猜。
                    print(f"ERROR: invalid run plan for job {_jid}:",
                          file=sys.stderr)
                    for _e in _errs:
                        print(f"  - {_e}", file=sys.stderr)
                print("ERROR: invalid path plan(s); nothing was dispatched.",
                      file=sys.stderr)
                sys.exit(82)

            try:
                from run_plan import detect_cross_job_conflicts as _detect
            except ImportError:
                _detect = None
            if _detect is None:
                logger.error("BATCH_PLAN_MODULE_MISSING",
                             "conflict detection unavailable; refusing to "
                             "dispatch without the check")
                sys.exit(82)

            _conflicts = _detect(_plans)
            if _conflicts:
                for _c in _conflicts:
                    logger.error("BATCH_PLAN_CONFLICT", _c)
                print("ERROR: the jobs in this batch would collide "
                      f"({len(_conflicts)} conflict(s)); nothing was dispatched. "
                      "See the log for detail.", file=sys.stderr)
                sys.exit(82)
            logger.info("BATCH_PLAN_OK",
                        "All jobs' path plans were resolved with the same "
                        "merge rules as job execution and checked before "
                        "dispatch; no collisions",
                        jobs=len(_plans),
                        write_roots=_plan_write_roots(args, config)
                        or "unrestricted")

    # ---- R16-04 / R17-06：**先校验，再改写 manifest** ----
    # 审核指出"manifest ownership 写入早于完整计划检查"。claim 会**改写
    # manifest 文件本身**，所以它必须排在批次计划校验之后：
    # 计划不成立时，一个字节都不该写进 manifest。
    # （注意：不能给 manifest 强加"必须在写根内"的约束 —— `--allow-write-root`
    #  管的是媒体/临时/日志产物，manifest 是用户在命令行显式指定的，
    #  实测那样会把合法的"manifest 在根外、产物在根内"批次也拒掉。）
    run_generation = acquire_run_ownership(
        manifest_path, _claimed_ids, owner=run_owner)
    if run_generation is None:
        # R3-11：领取失败意味着"结果无处可写"，必须停止，不能照常执行并报成功。
        logger.error("RUN_CLAIM_FAILED",
                     "Could not claim this run's jobs in the manifest; "
                     "refusing to report success later")
        print("ERROR: could not claim this run's jobs in the manifest; "
              "not producing artifacts that cannot be recorded.",
              file=sys.stderr)
        sys.exit(82)

    # Parallel worker pool
    workers = max(1, args.workers)
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)

        # Build task contexts for picklable worker submission
        # BUG-040 fix: convert argparse.Namespace to plain dict for spawn safety
        # FIX-V08: one attempt group per pipeline run so late results from an
        # older run cannot overwrite this run's outcomes.
        run_attempt_group = f"run-{os.getpid()}-{uuid.uuid4().hex}"
        task_contexts = []
        for job in jobs:
            task_contexts.append({
                "job": job,
                "config_path": getattr(args, "config"),
                "log_dir": log_dir,
                "manifest_path": manifest_path,
                "manifest_attempt_group": run_attempt_group,
                # RV-08: 运行归属凭据（generation 由父进程领取后下发）
                "manifest_owner": run_owner,
                "manifest_generation": run_generation,
                "args": {
                    "config": getattr(args, "config"),
                    "output": getattr(args, "output", None),
                    "no_wsl_redirect": getattr(args, "no_wsl_redirect", False),
                    "keep_intermediates": getattr(args, "keep_intermediates", False),
                    "gpu": getattr(args, "gpu", None),
                    "stage": getattr(args, "stage", None),
                    # ISSUE-V-MAIN-001 / BUG-047: AI CLI args must propagate to workers
                    "ai_enabled": getattr(args, "ai_enabled", None),
                    "ai_upscale_factor": getattr(args, "ai_upscale_factor", None),
                    "face_restore": getattr(args, "face_restore", None),
                    "ai_model": getattr(args, "ai_model", None),
                    # ---- R16-04：写入边界也必须传给 worker ----
                    # 漏掉这两个，worker 里的 `_plan_write_roots()` 拿不到
                    # 父进程的授权根：父进程以为受限，worker 其实是"无限制"，
                    # 于是计划解析与冲突判定父子不一致。
                    "allow_write_root": getattr(args, "allow_write_root", None),
                    "relocate_output": getattr(args, "relocate_output", None),
                    "fps_interpret": getattr(args, "fps_interpret", None),
                    "force_allow_banned_upscale": getattr(
                        args, "force_allow_banned_upscale", False),
                },
            })

        results: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker_wrapper, ctx): ctx for ctx in task_contexts}
            for future in as_completed(futures):
                ctx = futures[future]
                try:
                    res = future.result()
                    if not isinstance(res, dict):
                        res = _result(ctx["job"]["job_id"], JOB_FAILED, 86,
                                      f"unexpected worker result {res!r}")
                    results.append(res)
                except Exception as e:
                    logger.error("WORKER_ERROR", f"Job {ctx['job']['job_id']} failed in worker: {e}")
                    update_job_status(manifest, ctx["job"]["job_id"], "FAILED")
                    # R3-11：父进程兜底也必须走统一收口 ——
                    # save_manifest_job_result 会抛异常（磁盘拒绝写入等），
                    # 直接调用会让 WORKBENCH 的兜底自己崩掉，
                    # 既没有 persist-failed 也没有 82。
                    saved, save_reason = _persist_job_result(
                        manifest_path, ctx["job"]["job_id"], "FAILED",
                        error_message=str(e)[:2000],
                        attempt_group=run_attempt_group,
                        owner=run_owner, generation=run_generation,
                        logger=logger)
                    if not saved:
                        logger.error("WORKER_PERSIST_FAILED",
                                     f"Could not record the failure of job "
                                     f"{ctx['job']['job_id']}: {save_reason}")
                    results.append(_result(
                        ctx["job"]["job_id"],
                        JOB_FAILED if saved else JOB_PERSIST_FAILED,
                        86 if saved else EXIT_PERSIST_FAILED, str(e)[:2000]))
    else:
        results = []
        for job in jobs:
            results.append(_run_single_job(
                job, config, args, logger, manifest, manifest_path,
                attempt_group=f"seq-{os.getpid()}-{uuid.uuid4().hex}",
                owner=run_owner, generation=run_generation))

    # RV-10: 顶层聚合 —— 有任何失败/取消/落盘失败就不能以 0 退出
    rc = aggregate_results(results, logger)
    if rc != 0:
        for r in results:
            if r["status"] != JOB_DONE:
                print(f"[{r['job_id']}] {r['status']} "
                      f"(exit {r['exit_code']})", file=sys.stderr)
        logger.info("PIPELINE_DONE", f"Video pipeline finished with failures (rc={rc})")
        print(f"\nFinished with failures: exit {rc}", file=sys.stderr)
        sys.exit(rc)

    logger.info("PIPELINE_DONE", "Video pipeline complete")
    print("Done.")


if __name__ == "__main__":
    main()
