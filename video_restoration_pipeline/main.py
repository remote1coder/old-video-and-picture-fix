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


def parse_args():
    parser = argparse.ArgumentParser(description="Video Restoration Pipeline")
    parser.add_argument("--input", "-i", nargs="+", help="Input directories")
    parser.add_argument("--output", "-o", help="Output directory")
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
    paths = config.get("paths", {})
    if path_key in paths:
        for k, v in paths[path_key].items():
            _set_param(k, v, "paths")

    # 4. Override with audio_override (CORRECT POSITION: after paths, before recommended_params)
    #    ORIGINAL: audio_override was applied after user_overrides (line 147-154),
    #    causing it to override user_overrides. Now correctly between paths and recommended_params.
    audio_override = config.get("audio_override", {})
    if path_key in audio_override:
        for k, v in audio_override[path_key].items():
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


def _calculate_timeout(video_duration: float, has_ai: bool = False) -> int:
    """Calculate stage timeout based on video duration and complexity.

    Base: 10 minutes setup + 2 min per minute of video
    AI upscale: +5 min per minute (frame extraction + model inference)
    Max: 4 hours
    """
    base = 600  # 10 min setup
    per_minute = 120  # 2 min per video minute
    if has_ai:
        per_minute = 420  # 7 min per video minute (2 + 5 for AI)

    timeout = base + int(video_duration / 60) * per_minute
    return min(timeout, 3600 * 4)  # Cap at 4 hours


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
    import tempfile
    params_json = json.dumps(params, default=str)
    params_file = None
    # Windows cmd limit is 32767; leave safety margin. Also use file for any long dict.
    if len(params_json) > 30000:
        try:
            fd, params_file = tempfile.mkstemp(suffix=".json", prefix="vparams_")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(params_json)
            cmd.extend(["--params-file", params_file])
        except Exception:
            # Fallback to inline JSON if temp file fails
            cmd.extend(["--params", params_json])
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
    timeout = _calculate_timeout(video_duration, has_ai)

    # Run with timeout
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SCRIPT_DIR.parent)

        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, env=env, shell=False)
        stderr = r.stderr if r.stderr else ""
        return r.returncode, stderr
    except subprocess.TimeoutExpired:
        return 87, f"Stage timed out after {timeout}s"  # Resource不足 (timeout)
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
            capture_output=True, text=True, timeout=30, shell=False
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


def build_params_identity(config: Dict[str, Any], job: Dict[str, Any],
                          logger=None) -> str:
    """RV-09: digest of the *effective* parameters of this job.

    Uses the real 6-layer merge (merge_params) so a changed config or a changed
    user override invalidates previously produced artifacts.
    """
    try:
        params = merge_params(job, config, {}, logger)
    except Exception:
        params = dict(job or {})
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False,
                         default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def completion_marker_path(final_path: str) -> str:
    return str(final_path) + ".done.json"


def write_completion_marker(final_path: str, job: Dict[str, Any],
                            config: Dict[str, Any], logger=None) -> bool:
    """RV-09: record *what* this artifact is, next to the artifact itself.

    Without it, resume can only check "a file with this name exists and
    decodes" — which is satisfied by a different input, a different config or
    a different job that happens to share the basename.
    """
    payload = {
        "impl": VIDEO_IMPL_VERSION,
        "job_id": job.get("job_id", ""),
        "input_path": job.get("input_path", ""),
        "input_identity": build_input_identity(job.get("input_path", "")),
        "params_identity": build_params_identity(config, job, logger),
        "output_path": str(final_path),
        "output_size": (os.path.getsize(final_path)
                        if os.path.exists(final_path) else None),
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    try:
        tmp = completion_marker_path(final_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True, indent=2)
        os.replace(tmp, completion_marker_path(final_path))
        return True
    except Exception as e:
        if logger:
            logger.warning("COMPLETION_MARKER_FAIL",
                           f"Could not record completion marker: {e}")
        return False


def resume_artifact_matches(final_path: str, job: Dict[str, Any],
                            config: Dict[str, Any],
                            logger=None) -> Tuple[bool, str]:
    """RV-09: decide whether `final_path` really belongs to *this* job."""
    marker = completion_marker_path(final_path)
    if not os.path.exists(marker):
        return False, ("no completion marker; the artifact cannot be bound to "
                       "this job's input/params")
    try:
        with open(marker, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"unreadable completion marker: {e}"
    if str(data.get("impl", "")) != VIDEO_IMPL_VERSION:
        return False, (f"produced by implementation {data.get('impl')!r}, "
                       f"current is {VIDEO_IMPL_VERSION!r}")
    if str(data.get("job_id", "")) != str(job.get("job_id", "")):
        return False, (f"belongs to job {data.get('job_id')!r}, "
                       f"not {job.get('job_id')!r}")
    recorded = data.get("input_identity") or {}
    current = build_input_identity(job.get("input_path", ""))
    # R2-07：身份"拿不到"与"对不上"都要拒绝，不能因为缺摘要就放行。
    status = _input_identity_status(recorded, current)
    if status == "changed":
        return False, "input content changed since the artifact was produced"
    if status == "unproven":
        return False, ("input identity cannot be verified (missing/unreadable "
                       "on one side); refusing to reuse the artifact")
    if str(data.get("params_identity", "")) and \
            data["params_identity"] != build_params_identity(config, job, logger):
        return False, "effective parameters changed since it was produced"
    return True, "identity-match"


def find_final_output(job: Dict[str, Any], config: Dict[str, Any],
                      args) -> List[str]:
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
    explicit = job.get("output_path") or ""
    if explicit and os.path.exists(explicit):
        candidates.append(explicit)
    output_dir = ""
    if isinstance(args, dict):
        output_dir = args.get("output") or ""
    else:
        output_dir = getattr(args, "output", "") or ""
    if not output_dir:
        output_dir = config.get("global", {}).get("output_dir", "./output")
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


def process_job(job: Dict[str, Any], config: Dict[str, Any],
                args, logger) -> Tuple[int, str]:
    """Process a single job through all stages. Returns (exit_code, stderr)."""
    job_id = job["job_id"]
    input_path = job["input_path"]

    logger.info("JOB_START", f"Starting job {job_id}",
                job_id=job_id, input_path=input_path)

    # BUG-042 fix: normalize args to dict early so all downstream code uses dict API
    if not isinstance(args, dict):
        args = vars(args)

    # Build CLI overrides
    cli_overrides = {}
    _output = args.get("output")
    _no_wsl = args.get("no_wsl_redirect", False)
    _keep = args.get("keep_intermediates", False)
    _gpu = args.get("gpu")
    _ai_enabled = args.get("ai_enabled")
    _ai_scale = args.get("ai_upscale_factor")
    _face_restore = args.get("face_restore")
    _ai_model = args.get("ai_model")
    if _output:
        cli_overrides["output_dir"] = _output
    if _no_wsl:
        cli_overrides["no_wsl_redirect"] = True
    if _keep:
        cli_overrides["cleanup_intermediates"] = False
    if _gpu:
        cli_overrides["gpu_backend"] = _gpu
    # ISSUE-V-MAIN-001 / BUG-047: propagate AI CLI overrides into params
    # FIX-P5-1: 必须使用 ai_ 前缀键名，与 merge_params() 中 ai_config 展开逻辑保持一致
    if _ai_enabled is True:
        cli_overrides["ai_enabled"] = True
    if _ai_scale is not None:
        cli_overrides["ai_upscale_scale"] = _ai_scale
    if _face_restore is not None:
        cli_overrides["ai_face_restore"] = _face_restore
    if _ai_model is not None:
        cli_overrides["ai_upscale_model"] = _ai_model

    # Merge params
    params = merge_params(job, config, cli_overrides, logger)
    params["output_path"] = job.get("output_path", "")

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

    log_dir = params.get("log_dir", "./logs")
    job_logger = get_logger(log_dir=log_dir, job_id=job_id)

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
        # BUG-044 fix: resume from last completed stage + 1 when available
        resume_from = job.get("resume_from_stage")
        if resume_from is not None:
            stages = [(n, s) for n, s in stages if n >= resume_from]
            job_logger.info("RESUME_STAGE", f"Resuming job from stage {resume_from}")
            # Clear one-shot resume marker so re-runs don't keep skipping
            job.pop("resume_from_stage", None)
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
    for stage_num, script_name in stages:
        job_logger.info("STAGE_START", f"Stage {stage_num}: {script_name}")
        exit_code, stderr = run_stage(script_name, job, params, args.get("config"))
        if exit_code == 0:
            job_logger.info("STAGE_DONE", f"Stage {stage_num} complete")
            # R2-07 补完：给本 stage 刚产出的中间产物登记身份（源 + 参数 + 实现）。
            # stage7 的最终产物另有 completion marker，不重复登记。
            if stage_num != 7:
                try:
                    _td = params.get("temp_dir", "./temp")
                    for _art in find_stage_artifacts(_td, job_id, input_path,
                                                     stage_num):
                        record_stage_artifact_identity(
                            _art, job, config, stage_num, input_path,
                            job_logger)
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
            job_logger.error("STAGE_FAILED", f"Stage {stage_num} failed: {exit_code}")
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
                            cleanup_old_intermediates(temp_dir, job_id, keep_steps)
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
                        cleanup_old_intermediates(temp_dir, job_id, keep_steps)
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
        if params.get("cleanup_intermediates", True):
            temp_dir = params.get("temp_dir", "./temp")
            keep_steps = params.get("keep_steps", 2)
            # FIX-NEW-1: Removed spurious input_name arg — function signature is
            # cleanup_old_intermediates(temp_dir, job_id, keep_steps, current_step)
            # and matches files by job_id prefix; input_name is not a parameter.
            # FIX-NEW-V-07 / R6-10: 传 current_step，让判定走
            # "step < current_step 且不在保留集" 的显式语义，而不是
            # current_step=None 时"按 step 号降序取前 keep_steps 个"。
            cleanup_old_intermediates(temp_dir, job_id, keep_steps,
                                      current_step=stage_num)

    logger.info("JOB_DONE", f"Job {job_id} complete")
    # RV-09: 成功后写下"这个产物是用哪个输入/哪套参数做出来的"，
    # 否则下一次运行没有任何依据判断能否复用。
    try:
        finals = find_final_output(job, config, args)
        for cand in finals:
            if os.path.exists(cand) and os.path.getsize(cand) > 0:
                write_completion_marker(cand, job, config, job_logger)
                break
    except Exception as e:
        job_logger.warning("COMPLETION_MARKER_SKIP",
                           f"Could not record completion identity: {e}")
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


def record_stage_artifact_identity(artifact: str, job: Dict[str, Any],
                                   config: Dict[str, Any], stage_num: int,
                                   input_path: str, logger=None) -> bool:
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
                                  logger=None) -> Tuple[bool, str]:
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
    if recorded_params and recorded_params != build_params_identity(
            config, job, logger):
        return False, "effective parameters changed since it was produced"
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
                            cand, job, config, logger)
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
                # FIX-P5-2: 删除所有损坏的中间文件，并重新标记为 PENDING
                for stage_num in range(8):
                    for fpath in find_stage_artifacts(temp_dir, job_id,
                                                      input_path, stage_num):
                        try:
                            os.remove(fpath)
                            logger.info("FAILED_CLEANUP", f"Removed corrupt intermediate: {fpath}")
                        except Exception as e:
                            logger.warning("FAILED_CLEANUP_FAIL", f"Could not remove {fpath}: {e}")
                logger.info("FAILED_DOWNGRADE",
                           f"Job {job_id} no valid intermediates, cleaned up and downgraded to PENDING")
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
