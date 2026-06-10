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
import time
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple

# Add shared_modules to path
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from manifest_loader import load_manifest, filter_jobs, save_manifest, update_job_status  # noqa: E402
from checkpoint import cleanup_old_intermediates, estimate_disk_space  # noqa: E402


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
    parser.add_argument("--ai-model", choices=["RealESRGAN_x2plus", "RealESRGAN_x4plus",
                                                "DAT_x2", "DAT_x4"], default=None,
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
    CLI > user_overrides > paths > global > defaults
    Uses deep merge for nested dicts.
    Records PARAM_OVERRIDE events per v7.3 §3.10.12.
    """
    params: Dict[str, Any] = {}
    overrides_log = []

    def _log_override(key: str, old_val: Any, new_val: Any, source: str):
        if old_val != new_val:
            overrides_log.append({"key": key, "old": old_val, "new": new_val, "source": source})

    # Start with global
    base = config.get("global", {})
    params = _deep_merge(params, base)

    # Override with path-specific params
    detected_class = job.get("detected_class", "")
    path_key = detected_class.lower()
    paths = config.get("paths", {})
    if path_key in paths:
        path_params = paths[path_key]
        for k, v in path_params.items():
            if k != "audio_override":
                old = params.get(k)
                params[k] = v
                _log_override(k, old, v, "paths")

    # Override with job recommended_params
    rec_params = job.get("recommended_params", {})
    if rec_params:
        for k, v in rec_params.items():
            old = params.get(k)
            if isinstance(v, dict) and isinstance(params.get(k), dict):
                params[k] = _deep_merge(params.get(k, {}), v)
            else:
                params[k] = v
            _log_override(k, old, v, "recommended_params")

    # Override with user_overrides
    user_ov = job.get("user_overrides", {})
    if user_ov:
        for k, v in user_ov.items():
            old = params.get(k)
            if isinstance(v, dict) and isinstance(params.get(k), dict):
                params[k] = _deep_merge(params.get(k, {}), v)
            else:
                params[k] = v
            _log_override(k, old, v, "user_overrides")

    # Apply audio_override for the detected material type
    # FIX-P5-4: 优先级：CLI > user_overrides > audio_override > global.audio > defaults
    audio_override = config.get("audio_override", {})
    if path_key in audio_override:
        for k, v in audio_override[path_key].items():
            old = params.get(k)
            params[k] = v
            _log_override(k, old, v, "audio_override")

    # Override with CLI
    if cli_overrides:
        for k, v in cli_overrides.items():
            old = params.get(k)
            params[k] = v
            _log_override(k, old, v, "cli")

    # Merge nested config sections (audio, ai) as flat keys for stage script compatibility
    audio_config = config.get("audio", {})
    for k, v in audio_config.items():
        if k not in params:
            params[k] = v
        params[f"audio_{k}"] = v

    ai_config = config.get("ai", {})
    for k, v in ai_config.items():
        if k not in params:
            params[k] = v
        params[f"ai_{k}"] = v

    # grain_factory3 is accessed as nested dict by stage6: params.get("grain_factory3", {})
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
    params["original_width"] = job.get("original_width", 0)
    params["original_height"] = job.get("original_height", 0)
    params["ai_upscale_banned"] = job.get("ai_upscale_banned", False)
    params["job_id"] = job.get("job_id", "")
    params["input_path"] = job.get("input_path", "")
    params["field_segments"] = job.get("field_segments", [])
    params["duration"] = job.get("duration", 0)

    # V-P-006 fix: log PARAM_OVERRIDE events
    if logger and overrides_log:
        for entry in overrides_log:
            logger.info("PARAM_OVERRIDE",
                       f"{entry['key']}: {entry['old']} -> {entry['new']} (source={entry['source']})")

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
    if params.get("no_wsl_redirect", False):
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


def _check_disk_space(job: Dict[str, Any], config: Dict[str, Any], logger) -> bool:
    """Check if sufficient disk space is available before processing."""
    try:
        # Extract parameters for estimation
        detected_class = job.get("detected_class", "")
        width = job.get("original_width", 720)
        height = job.get("original_height", 480)
        duration = job.get("duration", 300)
        fps = 25.0 if detected_class == "HDV_AVCHD" else 29.97
        
        # Estimate using checkpoint function with correct signature
        required = estimate_disk_space(
            duration=duration,
            fps=fps,
            width=width,
            height=height,
        )
        
        temp_dir = config.get("global", {}).get("temp_dir", "./temp")
        stat = shutil.disk_usage(temp_dir)
        available = stat.free
        # Require 2x the estimated space as safety margin
        if available < required * 2 * 1024 * 1024:  # Convert MB to bytes
            logger.error("DISK_SPACE",
                        f"Insufficient disk space: need ~{required}MB, "
                        f"have {available // (1024*1024)}MB")
            return False
        return True
    except Exception as e:
        logger.warning("DISK_SPACE_CHECK", f"Disk space check failed: {e}")
        return True


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

    # V-P-008 fix: WSL2 path redirect for temp_dir and output_dir
    try:
        from checkpoint import wsl_redirect_path
        no_wsl_redirect = params.get("no_wsl_redirect", False)
        platform_val = params.get("platform", "auto")
        temp_dir = params.get("temp_dir", "./temp")
        output_dir = params.get("output_dir", "./output")
        redirected_temp, original_temp, _, _ = wsl_redirect_path(
            temp_dir, no_wsl_redirect=no_wsl_redirect, target_platform=platform_val
        )
        redirected_output, original_output, _, _ = wsl_redirect_path(
            output_dir, no_wsl_redirect=no_wsl_redirect, target_platform=platform_val
        )
        if original_temp:
            logger.info("WSL_REDIRECT", f"temp_dir: {temp_dir} -> {redirected_temp}",
                       original_path=temp_dir, redirected_path=redirected_temp)
            params["temp_dir"] = redirected_temp
            params["original_temp_dir"] = original_temp
        if original_output:
            logger.info("WSL_REDIRECT", f"output_dir: {output_dir} -> {redirected_output}",
                       original_path=output_dir, redirected_path=redirected_output)
            params["output_dir"] = redirected_output
            params["original_output_dir"] = original_output
    except Exception as e:
        logger.warning("WSL_REDIRECT_SKIP", f"WSL redirect check failed: {e}")

    # Disk space check
    if not _check_disk_space(job, config, logger):
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
    if ai_enabled and face_restore:
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

    # Run stages
    traditional_stage_completed = False
    for stage_num, script_name in stages:
        job_logger.info("STAGE_START", f"Stage {stage_num}: {script_name}")
        exit_code, stderr = run_stage(script_name, job, params, args.get("config"))
        if exit_code == 0:
            job_logger.info("STAGE_DONE", f"Stage {stage_num} complete")
            if stage_num == 2:
                # FIX-P5-5: 校验 stage2 输出文件存在、非空、可解码
                temp_dir = params.get("temp_dir", "./temp")
                input_name = Path(input_path).name
                pattern = f"{temp_dir}/{job_id}_{input_name}_step2*"
                import glob
                candidates = glob.glob(pattern)
                if not candidates:
                    job_logger.error("TRADITIONAL_GATE_BLOCK", "Stage 2 output file missing")
                    return 86, "Traditional restoration output missing"
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                step2_output = candidates[0]
                if os.path.getsize(step2_output) == 0:
                    job_logger.error("TRADITIONAL_GATE_BLOCK", "Stage 2 output file is empty")
                    return 86, "Traditional restoration output empty"
                try:
                    from checkpoint import verify_ffprobe_decodeable
                    if not verify_ffprobe_decodeable(step2_output):
                        job_logger.error("TRADITIONAL_GATE_BLOCK", "Stage 2 output not decodeable by ffprobe")
                        return 86, "Traditional restoration output not decodeable"
                except Exception as e:
                    job_logger.error("TRADITIONAL_GATE_BLOCK", f"Stage 2 output verification failed: {e}")
                    return 86, f"Traditional restoration verification failed: {e}"
                traditional_stage_completed = True
                job_logger.info("TRADITIONAL_GATE",
                               "Traditional restoration completed, proceeding to AI stages")
        elif exit_code == 3:
            job_logger.info("STAGE_SKIP", f"Stage {stage_num} skipped")
            if stage_num == 2:
                traditional_stage_completed = True
                job_logger.info("TRADITIONAL_GATE",
                               "Traditional restoration skipped (already done), proceeding to AI stages")
        else:
            job_logger.error("STAGE_FAILED", f"Stage {stage_num} failed: {exit_code}")
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
            input_name = Path(input_path).name
            keep_steps = params.get("keep_steps", 2)
            cleanup_old_intermediates(temp_dir, job_id, input_name, keep_steps)

    logger.info("JOB_DONE", f"Job {job_id} complete")
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
    jlogger = get_logger(log_dir=log_dir, level=jcfg.get("global", {}).get("log_level", "INFO"))
    jmanifest = load_manifest(manifest_path)
    _run_single_job(job, jcfg, args, jlogger, jmanifest, manifest_path)
    return job["job_id"]


def _run_single_job(job: Dict[str, Any], config: Dict[str, Any],
                    args, logger, manifest: Dict[str, Any], manifest_path: str):
    """Run a single job with retry logic and manifest update."""
    job_id = job["job_id"]
    retry_count = config.get("global", {}).get("retry_count", 2)
    success = False
    for attempt in range(retry_count + 1):
        exit_code, stderr = process_job(job, config, args, logger)
        if exit_code in (0, 3):
            update_job_status(manifest, job_id, "DONE")
            success = True
            break
        else:
            from checkpoint import is_retryable_error
            if is_retryable_error(exit_code, stderr):
                wait = 5 if attempt == 0 else 15
                logger.warning("JOB_RETRY", f"Retrying job {job_id} in {wait}s")
                time.sleep(wait)
            else:
                break
    if not success:
        update_job_status(manifest, job_id, "FAILED")
    save_manifest(manifest, manifest_path)


def main():
    args = parse_args()
    config = load_config(args.config)
    gcfg = config.get("global", {})
    log_dir = gcfg.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, level=gcfg.get("log_level", "INFO"))
    logger.info("PIPELINE_START", "Video restoration pipeline starting")

    # Dependency check
    try:
        from dependency_check import run_all_checks
        ok, issues = run_all_checks(config)
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
        manifest_path = args.manifest or str(SCRIPT_DIR / "batch_manifest.yaml")
        from manifest_generate import generate_manifest
        generate_manifest(args.input, manifest_path, config, "video")
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
        generate_manifest(args.input, manifest_path, config, "video")
        manifest = load_manifest(manifest_path)
        jobs = filter_jobs(manifest, ["PENDING"])
        logger.info("QUICK_START", f"Quick mode: {len(jobs)} jobs")
        print(f"Quick mode: scanning done, executing {len(jobs)} jobs...")
        for job in jobs:
            _run_single_job(job, config, args, logger, manifest, manifest_path)
        # Clean up temporary manifest
        try:
            Path(manifest_path).unlink(missing_ok=True)
        except Exception:
            pass
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
        # Run single job
        _run_single_job(job, config, args, logger, manifest, manifest_path)
        try:
            Path(manifest_path).unlink(missing_ok=True)
        except Exception:
            pass
        logger.info("PIPELINE_DONE", "Video pipeline complete (interactive mode)")
        print("Done.")
        return

    # Generate or load manifest
    if args.manifest:
        manifest_path = args.manifest
    elif args.input:
        manifest_path = str(SCRIPT_DIR / "batch_manifest.yaml")
        from manifest_generate import generate_manifest
        generate_manifest(args.input, manifest_path, config, "video")
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
            input_name = Path(job.get("input_path", "")).name
            last_completed = -1
            # FIX-P5-6: Scan step0..step7 to find highest valid intermediate, check all candidates
            import glob
            for stage_num in range(7, -1, -1):
                pattern = f"{temp_dir}/{job_id}_{input_name}_step{stage_num}*"
                candidates = glob.glob(pattern)
                if candidates:
                    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    for cand in candidates:
                        if verify_ffprobe_decodeable(cand):
                            last_completed = stage_num
                            break
                    if last_completed >= 0:
                        break
            job["last_completed_stage"] = last_completed
            if last_completed >= 0:
                logger.info("FAILED_DOWNGRADE",
                           f"Job {job_id} has valid step{last_completed}, downgrading to PENDING "
                           f"(will resume from stage {last_completed + 1})")
                job["status"] = "PENDING"
                job["resume_from_stage"] = last_completed + 1
            else:
                # FIX-P5-2: 删除所有损坏的中间文件，并重新标记为 PENDING
                for stage_num in range(8):
                    pattern = f"{temp_dir}/{job_id}_{input_name}_step{stage_num}*"
                    for fpath in glob.glob(pattern):
                        try:
                            os.remove(fpath)
                            logger.info("FAILED_CLEANUP", f"Removed corrupt intermediate: {fpath}")
                        except Exception as e:
                            logger.warning("FAILED_CLEANUP_FAIL", f"Could not remove {fpath}: {e}")
                job["status"] = "PENDING"
                job.pop("resume_from_stage", None)
                logger.info("FAILED_DOWNGRADE",
                           f"Job {job_id} no valid intermediates, cleaned up and downgraded to PENDING")
        save_manifest(manifest, manifest_path)
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
                input_name = Path(job.get("input_path", "")).name
                # Only mark DONE if step7 output exists and is valid
                step7_done = False
                import glob
                pattern = f"{temp_dir}/{job_id}_{input_name}_step7*"
                candidates = glob.glob(pattern)
                if candidates:
                    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    if verify_ffprobe_decodeable(candidates[0]):
                        step7_done = True
                        logger.info("RESUME_DONE",
                                   f"Job {job_id} has valid step7 output, marking DONE")
                        job["status"] = "DONE"
                if not step7_done:
                    # Determine highest valid completed stage for partial resume
                    last_completed = -1
                    for stage_num in range(6, -1, -1):
                        pattern = f"{temp_dir}/{job_id}_{input_name}_step{stage_num}*"
                        candidates = glob.glob(pattern)
                        if candidates:
                            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                            if verify_ffprobe_decodeable(candidates[0]):
                                last_completed = stage_num
                                break
                    job["last_completed_stage"] = last_completed
                    if last_completed >= 0:
                        job["resume_from_stage"] = last_completed + 1
                        logger.info("RESUME_PARTIAL",
                                   f"Job {job_id} resuming from stage {last_completed + 1}")
            save_manifest(manifest, manifest_path)
        except Exception as e:
            logger.warning("RESUME_CHECK", f"Resume check skipped: {e}")

    jobs = filter_jobs(manifest, target_statuses)
    if args.job_id:
        jobs = [j for j in jobs if j.get("job_id") == args.job_id]

    logger.info("BATCH_START", f"Processing {len(jobs)} video jobs")
    print(f"Processing {len(jobs)} video jobs...")

    # V-P-009 fix: enhanced preview mode with motion segments
    # Also handle mode == "preview"
    if args.preview_only or (mode == "preview"):
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

    # Parallel worker pool
    workers = max(1, args.workers)
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)

        # Build task contexts for picklable worker submission
        # BUG-040 fix: convert argparse.Namespace to plain dict for spawn safety
        task_contexts = []
        for job in jobs:
            task_contexts.append({
                "job": job,
                "config_path": getattr(args, "config"),
                "log_dir": log_dir,
                "manifest_path": manifest_path,
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

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker_wrapper, ctx): ctx for ctx in task_contexts}
            for future in as_completed(futures):
                ctx = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error("WORKER_ERROR", f"Job {ctx['job']['job_id']} failed in worker: {e}")
                    update_job_status(manifest, ctx["job"]["job_id"], "FAILED")
                    save_manifest(manifest, manifest_path)
    else:
        for job in jobs:
            _run_single_job(job, config, args, logger, manifest, manifest_path)

    logger.info("PIPELINE_DONE", "Video pipeline complete")
    print("Done.")


if __name__ == "__main__":
    main()
