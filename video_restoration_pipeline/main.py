#!/usr/bin/env python3
"""
Video Restoration Pipeline - Main Controller
Orchestrates video restoration stages with manifest-driven batch processing.

v7.3 Compliance:
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
    parser.add_argument("--preview", action="store_true", help="Preview mode")
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
                 cli_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge parameters with priority:
    CLI > user_overrides > paths > global > defaults
    Uses deep merge for nested dicts.
    """
    params: Dict[str, Any] = {}

    # Start with global
    params = _deep_merge(params, config.get("global", {}))

    # Override with path-specific params
    detected_class = job.get("detected_class", "")
    path_key = detected_class.lower()
    paths = config.get("paths", {})
    if path_key in paths:
        params = _deep_merge(params, paths[path_key])

    # Apply audio_override for the detected material type (v7.3 §3.1)
    # Priority: CLI > user_overrides > recommended_params > paths.audio_override > paths > global > defaults
    audio_override = paths.get("audio_override", {})
    if path_key in audio_override:
        # Flat-inject audio override params so stage scripts can read them directly
        for k, v in audio_override[path_key].items():
            params[k] = v

    # Override with job recommended_params
    rec_params = job.get("recommended_params", {})
    if rec_params:
        params = _deep_merge(params, rec_params)

    # Override with user_overrides
    user_ov = job.get("user_overrides", {})
    if user_ov:
        params = _deep_merge(params, user_ov)

    # Override with CLI
    if cli_overrides:
        params = _deep_merge(params, cli_overrides)

    # Merge nested config sections (audio, ai) as flat keys for stage script compatibility
    # Stage scripts read audio params as flat keys: params.get("hum_freq", 50)
    audio_config = config.get("audio", {})
    for k, v in audio_config.items():
        # Only inject if not already set by higher priority sources
        if k not in params:
            params[k] = v
        # Also inject with "audio_" prefix for backward compatibility in stage7
        params[f"audio_{k}"] = v

    ai_config = config.get("ai", {})
    for k, v in ai_config.items():
        if k not in params:
            params[k] = v

    # grain_factory3 is accessed as nested dict by stage6: params.get("grain_factory3", {})
    if "grain_factory3" in config:
        if "grain_factory3" not in params:
            params["grain_factory3"] = config["grain_factory3"]
        else:
            # Deep merge grain_factory3 to preserve per-class sub-keys
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

    # Add params as JSON
    import json
    params_json = json.dumps(params, default=str)
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

    # Build CLI overrides
    cli_overrides = {}
    if args.output:
        cli_overrides["output_dir"] = args.output
    if args.no_wsl_redirect:
        cli_overrides["no_wsl_redirect"] = True
    if args.keep_intermediates:
        cli_overrides["cleanup_intermediates"] = False
    if args.gpu:
        cli_overrides["gpu_backend"] = args.gpu

    # Merge params
    params = merge_params(job, config, cli_overrides)
    params["output_path"] = job.get("output_path", "")

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

    if ai_enabled and not (ai_banned and not force_allow):
        stages.append((3, "stage3_ai_upscale.py"))
        stages.append((4, "stage4_postprocess.py"))

    face_restore = params.get("face_restore", "")
    if ai_enabled and face_restore:
        stages.append((5, "stage5_ai_face.py"))

    stages.append((6, "stage6_grain_synthesis.py"))
    stages.append((7, "stage7_encode.py"))

    if args.stage is not None:
        stage_map = {s[0]: s[1] for s in stages}
        if args.stage in stage_map:
            stages = [(args.stage, stage_map[args.stage])]
        else:
            return 81, "Invalid stage number"

    # Run stages
    for stage_num, script_name in stages:
        job_logger.info("STAGE_START", f"Stage {stage_num}: {script_name}")
        exit_code, stderr = run_stage(script_name, job, params, args.config)
        if exit_code == 0:
            job_logger.info("STAGE_DONE", f"Stage {stage_num} complete")
        elif exit_code == 3:
            job_logger.info("STAGE_SKIP", f"Stage {stage_num} skipped")
        else:
            job_logger.error("STAGE_FAILED", f"Stage {stage_num} failed: {exit_code}")
            return exit_code, stderr

        # Cleanup old intermediates after successful stage
        if params.get("cleanup_intermediates", True):
            temp_dir = params.get("temp_dir", "./temp")
            keep_steps = params.get("keep_steps", 2)
            cleanup_old_intermediates(temp_dir, job_id, keep_steps)

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
    jobs = filter_jobs(manifest, ["PENDING", "FAILED"])
    if args.job_id:
        jobs = [j for j in jobs if j.get("job_id") == args.job_id]

    logger.info("BATCH_START", f"Processing {len(jobs)} video jobs")
    print(f"Processing {len(jobs)} video jobs...")

    # Preview mode
    if args.preview:
        from preview import extract_preview_frames
        for job in jobs[:3]:
            try:
                preview_dir = Path(gcfg.get("temp_dir", "./temp")) / "preview"
                frames = extract_preview_frames(job["input_path"], str(preview_dir), num_frames=3)
                logger.info("PREVIEW", f"Preview frames for {job['job_id']}: {frames}")
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
        task_contexts = []
        for job in jobs:
            task_contexts.append({
                "job": job,
                "config_path": args.config,
                "log_dir": log_dir,
                "manifest_path": manifest_path,
                "args": args,
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
