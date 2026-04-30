#!/usr/bin/env python3
"""
Video Restoration Pipeline - Main Controller
Orchestrates video restoration stages with manifest-driven batch processing.

Usage:
    python main.py --input /mnt/videos --config config.yaml
    python main.py --manifest batch_manifest.yaml
    python main.py --input /mnt/videos --preview
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple

# Add shared_utils to path
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_utils"))

from logger import get_logger  # noqa: E402
from manifest_loader import load_manifest, filter_jobs, save_manifest, update_job_status  # noqa: E402
from checkpoint import wsl_redirect_path, cleanup_old_intermediates  # noqa: E402


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
    parser.add_argument("--gpu", default="auto", help="GPU backend (auto/cuda/rocm/mps/cpu)")
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

    return params


def run_stage(script_name: str, job: Dict[str, Any], params: Dict[str, Any],
              config: Dict[str, Any]) -> Tuple[int, str]:
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

    # Add config path
    cmd.extend(["--config", str(SCRIPT_DIR / "config.yaml")])

    # Add flags
    if params.get("no_wsl_redirect", False):
        cmd.append("--no-wsl-redirect")
    if params.get("cleanup_intermediates", True):
        cmd.append("--cleanup-intermediates")

    # Run with timeout
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SCRIPT_DIR.parent)

        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=3600 * 4, env=env, shell=False)  # 4h timeout per stage
        stderr = r.stderr if r.stderr else ""
        return r.returncode, stderr
    except subprocess.TimeoutExpired:
        return 87, "Stage timed out after 4 hours"  # Resource不足 (timeout)
    except Exception as e:
        return 82, str(e)  # IO error


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
    original_output = job.get("output_path", "")
    params["output_path"] = original_output
    params["original_output_path"] = original_output

    # WSL2 path redirect for temp/output dirs
    temp_dir = params.get("temp_dir", "./temp")
    output_dir = params.get("output_dir", "./output")
    no_redirect = params.get("no_wsl_redirect", False)
    platform_val = params.get("platform", "auto")
    params["temp_dir"] = wsl_redirect_path(temp_dir, temp_dir, no_redirect, platform_val)
    params["output_dir"] = wsl_redirect_path(output_dir, temp_dir, no_redirect, platform_val)

    # Set log dir for this job
    log_dir = params.get("log_dir", "./logs")
    job_logger = get_logger(log_dir=log_dir, job_id=job_id)

    # Define stage execution order
    stages = [
        (0, "stage0_audio_extract.py"),
        (1, "stage1_preprocess.py"),
        (2, "stage2_traditional_check.py"),
    ]

    # Conditionally add AI stages
    ai_enabled = config.get("ai", {}).get("enabled", False)
    ai_banned = job.get("ai_upscale_banned", False)
    force_allow = config.get("ai", {}).get("force_allow_banned_upscale", False)

    if ai_enabled and not (ai_banned and not force_allow):
        stages.append((3, "stage3_ai_upscale.py"))
        stages.append((4, "stage4_postprocess.py"))

    # Face restore stage
    face_restore = config.get("ai", {}).get("face_restore", "")
    if ai_enabled and face_restore:
        stages.append((5, "stage5_ai_face.py"))

    # Grain synthesis
    grain_enabled = config.get("grain_factory3", {}).get("enabled", True)
    if grain_enabled:
        stages.append((6, "stage6_grain_synthesis.py"))

    # Final encode
    stages.append((7, "stage7_encode.py"))

    # Run single stage if requested
    if args.stage is not None:
        stage_map = {s[0]: s[1] for s in stages}
        if args.stage in stage_map:
            stages = [(args.stage, stage_map[args.stage])]
        else:
            logger.error("JOB_ERROR", f"Invalid stage {args.stage}")
            return 81, ""

    # Execute stages
    for stage_num, script_name in stages:
        job_logger.info("STAGE_START", f"Stage {stage_num}: {script_name}",
                       stage=stage_num, script=script_name)

        exit_code, stderr = run_stage(script_name, job, params, config)

        if exit_code == 0:
            job_logger.info("STAGE_DONE", f"Stage {stage_num} completed")
        elif exit_code == 3:
            job_logger.info("STAGE_SKIP", f"Stage {stage_num} skipped (already exists)")
        else:
            job_logger.error("STAGE_FAILED", f"Stage {stage_num} failed with code {exit_code}")
            return exit_code, stderr

        # Cleanup old intermediates after successful stage
        if params.get("cleanup_intermediates", True):
            temp_dir = params.get("temp_dir", "./temp")
            input_name = Path(input_path).name
            cleanup_old_intermediates(temp_dir, job_id, input_name, keep_steps=2)

    logger.info("JOB_DONE", f"Job {job_id} completed")
    return 0, ""


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    global_config = config.get("global", {})

    # Setup master logger
    log_dir = global_config.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, level=global_config.get("log_level", "INFO"))

    logger.info("PIPELINE_START", "Video restoration pipeline starting")

    # Dependency check
    try:
        from dependency_check import run_all_checks
        ok, issues = run_all_checks(config)
        if not ok:
            for issue in issues:
                logger.error("DEP_CHECK", issue)
            print("Dependency check failed. See logs for details.")
            sys.exit(80)
    except Exception as e:
        logger.warning("DEP_CHECK", f"Could not run dependency check: {e}")

    # Generate or load manifest
    if args.manifest:
        manifest_path = args.manifest
        logger.info("MANIFEST_LOAD", f"Loading manifest: {manifest_path}")
    elif args.input:
        manifest_path = str(SCRIPT_DIR / "batch_manifest.yaml")
        from manifest_generate import generate_manifest
        logger.info("MANIFEST_GENERATE", "Generating manifest from input dirs")
        generate_manifest(args.input, manifest_path, config, "video")
    else:
        print("Error: Provide --input or --manifest")
        sys.exit(81)

    manifest = load_manifest(manifest_path)

    # Filter jobs to process
    jobs = filter_jobs(manifest, ["PENDING", "FAILED"])

    if args.job_id:
        jobs = [j for j in jobs if j.get("job_id") == args.job_id]

    logger.info("BATCH_START", f"Processing {len(jobs)} jobs")
    print(f"Processing {len(jobs)} jobs...")

    # Process each job with retry
    retry_count = global_config.get("retry_count", 2)

    for job in jobs:
        job_id = job["job_id"]
        success = False

        for attempt in range(retry_count + 1):
            exit_code, stderr = process_job(job, config, args, logger)

            if exit_code == 0:
                update_job_status(manifest, job_id, "DONE")
                success = True
                break
            elif exit_code == 3:
                update_job_status(manifest, job_id, "DONE")
                success = True
                break
            else:
                # Check if retryable using exit code AND stderr content
                from checkpoint import is_retryable_error
                if is_retryable_error(exit_code, stderr):
                    wait_time = 5 if attempt == 0 else 15
                    logger.warning("JOB_RETRY",
                                   f"Job {job_id} attempt {attempt+1} failed (code={exit_code}), retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error("JOB_FAIL", f"Job {job_id} failed with code {exit_code}, not retryable",
                                stderr=stderr[:500])
                    break

        if not success:
            update_job_status(manifest, job_id, "FAILED",
                              f"Failed after {retry_count + 1} attempts")

        # Save manifest after each job
        save_manifest(manifest, manifest_path)

    logger.info("PIPELINE_DONE", "Video restoration pipeline completed")
    print("Done. Check logs for details.")


if __name__ == "__main__":
    main()
