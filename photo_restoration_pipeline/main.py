#!/usr/bin/env python3
"""
Photo Restoration Pipeline - Main Controller
Orchestrates photo restoration stages with manifest-driven batch processing.

Usage:
    python main.py --input /mnt/photos --config config.yaml
    python main.py --manifest batch_manifest.yaml
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_utils"))

from logger import get_logger  # noqa: E402
from manifest_loader import load_manifest, filter_jobs, save_manifest, update_job_status  # noqa: E402
from checkpoint import cleanup_old_intermediates  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Photo Restoration Pipeline")
    parser.add_argument("--input", "-i", nargs="+", help="Input directories")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--config", "-c", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--manifest", "-m", help="Use existing manifest")
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--no-wsl-redirect", action="store_true")
    parser.add_argument("--stage", type=int)
    parser.add_argument("--job-id")
    parser.add_argument("--gpu", default="auto")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    import yaml  # type: ignore[import-untyped]
    with open(path, "r", encoding="utf-8") as f:
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
                 cli: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    params = _deep_merge(params, config.get("global", {}))
    params = _deep_merge(params, config.get("photo", {}))
    rec = job.get("recommended_params", {})
    if rec:
        params = _deep_merge(params, rec)
    user_ov = job.get("user_overrides", {})
    if user_ov:
        params = _deep_merge(params, user_ov)
    if cli:
        params = _deep_merge(params, cli)
    return params


def run_stage(script_name: str, job: Dict[str, Any], params: Dict[str, Any]) -> tuple:
    script_path = SCRIPT_DIR / "stages" / script_name
    if not script_path.exists():
        return 80, "Script not found"
    cmd = [sys.executable, str(script_path)]
    cmd.extend(["--input", job["input_path"]])
    cmd.extend(["--job-id", job["job_id"]])
    import json
    cmd.extend(["--params", json.dumps(params, default=str)])
    cmd.extend(["--config", str(SCRIPT_DIR / "config.yaml")])
    if params.get("no_wsl_redirect"):
        cmd.append("--no-wsl-redirect")
    if params.get("cleanup_intermediates", True):
        cmd.append("--cleanup-intermediates")
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SCRIPT_DIR.parent)
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=3600 * 2, env=env, shell=False)
        stderr = r.stderr if r.stderr else ""
        return r.returncode, stderr
    except subprocess.TimeoutExpired:
        return 87, "Stage timed out after 2 hours"
    except Exception as e:
        return 82, str(e)


def process_job(job: Dict[str, Any], config: Dict[str, Any],
                args, logger) -> Tuple[int, str]:
    job_id = job["job_id"]
    logger.info("JOB_START", f"Photo job {job_id}", job_id=job_id)

    cli_ov = {}
    if args.output:
        cli_ov["output_dir"] = args.output
    if args.no_wsl_redirect:
        cli_ov["no_wsl_redirect"] = True
    if args.keep_intermediates:
        cli_ov["cleanup_intermediates"] = False
    if args.gpu:
        cli_ov["gpu_backend"] = args.gpu

    params = merge_params(job, config, cli_ov)
    params["output_path"] = job.get("output_path", "")

    # Merge AI config into params for stage script compatibility
    ai_config = config.get("ai", {})
    for k, v in ai_config.items():
        if k not in params:
            params[k] = v

    # Ensure ai_enabled / face_restore are available
    params["ai_enabled"] = ai_config.get("enabled", False)
    params["ai_upscale_banned"] = job.get("ai_upscale_banned", False)
    params["face_restore"] = ai_config.get("face_restore", "")

    # Merge grain_factory3 config
    if "grain_factory3" in config:
        if "grain_factory3" not in params:
            params["grain_factory3"] = config["grain_factory3"]
        else:
            params["grain_factory3"] = _deep_merge(
                config["grain_factory3"], params["grain_factory3"]
            )

    # Inject job core metadata fields (mirrors video main.py behavior)
    params["detected_class"] = job.get("detected_class", "SCAN_PRINT")
    params["original_width"] = job.get("original_width", 0)
    params["original_height"] = job.get("original_height", 0)
    params["native_width"] = job.get("native_width", 0)
    params["native_height"] = job.get("native_height", 0)
    params["native_dar"] = job.get("native_dar", "")
    params["file_size"] = job.get("file_size", 0)
    params["file_mtime"] = job.get("file_mtime", 0)
    params["file_hash_prefix"] = job.get("file_hash_prefix", "")
    params["dpi"] = job.get("dpi", 0)
    params["job_id"] = job.get("job_id", "")
    params["input_path"] = job.get("input_path", "")

    log_dir = params.get("log_dir", "./logs")
    job_logger = get_logger(log_dir=log_dir, job_id=job_id)

    stages = [
        (1, "stage1_preprocess.py"),
        (2, "stage2_traditional_check.py"),
    ]

    ai_enabled = config.get("ai", {}).get("enabled", False)
    ai_banned = job.get("ai_upscale_banned", False)
    force_allow = config.get("ai", {}).get("force_allow_banned_upscale", False)

    if ai_enabled and not (ai_banned and not force_allow):
        stages.append((3, "stage3_ai_upscale.py"))
        stages.append((4, "stage4_postprocess.py"))

    face_restore = config.get("ai", {}).get("face_restore", "")
    if ai_enabled and face_restore:
        stages.append((5, "stage5_ai_face.py"))

    stages.append((6, "stage6_encode.py"))

    if args.stage is not None:
        stage_map = {s[0]: s[1] for s in stages}
        if args.stage in stage_map:
            stages = [(args.stage, stage_map[args.stage])]
        else:
            return 81, "Invalid stage number"

    for stage_num, script_name in stages:
        job_logger.info("STAGE_START", f"Stage {stage_num}: {script_name}")
        exit_code, stderr = run_stage(script_name, job, params)
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
            input_name = Path(job["input_path"]).name
            cleanup_old_intermediates(temp_dir, job_id, input_name, keep_steps=2)

    logger.info("JOB_DONE", f"Job {job_id} complete")
    return 0, ""


def main():
    args = parse_args()
    config = load_config(args.config)
    gcfg = config.get("global", {})
    log_dir = gcfg.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, level=gcfg.get("log_level", "INFO"))
    logger.info("PIPELINE_START", "Photo restoration pipeline starting")

    try:
        from dependency_check import run_all_checks
        ok, issues = run_all_checks(config)
        if not ok:
            for issue in issues:
                logger.error("DEP_CHECK", issue)
            sys.exit(80)
    except Exception as e:
        logger.warning("DEP_CHECK", f"Check skipped: {e}")

    if args.manifest:
        manifest_path = args.manifest
    elif args.input:
        manifest_path = str(SCRIPT_DIR / "batch_manifest.yaml")
        from manifest_generate import generate_manifest
        generate_manifest(args.input, manifest_path, config, "photo")
    else:
        print("Error: Provide --input or --manifest")
        sys.exit(81)

    manifest = load_manifest(manifest_path)
    jobs = filter_jobs(manifest, ["PENDING", "FAILED"])
    if args.job_id:
        jobs = [j for j in jobs if j.get("job_id") == args.job_id]

    logger.info("BATCH_START", f"Processing {len(jobs)} photo jobs")
    print(f"Processing {len(jobs)} photo jobs...")

    retry_count = gcfg.get("retry_count", 2)
    for job in jobs:
        job_id = job["job_id"]
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

    logger.info("PIPELINE_DONE", "Photo pipeline complete")
    print("Done.")


if __name__ == "__main__":
    main()
