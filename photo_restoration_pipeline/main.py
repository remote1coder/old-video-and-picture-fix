#!/usr/bin/env python3
"""
Photo Restoration Pipeline - Main Controller
Single-file sequential execution mode.

v7.3 §1.1-1.6 Compliance:
- No multi-file parallel execution
- Stages execute sequentially per file
- Dynamic timeout based on image dimensions
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


SCRIPT_DIR = Path(__file__).parent.resolve()


def _load_config(config_path: str) -> Dict[str, Any]:
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Config load error: {e}", file=sys.stderr)
        sys.exit(81)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def merge_params(config: Dict[str, Any],
                 detected_class: str,
                 cli_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """6-layer parameter cascading per v7.3 §3.1.
    Priority: CLI > user_overrides > paths > global > defaults
    Uses deep merge for nested dicts.
    """
    # Layer 1: global
    params: Dict[str, Any] = {}
    params = _deep_merge(params, config.get("global", {}))

    # Layer 2: paths (type-dependent)
    paths_cfg = config.get("paths", {})
    if isinstance(paths_cfg, dict):
        type_key = detected_class.lower() if detected_class else "digital_photo"
        if type_key in paths_cfg:
            type_paths = paths_cfg[type_key]
            if isinstance(type_paths, dict):
                params = _deep_merge(params, type_paths)

    # Layer 3: photo-specific
    params = _deep_merge(params, config.get("photo", {}))

    # Layer 4: recommended params from classifier
    if "recommended_params" in cli_overrides:
        params = _deep_merge(params, cli_overrides.pop("recommended_params"))

    # Layer 5: user overrides from manifest
    if "user_overrides" in cli_overrides:
        params = _deep_merge(params, cli_overrides.pop("user_overrides"))

    # Layer 6: CLI
    params = _deep_merge(params, cli_overrides)

    return params


def _get_image_dimensions(input_path: str) -> tuple:
    """Get image width and height using PIL or ffprobe."""
    try:
        from PIL import Image
        with Image.open(input_path) as img:
            return img.size
    except Exception:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=p=0", input_path],
                capture_output=True, text=True, timeout=10, shell=False
            )
            if r.returncode == 0:
                parts = r.stdout.strip().split(",")
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
    return 1920, 1080  # Default fallback


def _calculate_timeout(params: Dict[str, Any],
                       input_path: str,
                       is_ai_stage: bool = False) -> int:
    """Calculate dynamic timeout based on image dimensions.
    
    Base timeout: 10 minutes
    Additional: proportional to image pixel count
    Max: 2 hours
    """
    w, h = _get_image_dimensions(input_path)
    pixel_count = w * h
    
    # Base: 10 minutes
    base = 600
    
    # Additional time based on pixel count
    # 1080p (~2M pixels) = +5 minutes for non-AI, +20 minutes for AI
    # 4K (~8M pixels) = +20 minutes for non-AI, +80 minutes for AI
    pixel_multiplier = 2.0 if is_ai_stage else 0.5
    additional = (pixel_count / (1920 * 1080)) * 300 * pixel_multiplier
    
    # Cap at 2 hours
    timeout = min(base + additional, 7200)
    return int(timeout)


def _run_stage(script_name: str,
               job_id: str,
               input_path: str,
               config_path: str,
               params: Dict[str, Any],
               stage_idx: int,
               is_retry: bool = False,
               is_ai_stage: bool = False,
               script_dir: Optional[str] = None) -> int:
    """Run a single stage with dynamic timeout."""
    if script_dir is None:
        script_dir = SCRIPT_DIR / "stages"
    script_path = script_dir / script_name
    if not script_path.exists():
        print(f"Stage script not found: {script_path}", file=sys.stderr)
        return 81

    params_json = json.dumps(params)
    cmd = [
        sys.executable, str(script_path),
        "--job-id", job_id,
        "--input", input_path,
        "--config", config_path,
        "--params", params_json,
    ]
    if is_retry:
        cmd.append("--is-retry")

    # Dynamic timeout calculation (Fix: replaces hardcoded 1200)
    timeout = _calculate_timeout(params, input_path, is_ai_stage=is_ai_stage)

    # Process isolation per v7.3 §4.2
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPT_DIR)

    print(f"[Stage {stage_idx}] Running: {script_name} (timeout={timeout}s)")

    try:
        r = subprocess.run(cmd, timeout=timeout, env=env, shell=False)
        return r.returncode
    except subprocess.TimeoutExpired:
        print(f"[Stage {stage_idx}] Timeout after {timeout}s", file=sys.stderr)
        return 87
    except Exception as e:
        print(f"[Stage {stage_idx}] Error: {e}", file=sys.stderr)
        return 84


def _check_disk_space(input_path: str, config: Dict[str, Any]) -> bool:
    """Check sufficient disk space with 2x safety margin."""
    try:
        stat = os.statvfs(os.path.dirname(input_path) or ".")
        available_mb = stat.f_bavail * stat.f_frsize / (1024 * 1024)

        # Estimate using checkpoint module
        sys.path.insert(0, str(SCRIPT_DIR))
        from checkpoint import estimate_disk_space

        required = estimate_disk_space(
            duration=0, fps=1,
            width=0, height=0,
            file_path=input_path,
            is_photo=True
        )

        if available_mb < required:
            print(f"Disk space insufficient: {available_mb:.0f}MB < {required}MB needed", file=sys.stderr)
            return False
        return True
    except Exception:
        # If check fails, allow to proceed
        return True


def cleanup_old_intermediates(temp_dir: str, job_id: str) -> None:
    """Clean old intermediate files."""
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from checkpoint import cleanup_old_intermediates as coi
        coi(temp_dir, job_id)
    except Exception:
        pass


def _is_retryable_error(returncode: int) -> bool:
    """Check if error is retryable (transient).

    Per v6.3 §3.7:
    - 82 (IO), 83 (GPU/OOM), 87 (resource/timeout) are retryable
    - 84 (encoding) only retryable if OOM/no space indicated in stderr
    - 80 (deps), 81 (params), 85 (perms), 86 (verify) are NOT retryable
    """
    retryable_codes = {82, 83, 87}
    return returncode in retryable_codes


def _run_single_job(job: Dict[str, Any],
                    config_path: str,
                    config: Dict[str, Any],
                    cli_params: Dict[str, Any]) -> int:
    """Run a single job with retry per v7.3 §4.5."""
    job_id = job.get("job_id", "unknown")
    input_path = job.get("input_path", "")
    detected_class = job.get("detected_class", "")
    confidence = job.get("confidence", 0)
    ai_upscale_banned = job.get("ai_upscale_banned", False)

    if not os.path.exists(input_path):
        print(f"[{job_id}] Input not found: {input_path}", file=sys.stderr)
        return 82

    # Parameter merge
    params = merge_params(config, detected_class, cli_params)
    params["detected_class"] = detected_class
    params["confidence"] = confidence
    params["ai_upscale_banned"] = ai_upscale_banned
    params["job_id"] = job_id
    params["input_path"] = input_path

    # Per-job log
    log_dir = params.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    # Add log config to params for stages
    params["log_dir"] = log_dir

    # Clean intermediates
    temp_dir = params.get("temp_dir", "./temp")
    cleanup_old_intermediates(temp_dir, job_id)

    # Disk space check
    if not _check_disk_space(input_path, config):
        return 87

    # Stage execution
    max_retries = params.get("retry_count", config.get("global", {}).get("retry_count", 2))
    retry_count = 0

    stages = [
        ("stage1_preprocess.py", False),
        ("stage2_traditional_check.py", False),
        ("stage3_ai_upscale.py", True),  # AI stage
        ("stage4_postprocess.py", False),  # Post-processing (not AI)
        ("stage5_ai_face.py", True),  # AI stage
        ("stage6_encode.py", False),
    ]

    for stage_script, is_ai_stage in stages:
        stage_name = stage_script.replace(".py", "")

        # Conditional stages
        if stage_name in ("stage3_ai_upscale", "stage4_postprocess"):
            if not params.get("ai_enabled", False) or ai_upscale_banned:
                continue
        elif stage_name == "stage5_ai_face":
            if not params.get("face_restore"):
                continue

        while True:
            is_retry = retry_count > 0
            rc = _run_stage(
                stage_script, job_id, input_path,
                config_path, params, len([s for s, _ in stages]),
                is_retry=is_retry, is_ai_stage=is_ai_stage
            )

            if rc == 0 or rc == 3:
                break  # Success or skip
            elif rc == 1:
                print(f"[{job_id}] {stage_name} non-retryable error", file=sys.stderr)
                return rc
            elif _is_retryable_error(rc):
                retry_count += 1
                if retry_count > max_retries:
                    print(f"[{job_id}] {stage_name} max retries exceeded", file=sys.stderr)
                    return rc
                time.sleep(2 ** retry_count)  # Exponential backoff
            else:
                return rc

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--temp-dir", default="./temp")
    parser.add_argument("--no-wsl-redirect", action="store_true")
    parser.add_argument("--cleanup-intermediates", action="store_true")
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--force-allow-banned-upscale", action="store_true")
    # Processing control
    parser.add_argument("--force-traditional-first", action="store_true",
                       help="Force traditional processing before AI (default: true)")
    parser.add_argument("--face-restore", choices=["GFPGAN", "CodeFormer", ""], default=None)
    # Quality overrides
    parser.add_argument("--crf", type=int, default=None)
    parser.add_argument("--preset", type=int, default=None)
    parser.add_argument("--aq-mode", type=int, default=None)
    # AI params
    parser.add_argument("--ai-enabled", action="store_true")
    parser.add_argument("--ai-upscale-factor", type=int, default=None)
    parser.add_argument("--ai-model", default=None)
    parser.add_argument("--ai-denoise-strength", type=float, default=None)
    parser.add_argument("--ai-pre-upscale-lanczos", action="store_true")

    args = parser.parse_args()

    config = _load_config(args.config)

    # Default force_traditional_first to True
    force_traditional = True if args.force_traditional_first is None else args.force_traditional_first

    # Classify
    sys.path.insert(0, str(SCRIPT_DIR))
    from classify import classify_photo
    classification = classify_photo(args.input)
    detected_class = classification.get("detected_class", "")

    cli_params = {}
    if args.output:
        cli_params["output_path"] = args.output
    if args.crf is not None:
        cli_params["crf"] = args.crf
    if args.preset is not None:
        cli_params["preset"] = args.preset
    if args.aq_mode is not None:
        cli_params["aq_mode"] = args.aq_mode
    if args.face_restore is not None:
        cli_params["face_restore"] = args.face_restore
    if args.ai_enabled:
        cli_params.setdefault("ai", {})["enabled"] = True
    if args.ai_upscale_factor:
        cli_params.setdefault("ai", {})["upscale_factor"] = args.ai_upscale_factor
    if args.ai_model:
        cli_params.setdefault("ai", {})["model"] = args.ai_model
    if args.ai_denoise_strength is not None:
        cli_params.setdefault("ai", {})["denoise_strength"] = args.ai_denoise_strength
    if args.ai_pre_upscale_lanczos:
        cli_params.setdefault("ai", {})["pre_upscale_lanczos"] = True
    if args.force_allow_banned_upscale:
        cli_params["force_allow_banned_upscale"] = True

    cli_params["cleanup_intermediates"] = not args.no_cleanup
    cli_params["temp_dir"] = args.temp_dir
    cli_params["log_dir"] = args.log_dir
    cli_params["force_traditional_first"] = force_traditional

    # Preview mode
    if args.preview:
        print("Preview mode: showing job plan")
        job = {
            "job_id": args.job_id or f"photo_{int(time.time())}",
            "input_path": args.input,
            "detected_class": detected_class,
            "confidence": classification.get("confidence", 0),
            "ai_upscale_banned": classification.get("ai_upscale_banned", False),
        }
        print(json.dumps(job, indent=2, ensure_ascii=False))
        return 0

    # Run job
    job = {
        "job_id": args.job_id or f"photo_{int(time.time())}",
        "input_path": args.input,
        "detected_class": detected_class,
        "confidence": classification.get("confidence", 0),
        "ai_upscale_banned": classification.get("ai_upscale_banned", False),
        "user_overrides": {},
    }

    rc = _run_single_job(job, args.config, config, cli_params)
    sys.exit(rc)


if __name__ == "__main__":
    main()
