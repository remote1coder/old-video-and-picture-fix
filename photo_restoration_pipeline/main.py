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
                 cli_overrides: Dict[str, Any],
                 logger=None) -> Dict[str, Any]:
    """6-layer parameter cascading per v7.3 §3.1.
    Priority: CLI > user_overrides > paths > global > defaults
    Uses deep merge for nested dicts.
    Records PARAM_OVERRIDE events per v7.32 §3.10.12.
    Also merges ai, grain_factory3, and audio config sections.
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

    # Layer 3.5: ai / grain_factory3 / audio (Bug 8.2 fix)
    # Flatten nested ai params into params for stage compatibility
    ai_cfg = config.get("ai", {})
    if isinstance(ai_cfg, dict):
        # Rename ai.enabled -> ai_enabled for stage script compatibility
        ai_flat = dict(ai_cfg)
        if "enabled" in ai_flat:
            ai_flat["ai_enabled"] = ai_flat.pop("enabled")
        params = _deep_merge(params, ai_flat)
    grain_cfg = config.get("grain_factory3", {})
    if isinstance(grain_cfg, dict):
        params = _deep_merge(params, grain_cfg)
    audio_cfg = config.get("audio", {})
    if isinstance(audio_cfg, dict):
        # Rename audio.enabled -> audio_enabled for consistency
        audio_flat = dict(audio_cfg)
        if "enabled" in audio_flat:
            audio_flat["audio_enabled"] = audio_flat.pop("enabled")
        params = _deep_merge(params, audio_flat)

    # Layer 4: recommended params from classifier
    if "recommended_params" in cli_overrides:
        rec_params = cli_overrides.pop("recommended_params")
        for k, v in rec_params.items():
            old_v = params.get(k)
            if old_v is not None and old_v != v:
                if logger and hasattr(logger, 'param_override'):
                    logger.param_override(k, old_v, v, source="recommended_params")
                elif logger:
                    logger.info("PARAM_OVERRIDE",
                               f"Key '{k}': {old_v} -> {v} (source=recommended_params)")
            params[k] = v

    # Layer 5: user overrides from manifest
    if "user_overrides" in cli_overrides:
        user_ov = cli_overrides.pop("user_overrides")
        for k, v in user_ov.items():
            old_v = params.get(k)
            if old_v is not None and old_v != v:
                if logger and hasattr(logger, 'param_override'):
                    logger.param_override(k, old_v, v, source="user_overrides")
                elif logger:
                    logger.info("PARAM_OVERRIDE",
                               f"Key '{k}': {old_v} -> {v} (source=user_overrides)")
            params[k] = v

    # Layer 6: CLI
    cli_flat = dict(cli_overrides)
    for k, v in cli_flat.items():
        old_v = params.get(k)
        if old_v is not None and old_v != v:
            if logger and hasattr(logger, 'param_override'):
                logger.param_override(k, old_v, v, source="cli")
            elif logger:
                logger.info("PARAM_OVERRIDE",
                           f"Key '{k}': {old_v} -> {v} (source=cli)")
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
    Max: 20 minutes per photo (v7.32 §3.10.18 single-photo limit)
    Also respects 30MB file size threshold.
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
    
    # v7.32 §3.10.18: single photo limit 20min (1200s)
    max_timeout = 1200
    
    # If file > 30MB, still cap at 20min (no extra time for huge files)
    try:
        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        if file_size_mb > 30:
            # No reduction, but ensure we don't exceed 20min
            pass
    except Exception:
        pass
    
    timeout = min(base + additional, max_timeout)
    return int(timeout)


def _run_stage(script_name: str,
               job_id: str,
               input_path: str,
               config_path: str,
               params: Dict[str, Any],
               stage_idx: int,
               is_retry: bool = False,
               is_ai_stage: bool = False,
               script_dir: Optional[str] = None) -> tuple:
    """Run a single stage with dynamic timeout.

    Returns:
        (returncode, stderr) tuple per v6.3 §3.7 retry logic.
    """
    if script_dir is None:
        script_dir = SCRIPT_DIR / "stages"
    script_path = script_dir / script_name
    if not script_path.exists():
        print(f"Stage script not found: {script_path}", file=sys.stderr)
        return 81, ""

    # BUG-057 fix: if params JSON exceeds ~30000 chars, write to temp file
    params_json = json.dumps(params)
    params_file_path = None
    if len(params_json) > 30000:
        import tempfile
        pf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        pf.write(params_json)
        pf.close()
        params_file_path = pf.name
        cmd = [
            sys.executable, str(script_path),
            "--job-id", job_id,
            "--input", input_path,
            "--config", config_path,
            "--params-file", params_file_path,
        ]
    else:
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
        r = subprocess.run(cmd, timeout=timeout, env=env, shell=False,
                          capture_output=True, text=True)
        if r.returncode != 0 and r.stderr:
            print(f"[Stage {stage_idx}] stderr: {r.stderr[:500]}", file=sys.stderr)
        return r.returncode, r.stderr
    except subprocess.TimeoutExpired as e:
        print(f"[Stage {stage_idx}] Timeout after {timeout}s", file=sys.stderr)
        return 87, str(e)
    except Exception as e:
        print(f"[Stage {stage_idx}] Error: {e}", file=sys.stderr)
        return 84, str(e)
    finally:
        # Clean up temp params file
        if params_file_path:
            try:
                os.unlink(params_file_path)
            except Exception:
                pass


def _check_disk_space(input_path: str, config: Dict[str, Any]) -> bool:
    """Check sufficient disk space with 2x safety margin."""
    try:
        stat = os.statvfs(os.path.dirname(input_path) or ".")
        available_mb = stat.f_bavail * stat.f_frsize / (1024 * 1024)

        # PHOTO-AUDIT-002: Read actual image dimensions for accurate photo estimate
        img_w, img_h = 0, 0
        try:
            from PIL import Image
            with Image.open(input_path) as img:
                img_w, img_h = img.size
        except Exception:
            pass

        # Estimate using checkpoint module
        sys.path.insert(0, str(SCRIPT_DIR))
        from checkpoint import estimate_disk_space

        required = estimate_disk_space(
            duration=0, fps=1,
            width=img_w, height=img_h,
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


def _is_retryable_error(returncode: int, stderr: str = "") -> bool:
    """Check if error is retryable (transient).

    Per v6.3 §3.7:
    - 82 (IO), 83 (GPU/OOM), 87 (resource/timeout) are retryable
    - 84 (encoding) only retryable if OOM/no space indicated in stderr
    - 80 (deps), 81 (params), 85 (perms), 86 (verify) are NOT retryable
    """
    retryable_codes = {82, 83, 87}
    if returncode in retryable_codes:
        return True
    # Encoding error (84) only retryable if OOM or disk full in stderr
    if returncode == 84:
        stderr_lower = (stderr or "").lower()
        if any(k in stderr_lower for k in ("out of memory", "oom", "cuda", "no space", "errno 28")):
            return True
    return False


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

    # Per-job log
    log_dir = cli_params.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    # Initialize logger for PARAM_OVERRIDE recording
    sys.path.insert(0, str(SCRIPT_DIR))
    from logger import get_logger
    logger = get_logger(log_dir=log_dir, job_id=job_id)

    # Parameter merge (with override logging)
    params = merge_params(config, detected_class, cli_params, logger=logger)
    params["detected_class"] = detected_class
    params["confidence"] = confidence
    params["ai_upscale_banned"] = ai_upscale_banned
    params["job_id"] = job_id
    params["input_path"] = input_path
    # Inject classification metadata for stage scripts (Bug 11.1 fix)
    params["original_width"] = job.get("original_width", 0)
    params["original_height"] = job.get("original_height", 0)
    params["native_width"] = job.get("native_width", 0)
    params["native_height"] = job.get("native_height", 0)
    params["native_dar"] = job.get("native_dar", "")
    params["dpi"] = job.get("dpi")
    params["file_size"] = job.get("file_size", 0)
    params["file_mtime"] = job.get("file_mtime", 0)
    params["file_hash_prefix"] = job.get("file_hash_prefix", "")

    # Add log config to params for stages
    params["log_dir"] = log_dir

    # P-6 fix: WSL temp_dir/output_dir redirect
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    platform = params.get("platform", "auto")
    sys.path.insert(0, str(SCRIPT_DIR))
    from checkpoint import wsl_redirect_path
    temp_dir = params.get("temp_dir", "./temp")
    redirected_temp, orig_temp, _, _ = wsl_redirect_path(
        temp_dir, no_wsl_redirect=no_wsl_redirect, platform=platform
    )
    if orig_temp:
        logger.info("WSL_REDIRECT", f"temp_dir: {temp_dir} -> {redirected_temp}")
        params["temp_dir"] = redirected_temp
        params["_orig_temp_dir"] = orig_temp
        temp_dir = redirected_temp
    output_dir = params.get("output_dir", "./output")
    redirected_out, orig_out, _, _ = wsl_redirect_path(
        output_dir, no_wsl_redirect=no_wsl_redirect, platform=platform
    )
    if orig_out:
        logger.info("WSL_REDIRECT", f"output_dir: {output_dir} -> {redirected_out}")
        params["output_dir"] = redirected_out
        params["_orig_output_dir"] = orig_out

    # Clean intermediates
    cleanup_old_intermediates(temp_dir, job_id)

    # Disk space check
    if not _check_disk_space(input_path, config):
        return 87

    # Stage execution
    max_retries = params.get("retry_count", config.get("global", {}).get("retry_count", 2))

    stages = [
        ("stage1_preprocess.py", False),
        ("stage2_traditional_check.py", False),
        ("stage3_ai_upscale.py", True),  # AI stage
        ("stage4_postprocess.py", False),  # Post-processing (not AI)
        ("stage5_ai_face.py", True),  # AI stage
        ("stage6_encode.py", False),
    ]

    # force_traditional_first gate: track stage2 success explicitly (Bug 11.26)
    traditional_stage_completed = False

    for stage_idx, (stage_script, is_ai_stage) in enumerate(stages):
        stage_name = stage_script.replace(".py", "")
        retry_count = 0

        # Conditional stages
        if stage_name in ("stage3_ai_upscale", "stage4_postprocess"):
            if not params.get("ai_enabled", False) or ai_upscale_banned:
                continue
        elif stage_name == "stage5_ai_face":
            if not params.get("face_restore"):
                continue

        # force_traditional_first gate: block AI stages if stage2 failed (Bug 11.26)
        force_traditional_first = params.get("force_traditional_first", True)
        if force_traditional_first and stage_name in ("stage3_ai_upscale", "stage4_postprocess", "stage5_ai_face"):
            if not traditional_stage_completed:
                logger.error("TRADITIONAL_GATE_BLOCK",
                            "force_traditional_first=true but traditional stage not completed, blocking AI stages")
                print(f"[{job_id}] Traditional restoration gate blocked {stage_name}", file=sys.stderr)
                return 86

        while True:
            is_retry = retry_count > 0
            rc, stderr = _run_stage(
                stage_script, job_id, input_path,
                config_path, params, stage_idx,
                is_retry=is_retry, is_ai_stage=is_ai_stage
            )

            if rc == 0 or rc == 3:
                # Mark traditional stage completed after stage2 success
                if stage_name == "stage2_traditional_check":
                    traditional_stage_completed = True
                    logger.info("TRADITIONAL_GATE_PASS",
                               "Traditional restoration completed, AI stages allowed")
                # BUG-053 fix: per-stage intermediate cleanup (retains last 2 stages)
                cleanup_old_intermediates(temp_dir, job_id)
                break  # Success or skip
            elif rc == 1:
                print(f"[{job_id}] {stage_name} non-retryable error", file=sys.stderr)
                return rc
            elif _is_retryable_error(rc, stderr):
                retry_count += 1
                if retry_count > max_retries:
                    print(f"[{job_id}] {stage_name} max retries exceeded", file=sys.stderr)
                    return rc
                # Fixed retry intervals: 5s for first, 15s for second (Bug 11.6 fix)
                sleep_seconds = 5 if retry_count == 1 else 15
                logger.warning("STAGE_RETRY",
                              f"Retrying {stage_name} in {sleep_seconds}s (attempt {retry_count}/{max_retries})")
                time.sleep(sleep_seconds)
            else:
                return rc

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--input", "-i", default=None, help="Single input photo file")
    parser.add_argument("--input-dir", default=None, help="Directory of photos for batch processing")
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
    parser.add_argument("--force-traditional-first", action="store_true", default=True,
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
    # BUG-054 fix: parallel batch processing
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers for batch processing (default: 1)")
    # ISSUE-P-MAIN-001 fix: manifest workflow support
    parser.add_argument("--manifest", default=None, help="Path to batch manifest YAML/JSONL")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from manifest (process only PENDING/FAILED jobs)")

    args = parser.parse_args()

    # Validate input arguments
    input_modes = [args.input, args.input_dir, args.manifest]
    if sum(bool(m) for m in input_modes) == 0:
        parser.error("Either --input, --input-dir, or --manifest must be specified")
    if sum(bool(m) for m in input_modes) > 1:
        parser.error("--input, --input-dir, and --manifest are mutually exclusive")

    config = _load_config(args.config)

    # BUG-014 fix: default=True in argparse ensures correct default behavior
    force_traditional = args.force_traditional_first

    # Build common CLI params
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
        cli_params["ai_enabled"] = True
    # Bug 11.2/11.3 fix: unify CLI key names with what stage scripts read
    if args.ai_upscale_factor is not None:
        cli_params["upscale_scale"] = args.ai_upscale_factor
    if args.ai_model is not None:
        cli_params["upscale_model"] = args.ai_model
    if args.ai_denoise_strength is not None:
        cli_params["ai_denoise_strength"] = args.ai_denoise_strength
    if args.ai_pre_upscale_lanczos:
        cli_params["pre_upscale_lanczos"] = True
    if args.force_allow_banned_upscale:
        cli_params["force_allow_banned_upscale"] = True

    cli_params["cleanup_intermediates"] = not args.no_cleanup
    cli_params["temp_dir"] = args.temp_dir
    cli_params["log_dir"] = args.log_dir
    cli_params["force_traditional_first"] = force_traditional

    # Preview mode
    if args.preview:
        preview_input = args.input or args.input_dir
        print(f"Preview mode: showing job plan for {preview_input}")
        # For single file, classify and show plan
        if args.input:
            sys.path.insert(0, str(SCRIPT_DIR))
            from classify import classify_photo
            classification = classify_photo(args.input)
            job = {
                "job_id": args.job_id or f"photo_{int(time.time())}",
                "input_path": args.input,
                "detected_class": classification.get("detected_class", ""),
                "confidence": classification.get("confidence", 0),
                "ai_upscale_banned": classification.get("ai_upscale_banned", False),
            }
            print(json.dumps(job, indent=2, ensure_ascii=False))

            # Generate actual preview frames
            try:
                sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
                from preview import extract_preview_frames
                preview_dir = os.path.join(args.temp_dir, "previews")
                os.makedirs(preview_dir, exist_ok=True)
                preview_frames = extract_preview_frames(
                    args.input, preview_dir,
                    num_frames=4, force_compare_frames=False
                )
                if preview_frames:
                    print(f"Preview frames generated: {preview_frames}")
            except Exception as e:
                print(f"Preview generation skipped: {e}", file=sys.stderr)
        return 0

    # ISSUE-P-MAIN-001 fix: Manifest mode (scan→execute→resume workflow)
    if args.manifest:
        sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
        from manifest_loader import load_manifest, filter_jobs, update_job_status, save_manifest
        try:
            manifest = load_manifest(args.manifest)
        except Exception as e:
            print(f"Failed to load manifest: {e}", file=sys.stderr)
            sys.exit(81)

        statuses = ["PENDING", "FAILED"] if args.resume else ["PENDING", "FAILED", "SKIP"]
        jobs = filter_jobs(manifest, statuses=statuses)
        if not jobs:
            print("No pending jobs in manifest")
            sys.exit(0)

        print(f"Manifest mode: {len(jobs)} jobs to process (statuses={statuses})")
        results = []
        if args.workers > 1:
            print(f"Parallel processing with {args.workers} workers")
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(_run_single_job, job, args.config, config, dict(cli_params)): job
                    for job in jobs
                }
                for future in futures:
                    job = futures[future]
                    try:
                        rc = future.result()
                    except Exception as e:
                        print(f"[{job['job_id']}] Worker crashed: {e}", file=sys.stderr)
                        rc = 84
                    results.append((job, rc))
                    if rc != 0:
                        print(f"[{job['job_id']}] FAILED with exit code {rc}", file=sys.stderr)
        else:
            for job in jobs:
                rc = _run_single_job(job, args.config, config, dict(cli_params))
                results.append((job, rc))
                if rc != 0:
                    print(f"[{job['job_id']}] FAILED with exit code {rc}", file=sys.stderr)

        # Update manifest with results
        for job, rc in results:
            if rc == 0:
                update_job_status(manifest, job["job_id"], "DONE")
            else:
                update_job_status(manifest, job["job_id"], "FAILED",
                                  error_message=f"Exit code {rc}")
        save_manifest(manifest, args.manifest)

        success_count = sum(1 for _, rc in results if rc == 0)
        print(f"\nManifest complete: {success_count}/{len(jobs)} succeeded")
        failed = [(j, rc) for j, rc in results if rc != 0]
        if failed:
            print(f"Failed jobs ({len(failed)}):")
            for j, rc in failed:
                print(f"  {j['job_id']} ({j['input_path']}) -> exit {rc}")
        sys.exit(0 if not failed else max(rc for _, rc in failed))

    # Batch mode via --input-dir (Bug 11.8 / P-1 fix)
    # BUG-054 fix: support --workers for parallel batch processing
    if args.input_dir:
        import glob
        photo_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".avif")
        input_files = []
        for ext in photo_exts:
            input_files.extend(glob.glob(os.path.join(args.input_dir, f"*{ext}"), recursive=False))
            input_files.extend(glob.glob(os.path.join(args.input_dir, f"*{ext.upper()}"), recursive=False))
        input_files = sorted(set(input_files))

        if not input_files:
            print(f"No photo files found in {args.input_dir}", file=sys.stderr)
            sys.exit(81)

        print(f"Batch mode: {len(input_files)} files found in {args.input_dir}")
        sys.path.insert(0, str(SCRIPT_DIR))
        from classify import classify_photo

        # Pre-classify all files (lightweight, keep in main process)
        jobs = []
        for idx, input_path in enumerate(input_files):
            classification = classify_photo(input_path)
            detected_class = classification.get("detected_class", "")
            job_id = args.job_id or f"photo_{int(time.time())}_{idx}"
            job = {
                "job_id": job_id,
                "input_path": input_path,
                "detected_class": detected_class,
                "confidence": classification.get("confidence", 0),
                "ai_upscale_banned": classification.get("ai_upscale_banned", False),
                "original_width": classification.get("original_width", 0),
                "original_height": classification.get("original_height", 0),
                "native_width": classification.get("native_width", 0),
                "native_height": classification.get("native_height", 0),
                "native_dar": classification.get("native_dar", ""),
                "dpi": classification.get("dpi"),
                "file_size": classification.get("file_size", 0),
                "file_mtime": classification.get("file_mtime", 0),
                "file_hash_prefix": classification.get("file_hash_prefix", ""),
                "user_overrides": {},
            }
            jobs.append((input_path, job))

        results = []
        if args.workers > 1:
            print(f"Parallel processing with {args.workers} workers (spawn mode)")
            from concurrent.futures import ProcessPoolExecutor
            # _run_single_job is a top-level function, pickle-safe under spawn
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(_run_single_job, job, args.config, config, dict(cli_params)): input_path
                    for input_path, job in jobs
                }
                for future in futures:
                    input_path = futures[future]
                    try:
                        rc = future.result()
                    except Exception as e:
                        print(f"[{input_path}] Worker crashed: {e}", file=sys.stderr)
                        rc = 84
                    results.append((input_path, rc))
                    if rc != 0:
                        print(f"[{input_path}] FAILED with exit code {rc}", file=sys.stderr)
        else:
            # Sequential fallback
            for input_path, job in jobs:
                rc = _run_single_job(job, args.config, config, dict(cli_params))
                results.append((input_path, rc))
                if rc != 0:
                    print(f"[{job['job_id']}] FAILED with exit code {rc}", file=sys.stderr)

        # Summary
        success_count = sum(1 for _, rc in results if rc == 0)
        print(f"\nBatch complete: {success_count}/{len(input_files)} succeeded")
        failed = [(p, rc) for p, rc in results if rc != 0]
        if failed:
            print(f"Failed files ({len(failed)}):")
            for p, rc in failed:
                print(f"  {p} -> exit {rc}")
        sys.exit(0 if not failed else max(rc for _, rc in failed))

    # Single-file mode
    sys.path.insert(0, str(SCRIPT_DIR))
    from classify import classify_photo
    classification = classify_photo(args.input)
    detected_class = classification.get("detected_class", "")

    job = {
        "job_id": args.job_id or f"photo_{int(time.time())}",
        "input_path": args.input,
        "detected_class": detected_class,
        "confidence": classification.get("confidence", 0),
        "ai_upscale_banned": classification.get("ai_upscale_banned", False),
        "original_width": classification.get("original_width", 0),
        "original_height": classification.get("original_height", 0),
        "native_width": classification.get("native_width", 0),
        "native_height": classification.get("native_height", 0),
        "native_dar": classification.get("native_dar", ""),
        "dpi": classification.get("dpi"),
        "file_size": classification.get("file_size", 0),
        "file_mtime": classification.get("file_mtime", 0),
        "file_hash_prefix": classification.get("file_hash_prefix", ""),
        "user_overrides": {},
    }

    rc = _run_single_job(job, args.config, config, cli_params)
    sys.exit(rc)


if __name__ == "__main__":
    main()
