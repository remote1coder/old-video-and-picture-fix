#!/usr/bin/env python3
"""
Stage 3: Photo AI Upscaling
Integer-scale AI super-resolution for photos.
Supports: Real-ESRGAN, DAT models.

v6.3 Compliance:
- Target resolution must be <= 1080p (1920x1080) for AI upscale
- Original input pixels and intermediate pixels are BANNED from AI upscale
- Lanczos preprocessing to AI is BANNED
- GPU memory pre-check with CPU fallback
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from checkpoint import compute_param_hash, build_intermediate_name, gpu_memory_check  # noqa: E402


def _get_model_basename(model: str) -> str:
    """Extract model family from full model name."""
    # e.g. "RealESRGAN_x2plus" -> "RealESRGAN", "DAT_x2" -> "DAT"
    if model.startswith("RealESRGAN"):
        return "RealESRGAN"
    elif model.startswith("DAT"):
        return "DAT"
    return "RealESRGAN"


def _get_model_for_scale(model: str, scale: int) -> str:
    """Get correct model name for a given scale, preserving model family."""
    family = _get_model_basename(model)
    if family == "RealESRGAN":
        return f"RealESRGAN_x{scale}plus"
    elif family == "DAT":
        return f"DAT_x{scale}"
    return f"RealESRGAN_x{scale}plus"


def _resolve_input_source(temp_dir: str, job_id: str, input_name: str,
                          ban_original: bool, ban_intermediate: bool) -> tuple:
    """Resolve valid input source for AI upscale.

    Rules (v6.3 §5.3):
    - Original input pixels are banned
    - Intermediate pixels from prior AI stages are banned
    - Only traditional-processed intermediates are allowed

    Returns: (path, source_type) or (None, "") if no valid source
    """
    stem = Path(input_name).stem
    # Stage 1 preprocess is the only allowed source for AI upscale
    pattern = f"{job_id}_{stem}_step1_preprocess*"
    candidates = glob.glob(os.path.join(temp_dir, pattern))
    if candidates:
        # Sort by mtime, pick most recent
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0], "traditional"
    return None, ""


def run_realesrgan(input_path: str, output_path: str, model: str,
                   scale: int, model_dir: str = "", use_gpu: bool = True) -> bool:
    """Run Real-ESRGAN on a single image via realesrgan-ncnn-vulkan or Python API fallback."""
    if not model_dir:
        project_root = SCRIPT_DIR.parent.resolve()
        model_dir = str(project_root / "models")
    model_path = os.path.join(model_dir, f"{model}.pth")
    if not os.path.exists(model_path):
        return False

    # Try realesrgan-ncnn-vulkan CLI first (preferred)
    vulkan_bin = shutil.which("realesrgan-ncnn-vulkan")
    if vulkan_bin:
        cmd = [
            vulkan_bin,
            "-i", input_path,
            "-o", output_path,
            "-n", model,
            "-s", str(scale),
        ]
        if not use_gpu:
            cmd.extend(["-g", "-1"])
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=1800, shell=False)
            return r.returncode == 0
        except Exception:
            pass

    # Fallback: Python API via realesrgan package
    try:
        cmd = [
            sys.executable, "-c",
            f"from realesrgan import RealESRGANer; "
            f"from basicsr.utils.download_util import load_file_from_url; "
            f"import torch; "
            f"model = RealESRGANer(scale={scale}, model_path='{model_path}', "
            f"prepad=10, half={'False' if not use_gpu else 'not torch.cuda.is_available()'}); "
            f"from PIL import Image; "
            f"img = Image.open('{input_path}').convert('RGB'); "
            f"import numpy as np; "
            f"output, _ = model.enhance(np.array(img)); "
            f"Image.fromarray(output).save('{output_path}')",
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=1800, shell=False)
        return r.returncode == 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--params")
    parser.add_argument("--config")
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()

    params = json.loads(args.params) if args.params else {}
    log_dir = params.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, job_id=args.job_id)

    ai_enabled = params.get("ai_enabled", False)
    if not ai_enabled:
        logger.info("STAGE3_SKIP", "AI not enabled")
        return 3

    ai_banned = params.get("ai_upscale_banned", False)
    force_allow = params.get("force_allow_banned_upscale", False)
    if ai_banned and not force_allow:
        logger.info("STAGE3_BANNED", "AI upscale banned")
        return 3

    scale = params.get("upscale_scale", 2)
    max_scale = params.get("max_scale", 4)
    # v6.3: Only integer scales 2x and 4x are allowed
    if scale not in (2, 4):
        logger.error("STAGE3_INVALID",
                     f"Invalid scale: {scale}. Only integer 2x or 4x allowed.")
        return 81
    if scale > max_scale:
        logger.error("STAGE3_INVALID", f"Scale {scale} exceeds max_scale={max_scale}")
        return 81

    model = params.get("upscale_model", "RealESRGAN_x2plus")
    # v6.3: Anime model banned for real photos unless explicitly allowed
    if "anime" in model.lower():
        logger.error("STAGE3_ANIME_BANNED",
                     f"Anime model {model} is banned for real photos. "
                     "Use RealESRGAN_x2plus/x4plus or DAT instead.")
        return 81

    # v6.3 §5.3: Target resolution check - AI upscale only available when target <= 1080p
    original_width = params.get("original_width", 0)
    original_height = params.get("original_height", 0)
    target_w = original_width * scale
    target_h = original_height * scale
    # 1080p threshold: 1920x1080
    if target_w > 1920 or target_h > 1080:
        logger.warning("STAGE3_RES_LIMIT",
                       f"Target resolution {target_w}x{target_h} exceeds 1080p, "
                       "skipping AI upscale per v6.3 §5.3")
        return 3

    # v6.3 §5.3: Enforce traditional-first if configured
    force_traditional_first = params.get("force_traditional_first", True)
    if force_traditional_first:
        temp_dir = params.get("temp_dir", "./temp")
        job_id = args.job_id
        input_name = Path(args.input).name
        # Check stage 1 output exists
        stage1_pattern = f"{job_id}_{Path(input_name).stem}_step1_preprocess*"
        stage1_candidates = glob.glob(os.path.join(temp_dir, stage1_pattern))
        if not stage1_candidates:
            logger.error("STAGE3_NO_TRADITIONAL",
                       "force_traditional_first=true but no stage1 output found")
            return 82

    # v6.3 §5.3: Ban original input pixels and intermediate pixels
    ban_original = params.get("ban_original_input_pixels", True)
    ban_intermediate = params.get("ban_intermediate_pixels", True)

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    input_name = Path(args.input).name

    # Resolve valid input source
    stage1_output, source_type = _resolve_input_source(
        temp_dir, job_id, input_name, ban_original, ban_intermediate
    )
    if not stage1_output:
        logger.error("STAGE3_NO_INPUT", "No valid traditional-processed input found for AI upscale")
        return 82

    logger.info("STAGE3_SOURCE", f"Using {source_type} source", path=stage1_output)

    # v6.3 §5.3: Lanczos target must be "auto" (no pre-resize to AI)
    lanczos_target = params.get("lanczos_target", "auto")
    if lanczos_target and lanczos_target != "auto":
        logger.warning("STAGE3_LANCZOS_BANNED",
                       "lanczos_target is set but v6.3 §5.3 prohibits Lanczos pre-processing to AI")

    # Output path
    param_hash = compute_param_hash(args.input, params)
    inter_name = build_intermediate_name(
        args.job_id, input_name, 3, "upscale", param_hash, "png"
    )
    output_path = os.path.join(temp_dir, inter_name)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info("STAGE3_SKIP", "Already upscaled")
        return 3

    model = params.get("upscale_model", "RealESRGAN_x2plus")
    logger.info("STAGE3_START", "AI upscaling",
                model=model, scale=scale, input=stage1_output)

    # Upscale directory
    upscale_dir = os.path.join(temp_dir, f"{job_id}_upscaled")
    os.makedirs(upscale_dir, exist_ok=True)
    output_file = os.path.join(upscale_dir, f"{Path(input_name).stem}_x{scale}.png")

    # GPU memory pre-check and fallback
    try:
        from PIL import Image
        with Image.open(stage1_output) as img:
            img_w, img_h = img.size
    except Exception:
        img_w, img_h = 1920, 1080  # fallback conservative estimate

    # Estimated VRAM: width * height * channels * 4 bytes * (2 + scale^2)
    vram_required = int(img_w * img_h * 3 * 4 * (2 + scale * scale))
    sufficient, backend = gpu_memory_check(vram_required)

    use_gpu = True
    actual_scale = scale
    actual_model = model

    if not sufficient:
        logger.warning("STAGE3_VRAM_LOW",
                       f"GPU VRAM insufficient for {scale}x (need ~{vram_required // (1024*1024)}MiB)")
        # Try downgrade, preserving model family
        if scale == 4:
            actual_scale = 2
            actual_model = _get_model_for_scale(model, 2)
            vram_required = int(img_w * img_h * 3 * 4 * (2 + actual_scale * actual_scale))
            sufficient, backend = gpu_memory_check(vram_required)
            if sufficient:
                logger.warning("STAGE3_DOWNSCALE", f"Downgraded from 4x to 2x ({actual_model}) due to VRAM limit")
            else:
                logger.warning("STAGE3_DOWNSCALE", "4x->2x still insufficient, will try CPU")
                use_gpu = False
        else:
            use_gpu = False

    model_dir = params.get("model_dir", "")
    success = run_realesrgan(
        stage1_output, output_file, actual_model, actual_scale, model_dir,
        use_gpu=use_gpu
    )

    if success and os.path.exists(output_file):
        import shutil
        shutil.copy2(output_file, output_path)
        logger.info("STAGE3_DONE", "Upscale complete",
                    output=output_path, scale=actual_scale,
                    backend=backend if use_gpu else "cpu")
        return 0

    logger.error("STAGE3_FAILED", "AI upscaling failed")
    return 83


if __name__ == "__main__":
    sys.exit(main() or 0)
