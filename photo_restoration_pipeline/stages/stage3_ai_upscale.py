#!/usr/bin/env python3
"""
Stage 3: Photo AI Upscaling
Integer-scale AI super-resolution for photos.
Supports: Real-ESRGAN, DAT models.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_utils"))

from logger import get_logger  # noqa: E402
from checkpoint import compute_param_hash, build_intermediate_name, gpu_memory_check  # noqa: E402


def run_realesrgan(input_path: str, output_path: str, model: str,
                   scale: int, model_dir: str = "", use_gpu: bool = True) -> bool:
    """Run Real-ESRGAN on a single image."""
    if not model_dir:
        project_root = SCRIPT_DIR.parent.resolve()
        model_dir = str(project_root / "models")
    model_path = os.path.join(model_dir, f"{model}.pth")
    if not os.path.exists(model_path):
        return False
    try:
        cmd = [
            sys.executable, "-m", "realesrgan",
            "-i", input_path,
            "-o", output_path,
            "-n", model,
            "-s", str(scale),
        ]
        if not use_gpu:
            cmd.extend(["-g", "-1"])
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
    if scale not in (2, 4):
        logger.error("STAGE3_INVALID", f"Invalid scale: {scale}")
        return 81

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    input_name = Path(args.input).name

    # Find stage1 output
    import glob
    pattern = f"{job_id}_{Path(input_name).stem}_step1_preprocess*"
    candidates = glob.glob(os.path.join(temp_dir, pattern))
    if not candidates:
        logger.error("STAGE3_NO_INPUT", "Stage 1 output not found")
        return 82

    stage1_output = candidates[0]

    # Output path
    param_hash = compute_param_hash(args.input, params)
    inter_name = build_intermediate_name(
        args.job_id, input_name, 3, "upscale", param_hash, "png"
    )
    output_path = os.path.join(temp_dir, inter_name)

    if os.path.exists(output_path):
        logger.info("STAGE3_SKIP", "Already upscaled")
        return 3

    model = params.get("upscale_model", "RealESRGAN_x2plus")
    logger.info("STAGE3_START", "AI upscaling",
                model=model, scale=scale, input=stage1_output)

    # Upscale directory
    upscale_dir = os.path.join(temp_dir, f"{job_id}_upscaled")
    os.makedirs(upscale_dir, exist_ok=True)

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
        # Try downgrade 4x -> 2x
        if scale == 4:
            actual_scale = 2
            actual_model = "RealESRGAN_x2plus"
            vram_required = int(img_w * img_h * 3 * 4 * (2 + actual_scale * actual_scale))
            sufficient, backend = gpu_memory_check(vram_required)
            if sufficient:
                logger.warning("STAGE3_DOWNSCALE", "Downgraded from 4x to 2x due to VRAM limit")
            else:
                logger.warning("STAGE3_DOWNSCALE", "4x->2x still insufficient, will try CPU")
                use_gpu = False
        else:
            use_gpu = False

    model_dir = params.get("model_dir", "")
    success = run_realesrgan(
        stage1_output, upscale_dir, actual_model, actual_scale, model_dir,
        use_gpu=use_gpu
    )

    if success:
        # Find output file
        upscaled_files = list(Path(upscale_dir).glob("*.png"))
        if upscaled_files:
            import shutil
            shutil.copy2(str(upscaled_files[0]), output_path)
            logger.info("STAGE3_DONE", "Upscale complete",
                        output=output_path, scale=actual_scale, backend=backend if use_gpu else "cpu")
            return 0

    logger.error("STAGE3_FAILED", "AI upscaling failed")
    return 83


if __name__ == "__main__":
    sys.exit(main() or 0)
