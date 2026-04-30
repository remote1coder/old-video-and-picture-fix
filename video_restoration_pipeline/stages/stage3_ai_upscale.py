#!/usr/bin/env python3
"""
Stage 3: AI Upscaling
Performs integer-scale AI super-resolution (2x or 4x only).
GPU memory pre-check with automatic fallback (4x->2x->CPU).

Exit codes: 0=success, 3=skipped, 83=GPU error
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


def estimate_gpu_memory(width: int, height: int, scale: int,
                        model_name: str) -> int:
    """Estimate GPU memory requirement in bytes."""
    # Base formula: width * height * channels * bytes * batch * (2 + scale^2)
    base = width * height * 3 * 4 * 1 * (2 + scale * scale)
    # Model weight overhead
    model_weights = {
        "RealESRGAN_x2plus": 64 * 1024 * 1024,
        "RealESRGAN_x4plus": 128 * 1024 * 1024,
        "DAT_x2": 32 * 1024 * 1024,
        "DAT_x4": 64 * 1024 * 1024,
    }
    weight = model_weights.get(model_name, 64 * 1024 * 1024)
    return base + weight


def check_gpu_memory(required_bytes: int) -> tuple:
    """Check GPU availability. Returns (sufficient: bool, backend: str)."""
    # Try nvidia-smi
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, shell=False
        )
        if r.returncode == 0:
            free_mib = float(r.stdout.strip().split("\n")[0].strip())
            free_bytes = int(free_mib * 1024 * 1024)
            if free_bytes >= required_bytes * 1.2:
                return True, "cuda"
    except Exception:
        pass

    # Try torch
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            if free >= required_bytes * 1.2:
                return True, "cuda"
    except Exception:
        pass

    return False, "cpu"


def run_realesrgan(input_path: str, output_dir: str, model_name: str,
                   scale: int, gpu_id: int = 0) -> bool:
    """Run Real-ESRGAN upscaling."""
    model_dir = SCRIPT_DIR.parent.resolve() / "models"
    model_path = str(model_dir / f"{model_name}.pth")
    if not os.path.exists(model_path):
        return False

    try:
        # Use Real-ESRGAN inference script
        cmd = [
            sys.executable, "-m", "realesrgan",
            "-i", input_path,
            "-o", output_dir,
            "-n", model_name,
            "-s", str(scale),
            "-g", str(gpu_id),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=3600, shell=False)
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

    # Check AI settings
    ai_enabled = params.get("ai_enabled", False)
    if not ai_enabled:
        logger.info("STAGE3_SKIP", "AI not enabled")
        return 3

    # Check upscale ban
    ai_banned = params.get("ai_upscale_banned", False)
    force_allow = params.get("force_allow_banned_upscale", False)
    if ai_banned and not force_allow:
        logger.info("STAGE3_BANNED", "AI upscale banned for this file")
        return 3

    scale = params.get("upscale_scale", 2)
    if scale not in (2, 4):
        logger.error("STAGE3_INVALID_SCALE", f"Invalid scale: {scale}")
        return 81

    model = params.get("upscale_model", "RealESRGAN_x2plus")
    temp_dir = params.get("temp_dir", "./temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Find stage1 output
    job_id = args.job_id
    input_name = Path(args.input).name
    pattern = f"{job_id}_{Path(input_name).stem}_step1_preprocess*"
    import glob
    candidates = glob.glob(os.path.join(temp_dir, pattern))
    if not candidates:
        logger.error("STAGE3_NO_INPUT", "Stage 1 output not found")
        return 82

    stage1_output = candidates[0]

    # Get dimensions
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "json", stage1_output],
            capture_output=True, text=True, timeout=30, shell=False
        )
        info = json.loads(r.stdout)
        stream = info["streams"][0]
        width = stream["width"]
        height = stream["height"]
    except Exception:
        width, height = 720, 480

    # GPU memory check
    mem_required = estimate_gpu_memory(width, height, scale, model)
    sufficient, backend = check_gpu_memory(mem_required)

    if not sufficient and backend == "cpu":
        # Try 2x if requested 4x
        if scale == 4:
            logger.warning("STAGE3_DOWNSCALE", "GPU memory insufficient for 4x, trying 2x")
            scale = 2
            mem_required = estimate_gpu_memory(width, height, scale, model)
            model = model.replace("x4", "x2")
            sufficient, backend = check_gpu_memory(mem_required)

        if not sufficient:
            logger.warning("STAGE3_CPU", "GPU unavailable, using CPU (very slow)")
            backend = "cpu"

    logger.info("STAGE3_START", "AI upscaling",
                model=model, scale=scale, backend=backend,
                input_width=width, input_height=height)

    # Extract frames
    frames_dir = os.path.join(temp_dir, f"{job_id}_frames")
    os.makedirs(frames_dir, exist_ok=True)

    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-i", stage1_output,
             "-pix_fmt", "rgb48be",
             os.path.join(frames_dir, "frame_%08d.png")],
            capture_output=True, timeout=600, shell=False
        )
    except Exception as e:
        logger.error("STAGE3_EXTRACT_FAIL", str(e))
        return 82

    # Run AI upscaling
    output_dir = os.path.join(temp_dir, f"{job_id}_upscaled")
    os.makedirs(output_dir, exist_ok=True)

    success = run_realesrgan(frames_dir, output_dir, model, scale)

    if not success and backend != "cpu":
        # Wait for GPU memory reclamation, then retry on CPU
        import time
        time.sleep(0.1)
        logger.warning("STAGE3_RETRY_CPU", "GPU inference failed, retrying on CPU")
        success = run_realesrgan(frames_dir, output_dir, model, scale, gpu_id=-1)

    if success:
        logger.info("STAGE3_DONE", "AI upscaling complete")
        return 0
    else:
        logger.error("STAGE3_FAILED", "AI upscaling failed")
        return 83


if __name__ == "__main__":
    sys.exit(main() or 0)
