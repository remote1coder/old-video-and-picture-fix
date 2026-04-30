#!/usr/bin/env python3
"""
Stage 5: Photo Face / Portrait Restoration
GFPGAN / CodeFormer for face restoration in photos.
Full-frame processing (no temporal constraints for static images).
"""

import os
import sys
import json
import argparse
import subprocess
import glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_utils"))

from logger import get_logger  # noqa: E402


def feather_blend(original_path: str, restored_path: str, output_path: str,
                  feather_width: int = 8) -> bool:
    """Blend restored face region back into original image with feathering.

    Uses per-pixel absolute difference to detect modified regions,
    then applies Gaussian blur to the mask edge for smooth transition.
    """
    try:
        import numpy as np
        from PIL import Image, ImageFilter

        orig = Image.open(original_path).convert("RGB")
        rest = Image.open(restored_path).convert("RGB")

        # Ensure same size
        if orig.size != rest.size:
            rest = rest.resize(orig.size, Image.LANCZOS)

        orig_arr = np.array(orig, dtype=np.float32)
        rest_arr = np.array(rest, dtype=np.float32)

        # Compute per-pixel absolute difference (average across RGB)
        diff = np.mean(np.abs(rest_arr - orig_arr), axis=2)

        # Threshold: consider pixels with >5% of max difference as modified
        threshold = 255.0 * 0.05
        mask = (diff > threshold).astype(np.float32)

        # If mask covers the whole image (extreme case), skip blending
        mask_ratio = np.mean(mask)
        if mask_ratio > 0.95 or mask_ratio < 0.01:
            import shutil
            shutil.copy2(restored_path, output_path)
            return True

        # Convert mask to PIL image for Gaussian blur
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_blurred = mask_img.filter(ImageFilter.GaussianBlur(radius=feather_width))
        mask_smooth = np.array(mask_blurred, dtype=np.float32) / 255.0
        mask_smooth = np.clip(mask_smooth, 0.0, 1.0)
        # Expand to 3 channels
        mask_smooth = np.stack([mask_smooth] * 3, axis=2)

        # Alpha blending
        blended = mask_smooth * rest_arr + (1.0 - mask_smooth) * orig_arr
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        result = Image.fromarray(blended)
        result.save(output_path)
        return True
    except Exception:
        # Fallback: direct copy if blending fails
        try:
            import shutil
            shutil.copy2(restored_path, output_path)
            return True
        except Exception:
            return False


def run_gfpgan(input_path: str, output_dir: str) -> bool:
    """Run GFPGAN on a single image."""
    try:
        cmd = [
            sys.executable, "-m", "gfpgan",
            "-i", input_path,
            "-o", output_dir,
            "-v", "1.4",
            "-s", "2",
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=600, shell=False)
        return r.returncode == 0
    except Exception:
        return False


def run_codeformer(input_path: str, output_dir: str) -> bool:
    """Run CodeFormer on a single image."""
    try:
        # CodeFormer inference via basicSR/codeformer
        cmd = [
            sys.executable, "-m", "codeformer",
            "-i", input_path,
            "-o", output_dir,
            "--face_upsample",
            "--bg_upsampler", "realesrgan",
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=600, shell=False)
        if r.returncode == 0:
            return True
        # Fallback: try inference_codeformer.py directly
        cmd2 = [
            sys.executable,
            "-c", "from codeformer.app import inference; inference('{}', '{}')".format(
                input_path, output_dir
            ),
        ]
        r2 = subprocess.run(cmd2, capture_output=True, timeout=600, shell=False)
        return r2.returncode == 0
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

    face_restore = params.get("face_restore", "")
    ai_enabled = params.get("ai_enabled", False)
    if not ai_enabled or not face_restore:
        logger.info("STAGE5_SKIP", "Face restore not enabled")
        return 3

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    input_name = Path(args.input).name

    # Find input: stage4 > stage3 > stage1
    input_source = None
    for pat in ["step4_postprocess*", "step3_upscale*", "step1_preprocess*"]:
        pattern = f"{job_id}_{Path(input_name).stem}_{pat}"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        if candidates:
            input_source = candidates[0]
            break

    if not input_source:
        logger.error("STAGE5_NO_INPUT", "No input found")
        return 82

    output_path = os.path.join(temp_dir,
        f"{job_id}_{Path(input_name).stem}_step5_face.png")

    if os.path.exists(output_path):
        return 3

    logger.info("STAGE5_START", f"Face restore: {face_restore}")

    if face_restore == "GFPGAN":
        face_dir = os.path.join(temp_dir, f"{job_id}_face")
        os.makedirs(face_dir, exist_ok=True)

        if run_gfpgan(input_source, face_dir):
            face_files = list(Path(face_dir).rglob("*.png"))
            if face_files:
                if feather_blend(input_source, str(face_files[0]), output_path, feather_width=8):
                    logger.info("STAGE5_DONE", "Face restore complete (GFPGAN)")
                    return 0
                else:
                    logger.error("STAGE5_BLEND_FAIL", "Feather blend failed")
                    return 83

    elif face_restore == "CodeFormer":
        face_dir = os.path.join(temp_dir, f"{job_id}_face")
        os.makedirs(face_dir, exist_ok=True)

        if run_codeformer(input_source, face_dir):
            face_files = list(Path(face_dir).rglob("*.png"))
            if face_files:
                if feather_blend(input_source, str(face_files[0]), output_path, feather_width=8):
                    logger.info("STAGE5_DONE", "Face restore complete (CodeFormer)")
                    return 0
                else:
                    logger.error("STAGE5_BLEND_FAIL", "Feather blend failed")
                    return 83

    logger.error("STAGE5_FAILED", "Face restore failed")
    return 83


if __name__ == "__main__":
    sys.exit(main() or 0)
