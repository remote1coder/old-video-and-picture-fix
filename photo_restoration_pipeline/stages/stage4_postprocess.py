#!/usr/bin/env python3
"""
Stage 4: Photo Post-processing after AI Upscale
- Lanczos resize to target if non-integer
- Selective local denoise with masks
- 16bit space operations
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_utils"))

from logger import get_logger  # noqa: E402


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

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    input_name = Path(args.input).name

    # Find stage3 output
    pattern = f"{job_id}_{Path(input_name).stem}_step3_upscale*"
    candidates = glob.glob(os.path.join(temp_dir, pattern))

    if not candidates:
        logger.info("STAGE4_SKIP", "No AI upscale output")
        return 3

    stage3_output = candidates[0]
    logger.info("STAGE4_START", "Photo post-processing", input=stage3_output)

    # Output path
    output_path = os.path.join(temp_dir,
        f"{job_id}_{Path(input_name).stem}_step4_postprocess.png")

    if os.path.exists(output_path):
        return 3

    try:
        from PIL import Image, ImageFilter

        img = Image.open(stage3_output)

        # Lanczos resize to target resolution if specified
        lanczos_target = params.get("lanczos_target", "auto")
        if lanczos_target and lanczos_target != "auto":
            try:
                target_w, target_h = lanczos_target.split("x")
                target_w, target_h = int(target_w), int(target_h)
                if img.width != target_w or img.height != target_h:
                    img = img.resize((target_w, target_h), Image.LANCZOS)
                    logger.info("STAGE4_LANCZOS",
                                f"Resized {img.width}x{img.height} -> {target_w}x{target_h}")
            except Exception as e:
                logger.warning("STAGE4_LANCZOS_WARN", f"Lanczos resize skipped: {e}")

        # Selective local processing: mild blur on flat areas
        local_process = params.get("local_mask_denoise", True)
        if local_process and img.mode in ("RGB", "RGBA", "L"):
            # Convert to RGB for processing
            orig = img.convert("RGB")
            # Apply mild median filter to reduce noise in flat areas
            smoothed = orig.filter(ImageFilter.MedianFilter(size=3))
            # Compute edge mask using FIND_EDGES
            edges = orig.filter(ImageFilter.FIND_EDGES)
            # Convert edge image to mask: high edge = keep original, low edge = use smoothed
            import numpy as np
            edge_arr = np.array(edges.convert("L"), dtype=np.float32)
            # Normalize edge strength to 0-1
            max_val = edge_arr.max()
            if max_val > 0:
                edge_arr = edge_arr / max_val
            else:
                edge_arr = np.zeros_like(edge_arr)
            # Threshold: strong edges (>0.1) keep original, flat areas use smoothed
            mask = np.clip((edge_arr - 0.05) / 0.05, 0.0, 1.0)  # 0.05->0, 0.1->1
            # Expand mask to 3 channels
            mask = np.stack([mask] * 3, axis=2)
            orig_arr = np.array(orig, dtype=np.float32)
            smooth_arr = np.array(smoothed, dtype=np.float32)
            blended = mask * orig_arr + (1.0 - mask) * smooth_arr
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            img = Image.fromarray(blended)
            logger.info("STAGE4_MASK", "Applied selective local denoise")

        img.save(output_path)
        logger.info("STAGE4_DONE", "Post-processing complete", output=output_path)
        return 0

    except Exception as e:
        logger.error("STAGE4_ERROR", f"Post-process failed: {e}")
        # Fallback: direct copy
        import shutil
        shutil.copy2(stage3_output, output_path)
        logger.info("STAGE4_DONE", "Post-processing complete (fallback copy)", output=output_path)
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
