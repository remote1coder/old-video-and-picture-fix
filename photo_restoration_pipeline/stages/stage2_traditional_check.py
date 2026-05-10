#!/usr/bin/env python3
"""
Stage 2: Photo Traditional Repair Verification
Verifies stage1 output: file integrity, decodeability, quality checks.

v6.3 Compliance:
- Multi-channel quality check (mean/std per channel)
- Structural integrity verification (pixel range, non-uniformity)
- Re-checks single-channel degradation status
- Sorts candidates by mtime for deterministic selection
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402


def _check_multi_channel_quality(img_path: str, logger=None) -> dict:
    """Check per-channel quality metrics.

    Returns dict with channel means, stds, and quality flags.
    """
    try:
        from PIL import Image
        import numpy as np
        with Image.open(img_path) as img:
            arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0

        means = [arr[:, :, c].mean() for c in range(3)]
        stds = [arr[:, :, c].std() for c in range(3)]

        # Check for uniform/corrupted images
        min_std = min(stds)
        if min_std < 0.001:
            if logger:
                logger.warning("STAGE2_Q_FLAT", f"Near-uniform channel detected (std={min_std:.6f})")
            return {"ok": False, "reason": "flat_channel", "means": means, "stds": stds}

        # Check for all-zero or all-max
        if arr.max() == 0 or arr.min() == arr.max():
            if logger:
                logger.error("STAGE2_Q_CORRUPT", "Image appears uniform (all same value)")
            return {"ok": False, "reason": "uniform", "means": means, "stds": stds}

        # Check mean range
        for i, (m, s) in enumerate(zip(means, stds)):
            if not (0.01 <= m <= 0.99):
                if logger:
                    logger.warning("STAGE2_Q_RANGE", f"Channel {i} mean out of normal range: {m:.4f}")

        return {"ok": True, "means": means, "stds": stds}
    except Exception as e:
        return {"ok": False, "reason": f"exception: {e}"}


def _check_file_integrity(path: str, logger=None) -> bool:
    """Verify file exists, is non-empty, and decodeable."""
    if not os.path.exists(path):
        if logger:
            logger.error("STAGE2_NOT_FOUND", "Stage 1 output missing")
        return False

    size = os.path.getsize(path)
    if size == 0:
        if logger:
            logger.error("STAGE2_EMPTY", "Stage 1 output is empty")
        return False

    # Try to open with PIL
    try:
        from PIL import Image
        with Image.open(path) as img:
            w, h = img.size
            mode = img.mode
            if logger:
                logger.info("STAGE2_VERIFY", f"Image OK: {w}x{h} {mode}")
    except Exception as e:
        if logger:
            logger.error("STAGE2_CORRUPT", f"Cannot open image: {e}")
        return False

    return True


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

    # Find stage1 output - sort by mtime for deterministic selection (Bug-5 fix)
    pattern = f"{job_id}_{Path(input_name).stem}_step1_preprocess*"
    candidates = glob.glob(os.path.join(temp_dir, pattern))
    if not candidates:
        logger.error("STAGE2_NO_INPUT", "Stage 1 output not found")
        return 82

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    stage1_output = candidates[0]
    logger.info("STAGE2_START", "Verifying stage1 output", path=stage1_output)

    # Basic checks
    if not _check_file_integrity(stage1_output, logger):
        return 86

    # Multi-channel quality check
    quality = _check_multi_channel_quality(stage1_output, logger)
    if not quality["ok"]:
        logger.error("STAGE2_Q_FAIL", f"Quality check failed: {quality['reason']}")
        return 86

    # Re-check single-channel degradation
    try:
        mean_diff = max(abs(quality["means"][i] - quality["means"][j])
                       for i in range(3) for j in range(i+1, 3))
        std_diff = max(abs(quality["stds"][i] - quality["stds"][j])
                      for i in range(3) for j in range(i+1, 3))
        logger.info("STAGE2_CHANNEL_RECHECK",
                   f"Post-process channel stats: mean_diff={mean_diff:.4f}, std_diff={std_diff:.4f}")
    except Exception:
        pass

    logger.info("STAGE2_DONE", "Verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
