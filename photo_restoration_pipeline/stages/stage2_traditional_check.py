#!/usr/bin/env python3
"""
Stage 2: Photo Traditional Repair Verification
Verifies stage1 output: file integrity, decodeability, basic quality checks.
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

    # Find stage1 output
    pattern = f"{job_id}_{Path(input_name).stem}_step1_preprocess*"
    candidates = glob.glob(os.path.join(temp_dir, pattern))
    if not candidates:
        logger.error("STAGE2_NO_INPUT", "Stage 1 output not found")
        return 82

    stage1_output = candidates[0]
    logger.info("STAGE2_START", "Verifying stage1 output", path=stage1_output)

    # Basic checks
    if not os.path.exists(stage1_output):
        logger.error("STAGE2_NOT_FOUND", "Stage 1 output missing")
        return 86

    size = os.path.getsize(stage1_output)
    if size == 0:
        logger.error("STAGE2_EMPTY", "Stage 1 output is empty")
        return 86

    # Try to open with PIL
    try:
        from PIL import Image
        with Image.open(stage1_output) as img:
            w, h = img.size
            mode = img.mode
            logger.info("STAGE2_VERIFY", f"Image OK: {w}x{h} {mode}")
    except Exception as e:
        logger.error("STAGE2_CORRUPT", f"Cannot open image: {e}")
        return 86

    logger.info("STAGE2_DONE", "Verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
