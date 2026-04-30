#!/usr/bin/env python3
"""
Stage 2: Traditional Repair Result Verification
Validates stage1 output for decodeability, frame integrity, and quality metrics.

Exit codes: 0=success, 86=verify failed
"""

import os
import sys
import json
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_utils"))

from logger import get_logger  # noqa: E402
from checkpoint import verify_ffprobe_decodeable, verify_frame_integrity  # noqa: E402


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

    # Find the stage1 intermediate
    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    input_name = Path(args.input).name

    # Look for stage1 output
    pattern = f"{job_id}_{Path(input_name).stem}_step1_preprocess*"
    import glob
    candidates = glob.glob(os.path.join(temp_dir, pattern))

    if not candidates:
        logger.error("STAGE2_NO_INPUT", "Stage 1 output not found")
        return 82

    stage1_output = candidates[0]
    logger.info("STAGE2_START", "Verifying stage1 output", path=stage1_output)

    # ffprobe decodeable check
    if not verify_ffprobe_decodeable(stage1_output):
        logger.error("STAGE2_DECODE_FAIL", "Stage 1 output not decodeable")
        return 86

    # Frame integrity check
    duration = params.get("duration", 300)
    vs_bit_depth = params.get("vs_bit_depth", 16)

    if not verify_frame_integrity(stage1_output, duration=duration,
                                   bit_depth=vs_bit_depth):
        logger.error("STAGE2_INTEGRITY_FAIL", "Frame integrity check failed")
        return 86

    logger.info("STAGE2_DONE", "Stage 1 output verified OK")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
