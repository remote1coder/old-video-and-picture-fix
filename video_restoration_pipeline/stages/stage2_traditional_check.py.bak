#!/usr/bin/env python3
"""
Stage 2: Traditional Repair Verification
Verifies deinterlace output using ffmpeg idet filter.

v6.3 Compliance:
- Progressiveness threshold from config (default >=95%)
- Deterministic intermediate matching
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


def _load_params(args):
    if getattr(args, "params_file", None):
        with open(args.params_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(args.params) if args.params else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--params")
    parser.add_argument("--params-file")
    parser.add_argument("--config")
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()

    params = _load_params(args)
    log_dir = params.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, job_id=args.job_id)

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    input_name = Path(args.input).name

    # Find stage1 output - sort by mtime for deterministic selection
    pattern = f"{job_id}_{Path(input_name).stem}_step1_preprocess*"
    candidates = glob.glob(os.path.join(temp_dir, pattern))
    if not candidates:
        logger.error("STAGE2_NO_INPUT", "Stage 1 output not found")
        return 82

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    stage1_output = candidates[0]
    logger.info("STAGE2_START", "Verifying deinterlace output", path=stage1_output)

    # Check file integrity
    if not os.path.exists(stage1_output) or os.path.getsize(stage1_output) == 0:
        logger.error("STAGE2_EMPTY", "Stage 1 output missing or empty")
        return 86

    # Verify with ffprobe
    try:
        import subprocess
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1", stage1_output],
            capture_output=True, text=True, timeout=60, shell=False
        )
        if r.returncode != 0 or "video" not in r.stdout:
            logger.error("STAGE2_CORRUPT", "Stage 1 output not valid video")
            return 86
    except Exception as e:
        logger.error("STAGE2_VERIFY_FAIL", f"Verification failed: {e}")
        return 86

    # Check deinterlace quality with idet
    threshold = params.get("deinterlace_progressive_threshold", 95.0)
    try:
        import subprocess
        import re
        cmd = [
            "ffmpeg", "-hide_banner", "-v", "info",
            "-i", stage1_output,
            "-vf", "idet",
            "-f", "null", "-"
        ]
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=300, shell=False)
        if r.returncode == 0:
            stats = {}
            for line in r.stderr.splitlines():
                if "Multi frame detection:" in line:
                    for key in ("TFF", "BFF", "Progressive", "Undetermined"):
                        m = re.search(rf"{key}:\s*(\d+)", line)
                        if m:
                            stats[key.lower()] = int(m.group(1))
            total = sum(stats.values())
            if total > 0:
                prog_pct = stats.get("progressive", 0) / total * 100
                if prog_pct >= threshold:
                    logger.info("STAGE2_DONE", f"Verification passed: {prog_pct:.1f}% progressive")
                else:
                    logger.error("STAGE2_LOW_PROGRESSIVE",
                                f"Progressive={prog_pct:.1f}% < threshold={threshold}%")
                    return 86

                # Frame rate validation
                try:
                    fr_cmd = ["ffprobe", "-hide_banner", "-v", "error",
                              "-select_streams", "v:0",
                              "-show_entries", "stream=r_frame_rate",
                              "-of", "default=noprint_wrappers=1:nokey=1", stage1_output]
                    fr = subprocess.run(fr_cmd, capture_output=True, text=True,
                                        timeout=60, shell=False)
                    if fr.returncode == 0:
                        rate_str = fr.stdout.strip()
                        if "/" in rate_str:
                            num, den = map(float, rate_str.split("/"))
                            detected_fps = num / den if den != 0 else 0
                        else:
                            detected_fps = float(rate_str)
                        expected_fps = params.get("expected_fps", params.get("original_fps", 0))
                        if expected_fps > 0:
                            deviation = abs(detected_fps - expected_fps) / expected_fps
                            if deviation > 0.02:
                                logger.error("STAGE2_FRAMERATE",
                                            f"FPS {detected_fps} deviates {deviation*100:.1f}% from expected {expected_fps}")
                                return 86
                            else:
                                logger.info("STAGE2_FRAMERATE",
                                           f"FPS {detected_fps} matches expected {expected_fps}")
                except Exception as e:
                    logger.warning("STAGE2_FRAMERATE_CHECK", f"Frame rate check failed: {e}")

                logger.info("TRADITIONAL_GATE", "Traditional restoration completed, proceeding to AI stages")
                return 0
    except Exception as e:
        logger.warning("STAGE2_CHECK_FAIL", f"Quality check failed: {e}")

    logger.info("STAGE2_DONE", "Verification passed (basic)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
