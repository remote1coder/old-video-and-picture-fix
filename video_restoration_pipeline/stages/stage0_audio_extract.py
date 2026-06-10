#!/usr/bin/env python3
"""
Stage 0: Audio Extraction
Extracts original audio stream from video without any processing.
Audio is stored temporarily for stage7 to process and mux.

Exit codes: 0=success, 3=skipped, 82=IO error
"""

import os
import sys
import json
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from audio_utils import extract_audio  # noqa: E402


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
    parser.add_argument("--no-wsl-redirect", action="store_true")
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()

    params = _load_params(args)
    log_dir = params.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, job_id=args.job_id)

    # Determine output path
    temp_dir = params.get("temp_dir", "./temp")
    os.makedirs(temp_dir, exist_ok=True)

    input_name = Path(args.input).stem
    # V-P-018 fix: use consistent extension with stage7 lookup
    output_audio = os.path.join(temp_dir, f"{args.job_id}_{input_name}.audio")

    # Check if already exists
    if os.path.exists(output_audio) and os.path.getsize(output_audio) > 0:
        logger.info("STAGE0_SKIP", "Audio already extracted", path=output_audio)
        return 3

    logger.info("STAGE0_START", "Extracting audio", input=args.input)

    # FIX-P7-3: Skip extraction if source has no audio stream
    from audio_utils import get_audio_info
    audio_info = get_audio_info(args.input)
    if audio_info.get("channels") is None:
        logger.info("STAGE0_NO_AUDIO", "Source has no audio stream, skipping extraction")
        return 0

    success = extract_audio(args.input, output_audio, codec="copy")

    if success:
        logger.info("STAGE0_DONE", "Audio extracted", output=output_audio)
        return 0
    else:
        logger.error("STAGE0_FAILED", "Audio extraction failed")
        return 82


if __name__ == "__main__":
    sys.exit(main() or 0)
