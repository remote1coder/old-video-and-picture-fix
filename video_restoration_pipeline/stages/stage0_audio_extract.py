#!/usr/bin/env python3
"""
Stage 0: Audio Extraction
Extracts original audio stream from video without any processing.
Audio is stored temporarily for stage7 to process and mux.

Exit codes: 0=success, 3=skipped, 82=IO error
"""

import sys
import json
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from audio_utils import extract_audio  # noqa: E402
from checkpoint import wsl_redirect_path  # noqa: E402


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

    # FIX-P7-4: WSL2 temp_dir redirect per v7.32 §3.10.10
    # ORIGINAL: no WSL2 redirect, could write to slow /mnt/ paths
    no_wsl_redirect = getattr(args, "no_wsl_redirect", False) or params.get("no_wsl_redirect", False)
    platform = params.get("platform", "auto")
    try:
        redirected_temp, _, _, _ = wsl_redirect_path(
            temp_dir, no_wsl_redirect=no_wsl_redirect, target_platform=platform
        )
        if redirected_temp != temp_dir:
            logger.info("STAGE0_WSL_REDIRECT",
                        f"WSL redirect: {temp_dir} -> {redirected_temp}")
            temp_dir = redirected_temp
    except Exception:
        pass  # wsl_redirect_path not available or not in WSL

    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    input_name = Path(args.input).stem
    # V-P-018 fix: use consistent extension with stage7 lookup
    # FIX-P7-9: Use pathlib.Path per CLAUDE.md §4
    # ORIGINAL: output_audio = os.path.join(temp_dir, f"...") / os.path.exists()
    # FIX-R7-S0-OBS: Use .mka (Matroska Audio) instead of the non-standard .audio
    # extension. .audio is not a recognized container, so ffmpeg could not
    # infer the muxer and stage7's ffprobe could fail to read channel_layout/
    # duration. .mka accepts any codec with -c:a copy per v7.32 §3.10.9.
    output_audio_obj = Path(temp_dir) / f"{args.job_id}_{input_name}.mka"

    # Check if already exists
    if output_audio_obj.exists() and output_audio_obj.stat().st_size > 0:
        logger.info("STAGE0_SKIP", "Audio already extracted", path=str(output_audio_obj))
        return 3

    logger.info("STAGE0_START", "Extracting audio", input=args.input)

    # FIX-P7-3: Skip extraction if source has no audio stream
    from audio_utils import get_audio_info
    audio_info = get_audio_info(args.input)
    if audio_info.get("channels") is None:
        logger.info("STAGE0_NO_AUDIO", "Source has no audio stream, skipping extraction")
        return 0

    output_audio = str(output_audio_obj)
    success = extract_audio(args.input, output_audio, codec="copy")

    if success:
        logger.info("STAGE0_DONE", "Audio extracted", output=output_audio)
        return 0
    else:
        logger.error("STAGE0_FAILED", "Audio extraction failed")
        return 82


if __name__ == "__main__":
    sys.exit(main() or 0)
