#!/usr/bin/env python3
"""
Stage 6: Photo Final Encoding
- AVIF encoding with avifenc (10bit YUV420 or 16bit RGB444)
- CICP metadata
- Archival mode: RGB444 lossless when archival_mode=true

v6.3/v7.3 Compliance:
- Supports both avif_params (YUV420 10-bit) and avif_rgb444_params (RGB444 16-bit lossless)
- Grain synthesis moved to stage4_postprocess.py per v7.3 spec
- Proper CICP tagging
"""

import os
import sys
import json
import argparse
import glob
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from checkpoint import wsl_redirect_path, wsl_copy_back  # noqa: E402


def _get_avifenc_version() -> tuple:
    """Parse avifenc version. Returns (major, minor, patch) or (0, 0, 0)."""
    try:
        r = subprocess.run(["avifenc", "--version"],
                           capture_output=True, text=True, timeout=10, shell=False)
        if r.returncode == 0:
            import re
            m = re.search(r'(\d+)\.(\d+)\.(\d+)', r.stdout)
            if m:
                return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        pass
    return (0, 0, 0)


def _resolve_jobs_value(jobs: str) -> str:
    """Resolve --jobs value based on avifenc version.

    libavif <1.1.0 does not support 'all' keyword; use CPU count instead.
    """
    if jobs != "all":
        return str(jobs)
    version = _get_avifenc_version()
    if version >= (1, 1, 0):
        return "all"
    return str(min(os.cpu_count() or 1, 32))


def _build_avifenc_cmd(input_path: str, output_path: str, params: dict, use_rgb444: bool = False) -> list:
    """Build avifenc command from config parameters.

    When use_rgb444=True, uses avif_rgb444_params (16-bit lossless).
    Otherwise uses avif_params (10-bit YUV420).
    """
    if use_rgb444:
        avif_params = params.get("avif_rgb444_params", {})
        # Defaults for archival RGB444
        depth = avif_params.get("depth", 16)
        yuv = avif_params.get("yuv", "444")
        rng = avif_params.get("range", "limited")
        cicp = avif_params.get("cicp", "1/13/0")
        speed = avif_params.get("speed", 4)
        jobs = avif_params.get("jobs", "all")
        lossless = avif_params.get("lossless", True)
    else:
        avif_params = params.get("avif_params", {})
        depth = avif_params.get("depth", 10)
        yuv = avif_params.get("yuv", "420")
        rng = avif_params.get("range", "limited")
        cicp = avif_params.get("cicp", "1/13/1")
        speed = avif_params.get("speed", 4)
        jobs = avif_params.get("jobs", "all")
        lossless = False

    cmd = [
        "avifenc",
        "--depth", str(depth),
        "--yuv", yuv,
        "--range", rng,
        "--cicp", cicp,
        "--speed", str(speed),
    ]

    if lossless:
        cmd.append("--lossless")

    cmd.extend(["--jobs", _resolve_jobs_value(jobs)])
    cmd.extend([input_path, output_path])

    return cmd


def run_avifenc(input_path: str, output_path: str, params: dict, use_rgb444: bool = False) -> bool:
    """Run avifenc with proper parameters."""
    cmd = _build_avifenc_cmd(input_path, output_path, params, use_rgb444)
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=600, shell=False)
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

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    input_name = Path(args.input).name

    # Find input: stage5 > stage4 > stage3 > stage1 - sort by mtime
    input_source = None
    for pat in ["step5_face*", "step4_postprocess*", "step3_upscale*", "step1_preprocess*"]:
        pattern = f"{job_id}_{Path(input_name).stem}_{pat}"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            input_source = candidates[0]
            break

    if not input_source:
        # Use original input
        input_source = args.input

    output_path = params.get("output_path", "")
    if not output_path:
        output_dir = params.get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir,
            f"{Path(input_name).stem}_restored.avif")

    # WSL2 redirect
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    platform = params.get("platform", "auto")
    redirected_path, original_output_path = wsl_redirect_path(output_path, no_wsl_redirect, platform)
    if original_output_path:
        logger.info("STAGE6_WSL_REDIRECT",
                   f"WSL redirect: {output_path} -> {redirected_path}")
        output_path = redirected_path

    logger.info("STAGE6_START", "Final photo encoding",
                input=input_source, output=output_path)

    working_path = input_source

    # Determine archival mode
    archival_mode = params.get("archival_mode", False)
    use_rgb444 = archival_mode

    # AVIF encoding
    logger.info("STAGE6_ENCODE", f"Encoding AVIF (archival={use_rgb444})")
    if run_avifenc(working_path, output_path, params, use_rgb444=use_rgb444):
        # WSL redirect: copy back to original Linux path
        if original_output_path:
            if wsl_copy_back(output_path, original_output_path):
                logger.info("STAGE6_WSL_COPYBACK",
                           f"Copied result back to {original_output_path}")
                output_path = original_output_path
            else:
                logger.error("STAGE6_WSL_COPYBACK_FAIL",
                            f"Failed to copy {output_path} back to {original_output_path}")
                return 82
        logger.info("STAGE6_DONE", "Photo encoding complete", output=output_path)
        return 0
    else:
        # Fallback to PNG if avifenc fails
        png_path = output_path.rsplit(".", 1)[0] + ".png"
        try:
            import shutil
            shutil.copy2(working_path, png_path)
            logger.warning("STAGE6_FALLBACK", f"avifenc failed, saved PNG: {png_path}")
            return 0
        except Exception as e:
            logger.error("STAGE6_FAILED", f"Encoding failed: {e}")
            return 84


if __name__ == "__main__":
    sys.exit(main() or 0)
