#!/usr/bin/env python3
"""
Stage 6: Photo Final Encoding
- Grain synthesis (G'MIC film_grain)
- AVIF encoding with avifenc (10bit YUV420 or 16bit RGB444)
- CICP metadata
"""

import os
import sys
import json
import argparse
import glob
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_utils"))

from logger import get_logger  # noqa: E402


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


def run_avifenc(input_path: str, output_path: str, config: dict) -> bool:
    """Run avifenc with proper parameters."""
    avif_params = config.get("avif_params") or {}
    depth = avif_params.get("depth", 10)
    yuv = avif_params.get("yuv", "420")
    rng = avif_params.get("range", "limited")
    cicp = avif_params.get("cicp", "1/13/1")
    speed = avif_params.get("speed", 4)
    jobs = avif_params.get("jobs", "all")

    cmd = [
        "avifenc",
        "--depth", str(depth),
        "--yuv", yuv,
        "--range", rng,
        "--cicp", cicp,
        "--speed", str(speed),
    ]

    # jobs handling: version-aware
    cmd.extend(["--jobs", _resolve_jobs_value(jobs)])

    # For RGB444 lossless archival
    if yuv == "444" and depth == 16:
        cmd.append("--lossless")
        # Use RGB CICP (matrix=0 Identity)
        cicp_rgb = avif_params.get("cicp_rgb444", "1/13/0")
        # Replace cicp parameter
        for i, arg in enumerate(cmd):
            if arg == "--cicp":
                cmd[i + 1] = cicp_rgb
                break

    cmd.extend([input_path, output_path])

    try:
        r = subprocess.run(cmd, capture_output=True, timeout=600, shell=False)
        return r.returncode == 0
    except Exception:
        return False


def add_grain_gmic(input_path: str, output_path: str,
                   strength: float, size: float, roughness: float) -> bool:
    """Add film grain using G'MIC."""
    try:
        cmd = [
            "gmic", input_path,
            "-film_grain", str(int(strength * 100)),
            str(size), str(roughness),
            "-o", output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=300, shell=False)
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

    # Find input: stage5 > stage4 > stage3 > stage1
    input_source = None
    for pat in ["step5_face*", "step4_postprocess*", "step3_upscale*", "step1_preprocess*"]:
        pattern = f"{job_id}_{Path(input_name).stem}_{pat}"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        if candidates:
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

    logger.info("STAGE6_START", "Final photo encoding",
                input=input_source, output=output_path)

    # Grain synthesis
    grain_enabled = params.get("grain_synthesis", True)
    working_path = input_source

    if grain_enabled:
        grain_strength = params.get("grain_strength", 0.3)
        grain_size = params.get("grain_size", 1.0)
        grain_roughness = params.get("grain_roughness", 0.5)

        grain_out = os.path.join(temp_dir,
            f"{job_id}_{Path(input_name).stem}_grain.png")

        logger.info("STAGE6_GRAIN", "Adding film grain")
        if add_grain_gmic(working_path, grain_out,
                         grain_strength, grain_size, grain_roughness):
            working_path = grain_out

    # AVIF encoding
    photo_config = {
        "avif_params": params.get("avif_params", {
            "depth": 10, "yuv": "420", "range": "limited",
            "cicp": "1/13/1", "speed": 4, "jobs": "all"
        }),
    }

    logger.info("STAGE6_ENCODE", "Encoding AVIF")
    if run_avifenc(working_path, output_path, photo_config):
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
