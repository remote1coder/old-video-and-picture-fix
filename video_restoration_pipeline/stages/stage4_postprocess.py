#!/usr/bin/env python3
"""
Stage 4: Post-processing after AI Upscale
Anti-aliasing and final video processing.

v6.3 Compliance:
- Uses standard FFmpeg filters (gblur/unsharp)
- Output format: FFV1 YUV420P16 (per v6.3 §3.6)
- Deterministic intermediate matching
"""

import os
import sys
import json
import hashlib
import argparse
import glob
import subprocess
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

    # Find stage3 output - deterministic by mtime
    pattern = f"{job_id}_{Path(input_name).stem}_step3_upscale*"
    candidates = glob.glob(os.path.join(temp_dir, pattern))
    # P8-11 fallback A: if no stage3 output, look for stage2 output
    if not candidates:
        pattern2 = f"{job_id}_{Path(input_name).stem}_step2_traditional*"
        candidates2 = glob.glob(os.path.join(temp_dir, pattern2))
        if candidates2:
            candidates2.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            stage3_output = candidates2[0]
            logger.info("STAGE4_FALLBACK", "No stage3 output, using stage2 output",
                        input=stage3_output)
        else:
            logger.info("STAGE4_SKIP", "No AI upscale or traditional output")
            return 3
    else:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        stage3_output = candidates[0]

    # P8-7: Compute param_hash for intermediate filename
    try:
        from checkpoint import compute_param_hash
    except Exception:
        import hashlib
        def compute_param_hash(input_path, params):
            param_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
            file_id = str(os.path.getsize(input_path)) + str(int(os.path.getmtime(input_path)))
            return hashlib.sha256((param_str + file_id).encode()).hexdigest()[:16]

    core_params = {
        "color_lift_r": params.get("color_lift_r", 0.02),
        "color_lift_g": params.get("color_lift_g", 0.01),
        "color_lift_b": params.get("color_lift_b", 0.01),
        "color_gamma_r": params.get("color_gamma_r", 0.98),
        "color_gamma_g": params.get("color_gamma_g", 0.98),
        "color_gamma_b": params.get("color_gamma_b", 0.98),
        "color_gain_r": params.get("color_gain_r", 1.02),
        "color_gain_g": params.get("color_gain_g", 1.02),
        "color_gain_b": params.get("color_gain_b", 1.02),
        "vs_bit_depth": params.get("vs_bit_depth", 16),
    }
    param_hash = compute_param_hash(stage3_output, core_params)

    output_path = os.path.join(temp_dir,
        f"{job_id}_{Path(input_name).stem}_step4_postprocess_p{param_hash}.mkv")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info("STAGE4_SKIP", "Post-process already done")
        return 3

    logger.info("STAGE4_START", "Post-processing", input=stage3_output)

    # ISSUE-V-S4-001 / ISSUE-V-S4-002 fix:
    # Selective local processing with gradient-variance region split +
    # feathered transition + final lift/gamma/gain color grading.
    # Per v6.32 / v7.32: flat regions -> mild gblur; textured -> unsharp;
    # feathered mask >=8px; colorbalance for lift/gamma/gain.
    lift_r = params.get("color_lift_r", 0.02)
    lift_g = params.get("color_lift_g", 0.01)
    lift_b = params.get("color_lift_b", 0.01)
    gamma_r = params.get("color_gamma_r", 0.98)
    gamma_g = params.get("color_gamma_g", 0.98)
    gamma_b = params.get("color_gamma_b", 0.98)
    gain_r = params.get("color_gain_r", 1.02)
    gain_g = params.get("color_gain_g", 1.02)
    gain_b = params.get("color_gain_b", 1.02)

    filter_chain = (
        "split[main][tmp];"
        # Edge-based region mask: high edge response -> textured (white in mask)
        "[tmp]edgedetect=mode=colormix:low=0.05:high=0.15,"
        "format=gray,"
        "geq=lum='if(lt(lum(X,Y),30),0,if(gt(lum(X,Y),90),255,255*(lum(X,Y)-30)/60))'[maskraw];"
        # Feathered mask via gaussian blur (sigma=6 gives ~12px transition, >=8px per spec)
        "[maskraw]gblur=sigma=6[mask];"
        # Flat branch: mild denoise/blur for smooth regions
        "[main]gblur=sigma=0.6[flat];"
        # Detail branch: mild sharpening for textured regions
        "[main]unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.4[detail];"
        # Masked merge with feathered transition
        "[flat][detail][mask]maskedmerge,"
        # Final color grading: lift/gamma/gain via colorbalance
        f"colorbalance=rs={lift_r}:gs={lift_g}:bs={lift_b}:"
        f"rm={gamma_r}:gm={gamma_g}:bm={gamma_b}:"
        f"rh={gain_r}:gh={gain_g}:bh={gain_b},"
        # Saturation/contrast fine tune
        "eq=saturation=1.05:contrast=1.02"
    )

    # v6.3 §3.6 / P8-8: Intermediate format must be FFV1 YUV420P16
    vs_bit_depth = params.get("vs_bit_depth", 16)
    if vs_bit_depth != 16:
        logger.warning("STAGE4_BITDEPTH_FORCE",
                       f"vs_bit_depth={vs_bit_depth}, forcing 16bit for intermediate")
    pix_fmt = "yuv420p16le"

    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", stage3_output,
        "-vf", filter_chain,
        "-c:v", "ffv1",
        "-pix_fmt", pix_fmt,
        "-an",
        output_path
    ]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=3600, shell=False)
        if r.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info("STAGE4_DONE", "Post-processing complete", output=output_path)
            return 0
        else:
            logger.error("STAGE4_FAILED", f"Post-processing failed: {r.stderr[:300]}")
            return 84
    except subprocess.TimeoutExpired:
        logger.error("STAGE4_TIMEOUT", "Post-processing timed out")
        return 87
    except Exception as e:
        logger.error("STAGE4_ERROR", str(e))
        return 82


if __name__ == "__main__":
    sys.exit(main() or 0)
