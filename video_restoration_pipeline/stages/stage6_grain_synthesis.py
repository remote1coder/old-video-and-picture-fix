#!/usr/bin/env python3
"""
Stage 6: Grain Synthesis (Video)
Applies GrainFactory3 after all processing, before final encode.
Receives input from stage5 (face restore) or stage4 (postprocess).

Parameters from config grain_factory3 section:
- analog_tape: g1str=1.0, g2str=0.2, g3str=0.2
- dv_digital: g1str=1.0, g2str=0.2, g3str=0.2
- hdv_avchd: g1str=0.8, g2str=0.2, g3str=0.2

Exit codes: 0=success, 3=skipped
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_utils"))

from logger import get_logger  # noqa: E402


def generate_grain_vpy(input_video: str, output_path: str,
                       g1str: float, g2str: float, g3str: float,
                       addgrainc_enabled: bool, addgrainc_ratio: float) -> str:
    """Generate VapourSynth script for grain synthesis."""
    lines = []
    lines.append("import vapoursynth as vs")
    lines.append("import havsfunc as haf")
    lines.append("core = vs.core")
    lines.append("")
    lines.append(f"# Load input: {input_video}")
    lines.append(f"clip = core.ffms2.Source(source={repr(input_video)})")
    lines.append("")
    lines.append("# Ensure 16bit YUV")
    lines.append("if clip.format.bits_per_sample != 16:")
    lines.append("    clip = core.resize.Bicubic(clip, format=vs.YUV420P16)")
    lines.append("")
    lines.append("# GrainFactory3 - only g1str/g2str/g3str (no taps/shutter/correlation)")
    lines.append(f"clip = haf.GrainFactory3(clip, g1str={g1str}, g2str={g2str}, g3str={g3str})")
    lines.append("")

    if addgrainc_enabled:
        add_str = g1str * addgrainc_ratio
        lines.append("# AddGrainC supplementary temporal grain")
        lines.append(f"clip = core.grain.Add(clip, var={add_str:.3f})")
        lines.append("")

    # 16bit -> 10bit error diffusion dithering (YUV444 first, then subsample to YUV420)
    lines.append("# 16bit to 10bit error diffusion")
    lines.append("clip = core.resize.Bicubic(clip, format=vs.YUV444P16)")
    lines.append('clip = core.resize.Bicubic(clip, format=vs.YUV444P10, dither_type="error_diffusion")')
    lines.append("clip = core.resize.Bicubic(clip, format=vs.YUV420P10)")
    lines.append("")

    lines.append("clip.set_output()")
    return "\n".join(lines)


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

    grain_enabled = params.get("grain_enabled", True)
    if not grain_enabled:
        logger.info("STAGE6_SKIP", "Grain synthesis disabled")
        return 3

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    input_name = Path(args.input).name

    # Determine input source: stage5 > stage4 > stage1
    import glob
    input_source = None
    for stage_pat in ["step5*", "step4*", "step1*"]:
        pattern = f"{job_id}_{Path(input_name).stem}_{stage_pat}"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        if candidates:
            input_source = candidates[0]
            break

    if not input_source:
        logger.error("STAGE6_NO_INPUT", "No input found from previous stages")
        return 82

    # Get grain parameters based on detected class
    detected_class = params.get("detected_class", "analog_tape")
    gf3_config = params.get("grain_factory3", {})

    if "hdv" in detected_class.lower() or "avchd" in detected_class.lower():
        grain_params = gf3_config.get("hdv_avchd", {"g1str": 0.8, "g2str": 0.2, "g3str": 0.2})
    elif "dv" in detected_class.lower():
        grain_params = gf3_config.get("dv_digital", {"g1str": 1.0, "g2str": 0.2, "g3str": 0.2})
    else:
        grain_params = gf3_config.get("analog_tape", {"g1str": 1.0, "g2str": 0.2, "g3str": 0.2})

    g1str = grain_params.get("g1str", 1.0)
    g2str = grain_params.get("g2str", 0.2)
    g3str = grain_params.get("g3str", 0.2)
    addgrainc_enabled = gf3_config.get("addgrainc_enabled", False)
    addgrainc_ratio = gf3_config.get("addgrainc_ratio", 0.2)

    logger.info("STAGE6_START", "Grain synthesis",
                g1str=g1str, g2str=g2str, g3str=g3str,
                source=input_source)

    # Generate VS script
    vpy_content = generate_grain_vpy(
        input_source,
        os.path.join(temp_dir, f"{job_id}_grain.mkv"),
        g1str, g2str, g3str,
        addgrainc_enabled, addgrainc_ratio
    )

    vpy_path = os.path.join(temp_dir, f"{job_id}_{Path(input_name).stem}_stage6.vpy")
    with open(vpy_path, "w", encoding="utf-8") as f:
        f.write(vpy_content)

    # Render
    output_path = os.path.join(temp_dir,
        f"{job_id}_{Path(input_name).stem}_step6_grain.mkv")

    try:
        vspipe_cmd = ["vspipe", "--y4m", vpy_path, "-"]
        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-v", "error",
            "-thread_queue_size", "1024",
            "-i", "-",
            "-c:v", "ffv1",
            "-an", "-y", output_path
        ]

        vp = subprocess.Popen(vspipe_cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, shell=False)
        ff = subprocess.Popen(ffmpeg_cmd, stdin=vp.stdout,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, shell=False)
        vp.stdout.close()
        _, stderr = ff.communicate(timeout=7200)

        if ff.returncode == 0 and os.path.exists(output_path):
            logger.info("STAGE6_DONE", "Grain synthesis complete", output=output_path)
            return 0
        else:
            logger.error("STAGE6_FAILED", f"Render error: {stderr.decode()[:300]}")
            return 84

    except subprocess.TimeoutExpired:
        logger.error("STAGE6_TIMEOUT", "Grain synthesis timed out")
        return 87
    except Exception as e:
        logger.error("STAGE6_ERROR", str(e))
        return 82


if __name__ == "__main__":
    sys.exit(main() or 0)
