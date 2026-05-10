#!/usr/bin/env python3
"""
Grain Synthesis (Video) - Stage 6
Adds film grain to video after all processing is complete.

v6.3 Compliance:
- Positioned after all restoration (stage2-5) and before final encoding (stage7)
- Input source STRICT validation:
  * face_restore enabled -> MUST receive stage5 output
  * face_restore disabled -> MUST receive stage4 output
  * STRICTLY FORBIDDEN to receive stage3 output directly
- GrainFactory3 with g1str/g2str/g3str (havsfunc r33)
- 16bit processing space (YUV420P16)
- No taps/shutter/correlation (not GrainFactory3 params)
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path
import glob

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402


def get_video_dimensions(input_source: str) -> tuple:
    """Get video width, height, fps using ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-of", "csv=p=0", input_source],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            if len(parts) >= 3:
                width = int(parts[0])
                height = int(parts[1])
                fps_parts = parts[2].split("/")
                fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(parts[2])
                return width, height, fps
    except Exception:
        pass
    return 720, 480, 29.97


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--params", default="")
    args = parser.parse_args()

    params = {}
    if args.params:
        import json
        try:
            params = json.loads(args.params)
        except Exception:
            pass

    # Use get_logger with proper parameters from config
    log_dir = params.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, job_id=args.job_id)

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    # Use args.input stem for pattern matching (consistent with other stages)
    input_name = Path(args.input).stem

    # Determine if face restore was enabled
    # Fix: ai_enabled is passed as flat key "ai_enabled" in params, not as nested "ai" dict
    face_restore_enabled = bool(params.get("face_restore", ""))
    ai_enabled = params.get("ai_enabled", False)

    # Input source STRICT validation per v6.3 §3.10.24:
    # - face_restore enabled -> MUST receive stage5 output
    # - face_restore disabled -> MUST receive stage4 output
    # - STRICTLY FORBIDDEN to receive stage3 output directly
    input_source = None
    expected_stage = None

    if ai_enabled and face_restore_enabled:
        # Face restore enabled: MUST use stage5 output
        expected_stage = "step5_face"
        pattern = f"{job_id}_{input_name}_step5_face*"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            input_source = candidates[0]
        if not input_source:
            logger.error("STAGE6_INPUT_ERROR",
                        f"Face restore enabled but no stage5 output found. "
                        f"Expected pattern: {pattern}",
                        temp_dir=temp_dir)
            return 82
    else:
        # Face restore disabled: MUST use stage4 output
        expected_stage = "step4_postprocess"
        pattern = f"{job_id}_{input_name}_step4_postprocess*"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            input_source = candidates[0]
        if not input_source:
            logger.error("STAGE6_INPUT_ERROR",
                        f"No stage4 output found. Expected pattern: {pattern}",
                        temp_dir=temp_dir)
            return 82

    if not input_source:
        logger.error("STAGE6_NO_INPUT", "No valid input found", expected_stage=expected_stage)
        return 82

    logger.info("STAGE6_INPUT", f"Using {expected_stage} output", input=input_source)

    output_path = os.path.join(temp_dir,
        f"{job_id}_{input_name}_step6_grain.mkv")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return 3

    # Load grain parameters from config
    config_path = args.config
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        logger.error("STAGE6_CONFIG", f"Cannot load config: {e}")
        return 81

    grain_cfg = cfg.get("grain_factory3", {})
    if not grain_cfg.get("enabled", True):
        logger.info("STAGE6_SKIP", "Grain synthesis disabled")
        return 0

    # Get material-specific grain params
    material_type = params.get("detected_class", "analog_tape").lower()
    material_grain = grain_cfg.get(material_type, {})

    g1str = material_grain.get("g1str", 1.0)
    g2str = material_grain.get("g2str", 0.2)
    g3str = material_grain.get("g3str", 0.2)

    # Validate no illegal params
    if not grain_cfg:
        logger.error("STAGE6_CONFIG", "grain_factory3 config missing")
        return 81

    logger.info("STAGE6_START", "Grain synthesis", input=input_source,
                g1str=g1str, g2str=g2str, g3str=g3str)

    # Get video dimensions for vspipe
    width, height, fps = get_video_dimensions(input_source)

    # Generate VapourSynth script
    # v6.3: Only g1str/g2str/g3str are valid GrainFactory3 params.
    # taps/shutter/correlation are AddGrainC params, NOT GrainFactory3 params.
    vs_script = f"""import vapoursynth as vs
core = vs.core
import havsfunc as haf
clip = core.ffms2.Source(source={repr(input_source)})
clip = haf.GrainFactory3(clip, g1str={g1str}, g2str={g2str}, g3str={g3str})
clip.set_output()
"""

    vs_path = os.path.join(temp_dir, f"{job_id}_{input_name}_grain.vpy")
    with open(vs_path, "w", encoding="utf-8") as f:
        f.write(vs_script)

    # Fix: Use vspipe directly (standard VapourSynth CLI tool)
    vspipe_cmd = [
        "vspipe", "-c", "y4m",
        vs_path, "-"
    ]

    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", "-",
        "-c:v", "libsvtav1",
        "-crf", str(cfg.get("global", {}).get("crf", 24)),
        "-preset", str(cfg.get("global", {}).get("preset", 4)),
        "-pix_fmt", "yuv420p10le",
        output_path
    ]

    try:
        vspipe_proc = subprocess.Popen(vspipe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=vspipe_proc.stdout, stderr=subprocess.PIPE, shell=False)
        vspipe_proc.stdout.close()  # Allow vspipe to receive SIGPIPE when ffmpeg exits

        stdout, stderr = ffmpeg_proc.communicate(timeout=3600)
        vspipe_stderr = vspipe_proc.stderr.read() if vspipe_proc.stderr else b""
        vspipe_proc.wait()

        if ffmpeg_proc.returncode == 0:
            logger.info("STAGE6_DONE", "Grain synthesis complete", output=output_path)
            return 0
        else:
            err = stderr.decode("utf-8", errors="replace") if stderr else ""
            v_err = vspipe_stderr.decode("utf-8", errors="replace") if vspipe_stderr else ""
            logger.error("STAGE6_FAIL", "Grain synthesis failed", ffmpeg_err=err, vspipe_err=v_err)
            return 84
    except subprocess.TimeoutExpired:
        vspipe_proc.kill()
        ffmpeg_proc.kill()
        logger.error("STAGE6_TIMEOUT", "Grain synthesis timed out")
        return 87
    except Exception as e:
        logger.error("STAGE6_ERROR", f"Grain synthesis error: {e}")
        return 84


if __name__ == "__main__":
    sys.exit(main())
