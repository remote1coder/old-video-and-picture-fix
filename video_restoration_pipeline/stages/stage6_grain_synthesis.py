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
import json
import subprocess
from pathlib import Path
import glob

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from checkpoint import wsl_redirect_path, select_media_candidates  # FIX-P9-3 / R3-01


def _build_vs_script(input_source: str, vs_format: str,
                     g1str: float, g2str: float, g3str: float,
                     addgrainc_ratio: float,
                     use_addgrainc: bool = False,
                     addgrainc_after_gf3: bool = False,
                     temp_avg: float = 0.0) -> str:
    """Build VapourSynth grain synthesis script.

    Module-level function for multiprocessing spawn compliance (CLAUDE.md §5).

    FIX-P9-4: per v7.32 §3.10.4 / v6.32 §7.17, AddGrainC has two modes:
    - use_addgrainc=True, addgrainc_after_gf3=False: standalone fallback
      (GrainFactory3 completely unavailable)
    - addgrainc_after_gf3=True: supplement mode, stacks AFTER GrainFactory3
      to enhance temporal continuity. Intensity ≤20% of g1str. Disabled by default.
    """
    if use_addgrainc and not addgrainc_after_gf3:
        # Standalone AddGrainC fallback (GrainFactory3 completely unavailable)
        var = max(0.0, g1str * addgrainc_ratio)
        uvar = max(0.0, g2str * addgrainc_ratio)
        vvar = max(0.0, g3str * addgrainc_ratio)
        return (
            f"import vapoursynth as vs\n"
            f"core = vs.core\n"
            f"clip = core.ffms2.Source(source={repr(input_source)})\n"
            f"clip = core.resize.Bicubic(clip, format={vs_format})\n"
            f"clip = core.grain.Add(clip, var={var}, uvar={uvar}, "
            f"vvar={vvar}, constant=False)\n"
            f"clip.set_output()\n"
        )

    # GrainFactory3 main path
    script = (
        f"import vapoursynth as vs\n"
        f"core = vs.core\n"
        f"import havsfunc as haf\n"
        f"clip = core.ffms2.Source(source={repr(input_source)})\n"
        f"clip = core.resize.Bicubic(clip, format={vs_format})\n"
        f"clip = haf.GrainFactory3(clip, g1str={g1str}, "
        f"g2str={g2str}, g3str={g3str}, temp_avg={temp_avg})\n"
    )
    if addgrainc_after_gf3:
        # FIX-P9-4: Stack AddGrainC after GrainFactory3 for temporal continuity
        # per v7.32 §3.10.4: intensity ≤20% of g1str, disabled by default
        var = max(0.0, g1str * addgrainc_ratio)
        uvar = max(0.0, g2str * addgrainc_ratio)
        vvar = max(0.0, g3str * addgrainc_ratio)
        script += (
            f"clip = core.grain.Add(clip, var={var}, uvar={uvar}, "
            f"vvar={vvar}, constant=False)\n"
        )
    script += "clip.set_output()\n"
    return script


def _run_grain_pipeline(vs_script_str: str, vs_path: str,
                        vspipe_cmd: list, ffmpeg_cmd: list,
                        output_path: str, logger,
                        fallback_label: str = "") -> tuple:
    """Run grain synthesis pipeline (vspipe → ffmpeg FFV1).

    Module-level function for multiprocessing spawn compliance (CLAUDE.md §5).
    Returns (status_code_or_str, detail_str) per BUG-036.
    """
    with open(vs_path, "w", encoding="utf-8") as f:
        f.write(vs_script_str)
    try:
        vspipe_proc = subprocess.Popen(
            vspipe_cmd, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, shell=False)
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=vspipe_proc.stdout,
            stderr=subprocess.PIPE, shell=False)
        vspipe_proc.stdout.close()
        stdout, stderr = ffmpeg_proc.communicate(timeout=3600)
        vspipe_stderr = (vspipe_proc.stderr.read()
                         if vspipe_proc.stderr else b"")
        vspipe_proc.wait()
        if ffmpeg_proc.returncode == 0:
            if fallback_label:
                logger.info("STAGE6_DONE",
                            f"Grain synthesis complete ({fallback_label})",
                            output=output_path)
            else:
                logger.info("STAGE6_DONE", "Grain synthesis complete",
                            output=output_path)
            return (0, "")
        err = stderr.decode("utf-8", errors="replace") if stderr else ""
        v_err_bytes = vspipe_stderr
        v_err_str = (v_err_bytes.decode("utf-8", errors="replace")
                     if isinstance(v_err_bytes, bytes) else str(v_err_bytes))
        return (err, v_err_str)
    except subprocess.TimeoutExpired:
        vspipe_proc.kill()
        ffmpeg_proc.kill()
        return ("TIMEOUT", "")
    except Exception as e:
        return (str(e), "")


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
                fps = (float(fps_parts[0]) / float(fps_parts[1])
                       if len(fps_parts) == 2 else float(parts[2]))
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

    # FIX-P9-3: WSL2 temp_dir redirect per v7.32 §3.10.10 / v6.32 §7.3
    temp_dir_raw = params.get("temp_dir", "./temp")
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    _, _, temp_dir, _ = wsl_redirect_path(
        output_path="", temp_dir=temp_dir_raw,
        no_wsl_redirect=no_wsl_redirect, logger=logger)
    job_id = args.job_id
    # Use args.input stem for pattern matching (consistent with other stages)
    input_name = Path(args.input).stem

    # FIX-W9-09: log ai_upscale_banned flag per checkproject rule F. Grain
    # synthesis is not AI so the flag doesn't change grain behavior; logged
    # for traceability. When banned, F9-02 fallback ensures stage4 input.
    ai_banned = params.get("ai_upscale_banned", False)
    if ai_banned:
        logger.info("STAGE6_BAN_FLAG",
                    "ai_upscale_banned=true noted (grain synthesis unaffected; "
                    "input sourced from stage4 per F9-02 fallback)")

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
        # FIX-F9-02: face_restore enabled but stage5 may have produced no
        # output (stage5 exits 3 when no static face segments — a common
        # case). Per v7.32 §3.10.24, fall back to stage4 output instead of
        # aborting the whole pipeline with return 82.
        expected_stage = "step5_face"
        pattern5 = f"{job_id}_{input_name}_step5_face*"
        # R3-01：只接受媒体产物，排除同前缀侧车
        cands = select_media_candidates(glob.glob(os.path.join(temp_dir, pattern5)))
        if cands:
            input_source = str(cands[0])
        if not input_source:
            # Fallback to stage4 output
            pattern4 = f"{job_id}_{input_name}_step4_postprocess*"
            cands4 = select_media_candidates(
                glob.glob(os.path.join(temp_dir, pattern4)))
            if cands4:
                input_source = str(cands4[0])
                expected_stage = "step4_postprocess(fallback)"
                logger.warning("STAGE6_FALLBACK_STEP4",
                               "face_restore enabled but no stage5 output "
                               "(stage5 likely skipped — no static face segments), "
                               "falling back to stage4 output per v7.32 §3.10.24")
            else:
                logger.error("STAGE6_INPUT_ERROR",
                            f"Neither stage5 nor stage4 output found. "
                            f"Patterns: {pattern5} / {pattern4}",
                            temp_dir=temp_dir)
                return 82
    else:
        # Face restore disabled: MUST use stage4 output
        expected_stage = "step4_postprocess"
        pattern = f"{job_id}_{input_name}_step4_postprocess*"
        # R3-01：只接受媒体产物，排除同前缀侧车
        cands = select_media_candidates(glob.glob(os.path.join(temp_dir, pattern)))
        if cands:
            input_source = str(cands[0])
        if not input_source:
            logger.error("STAGE6_INPUT_ERROR",
                        f"No stage4 output found. Expected pattern: {pattern}",
                        temp_dir=temp_dir)
            return 82

    if not input_source:
        logger.error("STAGE6_NO_INPUT", "No valid input found",
                     expected_stage=expected_stage)
        return 82

    logger.info("STAGE6_INPUT", f"Using {expected_stage} output",
                input=input_source)

    # FIX-P9-11: input file integrity check via ffprobe per v6.32 §7.9
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "json", input_source],
            capture_output=True, text=True, timeout=30, shell=False)
        if r.returncode != 0:
            logger.error("STAGE6_INPUT_CORRUPT",
                        f"Input file is not a valid video: {input_source}",
                        ffprobe_err=r.stderr[:300] if r.stderr else "")
            return 86
        probe_info = json.loads(r.stdout)
        streams = probe_info.get("streams", [])
        if not streams or "width" not in streams[0]:
            logger.error("STAGE6_INPUT_CORRUPT",
                        f"Input file has no valid video stream: {input_source}")
            return 86
    except subprocess.TimeoutExpired:
        logger.error("STAGE6_INPUT_CORRUPT",
                    f"ffprobe timed out on input: {input_source}")
        return 86
    except Exception as e:
        logger.error("STAGE6_INPUT_CORRUPT",
                    f"Cannot probe input file: {e}")
        return 86

    # FIX-P9-4: output filename must include param_hash per v7.3 §3.10.19
    param_hash = params.get("param_hash", job_id)
    output_path = os.path.join(
        temp_dir,
        f"{job_id}_{input_name}_step6_grain_p{param_hash}.mkv"
    )

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

    # FIX-P9-6: Read addgrainc_enabled and addgrainc_ratio from config
    # Per v7.3 §3.10.24: AddGrainC default disabled, intensity ≤20% of g1str
    addgrainc_enabled = grain_cfg.get("addgrainc_enabled", False)
    addgrainc_ratio = grain_cfg.get("addgrainc_ratio", 0.2)

    # FIX-P9-5: explicit detected_class → grain config key mapping per v7.32 §3.2
    # detected_class values may be uppercase (ANALOG_TAPE, DV_DIGITAL, HDV_AVCHD)
    _DETECTED_CLASS_TO_GRAIN_KEY = {
        "analog_tape": "analog_tape",
        "dv_digital": "dv_digital",
        "hdv_avchd": "hdv_avchd",
    }
    material_type_raw = params.get("detected_class", "analog_tape").lower()
    material_key = _DETECTED_CLASS_TO_GRAIN_KEY.get(material_type_raw, "analog_tape")
    material_grain = grain_cfg.get(material_key, {})
    if material_key != material_type_raw:
        logger.warning("STAGE6_GRAIN_MAPPING",
                      f"Unknown material type '{material_type_raw}', "
                      f"defaulting to '{material_key}'")

    g1str = material_grain.get("g1str", 1.0)
    g2str = material_grain.get("g2str", 0.2)
    g3str = material_grain.get("g3str", 0.2)
    # FIX-P1-13: pass temp_avg to GrainFactory3 (r33 legal param, controls
    # temporal correlation; default 0 = disabled, AddGrainC fallback used then).
    temp_avg = grain_cfg.get("temp_avg", material_grain.get("temp_avg", 0.0))

    # Validate no illegal params
    if not grain_cfg:
        logger.error("STAGE6_CONFIG", "grain_factory3 config missing")
        return 81

    logger.info("STAGE6_START", "Grain synthesis", input=input_source,
                g1str=g1str, g2str=g2str, g3str=g3str)

    # Get video dimensions for vspipe
    width, height, fps = get_video_dimensions(input_source)

    # V-P-046 fix: 16bit processing space
    # Generate VapourSynth script
    # v6.3: Only g1str/g2str/g3str are valid GrainFactory3 params.
    vs_bit_depth = params.get("vs_bit_depth", 16)
    if vs_bit_depth == 16:
        vs_format = "vs.YUV420P16"
        pix_fmt = "yuv420p16le"
    elif vs_bit_depth == 10:
        vs_format = "vs.YUV420P10"
        pix_fmt = "yuv420p10le"
    else:
        vs_format = "vs.YUV420P8"
        pix_fmt = "yuv420p"

    vs_path = os.path.join(temp_dir, f"{job_id}_{input_name}_grain.vpy")
    # FIX-P9-4: pass addgrainc_after_gf3 so AddGrainC can stack after GrainFactory3
    vs_script = _build_vs_script(
        input_source, vs_format, g1str, g2str, g3str,
        addgrainc_ratio, use_addgrainc=False,
        addgrainc_after_gf3=addgrainc_enabled,
        temp_avg=temp_avg)
    # FIX-P9-11: removed redundant write here — _run_grain_pipeline writes vs_path internally

    # Fix: Use vspipe directly (standard VapourSynth CLI tool)
    vspipe_cmd = [
        "vspipe", "-c", "y4m",
        vs_path, "-"
    ]

    # V-P-048 fix: lossless FFV1 intermediate instead of lossy SVT-AV1
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", "-",
        "-c:v", "ffv1",
        "-pix_fmt", pix_fmt,
        output_path
    ]

    result = _run_grain_pipeline(
        vs_script, vs_path, vspipe_cmd, ffmpeg_cmd,
        output_path, logger)
    if result[0] == 0:
        return 0

    # If GrainFactory3 failed, try AddGrainC fallback (if enabled)
    # FIX-P9-6: respect addgrainc_enabled config; if disabled, fail immediately
    # FIX-P9-9: AddGrainC standalone fallback uses halved intensity
    # per v6.32 §7.17: conservative fallback to avoid over-graining
    err_str, v_err_str = result
    if ("GrainFactory3" in err_str or "GrainFactory3" in v_err_str
            or "haf" in err_str or "havsfunc" in err_str):
        if not addgrainc_enabled:
            logger.critical("STAGE6_GF3_FAILED",
                           "GrainFactory3 failed. AddGrainC is disabled by config. "
                           "GrainFactory3 produces multi-scale structured grain; "
                           "AddGrainC fallback would produce lower quality grain.",
                           err=err_str[:500])
            return 84
        logger.warning("STAGE6_FALLBACK",
                       "GrainFactory3 failed, trying AddGrainC fallback "
                       "(halved intensity, lower quality)",
                       grainfactory3_err=err_str[:500])
        # FIX-P9-9: halved intensity for standalone fallback per v6.32 §7.17
        addgrain_script = _build_vs_script(
            input_source, vs_format, g1str * 0.5, g2str * 0.5, g3str * 0.5,
            addgrainc_ratio, use_addgrainc=True,
            temp_avg=temp_avg)
        result2 = _run_grain_pipeline(
            addgrain_script, vs_path, vspipe_cmd, ffmpeg_cmd,
            output_path, logger, fallback_label="AddGrainC")
        if result2[0] == 0:
            return 0
        err_str2 = result2[0] if isinstance(result2[0], str) else str(result2[0])
        logger.error("STAGE6_FAIL",
                    "Grain synthesis failed (AddGrainC fallback also failed)",
                    addgrainc_err=err_str2[:500])
        return 84
    else:
        logger.error("STAGE6_FAIL", "Grain synthesis failed",
                     ffmpeg_err=err_str[:500],
                     vspipe_err=v_err_str[:500])
        return 84


if __name__ == "__main__":
    sys.exit(main())
