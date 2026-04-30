#!/usr/bin/env python3
"""
Stage 1: Traditional Video Preprocessing
Generates VapourSynth script for:
- IVTC (VIVTC/sRestore)
- Deinterlacing (QTGMC)
- Denoising (TemporalDegrain2)
- Stabilization (DepanStabilize)
- Chroma shift repair
- Deblocking
- BT.601->BT.709 color space conversion
- Black border crop/restore

All processing in 16bit (YUV420P16), output as YUV420P16.
"""

import os
import sys
import json
import argparse
import hashlib
import subprocess
import re
from pathlib import Path
from typing import Tuple, Dict, Optional

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_utils"))

from logger import get_logger  # noqa: E402
from checkpoint import compute_param_hash, build_intermediate_name  # noqa: E402


def detect_black_borders(video_path: str, logger=None) -> dict:
    """Detect black borders using ffmpeg cropdetect."""
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "info",
        "-i", video_path,
        "-vf", "cropdetect=limit=16:round=2:reset=0",
        "-frames:v", "100",  # Analyze first 100 frames
        "-f", "null", "-"
    ]
    crop_info = {"left": 0, "right": 0, "top": 0, "bottom": 0,
                 "cropped_w": 0, "cropped_h": 0}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=120, shell=False)
        # Parse cropdetect output from stderr
        # Example: [Parsed_cropdetect_0 @ 0x...] x1:0 x2:719 y1:8 y2:471 w:720 h:464 x:0 y:8 pts:...
        # Or: crop=720:464:0:8
        crop_match = None
        for line in r.stderr.split("\n"):
            m = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", line)
            if m:
                crop_match = m
        if crop_match:
            cw, ch, cx, cy = map(int, crop_match.groups())
            # Get original dimensions
            w_cmd = [
                "ffprobe", "-hide_banner", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0",
                video_path
            ]
            wr = subprocess.run(w_cmd, capture_output=True, text=True,
                               timeout=30, shell=False)
            if wr.returncode == 0:
                ow, oh = map(int, wr.stdout.strip().split("x"))
                crop_info = {
                    "left": cx,
                    "right": ow - cx - cw,
                    "top": cy,
                    "bottom": oh - cy - ch,
                    "cropped_w": cw,
                    "cropped_h": ch,
                    "orig_w": ow,
                    "orig_h": oh,
                }
    except Exception as e:
        if logger:
            logger.warning("STAGE1_CROP_FAIL", f"cropdetect failed: {e}")
    return crop_info


def validate_crop(crop_info: dict, logger=None) -> bool:
    """Validate crop parameters. Returns True if valid."""
    if not crop_info or crop_info.get("cropped_w", 0) == 0:
        return True  # No crop detected, OK

    ow = crop_info.get("orig_w", 1)
    oh = crop_info.get("orig_h", 1)
    cw = crop_info.get("cropped_w", 0)
    ch = crop_info.get("cropped_h", 0)

    # Check minimum size: 240px in both dimensions
    if cw < 240 or ch < 240:
        if logger:
            logger.error("STAGE1_CROP_TOO_SMALL",
                        f"Cropped size {cw}x{ch} below minimum 240px")
        return False

    # Check crop ratio: must retain at least 50% of original
    orig_area = ow * oh
    crop_area = cw * ch
    if crop_area < orig_area * 0.5:
        if logger:
            logger.error("STAGE1_CROP_TOO_MUCH",
                        f"Cropped area {crop_area} < 50% of original {orig_area}")
        return False

    return True


def generate_vs_script(input_path: str, params: dict, job_id: str,
                       crop_info: Optional[dict] = None) -> str:
    """Generate VapourSynth script content."""
    field_order = params.get("field_order", "tt")
    needs_ivtc = params.get("needs_ivtc", False)
    is_psf = params.get("is_psf", False)

    ivtc = params.get("ivtc", "VIVTC")
    deinterlace = params.get("deinterlace", "QTGMC")
    denoise = params.get("denoise", "TemporalDegrain2")
    denoise_params = params.get("denoise_params", {})
    stabilize = params.get("stabilize", "DepanStabilize")
    stabilize_params = params.get("stabilize_params", {})
    chroma_shift = params.get("chroma_shift", False)
    deblock = params.get("deblock", "")
    target_res = params.get("target_resolution", "720x480")
    vs_bit_depth = params.get("vs_bit_depth", 16)

    # Parse resolution
    if "x" in target_res:
        _, target_h = map(int, target_res.split("x"))
    else:
        _, target_h = 720, 480

    # Determine matrix based on resolution
    if target_h <= 576:
        matrix_in = "470bg"  # PAL BT.601
    else:
        matrix_in = "170m"   # NTSC BT.601

    # VapourSynth format for 16bit
    vs_format = "vs.YUV420P16"

    # Determine range_in from params/config
    color_range = params.get("color_range", "limited")
    range_in_str = "vs.RANGE_FULL" if color_range == "full" else "vs.RANGE_LIMITED"
    range_out_str = "vs.RANGE_LIMITED"  # Output always limited

    # Black level for border restore (16bit limited range)
    # Y black = 4096, UV neutral = 32768 for 16bit limited
    black_y = 4096
    neutral_uv = 32768

    # Crop parameters
    has_crop = crop_info is not None and crop_info.get("cropped_w", 0) > 0
    crop_left = crop_info.get("left", 0) if crop_info else 0
    crop_right = crop_info.get("right", 0) if crop_info else 0
    crop_top = crop_info.get("top", 0) if crop_info else 0
    crop_bottom = crop_info.get("bottom", 0) if crop_info else 0

    lines = []
    lines.append("import vapoursynth as vs")
    lines.append("import havsfunc as haf")
    lines.append("core = vs.core")
    lines.append("")

    # Source
    lines.append(f"# Load source: {input_path}")
    lines.append(f"clip = core.ffms2.Source(source={repr(input_path)})")
    lines.append("")

    # Convert to 16bit
    lines.append(f"# Convert to {vs_bit_depth}bit processing space")
    lines.append(f"clip = core.resize.Bicubic(clip, format={vs_format})")
    lines.append("")

    # PSF handling
    if is_psf:
        lines.append("# PSF handling: SeparateFields + Weave")
        tff_bool = "True" if field_order == "tt" else "False"
        lines.append(f"clip = core.std.SeparateFields(clip, tff={tff_bool})")
        lines.append("clip = core.std.Weave(clip)")
        lines.append("")

    # IVTC (must be before deinterlace)
    if needs_ivtc and not is_psf:
        lines.append("# IVTC - must execute before QTGMC")
        if ivtc == "VIVTC":
            lines.append("clip = core.vivtc.VFM(clip, mode=1, ovr=0)")
            lines.append("# VDecimate must chain directly after VFM")
            lines.append("clip = core.vivtc.VDecimate(clip, cycle=5)")
        elif ivtc == "sRestore":
            lines.append("clip = haf.srestore(clip, frate=23.976)")
        lines.append("")

    # Black border crop (before color conversion)
    if has_crop and (crop_left > 0 or crop_right > 0 or crop_top > 0 or crop_bottom > 0):
        lines.append(f"# Auto-crop black borders: L={crop_left} R={crop_right} T={crop_top} B={crop_bottom}")
        lines.append(f"clip = core.std.CropRel(clip, left={crop_left}, right={crop_right}, top={crop_top}, bottom={crop_bottom})")
        lines.append("")

    # QTGMC parameters (used in both segmented and global modes)
    qtgmc_preset = params.get("qtgmc_preset", "Slow")
    fps_divisor = params.get("qtgmc_fps_divisor", 2)
    qtgmc_ezdenoise = params.get("qtgmc_ezdenoise", 2)
    qtgmc_sharpness = params.get("qtgmc_sharpness", 0.5)
    qtgmc_chromamotion = params.get("qtgmc_chromamotion", True)
    qtgmc_sourcematch = params.get("qtgmc_sourcematch", 3)
    qtgmc_input_type = params.get("qtgmc_input_type", 0)
    qtgmc_clear_fieldbased = params.get("qtgmc_clear_fieldbased", True)
    cm_str = "True" if qtgmc_chromamotion else "False"

    # Deinterlace (QTGMC) — segmented or global
    field_segments = params.get("field_segments", [])
    use_segmented = bool(field_segments) and not is_psf and deinterlace == "QTGMC"

    if use_segmented:
        lines.append("# Mixed field order: per-segment QTGMC deinterlacing")
        segment_vars: list[str] = []
        for idx, seg in enumerate(field_segments):
            seg_field = seg.get("field_order", "tt")
            if seg_field not in ("tt", "bb"):
                continue
            start_f = seg.get("start_frame", 0)
            end_f = seg.get("end_frame", 0)
            if end_f <= start_f:
                continue
            tff_bool = "True" if seg_field == "tt" else "False"
            var_name = f"seg{idx}"
            segment_vars.append(var_name)
            lines.append(f"{var_name} = core.std.Trim(clip, first={start_f}, last={end_f})")
            if qtgmc_input_type == 0 and qtgmc_clear_fieldbased:
                field_based_val = 2 if seg_field == "tt" else 1
                lines.append(f"{var_name} = core.std.SetFrameProp({var_name}, prop='_FieldBased', intval={field_based_val})")
            lines.append(f"{var_name} = haf.QTGMC({var_name}, Preset='{qtgmc_preset}', "
                        f"EZDenoise={qtgmc_ezdenoise}, Sharpness={qtgmc_sharpness}, "
                        f"ChromaMotion={cm_str}, SourceMatch={qtgmc_sourcematch}, "
                        f"InputType={qtgmc_input_type}, TFF={tff_bool}, "
                        f"FpsDivisor={fps_divisor})")
            lines.append("")
        if segment_vars:
            splice_args = ", ".join(segment_vars)
            lines.append(f"clip = core.std.Splice([{splice_args}], mismatch=False)")
            lines.append("")
        else:
            # Fallback to global if no valid segments
            use_segmented = False

    if not use_segmented and not is_psf and field_order in ("tt", "bb") and deinterlace == "QTGMC":
        lines.append("# Deinterlace with QTGMC")
        tff_bool = "True" if field_order == "tt" else "False"
        if qtgmc_input_type == 0 and qtgmc_clear_fieldbased:
            field_based_val = 2 if field_order == "tt" else 1
            lines.append("# Ensure correct field order property for QTGMC")
            lines.append(f"clip = core.std.SetFrameProp(clip, prop='_FieldBased', intval={field_based_val})")
        lines.append(f"clip = haf.QTGMC(clip, Preset='{qtgmc_preset}', "
                    f"EZDenoise={qtgmc_ezdenoise}, Sharpness={qtgmc_sharpness}, "
                    f"ChromaMotion={cm_str}, SourceMatch={qtgmc_sourcematch}, "
                    f"InputType={qtgmc_input_type}, TFF={tff_bool}, "
                    f"FpsDivisor={fps_divisor})")
        lines.append("")

    # Denoise
    if denoise == "TemporalDegrain2":
        degrain_tr = denoise_params.get("degrainTR", 2)
        grain_level = denoise_params.get("grainLevel", 2)
        lines.append("# Temporal denoising")
        lines.append(f"clip = haf.TemporalDegrain2(clip, "
                    f"degrainTR={degrain_tr}, grainLevel={grain_level})")
        lines.append("")

    # Deblocking
    if deblock == "Deblock_QED":
        deblock_quant = params.get("deblock_quant", 25)
        lines.append("# Deblocking")
        lines.append(f"clip = haf.Deblock_QED(clip, quant={deblock_quant})")
        lines.append("")
    elif deblock == "DeblockPP7":
        lines.append("# Deblocking (HDV)")
        lines.append("clip = core.deblock.Deblock(clip)")
        lines.append("")

    # Chroma shift
    if chroma_shift:
        lines.append("# Chroma shift correction")
        lines.append("clip = haf.ChromaShift(clip, cx=2)")
        lines.append("")

    # Stabilization
    if stabilize == "DepanStabilize":
        dxmax = stabilize_params.get("dxmax", 10)
        dymax = stabilize_params.get("dymax", 10)
        lines.append("# Temporal stabilization")
        lines.append(f"clip = haf.DepanStabilize(clip, dxmax={dxmax}, dymax={dymax})")
        lines.append("")

    # Color space conversion: BT.601 -> BT.709
    lines.append("# BT.601 -> BT.709 color space conversion")
    lines.append(f"clip = core.resize.Bicubic(clip, "
                f"matrix_in_s='{matrix_in}', matrix_s='709', "
                f"range_in={range_in_str}, range={range_out_str})")
    lines.append("")

    # Restore black borders after color conversion
    if has_crop and (crop_left > 0 or crop_right > 0 or crop_top > 0 or crop_bottom > 0):
        lines.append("# Restore black borders with target color space black levels")
        lines.append(f"clip = core.std.AddBorders(clip, left={crop_left}, right={crop_right}, top={crop_top}, bottom={crop_bottom}, "
                    f"color=[{black_y}, {neutral_uv}, {neutral_uv}])")
        lines.append("")

    # Output
    lines.append("# Output")
    lines.append("clip.set_output()")

    return "\n".join(lines)


def check_deinterlace_quality(video_path: str, logger=None) -> Tuple[bool, float]:
    """Check deinterlace output quality using ffmpeg idet.

    Returns (is_good, progressive_pct) where is_good=True if Progressive >= 95%.
    """
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-v", "info",
            "-i", video_path,
            "-vf", "idet",
            "-f", "null", "-"
        ]
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=300, shell=False)
        if r.returncode != 0:
            return False, 0.0

        stderr = r.stderr
        stats: dict[str, int] = {}
        for line in stderr.splitlines():
            if "Multi frame detection:" in line:
                for key in ("TFF", "BFF", "Progressive", "Undetermined"):
                    m = re.search(rf"{key}:\s*(\d+)", line)
                    if m:
                        stats[key.lower()] = int(m.group(1))

        total = sum(stats.values())
        if total == 0:
            return False, 0.0

        prog_pct = stats.get("progressive", 0) / total * 100
        is_good = prog_pct >= 95.0
        if logger:
            logger.info("STAGE1_DEINT_CHECK",
                       f"Deinterlace quality: Progressive={prog_pct:.1f}%")
        return is_good, prog_pct
    except Exception as e:
        if logger:
            logger.warning("STAGE1_DEINT_CHECK_FAIL", f"Failed to check deinterlace quality: {e}")
        return False, 0.0


def check_ivtc_quality(video_path: str, logger=None) -> Tuple[bool, Dict[str, int]]:
    """Check IVTC output quality using ffmpeg idet.

    Returns (is_good, stats) where is_good=True if TFF+BFF <= 10% and Progressive >= 90%.
    """
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-v", "info",
            "-i", video_path,
            "-vf", "idet",
            "-f", "null", "-"
        ]
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=300, shell=False)
        if r.returncode != 0:
            return False, {}

        stderr = r.stderr
        # Parse idet output lines like:
        # Single frame detection: TFF: 10 BFF: 5 Progressive: 1000 Undetermined: 5
        # Multi frame detection: TFF: 8 BFF: 4 Progressive: 1003 Undetermined: 5
        stats = {}
        for line in stderr.splitlines():
            if "Multi frame detection:" in line:
                for key in ("TFF", "BFF", "Progressive", "Undetermined"):
                    m = re.search(rf"{key}:\s*(\d+)", line)
                    if m:
                        stats[key.lower()] = int(m.group(1))

        total = sum(stats.values())
        if total == 0:
            return False, stats

        tff_pct = stats.get("tff", 0) / total * 100
        bff_pct = stats.get("bff", 0) / total * 100
        prog_pct = stats.get("progressive", 0) / total * 100

        is_good = (tff_pct + bff_pct) <= 10.0 and prog_pct >= 90.0
        if logger:
            logger.info("STAGE1_IVTC_CHECK",
                       f"IVTC quality: TFF={tff_pct:.1f}% BFF={bff_pct:.1f}% Progressive={prog_pct:.1f}%")
        return is_good, stats
    except Exception as e:
        if logger:
            logger.warning("STAGE1_IVTC_CHECK_FAIL", f"Failed to check IVTC quality: {e}")
        return False, {}


def check_frame_rate(video_path: str, expected_fps: float = 23.976,
                      tolerance: float = 0.01, logger=None) -> bool:
    """Verify video frame rate is within tolerance of expected."""
    try:
        cmd = [
            "ffprobe", "-hide_banner", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=30, shell=False)
        if r.returncode != 0:
            return False

        fps_str = r.stdout.strip()
        if "/" in fps_str:
            num, den = map(float, fps_str.split("/"))
            actual_fps = num / den if den != 0 else 0.0
        else:
            actual_fps = float(fps_str)

        deviation = abs(actual_fps - expected_fps) / expected_fps
        ok = deviation <= tolerance
        if logger:
            logger.info("STAGE1_FPS_CHECK",
                       f"Frame rate: {actual_fps:.3f}fps (expected {expected_fps}fps, dev={deviation:.2%})")
        return ok
    except Exception as e:
        if logger:
            logger.warning("STAGE1_FPS_CHECK_FAIL", f"Failed to check frame rate: {e}")
        return False


def render_vpy_script(vpy_path: str, output_path: str, logger=None) -> bool:
    """Render a VapourSynth script to video using vspipe + ffmpeg."""
    try:
        vspipe_cmd = ["vspipe", "--y4m", vpy_path, "-"]
        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-v", "error",
            "-thread_queue_size", "1024",
            "-i", "-",
            "-c:v", "ffv1",
            "-an", "-y", output_path
        ]
        vspipe_proc = subprocess.Popen(vspipe_cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, shell=False)
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=vspipe_proc.stdout,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, shell=False)
        if vspipe_proc.stdout:
            vspipe_proc.stdout.close()
        stdout, stderr = ffmpeg_proc.communicate(timeout=7200)
        return ffmpeg_proc.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        if logger:
            logger.error("STAGE1_RENDER_FAIL", f"Render failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--params")
    parser.add_argument("--config")
    parser.add_argument("--no-wsl-redirect", action="store_true")
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()

    params = json.loads(args.params) if args.params else {}
    log_dir = params.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, job_id=args.job_id)

    temp_dir = params.get("temp_dir", "./temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Build param_hash for checkpoint
    input_path = args.input
    input_name = Path(input_path).name
    param_hash = compute_param_hash(input_path, params)

    # Generate intermediate filename
    inter_name = build_intermediate_name(
        args.job_id, input_name, 1, "preprocess", param_hash, "mkv"
    )
    inter_path = os.path.join(temp_dir, inter_name)

    # Check existing
    if os.path.exists(inter_path) and os.path.getsize(inter_path) > 0:
        logger.info("STAGE1_SKIP", "Intermediate exists", path=inter_path)
        return 3

    # Detect and validate black borders
    crop_info = detect_black_borders(input_path, logger)
    if not validate_crop(crop_info, logger):
        logger.error("STAGE1_CROP_INVALID", "Black border crop validation failed")
        return 81  # Parameter error

    if crop_info and crop_info.get("cropped_w", 0) > 0:
        logger.info("STAGE1_CROP",
                   f"Detected borders: L={crop_info['left']} R={crop_info['right']} "
                   f"T={crop_info['top']} B={crop_info['bottom']} -> {crop_info['cropped_w']}x{crop_info['cropped_h']}")

    # Generate VapourSynth script
    vpy_content = generate_vs_script(input_path, params, args.job_id, crop_info)
    vpy_path = os.path.join(temp_dir, f"{args.job_id}_{Path(input_name).stem}_stage1.vpy")

    with open(vpy_path, "w", encoding="utf-8") as f:
        f.write(vpy_content)

    # Log script hash
    script_hash = hashlib.sha256(vpy_content.encode()).hexdigest()[:16]
    logger.info("STAGE1_VPY", f"VS script generated, hash={script_hash}",
                script_path=vpy_path, hash=script_hash)

    # Render using vspipe + ffmpeg
    logger.info("STAGE1_START", "Rendering VS script", vpy=vpy_path)

    success = False
    try:
        success = render_vpy_script(vpy_path, inter_path, logger)
    except subprocess.TimeoutExpired:
        logger.error("STAGE1_TIMEOUT", "Stage 1 timed out")
        return 87
    except Exception as e:
        logger.error("STAGE1_ERROR", str(e))
        return 82

    # IVTC fallback logic (v6.3)
    needs_ivtc = params.get("needs_ivtc", False)
    ivtc_method = params.get("ivtc", "VIVTC")
    ivtc_auto_fallback = params.get("ivtc_auto_fallback", True)

    if success and needs_ivtc and ivtc_method == "VIVTC" and ivtc_auto_fallback:
        logger.info("STAGE1_IVTC_VERIFY", "Checking IVTC output quality")
        is_good, stats = check_ivtc_quality(inter_path, logger)
        if not is_good:
            logger.warning("STAGE1_IVTC_FALLBACK",
                          "VFM output quality insufficient, falling back to sRestore")
            # Generate sRestore version
            fallback_params = dict(params)
            fallback_params["ivtc"] = "sRestore"
            vpy_fallback = generate_vs_script(input_path, fallback_params, args.job_id, crop_info)
            vpy_fb_path = os.path.join(temp_dir, f"{args.job_id}_{Path(input_name).stem}_stage1_fallback.vpy")
            with open(vpy_fb_path, "w", encoding="utf-8") as f:
                f.write(vpy_fallback)

            fb_hash = hashlib.sha256(vpy_fallback.encode()).hexdigest()[:16]
            logger.info("STAGE1_VPY_FB", f"Fallback VS script generated, hash={fb_hash}")

            # Remove old output and render fallback
            if os.path.exists(inter_path):
                os.remove(inter_path)
            try:
                success = render_vpy_script(vpy_fb_path, inter_path, logger)
            except Exception as e:
                logger.error("STAGE1_IVTC_FB_FAIL", f"sRestore fallback render failed: {e}")
                return 84

            if success:
                # Verify sRestore frame rate (should be ~23.976)
                fps_ok = check_frame_rate(inter_path, expected_fps=23.976,
                                          tolerance=0.01, logger=logger)
                if not fps_ok:
                    logger.error("STAGE1_IVTC_FPS_FAIL",
                                "sRestore output frame rate out of tolerance")
                    return 86  # Verification failure
            else:
                logger.error("STAGE1_IVTC_FB_FAIL", "sRestore fallback render failed")
                return 84

    # Segmented deinterlace fallback logic (v6.3)
    if success and params.get("field_segments") and params.get("deinterlace", "QTGMC") == "QTGMC":
        logger.info("STAGE1_DEINT_VERIFY", "Checking segmented deinterlace quality")
        is_good, prog_pct = check_deinterlace_quality(inter_path, logger)
        if not is_good:
            logger.warning("STAGE1_DEINT_FALLBACK",
                          f"Progressive={prog_pct:.1f}% < 95%, falling back to global conservative deinterlace")
            fallback_params = dict(params)
            fallback_params["field_segments"] = []  # Clear segments
            # Use global field_order if valid, otherwise default to tt
            if fallback_params.get("field_order") not in ("tt", "bb"):
                fallback_params["field_order"] = "tt"
            vpy_fallback = generate_vs_script(input_path, fallback_params, args.job_id, crop_info)
            vpy_fb_path = os.path.join(temp_dir, f"{args.job_id}_{Path(input_name).stem}_stage1_deint_fb.vpy")
            with open(vpy_fb_path, "w", encoding="utf-8") as f:
                f.write(vpy_fallback)

            fb_hash = hashlib.sha256(vpy_fallback.encode()).hexdigest()[:16]
            logger.info("STAGE1_VPY_DEINT_FB", f"Fallback VS script generated, hash={fb_hash}")

            # Remove old output and render fallback
            if os.path.exists(inter_path):
                os.remove(inter_path)
            try:
                success = render_vpy_script(vpy_fb_path, inter_path, logger)
            except Exception as e:
                logger.error("STAGE1_DEINT_FB_FAIL", f"Global deinterlace fallback render failed: {e}")
                return 84

            if not success:
                logger.error("STAGE1_DEINT_FB_FAIL", "Global deinterlace fallback render failed")
                return 84

    if success and os.path.exists(inter_path):
        logger.info("STAGE1_DONE", "Stage 1 complete", output=inter_path)
        return 0
    else:
        logger.error("STAGE1_FAILED", "Rendering failed")
        return 84


if __name__ == "__main__":
    sys.exit(main() or 0)
