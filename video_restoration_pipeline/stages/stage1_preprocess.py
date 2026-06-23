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

v6.3 Compliance:
- IVTC fallback threshold from config
- InputType=1 for damaged field order detection
- matrix_in_s from color metadata or resolution heuristic
- All params read from config (not hardcoded)
"""

import os
import sys
import json
import argparse
import subprocess
import re
from pathlib import Path
from typing import Tuple, Dict, Optional

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from checkpoint import compute_param_hash, build_intermediate_name  # noqa: E402


def _get_video_color_metadata(video_path: str) -> dict:
    """Get color metadata from video stream using ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=color_primaries,color_trc,colorspace,color_range",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            import json as _json
            data = _json.loads(r.stdout)
            if data.get("streams"):
                return data["streams"][0]
    except Exception:
        pass
    return {}


def detect_black_borders(video_path: str, logger=None) -> dict:
    """Detect black borders using ffmpeg cropdetect."""
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "info",
        "-i", video_path,
        "-vf", "cropdetect=limit=16:round=2:reset=0",
        "-frames:v", "100",
        "-f", "null", "-"
    ]
    crop_info = {"left": 0, "right": 0, "top": 0, "bottom": 0,
                 "cropped_w": 0, "cropped_h": 0}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=120, shell=False)
        crop_match = None
        for line in r.stderr.split("\n"):
            m = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", line)
            if m:
                crop_match = m
        if crop_match:
            cw, ch, cx, cy = map(int, crop_match.groups())
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
                parts = wr.stdout.strip().split("x")
                if len(parts) == 2:
                    ow, oh = map(int, parts)
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
        return True

    ow = crop_info.get("orig_w", 1)
    oh = crop_info.get("orig_h", 1)
    cw = crop_info.get("cropped_w", 0)
    ch = crop_info.get("cropped_h", 0)

    if cw < 240 or ch < 240:
        if logger:
            logger.error("STAGE1_CROP_TOO_SMALL",
                        f"Cropped size {cw}x{ch} below minimum 240px")
        return False

    orig_area = ow * oh
    crop_area = cw * ch
    if crop_area < orig_area * 0.5:
        if logger:
            logger.error("STAGE1_CROP_TOO_MUCH",
                        f"Cropped area {crop_area} < 50% of original {orig_area}")
        return False

    return True


def _get_matrix_in(video_path: str, target_res: str, params: dict) -> str:
    """Determine matrix_in_s based on color metadata or resolution fallback.

    Priority:
    1. color_primaries from ffprobe (bt470bg->470bg, smpte170m->170m)
    2. Resolution heuristic using SOURCE video height (not target)

    FIX-P6-4: Resolution fallback must use SOURCE video height per v7.32 §3.10.3.
    ORIGINAL: Used target_res to determine PAL/NTSC, which is wrong for
    PAL 720x576 source with target_resolution="1280x720" (would return 170m).
    """
    # FIX-P6-7: Accept pre-fetched color_meta from params to avoid duplicate ffprobe call
    # ORIGINAL: Always called _get_video_color_metadata() — duplicate call in generate_vs_script
    color_meta = params.get("_color_meta_cache")
    if color_meta is None:
        color_meta = _get_video_color_metadata(video_path)
    return _get_matrix_in_from_meta(video_path, color_meta, target_res, params)


def _get_matrix_in_from_meta(video_path: str, color_meta: dict,
                              target_res: str, params: dict) -> str:
    """Determine matrix_in_s from pre-fetched color metadata or resolution fallback.

    FIX-P6-7: Extracted from _get_matrix_in to support single ffprobe call pattern.
    """
    cp = color_meta.get("color_primaries", "")
    # FIX-P6-3: bt470m is NTSC (BT.470 System M) -> 170m, not PAL 470bg per v7.32 §3.10.3
    # ORIGINAL: if cp in ("bt470bg", "bt470m"): return "470bg"
    # bt470m was incorrectly mapped to PAL 470bg; it's actually NTSC
    if cp == "bt470bg":
        return "470bg"
    if cp in ("smpte170m", "bt470m"):
        return "170m"

    # FIX-P6-2: BT.709 source — identity, no conversion needed per v7.32 §3.10.3
    # ORIGINAL: No handling for bt709; BT.709 source was forced through BT.601→BT.709 conversion
    cs = color_meta.get("colorspace", "")
    if cp in ("bt709",) or cs in ("bt709",):
        return "709"

    # FIX-P6-4: Resolution fallback: use SOURCE video height, not target_res
    source_h = 480
    try:
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=height",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            if data.get("streams"):
                source_h = int(data["streams"][0].get("height", 480))
    except Exception:
        source_h = 480

    # FIX-P6-1: Only exact 576-line = PAL SD BT.601 per v7.32 §3.10.3
    # ORIGINAL: if source_h <= 576: return "470bg" — NTSC 480-line incorrectly mapped to PAL
    if source_h == 576:
        return "470bg"  # PAL SD BT.601
    return "170m"  # NTSC BT.601 (480-line NTSC SD and all other resolutions)


def generate_vs_script(input_path: str, params: dict, job_id: str,
                       crop_info: Optional[dict] = None,
                       ivtc_stats: Optional[dict] = None) -> str:
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

    # FIX-P6-7: Fetch color metadata ONCE, cache for _get_matrix_in to reuse
    # ORIGINAL: _get_video_color_metadata called twice (in _get_matrix_in + here)
    color_meta = _get_video_color_metadata(input_path)
    params["_color_meta_cache"] = color_meta

    # Determine matrix based on metadata + resolution
    matrix_in = _get_matrix_in(input_path, target_res, params)

    # Clean up cache (don't leak into downstream params)
    params.pop("_color_meta_cache", None)

    # VapourSynth format based on bit depth
    if vs_bit_depth == 16:
        vs_format = "vs.YUV420P16"
    elif vs_bit_depth == 10:
        vs_format = "vs.YUV420P10"
    else:
        vs_format = "vs.YUV420P8"

    # Determine range from pre-fetched color metadata (FIX-P6-2 / FIX-P6-7)
    detected_range = color_meta.get("color_range", "")
    if detected_range == "tv":
        color_range = "limited"
    elif detected_range == "pc":
        color_range = "full"
    else:
        # FIX-P6-3: Log WARNING when color_range cannot be detected per v6.32 §7.19
        # "若无法读取源 range 属性，默认假设老影像为 limited range，但记录 WARNING"
        color_range = params.get("color_range", "limited")
        if detected_range not in ("tv", "pc"):
            # Will be logged by caller (main()) since generate_vs_script
            # doesn't have access to logger. Mark in params for downstream logging.
            params["_color_range_unknown"] = True
    range_in_str = "vs.RANGE_FULL" if color_range == "full" else "vs.RANGE_LIMITED"
    range_out_str = "vs.RANGE_LIMITED"

    # Black level dynamic calculation per v6.3 §3.10.3
    if color_range == "full":
        black_y = 0
        neutral_uv = 128 * (2 ** (vs_bit_depth - 8))
    else:
        if vs_bit_depth == 8:
            black_y, neutral_uv = 16, 128
        elif vs_bit_depth == 10:
            black_y, neutral_uv = 64, 512
        else:
            # Generic formula for any bit depth (including 16bit)
            black_y = int(16 * (2 ** (vs_bit_depth - 8)))
            neutral_uv = int(128 * (2 ** (vs_bit_depth - 8)))

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

    # Convert to target bit depth
    lines.append(f"# Convert to {vs_bit_depth}bit processing space")
    lines.append(f"clip = core.resize.Bicubic(clip, format={vs_format})")
    lines.append("")

    # Default tff_bool for use throughout script
    tff_bool = "True" if field_order == "tt" else "False"

    # QTGMC parameters (moved here so PSF path can use them) FIX-P6-3/P6-11
    qtgmc_preset = params.get("qtgmc_preset", "Slow")
    # Validate preset
    valid_presets = {"Placebo", "Very Slow", "Slower", "Slow", "Medium", "Fast", "Faster",
                     "Very Fast", "Super Fast", "Ultra Fast", "Draft"}
    if qtgmc_preset not in valid_presets:
        qtgmc_preset = "Slow"

    fps_divisor = params.get("qtgmc_fps_divisor", 1)  # FIX-P6-3: default keep double frame rate
    qtgmc_ezdenoise = params.get("qtgmc_ezdenoise", 2)
    # Validate EZDenoise range
    qtgmc_ezdenoise = max(0, min(qtgmc_ezdenoise, 10))

    qtgmc_sharpness = params.get("qtgmc_sharpness", 0.5)
    qtgmc_chromamotion = params.get("qtgmc_chromamotion", True)
    qtgmc_sourcematch = params.get("qtgmc_sourcematch", 3)
    qtgmc_input_type = params.get("qtgmc_input_type", 0)
    qtgmc_clear_fieldbased = params.get("qtgmc_clear_fieldbased", True)
    cm_str = "True" if qtgmc_chromamotion else "False"

    # PSF handling
    if is_psf:
        lines.append("# PSF handling: SeparateFields + Weave")
        lines.append(f"clip = core.std.SeparateFields(clip, tff={tff_bool})")
        lines.append("clip = core.std.Weave(clip)")
        # FIX-P6-11: Optional progressive repair for PSF
        if params.get("psf_use_input_type1", False):
            lines.append("# Optional progressive repair for PSF")
            # FIX-P6-2: PSF FpsDivisor=1 per v7.32 §3.10.1
            # ORIGINAL: FpsDivisor={fps_divisor} — PSF restore already produces correct frame rate,
            # must NOT halve it. FpsDivisor=1 keeps the native PSF frame rate.
            lines.append(f"clip = haf.QTGMC(clip, Preset='{qtgmc_preset}', "
                        f"EZDenoise={qtgmc_ezdenoise}, Sharpness={qtgmc_sharpness}, "
                        f"ChromaMotion={cm_str}, SourceMatch={qtgmc_sourcematch}, "
                        f"InputType=1, FpsDivisor=1)")
        lines.append("")

    # IVTC (must be before deinterlace)
    if needs_ivtc and not is_psf:
        lines.append("# IVTC - must execute before QTGMC")
        if ivtc == "VIVTC":
            # FIX-P6-8: VFM mode/VDecimate cycle from config per v6.32 §7.9 / v7.32 §3.10.13
            # ORIGINAL: hardcoded mode=1, cycle=5
            vfm_mode = params.get("vfm_mode", 1)
            vdecimate_cycle = params.get("vdecimate_cycle", 5)
            # V-P-019 fix: VFM and VDecimate must be chained directly per v7.32 §3.10.13
            # FIX-P6-7: _Combed diagnostic block between VFM and VDecimate per v7.32 §3.10.13
            lines.append(f"clip = core.vivtc.VFM(clip, mode={vfm_mode}, ovr=0)")
            lines.append(f"# _Combed statistics collection (sidecar export) per v7.32 §3.10.13")
            lines.append(f"_combed_stats = " + r'{"combed_frames": [], "total": 0, "mode": ' + str(vfm_mode) + r'}')
            lines.append("def _collect_combed(n, f):")
            lines.append("    if f.props.get('_Combed', 0):")
            lines.append("        _combed_stats['combed_frames'].append(n)")
            lines.append("    _combed_stats['total'] += 1")
            lines.append("    return f")
            # FIX-E1: ModifyFrame is EXEMPT from the "no filters between VFM/VDecimate" rule
            # because it only reads frame properties (_Combed) without modifying pixels or
            # frame attributes. VDecimate's combed-frame detection relies solely on VFM's
            # output pixel data and _Combed prop, neither of which ModifyFrame alters.
            # Per v7.32 §3.10.13, the prohibition targets "修改帧内容/属性的滤镜" only.
            lines.append("_diag = core.std.ModifyFrame(clip, clip, _collect_combed)")
            lines.append("# Eagerly request first 300 frames to trigger callbacks")
            lines.append("_max_diag = min(300, clip.num_frames)")
            lines.append("for _ci in range(_max_diag):")
            lines.append("    try:")
            lines.append("        _diag.get_frame(_ci)")
            lines.append("    except Exception:")
            lines.append("        break")
            lines.append("import os as _os, json as _json")
            lines.append("_sidecar_path = _os.environ.get('VS_COMBED_STATS', '')")
            lines.append("if _sidecar_path:")
            lines.append("    try:")
            lines.append("        with open(_sidecar_path, 'w', encoding='utf-8') as _sf:")
            lines.append("            _json.dump(_combed_stats, _sf)")
            lines.append("    except Exception:")
            lines.append("        pass")
            lines.append(f"clip = core.vivtc.VDecimate(clip, cycle={vdecimate_cycle})")
        elif ivtc == "sRestore":
            lines.append("clip = haf.srestore(clip, frate=23.976)")
        lines.append("")

    # Black border crop (before color conversion)
    if has_crop and (crop_left > 0 or crop_right > 0 or crop_top > 0 or crop_bottom > 0):
        lines.append("# Auto-crop black borders")
        lines.append(f"clip = core.std.CropRel(clip, left={crop_left}, right={crop_right}, top={crop_top}, bottom={crop_bottom})")
        lines.append("")

    # V-P-025 fix: Remove InputType=1 test code from generated script
    # (damaged field order detection should be done in classify.py, not in VS script)

    # Deinterlace (QTGMC)
    field_segments = params.get("field_segments", [])
    use_segmented = bool(field_segments) and not is_psf and deinterlace == "QTGMC"

    if use_segmented:
        lines.append("# Mixed field order: per-segment QTGMC deinterlacing")
        segment_vars = []
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
            lines.append("clip = core.std.SetFieldBased(clip, 0)")  # FIX-P6-6
            lines.append("")
        else:
            use_segmented = False

    if not use_segmented and not is_psf and field_order in ("tt", "bb") and deinterlace == "QTGMC":
        lines.append("# Deinterlace with QTGMC")
        if qtgmc_input_type == 0 and qtgmc_clear_fieldbased:
            field_based_val = 2 if field_order == "tt" else 1
            lines.append("# Ensure correct field order property for QTGMC")
            lines.append(f"clip = core.std.SetFrameProp(clip, prop='_FieldBased', intval={field_based_val})")
        lines.append(f"clip = haf.QTGMC(clip, Preset='{qtgmc_preset}', "
                    f"EZDenoise={qtgmc_ezdenoise}, Sharpness={qtgmc_sharpness}, "
                    f"ChromaMotion={cm_str}, SourceMatch={qtgmc_sourcematch}, "
                    f"InputType={qtgmc_input_type}, TFF={tff_bool}, "
                    f"FpsDivisor={fps_divisor})")
        # v6.3: Reset _FieldBased to 0 after InputType=0 processing
        if qtgmc_input_type == 0:
            lines.append("clip = core.std.SetFieldBased(clip, 0)")
        lines.append("")
    elif field_order == "unknown" and not is_psf:
        # v6.3: unknown field_order -> skip QTGMC InputType=0, conservative handling
        # FIX-P6-5: Clear _FieldBased attribute per v7.32 §3.10.1
        # ORIGINAL: Only added comment, did not clear _FieldBased property
        lines.append("# field_order=unknown: skipping QTGMC deinterlace (conservative)")
        lines.append("clip = core.std.SetFieldBased(clip, 0)")
        lines.append("")
    elif field_order == "progressive" and not is_psf:
        # FIX-P6-9: explicit comment for progressive per v7.32 §3.10.1
        # field_order=progressive: native progressive, skipping QTGMC deinterlace
        lines.append("# field_order=progressive: native progressive, skipping QTGMC deinterlace")
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
        # FIX-P6-12: Support configurable DeblockPP7 parameters per v7.32 §3.10.1
        # ORIGINAL: Hardcoded call with no parameters
        deblock_pp7_params = params.get("deblock_pp7_params", {})
        lines.append("# Deblocking (HDV)")
        if deblock_pp7_params:
            pp7_args = ", ".join(f"{k}={v}" for k, v in deblock_pp7_params.items())
            lines.append(f"clip = core.deblock.Deblock(clip, {pp7_args})")
        else:
            lines.append("clip = core.deblock.Deblock(clip)")
        lines.append("")

    # Chroma shift
    if chroma_shift:
        lines.append("# Chroma shift correction")
        lines.append("clip = haf.ChromaShift(clip, cx=2)")
        lines.append("")

    # Stabilization
    if stabilize == "DepanStabilize":
        dxmax = max(0, min(stabilize_params.get("dxmax", 10), 20))  # FIX-P6-9
        dymax = max(0, min(stabilize_params.get("dymax", 10), 20))  # FIX-P6-9
        lines.append("# Temporal stabilization")
        lines.append(f"clip = haf.DepanStabilize(clip, dxmax={dxmax}, dymax={dymax})")
        lines.append("")

    # Color space conversion: BT.601 -> BT.709
    # FIX-P6-2: Skip conversion if source is already BT.709 per v7.32 §3.10.3
    # ORIGINAL: Always applied BT.601→BT.709 conversion even for BT.709 sources
    if matrix_in == "709":
        lines.append("# Source is already BT.709, no color conversion needed")
    else:
        lines.append("# BT.601 -> BT.709 color space conversion")
        lines.append(f"clip = core.resize.Bicubic(clip, "
                    f"matrix_in_s='{matrix_in}', matrix_s='709', "
                    f"range_in={range_in_str}, range={range_out_str})")
    lines.append("")

    # Restore black borders
    if has_crop and (crop_left > 0 or crop_right > 0 or crop_top > 0 or crop_bottom > 0):
        lines.append("# Restore black borders")
        lines.append(f"clip = core.std.AddBorders(clip, left={crop_left}, right={crop_right}, top={crop_top}, bottom={crop_bottom}, "
                    f"color=[{black_y}, {neutral_uv}, {neutral_uv}])")
        lines.append("")

    # Output
    lines.append("# Output")
    lines.append("clip.set_output()")

    return "\n".join(lines)


def _check_combed_and_fallback(vs_path: Path, params: dict, logger) -> bool:
    """Run quick _Combed diagnostic and return True if sRestore fallback needed.

    FIX-P6-7: Per v7.32 §3.10.13, VFM output must be validated for _Combed
    markers. If combed_ratio > threshold, sRestore fallback is required.

    Uses the _Combed diagnostic block embedded in the .vpy script.
    Runs a quick vspipe pass (first 300 frames) to populate sidecar JSON.
    """
    combed_stats_path = vs_path.with_suffix(".combed.json")
    env = {**os.environ, "VS_COMBED_STATS": str(combed_stats_path)}

    # Quick diagnostic pass: request frames 0-299 to populate sidecar
    # FIX-P6-9: Write y4m stdout to /dev/null instead of capturing to memory
    # ORIGINAL: capture_output=True stored up to 100MB+ of y4m data in RAM
    try:
        with open(os.devnull, 'wb') as devnull_out:
            r_diag = subprocess.run(
                ["vspipe", str(vs_path), "-", "-s", "0", "-e", "299"],
                env=env, stdout=devnull_out, stderr=subprocess.PIPE,
                shell=False, timeout=120
            )
        # Output discarded; sidecar file is the only channel we care about
    except Exception as e:
        logger.warning("IVTC_COMBED_DIAG_FAIL",
                      f"_Combed diagnostic pass failed: {e}")
        return False

    needs_srestore = False
    if combed_stats_path.exists():
        try:
            with open(combed_stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            total = stats.get("total", 0)
            combed_count = len(stats.get("combed_frames", []))
            if total > 0:
                ratio = combed_count / total
                # Threshold: v7.32 §3.10.13 "continuous multi-frame failure"
                # 10% is conservative; normal 3:2 Pulldown _Combed < 2%
                threshold = params.get("ivtc_combed_threshold", 0.10)
                if ratio > threshold:
                    logger.warning("IVTC_COMBED_FAIL",
                                  f"VFM _Combed ratio {ratio:.1%} > threshold "
                                  f"{threshold:.1%}, sRestore fallback required",
                                  combed_frames=stats.get("combed_frames")[:20])
                    needs_srestore = True
        except Exception as e:
            logger.warning("IVTC_COMBED_READ_FAIL",
                          f"Failed to read _Combed sidecar: {e}")
        finally:
            # Clean up sidecar
            try:
                combed_stats_path.unlink()
            except Exception:
                pass

    return needs_srestore


def check_deinterlace_quality(video_path: str, logger=None) -> Tuple[bool, float]:
    """Check deinterlace output quality using ffmpeg idet."""
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
        stats = {}
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
            logger.warning("STAGE1_DEINT_CHECK_FAIL", f"Failed: {e}")
        return False, 0.0


def check_ivtc_quality(video_path: str, logger=None,
                       ivtc_fallback_threshold: Optional[dict] = None) -> Tuple[bool, Dict[str, int]]:
    """Check IVTC output quality using ffmpeg idet.

    Uses ivtc_fallback_threshold config for decision:
    - tff_bff_percent: if TFF+BFF > threshold, IVTC failed
    - progressive_percent: must be >= threshold for success
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

        # Use config thresholds if available
        if ivtc_fallback_threshold:
            tff_bff_limit = ivtc_fallback_threshold.get("tff_bff_percent", 10.0)
            prog_min = ivtc_fallback_threshold.get("progressive_percent", 90.0)
            is_good = (tff_pct + bff_pct) <= tff_bff_limit and prog_pct >= prog_min
        else:
            is_good = (tff_pct + bff_pct) <= 10.0 and prog_pct >= 90.0

        if logger:
            logger.info("STAGE1_IVTC_CHECK",
                       f"TFF={tff_pct:.1f}% BFF={bff_pct:.1f}% Progressive={prog_pct:.1f}%")
        return is_good, stats
    except Exception as e:
        if logger:
            logger.warning("STAGE1_IVTC_CHECK_FAIL", f"Failed: {e}")
        return False, {}


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

    temp_dir = params.get("temp_dir", "./temp")
    # FIX-P6-5: WSL2 defensive check per v7.32 §3.10.10
    # main.py handles primary WSL2 redirect; this is a defensive fallback
    if temp_dir.startswith("/mnt/") and not params.get("no_wsl_redirect"):
        logger.warning("STAGE1_WSL_UNREDIRECTED",
                      f"temp_dir still on /mnt/: {temp_dir}. "
                      "Expected main.py to redirect before calling stage.")
        try:
            from checkpoint import wsl_redirect_path
            redirected = wsl_redirect_path(temp_dir)
            if isinstance(redirected, tuple):
                redirected = redirected[0]
            if redirected != temp_dir:
                logger.info("STAGE1_WSL_REDIRECT",
                           f"temp_dir redirected: {temp_dir} -> {redirected}")
                temp_dir = redirected
        except ImportError:
            pass
    # FIX-P6-10: Use pathlib.Path consistently per project convention
    # ORIGINAL: os.makedirs / os.path.* mixed usage
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    input_path = args.input
    input_name = Path(input_path).name

    # Check for existing intermediate
    param_hash = compute_param_hash(input_path, params)
    inter_name = build_intermediate_name(
        args.job_id, input_name, 1, "preprocess", param_hash, "mkv"
    )
    inter_path = str(Path(temp_dir) / inter_name)

    # FIX-P6-10: pathlib consistency
    if Path(inter_path).exists() and Path(inter_path).stat().st_size > 0:
        logger.info("STAGE1_SKIP", "Intermediate exists", path=inter_path)
        return 3

    logger.info("STAGE1_START", "Traditional video preprocessing", input=input_path)

    # Detect black borders
    crop_info = detect_black_borders(input_path, logger)
    if not validate_crop(crop_info, logger):
        # v6.3 §7.19 / v7.32 §3.10.3: invalid crop must mark FAILED, not fallback to full frame
        logger.error("STAGE1_CROP_FAILED", "Black border crop validation failed, aborting")
        return 81

    # Generate VapourSynth script
    vs_script = generate_vs_script(input_path, params, args.job_id, crop_info)

    # FIX-P6-3: Log WARNING when color_range cannot be detected per v6.32 §7.19
    if params.get("_color_range_unknown"):
        logger.warning("STAGE1_COLOR_RANGE_UNKNOWN",
                      "Cannot detect color_range from ffprobe, "
                      "defaulting to limited per v6.32 §7.19")
        params.pop("_color_range_unknown", None)

    # FIX-P6-8: Log warning for unknown field_order
    if params.get("field_order") == "unknown":
        logger.warning("STAGE1_UNKNOWN_FIELD_ORDER",
                      "field_order=unknown, skipping QTGMC deinterlace (conservative)")

    # ISSUE-039 fix: hard validation of IVTC-before-QTGMC ordering in generated script
    needs_ivtc = params.get("needs_ivtc", False)
    if needs_ivtc:
        ivtc_pos = vs_script.find("core.vivtc.VFM")
        if ivtc_pos == -1:
            ivtc_pos = vs_script.find("haf.srestore")
        qtgmc_pos = vs_script.find("haf.QTGMC")
        if qtgmc_pos != -1 and ivtc_pos != -1 and qtgmc_pos < ivtc_pos:
            logger.error("STAGE1_ORDER_FAIL",
                        "QTGMC appears before IVTC in generated script; violates v6.3 hard constraint")
            return 81

    vs_path = str(Path(temp_dir) / f"{args.job_id}_{input_name}.vpy")
    with open(vs_path, "w", encoding="utf-8") as f:
        f.write(vs_script)

    # V-P-020 fix: record SHA-256 hash of generated script
    import hashlib
    script_hash = hashlib.sha256(vs_script.encode("utf-8")).hexdigest()
    logger.info("STAGE1_VPY", f"VS script written: {vs_path}", script_sha256=script_hash)

    # FIX-P6-7: _Combed pre-check for VIVTC path per v7.32 §3.10.13
    # Run quick diagnostic before main render; if combed_ratio > threshold,
    # return 86 so main.py can re-run with sRestore
    ivtc_method = params.get("ivtc", "VIVTC")
    if needs_ivtc and ivtc_method == "VIVTC" and params.get("ivtc_auto_fallback", True):
        if _check_combed_and_fallback(Path(vs_path), params, logger):
            logger.warning("IVTC_COMBED_PRECHECK",
                          "VFM _Combed pre-check failed, requesting sRestore fallback")
            return 86

    # Render with vspipe -> ffmpeg
    try:
        vspipe_cmd = ["vspipe", "--y4m", vs_path, "-"]
        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-v", "error",
            "-thread_queue_size", "1024",
            "-i", "-",
            "-c:v", "ffv1",
            "-an", "-y", inter_path
        ]

        vp = subprocess.Popen(vspipe_cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, shell=False)
        ff = subprocess.Popen(ffmpeg_cmd, stdin=vp.stdout,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, shell=False)
        vp.stdout.close()
        _, stderr = ff.communicate(timeout=7200)

        if ff.returncode == 0 and Path(inter_path).exists():
            # V-P-022 fix: IVTC auto-fallback check
            needs_ivtc = params.get("needs_ivtc", False)
            ivtc = params.get("ivtc", "VIVTC")
            ivtc_auto_fallback = params.get("ivtc_auto_fallback", True)
            if needs_ivtc and ivtc == "VIVTC" and ivtc_auto_fallback:
                ivtc_threshold = params.get("ivtc_fallback_threshold", {})
                is_good, stats = check_ivtc_quality(inter_path, logger, ivtc_threshold)
                if not is_good:
                    logger.warning("STAGE1_IVTC_FALLBACK",
                                  f"IVTC quality check failed (stats={stats}), regenerating with sRestore")
                    # Regenerate with sRestore
                    srestore_params = dict(params)
                    srestore_params["ivtc"] = "sRestore"
                    vs_script_srestore = generate_vs_script(input_path, srestore_params, args.job_id, crop_info)
                    vs_path_srestore = str(Path(temp_dir) / f"{args.job_id}_{input_name}_srestore.vpy")
                    with open(vs_path_srestore, "w", encoding="utf-8") as f:
                        f.write(vs_script_srestore)
                    script_hash_sr = hashlib.sha256(vs_script_srestore.encode("utf-8")).hexdigest()
                    logger.info("STAGE1_VPY_SRESTORE", f"sRestore script written: {vs_path_srestore}", script_sha256=script_hash_sr)

                    # FIX-P6-4: Render sRestore to temporary file first (atomic safety)
                    # ORIGINAL: Rendered directly to inter_path, corrupting VIVTC output on failure
                    srestore_inter_path = inter_path + ".srestore.tmp"
                    ffmpeg_cmd_srestore = [
                        "ffmpeg", "-hide_banner", "-v", "error",
                        "-thread_queue_size", "1024",
                        "-i", "-",
                        "-c:v", "ffv1",
                        "-an", "-y", srestore_inter_path
                    ]
                    vp2 = subprocess.Popen(["vspipe", "--y4m", vs_path_srestore, "-"],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
                    ff2 = subprocess.Popen(ffmpeg_cmd_srestore, stdin=vp2.stdout,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
                    vp2.stdout.close()
                    _, stderr2 = ff2.communicate(timeout=7200)
                    if ff2.returncode == 0 and Path(srestore_inter_path).exists():
                        # Atomic replace: only overwrite original on success
                        import shutil
                        try:
                            shutil.move(srestore_inter_path, inter_path)
                        except OSError:
                            # Cross-filesystem fallback
                            shutil.copyfile(srestore_inter_path, inter_path)
                            try:
                                os.unlink(srestore_inter_path)
                            except Exception:
                                pass
                        # V-P-022: verify sRestore frame rate
                        try:
                            fr_cmd = ["ffprobe", "-hide_banner", "-v", "error",
                                      "-select_streams", "v:0",
                                      "-show_entries", "stream=r_frame_rate",
                                      "-of", "default=noprint_wrappers=1:nokey=1", inter_path]
                            fr_r = subprocess.run(fr_cmd, capture_output=True, text=True, timeout=30, shell=False)
                            if fr_r.returncode == 0:
                                rate_str = fr_r.stdout.strip()
                                if "/" in rate_str:
                                    num, den = map(float, rate_str.split("/"))
                                    detected_fps = num / den if den != 0 else 0
                                else:
                                    detected_fps = float(rate_str)
                                expected_fps = 23.976
                                if abs(detected_fps - expected_fps) / expected_fps > 0.01:
                                    logger.error("STAGE1_IVTC_FRAMERATE",
                                                f"sRestore output fps {detected_fps} deviates from expected {expected_fps}")
                                    return 86
                        except Exception as e:
                            logger.warning("STAGE1_FRAMERATE_CHECK", f"Frame rate check failed: {e}")
                        logger.info("STAGE1_SRESTORE_DONE", "sRestore IVTC fallback complete", output=inter_path)
                    else:
                        # Clean up failed sRestore temp file
                        try:
                            if Path(srestore_inter_path).exists():
                                Path(srestore_inter_path).unlink()
                        except Exception:
                            pass
                        err_msg = stderr2.decode()[:300] if stderr2 else "Unknown error"
                        logger.error("STAGE1_SRESTORE_FAIL", f"sRestore render error: {err_msg}")
                        return 84

            # V-P-023 fix: deinterlace quality check for segmented field order
            field_segments = params.get("field_segments", [])
            if field_segments:
                is_good, prog_pct = check_deinterlace_quality(inter_path, logger)
                if not is_good:
                    logger.warning("STAGE1_DEINT_FALLBACK",
                                  f"Deinterlace quality {prog_pct:.1f}% < 95%, progressive ratio too low")
                    # Conservative fallback: result is usable but flagged

            # V-P-024 fix: PSF frame rate verification
            is_psf = params.get("is_psf", False)
            if is_psf:
                try:
                    fr_cmd = ["ffprobe", "-hide_banner", "-v", "error",
                              "-select_streams", "v:0",
                              "-show_entries", "stream=r_frame_rate",
                              "-of", "default=noprint_wrappers=1:nokey=1", inter_path]
                    fr_r = subprocess.run(fr_cmd, capture_output=True, text=True, timeout=30, shell=False)
                    if fr_r.returncode == 0:
                        rate_str = fr_r.stdout.strip()
                        if "/" in rate_str:
                            num, den = map(float, rate_str.split("/"))
                            detected_fps = num / den if den != 0 else 0
                        else:
                            detected_fps = float(rate_str)
                        # FIX-P6-6: Also check "fps" key from classify output per v7.32 §3.10.1
                        # ORIGINAL: Only checked "original_fps" which is never set by main.py
                        original_fps = params.get("original_fps", params.get("fps", 0))
                        if original_fps > 0 and abs(detected_fps - original_fps) > 0.5:
                            logger.error("STAGE1_PSF_FRAMERATE",
                                        f"PSF output fps {detected_fps} differs from original {original_fps}")
                            return 86
                except Exception as e:
                    logger.warning("STAGE1_PSF_FRAMERATE_CHECK", f"PSF frame rate check failed: {e}")

            # FIX-P6-12: --cleanup-intermediates removes VS script per v7.32 §3.10.17
            if args.cleanup_intermediates:
                try:
                    vs_path_obj = Path(vs_path)
                    if vs_path_obj.exists():
                        vs_path_obj.unlink()
                        logger.info("STAGE1_CLEANUP", f"Removed VS script: {vs_path}")
                    # Also remove sRestore script if it exists
                    srestore_vs = Path(f"{vs_path_obj.stem}_srestore{vs_path_obj.suffix}")
                    if srestore_vs.exists():
                        srestore_vs.unlink()
                        logger.info("STAGE1_CLEANUP", f"Removed sRestore VS script: {srestore_vs}")
                except Exception as e:
                    logger.warning("STAGE1_CLEANUP_FAIL", f"Failed to cleanup: {e}")

            logger.info("STAGE1_DONE", "Preprocessing complete", output=inter_path)
            return 0
        else:
            err_msg = stderr.decode()[:300] if stderr else "Unknown error"
            logger.error("STAGE1_FAILED", f"Render error: {err_msg}")
            return 84
    except subprocess.TimeoutExpired:
        logger.error("STAGE1_TIMEOUT", "Preprocessing timed out")
        return 87
    except Exception as e:
        logger.error("STAGE1_ERROR", str(e))
        return 82


if __name__ == "__main__":
    sys.exit(main() or 0)
