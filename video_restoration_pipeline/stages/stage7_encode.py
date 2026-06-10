#!/usr/bin/env python3
"""
Stage 7: Final Encoding
- SVT-AV1 10-bit encoding with CRF
- FLAC audio
- Original metadata preserved
- SRT subtitle embedding
- SVT-AV1 tile settings

v6.3 Compliance:
- SVT-AV1 tile 2x2 or 1x1
- audio hum_energy_threshold used
- Deterministic intermediate matching
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


def _get_audio_layout(audio_path: str) -> dict:
    """Get audio channel layout info using ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error",
             "-select_streams", "a:0",
             "-show_entries", "stream=channels,channel_layout",
             "-of", "json", audio_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            streams = data.get("streams", [])
            if streams:
                return {
                    "channels": int(streams[0].get("channels", 2)),
                    "layout": streams[0].get("channel_layout", "stereo"),
                }
    except Exception:
        pass
    return {"channels": 2, "layout": "stereo"}


def _build_audio_filter(audio_path: str, params: dict, logger=None,
                          tempo_ratio: float = 1.0,
                          audio_layout: dict = None) -> str:
    """Build audio filter chain with hum removal and denoising.
    
    CRITICAL: bandreject filters must come BEFORE afftdn/arnndn.
    Covers fundamental + 2nd + 3rd harmonic, all width=2Hz.
    
    Scene 1 (duration unchanged): asetpts=PTS at end.
    Scene 2 (duration changed): atempo before asetpts=PTS.
    """
    audio_filters = []

    # V-P-051 fix: FFmpeg version check for bandreject compatibility
    ffmpeg_version_ok = True
    try:
        vr = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10, shell=False)
        if vr.returncode == 0:
            version_line = vr.stdout.splitlines()[0]
            # Extract version number like "4.4.2"
            import re
            vm = re.search(r"ffmpeg version (\d+)\.(\d+)", version_line)
            if vm:
                major, minor = int(vm.group(1)), int(vm.group(2))
                if major < 4 or (major == 4 and minor < 4):
                    ffmpeg_version_ok = False
                    logger.warning("STAGE7_FFMPEG_OLD",
                                  f"FFmpeg {major}.{minor} < 4.4, bandreject width_type=h may fail; will fallback to arnndn")
    except Exception:
        pass

    # Hum removal FIRST (before noise reduction)
    hum_freq = params.get("hum_freq", 50)
    hum_dual = params.get("hum_dual", False)
    auto_detect = params.get("auto_detect_hum", True)

    detected_freq = None
    if auto_detect:
        # Use audio_utils for proper FFT-based detection
        try:
            sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
            from audio_utils import detect_hum_frequency
            hum_result = detect_hum_frequency(audio_path, config=params)
            if hum_result.get("skip_bandreject", False):
                logger.info("STAGE7_HUM_SKIP", "Hum energy below threshold, skipping bandreject")
                detected_freq = None
            else:
                detected_freq = hum_result.get("primary_freq", hum_freq)
        except Exception:
            detected_freq = hum_freq
    
    if detected_freq and ffmpeg_version_ok:
        # Fundamental + 2nd + 3rd harmonic, all width=2Hz
        for f in [detected_freq, detected_freq * 2, detected_freq * 3]:
            audio_filters.append(f"bandreject=f={f}:width_type=h:width=2")
        
        # Dual hum support
        if hum_dual:
            other = 60 if detected_freq == 50 else 50
            for f in [other, other * 2, other * 3]:
                audio_filters.append(f"bandreject=f={f}:width_type=h:width=2")
    elif detected_freq and not ffmpeg_version_ok:
        # Skip bandreject, use arnndn only for hum+noise reduction
        logger.info("STAGE7_HUM_FALLBACK", "Skipping bandreject due to old FFmpeg, relying on arnndn")

    # Noise reduction AFTER bandreject
    noise_reduction = params.get("noise_reduction", True)
    if noise_reduction:
        arnndn_model = params.get("arnndn_model", "")
        # FIX-P7-2: arnndn requires mono input for some CPUs
        # Convert to mono before arnndn, then explicitly restore original layout
        if arnndn_model and Path(arnndn_model).exists():
            audio_filters.append("aformat=channel_layouts=mono")
            audio_filters.append(f"arnndn=m='{arnndn_model}'")
            # Explicitly restore original channel layout inside filter chain
            if audio_layout is None:
                audio_layout = _get_audio_layout(audio_path)
            layout = audio_layout.get("layout", "stereo")
            audio_filters.append(f"aformat=channel_layouts={layout}")
        else:
            audio_filters.append("afftdn=nr=10:nf=-20")

    # V-P-053 fix: compand dynamic range compression
    use_compand = params.get("audio_compand", True)
    if use_compand:
        audio_filters.append("compand=0.3:1:-90/-900|-70/-70|-30/-9|0/-3:6:0:0:0.5")

    # Scene 2: tempo adjustment if video duration changed
    if tempo_ratio != 1.0:
        # atempo valid range [0.5, 100.0]; chain if >2.0
        if 0.5 <= tempo_ratio <= 100.0:
            remaining = tempo_ratio
            atempo_filters = []
            while remaining > 2.0:
                atempo_filters.append("atempo=2.0")
                remaining /= 2.0
            atempo_filters.append(f"atempo={remaining:.4f}")
            audio_filters.extend(atempo_filters)
            logger.info("STAGE7_ATEMPO",
                       f"Scene 2: audio tempo adjusted by ratio {tempo_ratio:.4f}",
                       tempo_ratio=tempo_ratio)
            # CRITICAL log if outside safe 0.8-1.2 range
            if tempo_ratio < 0.8 or tempo_ratio > 1.2:
                logger.critical("STAGE7_ATEMPO_RISK",
                               f"Tempo ratio {tempo_ratio:.4f} outside safe range 0.8-1.2; audio quality risk")
        else:
            logger.error("STAGE7_ATEMPO_FAIL",
                        f"Tempo ratio {tempo_ratio:.4f} outside atempo valid range [0.5, 100.0]")
            return "__INVALID_TEMPO__"

    # Reset PTS for A/V sync (v6.3 §3.10.9)
    audio_filters.append("asetpts=PTS")

    return ",".join(audio_filters) if audio_filters else "anull"


def _get_stream_duration(path: str, stream_type: str = "v") -> float:
    """Get video or audio stream duration using ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", f"{stream_type}:0",
             "-show_entries", "stream=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            dur = r.stdout.strip()
            if dur and dur != "N/A":
                return float(dur)
    except Exception:
        pass
    return 0.0


def _get_source_color_info(video_path: str) -> tuple:
    """Probe source color_range and color_primaries via ffprobe.

    Returns (source_color_range, color_standard) where:
    - source_color_range: 'limited' or 'full'
    - color_standard: 'PAL' or 'NTSC'
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_range,color_primaries",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            streams = data.get("streams", [])
            if streams:
                cr = streams[0].get("color_range", "")
                cp = streams[0].get("color_primaries", "")
                source_range = "full" if cr == "pc" else "limited"
                # Map color_primaries to PAL/NTSC standard
                if cp in ("bt470bg", "5"):
                    color_std = "PAL"
                elif cp in ("smpte170m", "bt470m", "6", "4"):
                    color_std = "NTSC"
                else:
                    color_std = "NTSC"  # Default fallback for old footage
                return source_range, color_std
    except Exception:
        pass
    return "limited", "NTSC"


def _validate_av_sync(muxed_path: str, logger) -> int:
    """Validate A/V sync on the final muxed output file.

    Per v6.3 §3.10.9 / v7.32 §3.10.9:
    - diff > 100ms: WARNING
    - diff > 300ms: sync processing needed
    - diff > 500ms: ERROR/FAILED (return 86)
    """
    video_dur = _get_stream_duration(muxed_path, "v")
    audio_dur = _get_stream_duration(muxed_path, "a")

    if video_dur == 0 or audio_dur == 0:
        logger.warning("STAGE7_SYNC_SKIP", "Cannot determine durations for sync check")
        return 0  # Continue anyway

    diff_ms = abs(video_dur - audio_dur) * 1000

    if diff_ms > 500:
        logger.error("STAGE7_SYNC_FAIL",
                    f"AV sync difference {diff_ms:.0f}ms exceeds 500ms threshold",
                    video_duration=video_dur, audio_duration=audio_dur, diff_ms=diff_ms)
        return 86
    elif diff_ms > 300:
        logger.warning("STAGE7_SYNC_WARN",
                      f"AV sync difference {diff_ms:.0f}ms exceeds 300ms threshold, sync processing recommended",
                      video_duration=video_dur, audio_duration=audio_dur, diff_ms=diff_ms)
        return 0  # Continue but warn
    elif diff_ms > 100:
        logger.info("STAGE7_SYNC_INFO",
                   f"AV sync difference {diff_ms:.0f}ms exceeds 100ms threshold",
                   video_duration=video_dur, audio_duration=audio_dur, diff_ms=diff_ms)
        return 0  # Continue

    return 0


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

    # Find input - deterministic by mtime
    input_source = None
    for pat in ["step6_grain*", "step5_face*", "step4_postprocess*", "step3_upscale*", "step2_traditional*", "step1_preprocess*"]:
        pattern = f"{job_id}_{Path(input_name).stem}_{pat}"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            input_source = candidates[0]
            break

    if not input_source:
        logger.error("STAGE7_NO_INPUT", "No input found")
        return 82

    output_path = params.get("output_path", "")
    if not output_path:
        output_dir = params.get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir,
            f"{Path(input_name).stem}_restored.mkv")

    # WSL2 redirect: if running in WSL and Windows tools need Windows paths
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    platform = params.get("platform", "auto")
    redirected_path, original_output_path, _, _ = wsl_redirect_path(
        output_path, no_wsl_redirect=no_wsl_redirect, target_platform=platform
    )
    if original_output_path:
        logger.info("STAGE7_WSL_REDIRECT",
                   f"WSL redirect: {output_path} -> {redirected_path}")
        output_path = redirected_path

    logger.info("STAGE7_START", "Final encoding",
                input=input_source, output=output_path)

    # SVT-AV1 parameters
    crf = params.get("crf", 24)
    preset = params.get("preset", 4)
    pix_fmt = params.get("pix_fmt", "yuv420p10le")
    color_range = params.get("color_range", "limited")

    # FIX-P7-6: CRF adaptive logic per v6.32 §7.10 / v7.32 §3.10.20
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "json", input_source],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            info = json.loads(r.stdout)
            stream = info["streams"][0]
            vid_w = stream.get("width", 1920)
            vid_h = stream.get("height", 1080)

            # Base CRF by resolution
            if vid_w > 1920 or vid_h > 1080:
                crf = 26
            else:
                crf = 24  # 480p/576p/720p unified default

            # Grain adaptive (if upstream stages passed grain info)
            grain_strength = params.get("detected_grain_strength", None)
            if grain_strength is None and "noise_score" in params:
                grain_strength = params["noise_score"] * 10
            if grain_strength is not None:
                if grain_strength > 5.0:
                    crf -= 1
                    logger.info("STAGE7_CRF_GRAIN", f"High grain {grain_strength:.2f}, CRF -> {crf}")
                elif grain_strength < 1.0:
                    crf += 1
                    logger.info("STAGE7_CRF_CLEAN", f"Low grain {grain_strength:.2f}, CRF -> {crf}")

            detected_class = params.get("detected_class", "").lower()
            if "animation" in detected_class or "anime" in detected_class:
                crf += 2
                logger.info("STAGE7_ANIMATION_CRF", f"Animation detected, CRF adjusted to {crf}")
            logger.info("STAGE7_ADAPTIVE",
                        f"Adaptive encoding: {vid_w}x{vid_h} -> preset={preset}, crf={crf}")
    except Exception as e:
        logger.warning("STAGE7_ADAPTIVE_SKIP", f"Adaptive preset/CRF failed: {e}")

    # v6.3: tile settings
    tile_rows = params.get("svt_tile_rows", 2)
    tile_cols = params.get("svt_tile_cols", 2)

    # Video encoding with audio sync flags
    # BUG-041 fix: defer svtav1-params construction until all parameters are known
    video_cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-fflags", "+genpts",  # Generate missing presentation timestamps
        "-i", input_source,
        "-c:v", "libsvtav1",
        "-crf", str(crf),
        "-preset", str(preset),
        "-pix_fmt", pix_fmt,
    ]

    # FIX-P7-1: BT.601→BT.709 color space conversion with range handling
    source_color_range, color_standard = _get_source_color_info(args.input)
    matrix_in = "bt470bg" if color_standard == "PAL" else "smpte170m"
    rangein = "full" if source_color_range == "full" else "limited"
    vf_parts = []
    if not params.get("colorspace_converted", False):
        vf_parts.append(f"zscale=matrixin={matrix_in}:matrix=bt709:rangein={rangein}:range=limited")
        logger.info("STAGE7_COLORSPACE",
                   f"BT.601({color_standard})->BT.709 zscale: matrixin={matrix_in}, rangein={rangein}")

    # FIX-P7-7: 16bit→10bit error diffusion dithering before final encode
    input_depth = params.get("input_bit_depth", 16)
    if input_depth > 10:
        vf_parts.append("format=yuv444p16,scale=out_range=limited:out_format=yuv444p10,format=yuv420p10le")
        logger.info("STAGE7_DITHER", "Applying 16bit->10bit error diffusion dithering via YUV444")

    if vf_parts:
        video_cmd.extend(["-vf", ",".join(vf_parts)])

    # Scene detection for audio handling (v6.3 §3.10.9)
    # Scene 1: duration unchanged -> asetpts=PTS only
    # Scene 2: duration changed -> atempo adjustment needed
    original_duration = params.get("duration", 0)
    processed_duration = _get_stream_duration(input_source, "v")
    tempo_ratio = 1.0
    if original_duration > 0 and processed_duration > 0:
        diff_ms = abs(original_duration - processed_duration) * 1000
        if diff_ms > 300:
            tempo_ratio = original_duration / processed_duration
            logger.info("STAGE7_SCENE2",
                       f"Detected Scene 2: video duration changed by {diff_ms:.0f}ms (ratio={tempo_ratio:.4f})")
        else:
            logger.info("STAGE7_SCENE1",
                       f"Detected Scene 1: duration diff {diff_ms:.0f}ms <= 300ms")
    else:
        logger.info("STAGE7_SCENE1", "Duration info unavailable, defaulting to Scene 1")

    # Audio (extract from original if available)
    # V-P-049 fix: match stage0 output naming convention
    audio_path = os.path.join(temp_dir, f"{job_id}_{input_name}.audio")
    # Fallback: also check the old naming for backward compatibility
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        audio_path_fallback = os.path.join(temp_dir, f"{job_id}_{input_name}_audio_raw.wav")
        if os.path.exists(audio_path_fallback) and os.path.getsize(audio_path_fallback) > 0:
            audio_path = audio_path_fallback
    has_audio = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
    if has_audio:
        # FIX-P7-2: multichannel preserve + explicit layout restore in filter chain
        audio_layout = _get_audio_layout(audio_path)
        channels = audio_layout.get("channels", 2)
        audio_filter = _build_audio_filter(audio_path, params, logger,
                                             tempo_ratio=tempo_ratio,
                                             audio_layout=audio_layout)
        # Invalid tempo check
        if audio_filter == "__INVALID_TEMPO__":
            logger.error("STAGE7_AUDIO_FAIL", "Invalid tempo ratio, cannot process audio")
            return 86

        multichannel_preserve = params.get("multichannel_preserve", True)
        if multichannel_preserve and channels > 2:
            audio_codec = "flac"
            audio_bitrate = None
            logger.info("STAGE7_MULTICHANNEL",
                       f"Detected {channels} channels, using FLAC to preserve layout")
        else:
            audio_codec = params.get("audio_codec", params.get("codec", "aac"))
            audio_bitrate = params.get("audio_bitrate", params.get("bitrate", "320k"))

        video_cmd.extend([
            "-i", audio_path,
            "-af", audio_filter,
            "-map", "0:v:0",  # Map video from first input
            "-map", "1:a:0",  # Map audio from second input
        ])
        # FIX-P7-2: Layout restored inside audio filter chain via aformat;
        # -ac fallback removed to avoid overriding explicit layout

        # Select codec per config (v6.3 §3.10.9)
        if audio_codec == "aac":
            video_cmd.extend(["-c:a", "aac", "-b:a", audio_bitrate])
        elif audio_codec == "flac":
            video_cmd.extend(["-c:a", "flac"])
        else:
            video_cmd.extend(["-c:a", audio_codec])
            if audio_bitrate:
                video_cmd.extend(["-b:a", audio_bitrate])
    else:
        # FIX-P7-8: Log when source has no audio stream
        logger.info("STAGE7_NO_AUDIO", "Source has no audio stream, outputting video only")
        video_cmd.append("-an")

    # Color metadata (v6.3: double marking)
    # SVT-AV1 uses numeric values, not string names
    color_primaries = params.get("color_primaries", "bt709")
    color_trc = params.get("color_trc", "bt709")
    colorspace = params.get("colorspace", "bt709")

    # FIX-P7-9: FFmpeg -color_range uses 'tv'/'pc', not 'limited'/'full'
    ffmpeg_range = "tv" if color_range == "limited" else "pc"

    # Map color names to SVT-AV1 numeric values
    color_map = {
        "bt709": "1", "bt2020": "9", "bt470bg": "5", "smpte170m": "6",
        "bt601": "6", "bt470m": "4"
    }
    svt_primaries = color_map.get(color_primaries, "1")
    svt_trc = color_map.get(color_trc, "1")
    svt_matrix = color_map.get(colorspace, "1")
    svt_range = "0" if color_range == "limited" else "1"

    # FFmpeg global color metadata (string names)
    video_cmd.extend([
        "-color_primaries", color_primaries,
        "-color_trc", color_trc,
        "-colorspace", colorspace,
        "-color_range", ffmpeg_range,
    ])
    
    # BUG-041 fix: build svtav1-params once after all parameters are known
    svt_params = f"tile-rows={tile_rows}:tile-columns={tile_cols}"
    svt_params += f":color-primaries={svt_primaries}:transfer-characteristics={svt_trc}"
    svt_params += f":matrix-coefficients={svt_matrix}:color-range={svt_range}"
    video_cmd.extend(["-svtav1-params", svt_params])

    # Metadata and output
    video_cmd.extend([
        "-metadata", "restored=true",
        "-metadata", f"restored_at={__import__('datetime').datetime.now().isoformat()}",
        "-y", output_path,
    ])

    try:
        r = subprocess.run(video_cmd, capture_output=True, text=True,
                          timeout=3600*2, shell=False)
        if r.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            # FIX-P7-4: Validate A/V sync on final muxed output (not separate files)
            if has_audio:
                sync_result = _validate_av_sync(output_path, logger)
                if sync_result != 0:
                    return sync_result
            
            # WSL redirect: copy back to original Linux path
            if original_output_path:
                if wsl_copy_back(output_path, original_output_path):
                    logger.info("STAGE7_WSL_COPYBACK",
                               f"Copied result back to {original_output_path}")
                    output_path = original_output_path
                else:
                    logger.error("STAGE7_WSL_COPYBACK_FAIL",
                                f"Failed to copy {output_path} back to {original_output_path}")
                    return 82
            
            logger.info("STAGE7_DONE", "Encoding complete", output=output_path)
            return 0
        else:
            logger.error("STAGE7_FAILED", f"Encoding failed: {r.stderr[:300]}")
            return 84
    except subprocess.TimeoutExpired:
        logger.error("STAGE7_TIMEOUT", "Encoding timed out")
        return 87
    except Exception as e:
        logger.error("STAGE7_ERROR", str(e))
        return 82


if __name__ == "__main__":
    sys.exit(main() or 0)
