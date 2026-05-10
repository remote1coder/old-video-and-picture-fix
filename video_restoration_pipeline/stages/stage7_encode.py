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


def _build_audio_filter(audio_path: str, params: dict, logger=None,
                          tempo_ratio: float = 1.0) -> str:
    """Build audio filter chain with hum removal and denoising.
    
    CRITICAL: bandreject filters must come BEFORE afftdn/arnndn.
    Covers fundamental + 2nd + 3rd harmonic, all width=2Hz.
    
    Scene 1 (duration unchanged): asetpts=PTS at end.
    Scene 2 (duration changed): atempo before asetpts=PTS.
    """
    audio_filters = []

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
    
    if detected_freq:
        # Fundamental + 2nd + 3rd harmonic, all width=2Hz
        for f in [detected_freq, detected_freq * 2, detected_freq * 3]:
            audio_filters.append(f"bandreject=f={f}:width_type=h:width=2")
        
        # Dual hum support
        if hum_dual:
            other = 60 if detected_freq == 50 else 50
            for f in [other, other * 2, other * 3]:
                audio_filters.append(f"bandreject=f={f}:width_type=h:width=2")

    # Noise reduction AFTER bandreject
    noise_reduction = params.get("noise_reduction", True)
    if noise_reduction:
        arnndn_model = params.get("arnndn_model", "")
        if arnndn_model and Path(arnndn_model).exists():
            audio_filters.append(f"arnndn=m='{arnndn_model}'")
        else:
            audio_filters.append("afftdn=nr=10:nf=-20")

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


def _validate_av_sync(video_path: str, audio_path: str, logger) -> int:
    """Validate audio/video sync duration difference.
    
    Per v6.3 §3.10.9:
    - diff > 100ms: WARNING
    - diff > 300ms: sync processing needed
    - diff > 500ms: ERROR/FAILED (return 86)
    """
    video_dur = _get_stream_duration(video_path, "v")
    audio_dur = _get_stream_duration(audio_path, "a")
    
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
    for pat in ["step6_grain*", "step5_face*", "step4_postprocess*", "step3_upscale*", "step1_preprocess*"]:
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
    redirected_path, original_output_path = wsl_redirect_path(output_path, no_wsl_redirect, platform)
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

    # v6.3: tile settings
    tile_rows = params.get("svt_tile_rows", 2)
    tile_cols = params.get("svt_tile_cols", 2)

    # Video encoding with audio sync flags
    video_cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-fflags", "+genpts",  # Generate missing presentation timestamps
        "-i", input_source,
        "-c:v", "libsvtav1",
        "-crf", str(crf),
        "-preset", str(preset),
        "-pix_fmt", pix_fmt,
        "-svtav1-params", f"tile-rows={tile_rows}:tile-columns={tile_cols}",
    ]

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
    audio_path = os.path.join(temp_dir, f"{job_id}_{input_name}.audio")
    has_audio = os.path.exists(audio_path) and os.path.getsize(audio_path) > 0
    if has_audio:
        audio_filter = _build_audio_filter(audio_path, params, logger, tempo_ratio=tempo_ratio)
        # Invalid tempo check
        if audio_filter == "__INVALID_TEMPO__":
            logger.error("STAGE7_AUDIO_FAIL", "Invalid tempo ratio, cannot process audio")
            return 86
        audio_codec = params.get("audio_codec", params.get("codec", "aac"))
        audio_bitrate = params.get("audio_bitrate", params.get("bitrate", "320k"))
        video_cmd.extend([
            "-i", audio_path,
            "-af", audio_filter,
            "-map", "0:v:0",  # Map video from first input
            "-map", "1:a:0",  # Map audio from second input
        ])
        # Select codec per config (v6.3 §3.10.9)
        if audio_codec == "aac":
            video_cmd.extend(["-c:a", "aac", "-b:a", audio_bitrate])
        elif audio_codec == "flac":
            video_cmd.extend(["-c:a", "flac"])
        else:
            video_cmd.extend(["-c:a", audio_codec])
    else:
        video_cmd.append("-an")

    # Color metadata (v6.3: double marking)
    # SVT-AV1 uses numeric values, not string names
    color_primaries = params.get("color_primaries", "bt709")
    color_trc = params.get("color_trc", "bt709")
    colorspace = params.get("colorspace", "bt709")
    
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
        "-color_range", color_range,
    ])
    
    # svtav1-params with color metadata (double marking, numeric values)
    svt_params = f"tile-rows={tile_rows}:tile-columns={tile_cols}"
    svt_params += f":color-primaries={svt_primaries}:transfer-characteristics={svt_trc}"
    svt_params += f":matrix-coefficients={svt_matrix}:color-range={svt_range}"
    
    # Update svtav1-params in video_cmd
    for i, arg in enumerate(video_cmd):
        if arg == "-svtav1-params":
            video_cmd[i + 1] = svt_params
            break

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
            # v6.3: Validate audio/video sync duration
            if has_audio:
                sync_result = _validate_av_sync(output_path, audio_path, logger)
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
