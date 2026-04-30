#!/usr/bin/env python3
"""
Stage 7: Final Encoding + Audio Processing + Muxing
- Audio processing: hum removal (bandreject) + denoising (afftdn/arnndn)
- Video encoding: SVT-AV1 or x265, 10bit, BT.709
- A/V sync verification
- Final MKV muxing

Exit codes: 0=success, 84=encoding error
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
from audio_utils import process_audio, check_av_sync, build_atempo_chain  # noqa: E402
from checkpoint import wsl_copy_back  # noqa: E402


def _get_media_duration(path: str) -> float:
    """Get media duration in seconds using ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return 0.0


def _adaptive_crf(base_crf: int, params: dict) -> int:
    """Adjust CRF based on grain strength heuristic.

    High grain (g1str >= 1.0) -> CRF -1 (higher bitrate)
    Low grain (g1str <= 0.5 or disabled) -> CRF +1 (lower bitrate)
    """
    grain_enabled = params.get("grain_enabled", True)
    g1str = params.get("g1str", 1.0)
    if not grain_enabled or g1str <= 0.5:
        return base_crf + 1
    elif g1str >= 1.0:
        return max(1, base_crf - 1)
    return base_crf


def build_video_encode_cmd(input_video: str, output_video: str,
                           params: dict) -> list:
    """Build FFmpeg command for video encoding."""
    encoder = params.get("encoder", "svtav1")
    base_crf = params.get("crf", 24)
    crf = _adaptive_crf(base_crf, params)
    preset = params.get("preset", 4)
    pix_fmt = params.get("pix_fmt", "yuv420p10le")

    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", input_video,
        "-an",  # No audio in this pass
        "-pix_fmt", pix_fmt,
    ]

    if encoder == "svtav1":
        cmd.extend([
            "-c:v", "libsvtav1",
            "-crf", str(crf),
            "-preset", str(preset),
            "-svtav1-params", "color-primaries=1:transfer-characteristics=1:matrix-coefficients=1:color-range=0",
        ])
    elif encoder == "x265":
        x265_base = params.get("x265_crf", params.get("crf", 18))
        x265_crf = _adaptive_crf(x265_base, params)
        cmd.extend([
            "-c:v", "libx265",
            "-crf", str(x265_crf),
            "-preset", "slow",
        ])
    else:
        cmd.extend(["-c:v", "libsvtav1", "-crf", str(crf)])

    # Color metadata (container level)
    cmd.extend([
        "-color_range", "tv",
        "-color_primaries:v", "bt709",
        "-color_trc:v", "bt709",
        "-colorspace:v", "bt709",
    ])

    cmd.append(output_video)
    return cmd


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

    # Find video input (stage6 > stage4 > stage1)
    import glob
    video_source = None
    for stage_pat in ["step6_grain*", "step4_postprocess*", "step1_preprocess*"]:
        pattern = f"{job_id}_{Path(input_name).stem}_{stage_pat}"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        if candidates:
            video_source = candidates[0]
            break

    if not video_source:
        logger.error("STAGE7_NO_VIDEO", "No video input found")
        return 82

    # Find audio input (stage0)
    audio_pattern = f"{job_id}_{Path(input_name).stem}_audio_raw*"
    audio_candidates = glob.glob(os.path.join(temp_dir, audio_pattern))
    audio_source = audio_candidates[0] if audio_candidates else None

    output_path = params.get("output_path", "")
    if not output_path:
        output_dir = params.get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{Path(input_name).stem}_restored.mkv")

    logger.info("STAGE7_START", "Final encoding",
                video_source=video_source,
                audio_source=audio_source,
                output=output_path)

    # Step 1: Process audio if available
    processed_audio = None
    if audio_source and os.path.exists(audio_source):
        audio_config = {
            "hum_freq": params.get("hum_freq", 50),
            "hum_harmonics": params.get("hum_harmonics", 3),
            "hum_bandwidth": params.get("hum_bandwidth", 2),
            "hum_width_type": params.get("hum_width_type", "h"),
            "denoise": params.get("denoise", "afftdn"),
            "codec": params.get("codec", "aac"),
            "bitrate": params.get("bitrate", "320k"),
            "arnndn_model_50hz": params.get("arnndn_model_50hz", "arnndn_50Hz.pb"),
            "arnndn_model_60hz": params.get("arnndn_model_60hz", "arnndn_60Hz.pb"),
            "hum_auto_detect_seconds": params.get("hum_auto_detect_seconds", 30),
            "hum_energy_threshold": params.get("hum_energy_threshold", 0.005),
        }

        processed_audio = os.path.join(temp_dir,
            f"{job_id}_{Path(input_name).stem}_audio_processed.aac")

        audio_ok = process_audio(audio_source, processed_audio, audio_config, logger=logger)
        if not audio_ok:
            logger.warning("STAGE7_AUDIO_FAIL", "Audio processing failed, will use original")
            processed_audio = audio_source

        # Scene 2: if video duration differs significantly from audio,
        # use atempo to linearly adjust audio speed (v6.3/v7.3)
        video_dur = _get_media_duration(video_source)
        audio_dur = _get_media_duration(processed_audio) if processed_audio else 0.0
        if video_dur > 0 and audio_dur > 0:
            diff_ms = abs(video_dur - audio_dur) * 1000
            ratio = video_dur / audio_dur
            if diff_ms > 300 and (ratio < 0.8 or ratio > 1.2):
                logger.warning("STAGE7_SYNC_SCENE2",
                              f"A/V duration diff={diff_ms:.0f}ms, applying atempo ratio={ratio:.3f}")
                try:
                    atempo_filter = build_atempo_chain(ratio)
                    temp_atempo = processed_audio + ".atempo.aac"
                    atempo_cmd = [
                        "ffmpeg", "-hide_banner", "-v", "error", "-y",
                        "-i", processed_audio,
                        "-af", atempo_filter,
                        "-c:a", params.get("codec", "aac"),
                        "-b:a", params.get("bitrate", "320k"),
                        temp_atempo,
                    ]
                    r = subprocess.run(atempo_cmd, capture_output=True,
                                      timeout=300, shell=False)
                    if r.returncode == 0 and os.path.exists(temp_atempo):
                        processed_audio = temp_atempo
                        logger.info("STAGE7_ATEMPO", f"atempo applied ratio={ratio:.3f}")
                    else:
                        logger.warning("STAGE7_ATEMPO_FAIL", "atempo adjustment failed")
                except Exception as e:
                    logger.warning("STAGE7_ATEMPO_ERR", str(e))
    else:
        logger.info("STAGE7_NO_AUDIO", "No audio stream found")

    # Step 2: Encode video
    video_temp = os.path.join(temp_dir,
        f"{job_id}_{Path(input_name).stem}_video_temp.mkv")

    try:
        cmd = build_video_encode_cmd(video_source, video_temp, params)
        r = subprocess.run(cmd, capture_output=True, timeout=7200, shell=False)

        if r.returncode != 0:
            logger.error("STAGE7_ENCODE_FAIL",
                        f"Video encoding failed: {r.stderr.decode()[:500]}")
            return 84

    except subprocess.TimeoutExpired:
        logger.error("STAGE7_ENCODE_TIMEOUT", "Video encoding timed out")
        return 87
    except Exception as e:
        logger.error("STAGE7_ENCODE_ERROR", str(e))
        return 84

    # Step 3: Mux video + audio
    try:
        mux_cmd = [
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-i", video_temp,
        ]

        if processed_audio and os.path.exists(processed_audio):
            mux_cmd.extend(["-i", processed_audio])
            # Copy audio or re-encode
            audio_codec = params.get("codec", "aac")
            if audio_codec == "copy":
                mux_cmd.extend(["-c:a", "copy"])
            else:
                bitrate = params.get("bitrate", "320k")
                mux_cmd.extend(["-c:a", audio_codec, "-b:a", bitrate])
        else:
            mux_cmd.append("-an")

        mux_cmd.extend([
            "-c:v", "copy",
            "-fflags", "+genpts",
            "-movflags", "+faststart",
            output_path
        ])

        r = subprocess.run(mux_cmd, capture_output=True, timeout=600, shell=False)

        if r.returncode != 0:
            logger.error("STAGE7_MUX_FAIL",
                        f"Mux failed: {r.stderr.decode()[:500]}")
            return 84

    except Exception as e:
        logger.error("STAGE7_MUX_ERROR", str(e))
        return 84

    # Step 4: A/V sync check
    if processed_audio and os.path.exists(processed_audio):
        try:
            diff_ms, status = check_av_sync(output_path, processed_audio)
            logger.info("STAGE7_SYNC", f"A/V sync: {diff_ms:.1f}ms ({status})")
            if status == "ERROR":
                logger.error("STAGE7_SYNC_FAIL", "A/V sync error detected")
        except Exception:
            pass

    # Step 5: Verify output
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info("STAGE7_DONE", "Final output complete", output=output_path)

        # WSL copy back if needed
        if params.get("platform") == "wsl2" and not params.get("no_wsl_redirect"):
            orig_path = params.get("original_output_path", output_path)
            if orig_path != output_path:
                if not wsl_copy_back(output_path, orig_path):
                    logger.error("STAGE7_WSL_FAIL",
                                f"Failed to copy output back to {orig_path}")
                    return 82  # IO error

        return 0
    else:
        logger.error("STAGE7_NO_OUTPUT", "Output file not created")
        return 84


if __name__ == "__main__":
    sys.exit(main() or 0)
