#!/usr/bin/env python3
"""
Audio Processing Utilities
Handles audio extraction, hum detection/removal, denoising, and sync verification.
"""

import subprocess
import numpy as np
from typing import Dict, Any, Tuple, Optional


def extract_audio(input_path: str, output_path: str,
                  codec: str = "copy") -> bool:
    """Extract audio stream from video without processing.

    Args:
        input_path: Source video path
        output_path: Output audio path
        codec: Audio codec ("copy" for no re-encode)
    """
    try:
        cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y",
               "-i", input_path, "-vn"]
        if codec == "copy":
            cmd.extend(["-acodec", "copy"])
        cmd.append(output_path)

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, shell=False)
        return r.returncode == 0
    except Exception:
        return False


def get_audio_duration(path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def get_audio_info(path: str) -> Dict[str, Any]:
    """Get audio stream info: sample_rate, channels, channel_layout, codec.

    Returns None for values if ffprobe fails or no audio stream found.
    Callers must check for None before using.
    """
    info: Dict[str, Any] = {"sample_rate": None, "channels": None, "channel_layout": None, "codec": None}
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels,channel_layout,codec_name",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            import json
            data = json.loads(r.stdout)
            stream = data.get("streams", [{}])[0]
            if stream.get("sample_rate"):
                info["sample_rate"] = int(stream["sample_rate"])
            if stream.get("channels"):
                info["channels"] = int(stream["channels"])
            if stream.get("channel_layout"):
                info["channel_layout"] = stream["channel_layout"]
            if stream.get("codec_name"):
                info["codec"] = stream["codec_name"]
    except Exception:
        pass
    return info


def detect_hum_frequency(audio_path: str, config: Dict[str, Any]) -> Tuple[Optional[int], float, bool]:
    """Detect power line hum frequency (50Hz or 60Hz) from audio.

    Analyzes first 30 seconds (or full audio if shorter).
    Returns (detected_freq or None, energy_ratio, is_dual_hum).

    Dual hum detection: if both 50Hz and 60Hz bands exceed thresholds,
    is_dual_hum is True and the primary (stronger) freq is returned.
    """
    auto_detect_sec = config.get("hum_auto_detect_seconds", 30)
    energy_threshold = config.get("hum_energy_threshold", 0.005)
    dual_ratio_threshold = config.get("hum_dual_ratio_threshold", 3.0)

    try:
        # Get audio info
        info = get_audio_info(audio_path)
        sample_rate = info.get("sample_rate")
        if sample_rate is None:
            sample_rate = 48000  # safe fallback for hum detection
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error",
             "-i", audio_path,
             "-t", str(auto_detect_sec),
             "-ar", str(sample_rate),
             "-ac", "1",  # mono
             "-f", "s16le", "-"],
            capture_output=True, timeout=60, shell=False
        )

        if r.returncode != 0:
            return None, 0.0, False

        samples = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float64)
        if len(samples) == 0:
            return None, 0.0, False

        # Normalize
        samples = samples / 32768.0

        # Adaptive FFT window: 2^ceil(log2(sample_rate * 0.085))
        win_size = 2 ** int(np.ceil(np.log2(sample_rate * 0.085)))
        hop = win_size // 2  # 50% overlap

        # Compute STFT power
        total_power = 0.0
        band_45_65_power = 0.0
        band_48_52_power = 0.0
        band_58_62_power = 0.0
        band_45_180_power = 0.0

        freqs = np.fft.rfftfreq(win_size, 1.0 / sample_rate)
        idx_45 = np.argmin(np.abs(freqs - 45))
        idx_65 = np.argmin(np.abs(freqs - 65))
        idx_48 = np.argmin(np.abs(freqs - 48))
        idx_52 = np.argmin(np.abs(freqs - 52))
        idx_58 = np.argmin(np.abs(freqs - 58))
        idx_62 = np.argmin(np.abs(freqs - 62))
        idx_180 = np.argmin(np.abs(freqs - 180))

        # Hann window
        hann_win = np.hanning(win_size)

        pos = 0
        frame_count = 0
        while pos + win_size <= len(samples):
            frame = samples[pos:pos + win_size] * hann_win
            fft = np.fft.rfft(frame)
            power = np.abs(fft) ** 2

            total_power += np.sum(power)
            band_45_65_power += np.sum(power[idx_45:idx_65 + 1])
            band_48_52_power += np.sum(power[idx_48:idx_52 + 1])
            band_58_62_power += np.sum(power[idx_58:idx_62 + 1])
            band_45_180_power += np.sum(power[idx_45:idx_180 + 1])

            pos += hop
            frame_count += 1

        if total_power == 0 or frame_count == 0:
            return None, 0.0, False

        energy_ratio = band_45_65_power / total_power

        if energy_ratio < energy_threshold:
            return None, energy_ratio, False

        # Compare 48-52Hz vs 58-62Hz
        ratio_50 = band_48_52_power / total_power
        ratio_60 = band_58_62_power / total_power
        low_freq_ratio = band_45_180_power / total_power

        # Dual hum detection
        is_dual = False
        if (ratio_50 >= energy_threshold and ratio_60 >= energy_threshold and
            ratio_50 >= low_freq_ratio * 0.3 and ratio_60 >= low_freq_ratio * 0.3 and
            max(ratio_50, ratio_60) / min(ratio_50, ratio_60) < dual_ratio_threshold):
            is_dual = True

        if ratio_50 > ratio_60:
            return 50, energy_ratio, is_dual
        else:
            return 60, energy_ratio, is_dual

    except Exception:
        return None, 0.0, False


def build_hum_filter_chain(freq: int, config: Dict[str, Any],
                             secondary_freq: Optional[int] = None) -> str:
    """Build FFmpeg bandreject filter chain for hum removal.

    Creates multiple bandreject filters for fundamental + harmonics.
    e.g., for 50Hz: 50, 100, 150Hz

    If secondary_freq is provided and hum_dual is True, both filter chains
    are concatenated (primary first, then secondary).
    """
    harmonics = config.get("hum_harmonics", 3)
    bandwidth = config.get("hum_bandwidth", 2)
    width_type = config.get("hum_width_type", "h")

    def _chain(f0: int) -> list:
        return [f"bandreject=f={f0 * h}:width_type={width_type}:width={bandwidth}"
                for h in range(1, harmonics + 1)]

    filters = _chain(freq)

    if secondary_freq is not None and config.get("hum_dual", False):
        filters.extend(_chain(secondary_freq))

    return ",".join(filters)


def process_audio(audio_input: str, audio_output: str,
                  config: Dict[str, Any], video_duration: float = 0,
                  logger=None) -> bool:
    """Process audio: hum removal + denoising.

    Args:
        audio_input: Input audio path (from stage0)
        audio_output: Output processed audio path
        config: Audio section from config.yaml
        video_duration: Final video duration for sync check
    """
    hum_freq = config.get("hum_freq", 50)
    auto_detect = True  # Always try auto-detection first

    # Detect hum if auto-detection enabled
    if auto_detect:
        detected_freq, energy_ratio, is_dual = detect_hum_frequency(audio_input, config)
        if detected_freq is None:
            # No significant hum detected, skip bandreject
            hum_filters = ""
        else:
            secondary = None
            if is_dual:
                secondary = 60 if detected_freq == 50 else 50
                if config.get("hum_dual", False):
                    if logger:
                        logger.warning("HUM_DUAL", "Dual hum detected (50Hz + 60Hz), processing both bands")
                else:
                    if logger:
                        logger.warning("HUM_DUAL", "Dual hum detected, processing primary band only. Set hum_dual=true to process both (risky).")
            hum_filters = build_hum_filter_chain(detected_freq, config, secondary)
    else:
        hum_filters = build_hum_filter_chain(hum_freq, config)

    # Build denoise filter
    denoise = config.get("denoise", "afftdn")
    if denoise == "afftdn":
        denoise_filter = "afftdn=nf=-25:tnnn=80"
    elif denoise == "arnndn":
        model_50 = config.get("arnndn_model_50hz", "arnndn_50Hz.pb")
        model_60 = config.get("arnndn_model_60hz", "arnndn_60Hz.pb")
        # Use detected_freq if available, fallback to config hum_freq
        _freq = hum_freq
        if auto_detect:
            _freq = detected_freq if detected_freq is not None else hum_freq
        model_file = model_50 if _freq == 50 else model_60
        denoise_filter = f"arnndn=m={model_file}"
    else:
        denoise_filter = "afftdn"

    # Build full filter chain
    filters = []
    if hum_filters:
        filters.append(hum_filters)
    filters.append(denoise_filter)
    # Reset PTS for sync
    filters.append("asetpts=PTS")

    filter_chain = ",".join(filters)

    # Determine codec
    codec = config.get("codec", "aac")
    bitrate = config.get("bitrate", "320k")

    try:
        cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y",
               "-i", audio_input,
               "-af", filter_chain]

        if codec == "aac":
            cmd.extend(["-c:a", "aac", "-b:a", bitrate])
        elif codec == "flac":
            cmd.extend(["-c:a", "flac"])
        else:
            cmd.extend(["-c:a", codec])

        cmd.append(audio_output)

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, shell=False)
        return r.returncode == 0
    except Exception:
        return False


def check_av_sync(video_path: str, audio_path: str) -> Tuple[float, str]:
    """Check audio-video duration difference.

    Returns:
        (duration_diff_ms, status)
        status: "OK", "WARNING", "ERROR"
    """
    try:
        # Get video duration
        rv = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        video_dur = float(rv.stdout.strip())

        # Get audio duration
        ra = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        audio_dur = float(ra.stdout.strip())

        diff_ms = abs(video_dur - audio_dur) * 1000

        if diff_ms > 500:
            return diff_ms, "ERROR"
        elif diff_ms > 300:
            return diff_ms, "WARNING"
        elif diff_ms > 100:
            return diff_ms, "WARNING"
        else:
            return diff_ms, "OK"

    except Exception as e:
        return -1, f"ERROR: {e}"


def build_atempo_chain(ratio: float) -> str:
    """Build atempo filter chain for audio speed adjustment.

    atempo range: [0.5, 100.0], chainable for > 2.0x
    """
    if ratio < 0.5 or ratio > 100.0:
        raise ValueError(f"atempo ratio {ratio} out of range [0.5, 100.0]")

    filters = []
    remaining = ratio
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")

    return ",".join(filters)
