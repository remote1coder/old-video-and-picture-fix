#!/usr/bin/env python3
"""
Audio Restoration Utilities
Shared audio processing for both video and photo pipelines.
"""

import sys
from pathlib import Path
import subprocess
from typing import Dict, Any, List

import math

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from logger import get_logger  # noqa: E402


def _compute_fft_window_size(sample_rate: int) -> int:
    """Compute adaptive FFT window size for ~85ms duration.
    
    Formula: 2^ceil(log2(sample_rate * 0.085))
    Examples: 48kHz -> 4096, 44.1kHz -> 4096, 96kHz -> 8192
    """
    target_samples = sample_rate * 0.085
    return int(2 ** math.ceil(math.log2(target_samples)))


def detect_hum_frequency(audio_path: str, sample_rate: int = 48000,
                         config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Detect 50Hz or 60Hz hum in audio using FFT analysis.
    
    Uses adaptive window (85ms), Hann window, 50% overlap.
    Returns dict with has_hum, primary_freq, hum_ratio, is_dual_hum, skip_bandreject.
    """
    import numpy as np
    
    if config is None:
        config = {}
    
    analyze_seconds = config.get("hum_auto_detect_seconds", 30)
    
    result = {
        "has_hum": False,
        "primary_freq": 50,
        "hum_ratio": 0.0,
        "is_dual_hum": False,
        "skip_bandreject": True,
    }
    
    try:
        # Extract audio to raw PCM for FFT analysis
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-t", str(analyze_seconds),
            "-ar", str(sample_rate),
            "-ac", "1",  # mono for analysis
            "-f", "s16le", "-"
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=60, shell=False)
        if proc.returncode != 0:
            return result
        
        samples = np.frombuffer(proc.stdout, dtype=np.int16)
        if len(samples) < 1024:
            return result
        
        # Adaptive FFT window: 2^ceil(log2(sample_rate * 0.085))
        n_fft = _compute_fft_window_size(sample_rate)
        hop_length = n_fft // 2  # 50% overlap
        
        # Hann window
        window = np.hanning(n_fft)
        
        num_frames = (len(samples) - n_fft) // hop_length + 1
        if num_frames < 1:
            return result
        
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        power_spectrum = np.zeros(len(freqs))
        
        for i in range(num_frames):
            start = i * hop_length
            frame = samples[start:start + n_fft]
            if len(frame) < n_fft:
                break
            windowed = frame * window
            fft = np.fft.rfft(windowed)
            power_spectrum += np.abs(fft) ** 2
        
        power_spectrum /= num_frames
        
        total_power = np.sum(power_spectrum)
        if total_power < 1e-12:
            return result
        
        # 45-65Hz band power
        hum_mask = (freqs >= 45) & (freqs <= 65)
        hum_power = np.sum(power_spectrum[hum_mask])
        hum_ratio = hum_power / total_power
        result["hum_ratio"] = hum_ratio
        
        # Energy threshold: < 0.5% means no significant hum -> skip bandreject
        if hum_ratio < 0.005:
            result["skip_bandreject"] = True
            result["has_hum"] = False
            return result
        
        result["has_hum"] = True
        result["skip_bandreject"] = False
        
        # Detect 50Hz vs 60Hz
        mask_50 = (freqs >= 48) & (freqs <= 52)
        power_50 = np.sum(power_spectrum[mask_50])
        mask_60 = (freqs >= 58) & (freqs <= 62)
        power_60 = np.sum(power_spectrum[mask_60])
        
        # 45-180Hz lowfreq total
        mask_low = (freqs >= 45) & (freqs <= 180)
        power_low = np.sum(power_spectrum[mask_low])
        
        # Dual-hum detection
        ratio_50 = power_50 / total_power
        ratio_60 = power_60 / total_power
        
        if (ratio_50 >= 0.005 and ratio_60 >= 0.005 and
            power_low > 0 and
            (power_50 / power_low) >= 0.30 and
            (power_60 / power_low) >= 0.30 and
            max(ratio_50, ratio_60) / min(ratio_50, ratio_60) < 3.0):
            result["is_dual_hum"] = True
            result["primary_freq"] = 50 if power_50 >= power_60 else 60
        else:
            result["is_dual_hum"] = False
            result["primary_freq"] = 50 if power_50 >= power_60 else 60
        
        return result
    
    except Exception:
        return result


def get_audio_channel_layout(audio_path: str) -> str:
    """Get audio channel layout using ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channel_layout",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            layout = r.stdout.strip()
            if layout and layout != "unknown":
                return layout
    except Exception:
        pass
    return "stereo"


def build_hum_filters(freq: int, dual_hum: bool = False,
                      config: Dict[str, Any] = None) -> List[str]:
    """Build bandreject filter chain for hum removal.
    
    Covers fundamental + second harmonic + third harmonic.
    All filters use width_type=h:width=2 (two Hz total bandwidth).
    """
    if config is None:
        config = {}
    
    filters = []
    
    # Primary frequency: fundamental + second + third harmonic, all width=2
    for f in [freq, freq * 2, freq * 3]:
        filters.append(f"bandreject=f={f}:width_type=h:width=2")
    
    # Dual hum: only if explicitly enabled
    if dual_hum and config.get("hum_dual", False):
        other_freq = 60 if freq == 50 else 50
        for f in [other_freq, other_freq * 2, other_freq * 3]:
            filters.append(f"bandreject=f={f}:width_type=h:width=2")
    
    return filters


def process_audio(audio_path: str, output_path: str,
                  config: Dict[str, Any], logger=None) -> bool:
    """Process audio for restoration.

    Steps:
    1. Detect hum energy and frequency
    2. If hum detected: bandreject filters (fundamental + second + third harmonic)
    3. Noise reduction (afftdn or arnndn)
    4. Preserve multi-channel layout
    
    Filter order is CRITICAL: bandreject FIRST, then afftdn/arnndn.
    """
    if logger is None:
        logger = get_logger()

    audio_config = config.get("audio", {})
    if not audio_config.get("enabled", False):
        logger.info("AUDIO_SKIP", "Audio processing disabled")
        return True

    try:
        # Parse audio settings
        sample_rate = audio_config.get("sample_rate", 48000)
        noise_reduction = audio_config.get("noise_reduction", True)
        codec = audio_config.get("codec", "flac")
        bitrate = audio_config.get("bitrate", "320k")
        arnndn_model = audio_config.get("arnndn_model", "")
        hum_dual_config = audio_config.get("hum_dual", False)

        # Get channel layout for multi-channel preservation
        channel_layout = get_audio_channel_layout(audio_path)
        num_channels = 2
        if channel_layout in ("5.1", "6.1", "7.1"):
            num_channels = 6 if channel_layout == "5.1" else (8 if channel_layout == "7.1" else 7)
            # For multi-channel, use FLAC to preserve layout (AAC may downmix)
            if codec == "aac":
                codec = "flac"
                logger.info("AUDIO_MULTICHANNEL",
                           f"Detected {channel_layout}, switching to FLAC to preserve layout")

        # Build filter chain
        # ORDER IS CRITICAL: bandreject FIRST (removes strong single-freq noise),
        # then afftdn/arnndn (noise spectrum estimation needs clean signal)
        filters = []

        # Step 1: Hum detection and removal (BEFORE noise reduction)
        hum_result = detect_hum_frequency(audio_path, sample_rate, audio_config)
        
        if hum_result["skip_bandreject"]:
            logger.info("AUDIO_HUM_SKIP",
                       f"No significant hum detected (ratio={hum_result['hum_ratio']:.4f}), skipping bandreject")
        else:
            primary_freq = hum_result["primary_freq"]
            is_dual = hum_result["is_dual_hum"]
            
            if is_dual and not hum_dual_config:
                logger.warning("AUDIO_DUAL_HUM",
                              f"Dual hum detected (50Hz+60Hz). Processing primary frequency {primary_freq}Hz only. "
                              f"Set hum_dual=true to process both.")
            
            hum_filters = build_hum_filters(primary_freq, is_dual and hum_dual_config, audio_config)
            filters.extend(hum_filters)
            logger.info("AUDIO_HUM_FILTERS",
                       f"Applying bandreject for {primary_freq}Hz hum",
                       filters=hum_filters,
                       is_dual=is_dual)

        # Step 2: Noise reduction (AFTER bandreject)
        if noise_reduction:
            if arnndn_model and Path(arnndn_model).exists():
                filters.append(f"arnndn=m='{arnndn_model}'")
                logger.info("AUDIO_DENOISE", "Using arnndn noise reduction", model=arnndn_model)
            else:
                # Fallback to afftdn
                filters.append("afftdn=nr=10:nf=-20")
                logger.info("AUDIO_DENOISE", "Using afftdn noise reduction")

        # Step 3: Reset PTS to ensure A/V sync (v6.3 §3.10.9)
        filters.append("asetpts=PTS")

        # Build command
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-vn",
            "-af", ",".join(filters) if filters else "anull",
            "-ar", str(sample_rate),
        ]

        # Audio codec selection
        if codec == "aac":
            cmd.extend(["-c:a", "aac", "-b:a", bitrate])
        elif codec == "flac":
            cmd.extend(["-c:a", "flac"])
        else:
            cmd.extend(["-c:a", codec])

        # Preserve channel layout for multi-channel
        if num_channels > 2:
            cmd.extend(["-ac", str(num_channels)])

        cmd.append(output_path)

        r = subprocess.run(cmd, capture_output=True, timeout=300, shell=False)
        if r.returncode == 0:
            logger.info("AUDIO_DONE", "Audio processing complete",
                       output=output_path, codec=codec, channels=num_channels)
            return True
        else:
            logger.error("AUDIO_FAIL", "Audio processing failed", 
                        stderr=r.stderr.decode("utf-8", errors="replace"))
            return False

    except Exception as e:
        logger.error("AUDIO_ERROR", f"Audio processing error: {e}")
        return False


# Audio filter presets for different restoration scenarios
AUDIO_PRESETS = {
    "tape_hiss": {
        "filters": ["highpass=f=80", "afftdn=nr=15:nf=-25"],
        "description": "Tape hiss removal with highpass",
    },
    "vinyl_crackle": {
        "filters": ["acrusher=mix=0.05:level_in=1:level_out=1",
                    "highpass=f=30",
                    "afftdn=nr=8:nf=-30"],
        "description": "Vinyl crackle and surface noise",
    },
    "hum_50hz": {
        "filters": ["bandreject=f=50:width_type=h:width=2",
                    "bandreject=f=100:width_type=h:width=2",
                    "bandreject=f=150:width_type=h:width=2"],
        "description": "50Hz mains hum and harmonics",
    },
    "hum_60hz": {
        "filters": ["bandreject=f=60:width_type=h:width=2",
                    "bandreject=f=120:width_type=h:width=2",
                    "bandreject=f=180:width_type=h:width=2"],
        "description": "60Hz mains hum and harmonics",
    },
    "general_denoise": {
        "filters": ["afftdn=nr=10:nf=-20"],
        "description": "General noise reduction",
    },
}


def extract_audio(video_path: str, output_path: str, codec: str = "copy",
                  logger=None) -> bool:
    """Extract audio stream from video file without processing.
    
    Uses ffmpeg -vn -acodec copy to extract original audio unchanged.
    This is Stage 0 operation - audio processing happens in Stage 7.
    """
    if logger is None:
        logger = get_logger()
    
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",  # No video
            "-acodec", codec,  # Copy audio codec (or "copy" for lossless)
            output_path
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=120, shell=False)
        if r.returncode == 0:
            logger.info("AUDIO_EXTRACT", "Audio extracted", 
                       input=video_path, output=output_path, codec=codec)
            return True
        else:
            logger.error("AUDIO_EXTRACT_FAIL", "Audio extraction failed",
                        stderr=r.stderr.decode("utf-8", errors="replace"))
            return False
    except Exception as e:
        logger.error("AUDIO_EXTRACT_ERROR", f"Audio extraction error: {e}")
        return False


def get_audio_preset(preset_name: str) -> Dict[str, Any]:
    """Get audio filter preset by name."""
    return AUDIO_PRESETS.get(preset_name, AUDIO_PRESETS["general_denoise"])


def get_ffmpeg_audio_filter(preset: str, custom_filters: List[str] = None) -> str:
    """Build complete FFmpeg audio filter string from preset + custom."""
    preset_data = get_audio_preset(preset)
    filters = list(preset_data.get("filters", []))
    if custom_filters:
        filters.extend(custom_filters)
    return ",".join(filters)
