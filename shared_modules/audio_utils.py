#!/usr/bin/env python3
"""
Audio Restoration Utilities
Shared audio processing for both video and photo pipelines.
"""

import sys
from pathlib import Path
import subprocess
from typing import Dict, Any, List, Optional, Tuple

import math

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from logger import get_logger  # noqa: E402


def _compute_fft_window_size(sample_rate: int) -> int:
    """Compute adaptive FFT window size for ~85ms duration.

    Formula: 2^ceil(log2(sample_rate * 0.085))
    Minimum: max(result, 4096) per v7.32 §3.10.22
    Examples: 48kHz -> 4096, 44.1kHz -> 4096, 96kHz -> 8192, 8kHz -> 4096 (clamped)
    """
    target_samples = sample_rate * 0.085
    n = int(2 ** math.ceil(math.log2(target_samples)))
    # FIX-P3-4: Per v7.32 §3.10.22: enforce minimum N for low-frequency
    # resolution. At 4096 points, 48kHz gives ~11.7Hz/bin, sufficient for
    # 45-65Hz analysis.
    return max(n, 4096)


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
        # FIX-WARN-4: low_freq_absent flag per v7.32 §3.10.22 ("无低频内容
        # (峰值低于阈值),默认回退至 50Hz 并记录 WARNING"). Set when there is
        # signal but the 45-65Hz hum band energy is negligible (absolute or
        # relative). ORIGINAL: only the total_power<1e-12 (truly silent) path
        # tripped the WARNING via primary_freq==50 & hum_ratio==0.0; a tiny
        # but non-zero hum_ratio (<0.5%) returned skip_bandreject=True without
        # setting low_freq_absent, so the WARNING never fired for "near-silent
        # low-freq" material. process_audio keys off this flag for the WARNING.
        "low_freq_absent": False,
    }
    
    try:
        # FIX-P3-2: Probe audio duration; if shorter than analyze_seconds,
        # analyze full audio per v7.32 §3.10.9: "分析音频前 30 秒（若时长
        # <30s 则分析全音频）"
        audio_duration = None
        try:
            r_dur = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1",
                 audio_path],
                capture_output=True, text=True, timeout=10,
                shell=False
            )
            if r_dur.returncode == 0 and r_dur.stdout.strip():
                audio_duration = float(r_dur.stdout.strip())
        except Exception:
            pass

        # Extract audio to raw PCM for FFT analysis
        cmd = ["ffmpeg", "-y", "-i", audio_path]
        if audio_duration is None or audio_duration > analyze_seconds:
            cmd.extend(["-t", str(analyze_seconds)])
        cmd.extend([
            "-ar", str(sample_rate),
            "-ac", "1",  # mono for analysis
            "-f", "s16le", "-"
        ])
        proc = subprocess.run(cmd, capture_output=True, timeout=60,
                              shell=False)
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
            # FIX-P3-4: Default to 50Hz when no low-frequency content per v7.32 §3.10.22
            # FIX-WARN-4: mark low_freq_absent so process_audio emits WARNING.
            result["primary_freq"] = 50
            result["has_hum"] = False
            result["skip_bandreject"] = True
            result["low_freq_absent"] = True
            return result

        # 45-65Hz band power
        hum_mask = (freqs >= 45) & (freqs <= 65)
        hum_power = np.sum(power_spectrum[hum_mask])
        hum_ratio = hum_power / total_power
        result["hum_ratio"] = hum_ratio

        # Energy threshold: < 0.5% means no significant hum -> skip bandreject
        # FIX-WARN-4: also mark low_freq_absent so process_audio emits the
        # "no low-frequency content" WARNING per v7.32 §3.10.22. ORIGINAL
        # only set skip_bandreject and returned, so a tiny non-zero hum_ratio
        # fell through to the INFO skip log and never produced a WARNING.
        if hum_ratio < 0.005:
            result["skip_bandreject"] = True
            result["has_hum"] = False
            result["low_freq_absent"] = True
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


def get_audio_info(audio_path: str) -> Dict[str, Any]:
    """Get audio stream information using ffprobe.

    Returns a dict with sample_rate, channels, channel_layout, codec.
    Uses None for unknown values instead of hard-coded defaults.
    """
    info: Dict[str, Any] = {
        "sample_rate": None,
        "channels": None,
        "channel_layout": None,
        "codec": None,
    }
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels,codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            lines = [line.strip() for line in r.stdout.strip().splitlines() if line.strip()]
            if len(lines) >= 3:
                # sample_rate
                sr = lines[0]
                if sr.isdigit():
                    info["sample_rate"] = int(sr)
                # channels
                ch = lines[1]
                if ch.isdigit():
                    info["channels"] = int(ch)
                # codec
                info["codec"] = lines[2] if lines[2] else None
        # channel_layout from separate probe (more reliable)
        info["channel_layout"] = get_audio_channel_layout(audio_path)
    except Exception:
        pass
    return info


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


def _build_afftdn_filter(num_channels: int, nr: int = 12) -> str:
    """Build a spec-compliant afftdn filter string.

    FIX-WARN-1: Per v7.32 §3.10.22 (afftdn 参数语义):
      - nf (noise_floor): default -50 dB. Aggressive values like -25/-20
        damage speech and are explicitly forbidden by spec. ORIGINAL code
        used nf=-25 / nf=-20 (forbidden).
      - nr (noise_reduction): default 12 dB (caller may pass a gentler
        value for hum-removal fallback).
      - noise_link: default 'min' couples noise estimation across channels.
        For multi-channel independent per-channel processing, spec requires
        explicit noise_link=none. Added when num_channels > 1.
    afftdn is a generic denoiser (NOT a hum remover); hum removal stays
    the responsibility of bandreject.
    """
    filt = f"afftdn=nf=-50:nr={nr}"
    if num_channels > 1:
        filt += ":noise_link=none"
    return filt


def _get_ffmpeg_version() -> Tuple[bool, str]:
    """Check if FFmpeg version >= 4.4. Returns (ok, version_str)."""
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True,
                          timeout=10, shell=False)
        if r.returncode == 0:
            first = r.stdout.splitlines()[0]
            import re
            m = re.search(r'version\s+(\d+\.\d+)', first)
            if m:
                v = m.group(1)
                parts = v.split(".")
                major = int(parts[0])
                minor = int(parts[1]) if len(parts) > 1 else 0
                return (major > 4 or (major == 4 and minor >= 4)), v
    except Exception:
        pass
    return False, "unknown"


def _get_arnndn_model_path(config: Dict[str, Any], primary_freq: int) -> Optional[str]:
    """Get arnndn model path.

    FIX-P1-8: arnndn uses official std.rnnn (Xiph RNNoise original generic
    model, .rnnn format — NOT frequency-based .pb files). The primary_freq
    argument is retained for signature compatibility but no longer selects
    a per-frequency model (RNNoise is a generic denoiser, not a hum filter;
    hum removal is done by bandreject).
    """
    models_dir = config.get("arnndn_model_dir", "./models")
    audio_cfg = config.get("audio", {}) if isinstance(config.get("audio"), dict) else {}
    model_name = audio_cfg.get("arnndn_model", config.get("arnndn_model", "std.rnnn"))
    model_path = Path(models_dir) / model_name
    if model_path.exists():
        return str(model_path)
    return None


def process_audio(audio_path: str, output_path: str,
                  config: Dict[str, Any], logger=None) -> bool:
    """Process audio for restoration.

    Steps:
    1. Detect FFmpeg version; if <4.4, force arnndn fallback
    2. Detect hum energy and frequency
    3. If hum detected: bandreject filters (fundamental + second + third harmonic)
    4. Noise reduction (afftdn or arnndn)
    5. Optional compand dynamic range compression
    6. Preserve multi-channel layout
    
    Filter order is CRITICAL: bandreject FIRST, then afftdn/arnndn.
    """
    if logger is None:
        logger = get_logger()

    audio_config = config.get("audio", {})
    if not audio_config.get("enabled", False):
        logger.info("AUDIO_SKIP", "Audio processing disabled")
        return True

    try:
        # ISSUE-SHARED-AU-001: Check if audio stream exists before processing
        audio_info = get_audio_info(audio_path)
        if audio_info.get("channels") is None:
            logger.info("AUDIO_NO_STREAM",
                       "Source file has no audio stream, skipping audio processing")
            return True

        # Parse audio settings
        # Per v7.32 §3.10.9: sample_rate can be "auto" — detect from actual file
        sample_rate_raw = audio_config.get("sample_rate", 48000)
        if sample_rate_raw == "auto" or sample_rate_raw is None:
            detected_sr = audio_info.get("sample_rate")
            if detected_sr:
                sample_rate = int(detected_sr)
                logger.info("AUDIO_SAMPLE_RATE",
                           f"Auto-detected sample rate: {sample_rate}Hz")
            else:
                sample_rate = 48000
                logger.warning("AUDIO_SAMPLE_RATE_DEFAULT",
                              "Could not detect sample rate, defaulting to 48000Hz")
        else:
            sample_rate = int(sample_rate_raw)
        
        # Unified key: 'denoise' per spec v7.32, fallback to 'noise_reduction'
        denoise = audio_config.get("denoise", audio_config.get("noise_reduction", True))
        codec = audio_config.get("codec", "aac")  # Default 'aac' per spec
        bitrate = audio_config.get("bitrate", "320k")
        hum_dual_config = audio_config.get("hum_dual", False)
        use_compand = audio_config.get("compand", False)

        # Check FFmpeg version for bandreject compatibility
        ffmpeg_ok, ffmpeg_ver = _get_ffmpeg_version()
        if not ffmpeg_ok:
            logger.warning("AUDIO_FFMPEG_OLD",
                          f"FFmpeg {ffmpeg_ver} < 4.4, bandreject unavailable. "
                          "Will use arnndn fallback for hum removal.")

        # BUG-059 fix: Use channel count instead of exact layout string matching
        # to handle variants like "5.1(side)", "7.1(wide)", "5.1(back)", etc.
        channel_layout = audio_info.get("channel_layout", "stereo") or "stereo"
        num_channels = audio_info.get("channels", 2) or 2
        if num_channels >= 6:
            # For multi-channel, use FLAC to preserve layout (AAC may downmix)
            if codec == "aac":
                codec = "flac"
                logger.info("AUDIO_MULTICHANNEL",
                           f"Detected {num_channels}ch ({channel_layout}), switching to FLAC to preserve layout")

        # Build filter chain
        # ORDER IS CRITICAL: bandreject FIRST (removes strong single-freq noise),
        # then afftdn/arnndn (noise spectrum estimation needs clean signal)
        filters = []
        # FIX-P3-2: Promote filter_complex/final_label to function scope so
        # Step 1 (hum) and Step 2 (denoise) can share the same filter_complex
        filter_complex = None
        final_label = "[aout]"

        # Step 1: Hum detection and removal (BEFORE noise reduction)
        hum_result = detect_hum_frequency(audio_path, sample_rate, audio_config)
        primary_freq = hum_result["primary_freq"]

        # FIX-P3-1: Per v7.32 §3.10.22 / v6.32 §7.12: Allow user to override hum_freq
        user_hum_freq = audio_config.get("hum_freq")
        if user_hum_freq is not None:
            user_freq = int(user_hum_freq)
            if user_freq in (50, 60):
                if primary_freq != user_freq:
                    logger.info("AUDIO_HUM_OVERRIDE",
                               f"User override hum_freq={user_freq}Hz "
                               f"(auto-detected={primary_freq}Hz)")
                primary_freq = user_freq
            else:
                logger.warning("AUDIO_HUM_OVERRIDE_INVALID",
                              f"Invalid hum_freq={user_freq}, must be 50 or 60. "
                              f"Using auto-detected {primary_freq}Hz")

        # FIX-P3-4: Log warning when no low-frequency content detected
        # FIX-R7-S7-3a: ORIGINAL wording "defaulting hum_freq to 50Hz"
        # contradicted the revised spec (v7.32 §3.10.22 / v6.32 §7.12:
        # hum_freq defaults to null = auto-detect). When auto-detect finds
        # no low-frequency content, hum_freq stays null (auto) and
        # bandreject is skipped — de-humming falls back to afftdn/arnndn
        # broadband denoise. Wording corrected to match the null default;
        # detect_hum_frequency still sets primary_freq=50 as an internal
        # placeholder, but skip_bandreject=True so no bandreject runs.
        # FIX-WARN-4: Use the low_freq_absent flag (set for both the
        # total_power<1e-12 and hum_ratio<0.5% cases) so a tiny non-zero
        # hum_ratio also triggers the WARNING per v7.32 §3.10.22. ORIGINAL
        # keyed off `primary_freq==50 and hum_ratio==0.0`, which missed the
        # near-silent-but-non-zero case (hum_ratio in (0, 0.005)).
        if hum_result.get("low_freq_absent"):
            logger.warning("AUDIO_HUM_DEFAULT",
                          "No low-frequency content detected; hum_freq stays "
                          "null (auto), bandreject skipped per v7.32 §3.10.22")
        
        if hum_result["skip_bandreject"]:
            logger.info("AUDIO_HUM_SKIP",
                       f"No significant hum detected (ratio={hum_result['hum_ratio']:.4f}), skipping bandreject")
        else:
            is_dual = hum_result["is_dual_hum"]
            
            if is_dual and not hum_dual_config:
                logger.warning("AUDIO_DUAL_HUM",
                              f"Dual hum detected (50Hz+60Hz). Processing primary frequency {primary_freq}Hz only. "
                              f"Set hum_dual=true to process both.")

            # FIX-FAIL-1: Per v7.32 §3.10.9-2 / §3.10.22 (and v6.32 §7.12):
            # when the user explicitly configures hum_dual=true AND dual hum
            # is detected, both 50/100/150 + 60/120/180Hz bandreject groups
            # are chained. This is an audio-quality risk, so a CRITICAL log
            # is mandatory. ORIGINAL only logged INFO (AUDIO_HUM_FILTERS) and
            # a WARNING for the single-group default case — no CRITICAL
            # anywhere in the file. Added here per spec.
            if is_dual and hum_dual_config:
                logger.critical(
                    "AUDIO_HUM_DUAL_ENABLED",
                    f"Dual hum processing ENABLED (hum_dual=true): chaining "
                    f"50/100/150 + 60/120/180Hz bandreject groups. Audio "
                    f"quality risk: may damage low-frequency content near "
                    f"50/60Hz; recommended only for confirmed cross-region "
                    f"material with dual mains hum.",
                    primary_freq=primary_freq,
                )

            if ffmpeg_ok:
                hum_filters = build_hum_filters(primary_freq, is_dual and hum_dual_config, audio_config)
                filters.extend(hum_filters)
                logger.info("AUDIO_HUM_FILTERS",
                           f"Applying bandreject for {primary_freq}Hz hum",
                           filters=hum_filters,
                           is_dual=is_dual)
            else:
                # FIX-P3-2: FFmpeg <4.4, use arnndn for hum removal per v7.32 §3.10.22
                # Per v7.32 §3.10.9: per-channel processing, no downmix
                arnndn_model = _get_arnndn_model_path(audio_config, primary_freq)
                if not arnndn_model:
                    arnndn_model = audio_config.get("arnndn_model", "")
                if arnndn_model and Path(arnndn_model).exists():
                    if num_channels > 1:
                        # Per-channel arnndn using filter_complex (no downmix!)
                        # FIX-P3-6: Extended channel_map for 3ch/4ch per
                        # v7.32 §3.10.9: "严禁混缩为立体声"
                        channel_map = {
                            1: ["FC"],
                            2: ["FL", "FR"],
                            3: ["FL", "FR", "LFE"],          # 2.1
                            4: ["FL", "FR", "BL", "BR"],     # Quad / 3.1
                            6: ["FL", "FR", "FC", "LFE", "BL", "BR"],
                            8: ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
                        }
                        ch_names = channel_map.get(num_channels, ["FL", "FR"])
                        fc_parts = []
                        for i, ch in enumerate(ch_names):
                            fc_parts.append(f"[0:a]pan=mono|c0={ch}[ch{i}]")
                            fc_parts.append(
                                f"[ch{i}]arnndn=m={arnndn_model}[ch{i}_h]"
                            )
                        merge_inputs = "".join(
                            f"[ch{i}_h]" for i in range(len(ch_names))
                        )
                        fc_parts.append(
                            f"{merge_inputs}amerge=inputs={len(ch_names)}[aout]"
                        )
                        filter_complex = ";".join(fc_parts)
                        final_label = "[aout]"
                        logger.info("AUDIO_HUM_ARNNDN_MULTICHANNEL",
                                   f"Per-channel arnndn hum removal for "
                                   f"{num_channels}ch (FFmpeg <4.4)",
                                   model=arnndn_model)
                    else:
                        filters.append(f"arnndn=m={arnndn_model}")
                        logger.info("AUDIO_HUM_ARNNDN",
                                   f"Using arnndn for {primary_freq}Hz hum "
                                   f"removal (FFmpeg <4.4)",
                                   model=arnndn_model)
                else:
                    # FIX-P3-3: Per v7.32 §3.10.22: afftdn fallback when
                    # arnndn unavailable
                    # FIX-P3-5: Step 1 hum removal fallback — gentler noise
                    # reduction to avoid damaging speech when replacing
                    # targeted bandreject filters
                    logger.warning("AUDIO_HUM_ARNNDN_FALLBACK",
                                  f"arnndn model not found for "
                                  f"{primary_freq}Hz, falling back to afftdn "
                                  "for hum removal (FFmpeg <4.4)")
                    # FIX-WARN-1: spec-compliant afftdn (nf=-50, not -25;
                    # noise_link=none for multi-channel per v7.32 §3.10.22).
                    # nr kept gentle (6) for the hum-removal fallback.
                    filters.append(_build_afftdn_filter(num_channels, nr=6))

        # Step 2: Noise reduction (AFTER bandreject)
        if denoise:
            arnndn_model = _get_arnndn_model_path(audio_config, primary_freq)
            if not arnndn_model:
                # Fallback to config-provided path
                arnndn_model = audio_config.get("arnndn_model", "")

            if arnndn_model and Path(arnndn_model).exists():
                # FIX-P3-2: Per-channel arnndn to preserve multi-channel
                # layout per v7.32 §3.10.9
                # FIX-P3-1: Per v6.32 §7.30: arnndn stereo input has
                # community-reported bugs on some CPUs. Use per-channel
                # filter_complex for ALL multi-channel arnndn processing,
                # regardless of FFmpeg version.
                # FIX-P3-6: Extended channel_map for 3ch/4ch per
                # v7.32 §3.10.9: "严禁混缩为立体声"
                if num_channels > 1:
                    # Migrate bandreject from -af to filter_complex so
                    # they chain correctly with per-channel arnndn
                    bandreject_filters = [
                        f for f in filters
                        if f.startswith("bandreject")
                    ]
                    for br in bandreject_filters:
                        filters.remove(br)

                    channel_map = {
                        1: ["FC"],
                        2: ["FL", "FR"],
                        3: ["FL", "FR", "LFE"],          # 2.1
                        4: ["FL", "FR", "BL", "BR"],     # Quad / 3.1
                        6: ["FL", "FR", "FC", "LFE", "BL", "BR"],
                        8: ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
                    }
                    ch_names = channel_map.get(num_channels, ["FL", "FR"])

                    if filter_complex is not None:
                        # Append denoise to existing filter_complex
                        # (from Step 1 FFmpeg<4.4 per-channel arnndn hum)
                        fc_parts = [filter_complex]
                        for i, ch in enumerate(ch_names):
                            fc_parts.append(
                                f"[{final_label[1:-1]}]pan=mono|c0=c{i}"
                                f"[dch{i}]"
                            )
                            fc_parts.append(
                                f"[dch{i}]arnndn=m={arnndn_model}"
                                f"[dch{i}_out]"
                            )
                        merge_inputs = "".join(
                            f"[dch{i}_out]" for i in range(len(ch_names))
                        )
                        fc_parts.append(
                            f"{merge_inputs}amerge=inputs="
                            f"{len(ch_names)}[aout2]"
                        )
                        filter_complex = ";".join(fc_parts)
                        final_label = "[aout2]"
                        logger.info(
                            "AUDIO_ARNNDN_MULTICHANNEL_APPEND",
                            f"Appended per-channel arnndn denoise to "
                            f"existing filter_complex for {num_channels}ch",
                            model=arnndn_model)
                    else:
                        # Build new filter_complex with bandreject +
                        # arnndn per channel
                        fc_parts = []
                        for i, ch in enumerate(ch_names):
                            fc_parts.append(
                                f"[0:a]pan=mono|c0={ch}[ch{i}]"
                            )
                            if bandreject_filters:
                                br_chain = ",".join(bandreject_filters)
                                fc_parts.append(
                                    f"[ch{i}]{br_chain}[ch{i}_br]"
                                )
                                fc_parts.append(
                                    f"[ch{i}_br]arnndn=m={arnndn_model}"
                                    f"[ch{i}_d]"
                                )
                            else:
                                fc_parts.append(
                                    f"[ch{i}]arnndn=m={arnndn_model}"
                                    f"[ch{i}_d]"
                                )
                        merge_inputs = "".join(
                            f"[ch{i}_d]" for i in range(len(ch_names))
                        )
                        fc_parts.append(
                            f"{merge_inputs}amerge=inputs="
                            f"{len(ch_names)}[aout]"
                        )
                        filter_complex = ";".join(fc_parts)
                        final_label = "[aout]"
                        logger.info(
                            "AUDIO_ARNNDN_MULTICHANNEL",
                            f"Per-channel arnndn processing for "
                            f"{num_channels}ch",
                            model=arnndn_model)
                else:
                    filters.append(f"arnndn=m={arnndn_model}")
                    logger.info("AUDIO_DENOISE",
                               "Using arnndn noise reduction",
                               model=arnndn_model)
            else:
                # Fallback to afftdn
                # FIX-P3-5: Step 2 broad-spectrum denoise fallback — slightly
                # stronger settings than hum fallback, but still conservative
                # to avoid artifacts
                if not ffmpeg_ok:
                    logger.warning("AUDIO_ARNNDN_MISSING",
                                  f"arnndn model not found for "
                                  f"{primary_freq}Hz, using afftdn fallback")
                # FIX-WARN-1: spec-compliant afftdn (nf=-50, not -20;
                # noise_link=none for multi-channel per v7.32 §3.10.22).
                # nr=12 is the spec default for broad-spectrum denoise.
                filters.append(_build_afftdn_filter(num_channels, nr=12))
                logger.info("AUDIO_DENOISE", "Using afftdn noise reduction")

        # Step 3: Optional compand dynamic range compression
        if use_compand:
            # Default gentle compand curve per v6.32 §7.27
            compand_points = audio_config.get("compand_points",
                                              "0.3:1:-90/-900|-70/-70|-30/-9|0/-3:6:0:0:0.5")
            filters.append(f"compand={compand_points}")
            logger.info("AUDIO_COMPAND", "Applying compand dynamic range compression")

        # Step 4: Reset PTS to ensure A/V sync (v6.3 §3.10.9)
        filters.append("asetpts=PTS")

        # Build command
        if filter_complex:
            # FIX-P3-2: Append remaining filters (compand, asetpts) to
            # filter_complex using tracked final_label
            # FIX-P3-9: When filter_complex is active, arnndn/bandreject are
            # already inside it. Only append "post-processing" filters
            # (compand, asetpts) that were added after the per-channel
            # filter_complex was built.
            # CAUTION: If a non-per-channel arnndn is ever added to filters[],
            # this exclusion logic must be revisited — currently safe because
            # mono arnndn only occurs when filter_complex is None.
            post_filters = [
                f for f in filters
                if not f.startswith("bandreject")
                and not f.startswith("arnndn")
                and not f.startswith("aformat")
            ]
            if post_filters:
                filter_complex += (
                    f";{final_label}{','.join(post_filters)}[aout_final]"
                )
                map_label = "[aout_final]"
            else:
                map_label = final_label
            cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-vn",
                "-filter_complex", filter_complex,
                "-map", map_label,
                "-ar", str(sample_rate),
            ]
        else:
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

        # Preserve channel layout for multi-channel per v7.32 §3.10.9
        # (read via ffprobe and retained; "严禁混缩为立体声").
        # FIX-WARN-2: ORIGINAL only passed `-ac <num_channels>`, leaving the
        # output channel_layout to ffmpeg's inference. After a per-channel
        # pan/amerge chain the layout can come out as "unknown", which some
        # muxers/players then mis-handle. When the source layout is a known
        # non-stereo layout, explicitly pass `-channel_layout` so the output
        # carries the correct label (e.g. "5.1"). The pan-filter rebuild
        # (needed only if amerge's default layout disagrees with the source
        # label) is deferred pending confirmation of amerge's output-layout
        # behavior — see round_3 WARN-2 / "需联网确认" item 6.
        if num_channels > 2:
            cmd.extend(["-ac", str(num_channels)])
            if channel_layout and channel_layout != "stereo":
                cmd.extend(["-channel_layout", channel_layout])

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
    
    Uses ffmpeg -vn -c:a copy to extract original audio unchanged.
    This is Stage 0 operation - audio processing happens in Stage 7.
    """
    if logger is None:
        logger = get_logger()
    
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",  # No video
            "-c:a", codec,  # Copy audio codec
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


def check_av_sync(video_path: str, audio_path: str,
                  logger=None) -> Tuple[bool, float, str]:
    """Check A/V sync by comparing processed video and audio durations.

    Returns:
        (is_acceptable, diff_ms, action)
        is_acceptable: True if diff <= 300ms (stage7 should proceed)
        diff_ms: absolute difference in milliseconds
        action: "ok" / "warning" / "sync_required" / "failed"
    FIX-P3-5 per v7.32 §3.10.9
    """
    if logger is None:
        logger = get_logger()

    def _get_duration(path: str, is_video: bool = False) -> float:
        """Get media duration. For video, try stream-level first
        (more accurate for VFR content)."""
        if is_video:
            # FIX-P3-9: Try stream-level duration first for VFR accuracy
            try:
                r = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", path],
                    capture_output=True, text=True, timeout=30, shell=False
                )
                if r.returncode == 0:
                    val = r.stdout.strip()
                    if val and val != "N/A":
                        return float(val)
            except Exception:
                pass
        # Fallback to format-level duration
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=30, shell=False
            )
            if r.returncode == 0:
                return float(r.stdout.strip())
        except Exception:
            pass
        return 0.0

    v_dur = _get_duration(video_path, is_video=True)
    a_dur = _get_duration(audio_path, is_video=False)

    if v_dur <= 0 or a_dur <= 0:
        logger.error("AV_SYNC_FAIL", "Cannot determine video or audio duration")
        return False, 0.0, "failed"

    diff_ms = abs((v_dur - a_dur) * 1000)
    action = "ok"
    # FIX-WARN-3: Per v7.32 §3.10.9 thresholds: >100ms WARNING, >300ms sync
    # required, >500ms ERROR + FAILED. ORIGINAL set the action correctly but
    # logged every branch at INFO, so >500ms failures produced no ERROR-level
    # log and 100-500ms produced no WARNING. Emit the spec-mandated log level
    # for each band (the FAILED status flag is still applied by the stage7
    # caller; here we only align the shared-layer log level to spec).
    if diff_ms > 500:
        action = "failed"
        logger.error("AV_SYNC_FAILED",
                     f"A/V sync diff {diff_ms:.1f}ms > 500ms, marking FAILED",
                     video_dur=v_dur, audio_dur=a_dur, diff_ms=diff_ms)
    elif diff_ms > 300:
        action = "sync_required"
        logger.warning("AV_SYNC_SYNC_REQUIRED",
                       f"A/V sync diff {diff_ms:.1f}ms > 300ms, sync required",
                       diff_ms=diff_ms)
    elif diff_ms > 100:
        action = "warning"
        logger.warning("AV_SYNC_WARNING",
                       f"A/V sync diff {diff_ms:.1f}ms > 100ms",
                       diff_ms=diff_ms)
    else:
        logger.info("AV_SYNC_CHECK",
                   f"Video={v_dur:.3f}s, Audio={a_dur:.3f}s, Diff={diff_ms:.1f}ms",
                   action=action)
    return action in ("ok", "warning"), diff_ms, action
