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

import sys
import json
import argparse
import subprocess
import datetime
import re
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

    FIX-P7-8: DEPRECATION NOTICE — This function duplicates logic in
    shared_modules/audio_utils.process_audio(). Long-term plan: replace
    _build_audio_filter with a call to audio_utils.process_audio() to
    generate an intermediate processed audio file, then stage7 only
    handles muxing/encoding. For now, both copies are maintained with
    synchronized bug fixes (P7-1 through P7-7).
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

    # FIX-P7-3: Per v7.32 §3.10.9: preserve channel layout, no downmix
    # bandreject/afftdn are per-channel internally in FFmpeg -af (no downmix needed).
    # Only arnndn needs filter_complex per-channel (model trained mono, stereo bug).
    # FIX-P7-4: For >2ch with FFmpeg<4.4, add afftdn as hum fallback since
    # arnndn cannot process multi-channel without downmix (per v7.32 §3.10.9).
    if audio_layout and audio_layout.get("channels", 2) > 2:
        num_ch = audio_layout["channels"]
        arnndn_model_val = params.get("arnndn_model", "")
        if arnndn_model_val and Path(arnndn_model_val).exists() and not ffmpeg_version_ok:
            # arnndn path: must use filter_complex per-channel, but that's
            # complex; use afftdn as hum fallback for multi-channel per §3.10.9
            logger.info("STAGE7_MULTICHANNEL_HUM_AFFTDN",
                        f"FFmpeg <4.4 + {num_ch}ch audio: using afftdn for hum "
                        "removal (arnndn requires mono, would violate §3.10.9)")
        else:
            # bandreject/afftdn: unified -af is safe (FFmpeg processes each
            # channel independently, no downmix)
            logger.info("STAGE7_MULTICHANNEL_UNIFIED",
                        f"Unified filter chain for {num_ch}ch audio "
                        "(bandreject/afftdn are per-channel internally)")

    # FIX-P7-1: Per v6.32 §7.12 / v7.32 §3.10.22:
    # hum_freq=None means auto-detect; explicit 50/60 means user override
    # ORIGINAL: hum_freq = params.get("hum_freq", 50) with auto_detect flag
    hum_freq = params.get("hum_freq")  # None = auto-detect; 50/60 = user override
    hum_dual = params.get("hum_dual", False)

    detected_freq = None
    is_dual_hum = False  # FIX-P7-9: track dual hum detection result
    if hum_freq is not None:
        # User explicitly configured hum_freq, skip auto-detection
        detected_freq = int(hum_freq)
        logger.info("STAGE7_HUM_USER_OVERRIDE",
                    f"User forced hum_freq={detected_freq}Hz, skipping auto-detect")
    else:
        # Auto-detect path
        try:
            sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
            from audio_utils import detect_hum_frequency
            hum_result = detect_hum_frequency(audio_path, config=params)
            if hum_result.get("skip_bandreject", False):
                logger.info("STAGE7_HUM_SKIP", "Hum energy below threshold, skipping bandreject")
                # FIX-R7-S7-3a: explicit WARNING when no low-frequency content is
                # detected (hum_ratio==0.0) per v7.32 §3.10.22 — hum_freq stays
                # null (auto), bandreject skipped, hum removal deferred to afftdn.
                if float(hum_result.get("hum_ratio", 0.0)) == 0.0:
                    logger.warning("STAGE7_HUM_NO_LOWFREQ",
                                   "No low-frequency content detected; hum_freq stays "
                                   "null (auto), bandreject skipped per v7.32 §3.10.22")
                detected_freq = None
            else:
                detected_freq = hum_result.get("primary_freq", 50)
                # FIX-P7-9: Per v7.32 §3.10.22: save is_dual_hum from detection
                is_dual_hum = hum_result.get("is_dual_hum", False)
        except Exception as e:
            # FIX-P7-11: Per v7.32 §3.10.22: default to 50Hz with WARNING
            detected_freq = 50
            logger.warning("STAGE7_HUM_DETECT_FAIL",
                           f"Hum frequency detection failed: {e}. "
                           "Defaulting to 50Hz per v7.32 §3.10.22")
    
    if detected_freq and ffmpeg_version_ok:
        # Fundamental + 2nd + 3rd harmonic, all width=2Hz
        for f in [detected_freq, detected_freq * 2, detected_freq * 3]:
            audio_filters.append(f"bandreject=f={f}:width_type=h:width=2")
        
        # Dual hum support
        # FIX-P7-9: Per v7.32 §3.10.22: dual hum must be detected AND user-enabled.
        # Default: only process primary freq group, log WARNING if dual detected.
        # Only when hum_dual=true AND is_dual_hum=true: process both groups with CRITICAL.
        if hum_dual and is_dual_hum:
            # User explicitly enabled + actual dual hum detected
            other = 60 if detected_freq == 50 else 50
            for f in [other, other * 2, other * 3]:
                audio_filters.append(f"bandreject=f={f}:width_type=h:width=2")
            logger.critical("STAGE7_DUAL_HUM_RISK",
                            "Processing BOTH 50Hz+60Hz hum groups per "
                            "hum_dual=true. Audio quality risk — dual "
                            "filtering may over-process.",
                            primary=detected_freq, secondary=other)
        elif is_dual_hum and not hum_dual:
            logger.warning("STAGE7_DUAL_HUM_DEFAULT",
                           f"Dual hum detected but processing primary "
                           f"{detected_freq}Hz only. Set hum_dual=true to "
                           f"process both groups (audio quality risk).")
    elif detected_freq and not ffmpeg_version_ok:
        # FIX-P7-1: Per v6.32 §7.12 / v7.32 §3.10.22:
        # FFmpeg <4.4: arnndn replaces bandreject for hum removal
        # ORIGINAL: only logged, arnndn filter not inserted into audio_filters
        arnndn_model = params.get("arnndn_model", "")
        if arnndn_model and Path(arnndn_model).exists():
            if audio_layout is None:
                audio_layout = _get_audio_layout(audio_path)
            num_ch = audio_layout.get("channels", 2)
            if num_ch > 2:
                # Multi-channel: arnndn needs mono, skip for hum removal
                logger.warning("STAGE7_HUM_ARNNDN_MULTICH",
                               f"FFmpeg <4.4, arnndn requires mono input, cannot "
                               f"remove {detected_freq}Hz hum from {num_ch}ch. "
                               f"afftdn will provide general noise reduction only.")
            elif num_ch == 2:
                audio_filters.append("aformat=channel_layouts=mono")
                # FIX-P7-7: No single-quote wrapping per v6.32 §7.30
                audio_filters.append(f"arnndn=m={arnndn_model}")
                layout = audio_layout.get("layout", "stereo")
                audio_filters.append(f"aformat=channel_layouts={layout}")
            else:
                # Mono: use arnndn directly
                audio_filters.append(f"arnndn=m={arnndn_model}")
            logger.info("STAGE7_HUM_ARNNDN",
                        f"FFmpeg <4.4, using arnndn for {detected_freq}Hz hum removal",
                        model=arnndn_model)
        else:
            logger.warning("STAGE7_HUM_NO_ARNNDN",
                           f"FFmpeg <4.4 and arnndn model not found at '{arnndn_model}', "
                           f"hum at {detected_freq}Hz will NOT be removed. "
                           f"afftdn will provide general noise reduction only. "
                           f"Download arnndn models from https://github.com/richardpl/arnndn-models (std.rnnn)")

    # Noise reduction AFTER bandreject
    noise_reduction = params.get("noise_reduction", True)
    if noise_reduction:
        arnndn_model = params.get("arnndn_model", "")
        if audio_layout is None:
            audio_layout = _get_audio_layout(audio_path)
        num_channels = audio_layout.get("channels", 2)

        if arnndn_model and Path(arnndn_model).exists():
            # FIX-P7-3: Per v7.32 §3.10.9 "严禁混缩为立体声":
            # - Mono: use arnndn directly (no layout change needed)
            # - Stereo: aformat=mono → arnndn → restore (acceptable per §7.30
            #   mono input recommendation, preserves stereo on output)
            # - Multi-channel (>2): use afftdn instead to avoid downmix;
            #   arnndn requires mono which would destroy spatial info in 5.1/7.1
            # ORIGINAL: always used aformat=mono → arnndn for all channel counts,
            # which downmixes multi-channel audio violating the spec.
            if num_channels > 2:
                logger.warning("STAGE7_ARNNDN_MULTICHANNEL_SKIP",
                               f"arnndn requires mono input, would downmix "
                               f"{num_channels}ch audio. Using afftdn instead "
                               f"per v7.32 §3.10.9 (no downmix allowed).")
                audio_filters.append("afftdn=nr=12:nf=-50")
            elif num_channels == 2:
                # Stereo: arnndn mono limitation, restore after processing
                audio_filters.append("aformat=channel_layouts=mono")
                # FIX-P7-7: No single-quote wrapping per v6.32 §7.30
                audio_filters.append(f"arnndn=m={arnndn_model}")
                layout = audio_layout.get("layout", "stereo")
                audio_filters.append(f"aformat=channel_layouts={layout}")
            else:
                # Mono: use arnndn directly
                audio_filters.append(f"arnndn=m={arnndn_model}")
        else:
            audio_filters.append("afftdn=nr=12:nf=-50")

    # V-P-053 fix: compand dynamic range compression
    # FIX-P7-8: Default changed to False per v7.32 §3.10.9 (not mandatory)
    # ORIGINAL: audio_compand defaulted to True
    use_compand = params.get("audio_compand", False)
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


def _detect_black_bars_stage7(video_path: str, logger=None) -> dict:
    """Detect black bars on the stage7 input for crop/restore around color
    conversion per v7.32 §3.10.3 ("色域转换前应先自动检测并裁剪黑边,转换完成后
    还原黑边"). Bit-depth-aware cropdetect limit (limited-range black level
    scales with bit depth: 16 * 2^(bd-8)).
    """
    info = {"left": 0, "right": 0, "top": 0, "bottom": 0,
            "cropped_w": 0, "cropped_h": 0, "orig_w": 0, "orig_h": 0,
            "has_bars": False}
    try:
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,pix_fmt,bits_per_raw_sample",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=30, shell=False)
        ow = oh = 0
        bd = 8
        if pr.returncode == 0:
            data = json.loads(pr.stdout)
            streams = data.get("streams", [])
            if streams:
                ow = int(streams[0].get("width", 0) or 0)
                oh = int(streams[0].get("height", 0) or 0)
                bpr = streams[0].get("bits_per_raw_sample", 0)
                pf = streams[0].get("pix_fmt", "")
                try:
                    bd = int(bpr) if bpr else 8
                except (ValueError, TypeError):
                    bd = 8
                if bd < 8:
                    bd = 8
                if not bpr:
                    if "p16" in pf or "16le" in pf or "16be" in pf:
                        bd = 16
                    elif "p10" in pf:
                        bd = 10
        info["orig_w"] = ow
        info["orig_h"] = oh
        if ow == 0 or oh == 0:
            return info
        # Bit-depth-aware cropdetect limit (limited-range black = 16 * 2^(bd-8))
        limit = 16 * (2 ** (bd - 8))
        cmd = [
            "ffmpeg", "-hide_banner", "-v", "info",
            "-i", video_path,
            "-vf", f"cropdetect=limit={limit}:round=2:reset=0",
            "-frames:v", "100", "-f", "null", "-"
        ]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=120, shell=False)
        crop_match = None
        for line in r.stderr.split("\n"):
            m = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", line)
            if m:
                crop_match = m
        if crop_match:
            cw, ch, cx, cy = map(int, crop_match.groups())
            if cw < ow or ch < oh:
                info.update({
                    "left": cx, "right": ow - cx - cw,
                    "top": cy, "bottom": oh - cy - ch,
                    "cropped_w": cw, "cropped_h": ch,
                    "has_bars": True,
                })
    except Exception as e:
        if logger:
            logger.warning("STAGE7_CROP_FAIL", f"cropdetect failed: {e}")
    return info


def _get_source_color_info(video_path: str) -> tuple:
    """Probe source color_range and color_primaries via ffprobe.

    Returns (source_color_range, color_standard, range_known, primaries_known)
    where:
    - source_color_range: 'limited' or 'full'
    - color_standard: 'PAL', 'NTSC', or 'BT709'
    - range_known: True if ffprobe reported color_range
    - primaries_known: True if ffprobe reported recognizable primaries

    FIX-P7-2/P7-7: Added BT.709 recognition per v7.32 §3.10.3.
    FIX-R7-S7-A1: return range_known/primaries_known flags so the caller can
    log WARNING for unknown range/primaries per v6.32 §7.19 ("若无法读取源 range
    属性,默认假设老影像为 limited range,但记录 WARNING").
    ORIGINAL: only recognized PAL/NTSC, defaulted unknown to NTSC silently.
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
                range_known = bool(cr)
                source_range = "full" if cr == "pc" else "limited"
                primaries_known = False
                # FIX-P7-2/P7-7: Add BT.709 recognition per v7.32 §3.10.3
                if cp in ("bt709", "1", "9"):
                    color_std = "BT709"
                    primaries_known = True
                elif cp in ("bt470bg", "5"):
                    color_std = "PAL"
                    primaries_known = True
                elif cp in ("smpte170m", "bt470m", "6", "4"):
                    color_std = "NTSC"
                    primaries_known = True
                else:
                    # Unknown primaries — default NTSC for old footage
                    color_std = "NTSC"
                return source_range, color_std, range_known, primaries_known
    except Exception:
        pass
    return "limited", "NTSC", False, False


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
        # FIX-R7-S7-11: >100ms must be WARNING per v7.32 §3.10.9 ("音视频时长
        # 差值 > 100ms 触发 WARNING"). ORIGINAL logged INFO.
        logger.warning("STAGE7_SYNC_WARN_100",
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
    # FIX-P7-10: Use .stem (without extension) to match stage0 naming
    # ORIGINAL: Path(args.input).name included extension, mismatch with stage0
    input_name = Path(args.input).stem

    # FIX-P7-9: Use pathlib.Path instead of os.path per CLAUDE.md §4
    # Find input - deterministic by mtime
    input_source = None
    for pat in ["step6_grain*", "step5_face*", "step4_postprocess*",
                "step3_upscale*", "step3_forced_upscale*",
                "step2_traditional*", "step1_preprocess*"]:
        pattern = f"{job_id}_{Path(input_name).stem}_{pat}"
        # ORIGINAL: glob.glob(os.path.join(temp_dir, pattern))
        candidates = list(Path(temp_dir).glob(pattern))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            input_source = str(candidates[0])
            break

    if not input_source:
        logger.error("STAGE7_NO_INPUT", "No input found")
        return 82

    output_path = params.get("output_path", "")
    if not output_path:
        # FIX-P7-9: Use pathlib.Path per CLAUDE.md §4
        output_dir = Path(params.get("output_dir", "./output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{Path(input_name).stem}_restored.mkv")

    # WSL2 redirect: if running in WSL and Windows tools need Windows paths
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    platform = params.get("platform", "auto")

    # FIX-P7-13: Also redirect temp_dir per v7.32 §3.10.10 / CLAUDE.md §4
    # ORIGINAL: only output_path was redirected, temp_dir on /mnt/ caused slow I/O
    try:
        redirected_temp, _, _, _ = wsl_redirect_path(
            temp_dir, no_wsl_redirect=no_wsl_redirect, target_platform=platform
        )
        if redirected_temp != temp_dir:
            logger.info("STAGE7_WSL_TEMP_REDIRECT",
                       f"WSL temp redirect: {temp_dir} -> {redirected_temp}")
            temp_dir = redirected_temp
    except Exception:
        pass  # wsl_redirect_path not available or not in WSL

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

    # FIX-P7-5: Preset adaptive per v6.32 §7.10:
    # On machines with ≥16 CPU cores, lower preset for better quality
    if params.get("preset_max_quality", False):
        import multiprocessing
        try:
            cpu_count = multiprocessing.cpu_count()
            if cpu_count >= 16:
                original_preset = preset
                preset = min(preset, 3)  # Try preset 2-3 for better quality
                logger.info("STAGE7_PRESET_ADAPTIVE",
                            f"CPU cores={cpu_count} >= 16, "
                            f"preset {original_preset} -> {preset}")
        except NotImplementedError:
            pass  # cpu_count not available on this platform

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
            # FIX-P7-5: Per v7.32 §3.10.20, grain_strength formula is
            # std(Y) / (1 + 0.1 × motion_index), NOT noise_score * 10.
            # noise_score*10 is an approximation only.
            grain_strength = params.get("detected_grain_strength", None)
            if grain_strength is None and "noise_score" in params:
                grain_strength = params["noise_score"] * 10
                logger.warning("STAGE7_GRAIN_APPROX",
                              "Using noise_score*10 as grain_strength "
                              "approximation. Spec v7.32 §3.10.20 requires "
                              "std(Y)/(1+0.1*motion).")
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
    # FIX-R7-S7-A4: -fflags +genpts is an OUTPUT option ("封装时使用 -fflags +genpts"
    # per v7.32 §3.10.9). ORIGINAL placed it before -i (input side). It is now
    # added on the output side below, before the output path.
    video_cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", input_source,
        "-c:v", "libsvtav1",
        "-crf", str(crf),
        "-preset", str(preset),
        "-pix_fmt", pix_fmt,
    ]

    # FIX-P7-1: BT.601→BT.709 color space conversion with range handling
    # FIX-P7-3: Probe stage output (input_source), not original input (args.input)
    # per v7.32 §3.10.3: conversion should be based on the video being encoded
    # ORIGINAL: _get_source_color_info(args.input) probed original input
    source_color_range, color_standard, range_known, primaries_known = _get_source_color_info(input_source)

    # FIX-R7-S7-A1: log WARNING for unknown source range/primaries per v6.32 §7.19
    if not range_known:
        logger.warning("STAGE7_COLOR_RANGE_UNKNOWN",
                      "Cannot read source color_range from ffprobe, assuming "
                      "limited range per v6.32 §7.19")
    if not primaries_known:
        logger.warning("STAGE7_COLOR_PRIMARIES_UNKNOWN",
                      "Cannot recognize source color_primaries from ffprobe, "
                      "defaulting to NTSC (BT.601) per v7.32 §3.10.3")

    # FIX-R7-S7-16: detect black bars to wrap color conversion with crop/restore
    # per v7.32 §3.10.3 ("色域转换前应先自动检测并裁剪黑边,转换完成后还原黑边").
    # ORIGINAL vf_parts only had zscale+dither, no crop/restore — black bars got
    # color-converted and ended up at the wrong black level.
    bar_info = _detect_black_bars_stage7(input_source, logger)
    _crop_filter = None
    _pad_filter = None
    if bar_info["has_bars"]:
        cw = bar_info["cropped_w"]
        ch = bar_info["cropped_h"]
        ow = bar_info["orig_w"]
        oh = bar_info["orig_h"]
        # §3.10.3 / §7.26: effective area <50% original OR either dim <240 -> FAILED
        if (cw * ch) < (ow * oh) * 0.5 or cw < 240 or ch < 240:
            logger.error("STAGE7_CROP_FAILED",
                        f"Black-bar crop {cw}x{ch} invalid (orig {ow}x{oh}); "
                        f"effective area <50% or dim <240px per v7.32 §3.10.3")
            return 86
        _crop_filter = f"crop={cw}:{ch}:{bar_info['left']}:{bar_info['top']}"
        # pad restores black bars at the OUTPUT format's limited-range black level
        # (ffmpeg pad default 'black' = 16/64/4096 for 8/10/16bit limited YUV).
        _pad_filter = f"pad={ow}:{oh}:{bar_info['left']}:{bar_info['top']}:black"
        logger.info("STAGE7_CROP_RESTORE",
                   f"Black bars detected: crop {cw}x{ch}, restore to {ow}x{oh}")

    # FIX-P7-2: Skip colorspace conversion if source is already BT.709
    # per v7.32 §3.10.3: only BT.601 sources need conversion
    # ORIGINAL: always applied zscale, causing BT.709 sources to get color shift
    vf_parts = []
    # FIX-R7-S7-16: crop black bars BEFORE color conversion
    if _crop_filter:
        vf_parts.append(_crop_filter)
    if color_standard == "BT709":
        logger.info("STAGE7_COLORSPACE_SKIP",
                    "Source already BT.709, skipping colorspace conversion")
    elif not params.get("colorspace_converted", False):
        matrix_in = "bt470bg" if color_standard == "PAL" else "smpte170m"
        rangein = "full" if source_color_range == "full" else "limited"
        vf_parts.append(f"zscale=matrixin={matrix_in}:matrix=bt709:rangein={rangein}:range=limited")
        logger.info("STAGE7_COLORSPACE",
                   f"BT.601({color_standard})->BT.709 zscale: matrixin={matrix_in}, rangein={rangein}")

    # FIX-P7-6: 16bit→10bit error diffusion dithering per v6.32 §阶段3
    # Per fix doc review: zscale error_diffusion requires zimg ≥3.0;
    # Ubuntu 22.04's zimg may not support it, so check availability first.
    # ORIGINAL: used simple scale without proper dithering
    input_depth = params.get("input_bit_depth", 16)
    if input_depth > 10:
        # FIX-R7-S7-18: 16bit limited range (black=4096/white=60160) is industry
        # convention, NOT ITU standard — must log WARNING per v7.32 §3.10.26.
        logger.warning("STAGE7_16BIT_QUANT",
                       "16bit limited range (black=4096/white=60160) is industry "
                       "convention, NOT ITU standard (per v7.32 §3.10.26)")
        # Check if zscale supports error_diffusion dithering
        _zscale_ed_available = False
        try:
            _zd = subprocess.run(
                ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "color=c=black:s=64x64",
                 "-vf", "zscale=dither=error_diffusion", "-f", "null", "-"],
                capture_output=True, shell=False, timeout=10
            )
            _zscale_ed_available = (_zd.returncode == 0)
        except Exception:
            pass

        if _zscale_ed_available:
            # Per v6.32 §阶段3: error_diffusion dithering in YUV444 space
            vf_parts.append("format=yuv444p16")
            vf_parts.append("zscale=rangein=limited:range=limited:dither=error_diffusion:format=yuv444p10")
            vf_parts.append("format=yuv420p10le")
            logger.info("STAGE7_DITHER", "Using error_diffusion dithering (16bit->10bit)")
        else:
            # Fallback: ordered dithering (available on all zimg versions)
            vf_parts.append("format=yuv444p16")
            vf_parts.append("zscale=rangein=limited:range=limited:dither=ordered:format=yuv444p10")
            vf_parts.append("format=yuv420p10le")
            logger.warning("STAGE7_DITHER_FALLBACK",
                          "error_diffusion not available in zscale, "
                          "using ordered dithering (16bit->10bit)")

    # FIX-R7-S7-16: restore black bars AFTER color conversion + dither so they
    # use the target bit-depth/range black level per v7.32 §3.10.3.
    if _pad_filter:
        vf_parts.append(_pad_filter)

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
    # FIX-P7-9: Use pathlib.Path per CLAUDE.md §4
    audio_path_obj = Path(temp_dir) / f"{job_id}_{input_name}.audio"
    # Fallback: also check the old naming for backward compatibility
    if not audio_path_obj.exists() or audio_path_obj.stat().st_size == 0:
        audio_path_fallback = Path(temp_dir) / f"{job_id}_{input_name}_audio_raw.wav"
        if audio_path_fallback.exists() and audio_path_fallback.stat().st_size > 0:
            audio_path_obj = audio_path_fallback
    audio_path = str(audio_path_obj)
    has_audio = audio_path_obj.exists() and audio_path_obj.stat().st_size > 0
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
        # FIX-P7-12: Use proper datetime import per CLAUDE.md §4
        # ORIGINAL: __import__('datetime').datetime.now().isoformat()
        "-metadata", f"restored_at={datetime.datetime.now().isoformat()}",
        # FIX-R7-S7-A4: -fflags +genpts is an OUTPUT option ("封装时使用 -fflags
        # +genpts" per v7.32 §3.10.9) — placed here before the output path so
        # FFmpeg regenerates container PTS at mux time without modifying samples.
        "-fflags", "+genpts",
        "-y", output_path,
    ])

    try:
        r = subprocess.run(video_cmd, capture_output=True, text=True,
                          timeout=3600*2, shell=False)
        # FIX-P7-9: Use pathlib.Path per CLAUDE.md §4
        if r.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 0:
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
