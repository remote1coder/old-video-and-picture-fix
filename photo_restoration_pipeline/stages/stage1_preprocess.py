#!/usr/bin/env python3
"""
Stage 1: Photo Traditional Preprocessing
- Color space conversion (L*a*b*, CMYK, Grayscale -> RGB)
- Scratch removal (G'MIC)
- DCT Deblocking (G'MIC -deblock, operates in DCT transform domain)
- Debanding (G'MIC)
- Grain extraction via frequency separation (high-pass filter)
- AI/artificial coloring detection (posterization + saturation heuristics)
- Single-channel degradation detection on processed image
- Single-channel correction in linear RGB space (AI not enabled only)

v6.3 Compliance:
- Supports L*a*b*, CMYK, Grayscale input conversion to RGB
- DCT deblock via G'MIC -deblock (transform domain)
- Channel degradation detection runs on processed working_path
- All thresholds read from config (not hardcoded)
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from checkpoint import compute_param_hash, build_intermediate_name  # noqa: E402

# FIX-P11-12: Photo-specific exit codes per v6.32 §7.7 / v7.32 exit code scheme
# All custom exit codes must be in 80-113 range
# ORIGINAL: mixed generic codes (82, 84) without clear semantics
PHOTO_ERR_GMIC_FAIL = 84    # G'MIC / external tool processing error
PHOTO_ERR_COLOR_CONV = 82   # I/O error during color conversion
PHOTO_ERR_OUTPUT = 86       # Output validation failure
PHOTO_ERR_COPY_FAIL = 82    # I/O error during file copy


def _detect_input_color_space(input_path: str, params: dict,
                               logger=None) -> str:
    """Detect embedded ICC profile and warn if missing/mismatched.

    FIX-P11-10: Per v6.32 §4.2.1: detect-then-warn, do NOT auto-convert.
    High DPI scans without ICC profile are flagged for user verification.
    """
    try:
        from PIL import Image
        img = Image.open(input_path)
        icc = img.info.get("icc_profile")
        if icc:
            # FIX-P11-11: Parse ICC profile content per v6.32 §4.2.1
            # ORIGINAL: only returned "embedded" without identifying profile
            try:
                import io
                from PIL import ImageCms
                profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                profile_name = ImageCms.getProfileName(
                    profile.profile).lower()
                if "adobe" in profile_name:
                    cs_name = "AdobeRGB"
                elif "prophoto" in profile_name or "romm" in profile_name:
                    cs_name = "ProPhoto"
                elif "srgb" in profile_name:
                    cs_name = "sRGB"
                else:
                    cs_name = "embedded"
            except Exception:
                cs_name = "embedded"
            logger.info("STAGE1_ICC_DETECTED",
                       f"ICC profile detected: {cs_name}",
                       profile_size=len(icc))
            return cs_name
        # No ICC profile: warn based on DPI
        dpi = params.get("dpi", 0)
        if dpi and dpi > 1200:
            logger.warning("STAGE1_ICC_MISSING",
                          f"High DPI ({dpi}) scan without ICC profile. "
                          f"Color space may be Adobe RGB or ProPhoto. "
                          f"Please verify in RawTherapee/GIMP before "
                          f"processing.")
        else:
            logger.info("STAGE1_ICC_ASSUME_SRGB",
                       "No ICC profile, assuming sRGB (low DPI scan)")
        return "unknown"
    except Exception:
        return "unknown"


def _detect_color_space(img_path: str) -> tuple:
    """Detect image color space and return (mode, needs_conversion).

    Returns:
        (detected_mode, target_mode_for_pil)
    """
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            mode = img.mode
            if mode == "RGB":
                return "RGB", None
            elif mode in ("RGBA", "P", "PA"):
                return mode, "RGB"
            elif mode == "L":
                return "Grayscale", "RGB"
            elif mode in ("CMYK", "YCbCr"):
                return mode, "RGB"
            elif mode == "LAB":
                return "L*a*b*", "RGB"
            elif mode in ("I;16", "I;16B"):
                return "16-bit Gray", "RGB"
            elif mode == "HSV":
                return "HSV", "RGB"
            else:
                return mode, "RGB"
    except Exception:
        return "Unknown", "RGB"


def _convert_to_rgb(input_path: str, output_path: str, logger=None) -> bool:
    """Convert image to RGB color space if needed.

    Bug P-12 fix: preserves ICC profile when converting color spaces.
    """
    try:
        from PIL import Image
        with Image.open(input_path) as img:
            mode = img.mode
            # Extract ICC profile before conversion
            icc_profile = img.info.get("icc_profile")

            if mode == "RGB":
                # Already RGB, just copy (preserve original ICC)
                # FIX-P11-9: Use copyfile (not copy2) per v7.32 §3.10.10
                # copy2 copies metadata which causes PermissionError on WSL2 NTFS
                shutil.copyfile(input_path, output_path)
                try:
                    src_stat = os.stat(input_path)
                    os.utime(output_path,
                            (src_stat.st_atime, src_stat.st_mtime))
                except OSError:
                    pass
                return True
            elif mode in ("RGBA", "P", "PA"):
                rgb_img = img.convert("RGB")
                save_kwargs = {}
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile
                rgb_img.save(output_path, **save_kwargs)
                if logger:
                    logger.info("STAGE1_COLOR", f"Converted {mode} -> RGB (ICC preserved={bool(icc_profile)})")
                return True
            elif mode == "L":
                rgb_img = img.convert("RGB")
                save_kwargs = {}
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile
                rgb_img.save(output_path, **save_kwargs)
                if logger:
                    logger.info("STAGE1_COLOR", "Converted Grayscale -> RGB")
                return True
            elif mode in ("CMYK", "YCbCr"):
                rgb_img = img.convert("RGB")
                save_kwargs = {}
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile
                rgb_img.save(output_path, **save_kwargs)
                if logger:
                    logger.info("STAGE1_COLOR", f"Converted {mode} -> RGB (ICC preserved={bool(icc_profile)})")
                return True
            elif mode == "LAB":
                rgb_img = img.convert("RGB")
                save_kwargs = {}
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile
                rgb_img.save(output_path, **save_kwargs)
                if logger:
                    logger.info("STAGE1_COLOR", "Converted L*a*b* -> RGB")
                return True
            elif mode in ("I;16", "I;16B"):
                # 16-bit grayscale: preserve full precision (Bug 11.9 fix)
                # BUG-048 fix: I;16 mode only supports 2D arrays in PIL.
                # For 3-channel 16-bit RGB, construct image safely.
                import numpy as np
                arr_16 = np.array(img, dtype=np.uint16)
                rgb_16 = np.stack([arr_16, arr_16, arr_16], axis=-1).astype(np.uint16)
                try:
                    # Try direct fromarray (works in newer Pillow versions for TIFF)
                    out_img = Image.fromarray(rgb_16)
                except (ValueError, TypeError):
                    # FIX-P11-9: Fallback must produce RGB output, not single-channel I;16
                    # per v6.32 §4.2.1: all outputs must be RGB for downstream processing
                    # ORIGINAL: fell back to Image.fromarray(arr_16, mode='I;16') which
                    # saves as single-channel, breaking downstream RGB pipeline
                    rgb_16_reshaped = rgb_16.reshape((rgb_16.shape[0], rgb_16.shape[1], 3))
                    try:
                        out_img = Image.fromarray(rgb_16_reshaped)
                    except (ValueError, TypeError):
                        # Final fallback: use imageio for proper 16-bit RGB TIFF
                        import imageio.v3 as iio
                        iio.imwrite(output_path, rgb_16_reshaped)
                        if logger:
                            logger.info("STAGE1_COLOR",
                                       "Converted 16-bit -> RGB via imageio fallback")
                        return True
                out_img.save(output_path)
                if logger:
                    logger.info("STAGE1_COLOR",
                               "Converted 16-bit Gray -> RGB (16bit preserved, BUG-048 safe)")
                return True
            elif mode == "HSV":
                rgb_img = img.convert("RGB")
                save_kwargs = {}
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile
                rgb_img.save(output_path, **save_kwargs)
                if logger:
                    logger.info("STAGE1_COLOR", "Converted HSV -> RGB")
                return True
            else:
                # Fallback: try converting any mode to RGB
                rgb_img = img.convert("RGB")
                save_kwargs = {}
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile
                rgb_img.save(output_path, **save_kwargs)
                if logger:
                    logger.info("STAGE1_COLOR", f"Converted {mode} -> RGB (fallback, ICC preserved={bool(icc_profile)})")
                return True
    except Exception as e:
        if logger:
            logger.error("STAGE1_COLOR_FAIL", f"Color space conversion failed: {e}")
        return False


def run_gmic(input_path: str, output_path: str, *gmic_args) -> tuple:
    """Run G'MIC with shell=False safety.

    Returns:
        (success: bool, stderr: str) tuple per Bug 11.20 fix.
    """
    cmd = ["gmic", input_path]
    cmd.extend(list(gmic_args))
    cmd.extend(["-o", output_path])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, shell=False)
        if r.returncode != 0 and r.stderr:
            return False, r.stderr[:1000]
        return r.returncode == 0, ""
    except Exception as e:
        return False, str(e)


def detect_channel_degradation(img_path: str, params: dict) -> dict:
    """Detect single-channel degradation in photo.

    Uses config-driven thresholds (not hardcoded).
    Handles 16-bit I;16 images with full precision.
    """
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(img_path)
        # Preserve 16-bit precision for I;16 images
        if img.mode == "I;16":
            arr = np.array(img).astype(np.float32) / 65535.0
        else:
            arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0

        # Compute per-channel stats
        means = [arr[:, :, c].mean() for c in range(3)]
        stds = [arr[:, :, c].std() for c in range(3)]

        result = {
            "degraded": False,
            "channels": [],
            "mean_diff": 0.0,
            "std_diff": 0.0,
        }

        # Check mean differences using config thresholds
        mean_threshold = params.get("channel_degradation_mean_threshold", 0.08)
        std_threshold = params.get("channel_degradation_std_threshold", 0.15)

        max_mean_diff = max(abs(means[i] - means[j])
                           for i in range(3) for j in range(i+1, 3))
        # FIX-P11-5: Relative std difference per v6.32 §4.2.1
        # Spec: "任意两通道 std 差异 > 15%（相对值）"
        # ORIGINAL: max_std_diff = max(abs(stds[i] - stds[j])) was absolute
        max_std_diff = 0.0
        for i in range(3):
            for j in range(i+1, 3):
                avg_std = (stds[i] + stds[j]) / 2.0
                if avg_std > 1e-8:
                    rel_diff = abs(stds[i] - stds[j]) / avg_std
                    max_std_diff = max(max_std_diff, rel_diff)

        if max_mean_diff > mean_threshold or max_std_diff > std_threshold:
            result["degraded"] = True
            result["mean_diff"] = max_mean_diff
            result["std_diff"] = max_std_diff

        return result
    except Exception:
        return {"degraded": False}


def _sRGB_to_linear(c: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear RGB using IEC 61966-2-1."""
    c = np.clip(c, 0.0, 1.0)
    mask = c <= 0.04045
    linear = np.empty_like(c, dtype=np.float32)
    linear[mask] = c[mask] / 12.92
    linear[~mask] = np.power((c[~mask] + 0.055) / 1.055, 2.4)
    return linear


def _linear_to_sRGB(c: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB using IEC 61966-2-1."""
    c = np.clip(c, 0.0, 1.0)
    mask = c <= 0.0031308
    srgb = np.empty_like(c, dtype=np.float32)
    srgb[mask] = c[mask] * 12.92
    srgb[~mask] = 1.055 * np.power(c[~mask], 1.0 / 2.4) - 0.055
    return srgb


def _reinhard_tone_map(c: np.ndarray, white_point: float = 2.0) -> np.ndarray:
    """Apply Reinhard tone mapping to compress highlights."""
    return c * (1.0 + c / (white_point * white_point)) / (1.0 + c)


def extract_grain_pattern(img_path: str, output_path: str,
                            params: dict = None, logger=None) -> bool:
    """Extract film grain pattern using high-pass filtering.
    
    Per v6.3 §4.2.1: grain extraction for analysis and later synthesis.
    Uses frequency separation to isolate high-frequency grain from low-frequency image.
    """
    try:
        from PIL import Image
        import numpy as np
        from scipy.ndimage import gaussian_filter
        
        img = Image.open(img_path)
        if img.mode == "I;16":
            arr = np.array(img, dtype=np.float32) / 65535.0
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
        else:
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        
        # Frequency separation: low-pass = image structure, high-pass = grain
        grain_strength = params.get("grain_strength", 0.3) if params else 0.3
        grain_size = params.get("grain_size", 0.5) if params else 0.5
        grain_roughness = params.get("grain_roughness", 0) if params else 0
        # FIX-P11-3: grain_size maps to sigma per v7.32 §3.10.4
        # grain_size (0.5-2.0) → sigma: larger = coarser grain
        # ORIGINAL: hardcoded sigma=2.0, ignored grain_size
        sigma = grain_size * 2.0  # grain_size=0.5→1.0, grain_size=1.0→2.0, grain_size=2.0→4.0

        # FIX-P11-9: Laplacian pyramid frequency separation per v6.32 §4.2.1
        # ORIGINAL: single Gaussian high-pass, not true frequency separation
        levels = max(1, int(grain_size * 3))
        grain = np.zeros_like(arr)
        for c in range(3):
            channel = arr[:, :, c]
            pyramid = []
            current = channel.copy()
            for _ in range(levels):
                low = gaussian_filter(current, sigma=sigma)
                high = current - low
                pyramid.append(high)
                current = low

            # Weight levels by roughness (0=pure fine/high-freq, 1=pure coarse/low-freq)
            total_weight = 0.0
            for i, level in enumerate(pyramid):
                weight = (1.0 - grain_roughness) ** i  # roughness=0 → all weight on finest
                grain[:, :, c] += level * weight * grain_strength
                total_weight += weight

            if total_weight > 1e-8:
                grain[:, :, c] /= total_weight  # Normalize weights
        
        # Save grain pattern as 16-bit TIFF for precision
        grain_clipped = np.clip(grain + 0.5, 0.0, 1.0)
        if img.mode == "I;16":
            grain_uint16 = (grain_clipped * 65535).astype(np.uint16)
            # FIX-P11-11: I;16 single-channel breaks downstream RGB pipeline
            # Save as 3-channel 16-bit TIFF instead
            try:
                import imageio.v3 as iio
                iio.imwrite(output_path, grain_uint16)
            except Exception:
                # Fallback: save as 8-bit (lossy but avoids I;16 single-channel)
                grain_uint8 = (grain_clipped * 255).astype(np.uint8)
                out_img = Image.fromarray(grain_uint8)
                out_img.save(output_path)
        else:
            grain_uint8 = (grain_clipped * 255).astype(np.uint8)
            out_img = Image.fromarray(grain_uint8)
            out_img.save(output_path)
        
        if logger:
            logger.info("STAGE1_GRAIN_EXTRACT", f"Grain pattern saved: {output_path}")
        return True
    except Exception as e:
        if logger:
            logger.warning("STAGE1_GRAIN_EXTRACT_FAIL", f"Grain extraction failed: {e}")
        return False


def detect_ai_coloring(img_path: str, params: dict = None,
                       logger=None) -> dict:
    """Detect signs of artificial/AI coloring in photos.

    Heuristics:
    - Unnatural color saturation spikes (posterization)
    - Abrupt hue transitions
    - High chroma noise in supposedly uniform regions

    Returns dict with detected (bool) and confidence (float).
    """
    # FIX-P11-6: Config-driven thresholds per v7.32 §3.10.16
    # ORIGINAL: hardcoded 0.3 and 0.5
    params = params or {}
    posterization_threshold = params.get(
        "ai_coloring_posterization_threshold", 0.3)
    detection_threshold = params.get(
        "ai_coloring_detection_threshold", 0.5)

    try:
        from PIL import Image
        import numpy as np

        img = Image.open(img_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0

        # Convert to HSV-like space for analysis
        max_c = arr.max(axis=2)
        min_c = arr.min(axis=2)
        delta = max_c - min_c

        # Saturation
        sat = np.where(max_c > 0, delta / (max_c + 1e-8), 0)

        # Detect posterization: count unique colors per channel
        posterization_score = 0.0
        for c in range(3):
            unique_ratio = len(np.unique(
                (arr[:, :, c] * 255).astype(np.uint8))) / 256.0
            if unique_ratio < posterization_threshold:
                posterization_score += 1.0
        posterization_score /= 3.0

        # Detect unnatural hue transitions
        gray_mask = delta < 0.05
        unnatural_sat = np.mean(sat[gray_mask]) if gray_mask.any() else 0.0

        # Confidence: combination of posterization + unnatural saturation
        confidence = min(1.0, posterization_score * 0.6 + unnatural_sat * 4.0)
        detected = confidence > detection_threshold

        if logger:
            logger.info("STAGE1_COLORING_DETECT",
                       f"AI coloring detection: detected={detected}, "
                       f"confidence={confidence:.2f}, "
                       f"posterization={posterization_score:.2f}, "
                       f"unnatural_sat={unnatural_sat:.4f}")

        return {"detected": detected, "confidence": confidence,
                "posterization_score": posterization_score,
                "unnatural_saturation": float(unnatural_sat)}
    except Exception as e:
        if logger:
            logger.warning("STAGE1_COLORING_DETECT_FAIL",
                          f"Coloring detection failed: {e}")
        return {"detected": False, "confidence": 0.0}


def apply_channel_correction(img_path: str, output_path: str,
                              params: dict, logger=None) -> bool:
    """Apply single-channel correction in linear RGB space.

    Only applied when degradation is severe and AI is NOT enabled.
    Uses IEC 61966-2-1 color space conversion.
    Respects all channel_correction_* config parameters.
    """
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(img_path)
        # Preserve 16-bit if available
        if img.mode == "I;16":
            arr = np.array(img).astype(np.float32) / 65535.0
        else:
            arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0

        # Convert to linear space
        formula = params.get("channel_correction_formula", "iec61966-2-1")
        if formula == "iec61966-2-1":
            linear = _sRGB_to_linear(arr)
        else:
            # Fallback: simple gamma 2.2
            linear = np.power(np.clip(arr, 0.0, 1.0), 2.2)

        # Compute per-channel stats in linear space
        means = [linear[:, :, c].mean() for c in range(3)]
        stds = [linear[:, :, c].std() for c in range(3)]

        # Identify degraded channel(s)
        mean_threshold = params.get("channel_degradation_mean_threshold_severe", 0.15)
        std_threshold = params.get("channel_degradation_std_threshold_severe", 0.25)

        max_mean_diff = max(abs(means[i] - means[j])
                           for i in range(3) for j in range(i+1, 3))
        # FIX-P11-5: Relative std difference per v6.32 §4.2.1 (consistent with detection)
        max_std_diff = 0.0
        for i in range(3):
            for j in range(i+1, 3):
                avg_std = (stds[i] + stds[j]) / 2.0
                if avg_std > 1e-8:
                    rel_diff = abs(stds[i] - stds[j]) / avg_std
                    max_std_diff = max(max_std_diff, rel_diff)

        if max_mean_diff < mean_threshold and max_std_diff < std_threshold:
            if logger:
                logger.info("STAGE1_CHANNEL_SKIP",
                           "Degradation below severe threshold, skipping correction")
            return False

        # Find target reference: use median mean/std as reference
        sorted_means = sorted(means)
        target_mean = sorted_means[1]  # median
        sorted_stds = sorted(stds)
        target_std = sorted_stds[1]

        corrected = linear.copy()
        # PHOTO-AUDIT-008 fix: independent denoise + level correction per v6.32 §4.2.1
        # For each degraded channel, apply median denoise then histogram-level correction
        # against the reference (median) channel
        ref_channel_idx = np.argsort(means)[1]  # median mean channel as reference
        ref_ch = corrected[:, :, ref_channel_idx]
        for c in range(3):
            ch = corrected[:, :, c]
            ch_mean = means[c]
            ch_std = stds[c]

            # Skip if close to reference
            if abs(ch_mean - target_mean) < 0.01 and abs(ch_std - target_std) < 0.01:
                continue

            # Step 1: Independent denoise (median filter to remove impulse noise)
            try:
                from scipy.ndimage import median_filter
                ch_denoised = median_filter(ch, size=3)
            except Exception:
                ch_denoised = ch

            # Step 2: Level correction using percentile-based histogram matching to reference
            p_low, p_high = np.percentile(ch_denoised, [1, 99])
            ref_low, ref_high = np.percentile(ref_ch, [1, 99])
            if p_high > p_low and ref_high > ref_low:
                ch_corrected = (ch_denoised - p_low) / (p_high - p_low) * (ref_high - ref_low) + ref_low
            else:
                ch_corrected = ch_denoised

            corrected[:, :, c] = ch_corrected

        # Normalize if configured
        if params.get("channel_correction_normalize", True):
            # FIX-P11-2: Only normalize channels that deviate from reference
            # ORIGINAL: applied to all channels unconditionally + wrong formula
            # mapped to [target_mean, target_mean+target_std] instead of centered
            # Per v6.32 §7.16: standardize to reference statistics
            for c in range(3):
                ch = corrected[:, :, c]
                ch_mean = ch.mean()
                ch_std = ch.std()
                if abs(ch_mean - target_mean) > 0.01 or abs(ch_std - target_std) > 0.01:
                    if ch_std > 1e-8:
                        # Standardize to reference statistics
                        corrected[:, :, c] = (ch - ch_mean) / ch_std * target_std + target_mean
                    else:
                        # Near-constant channel: just shift mean
                        corrected[:, :, c] = ch - ch_mean + target_mean

        # Clamp to valid range
        corrected = np.clip(corrected, 0.0, 1.0)

        # Optional soft compression
        if params.get("channel_correction_soft_compress", True):
            corrected = _reinhard_tone_map(corrected)
            corrected = np.clip(corrected, 0.0, 1.0)

        # Convert back to sRGB
        if formula == "iec61966-2-1":
            srgb = _linear_to_sRGB(corrected)
        else:
            srgb = np.power(np.clip(corrected, 0.0, 1.0), 1.0 / 2.2)
        srgb = np.clip(srgb, 0.0, 1.0)

        # Round and denormalize to configured bit depth
        bit_depth = params.get("photo_bit_depth", 16)
        if bit_depth == 16:
            max_val = 65535.0
            dtype = np.uint16
        else:
            max_val = 255.0
            dtype = np.uint8

        # Apply rounding if configured
        if params.get("channel_correction_round", True):
            out_arr = np.rint(srgb * max_val).astype(dtype)
        else:
            out_arr = (srgb * max_val).astype(dtype)

        # Apply clamp if configured
        if params.get("channel_correction_clamp", True):
            if bit_depth == 16:
                out_arr = np.clip(out_arr, 0, 65535).astype(dtype)
            else:
                out_arr = np.clip(out_arr, 0, 255).astype(dtype)

        out_img = Image.fromarray(out_arr)
        out_img.save(output_path)

        if logger:
            logger.info("STAGE1_CHANNEL_CORRECT",
                       f"Channel correction applied: mean_diff={max_mean_diff:.4f}, std_diff={max_std_diff:.4f}")
        return True
    except Exception as e:
        if logger:
            logger.warning("STAGE1_CHANNEL_FAIL", f"Channel correction failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--params")
    parser.add_argument("--params-file")
    # FIX-P11-7: --config is passed by main.py but unused here;
    # all config is provided via --params JSON. Kept for argparse compat.
    parser.add_argument("--config")
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()

    # BUG-057 support: --params-file fallback for long JSON
    if getattr(args, "params_file", None):
        with open(args.params_file, "r", encoding="utf-8") as f:
            params = json.load(f)
    else:
        params = json.loads(args.params) if args.params else {}
    log_dir = params.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, job_id=args.job_id)

    temp_dir = params.get("temp_dir", "./temp")
    # FIX-P11-6: v7.32 §3.10.10: WSL2 path redirect for temp_dir
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    platform = params.get("platform", "auto")
    try:
        from checkpoint import wsl_redirect_path
        redirected_temp, orig_temp, _, _ = wsl_redirect_path(
            temp_dir, no_wsl_redirect=no_wsl_redirect,
            target_platform=platform
        )
        if redirected_temp != temp_dir:
            logger.info("STAGE1_WSL_REDIRECT",
                        f"temp_dir: {temp_dir} -> {redirected_temp}")
            temp_dir = redirected_temp
            params["temp_dir"] = temp_dir
    except Exception:
        pass  # wsl_redirect_path may not be available on native Windows
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    input_path = args.input
    input_name = Path(input_path).name

    # Check for existing intermediate
    param_hash = compute_param_hash(input_path, params)
    inter_name = build_intermediate_name(
        args.job_id, input_name, 1, "preprocess", param_hash, "tif"
    )
    # FIX-P10-11/P11-8: Use pathlib.Path consistently per CLAUDE.md §4
    # ORIGINAL: os.path.join + os.path.exists + os.path.getsize
    inter_path = str(Path(temp_dir) / inter_name)
    inter_path_p = Path(temp_dir) / inter_name

    if inter_path_p.exists() and inter_path_p.stat().st_size > 0:
        logger.info("STAGE1_SKIP", "Intermediate exists", path=inter_path)
        return 3

    logger.info("STAGE1_START", "Photo preprocessing", input=input_path)

    # Step 0: Detect and convert color space if needed
    color_mode, needs_conv = _detect_color_space(input_path)
    if needs_conv:
        logger.info("STAGE1_COLOR_DETECT", f"Detected {color_mode}, converting to RGB")
        rgb_path = str(Path(temp_dir) /
            f"{args.job_id}_{Path(input_name).stem}_rgb.tif")
        if not _convert_to_rgb(input_path, rgb_path, logger):
            logger.error("STAGE1_COLOR_FAIL", "Color conversion failed")
            return PHOTO_ERR_COLOR_CONV
        working_input = rgb_path
    else:
        working_input = input_path
        logger.info("STAGE1_COLOR_OK", f"Input is already {color_mode}")

    # FIX-P11-10: v6.32 §4.2.1 — detect ICC profile and warn if missing
    # High-DPI scans without ICC need user verification; no auto-conversion
    # Respect user override via config input_color_space
    user_cs = params.get("input_color_space", "auto")
    if user_cs == "auto":
        detected_cs = _detect_input_color_space(working_input, params, logger)
    else:
        detected_cs = user_cs
        logger.info("STAGE1_ICC_USER_OVERRIDE",
                   f"User specified color space: {detected_cs}")

    # Save color space info to sidecar JSON for downstream stages
    # FIX-P11-11: icc_profile_embedded now checks for any recognized ICC name
    # ORIGINAL: only checked == "embedded", missed AdobeRGB/ProPhoto/sRGB
    icc_names = {"embedded", "AdobeRGB", "ProPhoto", "sRGB"}
    color_meta = {
        "input_color_space": detected_cs,
        "icc_profile_embedded": (detected_cs in icc_names),
        "user_override": (user_cs != "auto"),
    }
    color_meta_path = str(Path(temp_dir) /
        f"{args.job_id}_{Path(input_name).stem}_color_meta.json")
    try:
        with open(color_meta_path, "w", encoding="utf-8") as f:
            json.dump(color_meta, f, indent=2)
        logger.info("STAGE1_COLOR_META",
                   f"Saved color space metadata: {color_meta_path}")
    except Exception:
        pass

    # FIX-P11-4: Channel degradation detection BEFORE destructive repairs
    # v7.32 §3.10.6: 前置补偿 in stage1 传统修复前
    # ORIGINAL: channel degradation detection was after all repairs (line 667-695)
    degradation = detect_channel_degradation(working_input, params)

    # Save channel metadata
    channel_meta = {
        "channel_degradation": degradation.get("degraded", False),
        "mean_diff": degradation.get("mean_diff", 0.0),
        "std_diff": degradation.get("std_diff", 0.0),
    }
    meta_path = str(Path(temp_dir) /
        f"{args.job_id}_{Path(input_name).stem}_channel_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(channel_meta, f, indent=2)
    logger.info("STAGE1_CHANNEL_META", f"Saved channel metadata: {meta_path}")

    if degradation["degraded"]:
        ai_enabled = params.get("ai_enabled", False)
        # v7.32 §3.10.6: 前置补偿 condition
        mean_severe = degradation.get("mean_diff", 0) > params.get(
            "channel_degradation_mean_threshold_severe", 0.15)
        std_severe = degradation.get("std_diff", 0) > params.get(
            "channel_degradation_std_threshold_severe", 0.25)
        if not ai_enabled and (mean_severe or std_severe):
            logger.info("STAGE1_CHANNEL_CORR",
                       "Applying pre-compensation before repairs")
            corr_out = working_input + ".channel.tif"
            if apply_channel_correction(working_input, corr_out, params, logger):
                if Path(corr_out).exists():
                    working_input = corr_out

    # Convert to 16bit TIFF for processing
    working_path = str(Path(temp_dir) /
        f"{args.job_id}_{Path(input_name).stem}_16bit.tif")

    # BUG-049 fix: Explicitly check gmic returncode
    try:
        r = subprocess.run(
            ["gmic", working_input,
             "-type", "ushort",
             "-o", working_path],
            capture_output=True, timeout=120, shell=False
        )
        if r.returncode != 0:
            stderr = (r.stderr or b"").decode("utf-8", errors="replace")[:500]
            logger.error("STAGE1_CONVERT_FAIL", f"gmic exited with code {r.returncode}: {stderr}")
            return PHOTO_ERR_GMIC_FAIL
    except Exception as e:
        logger.error("STAGE1_CONVERT_FAIL", str(e))
        return PHOTO_ERR_COLOR_CONV

    # FIX-P11-3: Grain extraction BEFORE destructive repairs per v6.32 §4.2.1
    # "颗粒保护前置：去划痕/去噪前，若需保留原生胶片颗粒，先用频率分离将高频颗粒层隔离"
    # ORIGINAL: grain extraction was after scratch removal (line 609-613), wrong order
    if params.get("grain_synthesis", True):
        grain_out = str(Path(temp_dir) /
            f"{args.job_id}_{Path(input_name).stem}_grain.tif")
        extract_grain_pattern(working_path, grain_out, params, logger)

    # Scratch removal
    if params.get("scratch_remove", True):
        logger.info("STAGE1_SCRATCH", "Removing scratches")
        scratch_out = working_path + ".scratch.tif"
        # FIX-P11-1: G'MIC commands require '-' prefix per CLI syntax
        # ORIGINAL: ["fx_remove_scratches", ...] → G'MIC interprets as filename
        scratch_strength = str(params.get("scratch_remove_strength", 20))
        success = False
        stderr = ""
        for cmd_name in ["-fx_remove_scratches", "-remove_scratches"]:
            success, stderr = run_gmic(working_path, scratch_out,
                                       cmd_name, scratch_strength)
            if success:
                break
            logger.warning("STAGE1_SCRATCH_CMD",
                          f"G'MIC {cmd_name} failed, trying next: "
                          f"{stderr[:200]}")

        # Fallback to despeckle if all scratch commands fail
        if not success:
            logger.warning("STAGE1_SCRATCH_FALLBACK",
                          "fx_remove_scratches unavailable, "
                          "falling back to -despeckle")
            success, stderr = run_gmic(working_path, scratch_out,
                                       "-despeckle", scratch_strength)

        if success and Path(scratch_out).exists():
            working_path = scratch_out
        elif not success:
            logger.error("STAGE1_SCRATCH_FAIL",
                        f"G'MIC scratch removal failed: {stderr}")
            return PHOTO_ERR_GMIC_FAIL

    # Deblock (DCT-based via G'MIC)
    if params.get("deblock_strength", 0) > 0:
        logger.info("STAGE1_DEBLOCK", "DCT deblocking")
        deblock_out = working_path + ".deblock.tif"
        # G'MIC -deblock operates in DCT transform domain for JPEG-like artifacts
        success, stderr = run_gmic(working_path, deblock_out,
                          "-deblock",
                          str(params.get("deblock_strength", 25)))
        if success and Path(deblock_out).exists():
            working_path = deblock_out
        elif not success:
            logger.error("STAGE1_DEBLOCK_FAIL", f"G'MIC -deblock failed: {stderr}")
            return PHOTO_ERR_GMIC_FAIL

    # Deband
    if params.get("deband_threshold", 0) > 0:
        logger.info("STAGE1_DEBAND", "Debanding")
        deband_out = working_path + ".deband.tif"
        success, stderr = run_gmic(working_path, deband_out,
                          "-deband",
                          str(params.get("deband_threshold", 1.2)))
        if success and Path(deband_out).exists():
            working_path = deband_out
        elif not success:
            logger.error("STAGE1_DEBAND_FAIL", f"G'MIC -deband failed: {stderr}")
            return PHOTO_ERR_GMIC_FAIL

    # Grain extraction moved BEFORE scratch removal per v6.32 §4.2.1 (FIX-P11-3)
    # Original location was here (after deband), now extracted earlier

    # AI/artificial coloring detection (v6.3 §4.2.1)
    # FIX-P11-6: Pass params for config-driven thresholds
    coloring_info = detect_ai_coloring(working_path, params, logger)
    if coloring_info["detected"]:
        logger.warning("STAGE1_AI_COLORING",
                      f"Possible AI/artificial coloring detected (confidence={coloring_info['confidence']:.2f})")

    # NOTE: Channel degradation detection moved BEFORE repairs per v7.32 §3.10.6 (FIX-P11-4)
    # See Step 0.5 above

    # FIX-P11-10: Use shutil.copyfile (not copy2) per v7.32 §3.10.10
    # copy2 copies metadata which causes PermissionError on WSL2 NTFS mounts
    # with metadata=disabled. copyfile only copies data content.
    # ORIGINAL: shutil.copy2(working_path, inter_path)
    shutil.copyfile(working_path, inter_path)
    # Verify copy integrity
    # FIX-P10-11/P11-8: Use Path consistently per CLAUDE.md §4
    if Path(working_path).stat().st_size != inter_path_p.stat().st_size:
        logger.error("STAGE1_COPY_FAIL",
                    "Intermediate file size mismatch after copy")
        return PHOTO_ERR_COPY_FAIL
    # Sync mtime/atime manually (equivalent to copy2 metadata preservation)
    # FIX-P11-10: Use Path.stat() per CLAUDE.md §4 (not os.stat)
    try:
        src_stat = Path(working_path).stat()
        os.utime(inter_path, (src_stat.st_atime, src_stat.st_mtime))
    except OSError:
        pass  # Non-critical: mtime sync failure on WSL2 cross-filesystem

    if inter_path_p.exists():
        # FIX-P11-8: Cleanup intermediates if requested per v7.32 §3.10.17
        if args.cleanup_intermediates:
            stem = f"{args.job_id}_{Path(input_name).stem}"
            for suffix in [".channel.tif", ".scratch.tif", ".deblock.tif",
                           ".deband.tif", "_grain.tif", "_rgb.tif",
                           "_16bit.tif", "_color_meta.json",
                           "_channel_meta.json"]:
                p = Path(temp_dir) / f"{stem}{suffix}"
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
        logger.info("STAGE1_DONE", "Preprocessing complete", output=inter_path)
        return 0
    else:
        logger.error("STAGE1_FAILED", "Output not created")
        return PHOTO_ERR_OUTPUT


if __name__ == "__main__":
    # FIX-P11-5: main() must return explicit int exit code per v7.32 §3.5
    # ORIGINAL: sys.exit(main() or 0) — "or 0" masks None returns
    rc = main()
    if rc is None:
        rc = 86  # validation failure — main() must always return explicit exit code
    sys.exit(rc)
