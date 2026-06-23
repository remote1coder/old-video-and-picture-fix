#!/usr/bin/env python3
"""
Stage 2: Photo AI Pre-processing — Single-Channel Correction
IEC 61966-2-1 compliant per-channel correction when AI is enabled
and channel degradation is detected.

# FIX-P10-2: New stage per v7.32 §3.10.6 / v6.32 §7.16
# Runs between stage1_preprocess and stage3_ai_upscale
# Only active when ai_enabled=true AND channel degradation detected

Spec reference:
- v7.32 §3.1: stage2_ai_pre.py = "AI 超分前预处理：单通道校正"
- v7.32 §3.10.6: single-channel correction workflow
- v6.32 §7.16: IEC 61966-2-1 color space specification for channel correction

Workflow:
1. Read stage1 channel_degradation metadata
2. If no degradation detected → skip (exit 3)
3. Normalize to 0-1 float32
4. IEC 61966-2-1 sRGB → linear conversion
5. Per-channel independent denoise + level correction
6. IEC 61966-2-1 linear → sRGB conversion
7. Clamp + round (not truncate) back to original bit depth
8. Save corrected intermediate
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from checkpoint import compute_param_hash, build_intermediate_name  # noqa: E402


def _sRGB_to_linear(c: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear RGB using IEC 61966-2-1.

    Per v6.32 §7.16:
    - C_linear = C_sRGB / 12.92  (if C_sRGB <= 0.04045)
    - C_linear = ((C_sRGB + 0.055) / 1.055) ^ 2.4  (if C_sRGB > 0.04045)
    """
    c = np.clip(c, 0.0, 1.0)
    mask = c <= 0.04045
    linear = np.empty_like(c, dtype=np.float32)
    linear[mask] = c[mask] / 12.92
    linear[~mask] = np.power((c[~mask] + 0.055) / 1.055, 2.4)
    return linear


def _linear_to_sRGB(c: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB using IEC 61966-2-1.

    Per v6.32 §7.16:
    - C_sRGB = C_linear * 12.92  (if C_linear <= 0.0031308)
    - C_sRGB = 1.055 * C_linear^(1/2.4) - 0.055  (if C_linear > 0.0031308)
    """
    c = np.clip(c, 0.0, 1.0)
    mask = c <= 0.0031308
    srgb = np.empty_like(c, dtype=np.float32)
    srgb[mask] = c[mask] * 12.92
    srgb[~mask] = 1.055 * np.power(c[~mask], 1.0 / 2.4) - 0.055
    return srgb


def _reinhard_tone_map(c: np.ndarray,
                       white_point: float = 2.0) -> np.ndarray:
    """Apply Reinhard tone mapping to compress highlights.

    Per v6.32 §7.16: optional smooth compression for severely
    out-of-range pixel values after correction.
    """
    return c * (1.0 + c / (white_point * white_point)) / (1.0 + c)


def correct_channel(ch_linear: np.ndarray, params: dict,
                    ref_linear: np.ndarray,
                    ref_mean: float, ref_std: float) -> np.ndarray:
    """Correct a single degraded channel in linear space.

    Per v6.32 §7.16:
    1. Independent denoise (median filter for impulse noise)
    2. Percentile-based histogram level correction against reference
    3. Optional normalization
    4. Optional soft compression (Reinhard tone mapping)
    """
    # Step 1: Independent denoise
    try:
        from scipy.ndimage import median_filter
        ch_denoised = median_filter(ch_linear, size=3)
    except Exception:
        ch_denoised = ch_linear

    # Step 2: Percentile-based histogram matching to reference
    p_low, p_high = np.percentile(ch_denoised, [1, 99])
    ref_low, ref_high = np.percentile(ref_linear, [1, 99])
    if p_high > p_low and ref_high > ref_low:
        ch_corrected = (
            (ch_denoised - p_low) / (p_high - p_low)
            * (ref_high - ref_low) + ref_low
        )
    else:
        ch_corrected = ch_denoised

    # Step 3: Normalize if configured
    if params.get("channel_correction_normalize", True):
        ch_min, ch_max = ch_corrected.min(), ch_corrected.max()
        if ch_max > ch_min:
            ch_corrected = (
                (ch_corrected - ch_min) / (ch_max - ch_min)
                * ref_std + ref_mean
            )

    # Step 4: Soft compression if configured
    if params.get("channel_correction_soft_compress", True):
        ch_corrected = _reinhard_tone_map(ch_corrected)

    return ch_corrected


def detect_channel_degradation(img_path: str,
                               params: dict) -> dict:
    """Detect single-channel degradation from image.

    Uses config-driven thresholds.
    Handles 16-bit I;16 images with full precision.
    """
    try:
        from PIL import Image
        img = Image.open(img_path)
        if img.mode == "I;16":
            arr = np.array(img).astype(np.float32) / 65535.0
        else:
            arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0

        means = [arr[:, :, c].mean() for c in range(3)]
        stds = [arr[:, :, c].std() for c in range(3)]

        mean_threshold = params.get(
            "channel_degradation_mean_threshold", 0.08)
        std_threshold = params.get(
            "channel_degradation_std_threshold", 0.15)

        max_mean_diff = max(
            abs(means[i] - means[j])
            for i in range(3) for j in range(i + 1, 3))
        max_std_diff = max(
            abs(stds[i] - stds[j])
            for i in range(3) for j in range(i + 1, 3))

        degraded_channels = []
        for c in range(3):
            if abs(means[c] - sorted(means)[1]) > mean_threshold:
                degraded_channels.append(c)
            elif abs(stds[c] - sorted(stds)[1]) > std_threshold:
                degraded_channels.append(c)

        return {
            "degraded": max_mean_diff > mean_threshold
            or max_std_diff > std_threshold,
            "channels": degraded_channels,
            "mean_diff": max_mean_diff,
            "std_diff": max_std_diff,
            "means": means,
            "stds": stds,
        }
    except Exception:
        return {"degraded": False, "channels": []}


def run_stage2_ai_pre(input_path: str, output_path: str,
                      params: dict, logger=None) -> int:
    """Run single-channel correction per v7.32 §3.10.6 / v6.32 §7.16.

    Returns:
        0 = success, 3 = skip, 82 = IO error, 86 = validation failure
    """
    # Step 1: Detect channel degradation
    degradation = detect_channel_degradation(input_path, params)
    if not degradation.get("degraded"):
        if logger:
            logger.info("STAGE2_CHANNEL_SKIP",
                        "No channel degradation detected, skipping correction")
        return 3

    degraded_channels = degradation.get("channels", [])
    if not degraded_channels:
        if logger:
            logger.info("STAGE2_CHANNEL_SKIP",
                        "No degraded channels identified")
        return 3

    if logger:
        logger.info("STAGE2_CHANNEL_DEG",
                    f"Channel degradation detected: "
                    f"channels={degraded_channels}, "
                    f"mean_diff={degradation['mean_diff']:.4f}, "
                    f"std_diff={degradation['std_diff']:.4f}")

    # Step 2: Load image and normalize to 0-1
    try:
        from PIL import Image
        img = Image.open(input_path)
        if img.mode == "I;16":
            arr = np.array(img).astype(np.float32) / 65535.0
            max_val = 65535.0
            dtype = np.uint16
        else:
            arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            max_val = 255.0
            dtype = np.uint8
    except Exception as e:
        if logger:
            logger.error("STAGE2_LOAD_FAIL", f"Cannot load image: {e}")
        return 82

    # Step 3: IEC 61966-2-1 sRGB → linear conversion
    formula = params.get("channel_correction_formula", "iec61966-2-1")
    if formula == "iec61966-2-1":
        linear = _sRGB_to_linear(arr)
    else:
        linear = np.power(np.clip(arr, 0.0, 1.0), 2.2)

    # Step 4: Per-channel correction
    means_linear = [linear[:, :, c].mean() for c in range(3)]
    stds_linear = [linear[:, :, c].std() for c in range(3)]

    # Use median channel as reference
    ref_idx = np.argsort(means_linear)[1]
    ref_mean = means_linear[ref_idx]
    ref_std = stds_linear[ref_idx]
    ref_ch = linear[:, :, ref_idx]

    corrected = linear.copy()
    for c in degraded_channels:
        if c == ref_idx:
            continue
        if logger:
            logger.info("STAGE2_CHANNEL_CORRECT",
                        f"Correcting channel {c} against reference {ref_idx}")
        corrected[:, :, c] = correct_channel(
            linear[:, :, c], params, ref_ch, ref_mean, ref_std)

    # Step 5: Clamp
    corrected = np.clip(corrected, 0.0, 1.0)

    # Step 6: IEC 61966-2-1 linear → sRGB conversion
    if formula == "iec61966-2-1":
        srgb = _linear_to_sRGB(corrected)
    else:
        srgb = np.power(np.clip(corrected, 0.0, 1.0), 1.0 / 2.2)
    srgb = np.clip(srgb, 0.0, 1.0)

    # Step 7: Round (not truncate) and denormalize per v6.32 §7.16
    if params.get("channel_correction_round", True):
        out_arr = np.rint(srgb * max_val).astype(dtype)
    else:
        out_arr = (srgb * max_val).astype(dtype)

    # Step 8: Clamp to valid integer range
    if params.get("channel_correction_clamp", True):
        if max_val == 65535.0:
            out_arr = np.clip(out_arr, 0, 65535).astype(dtype)
        else:
            out_arr = np.clip(out_arr, 0, 255).astype(dtype)

    # Step 9: Save
    try:
        out_img = Image.fromarray(out_arr)
        out_img.save(output_path)
    except Exception as e:
        if logger:
            logger.error("STAGE2_SAVE_FAIL", f"Cannot save output: {e}")
        return 82

    if logger:
        logger.info("STAGE2_DONE",
                    f"Channel correction applied: "
                    f"degraded_channels={degraded_channels}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--params")
    parser.add_argument("--params-file")
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

    ai_enabled = params.get("ai_enabled", False)
    if not ai_enabled:
        logger.info("STAGE2_SKIP", "AI not enabled, skipping AI pre-processing")
        return 3

    temp_dir = params.get("temp_dir", "./temp")
    os.makedirs(temp_dir, exist_ok=True)
    job_id = args.job_id
    input_name = Path(args.input).name

    # Find stage1 output - sort by mtime for deterministic selection
    pattern = f"{job_id}_{Path(input_name).stem}_step1_preprocess*"
    candidates = glob.glob(os.path.join(temp_dir, pattern))
    if not candidates:
        logger.error("STAGE2_NO_INPUT", "Stage 1 output not found")
        return 82

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    stage1_output = candidates[0]
    logger.info("STAGE2_START", "AI pre-processing (channel correction)",
                input=stage1_output)

    # Check for existing intermediate
    param_hash = compute_param_hash(stage1_output, params)
    inter_name = build_intermediate_name(
        job_id, input_name, 2, "ai_pre", param_hash, "tif"
    )
    inter_path = os.path.join(temp_dir, inter_name)

    if os.path.exists(inter_path) and os.path.getsize(inter_path) > 0:
        logger.info("STAGE2_SKIP", "Intermediate exists", path=inter_path)
        return 3

    # Run channel correction
    rc = run_stage2_ai_pre(stage1_output, inter_path, params, logger)
    return rc


if __name__ == "__main__":
    # FIX-P11-5: main() must return explicit int exit code per v7.32 §3.5
    rc = main()
    if rc is None:
        rc = 86
    sys.exit(rc)
