#!/usr/bin/env python3
"""
Stage 1: Photo Traditional Preprocessing
- Scratch removal (G'MIC)
- Deblocking
- Debanding
- Single-channel degradation detection
- Color space conversion if needed
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_utils"))

from logger import get_logger  # noqa: E402
from checkpoint import compute_param_hash, build_intermediate_name  # noqa: E402


def run_gmic(input_path: str, output_path: str, *gmic_args) -> bool:
    """Run G'MIC with shell=False safety."""
    cmd = ["gmic", input_path]
    cmd.extend(list(gmic_args))
    cmd.extend(["-o", output_path])
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300, shell=False)
        return r.returncode == 0
    except Exception:
        return False


def detect_channel_degradation(img_path: str) -> dict:
    """Detect single-channel degradation in photo."""
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0

        # Compute per-channel stats
        means = [arr[:, :, c].mean() for c in range(3)]
        stds = [arr[:, :, c].std() for c in range(3)]

        result = {
            "degraded": False,
            "channels": [],
            "mean_diff": 0.0,
            "std_diff": 0.0,
        }

        # Check mean differences
        max_mean_diff = max(abs(means[i] - means[j])
                           for i in range(3) for j in range(i+1, 3))
        max_std_diff = max(abs(stds[i] - stds[j])
                          for i in range(3) for j in range(i+1, 3))

        if max_mean_diff > 0.08 or max_std_diff > 0.15:
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


def apply_channel_correction(img_path: str, output_path: str,
                              params: dict, logger=None) -> bool:
    """Apply single-channel correction in linear RGB space.

    Only applied when degradation is severe and AI is NOT enabled.
    Uses IEC 61966-2-1 color space conversion.
    """
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(img_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0

        # Convert to linear space
        linear = _sRGB_to_linear(arr)

        # Compute per-channel stats in linear space
        means = [linear[:, :, c].mean() for c in range(3)]
        stds = [linear[:, :, c].std() for c in range(3)]

        # Identify degraded channel(s)
        mean_threshold = params.get("channel_degradation_mean_threshold_severe", 0.15)
        std_threshold = params.get("channel_degradation_std_threshold_severe", 0.25)

        max_mean_diff = max(abs(means[i] - means[j])
                           for i in range(3) for j in range(i+1, 3))
        max_std_diff = max(abs(stds[i] - stds[j])
                          for i in range(3) for j in range(i+1, 3))

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
        for c in range(3):
            ch = corrected[:, :, c]
            ch_mean = means[c]
            ch_std = stds[c]

            # Skip if close to reference
            if abs(ch_mean - target_mean) < 0.01 and abs(ch_std - target_std) < 0.01:
                continue

            # Linear stretch to match target mean/std
            if ch_std > 1e-6:
                ch = (ch - ch_mean) * (target_std / ch_std) + target_mean
            else:
                ch = ch - ch_mean + target_mean
            corrected[:, :, c] = ch

        # Clamp to valid range
        corrected = np.clip(corrected, 0.0, 1.0)

        # Optional soft compression
        if params.get("channel_correction_soft_compress", True):
            corrected = _reinhard_tone_map(corrected)
            corrected = np.clip(corrected, 0.0, 1.0)

        # Convert back to sRGB
        srgb = _linear_to_sRGB(corrected)
        srgb = np.clip(srgb, 0.0, 1.0)

        # Round (not truncate) and denormalize to 8/16bit
        # Use original bit depth if available, else 8bit
        bit_depth = params.get("photo_bit_depth", 8)
        if bit_depth == 16:
            max_val = 65535.0
        else:
            max_val = 255.0

        out_arr = np.rint(srgb * max_val).astype(np.uint16 if bit_depth == 16 else np.uint8)

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
    parser.add_argument("--config")
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()

    params = json.loads(args.params) if args.params else {}
    log_dir = params.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, job_id=args.job_id)

    temp_dir = params.get("temp_dir", "./temp")
    os.makedirs(temp_dir, exist_ok=True)

    input_path = args.input
    input_name = Path(input_path).name

    # Check for existing intermediate
    param_hash = compute_param_hash(input_path, params)
    inter_name = build_intermediate_name(
        args.job_id, input_name, 1, "preprocess", param_hash, "tif"
    )
    inter_path = os.path.join(temp_dir, inter_name)

    if os.path.exists(inter_path) and os.path.getsize(inter_path) > 0:
        logger.info("STAGE1_SKIP", "Intermediate exists", path=inter_path)
        return 3

    logger.info("STAGE1_START", "Photo preprocessing", input=input_path)

    # Convert to 16bit TIFF for processing
    working_path = os.path.join(temp_dir,
        f"{args.job_id}_{Path(input_name).stem}_16bit.tif")

    try:
        subprocess.run(
            ["gmic", input_path,
             "-type", "ushort",
             "-o", working_path],
            capture_output=True, timeout=120, shell=False
        )
    except Exception as e:
        logger.error("STAGE1_CONVERT_FAIL", str(e))
        return 82

    # Scratch removal
    if params.get("scratch_remove", True):
        logger.info("STAGE1_SCRATCH", "Removing scratches")
        scratch_out = working_path + ".scratch.tif"
        success = run_gmic(working_path, scratch_out,
                          "-repair", str(params.get("scratch_remove_strength", 20)))
        if success and os.path.exists(scratch_out):
            working_path = scratch_out

    # Deblock
    if params.get("deblock_strength", 0) > 0:
        logger.info("STAGE1_DEBLOCK", "Deblocking")
        deblock_out = working_path + ".deblock.tif"
        success = run_gmic(working_path, deblock_out,
                          "-deblock",
                          str(params.get("deblock_strength", 25)))
        if success and os.path.exists(deblock_out):
            working_path = deblock_out

    # Deband
    if params.get("deband_threshold", 0) > 0:
        logger.info("STAGE1_DEBAND", "Debanding")
        deband_out = working_path + ".deband.tif"
        success = run_gmic(working_path, deband_out,
                          "-deband",
                          str(params.get("deband_threshold", 1.2)))
        if success and os.path.exists(deband_out):
            working_path = deband_out

    # Channel degradation detection and correction
    degradation = detect_channel_degradation(input_path)
    if degradation["degraded"]:
        logger.info("STAGE1_CHANNEL_DEG",
                   f"Channel degradation detected: mean_diff={degradation['mean_diff']:.4f}")
        # Apply correction only when AI is NOT enabled (v6.3 requirement)
        ai_enabled = params.get("ai_enabled", False)
        if not ai_enabled:
            logger.info("STAGE1_CHANNEL_CORR", "Applying single-channel correction (no AI)")
            corr_out = working_path + ".channel.tif"
            if apply_channel_correction(working_path, corr_out, params, logger):
                if os.path.exists(corr_out):
                    working_path = corr_out

    # Save final output
    import shutil
    shutil.copy2(working_path, inter_path)

    if os.path.exists(inter_path):
        logger.info("STAGE1_DONE", "Preprocessing complete", output=inter_path)
        return 0
    else:
        logger.error("STAGE1_FAILED", "Output not created")
        return 82


if __name__ == "__main__":
    sys.exit(main() or 0)
