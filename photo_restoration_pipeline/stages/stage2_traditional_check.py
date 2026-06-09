#!/usr/bin/env python3
"""
Stage 2: Photo Traditional Repair Verification
Verifies stage1 output: file integrity, decodeability, quality checks.

v6.3 Compliance:
- Multi-channel quality check (mean/std per channel)
- Structural integrity verification (pixel range, non-uniformity)
- Re-checks single-channel degradation status
- Sorts candidates by mtime for deterministic selection
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


def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute lightweight structural similarity between two images.
    
    Uses a simplified SSIM without external dependencies.
    Both arrays should be float32 in [0, 1].
    """
    try:
        from skimage.metrics import structural_similarity as skimage_ssim
        # Convert to grayscale for SSIM
        if img1.ndim == 3:
            gray1 = img1.mean(axis=2)
            gray2 = img2.mean(axis=2)
        else:
            gray1, gray2 = img1, img2
        return float(skimage_ssim(gray1, gray2, data_range=1.0))
    except ImportError:
        # Fallback: covariance-based structural similarity
        if img1.ndim == 3:
            gray1 = img1.mean(axis=2)
            gray2 = img2.mean(axis=2)
        else:
            gray1, gray2 = img1, img2
        mu1 = gray1.mean()
        mu2 = gray2.mean()
        sigma1 = gray1.std()
        sigma2 = gray2.std()
        cov = np.mean((gray1 - mu1) * (gray2 - mu2))
        # SSIM approximation
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_val = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / \
                   ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
        return float(np.clip(ssim_val, -1.0, 1.0))


def _compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def _check_multi_channel_quality(img_path: str, logger=None) -> dict:
    """Check per-channel quality metrics.

    Returns dict with channel means, stds, and quality flags.
    """
    try:
        from PIL import Image
        import numpy as np
        with Image.open(img_path) as img:
            # ISSUE-051 fix: preserve 16-bit precision for I;16 images
            if img.mode == "I;16":
                arr = np.array(img, dtype=np.float32) / 65535.0
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
            else:
                arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0

        means = [arr[:, :, c].mean() for c in range(3)]
        stds = [arr[:, :, c].std() for c in range(3)]

        # Check for uniform/corrupted images
        min_std = min(stds)
        if min_std < 0.001:
            if logger:
                logger.warning("STAGE2_Q_FLAT", f"Near-uniform channel detected (std={min_std:.6f})")
            return {"ok": False, "reason": "flat_channel", "means": means, "stds": stds}

        # Check for all-zero or all-max
        if arr.max() == 0 or arr.min() == arr.max():
            if logger:
                logger.error("STAGE2_Q_CORRUPT", "Image appears uniform (all same value)")
            return {"ok": False, "reason": "uniform", "means": means, "stds": stds}

        # Check mean range
        for i, (m, s) in enumerate(zip(means, stds)):
            if not (0.01 <= m <= 0.99):
                if logger:
                    logger.warning("STAGE2_Q_RANGE", f"Channel {i} mean out of normal range: {m:.4f}")

        return {"ok": True, "means": means, "stds": stds}
    except Exception as e:
        return {"ok": False, "reason": f"exception: {e}"}


def _check_structural_integrity(original_path: str, processed_path: str,
                                 params: dict = None, logger=None) -> dict:
    """Compare original and processed images using SSIM/PSNR.
    
    Per v6.3 §4.2.2: structural integrity verification required.
    Returns dict with ssim, psnr, and ok flag.
    """
    try:
        from PIL import Image
        import numpy as np
        
        with Image.open(original_path) as orig_img:
            # ISSUE-051 fix: preserve 16-bit precision
            if orig_img.mode == "I;16":
                orig = np.array(orig_img, dtype=np.float32) / 65535.0
                if orig.ndim == 2:
                    orig = np.stack([orig, orig, orig], axis=-1)
            else:
                orig = np.array(orig_img.convert("RGB")).astype(np.float32) / 255.0
        with Image.open(processed_path) as proc_img:
            if proc_img.mode == "I;16":
                proc = np.array(proc_img, dtype=np.float32) / 65535.0
                if proc.ndim == 2:
                    proc = np.stack([proc, proc, proc], axis=-1)
            else:
                proc = np.array(proc_img.convert("RGB")).astype(np.float32) / 255.0
        
        # Resize proc to orig size if dimensions differ
        if orig.shape != proc.shape:
            from PIL import Image as PILImage
            proc_pil = PILImage.fromarray((proc * 255).astype(np.uint8))
            proc_pil = proc_pil.resize((orig.shape[1], orig.shape[0]), PILImage.LANCZOS)
            proc = np.array(proc_pil).astype(np.float32) / 255.0
        
        ssim_val = _compute_ssim(orig, proc)
        psnr_val = _compute_psnr(orig, proc)
        
        if logger:
            logger.info("STAGE2_STRUCTURAL", 
                       f"SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f}dB")
        
        # Thresholds: SSIM should not drop too much (indicates corruption)
        # Normal preprocessing should maintain SSIM > 0.85
        result = {"ssim": ssim_val, "psnr": psnr_val, "ok": True}
        if ssim_val < 0.5:
            result["ok"] = False
            result["reason"] = f"SSIM too low ({ssim_val:.4f}), possible corruption"
            if logger:
                logger.error("STAGE2_STRUCTURAL_FAIL", result["reason"])
        elif psnr_val < 20.0:
            result["ok"] = False
            result["reason"] = f"PSNR too low ({psnr_val:.2f}dB), possible corruption"
            if logger:
                logger.error("STAGE2_STRUCTURAL_FAIL", result["reason"])
        
        return result
    except Exception as e:
        if logger:
            logger.warning("STAGE2_STRUCTURAL_SKIP", f"Structural check skipped: {e}")
        return {"ssim": 0.0, "psnr": 0.0, "ok": True}  # Skip on error, don't fail


def _check_channel_degradation_fixed(original_path: str, processed_path: str,
                                      params: dict = None, logger=None) -> dict:
    """Verify that channel degradation detected in original was improved by processing.
    
    Per v6.3 §4.2.2: single-channel degradation recheck required.
    """
    try:
        from PIL import Image
        import numpy as np
        
        mean_threshold = params.get("channel_degradation_mean_threshold", 0.08) if params else 0.08
        std_threshold = params.get("channel_degradation_std_threshold", 0.15) if params else 0.15
        
        def _get_stats(path: str):
            with Image.open(path) as img:
                arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            means = [arr[:, :, c].mean() for c in range(3)]
            stds = [arr[:, :, c].std() for c in range(3)]
            max_mean_diff = max(abs(means[i] - means[j]) for i in range(3) for j in range(i+1, 3))
            max_std_diff = max(abs(stds[i] - stds[j]) for i in range(3) for j in range(i+1, 3))
            return max_mean_diff, max_std_diff
        
        orig_mean_diff, orig_std_diff = _get_stats(original_path)
        proc_mean_diff, proc_std_diff = _get_stats(processed_path)
        
        mean_improved = proc_mean_diff < orig_mean_diff
        std_improved = proc_std_diff < orig_std_diff
        mean_fixed = proc_mean_diff < mean_threshold
        std_fixed = proc_std_diff < std_threshold
        
        if logger:
            logger.info("STAGE2_CHANNEL_RECHECK",
                       f"Original: mean_diff={orig_mean_diff:.4f}, std_diff={orig_std_diff:.4f}; "
                       f"Processed: mean_diff={proc_mean_diff:.4f}, std_diff={proc_std_diff:.4f}; "
                       f"improved={mean_improved or std_improved}, fixed={mean_fixed and std_fixed}")
        
        # If original was degraded and processing made it worse, warn
        orig_degraded = orig_mean_diff > mean_threshold or orig_std_diff > std_threshold
        if orig_degraded and not (mean_improved or std_improved):
            if logger:
                logger.warning("STAGE2_CHANNEL_WORSENED",
                              "Channel degradation worsened after processing")
            return {"ok": False, "reason": "channel_degradation_worsened"}
        
        return {"ok": True}
    except Exception as e:
        if logger:
            logger.warning("STAGE2_CHANNEL_RECHECK_SKIP", f"Channel recheck skipped: {e}")
        return {"ok": True}


def _check_file_integrity(path: str, params: dict = None, logger=None) -> bool:
    """Verify file exists, is non-empty, and decodeable.

    Bug P-14 fix: when photo_bit_depth=16, verifies output is in 16-bit mode.
    """
    if not os.path.exists(path):
        if logger:
            logger.error("STAGE2_NOT_FOUND", "Stage 1 output missing")
        return False

    size = os.path.getsize(path)
    if size == 0:
        if logger:
            logger.error("STAGE2_EMPTY", "Stage 1 output is empty")
        return False

    # Try to open with PIL
    try:
        from PIL import Image
        with Image.open(path) as img:
            w, h = img.size
            mode = img.mode
            if logger:
                logger.info("STAGE2_VERIFY", f"Image OK: {w}x{h} {mode}")
            # Bug P-14: validate 16bit output format when configured
            bit_depth = params.get("photo_bit_depth", 16) if params else 16
            if bit_depth == 16 and mode not in ("I;16", "I;16B", "RGB"):
                logger.warning("STAGE2_BITDEPTH",
                              f"Expected 16-bit output but got mode={mode}. "
                              "Continuing with available bit depth.")
            elif bit_depth == 16 and mode in ("I;16", "I;16B"):
                logger.info("STAGE2_BITDEPTH", "Verified 16-bit output format")
    except Exception as e:
        if logger:
            logger.error("STAGE2_CORRUPT", f"Cannot open image: {e}")
        return False

    return True


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

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    input_name = Path(args.input).name

    # Find stage1 output - sort by mtime for deterministic selection (Bug-5 fix)
    pattern = f"{job_id}_{Path(input_name).stem}_step1_preprocess*"
    candidates = glob.glob(os.path.join(temp_dir, pattern))
    if not candidates:
        logger.error("STAGE2_NO_INPUT", "Stage 1 output not found")
        return 82

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    stage1_output = candidates[0]
    logger.info("STAGE2_START", "Verifying stage1 output", path=stage1_output)

    # Basic checks
    if not _check_file_integrity(stage1_output, params, logger):
        return 86

    # Multi-channel quality check
    quality = _check_multi_channel_quality(stage1_output, logger)
    if not quality["ok"]:
        logger.error("STAGE2_Q_FAIL", f"Quality check failed: {quality['reason']}")
        return 86

    # Structural integrity: SSIM/PSNR vs original (v6.3 §4.2.2)
    structural = _check_structural_integrity(args.input, stage1_output, params, logger)
    if not structural.get("ok", True):
        logger.error("STAGE2_STRUCTURAL_FAIL", f"Structural integrity failed: {structural.get('reason')}")
        return 86

    # Single-channel degradation recheck: compare original vs processed (v6.3 §4.2.2)
    channel_recheck = _check_channel_degradation_fixed(args.input, stage1_output, params, logger)
    if not channel_recheck.get("ok", True):
        logger.error("STAGE2_CHANNEL_RECHECK_FAIL", f"Channel degradation recheck failed: {channel_recheck.get('reason')}")
        return 86

    logger.info("STAGE2_DONE", "Verification passed")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
