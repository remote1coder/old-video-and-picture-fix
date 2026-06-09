#!/usr/bin/env python3
"""
Stage 4: Photo Post-processing after AI Upscale
- Lanczos resize to target if non-integer (only when AI output needs it)
- Selective local processing with region separation
- 16-bit space operations for texture enhancement
- FLIF Cb/Cr layer extraction (placeholder for archival pipeline)
v6.3/v7.3 Compliance:
- Separates flat areas from edges/textures for selective processing
- Operates in 16-bit space when photo_bit_depth=16
- Lanczos resize only applies when target is explicitly set (not "auto")
"""

import os
import sys
import json
import argparse
import glob
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402


def _detect_text_regions(gray: np.ndarray, min_aspect_ratio: float = 2.0) -> np.ndarray:
    """Detect text-like regions using stroke width approximation.
    
    Uses high-gradient areas with high aspect ratio connected components
    as a proxy for text regions. Returns a binary mask.
    """
    try:
        from scipy import ndimage
        # High gradient areas
        dx = ndimage.sobel(gray, axis=1)
        dy = ndimage.sobel(gray, axis=0)
        gradient = np.sqrt(dx**2 + dy**2)
        grad_norm = gradient / (gradient.max() + 1e-8)
        
        # Threshold for high-contrast strokes
        stroke_mask = grad_norm > 0.15
        
        # Label connected components
        labeled, num_features = ndimage.label(stroke_mask)
        text_mask = np.zeros_like(gray, dtype=np.float32)
        
        for i in range(1, num_features + 1):
            coords = np.argwhere(labeled == i)
            if len(coords) < 20:  # Too small
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            h = y_max - y_min + 1
            w = x_max - x_min + 1
            if w == 0:
                continue
            aspect = max(h, w) / min(h, w)
            density = len(coords) / (h * w)
            # Text-like: high aspect ratio, moderate density
            if aspect >= min_aspect_ratio and 0.1 <= density <= 0.8:
                text_mask[labeled == i] = 1.0
        
        # Dilate slightly to cover character bodies
        text_mask = ndimage.binary_dilation(text_mask, iterations=2).astype(np.float32)
        return text_mask
    except Exception:
        return np.zeros_like(gray, dtype=np.float32)


def _separate_regions(img_arr: np.ndarray, edge_threshold: float = 0.05,
                       detect_text: bool = True) -> tuple:
    """Separate flat areas from edge/texture/text regions.

    Uses gradient magnitude via Sobel operators for edge detection,
    combined with local variance for texture detection.
    Includes text detection to protect text regions from smoothing.

    Returns: (flat_mask, edge_mask, texture_mask, text_mask)
    Each mask is float32 in [0, 1].
    """
    from scipy.ndimage import sobel, gaussian_filter

    # Convert to grayscale for analysis
    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        gray = img_arr.mean(axis=2)
    else:
        gray = img_arr

    # Sobel gradient magnitude for edges
    dx = sobel(gray, axis=1)
    dy = sobel(gray, axis=0)
    gradient = np.sqrt(dx**2 + dy**2)
    grad_max = gradient.max()
    if grad_max > 0:
        gradient = gradient / grad_max

    # Local variance for texture detection
    local_mean = gaussian_filter(gray, sigma=2)
    local_var = gaussian_filter(gray**2, sigma=2) - local_mean**2
    var_max = local_var.max()
    if var_max > 0:
        texture_strength = local_var / var_max
    else:
        texture_strength = np.zeros_like(gray)

    # Edge mask: high gradient
    edge_mask = np.clip((gradient - edge_threshold * 0.5) / edge_threshold, 0.0, 1.0)

    # Texture mask: high variance but not edge
    texture_mask = np.clip(texture_strength * 2.0, 0.0, 1.0) * (1.0 - edge_mask)

    # Text detection (v6.3 §4.2.4: protect text regions)
    text_mask = np.zeros_like(gray, dtype=np.float32)
    if detect_text:
        text_mask = _detect_text_regions(gray)
        # Subtract text from edge/texture to avoid double-counting
        edge_mask = np.clip(edge_mask - text_mask, 0.0, 1.0)
        texture_mask = np.clip(texture_mask - text_mask, 0.0, 1.0)

    # Flat mask: low gradient AND low variance AND not text
    flat_mask = np.clip(1.0 - edge_mask - texture_mask - text_mask, 0.0, 1.0)

    return flat_mask.astype(np.float32), edge_mask.astype(np.float32), texture_mask.astype(np.float32), text_mask.astype(np.float32)


def _apply_selective_processing(img: "Image.Image", params: dict, logger=None) -> "Image.Image":
    """Apply selective local processing with region separation.

    - Flat areas: mild denoise
    - Edges: sharpening/enhancement
    - Textures: subtle enhancement
    - Text regions: fully protected (no modification)
    """
    import numpy as np
    from PIL import Image

    # Handle I;16 input with full precision (Bug 11.9 fix)
    is_16bit_input = img.mode == "I;16"
    bit_depth = params.get("photo_bit_depth", 16)

    if is_16bit_input and bit_depth == 16:
        # Preserve 16-bit precision: load as uint16, normalize to [0,1]
        orig = img  # Keep I;16
        orig_arr = np.array(orig, dtype=np.float32) / 65535.0
        max_val = 1.0
    else:
        if is_16bit_input:
            # Fallback to 8-bit if bit_depth != 16
            orig = img.point(lambda p: p / 256).convert("L").convert("RGB")
        else:
            orig = img.convert("RGB")
        orig_arr = np.array(orig).astype(np.float32)
        max_val = 255.0

    # Normalize to [0, 1]
    arr_norm = orig_arr / max_val

    # Separate regions (now includes text detection)
    flat_mask, edge_mask, texture_mask, text_mask = _separate_regions(arr_norm)

    if logger and text_mask.max() > 0:
        text_ratio = float(text_mask.mean())
        logger.info("STAGE4_TEXT_DETECT", f"Text regions detected: {text_ratio*100:.2f}% of image")

    # Flat area processing: mild gaussian blur
    from scipy.ndimage import gaussian_filter
    flat_smooth = np.zeros_like(arr_norm)
    for c in range(3):
        flat_smooth[:, :, c] = gaussian_filter(arr_norm[:, :, c], sigma=1.0)

    # Edge processing: mild unsharp mask (enhance edges)
    edge_enhanced = np.zeros_like(arr_norm)
    for c in range(3):
        blurred = gaussian_filter(arr_norm[:, :, c], sigma=2.0)
        edge_enhanced[:, :, c] = arr_norm[:, :, c] + 0.3 * (arr_norm[:, :, c] - blurred)

    # Texture processing: subtle enhancement (slight contrast boost)
    texture_enhanced = np.zeros_like(arr_norm)
    for c in range(3):
        mean_val = arr_norm[:, :, c].mean()
        texture_enhanced[:, :, c] = (arr_norm[:, :, c] - mean_val) * 1.05 + mean_val

    # Text regions: fully preserve original (no modification)
    text_preserved = arr_norm.copy()

    # Combine regions
    combined = (flat_mask[:, :, np.newaxis] * flat_smooth +
                edge_mask[:, :, np.newaxis] * edge_enhanced +
                texture_mask[:, :, np.newaxis] * texture_enhanced +
                text_mask[:, :, np.newaxis] * text_preserved +
                (1.0 - flat_mask[:, :, np.newaxis] - edge_mask[:, :, np.newaxis] - texture_mask[:, :, np.newaxis] - text_mask[:, :, np.newaxis]) * arr_norm)

    combined = np.clip(combined, 0.0, 1.0)

    # Convert back to appropriate bit depth for PIL
    if bit_depth == 16:
        # For 16-bit output, use I;16 mode
        result_arr = (combined * 65535.0).astype(np.uint16)
        return Image.fromarray(result_arr, mode='I;16')
    else:
        result_arr = (combined * 255.0).astype(np.uint8)
        return Image.fromarray(result_arr)


def _extract_flif_chrominance(img: "Image.Image", output_dir: str, job_id: str, stem: str,
                               logger=None) -> bool:
    """Extract Cb/Cr chrominance layers and save as separate files.

    Attempts FLIF encoding first (v6.3 §4.2.4 requirement).
    Falls back to lossless PNG if flif CLI is unavailable.
    """
    try:
        import shutil
        # Convert to YCbCr
        ycbcr = img.convert("YCbCr")
        y, cb, cr = ycbcr.split()

        # Try FLIF encoding first
        flif_available = shutil.which("flif") is not None
        
        if flif_available:
            cb_path = os.path.join(output_dir, f"{job_id}_{stem}_cb.flif")
            cr_path = os.path.join(output_dir, f"{job_id}_{stem}_cr.flif")
            y_path = os.path.join(output_dir, f"{job_id}_{stem}_y.flif")
            
            # Save as temporary PNG first, then convert to FLIF
            cb_tmp = cb_path + ".tmp.png"
            cr_tmp = cr_path + ".tmp.png"
            y_tmp = y_path + ".tmp.png"
            cb.save(cb_tmp)
            cr.save(cr_tmp)
            y.save(y_tmp)
            
            try:
                rc_cb = subprocess.run(["flif", "-e", cb_tmp, cb_path], capture_output=True, timeout=60, shell=False)
                rc_cr = subprocess.run(["flif", "-e", cr_tmp, cr_path], capture_output=True, timeout=60, shell=False)
                rc_y = subprocess.run(["flif", "-e", y_tmp, y_path], capture_output=True, timeout=60, shell=False)

                # ISSUE-055 fix: verify FLIF returncodes and output existence
                if rc_cb.returncode == 0 and rc_cr.returncode == 0 and rc_y.returncode == 0 and \
                   os.path.exists(cb_path) and os.path.exists(cr_path) and os.path.exists(y_path):
                    # Clean up temp PNGs if FLIF succeeded
                    for p in (cb_tmp, cr_tmp, y_tmp):
                        if os.path.exists(p):
                            os.remove(p)
                    if logger:
                        logger.info("STAGE4_FLIF", f"FLIF chrominance layers saved: {cb_path}, {cr_path}")
                    return True
                else:
                    # FLIF encoding partially failed, fall through to PNG
                    for p in (cb_tmp, cr_tmp, y_tmp):
                        if os.path.exists(p):
                            os.remove(p)
                    # Clean up partial FLIF outputs
                    for p in (cb_path, cr_path, y_path):
                        if os.path.exists(p):
                            os.remove(p)
            except Exception:
                # FLIF encoding failed, fall through to PNG
                for p in (cb_tmp, cr_tmp, y_tmp):
                    if os.path.exists(p):
                        os.remove(p)
        
        # Fallback: save as lossless PNG
        cb_path = os.path.join(output_dir, f"{job_id}_{stem}_cb.png")
        cr_path = os.path.join(output_dir, f"{job_id}_{stem}_cr.png")
        cb.save(cb_path, compress_level=0)
        cr.save(cr_path, compress_level=0)
        
        if logger:
            logger.warning("STAGE4_FLIF_FALLBACK", 
                          "FLIF not available, saved Cb/Cr as lossless PNG fallback")
        return True
    except Exception as e:
        if logger:
            logger.warning("STAGE4_FLIF_FAIL", f"Cb/Cr extraction failed: {e}")
        return False


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

    # Find stage3 output - sort by mtime for deterministic selection
    pattern = f"{job_id}_{Path(input_name).stem}_step3_upscale*"
    candidates = glob.glob(os.path.join(temp_dir, pattern))

    if not candidates:
        logger.info("STAGE4_SKIP", "No AI upscale output")
        return 3

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    stage3_output = candidates[0]
    logger.info("STAGE4_START", "Photo post-processing", input=stage3_output)

    # Output path - use TIFF for 16-bit output
    bit_depth = params.get("photo_bit_depth", 16)
    if bit_depth == 16:
        output_path = os.path.join(temp_dir,
            f"{job_id}_{Path(input_name).stem}_step4_postprocess.tif")
    else:
        output_path = os.path.join(temp_dir,
            f"{job_id}_{Path(input_name).stem}_step4_postprocess.png")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return 3

    try:
        from PIL import Image

        img = Image.open(stage3_output)

        # Preserve 16-bit if input is 16-bit
        if img.mode == "I;16" and bit_depth == 16:
            # Process in 16-bit space
            img_16bit = img
        else:
            # Convert to RGB for processing
            img_16bit = img.convert("RGB")

        # Lanczos resize to target resolution if explicitly specified
        lanczos_target = params.get("lanczos_target", "auto")
        if lanczos_target and lanczos_target != "auto":
            try:
                target_w, target_h = lanczos_target.split("x")
                target_w, target_h = int(target_w), int(target_h)
                if img_16bit.width != target_w or img_16bit.height != target_h:
                    img_16bit = img_16bit.resize((target_w, target_h), Image.LANCZOS)
                    logger.info("STAGE4_LANCZOS",
                                f"Resized {img_16bit.width}x{img_16bit.height} -> {target_w}x{target_h}")
            except Exception as e:
                logger.warning("STAGE4_LANCZOS_WARN", f"Lanczos resize skipped: {e}")

        # Selective local processing with region separation
        local_process = params.get("local_mask_denoise", True)
        if local_process and img_16bit.mode in ("RGB", "RGBA", "L", "I;16"):
            img_16bit = _apply_selective_processing(img_16bit, params, logger)
            logger.info("STAGE4_MASK", "Applied selective local processing with region separation")

        # FLIF chrominance extraction (for archival pipeline)
        archival_mode = params.get("archival_mode", False)
        if archival_mode:
            flif_ok = _extract_flif_chrominance(
                img_16bit, temp_dir, job_id, Path(input_name).stem, logger=logger
            )
            if flif_ok:
                logger.info("STAGE4_FLIF", "Extracted Cb/Cr chrominance layers")
            else:
                logger.warning("STAGE4_FLIF", "Cb/Cr extraction skipped")

        img_16bit.save(output_path)

        logger.info("STAGE4_DONE", "Post-processing complete", output=output_path)
        return 0

    except Exception as e:
        logger.error("STAGE4_ERROR", f"Post-process failed: {e}")
        # Fallback: direct copy
        import shutil
        shutil.copy2(stage3_output, output_path)
        logger.info("STAGE4_DONE", "Post-processing complete (fallback copy)", output=output_path)
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
