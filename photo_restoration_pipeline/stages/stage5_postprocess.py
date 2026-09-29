#!/usr/bin/env python3
"""
Stage 5: Photo Post-processing
- Lanczos resize to target if non-integer (only when AI output needs it)
- Selective local processing with region separation
- Pure Python Laplacian pyramid grain synthesis (v7.32 §3.10.4)
- Half-body portrait color matching (v7.32 §3.10.7)
- 16-bit space operations for texture enhancement
- FLIF Cb/Cr layer extraction (placeholder for archival pipeline)

# FIX-P10-1: Renamed from stage4_postprocess.py to stage5_postprocess.py per v7.32 §3.1
# Stage order is now: stage1→stage2_ai_pre→stage3_ai_upscale→stage4_ai_face→stage5_postprocess→stage6_encode

v6.3/v7.3 Compliance:
- Separates flat areas from edges/textures for selective processing
- Operates in 16-bit space when photo_bit_depth=16
- Lanczos resize only applies when target is explicitly set (not "auto")
- Grain synthesis via Laplacian pyramid (all AI processing must complete first)
  # FIX-R13-1: Replaced non-existent G'MIC -film_grain with pure Python
  # Laplacian pyramid implementation. Algorithm matches DaVinci Resolve
  # Film Grain / Adobe Camera Raw Grain core approach.
"""

import os
import sys
import json
import argparse
import glob
import subprocess
import shutil
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
        text_mask = ndimage.binary_dilation(text_mask, iterations=2).astype(
            np.float32)
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
    edge_mask = np.clip(
        (gradient - edge_threshold * 0.5) / edge_threshold, 0.0, 1.0)

    # Texture mask: high variance but not edge
    texture_mask = np.clip(
        texture_strength * 2.0, 0.0, 1.0) * (1.0 - edge_mask)

    # Text detection (v6.3 §4.2.4: protect text regions)
    text_mask = np.zeros_like(gray, dtype=np.float32)
    if detect_text:
        text_mask = _detect_text_regions(gray)
        # Subtract text from edge/texture to avoid double-counting
        edge_mask = np.clip(edge_mask - text_mask, 0.0, 1.0)
        texture_mask = np.clip(texture_mask - text_mask, 0.0, 1.0)

    # Flat mask: low gradient AND low variance AND not text
    flat_mask = np.clip(
        1.0 - edge_mask - texture_mask - text_mask, 0.0, 1.0)

    return (flat_mask.astype(np.float32), edge_mask.astype(np.float32),
            texture_mask.astype(np.float32), text_mask.astype(np.float32))


def _apply_selective_processing(img: "Image.Image", params: dict,
                                 logger=None) -> "Image.Image":
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
        raw_arr = np.array(orig, dtype=np.float32) / 65535.0
        # I;16 images are 2D (H, W) — expand to 3-channel (H, W, 3)
        # so downstream region separation and per-channel processing work
        if raw_arr.ndim == 2:
            raw_arr = np.stack([raw_arr, raw_arr, raw_arr], axis=2)
        orig_arr = raw_arr
        max_val = 1.0
    else:
        if is_16bit_input:
            # Fallback to 8-bit if bit_depth != 16
            # FIX-P12-4: do NOT convert to L first, that destroys color
            orig = img.point(lambda p: p / 256).convert("RGB")
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
        logger.info("STAGE5_TEXT_DETECT",
                    f"Text regions detected: {text_ratio*100:.2f}% of image")

    # Flat area processing: mild gaussian blur
    from scipy.ndimage import gaussian_filter
    flat_smooth = np.zeros_like(arr_norm)
    for c in range(3):
        flat_smooth[:, :, c] = gaussian_filter(arr_norm[:, :, c], sigma=1.0)

    # Edge processing: mild unsharp mask (enhance edges)
    edge_enhanced = np.zeros_like(arr_norm)
    for c in range(3):
        blurred = gaussian_filter(arr_norm[:, :, c], sigma=2.0)
        edge_enhanced[:, :, c] = (
            arr_norm[:, :, c] + 0.3 * (arr_norm[:, :, c] - blurred))

    # Texture processing: subtle enhancement (slight contrast boost)
    texture_enhanced = np.zeros_like(arr_norm)
    for c in range(3):
        mean_val = arr_norm[:, :, c].mean()
        texture_enhanced[:, :, c] = (
            (arr_norm[:, :, c] - mean_val) * 1.05 + mean_val)

    # Text regions: fully preserve original (no modification)
    text_preserved = arr_norm.copy()

    # Combine regions
    combined = (
        flat_mask[:, :, np.newaxis] * flat_smooth +
        edge_mask[:, :, np.newaxis] * edge_enhanced +
        texture_mask[:, :, np.newaxis] * texture_enhanced +
        text_mask[:, :, np.newaxis] * text_preserved +
        (1.0 - flat_mask[:, :, np.newaxis]
         - edge_mask[:, :, np.newaxis]
         - texture_mask[:, :, np.newaxis]
         - text_mask[:, :, np.newaxis]) * arr_norm)

    combined = np.clip(combined, 0.0, 1.0)

    # Convert back to appropriate bit depth for PIL
    if bit_depth == 16:
        # FIX-P12-5/P12-6: I;16 only supports 2D arrays in PIL.
        # For 3-channel 16bit, return numpy array for caller to save as TIFF.
        result_arr = (combined * 65535.0).astype(np.uint16)
        return result_arr, 16  # Return (array, bit_depth) tuple
    else:
        result_arr = (combined * 255.0).astype(np.uint8)
        return Image.fromarray(result_arr), 8


def _chin_y_from_landmarks_photo(landmarks, face_box) -> int:
    """Estimate chin y from 5-point landmarks (photo pipeline, RGB float32).

    FIX-H-12-06: Per v7.32 §3.10.7, use landmark keypoints (not the face-box
    lower edge) to locate the neck ROI upper boundary. Returns -1 when
    landmarks are unavailable (caller falls back to geometric face-box edge).
    """
    if not isinstance(landmarks, dict):
        return -1
    try:
        ys = []
        for k in ("mouth_left", "mouth_right"):
            pt = landmarks.get(k)
            if pt and len(pt) >= 2:
                ys.append(float(pt[1]))
        if not ys:
            return -1
        mouth_y = max(ys)
        fh = float(face_box[3]) if face_box and len(face_box) > 3 else 0.0
        margin = max(4.0, fh * 0.12)
        return int(mouth_y + margin)
    except Exception:
        return -1


def _hand_skin_roi_photo(img_arr: np.ndarray, face_box: tuple) -> np.ndarray:
    """Sample hand skin-tone pixels below the face (photo pipeline, RGB float32).

    FIX-H-12-06: Per v7.32 §3.10.7, color matching should reference hand ROI in
    addition to neck. Hands are not directly detected (no hand model); sample
    skin-tone pixels in the region below the face box. img_arr is RGB float32
    in [0,255]. Returns N×3 float32 array (possibly empty).
    """
    try:
        if img_arr is None or img_arr.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        h, w = img_arr.shape[:2]
        fx, fy, fw, fh = (int(face_box[0]), int(face_box[1]),
                          int(face_box[2]), int(face_box[3]))
        x1 = max(0, fx - fw // 2)
        x2 = min(w, fx + fw + fw // 2)
        y1 = min(h, fy + fh + fh // 4)
        y2 = min(h, y1 + fh * 2)
        if y2 <= y1 or x2 <= x1:
            return np.empty((0, 3), dtype=np.float32)
        region = img_arr[y1:y2, x1:x2]
        if region.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        r = region[:, :, 0]
        g = region[:, :, 1]
        b = region[:, :, 2]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        skin_mask = (r > 95) & (g > 40) & (b > 20) & (mx - mn > 15) & (np.abs(r - g) > 15)
        return region[skin_mask]
    except Exception:
        return np.empty((0, 3), dtype=np.float32)


def _match_portrait_color(image_path: str, face_boxes: list,
                          output_path: str, logger=None,
                          landmarks: list = None) -> bool:
    """Automated half-body portrait color matching per v7.32 §3.10.7.

    FIX-P12-6: Migrated from stage4_ai_face.py to stage5_postprocess.py.
    Color matching must execute after all AI processing (face restore + upscale)
    and before grain synthesis, per spec order.

    FIX-H-12-06: When stage4 persisted face landmarks (RetinaFace/MTCNN), use
    the chin point to locate the neck ROI upper boundary (instead of the
    face-box lower edge) and fold hand skin-tone pixels into the reference so
    the restored face matches body skin, not just neck. Falls back to geometric
    ROI when landmarks are None (YOLOv8-face).
    """
    try:
        import numpy as np
        from PIL import Image, ImageFilter
        import cv2

        img = Image.open(image_path).convert("RGB")
        img_arr = np.array(img, dtype=np.float32)
        h_img, w_img = img_arr.shape[:2]

        # Build a combined mask for all face regions
        face_mask = np.zeros((h_img, w_img), dtype=np.float32)
        corrected = img_arr.copy()

        for box_idx, (fx, fy, fw, fh) in enumerate(face_boxes):
            # Clamp box to image bounds
            x1 = max(0, fx)
            y1 = max(0, fy)
            x2 = min(w_img, fx + fw)
            y2 = min(h_img, fy + fh)
            if x2 <= x1 or y2 <= y1:
                continue

            # Estimate neck region: directly below face,
            # same width, 0.4x face height.
            # FIX-H-12-06: when landmarks available, start neck ROI at the chin
            # (mouth_y + margin) instead of the face-box lower edge - the box
            # edge often cuts into the jaw, biasing the reference.
            _lm = (landmarks[box_idx]
                   if landmarks and box_idx < len(landmarks) else None)
            _chin = _chin_y_from_landmarks_photo(_lm, (fx, fy, fw, fh))
            ny1 = max(y2, _chin) if _chin > 0 else y2
            ny2 = min(h_img, ny1 + int(fh * 0.4))
            nx1 = x1
            nx2 = x2
            if ny2 <= ny1:
                continue

            face_roi = img_arr[y1:y2, x1:x2]
            neck_roi = img_arr[ny1:ny2, nx1:nx2]
            if neck_roi.size == 0:
                continue

            # FIX-H-12-06: sample hand skin-tone below the face and fold into
            # the neck reference so the match target reflects body skin.
            hand_skin = _hand_skin_roi_photo(img_arr, (fx, fy, fw, fh))

            # Convert to LAB using OpenCV
            # FIX-M12-11: Do LAB conversion on float32 (not uint8) to avoid
            # truncation per round_12 M-12-11. cv2 float32 LAB uses ranges
            # L∈[0,100], a/b∈[-128,127] (uint8 LAB offsets a/b to [0,255]).
            # ORIGINAL: astype(np.uint8) truncated before LAB + clamped all
            # channels to [0,255] (wrong for float a/b which can be negative).
            face_bgr = cv2.cvtColor(
                face_roi.astype(np.float32), cv2.COLOR_RGB2BGR)
            neck_bgr = cv2.cvtColor(
                neck_roi.astype(np.float32), cv2.COLOR_RGB2BGR)
            face_lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
            neck_lab = cv2.cvtColor(neck_bgr, cv2.COLOR_BGR2LAB)

            # FIX-H-12-06: combine neck + hand skin pixels for reference stats
            # so the restored face matches body skin, not just neck. Convert
            # hand skin (RGB float32) -> BGR -> LAB and concatenate.
            if hand_skin.size > 0:
                try:
                    hand_bgr = cv2.cvtColor(
                        hand_skin.reshape(-1, 1, 3), cv2.COLOR_RGB2BGR)
                    hand_lab = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2LAB)
                    ref_pix = np.concatenate(
                        [neck_lab.reshape(-1, 3), hand_lab.reshape(-1, 3)],
                        axis=0)
                except Exception:
                    ref_pix = neck_lab.reshape(-1, 3)
            else:
                ref_pix = neck_lab.reshape(-1, 3)

            # Mean/std matching per channel (range-agnostic)
            matched_lab = face_lab.copy()
            for c in range(3):
                f_mean = face_lab[:, :, c].mean()
                f_std = face_lab[:, :, c].std() + 1e-6
                n_mean = float(ref_pix[:, c].mean())
                n_std = float(ref_pix[:, c].std()) + 1e-6
                matched_lab[:, :, c] = (
                    (matched_lab[:, :, c] - f_mean) * (n_std / f_std)
                    + n_mean)

            # Clamp LAB valid ranges (float32 convention)
            matched_lab[:, :, 0] = np.clip(matched_lab[:, :, 0], 0, 100)
            matched_lab[:, :, 1] = np.clip(matched_lab[:, :, 1], -128, 127)
            matched_lab[:, :, 2] = np.clip(matched_lab[:, :, 2], -128, 127)

            # Convert back to RGB (float32, no uint8 cast)
            matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
            matched_rgb = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB)

            corrected[y1:y2, x1:x2] = matched_rgb
            face_mask[y1:y2, x1:x2] = 1.0

        # If no valid face was processed, skip
        if face_mask.max() < 0.5:
            # FIX-P12-10: Use copyfile (not copy2) per v7.32 §3.10.10
            shutil.copyfile(image_path, output_path)
            return True

        # Feather the mask edge
        mask_img = Image.fromarray((face_mask * 255).astype(np.uint8))
        mask_blurred = mask_img.filter(ImageFilter.GaussianBlur(radius=8))
        mask_smooth = np.array(mask_blurred, dtype=np.float32) / 255.0
        mask_smooth = np.clip(mask_smooth, 0.0, 1.0)
        mask_smooth = np.stack([mask_smooth] * 3, axis=2)

        blended = mask_smooth * corrected + (1.0 - mask_smooth) * img_arr
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        Image.fromarray(blended).save(output_path)
        return True
    except Exception as e:
        if logger:
            logger.warning("STAGE5_COLOR_MATCH_FAIL",
                          f"Portrait color matching failed: {e}")
        # FIX-H12-04: Scheme C half-automatic fallback per v7.32 §3.10.7.
        # Output the face-restored image as _face_restored.tif so the user
        # can manually Match Color in GIMP/PS and drop it back. The main
        # flow continues using the original (uncolor-matched) image.
        # ORIGINAL: silently copied the original to output_path with no
        # handoff artifact, giving the user no way to finish color matching.
        try:
            fallback_tif = output_path.rsplit(".", 1)[0] + "_face_restored.tif"
            shutil.copyfile(image_path, fallback_tif)
            if logger:
                logger.warning("STAGE5_FACE_RESTORED_TIF",
                               f"半自动兜底：已输出 {fallback_tif}，请在 "
                               f"GIMP/PS 手工 Match Color 后放回继续流程 "
                               f"(v7.32 §3.10.7 方案C)")
            # FIX-P12-10: Use copyfile (not copy2) per v7.32 §3.10.10
            shutil.copyfile(image_path, output_path)
            return True
        except Exception:
            return False


def _add_grain_python(input_path: str, output_path: str,
                       strength: float, size: float,
                       roughness: float, bit_depth: int = 16) -> bool:
    """Pure Python film grain synthesis via Laplacian pyramid.

    FIX-R13-1: Replaces non-existent G'MIC -film_grain command.
    Algorithm matches DaVinci Resolve Film Grain / Adobe Camera Raw Grain:
      1. Laplacian pyramid decomposition (multi-scale frequency bands)
      2. Inject per-level Gaussian noise with sigma proportional to strength
      3. Roughness controls per-level weight (fine vs coarse grain ratio)
      4. Reconstruct pyramid = grain-synthesized image

    Grain is applied primarily to luminance (Y channel) per film grain
    physics — chrominance grain is kept at 15% of luminance intensity,
    matching real film behavior where UV grain is 10-20% of Y grain.
    In RGB space this is approximated by applying a shared base noise
    (luminance) plus small per-channel variation (chrominance).

    Args:
        input_path: Input image path (PNG or TIFF)
        output_path: Output image path
        strength: Overall grain intensity [0, 1], default 0.3
        size: Grain spatial scale [0.5, 2.0], default 0.5
        roughness: Fine/coarse grain ratio [0, 1], default 0
            0 = pure fine grain (natural film look)
            0.5 = balanced
            1 = mostly coarse grain (DV/tape noise look)
        bit_depth: Output bit depth (8 or 16)

    Returns:
        True if successful, False otherwise
    """
    try:
        from PIL import Image
        from scipy.ndimage import gaussian_filter, zoom

        if strength <= 0:
            # No grain requested — just copy
            shutil.copyfile(input_path, output_path)
            return True

        # Load image — prefer tifffile for 16-bit TIFF to avoid
        # PIL I;16 3-channel data loss (PIL I;16 only supports 2D arrays)
        arr = None
        input_depth = 8
        try:
            import tifffile
            loaded = tifffile.imread(input_path)
            if loaded.dtype == np.uint16:
                arr = loaded.astype(np.float32) / 65535.0
                input_depth = 16
            elif loaded.dtype == np.uint8:
                arr = loaded.astype(np.float32) / 255.0
                input_depth = 8
            else:
                # Other dtypes — normalize to [0,1]
                arr = loaded.astype(np.float32)
                arr = arr / (arr.max() + 1e-8)
                input_depth = 16 if loaded.dtype.itemsize >= 2 else 8
            # Ensure 3-channel
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=2)
        except (ImportError, Exception):
            # Fallback to PIL
            img = Image.open(input_path)
            if img.mode == "I;16":
                arr = np.array(img, dtype=np.float32) / 65535.0
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=2)
                input_depth = 16
            else:
                arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
                input_depth = 8

        h, w = arr.shape[:2]
        if h < 8 or w < 8:
            # Image too small for pyramid decomposition
            shutil.copyfile(input_path, output_path)
            return True

        # --- Laplacian Pyramid Decomposition ---
        # Number of levels based on grain_size and image dimensions
        max_levels = int(np.log2(min(h, w)) - 2)
        max_levels = max(3, min(max_levels, 8))
        n_levels = max(3, int(max_levels * min(size, 2.0)))

        # Build Gaussian pyramid
        gauss_pyramid = [arr.copy()]
        for _ in range(n_levels):
            prev = gauss_pyramid[-1]
            # Blur and downsample by 2x
            blurred = np.zeros_like(prev)
            for c in range(3):
                blurred[:, :, c] = gaussian_filter(
                    prev[:, :, c], sigma=2.0)
            # Downsample: take every other pixel
            gauss_pyramid.append(blurred[::2, ::2, :])

        # Build Laplacian pyramid (difference between adjacent levels)
        lap_pyramid = []
        for i in range(n_levels):
            small = gauss_pyramid[i + 1]
            # Upsample small to current level size
            up_h, up_w = gauss_pyramid[i].shape[0], gauss_pyramid[i].shape[1]
            up = zoom(small, (up_h / small.shape[0],
                              up_w / small.shape[1], 1.0),
                      order=1)  # bilinear
            # Ensure exact size match after zoom — pad if too small,
            # truncate if too large (handles odd-dimension rounding)
            if up.shape[0] < up_h or up.shape[1] < up_w:
                # Pad with edge values to reach target size
                padded = np.zeros((up_h, up_w, 3), dtype=np.float32)
                padded[:up.shape[0], :up.shape[1], :] = up
                # Replicate edge for padded rows/cols
                if up.shape[0] < up_h:
                    padded[up.shape[0]:, :up.shape[1], :] = up[-1:, :, :]
                if up.shape[1] < up_w:
                    padded[:up.shape[0], up.shape[1]:, :] = up[:, -1:, :]
                if up.shape[0] < up_h and up.shape[1] < up_w:
                    padded[up.shape[0]:, up.shape[1]:, :] = up[-1, -1, :]
                up = padded
            elif up.shape[0] > up_h or up.shape[1] > up_w:
                up = up[:up_h, :up_w, :]
            lap = gauss_pyramid[i] - up
            lap_pyramid.append(lap)

        # --- Inject Grain at Each Level ---
        # Level 0 = finest (high freq), Level n-1 = coarsest (low freq)
        for i in range(n_levels):
            level_h, level_w = lap_pyramid[i].shape[:2]

            # Per-level sigma: finer levels get relatively more noise
            # roughness shifts weight from fine to coarse levels
            level_ratio = i / max(n_levels - 1, 1)

            # Fine grain weight (decreases with level)
            fine_weight = (1.0 - roughness) * (1.0 - level_ratio * 0.6)
            # Coarse grain weight (increases with level)
            coarse_weight = roughness * (0.3 + level_ratio * 0.7)
            # Combined per-level sigma
            level_sigma = strength * 0.05 * (fine_weight + coarse_weight)

            if level_sigma <= 0:
                continue

            # Generate Gaussian noise with luminance-weighted application
            # Film grain is primarily in luminance (Y), with chroma at 15%
            # Since we operate in RGB space, we approximate by applying
            # a shared base noise (luminance component) plus small
            # per-channel variation (chrominance component).
            # This produces correlated grain similar to real film where
            # R/G/B grain is ~85% shared (luminance) and ~15% independent.
            noise_luma = np.random.normal(
                0, level_sigma, (level_h, level_w)).astype(np.float32)
            # Per-channel chrominance variation (15% of luma)
            noise_ch_r = np.random.normal(
                0, level_sigma * 0.15, (level_h, level_w)).astype(np.float32)
            noise_ch_g = np.random.normal(
                0, level_sigma * 0.15, (level_h, level_w)).astype(np.float32)
            noise_ch_b = np.random.normal(
                0, level_sigma * 0.15, (level_h, level_w)).astype(np.float32)

            # Combine: base luminance noise + per-channel chroma variation
            noise = np.stack([noise_luma + noise_ch_r,
                              noise_luma + noise_ch_g,
                              noise_luma + noise_ch_b], axis=2)
            lap_pyramid[i] = lap_pyramid[i] + noise

        # --- Reconstruct Pyramid ---
        result = gauss_pyramid[-1].copy()  # Start from coarsest
        for i in range(n_levels - 1, -1, -1):
            target_h, target_w = gauss_pyramid[i].shape[0], gauss_pyramid[i].shape[1]
            # Upsample result to current level size
            up = zoom(result, (target_h / result.shape[0],
                                target_w / result.shape[1], 1.0),
                      order=1)
            # Handle size mismatch (same padding/truncation logic as decomposition)
            if up.shape[0] < target_h or up.shape[1] < target_w:
                padded = np.zeros((target_h, target_w, 3), dtype=np.float32)
                padded[:up.shape[0], :up.shape[1], :] = up
                if up.shape[0] < target_h:
                    padded[up.shape[0]:, :up.shape[1], :] = up[-1:, :, :]
                if up.shape[1] < target_w:
                    padded[:up.shape[0], up.shape[1]:, :] = up[:, -1:, :]
                if up.shape[0] < target_h and up.shape[1] < target_w:
                    padded[up.shape[0]:, up.shape[1]:, :] = up[-1, -1, :]
                up = padded
            elif up.shape[0] > target_h or up.shape[1] > target_w:
                up = up[:target_h, :target_w, :]
            result = up + lap_pyramid[i]

        # Clip and save
        result = np.clip(result, 0.0, 1.0)

        out_depth = bit_depth if bit_depth in (8, 16) else input_depth
        if out_depth == 16:
            out_arr = (result * 65535.0).astype(np.uint16)
            try:
                import tifffile
                tifffile.imwrite(output_path, out_arr, compression="zlib")
            except ImportError:
                # Fallback: save as 8-bit (significant quality loss)
                import warnings
                warnings.warn(
                    "tifffile not available; saving 16-bit grain output "
                    "as 8-bit. Install tifffile for full 16-bit support.")
                out_8 = (result * 255.0).astype(np.uint8)
                Image.fromarray(out_8).save(output_path)
        else:
            out_arr = (result * 255.0).astype(np.uint8)
            Image.fromarray(out_arr).save(output_path)

        return True
    except Exception:
        return False


def _add_grain(input_path: str, output_path: str,
               strength: float, size: float,
               roughness: float, bit_depth: int = 16) -> bool:
    """Add film grain via pure Python Laplacian pyramid.

    FIX-A1: Removed dead G'MIC -film_grain fallback (G'MIC stdlib has no
    -film_grain command, verified gmic_stdlib.gmic 3.7.6). Pure Python
    Laplacian pyramid is the sole implementation per v7.32 §3.10.4.
    """
    return _add_grain_python(input_path, output_path, strength, size,
                             roughness, bit_depth)


def _extract_flif_chrominance(img: "Image.Image", output_dir: str,
                               job_id: str, stem: str,
                               logger=None) -> bool:
    """Extract Cb/Cr chrominance layers and save as separate files.

    Attempts FLIF encoding first (v6.3 §4.2.4 requirement).
    Falls back to lossless PNG if flif CLI is unavailable.
    """
    try:
        import shutil as _shutil
        # Convert to YCbCr
        ycbcr = img.convert("YCbCr")
        y, cb, cr = ycbcr.split()

        # Try FLIF encoding first
        flif_available = _shutil.which("flif") is not None

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
                rc_cb = subprocess.run(
                    ["flif", "-e", cb_tmp, cb_path],
                    capture_output=True, timeout=60, shell=False)
                rc_cr = subprocess.run(
                    ["flif", "-e", cr_tmp, cr_path],
                    capture_output=True, timeout=60, shell=False)
                rc_y = subprocess.run(
                    ["flif", "-e", y_tmp, y_path],
                    capture_output=True, timeout=60, shell=False)

                # ISSUE-055 fix: verify FLIF returncodes and output existence
                if (rc_cb.returncode == 0 and rc_cr.returncode == 0
                        and rc_y.returncode == 0
                        and os.path.exists(cb_path)
                        and os.path.exists(cr_path)
                        and os.path.exists(y_path)):
                    # Clean up temp PNGs if FLIF succeeded
                    for p in (cb_tmp, cr_tmp, y_tmp):
                        if os.path.exists(p):
                            os.remove(p)
                    if logger:
                        logger.info("STAGE5_FLIF",
                                    f"FLIF chrominance layers saved: "
                                    f"{cb_path}, {cr_path}")
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
            logger.warning("STAGE5_FLIF_FALLBACK",
                          "FLIF not available, saved Cb/Cr as lossless PNG fallback")
        return True
    except Exception as e:
        if logger:
            logger.warning("STAGE5_FLIF_FAIL",
                          f"Cb/Cr extraction failed: {e}")
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
    # FIX-P12-3: WSL2 redirect for temp_dir per v7.32 §3.10.10
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    platform = params.get("platform", "auto")
    try:
        from checkpoint import wsl_redirect_path
        redirected_temp, _, _, _ = wsl_redirect_path(
            temp_dir, no_wsl_redirect=no_wsl_redirect,
            target_platform=platform
        )
        if redirected_temp != temp_dir:
            logger.info("STAGE5_WSL_REDIRECT",
                       f"WSL temp_dir redirect: {temp_dir} -> {redirected_temp}")
            temp_dir = redirected_temp
            params["temp_dir"] = temp_dir
    except Exception:
        pass
    job_id = args.job_id
    input_name = Path(args.input).name

    # FIX-P10-1: Input is now stage4_ai_face output (or stage3_upscale if no face)
    # Sort by mtime for deterministic selection; prefer face-restored output
    input_source = None
    for pat in ["step4_face*", "step3_upscale*", "step2_ai_pre*",
                "step1_preprocess*"]:
        pattern = f"{job_id}_{Path(input_name).stem}_{pat}"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        if candidates:
            candidates.sort(
                key=lambda p: os.path.getmtime(p), reverse=True)
            input_source = candidates[0]
            break

    if not input_source:
        logger.info("STAGE5_SKIP", "No input from prior stages")
        return 3

    logger.info("STAGE5_START", "Photo post-processing", input=input_source)

    # Output path - use TIFF for 16-bit output
    # FIX-P10-1: step number is now 5
    bit_depth = params.get("photo_bit_depth", 16)
    if bit_depth == 16:
        output_path = os.path.join(temp_dir,
            f"{job_id}_{Path(input_name).stem}_step5_postprocess.tif")
    else:
        output_path = os.path.join(temp_dir,
            f"{job_id}_{Path(input_name).stem}_step5_postprocess.png")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return 3

    try:
        from PIL import Image

        img = Image.open(input_source)

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
                    img_16bit = img_16bit.resize(
                        (target_w, target_h), Image.LANCZOS)
                    logger.info("STAGE5_LANCZOS",
                                f"Resized {img_16bit.width}x{img_16bit.height} "
                                f"-> {target_w}x{target_h}")
            except Exception as e:
                logger.warning("STAGE5_LANCZOS_WARN",
                              f"Lanczos resize skipped: {e}")

        # Selective local processing with region separation
        # FIX-R13-2: Track image data as numpy array for 16-bit path
        # to avoid PIL I;16 3-channel data loss (PIL I;16 only supports 2D)
        current_arr = None  # numpy float32 [0,1] when 16-bit path active
        current_img = img_16bit  # PIL Image for 8-bit path
        local_process = params.get("local_mask_denoise", True)
        if local_process and img_16bit.mode in ("RGB", "RGBA", "L", "I;16"):
            result = _apply_selective_processing(img_16bit, params, logger)
            # FIX-P12-6: Handle tuple return (array, bit_depth) for 16-bit
            if isinstance(result, tuple):
                arr_result, result_depth = result
                if result_depth == 16:
                    # Keep as numpy array — do NOT go through PIL for 16-bit
                    # PIL I;16 mode only supports 2D arrays; 3-channel uint16
                    # TIFF round-trips through PIL lose color data.
                    current_arr = arr_result.astype(np.float32) / 65535.0
                    current_img = None  # Signal: use array path
                else:
                    current_img = arr_result  # PIL Image for 8-bit
            else:
                current_img = result  # Legacy PIL Image return
            logger.info("STAGE5_MASK",
                        "Applied selective local processing with region separation")

        # FLIF chrominance extraction (for archival pipeline)
        archival_mode = params.get("archival_mode", False)
        if archival_mode:
            flif_img = current_img if current_img is not None else None
            if flif_img is None and current_arr is not None:
                # Convert array to PIL Image for FLIF extraction
                flif_arr = (current_arr * 255.0).clip(0, 255).astype(
                    np.uint8)
                flif_img = Image.fromarray(flif_arr)
            if flif_img is not None:
                flif_ok = _extract_flif_chrominance(
                    flif_img, temp_dir, job_id, Path(input_name).stem,
                    logger=logger
                )
            else:
                flif_ok = False
            if flif_ok:
                logger.info("STAGE5_FLIF",
                            "Extracted Cb/Cr chrominance layers")
            else:
                logger.warning("STAGE5_FLIF", "Cb/Cr extraction skipped")

        # FIX-P12-6: Half-body portrait color matching per v7.32 §3.10.7
        # Must run after face restore (stage4) and before grain synthesis
        enable_color_match = params.get("portrait_color_match", True)
        face_boxes = []
        face_landmarks = []  # FIX-H-12-06: parallel to face_boxes
        # Read face_boxes from stage4 metadata file
        face_meta_path = os.path.join(
            temp_dir, f"{job_id}_{Path(input_name).stem}_face_boxes.json")
        if os.path.exists(face_meta_path):
            try:
                with open(face_meta_path, "r", encoding="utf-8") as f:
                    face_meta = json.load(f)
                face_boxes = face_meta.get("face_boxes", [])
                # FIX-H-12-06: read landmarks written by stage4 for chin-based
                # neck ROI + hand skin-tone reference per v7.32 §3.10.7.
                face_landmarks = face_meta.get("face_landmarks", [])
            except Exception:
                face_boxes = []
                face_landmarks = []
        else:
            face_landmarks = []

        if enable_color_match and face_boxes:
            # Save current state before color matching
            color_match_temp = output_path + ".pre_color_match"
            if current_arr is not None:
                # Save 16-bit array as TIFF for color matching
                arr_u16 = (current_arr * 65535.0).clip(0, 65535).astype(
                    np.uint16)
                try:
                    import tifffile
                    tifffile.imwrite(color_match_temp, arr_u16,
                                    compression="zlib")
                except ImportError:
                    # Fallback: save as 8-bit for color matching
                    arr_8 = (current_arr * 255.0).clip(0, 255).astype(
                        np.uint8)
                    Image.fromarray(arr_8).save(color_match_temp)
            else:
                current_img.save(color_match_temp)
            if _match_portrait_color(color_match_temp, face_boxes,
                                     output_path + ".color_matched", logger,
                                     landmarks=face_landmarks):
                # Reload the color-matched image
                # FIX-R13-2: For 16-bit path, load directly as numpy array
                # to avoid PIL I;16 3-channel data loss
                try:
                    if current_arr is not None:
                        # Try tifffile first for 16-bit
                        try:
                            import tifffile
                            loaded = tifffile.imread(
                                output_path + ".color_matched")
                            if loaded.dtype == np.uint16:
                                current_arr = loaded.astype(
                                    np.float32) / 65535.0
                            else:
                                current_arr = loaded.astype(
                                    np.float32) / 255.0
                        except (ImportError, Exception):
                            # Fallback to PIL (may lose 16-bit precision)
                            cm_img = Image.open(
                                output_path + ".color_matched")
                            cm_arr = np.array(cm_img.convert("RGB"),
                                              dtype=np.float32) / 255.0
                            current_arr = cm_arr
                    else:
                        current_img = Image.open(
                            output_path + ".color_matched")
                    logger.info("STAGE5_COLOR_MATCH",
                                "Portrait color matching applied per v7.32 §3.10.7")
                except Exception:
                    logger.warning("STAGE5_COLOR_MATCH_RELOAD",
                                  "Failed to reload color-matched image")
                # Clean up temp files
                for tmp in [color_match_temp,
                            output_path + ".color_matched"]:
                    if os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
            else:
                logger.warning("STAGE5_COLOR_MATCH_SKIP",
                              "Color matching failed, continuing without it")
                color_match_temp = output_path + ".pre_color_match"
                if os.path.exists(color_match_temp):
                    try:
                        os.remove(color_match_temp)
                    except Exception:
                        pass

        # Save before grain
        # FIX-P12-4: Ensure grain is applied in 16bit space per v7.32 §3.10.4
        # FIX-R13-2: For 16-bit path, save directly from numpy array via
        # tifffile to avoid PIL I;16 3-channel data loss. If tifffile is
        # unavailable, warn and save as 8-bit (quality loss).
        grain_temp = output_path + ".pre_grain.tif"  # TIFF for 16bit
        if current_arr is not None:
            # 16-bit array path — save via tifffile (preferred)
            arr_u16 = (current_arr * 65535.0).clip(0, 65535).astype(
                np.uint16)
            try:
                import tifffile
                tifffile.imwrite(grain_temp, arr_u16, compression="zlib")
            except ImportError:
                # No tifffile: must save as 8-bit (quality loss)
                logger.warning("STAGE5_NO_TIFFFILE",
                              "tifffile not available; saving 16-bit "
                              "pre-grain image as 8-bit. Install "
                              "tifffile for full 16-bit support.")
                arr_8 = (current_arr * 255.0).clip(0, 255).astype(
                    np.uint8)
                Image.fromarray(arr_8).save(grain_temp)
        elif bit_depth == 16:
            # PIL Image path (non-I;16, e.g. RGB mode)
            if current_img is not None and current_img.mode == "I;16":
                # Single-channel I;16 — PIL can save as TIFF directly
                current_img.save(grain_temp, format="TIFF",
                                compression="tiff_lzw")
            elif current_img is not None:
                # Convert 8-bit PIL image to 16-bit for grain in 16bit space
                arr = np.array(current_img.convert("RGB"),
                              dtype=np.float32) / 255.0
                arr_16 = (arr * 65535.0).clip(0, 65535).astype(np.uint16)
                try:
                    import tifffile
                    tifffile.imwrite(grain_temp, arr_16, compression="zlib")
                except ImportError:
                    logger.warning("STAGE5_NO_TIFFFILE",
                                  "tifffile not available; saving 16-bit "
                                  "pre-grain image as 8-bit. Install "
                                  "tifffile for full 16-bit support.")
                    # Fallback: save as 8-bit (quality loss)
                    r = Image.fromarray(arr_16[:, :, 0], mode='I;16')
                    g = Image.fromarray(arr_16[:, :, 1], mode='I;16')
                    b = Image.fromarray(arr_16[:, :, 2], mode='I;16')
                    rgb_8 = Image.merge("RGB", (
                        r.point(lambda p: p // 256),
                        g.point(lambda p: p // 256),
                        b.point(lambda p: p // 256),
                    ))
                    rgb_8.save(grain_temp)
        else:
            # 8-bit path — save PIL Image directly
            if current_img is not None:
                current_img.save(grain_temp)
            elif current_arr is not None:
                # Should not happen in 8-bit path, but handle gracefully
                arr_8 = (current_arr * 255.0).clip(0, 255).astype(
                    np.uint8)
                Image.fromarray(arr_8).save(grain_temp)

        # FIX-P10-1 / FIX-P10-4: Grain synthesis in stage5
        # (v7.32 §3.1: grain after all AI + post-processing)
        # FIX-P10-4: This stage always runs regardless of ai_enabled
        # FIX-R13-1: Replaced G'MIC -film_grain with pure Python Laplacian
        # pyramid implementation (G'MIC -film_grain does not exist)
        grain_enabled = params.get("grain_synthesis", True)
        if grain_enabled:
            grain_strength = params.get("grain_strength", 0.3)
            grain_size = params.get("grain_size", 0.5)  # FIX-R13-1: 0.5 per v7.32 §3.10.4
            grain_roughness = params.get("grain_roughness", 0)  # FIX-P12-12: default 0 per v7.32 §3.10.4
            logger.info("STAGE5_GRAIN",
                        "Adding film grain via Laplacian pyramid (post-AI, post-postprocess)")
            if _add_grain(grain_temp, output_path,
                          grain_strength, grain_size, grain_roughness,
                          bit_depth=bit_depth):
                logger.info("STAGE5_GRAIN_DONE",
                            "Film grain applied in post-processing stage")
                try:
                    os.remove(grain_temp)
                except Exception:
                    pass
            else:
                logger.warning("STAGE5_GRAIN_FAIL",
                              "Grain synthesis failed, using ungrained output")
                shutil.move(grain_temp, output_path)
        else:
            shutil.move(grain_temp, output_path)

        logger.info("STAGE5_DONE", "Post-processing complete",
                    output=output_path)
        return 0

    except Exception as e:
        logger.error("STAGE5_ERROR", f"Post-process failed: {e}")
        # Fallback: direct copy
        if input_source:
            # FIX-P12-11: Use copyfile (not copy2) per v7.32 §3.10.10
            shutil.copyfile(input_source, output_path)
            logger.info("STAGE5_DONE",
                        "Post-processing complete (fallback copy)",
                        output=output_path)
        return 0


if __name__ == "__main__":
    # FIX-P11-5: main() must return explicit int exit code per v7.32 §3.5
    # ORIGINAL: sys.exit(main() or 0) — "or 0" masks None returns
    rc = main()
    if rc is None:
        rc = 86  # validation failure — main() must always return explicit exit code
    sys.exit(rc)
