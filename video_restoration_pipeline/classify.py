#!/usr/bin/env python3
"""
Video Material Classifier
Classifies videos into: ANALOG_TAPE, DV_DIGITAL, HDV_AVCHD
Detects field order, PSF, IVTC requirements.
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple
import json

import numpy as np


def run_ffprobe(input_path: str) -> Dict[str, Any]:
    """Run ffprobe and return parsed stream/format info."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format:stream",
             "-of", "json", input_path],
            capture_output=True, text=True, timeout=60, shell=False
        )
        if r.returncode == 0:
            return json.loads(r.stdout)
    except Exception:
        pass
    return {}


def _parse_idet_counts(idet_output: str) -> Tuple[int, int, int, int]:
    """Parse ffmpeg idet stderr for TFF/BFF/Progressive/Undetermined counts."""
    tff_count = 0
    bff_count = 0
    progressive_count = 0
    undetermined_count = 0
    for line in idet_output.split("\n"):
        if "Multi frame detection:" in line:
            parts = line.split("TFF:")
            if len(parts) > 1:
                try:
                    tff_count = int(parts[1].split()[0])
                except ValueError:
                    pass
            parts = line.split("BFF:")
            if len(parts) > 1:
                try:
                    bff_count = int(parts[1].split()[0])
                except ValueError:
                    pass
            parts = line.split("Progressive:")
            if len(parts) > 1:
                try:
                    progressive_count = int(parts[1].split()[0])
                except ValueError:
                    pass
            parts = line.split("Undetermined:")
            if len(parts) > 1:
                try:
                    undetermined_count = int(parts[1].split()[0])
                except ValueError:
                    pass
    return tff_count, bff_count, progressive_count, undetermined_count


def detect_field_order_segments(input_path: str) -> Tuple[str, float, list]:
    """Detect field order with variable-field-order segmentation.

    Uses sliding-window idet analysis. Splits video into 2-second windows
    with 1-second step, runs idet on each window, and merges consecutive
    windows with the same dominant field order into segments.

    Returns: (field_order, confidence, segments)
    - field_order: "tt", "bb", "psf", "progressive", "unknown", or "mixed"
    - confidence: float 0-1
    - segments: list of dicts with {start_frame, end_frame, field_order}
      Empty list if no mixed field order detected.
    """
    # Get video info
    info = run_ffprobe(input_path)
    streams = info.get("streams", [])
    video_stream = None
    for s in streams:
        if s.get("codec_type") == "video":
            video_stream = s
            break

    fo_meta = video_stream.get("field_order", "unknown") if video_stream else "unknown"

    # Get duration and fps
    duration = 0.0
    fps = 30.0
    try:
        format_info = info.get("format", {})
        duration = float(format_info.get("duration", 0))
        if video_stream:
            avg_fps = video_stream.get("avg_frame_rate", "30/1")
            num, den = avg_fps.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 30.0
    except Exception:
        pass

    # Run global idet first
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "info", "-i", input_path,
             "-vf", "idet", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120, shell=False
        )
        global_idet = r.stderr
    except Exception:
        global_idet = ""

    gtff, gbff, gprog, gund = _parse_idet_counts(global_idet)
    gtotal = gtff + gbff + gprog + gund
    if gtotal == 0:
        return "unknown", 0.0, []

    gprog_ratio = gprog / gtotal

    # PSF check
    if gprog_ratio > 0.90 and fo_meta in ("tt", "bb", "tff", "bff"):
        if _check_psf_ssim(input_path):
            return "psf", gprog_ratio, []

    # If too short or no video stream, skip segmentation
    if duration < 3.0 or not video_stream:
        if gprog_ratio > 0.95:
            return "progressive", gprog_ratio, []
        elif gtff / gtotal > 0.8:
            return "tt", gtff / gtotal, []
        elif gbff / gtotal > 0.8:
            return "bb", gbff / gtotal, []
        elif gprog_ratio > 0.5:
            return "progressive", gprog_ratio, []
        else:
            return "unknown", 0.3, []

    # Sliding window analysis: 2-second windows, 1-second step
    window_duration = 2.0  # seconds
    step = 1.0             # seconds
    windows = []

    num_windows = max(1, int((duration - window_duration) / step) + 1)
    for i in range(num_windows):
        start = i * step
        if start + window_duration > duration:
            break
        try:
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-v", "info",
                 "-ss", str(start), "-t", str(window_duration),
                 "-i", input_path,
                 "-vf", "idet", "-f", "null", "-"],
                capture_output=True, text=True, timeout=60, shell=False
            )
            wtff, wbff, wprog, wund = _parse_idet_counts(r.stderr)
            wtotal = wtff + wbff + wprog + wund
            if wtotal == 0:
                continue
            # Determine dominant field order for this window
            ratios = {
                "progressive": wprog / wtotal,
                "tt": wtff / wtotal,
                "bb": wbff / wtotal,
            }
            dom_order = max(ratios, key=ratios.get)
            dom_conf = ratios[dom_order]
            # Only accept if confidence > 0.5
            if dom_conf > 0.5:
                windows.append({
                    "start": start,
                    "end": start + window_duration,
                    "field_order": dom_order,
                    "confidence": dom_conf,
                })
        except Exception:
            continue

    if not windows:
        # Fallback to global result
        if gprog_ratio > 0.95:
            return "progressive", gprog_ratio, []
        elif gtff / gtotal > 0.8:
            return "tt", gtff / gtotal, []
        elif gbff / gtotal > 0.8:
            return "bb", gbff / gtotal, []
        elif gprog_ratio > 0.5:
            return "progressive", gprog_ratio, []
        else:
            return "unknown", 0.3, []

    # Merge consecutive windows with same field order
    segments = []
    current = {
        "start": windows[0]["start"],
        "end": windows[0]["end"],
        "field_order": windows[0]["field_order"],
    }
    for w in windows[1:]:
        if w["field_order"] == current["field_order"]:
            current["end"] = w["end"]
        else:
            segments.append(current.copy())
            current = {
                "start": w["start"],
                "end": w["end"],
                "field_order": w["field_order"],
            }
    segments.append(current)

    # Convert time-based segments to frame-based
    frame_segments = []
    for seg in segments:
        start_frame = int(seg["start"] * fps)
        end_frame = int(seg["end"] * fps)
        if end_frame <= start_frame:
            end_frame = start_frame + int(window_duration * fps)
        frame_segments.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "field_order": seg["field_order"],
        })

    # Check if mixed (has both tt and bb segments)
    has_tt = any(s["field_order"] == "tt" for s in frame_segments)
    has_bb = any(s["field_order"] == "bb" for s in frame_segments)

    if has_tt and has_bb:
        # Mixed field order detected
        # Overall confidence = average of segment confidences (approximate)
        # v6.3: field_order valid values are tt/bb/psf/progressive/unknown ONLY.
        # "mixed" is not a valid field_order; use "unknown" with field_segments.
        avg_conf = sum(w["confidence"] for w in windows) / len(windows)
        return "unknown", avg_conf, frame_segments

    # Not mixed, return global result
    if gprog_ratio > 0.95:
        return "progressive", gprog_ratio, []
    elif gtff / gtotal > 0.8:
        return "tt", gtff / gtotal, []
    elif gbff / gtotal > 0.8:
        return "bb", gbff / gtotal, []
    elif gprog_ratio > 0.5:
        return "progressive", gprog_ratio, []
    else:
        return "unknown", 0.3, []


def detect_field_order(input_path: str) -> Tuple[str, float]:
    """Detect field order using ffprobe + idet.

    Returns: (field_order, confidence)
    Valid values: tt, bb, psf, progressive, unknown
    """
    # Delegate to segment-aware detector, discard segments for simple callers
    field_order, confidence, _ = detect_field_order_segments(input_path)
    return field_order, confidence


def _detect_black_borders_frame(arr: np.ndarray) -> Tuple[int, int, int, int]:
    """Detect black borders in a single grayscale frame.

    Returns (crop_top, crop_bottom, crop_left, crop_right) in pixels.
    Threshold: brightness < 16 and variance < 0.1 for consecutive rows/cols.
    """
    # numpy imported at module level
    h, w = arr.shape
    crop_top = 0
    crop_bottom = 0
    crop_left = 0
    crop_right = 0

    # Top border
    for y in range(h):
        row = arr[y, :]
        if row.mean() < 16 and row.var() < 0.1:
            crop_top += 1
        else:
            break

    # Bottom border
    for y in range(h - 1, -1, -1):
        row = arr[y, :]
        if row.mean() < 16 and row.var() < 0.1:
            crop_bottom += 1
        else:
            break

    # Left border
    for x in range(w):
        col = arr[:, x]
        if col.mean() < 16 and col.var() < 0.1:
            crop_left += 1
        else:
            break

    # Right border
    for x in range(w - 1, -1, -1):
        col = arr[:, x]
        if col.mean() < 16 and col.var() < 0.1:
            crop_right += 1
        else:
            break

    return crop_top, crop_bottom, crop_left, crop_right


def _compute_ssim_windowed(a: np.ndarray, b: np.ndarray,
                            win_size: int) -> float:
    """Compute SSIM between two arrays using sliding window average."""
    # numpy imported at module level
    h, w = a.shape
    if h < win_size or w < win_size:
        # Fall back to global SSIM if window too large
        mu1 = a.mean()
        mu2 = b.mean()
        sigma1 = a.std()
        sigma2 = b.std()
        sigma12 = np.mean((a - mu1) * (b - mu2))
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
        return float(ssim)

    # Sliding window SSIM
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim_values = []
    step = max(1, win_size // 2)
    for y in range(0, h - win_size + 1, step):
        for x in range(0, w - win_size + 1, step):
            wa = a[y:y+win_size, x:x+win_size]
            wb = b[y:y+win_size, x:x+win_size]
            mu1 = wa.mean()
            mu2 = wb.mean()
            sigma1 = wa.std()
            sigma2 = wb.std()
            if sigma1 < 1e-6 and sigma2 < 1e-6:
                ssim_values.append(1.0)
                continue
            sigma12 = np.mean((wa - mu1) * (wb - mu2))
            ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                       ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
            ssim_values.append(float(ssim_val))
    return float(np.mean(ssim_values)) if ssim_values else 0.0


def _check_psf_ssim(input_path: str) -> bool:
    """Check if fields have high SSIM (PSF indicator).

    Extracts multiple frames, detects and crops black borders,
    computes adaptive-window SSIM on Y channel.
    Returns True if SSIM > 0.98 (likely PSF).
    Uses 3-frame average for robustness.
    """
    try:
        import tempfile
        from PIL import Image

        # Extract 3 frames (beginning, middle, end of first 100 frames)
        frame_indices = [0, 25, 50]
        ssim_values = []

        for idx in frame_indices:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()

            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-v", "error",
                 "-i", input_path, "-vf", f"select=eq(n\\,{idx})",
                 "-frames:v", "1", "-f", "image2", "-y", tmp.name],
                capture_output=True, timeout=30, shell=False
            )

            if r.returncode != 0:
                continue

            img = Image.open(tmp.name).convert("L")
            arr = np.array(img, dtype=np.float64)

            # Detect and crop black borders
            crop_top, crop_bottom, crop_left, crop_right = _detect_black_borders_frame(arr)
            # Ensure we keep at least 50% of the frame
            min_h = arr.shape[0] // 2
            min_w = arr.shape[1] // 2
            effective_h = arr.shape[0] - crop_top - crop_bottom
            effective_w = arr.shape[1] - crop_left - crop_right
            if effective_h < min_h or effective_w < min_w:
                # Too aggressive crop, use minimal crop
                crop_top = min(crop_top, arr.shape[0] // 4)
                crop_bottom = min(crop_bottom, arr.shape[0] // 4)
                crop_left = min(crop_left, arr.shape[1] // 4)
                crop_right = min(crop_right, arr.shape[1] // 4)

            if crop_top + crop_bottom < arr.shape[0] and crop_left + crop_right < arr.shape[1]:
                arr = arr[crop_top:arr.shape[0]-crop_bottom,
                          crop_left:arr.shape[1]-crop_right]

            # Separate fields
            top_field = arr[0::2, :]
            bottom_field = arr[1::2, :]

            # Handle odd height
            h = min(top_field.shape[0], bottom_field.shape[0])
            top_field = top_field[:h, :]
            bottom_field = bottom_field[:h, :]

            if top_field.size == 0 or bottom_field.size == 0:
                continue

            # Adaptive window size
            win_size = max(3, min(11, (arr.shape[1] // 40) | 1, (arr.shape[0] // 40) | 1))
            win_size = min(win_size, min(arr.shape[0], arr.shape[1]))

            ssim = _compute_ssim_windowed(top_field, bottom_field, win_size)
            ssim_values.append(ssim)

        if not ssim_values:
            return False

        avg_ssim = float(np.mean(ssim_values))

        # Primary threshold: > 0.98 = PSF
        if avg_ssim > 0.98:
            return True
        # Secondary check: 0.95-0.98 range requires all 3 frames to agree
        if avg_ssim >= 0.95:
            min_ssim = min(ssim_values)
            return min_ssim > 0.95
        return False

    except Exception:
        return False


def classify_video(input_path: str) -> Dict[str, Any]:
    """Classify video and return full metadata.

    Returns dict with:
    - detected_class: ANALOG_TAPE | DV_DIGITAL | HDV_AVCHD
    - confidence: float 0-1
    - field_order: tt | bb | psf | progressive | unknown
    - needs_ivtc: bool
    - is_psf: bool
    - original_width, original_height
    - native_width, native_height, native_dar
    """
    result: Dict[str, Any] = {
        "detected_class": "UNKNOWN",
        "confidence": 0.0,
        "field_order": "unknown",
        "needs_ivtc": False,
        "is_psf": False,
        "original_width": 0,
        "original_height": 0,
        "native_width": 0,
        "native_height": 0,
        "native_dar": "",
        "file_size": 0,
        "file_mtime": 0,
        "file_hash_prefix": "",
    }

    if not Path(input_path).exists():
        return result

    # File stats
    stat = Path(input_path).stat()
    result["file_size"] = stat.st_size
    result["file_mtime"] = int(stat.st_mtime)

    # Compute hash prefix
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "shared_modules"))
        from checkpoint import get_file_xxhash
        result["file_hash_prefix"] = get_file_xxhash(input_path)
    except Exception:
        pass

    # ffprobe analysis
    info = run_ffprobe(input_path)
    streams = info.get("streams", [])
    video_stream = None
    for s in streams:
        if s.get("codec_type") == "video":
            video_stream = s
            break

    if not video_stream:
        return result

    # Dimensions
    result["original_width"] = video_stream.get("width", 0)
    result["original_height"] = video_stream.get("height", 0)
    result["native_width"] = result["original_width"]
    result["native_height"] = result["original_height"]

    # DAR from display_aspect_ratio
    dar = video_stream.get("display_aspect_ratio", "")
    if dar and dar != "0:1":
        result["native_dar"] = dar

    # Codec-based classification
    codec = video_stream.get("codec_name", "").lower()
    width = int(result["original_width"] or 0)
    height = int(result["original_height"] or 0)

    # DV detection
    if "dv" in codec or "dvvideo" in codec:
        result["detected_class"] = "DV_DIGITAL"
        result["confidence"] = 0.95
    # HDV/AVCHD detection
    elif width >= 1280 or height >= 720:
        if "h264" in codec or "avc" in codec or "mpeg2" in codec:
            result["detected_class"] = "HDV_AVCHD"
            result["confidence"] = 0.90
    # MPEG-2 at SD resolution -> likely analog capture
    elif "mpeg2" in codec or "mpeg1" in codec:
        result["detected_class"] = "ANALOG_TAPE"
        result["confidence"] = 0.85
    # AVI with common analog codecs
    elif codec in ("mjpeg", "huffyuv", "ffv1"):
        result["detected_class"] = "ANALOG_TAPE"
        result["confidence"] = 0.80
    else:
        # Default: assume analog tape for older-looking content
        result["detected_class"] = "ANALOG_TAPE"
        result["confidence"] = 0.70

    # Field order detection (with segment-aware mixed field order support)
    field_order, fo_confidence, field_segments = detect_field_order_segments(input_path)
    result["field_order"] = field_order
    result["is_psf"] = (field_order == "psf")

    # Store segments if mixed field order detected
    if field_segments:
        result["field_segments"] = field_segments

    # IVTC detection: analog tape with TFF/BFF likely needs IVTC
    # Mixed field order materials skip IVTC (IVTC assumes consistent field order globally)
    interlaced_orders = ("tt", "bb")
    if field_order == "unknown" and field_segments:
        # Skip IVTC for mixed field order; per-segment QTGMC handles deinterlacing
        result["needs_ivtc"] = False
    elif result["detected_class"] == "ANALOG_TAPE" and field_order in interlaced_orders:
        result["needs_ivtc"] = True
    elif result["detected_class"] == "DV_DIGITAL" and field_order in interlaced_orders:
        # DV may need IVTC if it has pulldown
        result["needs_ivtc"] = True  # Auto-detect will verify

    # AI upscale banned: HDV/AVCHD 1080p or any video >=1080p height
    # v6.3: base ban on original input pixel dimensions
    if height >= 1080 or width >= 1920:
        result["ai_upscale_banned"] = True
    else:
        result["ai_upscale_banned"] = False

    # Adjust confidence with field order info
    result["confidence"] = min(float(result.get("confidence", 0.0)), fo_confidence + 0.1)

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python classify.py <video_file>")
        sys.exit(81)

    result = classify_video(sys.argv[1])
    print(json.dumps(result, indent=2, ensure_ascii=False))
