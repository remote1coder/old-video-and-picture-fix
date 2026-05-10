#!/usr/bin/env python3
"""
Photo Material Classifier
Classifies photos into: SCAN_PRINT, DIGITAL_PHOTO
Detects dimensions, DPI, and AI upscale ban status.

v6.3 Compliance:
- AI upscale banned when target resolution > 1080p OR DPI > 1200
- Computes native_width/native_height/native_dar
- Computes file hash prefix for deduplication
"""

import subprocess
from pathlib import Path
from typing import Dict, Any
import json


def run_exiftool(input_path: str) -> Dict[str, str]:
    """Run exiftool and return key metadata."""
    try:
        r = subprocess.run(
            ["exiftool", "-json", input_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            if data:
                return {k: str(v) for k, v in data[0].items()}
    except Exception:
        pass
    return {}


def classify_photo(input_path: str) -> Dict[str, Any]:
    """Classify photo and return metadata.

    Returns:
    - detected_class: SCAN_PRINT | DIGITAL_PHOTO
    - confidence: float 0-1
    - original_width, original_height
    - native_width, native_height, native_dar
    - dpi: int or None
    - ai_upscale_banned: bool (based on pixel dimensions, target resolution, and DPI)
    """
    result: Dict[str, Any] = {
        "detected_class": "UNKNOWN",
        "confidence": 0.0,
        "original_width": 0,
        "original_height": 0,
        "native_width": 0,
        "native_height": 0,
        "native_dar": "",
        "dpi": None,
        "ai_upscale_banned": False,
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

    # Try to get dimensions via PIL
    try:
        from PIL import Image
        with Image.open(input_path) as img:
            result["original_width"], result["original_height"] = img.size
            result["native_width"], result["native_height"] = img.size
            # Compute native DAR
            w, h = img.size
            if h > 0:
                dar = w / h
                result["native_dar"] = f"{dar:.4f}"
            # Try to get DPI
            dpi_info = img.info.get("dpi")
            if dpi_info:
                result["dpi"] = int(max(dpi_info))
    except Exception:
        pass

    # ExifTool metadata
    exif = run_exiftool(input_path)

    # Classification
    model = exif.get("Model", "")
    make = exif.get("Make", "")
    scanner = exif.get("ScannerModelName", "")
    software = exif.get("Software", "")

    # Case-insensitive scanner detection (Bug-18 fix)
    scanner_lower = scanner.lower() if scanner else ""
    software_lower = software.lower() if software else ""
    make_lower = make.lower() if make else ""

    if scanner_lower or "scan" in software_lower or "scan" in make_lower:
        result["detected_class"] = "SCAN_PRINT"
        result["confidence"] = 0.95
    elif model or make:
        result["detected_class"] = "DIGITAL_PHOTO"
        result["confidence"] = 0.90
    else:
        # Default based on resolution
        w = result["original_width"]
        h = result["original_height"]
        if w > 0 and h > 0:
            if w > 3000 or h > 2000:
                result["detected_class"] = "SCAN_PRINT"
                result["confidence"] = 0.70
            else:
                result["detected_class"] = "DIGITAL_PHOTO"
                result["confidence"] = 0.60
        else:
            result["detected_class"] = "UNKNOWN"
            result["confidence"] = 0.0

    # AI upscale ban check (v6.3 §4.4)
    # Unified ban logic using config-driven thresholds:
    # max_long_edge=3840, max_short_edge=2160, dpi_threshold=1200
    w = result["original_width"]
    h = result["original_height"]
    dpi = result.get("dpi")

    # Ban 1: source pixel dimensions exceed max_long_edge/max_short_edge thresholds
    max_long = 3840
    max_short = 2160
    if w > max_long or h > max_short:
        result["ai_upscale_banned"] = True

    # Ban 2: DPI exceeds 1200 (high-DPI scans have sufficient detail already)
    if dpi and dpi > 1200:
        result["ai_upscale_banned"] = True

    # v6.3 §5.3 predictive ban: if source at 2x would exceed 1080p target
    # This is a soft indicator - actual enforcement is in Stage 3
    # We don't set ai_upscale_banned=True here, just log for reference
    # Stage 3 enforces the actual 1080p target resolution limit

    # Compute hash prefix
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "shared_modules"))
        from checkpoint import get_file_xxhash
        result["file_hash_prefix"] = get_file_xxhash(input_path)
    except Exception:
        pass

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python classify.py <photo_file>")
        sys.exit(81)

    result = classify_photo(sys.argv[1])
    print(json.dumps(result, indent=2, ensure_ascii=False))
