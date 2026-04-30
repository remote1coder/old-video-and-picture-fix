#!/usr/bin/env python3
"""
Photo Material Classifier
Classifies photos into: SCAN_PRINT, DIGITAL_PHOTO
Detects dimensions, DPI, and AI upscale ban status.
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
    - dpi: int or None
    - ai_upscale_banned: bool (based on pixel dimensions)
    """
    result: Dict[str, Any] = {
        "detected_class": "UNKNOWN",
        "confidence": 0.0,
        "original_width": 0,
        "original_height": 0,
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

    if scanner or "scan" in software.lower() or "scan" in make.lower():
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

    # AI upscale ban check (pixel size AND DPI threshold)
    w = result["original_width"]
    h = result["original_height"]
    # Ban 1: pixel dimensions exceed threshold
    if w > 3840 or h > 2160:
        result["ai_upscale_banned"] = True
    # Ban 2: DPI exceeds 1200 (high-DPI scans have sufficient detail already)
    dpi = result.get("dpi")
    if dpi and dpi > 1200:
        result["ai_upscale_banned"] = True

    # Compute hash prefix
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "shared_utils"))
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
