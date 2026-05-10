#!/usr/bin/env python3
"""
Preview Mode Module
Generates preview frames at key positions for visual inspection before full processing.
Supports both video and photo inputs.

v7.3 Compliance:
- Preview mode extracts representative frames for user inspection
- Photo preview: extracts thumbnail and center crop
- Video preview: time-spaced frame extraction
"""

import subprocess
from pathlib import Path
from typing import List, Optional


def _is_photo(path: str) -> bool:
    """Check if path is a photo file (by extension)."""
    ext = Path(path).suffix.lower()
    return ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".webp", ".avif")


def extract_preview_frames(input_path: str, output_dir: str,
                           num_frames: int = 6,
                           target_fps: Optional[float] = None) -> List[str]:
    """Extract preview frames at evenly spaced positions.

    For photos: generates a thumbnail and a center crop.
    For videos: time-spaced frame extraction via ffmpeg.

    Args:
        input_path: Source path (video or photo)
        output_dir: Directory to save preview frames
        num_frames: Number of frames to extract (default: 6)
        target_fps: If set, adapt frame positions for VFR content

    Returns:
        List of extracted frame file paths
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(input_path).stem
    frames = []

    if _is_photo(input_path):
        # Photo preview: thumbnail and center crop via PIL
        try:
            from PIL import Image
            with Image.open(input_path) as img:
                # Preview 1: Full thumbnail (max 640px long edge)
                thumb = img.copy()
                thumb.thumbnail((640, 640))
                thumb_file = out_path / f"{stem}_preview_thumb.png"
                thumb.save(thumb_file)
                if thumb_file.exists():
                    frames.append(str(thumb_file))

                # Preview 2: Center crop (512x512)
                w, h = img.size
                crop_size = min(512, w, h)
                left = (w - crop_size) // 2
                top = (h - crop_size) // 2
                center = img.crop((left, top, left + crop_size, top + crop_size))
                center_file = out_path / f"{stem}_preview_center.png"
                center.save(center_file)
                if center_file.exists():
                    frames.append(str(center_file))

                # Preview 3: Top-left crop (detail check)
                detail = img.crop((0, 0, min(512, w), min(512, h)))
                detail_file = out_path / f"{stem}_preview_detail.png"
                detail.save(detail_file)
                if detail_file.exists():
                    frames.append(str(detail_file))
        except Exception:
            pass
        return frames

    # Video path
    # Get video duration
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", input_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        duration = float(r.stdout.strip())
    except Exception:
        duration = 0

    if duration <= 0:
        # Fallback: extract first N seconds
        duration = num_frames * 2.0

    for i in range(num_frames):
        pos = (i / max(num_frames - 1, 1)) * duration
        pos = min(pos, duration - 0.1)

        out_file = out_path / f"{stem}_preview_{i:02d}_at_{pos:.1f}s.png"

        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-v", "error",
                 "-ss", str(pos),
                 "-i", input_path,
                 "-vf", "select=eq(n\\,0)",
                 "-frames:v", "1",
                 "-f", "image2",
                 "-y", str(out_file)],
                capture_output=True, timeout=30, shell=False
            )
            if out_file.exists():
                frames.append(str(out_file))
        except Exception:
            continue

    return frames


def generate_preview_clip(input_path: str, output_path: str,
                          start_sec: float = 0, duration_sec: float = 10,
                          target_width: int = 640) -> bool:
    """Generate a short preview clip (e.g., 10s) for quick inspection.

    Args:
        input_path: Source video path
        output_path: Output preview clip path
        start_sec: Start position in seconds
        duration_sec: Clip duration
        target_width: Target width (height auto-scaled)
    """
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error",
             "-ss", str(start_sec),
             "-i", input_path,
             "-t", str(duration_sec),
             "-vf", f"scale={target_width}:-2",
             "-c:v", "libx264", "-crf", "23", "-preset", "fast",
             "-an", "-y", output_path],
            capture_output=True, timeout=120, shell=False
        )
        return r.returncode == 0
    except Exception:
        return False
