#!/usr/bin/env python3
"""
Preview Mode Module
Generates preview frames at key positions for visual inspection before full processing.
Supports both video and photo inputs.

v7.3 Compliance:
- Preview mode extracts representative frames for user inspection
- Photo preview: extracts thumbnail and center crop
- Video preview: time-spaced frame extraction with output seeking for accuracy
- 0.5s fixed-duration motion segments with dynamic frame count by fps
- WebP animation output for motion preview
- Forced upscale comparison frames
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from logger import get_logger


def _is_photo(path: str) -> bool:
    """Check if path is a photo file (by extension)."""
    ext = Path(path).suffix.lower()
    return ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".webp", ".avif")


def _get_video_fps(input_path: str) -> float:
    """Get video fps using ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", input_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            rate = r.stdout.strip()
            if "/" in rate:
                num, den = rate.split("/")
                return float(num) / float(den)
            return float(rate)
    except Exception:
        pass
    return 25.0


def _is_vfr(input_path: str) -> bool:
    """Detect Variable Frame Rate (VFR) by comparing r_frame_rate vs avg_frame_rate.

    Per v7.32 §3.8 / v6.32: VFR material requires frame-number-based extraction
    instead of time-based seeking for accurate positioning.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate,avg_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", input_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            lines = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
            if len(lines) >= 2:
                r_rate = lines[0]
                avg_rate = lines[1]
                # Parse both rates
                def _parse_rate(s: str) -> float:
                    if "/" in s:
                        n, d = s.split("/")
                        return float(n) / float(d)
                    return float(s)
                r_fps = _parse_rate(r_rate)
                avg_fps = _parse_rate(avg_rate)
                # If difference > 1% or avg is unavailable (0), treat as VFR
                if avg_fps <= 0 or abs(r_fps - avg_fps) > 0.01 * max(r_fps, avg_fps):
                    return True
    except Exception:
        pass
    return False


def _check_pts_monotonic(input_path: str, max_frames: int = 200) -> bool:
    """Check if video PTS is monotonically increasing.

    Per v7.32 §3.8(注): Validate PTS monotonicity before frame-number extraction.
    FIX-P3-7
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "frame=pkt_pts_time",
             "-of", "csv=p=0", input_path],
            capture_output=True, text=True, timeout=60, shell=False
        )
        if r.returncode != 0:
            return False
        pts_list = []
        for line in r.stdout.strip().splitlines():
            line = line.strip()
            if line:
                try:
                    pts_list.append(float(line))
                except ValueError:
                    continue
            if len(pts_list) >= max_frames:
                break
        for i in range(1, len(pts_list)):
            # FIX-WARN-6: Per v7.32 §3.8(注): validate PTS is monotonically
            # increasing. ORIGINAL used strict `<` (only flagged decreasing
            # PTS), so equal/duplicate PTS were treated as valid and could
            # break frame-number extraction positioning. Use `<=` so equal
            # PTS also fails the check -> caller falls back to PTS-based
            # seeking (extract_preview_frames -ss path), which is still
            # correct, just slightly slower.
            if pts_list[i] <= pts_list[i - 1]:
                return False
        return True
    except Exception:
        return False


def _get_total_frames(input_path: str) -> int:
    """Estimate total frame count for VFR frame-number-based extraction."""
    # First try nb_frames from stream metadata
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames",
             "-of", "default=noprint_wrappers=1:nokey=1", input_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            val = r.stdout.strip()
            if val and val != "N/A":
                return int(val)
    except Exception:
        pass
    # Fallback: duration * avg_frame_rate
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=duration,stream=avg_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", input_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            lines = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
            if len(lines) >= 2:
                duration = float(lines[0])
                avg_rate = lines[1]
                if "/" in avg_rate:
                    n, d = avg_rate.split("/")
                    avg_fps = float(n) / float(d)
                else:
                    avg_fps = float(avg_rate)
                if duration > 0 and avg_fps > 0:
                    return int(duration * avg_fps)
    except Exception:
        pass
    return 0


def _get_video_bit_depth(input_path: str) -> Optional[int]:
    """Get video bit depth using ffprobe.

    FIX-P3-7: Avoid wasting space by forcing rgb48be for 8bit sources.
    Returns None if bit depth cannot be determined (conservative fallback
    to rgb48be).
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=bits_per_raw_sample",
             "-of", "default=noprint_wrappers=1:nokey=1", input_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            val = r.stdout.strip()
            if val and val != "N/A":
                return int(val)
    except Exception:
        pass
    return None


def extract_preview_frames(input_path: str, output_dir: str,
                           num_frames: int = 6,
                           target_fps: Optional[float] = None) -> List[str]:
    """Extract preview frames at key positions.

    For photos: generates a thumbnail and crops.
    For videos: time-spaced frame extraction via ffmpeg with output seeking.

    Args:
        input_path: Source path (video or photo)
        output_dir: Directory to save preview frames
        num_frames: Number of frames to extract (default: 6)
        target_fps: If set, adapt frame positions for VFR content

    Returns:
        List of extracted frame file paths
    """
    out_path = Path(output_dir) / "preview"
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(input_path).stem
    frames = []

    if _is_photo(input_path):
        # Photo preview: thumbnail and crops via PIL
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
        duration = num_frames * 2.0

    # Extract frames at 10%, 50%, 90% positions (v7.32 spec)
    positions = [0.1, 0.5, 0.9]
    if num_frames > 3:
        # Add evenly spaced additional frames
        for i in range(1, num_frames - 2):
            pos = i / (num_frames - 1)
            if pos not in positions:
                positions.append(pos)
        positions = sorted(set(positions))[:num_frames]

    # ISSUE-V-PREV-001 fix: detect VFR and switch to frame-number extraction
    vfr = _is_vfr(input_path)
    pts_ok = True
    total_frames = 0
    if vfr:
        pts_ok = _check_pts_monotonic(input_path)
        if not pts_ok:
            logger = get_logger()
            logger.warning("PREVIEW_VFR_PTS_BAD",
                          "VFR detected with non-monotonic PTS, "
                          "switching to PTS-coverage extraction")
        total_frames = _get_total_frames(input_path)

    # FIX-P3-7: Use dynamic pix_fmt based on source bit depth.
    # 8bit sources -> rgb24 (saves ~50% PNG space), 10bit+ -> rgb48be
    bit_depth = _get_video_bit_depth(input_path)
    pix_fmt = "rgb24" if (bit_depth and bit_depth <= 8) else "rgb48be"

    for pos in positions:
        out_file = out_path / f"{stem}_preview_{pos:.2f}.png"

        try:
            # FIX-P3-5: Per v7.32 §3.8(注): Use frame-number extraction only
            # when VFR + valid PTS + known frame count. If PTS is
            # non-monotonic, fall back to PTS-based seeking.
            if vfr and pts_ok and total_frames > 0:
                # Frame-number-based extraction for VFR accuracy
                target_frame = int(pos * total_frames)
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-v", "error",
                     "-i", input_path,
                     # FIX-P3-4: shell=False no escaping needed
                     "-vf", f"select=eq(n,{target_frame})",
                     "-vsync", "vfr",
                     "-frames:v", "1",
                     "-pix_fmt", pix_fmt,
                     "-compression_level", "9",
                     "-f", "image2",
                     "-y", str(out_file)],
                    capture_output=True, timeout=30, shell=False
                )
            else:
                # Standard PTS-based output seeking (CFR or VFR with bad PTS)
                target_time = pos * duration
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-v", "error",
                     "-i", input_path,
                     "-ss", str(target_time),
                     "-frames:v", "1",
                     "-pix_fmt", pix_fmt,
                     "-compression_level", "9",
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
                          start_sec: float = 0, duration_sec: float = 0.5,
                          target_width: int = 640) -> bool:
    """Generate a short preview clip with fixed 0.5s duration.
    
    Uses WebP animation for motion preview per v7.32 §3.10.11.
    Dynamic frame count based on source fps:
    - 25fps -> 12 frames
    - 50fps -> 25 frames
    - 60fps -> 30 frames

    Args:
        input_path: Source video path
        output_path: Output preview clip path (should end in .webp)
        start_sec: Start position in seconds
        duration_sec: Clip duration (default 0.5s per spec)
        target_width: Target width (height auto-scaled)
    """
    try:
        fps = _get_video_fps(input_path)
        frame_count = max(1, int(fps * duration_sec))
        
        # ISSUE-SHARED-PV-001 fix: Single-stage ffmpeg directly to animated WebP.
        # FIX-P3-3: Per v7.32 §3.8: Use output seeking (-ss AFTER -i) for
        # accuracy. Input seeking before -i can misposition on VFR content.
        # Falls back to two-stage PNG extraction if single-stage fails.
        # FIX-P3-6: Dynamic frame count by fps per v7.32 §3.10.21
        vf = f"fps={fps},scale={target_width}:-2"
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error",
             "-i", input_path,
             "-ss", str(start_sec),      # ← output seeking (after -i)
             "-t", str(duration_sec),
             "-vf", vf,
             "-c:v", "libwebp",
             "-lossless", "0", "-q:v", "80",
             "-loop", "0",
             "-an",  # No audio
             "-y", output_path],
            capture_output=True, timeout=120, shell=False
        )
        
        if r.returncode == 0:
            return True
        
        # Fallback: two-stage PNG extraction + WebP assembly
        frame_count = max(1, int(fps * duration_sec))
        temp_dir = Path(tempfile.mkdtemp(prefix="preview_"))
        stem = Path(input_path).stem
        
        try:
            r_extract = subprocess.run(
                ["ffmpeg", "-hide_banner", "-v", "error",
                 "-i", input_path,
                 "-ss", str(start_sec),      # ← output seeking (after -i)
                 "-t", str(duration_sec),
                 "-vf", f"scale={target_width}:-2",
                 "-vsync", "vfr",
                 "-y", str(temp_dir / f"{stem}_frame_%03d.png")],
                capture_output=True, timeout=60, shell=False
            )

            # FIX-P3-8: Verify stage 1 succeeded before attempting stage 2
            if r_extract.returncode != 0:
                return False

            # Verify at least one PNG was extracted
            png_files = sorted(temp_dir.glob(f"{stem}_frame_*.png"))
            if not png_files:
                return False

            r2 = subprocess.run(
                ["ffmpeg", "-hide_banner", "-v", "error",
                 "-framerate", str(fps),
                 "-i", str(temp_dir / f"{stem}_frame_%03d.png"),
                 "-vf", f"scale={target_width}:-2",
                 "-lossless", "0", "-q:v", "80",
                 "-loop", "0",
                 "-y", output_path],
                capture_output=True, timeout=120, shell=False
            )

            return r2.returncode == 0
        finally:
            # Cleanup temp frames
            for f in temp_dir.glob(f"{stem}_frame_*.png"):
                f.unlink(missing_ok=True)
            temp_dir.rmdir()
    except Exception:
        return False


def generate_comparison_frames(original_path: str, processed_path: str,
                               output_dir: str, num_pairs: int = 3) -> List[str]:
    """Generate side-by-side comparison frames for forced upscale review.
    
    Args:
        original_path: Original input path
        processed_path: Processed output path
        output_dir: Directory to save comparison frames
        num_pairs: Number of comparison pairs
    
    Returns:
        List of comparison frame paths
    """
    out_path = Path(output_dir) / "preview"
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(original_path).stem
    results = []
    
    if _is_photo(original_path):
        try:
            from PIL import Image
            # FIX-P3-7: Use context manager to properly close image resources
            with Image.open(original_path) as orig, \
                 Image.open(processed_path) as proc:
                w = max(orig.width, proc.width)
                h = max(orig.height, proc.height)
                combined = Image.new("RGB", (w * 2, h))
                combined.paste(orig.resize((w, h)), (0, 0))
                combined.paste(proc.resize((w, h)), (w, 0))

                out_file = out_path / f"{stem}_compare.png"
                combined.save(out_file)
                if out_file.exists():
                    results.append(str(out_file))
        except Exception:
            pass
    else:
        # Video: extract at 10%, 50%, 90% and create side-by-side
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", original_path],
                capture_output=True, text=True, timeout=30, shell=False
            )
            duration = float(r.stdout.strip())
            positions = [0.1, 0.5, 0.9][:num_pairs]
            
            for pos in positions:
                t = pos * duration
                orig_frame = out_path / f"{stem}_orig_{pos:.2f}.png"
                proc_frame = out_path / f"{stem}_proc_{pos:.2f}.png"
                
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-v", "error",
                     "-i", original_path, "-ss", str(t), "-frames:v", "1", "-y", str(orig_frame)],
                    capture_output=True, timeout=30, shell=False
                )
                subprocess.run(
                    ["ffmpeg", "-hide_banner", "-v", "error",
                     "-i", processed_path, "-ss", str(t), "-frames:v", "1", "-y", str(proc_frame)],
                    capture_output=True, timeout=30, shell=False
                )
                
                if orig_frame.exists() and proc_frame.exists():
                    compare_file = out_path / f"{stem}_compare_{pos:.2f}.png"
                    # Per v7.32 §3.8: scale both inputs to same size before hstack
                    # to handle resolution changes after processing (e.g., upscale)
                    subprocess.run(
                        ["ffmpeg", "-hide_banner", "-v", "error",
                         "-i", str(orig_frame), "-i", str(proc_frame),
                         "-filter_complex",
                         "[0:v]scale=640:-2:flags=lanczos[orig];"
                         "[1:v]scale=640:-2:flags=lanczos[proc];"
                         "[orig][proc]hstack",
                         "-y", str(compare_file)],
                        capture_output=True, timeout=30, shell=False
                    )
                    if compare_file.exists():
                        results.append(str(compare_file))
        except Exception:
            pass
    
    return results


def detect_motion_segments(input_path: str, num_segments: int = 3,
                           segment_duration: float = 0.5) -> List[Tuple[float, float]]:
    """Detect motion magnitude and return top N most active segments.
    
    Per v7.32 §3.8: Detect motion magnitude, extract 3 most intense
    0.5-second duration sequences.
    
    Uses sparse frame sampling (every ~0.5s) and pixel difference (MAD)
    as motion metric. Avoids heavy optical flow computation.
    
    Returns:
        List of (start_sec, motion_score) tuples, sorted by motion_score
        descending. Non-overlapping segments are selected.
    """
    try:
        # Get duration
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", input_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        duration = float(r.stdout.strip())
        if duration <= 0:
            # FIX-P3-8: Return empty list instead of misleading
            # [(0.0, 0.0)] — caller can distinguish "no segments"
            # from "segment at t=0.0 with score 0.0"
            return []
        
        # Sample frames every ~0.5s for motion analysis
        sample_interval = max(0.5, segment_duration)
        num_samples = int(duration / sample_interval)
        if num_samples < 2:
            # FIX-P3-8: Return empty list — too short for motion analysis
            return []
        
        # BUG-061 fix: Use system temp directory instead of input file's parent
        # to avoid PermissionError on read-only locations (e.g., optical drives,
        # network shares, system-protected directories).
        temp_dir = Path(tempfile.mkdtemp(prefix="motion_"))
        
        frames = []
        for i in range(num_samples):
            t = i * sample_interval
            frame_path = temp_dir / f"sample_{i:04d}.png"
            # FIX-WARN-7: ORIGINAL used input seeking (`-ss <t> -i input`),
            # which is keyframe-aligned and mispositions on VFR content —
            # inconsistent with generate_preview_clip (L357) which uses
            # output seeking. Per v7.32 §3.8, switch to output seeking
            # (`-i input -ss <t>`) so motion-segment sampling lands on the
            # same frame as the static preview extraction. Cost: output
            # seeking decodes from the start to the target, but motion
            # detection only samples one frame every ~0.5s, so acceptable.
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-v", "error",
                 "-i", input_path,
                 "-ss", str(t),          # output seek (after -i)
                 "-frames:v", "1", "-y", str(frame_path)],
                capture_output=True, timeout=30, shell=False
            )
            if frame_path.exists():
                frames.append((t, frame_path))
        
        if len(frames) < 2:
            # Cleanup and fallback
            for _, fp in frames:
                fp.unlink(missing_ok=True)
            temp_dir.rmdir()
            # FIX-P3-8: Return empty list — insufficient frames
            return []
        
        # Calculate motion scores (MAD between adjacent frames)
        from PIL import Image
        import numpy as np
        
        motion_scores = []
        for i in range(len(frames) - 1):
            t1, fp1 = frames[i]
            try:
                img1 = np.array(Image.open(fp1).convert("L"))
                img2 = np.array(Image.open(frames[i + 1][1]).convert("L"))
                # Resize to same size if needed (handle resolution changes)
                if img1.shape != img2.shape:
                    h = min(img1.shape[0], img2.shape[0])
                    w = min(img1.shape[1], img2.shape[1])
                    img1 = img1[:h, :w]
                    img2 = img2[:h, :w]
                mad = np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))
                motion_scores.append((t1, float(mad)))
            except Exception:
                motion_scores.append((t1, 0.0))
        
        # Cleanup temp frames
        for _, fp in frames:
            fp.unlink(missing_ok=True)
        temp_dir.rmdir()
        
        # Sort by motion score descending
        motion_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top N non-overlapping segments
        selected = []
        min_gap = segment_duration * 2  # Avoid overlapping segments
        for t, score in motion_scores:
            overlap = False
            for sel_t, _ in selected:
                if abs(t - sel_t) < min_gap:
                    overlap = True
                    break
            if not overlap:
                selected.append((t, score))
                if len(selected) >= num_segments:
                    break
        
        if not selected:
            # FIX-P3-8: Return empty list instead of misleading placeholder
            return []
        
        return selected
    
    except Exception:
        # FIX-P3-8: Return empty list on error
        return []


def generate_motion_preview_clips(input_path: str, output_dir: str,
                                  num_segments: int = 3,
                                  segment_duration: float = 0.5,
                                  target_width: int = 640) -> List[str]:
    """Generate preview clips for the most motion-active segments.
    
    Per v7.32 §3.8: Detect motion magnitude, extract 3 most intense
    0.5-second segments, output as WebP animations.
    
    Args:
        input_path: Source video path
        output_dir: Directory to save preview clips
        num_segments: Number of motion segments to extract (default 3)
        segment_duration: Duration of each segment in seconds (default 0.5)
        target_width: Target width for preview clips
    
    Returns:
        List of generated WebP clip paths
    """
    out_path = Path(output_dir) / "preview"
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    
    segments = detect_motion_segments(input_path, num_segments, segment_duration)
    # FIX-P3-8: Handle empty segment list gracefully
    if not segments:
        return []
    results = []
    
    for idx, (start_sec, score) in enumerate(segments):
        out_file = out_path / f"{stem}_motion_{idx}_{start_sec:.2f}s.webp"
        if generate_preview_clip(input_path, str(out_file), start_sec,
                                 segment_duration, target_width):
            results.append(str(out_file))

    return results


def auto_generate_comparison_if_needed(
    original_path: str,
    processed_path: str,
    output_dir: str,
    config: dict = None,
    num_pairs: int = 3
) -> List[str]:
    """Auto-generate comparison frames when force_allow_banned_upscale is enabled.

    Per v7.32 §3.8: Automatically generate before/after comparison frames
    when force_allow_banned_upscale: true.

    Args:
        original_path: Original input path
        processed_path: Processed output path
        output_dir: Directory to save comparison frames
        config: Full pipeline config dict
        num_pairs: Number of comparison pairs

    Returns:
        List of comparison frame paths (empty if not triggered)
    """
    if config is None:
        return []

    # Check force_allow_banned_upscale in ai config hierarchy
    ai_config = config.get("ai", {})
    if not ai_config.get("force_allow_banned_upscale", False):
        return []

    return generate_comparison_frames(
        original_path, processed_path, output_dir, num_pairs
    )
