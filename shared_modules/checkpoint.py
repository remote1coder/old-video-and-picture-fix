#!/usr/bin/env python3
"""
Checkpoint / Resume Support Module
Manages param_hash calculation, intermediate file naming,
integrity verification, and disk space estimation.

v6.3 §3.7-3.8 Compliance:
- File identity (size, mtime, hash) always included in param_hash
- Only quality-related parameters included
- Job-specific metadata excluded (output_path, etc.)

v7.3 §4.4-4.5 Compliance:
- Disk space estimation with 2x safety margin
- Frame integrity verification
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image


def get_file_xxhash(filepath: str, hash_length: int = 16) -> str:
    """Get xxHash of file. Falls back to MD5 if xxhash module unavailable."""
    try:
        import xxhash
        h = xxhash.xxh3_64()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()[:hash_length]
    except ImportError:
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()[:hash_length]


def _hash_dict(d: Dict[str, Any]) -> str:
    """Sort + canonical JSON -> SHA-256, return first 16 chars."""
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# Quality parameters that affect processing output.
# Must match v7.32 §3.7 / §3.10.19 specification.
CORE_QUALITY_PARAMS = frozenset([
    # Denoising
    "degrainTR", "grainLevel", "EZDenoise",
    # Deblocking
    "deblock_quant", "deblock_strength",
    # QTGMC
    "qtgmc_preset", "qtgmc_ezdenoise", "qtgmc_sharpness",
    "qtgmc_chromamotion", "qtgmc_sourcematch", "qtgmc_fps_divisor",
    # IVTC
    "ivtc", "VFM_mode", "VDecimate_cycle",
    # Color space
    "matrix_in_s", "matrix_s",
    # AI upscaling
    "upscale_scale", "upscale_model", "face_restore",
    # Grain
    "g1str", "g2str", "g3str", "grain_strength",
    # Bit depth / pix fmt
    "vs_bit_depth", "pix_fmt",
    # Encoding
    "crf", "preset",
    # Face restoration
    "face_detect_threshold", "face_detect_threshold_lowres", "static_segment_duration",
    # Crop / shift / stabilize
    "crop_left", "crop_right", "crop_top", "crop_bottom",
    "chroma_shift", "dxmax", "dymax",
    # Audio
    "hum_freq", "denoise", "ivtc_fallback_threshold",
])


def compute_param_hash(input_path: str, params: Dict[str, Any]) -> str:
    """
    Compute param_hash that uniquely identifies the processing configuration.
    Includes file identity + quality parameters only (excludes job-specific metadata).
    """
    # Compute file identity from input_path
    file_size = 0
    file_mtime = 0
    file_hash_prefix = ""
    try:
        p = Path(input_path)
        if p.exists():
            st = p.stat()
            file_size = st.st_size
            file_mtime = int(st.st_mtime)
            file_hash_prefix = get_file_xxhash(input_path)
    except Exception:
        pass

    identity = {
        "file_size": file_size,
        "file_mtime": file_mtime,
        "file_hash_prefix": file_hash_prefix,
    }

    # Extract only quality-related params
    quality_params: Dict[str, Any] = {}
    for key, value in params.items():
        if key in CORE_QUALITY_PARAMS:
            quality_params[key] = value
        elif isinstance(value, dict):
            # Recurse into nested quality params
            nested = {}
            for nk, nv in value.items():
                if nk in CORE_QUALITY_PARAMS:
                    nested[nk] = nv
            if nested:
                quality_params[key] = nested

    return _hash_dict({
        "identity": identity,
        "quality": quality_params,
    })


def build_intermediate_name(job_id: str,
                            input_path: str,
                            step: int,
                            operation: str,
                            param_hash: str,
                            ext: str = "mkv") -> str:
    """Build intermediate file name: {job_id}_{input_name}_step{N}_{operation}_p{param_hash}.{ext}"""
    safe_name = Path(input_path).stem.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return f"{job_id}_{safe_name}_step{step}_{operation}_p{param_hash}.{ext}"


def cleanup_old_intermediates(temp_dir: str,
                              job_id: str,
                              keep_steps: int = 5,
                              current_step: int = 0) -> None:
    """Remove old intermediate files, keeping the most recent keep_steps stages."""
    temp = Path(temp_dir)
    if not temp.exists():
        return

    # Collect intermediate files for this job
    intermediates = []
    for f in temp.iterdir():
        if f.is_file() and f.name.startswith(f"{job_id}_"):
            # Parse step number from filename
            parts = f.name.split("_")
            for p in parts:
                if p.startswith("step"):
                    try:
                        step = int(p[4:])
                        intermediates.append((step, f))
                    except ValueError:
                        pass
                    break

    if not intermediates:
        return

    # Sort by step, keep the most recent keep_steps
    intermediates.sort(key=lambda x: x[0], reverse=True)
    steps_to_keep = {s for s, _ in intermediates[:keep_steps]}

    for step, f in intermediates:
        if step not in steps_to_keep and step < current_step:
            try:
                f.unlink()
            except OSError:
                pass


def _read_frame_info(filepath: str, position: float) -> Optional[Dict[str, Any]]:
    """Read frame at position (0-1) using ffprobe or PIL."""
    ext = Path(filepath).suffix.lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
        # Photo: read directly with PIL
        try:
            img = Image.open(filepath)
            arr = np.array(img)
            if arr.ndim >= 2:
                return {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "height": arr.shape[0] if arr.ndim > 1 else 1,
                    "width": arr.shape[1] if arr.ndim > 1 else arr.shape[0],
                }
        except Exception:
            pass
        return None

    # Video: use ffprobe
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "csv=p=0", filepath],
            capture_output=True, text=True, timeout=10, shell=False
        )
        duration = float(r.stdout.strip()) if r.returncode == 0 else 0
        if duration <= 0:
            return None

        target_time = duration * position
        r2 = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error",
             "-ss", str(target_time),
             "-i", filepath,
             "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-"],
            capture_output=True, timeout=30, shell=False
        )
        if r2.returncode != 0:
            return None

        w, h = _get_video_dimensions(filepath)
        if w <= 0 or h <= 0:
            return None

        data = r2.stdout
        expected = w * h * 3
        if len(data) < expected:
            return None

        arr = np.frombuffer(data[:expected], dtype=np.uint8).reshape((h, w, 3))
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "height": h,
            "width": w,
        }
    except Exception:
        return None


def _get_video_dimensions(filepath: str) -> Tuple[int, int]:
    """Get video dimensions."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", filepath],
            capture_output=True, text=True, timeout=10, shell=False
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 0, 0


def verify_frame_integrity(filepath: str,
                           is_retry: bool = False,
                           samples: int = 4,
                           bit_depth: int = 8,
                           color_range: str = "limited") -> bool:
    """Verify video/photo file integrity by checking multiple positions.

    Per v6.3 §3.7 / §3.10.19:
    - Mandatory positions: 0-5%, 45-55%, 95-100% (at least 1 frame each)
    - Additional: 1 frame per 10 minutes of duration
    - Brightness range validation (allows legal super-black/white):
      * 8bit limited: 10-245
      * 10bit limited: 40-980
      * full range: 0-max
    """
    ext = Path(filepath).suffix.lower()
    is_photo = ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

    if is_photo:
        try:
            img = Image.open(filepath)
            img.verify()
            img = Image.open(filepath)
            img.load()
            arr = np.array(img)
            if arr.size > 0:
                mean_val = float(np.mean(arr))
                if mean_val < 1.0:
                    return False
            return True
        except Exception:
            return False

    # Determine duration for per-10min sampling
    duration = 0.0
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", filepath],
            capture_output=True, text=True, timeout=10, shell=False
        )
        if r.returncode == 0:
            duration = float(r.stdout.strip())
    except Exception:
        pass

    # Build mandatory + per-10min sample positions
    # Mandatory: ~0.025 (2.5%), ~0.5 (50%), ~0.975 (97.5%)
    positions = [0.025, 0.5, 0.975]
    if duration > 600:
        extra = int(duration // 600)
        for i in range(1, extra + 1):
            pos = min(i * 600 / duration, 0.99)
            if pos not in positions:
                positions.append(pos)

    if is_retry:
        # Add more samples on retry
        positions.extend([0.1, 0.3, 0.7, 0.9])
        positions = sorted(set(positions))

    # Brightness validity range per bit_depth and color_range
    if color_range == "full":
        y_min, y_max = 0, (2 ** bit_depth) - 1
    else:
        if bit_depth == 8:
            y_min, y_max = 10, 245
        elif bit_depth == 10:
            y_min, y_max = 40, 980
        elif bit_depth == 16:
            y_min, y_max = 2560, 37632  # 16*160=2560, 235*160=37600 approx
        else:
            y_min, y_max = 10, 245

    frame_means = []
    for pos in positions:
        info = _read_frame_info(filepath, pos)
        if info is None:
            return False
        mean_y = info.get("mean", 0)
        # Validate pixel values are in legal range (allows super-black/white)
        if not (y_min <= mean_y <= y_max):
            # If mean is outside range, check if it's an edge case (very dark/bright scene)
            # by checking std; high std with extreme mean may still be valid
            std_y = info.get("std", 0)
            if std_y < 1.0:
                return False
        frame_means.append(mean_y)

    if not frame_means:
        return False

    mean_of_means = sum(frame_means) / len(frame_means)

    # Check for all-black or all-white (std near 0 with extreme mean)
    if mean_of_means < y_min + 1 and all(m < y_min + 5 for m in frame_means):
        return False

    if is_retry:
        if any(abs(m - mean_of_means) > 60 for m in frame_means):
            return False
    else:
        if any(abs(m - mean_of_means) > 80 for m in frame_means):
            return False

    return True


def gpu_memory_check(vram_required: int) -> Tuple[bool, str]:
    """Check GPU memory.
    Returns (sufficient: bool, backend_name: str).
    Backend: 'trt' (TensorRT if >6GB), 'cuda' (3-6GB), 'cpu' (<3GB).
    """
    try:
        import torch
        total_vram = torch.cuda.get_device_properties(0).total_memory
        total_vram_mb = total_vram / (1024 * 1024)
        if total_vram_mb >= vram_required:
            if total_vram_mb > 6000:
                return True, "trt"
            elif total_vram_mb > 3000:
                return True, "cuda"
            else:
                return False, "cpu"
        else:
            return False, "cpu"
    except Exception:
        return False, "cpu"


def is_retryable_error(exit_code: int, stderr: str = "") -> bool:
    """Check if error is retryable (transient).

    Per v6.3 §3.7 / v7.3 §4.5:
    - 82 (IO), 83 (GPU/OOM), 87 (resource/timeout) are retryable
    - 84 (encoding) is retryable ONLY if stderr indicates OOM / no space
    - 80 (deps), 81 (params), 85 (perms), 86 (verify) are NOT retryable
    """
    if exit_code in (82, 83, 87):
        return True
    if exit_code == 84 and stderr:
        stderr_lower = stderr.lower()
        if any(k in stderr_lower for k in ("out of memory", "oom", "no space",
                                            "errno 28", "cannot allocate")):
            return True
    return False


def verify_ffprobe_decodeable(filepath: str) -> bool:
    """Verify that ffprobe can decode the file (basic format check)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1", filepath],
            capture_output=True, text=True, timeout=30, shell=False
        )
        return r.returncode == 0 and r.stdout.strip() != ""
    except Exception:
        return False


def is_wsl() -> bool:
    """Detect if running inside Windows Subsystem for Linux."""
    return (
        Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists()
        or os.environ.get("WSL_DISTRO_NAME", "") != ""
    )


def wsl_redirect_path(output_path: str, no_wsl_redirect: bool = False,
                         platform: str = "auto") -> Tuple[str, Optional[str]]:
    """Redirect output path for WSL2 compatibility.
    
    Returns (redirected_path, original_output_path_or_None).
    If redirect is active, original_output_path is saved so caller
    can copy the result back after processing.
    
    Args:
        platform: "auto" | "wsl2" | "linux" | "windows". Only "wsl2" and "auto" allow redirect.
    """
    if no_wsl_redirect:
        return output_path, None
    if platform not in ("wsl2", "auto"):
        return output_path, None
    if not is_wsl():
        return output_path, None
    
    # If already a WSL mount path, no conversion needed
    if output_path.startswith("/mnt/"):
        return output_path, None
    
    # Try wslpath to convert Linux path to Windows path
    try:
        r = subprocess.run(
            ["wslpath", "-w", output_path],
            capture_output=True, text=True, timeout=5, shell=False
        )
        if r.returncode == 0:
            windows_path = r.stdout.strip()
            return windows_path, output_path
    except Exception:
        pass
    
    return output_path, None


def wsl_copy_back(redirected_path: str, original_path: str) -> bool:
    """Copy result from WSL redirected path back to original Linux path.
    
    Returns True on success, False on failure.
    """
    try:
        import shutil
        os.makedirs(os.path.dirname(original_path) or ".", exist_ok=True)
        shutil.copy2(redirected_path, original_path)
        return True
    except Exception:
        return False


def estimate_disk_space(duration: float = 300,
                        fps: float = 29.97,
                        width: int = 720,
                        height: int = 480,
                        file_path: Optional[str] = None,
                        is_photo: bool = False) -> int:
    """
    Estimate peak disk space in MB needed for restoration pipeline.
    
    For photos: uses actual file size as base (no duration/fps needed).
    For videos: uses duration * fps calculation.
    
    Adds 2x safety margin.
    """
    if is_photo and file_path and os.path.exists(file_path):
        # Use actual file size as base for photos
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        # Photo processing typically produces intermediates 3-5x original size
        # Max intermediate count: ~5 stages
        peak = file_size_mb * 5 * 5  # 5 stages, 5x each
    else:
        # Video estimation (original logic)
        if duration <= 0:
            duration = 300  # Default 5 minutes
        if fps <= 0:
            fps = 29.97

        uncompressed_frame = width * height * 3 / (1024 * 1024)  # MB
        total_frames = duration * fps

        # Peak intermediate: ~3 copies (original + preprocess + AI output)
        peak = uncompressed_frame * total_frames * 3

    # 2x safety margin
    return int(peak * 2)
