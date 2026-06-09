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
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image


def get_file_xxhash(filepath: str, hash_length: int = 16) -> str:
    """Get xxHash of file. Falls back to MD5 if xxhash module unavailable.
    
    Per v7.32 §3.10.19: For files >1GB, sample head/tail/middle blocks
    to avoid long hashing times. Uses xxhash64 (not xxh3_64).
    """
    # BUG-012 fix: log WARNING when falling back to MD5
    try:
        import xxhash
        h = xxhash.xxh64()
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "CHECKPOINT_HASH_FALLBACK",
            "xxhash module unavailable, falling back to MD5 for file identity hashing"
        )
        h = hashlib.md5()
    
    file_size = os.path.getsize(filepath)
    block_size = 65536
    
    with open(filepath, "rb") as f:
        if file_size > (1024 * 1024 * 1024):  # > 1GB
            # Sample head, middle, tail
            sample_size = block_size * 16  # 1MB per region
            # Head
            f.seek(0)
            h.update(f.read(sample_size))
            # Middle
            f.seek(file_size // 2)
            h.update(f.read(sample_size))
            # Tail
            f.seek(max(0, file_size - sample_size))
            h.update(f.read(sample_size))
            # Also include file size in hash context
            h.update(str(file_size).encode())
        else:
            while chunk := f.read(block_size):
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
    "degrainTR", "grainLevel",
    # Deblocking
    "deblock_quant", "deblock_strength",
    # QTGMC
    "qtgmc_preset", "qtgmc_ezdenoise", "qtgmc_sharpness",
    "qtgmc_chromamotion", "qtgmc_sourcematch", "qtgmc_fps_divisor",
    # IVTC
    "ivtc",
    # Color space
    "matrix_in_s", "matrix_s",
    # AI upscaling
    "upscale_scale", "upscale_model", "face_restore",
    # Grain
    "g1str", "g2str", "g3str", "grain_strength",
    # Bit depth / pix fmt
    "vs_bit_depth", "pix_fmt",
    # Encoding
    "crf", "preset", "preset_max_quality",
    # Face restoration
    "face_detect_threshold", "face_detect_threshold_lowres",
    "face_detect_lowres_max_dimension", "face_detect_fallback",
    "static_segment_duration",
    # Crop / shift / stabilize
    "crop_left", "crop_right", "crop_top", "crop_bottom",
    "chroma_shift", "dxmax", "dymax",
    # Audio
    "hum_freq", "hum_harmonics", "hum_dual", "hum_bandwidth",
    "denoise", "codec", "bitrate",
    "ivtc_fallback_threshold",
    # Photo-specific parameters (v6.32 §4.2.1 / §7.7)
    "scratch_remove", "scratch_remove_strength",
    "deblock_strength", "deband_threshold",
    "grain_synthesis", "grain_strength", "grain_size", "grain_roughness",
    "photo_bit_depth",
    "channel_correction_formula", "channel_correction_normalize",
    "channel_correction_round", "channel_correction_clamp",
    "channel_degradation_mean_threshold", "channel_degradation_std_threshold",
    "channel_degradation_mean_threshold_severe", "channel_degradation_std_threshold_severe",
    "channel_correction_linear_space", "channel_correction_soft_compress",
    "channel_correction_float_precision",
    "archival_mode", "local_mask_denoise",
    # QTGMC / field order behavior
    "qtgmc_input_type", "qtgmc_input_type_rule", "qtgmc_tff_required", "qtgmc_clear_fieldbased",
    # IVTC / PSF / deblock / deinterlace method selection
    "ivtc_auto_fallback", "psf_handling", "deblock", "deinterlace",
    # AI upscale banning and gating
    "max_scale", "force_traditional_first", "force_allow_banned_upscale",
    "ban_original_input_pixels", "ban_intermediate_pixels",
    # GrainFactory3 behavior
    "addgrainc_enabled", "addgrainc_ratio",
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

    # Extract only quality-related params (recursive nested traversal)
    def _extract_quality(obj: Any) -> Any:
        if isinstance(obj, dict):
            result: Dict[str, Any] = {}
            for k, v in obj.items():
                if k in CORE_QUALITY_PARAMS:
                    result[k] = v
                else:
                    nested = _extract_quality(v)
                    if nested is not None and nested != {}:
                        result[k] = nested
            return result
        return None

    quality_params = _extract_quality(params) or {}

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
                              keep_steps: int = 2,
                              current_step: Optional[int] = None) -> None:
    """Remove old intermediate files, keeping the most recent keep_steps stages.
    
    Args:
        current_step: The current stage number. If None, keeps the most recent
            keep_steps files and deletes all others (useful at job start).
            If specified, deletes files whose step < current_step - keep_steps.
    """
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

    # Sort by step descending (newest first)
    intermediates.sort(key=lambda x: x[0], reverse=True)

    if current_step is None:
        # BUG-058 fix: When called without current_step, keep the most recent
        # keep_steps files and delete everything else.
        for _, f in intermediates[keep_steps:]:
            try:
                f.unlink()
            except OSError:
                pass
    else:
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
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".avif"]:
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


def _probe_bit_depth_from_pixfmt(filepath: str) -> int:
    """Probe bit depth from video pixel format using ffprobe.
    
    Returns detected bit depth or 8 as fallback.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=pix_fmt",
             "-of", "default=noprint_wrappers=1:nokey=1", filepath],
            capture_output=True, text=True, timeout=10, shell=False
        )
        if r.returncode == 0:
            pix_fmt = r.stdout.strip().lower()
            # Map common pixel formats to bit depth
            if any(x in pix_fmt for x in ["p10", "10le", "10be", "yuv420p10", "yuv422p10", "yuv444p10", "rgb30"]):
                return 10
            if any(x in pix_fmt for x in ["p12", "12le", "12be"]):
                return 12
            if any(x in pix_fmt for x in ["p16", "16le", "16be", "rgb48", "rgba64", "yuv420p16", "yuv422p16", "yuv444p16"]):
                return 16
            if any(x in pix_fmt for x in ["p14", "14le", "14be"]):
                return 14
            if "p9" in pix_fmt or "9le" in pix_fmt or "9be" in pix_fmt:
                return 9
            # 8bit formats: yuv420p, yuv422p, yuv444p, rgb24, bgr24, etc.
            return 8
    except Exception:
        pass
    return 8


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
      * 16bit limited: 4096-60160 (industry convention: 16*256, 235*256)
      * full range: 0-max
    
    ISSUE-SHARED-CK-001 fix: If bit_depth is not explicitly overridden (default 8),
    auto-detect from the file's pixel format via ffprobe.
    """
    ext = Path(filepath).suffix.lower()
    is_photo = ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".avif"]
    
    # ISSUE-SHARED-CK-001: Auto-detect bit depth from pixel format for video files
    if not is_photo and bit_depth == 8:
        detected = _probe_bit_depth_from_pixfmt(filepath)
        if detected != 8:
            bit_depth = detected

    if is_photo:
        try:
            img = Image.open(filepath)
            img.verify()
            img = Image.open(filepath)
            img.load()
            arr = np.array(img)
            if arr.size > 0:
                # Per v6.32 §3.7 / v7.32 §3.10.19: do NOT reject photos based on
                # content (dark scenes, solid colors are legitimate). Only reject
                # if PIL verify/load fails or array is empty.
                return True
            return False
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
            y_min, y_max = 4096, 60160  # industry convention: 16*256, 235*256
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


def _check_nvidia_smi_vram() -> Optional[int]:
    """Query available VRAM via nvidia-smi. Returns bytes or None."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, shell=False
        )
        if r.returncode == 0:
            line = r.stdout.strip().splitlines()[0]
            mib = float(line)
            return int(mib * 1024 * 1024)
    except Exception:
        pass
    # Fallback: parse full table output
    try:
        r = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10, shell=False
        )
        if r.returncode == 0:
            import re
            for line in r.stdout.splitlines():
                m = re.search(r'(\d+)\s*MiB\s*/\s*\d+\s*MiB', line)
                if m:
                    mib = float(m.group(1))
                    return int(mib * 1024 * 1024)
    except Exception:
        pass
    return None


def _probe_cuda_paths() -> Optional[str]:
    """Probe common CUDA library paths and return LD_LIBRARY_PATH addition."""
    cuda_paths = [
        "/usr/local/cuda/lib64",
        "/usr/lib/wsl/lib",
        "/opt/cuda/lib64",
    ]
    found = []
    for p in cuda_paths:
        if os.path.exists(os.path.join(p, "libcuda.so.1")) or \
           os.path.exists(os.path.join(p, "libnvidia-ml.so.1")):
            found.append(p)
    if found:
        return ":".join(found)
    return None


def gpu_memory_check(vram_required_mb: int,
                        gpu_priority: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Check GPU memory.

    Args:
        vram_required_mb: Required VRAM in MiB (megabytes).
            Per v6.32 §7.14 / v7.32 §3.10.23, callers must pass MiB values
            (e.g., 4096 for 4GB, 8192 for 8GB).
        gpu_priority: Backend priority list (default: CUDA → ROCm → MPS → CPU).

    Returns:
        (sufficient: bool, backend_name: str).
    """
    if gpu_priority is None:
        gpu_priority = ["cuda", "rocm", "mps", "cpu"]
    
    # Normalize priority list
    priority = [p.lower() for p in gpu_priority]
    
    # Try PyTorch first
    try:
        import torch
        for backend in priority:
            if backend == "cuda":
                if torch.cuda.is_available() and getattr(torch.version, "hip", None) is None:
                    free_vram = None
                    try:
                        if hasattr(torch.cuda, "mem_get_info"):
                            free_vram, _ = torch.cuda.mem_get_info(0)
                        else:
                            free_vram = _check_nvidia_smi_vram()
                    except Exception:
                        free_vram = _check_nvidia_smi_vram()
                    
                    if free_vram is not None:
                        free_vram_mb = free_vram / (1024 * 1024)
                        if free_vram_mb >= vram_required_mb:
                            # Check TensorRT only if explicitly installed
                            actual_backend = "cuda"
                            if free_vram_mb > 6000:
                                try:
                                    import torch_tensorrt  # noqa: F401
                                    actual_backend = "trt"
                                except ImportError:
                                    actual_backend = "cuda"
                            return True, actual_backend
                        else:
                            continue
            elif backend == "rocm":
                if torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None:
                    try:
                        free_vram, _ = torch.cuda.mem_get_info(0)
                        free_vram_mb = free_vram / (1024 * 1024)
                        if free_vram_mb >= vram_required_mb:
                            return True, "rocm"
                    except Exception:
                        pass
            elif backend == "mps":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    # MPS has no direct memory query; assume sufficient if available
                    return True, "mps"
            elif backend == "cpu":
                return False, "cpu"
    except ImportError:
        pass
    
    # Fallback: nvidia-smi without PyTorch (only if cuda is in priority)
    if "cuda" in priority:
        free_vram = _check_nvidia_smi_vram()
        if free_vram is not None:
            free_vram_mb = free_vram / (1024 * 1024)
            if free_vram_mb >= vram_required_mb:
                return True, "cuda"
    
    return False, "cpu"


def is_retryable_error(exit_code: int, stderr: str = "") -> bool:
    """Check if error is retryable (transient).

    Per v6.3 §3.7 / v7.3 §4.5:
    - 82 (IO), 83 (GPU/OOM), 87 (resource/timeout) are retryable
    - 84 (encoding) is retryable ONLY if stderr indicates OOM / no space
    - 80 (deps), 81 (params), 85 (perms), 86 (verify) are NOT retryable
    - Cross-platform signal codes (Linux/Windows) are also retryable
    """
    if exit_code in (82, 83, 87):
        return True
    if exit_code == 84 and stderr:
        stderr_lower = stderr.lower()
        if any(k in stderr_lower for k in ("out of memory", "oom", "no space",
                                            "errno 28", "cannot allocate")):
            return True
    
    # Cross-platform signal codes
    sys_platform = platform.system()
    if sys_platform in ("Linux", "Darwin"):
        # SIGSEGV=139 (128+11), SIGKILL=137 (128+9), SIGABRT=134 (128+6)
        if exit_code in (134, 137, 139):
            return True
    elif sys_platform == "Windows":
        # 0xC0000005 = -1073741819 (access violation)
        # 0xC000013A = -1073741510 (process terminated)
        # 0xE0434352 = -532462766 (CLR exception)
        if exit_code in (-1073741819, -1073741510, -532462766):
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


def wsl_redirect_path(output_path: str, temp_dir: Optional[str] = None,
                         no_wsl_redirect: bool = False,
                         platform: str = "auto") -> Tuple[str, Optional[str], str, Optional[str]]:
    """Redirect output/temp paths for WSL2 compatibility.
    
    Per v6.32 §7.3 / v7.32 §3.10.10: When output_path OR temp_dir is under /mnt/
    (Windows mount), redirect to Linux native partition to avoid WSL2 I/O
    performance bottleneck.
    
    Returns (redirected_output, original_output_or_None,
             redirected_temp, original_temp_or_None).
    If redirect is active, original paths are saved so caller
    can copy the result back after processing.
    """
    if no_wsl_redirect:
        return output_path, None, temp_dir or "", None
    if platform not in ("wsl2", "auto"):
        return output_path, None, temp_dir or "", None
    if not is_wsl():
        return output_path, None, temp_dir or "", None
    
    def _redirect_single(path: Optional[str]) -> Tuple[str, Optional[str]]:
        if not path or not path.startswith("/mnt/"):
            return path or "", None
        parts = path.split("/")
        if len(parts) >= 3 and parts[2]:
            drive = parts[2]
            rel_path = "/".join(parts[3:])
            home = os.environ.get("HOME", "/tmp")
            redirected = f"{home}/wsl_redirect/{drive}/{rel_path}"
            return redirected, path
        return path or "", None
    
    out_redirected, out_original = _redirect_single(output_path)
    temp_redirected, temp_original = _redirect_single(temp_dir)
    
    return out_redirected, out_original, temp_redirected, temp_original


def wsl_copy_back(redirected_path: str, original_path: str,
                  logger=None) -> bool:
    """Copy result from WSL redirected path back to original Linux path.
    
    Per v6.32 §7.3:
    1. Use shutil.copyfile (not copy2) to avoid PermissionError on NTFS mounts
    2. Manual utime sync
    3. MD5 + size + ffprobe decodeable triple verification
    4. Atomic rename on same filesystem, copyfile on cross-filesystem
    5. Up to 3 retries
    6. Keep source file if verification fails
    """
    import shutil
    
    if logger is None:
        from logger import get_logger
        logger = get_logger()
    
    def _md5(path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    
    def _copy_and_verify(src: str, dst: str) -> bool:
        try:
            shutil.copyfile(src, dst)
            # Sync mtime
            try:
                st = os.stat(src)
                os.utime(dst, (st.st_atime, st.st_mtime))
            except OSError:
                pass
            # Triple verification
            if os.path.getsize(src) != os.path.getsize(dst):
                return False
            if _md5(src) != _md5(dst):
                return False
            if not verify_ffprobe_decodeable(dst):
                # ffprobe decodeable check may fail for photos; skip for non-video
                ext = Path(dst).suffix.lower()
                if ext in (".mkv", ".mp4", ".avi", ".mov", ".mxf"):
                    return False
            return True
        except Exception:
            return False
    
    tmp_path = original_path + ".tmp"
    
    for attempt in range(1, 4):
        try:
            os.makedirs(os.path.dirname(original_path) or ".", exist_ok=True)
            
            # Step 1: copy to .tmp
            if not _copy_and_verify(redirected_path, tmp_path):
                logger.warning("WSL_COPY_VERIFY_FAIL",
                             f"Copy verification failed (attempt {attempt})")
                continue
            
            # Step 2: atomic move or overwrite
            src_dev = os.stat(tmp_path).st_dev
            dst_dev = os.stat(os.path.dirname(original_path) or ".").st_dev
            
            if src_dev == dst_dev:
                # Same filesystem: atomic rename
                os.replace(tmp_path, original_path)
            else:
                # Cross filesystem: copyfile directly (rename would fail)
                # Per v6.32 §7.3: backup old file as .bak before overwriting
                if os.path.exists(original_path):
                    bak_path = original_path + ".bak"
                    try:
                        shutil.copyfile(original_path, bak_path)
                        os.utime(bak_path, (os.stat(original_path).st_atime,
                                            os.stat(original_path).st_mtime))
                    except OSError:
                        pass
                if not _copy_and_verify(tmp_path, original_path):
                    logger.warning("WSL_COPY_CROSSFS_FAIL",
                                 f"Cross-filesystem copy failed (attempt {attempt})")
                    continue
                os.unlink(tmp_path)
            
            logger.info("WSL_COPY_BACK", "Successfully copied result back",
                       original=original_path, redirected=redirected_path)
            return True
        except Exception as e:
            logger.warning("WSL_COPY_BACK_ERROR",
                         f"Copy back error (attempt {attempt}): {e}")
    
    # All retries failed: keep source, mark FAILED
    logger.error("WSL_COPY_BACK_FAIL",
                f"Failed to copy back after 3 attempts. Source preserved at {redirected_path}")
    return False


def estimate_disk_space(duration: float = 300,
                        fps: float = 29.97,
                        width: int = 720,
                        height: int = 480,
                        bit_depth: int = 8,
                        target_bitrate_mbps: float = 10.0,
                        audio_bitrate_kbps: float = 320.0,
                        file_path: Optional[str] = None,
                        is_photo: bool = False) -> int:
    """
    Estimate peak disk space in MB needed for restoration pipeline.
    
    Per v6.32 §7.3:
    - Method 1: width * height * bit_depth/8 * fps * duration * 1.5 (frame sequence)
    - Method 2: duration * target_bitrate_mbps * 1.5 + duration * audio_bitrate_kbps / 8
    - Uses max of both methods
    """
    if is_photo and file_path and os.path.exists(file_path):
        # Use actual file size as base for photos
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        # Photo processing typically produces intermediates 3-5x original size
        # Max intermediate count: ~5 stages
        peak = file_size_mb * 5 * 5  # 5 stages, 5x each
    else:
        # Video estimation
        if duration <= 0:
            duration = 300  # Default 5 minutes
        if fps <= 0:
            fps = 29.97
        
        # Method 1: uncompressed frame sequence (float division for bit_depth)
        bytes_per_pixel = bit_depth / 8.0
        uncompressed_frame_mb = width * height * bytes_per_pixel / (1024 * 1024)
        total_frames = duration * fps
        method1 = uncompressed_frame_mb * total_frames * 1.5
        
        # Method 2: bitrate-based estimation (float division)
        method2 = (duration * target_bitrate_mbps * 1.5 * 1024 * 1024 / 8 +
                   duration * audio_bitrate_kbps * 1024 / 8) / (1024 * 1024)
        
        peak = max(method1, method2)
    
    # 2x safety margin
    return int(peak * 2)
