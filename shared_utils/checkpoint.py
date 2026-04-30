#!/usr/bin/env python3
"""
Checkpoint Module - Breakpoint Resume Support
Handles intermediate file naming, param_hash computation, integrity verification,
and automatic cleanup of old intermediate files.
"""

import os
import hashlib
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np


# Quality-related parameters that affect param_hash
CORE_QUALITY_PARAMS = {
    "degrainTR", "grainLevel", "qtgmc_ezdenoise",
    "deblock_quant", "deblock_strength",
    "qtgmc_preset", "qtgmc_sharpness", "qtgmc_chromamotion",
    "qtgmc_sourcematch", "qtgmc_fps_divisor",
    "ivtc", "VFM_mode", "VDecimate_cycle",
    "matrix_in_s", "matrix_s",
    "upscale_scale",
    "upscale_model", "face_restore",
    "g1str", "g2str", "g3str", "grain_strength",
    "vs_bit_depth", "pix_fmt",
    "crf", "preset",
    "face_detect_threshold", "face_detect_threshold_lowres",
    "static_segment_duration",
    # v6.0 additions
    "crop_left", "crop_right", "crop_top", "crop_bottom",
    "chroma_shift",
    "dxmax", "dymax",
    "hum_freq", "denoise",
    "ivtc_fallback_threshold",
}


def get_file_xxhash(path: str, max_size: int = 1024 * 1024 * 1024) -> str:
    """Compute xxhash64 of file. For files > 1GB, use block sampling."""
    try:
        import xxhash
    except ImportError:
        # Fallback to MD5 if xxhash not installed
        return get_file_md5_prefix(path)

    p = Path(path)
    size = p.stat().st_size

    if size <= max_size:
        # Full file hash
        h = xxhash.xxh64()
        with open(p, "rb") as f:
            while True:
                chunk = f.read(8 * 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()[:16]
    else:
        # Block sampling: first 1MB + middle 1MB + last 1MB
        h = xxhash.xxh64()
        # First 1MB
        with open(p, "rb") as f:
            h.update(f.read(1024 * 1024))
        # Middle 1MB
        mid = size // 2 - 512 * 1024
        with open(p, "rb") as f:
            f.seek(mid)
            h.update(f.read(1024 * 1024))
        # Last 1MB
        with open(p, "rb") as f:
            f.seek(size - 1024 * 1024)
            h.update(f.read(1024 * 1024))
        return h.hexdigest()[:16]


def get_file_md5_prefix(path: str) -> str:
    """Compute MD5 hash prefix of file (fallback when xxhash unavailable)."""
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


def compute_file_identity(input_path: str) -> Dict[str, Any]:
    """Compute input file unique identity for param_hash."""
    p = Path(input_path)
    try:
        stat = p.stat()
        file_size = stat.st_size
        file_mtime = int(stat.st_mtime)
    except (OSError, FileNotFoundError):
        # File may not exist in test scenarios; return safe defaults
        file_size = 0
        file_mtime = 0

    try:
        if p.exists():
            file_hash = get_file_xxhash(input_path)
        else:
            file_hash = "0000000000000000"
    except Exception:
        file_hash = get_file_md5_prefix(input_path) if p.exists() else "0000000000000000"

    return {
        "file_size": file_size,
        "file_mtime": file_mtime,
        "file_hash_prefix": file_hash,
    }


def compute_param_hash(input_path: str,
                       params: Dict[str, Any]) -> str:
    """Compute SHA-256 based param_hash (first 16 chars).

    Incorporates:
    - Input file identity (size, mtime, hash)
    - Core quality parameters only
    - Excludes: log_level, output_path, temp_path, job_id, etc.

    NOTE: job_id is intentionally excluded from param_hash so that
    the same input file + same parameters can reuse intermediates
    even if the manifest is regenerated (which changes job_id).
    Filename uniqueness is ensured by build_intermediate_name().
    """
    file_id = compute_file_identity(input_path)

    # Filter to core quality params only
    core_params = {}
    for k, v in params.items():
        if k in CORE_QUALITY_PARAMS:
            core_params[k] = v

    # Build normalized string (job_id intentionally excluded)
    parts = [
        f"file_size={file_id['file_size']}",
        f"file_mtime={file_id['file_mtime']}",
        f"file_hash={file_id['file_hash_prefix']}",
    ]
    # Sort params for consistency
    for k in sorted(core_params.keys()):
        parts.append(f"{k}={core_params[k]}")

    param_str = "|".join(parts)
    h = hashlib.sha256(param_str.encode("utf-8")).hexdigest()[:16]
    return h


def build_intermediate_name(job_id: str, input_name: str, step: int,
                            operation: str, param_hash: str,
                            ext: str = "mkv") -> str:
    """Build intermediate file name: {job_id}_{input_name}_step{N}_{operation}_p{param_hash}.{ext}"""
    # Sanitize input_name (remove path, keep stem)
    safe_name = Path(input_name).stem
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-.")
    return f"{job_id}_{safe_name}_step{step}_{operation}_p{param_hash}.{ext}"


def find_existing_intermediate(temp_dir: str, job_id: str, input_name: str,
                               step: int, operation: str,
                               param_hash: str) -> Optional[Path]:
    """Find existing intermediate file matching the param_hash."""
    temp_path = Path(temp_dir)
    safe_name = Path(input_name).stem
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-.")
    pattern = f"{job_id}_{safe_name}_step{step}_{operation}_p{param_hash}.*"

    for f in temp_path.glob(pattern):
        if f.is_file():
            return f
    return None


def verify_ffprobe_decodeable(path: str) -> bool:
    """Verify file is decodeable by ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1",
             path],
            capture_output=True, text=True, timeout=60, shell=False
        )
        return r.returncode == 0 and "video" in r.stdout
    except Exception:
        return False


def verify_frame_integrity(path: str, duration: float = 0.0,
                           bit_depth: int = 10,
                           color_range: str = "limited") -> bool:
    """Verify frame data integrity by sampling frames at multiple positions.

    Samples: 0-5%, 45-55%, 95-100% of duration.
    For videos > 10min, samples every 10min.
    Validates: decode success, non-zero/non-max pixels, luma mean in range.
    """
    if not verify_ffprobe_decodeable(path):
        return False

    # Auto-detect duration if not provided
    if duration <= 0:
        try:
            r = subprocess.run(
                ["ffprobe", "-hide_banner", "-v", "error",
                 "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=30, shell=False
            )
            if r.returncode == 0:
                duration = float(r.stdout.strip())
        except Exception:
            duration = 300  # fallback: 5 minutes

    # Define sample positions
    positions = [0.025, 0.5, 0.975]  # start, middle, end
    if duration > 600:  # > 10 minutes
        extra = []
        for t in range(600, int(duration), 600):
            extra.append(t / duration)
        positions.extend(extra)

    # Luma range based on bit depth and range
    if color_range == "limited":
        if bit_depth == 8:
            luma_min, luma_max = 10, 245
        elif bit_depth == 10:
            luma_min, luma_max = 40, 980
        elif bit_depth == 16:
            luma_min, luma_max = 4096, 60160
        else:
            luma_min, luma_max = 10, 245
    else:  # full
        luma_max = (1 << bit_depth) - 1
        luma_min, luma_max = 0, luma_max

    for pos in positions:
        try:
            # Extract 1 frame at position using ffmpeg
            tmp_frame = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_frame.close()

            proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-v", "error",
                 "-ss", str(pos * duration) if duration > 0 else "0",
                 "-i", path,
                 "-vf", "select=eq(n\\,0)",
                 "-frames:v", "1",
                 "-f", "image2", tmp_frame.name],
                capture_output=True, timeout=60, shell=False
            )

            if proc.returncode != 0 or not os.path.exists(tmp_frame.name):
                os.unlink(tmp_frame.name)
                return False

            # Validate pixel data
            from PIL import Image
            img = Image.open(tmp_frame.name).convert("L")
            arr = np.array(img)

            # Check not all zeros or all max
            if arr.max() == 0 or arr.min() == arr.max():
                os.unlink(tmp_frame.name)
                return False

            # Check luma mean in valid range
            mean_luma = arr.mean()
            if not (luma_min <= mean_luma <= luma_max):
                os.unlink(tmp_frame.name)
                return False

            os.unlink(tmp_frame.name)

        except Exception:
            return False

    return True


def check_and_reuse(temp_dir: str, job_id: str, input_name: str,
                    step: int, operation: str, param_hash: str,
                    duration: float = 0.0) -> Optional[Path]:
    """Check if valid intermediate exists. Returns path if valid, None otherwise."""
    existing = find_existing_intermediate(
        temp_dir, job_id, input_name, step, operation, param_hash
    )
    if existing is None:
        return None

    # Verify decodeable
    if not verify_ffprobe_decodeable(str(existing)):
        existing.unlink(missing_ok=True)
        return None

    return existing


def cleanup_old_intermediates(temp_dir: str, job_id: str, input_name: str,
                              keep_steps: int = 2):
    """Cleanup old intermediate files, keeping the most recent N stages.

    Scans for all intermediates matching job_id + input_name,
    sorts by step number, and deletes files from steps older than
    the most recent `keep_steps` stages.
    """
    temp_path = Path(temp_dir)
    safe_name = Path(input_name).stem
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-.")
    pattern = f"{job_id}_{safe_name}_step*"

    files = []
    for f in temp_path.glob(pattern):
        if f.is_file():
            # Extract step number from filename
            try:
                stem = f.stem
                step_part = [p for p in stem.split("_") if p.startswith("step")]
                if step_part:
                    step_num = int(step_part[0].replace("step", ""))
                    files.append((step_num, f))
            except ValueError:
                continue

    if not files:
        return

    files.sort(key=lambda x: x[0])
    max_step = files[-1][0]

    # Keep files from max_step, max_step-1, ..., up to keep_steps
    cutoff = max_step - keep_steps

    for step_num, f in files:
        if step_num <= cutoff:
            try:
                f.unlink()
            except OSError:
                pass


def wsl_redirect_path(original_path: str, temp_dir: str,
                      no_wsl_redirect: bool = False,
                      platform: str = "auto") -> str:
    """Redirect /mnt/ paths to Linux native partition in WSL2.

    Returns the redirected path. If not WSL or redirect disabled, returns original.
    """
    if no_wsl_redirect:
        return original_path

    # Check platform parameter first
    if platform not in ("wsl2", "auto"):
        return original_path

    # Detect WSL
    if not Path("/proc/sys/kernel/osrelease").exists():
        return original_path

    try:
        with open("/proc/sys/kernel/osrelease", "r") as f:
            osrel = f.read()
            if "microsoft" not in osrel.lower() and "wsl" not in osrel.lower():
                return original_path
    except Exception:
        return original_path

    # Redirect /mnt/ paths
    p = Path(original_path).resolve()
    p_str = str(p)
    if p_str.startswith("/mnt/"):
        # Use temp_dir as base for redirected path
        redirected = Path(temp_dir) / "wsl_redirect" / p.name
        redirected.parent.mkdir(parents=True, exist_ok=True)
        return str(redirected)

    return original_path


def wsl_copy_back(src_path: str, dst_path: str, max_retries: int = 3) -> bool:
    """Copy processed file back to /mnt/ path in WSL2.

    Uses shutil.copyfile (not copy2) to avoid Linux extended metadata issues.
    Performs MD5 + size + ffprobe verification before atomic rename.
    """
    import hashlib

    src = Path(src_path)
    dst = Path(dst_path)

    if not src.exists():
        return False

    # Compute source MD5
    src_md5 = hashlib.md5()
    with open(src, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            src_md5.update(chunk)
    src_hash = src_md5.hexdigest()
    src_size = src.stat().st_size

    for attempt in range(max_retries):
        try:
            # Step 1: Copy to temp file
            tmp_dst = dst.parent / (dst.name + ".tmp")
            shutil.copyfile(str(src), str(tmp_dst))

            # Sync timestamps
            src_stat = src.stat()
            os.utime(str(tmp_dst), (src_stat.st_atime, src_stat.st_mtime))

            # Step 2: Verify
            dst_md5 = hashlib.md5()
            with open(tmp_dst, "rb") as f:
                while True:
                    chunk = f.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dst_md5.update(chunk)
            dst_hash = dst_md5.hexdigest()
            dst_size = tmp_dst.stat().st_size

            if dst_hash != src_hash or dst_size != src_size:
                tmp_dst.unlink(missing_ok=True)
                continue

            # ffprobe decodeable check
            if not verify_ffprobe_decodeable(str(tmp_dst)):
                tmp_dst.unlink(missing_ok=True)
                continue

            # Step 3: Atomic rename or copy
            src_dev = os.stat(str(tmp_dst)).st_dev
            try:
                dst_dev = os.stat(str(dst.parent)).st_dev
            except FileNotFoundError:
                dst_dev = src_dev  # Target dir doesn't exist yet

            if src_dev == dst_dev:
                os.replace(str(tmp_dst), str(dst))
            else:
                # Cross-device: backup old file, then copy
                if dst.exists():
                    bak = dst.parent / (dst.name + ".bak")
                    shutil.copyfile(str(dst), str(bak))
                shutil.copyfile(str(tmp_dst), str(dst))
                tmp_dst.unlink(missing_ok=True)

            return True

        except Exception:
            if attempt == max_retries - 1:
                return False
            import time
            time.sleep(2 ** attempt)

    return False


def estimate_disk_space(job: Dict[str, Any], config: Dict[str, Any]) -> int:
    """Estimate required disk space in bytes.

    Returns the larger of:
    - width * height * bit_depth/8 * fps * duration * 1.5
    - duration * target_bitrate * 1.5 + audio_duration * audio_bitrate / 8
    - Heuristic: source_size * factor (10/25/50 based on AI upscale)
    """
    input_path = job.get("input_path", "")
    try:
        src_size = Path(input_path).stat().st_size
    except Exception:
        src_size = 0

    # Get dimensions and duration
    width = job.get("original_width", 720)
    height = job.get("original_height", 480)
    duration = job.get("duration", 300)  # default 5 min

    # Bit depth
    bit_depth = config.get("vs_bit_depth", 16)

    # FPS (approximate)
    fps = 30

    # Method 1: Frame-based estimation
    method1 = int(width * height * (bit_depth / 8) * fps * int(duration) * 1.5)

    # Method 2: Bitrate-based estimation
    ai_config = config.get("ai", {})
    upscale_scale = ai_config.get("upscale_scale", 1)
    target_bitrate_mbps = 15 if upscale_scale >= 2 else 8
    audio_bitrate_kbps = 320
    method2 = int(int(duration) * target_bitrate_mbps * 1.5 * 1024 * 1024 / 8 +
                  int(duration) * audio_bitrate_kbps * 1024 / 8)

    # Method 3: Heuristic based on source size
    if upscale_scale >= 4:
        factor = 50
    elif upscale_scale >= 2:
        factor = 25
    else:
        factor = 10
    method3 = src_size * factor

    return int(max(method1, method2, method3))


def gpu_memory_check(required_bytes: int) -> Tuple[bool, str]:
    """Check if GPU has enough free memory.

    Tries: nvidia-smi (with unit conversion) -> torch.cuda.mem_get_info() -> pynvml
    Returns (sufficient, backend_name)
    """
    # Try nvidia-smi first
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, shell=False
        )
        if r.returncode == 0:
            free_mib = float(r.stdout.strip().split("\n")[0].strip())
            free_bytes = int(free_mib * 1024 * 1024)
            if free_bytes >= required_bytes * 1.2:
                return True, "cuda"
    except Exception:
        pass

    # Try torch.cuda
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            if free_bytes >= required_bytes * 1.2:
                return True, "cuda"
    except Exception:
        pass

    # Try pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if info.free >= required_bytes * 1.2:
            return True, "cuda"
    except Exception:
        pass

    # Try ROCm (AMD GPU via torch HIP)
    try:
        import torch
        if torch.cuda.is_available() and hasattr(torch.version, "hip") and torch.version.hip:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            if free_bytes >= required_bytes * 1.2:
                return True, "rocm"
    except Exception:
        pass

    # Try MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS does not expose free memory via torch API; assume available
            # if MPS is available and required_bytes is modest (< 8GB heuristic)
            if required_bytes < 8 * 1024 * 1024 * 1024:
                return True, "mps"
    except Exception:
        pass

    return False, "cpu"


def is_retryable_error(exit_code: int, stderr: str = "") -> bool:
    """Determine if an error is retryable based on exit code and stderr.

    Retryable: GPU OOM, network timeout, ffprobe IO error, VS plugin timeout
    Not retryable: parameter error, dependency missing, file not found,
                   permission denied, disk full (errno 28)
    """
    import platform
    system = platform.system()
    stderr_lower = stderr.lower()

    # Check stderr content FIRST (overrides exit code classification)
    if "no space" in stderr_lower or "errno 28" in stderr_lower:
        return False  # Disk full is NEVER retryable
    if "permission denied" in stderr_lower:
        return False
    if "not found" in stderr_lower or "no such file" in stderr_lower:
        return False
    if "out of memory" in stderr_lower:
        return True

    # By exit code
    if exit_code in (80, 81, 84, 85, 86):  # dep missing, param error, encoding error, permission, verify fail
        return False
    if exit_code == 87:  # resource不足
        return True
    if exit_code == 83:  # GPU error - retryable with降级
        return True
    if exit_code == 82:  # IO error - may be temporary
        return True

    # By platform-specific signals
    if system in ("Linux", "Darwin"):
        if exit_code in (137, 139, 134):  # SIGKILL, SIGSEGV, SIGABRT
            return True
    elif system == "Windows":
        if exit_code in (-1073741819, -1073741510, -532462766):
            return True

    return False
