#!/usr/bin/env python3
"""
Stage 3: AI Upscaling
Performs integer-scale AI super-resolution (2x or 4x only).
GPU memory pre-check with automatic fallback (4x->2x->CPU).

v6.3 Compliance:
- 256-frame sliding window
- VFR (Variable Frame Rate) support via PTS
- AI upscale banned for 720x480 DV
- Proper Real-ESRGAN CLI invocation (ncnn-vulkan or Python API)

P8 Fixes Applied:
- P8-1: Original input dims in dual-basis ban per v7.32 §3.10.2
- P8-2: Dual-basis three-layer ban check per v7.32 §3.10.2
- P8-3: Force override WARNING log per v7.32 §3.10.2 L3
- P8-4: Exit code 1 -> 83 (GPU error, per v7.32 §3.5)
- P8-5: WSL2 temp_dir/preview_dir redirect per v6.32 §7.3
- P8-6: ncnn-vulkan tried first, .pth fallback only
- P8-7: OOM downgrade only on actual OOM (three-state return)
- P8-8: VFR PTS preservation + bitdepth self-check
- P8-9: Batch model loading for Python API path
- P8-10: pathlib.Path consistent usage
- P8-11: ncnn-vulkan OOM detection from stderr per v6.32 §7.14
- P8-12: vs_bit_depth in param_hash core_params per v7.32 §3.10.19
- P8-13: pts_time first (FFmpeg>=7.0), fallback pkt_pts_time
- P8-14: VFR seek uses actual PTS, not CFR assumption
- P8-15: Enforce checkpoint.compute_param_hash (no local fallback)
- P8-16: MPS unified memory check per v6.32 §7.14
"""

import os
import sys
import json
import argparse
import subprocess
import re
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Optional, Union

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402


def estimate_gpu_memory(width: int, height: int, scale: int,
                        model_name: str) -> int:
    """Estimate GPU memory requirement in bytes."""
    base = width * height * 3 * 4 * 1 * (2 + scale * scale)
    model_weights = {
        "realesr-general-x4v3": 18 * 1024 * 1024,   # ncnn .bin ~10-18MB
        "RealESRGAN_x2plus": 18 * 1024 * 1024,       # ncnn .bin
        "RealESRGAN_x4plus": 18 * 1024 * 1024,       # ncnn .bin
        "DAT_x2": 18 * 1024 * 1024,                   # ncnn .bin
        "DAT_x4": 18 * 1024 * 1024,                   # ncnn .bin
        "DAT_2_x2": 18 * 1024 * 1024,
        "DAT_2_x4": 18 * 1024 * 1024,
    }
    weight = model_weights.get(model_name, 64 * 1024 * 1024)
    return base + weight


def check_gpu_memory(required_bytes: int, logger=None) -> tuple:
    """Check GPU availability. Returns (sufficient: bool, backend: str).

    Backend priority: cuda > rocm > mps > cpu.
    Bug 13.30/13.31 fix: Detect PyTorch version, ROCm, and MPS.
    """
    # 1. nvidia-smi (native CUDA)
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

    # 2. PyTorch backends (cuda / rocm / mps)
    try:
        import torch
        torch_ver = torch.__version__

        # Parse version for >=1.10 check (P8-4)
        ver_parts = torch_ver.split("+")[0].split(".")
        ver_major = int(ver_parts[0])
        ver_minor = int(ver_parts[1]) if len(ver_parts) > 1 else 0
        has_mem_get_info = (ver_major > 1) or (ver_major == 1 and ver_minor >= 10)

        # CUDA
        if torch.cuda.is_available():
            if has_mem_get_info:
                try:
                    free, total = torch.cuda.mem_get_info()
                    if free >= required_bytes * 1.2:
                        return True, "cuda"
                except Exception:
                    pass
            else:
                if logger:
                    logger.warning("STAGE3_PYTORCH_OLD",
                                   f"PyTorch {torch_ver} < 1.10, "
                                   "skipping torch.cuda.mem_get_info()")

        # ROCm (AMD GPU)
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            if has_mem_get_info:
                try:
                    free, total = torch.cuda.mem_get_info()
                    if free >= required_bytes * 1.2:
                        return True, "rocm"
                except Exception:
                    pass

        # Apple Metal (MPS)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # FIX-P8-11: Check unified memory availability for MPS
            # ORIGINAL: Assumed sufficient without any memory check
            # Per v6.32 §7.14: must verify available memory >= required * 1.2
            try:
                import os as _os
                if hasattr(_os, 'sysconf'):
                    total_mem = (_os.sysconf('SC_PAGE_SIZE')
                                 * _os.sysconf('SC_PHYS_PAGES'))
                    if total_mem * 0.7 >= required_bytes * 1.2:
                        return True, "mps"
                else:
                    # Windows/other: assume sufficient (no API available)
                    return True, "mps"
            except Exception:
                pass
            return False, "cpu"

        # CPU-only PyTorch detection
        if "cpu" in torch_ver.lower():
            return False, "cpu"
    except Exception:
        pass

    return False, "cpu"


def _get_model_for_scale(model: str, scale: int) -> str:
    """Get correct model name for a given scale, preserving model family.

    FIX-P1-15: realesr-general-x4v3 is a 4× ncnn model; for 2× output it
    stays the same model (ncnn -s 2 downscales). DAT_2 is default.
    """
    if model.startswith("realesr-general"):
        return "realesr-general-x4v3"
    elif model.startswith("RealESRGAN"):
        return f"RealESRGAN_x{scale}plus"
    elif model.startswith("DAT_2"):
        return f"DAT_2_x{scale}"
    elif model.startswith("DAT"):
        return f"DAT_x{scale}"
    return "realesr-general-x4v3"


# FIX-P8-6+P8-1: Restructured run_realesrgan
# ORIGINAL: Checked .pth first (blocking ncnn-vulkan), returned bool,
#           had broken OOM cleanup in finally block (del model on string)
# DEPRECATED: Use run_realesrgan_batch instead.
# This single-window function is kept for unit testing only.
def run_realesrgan(input_path: str, output_dir: str, model: str,
                   scale: int, gpu_id: int = 0) -> Union[bool, str]:
    """Run Real-ESRGAN upscaling. Returns True/False/"OOM".

    FIX-P8-6: ncnn-vulkan tried first (doesn't need .pth), then Python API.
    FIX-P8-1: OOM detection via child process exit code 83 per v6.32 §7.14.
    """
    model_dir = SCRIPT_DIR.parent.resolve() / "models"

    # FIX-P8-6: 1. Try realesrgan-ncnn-vulkan CLI first (doesn't need .pth)
    vulkan_bin = shutil.which("realesrgan-ncnn-vulkan")
    if vulkan_bin:
        cmd = [
            vulkan_bin,
            "-i", input_path,
            "-o", output_dir,
            "-n", model,
            "-s", str(scale),
            "-g", str(gpu_id),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=3600, shell=False)
            if r.returncode == 0:
                return True
        except Exception:
            pass

    # FIX-P8-6: 2. Python API fallback - needs .pth
    model_path = str(model_dir / f"{model}.pth")
    if not Path(model_path).exists():
        return False

    try:
        env = os.environ.copy()
        if gpu_id < 0:
            env["CUDA_VISIBLE_DEVICES"] = ""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                          delete=False) as f:
            script_path = f.name
            # FIX-P8-1: Temp script with OOM detection (exit code 83)
            f.write(
                "import sys\n"
                "from realesrgan import RealESRGANer\n"
                "import torch\n"
                "from PIL import Image\n"
                "import os\n"
                "import glob\n"
                "import numpy as np\n"
                "import gc\n"
                f"scale = {scale}\n"
                f"model_path = {repr(model_path)}\n"
                f"input_path = {repr(input_path)}\n"
                f"output_dir = {repr(output_dir)}\n"
                f"use_half = not torch.cuda.is_available() "
                f"if {gpu_id} >= 0 else False\n"
                "os.makedirs(output_dir, exist_ok=True)\n"
                "try:\n"
                "    upsampler = RealESRGANer(scale=scale, "
                "model_path=model_path, prepad=10, half=use_half)\n"
                "    for fp in sorted(glob.glob("
                "os.path.join(input_path, '*.png'))):\n"
                "        img = Image.open(fp).convert('RGB')\n"
                "        output, _ = upsampler.enhance(np.array(img))\n"
                "        Image.fromarray(output).save("
                "os.path.join(output_dir, os.path.basename(fp)))\n"
                "    del upsampler\n"
                "    gc.collect()\n"
                "    if torch.cuda.is_available():\n"
                "        torch.cuda.empty_cache()\n"
                "except RuntimeError as e:\n"
                "    if 'out of memory' in str(e).lower():\n"
                "        try:\n"
                "            del upsampler\n"
                "        except NameError:\n"
                "            pass\n"
                "        gc.collect()\n"
                "        if torch.cuda.is_available():\n"
                "            torch.cuda.empty_cache()\n"
                "        sys.exit(83)  # FIX-P8-1: GPU OOM per v6.32 §7.14\n"
                "    raise\n"
            )
        cmd = [sys.executable, script_path]
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=3600, shell=False, env=env)
        try:
            # FIX-P8-10: Use Path.unlink instead of os.unlink
            Path(script_path).unlink(missing_ok=True)
        except Exception:
            pass
        # FIX-P8-1: Check OOM exit code
        if r.returncode == 83:
            return "OOM"
        return r.returncode == 0
    except Exception:
        return False


# FIX-P8-9: Batch model loading for Python API path
# ORIGINAL: Per-window model loading via run_realesrgan (slow for Python API)
def run_realesrgan_batch(window_dirs: List[Tuple[str, str]], model: str,
                         scale: int, gpu_id: int = 0,
                         logger=None) -> Union[bool, str]:
    """Run Real-ESRGAN on all windows in a single model-load session.

    FIX-P8-9: Loads model once for Python API path, processing all windows.
    ncnn-vulkan still per-window (loads fast natively).
    Returns True/False/"OOM".
    """
    model_dir = SCRIPT_DIR.parent.resolve() / "models"

    # ncnn-vulkan: per-window (ncnn loads fast)
    vulkan_bin = shutil.which("realesrgan-ncnn-vulkan")
    if vulkan_bin:
        all_ok = True
        got_oom = False
        for frames_dir, out_dir in window_dirs:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            cmd = [vulkan_bin, "-i", frames_dir, "-o", out_dir,
                   "-n", model, "-s", str(scale), "-g", str(gpu_id)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=3600, shell=False)
                if r.returncode != 0:
                    all_ok = False
                    # FIX-P8-3: Detect ncnn-vulkan OOM from stderr
                    # ORIGINAL: Only returned bool, OOM downgrade never triggered
                    # Per v6.32 §7.14: OOM must trigger 4x→2x→CPU downgrade
                    stderr_lower = (r.stderr or "").lower()
                    if ("out of memory" in stderr_lower or
                        ("cuda" in stderr_lower and "error" in stderr_lower) or
                        ("vulkan" in stderr_lower and "memory" in stderr_lower)):
                        got_oom = True
            except Exception:
                all_ok = False
        if got_oom:
            return "OOM"
        return all_ok

    # Python API: single load, process all windows
    model_path = str(model_dir / f"{model}.pth")
    if not Path(model_path).exists():
        return False

    try:
        env = os.environ.copy()
        if gpu_id < 0:
            env["CUDA_VISIBLE_DEVICES"] = ""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                          delete=False) as f:
            script_path = f.name
            window_repr = repr([(d, o) for d, o in window_dirs])
            f.write(
                "import sys\n"
                "from realesrgan import RealESRGANer\n"
                "import torch\n"
                "from PIL import Image\n"
                "import os\n"
                "import glob\n"
                "import numpy as np\n"
                "import gc\n"
                f"model_path = {repr(model_path)}\n"
                f"scale = {scale}\n"
                f"use_half = not torch.cuda.is_available() "
                f"if {gpu_id} >= 0 else False\n"
                f"windows = {window_repr}\n"
                "# FIX-P8-9: Load model ONCE for all windows\n"
                "try:\n"
                "    upsampler = RealESRGANer(scale=scale, "
                "model_path=model_path, prepad=10, half=use_half)\n"
                "    for frames_dir, output_dir in windows:\n"
                "        os.makedirs(output_dir, exist_ok=True)\n"
                "        for fp in sorted(glob.glob("
                "os.path.join(frames_dir, '*.png'))):\n"
                "            img = Image.open(fp).convert('RGB')\n"
                "            output, _ = upsampler.enhance(np.array(img))\n"
                "            Image.fromarray(output).save("
                "os.path.join(output_dir, os.path.basename(fp)))\n"
                "    del upsampler\n"
                "    gc.collect()\n"
                "    if torch.cuda.is_available():\n"
                "        torch.cuda.empty_cache()\n"
                "except RuntimeError as e:\n"
                "    if 'out of memory' in str(e).lower():\n"
                "        try:\n"
                "            del upsampler\n"
                "        except NameError:\n"
                "            pass\n"
                "        gc.collect()\n"
                "        if torch.cuda.is_available():\n"
                "            torch.cuda.empty_cache()\n"
                "        sys.exit(83)  # FIX-P8-1: GPU OOM exit code\n"
                "    raise\n"
            )
        cmd = [sys.executable, script_path]
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=7200, shell=False, env=env)  # Longer timeout
        try:
            # FIX-P8-10: Use Path.unlink instead of os.unlink
            Path(script_path).unlink(missing_ok=True)
        except Exception:
            pass
        # FIX-P8-1: Check OOM exit code
        if r.returncode == 83:
            return "OOM"
        return r.returncode == 0
    except Exception:
        return False


def _query_all_pts(video_path: str, logger=None) -> dict:
    """Query PTS for ALL frames in a video, one-time.

    FIX-N8-2: Extracted from extract_frames_with_pts to avoid
    repeated full-video ffprobe calls per window.
    Returns {frame_index: pts_seconds}.
    """
    pts_map: dict = {}
    try:
        pts_cmd = [
            "ffprobe", "-hide_banner", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "frame=pts_time,pkt_pts_time",
            "-of", "json",
            video_path
        ]
        pts_proc = subprocess.run(pts_cmd, capture_output=True, text=True,
                                 timeout=300, shell=False)
        if pts_proc.returncode == 0:
            data = json.loads(pts_proc.stdout)
            for i, frame_info in enumerate(data.get("frames", [])):
                # FIX-P8-6: pts_time (FFmpeg>=7.0) first,
                # fallback to pkt_pts_time (FFmpeg<7.0)
                # ORIGINAL: Only used pkt_pts_time, fails on FFmpeg 7.0+
                pts_str = (frame_info.get("pts_time")
                           or frame_info.get("pkt_pts_time", ""))
                if pts_str:
                    try:
                        pts_map[i] = float(pts_str)
                    except ValueError:
                        pts_map[i] = i / 25.0
    except Exception:
        if logger:
            logger.warning("STAGE3_PTS_QUERY_FAIL",
                           "Failed to query PTS, using frame-index fallback")
    return pts_map


def extract_frames_with_pts(video_path: str, output_dir: str,
                              win_start: int = 0, win_size: int = 256,
                              logger=None, fps: float = 30.0,
                              pts_map: Optional[dict] = None
                            ) -> List[Tuple[str, float]]:
    """Extract frames in a sliding window with PTS support for VFR.

    FIX-N8-2: Accepts pre-built pts_map to avoid per-window ffprobe calls.
    Returns list of (frame_path, pts_seconds) ordered by PTS.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # V-P-033 fix: extract 16bit PNG for preservation
    # Use ffmpeg with -vsync 0 to preserve original timestamps
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", video_path,
    ]
    if win_start > 0:
        # FIX-P8-7: Use actual PTS for VFR seek per v6.32 §4B
        # ORIGINAL: Used CFR assumption (win_start / fps), wrong for VFR
        if pts_map and win_start in pts_map:
            seek_time = pts_map[win_start]
        else:
            seek_time = win_start / fps  # CFR fallback
        cmd.extend(["-ss", str(seek_time)])
    if win_size > 0:
        cmd.extend(["-frames:v", str(win_size)])
    cmd.extend([
        "-vsync", "0",
        "-frame_pts", "1",
        "-pix_fmt", "rgb48",
        str(Path(output_dir) / "frame_%08d.png")
    ])

    try:
        r = subprocess.run(cmd, capture_output=True, timeout=600, shell=False)
        if r.returncode != 0:
            if logger:
                logger.error("STAGE3_EXTRACT_FAIL",
                             f"Frame extract: {r.stderr.decode()[:200]}")
            return []
    except Exception as e:
        if logger:
            logger.error("STAGE3_EXTRACT_EXCEPT", str(e))
        return []

    # FIX-N8-2: Use pre-built pts_map instead of per-window ffprobe
    # Build ordered list by actual frame number
    frames: List[Tuple[str, float]] = []
    frame_files = sorted(Path(output_dir).glob("frame_*.png"))
    for i, fp in enumerate(frame_files):
        # Map window-local index to global index
        global_idx = win_start + i
        if pts_map:
            pts = pts_map.get(global_idx, global_idx / 25.0)
        else:
            pts = global_idx / 25.0
        frames.append((str(fp), pts))

    return frames


def _count_video_frames(video_path: str) -> int:
    """Count total frames in video using ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error",
             "-select_streams", "v:0",
             "-count_frames",
             "-show_entries", "stream=nb_read_frames",
             "-of", "default=noprint_wrappers=1:nokey=1",
             video_path],
            capture_output=True, text=True, timeout=60, shell=False
        )
        if r.returncode == 0:
            return int(r.stdout.strip())
    except Exception:
        pass
    # Fallback: use duration * fps
    try:
        dur_r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             video_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        fps_r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1",
             video_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if dur_r.returncode == 0 and fps_r.returncode == 0:
            duration = float(dur_r.stdout.strip())
            fps_parts = fps_r.stdout.strip().split("/")
            if len(fps_parts) == 2:
                fps = float(fps_parts[0]) / float(fps_parts[1])
            else:
                fps = float(fps_parts[0])
            return int(duration * fps)
    except Exception:
        pass
    return 0


def _load_params(args):
    if getattr(args, "params_file", None):
        with open(args.params_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(args.params) if args.params else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--params")
    parser.add_argument("--params-file")
    parser.add_argument("--config")
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()

    params = _load_params(args)
    log_dir = params.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, job_id=args.job_id)

    # Check AI settings
    ai_enabled = params.get("ai_enabled", False)
    if not ai_enabled:
        logger.info("STAGE3_SKIP", "AI not enabled")
        return 3

    scale = params.get("upscale_scale", 2)
    max_scale = params.get("max_scale", 4)

    # --- Layer 1 (hard): scale not 2/4 ---
    if scale not in (2, 4):
        logger.error("STAGE3_INVALID_SCALE",
                     f"Invalid scale: {scale}. Only integer 2x or 4x allowed.")
        return 81
    if scale > max_scale:
        logger.error("STAGE3_INVALID_SCALE",
                     f"Scale {scale} exceeds max_scale={max_scale}")
        return 81

    # Bug 13.30 fix: PyTorch version / backend check
    require_cuda = params.get("require_cuda", True)
    if require_cuda:
        try:
            import torch
            torch_ver = torch.__version__
            if "cpu" in torch_ver.lower():
                logger.error("STAGE3_CPU_TORCH",
                             f"PyTorch {torch_ver} is CPU-only "
                             "but require_cuda=True")
                # FIX-P8-4: Exit code 1 -> 83 (GPU error)
                # ORIGINAL: return 1  (outside 80-113 legal range)
                # Per v7.32 §3.5: 83 = GPU error
                return 83
        except Exception:
            pass

    # FIX-P8-5: WSL2 temp_dir redirect per v6.32 §7.3
    model = params.get("upscale_model", "realesr-general-x4v3")
    temp_dir = params.get("temp_dir", "./temp")
    try:
        from checkpoint import wsl_redirect_path
        # wsl_redirect_path returns (redirected_out, orig_out,
        #                            redirected_temp, orig_temp)
        _, _, temp_dir, _ = wsl_redirect_path("", temp_dir=temp_dir,
                                               logger=logger)
    except Exception:
        pass
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # Find stage2 output - deterministic by mtime (v6.32 §7.1)
    job_id = args.job_id
    input_name = Path(args.input).name
    pattern = f"{job_id}_{Path(input_name).stem}_step2_traditional*"
    candidates = list(Path(temp_dir).glob(pattern))
    if not candidates:
        logger.error("STAGE3_NO_INPUT",
                     "Stage 2 traditional output not found")
        return 82

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    stage2_output = str(candidates[0])

    # Validate stage2 output (v6.32 §7.1 mandatory)
    if Path(stage2_output).stat().st_size == 0:
        logger.error("STAGE3_STAGE2_EMPTY", "Stage 2 output is empty")
        return 82
    try:
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error", stage2_output],
            capture_output=True, timeout=30, shell=False
        )
        if r.returncode != 0:
            logger.error("STAGE3_STAGE2_UNDECODABLE",
                         "Stage 2 output undecodable")
            return 82
    except Exception as e:
        logger.error("STAGE3_STAGE2_CHECK_FAIL", str(e))
        return 82

    # Get intermediate dimensions for supplementary ban check
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "json", stage2_output],
            capture_output=True, text=True, timeout=30, shell=False
        )
        info = json.loads(r.stdout)
        stream = info["streams"][0]
        width = stream["width"]
        height = stream["height"]
    except Exception:
        width, height = 720, 480

    target_w = width * scale
    target_h = height * scale

    # FIX-P8-2: AI Upscale Ban Check - Three-Layer Verification
    # Per v7.32 §3.10.2: dual-basis (manifest flag + intermediate dims)
    # ORIGINAL: Only checked ai_banned manifest flag, no supplementary
    #           ban on intermediate dimensions, no three-layer logic
    ai_banned = params.get("ai_upscale_banned", False)
    force_allow = params.get("force_allow_banned_upscale", False)

    # FIX-P8-1: Dual-basis ban per v7.32 §3.10.2
    # Basis 1: Original input file pixel dimensions (permanent basis)
    # ORIGINAL: Only checked manifest flag + intermediate dims,
    #           missing original input dimensions check.
    # Per v7.32 §3.10.2: original file is the permanent baseline;
    # if it meets ban threshold, AI is banned regardless of
    # intermediate resizing/cropping.
    orig_w = params.get("original_width", 0)
    orig_h = params.get("original_height", 0)
    original_banned = False
    if orig_w > 0 and orig_h > 0:
        # HDV 1080p or larger: original already meets ban threshold
        if orig_w >= 1920 and orig_h >= 1080:
            original_banned = True
            logger.info("STAGE3_ORIGINAL_BANNED",
                        f"Original input {orig_w}x{orig_h} meets "
                        "ban threshold (>=1920x1080)")

    # FIX-P8-2: Analog tape 4x hard ban per v6.32 §4B
    # v6.32 §4B: "RealESRGAN 4x...strictly prohibited for analog tape"
    detected_class = params.get("detected_class", "")
    if detected_class == "ANALOG_TAPE" and scale == 4:
        logger.error("STAGE3_ANALOG_4X_BANNED",
                    "Real-ESRGAN 4x is strictly prohibited for analog tape "
                    "(VHS/Hi8/Betamax) per v6.32 §4B. Maximum upscale for "
                    "analog tape is 2x to 960p. "
                    "Use force_allow_banned_upscale to override.")
        if not force_allow:
            return 81  # Parameter error

    # Dual-basis ban: manifest flag OR original dims OR intermediate dims
    # Per v7.32 §3.10.2: either original or intermediate triggers ban
    intermediate_banned = (target_w > 1920 or target_h > 1080)
    combined_banned = ai_banned or original_banned or intermediate_banned

    # Layer 2 (default guard): skip when banned and not forced
    if combined_banned and not force_allow:
        logger.error("STAGE3_BANNED",
                     f"AI upscale banned: manifest={ai_banned}, "
                     f"intermediate_res_banned={intermediate_banned} "
                     f"({target_w}x{target_h})")
        return 3

    # FIX-P8-3: Layer 3 (force override): CRITICAL + WARNING dual log
    # Per v7.32 §3.10.2 L3: must emit CRITICAL + WARNING
    # ORIGINAL: Only had CRITICAL log, missing WARNING for quality risk
    if combined_banned and force_allow:
        logger.critical("STAGE3_FORCE",
                       "Force-allowing banned upscale per user override",
                       original_width=params.get("original_width"),
                       original_height=params.get("original_height"),
                       intermediate_width=width,
                       intermediate_height=height,
                       upscale_scale=scale)
        logger.warning("STAGE3_FORCE_RISK",
                      "Quality risk: forced upscale may produce AI "
                      "artifacts, plastic textures, temporal flickering, "
                      "or loss of historical authenticity")

    # GPU memory check
    mem_required = estimate_gpu_memory(width, height, scale, model)
    sufficient, backend = check_gpu_memory(mem_required, logger)

    if not sufficient and backend == "cpu":
        # Try 2x if requested 4x, preserving model family
        if scale == 4:
            logger.warning("STAGE3_DOWNSCALE",
                           "GPU memory insufficient for 4x, trying 2x")
            scale = 2
            mem_required = estimate_gpu_memory(width, height, scale, model)
            model = _get_model_for_scale(model, 2)
            sufficient, backend = check_gpu_memory(mem_required, logger)

        if not sufficient:
            logger.warning("STAGE3_CPU",
                           "GPU unavailable, using CPU (very slow)")
            backend = "cpu"

    # P8-6: Compute param_hash for intermediate filename
    # FIX-P8-8: Enforce checkpoint.compute_param_hash per v7.32 §3.10.19
    # ORIGINAL: Local fallback used SHA-256 without file content hash,
    #           silently degrading param_hash quality
    # checkpoint is a hard dependency - fail fast if not available
    from checkpoint import compute_param_hash

    core_params = {
        "upscale_scale": scale,
        "upscale_model": model,
        "lanczos_target": params.get("lanczos_target", "auto"),
        "ai_enabled": ai_enabled,
        "vs_bit_depth": params.get("vs_bit_depth", 16),  # FIX-P8-5
    }
    param_hash = compute_param_hash(stage2_output, core_params)

    logger.info("STAGE3_START", "AI upscaling",
                model=model, scale=scale, backend=backend,
                input_width=width, input_height=height)

    # 256-frame sliding window processing
    win_size = params.get("ai_window_size", 256)
    total_frames = _count_video_frames(stage2_output)
    if total_frames == 0:
        total_frames = 30000  # fallback: ~20min @ 25fps

    output_dir = str(Path(temp_dir) / f"{job_id}_upscaled")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get actual fps for frame extraction (v6.32 §4B)
    fps_cmd = [
        "ffprobe", "-hide_banner", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        stage2_output,
    ]
    actual_fps = 30.0
    try:
        r = subprocess.run(fps_cmd, capture_output=True, text=True,
                          timeout=10, shell=False)
        if r.returncode == 0:
            fps_str_val = r.stdout.strip()
            if "/" in fps_str_val:
                num, den = fps_str_val.split("/")
                actual_fps = int(num) / int(den)
            else:
                actual_fps = float(fps_str_val)
    except Exception:
        pass

    # FIX-N8-2: Query all PTS once before window loop (not per-window)
    # ORIGINAL: extract_frames_with_pts called ffprobe for every window
    all_pts_map = _query_all_pts(stage2_output, logger)

    # FIX-P8-9: Collect window dirs for batch processing
    # ORIGINAL: Per-window run_realesrgan (model loaded per window)
    # FIX-P8-8: Collect PTS for VFR preservation
    all_success = True
    gpu_oom_retry_done = False
    window_dirs: List[Tuple[str, str]] = []
    all_pts: List[float] = []  # FIX-P8-8: PTS for VFR
    all_frame_mappings: List[Tuple[int, int]] = []  # (win_start, count)

    for win_start in range(0, total_frames, win_size):
        win_end = min(win_start + win_size, total_frames)
        actual_win_size = win_end - win_start

        logger.info("STAGE3_WINDOW",
                    f"Processing frames {win_start}-{win_end}")

        # Extract window frames
        frames_dir = str(Path(temp_dir) / f"{job_id}_frames_{win_start}")
        # FIX-N8-2: Pass pre-built pts_map to avoid per-window ffprobe
        frames = extract_frames_with_pts(stage2_output, frames_dir,
                                         win_start, actual_win_size,
                                         logger, actual_fps,
                                         pts_map=all_pts_map)
        if not frames:
            logger.warning("STAGE3_WINDOW_EMPTY",
                           f"No frames in window {win_start}")
            continue

        # FIX-P8-8: Collect PTS for VFR encoding
        for _, pts in frames:
            all_pts.append(pts)

        win_output_dir = str(Path(temp_dir) / f"{job_id}_upscaled_{win_start}")
        Path(win_output_dir).mkdir(parents=True, exist_ok=True)
        window_dirs.append((frames_dir, win_output_dir))
        all_frame_mappings.append((win_start, len(frames)))

    # FIX-P8-9: Batch process all windows with single model load
    if window_dirs:
        result = run_realesrgan_batch(window_dirs, model, scale,
                                       gpu_id=0, logger=logger)

        # FIX-P8-7+P8-1: OOM downgrade only on actual OOM
        # ORIGINAL: OOM downgrade triggered on any failure (not just OOM),
        #           and model cleanup code was broken (del model on string)
        if result == "OOM" and backend != "cpu" and not gpu_oom_retry_done:
            # Step 1: 4x -> 2x downgrade
            if scale == 4:
                logger.warning("STAGE3_OOM_DOWNSCALE",
                              "GPU OOM detected, downgrading 4x->2x")
                time.sleep(0.1)  # v6.32 §7.14 mandatory 100ms wait
                scale = 2
                model = _get_model_for_scale(model, 2)
                result = run_realesrgan_batch(window_dirs, model, scale,
                                               gpu_id=0, logger=logger)

            # Step 2: GPU -> CPU downgrade
            if result == "OOM" or (isinstance(result, bool) and not result):
                if backend != "cpu":
                    logger.warning("STAGE3_OOM_CPU",
                                  "GPU still failing, downgrading to CPU")
                    time.sleep(0.1)
                    result = run_realesrgan_batch(
                        window_dirs, model, scale,
                        gpu_id=-1, logger=logger)
                    if result is True:
                        backend = "cpu"

            gpu_oom_retry_done = True

        # Non-OOM failure
        if result is not True:
            if result != "OOM":
                logger.error("STAGE3_WINDOW_FAIL",
                            "AI upscale failed (non-OOM)")
            else:
                logger.error("STAGE3_WINDOW_OOM",
                            "GPU OOM after retry exhausted")
            all_success = False

    # Collect upscaled frames from window outputs
    for win_start, frame_count in all_frame_mappings:
        win_output_dir = str(Path(temp_dir) / f"{job_id}_upscaled_{win_start}")
        upscaled = sorted(Path(win_output_dir).glob("*.png"),
                         key=lambda p: p.name)
        for src in upscaled:
            match = re.search(r'\d+', src.name)
            if not match:
                continue
            global_idx = win_start + int(match.group()) - 1
            dst = Path(output_dir) / f"frame_{global_idx:08d}.png"
            shutil.copyfile(str(src), str(dst))
            try:
                st = src.stat()
                os.utime(str(dst), (st.st_atime, st.st_mtime))
            except Exception:
                pass

        # Cleanup window frames to save disk space
        if params.get("cleanup_intermediates", True):
            frames_dir = str(Path(temp_dir) / f"{job_id}_frames_{win_start}")
            shutil.rmtree(frames_dir, ignore_errors=True)
            shutil.rmtree(win_output_dir, ignore_errors=True)

    if not all_success:
        logger.error("STAGE3_PARTIAL_FAIL",
                     "Some windows failed to upscale")

    # Verify we have output frames
    output_frames = sorted(Path(output_dir).glob("*.png"))
    if not output_frames:
        logger.error("STAGE3_NO_OUTPUT", "No upscaled frames produced")
        return 83

    # V-P-034 fix: Lanczos secondary scaling when target != AI output res
    lanczos_target = params.get("lanczos_target", "auto")
    if lanczos_target != "auto" and lanczos_target:
        try:
            if "x" in lanczos_target:
                target_lw, target_lh = map(int, lanczos_target.split("x"))
                if target_lw != target_w or target_lh != target_h:
                    logger.info("STAGE3_LANCZOS",
                               f"Scaling AI output {target_w}x{target_h} "
                               f"-> {target_lw}x{target_lh}")
                    lanczos_dir = str(Path(temp_dir) / f"{job_id}_lanczos")
                    Path(lanczos_dir).mkdir(parents=True, exist_ok=True)
                    for src in sorted(Path(output_dir).glob("*.png")):
                        lanczos_cmd = [
                            "ffmpeg", "-hide_banner", "-v", "error", "-y",
                            "-i", str(src),
                            "-vf",
                            f"zscale=w={target_lw}:h={target_lh}:f=lanczos",
                            "-pix_fmt", "rgb48",
                            str(Path(lanczos_dir) / src.name)
                        ]
                        subprocess.run(lanczos_cmd, capture_output=True,
                                      timeout=60, shell=False)
                    if list(Path(lanczos_dir).glob("*.png")):
                        shutil.rmtree(output_dir, ignore_errors=True)
                        # FIX-P8-10: Use shutil.move instead of os.rename
                        # os.rename fails across filesystems (OSError 18)
                        shutil.move(lanczos_dir, output_dir)
                        target_w, target_h = target_lw, target_lh
        except Exception as e:
            logger.warning("STAGE3_LANCZOS_FAIL",
                           f"Lanczos scaling failed: {e}")

    # V-P-035 fix: preserve original frame rate
    fps_cmd = [
        "ffprobe", "-hide_banner", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        stage2_output,
    ]
    fps_str = "25"
    try:
        r = subprocess.run(fps_cmd, capture_output=True, text=True,
                          timeout=10, shell=False)
        if r.returncode == 0:
            fps_str = r.stdout.strip()
    except Exception:
        pass

    # V-P-031 fix: force_allow suffix (P8-2 + N8-1 corrected)
    inter_name = f"{job_id}_{Path(input_name).stem}" \
                 f"_step3_upscale_p{param_hash}"
    # FIX-N8-1: Use combined_banned (not just ai_banned) per v7.32 §3.10.2 L3
    # ORIGINAL: only ai_banned triggered suffix; intermediate_banned also
    #           forces override and should add _forced_upscale suffix
    if combined_banned and force_allow:
        inter_name = (f"{job_id}_{Path(input_name).stem}"
                      f"_step3_forced_upscale_p{param_hash}")

    output_path = str(Path(temp_dir) / f"{inter_name}.mkv")

    # FIX-P8-8: VFR PTS preservation using concat demuxer
    # ORIGINAL: CFR encoding only (lost VFR timestamps)
    # Per v6.32 §4B: VFR support required
    is_vfr = False
    if len(all_pts) >= 3:
        intervals = [all_pts[i + 1] - all_pts[i]
                     for i in range(len(all_pts) - 1)]
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            max_deviation = max(abs(iv - avg_interval) for iv in intervals)
            # If max deviation > 10% of average, treat as VFR
            is_vfr = max_deviation > avg_interval * 0.1

    concat_path: Optional[str] = None
    if is_vfr and all_pts:
        # Use concat demuxer to preserve VFR timestamps
        concat_path = str(Path(temp_dir) / f"{job_id}_concat.txt")
        with open(concat_path, "w") as f:
            f.write("ffconcat version 1.0\n")
            for i, pts in enumerate(all_pts):
                if i < len(all_pts) - 1:
                    duration = all_pts[i + 1] - pts
                else:
                    duration = 1.0 / actual_fps
                frame_file = Path(output_dir) / f"frame_{i:08d}.png"
                f.write(f"file '{frame_file}'\n")
                f.write(f"duration {duration:.6f}\n")
            # FFmpeg concat demuxer needs last frame without duration
            if all_pts:
                last_frame = (Path(output_dir)
                              / f"frame_{len(all_pts)-1:08d}.png")
                f.write(f"file '{last_frame}'\n")

        encode_cmd = [
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_path,
            "-c:v", "ffv1",
            "-pix_fmt", "yuv420p16le",
            "-an",
            output_path,
        ]
    else:
        # CFR encoding (original path)
        encode_cmd = [
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-framerate", fps_str,
            "-i", str(Path(output_dir) / "frame_%08d.png"),
            "-c:v", "ffv1",
            "-pix_fmt", "yuv420p16le",
            "-an",
            output_path,
        ]

    try:
        r = subprocess.run(encode_cmd, capture_output=True, text=True,
                          timeout=3600, shell=False)
        if r.returncode != 0 or not Path(output_path).exists():
            logger.error("STAGE3_ENCODE_FAIL",
                         f"Failed to encode MKV: {r.stderr[:300]}")
            return 84
    except Exception as e:
        logger.error("STAGE3_ENCODE_ERROR", str(e))
        return 82

    # FIX-P8-8: Bitdepth self-check for intermediate file
    # Verify intermediate is actually 16bit per v6.32 processing space req
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=pix_fmt",
             "-of", "default=noprint_wrappers=1:nokey=1",
             output_path],
            capture_output=True, text=True, shell=False
        )
        pix_fmt_out = r.stdout.strip()
        if "p16" not in pix_fmt_out and "16" not in pix_fmt_out:
            logger.error("STAGE3_BITDEPTH_MISMATCH",
                        f"Intermediate file not 16bit: {pix_fmt_out}")
            return 86
    except Exception:
        pass

    # Generate forced upscale comparison preview
    # FIX-N8-1: Use combined_banned (not just ai_banned) to cover
    # intermediate_banned force override scenarios too
    if combined_banned and force_allow:
        preview_dir = params.get("preview_dir", "./preview")
        # FIX-P8-5: WSL2 redirect for preview_dir
        try:
            from checkpoint import wsl_redirect_path
            _, _, preview_dir, _ = wsl_redirect_path(
                "", temp_dir=preview_dir, logger=logger)
        except Exception:
            pass
        Path(preview_dir).mkdir(parents=True, exist_ok=True)
        try:
            from preview import generate_before_after_comparison
            generate_before_after_comparison(
                stage2_output, output_path,
                str(Path(preview_dir)
                    / f"{job_id}_forced_upscale_comparison.png")
            )
            logger.info("STAGE3_PREVIEW",
                        "Generated forced upscale comparison frames")
        except Exception as e:
            logger.warning("STAGE3_PREVIEW_FAIL",
                           f"Failed to generate preview: {e}")

    # Cleanup concat file if created
    if concat_path:
        try:
            Path(concat_path).unlink(missing_ok=True)
        except Exception:
            pass

    logger.info("STAGE3_DONE", "AI upscaling complete",
                total_frames=len(output_frames), output_path=output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
