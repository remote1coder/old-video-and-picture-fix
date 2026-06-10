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
"""

import os
import sys
import json
import hashlib
import argparse
import subprocess
import re
import glob
import shutil
from pathlib import Path
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402


def estimate_gpu_memory(width: int, height: int, scale: int,
                        model_name: str) -> int:
    """Estimate GPU memory requirement in bytes."""
    base = width * height * 3 * 4 * 1 * (2 + scale * scale)
    model_weights = {
        "RealESRGAN_x2plus": 64 * 1024 * 1024,
        "RealESRGAN_x4plus": 128 * 1024 * 1024,
        "DAT_x2": 32 * 1024 * 1024,
        "DAT_x4": 64 * 1024 * 1024,
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
                                   f"PyTorch {torch_ver} < 1.10, skipping torch.cuda.mem_get_info()")

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
            # MPS does not expose free memory via API; assume sufficient
            return True, "mps"

        # CPU-only PyTorch detection
        if "cpu" in torch_ver.lower():
            return False, "cpu"
    except Exception:
        pass

    return False, "cpu"


def _get_model_for_scale(model: str, scale: int) -> str:
    """Get correct model name for a given scale, preserving model family."""
    if model.startswith("RealESRGAN"):
        return f"RealESRGAN_x{scale}plus"
    elif model.startswith("DAT"):
        return f"DAT_x{scale}"
    return f"RealESRGAN_x{scale}plus"


def run_realesrgan(input_path: str, output_dir: str, model: str,
                   scale: int, gpu_id: int = 0) -> bool:
    """Run Real-ESRGAN upscaling via ncnn-vulkan or Python API."""
    model_dir = SCRIPT_DIR.parent.resolve() / "models"
    model_path = str(model_dir / f"{model}.pth")
    if not os.path.exists(model_path):
        return False

    # Try realesrgan-ncnn-vulkan CLI first
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
            return r.returncode == 0
        except Exception:
            pass

    # V-P-036 fix: Python API via dedicated temp script instead of inline -c
    try:
        env = os.environ.copy()
        if gpu_id < 0:
            env["CUDA_VISIBLE_DEVICES"] = ""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            script_path = f.name
            f.write(
                "from realesrgan import RealESRGANer\n"
                "import torch\n"
                "from PIL import Image\n"
                "import os\n"
                "import glob\n"
                "import numpy as np\n"
                f"scale = {scale}\n"
                f"model_path = {repr(model_path)}\n"
                f"input_path = {repr(input_path)}\n"
                f"output_dir = {repr(output_dir)}\n"
                f"use_half = not torch.cuda.is_available() if {gpu_id} >= 0 else False\n"
                "os.makedirs(output_dir, exist_ok=True)\n"
                "model = RealESRGANer(scale=scale, model_path=model_path, prepad=10, half=use_half)\n"
                "for fp in sorted(glob.glob(os.path.join(input_path, '*.png'))):\n"
                "    img = Image.open(fp).convert('RGB')\n"
                "    output, _ = model.enhance(np.array(img))\n"
                "    Image.fromarray(output).save(os.path.join(output_dir, os.path.basename(fp)))\n"
                "del model\n"
                "import gc\n"
                "gc.collect()\n"
                "import torch\n"
                "torch.cuda.empty_cache()\n"
            )
        cmd = [sys.executable, script_path]
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=3600, shell=False, env=env)
        try:
            os.unlink(script_path)
        except Exception:
            pass
        return r.returncode == 0
    except Exception:
        return False
    finally:
        # P8-3: Explicitly release model in Python API path
        try:
            del model
            import gc
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def extract_frames_with_pts(video_path: str, output_dir: str,
                              win_start: int = 0, win_size: int = 256,
                              logger=None, fps: float = 30.0) -> List[Tuple[str, float]]:
    """Extract frames in a sliding window with PTS support for VFR.

    Returns list of (frame_path, pts_seconds) ordered by PTS.
    """
    os.makedirs(output_dir, exist_ok=True)

    # V-P-033 fix: extract 16bit PNG for preservation
    # Use ffmpeg with -vsync 0 to preserve original timestamps
    # Extract only the window range
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", video_path,
    ]
    if win_start > 0:
        cmd.extend(["-ss", str(win_start / fps)])  # v6.32 §4B: use actual fps
    if win_size > 0:
        cmd.extend(["-frames:v", str(win_size)])
    cmd.extend([
        "-vsync", "0",
        "-frame_pts", "1",
        "-pix_fmt", "rgb48",
        os.path.join(output_dir, "frame_%08d.png")
    ])

    try:
        r = subprocess.run(cmd, capture_output=True, timeout=600, shell=False)
        if r.returncode != 0:
            if logger:
                logger.error("STAGE3_EXTRACT_FAIL", f"Frame extract: {r.stderr.decode()[:200]}")
            return []
    except Exception as e:
        if logger:
            logger.error("STAGE3_EXTRACT_EXCEPT", str(e))
        return []

    # Get PTS for each extracted frame
    pts_map = {}
    try:
        pts_cmd = [
            "ffprobe", "-hide_banner", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "frame=pkt_pts_time",
            "-of", "json",
            video_path
        ]
        pts_proc = subprocess.run(pts_cmd, capture_output=True, text=True,
                                 timeout=300, shell=False)
        if pts_proc.returncode == 0:
            data = json.loads(pts_proc.stdout)
            for i, frame_info in enumerate(data.get("frames", [])):
                pts_str = frame_info.get("pkt_pts_time", "")
                if pts_str:
                    try:
                        pts_map[i] = float(pts_str)
                    except ValueError:
                        pts_map[i] = i / 25.0
    except Exception:
        pass

    # Build ordered list by actual frame number
    frames = []
    frame_files = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
    for i, fp in enumerate(frame_files):
        # Map window-local index to global index
        global_idx = win_start + i
        pts = pts_map.get(global_idx, global_idx / 25.0)
        frames.append((fp, pts))

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
    # v6.3: Only integer scales 2x and 4x are allowed
    if scale not in (2, 4):
        logger.error("STAGE3_INVALID_SCALE",
                     f"Invalid scale: {scale}. Only integer 2x or 4x allowed.")
        return 81
    if scale > max_scale:
        logger.error("STAGE3_INVALID_SCALE", f"Scale {scale} exceeds max_scale={max_scale}")
        return 81

    # Bug 13.30 fix: PyTorch version / backend check
    require_cuda = params.get("require_cuda", True)
    if require_cuda:
        try:
            import torch
            torch_ver = torch.__version__
            if "cpu" in torch_ver.lower():
                logger.error("STAGE3_CPU_TORCH",
                             f"PyTorch {torch_ver} is CPU-only but require_cuda=True")
                return 1
        except Exception:
            pass

    # Check upscale ban
    ai_banned = params.get("ai_upscale_banned", False)
    force_allow = params.get("force_allow_banned_upscale", False)
    if ai_banned and not force_allow:
        logger.info("STAGE3_BANNED", "AI upscale banned for this file")
        return 3
    # V-P-030 fix: force_allow CRITICAL log
    if ai_banned and force_allow:
        logger.critical("STAGE3_FORCE",
                       "Force-allowing banned upscale per user override",
                       original_width=params.get("original_width"),
                       original_height=params.get("original_height"),
                       upscale_scale=scale)

    model = params.get("upscale_model", "RealESRGAN_x2plus")
    temp_dir = params.get("temp_dir", "./temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Find stage2 output - deterministic by mtime (v6.32 §7.1)
    job_id = args.job_id
    input_name = Path(args.input).name
    pattern = f"{job_id}_{Path(input_name).stem}_step2_traditional*"
    candidates = glob.glob(os.path.join(temp_dir, pattern))
    if not candidates:
        logger.error("STAGE3_NO_INPUT", "Stage 2 traditional output not found")
        return 82

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    stage2_output = candidates[0]

    # Validate stage2 output (v6.32 §7.1 mandatory)
    if os.path.getsize(stage2_output) == 0:
        logger.error("STAGE3_STAGE2_EMPTY", "Stage 2 output is empty")
        return 82
    try:
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error", stage2_output],
            capture_output=True, timeout=30, shell=False
        )
        if r.returncode != 0:
            logger.error("STAGE3_STAGE2_UNDECODABLE", "Stage 2 output undecodable")
            return 82
    except Exception as e:
        logger.error("STAGE3_STAGE2_CHECK_FAIL", str(e))
        return 82

    # Get dimensions
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

    # v6.3 §5.3: Target resolution check - AI upscale only available when target <= 1080p
    target_w = width * scale
    target_h = height * scale
    if target_w > 1920 or target_h > 1080:
        logger.warning("STAGE3_RES_LIMIT",
                       f"Target resolution {target_w}x{target_h} exceeds 1080p, "
                       "skipping AI upscale per v6.3 §5.3")
        return 3

    # GPU memory check
    mem_required = estimate_gpu_memory(width, height, scale, model)
    sufficient, backend = check_gpu_memory(mem_required, logger)

    if not sufficient and backend == "cpu":
        # Try 2x if requested 4x, preserving model family
        if scale == 4:
            logger.warning("STAGE3_DOWNSCALE", "GPU memory insufficient for 4x, trying 2x")
            scale = 2
            mem_required = estimate_gpu_memory(width, height, scale, model)
            model = _get_model_for_scale(model, 2)
            sufficient, backend = check_gpu_memory(mem_required, logger)

        if not sufficient:
            logger.warning("STAGE3_CPU", "GPU unavailable, using CPU (very slow)")
            backend = "cpu"

    # P8-6: Compute param_hash for intermediate filename
    try:
        from checkpoint import compute_param_hash
    except Exception:
        # Fallback if checkpoint not available
        def compute_param_hash(input_path, params):
            import hashlib
            param_str = json.dumps(params, sort_keys=True, ensure_ascii=False)
            file_id = str(os.path.getsize(input_path)) + str(int(os.path.getmtime(input_path)))
            return hashlib.sha256((param_str + file_id).encode()).hexdigest()[:16]

    core_params = {
        "upscale_scale": scale,
        "upscale_model": model,
        "lanczos_target": params.get("lanczos_target", "auto"),
        "ai_enabled": ai_enabled,
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

    output_dir = os.path.join(temp_dir, f"{job_id}_upscaled")
    os.makedirs(output_dir, exist_ok=True)

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
        r = subprocess.run(fps_cmd, capture_output=True, text=True, timeout=10, shell=False)
        if r.returncode == 0:
            fps_str = r.stdout.strip()
            if "/" in fps_str:
                num, den = fps_str.split("/")
                actual_fps = int(num) / int(den)
            else:
                actual_fps = float(fps_str)
    except Exception:
        pass

    all_success = True
    gpu_oom_retry_done = False  # v6.3 §7.14: max 1 OOM retry per job
    for win_start in range(0, total_frames, win_size):
        win_end = min(win_start + win_size, total_frames)
        actual_win_size = win_end - win_start

        logger.info("STAGE3_WINDOW", f"Processing frames {win_start}-{win_end}")

        # Extract window frames
        frames_dir = os.path.join(temp_dir, f"{job_id}_frames_{win_start}")
        frames = extract_frames_with_pts(stage2_output, frames_dir,
                                         win_start, actual_win_size, logger, actual_fps)
        if not frames:
            logger.warning("STAGE3_WINDOW_EMPTY", f"No frames in window {win_start}")
            continue

        # Run AI upscaling on this window
        win_output_dir = os.path.join(temp_dir, f"{job_id}_upscaled_{win_start}")
        os.makedirs(win_output_dir, exist_ok=True)

        success = run_realesrgan(frames_dir, win_output_dir, model, scale)

        # ISSUE-043 fix: runtime GPU OOM downgrade (4x->2x->CPU) with del model + empty_cache + 100ms
        if not success and backend != "cpu" and not gpu_oom_retry_done:
            # Step 1: 4x -> 2x downgrade
            if scale == 4:
                logger.warning("STAGE3_OOM_DOWNSCALE", "GPU OOM detected, downgrading 4x->2x")
                try:
                    import torch
                    _saved_model = model
                    del model
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    model = _saved_model
                except Exception:
                    pass
                import time
                time.sleep(0.1)  # v6.32 §7.14 mandatory 100ms wait
                scale = 2
                model = _get_model_for_scale(model, 2)
                success = run_realesrgan(frames_dir, win_output_dir, model, scale)

            # Step 2: GPU -> CPU downgrade
            if not success:
                logger.warning("STAGE3_OOM_CPU", "GPU still failing, downgrading to CPU")
                try:
                    import torch
                    _saved_model = model
                    del model
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    model = _saved_model
                except Exception:
                    pass
                import time
                time.sleep(0.1)
                success = run_realesrgan(frames_dir, win_output_dir, model, scale, gpu_id=-1)
                if success:
                    backend = "cpu"

            gpu_oom_retry_done = True  # v6.3: limit 1 retry per job

        if not success:
            logger.error("STAGE3_WINDOW_FAIL", f"AI upscale failed for window {win_start}")
            all_success = False
            continue

        # Collect upscaled frames from window output
        upscaled = sorted(Path(win_output_dir).glob("*.png"),
                         key=lambda p: p.name)
        for src in upscaled:
            # Map to global frame numbering
            global_idx = win_start + int(re.search(r'\d+', src.name).group()) - 1
            dst = Path(output_dir) / f"frame_{global_idx:08d}.png"
            shutil.copyfile(str(src), str(dst))
            try:
                st = os.stat(str(src))
                os.utime(str(dst), (st.st_atime, st.st_mtime))
            except Exception:
                pass

        # Cleanup window frames to save disk space
        if params.get("cleanup_intermediates", True):
            shutil.rmtree(frames_dir, ignore_errors=True)
            shutil.rmtree(win_output_dir, ignore_errors=True)

    if not all_success:
        logger.error("STAGE3_PARTIAL_FAIL", "Some windows failed to upscale")

    # Verify we have output frames
    output_frames = sorted(Path(output_dir).glob("*.png"))
    if not output_frames:
        logger.error("STAGE3_NO_OUTPUT", "No upscaled frames produced")
        return 83

    # V-P-034 fix: Lanczos secondary scaling when target != AI output resolution
    lanczos_target = params.get("lanczos_target", "auto")
    if lanczos_target != "auto" and lanczos_target:
        try:
            if "x" in lanczos_target:
                target_lw, target_lh = map(int, lanczos_target.split("x"))
                if target_lw != target_w or target_lh != target_h:
                    logger.info("STAGE3_LANCZOS",
                               f"Scaling AI output {target_w}x{target_h} -> {target_lw}x{target_lh}")
                    lanczos_dir = os.path.join(temp_dir, f"{job_id}_lanczos")
                    os.makedirs(lanczos_dir, exist_ok=True)
                    for src in sorted(Path(output_dir).glob("*.png")):
                        lanczos_cmd = [
                            "ffmpeg", "-hide_banner", "-v", "error", "-y",
                            "-i", str(src),
                            "-vf", f"zscale=w={target_lw}:h={target_lh}:f=lanczos",
                            "-pix_fmt", "rgb48",
                            str(Path(lanczos_dir) / src.name)
                        ]
                        subprocess.run(lanczos_cmd, capture_output=True, timeout=60, shell=False)
                    if list(Path(lanczos_dir).glob("*.png")):
                        import shutil as _shutil
                        _shutil.rmtree(output_dir, ignore_errors=True)
                        os.rename(lanczos_dir, output_dir)
                        target_w, target_h = target_lw, target_lh
        except Exception as e:
            logger.warning("STAGE3_LANCZOS_FAIL", f"Lanczos scaling failed: {e}")

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
        r = subprocess.run(fps_cmd, capture_output=True, text=True, timeout=10, shell=False)
        if r.returncode == 0:
            fps_str = r.stdout.strip()
    except Exception:
        pass

    # V-P-031 fix: force_allow suffix (P8-2 corrected)
    inter_name = f"{job_id}_{Path(input_name).stem}_step3_upscale_p{param_hash}"
    if ai_banned and force_allow:
        inter_name = f"{job_id}_{Path(input_name).stem}_step3_forced_upscale_p{param_hash}"

    # Encode PNG frames to intermediate MKV for stage4 consumption
    output_path = os.path.join(temp_dir, f"{inter_name}.mkv")
    encode_cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-framerate", fps_str,
        "-i", os.path.join(output_dir, "frame_%08d.png"),
        "-c:v", "ffv1",
        "-pix_fmt", "yuv420p16le",
        "-an",
        output_path,
    ]
    try:
        r = subprocess.run(encode_cmd, capture_output=True, text=True,
                          timeout=3600, shell=False)
        if r.returncode != 0 or not os.path.exists(output_path):
            logger.error("STAGE3_ENCODE_FAIL", f"Failed to encode MKV: {r.stderr[:300]}")
            return 84
    except Exception as e:
        logger.error("STAGE3_ENCODE_ERROR", str(e))
        return 82

    # P8-9: Generate forced upscale comparison preview
    if ai_banned and force_allow:
        preview_dir = params.get("preview_dir", "./preview")
        os.makedirs(preview_dir, exist_ok=True)
        try:
            from preview import generate_before_after_comparison
            generate_before_after_comparison(
                stage2_output, output_path,
                os.path.join(preview_dir, f"{job_id}_forced_upscale_comparison.png")
            )
            logger.info("STAGE3_PREVIEW", "Generated forced upscale comparison frames")
        except Exception as e:
            logger.warning("STAGE3_PREVIEW_FAIL", f"Failed to generate preview: {e}")

    logger.info("STAGE3_DONE", "AI upscaling complete",
                total_frames=len(output_frames), output_path=output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
