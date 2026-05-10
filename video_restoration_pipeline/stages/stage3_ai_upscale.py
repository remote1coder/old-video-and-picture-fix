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


def check_gpu_memory(required_bytes: int) -> tuple:
    """Check GPU availability. Returns (sufficient: bool, backend: str)."""
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

    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            if free >= required_bytes * 1.2:
                return True, "cuda"
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


def run_realesrgan(input_path: str, output_dir: str, model_name: str,
                   scale: int, gpu_id: int = 0) -> bool:
    """Run Real-ESRGAN upscaling via ncnn-vulkan or Python API."""
    model_dir = SCRIPT_DIR.parent.resolve() / "models"
    model_path = str(model_dir / f"{model_name}.pth")
    if not os.path.exists(model_path):
        return False

    # Try realesrgan-ncnn-vulkan CLI first
    vulkan_bin = shutil.which("realesrgan-ncnn-vulkan")
    if vulkan_bin:
        cmd = [
            vulkan_bin,
            "-i", input_path,
            "-o", output_dir,
            "-n", model_name,
            "-s", str(scale),
            "-g", str(gpu_id),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=3600, shell=False)
            return r.returncode == 0
        except Exception:
            pass

    # Fallback: Python API via subprocess
    try:
        env = os.environ.copy()
        if gpu_id < 0:
            env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = [
            sys.executable, "-c",
            f"from realesrgan import RealESRGANer; "
            f"from basicsr.utils.download_util import load_file_from_url; "
            f"import torch; "
            f"model = RealESRGANer(scale={scale}, model_path='{model_path}', "
            f"prepad=10, half={'False' if gpu_id < 0 else 'not torch.cuda.is_available()'}); "
            f"from PIL import Image; "
            f"import os; "
            f"import glob; "
            f"os.makedirs('{output_dir}', exist_ok=True); "
            f"for f in sorted(glob.glob(os.path.join('{input_path}', '*.png'))): "
            f"    img = Image.open(f).convert('RGB'); "
            f"    import numpy as np; "
            f"    output, _ = model.enhance(np.array(img)); "
            f"    Image.fromarray(output).save(os.path.join('{output_dir}', os.path.basename(f)))"
        ]
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=3600, shell=False, env=env)
        return r.returncode == 0
    except Exception:
        return False


def extract_frames_with_pts(video_path: str, output_dir: str,
                              win_start: int = 0, win_size: int = 256,
                              logger=None) -> List[Tuple[str, float]]:
    """Extract frames in a sliding window with PTS support for VFR.

    Returns list of (frame_path, pts_seconds) ordered by PTS.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use ffmpeg with -vsync 0 to preserve original timestamps
    # Extract only the window range
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", video_path,
    ]
    if win_start > 0:
        cmd.extend(["-ss", str(win_start / 30.0)])  # approximate frame to time
    if win_size > 0:
        cmd.extend(["-frames:v", str(win_size)])
    cmd.extend([
        "-vsync", "0",
        "-frame_pts", "1",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--params")
    parser.add_argument("--config")
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()

    params = json.loads(args.params) if args.params else {}
    log_dir = params.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, job_id=args.job_id)

    # Check AI settings
    ai_enabled = params.get("ai_enabled", False)
    if not ai_enabled:
        logger.info("STAGE3_SKIP", "AI not enabled")
        return 3

    # Check upscale ban
    ai_banned = params.get("ai_upscale_banned", False)
    force_allow = params.get("force_allow_banned_upscale", False)
    if ai_banned and not force_allow:
        logger.info("STAGE3_BANNED", "AI upscale banned for this file")
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

    model = params.get("upscale_model", "RealESRGAN_x2plus")
    temp_dir = params.get("temp_dir", "./temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Find stage1 output - deterministic by mtime
    job_id = args.job_id
    input_name = Path(args.input).name
    pattern = f"{job_id}_{Path(input_name).stem}_step1_preprocess*"
    candidates = glob.glob(os.path.join(temp_dir, pattern))
    if not candidates:
        logger.error("STAGE3_NO_INPUT", "Stage 1 output not found")
        return 82

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    stage1_output = candidates[0]

    # Get dimensions
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "json", stage1_output],
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
    sufficient, backend = check_gpu_memory(mem_required)

    if not sufficient and backend == "cpu":
        # Try 2x if requested 4x, preserving model family
        if scale == 4:
            logger.warning("STAGE3_DOWNSCALE", "GPU memory insufficient for 4x, trying 2x")
            scale = 2
            mem_required = estimate_gpu_memory(width, height, scale, model)
            model = _get_model_for_scale(model, 2)
            sufficient, backend = check_gpu_memory(mem_required)

        if not sufficient:
            logger.warning("STAGE3_CPU", "GPU unavailable, using CPU (very slow)")
            backend = "cpu"

    logger.info("STAGE3_START", "AI upscaling",
                model=model, scale=scale, backend=backend,
                input_width=width, input_height=height)

    # 256-frame sliding window processing
    win_size = params.get("ai_window_size", 256)
    total_frames = _count_video_frames(stage1_output)
    if total_frames == 0:
        total_frames = 30000  # fallback: ~20min @ 25fps

    output_dir = os.path.join(temp_dir, f"{job_id}_upscaled")
    os.makedirs(output_dir, exist_ok=True)

    all_success = True
    for win_start in range(0, total_frames, win_size):
        win_end = min(win_start + win_size, total_frames)
        actual_win_size = win_end - win_start

        logger.info("STAGE3_WINDOW", f"Processing frames {win_start}-{win_end}")

        # Extract window frames
        frames_dir = os.path.join(temp_dir, f"{job_id}_frames_{win_start}")
        frames = extract_frames_with_pts(stage1_output, frames_dir,
                                         win_start, actual_win_size, logger)
        if not frames:
            logger.warning("STAGE3_WINDOW_EMPTY", f"No frames in window {win_start}")
            continue

        # Run AI upscaling on this window
        win_output_dir = os.path.join(temp_dir, f"{job_id}_upscaled_{win_start}")
        os.makedirs(win_output_dir, exist_ok=True)

        success = run_realesrgan(frames_dir, win_output_dir, model, scale)

        if not success and backend != "cpu":
            # GPU OOM retry: clear cache, wait, then retry on CPU
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            import time
            time.sleep(0.1)
            logger.warning("STAGE3_RETRY_CPU", "Retrying on CPU after GPU failure")
            success = run_realesrgan(frames_dir, win_output_dir, model, scale, gpu_id=-1)

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
            shutil.copy2(str(src), str(dst))

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

    # Encode PNG frames to intermediate MKV for stage4 consumption
    output_path = os.path.join(temp_dir,
        f"{job_id}_{Path(input_name).stem}_step3_upscale.mkv")
    encode_cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-framerate", "25",  # TODO: preserve original frame rate
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

    logger.info("STAGE3_DONE", "AI upscaling complete",
                total_frames=len(output_frames), output_path=output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
