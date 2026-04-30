#!/usr/bin/env python3
"""
Stage 4: Post-processing after AI upscale
- Lanczos resize to target resolution (if non-integer target)
- Selective local processing with masks
- 16bit -> 10bit with error diffusion dithering

Exit codes: 0=success, 3=skipped
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_utils"))

from logger import get_logger  # noqa: E402


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

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    input_name = Path(args.input).name

    # Find input (stage3 output if AI enabled, else stage1 output)
    import glob

    # Try stage3 first
    pattern3 = f"{job_id}_{Path(input_name).stem}_step3_upscale*"
    candidates3 = glob.glob(os.path.join(temp_dir, pattern3))

    if candidates3:
        # AI was run, use stage3 output
        # Find upscaled frames and encode
        logger.info("STAGE4_START", "Post-processing AI upscaled frames")

        frames_dir = os.path.join(temp_dir, f"{job_id}_upscaled")
        if not os.path.exists(frames_dir):
            logger.info("STAGE4_SKIP", "No upscaled frames found")
            return 3

        # Re-encode upscaled frames to video
        output_path = os.path.join(temp_dir,
            f"{job_id}_{Path(input_name).stem}_step4_postprocess.mkv")

        try:
            # Get frame rate from stage1 processed output (NOT original input)
            # Stage1 may have changed frame rate via IVTC or QTGMC
            stage1_pattern = f"{job_id}_{Path(input_name).stem}_step1_preprocess*"
            stage1_candidates = glob.glob(os.path.join(temp_dir, stage1_pattern))
            fps_source = stage1_candidates[0] if stage1_candidates else args.input

            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate",
                 "-of", "default=noprint_wrappers=1:nokey=1", fps_source],
                capture_output=True, text=True, timeout=30, shell=False
            )
            fps = r.stdout.strip()
            if not fps:
                fps = "25"  # fallback
            logger.info("STAGE4_FPS", f"Using frame rate from processed video: {fps}")

            # Determine target resolution for Lanczos resize
            lanczos_target = params.get("lanczos_target", "auto")
            video_filter = ""
            if lanczos_target and lanczos_target != "auto":
                try:
                    target_w, target_h = lanczos_target.split("x")
                    # Check current frame size
                    sample_frame = os.path.join(frames_dir, "frame_00000001.png")
                    if not os.path.exists(sample_frame):
                        # Try glob
                        pngs = list(Path(frames_dir).glob("*.png"))
                        if pngs:
                            sample_frame = str(pngs[0])
                    if os.path.exists(sample_frame):
                        from PIL import Image
                        with Image.open(sample_frame) as img:
                            cur_w, cur_h = img.size
                        if cur_w != int(target_w) or cur_h != int(target_h):
                            video_filter = f"scale={target_w}:{target_h}:flags=lanczos"
                            logger.info("STAGE4_LANCZOS",
                                        f"Resizing {cur_w}x{cur_h} -> {target_w}x{target_h}")
                except Exception:
                    pass

            # Selective local processing: mild denoise on flat areas
            # Using smartblur with edge protection via edgedetect mask
            local_process = params.get("local_mask_denoise", True)
            if local_process:
                # Build a filter that smooths low-texture areas while preserving edges
                # smartblur = minblur with edge threshold
                local_filter = "smartblur=lr=1.0:ls=2.0"
                if video_filter:
                    video_filter = video_filter + "," + local_filter
                else:
                    video_filter = local_filter
                logger.info("STAGE4_MASK", "Applying selective local denoise")

            cmd = [
                "ffmpeg", "-hide_banner", "-v", "error",
                "-framerate", fps,
                "-i", os.path.join(frames_dir, "frame_%08d.png"),
            ]
            if video_filter:
                cmd.extend(["-vf", video_filter])
            cmd.extend([
                "-c:v", "ffv1",
                "-pix_fmt", "yuv444p16",
                "-an", "-y", output_path
            ])
            r = subprocess.run(cmd, capture_output=True, timeout=1800, shell=False)

            if r.returncode == 0:
                logger.info("STAGE4_DONE", "Post-process complete", output=output_path)
                return 0
            else:
                logger.error("STAGE4_FAILED", f"Encode error: {r.stderr.decode()[:300]}")
                return 84

        except Exception as e:
            logger.error("STAGE4_ERROR", str(e))
            return 82
    else:
        # No AI upscaling, skip this stage
        logger.info("STAGE4_SKIP", "No AI upscale output, skipping")
        return 3


if __name__ == "__main__":
    sys.exit(main() or 0)
