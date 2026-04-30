#!/usr/bin/env python3
"""
Stage 5: AI Face Restoration (Video)
GFPGAN/CodeFormer face restoration for static face segments only.

Static segment criteria (v6.3):
- Duration >= 2 seconds (VFR-aware via PTS accumulation)
- Face detection confidence >= 0.8 (DV low-res fallback to 0.6)
- Face center displacement < 5% of face box size AND < max(10px, min(50px, 5%))
- Face area > 15% of frame area
- Global motion < 3% (exclude camera movement)
- Feathered blending (gaussian blur mask, >=8px transition)

Exit codes: 0=success, 3=skipped, 83=GPU error
"""

import os
import sys
import json
import argparse
import subprocess
import glob
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_utils"))

from logger import get_logger  # noqa: E402


def extract_frames_with_pts(video_path: str, output_dir: str,
                            logger=None) -> List[Tuple[str, float]]:
    """Extract frames and return list of (frame_path, pts_seconds)."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames at native rate
    frame_pattern = os.path.join(output_dir, "frame_%08d.png")
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", video_path,
        "-vsync", "0",  # Preserve original timestamps
        "-frame_pts", "1",
        frame_pattern
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=1800, shell=False)
        if r.returncode != 0:
            if logger:
                logger.error("STAGE5_FF_FAIL", f"Frame extract failed: {r.stderr.decode()[:200]}")
            return []
    except Exception as e:
        if logger:
            logger.error("STAGE5_FF_EXCEPT", f"Frame extract exception: {e}")
        return []

    # Get PTS for each frame using ffprobe
    pts_cmd = [
        "ffprobe", "-hide_banner", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "frame=pkt_pts_time",
        "-of", "json",
        video_path
    ]
    pts_map = {}
    try:
        pts_proc: subprocess.CompletedProcess[str] = subprocess.run(
            pts_cmd, capture_output=True, text=True,
            timeout=300, shell=False)
        if pts_proc.returncode == 0:
            data = json.loads(pts_proc.stdout)
            for i, frame_info in enumerate(data.get("frames", [])):
                pts_str = frame_info.get("pkt_pts_time", "")
                if pts_str:
                    try:
                        pts_map[i] = float(pts_str)
                    except ValueError:
                        pts_map[i] = i / 25.0  # fallback
    except Exception:
        pass

    # Build ordered list
    frames = []
    frame_files = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
    for i, fp in enumerate(frame_files):
        pts = pts_map.get(i, i / 25.0)
        frames.append((fp, pts))

    return frames


def detect_faces(frame_path: str, model: str = "retinaface",
                 threshold: float = 0.8,
                 threshold_lowres: float = 0.6,
                 lowres_max_dim: int = 720) -> List[Tuple[int, int, int, int, float]]:
    """Detect faces in a frame. Returns list of (x, y, w, h, confidence)."""
    faces = []

    # Determine effective threshold based on resolution
    try:
        from PIL import Image
        with Image.open(frame_path) as img:
            w, h = img.size
            max_dim = max(w, h)
            effective_threshold = threshold_lowres if max_dim <= lowres_max_dim else threshold
    except Exception:
        effective_threshold = threshold

    try:
        if model == "retinaface":
            from retina_face import RetinaFace
            resp = RetinaFace.detect_faces(frame_path)
            if resp:
                for key, face in resp.items():
                    bbox = face.get("facial_area", [0, 0, 0, 0])
                    score = face.get("score", 0)
                    if score >= effective_threshold:
                        faces.append((bbox[0], bbox[1], bbox[2] - bbox[0],
                                     bbox[3] - bbox[1], score))
        elif model == "yolov8":
            from ultralytics import YOLO
            detector = YOLO("yolov8n-face.pt")
            results = detector(frame_path, verbose=False)
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf >= effective_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        faces.append((x1, y1, x2 - x1, y2 - y1, conf))
        elif model == "mtcnn":
            from mtcnn import MTCNN
            import cv2
            detector = MTCNN()
            img = cv2.imread(frame_path)
            if img is not None:
                results = detector.detect_faces(img)
                for r in results:
                    if r["confidence"] >= effective_threshold:
                        b = r["box"]
                        faces.append((b[0], b[1], b[2], b[3], r["confidence"]))
    except Exception:
        pass

    return faces


def detect_faces_with_fallback(frame_path: str,
                               models: List[str],
                               threshold: float = 0.8,
                               threshold_lowres: float = 0.6,
                               lowres_max_dim: int = 720) -> List[Tuple[int, int, int, int, float]]:
    """Try face detection models in order, return first success."""
    for model in models:
        faces = detect_faces(frame_path, model, threshold, threshold_lowres, lowres_max_dim)
        if faces:
            return faces
    return []


def match_face(prev_box: Tuple, faces: List[Tuple]) -> Optional[Tuple]:
    """Find the face in `faces` closest to `prev_box` by center distance."""
    if not faces:
        return None
    px, py, pw, ph, _ = prev_box
    pcx, pcy = px + pw // 2, py + ph // 2

    best_match = None
    best_dist = float("inf")
    for face in faces:
        fx, fy, fw, fh, fc = face
        cx, cy = fx + fw // 2, fy + fh // 2
        dist = (cx - pcx) ** 2 + (cy - pcy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_match = face
    return best_match


def compute_global_motion(frame_a: str, frame_b: str) -> float:
    """Compute global motion ratio between two frames using simple frame diff."""
    try:
        import cv2
        import numpy as np
        a = cv2.imread(frame_a, cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(frame_b, cv2.IMREAD_GRAYSCALE)
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        diff = cv2.absdiff(a, b)
        motion_pixels = np.count_nonzero(diff > 20)
        total_pixels = a.shape[0] * a.shape[1]
        return motion_pixels / total_pixels
    except Exception:
        return 0.0


def compute_frame_ssim(frame_a: str, frame_b: str) -> float:
    """Compute SSIM between two frames. Returns 0.0-1.0."""
    try:
        import cv2
        a = cv2.imread(frame_a, cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(frame_b, cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            return 0.0
        if a.shape != b.shape:
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            a = a[:h, :w]
            b = b[:h, :w]
        # Simple SSIM approximation using mean/std correlation
        mu1 = a.mean()
        mu2 = b.mean()
        sigma1 = a.std()
        sigma2 = b.std()
        sigma12 = ((a - mu1) * (b - mu2)).mean()
        c1 = 0.01 * 255 * 0.01 * 255
        c2 = 0.03 * 255 * 0.03 * 255
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 * mu1 + mu2 * mu2 + c1) * (sigma1 * sigma1 + sigma2 * sigma2 + c2))
        return float(ssim)
    except Exception:
        return 0.0


def find_static_frame_ranges(frames: List[Tuple[str, float]],
                              min_frames: int = 25,
                              ssim_threshold: float = 0.99) -> List[Tuple[int, int]]:
    """Find ranges of consecutive frames with SSIM > threshold.

    Returns list of (start_idx, end_idx) tuples.
    """
    ranges: List[Tuple[int, int]] = []
    if len(frames) < min_frames:
        return ranges

    start = 0
    for i in range(1, len(frames)):
        ssim = compute_frame_ssim(frames[i - 1][0], frames[i][0])
        if ssim < ssim_threshold:
            # End of static range
            length = i - start
            if length >= min_frames:
                ranges.append((start, i - 1))
            start = i
    # Check final range
    if start < len(frames) - 1:
        length = len(frames) - start
        if length >= min_frames:
            ranges.append((start, len(frames) - 1))
    return ranges


def feather_blend(restored_face: str, original_frame: str,
                  face_box: Tuple[int, int, int, int],
                  output_path: str, feather_width: int = 8) -> bool:
    """Blend restored face back into original frame with feathered edges."""
    try:
        import cv2
        import numpy as np

        orig = cv2.imread(original_frame)
        restored = cv2.imread(restored_face)
        if orig is None or restored is None:
            return False

        fx, fy, fw, fh = face_box

        # Ensure restored face matches expected size
        if restored.shape[0] != fh or restored.shape[1] != fw:
            restored = cv2.resize(restored, (fw, fh))

        # Create feather mask
        mask = np.zeros((fh, fw), dtype=np.float32)
        mask[feather_width:-feather_width, feather_width:-feather_width] = 1.0
        # Gaussian blur the mask edges
        mask = cv2.GaussianBlur(mask, (feather_width * 2 + 1, feather_width * 2 + 1), 0)
        mask = np.clip(mask, 0, 1)
        mask = mask[:, :, np.newaxis]

        # Blend
        roi = orig[fy:fy+fh, fx:fx+fw].astype(np.float32)
        restored_float = restored.astype(np.float32)
        blended = roi * (1 - mask) + restored_float * mask
        orig[fy:fy+fh, fx:fx+fw] = np.clip(blended, 0, 255).astype(np.uint8)

        cv2.imwrite(output_path, orig)
        return True
    except Exception:
        return False


def run_gfpgan_on_frames(input_dir: str, output_dir: str,
                         model_version: str = "1.4",
                         gpu_id: int = 0) -> bool:
    """Run GFPGAN on a directory of frames."""
    try:
        cmd = [
            sys.executable, "-m", "gfpgan",
            "-i", input_dir,
            "-o", output_dir,
            "-v", model_version,
            "-s", "2",
            "-g", str(gpu_id),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=3600, shell=False)
        return r.returncode == 0
    except Exception:
        return False


def run_codeformer_on_frames(input_dir: str, output_dir: str,
                              gpu_id: int = 0) -> bool:
    """Run CodeFormer on a directory of frames."""
    try:
        cmd = [
            sys.executable, "-m", "codeformer",
            "-i", input_dir,
            "-o", output_dir,
            "--face_upsample",
            "--bg_upsampler", "realesrgan",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=3600, shell=False)
        if r.returncode == 0:
            return True
        # Fallback: try inference via Python one-liner
        cmd2 = [
            sys.executable,
            "-c",
            (
                "import os, glob; "
                "from codeformer.app import inference; "
                "os.makedirs('{}', exist_ok=True); "
                "[inference(f, '{}') for f in glob.glob('{}/*.png')]"
            ).format(output_dir, output_dir, input_dir),
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True,
                           timeout=3600, shell=False)
        return r2.returncode == 0
    except Exception:
        return False


def find_static_face_segments(frames: List[Tuple[str, float]],
                               params: dict, logger) -> List[Dict]:
    """Find static face segments meeting v6.3 criteria."""
    segments: List[Dict[str, Any]] = []
    if len(frames) < 2:
        return segments

    # Config thresholds
    threshold = params.get("face_detect_threshold", 0.8)
    threshold_lowres = params.get("face_detect_threshold_lowres", 0.6)
    lowres_max_dim = params.get("face_detect_lowres_max_dimension", 720)
    min_duration = 2.0  # seconds
    displacement_ratio_threshold = 0.05
    min_face_area_ratio = 0.15
    max_global_motion = 0.03
    models = params.get("face_detect_fallback", ["retinaface", "yolov8", "mtcnn"])

    # Get frame dimensions for area ratio check
    try:
        from PIL import Image
        with Image.open(frames[0][0]) as img:
            frame_w, frame_h = img.size
            frame_area = frame_w * frame_h
    except Exception:
        frame_area = 1920 * 1080

    # Sliding window: accumulate PTS to find >=2s segments
    i = 0
    while i < len(frames):
        fp, pts = frames[i]
        faces = detect_faces_with_fallback(fp, models, threshold, threshold_lowres, lowres_max_dim)

        if not faces:
            i += 1
            continue

        # Filter faces by area ratio (must be > 15% of frame)
        valid_faces = [f for f in faces
                       if (f[2] * f[3]) / frame_area > min_face_area_ratio]
        if not valid_faces:
            i += 1
            continue

        # Try each valid face as a candidate
        for face in valid_faces:
            fx, fy, fw, fh, fc = face
            prev_box = face
            segment_frames = [(fp, face)]
            start_pts = pts
            j = i + 1
            last_valid_pts = pts
            avg_interval = None
            effective_duration = 0.0

            while j < len(frames):
                fp_next, pts_next = frames[j]

                # PTS anomaly: skip if gap > 2x normal interval
                interval = pts_next - last_valid_pts
                if avg_interval is None and j > i + 1:
                    avg_interval = (pts_next - start_pts) / (j - i)
                if avg_interval and interval > avg_interval * 2:
                    break

                # PTS rollback
                if pts_next < last_valid_pts:
                    if logger:
                        logger.error("STAGE5_PTS_ROLLBACK",
                                    f"PTS rollback at frame {j}: {last_valid_pts:.3f} -> {pts_next:.3f}")
                    break

                # Duplicate frame: use avg_interval for effective duration
                effective_interval = interval
                if interval == 0 and avg_interval:
                    effective_interval = avg_interval

                # Detect faces in next frame
                next_faces = detect_faces_with_fallback(
                    fp_next, models, threshold, threshold_lowres, lowres_max_dim
                )
                if not next_faces:
                    break

                # Match to same face
                matched = match_face(prev_box, next_faces)
                if not matched:
                    break

                mx, my, mw, mh, mc = matched

                # Check confidence
                if mc < threshold:
                    break

                # Check displacement relative to face size
                pcx, pcy = prev_box[0] + prev_box[2] // 2, prev_box[1] + prev_box[3] // 2
                mcx, mcy = mx + mw // 2, my + mh // 2
                disp_px = ((mcx - pcx) ** 2 + (mcy - pcy) ** 2) ** 0.5
                face_size = (prev_box[2] + prev_box[3]) / 2
                disp_ratio = disp_px / face_size if face_size > 0 else 1.0

                # Absolute pixel displacement with max(10px, min(50px, 5%))
                abs_threshold = max(10, min(50, face_size * 0.05))
                if disp_px > abs_threshold or disp_ratio > displacement_ratio_threshold:
                    break

                # Check global motion
                global_motion = compute_global_motion(segment_frames[-1][0], fp_next)
                if global_motion > max_global_motion:
                    break

                segment_frames.append((fp_next, matched))
                prev_box = matched
                last_valid_pts = pts_next
                effective_duration += effective_interval
                j += 1

                # Check duration (use effective_duration for duplicate frame tolerance)
                duration = max(last_valid_pts - start_pts, effective_duration)
                if duration >= min_duration:
                    segments.append({
                        "start_idx": i,
                        "end_idx": j - 1,
                        "duration": duration,
                        "frames": segment_frames.copy(),
                        "face": face,
                    })
                    break  # Found one segment starting at i with this face

            # If we found a segment, skip past it
            if segments and segments[-1]["start_idx"] == i:
                i = segments[-1]["end_idx"] + 1
                break
        else:
            i += 1

    return segments


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

    ai_enabled = params.get("ai_enabled", False)
    face_restore = params.get("face_restore", "")
    if not ai_enabled or not face_restore:
        logger.info("STAGE5_SKIP", "Face restore not enabled")
        return 3

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    input_name = Path(args.input).name

    # Find input video: stage4 > stage3 > stage1
    input_video = None
    for pat in ["step4_postprocess*", "step3_upscale*", "step1_preprocess*"]:
        pattern = f"{job_id}_{Path(input_name).stem}_{pat}"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        if candidates:
            input_video = candidates[0]
            break

    if not input_video or not os.path.exists(input_video):
        logger.info("STAGE5_SKIP", "No input video found")
        return 3

    logger.info("STAGE5_START", f"Face restore: {face_restore}")

    # Extract frames with PTS
    frames_dir = os.path.join(temp_dir, f"{job_id}_face_frames")
    frames = extract_frames_with_pts(input_video, frames_dir, logger)
    if not frames:
        logger.error("STAGE5_NO_FRAMES", "Failed to extract frames")
        return 82

    logger.info("STAGE5_FRAMES", f"Extracted {len(frames)} frames")

    # Find static face segments (PTS-based)
    segments = find_static_face_segments(frames, params, logger)

    # Supplement with static-frame-range detection (SSIM-based)
    # v6.3: consecutive >=25 frames with SSIM > 0.99 are treated as static regardless of PTS
    static_ranges = find_static_frame_ranges(frames, min_frames=25, ssim_threshold=0.99)
    if static_ranges:
        for start_idx, end_idx in static_ranges:
            # Check if this range already overlaps with a found segment
            overlap = False
            for seg in segments:
                if not (end_idx < seg["start_idx"] or start_idx > seg["end_idx"]):
                    overlap = True
                    break
            if overlap:
                continue
            # Detect face in the middle frame of this static range
            mid_idx = (start_idx + end_idx) // 2
            mid_fp = frames[mid_idx][0]
            mid_faces = detect_faces_with_fallback(
                mid_fp,
                params.get("face_detect_fallback", ["retinaface", "yolov8", "mtcnn"]),
                params.get("face_detect_threshold", 0.8),
                params.get("face_detect_threshold_lowres", 0.6),
                params.get("face_detect_lowres_max_dimension", 720)
            )
            if mid_faces:
                # Use the first/largest face
                face = mid_faces[0]
                seg_frames = [(frames[k][0], face) for k in range(start_idx, end_idx + 1)]
                duration = params.get("static_segment_duration", 2.0)
                segments.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "duration": duration,
                    "frames": seg_frames,
                    "face": face,
                    "source": "ssim_static_range",
                })
                logger.info("STAGE5_STATIC_RANGE",
                           f"Static frame range {start_idx}-{end_idx} added via SSIM detection")

    if not segments:
        logger.info("STAGE5_NO_SEGMENTS", "No static face segments found")
        return 3

    logger.info("STAGE5_SEGMENTS", f"Found {len(segments)} static face segments")

    # Process each segment
    output_dir = os.path.join(temp_dir, f"{job_id}_face_restored")
    os.makedirs(output_dir, exist_ok=True)

    processed_any = False
    for seg_idx, seg in enumerate(segments):
        seg_frames = seg["frames"]
        if not seg_frames:
            continue

        # Prepare input directory for GFPGAN
        seg_input_dir = os.path.join(temp_dir, f"{job_id}_seg{seg_idx}_input")
        seg_output_dir = os.path.join(temp_dir, f"{job_id}_seg{seg_idx}_output")
        os.makedirs(seg_input_dir, exist_ok=True)
        os.makedirs(seg_output_dir, exist_ok=True)

        # Copy frames to input dir
        for fidx, (fp, face) in enumerate(seg_frames):
            shutil.copy2(fp, os.path.join(seg_input_dir, f"frame_{fidx:04d}.png"))

        # Run face restoration
        if face_restore == "GFPGAN":
            ok = run_gfpgan_on_frames(seg_input_dir, seg_output_dir)
        elif face_restore == "CodeFormer":
            ok = run_codeformer_on_frames(seg_input_dir, seg_output_dir)
        else:
            logger.warning("STAGE5_MODEL_SKIP", f"{face_restore} not supported")
            ok = False

        if not ok:
            continue

        # Find restored frames
        restored_files = sorted(glob.glob(os.path.join(seg_output_dir, "**", "*.png"),
                                           recursive=True))
        if not restored_files:
            continue

        # Feather blend each restored face back
        for fidx, (orig_fp, face_box) in enumerate(seg_frames):
            if fidx >= len(restored_files):
                break
            # Overwrite original frame with blended version
            feather_blend(restored_files[fidx], orig_fp,
                          (face_box[0], face_box[1], face_box[2], face_box[3]),
                          orig_fp, feather_width=8)

        processed_any = True
        logger.info("STAGE5_SEGMENT_DONE",
                   f"Segment {seg_idx}: {len(seg_frames)} frames restored")

    if not processed_any:
        logger.info("STAGE5_NO_RESTORE", "No segments were successfully restored")
        return 3

    # Re-encode frames to video
    output_video = os.path.join(temp_dir,
        f"{job_id}_{Path(input_name).stem}_step5_face.mkv")

    # Get frame rate from input video
    fps_cmd = [
        "ffprobe", "-hide_banner", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_video
    ]
    fps_str = "25"
    try:
        r = subprocess.run(fps_cmd, capture_output=True, text=True,
                          timeout=30, shell=False)
        if r.returncode == 0:
            fps_str = r.stdout.strip()
    except Exception:
        pass

    encode_cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-framerate", fps_str,
        "-i", os.path.join(frames_dir, "frame_%08d.png"),
        "-c:v", "ffv1",
        "-pix_fmt", "yuv420p16le",
        output_video
    ]
    try:
        r = subprocess.run(encode_cmd, capture_output=True, timeout=3600, shell=False)
        if r.returncode != 0:
            logger.error("STAGE5_ENCODE_FAIL", f"Re-encode failed: {r.stderr.decode()[:200]}")
            return 84
    except Exception as e:
        logger.error("STAGE5_ENCODE_EXCEPT", f"Re-encode exception: {e}")
        return 84

    if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
        logger.info("STAGE5_DONE", "Face restore complete", output=output_video)
        return 0
    else:
        logger.error("STAGE5_NO_OUTPUT", "Output video not created")
        return 82


if __name__ == "__main__":
    sys.exit(main() or 0)
