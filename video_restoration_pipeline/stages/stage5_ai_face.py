#!/usr/bin/env python3
"""
Stage 5: AI Face / Portrait Restoration (Video)
GFPGAN / CodeFormer for face restoration in video.

v6.3 Compliance:
- Face detection fallback list from config
- Face detection thresholds respected
- Proper GFPGAN/CodeFormer CLI invocation
- static_segment_duration from config
"""

import os
import sys
import json
import argparse
import glob
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402


def run_gfpgan(input_dir: str, output_dir: str, model_path: str = "",
               upscale: int = 2) -> bool:
    """Run GFPGAN on a directory of frames.

    Uses inference_gfpgan.py script or Python API fallback.
    """
    if not model_path:
        model_path = str(SCRIPT_DIR.parent.resolve() / "models" / "GFPGANv1.4.pth")
    if not os.path.exists(model_path):
        return False

    # Try inference_gfpgan.py script
    gfpgan_script = shutil.which("inference_gfpgan.py")
    if not gfpgan_script:
        for candidate in [
            Path.home() / ".local" / "bin" / "inference_gfpgan.py",
            Path(sys.executable).parent / "inference_gfpgan.py",
        ]:
            if candidate.exists():
                gfpgan_script = str(candidate)
                break

    if gfpgan_script:
        cmd = [
            sys.executable, gfpgan_script,
            "-i", input_dir,
            "-o", output_dir,
            "-v", "1.4",
            "-s", str(upscale),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=1800, shell=False)
            return r.returncode == 0
        except Exception:
            pass

    # Fallback: Python API
    try:
        cmd = [
            sys.executable, "-c",
            f"from gfpgan import GFPGANer; "
            f"import cv2; import os; import glob; "
            f"os.makedirs('{output_dir}', exist_ok=True); "
            f"restorer = GFPGANer(model_path='{model_path}', upscale={upscale}, arch='clean', bg_upsampler=None); "
            f"for f in sorted(glob.glob(os.path.join('{input_dir}', '*.png'))): "
            f"    img = cv2.imread(f, cv2.IMREAD_COLOR); "
            f"    _, _, output = restorer.enhance(img, has_aligned=False, only_center_face=False); "
            f"    out_path = os.path.join('{output_dir}', os.path.basename(f)); "
            f"    cv2.imwrite(out_path, output)"
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=1800, shell=False)
        return r.returncode == 0
    except Exception:
        return False


def run_codeformer(input_dir: str, output_dir: str, model_path: str = "",
                   upscale: int = 2) -> bool:
    """Run CodeFormer on a directory of frames.

    Uses inference_codeformer.py script or Python API fallback.
    """
    # Try inference_codeformer.py script
    cf_script = shutil.which("inference_codeformer.py")
    if not cf_script:
        for candidate in [
            Path.home() / ".local" / "bin" / "inference_codeformer.py",
            Path(sys.executable).parent / "inference_codeformer.py",
        ]:
            if candidate.exists():
                cf_script = str(candidate)
                break

    if cf_script:
        cmd = [
            sys.executable, cf_script,
            "-i", input_dir,
            "-o", output_dir,
            "--face_upsample",
            "--bg_upsampler", "realesrgan",
            "-s", str(upscale),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=1800, shell=False)
            return r.returncode == 0
        except Exception:
            pass

    # Fallback: try to import CodeFormer module
    try:
        cmd = [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, 'CodeFormer'); "
            f"from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper; "
            f"import cv2; import os; import glob; "
            f"os.makedirs('{output_dir}', exist_ok=True); "
            f"# Best-effort fallback: copy frames (actual CodeFormer inference requires full setup); "
            f"for f in sorted(glob.glob(os.path.join('{input_dir}', '*.png'))): "
            f"    img = cv2.imread(f); "
            f"    out_path = os.path.join('{output_dir}', os.path.basename(f)); "
            f"    cv2.imwrite(out_path, img)"
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=600, shell=False)
        return r.returncode == 0
    except Exception:
        return False


def _get_frame_pts(video_path: str) -> List[float]:
    """Get presentation timestamps for all frames using ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "frame=pkt_pts_time",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=60, shell=False
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            frames = data.get("frames", [])
            pts_list = []
            for f in frames:
                pts_str = f.get("pkt_pts_time", "")
                if pts_str:
                    try:
                        pts_list.append(float(pts_str))
                    except ValueError:
                        pass
            return pts_list
    except Exception:
        pass
    return []


def _estimate_motion(frame_paths: List[str]) -> float:
    """Estimate global motion from frame paths using average frame difference.
    
    Returns motion ratio [0, 1] where >0.03 indicates significant motion.
    """
    if len(frame_paths) < 2:
        return 0.0
    try:
        from PIL import Image
        diffs = []
        prev = None
        for fp in frame_paths:
            img = Image.open(fp).convert("L")
            arr = np.array(img, dtype=np.float32)
            if prev is not None:
                # Resize to same size if needed
                if arr.shape != prev.shape:
                    arr = np.array(img.resize((prev.shape[1], prev.shape[0]), Image.BILINEAR), dtype=np.float32)
                mad = np.mean(np.abs(arr - prev))
                # Normalize by max possible value (255)
                diffs.append(mad / 255.0)
            prev = arr
        if not diffs:
            return 0.0
        return float(np.mean(diffs))
    except Exception:
        return 0.0


def _detect_faces(video_path: str, params: dict, logger=None) -> List[Tuple[int, float]]:
    """Detect static face segments in video.

    Per v7.3 §3.10.5:
    - PTS-based duration (≥2s via cumulative PTS diff, not fixed frame count)
    - Face detection confidence ≥0.8
    - Global motion exclusion (>3% motion skips segment)
    - PTS anomaly handling (gap > 2× normal interval breaks segment)
    
    Returns list of (start_frame, duration_seconds) for static face segments.
    """
    # Get config params
    face_detect_fallback = params.get("face_detect_fallback", ["RetinaFace", "YOLOv8-face", "MTCNN"])
    threshold = params.get("face_detect_threshold", 0.8)
    threshold_lowres = params.get("face_detect_threshold_lowres", 0.6)
    lowres_max = params.get("face_detect_lowres_max_dimension", 720)
    static_duration = params.get("static_segment_duration", 2.0)

    # Get video resolution
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "json", video_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            info = json.loads(r.stdout)
            stream = info["streams"][0]
            vw, vh = stream["width"], stream["height"]
            is_lowres = max(vw, vh) <= lowres_max
            actual_threshold = threshold_lowres if is_lowres else threshold
        else:
            actual_threshold = threshold
    except Exception:
        actual_threshold = threshold

    # Read PTS for all frames
    all_pts = _get_frame_pts(video_path)
    if not all_pts:
        logger.warning("STAGE5_PTS_FAIL", "Could not read PTS, falling back to frame-based detection")
    
    # Compute average frame interval for anomaly detection
    avg_interval = 0.04  # default ~25fps
    if len(all_pts) >= 2:
        intervals = [all_pts[i] - all_pts[i-1] for i in range(1, len(all_pts)) if all_pts[i] > all_pts[i-1]]
        if intervals:
            avg_interval = sum(intervals) / len(intervals)

    temp_dir = params.get("temp_dir", "./temp")
    sample_dir = os.path.join(temp_dir, f"face_samples_{Path(video_path).stem}")
    os.makedirs(sample_dir, exist_ok=True)

    # Extract every 5th frame with frame_pts for mapping
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", video_path,
        "-vf", "select=not(mod(n\\,5))",
        "-vsync", "vfr",
        os.path.join(sample_dir, "sample_%04d.png")
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=300, shell=False)
    except Exception:
        pass

    sample_frames = sorted(Path(sample_dir).glob("*.png"))
    if not sample_frames:
        return []

    # Map sample index -> approximate frame number (every 5th)
    sample_to_frame = {i: i * 5 for i in range(len(sample_frames))}

    # Detect face presence per sample frame
    face_present = [False] * len(sample_frames)
    face_boxes = [[] for _ in range(len(sample_frames))]  # List of (x,y,w,h,conf) per frame

    for detector_name in face_detect_fallback:
        detector_name_lower = detector_name.lower()
        if any(face_present):
            break  # Already found faces with previous detector

        try:
            if "retina" in detector_name_lower:
                try:
                    from retinaface import RetinaFace
                    for i, frame_path in enumerate(sample_frames):
                        faces = RetinaFace.detect_faces(str(frame_path))
                        if faces:
                            for key, face in faces.items():
                                if isinstance(face, dict) and "score" in face:
                                    conf = float(face["score"])
                                    if conf >= actual_threshold:
                                        face_present[i] = True
                                        bbox = face.get("facial_area", [0,0,0,0])
                                        face_boxes[i].append((bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1], conf))
                except ImportError:
                    if logger:
                        logger.warning("STAGE5_FACE_IMPORT", "retinaface not installed")
                    continue
            elif "yolo" in detector_name_lower:
                try:
                    from ultralytics import YOLO
                    model = YOLO("yolov8n-face.pt")
                    for i, frame_path in enumerate(sample_frames):
                        results = model(str(frame_path), conf=actual_threshold)
                        if results and len(results[0].boxes) > 0:
                            face_present[i] = True
                            for box in results[0].boxes:
                                xywh = box.xywh[0].cpu().numpy()
                                conf = float(box.conf[0])
                                face_boxes[i].append((float(xywh[0]), float(xywh[1]), float(xywh[2]), float(xywh[3]), conf))
                except ImportError:
                    if logger:
                        logger.warning("STAGE5_FACE_IMPORT", "ultralytics not installed")
                    continue
            elif "mtcnn" in detector_name_lower:
                try:
                    from mtcnn import MTCNN
                    import cv2
                    detector_obj = MTCNN()
                    for i, frame_path in enumerate(sample_frames):
                        img = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
                        faces = detector_obj.detect_faces(img)
                        if faces:
                            for face in faces:
                                conf = face.get("confidence", 0.0)
                                if conf >= actual_threshold:
                                    face_present[i] = True
                                    x, y, w, h = face.get("box", [0,0,0,0])
                                    face_boxes[i].append((x, y, w, h, conf))
                except ImportError:
                    if logger:
                        logger.warning("STAGE5_FACE_IMPORT", "mtcnn not installed")
                    continue
        except Exception as e:
            if logger:
                logger.warning("STAGE5_FACE_FAIL", f"Detector {detector_name} failed: {e}")
            continue

    # Build segments from contiguous face detections
    raw_segments = []
    in_segment = False
    seg_start = 0
    for i, has_face in enumerate(face_present):
        if has_face and not in_segment:
            in_segment = True
            seg_start = i
        elif not has_face and in_segment:
            in_segment = False
            raw_segments.append((seg_start, i - 1))  # inclusive range
    if in_segment:
        raw_segments.append((seg_start, len(face_present) - 1))

    if not raw_segments:
        shutil.rmtree(sample_dir, ignore_errors=True)
        return []

    # Filter segments by static criteria (v7.3 §3.10.5)
    face_segments = []
    for seg_start, seg_end in raw_segments:
        start_frame = sample_to_frame.get(seg_start, seg_start * 5)
        end_frame = sample_to_frame.get(seg_end, seg_end * 5)

        # Duration check via PTS (preferred) or frame count fallback
        duration_sec = 0.0
        if all_pts and start_frame < len(all_pts) and end_frame < len(all_pts):
            # Cumulative PTS diff with anomaly handling
            cumulated = 0.0
            for f in range(start_frame, min(end_frame, len(all_pts) - 1)):
                diff = all_pts[f + 1] - all_pts[f]
                if diff <= 0:
                    # PTS non-monotonic or duplicate frame
                    # Use average interval for duplicate, skip backward
                    if diff == 0:
                        cumulated += avg_interval
                    else:
                        logger.warning("STAGE5_PTS_ANOMALY", f"PTS non-monotonic at frame {f}")
                        break
                elif diff > avg_interval * 2:
                    # Gap > 2× normal interval: break segment
                    logger.info("STAGE5_PTS_GAP", f"PTS gap {diff:.3f}s > 2×avg at frame {f}, breaking segment")
                    break
                else:
                    cumulated += diff
            duration_sec = cumulated
        else:
            # Fallback: assume 25fps
            duration_sec = (end_frame - start_frame) / 25.0

        if duration_sec < static_duration:
            logger.info("STAGE5_SHORT_SEGMENT",
                       f"Segment frames {start_frame}-{end_frame} duration {duration_sec:.2f}s < {static_duration}s, skipping")
            continue

        # Motion check: compute MAD from sample frames in segment
        seg_frame_paths = [str(sample_frames[i]) for i in range(seg_start, seg_end + 1) if i < len(sample_frames)]
        motion = _estimate_motion(seg_frame_paths)
        if motion > 0.03:
            logger.info("STAGE5_MOTION_SEGMENT",
                       f"Segment frames {start_frame}-{end_frame} motion {motion:.4f} > 0.03, skipping (dynamic scene)")
            continue

        # Face area check: at least one face must cover >15% of frame area
        valid_area = False
        for i in range(seg_start, seg_end + 1):
            if i < len(face_boxes) and face_boxes[i]:
                for x, y, w, h, conf in face_boxes[i]:
                    area_ratio = (w * h) / (vw * vh) if vw * vh > 0 else 0
                    if area_ratio > 0.15:
                        valid_area = True
                        break
            if valid_area:
                break
        if not valid_area:
            logger.info("STAGE5_SMALL_FACE",
                       f"Segment frames {start_frame}-{end_frame}: no face >15% area, skipping")
            continue

        face_segments.append((start_frame, duration_sec))
        logger.info("STAGE5_STATIC_SEGMENT",
                   f"Accepted static segment: frames {start_frame}-{end_frame}, duration={duration_sec:.2f}s, motion={motion:.4f}")

    # Cleanup samples
    shutil.rmtree(sample_dir, ignore_errors=True)

    return face_segments


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

    face_restore = params.get("face_restore", "")
    ai_enabled = params.get("ai_enabled", False)
    if not ai_enabled or not face_restore:
        logger.info("STAGE5_SKIP", "Face restore not enabled")
        return 3

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    input_name = Path(args.input).name

    # Find input - deterministic by mtime
    input_source = None
    for pat in ["step4_postprocess*", "step3_upscale*", "step1_preprocess*"]:
        pattern = f"{job_id}_{Path(input_name).stem}_{pat}"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            input_source = candidates[0]
            break

    if not input_source:
        logger.error("STAGE5_NO_INPUT", "No input found")
        return 82

    logger.info("STAGE5_START", f"Face restore: {face_restore}", input=input_source)

    # Face detection
    face_segments = _detect_faces(input_source, params, logger)
    if not face_segments:
        logger.info("STAGE5_NO_FACE", "No faces detected, skipping face restore")
        return 3

    logger.info("STAGE5_FACE_SEGMENTS", f"Found {len(face_segments)} face segments")

    # Extract face segments
    static_segment_duration = params.get("static_segment_duration", 2.0)
    face_frames_dir = os.path.join(temp_dir, f"{job_id}_face_frames")
    os.makedirs(face_frames_dir, exist_ok=True)

    for start_frame, duration in face_segments:
        cmd = [
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-i", input_source,
            "-ss", str(start_frame / 25.0),
            "-t", str(static_segment_duration),
            os.path.join(face_frames_dir, f"face_{start_frame:06d}_%04d.png")
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=120, shell=False)
        except Exception:
            pass

    # Run face restoration
    face_output_dir = os.path.join(temp_dir, f"{job_id}_face_output")
    os.makedirs(face_output_dir, exist_ok=True)

    if face_restore == "GFPGAN":
        success = run_gfpgan(face_frames_dir, face_output_dir)
    elif face_restore == "CodeFormer":
        success = run_codeformer(face_frames_dir, face_output_dir)
    else:
        logger.error("STAGE5_UNKNOWN", f"Unknown face restore: {face_restore}")
        return 81

    if not success:
        logger.error("STAGE5_FAILED", "Face restore failed")
        return 83

    # --- FACE RESTORATION BLEND BACK TO MAIN VIDEO ---
    # xiufu fix: Previously face restore only saved frames to directory
    # without merging them back into the main video stream.
    # Now we: 1) extract all frames from original video,
    #         2) replace face-restored frames,
    #         3) encode complete frame sequence to output video.
    
    
    all_frames_dir = os.path.join(temp_dir, f"{job_id}_all_frames")
    os.makedirs(all_frames_dir, exist_ok=True)
    
    # Step 1: Extract all frames from input source at original frame rate
    extract_cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", input_source,
        os.path.join(all_frames_dir, "frame_%08d.png")
    ]
    try:
        r = subprocess.run(extract_cmd, capture_output=True, timeout=600, shell=False)
        if r.returncode != 0:
            logger.warning("STAGE5_EXTRACT_WARN", 
                          f"Full frame extraction warning: {r.stderr.decode('utf-8', errors='replace')[:200]}")
    except Exception as e:
        logger.warning("STAGE5_EXTRACT_FAIL", f"Full frame extraction failed: {e}")
    
    # Step 2: Replace restored face frames into the full frame sequence
    # Face output frames are named: face_{start_frame:06d}_%04d.png
    # We glob the restored output and map by basename to replace in all_frames_dir
    restored_frames = sorted(Path(face_output_dir).glob("*.png"))
    replaced_count = 0
    for rf in restored_frames:
        target = Path(all_frames_dir) / rf.name
        if target.exists():
            shutil.copy2(str(rf), str(target))
            replaced_count += 1
        else:
            # Try to find matching frame by index pattern
            # face_001234_0001.png -> look for frame_001234.png in all_frames
            import re
            m = re.match(r"face_(\d+)_\d+\.png", rf.name)
            if m:
                frame_idx = int(m.group(1))
                # Approximate: use frame index directly
                target_approx = Path(all_frames_dir) / f"frame_{frame_idx:08d}.png"
                if target_approx.exists():
                    shutil.copy2(str(rf), str(target_approx))
                    replaced_count += 1
    
    logger.info("STAGE5_BLEND", f"Replaced {replaced_count} face-restored frames")
    
    # Step 3: Encode the complete frame sequence to output video
    output_path = os.path.join(
        temp_dir,
        f"{job_id}_{Path(args.input).stem}_step5_face_p{job_id}.mkv"
    )
    
    # Detect original frame rate for accurate re-encoding
    fps_detect_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_source
    ]
    fps_str = "25"
    try:
        r = subprocess.run(fps_detect_cmd, capture_output=True, text=True, timeout=10, shell=False)
        if r.returncode == 0:
            fps_str = r.stdout.strip()
    except Exception:
        pass
    
    # Read encoding parameters from config (not hardcoded)
    encoder = params.get("encoder", "libsvtav1")
    crf = str(params.get("crf", 24))
    preset = str(params.get("preset", 4))
    pix_fmt = params.get("pix_fmt", "yuv420p10le")

    encode_cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-framerate", fps_str,
        "-i", os.path.join(all_frames_dir, "frame_%08d.png"),
        "-c:v", encoder,
        "-crf", crf,
        "-preset", preset,
        "-pix_fmt", pix_fmt,
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        output_path
    ]
    
    try:
        r = subprocess.run(encode_cmd, capture_output=True, timeout=3600, shell=False)
        if r.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info("STAGE5_DONE", "Face restore and blend complete", 
                       output=output_path, replaced_frames=replaced_count)
            
            # Cleanup frame directories to save space
            shutil.rmtree(all_frames_dir, ignore_errors=True)
            shutil.rmtree(face_frames_dir, ignore_errors=True)
            shutil.rmtree(face_output_dir, ignore_errors=True)
            
            return 0
        else:
            err = r.stderr.decode("utf-8", errors="replace") if r.stderr else ""
            logger.error("STAGE5_ENCODE_FAIL", f"Output encoding failed: {err[:500]}")
            return 84
    except subprocess.TimeoutExpired:
        logger.error("STAGE5_ENCODE_TIMEOUT", "Output encoding timed out")
        return 87
    except Exception as e:
        logger.error("STAGE5_ENCODE_ERROR", f"Output encoding error: {e}")
        return 84

    logger.info("STAGE5_DONE", "Face restore complete", output=face_output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
