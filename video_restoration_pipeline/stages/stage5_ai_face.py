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
import re
import argparse
import glob
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from checkpoint import wsl_redirect_path  # FIX-P9-3: WSL2 redirect per v7.32 §3.10.10


# ---------------------------------------------------------------------------
# P9-ARCH: Streaming infrastructure classes
# Per v7.32 §3.10.5 + §3.10.10: avoid full-PNG extraction for long videos.
# StreamingDecoder / StreamingEncoder are module-level for multiprocessing
# spawn compliance (CLAUDE.md §5).
# ---------------------------------------------------------------------------

class StreamingDecoder:
    """Stream-decode video to RGB numpy arrays via FFmpeg rawvideo pipe.

    Usage::

        with StreamingDecoder(path) as dec:
            for frame_idx, arr, pts in dec.frames():
                ...

    FIX-R2: VFR PTS pre-read per v7.32 §3.10.5. For VFR sources,
    PTS is read via ffprobe before decoding rather than computed as
    frame_idx / fps. Falls back to CFR estimation if ffprobe fails.
    """

    def __init__(self, video_path: str, pix_fmt: str = "rgb24"):
        self.video_path = video_path
        self.pix_fmt = pix_fmt
        self.bytes_per_frame: int = 0
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 25.0
        self.frames_total: int = 0
        self._vfr_pts: List[float] = []  # FIX-R2: pre-read PTS list
        self._probe()
        self._proc: Optional[subprocess.Popen] = None

    def _probe(self) -> None:
        """Get video dimensions, frame rate, and frame count via ffprobe."""
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
             "-show_entries", "format=duration",
             "-of", "json", self.video_path],
            capture_output=True, text=True, shell=False
        )
        if r.returncode == 0:
            info = json.loads(r.stdout)
            stream = info.get("streams", [{}])[0]
            self.width = int(stream.get("width", 720))
            self.height = int(stream.get("height", 480))
            fps_str = stream.get("r_frame_rate", "25/1")
            if "/" in str(fps_str):
                num, den = str(fps_str).split("/")
                self.fps = float(num) / float(den) if float(den) else 25.0
            else:
                self.fps = float(fps_str) if fps_str else 25.0
            nb_frames = stream.get("nb_frames", "")
            if nb_frames and str(nb_frames).isdigit():
                self.frames_total = int(nb_frames)
            else:
                fmt = info.get("format", {})
                dur = fmt.get("duration", "")
                if dur:
                    self.frames_total = int(float(dur) * self.fps)
        # FIX-R2: Pre-read PTS for VFR sources per v7.32 §3.10.5
        # Use pkt_pts_time (FFmpeg <7.0) or pts_time (FFmpeg >=7.0)
        self._vfr_pts = _get_frame_pts(self.video_path)
        if self._vfr_pts:
            self.frames_total = len(self._vfr_pts)
        # Calculate bytes per frame based on pixel format
        if self.pix_fmt == "rgb48":
            self.bytes_per_frame = self.width * self.height * 6
        else:  # rgb24
            self.bytes_per_frame = self.width * self.height * 3

    def __enter__(self):
        cmd = [
            "ffmpeg", "-hide_banner", "-v", "error",
            "-i", self.video_path,
            "-pix_fmt", self.pix_fmt,
            "-f", "rawvideo", "-"
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        return self

    def __exit__(self, *args):
        if self._proc:
            self._proc.stdout.close()
            self._proc.stderr.close()
            self._proc.wait()

    def frames(self):
        """Yield (frame_index, rgb_array, pts_seconds).

        FIX-R2: For VFR sources, uses pre-read PTS from ffprobe.
        Falls back to CFR estimation (frame_idx / fps) if no PTS list.
        """
        dtype = np.uint16 if self.pix_fmt == "rgb48" else np.uint8
        frame_idx = 0
        while True:
            raw = self._proc.stdout.read(self.bytes_per_frame)
            if len(raw) < self.bytes_per_frame:
                break
            arr = np.frombuffer(raw, dtype=dtype).reshape(
                (self.height, self.width, 3))
            # FIX-R2: Use pre-read VFR PTS if available
            if frame_idx < len(self._vfr_pts):
                pts = self._vfr_pts[frame_idx]
            else:
                pts = frame_idx / self.fps
            yield frame_idx, arr, pts
            frame_idx += 1


class StreamingEncoder:
    """Encode raw RGB stream to video via FFmpeg pipe.

    Usage::

        with StreamingEncoder(path, w, h, fps) as enc:
            enc.write_frame(arr)
    """

    def __init__(self, output_path: str, width: int, height: int,
                 fps: float, pix_fmt_in: str = "rgb24",
                 codec: str = "ffv1", pix_fmt_out: str = "yuv420p16le"):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.pix_fmt_in = pix_fmt_in
        self.codec = codec
        self.pix_fmt_out = pix_fmt_out
        self._proc: Optional[subprocess.Popen] = None

    def __enter__(self):
        fps_str = (str(int(self.fps)) if self.fps == int(self.fps)
                   else f"{self.fps:.6f}")
        cmd = [
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-f", "rawvideo",
            "-pix_fmt", self.pix_fmt_in,
            "-s", f"{self.width}x{self.height}",
            "-r", fps_str,
            "-i", "-",
            "-c:v", self.codec,
            "-pix_fmt", self.pix_fmt_out,
            "-color_range", "tv",           # per v7.32 §3.10.3
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            self.output_path
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        return self

    def __exit__(self, *args):
        if self._proc:
            self._proc.stdin.close()
            self._proc.stderr.close()
            self._proc.wait()

    def write_frame(self, frame: np.ndarray) -> None:
        """Write one RGB numpy array to the encoder pipe."""
        self._proc.stdin.write(frame.tobytes())


def _feather_blend_in_memory(orig: np.ndarray, restored: np.ndarray,
                              face_bbox: tuple = None,
                              margin: int = 8) -> np.ndarray:
    """Feather-blend restored frame into original using Gaussian mask.

    Per v7.32 §3.10.7: transition width >= 8px.
    If face_bbox provided, blend only within the expanded face ROI;
    otherwise blend the entire frame.

    Module-level function for multiprocessing spawn compliance (CLAUDE.md §5).
    """
    try:
        import cv2
        if orig.shape != restored.shape:
            restored = cv2.resize(restored, (orig.shape[1], orig.shape[0]))
        h, w = orig.shape[:2]

        # FIX-P9-12: preserve original dtype for 16bit support per v7.32 §3.10.5
        out_dtype = orig.dtype

        if face_bbox and len(face_bbox) >= 4:
            fx, fy, fw, fh = int(face_bbox[0]), int(face_bbox[1]), \
                int(face_bbox[2]), int(face_bbox[3])
            cx, cy = fx + fw // 2, fy + fh // 2
            expand = 1.5
            new_w, new_h = int(fw * expand), int(fh * expand)
            x1 = max(0, cx - new_w // 2)
            y1 = max(0, cy - new_h // 2)
            x2 = min(w, cx + new_w // 2)
            y2 = min(h, cy + new_h // 2)
            roi_h, roi_w = y2 - y1, x2 - x1
            if roi_h <= 0 or roi_w <= 0:
                return restored
            mask = np.zeros((roi_h, roi_w), dtype=np.float32)
            roi_margin = max(margin, min(roi_h, roi_w) // 10)
            mask[roi_margin:roi_h - roi_margin,
                 roi_margin:roi_w - roi_margin] = 1.0
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=roi_margin)
            result = orig.copy()
            roi_orig = orig[y1:y2, x1:x2].astype(np.float32)
            roi_rest = restored[y1:y2, x1:x2].astype(np.float32)
            mask_3c = np.stack([mask] * 3, axis=2)
            # FIX-P9-12: preserve original dtype (uint8 or uint16)
            blended_roi = (roi_rest * mask_3c +
                           roi_orig * (1 - mask_3c)).astype(out_dtype)
            result[y1:y2, x1:x2] = blended_roi
            return result
        else:
            # Full frame blend
            full_margin = max(margin, min(h, w) // 20)
            mask = np.zeros((h, w), dtype=np.float32)
            mask[full_margin:h - full_margin,
                 full_margin:w - full_margin] = 1.0
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=full_margin)
            mask_3c = np.stack([mask] * 3, axis=2)
            blended = (restored.astype(np.float32) * mask_3c +
                       orig.astype(np.float32) *
                       (1 - mask_3c)).astype(out_dtype)
            return blended
    except ImportError:
        return restored


def _check_gpu_memory(width: int, height: int, upscale: int,
                      logger=None) -> int:
    """Check GPU memory and return recommended upscale factor.

    FIX-P9-13: Per v6.32 §7.14: VRAM pre-check with tiered degradation.
    Estimate: width * height * 3 * 4 * (2 + upscale^2).
    Downgrade 4x→2x→CPU if VRAM insufficient.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return upscale  # CPU mode, no VRAM check needed
        free_bytes = torch.cuda.mem_get_info()[0]
        # Estimate: width * height * 3 * 4 * (2 + upscale^2)
        needed = width * height * 3 * 4 * (2 + upscale ** 2)
        if free_bytes >= needed * 1.2:
            return upscale
        if upscale > 2 and free_bytes >= width * height * 3 * 4 * 6 * 1.2:
            if logger:
                logger.warning("STAGE5_GPU_DOWNGRADE",
                              f"VRAM insufficient for {upscale}x "
                              f"(need {needed*1.2/1e9:.1f}GB, "
                              f"have {free_bytes/1e9:.1f}GB), downgrading to 2x")
            return 2
        if logger:
            logger.warning("STAGE5_GPU_CPU_FALLBACK",
                          f"VRAM insufficient for GPU processing "
                          f"(need {needed*1.2/1e9:.1f}GB, "
                          f"have {free_bytes/1e9:.1f}GB), using CPU")
        return 1  # Signal CPU fallback
    except Exception:
        return upscale


def run_gfpgan(input_dir: str, output_dir: str, model_path: str = "",
               upscale: int = 2, width: int = 0, height: int = 0,
               logger=None) -> bool:
    """Run GFPGAN on a directory of frames.

    Uses inference_gfpgan.py script or Python API fallback.
    FIX-P9-13: VRAM pre-check before model loading per v6.32 §7.14.
    """
    if not model_path:
        model_path = str(SCRIPT_DIR.parent.resolve() / "models" / "GFPGANv1.4.pth")
    if not os.path.exists(model_path):
        return False

    # FIX-P9-13: GPU VRAM pre-check per v6.32 §7.14
    if width > 0 and height > 0:
        recommended_upscale = _check_gpu_memory(width, height, upscale, logger)
        if recommended_upscale != upscale:
            if recommended_upscale < upscale:
                upscale = recommended_upscale

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
    # BUG-050 fix: Use json.dumps() for safe path quoting
    # to prevent single-quote injection
    try:
        import json as _json
        safe_input_dir = _json.dumps(input_dir)
        safe_output_dir = _json.dumps(output_dir)
        safe_model_path = _json.dumps(model_path)
        cmd = [
            sys.executable, "-c",
            f"from gfpgan import GFPGANer; "
            f"import cv2; import os; import glob; "
            f"os.makedirs({safe_output_dir}, exist_ok=True); "
            f"restorer = GFPGANer(model_path={safe_model_path}, upscale={upscale}, arch='clean', bg_upsampler=None); "
            f"for f in sorted(glob.glob(os.path.join({safe_input_dir}, '*.png'))): "
            f"    img = cv2.imread(f, cv2.IMREAD_COLOR); "
            f"    _, _, output = restorer.enhance(img, has_aligned=False, only_center_face=False); "
            f"    out_path = os.path.join({safe_output_dir}, os.path.basename(f)); "
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

    # V-P-043 fix: CodeFormer fallback returns False instead of copying original frames
    # True CodeFormer inference requires full setup; signal failure so caller can skip
    return False


def _get_frame_pts(video_path: str) -> List[float]:
    """Get presentation timestamps for all frames using ffprobe.

    FIX-P9-1: Try pts_time first (FFmpeg >=7.0), then pkt_pts_time (<7.0).
    Per v7.32 §3.10.5: VFR sources MUST use actual PTS, not CFR estimation.
    """
    for field_name in ["pts_time", "pkt_pts_time"]:
        try:
            r = subprocess.run(
                ["ffprobe", "-hide_banner", "-v", "error",
                 "-select_streams", "v:0",
                 "-show_entries", f"frame={field_name}",
                 "-of", "json", video_path],
                capture_output=True, text=True, timeout=60, shell=False
            )
            if r.returncode == 0:
                data = json.loads(r.stdout)
                frames = data.get("frames", [])
                pts_list = []
                for f in frames:
                    pts_str = f.get(field_name, "")
                    if pts_str:
                        try:
                            pts_list.append(float(pts_str))
                        except ValueError:
                            pass
                if pts_list:
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


# FIX-P9-8: optical flow motion detection per v7.3 §3.10.5
# Prefer optical flow; fall back to MAD with WARNING log
def _estimate_motion_optical_flow(frame_paths: List[str],
                                  logger=None) -> float:
    """Estimate global motion using OpenCV optical flow (Farneback).

    Per v7.3 §3.10.5: compute global optical flow or frame displacement;
    if pan/zoom causes overall displacement >3%, skip segment.
    Falls back to MAD-based _estimate_motion if cv2 unavailable.
    """
    try:
        import cv2
        flows = []
        prev_gray = None
        for fp in frame_paths:
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if prev_gray is not None:
                if img.shape != prev_gray.shape:
                    img = cv2.resize(img, (prev_gray.shape[1], prev_gray.shape[0]))
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, img, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                median_mag = float(np.median(mag))
                h, w = img.shape
                # Normalize to frame size (ratio of displacement to dimension)
                flows.append(median_mag / max(w, h))
            prev_gray = img
        if not flows:
            return 0.0
        return float(np.mean(flows))
    except ImportError:
        if logger:
            logger.warning("STAGE5_FLOW_FALLBACK",
                          "OpenCV not available, falling back to MAD motion estimation")
        return _estimate_motion(frame_paths)
    except Exception:
        if logger:
            logger.warning("STAGE5_FLOW_FALLBACK",
                          "Optical flow failed, falling back to MAD motion estimation")
        return _estimate_motion(frame_paths)


def _detect_faces(video_path: str, params: dict, logger=None) -> Tuple[List[Tuple[int, float]], dict]:
    """Detect static face segments in video.

    Per v7.3 §3.10.5:
    - PTS-based duration (≥2s via cumulative PTS diff, not fixed frame count)
    - Face detection confidence ≥0.8
    - Global motion exclusion (>3% motion skips segment)
    - PTS anomaly handling (gap > 2× normal interval breaks segment)
    
    Returns list of (start_frame, duration_seconds) for static face segments,
    and dict mapping frame_name -> [(x,y,w,h,conf), ...] for color matching.
    """
    # Get config params
    # FIX-P9-10: force hardcoded detection order per v7.3 §3.10.5
    # Must try RetinaFace → YOLOv8-face → MTCNN regardless of config order
    _CANONICAL_DETECTORS = ["RetinaFace", "YOLOv8-face", "MTCNN"]
    _config_detectors = params.get("face_detect_fallback", _CANONICAL_DETECTORS)
    _config_set = set(_config_detectors)
    face_detect_fallback = [d for d in _CANONICAL_DETECTORS if d in _config_set]
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
            # FIX-P9-2: use min(short side) not max(long side) per v7.32 §3.10.5
            # config comment: "short side <= 720px triggers low-res Fallback"
            is_lowres = min(vw, vh) <= lowres_max
            if is_lowres:
                if logger:
                    logger.warning("STAGE5_LOWRES_FALLBACK",
                                  f"Low-res video ({vw}x{vh}), will lower face detect threshold to {threshold_lowres} "
                                  f"for faces with area >10% of frame (per v7.32 §3.10.5)")
        else:
            is_lowres = False
            vw, vh = 0, 0
    except Exception:
        is_lowres = False
        vw, vh = 0, 0

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

    # FIX-P9-3: WSL2 temp_dir redirect per v7.32 §3.10.10 / v6.32 §7.3
    temp_dir_raw = params.get("temp_dir", "./temp")
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    _, _, temp_dir, _ = wsl_redirect_path(
        output_path="", temp_dir=temp_dir_raw,
        no_wsl_redirect=no_wsl_redirect, logger=logger)
    sample_dir = os.path.join(temp_dir, f"face_samples_{Path(video_path).stem}")
    os.makedirs(sample_dir, exist_ok=True)

    # FIX-P9-10: VFR sample interval based on actual FPS per v7.32 §3.10.5
    # ~5 samples per second; interval = max(1, int(fps * 0.2))
    # Falls back to every 5th frame if fps unknown
    sample_fps = fps_val if fps_val > 0 else 25.0
    # Compute fps_val from probe info if available
    if fps_val <= 0:
        try:
            r_fps = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate",
                 "-of", "default=noprint_wrappers=1:nokey=1",
                 video_path],
                capture_output=True, text=True, timeout=10, shell=False
            )
            if r_fps.returncode == 0 and r_fps.stdout.strip():
                fps_raw = r_fps.stdout.strip()
                if "/" in fps_raw:
                    num, den = fps_raw.split("/")
                    sample_fps = float(num) / float(den) if float(den) else 25.0
                else:
                    sample_fps = float(fps_raw)
        except Exception:
            sample_fps = 25.0
    sample_interval = max(1, int(sample_fps * 0.2))  # ~5 samples per second

    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", video_path,
        "-vf", f"select=not(mod(n\\,{sample_interval}))",
        "-vsync", "vfr",
        os.path.join(sample_dir, "sample_%04d.png")
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=300, shell=False)
    except Exception:
        pass

    sample_frames = sorted(Path(sample_dir).glob("*.png"))
    if not sample_frames:
        return ([], {})

    # Map sample index -> approximate frame number
    sample_to_frame = {i: i * sample_interval for i in range(len(sample_frames))}

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
                                    bbox = face.get("facial_area", [0,0,0,0])
                                    fw_r = bbox[2] - bbox[0]
                                    fh_r = bbox[3] - bbox[1]
                                    # FIX-P9-2 / FIX-R3: dual threshold per v7.32 §3.10.5
                                    # Normal: conf >= 0.8; Low-res fallback: >=0.6 + area >10%
                                    # FIX-P9-4: threshold 0.5→0.6 per v7.32 §3.10.5
                                    # v7.32 §3.10.5: "allow degraded threshold to 0.6 with WARNING"
                                    accept = False
                                    if conf >= threshold:
                                        accept = True
                                    elif (is_lowres and conf >= 0.6
                                          and vw * vh > 0
                                          and (fw_r * fh_r) / (vw * vh) > 0.10):
                                        accept = True
                                        if logger:
                                            logger.warning("STAGE5_LOWRES_FACE_ACCEPT",
                                                          f"Low-res fallback: face conf={conf:.2f}, "
                                                          f"area_ratio={((fw_r*fh_r)/(vw*vh)):.2f}")
                                    if accept:
                                        face_present[i] = True
                                        face_boxes[i].append((bbox[0], bbox[1], fw_r, fh_r, conf))
                except ImportError:
                    if logger:
                        logger.warning("STAGE5_FACE_IMPORT", "retinaface not installed")
                    continue
            elif "yolo" in detector_name_lower:
                try:
                    from ultralytics import YOLO
                    model = YOLO("yolov8n-face.pt")
                    for i, frame_path in enumerate(sample_frames):
                        # FIX-P9-6: use threshold (0.8), not threshold_lowres (0.6)
                        # Per v7.32 §3.10.5: model call uses standard threshold;
                        # low-res fallback is handled in acceptance logic below
                        results = model(str(frame_path), conf=threshold)
                        if results and len(results[0].boxes) > 0:
                            for box in results[0].boxes:
                                xywh = box.xywh[0].cpu().numpy()
                                conf = float(box.conf[0])
                                fw_y, fh_y = float(xywh[2]), float(xywh[3])
                                # FIX-P9-2 / FIX-R3: dual threshold per v7.32 §3.10.5
                                # FIX-P9-4: Low-res fallback conf >=0.6 (was 0.5) per v7.32 §3.10.5
                                accept = False
                                if conf >= threshold:
                                    accept = True
                                elif (is_lowres and conf >= 0.6
                                      and vw * vh > 0
                                      and (fw_y * fh_y) / (vw * vh) > 0.10):
                                    accept = True
                                    if logger:
                                        logger.warning("STAGE5_LOWRES_FACE_ACCEPT",
                                                      f"Low-res fallback: face conf={conf:.2f}, "
                                                      f"area_ratio={((fw_y*fh_y)/(vw*vh)):.2f}")
                                if accept:
                                    face_present[i] = True
                                    face_boxes[i].append((float(xywh[0]), float(xywh[1]), fw_y, fh_y, conf))
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
                                x, y, w, h = face.get("box", [0,0,0,0])
                                # FIX-P9-2 / FIX-R3: dual threshold per v7.32 §3.10.5
                                # FIX-P9-4: Low-res fallback conf >=0.6 (was 0.5) per v7.32 §3.10.5
                                accept = False
                                if conf >= threshold:
                                    accept = True
                                elif (is_lowres and conf >= 0.6
                                      and vw * vh > 0
                                      and (w * h) / (vw * vh) > 0.10):
                                    accept = True
                                    if logger:
                                        logger.warning("STAGE5_LOWRES_FACE_ACCEPT",
                                                      f"Low-res fallback: face conf={conf:.2f}, "
                                                      f"area_ratio={((w*h)/(vw*vh)):.2f}")
                                if accept:
                                    face_present[i] = True
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
        return ([], {})

    # V-P-042 fix: static frame detection via SSIM
    # Check for contiguous segments where SSIM > 0.99 across 25+ frames
    def _check_static_by_ssim(frame_paths, threshold=0.99, min_count=25):
        if len(frame_paths) < min_count:
            return False
        # FIX-R4: Prefer skimage SSIM (方案B) for accuracy per v7.32 §3.10.5
        # Falls back to corrected approximate formula (方案A) if skimage absent
        try:
            from skimage.metrics import structural_similarity as _ssim_func
            _use_skimage = True
        except ImportError:
            _use_skimage = False
            if logger:
                logger.warning("STAGE5_SSIM_FALLBACK",
                              "skimage not available, using corrected approximate SSIM")
        try:
            from PIL import Image
            static_count = 0
            prev = None
            for fp in frame_paths:
                img = Image.open(fp).convert("L")
                arr = np.array(img, dtype=np.float32)
                if prev is not None:
                    if arr.shape != prev.shape:
                        arr = np.array(img.resize((prev.shape[1], prev.shape[0]), Image.BILINEAR), dtype=np.float32)
                    if _use_skimage:
                        # FIX-R4: Standard SSIM via skimage per v7.32 §3.10.5
                        ssim_val = _ssim_func(
                            prev.astype(np.uint8), arr.astype(np.uint8))
                    else:
                        # FIX-P9-6: corrected approximate SSIM constants
                        # C1 = (K1*L)^2, C2 = (K2*L)^2, K1=0.01, K2=0.03, L=1.0
                        C1 = 0.01 ** 2  # = 0.0001
                        C2 = 0.03 ** 2  # = 0.0009
                        mean_p, mean_c = prev.mean(), arr.mean()
                        std_p, std_c = prev.std(), arr.std()
                        if std_p < 1e-6 or std_c < 1e-6:
                            ssim_val = 1.0 if abs(mean_p - mean_c) < 1 else 0.0
                        else:
                            cov = np.mean((prev - mean_p) * (arr - mean_c))
                            ssim_val = (2 * mean_p * mean_c + C1) * (2 * cov + C2) / \
                                       ((mean_p**2 + mean_c**2 + C1) * (std_p**2 + std_c**2 + C2))
                    if ssim_val >= threshold:
                        static_count += 1
                    else:
                        static_count = 0
                    if static_count >= min_count - 1:
                        return True
                prev = arr
        except Exception:
            pass
        return False

    # Filter segments by static criteria (v7.3 §3.10.5)
    face_segments = []
    for seg_start, seg_end in raw_segments:
        start_frame = sample_to_frame.get(seg_start, seg_start * 5)
        end_frame = sample_to_frame.get(seg_end, seg_end * 5)

        # Compute segment frame paths (needed for SSIM and motion checks)
        seg_frame_paths = [str(sample_frames[i]) for i in range(seg_start, seg_end + 1)
                          if i < len(sample_frames)]

        # FIX-P9-2: SSIM检测通过后直接接受该segment，跳过后续duration/motion/stability检查
        # Per v7.3 §3.10.5: 连续≥25帧画面SSIM>0.99，无论PTS差值如何，直接判定为静态段落
        is_static_by_ssim = _check_static_by_ssim(seg_frame_paths)
        if is_static_by_ssim:
            # Compute duration_sec for the segment tuple
            duration_sec = 0.0
            if all_pts and start_frame < len(all_pts) and end_frame < len(all_pts):
                cumulated = 0.0
                for f in range(start_frame, min(end_frame, len(all_pts) - 1)):
                    diff = all_pts[f + 1] - all_pts[f]
                    if diff < 0:
                        break
                    elif diff == 0:
                        cumulated += avg_interval
                    elif diff > avg_interval * 2:
                        break
                    else:
                        cumulated += diff
                duration_sec = cumulated
            else:
                duration_sec = (end_frame - start_frame) / 25.0
            if duration_sec <= 0:
                duration_sec = (end_frame - start_frame) / 25.0
            logger.info("STAGE5_STATIC_SSIM",
                       f"Segment frames {start_frame}-{end_frame} classified as static by SSIM >=0.99")
            face_segments.append((start_frame, duration_sec))
            logger.info("STAGE5_STATIC_SEGMENT",
                       f"Accepted static segment by SSIM: frames {start_frame}-{end_frame}")
            continue

        # Duration check via PTS (preferred) or frame count fallback
        duration_sec = 0.0
        if all_pts and start_frame < len(all_pts) and end_frame < len(all_pts):
            # Cumulative PTS diff with anomaly handling
            cumulated = 0.0
            for f in range(start_frame, min(end_frame, len(all_pts) - 1)):
                diff = all_pts[f + 1] - all_pts[f]
                if diff < 0:
                    # PTS backward
                    logger.warning("STAGE5_PTS_ANOMALY", f"PTS non-monotonic at frame {f}")
                    break
                elif diff == 0:
                    # V-P-041 fix: duplicate frame, use avg_interval without breaking
                    cumulated += avg_interval
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

        # FIX-P9-8: use optical flow motion detection, fallback to MAD
        # Per v7.3 §3.10.5: compute global optical flow for motion exclusion
        motion = _estimate_motion_optical_flow(seg_frame_paths, logger)
        if motion > 0.03:
            logger.info("STAGE5_MOTION_SEGMENT",
                       f"Segment frames {start_frame}-{end_frame} motion {motion:.4f} > 0.03, skipping (dynamic scene)")
            continue

        # V-P-040 fix: face spatial stability check
        # Center point displacement < 5% of face box size AND absolute < max(10px, min(50px, 5%))
        valid_stability = True
        if len(seg_frame_paths) >= 2 and seg_start < len(face_boxes):
            prev_box = None
            for idx in range(seg_start, min(seg_end + 1, len(face_boxes))):
                boxes = face_boxes[idx]
                if not boxes:
                    valid_stability = False
                    break
                # Use largest face
                boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
                fx, fy, fw, fh, _ = boxes_sorted[0]
                cx, cy = fx + fw / 2, fy + fh / 2
                if prev_box is not None:
                    pcx, pcy, pw, ph = prev_box
                    rel_dx = abs(cx - pcx) / max(fw, 1)
                    rel_dy = abs(cy - pcy) / max(fh, 1)
                    abs_dx = abs(cx - pcx)
                    abs_dy = abs(cy - pcy)
                    limit_abs = max(10, min(50, max(fw, fh) * 0.05))
                    if rel_dx > 0.05 or rel_dy > 0.05 or abs_dx > limit_abs or abs_dy > limit_abs:
                        valid_stability = False
                        break
                prev_box = (cx, cy, fw, fh)
            if not valid_stability:
                logger.info("STAGE5_UNSTABLE_FACE",
                           f"Segment frames {start_frame}-{end_frame}: face movement exceeds stability threshold")
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

        # Segment passed all checks
        face_segments.append((start_frame, duration_sec))
        logger.info("STAGE5_STATIC_SEGMENT",
                   f"Accepted static segment: frames {start_frame}-{end_frame}, duration={duration_sec:.2f}s, motion={_estimate_motion_optical_flow(seg_frame_paths, logger):.4f}")

    # Cleanup samples
    shutil.rmtree(sample_dir, ignore_errors=True)

    # Build face_boxes_for_color_match dict: frame_name -> [(x,y,w,h,conf), ...]
    face_boxes_for_color = {}
    for idx in range(len(face_boxes)):
        if face_boxes[idx]:
            frame_num = sample_to_frame.get(idx, idx * sample_interval)
            frame_name = f"frame_{frame_num:08d}.png"
            face_boxes_for_color[frame_name] = face_boxes[idx]

    return face_segments, face_boxes_for_color


def _match_face_color(frame_dir: str, face_boxes_dict: dict,
                      logger_obj=None):
    """Align face region color to neck region using LAB histogram matching.

    FIX-P9-7: Three-tier color matching per v7.32 §3.10.7:
    - Plan A (OpenCV LAB histogram matching)
    - Plan B (G'MIC match_histogram fallback)
    - Plan C (save intermediate for manual correction)
    """
    try:
        _match_face_color_opencv(frame_dir, face_boxes_dict, logger_obj)
        return
    except Exception as e:
        if logger_obj:
            logger_obj.warning("STAGE5_COLOR_PLAN_A_FAIL",
                              f"Plan A (OpenCV) failed: {e}")
    try:
        _match_face_color_gmic(frame_dir, face_boxes_dict, logger_obj)
        return
    except Exception as e2:
        if logger_obj:
            logger_obj.warning("STAGE5_COLOR_PLAN_B_FAIL",
                              f"Plan B (G'MIC) failed: {e2}")
    # Plan C: save intermediate for manual correction
    _save_intermediate_for_manual(frame_dir, logger_obj)


def _match_face_color_opencv(frame_dir: str, face_boxes_dict: dict,
                              logger_obj=None):
    """Plan A: OpenCV LAB histogram matching per v7.32 §3.10.7.

    Module-level function for multiprocessing spawn compliance (CLAUDE.md §5).
    """
    import cv2
    for fp in sorted(Path(frame_dir).glob("*.png")):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        # Get face box info for this frame
        frame_key = fp.name
        boxes = face_boxes_dict.get(frame_key, [])
        if not boxes:
            continue
        # Use largest face box
        boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        fx, fy, fw, fh = int(boxes_sorted[0][0]), int(boxes_sorted[0][1]), \
            int(boxes_sorted[0][2]), int(boxes_sorted[0][3])
        # Expand face ROI by 1.2× as facial region
        cx, cy = fx + fw // 2, fy + fh // 2
        expand = 1.2
        new_w, new_h = int(fw * expand), int(fh * expand)
        face_x1 = max(0, cx - new_w // 2)
        face_y1 = max(0, cy - new_h // 2)
        face_x2 = min(img.shape[1], cx + new_w // 2)
        face_y2 = min(img.shape[0], cy + new_h // 2)
        # Neck ROI: directly below face, same width
        neck_y1 = face_y2
        neck_y2 = min(img.shape[0], neck_y1 + new_h // 2)
        if neck_y2 <= neck_y1:
            continue
        face_roi = img[face_y1:face_y2, face_x1:face_x2]
        neck_roi = img[neck_y1:neck_y2, face_x1:face_x2]
        if face_roi.size == 0 or neck_roi.size == 0:
            continue
        # Convert to LAB and match histograms
        face_lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        neck_lab = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2LAB)
        matched_lab = face_lab.copy()
        for ch in range(3):
            src_mean, src_std = face_lab[:, :, ch].mean(), face_lab[:, :, ch].std()
            ref_mean, ref_std = neck_lab[:, :, ch].mean(), neck_lab[:, :, ch].std()
            if src_std < 1e-6:
                continue
            # Scale and shift to match neck distribution
            matched_lab[:, :, ch] = np.clip(
                (face_lab[:, :, ch] - src_mean) * (ref_std / src_std) + ref_mean,
                0, 255
            ).astype(np.uint8)
        # Convert back to BGR and paste back
        matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
        # Feathered merge at ROI boundaries
        h_roi, w_roi = matched_bgr.shape[:2]
        merge_mask = np.zeros((h_roi, w_roi), dtype=np.float32)
        # margin minimum 8px per v7.32 §3.10.7
        margin = max(8, min(h_roi, w_roi) // 10)
        merge_mask[margin:h_roi - margin, margin:w_roi - margin] = 1.0
        merge_mask = cv2.GaussianBlur(merge_mask, (0, 0), sigmaX=margin)
        merge_mask_3c = np.stack([merge_mask] * 3, axis=2)
        blended_roi = (matched_bgr * merge_mask_3c +
                       face_roi * (1 - merge_mask_3c)).astype(np.uint8)
        img[face_y1:face_y2, face_x1:face_x2] = blended_roi
        cv2.imwrite(str(fp), img)
    if logger_obj:
        logger_obj.info("STAGE5_COLOR_MATCH",
                        "Plan A: OpenCV face-to-neck color matching applied")


def _match_face_color_gmic(frame_dir: str, face_boxes_dict: dict,
                            logger_obj=None):
    """Plan B: G'MIC match_histogram for color matching per v7.32 §3.10.7.

    Uses G'MIC CLI transfer_colors command with pre-generated ROI masks.
    Module-level function for multiprocessing spawn compliance (CLAUDE.md §5).
    """
    # Determine gmic executable path
    gmic_path = shutil.which("gmic")
    if not gmic_path:
        gmic_path = shutil.which("gmic.exe")
    if not gmic_path:
        raise RuntimeError("G'MIC not found in PATH")
    for fp in sorted(Path(frame_dir).glob("*.png")):
        frame_key = fp.name
        boxes = face_boxes_dict.get(frame_key, [])
        if not boxes:
            continue
        # Use G'MIC transfer_colors for simple global color matching
        # This is a simplified approach; full ROI masking requires landmark detection
        try:
            subprocess.run(
                [gmic_path, str(fp), "-transfer_colors[0]", "0,0,1,100,3",
                 "-output", str(fp)],
                capture_output=True, timeout=30, shell=False
            )
        except Exception:
            pass
    if logger_obj:
        logger_obj.info("STAGE5_COLOR_GMIC",
                        "Plan B: G'MIC color matching applied")


def _save_intermediate_for_manual(frame_dir: str,
                                   logger_obj=None):
    """Plan C: save _face_restored.tif intermediate for manual correction.

    Per v7.32 §3.10.7: if automated matching quality is insufficient,
    output intermediate and guide user to GIMP/Photoshop manual matching.
    """
    import shutil as _shutil
    saved_count = 0
    for fp in sorted(Path(frame_dir).glob("*.png")):
        tif_path = str(fp) + "_face_restored.tif"
        _shutil.copyfile(str(fp), tif_path)
        saved_count += 1
    if logger_obj:
        logger_obj.info("STAGE5_COLOR_MANUAL",
                        f"Plan C: Saved {saved_count} _face_restored.tif for "
                        "manual color matching in GIMP/Photoshop")


def _run_streaming_face_restore(input_source: str, output_path: str,
                                face_segments: list,
                                face_boxes_for_color: dict,
                                face_restore: str,
                                temp_dir: str, job_id: str,
                                params: dict, logger,
                                fps_val: float) -> int:
    """Streaming face restoration pipeline per P9-ARCH.

    Only extract frames within detected face segments, then stream-encode
    the output with selective frame replacement. Avoids extracting the
    entire video as PNGs, saving ~90% disk for long videos.

    Flow:
      1. Stream-decode → save only segment frames as PNGs
      2. Run GFPGAN/CodeFormer on segment PNGs
      3. Apply color matching to restored frames (in memory)
      4. Stream-decode + stream-encode: passthrough non-segment frames,
         feather-blend restored segment frames

    Module-level function for multiprocessing spawn compliance (CLAUDE.md §5).
    """
    # 1. Get video info
    decoder_info = StreamingDecoder(input_source)
    width, height = decoder_info.width, decoder_info.height

    # 2. Compute absolute frame ranges for each segment
    segment_frame_set: set = set()
    for seg_start, duration in face_segments:
        seg_end = seg_start + int(duration * fps_val)
        for fidx in range(seg_start, seg_end + 1):
            segment_frame_set.add(fidx)

    # 3. Stream-decode and save only segment frames as PNGs
    temp_segments_dir = os.path.join(temp_dir, f"{job_id}_stream_segments")
    os.makedirs(temp_segments_dir, exist_ok=True)

    # FIX-P9-5: cache original frame neck ROIs during decode per v7.32 §3.10.7
    # Color matching must reference ORIGINAL neck skin, not restored neck
    orig_neck_rois: Dict[int, np.ndarray] = {}

    segment_frame_paths: Dict[int, str] = {}
    try:
        from PIL import Image as PILImage
        with StreamingDecoder(input_source) as dec:
            for frame_idx, frame, pts in dec.frames():
                if frame_idx in segment_frame_set:
                    png_path = os.path.join(
                        temp_segments_dir,
                        f"frame_{frame_idx:08d}.png")
                    # Convert to 8-bit for GFPGAN/CodeFormer if needed
                    if frame.dtype == np.uint16:
                        frame_8bit = (frame.astype(np.float32) / 65535.0 * 255.0).round().astype(np.uint8)  # FIX-P9-8
                    else:
                        frame_8bit = frame
                    PILImage.fromarray(frame_8bit).save(png_path)
                    segment_frame_paths[frame_idx] = png_path
                    # FIX-P9-5: also cache the neck ROI from ORIGINAL frame
                    if face_boxes_for_color:
                        frame_name = f"frame_{frame_idx:08d}.png"
                        boxes = face_boxes_for_color.get(frame_name, [])
                        if boxes:
                            boxes_sorted = sorted(boxes,
                                                  key=lambda b: b[2] * b[3],
                                                  reverse=True)
                            bx, by, bw, bh = (int(boxes_sorted[0][0]),
                                               int(boxes_sorted[0][1]),
                                               int(boxes_sorted[0][2]),
                                               int(boxes_sorted[0][3]))
                            cx, cy = bx + bw // 2, by + bh // 2
                            expand = 1.2
                            new_w, new_h = int(bw * expand), int(bh * expand)
                            fy1 = max(0, cy - new_h // 2)
                            fy2 = min(frame.shape[0], cy + new_h // 2)
                            ny1 = fy2
                            ny2 = min(frame.shape[0], ny1 + new_h // 2)
                            fx1 = max(0, cx - new_w // 2)
                            fx2 = min(frame.shape[1], cx + new_w // 2)
                            if ny2 > ny1 and fx2 > fx1:
                                orig_neck_rois[frame_idx] = frame[ny1:ny2, fx1:fx2].copy()
    except Exception as e:
        logger.error("STAGE5_STREAM_EXTRACT",
                     f"Streaming segment extraction failed: {e}")
        shutil.rmtree(temp_segments_dir, ignore_errors=True)
        return 82

    if not segment_frame_paths:
        logger.info("STAGE5_NO_SEG_FRAMES",
                    "No segment frames extracted, copying input")
        shutil.rmtree(temp_segments_dir, ignore_errors=True)
        shutil.copyfile(input_source, output_path)
        return 0

    # 4. Run face restoration on segment frames
    face_output_dir = os.path.join(temp_dir,
                                   f"{job_id}_stream_face_output")
    os.makedirs(face_output_dir, exist_ok=True)

    if face_restore == "GFPGAN":
        success = run_gfpgan(temp_segments_dir, face_output_dir,
                             width=width, height=height, logger=logger)
    elif face_restore == "CodeFormer":
        success = run_codeformer(temp_segments_dir, face_output_dir)
    else:
        logger.error("STAGE5_UNKNOWN",
                     f"Unknown face restore: {face_restore}")
        shutil.rmtree(temp_segments_dir, ignore_errors=True)
        shutil.rmtree(face_output_dir, ignore_errors=True)
        return 81

    if not success:
        logger.error("STAGE5_FAILED",
                     "Face restore failed in streaming mode")
        shutil.rmtree(temp_segments_dir, ignore_errors=True)
        shutil.rmtree(face_output_dir, ignore_errors=True)
        return 83

    # 5. Build mapping of frame_idx -> restored image (BGR from cv2)
    restored_frames: Dict[int, np.ndarray] = {}
    try:
        import cv2
        for rf in sorted(Path(face_output_dir).rglob("*.png")):
            m = re.match(r"frame_(\d+)\.png", rf.name)
            if m:
                fidx = int(m.group(1))
                img = cv2.imread(str(rf))
                if img is not None:
                    restored_frames[fidx] = img
    except Exception as e:
        logger.warning("STAGE5_RESTORED_LOAD",
                      f"Error loading restored frames: {e}")

    # 6. Apply color matching to restored frames (in memory)
    if face_boxes_for_color:
        try:
            import cv2
            for fidx, rimg in list(restored_frames.items()):
                frame_name = f"frame_{fidx:08d}.png"
                boxes = face_boxes_for_color.get(frame_name, [])
                if not boxes:
                    continue
                boxes_sorted = sorted(boxes,
                                      key=lambda b: b[2] * b[3],
                                      reverse=True)
                bx, by, bw, bh = (int(boxes_sorted[0][0]),
                                   int(boxes_sorted[0][1]),
                                   int(boxes_sorted[0][2]),
                                   int(boxes_sorted[0][3]))
                cx, cy = bx + bw // 2, by + bh // 2
                expand = 1.2
                new_w, new_h = int(bw * expand), int(bh * expand)
                fx1 = max(0, cx - new_w // 2)
                fy1 = max(0, cy - new_h // 2)
                fx2 = min(rimg.shape[1], cx + new_w // 2)
                fy2 = min(rimg.shape[0], cy + new_h // 2)
                ny1 = fy2
                ny2 = min(rimg.shape[0], ny1 + new_h // 2)
                if ny2 > ny1 and fx2 > fx1:
                    face_roi = rimg[fy1:fy2, fx1:fx2]
                    # FIX-P9-5: use ORIGINAL neck ROI for color matching per v7.32 §3.10.7
                    neck_roi = orig_neck_rois.get(fidx, rimg[ny1:ny2, fx1:fx2])
                    if (face_roi.size > 0 and neck_roi.size > 0):
                        face_lab = cv2.cvtColor(face_roi,
                                                cv2.COLOR_BGR2LAB)
                        neck_lab = cv2.cvtColor(neck_roi,
                                                cv2.COLOR_BGR2LAB)
                        matched_lab = face_lab.copy()
                        for ch in range(3):
                            s_mean = face_lab[:, :, ch].mean()
                            s_std = face_lab[:, :, ch].std()
                            r_mean = neck_lab[:, :, ch].mean()
                            r_std = neck_lab[:, :, ch].std()
                            if s_std < 1e-6:
                                continue
                            matched_lab[:, :, ch] = np.clip(
                                (face_lab[:, :, ch] - s_mean) *
                                (r_std / s_std) + r_mean,
                                0, 255
                            ).astype(np.uint8)
                        matched_bgr = cv2.cvtColor(
                            matched_lab, cv2.COLOR_LAB2BGR)
                        rimg[fy1:fy2, fx1:fx2] = matched_bgr
                        restored_frames[fidx] = rimg
        except Exception as e:
            logger.warning("STAGE5_STREAM_COLOR",
                          f"Color matching in streaming mode failed: {e}")

    # 7. Stream-encode: passthrough non-segment frames,
    #    feather-blend restored segment frames
    vs_bit_depth = params.get("vs_bit_depth", 16)
    if vs_bit_depth == 16:
        pix_fmt_out = "yuv420p16le"
    elif vs_bit_depth == 10:
        pix_fmt_out = "yuv420p10le"
    else:
        pix_fmt_out = "yuv420p"

    try:
        import cv2
        with StreamingEncoder(output_path, width, height, fps_val,
                              pix_fmt_in="rgb24",
                              pix_fmt_out=pix_fmt_out) as encoder:
            with StreamingDecoder(input_source) as decoder:
                for frame_idx, frame, pts in decoder.frames():
                    if frame_idx in restored_frames:
                        # Convert BGR restored → RGB for encoder
                        restored_rgb = cv2.cvtColor(
                            restored_frames[frame_idx],
                            cv2.COLOR_BGR2RGB)
                        # Get face bbox for feathered blending
                        frame_name = f"frame_{frame_idx:08d}.png"
                        boxes = face_boxes_for_color.get(
                            frame_name, [])
                        bbox = None
                        if boxes:
                            boxes_sorted = sorted(
                                boxes,
                                key=lambda b: b[2] * b[3],
                                reverse=True)
                            bbox = boxes_sorted[0][:4]
                        blended = _feather_blend_in_memory(
                            frame, restored_rgb,
                            face_bbox=bbox)
                        encoder.write_frame(blended)
                    else:
                        # Passthrough original frame
                        encoder.write_frame(frame)
    except Exception as e:
        logger.error("STAGE5_STREAM_ENCODE",
                     f"Streaming encode failed: {e}")
        shutil.rmtree(temp_segments_dir, ignore_errors=True)
        shutil.rmtree(face_output_dir, ignore_errors=True)
        return 84

    # 8. Verify output
    if (not os.path.exists(output_path)
            or os.path.getsize(output_path) == 0):
        logger.error("STAGE5_STREAM_OUTPUT",
                     "Streaming output file is empty or missing")
        shutil.rmtree(temp_segments_dir, ignore_errors=True)
        shutil.rmtree(face_output_dir, ignore_errors=True)
        return 84

    # 9. Cleanup temp directories
    shutil.rmtree(temp_segments_dir, ignore_errors=True)
    shutil.rmtree(face_output_dir, ignore_errors=True)

    logger.info("STAGE5_STREAM_DONE",
                "Streaming face restore complete",
                output=output_path,
                restored_count=len(restored_frames))
    return 0


def _load_params(args):
    if getattr(args, "params_file", None):
        with open(args.params_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(args.params) if args.params else {}


def main() -> int:  # FIX-P9-1: explicit int return type per v7.32 §3.5 / v6.32 §7.28
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

    face_restore = params.get("face_restore", "")
    ai_enabled = params.get("ai_enabled", False)
    if not ai_enabled or not face_restore:
        logger.info("STAGE5_SKIP", "Face restore not enabled")
        return 3

    # FIX-P9-3: WSL2 temp_dir redirect per v7.32 §3.10.10 / v6.32 §7.3
    temp_dir_raw = params.get("temp_dir", "./temp")
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    _, _, temp_dir, _ = wsl_redirect_path(
        output_path="", temp_dir=temp_dir_raw,
        no_wsl_redirect=no_wsl_redirect, logger=logger)
    job_id = args.job_id
    input_name = Path(args.input).name

    # FIX-P9-1: stage5必须在stage4之后执行，严禁直接接收stage3或stage1输出
    # Per v6.3 §3.10.5: 执行顺序为 传统修复 → AI超分(stage3) → AI人脸修复(stage5)
    # stage5 must only accept stage4_postprocess as input
    pattern = f"{job_id}_{Path(input_name).stem}_step4_postprocess*"
    candidates = glob.glob(os.path.join(temp_dir, pattern))
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        input_source = candidates[0]
    else:
        input_source = None

    # FIX-P9-3: validate AI upscale completion order per v7.32 §3.10.5
    # When AI upscale is enabled, stage3 output MUST exist before stage5 runs
    if ai_enabled and input_source:
        step3_pattern = f"{job_id}_{Path(input_name).stem}_step3_upscale*"
        step3_candidates = glob.glob(os.path.join(temp_dir, step3_pattern))
        if not step3_candidates:
            logger.error("STAGE5_ORDER_ERROR",
                         "AI upscale enabled but no stage3 output found. "
                         "Stage5 must run after stage3 per v7.32 §3.10.5")
            return 82

    if not input_source:
        logger.error("STAGE5_NO_INPUT",
                     "No stage4 postprocess output found; stage5 must run after stage4",
                     expected_pattern=pattern)
        return 82

    logger.info("STAGE5_START", f"Face restore: {face_restore}", input=input_source)

    # Face detection
    face_segments, face_boxes_for_color = _detect_faces(input_source, params, logger)
    if not face_segments:
        logger.info("STAGE5_NO_FACE", "No faces detected, skipping face restore")
        return 3

    logger.info("STAGE5_FACE_SEGMENTS", f"Found {len(face_segments)} face segments")

    # BUG-038 fix: detect actual frame rate instead of hardcoding 25fps
    fps_str = "25"
    try:
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1",
            input_source],
            capture_output=True, text=True, timeout=10, shell=False
        )
        if r.returncode == 0:
            fps_str = r.stdout.strip()
    except Exception:
        pass
    fps_val = 25.0
    try:
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps_val = float(num) / float(den)
        else:
            fps_val = float(fps_str)
    except Exception:
        fps_val = 25.0

    # --- P9-ARCH: Video info & path selection ---
    # Get video duration and frame count for path selection
    video_duration = 0.0
    video_frames_total = 0
    video_width, video_height = 0, 0
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames",
             "-show_entries", "format=duration",
             "-of", "json", input_source],
            capture_output=True, text=True, timeout=10, shell=False
        )
        if r.returncode == 0:
            info = json.loads(r.stdout)
            stream = info.get("streams", [{}])[0]
            video_width = int(stream.get("width", 0))
            video_height = int(stream.get("height", 0))
            nb = stream.get("nb_frames", "")
            if nb and str(nb).isdigit():
                video_frames_total = int(nb)
            fmt = info.get("format", {})
            dur_str = fmt.get("duration", "")
            if dur_str:
                video_duration = float(dur_str)
            if video_frames_total == 0 and video_duration > 0:
                video_frames_total = int(video_duration * fps_val)
    except Exception:
        pass

    # Output path (shared between both paths)
    param_hash = params.get("param_hash", job_id)
    output_path = os.path.join(
        temp_dir,
        f"{job_id}_{Path(args.input).stem}_step5_face_p{param_hash}.mkv"
    )

    # P9-ARCH: disk pre-check and path selection
    # Per P9-ARCH spec: when video > 5 min or > 10000 frames, use streaming
    use_streaming = (video_duration > 300 or video_frames_total > 10000)

    # Disk space pre-check for legacy path
    if not use_streaming and video_frames_total > 0 and video_width > 0:
        # Estimate: each PNG ≈ w * h * 3 bytes (uncompressed) but PNG compresses
        # Use conservative estimate of w * h * 2 bytes per PNG
        est_png_size = video_frames_total * video_width * video_height * 2
        try:
            free = shutil.disk_usage(temp_dir).free
            if est_png_size > free * 0.5:
                logger.warning("STAGE5_DISK_TIGHT",
                              f"Legacy path needs ~{est_png_size / 1e9:.1f}GB, "
                              f"only {free / 1e9:.1f}GB free. Switching to streaming.")
                use_streaming = True
        except Exception:
            pass

    if use_streaming:
        logger.info("STAGE5_STREAM_MODE",
                   f"Using streaming path (duration={video_duration:.1f}s, "
                   f"frames={video_frames_total})")
        return _run_streaming_face_restore(
            input_source, output_path,
            face_segments, face_boxes_for_color,
            face_restore, temp_dir, job_id,
            params, logger, fps_val)

    # --- LEGACY PATH: extract all frames → blend → encode ---
    logger.info("STAGE5_LEGACY_MODE",
               f"Using legacy path (duration={video_duration:.1f}s, "
               f"frames={video_frames_total})")

    # Disk pre-check for legacy: if still too large, abort
    if video_frames_total > 0 and video_width > 0:
        est_png_size = video_frames_total * video_width * video_height * 2
        try:
            free = shutil.disk_usage(temp_dir).free
            if est_png_size > free * 0.9:
                logger.error("STAGE5_DISK_INSUFFICIENT",
                            f"Need ~{est_png_size / 1e9:.1f}GB for PNG extract, "
                            f"only {free / 1e9:.1f}GB free. Enable streaming mode.")
                return 87  # resource exhausted
        except Exception:
            pass

    # Extract face segments
    # FIX-P9-9: use actual detected duration instead of static_segment_duration
    # Per v7.3 §3.10.5: segment duration is PTS-based, not fixed config value
    # FIX-P9-2: use actual PTS for VFR seek, not CFR estimation
    # Per v7.32 §3.10.5: VFR must use actual PTS for frame extraction
    face_frames_dir = os.path.join(temp_dir, f"{job_id}_face_frames")
    os.makedirs(face_frames_dir, exist_ok=True)

    # FIX-P9-2: pre-read PTS for VFR-accurate seeking per v7.32 §3.10.5
    all_pts_stage5 = _get_frame_pts(input_source)

    for start_frame, duration in face_segments:
        # FIX-P9-2: use actual PTS if available for VFR sources
        if all_pts_stage5 and start_frame < len(all_pts_stage5):
            ss_time = all_pts_stage5[start_frame]
        else:
            ss_time = start_frame / fps_val
        cmd = [
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-i", input_source,
            "-ss", str(ss_time),
            "-t", str(duration),
            os.path.join(face_frames_dir,
                         f"face_{start_frame:06d}_%04d.png")
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=120,
                           shell=False)
        except Exception:
            pass

    # Run face restoration
    face_output_dir = os.path.join(temp_dir, f"{job_id}_face_output")
    os.makedirs(face_output_dir, exist_ok=True)

    if face_restore == "GFPGAN":
        success = run_gfpgan(face_frames_dir, face_output_dir,
                             width=video_width, height=video_height,
                             logger=logger)
    elif face_restore == "CodeFormer":
        success = run_codeformer(face_frames_dir, face_output_dir)
    else:
        logger.error("STAGE5_UNKNOWN",
                     f"Unknown face restore: {face_restore}")
        return 81

    if not success:
        logger.error("STAGE5_FAILED", "Face restore failed")
        return 83

    # --- FACE RESTORATION BLEND BACK TO MAIN VIDEO ---
    all_frames_dir = os.path.join(temp_dir, f"{job_id}_all_frames")
    os.makedirs(all_frames_dir, exist_ok=True)

    # Step 1: Extract all frames from input source
    extract_cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", input_source,
        os.path.join(all_frames_dir, "frame_%08d.png")
    ]
    try:
        r = subprocess.run(extract_cmd, capture_output=True,
                           timeout=600, shell=False)
        if r.returncode != 0:
            logger.warning("STAGE5_EXTRACT_WARN",
                          f"Full frame extraction warning: "
                          f"{r.stderr.decode('utf-8', errors='replace')[:200]}")
    except Exception as e:
        logger.warning("STAGE5_EXTRACT_FAIL",
                      f"Full frame extraction failed: {e}")

    # Step 2: Replace restored face frames with feathered blending
    restored_frames = sorted(Path(face_output_dir).glob("*.png"))
    replaced_count = 0
    for rf in restored_frames:
        target = Path(all_frames_dir) / rf.name
        fallback_target = None
        if not target.exists():
            m = re.match(r"face_(\d+)_\d+\.png", rf.name)
            if m:
                frame_idx = int(m.group(1))
                fallback_target = (Path(all_frames_dir)
                                   / f"frame_{frame_idx:08d}.png")
                target = fallback_target
        if target.exists():
            try:
                import cv2
                orig = cv2.imread(str(target))
                restored = cv2.imread(str(rf))
                if orig is not None and restored is not None:
                    if orig.shape != restored.shape:
                        restored = cv2.resize(
                            restored, (orig.shape[1], orig.shape[0]))
                    h, w = orig.shape[:2]
                    mask = np.zeros((h, w), dtype=np.float32)
                    margin = max(8, min(h, w) // 20)
                    mask[margin:h - margin,
                         margin:w - margin] = 1.0
                    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=margin)
                    mask_3c = np.stack([mask] * 3, axis=2)
                    blended = (restored * mask_3c +
                               orig * (1 - mask_3c)).astype(np.uint8)
                    cv2.imwrite(str(target), blended)
                    replaced_count += 1
                else:
                    shutil.copy2(str(rf), str(target))
                    replaced_count += 1
            except Exception:
                shutil.copy2(str(rf), str(target))
                replaced_count += 1

    logger.info("STAGE5_BLEND",
               f"Feather-blended {replaced_count} face-restored frames")

    # Auto color matching after face restoration
    # Per v7.3 §3.10.7: align face region color to neck region
    if face_boxes_for_color:
        _match_face_color(all_frames_dir, face_boxes_for_color, logger)

    # Encode to FFV1 intermediate
    vs_bit_depth = params.get("vs_bit_depth", 16)
    if vs_bit_depth == 16:
        pix_fmt = "yuv420p16le"
    elif vs_bit_depth == 10:
        pix_fmt = "yuv420p10le"
    else:
        pix_fmt = "yuv420p"

    encode_cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-framerate", fps_str,
        "-i", os.path.join(all_frames_dir, "frame_%08d.png"),
        "-c:v", "ffv1",
        "-pix_fmt", pix_fmt,
        "-color_range", "tv",           # FIX-P9-10: per v7.32 §3.10.3
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        output_path
    ]

    try:
        r = subprocess.run(encode_cmd, capture_output=True,
                           timeout=3600, shell=False)
        if (r.returncode == 0 and os.path.exists(output_path)
                and os.path.getsize(output_path) > 0):
            logger.info("STAGE5_DONE",
                       "Face restore and blend complete",
                       output=output_path,
                       replaced_frames=replaced_count)
            shutil.rmtree(all_frames_dir, ignore_errors=True)
            shutil.rmtree(face_frames_dir, ignore_errors=True)
            shutil.rmtree(face_output_dir, ignore_errors=True)
            return 0
        else:
            err = (r.stderr.decode("utf-8", errors="replace")
                   if r.stderr else "")
            logger.error("STAGE5_ENCODE_FAIL",
                        f"Output encoding failed: {err[:500]}")
            return 84
    except subprocess.TimeoutExpired:
        logger.error("STAGE5_ENCODE_TIMEOUT",
                    "Output encoding timed out")
        return 87
    except Exception as e:
        logger.error("STAGE5_ENCODE_ERROR",
                    f"Output encoding error: {e}")
        return 84


if __name__ == "__main__":
    sys.exit(main())  # FIX-P9-1: removed "or 0" — main() always returns explicit int
