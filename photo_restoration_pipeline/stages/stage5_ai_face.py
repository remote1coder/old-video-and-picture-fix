#!/usr/bin/env python3
"""
Stage 5: Photo Face / Portrait Restoration
GFPGAN / CodeFormer for face restoration in photos.
Full-frame processing (no temporal constraints for static images).

v6.3 / v7.32 Compliance:
- Face detection fallback list from config is honored
- Face detection thresholds and low-res max dimension respected
- Bans original/intermediate pixels from AI face restore (only traditional-processed allowed)
- Portrait color matching (v7.32 §3.10.7): after face restoration, automatically aligns
  face white-balance to neck/hands using LAB mean/std matching with feather blending
"""

import os
import sys
import json
import argparse
import subprocess
import glob
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402


def feather_blend(original_path: str, restored_path: str, output_path: str,
                  feather_width: int = 8) -> bool:
    """Blend restored face region back into original image with feathering.

    Uses per-pixel absolute difference to detect modified regions,
    then applies Gaussian blur to the mask edge for smooth transition.
    """
    try:
        import numpy as np
        from PIL import Image, ImageFilter

        orig = Image.open(original_path).convert("RGB")
        rest = Image.open(restored_path).convert("RGB")

        # Ensure same size
        if orig.size != rest.size:
            rest = rest.resize(orig.size, Image.LANCZOS)

        orig_arr = np.array(orig, dtype=np.float32)
        rest_arr = np.array(rest, dtype=np.float32)

        # Compute per-pixel absolute difference (average across RGB)
        diff = np.mean(np.abs(rest_arr - orig_arr), axis=2)

        # Threshold: consider pixels with >5% of max difference as modified
        threshold = 255.0 * 0.05
        mask = (diff > threshold).astype(np.float32)

        # If mask covers the whole image (extreme case), skip blending
        mask_ratio = np.mean(mask)
        if mask_ratio > 0.95 or mask_ratio < 0.01:
            shutil.copy2(restored_path, output_path)
            return True

        # Convert mask to PIL image for Gaussian blur
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_blurred = mask_img.filter(ImageFilter.GaussianBlur(radius=feather_width))
        mask_smooth = np.array(mask_blurred, dtype=np.float32) / 255.0
        mask_smooth = np.clip(mask_smooth, 0.0, 1.0)
        # Expand to 3 channels
        mask_smooth = np.stack([mask_smooth] * 3, axis=2)

        # Alpha blending
        blended = mask_smooth * rest_arr + (1.0 - mask_smooth) * orig_arr
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        result = Image.fromarray(blended)
        result.save(output_path)
        return True
    except Exception:
        # Fallback: direct copy if blending fails
        try:
            shutil.copy2(restored_path, output_path)
            return True
        except Exception:
            return False


def _add_grain_gmic(input_path: str, output_path: str,
                      strength: float, size: float, roughness: float) -> bool:
    """Add film grain using G'MIC (v6.32 §3.10.4: run in stage5 after all AI processing)."""
    try:
        cmd = [
            "gmic", input_path,
            "-film_grain", str(int(strength * 100)),
            str(size), str(roughness),
            "-o", output_path,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=300, shell=False)
        return r.returncode == 0
    except Exception:
        return False


def _detect_faces(input_path: str, params: dict, logger=None) -> bool:
    """Detect faces in image using configured fallback list.

    Returns True if faces detected, False otherwise.
    """
    return len(_detect_face_boxes(input_path, params, logger)) > 0


def _detect_face_boxes(input_path: str, params: dict, logger=None) -> list:
    """Detect faces and return list of bounding boxes [(x, y, w, h), ...].

    Each detector is tried in fallback order until at least one face is found.
    Boxes are returned in the coordinate space of the input image.
    """
    face_detect_fallback = params.get("face_detect_fallback", ["RetinaFace", "YOLOv8-face", "MTCNN"])
    threshold = params.get("face_detect_threshold", 0.8)
    threshold_lowres = params.get("face_detect_threshold_lowres", 0.6)
    lowres_max = params.get("face_detect_lowres_max_dimension", 720)

    try:
        from PIL import Image
        with Image.open(input_path) as img:
            img_w, img_h = img.size
            is_lowres = max(img_w, img_h) <= lowres_max
            actual_threshold = threshold_lowres if is_lowres else threshold
    except Exception:
        actual_threshold = threshold
        img_w = img_h = None

    for detector in face_detect_fallback:
        try:
            if detector == "RetinaFace":
                try:
                    from retinaface import RetinaFace
                    faces = RetinaFace.detect_faces(input_path)
                    if faces and len(faces) > 0:
                        valid_boxes = []
                        for key, face in faces.items():
                            score = face.get("score", 0)
                            if score >= actual_threshold:
                                fa = face.get("facial_area", [0, 0, 0, 0])
                                valid_boxes.append((fa[0], fa[1], fa[2] - fa[0], fa[3] - fa[1]))
                        if valid_boxes:
                            if logger:
                                logger.info("STAGE5_FACE_DETECT", f"Faces found via RetinaFace: {len(valid_boxes)} (threshold={actual_threshold})")
                            return valid_boxes
                except Exception:
                    pass
            elif detector == "YOLOv8-face":
                try:
                    from ultralytics import YOLO
                    model = YOLO("yolov8n-face.pt")
                    results = model(input_path, conf=actual_threshold)
                    if results and len(results[0].boxes) > 0:
                        boxes = []
                        for box in results[0].boxes.xyxy.cpu().numpy():
                            x1, y1, x2, y2 = map(int, box)
                            boxes.append((x1, y1, x2 - x1, y2 - y1))
                        if boxes:
                            if logger:
                                logger.info("STAGE5_FACE_DETECT", f"Faces found via YOLOv8: {len(boxes)} (threshold={actual_threshold})")
                            return boxes
                except Exception:
                    pass
            elif detector == "MTCNN":
                try:
                    from mtcnn import MTCNN
                    import cv2
                    img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
                    detector_obj = MTCNN()
                    faces = detector_obj.detect_faces(img)
                    if faces and len(faces) > 0:
                        valid_boxes = []
                        for face in faces:
                            confidence = face.get("confidence", 0)
                            if confidence >= actual_threshold:
                                box = face.get("box", [0, 0, 0, 0])
                                valid_boxes.append((box[0], box[1], box[2], box[3]))
                        if valid_boxes:
                            if logger:
                                logger.info("STAGE5_FACE_DETECT", f"Faces found via MTCNN: {len(valid_boxes)} (threshold={actual_threshold})")
                            return valid_boxes
                except Exception:
                    pass
        except Exception:
            continue

    if logger:
        logger.warning("STAGE5_NO_FACE", "No faces detected with any fallback detector")
    return []


def _match_portrait_color(image_path: str, face_boxes: list,
                          output_path: str, logger=None) -> bool:
    """Automated half-body portrait color matching per v7.32 §3.10.7.

    After face restoration (GFPGAN/CodeFormer), the restored face often has
    different white-balance/color from the original neck/hands. This function
    aligns the face's LAB distribution to the neck region using mean/std
    matching (histogram specification simplified).

    Steps:
      1. Estimate neck ROI below each face box.
      2. Convert face and neck regions to LAB.
      3. For each LAB channel, match face mean/std to neck mean/std.
      4. Feather-blend corrected face back into the image.
    """
    try:
        import numpy as np
        from PIL import Image, ImageFilter
        import cv2

        img = Image.open(image_path).convert("RGB")
        img_arr = np.array(img, dtype=np.float32)
        h_img, w_img = img_arr.shape[:2]

        # Build a combined mask for all face regions
        face_mask = np.zeros((h_img, w_img), dtype=np.float32)
        corrected = img_arr.copy()

        for (fx, fy, fw, fh) in face_boxes:
            # Clamp box to image bounds
            x1 = max(0, fx)
            y1 = max(0, fy)
            x2 = min(w_img, fx + fw)
            y2 = min(h_img, fy + fh)
            if x2 <= x1 or y2 <= y1:
                continue

            # Estimate neck region: directly below face, same width, 0.4x face height
            ny1 = y2
            ny2 = min(h_img, y2 + int(fh * 0.4))
            nx1 = x1
            nx2 = x2
            if ny2 <= ny1:
                continue

            face_roi = img_arr[y1:y2, x1:x2]
            neck_roi = img_arr[ny1:ny2, nx1:nx2]
            if neck_roi.size == 0:
                continue

            # Convert to LAB using OpenCV
            face_bgr = cv2.cvtColor(face_roi.astype(np.uint8), cv2.COLOR_RGB2BGR)
            neck_bgr = cv2.cvtColor(neck_roi.astype(np.uint8), cv2.COLOR_RGB2BGR)
            face_lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            neck_lab = cv2.cvtColor(neck_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

            # Mean/std matching per channel
            matched_lab = face_lab.copy()
            for c in range(3):
                f_mean = face_lab[:, :, c].mean()
                f_std = face_lab[:, :, c].std() + 1e-6
                n_mean = neck_lab[:, :, c].mean()
                n_std = neck_lab[:, :, c].std() + 1e-6
                matched_lab[:, :, c] = (matched_lab[:, :, c] - f_mean) * (n_std / f_std) + n_mean

            # Clamp LAB valid ranges
            matched_lab[:, :, 0] = np.clip(matched_lab[:, :, 0], 0, 255)
            matched_lab[:, :, 1] = np.clip(matched_lab[:, :, 1], 0, 255)
            matched_lab[:, :, 2] = np.clip(matched_lab[:, :, 2], 0, 255)

            # Convert back to RGB
            matched_bgr = cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            matched_rgb = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

            corrected[y1:y2, x1:x2] = matched_rgb
            face_mask[y1:y2, x1:x2] = 1.0

        # If no valid face was processed, skip
        if face_mask.max() < 0.5:
            shutil.copy2(image_path, output_path)
            return True

        # Feather the mask edge
        mask_img = Image.fromarray((face_mask * 255).astype(np.uint8))
        mask_blurred = mask_img.filter(ImageFilter.GaussianBlur(radius=8))
        mask_smooth = np.array(mask_blurred, dtype=np.float32) / 255.0
        mask_smooth = np.clip(mask_smooth, 0.0, 1.0)
        mask_smooth = np.stack([mask_smooth] * 3, axis=2)

        blended = mask_smooth * corrected + (1.0 - mask_smooth) * img_arr
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        Image.fromarray(blended).save(output_path)
        return True
    except Exception as e:
        if logger:
            logger.warning("STAGE5_COLOR_MATCH_FAIL", f"Portrait color matching failed: {e}")
        try:
            shutil.copy2(image_path, output_path)
            return True
        except Exception:
            return False


def run_gfpgan(input_path: str, output_dir: str, params: dict = None) -> bool:
    """Run GFPGAN on a single image.

    Args:
        params: optional dict with 'face_restore_version' and 'face_restore_upscale'
    """
    # Bug 11.22 fix: read version and upscale from config instead of hardcoding
    version = "1.4"
    upscale = 2
    if params:
        version = params.get("face_restore_version", version)
        upscale = params.get("face_restore_upscale", upscale)

    try:
        # Try inference_gfpgan.py script if available
        gfpgan_script = shutil.which("inference_gfpgan.py")
        if not gfpgan_script:
            # Search common locations
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
                "-i", input_path,
                "-o", output_dir,
                "-v", version,
                "-s", str(upscale),
            ]
            r = subprocess.run(cmd, capture_output=True, timeout=600, shell=False)
            return r.returncode == 0

        # Fallback: Python API
        model_dir = params.get("model_dir", "") if params else ""
        if not model_dir:
            project_root = SCRIPT_DIR.parent.parent.resolve()
            model_dir = str(project_root / "models")
        model_path = os.path.join(model_dir, f"GFPGANv{version}.pth")

        # BUG-050 fix: Use json.dumps() for safe path quoting to prevent single-quote injection
        import json as _json
        safe_input = _json.dumps(input_path)
        safe_model = _json.dumps(model_path)
        safe_output_dir = _json.dumps(output_dir)
        cmd = [
            sys.executable, "-c",
            f"from gfpgan import GFPGANer; "
            f"from basicsr.utils import img2tensor, tensor2img; "
            f"from facexlib.utils.face_restoration_helper import FaceRestoreHelper; "
            f"import cv2; "
            f"import torch; "
            f"img = cv2.imread({safe_input}, cv2.IMREAD_COLOR); "
            f"restorer = GFPGANer(model_path={safe_model}, upscale={upscale}, arch='clean', "
            f"channel_only=False, bg_upsampler=None); "
            f"_, _, output = restorer.enhance(img, has_aligned=False, only_center_face=False); "
            f"import os; "
            f"out_path = os.path.join({safe_output_dir}, os.path.basename({safe_input})); "
            f"cv2.imwrite(out_path, output)",
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=600, shell=False)
        return r.returncode == 0
    except Exception:
        return False


def run_codeformer(input_path: str, output_dir: str, params: dict = None) -> bool:
    """Run CodeFormer on a single image.

    Bug 11.13 fix: fallback no longer copies original; it attempts real inference
    or returns False so the caller knows CodeFormer failed.
    """
    try:
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
                "-i", input_path,
                "-o", output_dir,
                "--face_upsample",
                "--bg_upsampler", "realesrgan",
            ]
            r = subprocess.run(cmd, capture_output=True, timeout=600, shell=False)
            if r.returncode == 0:
                return True

        # Fallback: attempt real CodeFormer Python API inference
        model_dir = params.get("model_dir", "") if params else ""
        if not model_dir:
            project_root = SCRIPT_DIR.parent.parent.resolve()
            model_dir = str(project_root / "models")
        model_path = os.path.join(model_dir, "CodeFormer.pth")
        if not os.path.exists(model_path):
            return False  # No model available, cannot perform inference

        # BUG-051 fix: Use json.dumps() for safe path quoting to prevent single-quote injection
        import json as _json
        safe_input = _json.dumps(input_path)
        safe_model = _json.dumps(model_path)
        safe_output_dir = _json.dumps(output_dir)
        cmd = [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, 'CodeFormer'); "
            f"from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper; "
            f"from codeformer.basicsr.utils.registry import ARCH_REGISTRY; "
            f"import cv2; import torch; import numpy as np; import os; "
            f"img = cv2.imread({safe_input}); "
            f"if img is None: raise RuntimeError('Failed to load image'); "
            f"# Attempt CodeFormer inference via FaceRestoreHelper + ARCH_REGISTRY; "
            f"try: "
            f"    device = 'cuda' if torch.cuda.is_available() else 'cpu'; "
            f"    helper = FaceRestoreHelper(upscale_factor=1, face_size=512); "
            f"    helper.read_image(img); "
            f"    helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5); "
            f"    helper.align_warp_face(); "
            f"    codeformer_net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, "
            f"        n_head=8, n_layers=9, connect_list=['32', '64', '128', '256']).to(device); "
            f"    ckpt = torch.load({safe_model}, map_location=device); "
            f"    codeformer_net.load_state_dict(ckpt['params_ema'] if 'params_ema' in ckpt else ckpt); "
            f"    codeformer_net.eval(); "
            f"    for idx, cropped_face in enumerate(helper.cropped_faces): "
            f"        face_t = torch.from_numpy(cropped_face / 255.).permute(2,0,1).unsqueeze(0).float().to(device); "
            f"        with torch.no_grad(): "
            f"            output = codeformer_net(face_t, w=0.9, adain=True)[0]; "
            f"        restored = (output.squeeze(0).permute(1,2,0).cpu().numpy() * 255).clip(0,255).astype(np.uint8); "
            f"        helper.add_restored_face(restored); "
            f"    helper.get_inverse_affine(None); "
            f"    result = helper.paste_faces_to_input_image(upsample_img=img); "
            f"    out_path = os.path.join({safe_output_dir}, os.path.basename({safe_input})); "
            f"    cv2.imwrite(out_path, result); "
            f"except Exception as e: "
            f"    print('CodeFormer fallback failed:', e, file=sys.stderr); "
            f"    sys.exit(1); ",
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=600, shell=False)
        return r.returncode == 0
    except Exception:
        return False


def _resolve_input_source(temp_dir: str, job_id: str, input_name: str) -> str:
    """Resolve valid input source for face restore.

    Only traditional-processed or post-processed sources are valid.
    Bans raw original input.
    """
    stem = Path(input_name).stem
    for pat in ["step4_postprocess*", "step3_upscale*", "step1_preprocess*"]:
        pattern = f"{job_id}_{stem}_{pat}"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--params")
    parser.add_argument("--params-file")
    parser.add_argument("--config")
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()

    # BUG-057 support: --params-file fallback for long JSON
    if getattr(args, "params_file", None):
        with open(args.params_file, "r", encoding="utf-8") as f:
            params = json.load(f)
    else:
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

    # Resolve input source
    input_source = _resolve_input_source(temp_dir, job_id, input_name)
    if not input_source:
        logger.error("STAGE5_NO_INPUT", "No input found")
        return 82

    output_path = os.path.join(temp_dir,
        f"{job_id}_{Path(input_name).stem}_step5_face.png")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return 3

    logger.info("STAGE5_START", f"Face restore: {face_restore}")

    # Pre-detect faces using configured fallback list; also get boxes for color matching
    face_boxes = _detect_face_boxes(input_source, params, logger)
    if not face_boxes:
        logger.info("STAGE5_NO_FACE_SKIP", "No faces detected, skipping face restore")
        return 3

    face_dir = os.path.join(temp_dir, f"{job_id}_face")
    os.makedirs(face_dir, exist_ok=True)
    blended_path = os.path.join(temp_dir,
        f"{job_id}_{Path(input_name).stem}_step5_face_blended.png")

    success = False
    if face_restore == "GFPGAN":
        if run_gfpgan(input_source, face_dir, params):
            face_files = sorted(Path(face_dir).rglob("*.png"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
            if face_files:
                if feather_blend(input_source, str(face_files[0]), blended_path, feather_width=8):
                    success = True
                else:
                    logger.error("STAGE5_BLEND_FAIL", "Feather blend failed")
                    return 83

    elif face_restore == "CodeFormer":
        if run_codeformer(input_source, face_dir, params):
            face_files = sorted(Path(face_dir).rglob("*.png"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
            if face_files:
                if feather_blend(input_source, str(face_files[0]), blended_path, feather_width=8):
                    success = True
                else:
                    logger.error("STAGE5_BLEND_FAIL", "Feather blend failed")
                    return 83

    if not success:
        logger.error("STAGE5_FAILED", "Face restore failed")
        return 83

    # P-26 fix: half-body portrait color matching per v7.32 §3.10.7
    enable_color_match = params.get("portrait_color_match", True)
    if enable_color_match and face_boxes:
        if _match_portrait_color(blended_path, face_boxes, output_path, logger):
            logger.info("STAGE5_DONE", f"Face restore complete ({face_restore}) with color matching")
        else:
            logger.warning("STAGE5_COLOR_MATCH_SKIP", "Color matching failed, using blended result")
            shutil.copy2(blended_path, output_path)
    else:
        shutil.copy2(blended_path, output_path)
        logger.info("STAGE5_DONE", f"Face restore complete ({face_restore})")

    # PHOTO-AUDIT-014 fix: Grain synthesis runs AFTER all AI processing (v6.32 §3.10.4)
    # G'MIC film_grain must execute in stage5 after face restore completes
    grain_enabled = params.get("grain_synthesis", True)
    if grain_enabled:
        grain_strength = params.get("grain_strength", 0.3)
        grain_size = params.get("grain_size", 1.0)
        grain_roughness = params.get("grain_roughness", 0.5)
        grain_temp = os.path.join(temp_dir,
            f"{job_id}_{Path(input_name).stem}_grain_temp.png")
        logger.info("STAGE5_GRAIN", "Adding film grain via G'MIC (post-AI)")
        if _add_grain_gmic(output_path, grain_temp,
                           grain_strength, grain_size, grain_roughness):
            shutil.move(grain_temp, output_path)
            logger.info("STAGE5_GRAIN_DONE", "Film grain applied after face restore")
        else:
            logger.warning("STAGE5_GRAIN_FAIL", "G'MIC grain synthesis failed, using ungrained output")
            if os.path.exists(grain_temp):
                os.remove(grain_temp)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
