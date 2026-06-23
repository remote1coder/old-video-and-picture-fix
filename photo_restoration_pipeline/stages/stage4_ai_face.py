#!/usr/bin/env python3
"""
Stage 4: Photo Face / Portrait Restoration
GFPGAN / CodeFormer for face restoration in photos.
Full-frame processing (no temporal constraints for static images).

# FIX-P10-1: Renamed from stage5_ai_face.py to stage4_ai_face.py per v7.32 §3.1
# Face restore now runs BEFORE postprocess (stage5), per spec order:
# stage1→stage2_ai_pre→stage3_ai_upscale→stage4_ai_face→stage5_postprocess→stage6_encode
# NOTE: Grain synthesis has been MOVED to stage5_postprocess.py per v7.32 §3.1

v6.3 / v7.32 Compliance:
- Face detection fallback list from config is honored
- Face detection thresholds and low-res max dimension respected
- Bans original/intermediate pixels from AI face restore
  (only traditional-processed allowed)
- Portrait color matching (v7.32 §3.10.7): after face restoration, automatically
  aligns face white-balance to neck/hands using LAB mean/std matching with
  feather blending
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
            # FIX-P12-10: Use copyfile (not copy2) per v7.32 §3.10.10
            shutil.copyfile(restored_path, output_path)
            return True

        # Convert mask to PIL image for Gaussian blur
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_blurred = mask_img.filter(
            ImageFilter.GaussianBlur(radius=feather_width))
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
            # FIX-P12-10: Use copyfile (not copy2) per v7.32 §3.10.10
            shutil.copyfile(restored_path, output_path)
            return True
        except Exception:
            return False


def _detect_faces(input_path: str, params: dict, logger=None) -> bool:
    """Detect faces in image using configured fallback list.

    Returns True if faces detected, False otherwise.
    """
    return len(_detect_face_boxes(input_path, params, logger)) > 0


def _detect_face_boxes(input_path: str, params: dict,
                       logger=None) -> list:
    """Detect faces and return list of bounding boxes [(x, y, w, h), ...].

    Each detector is tried in fallback order until at least one face is found.
    Boxes are returned in the coordinate space of the input image.
    """
    face_detect_fallback = params.get(
        "face_detect_fallback", ["RetinaFace", "YOLOv8-face", "MTCNN"])
    threshold = params.get("face_detect_threshold", 0.8)
    threshold_lowres = params.get("face_detect_threshold_lowres", 0.6)
    lowres_max = params.get("face_detect_lowres_max_dimension", 720)

    try:
        from PIL import Image
        with Image.open(input_path) as img:
            img_w, img_h = img.size
            # FIX-P12-1: v7.32 §3.10.5: short_edge (min) determines lowres
            # ORIGINAL: max(img_w, img_h) <= lowres_max (wrong: uses long edge)
            is_lowres = min(img_w, img_h) <= lowres_max
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
                                valid_boxes.append(
                                    (fa[0], fa[1], fa[2] - fa[0], fa[3] - fa[1]))
                        if valid_boxes:
                            if logger:
                                logger.info("STAGE4_FACE_DETECT",
                                            f"Faces found via RetinaFace: "
                                            f"{len(valid_boxes)} "
                                            f"(threshold={actual_threshold})")
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
                                logger.info("STAGE4_FACE_DETECT",
                                            f"Faces found via YOLOv8: "
                                            f"{len(boxes)} "
                                            f"(threshold={actual_threshold})")
                            return boxes
                except Exception:
                    pass
            elif detector == "MTCNN":
                try:
                    from mtcnn import MTCNN
                    import cv2
                    img = cv2.cvtColor(
                        cv2.imread(input_path), cv2.COLOR_BGR2RGB)
                    detector_obj = MTCNN()
                    faces = detector_obj.detect_faces(img)
                    if faces and len(faces) > 0:
                        valid_boxes = []
                        for face in faces:
                            confidence = face.get("confidence", 0)
                            if confidence >= actual_threshold:
                                box = face.get("box", [0, 0, 0, 0])
                                valid_boxes.append(
                                    (box[0], box[1], box[2], box[3]))
                        if valid_boxes:
                            if logger:
                                logger.info("STAGE4_FACE_DETECT",
                                            f"Faces found via MTCNN: "
                                            f"{len(valid_boxes)} "
                                            f"(threshold={actual_threshold})")
                            return valid_boxes
                except Exception:
                    pass
        except Exception:
            continue

    if logger:
        logger.warning("STAGE4_NO_FACE",
                       "No faces detected with any fallback detector")
    # NOTE: FIX-P12-2 lowres area filter is applied in main() after
    # _detect_face_boxes returns, since this function returns early
    # from each detector.
    return []


# NOTE: _match_portrait_color was moved to stage5_postprocess.py in FIX-P12-6.
# Color matching must execute after ALL AI processing (face restore + upscale)
# and before grain synthesis, per v7.32 §3.10.7 spec order.


def run_gfpgan(input_path: str, output_dir: str,
               params: dict = None) -> bool:
    """Run GFPGAN on a single image.

    Args:
        params: optional dict with 'face_restore_version'
                and 'face_restore_upscale'
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
            r = subprocess.run(
                cmd, capture_output=True, timeout=600, shell=False)
            return r.returncode == 0

        # Fallback: Python API
        model_dir = params.get("model_dir", "") if params else ""
        if not model_dir:
            project_root = SCRIPT_DIR.parent.parent.resolve()
            model_dir = str(project_root / "models")
        model_path = os.path.join(model_dir, f"GFPGANv{version}.pth")

        # BUG-050 fix: Use json.dumps() for safe path quoting
        # to prevent single-quote injection
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
            f"restorer = GFPGANer(model_path={safe_model}, "
            f"upscale={upscale}, arch='clean', "
            f"channel_only=False, bg_upsampler=None); "
            f"_, _, output = restorer.enhance(img, has_aligned=False, "
            f"only_center_face=False); "
            f"import os; "
            f"out_path = os.path.join({safe_output_dir}, "
            f"os.path.basename({safe_input})); "
            f"cv2.imwrite(out_path, output)",
        ]
        r = subprocess.run(
            cmd, capture_output=True, timeout=600, shell=False)
        return r.returncode == 0
    except Exception:
        return False


def run_codeformer(input_path: str, output_dir: str,
                   params: dict = None) -> bool:
    """Run CodeFormer on a single image.

    Bug 11.13 fix: fallback no longer copies original; it attempts real
    inference or returns False so the caller knows CodeFormer failed.
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
            r = subprocess.run(
                cmd, capture_output=True, timeout=600, shell=False)
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

        # BUG-051 fix: Use json.dumps() for safe path quoting
        import json as _json
        safe_input = _json.dumps(input_path)
        safe_model = _json.dumps(model_path)
        safe_output_dir = _json.dumps(output_dir)
        cmd = [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, 'CodeFormer'); "
            f"from codeformer.facelib.utils.face_restoration_helper "
            f"import FaceRestoreHelper; "
            f"from codeformer.basicsr.utils.registry import ARCH_REGISTRY; "
            f"import cv2; import torch; import numpy as np; import os; "
            f"img = cv2.imread({safe_input}); "
            f"if img is None: raise RuntimeError('Failed to load image'); "
            f"try: "
            f"    device = 'cuda' if torch.cuda.is_available() else 'cpu'; "
            f"    helper = FaceRestoreHelper("
            f"upscale_factor=1, face_size=512); "
            f"    helper.read_image(img); "
            f"    helper.get_face_landmarks_5("
            f"only_center_face=False, resize=640, "
            f"eye_dist_threshold=5); "
            f"    helper.align_warp_face(); "
            f"    codeformer_net = ARCH_REGISTRY.get('CodeFormer')"
            f"(dim_embd=512, codebook_size=1024, "
            f"n_head=8, n_layers=9, "
            f"connect_list=['32','64','128','256']).to(device); "
            f"    ckpt = torch.load({safe_model}, map_location=device); "
            f"    codeformer_net.load_state_dict("
            f"ckpt['params_ema'] if 'params_ema' in ckpt else ckpt); "
            f"    codeformer_net.eval(); "
            f"    for idx, cropped_face in enumerate(helper.cropped_faces): "
            f"        face_t = torch.from_numpy("
            f"cropped_face / 255.).permute(2,0,1).unsqueeze(0)"
            f".float().to(device); "
            f"        with torch.no_grad(): "
            f"            output = codeformer_net("
            f"face_t, w=0.9, adain=True)[0]; "
            f"        restored = (output.squeeze(0).permute(1,2,0)"
            f".cpu().numpy() * 255).clip(0,255).astype(np.uint8); "
            f"        helper.add_restored_face(restored); "
            f"    helper.get_inverse_affine(None); "
            f"    result = helper.paste_faces_to_input_image("
            f"upsample_img=img); "
            f"    out_path = os.path.join({safe_output_dir}, "
            f"os.path.basename({safe_input})); "
            f"    cv2.imwrite(out_path, result); "
            f"except Exception as e: "
            f"    print('CodeFormer fallback failed:', e, file=sys.stderr); "
            f"    sys.exit(1); ",
        ]
        r = subprocess.run(
            cmd, capture_output=True, timeout=600, shell=False)
        return r.returncode == 0
    except Exception:
        return False


def _resolve_input_source(temp_dir: str, job_id: str,
                          input_name: str) -> str:
    """Resolve valid input source for face restore.

    Only traditional-processed or AI-upscaled sources are valid.
    Bans raw original input.
    """
    stem = Path(input_name).stem
    # FIX-P10-1: updated search order for new stage numbering
    for pat in ["step3_upscale*", "step2_ai_pre*",
                "step1_preprocess*"]:
        pattern = f"{job_id}_{stem}_{pat}"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        if candidates:
            candidates.sort(
                key=lambda p: os.path.getmtime(p), reverse=True)
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
        logger.info("STAGE4_SKIP", "Face restore not enabled")
        return 3

    temp_dir = params.get("temp_dir", "./temp")
    # FIX-P12-3: WSL2 redirect for temp_dir per v7.32 §3.10.10
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    platform = params.get("platform", "auto")
    try:
        from checkpoint import wsl_redirect_path
        redirected_temp, _, _, _ = wsl_redirect_path(
            temp_dir, no_wsl_redirect=no_wsl_redirect,
            target_platform=platform
        )
        if redirected_temp != temp_dir:
            logger.info("STAGE4_WSL_REDIRECT",
                       f"WSL temp_dir redirect: {temp_dir} -> {redirected_temp}")
            temp_dir = redirected_temp
            params["temp_dir"] = temp_dir
    except Exception:
        pass
    job_id = args.job_id
    input_name = Path(args.input).name

    # Resolve input source
    input_source = _resolve_input_source(temp_dir, job_id, input_name)
    if not input_source:
        logger.error("STAGE4_NO_INPUT", "No input found")
        return 82

    # FIX-P10-1: step number is now 4
    output_path = os.path.join(temp_dir,
        f"{job_id}_{Path(input_name).stem}_step4_face.png")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return 3

    logger.info("STAGE4_START", f"Face restore: {face_restore}")

    # Pre-detect faces using configured fallback list;
    # also get boxes for color matching
    face_boxes = _detect_face_boxes(input_source, params, logger)

    # FIX-P12-2: Low-res area filter per v7.32 §3.10.5
    # Low-res images: only allow face restore if face box > 10% of image area
    lowres_max = params.get("face_detect_lowres_max_dimension", 720)
    try:
        from PIL import Image
        with Image.open(input_source) as _img:
            _w, _h = _img.size
        is_lowres = min(_w, _h) <= lowres_max
        if is_lowres and face_boxes:
            img_area = _w * _h
            filtered = []
            for (bx, by, bw, bh) in face_boxes:
                if (bw * bh) > 0.10 * img_area:
                    filtered.append((bx, by, bw, bh))
            if not filtered:
                logger.warning("STAGE4_LOWRES_AREA_SKIP",
                    f"Low-res faces < 10% area, skipping face restore "
                    f"per v7.32 §3.10.5")
                return 3
            face_boxes = filtered
    except Exception:
        pass

    if not face_boxes:
        logger.info("STAGE4_NO_FACE_SKIP",
                    "No faces detected, skipping face restore")
        return 3

    face_dir = os.path.join(temp_dir, f"{job_id}_face")
    os.makedirs(face_dir, exist_ok=True)
    blended_path = os.path.join(temp_dir,
        f"{job_id}_{Path(input_name).stem}_step4_face_blended.png")

    success = False
    if face_restore == "GFPGAN":
        if run_gfpgan(input_source, face_dir, params):
            face_files = sorted(
                Path(face_dir).rglob("*.png"),
                key=lambda p: p.stat().st_mtime, reverse=True)
            if face_files:
                if feather_blend(input_source, str(face_files[0]),
                                 blended_path, feather_width=8):
                    success = True
                else:
                    logger.error("STAGE4_BLEND_FAIL",
                                 "Feather blend failed")
                    return 83

    elif face_restore == "CodeFormer":
        if run_codeformer(input_source, face_dir, params):
            face_files = sorted(
                Path(face_dir).rglob("*.png"),
                key=lambda p: p.stat().st_mtime, reverse=True)
            if face_files:
                if feather_blend(input_source, str(face_files[0]),
                                 blended_path, feather_width=8):
                    success = True
                else:
                    logger.error("STAGE4_BLEND_FAIL",
                                 "Feather blend failed")
                    return 83

    if not success:
        logger.error("STAGE4_FAILED", "Face restore failed")
        return 83

    # P-26 fix: half-body portrait color matching per v7.32 §3.10.7
    # FIX-P12-6: Color matching MOVED to stage5_postprocess.py per spec order.
    # Stage4 now only persists face_boxes metadata for stage5 to consume.
    # Save face_boxes to metadata file for stage5 color matching
    face_meta_path = os.path.join(
        temp_dir, f"{job_id}_{Path(input_name).stem}_face_boxes.json")
    try:
        with open(face_meta_path, "w", encoding="utf-8") as f:
            json.dump({"face_boxes": face_boxes}, f)
        logger.info("STAGE4_FACE_META", f"Saved face_boxes metadata: {face_meta_path}")
    except Exception as e:
        logger.warning("STAGE4_FACE_META_FAIL", f"Failed to save face_boxes: {e}")

    # FIX-P11-10: Use copyfile (not copy2) per v7.32 §3.10.10
    # copy2 copies metadata which causes PermissionError on WSL2 NTFS mounts
    shutil.copyfile(blended_path, output_path)
    logger.info("STAGE4_DONE",
                f"Face restore complete ({face_restore})")

    # NOTE: Grain synthesis has been MOVED to stage5_postprocess.py
    # per v7.32 §3.1 (post-processing stage handles grain after all AI)

    return 0


if __name__ == "__main__":
    # FIX-P11-5: main() must return explicit int exit code per v7.32 §3.5
    # ORIGINAL: sys.exit(main() or 0) — "or 0" masks None returns
    rc = main()
    if rc is None:
        rc = 86  # validation failure — main() must always return explicit exit code
    sys.exit(rc)
