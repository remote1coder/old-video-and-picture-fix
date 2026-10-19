#!/usr/bin/env python3
"""
Photo Material Classifier
Classifies photos into: SCAN_PRINT, DIGITAL_PHOTO
Detects dimensions, DPI, and AI upscale ban status.

v6.3 Compliance:
- AI upscale banned when target resolution > 1080p OR DPI > 1200
- Computes native_width/native_height/native_dar
- Computes file hash prefix for deduplication
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional  # R4: 真实运行时 NameError: Dict 未导入


def run_exiftool(input_path: str) -> Dict[str, str]:
    """Run exiftool and return key metadata."""
    try:
        r = subprocess.run(
            ["exiftool", "-json", input_path],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            if data:
                return {k: str(v) for k, v in data[0].items()}
    except Exception:
        pass
    return {}


def run_ffprobe(input_path: str) -> Dict[str, Any]:
    """Run ffprobe and return video stream metadata.

    FIX-R11-7: Per v7.32 §3.9, native_width/native_height/native_dar must
    come from ffprobe (EXIF is only auxiliary). ffprobe is the authoritative
    source for native dimensions; EXIF original-size tags are a fallback.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,display_aspect_ratio",
             "-of", "json", input_path],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False
        )
        if r.returncode == 0:
            d = json.loads(r.stdout)
            streams = d.get("streams", [])
            if streams:
                s = streams[0]
                return {
                    "width": s.get("width"),
                    "height": s.get("height"),
                    "display_aspect_ratio": s.get("display_aspect_ratio"),
                }
    except Exception:
        pass
    return {}


def _analyze_pixel_features(input_path: str) -> Dict[str, Any]:
    """Analyze pixel-level features for better classification confidence.
    
    Detects:
    - Blockiness (JPEG/DCT artifact strength)
    - Blur amount (Laplacian variance)
    - Noise level (median filter residual)
    
    Returns dict with features and adjusted confidence hints.
    """
    try:
        from PIL import Image
        import numpy as np
        from scipy.ndimage import laplace, median_filter
        
        with Image.open(input_path) as img:
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        
        gray = arr.mean(axis=2)
        h, w = gray.shape
        
        # Blockiness: detect 8x8 DCT grid artifacts via gradient variance
        # FIX-P11-12: blockiness only meaningful for JPEG per v7.32 §3.9
        # ORIGINAL: computed for all formats (PNG/TIFF have no DCT artifacts)
        blockiness = 0.0
        if h >= 16 and w >= 16 and str(input_path).lower().endswith(
                ('.jpg', '.jpeg')):
            dx = np.diff(gray, axis=1)
            dy = np.diff(gray, axis=0)
            # Measure periodicity at 8-pixel intervals
            block_var_x = np.var(dx[:, ::8]) if w > 8 else 0.0
            block_var_y = np.var(dy[::8, :]) if h > 8 else 0.0
            blockiness = float((block_var_x + block_var_y) / 2.0)
        
        # Blur: Laplacian variance (lower = more blurred)
        lap_var = float(np.var(laplace(gray)))
        
        # Noise: median filter residual std
        med = median_filter(gray, size=3)
        noise_std = float(np.std(gray - med))
        
        return {
            "blockiness": blockiness,
            "blur_laplacian_var": lap_var,
            "noise_std": noise_std,
        }
    except Exception:
        return {"blockiness": 0.0, "blur_laplacian_var": 0.0, "noise_std": 0.0}


#: 扫描仪常见的"整数"输出分辨率（DPI）。
SCANNER_DPI_DISCRETE = {300, 600, 1200, 2400, 3200, 4800, 6400}


#: 只有"拍摄"才会产生的 EXIF 标签。
#:
#: 扫描仪也会写 Make/Model（EPSON、Brother、Canon 的扫描仪部门都写），
#: 但它**不会**写曝光时间/光圈/ISO/焦距 —— 那些是镜头和快门才有的量。
#: 所以这几个标签是区分"相机拍的"和"扫描仪扫的"最可靠的信号。
CAMERA_CAPTURE_TAGS = (
    "ExposureTime", "ShutterSpeedValue",
    "FNumber", "ApertureValue",
    "ISO", "ISOSpeedRatings", "PhotographicSensitivity",
    "FocalLength", "DigitalZoomRatio",
)


#: 扫描仪型号里常见的字样。
#:
#: 为什么需要这一层：`Make/Model + 无拍摄参数 + 300 DPI` 这个组合
#: **相机和扫描仪都可能出现**（相机直出也可能不带曝光参数、也被设 300 DPI）。
#: 实测：
#:   `EPSON / Perfection V600`  → 想判扫描件
#:   `Canon / EOS 5D`（无拍摄参数）→ 想判数码照片
#: 两者元数据形状一样，只能靠型号名里的关键词再分一层。
#:
#: **这个表不可能完备** —— 没列到的扫描仪会被判成 DIGITAL_PHOTO。
#: 这是有意的取舍：宁可漏判扫描件（只是参数组选得不够贴），
#: 也不要把相机照片误判成扫描件。
SCANNER_MODEL_KEYWORDS = (
    "scanner", "scanjet", "canoscan", "lidescan", "opticslim",
    "imagescan", "perfection", "v600", "v800", "v850", "es-", "ds-",
    "fl-", "snapscan", "plustek", "opticbook",
)


def _model_looks_like_scanner(model: str) -> bool:
    """型号名里是否带扫描仪特征字样。"""
    if not model:
        return False
    m = str(model).lower()
    return any(kw in m for kw in SCANNER_MODEL_KEYWORDS)


def _looks_like_camera_capture(exif: Dict[str, Any]) -> bool:
    """EXIF 里是否有"拍照"才有的参数。

    有任意一个即认为是相机拍摄。扫描仪不产生这些标签，
    所以 Make/Model 写着 EPSON 也不会被冤枉成相机。

    注意：截图、导出图、AI 生成图可能**一个都没有** —— 那不代表是扫描件，
    只是"没有拍摄信息"，调用方要按"证据不足"处理，不能直接判扫描。
    """
    if not exif:
        return False
    for tag in CAMERA_CAPTURE_TAGS:
        v = exif.get(tag)
        if v in (None, "", 0, "0"):
            continue
        return True
    return False


def _is_scanner_dpi(dpi: Any, tolerance: int = 5) -> bool:
    """DPI 是否落在扫描仪常见档位上。

    仅凭 DPI 不能断定是扫描件（相机照片也可能被设成 300 DPI 用于打印），
    所以调用方还要结合"有没有相机型号"一起判断。
    """
    if not dpi:
        return False
    try:
        d = float(dpi)
    except (TypeError, ValueError):
        return False
    return any(abs(d - sd) <= tolerance for sd in SCANNER_DPI_DISCRETE)


def classify_photo(input_path: str, config: dict = None) -> Dict[str, Any]:
    """Classify photo and return metadata.

    Args:
        input_path: Path to photo file.
        config: Optional config dict for threshold overrides.
            Reads max_long_edge, max_short_edge, dpi_threshold from config.

    Returns:
    - detected_class: SCAN_PRINT | DIGITAL_PHOTO
    - confidence: float 0-1
    - original_width, original_height
    - native_width, native_height, native_dar
    - dpi: int or None
    - ai_upscale_banned: bool (based on pixel dimensions, target resolution, and DPI)
    - field_order: per v7.32 §3.9, five valid values exist:
        "tt" (top-field-first), "bb" (bottom-field-first),
        "psf" (progressive segmented frame), "progressive" (native progressive),
        "unknown" (cannot determine).
      # FIX-R11-4: Corrected docstring to spec values (tt/bb/psf/progressive/
      # unknown). ORIGINAL: listed tff/bff/mixed which are NOT spec values.
      Photo classify only emits "progressive" (photos are inherently
      progressive) or "unknown" (unreadable file); tt/bb/psf are
      video-classify concerns.
    """
    # FIX-P11-2: Read thresholds from config instead of hardcoding per v6.32 §7.7
    cfg = config or {}
    max_long = cfg.get("max_long_edge", 3840)
    max_short = cfg.get("max_short_edge", 2160)
    dpi_threshold = cfg.get("dpi_threshold", 1200)

    result: Dict[str, Any] = {
        "detected_class": "UNKNOWN",
        "confidence": 0.0,
        "original_width": 0,
        "original_height": 0,
        "native_width": 0,
        "native_height": 0,
        "native_dar": "",
        "dpi": None,
        "ai_upscale_banned": False,
        "field_order": "progressive",  # FIX-P11-6: Photos are inherently progressive per v7.32 §3.9
        "file_size": 0,
        "file_mtime": 0,
        "file_hash_prefix": "",
    }

    if not Path(input_path).exists():
        return result

    # File stats
    stat = Path(input_path).stat()
    result["file_size"] = stat.st_size
    result["file_mtime"] = int(stat.st_mtime)

    # Try to get dimensions via PIL
    try:
        from PIL import Image
        with Image.open(input_path) as img:
            result["original_width"], result["original_height"] = img.size
            # PHOTO-AUDIT-005: Try to read EXIF original dimensions for native_*
            native_w, native_h = img.size
            try:
                exif_data = img._getexif()
                if exif_data:
                    # EXIF tag 0xA000 (ExifImageWidth) and 0xA001 (ExifImageHeight)
                    if 0xA000 in exif_data and 0xA001 in exif_data:
                        native_w = exif_data[0xA000]
                        native_h = exif_data[0xA001]
            except Exception:
                pass
            result["native_width"], result["native_height"] = native_w, native_h
            # FIX-P11-1: Compute native DAR from EXIF native dimensions, not current file size
            # per v7.32 §3.9: native_dar must use native_width/native_height (EXIF original)
            # ORIGINAL: w, h = img.size (used current file size, wrong for cropped images)
            w, h = native_w, native_h
            # FIX-P11-14: Check both w>0 and h>0 per v7.32 §3.9
            # ORIGINAL: only checked h>0, division by zero if w=0
            if w > 0 and h > 0:
                from math import gcd
                g = gcd(w, h)
                result["native_dar"] = f"{w//g}:{h//g}"
            else:
                result["native_dar"] = ""
            # Try to get DPI
            dpi_info = img.info.get("dpi")
            if dpi_info:
                result["dpi"] = int(max(dpi_info))
    except Exception:
        pass

    # FIX-R11-7: Per v7.32 §3.9, ffprobe is the authoritative source for
    # native_width/native_height/native_dar (EXIF above is only auxiliary).
    # Override with ffprobe values when available; keep EXIF as fallback.
    fp = run_ffprobe(input_path)
    if fp.get("width") and fp.get("height"):
        result["native_width"] = int(fp["width"])
        result["native_height"] = int(fp["height"])
        if fp.get("display_aspect_ratio"):
            result["native_dar"] = str(fp["display_aspect_ratio"])
        else:
            from math import gcd
            _w, _h = int(fp["width"]), int(fp["height"])
            if _w > 0 and _h > 0:
                _g = gcd(_w, _h)
                result["native_dar"] = f"{_w//_g}:{_h//_g}"

    # FIX-P11-2: field_order for unreadable images per v7.32 §3.9
    if result["original_width"] == 0 or result["original_height"] == 0:
        result["field_order"] = "unknown"  # Cannot determine if file unreadable

    # ExifTool metadata
    exif = run_exiftool(input_path)

    # FIX-P11-4: Try exiftool XResolution/YResolution if PIL DPI not found
    # Per v7.32 §3.9: classification uses exiftool metadata for DPI
    if result["dpi"] is None:
        x_res = exif.get("XResolution", "")
        y_res = exif.get("YResolution", "")
        try:
            res_str = x_res or y_res
            if "/" in str(res_str):
                num, den = str(res_str).split("/")
                result["dpi"] = int(int(num) / int(den))
            elif res_str:
                result["dpi"] = int(float(res_str))
        except (ValueError, ZeroDivisionError):
            pass

    # Pixel-level feature analysis for enhanced classification
    pixel_features = _analyze_pixel_features(input_path)
    
    # Classification
    model = exif.get("Model", "")
    make = exif.get("Make", "")
    scanner = exif.get("ScannerModelName", "")
    software = exif.get("Software", "")

    # Case-insensitive scanner detection (Bug-18 fix)
    scanner_lower = scanner.lower() if scanner else ""
    software_lower = software.lower() if software else ""
    make_lower = make.lower() if make else ""

    if scanner_lower or "scan" in software_lower or "scan" in make_lower:
        result["detected_class"] = "SCAN_PRINT"
        result["confidence"] = 0.95
    # FIX-R7-A1: DPI 分支**不能短路** `elif model or make`。
    #
    # 原实现是 `elif result.get("dpi"):` —— 只要 DPI 非空就进这个分支，
    # 然后在里面判断"是不是标准扫描 DPI"（300/600/...）；不匹配时**什么
    # 都不做**就结束了，后面的 `elif model or make:` 永远走不到。
    #
    # 实测后果（2026-10-20，一张带 Make=Canon / Model=EOS 5D 的图）：
    #   DPI=72  → 进入本分支、不匹配、什么都不做 → 最终 UNKNOWN / 0.0
    #             → main 判 LOW_CONFIDENCE_SKIP，整张图被跳过
    #   DPI=300 → 匹配 → 被判成 SCAN_PRINT / 0.80
    #             （相机照片被当成扫描件，走错参数组）
    #
    # 也就是说：**任何带常见 DPI 的数码相机照片都过不了分类**。
    #
    # 现在把"是否是标准扫描 DPI"收进条件本身，并且要求**没有相机型号**
    # 才按扫描件处理（扫描件通常不写 Make/Model）。
    # 已知局限：某些扫描仪会写 Make=EPSON/Brother，这类仍会被后面的
    # `model or make` 分支判成 DIGITAL_PHOTO —— 比"整张图被跳过"好，
    # 但没有品牌白名单就无法彻底区分，这里不引入白名单以免新的误判。
    elif model or make:
        # 有 Make/Model 时，用**有没有拍摄参数**来区分相机和扫描仪。
        #
        # 上一版只判断"有没有 Make/Model"，于是扫描仪写的 `Make=EPSON`
        # 会被判成 DIGITAL_PHOTO。加上拍摄参数这一层就能分开：
        #   Canon + ExposureTime/FNumber/ISO → 相机拍的
        #   EPSON + 只有 Make/Model         → 扫描仪扫的
        if _looks_like_camera_capture(exif):
            # 有曝光/光圈/ISO/焦距 → 一定是镜头拍的
            result["detected_class"] = "DIGITAL_PHOTO"
            result["confidence"] = 0.90
            result["classify_basis"] = "camera_make_model_with_capture_tags"
        elif _model_looks_like_scanner(str(model) if model else ""):
            # 型号名带 scanner/perfection/v600 等 → 扫描仪
            result["detected_class"] = "SCAN_PRINT"
            result["confidence"] = 0.80
            result["classify_basis"] = "scanner_model_keyword"
        else:
            # 有型号但没拍摄参数、型号也不像扫描仪。
            # 这类元数据"形状"和扫描仪一样，无法再细分 —— 保守判数码照片，
            # 理由见 SCANNER_MODEL_KEYWORDS 的注释（宁可漏判扫描件）。
            result["detected_class"] = "DIGITAL_PHOTO"
            result["confidence"] = 0.85
            result["classify_basis"] = ("make_model_without_capture_tags"
                                        "_not_scanner_like")
    elif (result.get("dpi") and _is_scanner_dpi(result["dpi"])):
        result["detected_class"] = "SCAN_PRINT"
        result["confidence"] = 0.80
        result["classify_basis"] = "scanner_dpi_no_make_model"
    else:
        # Default based on resolution + pixel features
        w = result["original_width"]
        h = result["original_height"]
        if w > 0 and h > 0:
            if w > 3000 or h > 2000:
                result["detected_class"] = "SCAN_PRINT"
                result["confidence"] = 0.70
                result["classify_basis"] = "resolution_fallback_large"
            else:
                result["detected_class"] = "DIGITAL_PHOTO"
                result["confidence"] = 0.60
                result["classify_basis"] = "resolution_fallback_small"
        else:
            result["detected_class"] = "UNKNOWN"
            result["confidence"] = 0.0
            result["classify_basis"] = "no_metadata_no_dimensions"
    
    # Adjust confidence based on pixel features
    # High blockiness suggests scanned/printed material (JPEG recompression)
    # High noise + moderate blur suggests old photo scan
    if pixel_features["blockiness"] > 0.01:
        result["confidence"] = min(0.99, result["confidence"] + 0.05)
    if pixel_features["noise_std"] > 0.02 and pixel_features["blur_laplacian_var"] < 0.01:
        # Old scanned photo pattern: noisy but soft
        if result["detected_class"] == "SCAN_PRINT":
            result["confidence"] = min(0.99, result["confidence"] + 0.03)

    # AI upscale ban check (v6.32 §4.2.1 / v7.32 §3.10.2)
    # FIX-P11-2: Unified ban logic using config-driven thresholds (not hardcoded)
    # FIX-P11-11: Corrected spec reference from "v6.3 §4.4" (nonexistent) to
    # "v6.32 §4.2.1 / v7.32 §3.10.2" per actual document structure
    # ORIGINAL: comment referenced "v6.3 §4.4" which does not exist
    w = result["original_width"]
    h = result["original_height"]
    dpi = result.get("dpi")

    # Ban 1: source pixel dimensions exceed max_long_edge/max_short_edge thresholds
    # FIX-P11-1: v7.32 §3.10.2: judge by long_edge/short_edge, NOT fixed width/height
    # ORIGINAL: if w > max_long or h > max_short: (wrong for portrait images)
    long_edge = max(w, h)
    short_edge = min(w, h)
    ban_reasons = []
    if long_edge > max_long or short_edge > max_short:
        result["ai_upscale_banned"] = True
        ban_reasons.append(f"pixel_exceeds:{long_edge}x{short_edge}>{max_long}x{max_short}")

    # Ban 2: DPI exceeds config threshold (high-DPI scans have sufficient detail)
    if dpi and dpi > dpi_threshold:
        result["ai_upscale_banned"] = True
        ban_reasons.append(f"dpi_exceeds:{dpi}>{dpi_threshold}")

    if ban_reasons:
        result["ban_reasons"] = ban_reasons

    # FIX-P11-11: v6.32 §4.2.1 / v7.32 §3.10.2 predictive ban: if source at 2x would exceed 1080p target
    # This is a soft indicator - actual enforcement is in Stage 3
    # We don't set ai_upscale_banned=True here, just log for reference
    # Stage 3 enforces the actual 1080p target resolution limit

    # Compute hash prefix
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "shared_modules"))
        from checkpoint import get_file_xxhash
        result["file_hash_prefix"] = get_file_xxhash(input_path)
    except ImportError:
        # NEW-V-01: 同 video/classify.py —— xxhash 缺失必须快速失败，
        # 否则 file_hash_prefix 恒为 ""，manifest_generate 的变更检测保护被静默绕过。
        raise
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "file_hash_prefix degraded for %s (%s: %s)",
            input_path, type(exc).__name__, exc,
        )

    # FIX-P11-3: v7.32 §3.9: Confidence threshold handling
    # Confidence < 80%: interactive mode requires manual confirmation,
    # batch mode marks SKIP in manifest
    # FIX-P11-12: UNKNOWN class gets special skip reason per v7.32 §3.9
    CONFIDENCE_THRESHOLD = 0.80
    if result["detected_class"] == "UNKNOWN":
        result["needs_confirmation"] = True
        result["skip_reason"] = (
            f"Cannot classify image (UNKNOWN), "
            f"confidence={result['confidence']:.2f}. "
            f"Manual classification required."
        )
    elif result["confidence"] < CONFIDENCE_THRESHOLD:
        result["needs_confirmation"] = True
        result["skip_reason"] = (
            f"Low confidence ({result['confidence']:.2f} < {CONFIDENCE_THRESHOLD}), "
            f"class={result['detected_class']}"
        )

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python classify.py <photo_file> [config_json]")
        sys.exit(81)

    config = None
    if len(sys.argv) >= 3:
        try:
            config = json.loads(sys.argv[2])
        except Exception:
            config = None

    result = classify_photo(sys.argv[1], config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
