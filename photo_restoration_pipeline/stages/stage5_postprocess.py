#!/usr/bin/env python3
"""
Stage 5: Photo Post-processing
- Lanczos resize to target if non-integer (only when AI output needs it)
- Selective local processing with region separation
- Pure Python Laplacian pyramid grain synthesis (v7.32 §3.10.4)
- Half-body portrait color matching (v7.32 §3.10.7)
- 16-bit space operations for texture enhancement
- FLIF Cb/Cr layer extraction (placeholder for archival pipeline)

# FIX-P10-1: Renamed from stage4_postprocess.py to stage5_postprocess.py per v7.32 §3.1
# Stage order is now: stage1→stage2_ai_pre→stage3_ai_upscale→stage4_ai_face→stage5_postprocess→stage6_encode

v6.3/v7.3 Compliance:
- Separates flat areas from edges/textures for selective processing
- Operates in 16-bit space when photo_bit_depth=16
- Lanczos resize only applies when target is explicitly set (not "auto")
- Grain synthesis via Laplacian pyramid (all AI processing must complete first)
  # FIX-R13-1: Replaced non-existent G'MIC -film_grain with pure Python
  # Laplacian pyramid implementation. Algorithm matches DaVinci Resolve
  # Film Grain / Adobe Camera Raw Grain core approach.
"""

import hashlib
import os
import sys
import json
import argparse
import glob
import subprocess
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
from checkpoint import safe_input_stem  # noqa: E402

from logger import get_logger  # noqa: E402

try:  # NEW-V-03: 独立于 Pillow 的真实位深读取通道
    import media_io as _media_io
except Exception:  # pragma: no cover - 缺失时退回旧行为并留痕
    _media_io = None

try:  # NEW-V-06: 取消/中断守卫（抓 KeyboardInterrupt，except Exception 抓不到）
    import cancel_guard as _cancel_guard
except Exception:  # pragma: no cover - 缺失时不阻断 stage，仅失去该保护
    _cancel_guard = None


def read_true_input_array(path: str):
    """Return ``(array, icc_bytes)`` for the input's real uint16 data.

    NEW-V-03: ``PIL.Image.open()`` on a 16-bit **RGB** TIFF returns mode
    ``'RGB'`` backed by **uint8** data truncated to the high byte
    (实测: 257 -> 1)。stage5 原本据此走 ``convert("RGB")``，
    于是 16-bit RGB 素材在 stage5 就被静默压成 8-bit，
    后面的 "16-bit 归档" 只剩容器是 16-bit、内容只有 8-bit 的信息量。
    这里改用 tifffile（media_io.read_image_array）判定真实位深。

    R2-02 / A03：同时把**源 ICC**带回来。旧实现丢弃 meta，
    于是输入 PNG 里 588 字节的 sRGB profile 在 stage5 输出里变成 0 字节 ——
    像素与元数据必须一起传递。

    Returns:
        ``(None, icc_or_None)`` 当输入不是 16-bit；读取彻底失败时同。
    """
    if _media_io is None:
        return None, None
    icc = None
    try:
        arr, meta = _media_io.read_image_array(path)
        icc = (meta or {}).get("icc_profile")
    except getattr(_media_io, "MediaPrecisionError", Exception):
        # R2-02 / A04：高精度读取失败是**必须向上抛**的硬错误。
        # 旧实现 `except Exception: return None`，于是调用方退回 Pillow 的
        # 8-bit 通道，把整幅图压成 257 的整数倍再装回 uint16，最后 rc=0。
        # 底层"响亮失败"没有贯穿调用方，等于没失败。
        raise
    except Exception:
        return None, None
    if arr is None:
        return None, icc
    if getattr(arr, "dtype", None) != np.uint16:
        return None, icc
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        return None, icc
    return arr, icc


#: R2-03：单维度上限。超过这个值基本可以断定是笔误/恶意输入，
#: 真要去分配这么大的中间缓冲也会把机器拖死，因此在参数层就拒绝。
_MAX_LANCZOS_DIMENSION = 32768


def _cached_was_degraded(cached_meta: Optional[Dict[str, Any]]) -> bool:
    """R2-03: whether a cache entry records operations that were *not* performed."""
    if not isinstance(cached_meta, dict):
        return False
    skipped = cached_meta.get("skipped_operations")
    return bool(skipped) if isinstance(skipped, (list, tuple)) else bool(skipped)


def parse_lanczos_target(value: Any) -> Tuple[Optional[int], Optional[int],
                                              Optional[str]]:
    """R2-03: validate an explicitly requested resize target.

    Returns ``(w, h, error)``；``error`` 非 None 表示请求非法。

    旧实现解析失败或解析出 0 时只记一条 warning 就跳过，然后**照常返回成功**
    （A14：`lanczos_target="0x0"` → rc=0，输出照写）。
    "请求了却没做"必须在参数层拒绝，不能靠后面"看起来没坏"蒙过去。
    """
    if value is None:
        return None, None, None
    text = str(value).strip()
    if text in ("", "auto"):
        return None, None, None
    if "x" not in text:
        return None, None, f"not in <width>x<height> form: {value!r}"
    w_text, _, h_text = text.partition("x")
    try:
        w = int(w_text)
        h = int(h_text)
    except (TypeError, ValueError):
        return None, None, f"non-integer dimensions: {value!r}"
    if w <= 0 or h <= 0:
        return None, None, f"dimensions must be positive: {value!r}"
    if w > _MAX_LANCZOS_DIMENSION or h > _MAX_LANCZOS_DIMENSION:
        return None, None, (f"dimension exceeds the supported maximum "
                            f"{_MAX_LANCZOS_DIMENSION}: {value!r}")
    return w, h, None


def _resize_array_high_depth(arr: "np.ndarray", target_w: int, target_h: int):
    """RV-06: resize a uint16 array *at* 16-bit precision.

    Returns ``(resized_array, method)`` or ``(None, reason)``. cv2's Lanczos4
    is preferred (it keeps uint16 end-to-end); scipy's cubic zoom is the
    fallback. Nothing is silently skipped — the caller decides between an
    authorised DEGRADED result and a hard failure.
    """
    import numpy as np
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] > 4:
        return None, f"unsupported channel count {arr.shape[2]}"
    try:
        import cv2  # type: ignore
        out = cv2.resize(arr, (int(target_w), int(target_h)),
                         interpolation=cv2.INTER_LANCZOS4)
        if out.dtype == np.uint16:
            if out.ndim == 2:
                out = out[:, :, None]
            return out, "cv2:INTER_LANCZOS4(uint16)"
    except Exception:
        pass
    try:
        from scipy import ndimage  # type: ignore
        zoom_factors = (target_h / arr.shape[0], target_w / arr.shape[1], 1)
        out = ndimage.zoom(arr.astype(np.float32), zoom_factors, order=3)
        out = np.clip(out, 0, 65535).astype(np.uint16)
        return out, "scipy:ndimage.zoom(order=3)"
    except Exception:
        pass
    return None, "no high-depth resampler available (cv2 and scipy missing)"


def _box_to_pixel_bounds(box, w_img: int, h_img: int):
    """把检测框规范化成**整数半开区间**并裁到图像内。

    R12-13：stage4 保存的是 **float** 坐标（facexlib 原样输出），stage5
    直接 `max(0, fx)` / `min(w, fx + fw)` 得到的仍是 float，
    `img[y1:y2, x1:x2]` 立刻抛 TypeError。项目内三次照片成功日志都在同一行
    记录了这个错误 —— 人像颜色匹配因此从未真正跑过。

    契约：
    - 非有限值（NaN/Inf）或非数值 → 返回 None，由调用方降级；
    - 起点 **floor**、终点 **ceil**（半开区间，保证不丢边缘像素）；
    - 裁到 ``[0, w]`` / ``[0, h]``；退化为空框时返回 None。

    Args:
        box: ``(x, y, w, h)``，允许 float。
        w_img/h_img: 图像宽高（像素）。

    Returns:
        ``(x1, y1, x2, y2)`` 全为 `int`，且 ``x2 > x1 and y2 > y1``；
        不合法时返回 ``None``。
    """
    import math

    try:
        fx, fy, fw, fh = (float(v) for v in box[:4])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (fx, fy, fw, fh)):
        return None
    if fw <= 0 or fh <= 0:
        return None

    x1 = max(0, int(math.floor(fx)))
    y1 = max(0, int(math.floor(fy)))
    x2 = min(int(w_img), int(math.ceil(fx + fw)))
    y2 = min(int(h_img), int(math.ceil(fy + fh)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _detect_text_regions(gray: np.ndarray, min_aspect_ratio: float = 2.0) -> np.ndarray:
    """Detect text-like regions using stroke width approximation.

    Uses high-gradient areas with high aspect ratio connected components
    as a proxy for text regions. Returns a binary mask.
    """
    try:
        from scipy import ndimage
        # High gradient areas
        dx = ndimage.sobel(gray, axis=1)
        dy = ndimage.sobel(gray, axis=0)
        gradient = np.sqrt(dx**2 + dy**2)
        grad_norm = gradient / (gradient.max() + 1e-8)

        # Threshold for high-contrast strokes
        stroke_mask = grad_norm > 0.15

        # Label connected components
        labeled, num_features = ndimage.label(stroke_mask)
        text_mask = np.zeros_like(gray, dtype=np.float32)

        # ---- 性能修复（2026-10-27，实测 20 分钟超时的根因）----
        # 原实现每个连通域都做两次**全图**扫描：
        #     coords = np.argwhere(labeled == i)   # O(全图)
        #     text_mask[labeled == i] = 1.0        # 又一次 O(全图)
        # 实测：AI 放大后的 8000×6000 照片有 **183,300** 个连通域，
        # 单次全图扫描 0.09s → 光这两行就约 268 分钟。
        # stage5 的超时是 1200s，于是**永远跑不完**，且中途没有任何
        # 进度日志（20 分钟零输出，看起来像"卡死"其实是算不完）。
        #
        # 改用 ndimage.find_objects 取每个连通域的 bounding box，
        # 只在 bbox 的局部切片内比较/写入 —— 复杂度从
        # O(域数 × 全图像素) 降到 O(全图像素 + Σ bbox 面积)。
        slices = ndimage.find_objects(labeled)
        for i, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            # 只在 bbox 内比较，避免全图扫描
            sub = (labeled[sl] == i)
            n_px = int(sub.sum())
            if n_px < 20:  # Too small
                continue
            h = sl[0].stop - sl[0].start
            w = sl[1].stop - sl[1].start
            if w == 0 or h == 0:
                continue
            aspect = max(h, w) / min(h, w)
            density = n_px / (h * w)
            # Text-like: high aspect ratio, moderate density
            if aspect >= min_aspect_ratio and 0.1 <= density <= 0.8:
                # 基础切片返回的是 view，按 sub 掩码写入会回写到 text_mask
                text_mask[sl][sub] = 1.0

        # Dilate slightly to cover character bodies
        text_mask = ndimage.binary_dilation(text_mask, iterations=2).astype(
            np.float32)
        return text_mask
    except Exception:
        return np.zeros_like(gray, dtype=np.float32)


def _separate_regions(img_arr: np.ndarray, edge_threshold: float = 0.05,
                       detect_text: bool = True) -> tuple:
    """Separate flat areas from edge/texture/text regions.

    Uses gradient magnitude via Sobel operators for edge detection,
    combined with local variance for texture detection.
    Includes text detection to protect text regions from smoothing.

    Returns: (flat_mask, edge_mask, texture_mask, text_mask)
    Each mask is float32 in [0, 1].
    """
    from scipy.ndimage import sobel, gaussian_filter

    # Convert to grayscale for analysis
    if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
        gray = img_arr.mean(axis=2)
    else:
        gray = img_arr

    # Sobel gradient magnitude for edges
    dx = sobel(gray, axis=1)
    dy = sobel(gray, axis=0)
    gradient = np.sqrt(dx**2 + dy**2)
    grad_max = gradient.max()
    if grad_max > 0:
        gradient = gradient / grad_max

    # Local variance for texture detection
    local_mean = gaussian_filter(gray, sigma=2)
    local_var = gaussian_filter(gray**2, sigma=2) - local_mean**2
    var_max = local_var.max()
    if var_max > 0:
        texture_strength = local_var / var_max
    else:
        texture_strength = np.zeros_like(gray)

    # Edge mask: high gradient
    edge_mask = np.clip(
        (gradient - edge_threshold * 0.5) / edge_threshold, 0.0, 1.0)

    # Texture mask: high variance but not edge
    texture_mask = np.clip(
        texture_strength * 2.0, 0.0, 1.0) * (1.0 - edge_mask)

    # Text detection (v6.3 §4.2.4: protect text regions)
    text_mask = np.zeros_like(gray, dtype=np.float32)
    if detect_text:
        text_mask = _detect_text_regions(gray)
        # Subtract text from edge/texture to avoid double-counting
        edge_mask = np.clip(edge_mask - text_mask, 0.0, 1.0)
        texture_mask = np.clip(texture_mask - text_mask, 0.0, 1.0)

    # Flat mask: low gradient AND low variance AND not text
    flat_mask = np.clip(
        1.0 - edge_mask - texture_mask - text_mask, 0.0, 1.0)

    return (flat_mask.astype(np.float32), edge_mask.astype(np.float32),
            texture_mask.astype(np.float32), text_mask.astype(np.float32))


def _apply_selective_processing(img: "Image.Image", params: dict,
                                 logger=None,
                                 src_array=None) -> "Image.Image":
    """Apply selective local processing with region separation.

    - Flat areas: mild denoise
    - Edges: sharpening/enhancement
    - Textures: subtle enhancement
    - Text regions: fully protected (no modification)

    NEW-V-03: ``src_array`` 允许调用方直接喂入 **真实 uint16** 数组，
    绕开 Pillow 把 16-bit RGB TIFF 读成 8-bit 的缺陷。
    """
    import numpy as np
    from PIL import Image

    # Handle I;16 input with full precision (Bug 11.9 fix)
    is_16bit_input = img.mode == "I;16" if img is not None else False
    bit_depth = params.get("photo_bit_depth", 16)

    if src_array is not None:
        # NEW-V-03: 真实 16-bit 素材（Pillow 无法表达的三通道 uint16）
        raw_arr = src_array.astype(np.float32) / 65535.0
        if raw_arr.ndim == 2:
            raw_arr = np.stack([raw_arr, raw_arr, raw_arr], axis=2)
        if raw_arr.shape[2] == 4:  # 丢弃 alpha，后续流程按 3 通道处理
            raw_arr = raw_arr[:, :, :3]
        orig_arr = raw_arr
        max_val = 1.0
    elif is_16bit_input and bit_depth == 16:
        # Preserve 16-bit precision: load as uint16, normalize to [0,1]
        orig = img  # Keep I;16
        raw_arr = np.array(orig, dtype=np.float32) / 65535.0
        # I;16 images are 2D (H, W) — expand to 3-channel (H, W, 3)
        # so downstream region separation and per-channel processing work
        if raw_arr.ndim == 2:
            raw_arr = np.stack([raw_arr, raw_arr, raw_arr], axis=2)
        orig_arr = raw_arr
        max_val = 1.0
    else:
        if is_16bit_input:
            # Fallback to 8-bit if bit_depth != 16
            # FIX-P12-4: do NOT convert to L first, that destroys color
            orig = img.point(lambda p: p / 256).convert("RGB")
        else:
            orig = img.convert("RGB")
        orig_arr = np.array(orig).astype(np.float32)
        max_val = 255.0

    # Normalize to [0, 1]
    arr_norm = orig_arr / max_val

    # Separate regions (now includes text detection)
    flat_mask, edge_mask, texture_mask, text_mask = _separate_regions(arr_norm)

    if logger and text_mask.max() > 0:
        text_ratio = float(text_mask.mean())
        logger.info("STAGE5_TEXT_DETECT",
                    f"Text regions detected: {text_ratio*100:.2f}% of image")

    # Flat area processing: mild gaussian blur
    from scipy.ndimage import gaussian_filter
    flat_smooth = np.zeros_like(arr_norm)
    for c in range(3):
        flat_smooth[:, :, c] = gaussian_filter(arr_norm[:, :, c], sigma=1.0)

    # Edge processing: mild unsharp mask (enhance edges)
    edge_enhanced = np.zeros_like(arr_norm)
    for c in range(3):
        blurred = gaussian_filter(arr_norm[:, :, c], sigma=2.0)
        edge_enhanced[:, :, c] = (
            arr_norm[:, :, c] + 0.3 * (arr_norm[:, :, c] - blurred))

    # Texture processing: subtle enhancement (slight contrast boost)
    texture_enhanced = np.zeros_like(arr_norm)
    for c in range(3):
        mean_val = arr_norm[:, :, c].mean()
        texture_enhanced[:, :, c] = (
            (arr_norm[:, :, c] - mean_val) * 1.05 + mean_val)

    # Text regions: fully preserve original (no modification)
    text_preserved = arr_norm.copy()

    # Combine regions
    combined = (
        flat_mask[:, :, np.newaxis] * flat_smooth +
        edge_mask[:, :, np.newaxis] * edge_enhanced +
        texture_mask[:, :, np.newaxis] * texture_enhanced +
        text_mask[:, :, np.newaxis] * text_preserved +
        (1.0 - flat_mask[:, :, np.newaxis]
         - edge_mask[:, :, np.newaxis]
         - texture_mask[:, :, np.newaxis]
         - text_mask[:, :, np.newaxis]) * arr_norm)

    combined = np.clip(combined, 0.0, 1.0)

    # Convert back to appropriate bit depth for PIL
    if bit_depth == 16:
        # FIX-P12-5/P12-6: I;16 only supports 2D arrays in PIL.
        # For 3-channel 16bit, return numpy array for caller to save as TIFF.
        result_arr = (combined * 65535.0).astype(np.uint16)
        return result_arr, 16  # Return (array, bit_depth) tuple
    else:
        result_arr = (combined * 255.0).astype(np.uint8)
        return Image.fromarray(result_arr), 8


def _chin_y_from_landmarks_photo(landmarks, face_box) -> int:
    """Estimate chin y from 5-point landmarks (photo pipeline, RGB float32).

    FIX-H-12-06: Per v7.32 §3.10.7, use landmark keypoints (not the face-box
    lower edge) to locate the neck ROI upper boundary. Returns -1 when
    landmarks are unavailable (caller falls back to geometric face-box edge).
    """
    if not isinstance(landmarks, dict):
        return -1
    try:
        ys = []
        for k in ("mouth_left", "mouth_right"):
            pt = landmarks.get(k)
            if pt and len(pt) >= 2:
                ys.append(float(pt[1]))
        if not ys:
            return -1
        mouth_y = max(ys)
        fh = float(face_box[3]) if face_box and len(face_box) > 3 else 0.0
        margin = max(4.0, fh * 0.12)
        return int(mouth_y + margin)
    except Exception:
        return -1


def _hand_skin_roi_photo(img_arr: np.ndarray, face_box: tuple) -> np.ndarray:
    """在脸部下方采样手部肤色像素。

    **R12-14：输入/输出统一为 0–1**（与 stage5 的调用方一致）。

    原实现把 docstring 写成 "RGB float32 in [0,255]"，阈值也用 0–255 的
    `r>95 / g>40 / b>20`；但唯一的调用方传进来的是**归一化到 0–1** 的图
    （`_arr / _scale_max`）。0–1 的像素永远不可能 > 95，于是手部参考**恒为空**，
    这条采样从来没生效过 —— 修正浮点框之后这个缺陷依然存在。

    Args:
        img_arr: RGB float32，范围 **0–1**。
        face_box: `(x, y, w, h)`，允许 float（stage4 原样输出 facexlib 坐标）。

    Returns:
        N×3 float32（同样 0–1），可能为空。
    """
    empty = np.empty((0, 3), dtype=np.float32)
    try:
        if img_arr is None or img_arr.size == 0:
            return empty
        h, w = img_arr.shape[:2]
        pb = _box_to_pixel_bounds(face_box, w, h)
        if pb is None:
            return empty
        bx1, by1, bx2, by2 = pb
        bw = bx2 - bx1
        bh = by2 - by1

        # 脸部下方：略宽于脸、向下延伸两倍脸高
        x1 = max(0, bx1 - bw // 2)
        x2 = min(w, bx2 + bw // 2)
        y1 = min(h, by2 + bh // 4)
        y2 = min(h, y1 + bh * 2)
        if y2 <= y1 or x2 <= x1:
            return empty
        region = img_arr[y1:y2, x1:x2]
        if region.size == 0:
            return empty

        r = region[:, :, 0]
        g = region[:, :, 1]
        b = region[:, :, 2]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        # 阈值原为 0–255 域（95 / 40 / 20 / 15 / 15），此处按 0–1 域等比缩放。
        # 这是项目启发式，不是普适肤色判据。
        _s = 1.0 / 255.0
        skin_mask = ((r > 95 * _s) & (g > 40 * _s) & (b > 20 * _s)
                     & ((mx - mn) > 15 * _s)
                     & (np.abs(r - g) > 15 * _s))
        return region[skin_mask]
    except Exception:
        return empty

def _carry_icc_to_output(output_path: str, icc: Optional[bytes],
                         logger=None) -> bool:
    """R3-07 / B06: attach the source ICC to the stage5 output **and verify it**.

    stage5 不做色彩空间转换，因此"未变化的 profile"应当原样带下去。
    实测（A03）：输入 RGB16 PNG 带 588 字节 sRGB ICC，真实 stage5 子进程
    rc=0、像素一致，但输出 TIFF 的 34675 标签是 0 字节。

    R3-07：第三轮只把失败记成 warning，调用方也不看返回值 ——
    注入真实嵌入失败后 stage5 仍然 rc=0、输出没有 ICC（实测 B06）。
    现在：写入后**回读验证**字节一致；不成功就是失败，由调用方决定
    （默认 `require_icc_preserved=True` → 不写 SUCCESS）。

    Returns:
        True 表示已嵌入且回读一致。
    """
    if not icc:
        return False
    if _media_io is None:
        return False
    ok = False
    try:
        ok = bool(_media_io.embed_icc_in_tiff(output_path, icc))
    except Exception as e:
        ok = False
        if logger:
            logger.error("STAGE5_ICC_EMBED_FAIL",
                         f"Could not embed the source ICC: {e}",
                         output=output_path)
    if ok:
        # 写成功不等于真的写进去了 —— 回读核对字节
        try:
            _arr, meta = _media_io.read_image_array(output_path)
            written = (meta or {}).get("icc_profile")
            if not written or bytes(written) != bytes(icc):
                ok = False
                if logger:
                    logger.error(
                        "STAGE5_ICC_VERIFY_FAIL",
                        "ICC was written but the file does not read back with "
                        "the same profile bytes",
                        expected_bytes=len(icc),
                        read_back_bytes=(len(written) if written else 0),
                        output=output_path)
        except Exception as e:
            ok = False
            if logger:
                logger.error("STAGE5_ICC_VERIFY_FAIL",
                             f"Could not read back the output to verify the "
                             f"ICC: {e}", output=output_path)
    if ok and logger:
        logger.info("STAGE5_ICC_CARRIED",
                    "Source ICC carried through stage5 and verified by "
                    "reading the output back",
                    icc_bytes=len(icc), output=output_path)
    return bool(ok)


def _promote_temp_to_output(temp_path: str, output_path: str) -> None:
    """把中间临时文件落到 output_path，**按扩展名决定最终格式**。

    FIX-2026-10-21（全链路实测，R23 病族）：中间产物用 `.pre_grain.tif`
    / `.pre_color_match.tif`（TIFF 能无损带 16-bit），而 8-bit 档位的
    输出名是 `..._step5_postprocess.png`。旧代码直接
    `_promote_temp_to_output(grain_temp, output_path)` —— **改名不改内容**，于是
    磁盘上是 TIFF 字节配 .png 名（实测文件头 `II*\\0`），avifenc 报
    `Unrecognized file format` → stage6 rc=84。

    扩展名相同才允许直接 move；不同就必须真正转格式。
    """
    import shutil as _sh
    src_ext = Path(temp_path).suffix.lower()
    dst_ext = Path(output_path).suffix.lower()
    if src_ext == dst_ext:
        _sh.move(temp_path, output_path)
        return
    if _media_io is not None:
        try:
            arr, _meta = _media_io.read_image_array(temp_path)
            if dst_ext in (".tif", ".tiff"):
                _media_io.write_tiff(output_path, arr)
            else:
                out = arr
                if getattr(arr, "dtype", None) == np.uint16:
                    # 目标是 8-bit 容器 → 真正降位，而不是改个名字
                    out = (np.asarray(arr, dtype=np.float64) / 257.0
                           ).clip(0, 255).astype(np.uint8)
                _media_io.write_png(output_path, out)
            os.remove(temp_path)
            return
        except Exception:
            pass
    # 兜底：PIL 按目标扩展名重新编码
    from PIL import Image as _PILImage
    with _PILImage.open(temp_path) as _im:
        _im.convert("RGB").save(output_path)
    try:
        os.remove(temp_path)
    except OSError:
        pass


def _write_color_matched(output_path: str, arr: "np.ndarray") -> None:
    """R2-02: 按数组真实位深写出色彩匹配结果（不用 Pillow 的 8-bit save）。

    FIX-2026-10-21（全链路实测，R23 病族第四次现身）：原实现按 **dtype**
    决定写 TIFF 还是 PNG，**完全不看 output_path 的扩展名**：

    ```python
    if arr.dtype == np.uint16:
        _media_io.write_tiff(output_path, arr)   # 哪怕路径叫 .png
    ```

    而 `photo_bit_depth=8` 档位的输出名正是 `..._step5_postprocess.png`。
    于是写出了 **TIFF 字节配 .png 名**（实测文件头 `II*\\0`），
    avifenc 直接报 `Unrecognized file format` → stage6 rc=84。

    现在按**扩展名**分派（与 R23 对 `run_dat_file` 的修法一致）：
    * `.tif/.tiff` → `write_tiff`（避开 PIL 对 .tif 默认 LZW）
    * 其它（PNG）→ `write_png`；8-bit 档位要求的是真 8-bit 文件，
      所以 uint16 输入先降到 uint8 再写（否则会写出 16-bit PNG 冒充 8-bit）。
    """
    ext = Path(output_path).suffix.lower()
    if _media_io is not None:
        try:
            if ext in (".tif", ".tiff"):
                _media_io.write_tiff(output_path, arr)
            else:
                out_arr = arr
                if getattr(arr, "dtype", None) == np.uint16:
                    # 8-bit 档位：真正降到 uint8（不是把 16-bit PNG 改个名）
                    out_arr = (np.asarray(arr, dtype=np.float64) / 257.0
                               ).clip(0, 255).astype(np.uint8)
                _media_io.write_png(output_path, out_arr)
            return
        except Exception:
            pass
    from PIL import Image as _PILImage
    _PILImage.fromarray(arr).save(output_path)


def _match_portrait_color(image_path: str, face_boxes: list,
                          output_path: str, logger=None,
                          landmarks: list = None) -> Dict[str, Any]:
    """Automated half-body portrait color matching per v7.32 §3.10.7.

    FIX-P12-6: Migrated from stage4_ai_face.py to stage5_postprocess.py.
    Color matching must execute after all AI processing (face restore + upscale)
    and before grain synthesis, per spec order.

    FIX-H-12-06: When stage4 persisted face landmarks (RetinaFace/MTCNN), use
    the chin point to locate the neck ROI upper boundary (instead of the
    face-box lower edge) and fold hand skin-tone pixels into the reference so
    the restored face matches body skin, not just neck. Falls back to geometric
    ROI when landmarks are None (YOLOv8-face).

    RV-06: returns a **structured result** ``{"applied", "skipped", "reason"}``.
    ORIGINAL returned ``True`` from the failure fallback that simply copied the
    input through, so the caller logged "color matching applied" for pixels
    that were never matched. A skipped/failed operation may never look like a
    successful one.
    """
    try:
        import numpy as np
        from PIL import Image, ImageFilter
        import cv2

        # R2-02 / A15：不能用 Pillow 读 —— 它对 16-bit TIFF/PNG 返回 uint8，
        # 于是整幅图（连人脸框外的未处理区域）都被量化成 8-bit 网格再写回。
        # 实测：左上角未处理像素 [39380,45372,27148] → [39321,45489,27242]。
        # 必须用 media_io.read_image_array 拿到真实位深的数组，全程按原生范围运算。
        _arr = None
        if _media_io is not None:
            try:
                _arr, _meta_icc = _media_io.read_image_array(image_path)
            except Exception:
                _arr = None
        if _arr is None:
            _pil = Image.open(image_path).convert("RGB")
            _arr = np.array(_pil)
        if _arr.ndim == 2:
            _arr = np.stack([_arr] * 3, axis=2)
        source_dtype = getattr(_arr, "dtype", None)
        _scale_max = 65535.0 if source_dtype == np.uint16 else 255.0
        # R3-02 / B05：cv2 对 **float32** 的非线性色彩空间转换（RGB↔Lab 等）
        # 规定输入范围为 **0–1**（见 OpenCV cvtColor 文档，2026-10-13 核对）。
        # 第三轮只把整数转成 float32 就送进去（16-bit 时值是 0–65535），
        # 逆转换之后又没乘回 —— 实测脸内 [30000,20000,10000] 变成 [1,1,0]，
        # 8-bit 的 [150,100,50] 同样变成 [1,1,0]，而函数仍返回 applied=True。
        # 这里统一：进入 LAB 前除以 _scale_max 归一，回到 RGB 后再乘回来。
        img_arr = _arr[:, :, :3].astype(np.float32) / _scale_max
        h_img, w_img = img_arr.shape[:2]

        # Build a combined mask for all face regions
        face_mask = np.zeros((h_img, w_img), dtype=np.float32)
        corrected = img_arr.copy()

        for box_idx, _box in enumerate(face_boxes):
            # R12-13：框可能是 float（stage4 原样保存 facexlib 输出），
            # 必须规范成整数半开区间才能做切片。非法框 → 跳过并记录，
            # 不静默吞掉。
            _pb = _box_to_pixel_bounds(_box, w_img, h_img)
            if _pb is None:
                if logger:
                    logger.warning(
                        "STAGE5_FACE_BOX_INVALID",
                        "Skipping a face box that is not a usable pixel "
                        "region (non-finite, empty, or fully outside)",
                        box=_box, image_size=[w_img, h_img])
                continue
            x1, y1, x2, y2 = _pb
            fx, fy, fw, fh = (float(v) for v in _box[:4])

            # Estimate neck region: directly below face,
            # same width, 0.4x face height.
            # FIX-H-12-06: when landmarks available, start neck ROI at the chin
            # (mouth_y + margin) instead of the face-box lower edge - the box
            # edge often cuts into the jaw, biasing the reference.
            _lm = (landmarks[box_idx]
                   if landmarks and box_idx < len(landmarks) else None)
            _chin = _chin_y_from_landmarks_photo(_lm, (fx, fy, fw, fh))
            # _chin 来自关键点推算，可能是 float —— 切片端点必须整数。
            ny1 = int(max(y2, _chin)) if _chin > 0 else int(y2)
            ny2 = min(int(h_img), int(ny1) + int(fh * 0.4))
            nx1 = x1
            nx2 = x2
            if ny2 <= ny1:
                continue

            face_roi = img_arr[y1:y2, x1:x2]
            neck_roi = img_arr[ny1:ny2, nx1:nx2]
            if neck_roi.size == 0:
                continue

            # FIX-H-12-06: sample hand skin-tone below the face and fold into
            # the neck reference so the match target reflects body skin.
            hand_skin = _hand_skin_roi_photo(img_arr, (fx, fy, fw, fh))

            # Convert to LAB using OpenCV
            # FIX-M12-11: Do LAB conversion on float32 (not uint8) to avoid
            # truncation per round_12 M-12-11. cv2 float32 LAB uses ranges
            # L∈[0,100], a/b∈[-128,127] (uint8 LAB offsets a/b to [0,255]).
            # ORIGINAL: astype(np.uint8) truncated before LAB + clamped all
            # channels to [0,255] (wrong for float a/b which can be negative).
            face_bgr = cv2.cvtColor(
                face_roi.astype(np.float32), cv2.COLOR_RGB2BGR)
            neck_bgr = cv2.cvtColor(
                neck_roi.astype(np.float32), cv2.COLOR_RGB2BGR)
            face_lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB)
            neck_lab = cv2.cvtColor(neck_bgr, cv2.COLOR_BGR2LAB)

            # FIX-H-12-06: combine neck + hand skin pixels for reference stats
            # so the restored face matches body skin, not just neck. Convert
            # hand skin (RGB float32) -> BGR -> LAB and concatenate.
            if hand_skin.size > 0:
                try:
                    hand_bgr = cv2.cvtColor(
                        hand_skin.reshape(-1, 1, 3), cv2.COLOR_RGB2BGR)
                    hand_lab = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2LAB)
                    ref_pix = np.concatenate(
                        [neck_lab.reshape(-1, 3), hand_lab.reshape(-1, 3)],
                        axis=0)
                except Exception:
                    ref_pix = neck_lab.reshape(-1, 3)
            else:
                ref_pix = neck_lab.reshape(-1, 3)

            # Mean/std matching per channel (range-agnostic)
            matched_lab = face_lab.copy()
            for c in range(3):
                f_mean = face_lab[:, :, c].mean()
                f_std = face_lab[:, :, c].std() + 1e-6
                n_mean = float(ref_pix[:, c].mean())
                n_std = float(ref_pix[:, c].std()) + 1e-6
                matched_lab[:, :, c] = (
                    (matched_lab[:, :, c] - f_mean) * (n_std / f_std)
                    + n_mean)

            # Clamp LAB valid ranges (float32 convention)
            matched_lab[:, :, 0] = np.clip(matched_lab[:, :, 0], 0, 100)
            matched_lab[:, :, 1] = np.clip(matched_lab[:, :, 1], -128, 127)
            matched_lab[:, :, 2] = np.clip(matched_lab[:, :, 2], -128, 127)

            # Convert back to RGB (float32, no uint8 cast)
            matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)
            matched_rgb = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB)

            corrected[y1:y2, x1:x2] = matched_rgb
            face_mask[y1:y2, x1:x2] = 1.0

        # If no valid face was processed, skip
        if face_mask.max() < 0.5:
            # FIX-P12-10: Use copyfile (not copy2) per v7.32 §3.10.10
            shutil.copyfile(image_path, output_path)
            # RV-06: 这是"没有可处理的区域"，不是"已应用匹配" —— 不能报成功
            return {"applied": False, "skipped": True,
                    "reason": "no usable face region; nothing was matched"}

        # Feather the mask edge
        mask_img = Image.fromarray((face_mask * 255).astype(np.uint8))
        mask_blurred = mask_img.filter(ImageFilter.GaussianBlur(radius=8))
        mask_smooth = np.array(mask_blurred, dtype=np.float32) / 255.0
        mask_smooth = np.clip(mask_smooth, 0.0, 1.0)
        mask_smooth = np.stack([mask_smooth] * 3, axis=2)

        blended = mask_smooth * corrected + (1.0 - mask_smooth) * img_arr
        # R2-02 / A15：按**输入真实位深**回写，不能一律压成 uint8。
        # R3-02：corrected / img_arr 都在 0–1 域；先按 0–1 裁剪，再乘回真实
        # 位深并就近取整（不能靠 astype 截断，否则暗部整体偏暗 0.5 LSB）。
        blended = np.clip(blended, 0.0, 1.0) * _scale_max
        out_dtype = np.uint16 if source_dtype == np.uint16 else np.uint8
        out_arr = np.rint(blended).astype(out_dtype)
        _write_color_matched(output_path, out_arr)
        return {"applied": True, "skipped": False, "reason": ""}
    except Exception as e:
        if logger:
            logger.warning("STAGE5_COLOR_MATCH_FAIL",
                          f"Portrait color matching failed: {e}")
        # FIX-H12-04: Scheme C half-automatic fallback per v7.32 §3.10.7.
        # Output the face-restored image as _face_restored.tif so the user
        # can manually Match Color in GIMP/PS and drop it back. The main
        # flow continues using the original (uncolor-matched) image.
        # ORIGINAL: silently copied the original to output_path with no
        # handoff artifact, giving the user no way to finish color matching.
        try:
            fallback_tif = output_path.rsplit(".", 1)[0] + "_face_restored.tif"
            shutil.copyfile(image_path, fallback_tif)
            if logger:
                logger.warning("STAGE5_FACE_RESTORED_TIF",
                               f"半自动兜底：已输出 {fallback_tif}，请在 "
                               f"GIMP/PS 手工 Match Color 后放回继续流程 "
                               f"(v7.32 §3.10.7 方案C)")
            # FIX-P12-10: Use copyfile (not copy2) per v7.32 §3.10.10
            shutil.copyfile(image_path, output_path)
            # RV-06: 兜底只是"把原图交回主流程"，**不是**完成了色彩匹配。
            # ORIGINAL 返回 True，调用方据此写入"已应用匹配"的日志，
            # 于是"请求了但没做"被记成了"做了"。
            return {"applied": False, "skipped": True,
                    "reason": f"portrait color matching failed: {e}"}
        except Exception:
            return {"applied": False, "skipped": True,
                    "reason": "portrait color matching failed and the "
                              "fallback could not be written"}


def _add_grain_python(input_path: str, output_path: str,
                       strength: float, size: float,
                       roughness: float, bit_depth: int = 16) -> bool:
    """Pure Python film grain synthesis via Laplacian pyramid.

    FIX-R13-1: Replaces non-existent G'MIC -film_grain command.
    Algorithm matches DaVinci Resolve Film Grain / Adobe Camera Raw Grain:
      1. Laplacian pyramid decomposition (multi-scale frequency bands)
      2. Inject per-level Gaussian noise with sigma proportional to strength
      3. Roughness controls per-level weight (fine vs coarse grain ratio)
      4. Reconstruct pyramid = grain-synthesized image

    Grain is applied primarily to luminance (Y channel) per film grain
    physics — chrominance grain is kept at 15% of luminance intensity,
    matching real film behavior where UV grain is 10-20% of Y grain.
    In RGB space this is approximated by applying a shared base noise
    (luminance) plus small per-channel variation (chrominance).

    Args:
        input_path: Input image path (PNG or TIFF)
        output_path: Output image path
        strength: Overall grain intensity [0, 1], default 0.3
        size: Grain spatial scale [0.5, 2.0], default 0.5
        roughness: Fine/coarse grain ratio [0, 1], default 0
            0 = pure fine grain (natural film look)
            0.5 = balanced
            1 = mostly coarse grain (DV/tape noise look)
        bit_depth: Output bit depth (8 or 16)

    Returns:
        True if successful, False otherwise
    """
    try:
        from PIL import Image
        from scipy.ndimage import gaussian_filter, zoom

        if strength <= 0:
            # No grain requested — just copy
            shutil.copyfile(input_path, output_path)
            return True

        # Load image — prefer tifffile for 16-bit TIFF to avoid
        # PIL I;16 3-channel data loss (PIL I;16 only supports 2D arrays)
        arr = None
        input_depth = 8
        try:
            import tifffile
            loaded = tifffile.imread(input_path)
            if loaded.dtype == np.uint16:
                arr = loaded.astype(np.float32) / 65535.0
                input_depth = 16
            elif loaded.dtype == np.uint8:
                arr = loaded.astype(np.float32) / 255.0
                input_depth = 8
            else:
                # Other dtypes — normalize to [0,1]
                arr = loaded.astype(np.float32)
                arr = arr / (arr.max() + 1e-8)
                input_depth = 16 if loaded.dtype.itemsize >= 2 else 8
            # Ensure 3-channel
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=2)
        except (ImportError, Exception):
            # Fallback to PIL
            img = Image.open(input_path)
            if img.mode == "I;16":
                arr = np.array(img, dtype=np.float32) / 65535.0
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=2)
                input_depth = 16
            else:
                arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
                input_depth = 8

        h, w = arr.shape[:2]
        if h < 8 or w < 8:
            # Image too small for pyramid decomposition
            shutil.copyfile(input_path, output_path)
            return True

        # --- Laplacian Pyramid Decomposition ---
        # Number of levels based on grain_size and image dimensions
        max_levels = int(np.log2(min(h, w)) - 2)
        max_levels = max(3, min(max_levels, 8))
        n_levels = max(3, int(max_levels * min(size, 2.0)))

        # Build Gaussian pyramid
        gauss_pyramid = [arr.copy()]
        for _ in range(n_levels):
            prev = gauss_pyramid[-1]
            # Blur and downsample by 2x
            blurred = np.zeros_like(prev)
            for c in range(3):
                blurred[:, :, c] = gaussian_filter(
                    prev[:, :, c], sigma=2.0)
            # Downsample: take every other pixel
            gauss_pyramid.append(blurred[::2, ::2, :])

        # Build Laplacian pyramid (difference between adjacent levels)
        lap_pyramid = []
        for i in range(n_levels):
            small = gauss_pyramid[i + 1]
            # Upsample small to current level size
            up_h, up_w = gauss_pyramid[i].shape[0], gauss_pyramid[i].shape[1]
            up = zoom(small, (up_h / small.shape[0],
                              up_w / small.shape[1], 1.0),
                      order=1)  # bilinear
            # Ensure exact size match after zoom — pad if too small,
            # truncate if too large (handles odd-dimension rounding)
            if up.shape[0] < up_h or up.shape[1] < up_w:
                # Pad with edge values to reach target size
                padded = np.zeros((up_h, up_w, 3), dtype=np.float32)
                padded[:up.shape[0], :up.shape[1], :] = up
                # Replicate edge for padded rows/cols
                if up.shape[0] < up_h:
                    padded[up.shape[0]:, :up.shape[1], :] = up[-1:, :, :]
                if up.shape[1] < up_w:
                    padded[:up.shape[0], up.shape[1]:, :] = up[:, -1:, :]
                if up.shape[0] < up_h and up.shape[1] < up_w:
                    padded[up.shape[0]:, up.shape[1]:, :] = up[-1, -1, :]
                up = padded
            elif up.shape[0] > up_h or up.shape[1] > up_w:
                up = up[:up_h, :up_w, :]
            lap = gauss_pyramid[i] - up
            lap_pyramid.append(lap)

        # --- Inject Grain at Each Level ---
        # Level 0 = finest (high freq), Level n-1 = coarsest (low freq)
        for i in range(n_levels):
            level_h, level_w = lap_pyramid[i].shape[:2]

            # Per-level sigma: finer levels get relatively more noise
            # roughness shifts weight from fine to coarse levels
            level_ratio = i / max(n_levels - 1, 1)

            # Fine grain weight (decreases with level)
            fine_weight = (1.0 - roughness) * (1.0 - level_ratio * 0.6)
            # Coarse grain weight (increases with level)
            coarse_weight = roughness * (0.3 + level_ratio * 0.7)
            # Combined per-level sigma
            level_sigma = strength * 0.05 * (fine_weight + coarse_weight)

            if level_sigma <= 0:
                continue

            # Generate Gaussian noise with luminance-weighted application
            # Film grain is primarily in luminance (Y), with chroma at 15%
            # Since we operate in RGB space, we approximate by applying
            # a shared base noise (luminance component) plus small
            # per-channel variation (chrominance component).
            # This produces correlated grain similar to real film where
            # R/G/B grain is ~85% shared (luminance) and ~15% independent.
            noise_luma = np.random.normal(
                0, level_sigma, (level_h, level_w)).astype(np.float32)
            # Per-channel chrominance variation (15% of luma)
            noise_ch_r = np.random.normal(
                0, level_sigma * 0.15, (level_h, level_w)).astype(np.float32)
            noise_ch_g = np.random.normal(
                0, level_sigma * 0.15, (level_h, level_w)).astype(np.float32)
            noise_ch_b = np.random.normal(
                0, level_sigma * 0.15, (level_h, level_w)).astype(np.float32)

            # Combine: base luminance noise + per-channel chroma variation
            noise = np.stack([noise_luma + noise_ch_r,
                              noise_luma + noise_ch_g,
                              noise_luma + noise_ch_b], axis=2)
            lap_pyramid[i] = lap_pyramid[i] + noise

        # --- Reconstruct Pyramid ---
        result = gauss_pyramid[-1].copy()  # Start from coarsest
        for i in range(n_levels - 1, -1, -1):
            target_h, target_w = gauss_pyramid[i].shape[0], gauss_pyramid[i].shape[1]
            # Upsample result to current level size
            up = zoom(result, (target_h / result.shape[0],
                                target_w / result.shape[1], 1.0),
                      order=1)
            # Handle size mismatch (same padding/truncation logic as decomposition)
            if up.shape[0] < target_h or up.shape[1] < target_w:
                padded = np.zeros((target_h, target_w, 3), dtype=np.float32)
                padded[:up.shape[0], :up.shape[1], :] = up
                if up.shape[0] < target_h:
                    padded[up.shape[0]:, :up.shape[1], :] = up[-1:, :, :]
                if up.shape[1] < target_w:
                    padded[:up.shape[0], up.shape[1]:, :] = up[:, -1:, :]
                if up.shape[0] < target_h and up.shape[1] < target_w:
                    padded[up.shape[0]:, up.shape[1]:, :] = up[-1, -1, :]
                up = padded
            elif up.shape[0] > target_h or up.shape[1] > target_w:
                up = up[:target_h, :target_w, :]
            result = up + lap_pyramid[i]

        # Clip and save
        result = np.clip(result, 0.0, 1.0)

        out_depth = bit_depth if bit_depth in (8, 16) else input_depth
        # FIX-2026-10-21（全链路实测，R23 病族）：**不能只看 out_depth**，
        # 必须同时看 output_path 的扩展名。`photo_bit_depth=8` 档位的输出名是
        # `..._step5_postprocess.png`，若 out_depth 被判成 16（bit_depth 没传
        # 进来时默认 16），旧代码会用 tifffile 写出 TIFF 字节配 .png 名
        # （实测文件头 `II*\0`），avifenc 报 `Unrecognized file format`
        # → stage6 rc=84。PNG 路径必须是真 PNG。
        _ext = os.path.splitext(output_path)[1].lower()
        if out_depth == 16 and _ext in (".tif", ".tiff"):
            out_arr = (result * 65535.0).astype(np.uint16)
            try:
                import tifffile
                tifffile.imwrite(output_path, out_arr, compression="zlib")
            except ImportError:
                # Fallback: save as 8-bit (significant quality loss)
                import warnings
                warnings.warn(
                    "tifffile not available; saving 16-bit grain output "
                    "as 8-bit. Install tifffile for full 16-bit support.")
                out_8 = (result * 255.0).astype(np.uint8)
                Image.fromarray(out_8).save(output_path)
        else:
            out_arr = (result * 255.0).astype(np.uint8)
            Image.fromarray(out_arr).save(output_path)

        return True
    except Exception:
        return False


def _add_grain(input_path: str, output_path: str,
               strength: float, size: float,
               roughness: float, bit_depth: int = 16) -> bool:
    """Add film grain via pure Python Laplacian pyramid.

    FIX-A1: Removed dead G'MIC -film_grain fallback (G'MIC stdlib has no
    -film_grain command, verified gmic_stdlib.gmic 3.7.6). Pure Python
    Laplacian pyramid is the sole implementation per v7.32 §3.10.4.
    """
    return _add_grain_python(input_path, output_path, strength, size,
                             roughness, bit_depth)


def _extract_flif_chrominance(img: "Image.Image", output_dir: str,
                               job_id: str, stem: str,
                               logger=None) -> bool:
    """Extract Cb/Cr chrominance layers and save as separate files.

    Attempts FLIF encoding first (v6.3 §4.2.4 requirement).
    Falls back to lossless PNG if flif CLI is unavailable.
    """
    try:
        import shutil as _shutil
        # Convert to YCbCr
        ycbcr = img.convert("YCbCr")
        y, cb, cr = ycbcr.split()

        # Try FLIF encoding first
        flif_available = _shutil.which("flif") is not None

        if flif_available:
            cb_path = os.path.join(output_dir, f"{job_id}_{stem}_cb.flif")
            cr_path = os.path.join(output_dir, f"{job_id}_{stem}_cr.flif")
            y_path = os.path.join(output_dir, f"{job_id}_{stem}_y.flif")

            # Save as temporary PNG first, then convert to FLIF
            cb_tmp = cb_path + ".tmp.png"
            cr_tmp = cr_path + ".tmp.png"
            y_tmp = y_path + ".tmp.png"
            cb.save(cb_tmp)
            cr.save(cr_tmp)
            y.save(y_tmp)

            try:
                rc_cb = subprocess.run(
                    ["flif", "-e", cb_tmp, cb_path],
                    capture_output=True, timeout=60, shell=False)
                rc_cr = subprocess.run(
                    ["flif", "-e", cr_tmp, cr_path],
                    capture_output=True, timeout=60, shell=False)
                rc_y = subprocess.run(
                    ["flif", "-e", y_tmp, y_path],
                    capture_output=True, timeout=60, shell=False)

                # ISSUE-055 fix: verify FLIF returncodes and output existence
                if (rc_cb.returncode == 0 and rc_cr.returncode == 0
                        and rc_y.returncode == 0
                        and os.path.exists(cb_path)
                        and os.path.exists(cr_path)
                        and os.path.exists(y_path)):
                    # Clean up temp PNGs if FLIF succeeded
                    for p in (cb_tmp, cr_tmp, y_tmp):
                        if os.path.exists(p):
                            os.remove(p)
                    if logger:
                        logger.info("STAGE5_FLIF",
                                    f"FLIF chrominance layers saved: "
                                    f"{cb_path}, {cr_path}")
                    return True
                else:
                    # FLIF encoding partially failed, fall through to PNG
                    for p in (cb_tmp, cr_tmp, y_tmp):
                        if os.path.exists(p):
                            os.remove(p)
                    # Clean up partial FLIF outputs
                    for p in (cb_path, cr_path, y_path):
                        if os.path.exists(p):
                            os.remove(p)
            except Exception:
                # FLIF encoding failed, fall through to PNG
                for p in (cb_tmp, cr_tmp, y_tmp):
                    if os.path.exists(p):
                        os.remove(p)

        # Fallback: save as lossless PNG
        cb_path = os.path.join(output_dir, f"{job_id}_{stem}_cb.png")
        cr_path = os.path.join(output_dir, f"{job_id}_{stem}_cr.png")
        cb.save(cb_path, compress_level=0)
        cr.save(cr_path, compress_level=0)

        if logger:
            logger.warning("STAGE5_FLIF_FALLBACK",
                          "FLIF not available, saved Cb/Cr as lossless PNG fallback")
        return True
    except Exception as e:
        if logger:
            logger.warning("STAGE5_FLIF_FAIL",
                          f"Cb/Cr extraction failed: {e}")
        return False


# --- FIX-V06/V07: stage result contract + cache identity -------------------
# 阶段结果状态（写入产物同名的 .status.json 旁车，供调度器/恢复逻辑读取）
STAGE_STATUS_SUCCESS = "SUCCESS"
STAGE_STATUS_DEGRADED = "DEGRADED"   # 明确策略接受的降级完成
STAGE_STATUS_FAILED = "FAILED"       # 必需步骤失败：绝不允许写成 DONE/成功

# FIX-V07: 影响本阶段像素输出的参数集合（不在此集合内的配置变化不应使缓存失效）
STAGE5_CACHE_PARAM_KEYS = (
    "photo_bit_depth", "archival_mode", "local_mask_denoise",
    "lanczos_target", "portrait_color_match", "grain_synthesis",
    "grain_strength", "grain_size", "grain_roughness",
    "scratch_remove", "scratch_remove_strength", "deband_threshold",
    "channel_correction_formula", "channel_correction_normalize",
    "channel_correction_round", "channel_correction_clamp",
    "channel_degradation_mean_threshold", "channel_degradation_std_threshold",
    "channel_degradation_mean_threshold_severe",
    "channel_degradation_std_threshold_severe",
    "channel_correction_linear_space", "channel_correction_soft_compress",
    "channel_correction_float_precision",
)

# FIX-V07: 实现/模型版本。任何会改变本阶段像素输出的实现调整都应递增，
# 否则旧产物会被当成"同一实现"而错误复用。
# RV-11/RV-06: 缓存身份与操作语义都变了，升版本以免误用旧 cache sidecar。
# R3-07 / B13：实现/缓存契约版本。**旧版本产出的坏缓存必须失效**。
# 实测：用第三轮修复**之前**的 stage5 生成一份丢 ICC 的缓存，
# 再让修复后的 stage5 处理完全相同的输入/参数 —— 旧缓存照样命中
# （返回 3、输出字节不变、ICC 仍不存在），修复逻辑根本没机会执行。
# 缓存身份里包含 impl，因此升版本即可让旧坏缓存全部重算。
# 语义变化点：ICC 贯穿 + 写后回读校验、人像色彩匹配量纲修正（R3-02）。
STAGE5_IMPL_VERSION = "photo-stage5-postprocess/4.0-r3-02-r3-07-icc-verified"


def _json_sidecar_path(artifact_path: str, kind: str) -> str:
    """Return sidecar path for an artifact (e.g. `.cache.json`)."""
    return f"{artifact_path}.{kind}.json"


def _write_json_sidecar(path: str, payload: dict) -> bool:
    """Write a JSON sidecar next to an artifact. Never raises."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True, indent=2)
        return True
    except Exception:
        return False


def _read_json_sidecar(path: str) -> Optional[dict]:
    """Read a JSON sidecar. Returns None when absent/unparseable/not an object."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _content_digest(path: str, chunk: int = 1 << 20) -> Optional[str]:
    """Dependency-free content fingerprint (head + tail + size).

    FIX-V07: checkpoint.get_file_xxhash needs the optional `xxhash` package and
    is unavailable in some environments; without a content fingerprint the
    identity degenerates to (size, mtime-int) and two different files written
    in the same second with equal size collide.
    """
    try:
        size = os.path.getsize(path)
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read(chunk))
            if size > chunk:
                f.seek(-chunk, os.SEEK_END)
                h.update(f.read(chunk))
        h.update(str(size).encode("ascii"))
        return h.hexdigest()
    except Exception:
        return None


def _file_identity(path: str) -> dict:
    """Identity of an input artifact: size + ns mtime + content fingerprint.

    FIX-V07: cache reuse must not be decided by "a non-empty file exists";
    it must be tied to the actual upstream bytes.
    """
    ident: Dict[str, Any] = {"size": None, "mtime_ns": None,
                             "xxhash": None, "digest": None}
    try:
        st = os.stat(path)
        ident["size"] = st.st_size
        ident["mtime_ns"] = st.st_mtime_ns
    except Exception:
        pass
    try:
        from checkpoint import get_file_xxhash  # type: ignore
        ident["xxhash"] = get_file_xxhash(path)
    except Exception:
        ident["xxhash"] = None
    ident["digest"] = _content_digest(path)
    return ident


def _canonical_metadata(value: Any) -> Any:
    """RV-11: stable representation of non-image inputs for the cache identity.

    ``None`` / missing is a state of its own (encoded as ``"__absent__"``), so
    "no metadata" → "metadata present" invalidates the cache instead of
    silently reusing an output produced without it.
    """
    if value is None:
        return "__absent__"
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            value = value.tolist()
    except Exception:
        pass
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except Exception:
        return repr(value)


def _read_face_metadata(temp_dir: str, job_id: str,
                        stem: str) -> Tuple[Optional[list], Optional[list]]:
    """Read stage4's face metadata (RV-11: it is an input to this stage).

    Returns ``(face_boxes, face_landmarks)``; ``None`` means "no metadata
    file / unusable", which is a distinct state from an empty list.
    """
    face_meta_path = os.path.join(temp_dir, f"{job_id}_{stem}_face_boxes.json")
    if not os.path.exists(face_meta_path):
        return None, None
    try:
        with open(face_meta_path, "r", encoding="utf-8") as f:
            face_meta = json.load(f)
        boxes = face_meta.get("face_boxes", [])
        landmarks = face_meta.get("face_landmarks", [])  # FIX-H-12-06
        return list(boxes or []), list(landmarks or [])
    except Exception:
        return None, None


def compute_stage5_cache_identity(input_source: str, params: dict,
                                  bit_depth: int,
                                  face_boxes: Optional[list] = None,
                                  face_landmarks: Optional[list] = None) -> dict:
    """FIX-V07 / RV-11: canonical cache identity for stage5.

    Covers (a) upstream artifact identity, (b) params that affect this stage's
    pixels, (c) implementation version, (d) the bit-depth contract and
    (e) **non-image inputs that actually change the output** — the face
    boxes/landmarks read from stage4's metadata.

    RV-11: ORIGINAL hashed image + params + impl only, so changing the face
    metadata (or adding it where none existed) kept the same ``cache_id`` and
    the stale output was reused. Metadata absence is itself a state and is
    encoded explicitly.
    """
    ident = _file_identity(input_source)
    # 只有内容相关的字段参与身份计算：mtime 被任何复制/重写改变，会导致永久失效，
    # 而只看 mtime 又会在"同一秒同尺寸"时误判命中。这里用 size + 内容指纹。
    content_ident = {k: ident.get(k) for k in ("size", "digest", "xxhash")}
    selected = {k: params.get(k) for k in STAGE5_CACHE_PARAM_KEYS
                if k in params}
    payload = {
        "impl": STAGE5_IMPL_VERSION,
        "bit_depth": bit_depth,
        "input": content_ident,
        "params": selected,
        # RV-11: 人脸元数据也影响像素 —— 必须参与身份
        "face_boxes": _canonical_metadata(face_boxes),
        "face_landmarks": _canonical_metadata(face_landmarks),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {"cache_id": digest, "input_mtime_ns": ident.get("mtime_ns"),
            **payload}


def _verify_output_readable(path: str) -> bool:
    """Lightweight product validation before a cache hit may be accepted.

    Existence + non-empty is NOT sufficient (a V06 failure copy or a truncated
    write can satisfy both). Deliberately dependency-light: tifffile when
    available, else Pillow.
    """
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return False
    except Exception:
        return False
    try:
        import tifffile  # type: ignore
        arr = tifffile.imread(path)
        return bool(getattr(arr, "size", 0)) and arr.ndim >= 2
    except Exception:
        pass
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False
    return False


def _discard_partial_output(output_path: str,
                            sidecars: Optional[list] = None) -> None:
    """Remove a partial/failed artifact so it can never be reused as a cache."""
    for p in [output_path] + (sidecars or []):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


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

    # NEW-V-06: 注册取消守卫。KeyboardInterrupt 继承 BaseException，
    # 本文件所有 `except Exception` 都抓不到它 —— 不装守卫时 Ctrl+C 会把
    # 半成品留在最终产物路径上且不留任何 FAILED 记录。
    if _cancel_guard is not None:
        _cancel_guard.install_cancel_guard(stage=5, logger=logger)

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
            logger.info("STAGE5_WSL_REDIRECT",
                       f"WSL temp_dir redirect: {temp_dir} -> {redirected_temp}")
            temp_dir = redirected_temp
            params["temp_dir"] = temp_dir
    except Exception:
        pass
    job_id = args.job_id
    input_name = Path(args.input).name

    # FIX-P10-1: Input is now stage4_ai_face output (or stage3_upscale if no face)
    # Sort by mtime for deterministic selection; prefer face-restored output
    input_source = None
    for pat in ["step4_face*", "step3_upscale*", "step2_ai_pre*",
                "step1_preprocess*"]:
        pattern = f"{job_id}_{safe_input_stem(input_name)}_{pat}"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        # NEW-V-04: 过滤掉 .status.json / .cache.json 之类的侧车文件
        candidates = [
            p for p in candidates
            if Path(p).suffix.lower() in set(
                getattr(_media_io, "IMAGE_EXTS", ()) or
                {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
            )
        ]
        if candidates:
            candidates.sort(
                key=lambda p: os.path.getmtime(p), reverse=True)
            input_source = candidates[0]
            break

    # FIX-NEW-V-13 / R15: 与 stage4 同型——stage3 的真实产物在
    # <temp>/<job_id>_upscaled/<stem>_x<N>.tif，三个 stepN_* glob 落空时
    # 必须回落到它，否则 stage5 记 STAGE5_SKIP 静默跳过（rc=3）。
    if not input_source:
        _up = os.path.join(temp_dir, f"{job_id}_upscaled")
        if os.path.isdir(_up):
            _c = glob.glob(os.path.join(
                _up, f"{safe_input_stem(input_name)}_x*"))
            _c = [p for p in _c if Path(p).suffix.lower()
                  in {".tif", ".tiff", ".png"}]
            if _c:
                _c.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                input_source = _c[0]

    if not input_source:
        logger.info("STAGE5_SKIP", "No input from prior stages")
        return 3

    logger.info("STAGE5_START", "Photo post-processing", input=input_source)

    # Output path - use TIFF for 16-bit output
    # FIX-P10-1: step number is now 5
    bit_depth = params.get("photo_bit_depth", 16)
    if bit_depth == 16:
        output_path = os.path.join(temp_dir,
            f"{job_id}_{safe_input_stem(input_name)}_step5_postprocess.tif")
    else:
        output_path = os.path.join(temp_dir,
            f"{job_id}_{safe_input_stem(input_name)}_step5_postprocess.png")

    # FIX-V07: 缓存命中必须比对身份，不再只判断"文件存在且非空"。
    # ORIGINAL: `if exists(output_path) and size>0: return 3` —— 改了颗粒/位深等
    # 参数仍会复用旧产物，且 V06 的失败副本也会命中。
    # RV-11: 人脸元数据也影响像素，所以必须先读出来再算身份。
    face_boxes, face_landmarks = _read_face_metadata(
        temp_dir, job_id, safe_input_stem(input_name))
    cache_identity = compute_stage5_cache_identity(
        input_source, params, bit_depth,
        face_boxes=face_boxes, face_landmarks=face_landmarks)
    cache_meta_path = _json_sidecar_path(output_path, "cache")
    # NEW-V-06: 必须在写入任何字节**之前**登记产物，否则首次运行被中断时
    # 守卫无东西可清理，半成品会留在最终产物路径上。
    if _cancel_guard is not None:
        _cancel_guard.watch_outputs(output_path,
                                    extra_paths=[cache_meta_path],
                                    logger=logger, stage=5)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        cached_meta = _read_json_sidecar(cache_meta_path)
        if cached_meta is None:
            logger.info("STAGE5_CACHE_UNTRUSTED",
                        "Existing output has no cache identity sidecar "
                        "(legacy or failed run); recomputing")
        elif cached_meta.get("cache_id") != cache_identity["cache_id"]:
            changed = sorted(
                k for k in STAGE5_CACHE_PARAM_KEYS
                if cached_meta.get("params", {}).get(k) !=
                cache_identity["params"].get(k))
            logger.info("STAGE5_CACHE_INVALIDATED",
                        "Cached output does not match current identity; "
                        "recomputing",
                        changed_params=changed,
                        cached_impl=cached_meta.get("impl"),
                        current_impl=cache_identity["impl"])
        elif _cached_was_degraded(cached_meta):
            # R2-03：上一次是"降级结果"（有操作没做成）却写了 SUCCESS 缓存。
            # 依赖恢复后（比如装上了 resampler）必须允许补做，不能一直命中旧缓存。
            logger.info("STAGE5_CACHE_DEGRADED_RECOMPUTE",
                        "Cached result was produced with skipped operations; "
                        "recomputing so the missing work can be retried",
                        cached_skipped=cached_meta.get("skipped_operations"))
        elif not _verify_output_readable(output_path):
            logger.info("STAGE5_CACHE_UNREADABLE",
                        "Cached output failed validation; recomputing")
        else:
            logger.info("STAGE5_CACHE_HIT",
                        "Reusing verified stage5 output",
                        cache_id=cache_identity["cache_id"])
            return 3
        # 复用失败：清掉旧产物与旧缓存元数据，避免半成品混入
        _discard_partial_output(output_path, [cache_meta_path])

    try:
        from PIL import Image

        img = Image.open(input_source)

        # NEW-V-03: Pillow 对 16-bit RGB TIFF 报的是 mode='RGB'（uint8 截断），
        # 直接 convert("RGB") 会让 16-bit 素材在 stage5 静默塌成 8-bit。
        true_arr16 = None
        if _media_io is not None:
            true_arr16, input_icc = read_true_input_array(input_source)
        else:
            input_icc = None
        # Pillow 也能拿到 ICC（PNG iCCP / TIFF 34675），作为兜底与交叉来源
        if input_icc is None:
            try:
                input_icc = (img.info or {}).get("icc_profile")
            except Exception:
                input_icc = None
        if true_arr16 is not None and logger is not None:
                logger.info(
                    "STAGE5_TRUE_16BIT_INPUT",
                    "Input is genuinely 16-bit; Pillow reports mode "
                    f"'{img.mode}'. Using the array path to keep precision.",
                    dtype=str(true_arr16.dtype),
                    shape=list(true_arr16.shape))

        # Preserve 16-bit if input is 16-bit
        if img.mode == "I;16" and bit_depth == 16:
            # Process in 16-bit space
            img_16bit = img
        elif true_arr16 is not None:
            # 真实 16-bit 多通道：PIL 无法承载，走纯数组路径
            img_16bit = None
        else:
            # Convert to RGB for processing
            img_16bit = img.convert("RGB")

        # Lanczos resize to target resolution if explicitly specified
        # RV-06: 请求了缩放就必须真的做（见下方具体分支）。
        skipped_operations: List[str] = []
        lanczos_target = params.get("lanczos_target", "auto")
        # R2-03：非法目标尺寸在参数层就拒绝（旧代码解析出 0 就直接跳过并成功）
        target_err = parse_lanczos_target(lanczos_target)[2]
        if target_err:
            logger.error("STAGE5_LANCZOS_INVALID_TARGET",
                         "Invalid lanczos_target; failing instead of silently "
                         "skipping the requested resize",
                         lanczos_target=lanczos_target, reason=target_err)
            _discard_partial_output(output_path, [cache_meta_path])
            _write_json_sidecar(
                _json_sidecar_path(output_path, "status"), {
                    "status": STAGE_STATUS_FAILED,
                    "stage": 5,
                    "failed_operation": "invalid_lanczos_target",
                    "reason": target_err,
                    "impl": STAGE5_IMPL_VERSION,
                })
            return 86

        if (lanczos_target and lanczos_target != "auto" and
                true_arr16 is not None):
            target_w, target_h = parse_lanczos_target(lanczos_target)[:2]
            if target_w and target_h and (true_arr16.shape[1] != target_w or
                                          true_arr16.shape[0] != target_h):
                resized, method = _resize_array_high_depth(
                    true_arr16, target_w, target_h)
                if resized is None:
                    if not params.get("allow_stage5_degrade", False):
                        logger.error(
                            "STAGE5_LANCZOS_UNSUPPORTED",
                            "A target size was requested but no high-depth "
                            "resampler is available; failing instead of "
                            "reporting a full success "
                            "(set allow_stage5_degrade=true to accept a "
                            "DEGRADED result)",
                            target=lanczos_target,
                            reason=method)
                        _discard_partial_output(output_path, [cache_meta_path])
                        _write_json_sidecar(
                            _json_sidecar_path(output_path, "status"), {
                                "status": STAGE_STATUS_FAILED,
                                "stage": 5,
                                "failed_operation": "lanczos_resize",
                                "reason": f"resize not performed: {method}",
                                "impl": STAGE5_IMPL_VERSION,
                            })
                        return 86
                    skipped_operations.append("lanczos_resize")
                    logger.warning("STAGE5_LANCZOS_DEGRADED",
                                   "Resize not performed on the 16-bit path; "
                                   "DEGRADED result accepted",
                                   reason=method,
                                   skipped_operations=skipped_operations)
                else:
                    true_arr16 = resized
                    logger.info("STAGE5_LANCZOS",
                                f"Resized 16-bit array -> "
                                f"{target_w}x{target_h} via {method}")
        elif lanczos_target and lanczos_target != "auto" and img_16bit is not None:
            target_w, target_h = parse_lanczos_target(lanczos_target)[:2]
            if target_w and target_h:
                try:
                    if img_16bit.width != target_w or img_16bit.height != target_h:
                        img_16bit = img_16bit.resize(
                            (target_w, target_h), Image.LANCZOS)
                        logger.info("STAGE5_LANCZOS",
                                    f"Resized {img_16bit.width}x{img_16bit.height} "
                                    f"-> {target_w}x{target_h}")
                except Exception as e:
                    logger.warning("STAGE5_LANCZOS_WARN",
                                  f"Lanczos resize skipped: {e}")
                    # RV-06: 跳过了就必须登记，不能假装做过了
                    if params.get("lanczos_target", "auto") not in (None, "auto", ""):
                        skipped_operations.append("lanczos_resize")

        # Selective local processing with region separation
        # FIX-R13-2: Track image data as numpy array for 16-bit path
        # to avoid PIL I;16 3-channel data loss (PIL I;16 only supports 2D)
        current_arr = None  # numpy float32 [0,1] when 16-bit path active
        current_img = img_16bit  # PIL Image for 8-bit path
        local_process = params.get("local_mask_denoise", True)
        if local_process and (true_arr16 is not None or
                              (img_16bit is not None and
                               img_16bit.mode in ("RGB", "RGBA", "L", "I;16"))):
            if true_arr16 is not None:
                result = _apply_selective_processing(
                    img_16bit, params, logger, src_array=true_arr16)
            else:
                # 保持 3 参数调用签名，兼容既有的 3 参数实现/测试替身
                result = _apply_selective_processing(img_16bit, params, logger)
            # FIX-P12-6: Handle tuple return (array, bit_depth) for 16-bit
            if isinstance(result, tuple):
                arr_result, result_depth = result
                if result_depth == 16:
                    # Keep as numpy array — do NOT go through PIL for 16-bit
                    # PIL I;16 mode only supports 2D arrays; 3-channel uint16
                    # TIFF round-trips through PIL lose color data.
                    current_arr = arr_result.astype(np.float32) / 65535.0
                    current_img = None  # Signal: use array path
                else:
                    current_img = arr_result  # PIL Image for 8-bit
            else:
                current_img = result  # Legacy PIL Image return
            logger.info("STAGE5_MASK",
                        "Applied selective local processing with region separation")
        elif true_arr16 is not None:
            # NEW-V-03: 未启用局部处理时也必须保住真实 16-bit
            current_arr = true_arr16.astype(np.float32) / 65535.0
            if current_arr.shape[2] == 4:
                current_arr = current_arr[:, :, :3]
            current_img = None

        # FLIF chrominance extraction (for archival pipeline)
        archival_mode = params.get("archival_mode", False)
        if archival_mode:
            flif_img = current_img if current_img is not None else None
            if flif_img is None and current_arr is not None:
                # Convert array to PIL Image for FLIF extraction
                flif_arr = (current_arr * 255.0).clip(0, 255).astype(
                    np.uint8)
                flif_img = Image.fromarray(flif_arr)
            if flif_img is not None:
                flif_ok = _extract_flif_chrominance(
                    flif_img, temp_dir, job_id, safe_input_stem(input_name),
                    logger=logger
                )
            else:
                flif_ok = False
            if flif_ok:
                logger.info("STAGE5_FLIF",
                            "Extracted Cb/Cr chrominance layers")
            else:
                logger.warning("STAGE5_FLIF", "Cb/Cr extraction skipped")

        # FIX-P12-6: Half-body portrait color matching per v7.32 §3.10.7
        # Must run after face restore (stage4) and before grain synthesis
        enable_color_match = params.get("portrait_color_match", True)
        # RV-11: 人脸元数据已在计算缓存身份前读入（见 _read_face_metadata）
        if face_boxes is None:
            face_boxes = []
        if face_landmarks is None:
            face_landmarks = []

        if enable_color_match and face_boxes:
            # Save current state before color matching
            # RV-06: 中间文件必须有**真实扩展名**。ORIGINAL 用
            # `output_path + ".pre_color_match"` / ".color_matched"，
            # Pillow 的 Image.save() 无法从这种名字推断格式 → ValueError
            # （Pillow 文档：format 缺省时按扩展名推断）。
            color_match_temp = output_path + ".pre_color_match.tif"
            # R2-02：匹配结果按真实位深写出（16-bit 走 TIFF），
            # 因此扩展名必须是 .tif —— 旧代码固定 .png，配合 8-bit save
            # 会把整图量化。
            color_matched_out = output_path + ".color_matched.tif"
            if current_arr is not None:
                # Save 16-bit array as TIFF for color matching
                arr_u16 = (current_arr * 65535.0).clip(0, 65535).astype(
                    np.uint16)
                try:
                    import tifffile
                    tifffile.imwrite(color_match_temp, arr_u16,
                                    compression="zlib")
                except ImportError:
                    # Fallback: save as 8-bit for color matching
                    color_match_temp = output_path + ".pre_color_match.png"
                    arr_8 = (current_arr * 255.0).clip(0, 255).astype(
                        np.uint8)
                    Image.fromarray(arr_8).save(color_match_temp)
            else:
                color_match_temp = output_path + ".pre_color_match.png"
                current_img.convert("RGB").save(color_match_temp)
            match_result = _match_portrait_color(color_match_temp, face_boxes,
                                                 color_matched_out, logger,
                                                 landmarks=face_landmarks)
            match_applied = bool(match_result.get("applied"))
            if match_applied:
                # Reload the color-matched image
                # FIX-R13-2: For 16-bit path, load directly as numpy array
                # to avoid PIL I;16 3-channel data loss
                try:
                    if current_arr is not None:
                        # Try tifffile first for 16-bit
                        try:
                            import tifffile
                            loaded = tifffile.imread(color_matched_out)
                            if loaded.dtype == np.uint16:
                                current_arr = loaded.astype(
                                    np.float32) / 65535.0
                            else:
                                current_arr = loaded.astype(
                                    np.float32) / 255.0
                        except (ImportError, Exception):
                            # Fallback to PIL (may lose 16-bit precision)
                            cm_img = Image.open(color_matched_out)
                            cm_arr = np.array(cm_img.convert("RGB"),
                                              dtype=np.float32) / 255.0
                            current_arr = cm_arr
                    else:
                        current_img = Image.open(color_matched_out)
                    logger.info("STAGE5_COLOR_MATCH",
                                "Portrait color matching applied per v7.32 §3.10.7")
                except Exception:
                    logger.warning("STAGE5_COLOR_MATCH_RELOAD",
                                  "Failed to reload color-matched image")
            else:
                # RV-06: 请求了但没做成 —— 必须留痕，不能等同于成功
                reason = match_result.get("reason") or "not applied"
                skipped_operations.append("portrait_color_match")
                logger.warning("STAGE5_COLOR_MATCH_SKIP",
                               f"Color matching was requested but not applied: "
                               f"{reason}; continuing with unmatched pixels",
                               skipped_operations=["portrait_color_match"])
            # Clean up temp files
            for tmp in [color_match_temp, color_matched_out]:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass

        # Save before grain
        # FIX-P12-4: Ensure grain is applied in 16bit space per v7.32 §3.10.4
        # FIX-R13-2: For 16-bit path, save directly from numpy array via
        # tifffile to avoid PIL I;16 3-channel data loss. If tifffile is
        # unavailable, warn and save as 8-bit (quality loss).
        grain_temp = output_path + ".pre_grain.tif"  # TIFF for 16bit
        if current_arr is not None:
            # 16-bit array path — save via tifffile (preferred)
            arr_u16 = (current_arr * 65535.0).clip(0, 65535).astype(
                np.uint16)
            try:
                import tifffile
                tifffile.imwrite(grain_temp, arr_u16, compression="zlib")
            except ImportError:
                # No tifffile: must save as 8-bit (quality loss)
                logger.warning("STAGE5_NO_TIFFFILE",
                              "tifffile not available; saving 16-bit "
                              "pre-grain image as 8-bit. Install "
                              "tifffile for full 16-bit support.")
                arr_8 = (current_arr * 255.0).clip(0, 255).astype(
                    np.uint8)
                Image.fromarray(arr_8).save(grain_temp)
        elif bit_depth == 16:
            # PIL Image path (non-I;16, e.g. RGB mode)
            if current_img is not None and current_img.mode == "I;16":
                # FIX-NEW-V-18 / R21: 不再用 PIL 的 tiff_lzw。
                # LZW 解码在 tifffile 里要 `imagecodecs`（本机未安装），
                # 于是 stage5 写出的 LZW 文件 stage5 自己都读不回来：
                #   MediaPrecisionError: <COMPRESSION.LZW: 5> requires the
                #   'imagecodecs' package ... refusing to return 8-bit data
                # → STAGE5_ERROR → rc=86。
                # **写 LZW 就必须能读 LZW**；这里改用 tifffile 的 zlib
                # （Python 内置 zlib，无需额外依赖），与下面 RGB 分支一致。
                try:
                    import numpy as _np
                    import tifffile as _tf
                    _tf.imwrite(grain_temp,
                                _np.array(current_img, dtype=_np.uint16),
                                compression="zlib")
                except ImportError:
                    current_img.save(grain_temp, format="TIFF")
            elif current_img is not None:
                # Convert 8-bit PIL image to 16-bit for grain in 16bit space
                arr = np.array(current_img.convert("RGB"),
                              dtype=np.float32) / 255.0
                arr_16 = (arr * 65535.0).clip(0, 65535).astype(np.uint16)
                try:
                    import tifffile
                    tifffile.imwrite(grain_temp, arr_16, compression="zlib")
                except ImportError:
                    logger.warning("STAGE5_NO_TIFFFILE",
                                  "tifffile not available; saving 16-bit "
                                  "pre-grain image as 8-bit. Install "
                                  "tifffile for full 16-bit support.")
                    # Fallback: save as 8-bit (quality loss)
                    r = Image.fromarray(arr_16[:, :, 0], mode='I;16')
                    g = Image.fromarray(arr_16[:, :, 1], mode='I;16')
                    b = Image.fromarray(arr_16[:, :, 2], mode='I;16')
                    rgb_8 = Image.merge("RGB", (
                        r.point(lambda p: p // 256),
                        g.point(lambda p: p // 256),
                        b.point(lambda p: p // 256),
                    ))
                    rgb_8.save(grain_temp)
        else:
            # 8-bit path — save PIL Image directly
            if current_img is not None:
                current_img.save(grain_temp)
            elif current_arr is not None:
                # Should not happen in 8-bit path, but handle gracefully
                arr_8 = (current_arr * 255.0).clip(0, 255).astype(
                    np.uint8)
                Image.fromarray(arr_8).save(grain_temp)

        # FIX-P10-1 / FIX-P10-4: Grain synthesis in stage5
        # (v7.32 §3.1: grain after all AI + post-processing)
        # FIX-P10-4: This stage always runs regardless of ai_enabled
        # FIX-R13-1: Replaced G'MIC -film_grain with pure Python Laplacian
        # pyramid implementation (G'MIC -film_grain does not exist)
        grain_enabled = params.get("grain_synthesis", True)
        if grain_enabled:
            grain_strength = params.get("grain_strength", 0.3)
            grain_size = params.get("grain_size", 0.5)  # FIX-R13-1: 0.5 per v7.32 §3.10.4
            grain_roughness = params.get("grain_roughness", 0)  # FIX-P12-12: default 0 per v7.32 §3.10.4
            logger.info("STAGE5_GRAIN",
                        "Adding film grain via Laplacian pyramid (post-AI, post-postprocess)")
            if _add_grain(grain_temp, output_path,
                          grain_strength, grain_size, grain_roughness,
                          bit_depth=bit_depth):
                logger.info("STAGE5_GRAIN_DONE",
                            "Film grain applied in post-processing stage")
                try:
                    os.remove(grain_temp)
                except Exception:
                    pass
            else:
                # FIX-V06: 颗粒合成是配置要求的必要步骤（grain_synthesis=True）。
                # ORIGINAL 直接 shutil.move 未加颗粒的中间图并继续写 STAGE5_DONE 返回 0，
                # 属于"必要处理没执行但报成功"。仅当显式允许降级时才接受，且必须留痕。
                if not params.get("allow_stage5_degrade", False):
                    logger.error("STAGE5_GRAIN_FAIL",
                                 "Grain synthesis failed and degrade is not "
                                 "authorized; failing stage5 "
                                 "(set allow_stage5_degrade=true to accept an "
                                 "explicit degraded result)")
                    _discard_partial_output(output_path, [cache_meta_path])
                    _write_json_sidecar(
                        _json_sidecar_path(output_path, "status"), {
                            "status": STAGE_STATUS_FAILED,
                            "stage": 5,
                            "failed_operation": "grain_synthesis",
                            "reason": "grain_synthesis failed",
                            "impl": STAGE5_IMPL_VERSION,
                        })
                    try:
                        os.remove(grain_temp)
                    except Exception:
                        pass
                    return 86
                logger.warning("STAGE5_DEGRADED",
                               "Grain synthesis failed; DEGRADED result "
                               "accepted via allow_stage5_degrade",
                               skipped_operations=["grain_synthesis"])
                _promote_temp_to_output(grain_temp, output_path)
                _carry_icc_to_output(output_path, input_icc, logger)
                # 降级结果不写缓存身份：下次运行必须重跑完整处理，不能被复用
                _write_json_sidecar(
                    _json_sidecar_path(output_path, "status"), {
                        "status": STAGE_STATUS_DEGRADED,
                        "stage": 5,
                        "skipped_operations": ["grain_synthesis"],
                        "impl": STAGE5_IMPL_VERSION,
                    })
                logger.info("STAGE5_DONE",
                            "Post-processing complete (DEGRADED)",
                            output=output_path)
                return 0
        else:
            _promote_temp_to_output(grain_temp, output_path)
        # R3-07 / B06：源 ICC 必须跟着像素走到最终产物；写不进去（或回读
        # 不一致）就不能算成功。默认 require_icc_preserved=True。
        if input_icc:
            icc_ok = _carry_icc_to_output(output_path, input_icc, logger)
            if not icc_ok and params.get("require_icc_preserved", True):
                logger.error("STAGE5_ICC_REQUIRED_FAIL",
                             "Source ICC could not be preserved in the output; "
                             "failing instead of reporting success "
                             "(set require_icc_preserved=false to accept it)")
                _discard_partial_output(output_path, [cache_meta_path])
                _write_json_sidecar(
                    _json_sidecar_path(output_path, "status"), {
                        "status": STAGE_STATUS_FAILED,
                        "stage": 5,
                        "failed_operation": "icc_preservation",
                        "reason": "source ICC could not be preserved",
                        "impl": STAGE5_IMPL_VERSION,
                    })
                return 86

        # FIX-V07: 只在完全成功后提交缓存身份（先验证产物可读）
        if not _verify_output_readable(output_path):
            logger.error("STAGE5_OUTPUT_INVALID",
                         "Stage5 output failed validation; failing stage")
            _discard_partial_output(output_path, [cache_meta_path])
            _write_json_sidecar(
                _json_sidecar_path(output_path, "status"), {
                    "status": STAGE_STATUS_FAILED,
                    "stage": 5,
                    "failed_operation": "output_validation",
                    "impl": STAGE5_IMPL_VERSION,
                })
            return 86
        # R2-03：缓存条目必须记录"哪些操作没做成"，否则降级结果与完整结果
        # 在缓存层无法区分，依赖恢复后也不会重做。
        cache_identity["skipped_operations"] = list(skipped_operations)
        _write_json_sidecar(cache_meta_path, cache_identity)
        _write_json_sidecar(
            _json_sidecar_path(output_path, "status"), {
                "status": STAGE_STATUS_SUCCESS,
                "stage": 5,
                # RV-06: 被跳过/未执行的操作必须出现在成功记录里，
                # 否则"没有 skipped"会被下游当成"全部都做了"。
                "skipped_operations": skipped_operations,
                "impl": STAGE5_IMPL_VERSION,
            })
        logger.info("STAGE5_DONE", "Post-processing complete",
                    output=output_path)
        return 0

    except Exception as e:
        # FIX-V06: 必需处理失败必须传播给调度器。
        # ORIGINAL: 捕获异常 → 复制输入到最终产物路径 → 写 STAGE5_DONE → return 0。
        # 孤立复现显示输出与输入逐字节相同且返回成功，调度器无法区分。
        # 现在：不留最终产物、状态写 FAILED、诊断副本使用独立名称、按异常类型返回码。
        logger.error("STAGE5_ERROR", f"Post-process failed: {e}")
        if isinstance(e, MemoryError):
            exit_code = 87      # resource exhausted
        elif isinstance(e, (OSError, IOError)):
            exit_code = 82      # I/O error
        else:
            exit_code = 86      # validation/processing failure
        _discard_partial_output(output_path, [cache_meta_path])
        _write_json_sidecar(
            _json_sidecar_path(output_path, "status"), {
                "status": STAGE_STATUS_FAILED,
                "stage": 5,
                "failed_operation": "postprocess",
                "reason": f"{type(e).__name__}: {e}",
                "exit_code": exit_code,
                "impl": STAGE5_IMPL_VERSION,
            })
        # 诊断用副本：独立名称，绝不占用最终成功产物路径
        if input_source and os.path.exists(input_source):
            diag_path = output_path + ".failed_input_copy"
            try:
                # FIX-P12-11: Use copyfile (not copy2) per v7.32 §3.10.10
                shutil.copyfile(input_source, diag_path)
                logger.info("STAGE5_DIAG_COPY",
                            "Input kept as diagnostic copy (NOT a stage5 "
                            "result)", input=input_source, output=diag_path)
            except Exception as copy_err:
                logger.warning("STAGE5_DIAG_COPY_FAIL",
                               f"Diagnostic copy failed: {copy_err}")
        return exit_code


if __name__ == "__main__":
    # FIX-P11-5: main() must return explicit int exit code per v7.32 §3.5
    # ORIGINAL: sys.exit(main() or 0) — "or 0" masks None returns
    rc = main()
    if rc is None:
        rc = 86  # validation failure — main() must always return explicit exit code
    sys.exit(rc)
