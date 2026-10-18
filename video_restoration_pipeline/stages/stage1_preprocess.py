#!/usr/bin/env python3
"""
Stage 1: Traditional Video Preprocessing
Generates VapourSynth script for:
- IVTC (VIVTC/sRestore)
- Deinterlacing (QTGMC)
- Denoising (TemporalDegrain2)
- Stabilization (DepanStabilize)
- Chroma shift repair
- Deblocking
- BT.601->BT.709 color space conversion
- Black border crop/restore

v6.3 Compliance:
- IVTC fallback threshold from config
- InputType=1 for damaged field order detection
- matrix_in_s from color metadata or resolution heuristic
- All params read from config (not hardcoded)
"""

import os
import sys
import json
import argparse
import subprocess
import re
from pathlib import Path
from typing import Tuple, Dict, Optional

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from checkpoint import compute_param_hash, build_intermediate_name  # noqa: E402

try:  # NEW-V-06: 取消/中断守卫（KeyboardInterrupt 继承 BaseException，
    # 本文件所有 `except Exception` 都抓不到它——不装守卫时 Ctrl+C 会把
    # 半截渲染留在 inter_path 上，且 vspipe/ffmpeg 两个子进程变孤儿）
    import cancel_guard as _cancel_guard
except Exception:  # pragma: no cover - 缺失时不阻断 stage，仅失去该保护
    _cancel_guard = None


def _get_video_color_metadata(video_path: str) -> dict:
    """Get color metadata from video stream using ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=color_primaries,color_trc,colorspace,color_range",
             "-of", "json", video_path],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False
        )
        if r.returncode == 0:
            import json as _json
            data = _json.loads(r.stdout)
            if data.get("streams"):
                return data["streams"][0]
    except Exception:
        pass
    return {}


def detect_black_borders(video_path: str, logger=None) -> dict:
    """Detect black borders using ffmpeg cropdetect."""
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "info",
        "-i", video_path,
        "-vf", "cropdetect=limit=16:round=2:reset=0",
        "-frames:v", "100",
        "-f", "null", "-"
    ]
    crop_info = {"left": 0, "right": 0, "top": 0, "bottom": 0,
                 "cropped_w": 0, "cropped_h": 0}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                          timeout=120, shell=False)
        crop_match = None
        for line in r.stderr.split("\n"):
            m = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", line)
            if m:
                crop_match = m
        if crop_match:
            cw, ch, cx, cy = map(int, crop_match.groups())
            w_cmd = [
                "ffprobe", "-hide_banner", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0",
                video_path
            ]
            wr = subprocess.run(w_cmd, capture_output=True, text=True, errors="replace",
                               timeout=30, shell=False)
            if wr.returncode == 0:
                parts = wr.stdout.strip().split("x")
                if len(parts) == 2:
                    ow, oh = map(int, parts)
                    crop_info = {
                        "left": cx,
                        "right": ow - cx - cw,
                        "top": cy,
                        "bottom": oh - cy - ch,
                        "cropped_w": cw,
                        "cropped_h": ch,
                        "orig_w": ow,
                        "orig_h": oh,
                    }
    except Exception as e:
        if logger:
            logger.warning("STAGE1_CROP_FAIL", f"cropdetect failed: {e}")
    return crop_info


def validate_crop(crop_info: dict, logger=None) -> bool:
    """Validate crop parameters. Returns True if valid."""
    if not crop_info or crop_info.get("cropped_w", 0) == 0:
        return True

    ow = crop_info.get("orig_w", 1)
    oh = crop_info.get("orig_h", 1)
    cw = crop_info.get("cropped_w", 0)
    ch = crop_info.get("cropped_h", 0)

    if cw < 240 or ch < 240:
        if logger:
            logger.error("STAGE1_CROP_TOO_SMALL",
                        f"Cropped size {cw}x{ch} below minimum 240px")
        return False

    orig_area = ow * oh
    crop_area = cw * ch
    if crop_area < orig_area * 0.5:
        if logger:
            logger.error("STAGE1_CROP_TOO_MUCH",
                        f"Cropped area {crop_area} < 50% of original {orig_area}")
        return False

    return True


def _get_matrix_in(video_path: str, target_res: str, params: dict) -> str:
    """Determine matrix_in_s based on color metadata or resolution fallback.

    Priority:
    1. color_primaries from ffprobe (bt470bg->470bg, smpte170m->170m)
    2. Resolution heuristic using SOURCE video height (not target)

    FIX-P6-4: Resolution fallback must use SOURCE video height per v7.32 §3.10.3.
    ORIGINAL: Used target_res to determine PAL/NTSC, which is wrong for
    PAL 720x576 source with target_resolution="1280x720" (would return 170m).
    """
    # FIX-P6-7: Accept pre-fetched color_meta from params to avoid duplicate ffprobe call
    # ORIGINAL: Always called _get_video_color_metadata() — duplicate call in generate_vs_script
    color_meta = params.get("_color_meta_cache")
    if color_meta is None:
        color_meta = _get_video_color_metadata(video_path)
    return _get_matrix_in_from_meta(video_path, color_meta, target_res, params)


def _get_matrix_in_from_meta(video_path: str, color_meta: dict,
                              target_res: str, params: dict) -> str:
    """Determine matrix_in_s from pre-fetched color metadata or resolution fallback.

    FIX-P6-7: Extracted from _get_matrix_in to support single ffprobe call pattern.
    """
    cp = color_meta.get("color_primaries", "")
    # FIX-P6-3: bt470m is NTSC (BT.470 System M) -> 170m, not PAL 470bg per v7.32 §3.10.3
    # ORIGINAL: if cp in ("bt470bg", "bt470m"): return "470bg"
    # bt470m was incorrectly mapped to PAL 470bg; it's actually NTSC
    if cp == "bt470bg":
        return "470bg"
    if cp in ("smpte170m", "bt470m"):
        return "170m"

    # FIX-P6-2: BT.709 source — identity, no conversion needed per v7.32 §3.10.3
    # ORIGINAL: No handling for bt709; BT.709 source was forced through BT.601→BT.709 conversion
    cs = color_meta.get("colorspace", "")
    if cp in ("bt709",) or cs in ("bt709",):
        return "709"

    # FIX-P6-4: Resolution fallback: use SOURCE video height, not target_res
    source_h = 480
    try:
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=height",
             "-of", "json", video_path],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            if data.get("streams"):
                source_h = int(data["streams"][0].get("height", 480))
    except Exception:
        source_h = 480

    # FIX-P6-1: Only exact 576-line = PAL SD BT.601 per v7.32 §3.10.3
    # ORIGINAL: if source_h <= 576: return "470bg" — NTSC 480-line incorrectly mapped to PAL
    if source_h == 576:
        return "470bg"  # PAL SD BT.601
    return "170m"  # NTSC BT.601 (480-line NTSC SD and all other resolutions)


def detect_shakiness(video_path: str, logger=None, samples: int = 12,
                     frames_per_sample: int = 40,
                     threshold_px: float = 1.0) -> dict:
    """抽样判定素材**是否真的需要防抖**（`stabilize: auto` 的决策依据）。

    背景：`haf.Stab`（mvtools depan）在 1080p 上约 **3.1 秒/帧**，实测占
    stage1 总耗时的 **89%**（QTGMC Slower 只要 0.25 秒/帧）。稳定素材
    （三脚架/演播室录制）位移≈0，防抖收益为零还可能裁掉边缘。故默认
    `auto`：先抽样测全局位移，再决定是否加 Stab。

    实现要点（两次踩坑后的最终方案）：
    * **不用** ffmpeg `vidstabdetect` 的 trf：它是**二进制** `TRF1` 格式，
      不是文本（早期用 awk/split 解析会得到假的"位移 0.03px"）。
    * **不用** `tblend=difference` 的帧间差：实测抖动素材的 YAVG 反而
      比稳定素材低（合成平移大部分区域相同），区分度不可靠。
    * **改为**：ffmpeg 抽低分辨率灰度帧 → `cv2.phaseCorrelate` 算相邻帧
      的**真实全局位移**（自测已知平移 5px 可精确检出 5.0）。

    抽样：在时长上均匀取 samples 段，每段 frames_per_sample 帧，缩到
    320px 宽做相位相关——实测单段毫秒级，总开销可忽略。

    返回：
      needs_stabilization: bool
      mean_abs_dx / mean_abs_dy / p95_abs：位移统计（像素，源分辨率）
      samples_used / reason / error
    探测失败时 **保守返回 needs_stabilization=True**（宁可多做，不误关）。
    """
    import tempfile

    out = {"needs_stabilization": True, "mean_abs_dx": 0.0,
           "mean_abs_dy": 0.0, "max_abs_dx": 0, "max_abs_dy": 0,
           "p95_abs": 0.0, "samples_used": 0, "reason": "", "error": None}

    # --- 总时长（用于均匀抽样） ---
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, errors="replace", timeout=30)
        total = float((probe.stdout or "0").strip() or 0)
        # 源宽度：相位相关在 320px 上算，位移要换算回源分辨率
        wp = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, errors="replace", timeout=30)
        probe_w = int((wp.stdout or "0").strip() or 0)
    except Exception as exc:
        out["error"] = f"duration probe failed: {exc}"
        out["reason"] = "时长未知，保守启用防抖"
        return out
    if total <= 0:
        out["error"] = "duration unavailable"
        out["reason"] = "时长未知，保守启用防抖"
        return out

    try:
        import cv2
        import numpy as np
    except Exception as exc:
        out["error"] = f"cv2/numpy unavailable: {exc}"
        out["reason"] = "缺少 cv2，无法做位移抽样，保守启用防抖"
        return out

    # --- 均匀抽样 ---
    step = max(total / max(samples, 1), 1.0)
    offsets, t = [], 0.0
    while t < total and len(offsets) < samples:
        offsets.append(t)
        t += step

    ANALYZE_W = 320
    dxs, dys = [], []
    segments = []          # 每段独立统计，用于报告"哪些段抖"
    for off in offsets:
        # 抽成 PNG 序列到临时目录（-ss 放在 -i 前做快速 seek）
        tmpdir = tempfile.mkdtemp(prefix="shake_")
        try:
            pattern = os.path.join(tmpdir, "f_%04d.png").replace(os.sep, "/")
            subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{off:.3f}",
                 "-i", video_path,
                 "-vf", f"scale={ANALYZE_W}:-2,format=gray",
                 "-frames:v", str(frames_per_sample),
                 "-y", pattern],
                capture_output=True, text=True, errors="replace", timeout=180)
            frames = sorted(f for f in os.listdir(tmpdir) if f.endswith(".png"))
            if len(frames) < 2:
                continue
            prev = None
            seg_dx, seg_dy = [], []
            for fn in frames:
                img = cv2.imread(os.path.join(tmpdir, fn),
                                 cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                cur = img.astype(np.float32)
                if prev is not None and prev.shape == cur.shape:
                    (sx, sy), resp = cv2.phaseCorrelate(prev, cur)
                    # 低响应视为不可信（画面几乎无特征），跳过
                    if resp >= 0.05:
                        # 换算回源分辨率
                        scale_back = (float(probe_w) / ANALYZE_W
                                      if probe_w else 1.0)
                        dxs.append(abs(sx) * scale_back)
                        dys.append(abs(sy) * scale_back)
                        seg_dx.append(abs(sx) * scale_back)
                        seg_dy.append(abs(sy) * scale_back)
                prev = cur
            if seg_dx:
                _m = (sum(seg_dx) / len(seg_dx) + sum(seg_dy) / len(seg_dy)) / 2
                segments.append({"start_s": round(off, 2),
                                 "mean_px": round(_m, 3),
                                 "shaky": _m >= threshold_px})
            out["samples_used"] += 1
        except Exception:
            pass
        finally:
            try:
                for f in os.listdir(tmpdir):
                    os.unlink(os.path.join(tmpdir, f))
                os.rmdir(tmpdir)
            except Exception:
                pass

    if not dxs:
        out["error"] = "phaseCorrelate produced no usable samples"
        out["reason"] = "抽样无有效位移数据，保守启用防抖"
        return out

    all_abs = dxs + dys
    all_abs_sorted = sorted(all_abs)
    p95 = all_abs_sorted[min(len(all_abs_sorted) - 1,
                             int(len(all_abs_sorted) * 0.95))]
    out["mean_abs_dx"] = round(sum(dxs) / len(dxs), 3)
    out["mean_abs_dy"] = round(sum(dys) / len(dys), 3)
    out["max_abs_dx"] = int(max(dxs))
    out["max_abs_dy"] = int(max(dys))
    out["p95_abs"] = round(float(p95), 3)

    mean_all = (out["mean_abs_dx"] + out["mean_abs_dy"]) / 2
    if mean_all < threshold_px and p95 < threshold_px * 3:
        out["needs_stabilization"] = False
        out["reason"] = (f"stable: mean|d|={mean_all:.2f}px p95={p95:.2f}px "
                         f"(< {threshold_px}px) -> skip Stab")
    else:
        out["needs_stabilization"] = True
        out["reason"] = (f"shaky: mean|d|={mean_all:.2f}px p95={p95:.2f}px "
                         f"(>= {threshold_px}px) -> enable Stab")

    out["segments"] = segments
    _shaky_segs = [g for g in segments if g.get("shaky")]
    out["shaky_segment_count"] = len(_shaky_segs)
    if _shaky_segs:
        out["shaky_segments"] = _shaky_segs
        out["reason"] += (f"; {len(_shaky_segs)}/{len(segments)} sampled "
                          f"segments exceed {threshold_px}px "
                          f"(e.g. at "
                          + ", ".join(f"{g['start_s']}s" for g in _shaky_segs[:5])
                          + ")")

    if logger:
        try:
            if _shaky_segs:
                # 局部抖动：整体判定可能为"稳定"，但某些段明显抖。
                # 这里单独告警，便于用户决定是否对全片开启防抖。
                logger.warning(
                    "STAGE1_SHAKE_LOCALIZED",
                    f"{len(_shaky_segs)}/{len(segments)} sampled segments are "
                    f"shaky (>={threshold_px}px) while the overall average is "
                    f"{(out['mean_abs_dx']+out['mean_abs_dy'])/2:.2f}px. "
                    "Stab is applied to the WHOLE clip, so enabling it for "
                    "these few segments would cost the full runtime; "
                    "set stabilize=DepanStabilize to force it on.",
                    shaky_segments=[g["start_s"] for g in _shaky_segs])
            logger.info("STAGE1_SHAKE_PROBE", "Shakiness sampled: "
                        + out["reason"],
                        samples_used=out["samples_used"],
                        mean_abs_dx=out["mean_abs_dx"],
                        mean_abs_dy=out["mean_abs_dy"],
                        p95_abs=out["p95_abs"],
                        needs_stabilization=out["needs_stabilization"])
        except Exception:
            pass
    return out


def _fps_to_rational(fps):
    """Convert a framerate float to (numerator, denominator) for AssumeFPS.

    FIX-W11: used to rebuild PTS after std.Splice of per-segment QTGMC output
    per v7.32 §3.10.1/§3.10.13. Maps common NTSC/PAL rates exactly; falls back
    to a limited-denominator rational approximation for unusual rates.
    """
    if not fps or fps <= 0:
        return 25, 1  # safe PAL default; caller ensures fps>0 in normal flow
    common = {
        23.976: (24000, 1001), 23.98: (24000, 1001), 24.0: (24, 1),
        25.0: (25, 1), 29.97: (30000, 1001), 30.0: (30, 1),
        50.0: (50, 1), 59.94: (60000, 1001), 60.0: (60, 1),
    }
    for v, nd in common.items():
        if abs(fps - v) < 0.02:
            return nd
    from fractions import Fraction
    f = Fraction(fps).limit_denominator(1001)
    return f.numerator, f.denominator


def generate_vs_script(input_path: str, params: dict, job_id: str,
                       crop_info: Optional[dict] = None,
                       ivtc_stats: Optional[dict] = None) -> str:
    """Generate VapourSynth script content."""
    # FIX-R5-06: vspipe 会**临时把 cwd 切到 vpy 脚本所在目录**（R55+ 默认，
    # 官方文档见 vspipe `-c/--preserve-cwd`："Don't temporarily change the
    # working directory to the script directory (R54 compatibility)"）。
    # 因此 vpy 里 `core.ffms2.Source(source=...)` 的相对路径按 **vpy 目录**
    # 解析而非进程 cwd。main.py 目前传的是绝对路径所以没暴露，但一旦传相对
    # 路径就会 `Index: Can't open`。这里统一转绝对路径防御。
    input_path = os.path.abspath(input_path)
    field_order = params.get("field_order", "tt")
    needs_ivtc = params.get("needs_ivtc", False)
    is_psf = params.get("is_psf", False)

    ivtc = params.get("ivtc", "VIVTC")
    deinterlace = params.get("deinterlace", "QTGMC")
    # FIX-spec: havsfunc VS function is TemporalDegrain (no "2" suffix) per
    # v7.32 §3.10.15; default fallback name aligned to revised spec.
    denoise = params.get("denoise", "TemporalDegrain")
    denoise_params = params.get("denoise_params", {})
    stabilize = params.get("stabilize", "DepanStabilize")
    stabilize_params = params.get("stabilize_params", {})
    chroma_shift = params.get("chroma_shift", False)
    deblock = params.get("deblock", "")
    target_res = params.get("target_resolution", "720x480")
    # FIX-W10: force 16bit processing space per v7.32 §3.10.1 ("强制 16bit 处理空间").
    # ORIGINAL allowed 8/10bit via config; sub-16bit violates the 16bit rule.
    _requested_bd = params.get("vs_bit_depth", 16)
    if _requested_bd < 16:
        params["_vs_bit_depth_upgraded"] = (_requested_bd, 16)
        vs_bit_depth = 16
    else:
        vs_bit_depth = 16

    # FIX-P6-7: Fetch color metadata ONCE, cache for _get_matrix_in to reuse
    # ORIGINAL: _get_video_color_metadata called twice (in _get_matrix_in + here)
    color_meta = _get_video_color_metadata(input_path)
    params["_color_meta_cache"] = color_meta

    # Determine matrix based on metadata + resolution
    matrix_in = _get_matrix_in(input_path, target_res, params)

    # Clean up cache (don't leak into downstream params)
    params.pop("_color_meta_cache", None)

    # VapourSynth format based on bit depth (FIX-W10: always 16bit)
    vs_format = "vs.YUV420P16"

    # Determine range from pre-fetched color metadata (FIX-P6-2 / FIX-P6-7)
    detected_range = color_meta.get("color_range", "")
    if detected_range == "tv":
        color_range = "limited"
    elif detected_range == "pc":
        color_range = "full"
    else:
        # FIX-P6-3: Log WARNING when color_range cannot be detected per v6.32 §7.19
        # "若无法读取源 range 属性，默认假设老影像为 limited range，但记录 WARNING"
        color_range = params.get("color_range", "limited")
        if detected_range not in ("tv", "pc"):
            # Will be logged by caller (main()) since generate_vs_script
            # doesn't have access to logger. Mark in params for downstream logging.
            params["_color_range_unknown"] = True
    range_in_str = "vs.RANGE_FULL" if color_range == "full" else "vs.RANGE_LIMITED"
    range_out_str = "vs.RANGE_LIMITED"

    # Black level dynamic calculation per v6.3 §3.10.3
    if color_range == "full":
        black_y = 0
        neutral_uv = 128 * (2 ** (vs_bit_depth - 8))
    else:
        if vs_bit_depth == 8:
            black_y, neutral_uv = 16, 128
        elif vs_bit_depth == 10:
            black_y, neutral_uv = 64, 512
        else:
            # Generic formula for any bit depth (including 16bit)
            black_y = int(16 * (2 ** (vs_bit_depth - 8)))
            neutral_uv = int(128 * (2 ** (vs_bit_depth - 8)))

    # Crop parameters
    has_crop = crop_info is not None and crop_info.get("cropped_w", 0) > 0
    crop_left = crop_info.get("left", 0) if crop_info else 0
    crop_right = crop_info.get("right", 0) if crop_info else 0
    crop_top = crop_info.get("top", 0) if crop_info else 0
    crop_bottom = crop_info.get("bottom", 0) if crop_info else 0

    lines = []
    lines.append("import vapoursynth as vs")
    lines.append("import havsfunc as haf")
    # R4: havsfunc r33 **没有** TemporalDegrain / DepanStabilize（实测 101 个函数里
    # 都没有），直接调用会 AttributeError。按实际可得性追加模块：
    #   TemporalDegrain2 -> G41Fun（doom9 实测其效果优于 v1：失焦区颗粒去得更干净
    #                      且保留对焦细节，v1 在运动场景会丢矢量抹掉细节）
    #   TemporalDegrain  -> fvsfunc（Avisynth TemporalDegrain 的 VS 移植）
    if denoise == "TemporalDegrain2":
        lines.append("import G41Fun")
    elif denoise == "TemporalDegrain":
        lines.append("import fvsfunc as fvf")
    lines.append("core = vs.core")
    lines.append("")

    # Source
    lines.append(f"# Load source: {input_path}")
    lines.append(f"clip = core.ffms2.Source(source={repr(input_path)})")
    lines.append("")

    # Convert to target bit depth
    lines.append(f"# Convert to {vs_bit_depth}bit processing space")
    lines.append(f"clip = core.resize.Bicubic(clip, format={vs_format})")
    lines.append("")

    # Default tff_bool for use throughout script
    tff_bool = "True" if field_order == "tt" else "False"

    # QTGMC parameters (moved here so PSF path can use them) FIX-P6-3/P6-11
    qtgmc_preset = params.get("qtgmc_preset", "Slow")
    # Validate preset
    valid_presets = {"Placebo", "Very Slow", "Slower", "Slow", "Medium", "Fast", "Faster",
                     "Very Fast", "Super Fast", "Ultra Fast", "Draft"}
    if qtgmc_preset not in valid_presets:
        qtgmc_preset = "Slow"

    fps_divisor = params.get("qtgmc_fps_divisor", 1)  # FIX-P6-3: default keep double frame rate
    qtgmc_ezdenoise = params.get("qtgmc_ezdenoise", 2)
    # Validate EZDenoise range
    qtgmc_ezdenoise = max(0, min(qtgmc_ezdenoise, 10))

    qtgmc_sharpness = params.get("qtgmc_sharpness", 0.5)
    qtgmc_chromamotion = params.get("qtgmc_chromamotion", True)
    qtgmc_sourcematch = params.get("qtgmc_sourcematch", 3)
    qtgmc_input_type = params.get("qtgmc_input_type", 0)
    qtgmc_clear_fieldbased = params.get("qtgmc_clear_fieldbased", True)
    cm_str = "True" if qtgmc_chromamotion else "False"

    # PSF handling
    if is_psf:
        # FIX-W2: use the preserved original TFF/BFF for SeparateFields per
        # v7.32 §3.10.1. ORIGINAL used tff_bool (computed from field_order=="tt"),
        # which is always False for PSF (field_order=="psf"), causing a 1-row
        # vertical offset on TFF PSF material after Weave. psf_tff is propagated
        # from classify.py via main.py (job -> params).
        psf_tff = params.get("psf_tff", False)
        psf_tff_str = "True" if psf_tff else "False"
        lines.append("# PSF handling: SeparateFields + Weave")
        lines.append(f"clip = core.std.SeparateFields(clip, tff={psf_tff_str})")
        lines.append("clip = core.std.Weave(clip)")
        # FIX-P6-11: Optional progressive repair for PSF
        if params.get("psf_use_input_type1", False):
            lines.append("# Optional progressive repair for PSF")
            # FIX-P6-2: PSF FpsDivisor=1 per v7.32 §3.10.1
            # ORIGINAL: FpsDivisor={fps_divisor} — PSF restore already produces correct frame rate,
            # must NOT halve it. FpsDivisor=1 keeps the native PSF frame rate.
            lines.append(f"clip = haf.QTGMC(clip, Preset='{qtgmc_preset}', "
                        f"EZDenoise={qtgmc_ezdenoise}, Sharpness={qtgmc_sharpness}, "
                        f"ChromaMotion={cm_str}, SourceMatch={qtgmc_sourcematch}, "
                        f"InputType=1, FPSDivisor=1)")  # havsfunc r33: FPSDivisor
        lines.append("")

    # IVTC (must be before deinterlace)
    if needs_ivtc and not is_psf:
        lines.append("# IVTC - must execute before QTGMC")
        if ivtc == "VIVTC":
            # FIX-P6-8: VFM mode/VDecimate cycle from config per v6.32 §7.9 / v7.32 §3.10.13
            # ORIGINAL: hardcoded mode=1, cycle=5
            vfm_mode = params.get("vfm_mode", 1)
            vdecimate_cycle = params.get("vdecimate_cycle", 5)
            # V-P-019 fix: VFM and VDecimate must be chained directly per v7.32 §3.10.13
            # FIX-P6-7: _Combed diagnostic block between VFM and VDecimate per v7.32 §3.10.13
            lines.append(f"clip = core.vivtc.VFM(clip, mode={vfm_mode}, ovr=0)")
            lines.append(f"# _Combed statistics collection (sidecar export) per v7.32 §3.10.13")
            lines.append(f"_combed_stats = " + r'{"combed_frames": [], "total": 0, "mode": ' + str(vfm_mode) + r'}')
            lines.append("def _collect_combed(n, f):")
            lines.append("    if f.props.get('_Combed', 0):")
            lines.append("        _combed_stats['combed_frames'].append(n)")
            lines.append("    _combed_stats['total'] += 1")
            lines.append("    return f")
            # FIX-E1: ModifyFrame is EXEMPT from the "no filters between VFM/VDecimate" rule
            # because it only reads frame properties (_Combed) without modifying pixels or
            # frame attributes. VDecimate's combed-frame detection relies solely on VFM's
            # output pixel data and _Combed prop, neither of which ModifyFrame alters.
            # Per v7.32 §3.10.13, the prohibition targets "修改帧内容/属性的滤镜" only.
            lines.append("_diag = core.std.ModifyFrame(clip, clip, _collect_combed)")
            lines.append("# Eagerly request first 300 frames to trigger callbacks")
            lines.append("_max_diag = min(300, clip.num_frames)")
            lines.append("for _ci in range(_max_diag):")
            lines.append("    try:")
            lines.append("        _diag.get_frame(_ci)")
            lines.append("    except Exception:")
            lines.append("        break")
            lines.append("import os as _os, json as _json")
            lines.append("_sidecar_path = _os.environ.get('VS_COMBED_STATS', '')")
            lines.append("if _sidecar_path:")
            lines.append("    try:")
            lines.append("        with open(_sidecar_path, 'w', encoding='utf-8') as _sf:")
            lines.append("            _json.dump(_combed_stats, _sf)")
            lines.append("    except Exception:")
            lines.append("        pass")
            lines.append(f"clip = core.vivtc.VDecimate(clip, cycle={vdecimate_cycle})")
        elif ivtc == "sRestore":
            lines.append("clip = haf.srestore(clip, frate=23.976)")
        lines.append("")

    # Black border crop (before color conversion)
    if has_crop and (crop_left > 0 or crop_right > 0 or crop_top > 0 or crop_bottom > 0):
        lines.append("# Auto-crop black borders")
        lines.append(f"clip = core.std.CropRel(clip, left={crop_left}, right={crop_right}, top={crop_top}, bottom={crop_bottom})")
        lines.append("")

    # V-P-025 fix: Remove InputType=1 test code from generated script
    # (damaged field order detection should be done in classify.py, not in VS script)

    # Deinterlace (QTGMC)
    field_segments = params.get("field_segments", [])
    use_segmented = bool(field_segments) and not is_psf and deinterlace == "QTGMC"

    if use_segmented:
        # FIX-W15: force InputType=0 for true interlaced (tt/bb) segments per
        # v7.32 §3.10.1. config may set qtgmc_input_type=1/2 (progressive/residual),
        # which is wrong for true interlaced and would cause deinterlace errors.
        if qtgmc_input_type != 0:
            params["_qtgmc_input_type_override"] = (qtgmc_input_type, 0)
            qtgmc_input_type = 0
        lines.append("# Mixed field order: per-segment QTGMC deinterlacing")
        segment_vars = []
        for idx, seg in enumerate(field_segments):
            seg_field = seg.get("field_order", "tt")
            if seg_field not in ("tt", "bb"):
                continue
            start_f = seg.get("start_frame", 0)
            end_f = seg.get("end_frame", 0)
            if end_f <= start_f:
                continue
            tff_bool = "True" if seg_field == "tt" else "False"
            var_name = f"seg{idx}"
            segment_vars.append(var_name)
            lines.append(f"{var_name} = core.std.Trim(clip, first={start_f}, last={end_f})")
            if qtgmc_input_type == 0 and qtgmc_clear_fieldbased:
                field_based_val = 2 if seg_field == "tt" else 1
                lines.append(f"{var_name} = core.std.SetFrameProp({var_name}, prop='_FieldBased', intval={field_based_val})")
            lines.append(f"{var_name} = haf.QTGMC({var_name}, Preset='{qtgmc_preset}', "
                        f"EZDenoise={qtgmc_ezdenoise}, Sharpness={qtgmc_sharpness}, "
                        f"ChromaMotion={cm_str}, SourceMatch={qtgmc_sourcematch}, "
                        f"InputType={qtgmc_input_type}, TFF={tff_bool}, "
                        f"FPSDivisor={fps_divisor})")  # havsfunc r33: FPSDivisor（全大写 FPS）
            lines.append("")
        if segment_vars:
            splice_args = ", ".join(segment_vars)
            lines.append(f"clip = core.std.Splice([{splice_args}], mismatch=False)")
            lines.append("clip = core.std.SetFieldBased(clip, 0)")  # FIX-P6-6
            # FIX-W11: enforce PTS continuity after Splice per v7.32 §3.10.1/§3.10.13
            # ("std.Splice() 拼接,同步修正 PTS"). AssumeFPS rebuilds PTS from the
            # source frame rate (QTGMC doubles; FpsDivisor=2 restores original rate,
            # FpsDivisor=1 keeps 2x). CFR assumption — VFR sources need fuller handling.
            _seg_fps = params.get("original_fps", params.get("fps", 0)) or 0
            if _seg_fps > 0 and fps_divisor != 2:
                _seg_fps = _seg_fps * 2  # QTGMC bob doubles; FpsDivisor!=2 keeps double
            _fps_num, _fps_den = _fps_to_rational(_seg_fps)
            lines.append(f"clip = core.std.AssumeFPS(clip, fpsnum={_fps_num}, fpsden={_fps_den})")
            lines.append("")
        else:
            use_segmented = False

    if not use_segmented and not is_psf and field_order in ("tt", "bb") and deinterlace == "QTGMC":
        # FIX-W15: force InputType=0 for true interlaced (tt/bb) per v7.32 §3.10.1.
        # config may set qtgmc_input_type=1/2 (progressive/residual), which is wrong
        # for true interlaced and would cause deinterlace errors.
        if qtgmc_input_type != 0:
            params["_qtgmc_input_type_override"] = (qtgmc_input_type, 0)
            qtgmc_input_type = 0
        lines.append("# Deinterlace with QTGMC")
        if qtgmc_input_type == 0 and qtgmc_clear_fieldbased:
            field_based_val = 2 if field_order == "tt" else 1
            lines.append("# Ensure correct field order property for QTGMC")
            lines.append(f"clip = core.std.SetFrameProp(clip, prop='_FieldBased', intval={field_based_val})")
        lines.append(f"clip = haf.QTGMC(clip, Preset='{qtgmc_preset}', "
                    f"EZDenoise={qtgmc_ezdenoise}, Sharpness={qtgmc_sharpness}, "
                    f"ChromaMotion={cm_str}, SourceMatch={qtgmc_sourcematch}, "
                    f"InputType={qtgmc_input_type}, TFF={tff_bool}, "
                    f"FPSDivisor={fps_divisor})")  # havsfunc r33: FPSDivisor（全大写 FPS）
        # v6.3: Reset _FieldBased to 0 after InputType=0 processing
        if qtgmc_input_type == 0:
            lines.append("clip = core.std.SetFieldBased(clip, 0)")
        lines.append("")
    elif field_order == "unknown" and not is_psf:
        # v6.3: unknown field_order -> skip QTGMC InputType=0, conservative handling
        # FIX-P6-5: Clear _FieldBased attribute per v7.32 §3.10.1
        # ORIGINAL: Only added comment, did not clear _FieldBased property
        lines.append("# field_order=unknown: skipping QTGMC deinterlace (conservative)")
        lines.append("clip = core.std.SetFieldBased(clip, 0)")
        lines.append("")
    elif field_order == "progressive" and not is_psf:
        # FIX-P6-9: explicit comment for progressive per v7.32 §3.10.1
        # field_order=progressive: native progressive, skipping QTGMC deinterlace
        lines.append("# field_order=progressive: native progressive, skipping QTGMC deinterlace")
        lines.append("")

    # Denoise
    # FIX-P1-9: havsfunc VS function is TemporalDegrain (no "2" suffix);
    # r33 params are sigma/blockWidth (NOT degrainTR/grainLevel which are
    # AviSynth TemporalDegrain2 params).
    # R4: 原来无论配置 TemporalDegrain 还是 TemporalDegrain2，都生成
    # `haf.TemporalDegrain(...)`，而 havsfunc r33 并没有这个函数 ——
    # 两个配置都会 AttributeError，且 TD2 被错误地降级成效果更差的 v1。
    if denoise == "TemporalDegrain2":
        # G41Fun 的 VS 版 TemporalDegrain2（havsfunc 里没有，r33 只有 SMDegrain）
        # R4: 实测 G41Fun 的签名为
        #   (clip, degrainTR=2, degrainPlane=4, meAlg=5, ..., postFFT=0,
        #    postTR=1, postSigma=1, ...)
        # **没有 grainLevel**（那是 Hybrid/Selur 分支版的写法）——
        # 传 grainLevel 会 TypeError。这里只传确定存在的参数。
        degrain_tr = max(0, min(int(denoise_params.get("degrainTR", 2)), 3))
        degrain_plane = int(denoise_params.get("degrainPlane", 4))
        lines.append("# Temporal denoising (TemporalDegrain2)")
        lines.append(f"clip = G41Fun.TemporalDegrain2(clip, "
                    f"degrainTR={degrain_tr}, degrainPlane={degrain_plane})")
        lines.append("")
    elif denoise == "TemporalDegrain":
        # v1 由 fvsfunc 提供（Avisynth TemporalDegrain 的 VS 移植）。
        # R4: fvsfunc 的签名是 (input_, sigma, bw, bh, ...)，
        # **没有 blockWidth** —— 实测传 blockWidth 会
        # TypeError: unexpected keyword argument 'blockWidth'。
        # 这里同时接受新名 bw/bh 与旧配置名 blockWidth。
        sigma = denoise_params.get("sigma", 16)
        bw = denoise_params.get("bw", denoise_params.get("blockWidth", 16))
        bh = denoise_params.get("bh", 16)
        lines.append("# Temporal denoising (TemporalDegrain v1)")
        lines.append(f"clip = fvf.TemporalDegrain(clip, "
                    f"sigma={sigma}, bw={bw}, bh={bh})")
        lines.append("")

    # Deblocking
    if deblock == "Deblock_QED":
        # FIX-NEW-V-15 / R17: havsfunc r33 的签名是
        # Deblock_QED(clp, quant1=24, quant2=26, aOff1=1, bOff1=2,
        #             aOff2=1, bOff2=2, uv=3)
        # **没有 quant 参数**。原来写 `quant=` → TypeError → vpy 求值失败
        # → vspipe rc=1 → stage1 STAGE1_FAILED rc=84。
        # 后果：`paths.dv_digital.deblock = "Deblock_QED"` 这条配置
        # **从未真正执行过**（此前 dv_digital 只在 v6/v7 之外的路径上跑过，
        # 且失败被当成"素材不支持"）。
        # 语义映射：单个 quant 值同时给 quant1/quant2（保持"整体强度"
        # 这一原有意图），默认值对齐 havsfunc 的 24。
        deblock_quant = int(params.get("deblock_quant", 24))
        lines.append("# Deblocking")
        lines.append(f"clip = haf.Deblock_QED(clip, quant1={deblock_quant}, "
                     f"quant2={deblock_quant})")
        lines.append("")
    elif deblock == "DeblockPP7":
        # FIX-P6-12: Support configurable DeblockPP7 parameters per v7.32 §3.10.1
        # ORIGINAL: Hardcoded call with no parameters
        deblock_pp7_params = params.get("deblock_pp7_params", {})
        lines.append("# Deblocking (HDV)")
        if deblock_pp7_params:
            pp7_args = ", ".join(f"{k}={v}" for k, v in deblock_pp7_params.items())
            lines.append(f"clip = core.deblock.Deblock(clip, {pp7_args})")
        else:
            lines.append("clip = core.deblock.Deblock(clip)")
        lines.append("")

    # Chroma shift
    if chroma_shift:
        # R4: **havsfunc r33 并没有 ChromaShift**（实测 8 个 haf.* 调用里它是
        # 缺失项之一），原来这行一执行就 AttributeError。
        # 改用 **VS 原生**实现，等价于 Avisynth 的 ChromaShiftSP
        # （doom9 上 DJATOM 给出的官方等价写法）：抽出 Y 平面，对整帧用
        # resize 的 src_left/src_top 做亚像素平移，再合并回 YUV。
        # 好处是不再依赖任何外部脚本，且支持亚像素精度。
        cx = float(params.get("chroma_shift_cx", 2))
        cy = float(params.get("chroma_shift_cy", 0))
        lines.append("# Chroma shift correction (VS-native, no havsfunc dependency)")
        lines.append("_cs_y = core.std.ShufflePlanes(clip, planes=0, "
                    "colorfamily=vs.GRAY)")
        lines.append(f"_cs_shift = core.resize.Spline16("
                    f"clip, width=clip.width, height=clip.height, "
                    f"src_left={cx}, src_top={cy}, "
                    f"src_width=clip.width + {cx}, "
                    f"src_height=clip.height + {cy})")
        lines.append("clip = core.std.ShufflePlanes("
                    "clips=[_cs_y, _cs_shift], planes=[0, 1, 2], "
                    "colorfamily=vs.YUV)")
        lines.append("")

    # Stabilization
    # FIX-效率：`stabilize` 支持四态，用户可任意手动覆盖：
    #   "auto"（默认）  —— 抽样测位移后自动决定（见 detect_shakiness）
    #   "DepanStabilize" / "Stab" —— 强制启用
    #   "" / "none" / "off" / false —— 强制关闭
    # 之所以引入 auto：`haf.Stab` 在 1080p 上约 3.1 秒/帧，占 stage1
    # 总耗时 89%（QTGMC Slower 仅 0.25 秒/帧）；而稳定素材位移≈0，
    # 防抖收益为零还可能裁掉边缘。实测同一素材：关 Stab 后 20 帧从
    # 70 秒降到 8 秒（8.75×）。
    _stab_requested = stabilize
    _stab_auto_note = ""
    if isinstance(_stab_requested, str) and _stab_requested.strip().lower() == "auto":
        _probe = detect_shakiness(
            input_path, logger,
            samples=int(params.get("shake_probe_samples", 12) or 12),
            frames_per_sample=int(params.get("shake_probe_frames", 40) or 40),
            threshold_px=float(params.get("shake_probe_threshold_px", 1.0) or 1.0))
        _stab_auto_note = _probe.get("reason", "")
        if _probe.get("needs_stabilization"):
            _stab_requested = "DepanStabilize"
        else:
            _stab_requested = ""
        if logger:
            logger.info("STAGE1_STABILIZE_DECISION",
                        f"stabilize=auto -> {'ON' if _stab_requested else 'OFF'}"
                        f" ({_stab_auto_note})",
                        samples_used=_probe.get("samples_used"),
                        mean_abs_dx=_probe.get("mean_abs_dx"),
                        mean_abs_dy=_probe.get("mean_abs_dy"),
                        p95_abs=_probe.get("p95_abs"))
    if _stab_requested is False or (
            isinstance(_stab_requested, str)
            and _stab_requested.strip().lower() in ("", "none", "off", "false")):
        _stab_requested = ""

    if _stab_requested == "DepanStabilize":
        # R4: havsfunc r33 **没有** DepanStabilize。但 depan 防抖本身是可得的：
        #   * core.mv.DepanEstimate/DepanStabilise 都在 mvtools 里（英式拼写 -ise）
        #   * havsfunc 的 haf.Stab 内部用的就是 mvtools 的 depan
        #     （r31 release note: "Stab: Use depan filters in mvtools plugin"）
        # 且 haf.Stab 签名恰为 (clp, dxmax, dymax, mirror) —— 与这里要传的
        # dxmax/dymax 完全对应，所以改用 haf.Stab 是等价且改动最小的做法。
        dxmax = max(0, min(stabilize_params.get("dxmax", 10), 20))  # FIX-P6-9
        dymax = max(0, min(stabilize_params.get("dymax", 10), 20))  # FIX-P6-9
        lines.append("# Temporal stabilization (depan via havsfunc Stab)")
        lines.append(f"clip = haf.Stab(clip, dxmax={dxmax}, dymax={dymax})")
        lines.append("")
    elif _stab_requested == "Stab":
        # FIX-W16: Provide Stab as a lightweight stabilization alternative per
        # v6.32 Part 4. DepanStabilize (mvtools) is the default and generally
        # superior, but Stab (havsfunc wrapper around core.std / internal) has
        # no motion-vector dependency and is useful when mvtools is unavailable
        # or for sources where DepanStabilize over-crops. Parameters mirror
        # haf.Stab defaults; amplify/dxmax override via stabilize_params.
        _amp = stabilize_params.get("amp", 10)
        _dxmax = max(0, min(stabilize_params.get("dxmax", 10), 20))
        lines.append("# Temporal stabilization (Stab lightweight alternative)")
        # R4: haf.Stab 签名是 (clp, dxmax=4, dymax=4, mirror=0)，**没有 amp
        # 参数**，原来的 amp= 会直接 TypeError。改为实际支持的 dxmax/dymax。
        _dymax = max(0, min(stabilize_params.get("dymax", 10), 20))
        lines.append(f"clip = haf.Stab(clip, dxmax={_dxmax}, dymax={_dymax})")
        lines.append("")

    # Color space conversion: BT.601 -> BT.709
    # FIX-W14: BT.709 source still needs range conversion if source is full range
    # per v7.32 §3.10.3 / v6.32 §7.19 ("严禁忽略 range 转换"). ORIGINAL skipped ALL
    # conversion for matrix_in=="709", leaving full-range BT.709 sources as full
    # range (output must be limited), causing black-level misalignment downstream.
    if matrix_in == "709" and color_range != "full":
        lines.append("# Source is already BT.709 limited range, no color conversion needed")
    elif matrix_in == "709" and color_range == "full":
        # BT.709 full range -> limited range (matrix identity, range conversion only)
        lines.append("# BT.709 source: range-only conversion (full -> limited)")
        lines.append(f"clip = core.resize.Bicubic(clip, "
                    f"matrix_in_s='709', matrix_s='709', "
                    f"range_in={range_in_str}, range={range_out_str})")
    else:
        lines.append("# BT.601 -> BT.709 color space conversion")
        lines.append(f"clip = core.resize.Bicubic(clip, "
                    f"matrix_in_s='{matrix_in}', matrix_s='709', "
                    f"range_in={range_in_str}, range={range_out_str})")
    lines.append("")

    # Restore black borders
    if has_crop and (crop_left > 0 or crop_right > 0 or crop_top > 0 or crop_bottom > 0):
        lines.append("# Restore black borders")
        lines.append(f"clip = core.std.AddBorders(clip, left={crop_left}, right={crop_right}, top={crop_top}, bottom={crop_bottom}, "
                    f"color=[{black_y}, {neutral_uv}, {neutral_uv}])")
        lines.append("")

    # Output
    lines.append("# Output")
    lines.append("clip.set_output()")

    return "\n".join(lines)


def _check_combed_and_fallback(vs_path: Path, params: dict, logger) -> bool:
    """Run quick _Combed diagnostic and return True if sRestore fallback needed.

    FIX-P6-7: Per v7.32 §3.10.13, VFM output must be validated for _Combed
    markers. If combed_ratio > threshold, sRestore fallback is required.

    Uses the _Combed diagnostic block embedded in the .vpy script.
    Runs a quick vspipe pass (first 300 frames) to populate sidecar JSON.
    """
    # FIX-R5-06: 这个 sidecar 是**由 vpy 脚本自己用 open() 写的**，而 vspipe
    # 会把 cwd 切到 vpy 所在目录（R55+ 默认行为）。原来传相对路径时会落到
    # `<temp>/<temp>/xxx.combed.json`（目录不存在 -> vpy 里
    # `except Exception: pass` 把异常吞掉），统计永远不落盘，
    # 于是 IVTC_COMBED_FAIL 的 sRestore 回退决策**从来没触发过**。
    # 必须传绝对路径。
    combed_stats_path = Path(os.path.abspath(str(vs_path))).with_suffix(
        ".combed.json")
    env = {**os.environ, "VS_COMBED_STATS": str(combed_stats_path)}

    # Quick diagnostic pass: request frames 0-299 to populate sidecar
    # FIX-P6-9: Write y4m stdout to /dev/null instead of capturing to memory
    # ORIGINAL: capture_output=True stored up to 100MB+ of y4m data in RAM
    try:
        with open(os.devnull, 'wb') as devnull_out:
            r_diag = subprocess.run(
                ["vspipe", str(vs_path), "-", "-s", "0", "-e", "299"],
                env=env, stdout=devnull_out, stderr=subprocess.PIPE,
                shell=False, timeout=120
            )
        # Output discarded; sidecar file is the only channel we care about
    except Exception as e:
        logger.warning("IVTC_COMBED_DIAG_FAIL",
                      f"_Combed diagnostic pass failed: {e}")
        return False

    needs_srestore = False
    if combed_stats_path.exists():
        try:
            with open(combed_stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            total = stats.get("total", 0)
            combed_count = len(stats.get("combed_frames", []))
            if total > 0:
                ratio = combed_count / total
                # Threshold: v7.32 §3.10.13 "continuous multi-frame failure"
                # 10% is conservative; normal 3:2 Pulldown _Combed < 2%
                threshold = params.get("ivtc_combed_threshold", 0.10)
                if ratio > threshold:
                    logger.warning("IVTC_COMBED_FAIL",
                                  f"VFM _Combed ratio {ratio:.1%} > threshold "
                                  f"{threshold:.1%}, sRestore fallback required",
                                  combed_frames=stats.get("combed_frames")[:20])
                    needs_srestore = True
        except Exception as e:
            logger.warning("IVTC_COMBED_READ_FAIL",
                          f"Failed to read _Combed sidecar: {e}")
        finally:
            # Clean up sidecar
            try:
                combed_stats_path.unlink()
            except Exception:
                pass

    return needs_srestore


def check_deinterlace_quality(video_path: str, logger=None) -> Tuple[bool, float]:
    """Check deinterlace output quality using ffmpeg idet."""
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-v", "info",
            "-i", video_path,
            "-vf", "idet",
            "-f", "null", "-"
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                          timeout=300, shell=False)
        if r.returncode != 0:
            return False, 0.0

        stderr = r.stderr
        stats = {}
        for line in stderr.splitlines():
            if "Multi frame detection:" in line:
                for key in ("TFF", "BFF", "Progressive", "Undetermined"):
                    m = re.search(rf"{key}:\s*(\d+)", line)
                    if m:
                        stats[key.lower()] = int(m.group(1))

        total = sum(stats.values())
        if total == 0:
            return False, 0.0

        prog_pct = stats.get("progressive", 0) / total * 100
        is_good = prog_pct >= 95.0
        if logger:
            logger.info("STAGE1_DEINT_CHECK",
                       f"Deinterlace quality: Progressive={prog_pct:.1f}%")
        return is_good, prog_pct
    except Exception as e:
        if logger:
            logger.warning("STAGE1_DEINT_CHECK_FAIL", f"Failed: {e}")
        return False, 0.0


def check_ivtc_quality(video_path: str, logger=None,
                       ivtc_fallback_threshold: Optional[dict] = None) -> Tuple[bool, Dict[str, int]]:
    """Check IVTC output quality using ffmpeg idet.

    Uses ivtc_fallback_threshold config for decision:
    - tff_bff_percent: if TFF+BFF > threshold, IVTC failed
    - progressive_percent: must be >= threshold for success
    """
    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-v", "info",
            "-i", video_path,
            "-vf", "idet",
            "-f", "null", "-"
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                          timeout=300, shell=False)
        if r.returncode != 0:
            return False, {}

        stderr = r.stderr
        stats = {}
        for line in stderr.splitlines():
            if "Multi frame detection:" in line:
                for key in ("TFF", "BFF", "Progressive", "Undetermined"):
                    m = re.search(rf"{key}:\s*(\d+)", line)
                    if m:
                        stats[key.lower()] = int(m.group(1))

        total = sum(stats.values())
        if total == 0:
            return False, stats

        tff_pct = stats.get("tff", 0) / total * 100
        bff_pct = stats.get("bff", 0) / total * 100
        prog_pct = stats.get("progressive", 0) / total * 100

        # Use config thresholds if available
        if ivtc_fallback_threshold:
            tff_bff_limit = ivtc_fallback_threshold.get("tff_bff_percent", 10.0)
            prog_min = ivtc_fallback_threshold.get("progressive_percent", 90.0)
            is_good = (tff_pct + bff_pct) <= tff_bff_limit and prog_pct >= prog_min
        else:
            is_good = (tff_pct + bff_pct) <= 10.0 and prog_pct >= 90.0

        if logger:
            logger.info("STAGE1_IVTC_CHECK",
                       f"TFF={tff_pct:.1f}% BFF={bff_pct:.1f}% Progressive={prog_pct:.1f}%")
        return is_good, stats
    except Exception as e:
        if logger:
            logger.warning("STAGE1_IVTC_CHECK_FAIL", f"Failed: {e}")
        return False, {}


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
    parser.add_argument("--no-wsl-redirect", action="store_true")
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()

    params = _load_params(args)
    log_dir = params.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, job_id=args.job_id)

    # NEW-V-06: 注册取消守卫（video main.py 认 rc=88 = EXIT_INTERRUPTED）。
    if _cancel_guard is not None:
        _cancel_guard.install_cancel_guard(stage=1, logger=logger)

    temp_dir = params.get("temp_dir", "./temp")
    # FIX-P6-5: WSL2 defensive check per v7.32 §3.10.10
    # main.py handles primary WSL2 redirect; this is a defensive fallback
    if temp_dir.startswith("/mnt/") and not params.get("no_wsl_redirect"):
        logger.warning("STAGE1_WSL_UNREDIRECTED",
                      f"temp_dir still on /mnt/: {temp_dir}. "
                      "Expected main.py to redirect before calling stage.")
        try:
            from checkpoint import wsl_redirect_path
            redirected = wsl_redirect_path(temp_dir)
            if isinstance(redirected, tuple):
                redirected = redirected[0]
            if redirected != temp_dir:
                logger.info("STAGE1_WSL_REDIRECT",
                           f"temp_dir redirected: {temp_dir} -> {redirected}")
                temp_dir = redirected
        except ImportError:
            pass
    # FIX-P6-10: Use pathlib.Path consistently per project convention
    # ORIGINAL: os.makedirs / os.path.* mixed usage
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    input_path = args.input
    input_name = Path(input_path).name

    # Check for existing intermediate
    param_hash = compute_param_hash(input_path, params)
    inter_name = build_intermediate_name(
        args.job_id, input_name, 1, "preprocess", param_hash, "mkv"
    )
    inter_path = str(Path(temp_dir) / inter_name)

    # FIX-P6-10: pathlib consistency
    if Path(inter_path).exists() and Path(inter_path).stat().st_size > 0:
        logger.info("STAGE1_SKIP", "Intermediate exists", path=inter_path)
        return 3

    # NEW-V-06: 渲染开始前登记产物——取消时删掉半截 mkv，
    # 否则上面的 exists()>0 缓存检查会把它当成功产物复用。
    if _cancel_guard is not None:
        _cancel_guard.watch_outputs(inter_path, logger=logger, stage=1)

    logger.info("STAGE1_START", "Traditional video preprocessing", input=input_path)

    # Detect black borders
    crop_info = detect_black_borders(input_path, logger)
    if not validate_crop(crop_info, logger):
        # v6.3 §7.19 / v7.32 §3.10.3: invalid crop must mark FAILED, not fallback to full frame
        logger.error("STAGE1_CROP_FAILED", "Black border crop validation failed, aborting")
        return 81

    # Generate VapourSynth script
    vs_script = generate_vs_script(input_path, params, args.job_id, crop_info)

    # FIX-P6-3: Log WARNING when color_range cannot be detected per v6.32 §7.19
    if params.get("_color_range_unknown"):
        logger.warning("STAGE1_COLOR_RANGE_UNKNOWN",
                      "Cannot detect color_range from ffprobe, "
                      "defaulting to limited per v6.32 §7.19")
        params.pop("_color_range_unknown", None)

    # FIX-W10: log when vs_bit_depth was upgraded to 16 per v7.32 §3.10.1
    if params.get("_vs_bit_depth_upgraded"):
        _req_bd, _act_bd = params["_vs_bit_depth_upgraded"]
        logger.warning("STAGE1_BITDEPTH_UPGRADED",
                      f"vs_bit_depth {_req_bd} < 16, upgraded to {_act_bd} "
                      "per v7.32 §3.10.1 (16bit processing space mandatory)")
        params.pop("_vs_bit_depth_upgraded", None)

    # FIX-W15: log when qtgmc_input_type was forced to 0 for tt/bb per v7.32 §3.10.1
    if params.get("_qtgmc_input_type_override"):
        _req_it, _act_it = params["_qtgmc_input_type_override"]
        logger.warning("STAGE1_INPUTTYPE_OVERRIDE",
                      f"qtgmc_input_type {_req_it} invalid for tt/bb true interlaced, "
                      f"forced to {_act_it} per v7.32 §3.10.1")
        params.pop("_qtgmc_input_type_override", None)

    # FIX-P6-8: Log warning for unknown field_order
    if params.get("field_order") == "unknown":
        logger.warning("STAGE1_UNKNOWN_FIELD_ORDER",
                      "field_order=unknown, skipping QTGMC deinterlace (conservative)")

    # ISSUE-039 fix: hard validation of IVTC-before-QTGMC ordering in generated script
    needs_ivtc = params.get("needs_ivtc", False)
    if needs_ivtc:
        ivtc_pos = vs_script.find("core.vivtc.VFM")
        if ivtc_pos == -1:
            ivtc_pos = vs_script.find("haf.srestore")
        qtgmc_pos = vs_script.find("haf.QTGMC")
        if qtgmc_pos != -1 and ivtc_pos != -1 and qtgmc_pos < ivtc_pos:
            logger.error("STAGE1_ORDER_FAIL",
                        "QTGMC appears before IVTC in generated script; violates v6.3 hard constraint")
            return 81

    vs_path = str(Path(temp_dir) / f"{args.job_id}_{input_name}.vpy")
    with open(vs_path, "w", encoding="utf-8") as f:
        f.write(vs_script)

    # V-P-020 fix: record SHA-256 hash of generated script
    import hashlib
    script_hash = hashlib.sha256(vs_script.encode("utf-8")).hexdigest()
    logger.info("STAGE1_VPY", f"VS script written: {vs_path}", script_sha256=script_hash)

    # FIX-P6-7: _Combed pre-check for VIVTC path per v7.32 §3.10.13
    # Run quick diagnostic before main render; if combed_ratio > threshold,
    # return 86 so main.py can re-run with sRestore
    ivtc_method = params.get("ivtc", "VIVTC")
    if needs_ivtc and ivtc_method == "VIVTC" and params.get("ivtc_auto_fallback", True):
        if _check_combed_and_fallback(Path(vs_path), params, logger):
            logger.warning("IVTC_COMBED_PRECHECK",
                          "VFM _Combed pre-check failed, requesting sRestore fallback")
            return 86

    # Render with vspipe -> ffmpeg
    try:
        # R4（关键修复）：**`--y4m` 这个参数根本不存在** —— 实测
        # VSPipe R72 与 R81 的 help 都是 `-c, --container <y4m/wav/w64/mkv>`。
        # 原写法让 vspipe 直接 "Unknown argument: --y4m" 退出 1，
        # 无任何 y4m 输出，ffmpeg 读 "-" 拿到无效数据 -> exit 84。
        vspipe_cmd = ["vspipe", "-c", "y4m", vs_path, "-"]
        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-v", "error",
            "-thread_queue_size", "1024",
            "-i", "-",
            "-c:v", "ffv1",
            "-an", "-y", inter_path
        ]

        # R4（关键修复）：原来给 vspipe 的 stderr 开了 subprocess.PIPE
        # 却**从未读取**它。vspipe 会持续往 stderr 写警告（本次实测每条
        # Cache/转义警告都会累积），管道缓冲区（约 64KB）一满 vspipe 就阻塞，
        # 不再向 stdout 输出 —— 于是 ffmpeg 从 "-" 读到无效数据：
        #   "Error opening input file -. Invalid data found when processing input"
        # 表现为 stage1 渲染失败 exit 84，而**单独跑 vspipe 却是成功的**，
        # 极容易误判成 VS 脚本有问题。
        # 修法：把 stderr 重定向到日志文件，既不阻塞又保留诊断信息。
        _vs_pipe_log = Path(vs_path).with_suffix(".vspipe.log")
        try:
            _vp_err = open(_vs_pipe_log, "wb")
        except Exception:
            _vp_err = subprocess.DEVNULL
        vp = subprocess.Popen(vspipe_cmd, stdout=subprocess.PIPE,
                             stderr=_vp_err, shell=False)
        ff = subprocess.Popen(ffmpeg_cmd, stdin=vp.stdout,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, shell=False)
        vp.stdout.close()
        # NEW-V-06/RV-12: 两个子进程都必须登记——取消时只杀 Python  stage
        # 的话，vspipe/ffmpeg 会继续把半截数据写回 inter_path（守卫刚删掉
        # 的文件被孤儿进程重新创建，取消就白做了）。
        if _cancel_guard is not None:
            _cancel_guard.watch_process(vp)
            _cancel_guard.watch_process(ff)
        try:
            _, stderr = ff.communicate(timeout=7200)
        finally:
            if _cancel_guard is not None:
                _cancel_guard.unwatch_process(vp)
                _cancel_guard.unwatch_process(ff)
        _vp_rc = vp.wait(timeout=60)
        try:
            if _vp_err is not subprocess.DEVNULL:
                _vp_err.close()
        except Exception:
            pass
        if _vp_rc != 0 and logger:
            logger.warning("VSPIPE_RC",
                           f"vspipe exited {_vp_rc}; see {_vs_pipe_log}")

        if ff.returncode == 0 and Path(inter_path).exists():
            # V-P-022 fix: IVTC auto-fallback check
            needs_ivtc = params.get("needs_ivtc", False)
            ivtc = params.get("ivtc", "VIVTC")
            ivtc_auto_fallback = params.get("ivtc_auto_fallback", True)
            if needs_ivtc and ivtc == "VIVTC" and ivtc_auto_fallback:
                ivtc_threshold = params.get("ivtc_fallback_threshold", {})
                is_good, stats = check_ivtc_quality(inter_path, logger, ivtc_threshold)
                if not is_good:
                    # FIX-W9: use the canonical IVTC_AUTO_FALLBACK event name per
                    # v7.32 §3.10.13 / v6.32 §7.13 ("日志记录 IVTC_AUTO_FALLBACK 事件,
                    # 包含残留率与回退原因"). ORIGINAL used STAGE1_IVTC_FALLBACK, which
                    # audit filters for IVTC_AUTO_FALLBACK would miss (the pre-check
                    # path already used IVTC_AUTO_FALLBACK via main.py).
                    _tff_bff_residual = stats.get("tff", 0) + stats.get("bff", 0)
                    _ivtc_total = sum(stats.values()) or 1
                    logger.warning("IVTC_AUTO_FALLBACK",
                                  "IVTC quality check failed, regenerating with sRestore",
                                  tff_bff_residual_pct=round(_tff_bff_residual / _ivtc_total * 100, 1),
                                  progressive_pct=round(stats.get("progressive", 0) / _ivtc_total * 100, 1),
                                  reason="VFM output idet TFF+BFF>10% or Progressive<90%")
                    # Regenerate with sRestore
                    srestore_params = dict(params)
                    srestore_params["ivtc"] = "sRestore"
                    vs_script_srestore = generate_vs_script(input_path, srestore_params, args.job_id, crop_info)
                    vs_path_srestore = str(Path(temp_dir) / f"{args.job_id}_{input_name}_srestore.vpy")
                    with open(vs_path_srestore, "w", encoding="utf-8") as f:
                        f.write(vs_script_srestore)
                    script_hash_sr = hashlib.sha256(vs_script_srestore.encode("utf-8")).hexdigest()
                    logger.info("STAGE1_VPY_SRESTORE", f"sRestore script written: {vs_path_srestore}", script_sha256=script_hash_sr)

                    # FIX-P6-4: Render sRestore to temporary file first (atomic safety)
                    # ORIGINAL: Rendered directly to inter_path, corrupting VIVTC output on failure
                    srestore_inter_path = inter_path + ".srestore.tmp"
                    ffmpeg_cmd_srestore = [
                        "ffmpeg", "-hide_banner", "-v", "error",
                        "-thread_queue_size", "1024",
                        "-i", "-",
                        "-c:v", "ffv1",
                        "-an", "-y", srestore_inter_path
                    ]
                    # R4: 同上，`--y4m` 不存在，应为 `-c y4m`
                    vp2 = subprocess.Popen(["vspipe", "-c", "y4m", vs_path_srestore, "-"],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
                    ff2 = subprocess.Popen(ffmpeg_cmd_srestore, stdin=vp2.stdout,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
                    vp2.stdout.close()
                    _, stderr2 = ff2.communicate(timeout=7200)
                    if ff2.returncode == 0 and Path(srestore_inter_path).exists():
                        # Atomic replace: only overwrite original on success
                        import shutil
                        try:
                            shutil.move(srestore_inter_path, inter_path)
                        except OSError:
                            # Cross-filesystem fallback
                            shutil.copyfile(srestore_inter_path, inter_path)
                            try:
                                os.unlink(srestore_inter_path)
                            except Exception:
                                pass
                        # V-P-022: verify sRestore frame rate
                        try:
                            fr_cmd = ["ffprobe", "-hide_banner", "-v", "error",
                                      "-select_streams", "v:0",
                                      "-show_entries", "stream=r_frame_rate",
                                      "-of", "default=noprint_wrappers=1:nokey=1", inter_path]
                            fr_r = subprocess.run(fr_cmd, capture_output=True, text=True, errors="replace", timeout=30, shell=False)
                            if fr_r.returncode == 0:
                                rate_str = fr_r.stdout.strip()
                                if "/" in rate_str:
                                    num, den = map(float, rate_str.split("/"))
                                    detected_fps = num / den if den != 0 else 0
                                else:
                                    detected_fps = float(rate_str)
                                expected_fps = 23.976
                                if abs(detected_fps - expected_fps) / expected_fps > 0.01:
                                    logger.error("STAGE1_IVTC_FRAMERATE",
                                                f"sRestore output fps {detected_fps} deviates from expected {expected_fps}")
                                    return 86
                        except Exception as e:
                            logger.warning("STAGE1_FRAMERATE_CHECK", f"Frame rate check failed: {e}")
                        logger.info("STAGE1_SRESTORE_DONE", "sRestore IVTC fallback complete", output=inter_path)
                    else:
                        # Clean up failed sRestore temp file
                        try:
                            if Path(srestore_inter_path).exists():
                                Path(srestore_inter_path).unlink()
                        except Exception:
                            pass
                        err_msg = stderr2.decode()[:300] if stderr2 else "Unknown error"
                        logger.error("STAGE1_SRESTORE_FAIL", f"sRestore render error: {err_msg}")
                        return 84

            # V-P-023 fix: deinterlace quality check for segmented field order
            field_segments = params.get("field_segments", [])
            if field_segments:
                is_good, prog_pct = check_deinterlace_quality(inter_path, logger)
                if not is_good:
                    logger.warning("STAGE1_DEINT_FALLBACK",
                                  f"Deinterlace quality {prog_pct:.1f}% < 95%, progressive ratio too low")
                    # FIX-W12: Per v7.32 §3.10.13, deinterlace quality <95% must
                    # trigger a conservative fallback (return 86 so main.py can
                    # re-run with global conservative processing / drop
                    # field_segments), not merely log a warning. ORIGINAL only
                    # logged WARNING and continued, producing a low-quality
                    # interlace artifact as if it were a clean progressive output.
                    return 86

            # V-P-024 fix: PSF frame rate verification
            is_psf = params.get("is_psf", False)
            if is_psf:
                try:
                    fr_cmd = ["ffprobe", "-hide_banner", "-v", "error",
                              "-select_streams", "v:0",
                              "-show_entries", "stream=r_frame_rate",
                              "-of", "default=noprint_wrappers=1:nokey=1", inter_path]
                    fr_r = subprocess.run(fr_cmd, capture_output=True, text=True, errors="replace", timeout=30, shell=False)
                    if fr_r.returncode == 0:
                        rate_str = fr_r.stdout.strip()
                        if "/" in rate_str:
                            num, den = map(float, rate_str.split("/"))
                            detected_fps = num / den if den != 0 else 0
                        else:
                            detected_fps = float(rate_str)
                        # FIX-P6-6: Also check "fps" key from classify output per v7.32 §3.10.1
                        # ORIGINAL: Only checked "original_fps" which is never set by main.py
                        original_fps = params.get("original_fps", params.get("fps", 0))
                        if original_fps > 0 and abs(detected_fps - original_fps) > 0.5:
                            logger.error("STAGE1_PSF_FRAMERATE",
                                        f"PSF output fps {detected_fps} differs from original {original_fps}")
                            return 86
                except Exception as e:
                    logger.warning("STAGE1_PSF_FRAMERATE_CHECK", f"PSF frame rate check failed: {e}")

            # FIX-P6-12: --cleanup-intermediates removes VS script per v7.32 §3.10.17
            if args.cleanup_intermediates:
                try:
                    vs_path_obj = Path(vs_path)
                    if vs_path_obj.exists():
                        vs_path_obj.unlink()
                        logger.info("STAGE1_CLEANUP", f"Removed VS script: {vs_path}")
                    # Also remove sRestore script if it exists
                    srestore_vs = Path(f"{vs_path_obj.stem}_srestore{vs_path_obj.suffix}")
                    if srestore_vs.exists():
                        srestore_vs.unlink()
                        logger.info("STAGE1_CLEANUP", f"Removed sRestore VS script: {srestore_vs}")
                except Exception as e:
                    logger.warning("STAGE1_CLEANUP_FAIL", f"Failed to cleanup: {e}")

            logger.info("STAGE1_DONE", "Preprocessing complete", output=inter_path)
            # NEW-V-06: 产物已完整落盘，取消守卫不再有权删它
            # （RV-12：发布是不可逆的成功事件）。
            if _cancel_guard is not None:
                _cancel_guard.release_outputs()
            return 0
        else:
            err_msg = stderr.decode()[:300] if stderr else "Unknown error"
            logger.error("STAGE1_FAILED", f"Render error: {err_msg}")
            return 84
    except subprocess.TimeoutExpired:
        logger.error("STAGE1_TIMEOUT", "Preprocessing timed out")
        return 87
    except Exception as e:
        logger.error("STAGE1_ERROR", str(e))
        return 82


if __name__ == "__main__":
    sys.exit(main() or 0)
