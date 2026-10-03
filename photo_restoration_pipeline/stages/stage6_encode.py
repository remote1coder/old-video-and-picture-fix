#!/usr/bin/env python3
"""
Stage 6: Photo Final Encoding
- AVIF encoding with avifenc (10bit YUV420 or RGB444)
- CICP metadata
- Archival mode: RGB444 lossless when archival_mode=true

v6.3/v7.3 Compliance:
- Supports both avif_params (YUV420 10-bit) and avif_rgb444_params (RGB444)
- Grain synthesis moved to stage5_postprocess.py per v7.32 §3.1
- Proper CICP tagging

FIX-V01/V02/V04/V05 (2026-10-02): the encode boundary now follows the
documented avifenc contract instead of assumptions:
  * input must be jpg/jpeg/png/y4m — TIFF intermediates are converted
    losslessly (16-bit preserved) in an explicit adapter (V05);
  * `--depth` only accepts 8/10/12; a 16-bit result requires the documented
    depth extension (e.g. `12,8`) on libavif >= 1.4.0 (V01);
  * the TIFF master is written at the *actual* pixel precision and only ever
    carries an ICC profile that matches the pixels it labels (V02);
  * wide-gamut conversion needs a real source profile and fails loudly instead
    of warning and continuing (V04).
Every run writes an `*.encode_contract.json` sidecar describing what was
actually encoded — not what was requested.
"""

import os
import sys
import json
import argparse
import glob
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from checkpoint import wsl_redirect_path, wsl_copy_back  # noqa: E402

try:  # FIX-V01/V05: encode-boundary contracts
    sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
    import media_io  # noqa: E402
except Exception:  # pragma: no cover - defensive; stage must fail loudly later
    media_io = None  # type: ignore[assignment]

try:  # NEW-V-06: 取消/中断守卫（except Exception 抓不到 KeyboardInterrupt）
    import cancel_guard  # noqa: E402
except Exception:  # pragma: no cover - 缺失时不阻断 stage，仅失去该保护
    cancel_guard = None  # type: ignore[assignment]


# FIX-V04: colour-space handling outcomes
COLOR_NOT_NEEDED = "not-needed"      # source already sRGB / no conversion required
COLOR_CONVERTED = "converted"        # real profile → real pixel transform → sRGB
COLOR_UNKNOWN = "unknown-source"     # source space not established
COLOR_DEGRADED = "degraded"          # conversion impossible, explicitly accepted
COLOR_FAILED = "failed"              # required conversion impossible → stage fails

WIDE_GAMUT_SPACES = ("adobergb", "prophoto", "displayp3", "p3")


def _get_avifenc_version() -> tuple:
    """Parse avifenc version. Returns (major, minor, patch) or (0, 0, 0)."""
    if media_io is not None:
        cap = media_io.detect_avifenc_capability()
        return cap.version
    return (0, 0, 0)


def _resolve_jobs_value(jobs: str) -> Optional[str]:
    """Resolve --jobs value based on avifenc version (FIX-P12-10)."""
    if media_io is not None:
        return media_io.resolve_jobs_value(jobs, media_io.detect_avifenc_capability())
    if jobs != "all":
        return str(jobs)
    version = _get_avifenc_version()
    if version >= (1, 1, 0):
        return "all"
    elif version >= (0, 9, 2):
        return str(min(os.cpu_count() or 1, 32))
    return None


def _build_avifenc_cmd(input_path: str, output_path: str, params: dict,
                       use_rgb444: bool = False) -> list:
    """Build a validated avifenc command from config parameters.

    FIX-V01: raises media_io.UnsupportedAvifConfig instead of emitting a bare
    ``--depth 16`` that no avifenc version accepts.
    """
    if media_io is None:
        raise RuntimeError("media_io module unavailable")
    cmd, _contract = media_io.build_avifenc_cmd(
        input_path, output_path, params, use_rgb444=use_rgb444)
    return cmd


def _calculate_avifenc_timeout(input_path: str, base_timeout: int = 600) -> int:
    """BUG-057 / ISSUE-P-S6-003: Dynamic timeout based on image pixel count."""
    try:
        from PIL import Image
        with Image.open(input_path) as img:
            w, h = img.size
        mp = (w * h) / 1_000_000
        dynamic = int(mp * 60)
        return max(base_timeout, dynamic)
    except Exception:
        return base_timeout


def run_avifenc(input_path: str, output_path: str, params: dict,
                use_rgb444: bool = False) -> Tuple[bool, str]:
    """Run avifenc. Returns (ok, stderr)."""
    cmd = _build_avifenc_cmd(input_path, output_path, params, use_rgb444)
    timeout = _calculate_avifenc_timeout(input_path)
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, shell=False)
        return r.returncode == 0, (r.stderr or b"").decode("utf-8", "replace")
    except Exception as e:
        return False, str(e)


# --- FIX-V04: colour contract ----------------------------------------------

def _read_color_meta(temp_dir: str, job_id: str, stem: str) -> Dict[str, Any]:
    """Read the stage1 colour sidecar if present."""
    path = Path(temp_dir) / f"{job_id}_{stem}_color_meta.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_source_icc(working_path: str) -> Optional[bytes]:
    """Return the ICC profile actually embedded in the working image."""
    try:
        from PIL import Image
        with Image.open(working_path) as im:
            return im.info.get("icc_profile")
    except Exception:
        return None


def _describe_profile(icc_bytes: bytes) -> str:
    try:
        import io
        from PIL import ImageCms
        prof = ImageCms.getOpenProfile(io.BytesIO(icc_bytes))
        return str(ImageCms.getProfileDescription(prof))
    except Exception:
        return ""


def resolve_color_space(working_path: str, params: Dict[str, Any],
                        temp_dir: str, job_id: str, stem: str,
                        logger) -> Dict[str, Any]:
    """Establish (and enforce) the colour contract before encoding (FIX-V04).

    ORIGINAL called ``ImageCms.createProfile("ADBE"/"ROMM")`` — Pillow only
    supports on-the-fly creation for LAB/XYZ/sRGB, so both calls raised and the
    code only warned before encoding and reporting success. Verified on this
    machine with Pillow 10.4.0 and 12.3.0: ``PyCMSError``.

    Returns a dict with keys: status, working_path, source_space, icc_path,
    cicp_override, precision, reason.
    """
    result: Dict[str, Any] = {
        "status": COLOR_NOT_NEEDED,
        "working_path": working_path,
        "source_space": "unknown",
        "icc_path": None,
        "cicp_override": None,
        "precision": None,
        "reason": None,
        "profile_description": None,
    }
    color_meta = _read_color_meta(temp_dir, job_id, stem)
    declared = str(params.get("input_color_space")
                   or color_meta.get("input_color_space") or "").lower()
    if not declared:
        # No sidecar/override: fall back to the profile embedded in the file.
        icc = _extract_source_icc(working_path)
        if icc:
            desc = _describe_profile(icc).lower()
            result["profile_description"] = _describe_profile(icc) or None
            if "adobe" in desc:
                declared = "adobergb"
            elif "prophoto" in desc or "romm" in desc:
                declared = "prophoto"
            elif "display p3" in desc or "displayp3" in desc:
                declared = "displayp3"
            elif "srgb" in desc or "iec61966" in desc:
                declared = "srgb"
            else:
                declared = "unknown"
        else:
            declared = "unknown"
    result["source_space"] = declared or "unknown"

    if result["source_space"] not in WIDE_GAMUT_SPACES:
        if result["source_space"] == "unknown":
            # FIX-V04: 未知源色彩不猜，也不假装是 sRGB
            result["status"] = COLOR_UNKNOWN
            result["reason"] = ("source colour space not established; no ICC "
                                "in file and no input_color_space metadata")
            logger.warning("STAGE6_COLOR_UNKNOWN",
                           "Source colour space unknown; encoding with the "
                           "configured CICP and recording it as unverified. "
                           "Set require_known_color_space=true to fail instead.")
            if params.get("require_known_color_space", False):
                result["status"] = COLOR_FAILED
                result["reason"] += " (require_known_color_space=true)"
                logger.error("STAGE6_COLOR_FAILED",
                             "Unknown source colour space is not accepted "
                             "(require_known_color_space=true)")
        return result

    # --- wide-gamut source: a real transform is mandatory -------------------
    allow_degrade = bool(params.get("allow_unconverted_colorspace", False))
    icc_bytes = _extract_source_icc(working_path)
    source_icc_file: Optional[str] = None
    src_profile = None
    try:
        from PIL import ImageCms
        import io as _io
        if icc_bytes:
            source_icc_file = str(Path(temp_dir) / f"{job_id}_{stem}_source.icc")
            with open(source_icc_file, "wb") as f:
                f.write(icc_bytes)
            src_profile = ImageCms.getOpenProfile(source_icc_file)
            result["profile_description"] = _describe_profile(icc_bytes) or None
        elif params.get("source_icc_path"):
            source_icc_file = str(params["source_icc_path"])
            src_profile = ImageCms.getOpenProfile(source_icc_file)
            result["profile_description"] = _describe_profile(
                Path(source_icc_file).read_bytes()) or None
    except Exception as e:
        src_profile = None
        result["reason"] = f"source ICC could not be opened: {e}"

    if src_profile is None:
        result["icc_path"] = source_icc_file
        if not allow_degrade:
            result["status"] = COLOR_FAILED
            result["reason"] = (result["reason"] or "no usable source ICC profile")
            logger.error("STAGE6_COLOR_FAILED",
                         f"{result['source_space']} source needs a real ICC "
                         "profile for conversion; none available. Refusing to "
                         "encode pixels that would be mislabelled as sRGB.")
            return result
        result["status"] = COLOR_DEGRADED
        result["cicp_override"] = "2/2/0"  # primaries+transfer unspecified
        result["reason"] = (result["reason"] or "no usable source ICC profile")
        logger.warning("STAGE6_COLOR_DEGRADED",
                       "Encoding without colour conversion; CICP set to "
                       "unspecified primaries/transfer so the sRGB label is "
                       "not claimed",
                       source_space=result["source_space"])
        return result

    try:
        from PIL import Image as _PILImg, ImageCms as _ImageCms
        img = _PILImg.open(working_path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        if img.mode == "L":
            img = img.convert("RGB")
        # Pillow's C-API conversion works on 8-bit data. Honest recording:
        result["precision"] = "8-bit"
        if params.get("require_16bit_color_conversion", False):
            result["status"] = COLOR_FAILED
            result["reason"] = "16-bit colour conversion not available via Pillow"
            logger.error("STAGE6_COLOR_FAILED",
                         "require_16bit_color_conversion=true but only 8-bit "
                         "CMS conversion is available")
            return result
        srgb_profile = _ImageCms.createProfile("sRGB")  # supported on-the-fly
        converted = _ImageCms.profileToProfile(img, src_profile, srgb_profile,
                                               outputMode="RGB")
        conv_path = str(Path(temp_dir) / f"{job_id}_{stem}_srgb.png")
        converted.save(conv_path, format="PNG")
        result["status"] = COLOR_CONVERTED
        result["working_path"] = conv_path
        result["cicp_override"] = None  # real sRGB pixels → configured CICP ok
        result["icc_path"] = None
        logger.info("STAGE6_COLOR_CONVERTED",
                    f"Converted {result['source_space']} -> sRGB with a real "
                    "ICC transform (8-bit CMS precision)",
                    source_profile=result["profile_description"])
        return result
    except Exception as e:
        if not allow_degrade:
            result["status"] = COLOR_FAILED
            result["reason"] = f"colour conversion failed: {e}"
            logger.error("STAGE6_COLOR_FAILED",
                         f"Mandatory colour conversion failed: {e}")
            return result
        result["status"] = COLOR_DEGRADED
        result["cicp_override"] = "2/2/0"
        result["icc_path"] = source_icc_file
        result["reason"] = f"colour conversion failed: {e}"
        logger.warning("STAGE6_COLOR_DEGRADED",
                       "Colour conversion failed; degraded encode accepted "
                       "with unspecified CICP + source ICC attached",
                       reason=str(e))
        return result


# --- FIX-V02: archival TIFF master -----------------------------------------

def effective_bit_depth(arr) -> int:
    """Estimate how many bits of *information* the samples really carry.

    NEW-V-03: 一个 uint16 TIFF 只说明"容器 16-bit"，不说明内容有 16-bit。
    典型伪装：上游把 8-bit 数据乘 257 塞进 uint16（stage5 的旧行为），
    此时所有样本都是 257 的倍数，真实信息量仍是 8 bit。
    做法：对采样值取最大公约数 g，则可区分电平数 = 2^16 / g，
    有效位深 = ceil(log2(电平数))。
    """
    import numpy as np
    a = np.asarray(arr)
    if a.dtype != np.uint16:
        return 8
    flat = a.reshape(-1)
    if flat.size == 0:
        return 0
    if flat.size > 200000:  # 大图抽样，够用且快
        idx = np.linspace(0, flat.size - 1, 200000).astype(np.int64)
        flat = flat[idx]
    g = int(np.gcd.reduce(flat.astype(np.uint32)))
    if g <= 0:
        return 0
    levels = max(1, 65536 // g)
    bits = int(np.ceil(np.log2(levels)))
    return max(1, min(16, bits))


def write_archival_master(working_path: str, tiff_path: str,
                          params: Dict[str, Any], logger) -> Dict[str, Any]:
    """Write the TIFF master at its *real* precision (FIX-V02).

    ORIGINAL: ``img.convert("RGB")`` then ``save(tiff)`` while logging "16bit
    TIFF"; the resulting file has BitsPerSample (8,8,8) — verified on this
    machine. It also attached an Adobe RGB / ProPhoto profile found by scanning
    the system colour directory, i.e. a label that does not match the pixels.
    """
    info: Dict[str, Any] = {
        "written": False,
        "requested_bit_depth": params.get("photo_bit_depth", 16),
        "actual_bit_depth": None,
        "precision_loss": False,
        "icc": "none",
        "writer": None,
        "dtype": None,
        "error": None,
    }
    if media_io is None:
        info["error"] = "media_io unavailable"
        return info
    try:
        import numpy as np
        arr, meta = media_io.read_image_array(working_path)
        arr = media_io._ensure_uint(arr)
        info["dtype"] = str(arr.dtype)
        info["actual_bit_depth"] = 16 if arr.dtype == np.uint16 else 8
        info["precision_loss"] = (info["requested_bit_depth"] == 16 and
                                  info["actual_bit_depth"] == 8)
        # NEW-V-03: 容器是 16-bit 不代表内容有 16-bit 信息量。
        # 上游若把 8-bit 数据乘 257 塞进 uint16（stage5 的老行为），
        # 所有样本都会是 257 的倍数 —— 有效位深其实只有 8 bit。
        info["effective_bit_depth"] = effective_bit_depth(arr)
        info["container_mismatch"] = (info["actual_bit_depth"] == 16 and
                                      info["effective_bit_depth"] <= 8)
        writer = media_io.write_tiff(tiff_path, arr)
        info["writer"] = writer
        info["written"] = True
        # Only an ICC that describes these very pixels may be attached.
        icc_bytes = meta.get("icc_profile")
        if icc_bytes:
            info["icc"] = ("source" if media_io.embed_icc_in_tiff(tiff_path, icc_bytes)
                           else "embed-failed")
        if info["precision_loss"]:
            logger.warning("STAGE6_TIFF_PRECISION_LOSS",
                           f"Declared {info['requested_bit_depth']}-bit master but "
                           f"the working image is {info['actual_bit_depth']}-bit; "
                           "the master is written at the real precision and the "
                           "loss is recorded (upstream stage dropped to 8-bit)",
                           dtype=info["dtype"])
        if info["container_mismatch"]:
            logger.warning(
                "STAGE6_TIFF_CONTAINER_MISMATCH",
                "Master is stored in a 16-bit container but carries only "
                f"{info['effective_bit_depth']}-bit of information (upstream "
                "upscaled 8-bit data into uint16); the master is kept but the "
                "real precision is recorded instead of being implied")
        if info["icc"] == "none":
            logger.info("STAGE6_TIFF_NO_SOURCE_ICC",
                        "No source ICC in the working image; TIFF master "
                        "carries no profile rather than a guessed one")
        logger.info("STAGE6_TIFF_MASTER",
                    f"TIFF master written ({info['actual_bit_depth']}-bit, "
                    f"writer={writer}, icc={info['icc']})",
                    output=tiff_path)
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        logger.warning("STAGE6_TIFF_FAIL", f"TIFF master failed: {e}")
    return info


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

    # NEW-V-06: 注册取消守卫（KeyboardInterrupt 继承 BaseException）。
    if cancel_guard is not None:
        cancel_guard.install_cancel_guard(stage=6, logger=logger)

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    input_name = Path(args.input).name

    # Bug 11.21 fix: pre-check avifenc existence
    if not shutil.which("avifenc"):
        logger.error("STAGE6_AVIFENC_MISSING", "avifenc not found in PATH")
        return 80
    if media_io is None:
        logger.error("STAGE6_MEDIA_IO_MISSING",
                     "shared_modules/media_io.py unavailable; refusing to "
                     "encode with unverified parameters")
        return 80

    capability = media_io.detect_avifenc_capability()
    logger.info("STAGE6_AVIFENC_CAPABILITY",
                f"avifenc version={'.'.join(map(str, capability.version))} "
                f"depth_extension={capability.supports_depth_extension}")

    # FIX-L12-16: WSL2 redirect for temp_dir per v7.32 §3.10.10
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    _platform = params.get("platform", "auto")
    try:
        _redir_temp, _orig_temp, _, _ = wsl_redirect_path(
            temp_dir, no_wsl_redirect=no_wsl_redirect,
            target_platform=_platform
        )
        if _redir_temp != temp_dir:
            logger.info("STAGE6_WSL_REDIRECT_TEMP",
                        f"temp_dir: {temp_dir} -> {_redir_temp}")
            temp_dir = _redir_temp
            params["temp_dir"] = temp_dir
    except Exception:
        pass  # wsl_redirect_path may be unavailable on native Windows

    # FIX-P10-1: input search order step5 > step4 > step3 > step2 > step1
    input_source = None
    for pat in ["step5_postprocess*", "step4_face*", "step3_upscale*", "step2_ai_pre*", "step1_preprocess*"]:
        pattern = f"{job_id}_{Path(input_name).stem}_{pat}"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        # NEW-V-04: 侧车文件（.status.json / .cache.json）也是 stepN_* 前缀，
        # 不过滤就会被当成"最新产物"选中，随后 Pillow 抛
        # UnidentifiedImageError。这里只保留真实图像扩展名。
        candidates = [
            p for p in candidates
            if Path(p).suffix.lower() in set(
                getattr(media_io, "IMAGE_EXTS", ()) or
                {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
            )
        ]
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            input_source = candidates[0]
            break

    if not input_source:
        # BUG-023 fix: require a processed intermediate
        logger.error("STAGE6_NO_INPUT", "No processed intermediate file found, refusing to encode raw input")
        return 82

    output_path = params.get("output_path", "")
    if not output_path:
        output_dir = params.get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir,
            f"{Path(input_name).stem}_restored.avif")

    # WSL2 redirect
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    platform = params.get("platform", "auto")
    redirected_path, original_output_path, _, _ = wsl_redirect_path(
        output_path, no_wsl_redirect=no_wsl_redirect, target_platform=platform
    )
    if original_output_path:
        logger.info("STAGE6_WSL_REDIRECT",
                   f"WSL redirect: {output_path} -> {redirected_path}")
        output_path = redirected_path

    logger.info("STAGE6_START", "Final photo encoding",
                input=input_source, output=output_path)

    # NEW-V-06: 写字节之前登记产物；中断时连契约 sidecar 一起清理，
    # 避免"半截 AVIF + 完整契约"这种互相矛盾的状态。
    if cancel_guard is not None:
        cancel_guard.watch_outputs(
            output_path,
            extra_paths=[f"{output_path}.encode_contract.json"],
            logger=logger, stage=6)

    # --- FIX-V04: 色彩契约（必需转换失败即阻断）---
    _stem = Path(input_name).stem
    color = resolve_color_space(input_source, params, temp_dir, job_id, _stem,
                                logger)
    if color["status"] == COLOR_FAILED:
        logger.error("STAGE6_COLOR_BLOCKED",
                     "Colour contract could not be satisfied; not writing a "
                     "success artifact",
                     source_space=color["source_space"], reason=color["reason"])
        return 86
    working_path = color["working_path"]

    # --- FIX-V05: 编码输入适配（avifenc 不接受 TIFF）---
    try:
        adapter = media_io.prepare_encoder_input(working_path, temp_dir, logger)
    except media_io.MediaWriteError as e:
        logger.error("STAGE6_INPUT_ADAPTER_FAIL",
                     f"Could not produce an avifenc-acceptable input: {e}")
        return 82
    encode_input = adapter["path"]
    if Path(encode_input).suffix.lower() not in media_io.AVIFENC_SUPPORTED_INPUT_EXTS:
        logger.error("STAGE6_INPUT_FORMAT_UNSUPPORTED",
                     f"avifenc accepts only "
                     f"{sorted(media_io.AVIFENC_SUPPORTED_INPUT_EXTS)}; got "
                     f"{Path(encode_input).suffix}")
        return 86

    # --- FIX-V02: 高位深母版（真实精度）---
    tiff_info: Dict[str, Any] = {"written": False}
    if params.get("photo_bit_depth", 16) == 16:
        tiff_path = output_path.rsplit(".", 1)[0] + ".tif"
        tiff_info = write_archival_master(working_path, tiff_path, params, logger)
        if not tiff_info.get("written") and params.get("archival_tiff_required", False):
            logger.error("STAGE6_TIFF_REQUIRED_FAIL",
                         "Archival TIFF master is required but could not be "
                         "written", error=tiff_info.get("error"))
            return 86

    # --- FIX-V01: 命令构造（不支持的组合在启动前失败）---
    archival_mode = params.get("archival_mode", False)
    use_rgb444 = archival_mode
    try:
        cmd, contract = media_io.build_avifenc_cmd(
            encode_input, output_path, params, use_rgb444=use_rgb444,
            capability=capability, icc_path=color.get("icc_path"))
    except media_io.UnsupportedAvifConfig as e:
        logger.error("STAGE6_AVIF_CONFIG_UNSUPPORTED", str(e))
        return 86

    # 实际编码契约：记录"实际会编出什么"，而不是"用户想要什么"
    contract.update({
        "color": {k: v for k, v in color.items() if k != "working_path"},
        "encoder_input": adapter,
        "tiff_master": tiff_info,
        "output_path": output_path,
    })
    logger.info("STAGE6_ENCODE_CONTRACT",
                f"depth_arg={contract['depth_arg']} "
                f"effective_depth={contract['effective_decode_depth']} "
                f"yuv={contract['yuv']} cicp={contract['cicp']} "
                f"lossless={contract['lossless']} "
                f"icc_attached={contract['icc_attached']}",
                input=encode_input)

    logger.info("STAGE6_ENCODE", f"Encoding AVIF (archival={use_rgb444})",
                cmd=" ".join(cmd))
    timeout = _calculate_avifenc_timeout(encode_input)
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, shell=False)
        ok = r.returncode == 0
        stderr = (r.stderr or b"").decode("utf-8", "replace")
    except Exception as e:
        ok, stderr = False, str(e)
    contract["exit_code"] = None if ok else 84
    if not ok:
        logger.error("STAGE6_AVIF_FAIL", "avifenc encoding failed",
                     stderr=stderr[:2000])
        # FIX-H12-05: encoding failure must not be masked by another format
        return 84
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        # 退出码 0 也不代表成功：必须验证产物本身
        logger.error("STAGE6_AVIF_OUTPUT_MISSING",
                     "avifenc returned 0 but the output is missing/empty",
                     output=output_path)
        return 84

    # WSL redirect: copy back to original Linux path
    if original_output_path:
        if wsl_copy_back(output_path, original_output_path):
            logger.info("STAGE6_WSL_COPYBACK",
                        f"Copied result back to {original_output_path}")
            output_path = original_output_path
        else:
            logger.error("STAGE6_WSL_COPYBACK_FAIL",
                         f"Failed to copy {output_path} back to {original_output_path}")
            return 82
        if tiff_info.get("written"):
            orig_tiff = original_output_path.rsplit(".", 1)[0] + ".tif"
            wsl_copy_back(str(Path(output_path).with_suffix(".tif")), orig_tiff)

    # 落盘实际编码契约，供验收与上游读取
    try:
        contract_path = output_path + ".encode_contract.json"
        with open(contract_path, "w", encoding="utf-8") as f:
            json.dump(contract, f, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception as e:
        logger.warning("STAGE6_CONTRACT_WRITE_FAIL", f"Could not write contract sidecar: {e}")

    logger.info("STAGE6_DONE", "Photo encoding complete", output=output_path)
    return 0


if __name__ == "__main__":
    # FIX-P11-5: main() must return explicit int exit code per v7.32 §3.5
    rc = main()
    if rc is None:
        rc = 86  # validation failure — main() must always return explicit exit code
    sys.exit(rc)
