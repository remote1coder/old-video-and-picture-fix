#!/usr/bin/env python3
"""
Stage 1: Photo Traditional Preprocessing
- Color space conversion (L*a*b*, CMYK, Grayscale -> RGB)
- Scratch removal (G'MIC)
- DCT Deblocking (G'MIC -deblock, operates in DCT transform domain)
- Debanding (G'MIC)
- Grain extraction via frequency separation (high-pass filter)
- AI/artificial coloring detection (posterization + saturation heuristics)
- Single-channel degradation detection on processed image
- Single-channel correction in linear RGB space (AI not enabled only)

v6.3 Compliance:
- Supports L*a*b*, CMYK, Grayscale input conversion to RGB
- DCT deblock via G'MIC -deblock (transform domain)
- Channel degradation detection runs on processed working_path
- All thresholds read from config (not hardcoded)
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import re
from pathlib import Path
from typing import List, Optional  # R4: 真实运行时 NameError: Optional 未导入

import numpy as np

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from checkpoint import (compute_param_hash, build_intermediate_name,  # noqa: E402
                         safe_input_stem)

# FIX-P11-12: Photo-specific exit codes per v6.32 §7.7 / v7.32 exit code scheme
# All custom exit codes must be in 80-113 range
# ORIGINAL: mixed generic codes (82, 84) without clear semantics
PHOTO_ERR_GMIC_FAIL = 84    # G'MIC / external tool processing error
PHOTO_ERR_COLOR_CONV = 82   # I/O error during color conversion
PHOTO_ERR_OUTPUT = 86       # Output validation failure
PHOTO_ERR_COPY_FAIL = 82    # I/O error during file copy


def _detect_input_color_space(input_path: str, params: dict,
                               logger=None) -> str:
    """Detect embedded ICC profile and warn if missing/mismatched.

    FIX-P11-10: Per v6.32 §4.2.1: detect-then-warn, do NOT auto-convert.
    High DPI scans without ICC profile are flagged for user verification.
    """
    try:
        from PIL import Image
        img = Image.open(input_path)
        icc = img.info.get("icc_profile")
        if icc:
            # FIX-P11-11: Parse ICC profile content per v6.32 §4.2.1
            # ORIGINAL: only returned "embedded" without identifying profile
            try:
                import io
                from PIL import ImageCms
                profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                profile_name = ImageCms.getProfileName(
                    profile.profile).lower()
                if "adobe" in profile_name:
                    cs_name = "AdobeRGB"
                elif "prophoto" in profile_name or "romm" in profile_name:
                    cs_name = "ProPhoto"
                elif "srgb" in profile_name:
                    cs_name = "sRGB"
                else:
                    cs_name = "embedded"
            except Exception:
                cs_name = "embedded"
            logger.info("STAGE1_ICC_DETECTED",
                       f"ICC profile detected: {cs_name}",
                       profile_size=len(icc))
            return cs_name
        # No ICC profile: warn based on DPI
        dpi = params.get("dpi", 0)
        if dpi and dpi > 1200:
            logger.warning("STAGE1_ICC_MISSING",
                          f"High DPI ({dpi}) scan without ICC profile. "
                          f"Color space may be Adobe RGB or ProPhoto. "
                          f"Please verify in RawTherapee/GIMP before "
                          f"processing.")
        else:
            logger.info("STAGE1_ICC_ASSUME_SRGB",
                       "No ICC profile, assuming sRGB (low DPI scan)")
        return "unknown"
    except Exception:
        return "unknown"


def _detect_color_space(img_path: str) -> tuple:
    """Detect image color space and return (mode, needs_conversion).

    Returns:
        (detected_mode, target_mode_for_pil)
    """
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            mode = img.mode
            if mode == "RGB":
                return "RGB", None
            elif mode in ("RGBA", "P", "PA"):
                return mode, "RGB"
            elif mode == "L":
                return "Grayscale", "RGB"
            elif mode in ("CMYK", "YCbCr"):
                return mode, "RGB"
            elif mode == "LAB":
                return "L*a*b*", "RGB"
            elif mode in ("I;16", "I;16B"):
                return "16-bit Gray", "RGB"
            elif mode == "HSV":
                return "HSV", "RGB"
            else:
                return mode, "RGB"
    except Exception:
        return "Unknown", "RGB"


def _convert_to_rgb(input_path: str, output_path: str, logger=None) -> bool:
    """Convert image to RGB color space if needed.

    Bug P-12 fix: preserves ICC profile when converting color spaces.
    """
    try:
        from PIL import Image
        with Image.open(input_path) as img:
            mode = img.mode
            # Extract ICC profile before conversion
            icc_profile = img.info.get("icc_profile")

            if mode == "RGB":
                # Already RGB, just copy (preserve original ICC)
                # FIX-P11-9: Use copyfile (not copy2) per v7.32 §3.10.10
                # copy2 copies metadata which causes PermissionError on WSL2 NTFS
                shutil.copyfile(input_path, output_path)
                try:
                    src_stat = os.stat(input_path)
                    os.utime(output_path,
                            (src_stat.st_atime, src_stat.st_mtime))
                except OSError:
                    pass
                return True
            elif mode in ("RGBA", "P", "PA"):
                rgb_img = img.convert("RGB")
                save_kwargs = {}
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile
                rgb_img.save(output_path, **save_kwargs)
                if logger:
                    logger.info("STAGE1_COLOR", f"Converted {mode} -> RGB (ICC preserved={bool(icc_profile)})")
                return True
            elif mode == "L":
                rgb_img = img.convert("RGB")
                save_kwargs = {}
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile
                rgb_img.save(output_path, **save_kwargs)
                if logger:
                    logger.info("STAGE1_COLOR", "Converted Grayscale -> RGB")
                return True
            elif mode in ("CMYK", "YCbCr"):
                rgb_img = img.convert("RGB")
                save_kwargs = {}
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile
                rgb_img.save(output_path, **save_kwargs)
                if logger:
                    logger.info("STAGE1_COLOR", f"Converted {mode} -> RGB (ICC preserved={bool(icc_profile)})")
                return True
            elif mode == "LAB":
                rgb_img = img.convert("RGB")
                save_kwargs = {}
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile
                rgb_img.save(output_path, **save_kwargs)
                if logger:
                    logger.info("STAGE1_COLOR", "Converted L*a*b* -> RGB")
                return True
            elif mode in ("I;16", "I;16B"):
                # 16-bit grayscale: preserve full precision (Bug 11.9 fix)
                # BUG-048 fix: I;16 mode only supports 2D arrays in PIL.
                # For 3-channel 16-bit RGB, construct image safely.
                import numpy as np
                arr_16 = np.array(img, dtype=np.uint16)
                rgb_16 = np.stack([arr_16, arr_16, arr_16], axis=-1).astype(np.uint16)
                try:
                    # Try direct fromarray (works in newer Pillow versions for TIFF)
                    out_img = Image.fromarray(rgb_16)
                except (ValueError, TypeError):
                    # FIX-P11-9: Fallback must produce RGB output, not single-channel I;16
                    # per v6.32 §4.2.1: all outputs must be RGB for downstream processing
                    # ORIGINAL: fell back to Image.fromarray(arr_16, mode='I;16') which
                    # saves as single-channel, breaking downstream RGB pipeline
                    rgb_16_reshaped = rgb_16.reshape((rgb_16.shape[0], rgb_16.shape[1], 3))
                    try:
                        out_img = Image.fromarray(rgb_16_reshaped)
                    except (ValueError, TypeError):
                        # Final fallback: use imageio for proper 16-bit RGB TIFF
                        import imageio.v3 as iio
                        iio.imwrite(output_path, rgb_16_reshaped)
                        if logger:
                            logger.info("STAGE1_COLOR",
                                       "Converted 16-bit -> RGB via imageio fallback")
                        return True
                out_img.save(output_path)
                if logger:
                    logger.info("STAGE1_COLOR",
                               "Converted 16-bit Gray -> RGB (16bit preserved, BUG-048 safe)")
                return True
            elif mode == "HSV":
                rgb_img = img.convert("RGB")
                save_kwargs = {}
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile
                rgb_img.save(output_path, **save_kwargs)
                if logger:
                    logger.info("STAGE1_COLOR", "Converted HSV -> RGB")
                return True
            else:
                # Fallback: try converting any mode to RGB
                rgb_img = img.convert("RGB")
                save_kwargs = {}
                if icc_profile:
                    save_kwargs["icc_profile"] = icc_profile
                rgb_img.save(output_path, **save_kwargs)
                if logger:
                    logger.info("STAGE1_COLOR", f"Converted {mode} -> RGB (fallback, ICC preserved={bool(icc_profile)})")
                return True
    except Exception as e:
        if logger:
            logger.error("STAGE1_COLOR_FAIL", f"Color space conversion failed: {e}")
        return False


def _build_gmic_whitelist_re(params: dict):
    """Build a path-whitelist regex for G'MIC args per v7.32 §3.10.25.

    FIX-R11-8: Reads gmic_windows_whitelist from config (a regex char class).
    Always permits ':' (Windows drive separator — structurally required and
    safe under shell=False list-arg invocation) by inserting it into the
    class, since the config value lacks it and would reject all Windows
    absolute paths. Falls back to a safe default when config is absent.
    """
    wl = (params or {}).get("gmic_windows_whitelist")
    if not wl or not isinstance(wl, str) or not wl.startswith("["):
        wl = r"[a-zA-Z0-9_\-\./\\]"
    # Insert ':' before the closing ']' if not already permitted.
    if ":" not in wl:
        wl = wl[:-1] + ":" + wl[-1:]
    try:
        return re.compile(wl + "+")
    except re.error:
        # Malformed config whitelist — fall back to safe default.
        return re.compile(r"[a-zA-Z0-9_\-\./\\:]+")


#: G'MIC 参数解析里**实证会坏**的字符。
#:
#: FIX-2026-10-22（真实素材实测，G'MIC 4.0.5 + `shell=False` 的 argv 传递）：
#:   `,`  → 失败（G'MIC 用它分隔参数列表）
#:   `(` `)` `[` `]` `#` `&` `'` `+` 空格 中文 → **全部正常**
#:
#: 原实现用**白名单** `[a-zA-Z0-9_\-./\\:]` 校验路径，于是真实照片里极其
#: 常见的括号名（Windows 重名文件 `001(1).JPG`）被判非法：
#:
#:   STAGE1_DEBAND_FAIL "G'MIC path contains forbidden characters:
#:                       ...\job_001_001(1)_16bit.tif"
#:   → stage1 rc=84 → 整批 8 张照片全部失败
#:
#: 白名单把"合法文件名"和"危险字符"混为一谈，代价是整批素材跑不动；
#: shell=False 下不存在 shell 注入面，真正要防的只有 G'MIC 自己的解析。
#: 故改为黑名单 + 路径分量首字符检查。
_GMIC_FORBIDDEN_CHARS = (",",)


def _validate_gmic_path(path: str, wl_re=None) -> str:
    """校验并返回喂给 G'MIC 的绝对路径。

    FIX-2026-10-22：判据由"白名单"改为"黑名单 + 首字符检查"，
    理由见 `_GMIC_FORBIDDEN_CHARS` 的实测记录。

    Args:
        wl_re: 可选的严格白名单正则。仅在 config 显式打开
            `gmic_strict_whitelist` 时传入（默认不传 = 不启用）。
    """
    p = str(Path(path).resolve())
    for ch in _GMIC_FORBIDDEN_CHARS:
        if ch in p:
            raise ValueError(
                f"G'MIC path contains {ch!r}, which breaks G'MIC argument "
                f"parsing: {p}")
    for part in Path(p).parts:
        if part.startswith("-"):
            raise ValueError(
                f"G'MIC path has a component starting with '-' "
                f"(G'MIC would read it as a command): {p}")
    if wl_re is not None and not wl_re.fullmatch(p):
        raise ValueError(
            f"G'MIC path rejected by gmic_strict_whitelist: {p}")
    return p


_GMIC_MAJOR_CACHE: dict = {}


def _gmic_major_version() -> Optional[int]:
    """Return the installed G'MIC major version, or None when unknown.

    R4：G'MIC 4.0 起 **`-type` 命令被移除**（实测 4.0.5：
    `Unknown command or filename '-type'`），且像素类型名从 `ushort` 改为
    `uint16`。而在 3.x 里 `-type ushort` 是标准写法。
    版本差异必须探测而不是猜，否则换台机器就会挂。
    """
    if "major" in _GMIC_MAJOR_CACHE:
        return _GMIC_MAJOR_CACHE["major"]
    major = None
    try:
        r = subprocess.run(["gmic", "-version"], capture_output=True,
                           text=True, timeout=30, shell=False)
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        m = re.search(r"G'MIC interpreter \(v\.(\d+)\.", out) or \
            re.search(r"[Vv]ersion:?\s*(\d+)\.(\d+)", out)
        if m:
            major = int(m.group(1))
    except Exception:
        major = None
    _GMIC_MAJOR_CACHE["major"] = major
    return major


def _gmic_16bit_output_args(output_path: str) -> list:
    """R4: build the G'MIC args that produce a 16-bit output file.

    实测（G'MIC 4.0.5）：
      * `-o X.tif` 默认输出 **float32**（不符合 16-bit 处理要求）；
      * `-o X.tif,uint16` 输出 **uint16** ✅
      * `-type ushort` 已移除。
    G'MIC 3.x 仍用 `-type ushort`。因此按版本选择，两者都能跑通。
    """
    if (_gmic_major_version() or 3) >= 4:
        return ["-o", f"{output_path},uint16"]
    return ["-type", "ushort", "-o", output_path]


#: 缓存"哪个 gmic 可执行文件能跑 denoise_patchpca"。
#:
#: `None` = 还没探测过；`""` = 探测过但都不行（用默认 PATH）。
_PATCHPCA_GMIC_CACHE = {"exe": None}


def _gmic_candidates() -> list:
    """按"优先默认、然后从新到旧"列出可执行文件。

    默认（PATH 上的 `gmic`）排第一 —— 如果哪天升级到修好这个 bug 的版本，
    直接用新的，不浪费一次失败。
    """
    import re as _re
    cands = ["gmic"]
    try:
        gdir = SCRIPT_DIR.parent / "tools_r4" / "gmic"
        found = []
        for d in gdir.glob("gmic-*-cli-win64"):
            exe = d / "gmic.exe"
            if not exe.exists():
                continue
            m = _re.search(r"gmic-(\d+)\.(\d+)\.(\d+)", d.name)
            ver = tuple(int(x) for x in m.groups()) if m else (0, 0, 0)
            found.append((ver, str(exe)))
        for _, exe in sorted(found, reverse=True):
            cands.append(exe)
    except Exception:
        pass
    return cands


def _run_gmic_once(exe: str, safe_in: str, safe_out: str,
                   gmic_args: tuple) -> tuple:
    cmd = [exe, safe_in]
    cmd.extend(list(gmic_args))
    cmd.extend(_gmic_16bit_output_args(safe_out))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           errors="replace", timeout=300, shell=False)
        if r.returncode != 0:
            return False, (r.stderr or "")[:1000]
        # 返回码 0 但没产出也算失败（实测过这种情况）
        if not Path(safe_out).exists():
            return False, (r.stderr or "")[:1000] or "no output produced"
        return True, ""
    except Exception as e:
        return False, str(e)


def run_gmic(input_path: str, output_path: str, *gmic_args,
             params: dict = None) -> tuple:
    """Run G'MIC with shell=False safety.

    Returns:
        (success: bool, stderr: str) tuple per Bug 11.20 fix.
    """
    # FIX-R11-8 / FIX-2026-10-22: 路径校验。默认走黑名单（放行括号/中文等
    # 合法文件名）；只有 config 显式打开 gmic_strict_whitelist 才启用白名单。
    wl_re = None
    if (params or {}).get("gmic_strict_whitelist"):
        wl_re = _build_gmic_whitelist_re(params)
    try:
        safe_in = _validate_gmic_path(input_path, wl_re)
        safe_out = _validate_gmic_path(output_path, wl_re)
    except ValueError as e:
        return False, str(e)
    # R4：输出也必须显式指定 16-bit。G'MIC 4.x 的 `-o` 若不指定像素类型会写出
    # **float32 TIFF**，而 Pillow 读不了 float TIFF —— 实测表现为
    # `UnidentifiedImageError` → OUTPUT_CORRUPT → 阻塞后续 AI 阶段。
    # 3.x 默认沿用输入类型，所以这个问题只在升级后暴露。

    # 只对 `denoise_patchpca` 做版本回落。
    #
    # 实测（2026-10-20）：G'MIC **4.0.6**（2026/10/14 发布）的
    # `denoise_patchpca` 在**所有**输入上都崩：
    #   `Operator 'vector[ind]': Invalid vector read`
    # 而项目里同时存在的 **4.0.5** 对同一输入正常产出。
    # 4.0.6 太新（发布 5 天），上游还没有相关 issue。
    #
    # 策略：先用默认（PATH 上的 4.0.6），失败后回落到旧版本重试一次。
    # 这样如果将来升级到修好的版本，会自动用新的，不浪费任何一次调用。
    wants_patchpca = any("denoise_patchpca" in str(a) for a in gmic_args)

    if not wants_patchpca:
        return _run_gmic_once("gmic", safe_in, safe_out, gmic_args)

    cached = _PATCHPCA_GMIC_CACHE["exe"]
    if cached is not None:
        # 已经探测过：直接用记下来的那个（"" 表示都不行，回到默认再试一次）
        return _run_gmic_once(cached or "gmic", safe_in, safe_out, gmic_args)

    last_err = ""
    for exe in _gmic_candidates():
        ok, err = _run_gmic_once(exe, safe_in, safe_out, gmic_args)
        if ok:
            _PATCHPCA_GMIC_CACHE["exe"] = exe
            return True, ""
        last_err = err
        # 默认版本成功就直接返回（上面已处理）；失败则继续试旧版本
    _PATCHPCA_GMIC_CACHE["exe"] = ""     # 都不行，别每次都重试一轮
    return False, last_err


def detect_channel_degradation(img_path: str, params: dict) -> dict:
    """Detect single-channel degradation in photo.

    Uses config-driven thresholds (not hardcoded).
    Handles 16-bit I;16 images with full precision.
    """
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(img_path)
        arr_raw = np.array(img)
        # FIX-R11-6: Unify to 3-channel and normalize by actual bit depth.
        # I;16 yields a 2D array; 16bit RGB TIFF PIL mode is unreliable.
        # ORIGINAL: I;16 branch produced a 2D array → arr[:,:,c] IndexError
        # was swallowed by the outer except → silently returned not-degraded.
        if arr_raw.ndim == 2:
            arr_raw = np.stack([arr_raw, arr_raw, arr_raw], axis=-1)
        elif arr_raw.ndim == 3 and arr_raw.shape[2] == 4:
            arr_raw = arr_raw[:, :, :3]
        if arr_raw.dtype == np.uint16:
            max_val = 65535.0
        elif arr_raw.dtype == np.uint8:
            max_val = 255.0
        else:
            # 'I' mode (int32) or other: infer from data range
            mx = float(arr_raw.max()) if arr_raw.size else 0.0
            max_val = 65535.0 if mx > 255 else 255.0
        arr = arr_raw.astype(np.float32) / max_val

        # Compute per-channel stats
        means = [float(arr[:, :, c].mean()) for c in range(3)]
        stds = [float(arr[:, :, c].std()) for c in range(3)]

        result = {
            "degraded": False,
            "channels": [],
            "mean_diff": 0.0,
            "std_diff": 0.0,
        }

        # Check mean differences using config thresholds
        mean_threshold = params.get("channel_degradation_mean_threshold", 0.08)
        std_threshold = params.get("channel_degradation_std_threshold", 0.15)

        # FIX-R11-11: Per v7.32 §3.10.6 "某通道均值与另两通道均值差异",
        # compute each channel's deviation from the mean of the OTHER TWO
        # channels, then take the max. ORIGINAL: pairwise max of abs diff.
        max_mean_diff = 0.0
        for c in range(3):
            others = [means[i] for i in range(3) if i != c]
            avg_others = sum(others) / 2.0
            max_mean_diff = max(max_mean_diff, abs(means[c] - avg_others))
        # FIX-P11-5: Relative std difference per v6.32 §4.2.1
        # Spec: "任意两通道 std 差异 > 15%（相对值）"
        # ORIGINAL: max_std_diff = max(abs(stds[i] - stds[j])) was absolute
        max_std_diff = 0.0
        for i in range(3):
            for j in range(i+1, 3):
                avg_std = (stds[i] + stds[j]) / 2.0
                if avg_std > 1e-8:
                    rel_diff = abs(stds[i] - stds[j]) / avg_std
                    max_std_diff = max(max_std_diff, rel_diff)

        if max_mean_diff > mean_threshold or max_std_diff > std_threshold:
            result["degraded"] = True
            result["mean_diff"] = max_mean_diff
            result["std_diff"] = max_std_diff
            # FIX-R11-1: Record which channels are degraded per v7.32 §3.10.6
            # ("记录退化通道"). A channel is degraded if its mean deviates from
            # the average of the other two beyond mean_threshold, or its std
            # relatively deviates beyond std_threshold. ORIGINAL: channels=[]
            # was initialized but never populated.
            degraded_channels = []
            for c in range(3):
                others_m = [means[i] for i in range(3) if i != c]
                avg_m = sum(others_m) / 2.0
                others_s = [stds[i] for i in range(3) if i != c]
                avg_s = sum(others_s) / 2.0
                mean_bad = abs(means[c] - avg_m) > mean_threshold
                std_bad = (avg_s > 1e-8
                           and abs(stds[c] - avg_s) / avg_s > std_threshold)
                if mean_bad or std_bad:
                    degraded_channels.append({
                        "channel": c,
                        "mean": means[c],
                        "std": stds[c],
                        "mean_degraded": mean_bad,
                        "std_degraded": std_bad,
                    })
            result["channels"] = degraded_channels

        return result
    except Exception:
        return {"degraded": False}


def _sRGB_to_linear(c: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear RGB using IEC 61966-2-1."""
    c = np.clip(c, 0.0, 1.0)
    mask = c <= 0.04045
    linear = np.empty_like(c, dtype=np.float32)
    linear[mask] = c[mask] / 12.92
    linear[~mask] = np.power((c[~mask] + 0.055) / 1.055, 2.4)
    return linear


def _linear_to_sRGB(c: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB using IEC 61966-2-1."""
    c = np.clip(c, 0.0, 1.0)
    mask = c <= 0.0031308
    srgb = np.empty_like(c, dtype=np.float32)
    srgb[mask] = c[mask] * 12.92
    srgb[~mask] = 1.055 * np.power(c[~mask], 1.0 / 2.4) - 0.055
    return srgb


def _reinhard_tone_map(c: np.ndarray, white_point: float = 2.0) -> np.ndarray:
    """Apply Reinhard tone mapping to compress highlights."""
    return c * (1.0 + c / (white_point * white_point)) / (1.0 + c)


def extract_grain_pattern(img_path: str, output_path: str,
                            params: dict = None, logger=None) -> bool:
    """Extract film grain pattern using high-pass filtering.
    
    Per v6.3 §4.2.1: grain extraction for analysis and later synthesis.
    Uses frequency separation to isolate high-frequency grain from low-frequency image.
    """
    try:
        from PIL import Image
        import numpy as np
        from scipy.ndimage import gaussian_filter
        
        img = Image.open(img_path)
        if img.mode == "I;16":
            arr = np.array(img, dtype=np.float32) / 65535.0
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
        else:
            arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        
        # Frequency separation: low-pass = image structure, high-pass = grain
        grain_strength = params.get("grain_strength", 0.3) if params else 0.3
        grain_size = params.get("grain_size", 0.5) if params else 0.5
        grain_roughness = params.get("grain_roughness", 0) if params else 0
        # FIX-P11-3: grain_size maps to sigma per v7.32 §3.10.4
        # grain_size (0.5-2.0) → sigma: larger = coarser grain
        # ORIGINAL: hardcoded sigma=2.0, ignored grain_size
        sigma = grain_size * 2.0  # grain_size=0.5→1.0, grain_size=1.0→2.0, grain_size=2.0→4.0

        # FIX-P11-9: Laplacian pyramid frequency separation per v6.32 §4.2.1
        # ORIGINAL: single Gaussian high-pass, not true frequency separation
        levels = max(1, int(grain_size * 3))
        grain = np.zeros_like(arr)
        for c in range(3):
            channel = arr[:, :, c]
            pyramid = []
            current = channel.copy()
            for _ in range(levels):
                low = gaussian_filter(current, sigma=sigma)
                high = current - low
                pyramid.append(high)
                current = low

            # Weight levels by roughness (0=pure fine/high-freq, 1=pure coarse/low-freq)
            total_weight = 0.0
            for i, level in enumerate(pyramid):
                weight = (1.0 - grain_roughness) ** i  # roughness=0 → all weight on finest
                grain[:, :, c] += level * weight * grain_strength
                total_weight += weight

            if total_weight > 1e-8:
                grain[:, :, c] /= total_weight  # Normalize weights
        
        # Save grain pattern.
        # FIX-R11-14: Save as 16-bit TIFF when photo_bit_depth==16 to preserve
        # precision for downstream grain synthesis per v6.32 §4.2.1. Use
        # tifffile (preferred) with imageio then PIL I;16 fallbacks.
        # ORIGINAL: 8bit-only save for non-I;16 inputs dropped precision for
        # 16bit RGB TIFF sources; I;16 path used imageio only.
        grain_clipped = np.clip(grain + 0.5, 0.0, 1.0)
        bit_depth = (params or {}).get("photo_bit_depth", 16)
        if bit_depth == 16:
            grain_uint16 = (grain_clipped * 65535).astype(np.uint16)
            saved_16 = False
            try:
                import tifffile
                tifffile.imwrite(output_path, grain_uint16)
                saved_16 = True
            except Exception:
                pass
            if not saved_16:
                try:
                    import imageio.v3 as iio
                    iio.imwrite(output_path, grain_uint16)
                    saved_16 = True
                except Exception:
                    pass
            if not saved_16:
                # Last resort: per-channel I;16 via PIL (single channel only)
                Image.fromarray(grain_uint16[:, :, 0], mode='I;16').save(
                    output_path)
        else:
            grain_uint8 = (grain_clipped * 255).astype(np.uint8)
            out_img = Image.fromarray(grain_uint8)
            out_img.save(output_path)
        
        if logger:
            logger.info("STAGE1_GRAIN_EXTRACT", f"Grain pattern saved: {output_path}")
        return True
    except Exception as e:
        if logger:
            logger.warning("STAGE1_GRAIN_EXTRACT_FAIL", f"Grain extraction failed: {e}")
        return False


def detect_ai_coloring(img_path: str, params: dict = None,
                       logger=None) -> dict:
    """Detect signs of artificial/AI coloring in photos.

    Heuristics:
    - Unnatural color saturation spikes (posterization)
    - Abrupt hue transitions
    - High chroma noise in supposedly uniform regions

    Returns dict with detected (bool) and confidence (float).
    """
    # FIX-P11-6: Config-driven thresholds per v7.32 §3.10.16
    # ORIGINAL: hardcoded 0.3 and 0.5
    params = params or {}
    posterization_threshold = params.get(
        "ai_coloring_posterization_threshold", 0.3)
    detection_threshold = params.get(
        "ai_coloring_detection_threshold", 0.5)

    try:
        from PIL import Image
        import numpy as np

        img = Image.open(img_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0

        # Convert to HSV-like space for analysis
        max_c = arr.max(axis=2)
        min_c = arr.min(axis=2)
        delta = max_c - min_c

        # Saturation
        sat = np.where(max_c > 0, delta / (max_c + 1e-8), 0)

        # Detect posterization: count unique colors per channel
        posterization_score = 0.0
        for c in range(3):
            unique_ratio = len(np.unique(
                (arr[:, :, c] * 255).astype(np.uint8))) / 256.0
            if unique_ratio < posterization_threshold:
                posterization_score += 1.0
        posterization_score /= 3.0

        # Detect unnatural hue transitions
        gray_mask = delta < 0.05
        unnatural_sat = np.mean(sat[gray_mask]) if gray_mask.any() else 0.0

        # Confidence: combination of posterization + unnatural saturation
        confidence = min(1.0, posterization_score * 0.6 + unnatural_sat * 4.0)
        detected = confidence > detection_threshold

        if logger:
            logger.info("STAGE1_COLORING_DETECT",
                       f"AI coloring detection: detected={detected}, "
                       f"confidence={confidence:.2f}, "
                       f"posterization={posterization_score:.2f}, "
                       f"unnatural_sat={unnatural_sat:.4f}")

        return {"detected": detected, "confidence": confidence,
                "posterization_score": posterization_score,
                "unnatural_saturation": float(unnatural_sat)}
    except Exception as e:
        if logger:
            logger.warning("STAGE1_COLORING_DETECT_FAIL",
                          f"Coloring detection failed: {e}")
        return {"detected": False, "confidence": 0.0}


def apply_channel_correction(img_path: str, output_path: str,
                              params: dict, logger=None) -> bool:
    """Apply single-channel correction in linear RGB space.

    Only applied when degradation is severe and AI is NOT enabled.
    Uses IEC 61966-2-1 color space conversion.
    Respects all channel_correction_* config parameters.
    """
    try:
        from PIL import Image
        import numpy as np

        # FIX-R11-13: Master toggle for pre-compensation per v7.32 §3.10.6.
        # When disabled, severe degradation does not trigger correction.
        if not params.get("channel_correction_enabled", True):
            if logger:
                logger.info("STAGE1_CHANNEL_DISABLED",
                            "channel_correction_enabled=false, skipping "
                            "pre-compensation per v7.32 §3.10.6")
            return False

        img = Image.open(img_path)
        # Preserve 16-bit if available
        # FIX-R11-13: Honor channel_correction_float_precision (float32/float64)
        # per config. ORIGINAL: hardcoded float32, ignored config key.
        precision = params.get("channel_correction_float_precision", "float32")
        dtype = np.float64 if precision == "float64" else np.float32
        if img.mode == "I;16":
            arr = np.array(img).astype(dtype) / 65535.0
        else:
            arr = np.array(img.convert("RGB")).astype(dtype) / 255.0

        # Convert to linear space
        # FIX-R11-2: Only IEC 61966-2-1 is allowed per v6.32 §7.16
        # ("严禁使用简单 pow(x, 2.2) 代替标准分段函数"). Non-IEC formula is
        # rejected rather than falling back to the forbidden pow(2.2).
        # ORIGINAL: fell back to np.power(arr, 2.2) for non-IEC formula.
        formula = params.get("channel_correction_formula", "iec61966-2-1")
        if formula != "iec61966-2-1":
            if logger:
                logger.error("STAGE1_CHANNEL_FORMULA",
                             f"Unsupported channel_correction_formula={formula}, "
                             f"only 'iec61966-2-1' is allowed per v6.32 §7.16")
            return False
        linear = _sRGB_to_linear(arr)

        # Compute per-channel stats in linear space
        means = [float(linear[:, :, c].mean()) for c in range(3)]
        stds = [float(linear[:, :, c].std()) for c in range(3)]

        # Identify degraded channel(s)
        mean_threshold = params.get("channel_degradation_mean_threshold_severe", 0.15)
        std_threshold = params.get("channel_degradation_std_threshold_severe", 0.25)

        # FIX-R11-11: "channel vs avg of other two" semantics, consistent
        # with detect_channel_degradation per v7.32 §3.10.6.
        # ORIGINAL: pairwise max(abs(means[i]-means[j])).
        max_mean_diff = 0.0
        for c in range(3):
            others = [means[i] for i in range(3) if i != c]
            avg_others = sum(others) / 2.0
            max_mean_diff = max(max_mean_diff, abs(means[c] - avg_others))
        # FIX-P11-5: Relative std difference per v6.32 §4.2.1 (consistent with detection)
        max_std_diff = 0.0
        for i in range(3):
            for j in range(i+1, 3):
                avg_std = (stds[i] + stds[j]) / 2.0
                if avg_std > 1e-8:
                    rel_diff = abs(stds[i] - stds[j]) / avg_std
                    max_std_diff = max(max_std_diff, rel_diff)

        if max_mean_diff < mean_threshold and max_std_diff < std_threshold:
            if logger:
                logger.info("STAGE1_CHANNEL_SKIP",
                           "Degradation below severe threshold, skipping correction")
            return False

        # Find target reference: use median mean/std as reference
        sorted_means = sorted(means)
        target_mean = sorted_means[1]  # median
        sorted_stds = sorted(stds)
        target_std = sorted_stds[1]

        corrected = linear.copy()
        # PHOTO-AUDIT-008 fix: independent denoise + level correction per v6.32 §4.2.1
        # For each degraded channel, apply median denoise then histogram-level correction
        # against the reference (median) channel
        ref_channel_idx = np.argsort(means)[1]  # median mean channel as reference
        ref_ch = corrected[:, :, ref_channel_idx]
        for c in range(3):
            ch = corrected[:, :, c]
            ch_mean = means[c]
            ch_std = stds[c]

            # Skip if close to reference
            if abs(ch_mean - target_mean) < 0.01 and abs(ch_std - target_std) < 0.01:
                continue

            # Step 1: Independent denoise (median filter to remove impulse noise)
            try:
                from scipy.ndimage import median_filter
                ch_denoised = median_filter(ch, size=3)
            except Exception:
                ch_denoised = ch

            # Step 2: Level correction using percentile-based histogram matching to reference
            p_low, p_high = np.percentile(ch_denoised, [1, 99])
            ref_low, ref_high = np.percentile(ref_ch, [1, 99])
            if p_high > p_low and ref_high > ref_low:
                ch_corrected = (ch_denoised - p_low) / (p_high - p_low) * (ref_high - ref_low) + ref_low
            else:
                ch_corrected = ch_denoised

            corrected[:, :, c] = ch_corrected

        # Normalize if configured
        if params.get("channel_correction_normalize", True):
            # FIX-P11-2: Only normalize channels that deviate from reference
            # ORIGINAL: applied to all channels unconditionally + wrong formula
            # mapped to [target_mean, target_mean+target_std] instead of centered
            # Per v6.32 §7.16: standardize to reference statistics
            for c in range(3):
                ch = corrected[:, :, c]
                ch_mean = ch.mean()
                ch_std = ch.std()
                if abs(ch_mean - target_mean) > 0.01 or abs(ch_std - target_std) > 0.01:
                    if ch_std > 1e-8:
                        # Standardize to reference statistics
                        corrected[:, :, c] = (ch - ch_mean) / ch_std * target_std + target_mean
                    else:
                        # Near-constant channel: just shift mean
                        corrected[:, :, c] = ch - ch_mean + target_mean

        # FIX-P1-3: Reinhard tone mapping BEFORE clamp (standard order per
        # v7.32 §3.10.6). Tone-map first to compress out-of-range highlights/
        # shadows, THEN clamp. Clamping first would hard-clip super-white/
        # super-black detail that tone mapping is meant to preserve.
        if params.get("channel_correction_soft_compress", True):
            corrected = _reinhard_tone_map(corrected)
        # Clamp to valid range (after tone mapping)
        corrected = np.clip(corrected, 0.0, 1.0)

        # Convert back to sRGB
        # FIX-R11-2: IEC 61966-2-1 only (no pow(1/2.2) fallback per v6.32 §7.16).
        # formula is guaranteed == "iec61966-2-1" (rejected earlier otherwise).
        srgb = _linear_to_sRGB(corrected)
        srgb = np.clip(srgb, 0.0, 1.0)

        # Round and denormalize to configured bit depth
        bit_depth = params.get("photo_bit_depth", 16)
        if bit_depth == 16:
            max_val = 65535.0
            dtype = np.uint16
        else:
            max_val = 255.0
            dtype = np.uint8

        # Apply rounding if configured
        if params.get("channel_correction_round", True):
            out_arr = np.rint(srgb * max_val).astype(dtype)
        else:
            out_arr = (srgb * max_val).astype(dtype)

        # Apply clamp if configured
        if params.get("channel_correction_clamp", True):
            if bit_depth == 16:
                out_arr = np.clip(out_arr, 0, 65535).astype(dtype)
            else:
                out_arr = np.clip(out_arr, 0, 255).astype(dtype)

        out_img = Image.fromarray(out_arr)
        out_img.save(output_path)

        if logger:
            logger.info("STAGE1_CHANNEL_CORRECT",
                       f"Channel correction applied: mean_diff={max_mean_diff:.4f}, std_diff={max_std_diff:.4f}")
        return True
    except Exception as e:
        if logger:
            logger.warning("STAGE1_CHANNEL_FAIL", f"Channel correction failed: {e}")
        return False


def detect_and_crop_black_bars(img_path: str, params: dict,
                                logger=None, out_dir: str = None) -> tuple:
    """Detect and crop black bars; return (cropped_path, crop_meta).

    FIX-R11-3: Per v7.32 §3.10.3 / v6.32 §7.26: black bars must be detected
    and cropped before color-space processing, restored after. Effective area
    <50% of original OR effective w/h <240px → FAILED (terminate).
    Returns:
      (img_path, None) — no black bars detected (unchanged)
      (cropped_path, {"cropped": True, "box": [...], "original_size": [...]})
      (None, {"failed": True, "reason": ...}) — FAILED condition
    """
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(img_path)
        arr = np.array(img.convert("RGB"), dtype=np.float32)
        h, w = arr.shape[:2]
        gray = arr.mean(axis=2)
        # FIX-R11-3: Luma threshold. Photos are 8bit-equivalent after
        # convert("RGB"); 16 is the limited-range 8bit black floor per
        # v7.32 §3.10.3 and a conservative near-black threshold for full range.
        thresh = float(params.get("black_bar_luma_threshold", 16))

        def _is_bar_row(r):
            # Row is a black bar if its mean is below threshold (uniform dark)
            return float(np.mean(r)) < thresh

        def _is_bar_col(c):
            return float(np.mean(c)) < thresh

        top = 0
        while top < h and _is_bar_row(gray[top]):
            top += 1
        bot = h
        while bot > top and _is_bar_row(gray[bot - 1]):
            bot -= 1
        left = 0
        while left < w and _is_bar_col(gray[:, left]):
            left += 1
        right = w
        while right > left and _is_bar_col(gray[:, right - 1]):
            right -= 1

        eff_w = right - left
        eff_h = bot - top
        eff_area = eff_w * eff_h
        orig_area = w * h

        # FIX-R11-3: Effective area <50% → FAILED
        if orig_area > 0 and eff_area < orig_area * 0.5:
            return None, {
                "failed": True,
                "reason": "effective_area_lt_50pct",
                "effective": f"{eff_w}x{eff_h}",
                "original": f"{w}x{h}",
            }
        # FIX-R11-3: Effective w/h <240px → FAILED
        if eff_w < 240 or eff_h < 240:
            return None, {
                "failed": True,
                "reason": "effective_dim_lt_240",
                "effective": f"{eff_w}x{eff_h}",
                "original": f"{w}x{h}",
            }
        # No black bars
        if top == 0 and bot == h and left == 0 and right == w:
            return img_path, None

        cropped = img.crop((left, top, right, bot))
        stem = Path(img_path).stem
        _out_dir = out_dir or str(Path(img_path).parent)
        out_path = str(Path(_out_dir) / f"{stem}_crop.tif")
        cropped.save(out_path)
        if logger:
            logger.info("STAGE1_BLACKBAR_CROP",
                        f"Cropped black bars: box=[{left},{top},{right},{bot}] "
                        f"effective={eff_w}x{eff_h} original={w}x{h}")
        return out_path, {
            "cropped": True,
            "box": [int(left), int(top), int(right), int(bot)],
            "original_size": [int(w), int(h)],
        }
    except Exception as e:
        if logger:
            logger.warning("STAGE1_BLACKBAR_DETECT_FAIL",
                           f"Black bar detection failed: {e}")
        return img_path, None


def restore_black_bars(processed_path: str, crop_meta: dict,
                        output_path: str, params: dict,
                        logger=None) -> bool:
    """Paste processed crop back into original frame size with black borders.

    FIX-R11-3: Per v7.32 §3.10.3, restore black bars after color-space
    processing using the bit-depth-appropriate black level:
      8bit limited → 16, 10bit limited → 64, 16bit limited → 4096,
      full range (any bit depth) → 0.
    For the photo RGB pipeline the Y black level is applied uniformly to all
    RGB channels (the U/V neutral table is video-YUV-oriented; photos are RGB).
    """
    if not crop_meta or not crop_meta.get("cropped"):
        shutil.copyfile(processed_path, output_path)
        return True
    try:
        from PIL import Image
        import numpy as np
        box = crop_meta["box"]
        orig_w, orig_h = crop_meta["original_size"]
        left, top, right, bot = box
        bit_depth = int(params.get("photo_bit_depth", 16))
        # FIX-R11-3: black level by bit depth + range per v7.32 §3.10.3
        range_mode = params.get("black_bar_range", "limited")
        if range_mode == "full":
            black_fill = 0
        else:
            black_fill = {8: 16, 10: 64, 16: 4096}.get(bit_depth, 0)

        if bit_depth == 16:
            try:
                import tifffile
                crop_arr = tifffile.imread(processed_path)
            except Exception:
                crop_arr = np.array(Image.open(processed_path))
            if crop_arr.ndim == 2:
                crop_arr = np.stack([crop_arr] * 3, axis=-1)
            canvas = np.full(
                (orig_h, orig_w, crop_arr.shape[2]), black_fill,
                dtype=crop_arr.dtype)
            canvas[top:bot, left:right] = crop_arr
            try:
                import tifffile
                tifffile.imwrite(output_path, canvas)
            except Exception:
                Image.fromarray(canvas[:, :, 0], mode='I;16').save(output_path)
        else:
            crop_arr = np.array(Image.open(processed_path).convert("RGB"))
            canvas = np.full((orig_h, orig_w, 3), black_fill, dtype=np.uint8)
            canvas[top:bot, left:right] = crop_arr
            Image.fromarray(canvas).save(output_path)
        if logger:
            logger.info("STAGE1_BLACKBAR_RESTORE",
                        f"Restored black bars: {orig_w}x{orig_h} "
                        f"black_fill={black_fill} bit_depth={bit_depth}")
        return True
    except Exception as e:
        if logger:
            logger.warning("STAGE1_BLACKBAR_RESTORE_FAIL", f"{e}")
        try:
            shutil.copyfile(processed_path, output_path)
        except Exception:
            return False
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--params")
    parser.add_argument("--params-file")
    # FIX-P11-7: --config is passed by main.py but unused here;
    # all config is provided via --params JSON. Kept for argparse compat.
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

    temp_dir = params.get("temp_dir", "./temp")
    # FIX-P11-6: v7.32 §3.10.10: WSL2 path redirect for temp_dir
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    platform = params.get("platform", "auto")
    try:
        from checkpoint import wsl_redirect_path
        redirected_temp, orig_temp, _, _ = wsl_redirect_path(
            temp_dir, no_wsl_redirect=no_wsl_redirect,
            target_platform=platform
        )
        if redirected_temp != temp_dir:
            logger.info("STAGE1_WSL_REDIRECT",
                        f"temp_dir: {temp_dir} -> {redirected_temp}")
            temp_dir = redirected_temp
            params["temp_dir"] = temp_dir
    except Exception:
        pass  # wsl_redirect_path may not be available on native Windows
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    input_path = args.input
    input_name = Path(input_path).name

    # Check for existing intermediate
    param_hash = compute_param_hash(input_path, params)
    inter_name = build_intermediate_name(
        args.job_id, input_name, 1, "preprocess", param_hash, "tif"
    )
    # FIX-P10-11/P11-8: Use pathlib.Path consistently per CLAUDE.md §4
    # ORIGINAL: os.path.join + os.path.exists + os.path.getsize
    inter_path = str(Path(temp_dir) / inter_name)
    inter_path_p = Path(temp_dir) / inter_name

    if inter_path_p.exists() and inter_path_p.stat().st_size > 0:
        logger.info("STAGE1_SKIP", "Intermediate exists", path=inter_path)
        return 3

    logger.info("STAGE1_START", "Photo preprocessing", input=input_path)

    # Step 0: Detect and convert color space if needed
    color_mode, needs_conv = _detect_color_space(input_path)
    if needs_conv:
        logger.info("STAGE1_COLOR_DETECT", f"Detected {color_mode}, converting to RGB")
        rgb_path = str(Path(temp_dir) /
            f"{args.job_id}_{safe_input_stem(input_name)}_rgb.tif")
        if not _convert_to_rgb(input_path, rgb_path, logger):
            logger.error("STAGE1_COLOR_FAIL", "Color conversion failed")
            return PHOTO_ERR_COLOR_CONV
        working_input = rgb_path
    else:
        working_input = input_path
        logger.info("STAGE1_COLOR_OK", f"Input is already {color_mode}")

    # FIX-P11-10: v6.32 §4.2.1 — detect ICC profile and warn if missing
    # High-DPI scans without ICC need user verification; no auto-conversion
    # Respect user override via config input_color_space
    user_cs = params.get("input_color_space", "auto")
    if user_cs == "auto":
        detected_cs = _detect_input_color_space(working_input, params, logger)
    else:
        detected_cs = user_cs
        logger.info("STAGE1_ICC_USER_OVERRIDE",
                   f"User specified color space: {detected_cs}")

    # Save color space info to sidecar JSON for downstream stages
    # FIX-P11-11: icc_profile_embedded now checks for any recognized ICC name
    # ORIGINAL: only checked == "embedded", missed AdobeRGB/ProPhoto/sRGB
    icc_names = {"embedded", "AdobeRGB", "ProPhoto", "sRGB"}
    color_meta = {
        "input_color_space": detected_cs,
        "icc_profile_embedded": (detected_cs in icc_names),
        "user_override": (user_cs != "auto"),
    }
    color_meta_path = str(Path(temp_dir) /
        f"{args.job_id}_{safe_input_stem(input_name)}_color_meta.json")

    # FIX-2026-10-20（G6 实测挖出的架构缺陷）：宽色域 → sRGB 的转换需要
    # ICC **二进制**，但 G'MIC 重写像素时不会带上它 —— 实测所有中间产物
    # （_16bit.tif / .deblock.tif / .deband.tif / step5_postprocess.tif）
    # 的 ICC 全部为无，于是 stage6 的 _extract_source_icc() 必然读不到，
    # 任何 AdobeRGB/ProPhoto/DisplayP3 素材都 COLOR_FAILED（rc=86）。
    # 而 stage1 只把 ICC 的**名字**写进了 color_meta，二进制从未传递。
    #
    # 修法：按 stage6 自己写盘的同一命名落一份 sidecar，stage6 读不到
    # 工作文件时回退读它（见 stage6_encode.py 的 STAGE6_ICC_SIDECAR）。
    icc_sidecar = str(Path(temp_dir) /
        f"{args.job_id}_{safe_input_stem(input_name)}_source.icc")
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(working_input) as _im:
            _icc_bytes = _im.info.get("icc_profile")
        if _icc_bytes:
            Path(icc_sidecar).write_bytes(_icc_bytes)
            color_meta["icc_path"] = icc_sidecar
            logger.info("STAGE1_ICC_SIDECAR",
                        "Saved source ICC binary for wide-gamut conversion",
                        path=icc_sidecar, profile_size=len(_icc_bytes))
    except Exception as _ice:
        # 拿不到二进制不阻断 stage1（sRGB 素材不需要它），但必须留痕
        logger.warning("STAGE1_ICC_SIDECAR_FAIL",
                       f"Could not save source ICC sidecar: {_ice}")

    try:
        with open(color_meta_path, "w", encoding="utf-8") as f:
            json.dump(color_meta, f, indent=2)
        logger.info("STAGE1_COLOR_META",
                   f"Saved color space metadata: {color_meta_path}")
    except Exception:
        pass

    # FIX-R11-3: Black bar detection + crop per v7.32 §3.10.3 / v6.32 §7.26.
    # Detect/crop before processing; restore at final output. Effective area
    # <50% or w/h <240px → FAILED (terminate) per spec.
    crop_meta = None
    if params.get("black_bar_detection", True):
        cropped_path, crop_meta = detect_and_crop_black_bars(
            working_input, params, logger, out_dir=temp_dir)
        if crop_meta and crop_meta.get("failed"):
            logger.error("STAGE1_BLACKBAR_FAIL",
                         f"Black bar crop FAILED: {crop_meta.get('reason')} "
                         f"(effective={crop_meta.get('effective')}, "
                         f"original={crop_meta.get('original')}) per v7.32 §3.10.3")
            return PHOTO_ERR_OUTPUT  # 86 validation failure
        if cropped_path and cropped_path != working_input:
            working_input = cropped_path

    # FIX-P11-4: Channel degradation detection BEFORE destructive repairs
    # v7.32 §3.10.6: 前置补偿 in stage1 传统修复前
    # ORIGINAL: channel degradation detection was after all repairs (line 667-695)
    degradation = detect_channel_degradation(working_input, params)

    # Save channel metadata
    # FIX-R11-1: Persist degraded_channels list per v7.32 §3.10.6.
    channel_meta = {
        "channel_degradation": degradation.get("degraded", False),
        "mean_diff": degradation.get("mean_diff", 0.0),
        "std_diff": degradation.get("std_diff", 0.0),
        "degraded_channels": degradation.get("channels", []),
    }
    meta_path = str(Path(temp_dir) /
        f"{args.job_id}_{safe_input_stem(input_name)}_channel_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(channel_meta, f, indent=2)
    logger.info("STAGE1_CHANNEL_META", f"Saved channel metadata: {meta_path}")

    if degradation["degraded"]:
        ai_enabled = params.get("ai_enabled", False)
        # v7.32 §3.10.6: 前置补偿 condition
        mean_severe = degradation.get("mean_diff", 0) > params.get(
            "channel_degradation_mean_threshold_severe", 0.15)
        std_severe = degradation.get("std_diff", 0) > params.get(
            "channel_degradation_std_threshold_severe", 0.25)
        if not ai_enabled and (mean_severe or std_severe):
            logger.info("STAGE1_CHANNEL_CORR",
                       "Applying pre-compensation before repairs")
            corr_out = working_input + ".channel.tif"
            if apply_channel_correction(working_input, corr_out, params, logger):
                if Path(corr_out).exists():
                    working_input = corr_out

    # Convert to 16bit TIFF for processing
    working_path = str(Path(temp_dir) /
        f"{args.job_id}_{safe_input_stem(input_name)}_16bit.tif")

    # FIX-2026-10-20（G6 实跑挖出，R16 修法没覆盖到的一角）：
    # G'MIC `-o X.tif,uint16` 只改**容器位深**，不拉伸值域 —— 8-bit 输入转出去
    # 仍是 0-255 装在 uint16 里（R16 记录的"名实不符"）。
    # R16 的修法（media_io.load_image_rgb_uint8 逐通道自适应）覆盖了 Python 侧
    # **所有读取方**，但覆盖不到 G'MIC 自己：紧接着的 denoise_haar /
    # denoise_patchpca 都按真正的 16-bit 尺度运算这个文件，产出**混合值域**
    # 数据（实测 deblock 后 ch0/ch2 仍是 0-244 而 ch1 冲到 65532）。
    # 最终 avifenc 拿到"16-bit 容器里装着 0-255"的图 → 成品纯黑。
    # 实测：普通 sRGB JPEG 与 ProPhoto TIFF **都中招**，全链 rc 一直是 0，
    # 契约、CICP、头部核验全部正常，只有像素是黑的。
    # 根因修法：8-bit 量级输入在转档时就把值域真正拉伸到 16-bit。
    _stretch: List[str] = []
    try:
        import media_io as _mio
        _arr, _ = _mio.read_image_array(working_input)
        if _arr is not None and int(_arr.max()) <= 255:
            _stretch = ["-mul", "257"]
            logger.info("STAGE1_BITSTRETCH",
                        "Source is 8-bit range; stretching to real 16-bit "
                        "before G'MIC processing (value range, not just "
                        "container depth)",
                        src_max=int(_arr.max()))
    except Exception as _bse:
        # 读不出就不拉伸（保持原行为），但必须留痕 —— 静默跳过会重现黑图
        logger.warning("STAGE1_BITSTRETCH_SKIPPED",
                       f"Could not determine source value range: {_bse}")

    # BUG-049 fix: Explicitly check gmic returncode
    try:
        # R4：`-type ushort` 在 G'MIC 4.x 已移除，改为按版本选择 16-bit 输出写法
        cmd = ["gmic", working_input] + _stretch + _gmic_16bit_output_args(working_path)
        r = subprocess.run(cmd, capture_output=True, timeout=120, shell=False)
        if r.returncode != 0:
            stderr = (r.stderr or b"").decode("utf-8", errors="replace")[:500]
            logger.error("STAGE1_CONVERT_FAIL", f"gmic exited with code {r.returncode}: {stderr}")
            return PHOTO_ERR_GMIC_FAIL
    except Exception as e:
        logger.error("STAGE1_CONVERT_FAIL", str(e))
        return PHOTO_ERR_COLOR_CONV

    # FIX-P11-3: Grain extraction BEFORE destructive repairs per v6.32 §4.2.1
    # "颗粒保护前置：去划痕/去噪前，若需保留原生胶片颗粒，先用频率分离将高频颗粒层隔离"
    # ORIGINAL: grain extraction was after scratch removal (line 609-613), wrong order
    if params.get("grain_synthesis", True):
        grain_out = str(Path(temp_dir) /
            f"{args.job_id}_{safe_input_stem(input_name)}_grain.tif")
        extract_grain_pattern(working_path, grain_out, params, logger)

    # Scratch removal
    if params.get("scratch_remove", True):
        logger.info("STAGE1_SCRATCH", "Removing scratches")
        scratch_out = working_path + ".scratch.tif"
        # FIX-A1: G'MIC stdlib has no -fx_remove_scratches/-remove_scratches/
        # -despeckle commands. Use real stdlib command -denoise_haar (Haar
        # wavelet denoise) as scratch/isolated-defect approximation.
        # scratch_remove_strength(0-100) → denoise_haar threshold(0.5-2.5)
        # FIX-R11-10: Validate 0-100 range per v7.32 §3.10.16.
        try:
            scratch_strength = int(params.get("scratch_remove_strength", 20))
        except (TypeError, ValueError):
            logger.error("STAGE1_SCRATCH_RANGE",
                         f"scratch_remove_strength must be int 0-100, got "
                         f"{params.get('scratch_remove_strength')!r}")
            return PHOTO_ERR_OUTPUT  # 86 validation failure
        if not 0 <= scratch_strength <= 100:
            logger.error("STAGE1_SCRATCH_RANGE",
                         f"scratch_remove_strength={scratch_strength} "
                         f"out of [0,100] per v7.32 §3.10.16")
            return PHOTO_ERR_OUTPUT  # 86 validation failure
        haar_thresh = 0.5 + (max(0, min(100, scratch_strength)) / 100.0) * 2.0
        success, stderr = run_gmic(working_path, scratch_out,
                                   "-denoise_haar", f"{haar_thresh:.3f}",
                                   params=params)
        if not success:
            logger.warning("STAGE1_SCRATCH_CMD",
                          f"G'MIC -denoise_haar failed: {stderr[:200]}")

        if success and Path(scratch_out).exists():
            working_path = scratch_out
        else:
            logger.error("STAGE1_SCRATCH_FAIL",
                        f"G'MIC scratch removal failed: {stderr}")
            return PHOTO_ERR_GMIC_FAIL

    # Deblock (JPEG-like block artifacts)
    # FIX-R11-9: Unify default — guard and value read from the SAME variable
    # (default 25). ORIGINAL: guard used default 0, value used default 25,
    # so absent-config silently disabled deblock while callers assumed 25.
    deblock_strength = params.get("deblock_strength", 25)
    if deblock_strength > 0:
        logger.info("STAGE1_DEBLOCK", "Deblocking")
        deblock_out = working_path + ".deblock.tif"
        # FIX-A1: G'MIC stdlib has no -deblock. Use real stdlib command
        # -denoise_patchpca (patch-PCA denoise) which targets block/patch
        # artifacts. deblock_strength(0-100) → threshold(0.5-3.0)
        pca_thresh = 0.5 + (max(0, min(100, deblock_strength)) / 100.0) * 2.5
        success, stderr = run_gmic(working_path, deblock_out,
                          "-denoise_patchpca", f"{pca_thresh:.3f}",
                          params=params)
        if success and Path(deblock_out).exists():
            working_path = deblock_out
        else:
            # **降级跳过，不阻断整张照片**。
            #
            # 原来的行为是 `return PHOTO_ERR_GMIC_FAIL` —— 一个**可选增强**
            # 失败，导致整张照片处理失败。实测发现 G'MIC 4.0.6 的
            # `denoise_patchpca` 在我们测过的**所有**输入上都会崩：
            #
            #   | 输入 | 结果 |
            #   |---|---|
            #   | 1600x1200 16bit 3ch | Invalid vector read |
            #   | 1600x1200 8bit  3ch | 同上 |
            #   |  400x300  16bit 3ch | 同上 |
            #   | 1600x1200 8bit  1ch | 同上 |
            #   | 官方示例语法 `,`     | 同上 |
            #
            # 报错是 `Error in ./denoise_patchpca/*foreach/ *** Command 'f':
            # [instance(...)] gmic<float32>: Operator 'vector[ind]':
            # Invalid vector read` —— G'MIC 内部越界，与我们的参数无关。
            #
            # 所以这里记 warning 后继续：deblock 只是"锦上添花"，
            # 没有它照片仍然可以完成整套修复（后面的 scratch/deblock 强度
            # 由配置控制，用户也可以直接把 deblock_strength 设 0）。
            logger.warning(
                "STAGE1_DEBLOCK_SKIP",
                "G'MIC denoise_patchpca failed; continuing WITHOUT deblock. "
                "Known issue: this command crashes inside G'MIC 4.0.6 "
                "('Invalid vector read') on every input shape we tested "
                "(8/16-bit, 1/3 channels, 400x300..1600x1200) and even with "
                "the upstream example syntax. Set deblock_strength=0 to "
                "avoid the wasted attempt.",
                strength=deblock_strength,
                threshold=round(pca_thresh, 3),
                stderr_tail=(stderr or "")[-400:])

    # Deband
    # FIX-R11-9: Unify default — guard and value read from the SAME variable
    # (default 1.2). ORIGINAL: guard used default 0, value used default 1.2.
    deband_threshold = params.get("deband_threshold", 1.2)
    if deband_threshold > 0:
        logger.info("STAGE1_DEBAND", "Debanding")
        deband_out = working_path + ".deband.tif"
        # FIX-A1: G'MIC stdlib has no -deband. Use real stdlib command
        # -denoise_haar (light smoothing) for banding. deband_threshold
        # (default 1.2) maps directly to denoise_haar threshold.
        success, stderr = run_gmic(working_path, deband_out,
                          "-denoise_haar",
                          str(deband_threshold),
                          params=params)
        if success and Path(deband_out).exists():
            working_path = deband_out
        elif not success:
            logger.error("STAGE1_DEBAND_FAIL", f"G'MIC -denoise_haar (deband) failed: {stderr}")
            return PHOTO_ERR_GMIC_FAIL

    # Grain extraction moved BEFORE scratch removal per v6.32 §4.2.1 (FIX-P11-3)
    # Original location was here (after deband), now extracted earlier

    # AI/artificial coloring detection (v6.3 §4.2.1)
    # FIX-P11-6: Pass params for config-driven thresholds
    coloring_info = detect_ai_coloring(working_path, params, logger)
    if coloring_info["detected"]:
        logger.warning("STAGE1_AI_COLORING",
                      f"Possible AI/artificial coloring detected (confidence={coloring_info['confidence']:.2f})")

    # NOTE: Channel degradation detection moved BEFORE repairs per v7.32 §3.10.6 (FIX-P11-4)
    # See Step 0.5 above

    # FIX-P11-10: Use shutil.copyfile (not copy2) per v7.32 §3.10.10
    # copy2 copies metadata which causes PermissionError on WSL2 NTFS mounts
    # with metadata=disabled. copyfile only copies data content.
    # ORIGINAL: shutil.copy2(working_path, inter_path)
    # FIX-R11-3: Restore black bars at final output per v7.32 §3.10.3
    # (paste processed crop back into original frame with bit-depth black
    # level). When no crop occurred, fall back to direct copy + size check.
    if crop_meta and crop_meta.get("cropped"):
        if not restore_black_bars(working_path, crop_meta, inter_path,
                                   params, logger):
            logger.warning("STAGE1_BLACKBAR_RESTORE_FALLBACK",
                           "restore_black_bars failed, copying cropped output")
            shutil.copyfile(working_path, inter_path)
    else:
        shutil.copyfile(working_path, inter_path)
        # Verify copy integrity (only meaningful for the no-crop direct copy)
        # FIX-P10-11/P11-8: Use Path consistently per CLAUDE.md §4
        if Path(working_path).stat().st_size != inter_path_p.stat().st_size:
            logger.error("STAGE1_COPY_FAIL",
                        "Intermediate file size mismatch after copy")
            return PHOTO_ERR_COPY_FAIL
    # Sync mtime/atime manually (equivalent to copy2 metadata preservation)
    # FIX-P11-10: Use Path.stat() per CLAUDE.md §4 (not os.stat)
    try:
        src_stat = Path(working_path).stat()
        os.utime(inter_path, (src_stat.st_atime, src_stat.st_mtime))
    except OSError:
        pass  # Non-critical: mtime sync failure on WSL2 cross-filesystem

    if inter_path_p.exists():
        # FIX-P11-8: Cleanup intermediates if requested per v7.32 §3.10.17
        if args.cleanup_intermediates:
            stem = f"{args.job_id}_{safe_input_stem(input_name)}"
            for suffix in [".channel.tif", ".scratch.tif", ".deblock.tif",
                           ".deband.tif", "_grain.tif", "_rgb.tif",
                           "_16bit.tif", "_color_meta.json",
                           "_channel_meta.json", "_crop.tif",
                           "_rgb_crop.tif"]:
                p = Path(temp_dir) / f"{stem}{suffix}"
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
        logger.info("STAGE1_DONE", "Preprocessing complete", output=inter_path)
        return 0
    else:
        logger.error("STAGE1_FAILED", "Output not created")
        return PHOTO_ERR_OUTPUT


if __name__ == "__main__":
    # FIX-P11-5: main() must return explicit int exit code per v7.32 §3.5
    # ORIGINAL: sys.exit(main() or 0) — "or 0" masks None returns
    rc = main()
    if rc is None:
        rc = 86  # validation failure — main() must always return explicit exit code
    sys.exit(rc)
