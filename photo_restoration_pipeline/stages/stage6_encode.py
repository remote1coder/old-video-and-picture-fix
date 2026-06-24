#!/usr/bin/env python3
"""
Stage 6: Photo Final Encoding
- AVIF encoding with avifenc (10bit YUV420 or 16bit RGB444)
- CICP metadata
- Archival mode: RGB444 lossless when archival_mode=true

v6.3/v7.3 Compliance:
- Supports both avif_params (YUV420 10-bit) and avif_rgb444_params (RGB444 16-bit lossless)
- Grain synthesis moved to stage5_postprocess.py per v7.32 §3.1
- Proper CICP tagging
"""

import os
import sys
import json
import argparse
import glob
import subprocess
import shutil
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from checkpoint import wsl_redirect_path, wsl_copy_back  # noqa: E402


def _get_avifenc_version() -> tuple:
    """Parse avifenc version. Returns (major, minor, patch) or (0, 0, 0)."""
    try:
        r = subprocess.run(["avifenc", "--version"],
                           capture_output=True, text=True, timeout=10, shell=False)
        if r.returncode == 0:
            import re
            # PHOTO-AUDIT-019: Support both X.Y.Z and X.Y version formats
            m = re.search(r'(\d+)\.(\d+)(?:\.(\d+))?', r.stdout)
            if m:
                major = int(m.group(1))
                minor = int(m.group(2))
                patch = int(m.group(3)) if m.group(3) else 0
                return (major, minor, patch)
    except Exception:
        pass
    return (0, 0, 0)


def _resolve_jobs_value(jobs: str) -> Optional[str]:
    """Resolve --jobs value based on avifenc version.

    FIX-P12-10: Per v6.32 §7.29:
    - <0.9.2: --jobs not supported, return None
    - 0.9.2–1.0.x: 'all' not supported, return cpu count
    - >=1.1.0: 'all' keyword supported
    """
    if jobs != "all":
        return str(jobs)
    version = _get_avifenc_version()
    if version >= (1, 1, 0):
        return "all"
    elif version >= (0, 9, 2):
        return str(min(os.cpu_count() or 1, 32))
    else:
        # <0.9.2 does not support --jobs at all
        return None


def _build_avifenc_cmd(input_path: str, output_path: str, params: dict, use_rgb444: bool = False) -> list:
    """Build avifenc command from config parameters.

    When use_rgb444=True, uses avif_rgb444_params (16-bit lossless).
    Otherwise uses avif_params (10-bit YUV420).
    """
    if use_rgb444:
        avif_params = params.get("avif_rgb444_params", {})
        # Defaults for archival RGB444
        depth = avif_params.get("depth", 16)
        yuv = avif_params.get("yuv", "444")
        rng = avif_params.get("range", "limited")
        cicp = avif_params.get("cicp", "1/13/0")
        speed = avif_params.get("speed", 4)
        jobs = avif_params.get("jobs", "all")
        lossless = avif_params.get("lossless", True)
    else:
        avif_params = params.get("avif_params", {})
        depth = avif_params.get("depth", 10)
        yuv = avif_params.get("yuv", "420")
        rng = avif_params.get("range", "limited")
        cicp = avif_params.get("cicp", "1/13/1")
        speed = avif_params.get("speed", 4)
        jobs = avif_params.get("jobs", "all")
        lossless = False

    cmd = [
        "avifenc",
        "--depth", str(depth),
        "--yuv", yuv,
        "--range", rng,
        "--cicp", cicp,
        "--speed", str(speed),
    ]

    if lossless:
        cmd.append("--lossless")

    # FIX-P12-10: --jobs may be None for avifenc <0.9.2
    jobs_val = _resolve_jobs_value(jobs)
    if jobs_val is not None:
        cmd.extend(["--jobs", jobs_val])
    cmd.extend([input_path, output_path])

    return cmd


def _calculate_avifenc_timeout(input_path: str, base_timeout: int = 600) -> int:
    """BUG-057 / ISSUE-P-S6-003: Dynamic timeout based on image pixel count.

    Estimates ~60 seconds per megapixel at speed 4.
    Minimum timeout is base_timeout (600s).
    """
    try:
        from PIL import Image
        with Image.open(input_path) as img:
            w, h = img.size
        mp = (w * h) / 1_000_000
        dynamic = int(mp * 60)
        return max(base_timeout, dynamic)
    except Exception:
        return base_timeout


def run_avifenc(input_path: str, output_path: str, params: dict, use_rgb444: bool = False) -> bool:
    """Run avifenc with proper parameters."""
    cmd = _build_avifenc_cmd(input_path, output_path, params, use_rgb444)
    timeout = _calculate_avifenc_timeout(input_path)
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, shell=False)
        return r.returncode == 0
    except Exception:
        return False


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

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    input_name = Path(args.input).name

    # Bug 11.21 fix: pre-check avifenc existence
    if not shutil.which("avifenc"):
        logger.error("STAGE6_AVIFENC_MISSING", "avifenc not found in PATH")
        # FIX-H12-05: avifenc missing is a dependency-missing failure → exit 80
        # per v7.32 §3.10.18. ORIGINAL: silently saved a PNG fallback and
        # returned 0, masking the missing dependency from the upper layer.
        return 80

    # FIX-L12-16: WSL2 redirect for temp_dir per v7.32 §3.10.10. stage6 reads
    # its input intermediate from temp_dir; if temp_dir is under /mnt/ it must
    # be redirected to a Linux native partition (matching stage1-5) so the
    # glob below finds files written by prior stages. ORIGINAL: only
    # output_path was redirected, so temp_dir globs could miss redirected
    # intermediates and return 82 (NO_INPUT).
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

    # FIX-P10-1: Updated search order for new stage numbering
    # Find input: stage5_postprocess > stage4_face > stage3_upscale > stage2_ai_pre > stage1_preprocess
    input_source = None
    for pat in ["step5_postprocess*", "step4_face*", "step3_upscale*", "step2_ai_pre*", "step1_preprocess*"]:
        pattern = f"{job_id}_{Path(input_name).stem}_{pat}"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            input_source = candidates[0]
            break

    if not input_source:
        # BUG-023 fix: do not silently fall back to raw input; require processed intermediate
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

    working_path = input_source

    # Determine archival mode
    archival_mode = params.get("archival_mode", False)
    use_rgb444 = archival_mode

    def _find_icc_profile(profile_name: str) -> Optional[str]:
        """Search system for ICC profile file."""
        search_paths = []
        if sys.platform == "win32":
            search_paths.append(Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32" / "spool" / "drivers" / "color")
        else:
            search_paths.extend([
                Path("/usr/share/color/icc"),
                Path("/usr/local/share/color/icc"),
                Path.home() / ".local" / "share" / "icc",
                Path.home() / ".color" / "icc",
            ])
        for base in search_paths:
            if not base.exists():
                continue
            for candidate in base.rglob("*.icc"):
                if profile_name.lower() in candidate.name.lower():
                    return str(candidate)
        return None

    # Bug 11.25 fix: also generate 16bit TIFF archive when photo_bit_depth=16
    # BUG-055 fix: Embed Adobe RGB or ProPhoto RGB ICC profile per v6.32 §4.2.6
    bit_depth = params.get("photo_bit_depth", 16)
    tiff_path = None
    if bit_depth == 16:
        tiff_path = output_path.rsplit(".", 1)[0] + ".tif"
        try:
            from PIL import Image
            img = Image.open(working_path)
            if img.mode != "I;16":
                img = img.convert("RGB")
            save_kwargs = {"format": "TIFF", "compression": "tiff_lzw"}
            # Try to embed Adobe RGB ICC profile
            icc_path = _find_icc_profile("Adobe RGB")
            if icc_path and os.path.exists(icc_path):
                with open(icc_path, "rb") as f:
                    save_kwargs["icc_profile"] = f.read()
                logger.info("STAGE6_TIFF_ICC", f"Embedded Adobe RGB ICC: {icc_path}")
            else:
                # Fallback: try ProPhoto RGB
                icc_path = _find_icc_profile("ProPhoto")
                if icc_path and os.path.exists(icc_path):
                    with open(icc_path, "rb") as f:
                        save_kwargs["icc_profile"] = f.read()
                    logger.info("STAGE6_TIFF_ICC", f"Embedded ProPhoto RGB ICC: {icc_path}")
                else:
                    logger.warning("STAGE6_TIFF_ICC",
                                   "No Adobe RGB or ProPhoto RGB ICC profile found on system. "
                                   "TIFF will use source color space. Install ICC profiles for archival accuracy.")
            img.save(tiff_path, **save_kwargs)
            logger.info("STAGE6_TIFF", f"16bit TIFF archive saved: {tiff_path}")
        except Exception as e:
            logger.warning("STAGE6_TIFF_FAIL", f"TIFF archive failed: {e}")

    # AVIF encoding
    logger.info("STAGE6_ENCODE", f"Encoding AVIF (archival={use_rgb444})")
    if run_avifenc(working_path, output_path, params, use_rgb444=use_rgb444):
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
            # Also copyback TIFF if generated
            if tiff_path and os.path.exists(tiff_path):
                orig_tiff = original_output_path.rsplit(".", 1)[0] + ".tif"
                wsl_copy_back(tiff_path, orig_tiff)
        logger.info("STAGE6_DONE", "Photo encoding complete", output=output_path)
        return 0
    else:
        # FIX-H12-05: avifenc encoding failure must return 84 (encoding error)
        # per v7.32 §3.10.18, NOT silently fall back to TIFF/PNG and return 0.
        # ORIGINAL: saved a TIFF/PNG fallback and returned 0, masking the
        # encoding failure from the upper layer (which could never tell the
        # AVIF was not produced). The 16bit TIFF archive (if any) is already
        # generated above before avifenc runs, so archival output survives.
        logger.error("STAGE6_AVIF_FAIL", "avifenc encoding failed")
        return 84


if __name__ == "__main__":
    # FIX-P11-5: main() must return explicit int exit code per v7.32 §3.5
    # ORIGINAL: sys.exit(main() or 0) — "or 0" masks None returns
    rc = main()
    if rc is None:
        rc = 86  # validation failure — main() must always return explicit exit code
    sys.exit(rc)
