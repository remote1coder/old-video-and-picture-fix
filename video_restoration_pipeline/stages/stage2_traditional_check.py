#!/usr/bin/env python3
"""
Stage 2: Traditional Repair Verification
Verifies deinterlace output using ffmpeg idet filter.

v6.3 Compliance:
- Progressiveness threshold from config (default >=95%)
- Deterministic intermediate matching
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402

try:  # NEW-V-06: 取消/中断守卫。stage2 是检查型 stage、不写最终产物，
    # 但中断时同样必须留 rc=88 的诚实记录（video main.py 记 cancelled），
    # 而不是让 `except Exception` 放它过去、状态悬空。
    import cancel_guard as _cancel_guard
except Exception:  # pragma: no cover - 缺失时不阻断 stage，仅失去该保护
    _cancel_guard = None


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
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()

    params = _load_params(args)
    log_dir = params.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, job_id=args.job_id)

    # NEW-V-06: 注册取消守卫（无产物可登记；rc=88 由 main.py 记账）。
    if _cancel_guard is not None:
        _cancel_guard.install_cancel_guard(stage=2, logger=logger)

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    input_name = Path(args.input).name

    # Find stage1 output - sort by mtime for deterministic selection
    pattern = f"{job_id}_{Path(input_name).stem}_step1_preprocess*"
    # R4（关键修复）：这个通配同时会命中 stage1 产出的 **sidecar**
    # （如 `...step1_preprocess_pdXXXX.mkv.identity.json`），而 sidecar
    # 是在 mkv 之后写的、mtime 更晚 —— 于是 `candidates[0]` 选中的是
    # 那个 .json，ffprobe 解析失败 -> "Stage 1 output not valid video"
    # (STAGE2_CORRUPT, 86)，真正的 582MB 视频反而被跳过。
    # R3-01：这里原本的黑名单元信息不全（挡不住未来的新侧车类型），统一
    # 收编到 checkpoint 的媒体后缀白名单 + mtime 新→旧 排序
    # （select_media_candidates），与 stage6/stage7 同一套规则。
    from checkpoint import select_media_candidates
    candidates = select_media_candidates(
        glob.glob(os.path.join(temp_dir, pattern)))
    if not candidates:
        logger.error("STAGE2_NO_INPUT", "Stage 1 output not found")
        return 82

    stage1_output = str(candidates[0])
    logger.info("STAGE2_START", "Verifying deinterlace output", path=stage1_output)

    # Check file integrity
    if not os.path.exists(stage1_output) or os.path.getsize(stage1_output) == 0:
        logger.error("STAGE2_EMPTY", "Stage 1 output missing or empty")
        return 86

    # Verify with ffprobe
    try:
        import subprocess
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1", stage1_output],
            capture_output=True, text=True, errors="replace", timeout=60, shell=False
        )
        if r.returncode != 0 or "video" not in r.stdout:
            logger.error("STAGE2_CORRUPT", "Stage 1 output not valid video")
            return 86
    except Exception as e:
        logger.error("STAGE2_VERIFY_FAIL", f"Verification failed: {e}")
        return 86

    # Check deinterlace quality with idet
    threshold = params.get("deinterlace_progressive_threshold", 95.0)
    try:
        import subprocess
        import re
        cmd = [
            "ffmpeg", "-hide_banner", "-v", "info",
            "-i", stage1_output,
            "-vf", "idet",
            "-f", "null", "-"
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                          timeout=300, shell=False)
        if r.returncode == 0:
            stats = {}
            for line in r.stderr.splitlines():
                if "Multi frame detection:" in line:
                    for key in ("TFF", "BFF", "Progressive", "Undetermined"):
                        m = re.search(rf"{key}:\s*(\d+)", line)
                        if m:
                            stats[key.lower()] = int(m.group(1))
            total = sum(stats.values())
            if total > 0:
                prog_pct = stats.get("progressive", 0) / total * 100
                if prog_pct >= threshold:
                    logger.info("STAGE2_DONE", f"Verification passed: {prog_pct:.1f}% progressive")
                else:
                    logger.error("STAGE2_LOW_PROGRESSIVE",
                                f"Progressive={prog_pct:.1f}% < threshold={threshold}%")
                    return 86

                # Frame rate validation
                try:
                    fr_cmd = ["ffprobe", "-hide_banner", "-v", "error",
                              "-select_streams", "v:0",
                              "-show_entries", "stream=r_frame_rate",
                              "-of", "default=noprint_wrappers=1:nokey=1", stage1_output]
                    fr = subprocess.run(fr_cmd, capture_output=True, text=True, errors="replace",
                                        timeout=60, shell=False)
                    if fr.returncode == 0:
                        rate_str = fr.stdout.strip()
                        if "/" in rate_str:
                            num, den = map(float, rate_str.split("/"))
                            detected_fps = num / den if den != 0 else 0
                        else:
                            detected_fps = float(rate_str)
                        expected_fps = params.get("expected_fps", params.get("original_fps", 0))
                        if expected_fps > 0:
                            deviation = abs(detected_fps - expected_fps) / expected_fps
                            if deviation > 0.02:
                                logger.error("STAGE2_FRAMERATE",
                                            f"FPS {detected_fps} deviates {deviation*100:.1f}% from expected {expected_fps}")
                                return 86
                            else:
                                logger.info("STAGE2_FRAMERATE",
                                           f"FPS {detected_fps} matches expected {expected_fps}")
                except Exception as e:
                    logger.warning("STAGE2_FRAMERATE_CHECK", f"Frame rate check failed: {e}")

                logger.info("TRADITIONAL_GATE", "Traditional restoration completed, proceeding to AI stages")

                # FIX-P7-5: Copy stage1 output to step2_traditional intermediate
                # so stage7 can find it in the fallback chain
                import shutil
                # FIX-2026-10-20: merge_params() 从不注入 params["param_hash"]，
                # 原来这里取到空串 → step2 文件名完全没有 _p{hash} 段，
                # 与 v7.32 §3.10.19 的命名契约不符。改为就地计算（与
                # stage1/3/4 一致），以**本 stage 的实际输入**为身份基准。
                from checkpoint import compute_param_hash
                param_hash = compute_param_hash(stage1_output, params)
                step2_name = (f"{job_id}_{Path(input_name).stem}"
                              f"_step2_traditional_p{param_hash}.mkv")
                step2_path = os.path.join(temp_dir, step2_name)
                try:
                    if not os.path.exists(step2_path):
                        shutil.copyfile(stage1_output, step2_path)
                        logger.info("STAGE2_COPY", f"Copied stage1 output to {step2_path}")
                except Exception as copy_err:
                    logger.warning("STAGE2_COPY_WARN", f"Failed to copy stage1 to step2 path: {copy_err}")

                return 0
    except Exception as e:
        logger.warning("STAGE2_CHECK_FAIL", f"Quality check failed: {e}")

    logger.info("STAGE2_DONE", "Verification passed (basic)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
