#!/usr/bin/env python3
"""
Stage 4: Post-processing after AI Upscale
Anti-aliasing and final video processing.

v6.3 Compliance:
- Uses standard FFmpeg filters (gblur/unsharp)
- Output format: FFV1 YUV420P16 (per v6.3 §3.6)
- Deterministic intermediate matching

P8 Fixes Applied:
- P8-2: Match _forced_upscale suffix per v7.32 §3.10.2 L3
- P8-4: Skin/hair protection in selective processing per v6.32 §3
- P8-5: WSL2 temp_dir redirect per v6.32 §7.3
- P8-8: Enforce checkpoint.compute_param_hash (no local fallback)
- P8-9: Removed unsharp from textured regions per v6.32 §3
- P8-10: Saturation/contrast conditional on config per v6.32 §3
- P8-13: Validate stage3 output with ffprobe per v6.32 §7.1
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402

try:  # NEW-V-06: 取消/中断守卫（KeyboardInterrupt 继承 BaseException，
    # 本文件所有 `except Exception` 都抓不到它——不装守卫时 Ctrl+C 会把
    # 半截 mkv 留在 temp 里被下游 stage 复用）
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

    # NEW-V-06: 注册取消守卫（video main.py 认 rc=88 = EXIT_INTERRUPTED）。
    if _cancel_guard is not None:
        _cancel_guard.install_cancel_guard(stage=4, logger=logger)

    # FIX-P8-5: WSL2 temp_dir redirect per v6.32 §7.3
    # ORIGINAL: No WSL2 redirect, causing I/O bottleneck on /mnt/ paths
    temp_dir = params.get("temp_dir", "./temp")
    try:
        from checkpoint import wsl_redirect_path
        _, _, temp_dir, _ = wsl_redirect_path("", temp_dir=temp_dir,
                                               logger=logger)
    except Exception:
        pass
    job_id = args.job_id
    input_name = Path(args.input).name

    # Find stage3 output - deterministic by mtime
    # FIX-P8-2: Match both step3_upscale and step3_forced_upscale per v7.32 §3.10.2 L3
    # ORIGINAL: Only matched step3_upscale*, missing _forced_upscale suffix
    # R3-01（关键修复）：两个通配都会命中 sidecar
    # （`...mkv.identity.json` 写在媒体之后、mtime 更晚），裸 sort 会把 .json
    # 选成 stage3 输入，ffprobe 直接 STAGE4_INPUT_UNDECODABLE (82)。
    # 统一走 checkpoint 媒体白名单 + mtime 新→旧。
    from checkpoint import select_media_candidates

    # ID-04（第七轮）：**精确产物优先**（与 stage5 同一约定）。
    # main 已验证过的那份路径通过 params["upstream_artifact"] 传下来；
    # 有它就绝不再 glob 重选，避免"验证 A、消费 B"。
    _pinned = str(params.get("upstream_artifact") or "")
    if _pinned:
        if not os.path.exists(_pinned):
            logger.error("STAGE4_PINNED_INPUT_MISSING",
                         "Orchestrator pinned an upstream artifact that does "
                         "not exist; refusing to silently glob for another "
                         "file", pinned=_pinned)
            return 82
        candidates = [_pinned]
        logger.info("STAGE4_INPUT_PINNED",
                    "Using the exact upstream artifact handed down by the "
                    "orchestrator", input=_pinned)
    else:
        pattern_upscale = f"{job_id}_{Path(input_name).stem}_step3_upscale*"
        pattern_forced = f"{job_id}_{Path(input_name).stem}_step3_forced_upscale*"
        candidates = select_media_candidates(
            list(Path(temp_dir).glob(pattern_upscale)) +
            list(Path(temp_dir).glob(pattern_forced)))
    # P8-11 fallback A: if no stage3 output, look for stage2 output
    if not candidates:
        pattern2 = f"{job_id}_{Path(input_name).stem}_step2_traditional*"
        candidates2 = select_media_candidates(
            Path(temp_dir).glob(pattern2))
        if candidates2:
            stage3_output = str(candidates2[0])
            logger.info("STAGE4_FALLBACK",
                        "No stage3 output, using stage2 output",
                        input=stage3_output)
        else:
            logger.info("STAGE4_SKIP",
                        "No AI upscale or traditional output")
            return 3
    else:
        stage3_output = str(candidates[0])

    # FIX-P8-13: Validate stage3 output with ffprobe per v6.32 §7.1
    # ORIGINAL: No validation, could process undecodable input
    # FIX-R8-W9: explicit size>0 check per v6.32 §7.1 (file existence + size>0
    # + ffprobe decodable). ORIGINAL only checked ffprobe returncode.
    stage3_path = Path(stage3_output)
    if not stage3_path.exists() or stage3_path.stat().st_size == 0:
        logger.error("STAGE4_INPUT_EMPTY",
                     f"Stage 3 output missing or empty: {stage3_output}")
        return 82
    try:
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error", stage3_output],
            capture_output=True, timeout=30, shell=False
        )
        if r.returncode != 0:
            logger.error("STAGE4_INPUT_UNDECODABLE",
                         "Stage 3 output undecodable")
            return 82
    except Exception as e:
        logger.error("STAGE4_INPUT_CHECK_FAIL", str(e))
        return 82

    # P8-7: Compute param_hash for intermediate filename
    # FIX-P8-8: Enforce checkpoint.compute_param_hash per v7.32 §3.10.19
    # ORIGINAL: Local fallback used SHA-256 without file content hash,
    #           silently degrading param_hash quality
    # checkpoint is a hard dependency - fail fast if not available
    from checkpoint import compute_param_hash

    core_params = {
        "color_lift_r": params.get("color_lift_r", 0.02),
        "color_lift_g": params.get("color_lift_g", 0.01),
        "color_lift_b": params.get("color_lift_b", 0.01),
        "color_gamma_r": params.get("color_gamma_r", 0.98),
        "color_gamma_g": params.get("color_gamma_g", 0.98),
        "color_gamma_b": params.get("color_gamma_b", 0.98),
        "color_gain_r": params.get("color_gain_r", 1.02),
        "color_gain_g": params.get("color_gain_g", 1.02),
        "color_gain_b": params.get("color_gain_b", 1.02),
        "vs_bit_depth": params.get("vs_bit_depth", 16),
    }
    param_hash = compute_param_hash(stage3_output, core_params)

    # FIX-P8-10: Use Path instead of os.path.join
    output_path = str(
        Path(temp_dir) /
        f"{job_id}_{Path(input_name).stem}"
        f"_step4_postprocess_p{param_hash}.mkv"
    )

    if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
        logger.info("STAGE4_SKIP", "Post-process already done")
        return 3

    # NEW-V-06: 写任何字节之前登记产物——取消时删掉半截 mkv，
    # 否则上面的 exists()>0 缓存检查会把它当成功产物复用。
    if _cancel_guard is not None:
        _cancel_guard.watch_outputs(output_path, logger=logger, stage=4)

    logger.info("STAGE4_START", "Post-processing", input=stage3_output)

    # ISSUE-V-S4-001 / ISSUE-V-S4-002 fix:
    # Selective local processing with gradient-variance region split +
    # feathered transition + final lift/gamma/gain color grading.
    # Per v6.32 / v7.32: flat regions -> mild gblur; textured -> preserve;
    # skin/hair -> completely excluded; feathered mask >=8px;
    # colorbalance for lift/gamma/gain.
    # FIX-P1-2: FFmpeg colorbalance rs/rm/rh are ADDITIVE OFFSETS in [-1,1]
    # (not multipliers). Config keys keep lift/gamma/gain multiplier-style
    # names for user familiarity, but values are mapped to offsets here:
    # offset = (multiplier - 1.0), clamped to [-1,1]. This prevents out-of-
    # range values (e.g. gain=1.02 → +0.02 offset, not 1.02 which exceeds
    # [-1,1] and would be rejected by FFmpeg).
    def _to_offset(v: float) -> float:
        return max(-1.0, min(1.0, float(v) - 1.0))
    lift_r = _to_offset(params.get("color_lift_r", 1.02))   # shadows offset
    lift_g = _to_offset(params.get("color_lift_g", 1.01))
    lift_b = _to_offset(params.get("color_lift_b", 1.01))
    gamma_r = _to_offset(params.get("color_gamma_r", 0.98))  # midtones offset
    gamma_g = _to_offset(params.get("color_gamma_g", 0.98))
    gamma_b = _to_offset(params.get("color_gamma_b", 0.98))
    gain_r = _to_offset(params.get("color_gain_r", 1.02))    # highlights offset
    gain_g = _to_offset(params.get("color_gain_g", 1.02))
    gain_b = _to_offset(params.get("color_gain_b", 1.02))

    # FIX-P8-4: Selective local processing per v6.32 §3
    # ORIGINAL: No skin/hair protection; unsharp applied to textured regions
    # Per v6.32 §3: "Human subjects, skin, hair regions are COMPLETELY
    # EXCLUDED from the mask." Flat regions: mild gblur denoise.
    # Textured regions: preserve original (no sharpening per spec).
    # FIX-R8-W8 (方案B): hair is high-frequency texture — edgedetect classifies
    # it as textured → enters the copy (preserve) branch, so it is implicitly
    # protected from gblur. Explicit hair segmentation (OpenCV/face parsing)
    # is deferred pending user decision on precision requirements (round_8 §5).
    filter_chain = (
        # FIX-R6-04: 多输入滤镜图必须走 filter_complex（-vf 只支持单入单出）。
        # 原链 `split[main][tmp]` 之后 `[main]` 被引用 4 次，但 split 只产出
        # 一份 main；[main] 还是 -vf 专属的输入别名，-filter_complex 下
        # 会报 "Invalid stream specifier: main"。实测（2026-10-18）该选择性
        # 处理图从未成功执行过：先卡 geq 选项逗号语法，修好后又报
        # "had 4 input(s) and 1 output(s)"。标准写法：[0:v]split=5 出 5 份，
        # 各分支独立标签，整图末尾统一 [vout] 供 -map 使用。
        "[0:v]split=5[main1][main2][main3][main4][tmp];"
        # Edge-based region mask (detect flat vs textured)
        "[tmp]edgedetect=mode=colormix:low=0.05:high=0.15,"
        "format=gray,"
        "geq=lum='if(lt(lum(X,Y),30),0,"
        "if(gt(lum(X,Y),90),255,"
        "255*(lum(X,Y)-30)/60))'[maskraw];"
        # Skin-tone detection mask (YCbCr skin range)
        # Per v6.32 §3: protect human subjects from any processing
        # FIX-R6-04: geq 的命名选项必须用 ':' 分隔（lum/cb/cr 是同一滤镜的
        # 三个选项）。原来用 ',' 会被解析成新滤镜 cb='128' → ffmpeg 报
        # "No option name near '128'"。
        "[main1]format=yuv420p,"
        "geq=lum='if(between(cb(X,Y),77,127)*between(cr(X,Y),133,173)"
        "*gt(lum(X,Y),80),255,0)':"
        "cb='128':cr='128'[skinmask];"
        # FIX-R8-W8 (方案A): explicit hair mask per v6.32 §3 ("人物主体、皮肤、
        # 头发区域完全排除"). Hair is typically dark (Y 16-80) but not pure black
        # (avoid protecting black borders). Blended with skin mask via screen
        # (union) so the person mask covers both skin and hair. ORIGINAL
        # (方案B) relied on edgedetect classifying hair as textured -> copy
        # branch, which protects it implicitly but fails on low-edge dark hair.
        "[main2]format=yuv420p,"
        "geq=lum='if(gt(lum(X,Y),16)*lt(lum(X,Y),80),255,0)':"
        "cb='128':cr='128'[hairmask];"
        "[skinmask][hairmask]blend=all_mode=screen[personmask];"
        # Invert: person(=skin+hair)=0 (protect), non-person=255 (allow)
        "[personmask]negate[noskin];"
        # Combine: flat non-skin areas only
        "[maskraw][noskin]blend=all_mode=multiply[combinedmask];"
        # Feathered mask via gaussian blur (sigma=6 -> ~12px, >=8px per spec)
        "[combinedmask]gblur=sigma=6[mask];"
        # Flat branch: mild denoise for smooth regions
        "[main3]gblur=sigma=0.6[flat];"
        # Detail branch: preserve original (no unsharp per v6.32 §3)
        # FIX-P8-9: Removed unsharp; textured regions kept as-is
        "[main4]copy[detail];"
        # Masked merge with feathered transition
        "[flat][detail][mask]maskedmerge,"
        # Final color grading: lift/gamma/gain via colorbalance
        f"colorbalance=rs={lift_r}:gs={lift_g}:bs={lift_b}:"
        f"rm={gamma_r}:gm={gamma_g}:bm={gamma_b}:"
        f"rh={gain_r}:gh={gain_g}:bh={gain_b}"
    )

    # FIX-P8-10: Only apply saturation/contrast if explicitly configured
    # ORIGINAL: Always applied eq=saturation=1.05:contrast=1.02
    # Per v6.32 §3: no implicit saturation/contrast changes
    saturation = params.get("post_saturation", 1.0)
    contrast = params.get("post_contrast", 1.0)
    if saturation != 1.0 or contrast != 1.0:
        filter_chain += (
            f",eq=saturation={saturation}:contrast={contrast}"
        )
    # FIX-R6-04: 整图输出标签（filter_complex + -map [vout]）
    filter_chain += "[vout]"

    # v6.3 §3.6: Intermediate format must be FFV1 YUV420P16
    vs_bit_depth = params.get("vs_bit_depth", 16)
    if vs_bit_depth != 16:
        logger.warning("STAGE4_BITDEPTH_FORCE",
                       f"vs_bit_depth={vs_bit_depth}, "
                       "forcing 16bit for intermediate")
    pix_fmt = "yuv420p16le"

    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", stage3_output,
        # FIX-R6-04: 多输入滤镜图用 -filter_complex + 显式 -map
        "-filter_complex", filter_chain,
        "-map", "[vout]",
        "-c:v", "ffv1",
        "-pix_fmt", pix_fmt,
        "-an",
        output_path
    ]

    try:
        # NEW-V-06/RV-12: 受管子进程——取消时 ffmpeg 一起收掉，
        # 否则守卫删掉的半截 mkv 会被孤儿 ffmpeg 重新写回来。
        if _cancel_guard is not None:
            r = _cancel_guard.run_managed(cmd, capture_output=True,
                                          text=True, timeout=3600,
                                          shell=False, logger=logger)
        else:  # pragma: no cover - 守卫缺失时退回原行为
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                               timeout=3600, shell=False)
        if (r.returncode == 0 and Path(output_path).exists()
                and Path(output_path).stat().st_size > 0):
            logger.info("STAGE4_DONE", "Post-processing complete",
                        output=output_path)
            # NEW-V-06: 产物已完整落盘，取消守卫不再有权删它（RV-12）。
            if _cancel_guard is not None:
                _cancel_guard.release_outputs()
            return 0
        else:
            logger.error("STAGE4_FAILED",
                         f"Post-processing failed: {r.stderr[:300]}")
            return 84
    except subprocess.TimeoutExpired:
        logger.error("STAGE4_TIMEOUT", "Post-processing timed out")
        return 87
    except Exception as e:
        logger.error("STAGE4_ERROR", str(e))
        return 82


if __name__ == "__main__":
    sys.exit(main() or 0)
