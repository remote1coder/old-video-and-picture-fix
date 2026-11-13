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


def _materialise_step2(src: str, dst: str) -> str:
    """把 stage1 产物挂成 step2 名字，**优先硬链接**。

    R23-LONG-02（2026-11-13，16 分钟 1080p 实跑）

    stage2 是**检查型**阶段：它只跑 idet 判断是否真的逐行、校验帧率，
    一个像素都不改。产物的唯一作用是给下游一个符合 `step2` 命名的路径
    （main 用 `{job}_{stem}_step2*` glob 校验，stage3 也可能 glob 兜底）。

    旧实现用 `shutil.copyfile`，对 16 分钟 1080p 意味着**再写 82 GB**
    （stage1 产物实测 82.4 GB）—— 整条链凭空多出一份全片写入，纯 I/O
    浪费，还多占 82 GB 空间。

    同卷时改用硬链接：**同一 inode 的第二个名字**，毫秒完成、零额外空间。
    下游按名字读、ffprobe 解码、哈希摘要全部照常；删掉 step1 也不会
    影响 step2（数据等两个名字都消失才释放），比复制更安全。

    Returns:
        ``"linked"`` 或 ``"copied"`` —— 让日志能证明到底走了哪条路，
        不要"看起来优化了"却没人能验证。
    """
    import shutil as _shutil

    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        # 这些都属于"本机/本卷就是不支持硬链接"，不是缺陷 —— 老实复制，
        # 行为与改动前完全一致。绝不因为优化失败而让 stage 挂掉：
        #   WinError 1    ERROR_INVALID_FUNCTION    （FAT32/exFAT 等）
        #   WinError 17   ERROR_NOT_SAME_DEVICE     （跨卷）
        #   WinError 1314 ERROR_PRIVILEGE_NOT_HELD  （无权限）
        #   POSIX: EXDEV / EPERM / EOPNOTSUPP
        _shutil.copyfile(src, dst)
        return "copied"


class _NoFileInput(Exception):
    """stream_full 的检查遍没有文件可探 —— 不是错误，是这条路径的正常形态。"""


class _IdetResult:
    __slots__ = ("returncode", "stderr")

    def __init__(self, returncode, stderr=""):
        self.returncode = int(returncode)
        self.stderr = str(stderr or "")


def _run_idet_on_vpy(vpy: str, timeout: float, vs_log_path: str = ""):
    """`vspipe(vpy) | ffmpeg -vf idet -f null -`，返回与 subprocess 同形的结果。

    ## 为什么必须并发排空 vspipe 的 stderr

    stage1 踩过这个坑（见其注释）：给 vspipe 开 PIPE 却不读，缓冲区
    （约 64KB）一满 vspipe 就阻塞、不再出帧，ffmpeg 于是读到无效数据并
    报 "Invalid data found"。所以这里把 stderr 落文件，绝不留在管道里。
    """
    import subprocess
    _err = subprocess.DEVNULL
    _fh = None
    if vs_log_path:
        try:
            _fh = open(vs_log_path, "wb")
            _err = _fh
        except OSError:
            _err = subprocess.DEVNULL
    vp = None
    try:
        vp = subprocess.Popen(["vspipe", "-c", "y4m", vpy, "-"],
                              stdout=subprocess.PIPE, stderr=_err,
                              shell=False)
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "info", "-i", "-",
             "-vf", "idet", "-f", "null", "-"],
            stdin=vp.stdout, capture_output=True, text=True,
            errors="replace", timeout=timeout, shell=False)
        return _IdetResult(r.returncode, r.stderr)
    except subprocess.TimeoutExpired:
        return _IdetResult(124, f"idet on the stage1 script timed out after "
                                f"{timeout:.0f}s")
    except OSError as exc:
        return _IdetResult(127, f"cannot run vspipe/ffmpeg: {exc}")
    finally:
        try:
            if vp is not None and vp.stdout is not None:
                vp.stdout.close()
        except Exception:
            pass
        try:
            if vp is not None:
                vp.wait(timeout=30)
        except Exception:
            try:
                if vp is not None:
                    vp.kill()
            except Exception:
                pass
        try:
            if _fh is not None:
                _fh.close()
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
    # ---- stream_full：检查遍直接吃 stage1 的 vpy（不读 step1 媒体）----
    # 设计 §5.1：检查遍 = 原源 → 同一 stage1 图 → 本阶段检查 → null sink。
    # "同一张图"由**同一个 vpy** 保证；不是"再算一遍碰巧一样"。
    _vpy = ""
    _stream = params.get("_stream_s1") or {}
    if isinstance(_stream, dict) and _stream.get("vs_script"):
        _vpy = str(_stream["vs_script"])
        if not os.path.exists(_vpy):
            logger.error("STAGE2_STREAM_VPY_MISSING",
                         "stream_full asked for a check pass, but the stage1 "
                         "VapourSynth script is gone", vs_script=_vpy)
            return 82
        stage1_output = ""
        logger.info("STAGE2_STREAM_START",
                    "Checking the deinterlace output straight from the stage1 "
                    "script (no step1 media is read or written)",
                    vs_script=_vpy)
    else:
        from checkpoint import select_media_candidates
        candidates = select_media_candidates(
            glob.glob(os.path.join(temp_dir, pattern)))
        if not candidates:
            logger.error("STAGE2_NO_INPUT", "Stage 1 output not found")
            return 82

        stage1_output = str(candidates[0])
    logger.info("STAGE2_START", "Verifying deinterlace output",
                path=stage1_output or _vpy)

    # Check file integrity（stream 模式没有文件可查，改由 vspipe 自己报错）
    if stage1_output and (not os.path.exists(stage1_output)
                          or os.path.getsize(stage1_output) == 0):
        logger.error("STAGE2_EMPTY", "Stage 1 output missing or empty")
        return 86

    # Verify with ffprobe
    try:
        if not stage1_output:
            raise _NoFileInput()
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
    except _NoFileInput:
        pass                     # stream 模式已在上面日志说明
    except Exception as e:
        logger.error("STAGE2_VERIFY_FAIL", f"Verification failed: {e}")
        return 86

    # Check deinterlace quality with idet
    threshold = params.get("deinterlace_progressive_threshold", 95.0)
    # FIX-2026-10-21（16 分钟 1080p 实跑）：**没做去隔行就不该验证去隔行质量**。
    #
    # stage1 在 `field_order == "unknown"` 时会跳过 QTGMC（保守策略，日志
    # STAGE1_UNKNOWN_FIELD_ORDER），vpy 里只有 SetFieldBased(clip, 0)。
    # 而 stage2 仍然跑 idet 门禁 —— 实测该 82GB 产物的元数据明明是
    # `field_order=progressive`、vpy 也显式 SetFieldBased(0)，idet 却判
    # TFF:2742 / Progressive:237（91% 隔行）。
    #
    # 原因是**防抖**：haf.Stab（DepanStabilize）对每帧做亚像素垂直位移，
    # 改变了行间相关性，而 idet 正是靠行间相关性判断隔行的。所以
    # "做过防抖的 progressive 输出"天生会被 idet 误判为隔行。
    #
    # 这个门禁的唯一目的是验证"QTGMC 把隔行去干净了"。既然没做去隔行，
    # 它就不适用 —— 记 INCONCLUSIVE 放行，而不是判失败。
    _fo_for_gate = str(params.get("field_order") or "")
    if _fo_for_gate == "unknown":
        logger.warning("STAGE2_IDET_INCONCLUSIVE",
                       "field_order=unknown: stage1 skipped QTGMC, so there "
                       "is no deinterlace to verify. The gate does not apply",
                       field_order=_fo_for_gate)
        stats = {}
    else:
        try:
            import os as _os
            import subprocess
            import re
            # FIX-2026-10-21（16 分钟 1080p 实跑）：固定 300 秒对长片必然不够 ——
            # idet 要解码整条流，实测 24GB / 28840 帧的 1080p 中间产物直接超时，
            # 而超时被当成"0% progressive"判失败。超时按大小自适应，且超时/报错
            # 一律记 IDET_INCONCLUSIVE（判不出来 ≠ 不合格），与 file segment 的
            # 场序检查保持同一条原则。
            _gb = 0.0
            try:
                if stage1_output:
                    _gb = _os.path.getsize(stage1_output) / float(1 << 30)
            except OSError:
                pass
            # stream 模式没有文件大小可依；按"整片解码"给足时间
            _idet_timeout = (min(3600.0, 300.0 + 25.0 * _gb)
                             if stage1_output else 3600.0)
            if _vpy:
                r = _run_idet_on_vpy(
                    _vpy, _idet_timeout,
                    vs_log_path=os.path.splitext(_vpy)[0] + ".vspipe.log")
            else:
                cmd = [
                    "ffmpeg", "-hide_banner", "-v", "info",
                    "-i", stage1_output,
                    "-vf", "idet",
                    "-f", "null", "-"
                ]
                r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                                  timeout=_idet_timeout, shell=False)
            if r.returncode != 0:
                logger.warning("STAGE2_IDET_INCONCLUSIVE",
                               "idet returned non-zero; the deinterlace gate does "
                               "not apply",
                               returncode=r.returncode, timeout=_idet_timeout)
                stats = {}
            else:
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
                    undet = stats.get("undetermined", 0)
                    undet_pct = undet / total * 100
                    # FIX-2026-10-21（全链路实测 + 联网核实 ffmpeg 语义）：
                    #
                    # `Undetermined` 是 idet 的**"无法分类"** —— 官方
                    # doc/filters.texi 写的是 "Cumulative number of frames that
                    # could not be classified"；vf_idet.c 里它既不满足
                    # TFF/BFF 的 interlace_threshold，也不满足 PROGRESSIVE 的
                    # progressive_threshold。
                    #
                    # 旧实现把 "不是 progressive 的帧" 一并算进分母，于是
                    # **任何 idet 判不出的素材都会被当成不合格误杀**。实测：
                    # 2 秒的逐行片与 2 秒的真隔行（field_order=tt）片，
                    # idet 都输出 TFF:0 BFF:0 Progressive:0 Undetermined:50 /
                    # :25 —— 门禁一律报 Progressive=0.0% 并 return 86。
                    #
                    # 官方判定口径也支持这一点：InterlaceInfo.Type() 只在
                    # Determined > Undetermined 时才下结论，且帧数 <250 直接
                    # 返回 ErrShortStream。
                    #
                    # 因此：Undetermined 占多数 = idet 没给结论，此时应记
                    # "未校验"并放行，而不是判失败。
                    inconclusive = float(params.get(
                        "deinterlace_idet_inconclusive_pct", 50.0))
                    if undet_pct >= inconclusive:
                        logger.warning(
                            "STAGE2_IDET_INCONCLUSIVE",
                            f"idet could not classify {undet_pct:.1f}% of "
                            f"frames (>= {inconclusive}%): the deinterlace "
                            "gate does not apply. Recording as unchecked, "
                            "not as a failure",
                            progressive_pct=round(prog_pct, 1),
                            undetermined_pct=round(undet_pct, 1),
                            frames=total,
                            stats=stats)
                    elif prog_pct >= threshold:
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
        except Exception as e:
            logger.warning("STAGE2_CHECK_FAIL", f"Quality check failed: {e}")

    # FIX-2026-10-21（16 分钟 1080p 实跑）：step2 中间产物的复制**必须无条件执行**。
    #
    # 旧结构把它放在 `if total > 0:` 的成功路径里（末尾 `return 0`），于是
    # idet 超时/报错走 except 时会直接 `return 0` —— **宣称成功却没有产物**，
    # main.py 随后报 `TRADITIONAL_GATE_BLOCK: Stage 2 output file missing`，
    # 整个 job 以 rc=86 结束。实测 24GB 中间产物 idet 300s 超时即命中。
    #
    # 只有"真的低质量并 return 86"那条路才不需要复制（stage1 会被 fallback
    # 重做）。
    #
    # R23-LONG-02：**这里不该 copyfile**。见 `_materialise_step2()` ——
    # stage2 一个像素都不改，产物只是换个名字给下游；同卷硬链接是同一
    # inode 的第二个名字，省掉一次全片写入（16 分钟 1080p = 82 GB）。
    # stream_full：检查遍不产 step2 媒体（设计 §5.2 S2：不创建 step2 媒体）
    if _vpy:
        logger.info("STAGE2_STREAM_NO_ARTIFACT",
                    "stream_full: the check pass produced conclusions only; "
                    "no step2 media was materialised")
        return 0
    try:
        from checkpoint import compute_param_hash
        param_hash = compute_param_hash(stage1_output, params)
        step2_name = (f"{job_id}_{Path(input_name).stem}"
                      f"_step2_traditional_p{param_hash}.mkv")
        step2_path = os.path.join(temp_dir, step2_name)
        if not os.path.exists(step2_path):
            _how = _materialise_step2(stage1_output, step2_path)
            logger.info("STAGE2_ARTIFACT",
                        f"step2 artifact ready ({_how}): {step2_path}",
                        materialise=_how, source=stage1_output)
    except Exception as copy_err:
        logger.warning("STAGE2_COPY_WARN", f"Failed to copy stage1 to step2 path: {copy_err}")

    logger.info("STAGE2_DONE", "Verification passed (basic)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
