#!/usr/bin/env python3
"""
Stage 7: Final Encoding
- SVT-AV1 10-bit encoding with CRF
- FLAC audio
- Original metadata preserved
- SRT subtitle embedding
- SVT-AV1 tile settings

v6.3 Compliance:
- SVT-AV1 tile 2x2 or 1x1
- audio hum_energy_threshold used
- Deterministic intermediate matching
"""

import sys
import json
import argparse
import subprocess
import datetime
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from checkpoint import (wsl_redirect_path, wsl_copy_back,
                        select_media_candidates)  # noqa: E402

try:  # NEW-V-06: 取消/中断守卫（KeyboardInterrupt 继承 BaseException，
    # 本文件所有 `except Exception` 都抓不到它——不装守卫时 Ctrl+C 会把
    # 半截成品留在最终产物路径上，与成功产物在文件系统层面无法区分）
    import cancel_guard as _cancel_guard
except Exception:  # pragma: no cover - 缺失时不阻断 stage，仅失去该保护
    _cancel_guard = None


def _get_audio_layout(audio_path: str) -> dict:
    """Get audio channel layout info using ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error",
             "-select_streams", "a:0",
             "-show_entries", "stream=channels,channel_layout",
             "-of", "json", audio_path],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            streams = data.get("streams", [])
            if streams:
                return {
                    "channels": int(streams[0].get("channels", 2)),
                    "layout": streams[0].get("channel_layout", "stereo"),
                }
    except Exception:
        pass
    return {"channels": 2, "layout": "stereo"}


#: 逐声道 arnndn 支持的声道数上限。超过这个数（罕见的多声道素材）退回
#: afftdn，因为每组都要起一个 arnndn 实例，实例数与声道数成正比。
_MAX_PER_CHANNEL_ARNNDN = 8


class AudioDegradeNotAllowed(Exception):
    """FIX-R3-08 补完：明确要求的音频算法不可用，且未授权降级。

    与 R3-08 / B12 同判据——"明确要求的算法不可用"必须让阶段失败，
    不能只记一条 WARNING 然后照样 rc=0。
    """


def _arnndn_filters(model: str, audio_layout: dict, escape,
                    logger=None, allow_degrade: bool = True) -> List[str]:
    """R2-10: build the arnndn filter chain **without merging channels**.

    ``allow_degrade``: 是否授权"降级到 afftdn"。仅当声道数
    ``> _MAX_PER_CHANNEL_ARNNDN`` 时才涉及降级决策。

    * 用户**明确要求** arnndn（``choice == "arnndn"``）时，调用方应传入
      ``params.get("allow_audio_degrade", False)``——未授权则抛
      ``AudioDegradeNotAllowed``，由调用方转成阶段失败（与 R3-08 / B12
      同判据：明确要求的算法不可用必须失败，不能静默降级）。
    * ``choice == "auto"`` 的语义本就是"有 arnndn 就用、没有就降级"，
      因此传 ``True``。

    旧实现（``aformat=channel_layouts=mono`` → arnndn → ``aformat=...stereo``）
    会把左右声道**下混成同一个单声道**再复制回去，左右差异永久丢失。
    实测（L=440Hz / R=880Hz 的立体声，本机 ffmpeg）：

    ==============================  ==========================
    链                              处理后 L 与 R 是否相同
    ==============================  ==========================
    mono→arnndn→stereo（旧）        **相同**（信息已合并）
    直接 arnndn（立体声）           不同
    channelsplit→逐声道→join（采用） 不同
    ==============================  ==========================

    采用逐声道方案而不是"直接喂立体声"，因为后者依赖 arnndn 对多声道的支持，
    各版本不一致；channelsplit 保证每个分支拿到的都是单声道，行为稳定。

    >2 声道：第三轮已实现逐声道处理（实测 5.1 rc=0 且六声道仍互不相同）。
    仅当声道数超过 `_MAX_PER_CHANNEL_ARNNDN`(8) 时才退回 afftdn 并记降级原因。
    """
    layout = (audio_layout or {}).get("layout", "stereo")
    channels = int((audio_layout or {}).get("channels", 1) or 1)
    arg = escape(model)

    if channels <= 1:
        return [f"arnndn=m={arg}"]

    if channels <= _MAX_PER_CHANNEL_ARNNDN:
        # 任意声道数都按"逐声道"处理：split → 每个声道各自 arnndn → join 回原布局。
        # 第三轮实测（合成 5.1，六声道各放 220/330/440/550/660/770 Hz）：
        # rc=0，处理后 6 个声道**仍两两不同**（未发生合并）。
        labels = [f"_dn{i}" for i in range(channels)]
        outs = [f"{lab}o" for lab in labels]
        chain = (
            f"channelsplit=channel_layout={layout}"
            + "".join(f"[{lab}]" for lab in labels) + ";"
            + "".join(f"[{lab}]arnndn=m={arg}[{out}];"
                      for lab, out in zip(labels, outs))
            + "".join(f"[{out}]" for out in outs)
            + f"join=inputs={channels}:channel_layout={layout}")
        if logger:
            logger.info("STAGE7_ARNNDN_PER_CHANNEL",
                        "arnndn applied per channel (no downmix)",
                        channels=channels, layout=layout)
        return [chain]

    # FIX-R3-08 补完 / >8ch：这里原本**无条件**退回 afftdn，只记一条
    # WARNING。但 `allow_audio_degrade` 是"是否授权降级"的开关，
    # 同一文件里"arnndn 模型缺失"的分支（见下方 `_resolve_audio_denoise`）
    # 已经正确遵守它——唯独这条 >8ch 路径绕过了开关。
    # 后果：用户明确要求 arnndn 且**未授权降级**时，>8 声道素材会被
    # 静默换成 afftdn，日志只有一条 WARNING，rc 仍是 0（与 B12 同族：
    # "明确要求的算法不可用"必须失败，而不是降级）。
    if not allow_degrade:
        if logger:
            logger.error("STAGE7_ARNNDN_TOO_MANY_CHANNELS",
                         f"per-channel arnndn for {channels}ch is not "
                         "implemented and no degradation is authorized; "
                         "failing the stage", channels=channels)
        raise AudioDegradeNotAllowed(
            f"arnndn requested for {channels}ch (> "
            f"{_MAX_PER_CHANNEL_ARNNDN}) but per-channel arnndn is not "
            "implemented; set allow_audio_degrade=true to accept afftdn")
    if logger:
        logger.warning("STAGE7_ARNNDN_MULTICHANNEL_DEGRADED",
                       f"per-channel arnndn for {channels}ch is not "
                       "implemented (needs per-layout labels and could not be "
                       "verified here); using afftdn, which processes channels "
                       "independently", channels=channels)
    return ["afftdn=nr=12:nf=-50"]


def _resolve_audio_denoise(params: dict) -> Tuple[str, str]:
    """R2-10: resolve the *effective* audio denoise choice.

    旧实现只看 ``params["noise_reduction"]``（默认 True），再按"模型在不在"选算法，
    于是主入口已经合并好的 ``audio_denoise=False`` 与 ``audio_denoise="afftdn"``
    都被忽略（独立用例 A18 复现：两种设置下链里都出现 arnndn）。

    优先级：``audio_denoise`` > ``noise_reduction``（旧别名）。
    取值语义：
      * ``False`` / ``"false"`` / ``"off"`` / ``"none"`` / ``0``  → ``"off"``
      * ``"arnndn"`` / ``"afftdn"``                               → 指定算法
      * ``True`` / ``"true"`` / ``1``                             → 默认策略
        （arnndn 模型可用则用 arnndn，否则 afftdn）

    Returns:
        ``(algorithm, source)``，algorithm ∈ {"off", "arnndn", "afftdn", "auto"}。
    """
    raw = None
    source = ""
    if "audio_denoise" in params:
        raw = params.get("audio_denoise")
        source = "audio_denoise"
    elif "noise_reduction" in params:
        raw = params.get("noise_reduction")
        source = "noise_reduction(legacy alias)"

    if raw is None:
        return "auto", "unset"
    if isinstance(raw, bool):
        return ("auto" if raw else "off"), source
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return ("auto" if raw else "off"), source
    text = str(raw).strip().lower()
    if text in ("", "false", "off", "none", "no", "0"):
        # 空串按"未设置"处理（与主入口的合并语义一致），其余都是明确关闭
        return ("auto" if text == "" else "off"), source
    if text in ("arnndn", "afftdn"):
        return text, source
    if text in ("true", "on", "yes", "1"):
        return "auto", source
    return "auto", source


def _build_audio_filter(audio_path: str, params: dict, logger=None,
                          tempo_ratio: float = 1.0,
                          audio_layout: dict = None) -> str:
    """Build audio filter chain with hum removal and denoising.

    CRITICAL: bandreject filters must come BEFORE afftdn/arnndn.
    Covers fundamental + 2nd + 3rd harmonic, all width=2Hz.

    Scene 1 (duration unchanged): asetpts=PTS at end.
    Scene 2 (duration changed): atempo before asetpts=PTS.

    FIX-P7-8: DEPRECATION NOTICE — This function duplicates logic in
    shared_modules/audio_utils.process_audio(). Long-term plan: replace
    _build_audio_filter with a call to audio_utils.process_audio() to
    generate an intermediate processed audio file, then stage7 only
    handles muxing/encoding. For now, both copies are maintained with
    synchronized bug fixes (P7-1 through P7-7).
    """
    audio_filters = []

    # V-P-051 fix: FFmpeg version check for bandreject compatibility
    ffmpeg_version_ok = True
    try:
        vr = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, errors="replace", timeout=10, shell=False)
        if vr.returncode == 0:
            version_line = vr.stdout.splitlines()[0]
            # Extract version number like "4.4.2"
            import re
            vm = re.search(r"ffmpeg version (\d+)\.(\d+)", version_line)
            if vm:
                major, minor = int(vm.group(1)), int(vm.group(2))
                if major < 4 or (major == 4 and minor < 4):
                    ffmpeg_version_ok = False
                    logger.warning("STAGE7_FFMPEG_OLD",
                                  f"FFmpeg {major}.{minor} < 4.4, bandreject width_type=h may fail; will fallback to arnndn")
    except Exception:
        pass

    # FIX-P7-3: Per v7.32 §3.10.9: preserve channel layout, no downmix
    # bandreject/afftdn are per-channel internally in FFmpeg -af (no downmix needed).
    # Only arnndn needs filter_complex per-channel (model trained mono, stereo bug).
    # FIX-P7-4: For >2ch with FFmpeg<4.4, add afftdn as hum fallback since
    # arnndn cannot process multi-channel without downmix (per v7.32 §3.10.9).
    if audio_layout and audio_layout.get("channels", 2) > 2:
        num_ch = audio_layout["channels"]
        arnndn_model_val = params.get("arnndn_model", "")
        if arnndn_model_val and Path(arnndn_model_val).exists() and not ffmpeg_version_ok:
            # arnndn path: must use filter_complex per-channel, but that's
            # complex; use afftdn as hum fallback for multi-channel per §3.10.9
            logger.info("STAGE7_MULTICHANNEL_HUM_AFFTDN",
                        f"FFmpeg <4.4 + {num_ch}ch audio: using afftdn for hum "
                        "removal (arnndn requires mono, would violate §3.10.9)")
        else:
            # bandreject/afftdn: unified -af is safe (FFmpeg processes each
            # channel independently, no downmix)
            logger.info("STAGE7_MULTICHANNEL_UNIFIED",
                        f"Unified filter chain for {num_ch}ch audio "
                        "(bandreject/afftdn are per-channel internally)")

    # FIX-P7-1: Per v6.32 §7.12 / v7.32 §3.10.22:
    # hum_freq=None means auto-detect; explicit 50/60 means user override
    # ORIGINAL: hum_freq = params.get("hum_freq", 50) with auto_detect flag
    hum_freq = params.get("hum_freq")  # None = auto-detect; 50/60 = user override
    hum_dual = params.get("hum_dual", False)

    detected_freq = None
    is_dual_hum = False  # FIX-P7-9: track dual hum detection result
    if hum_freq is not None:
        # User explicitly configured hum_freq, skip auto-detection
        detected_freq = int(hum_freq)
        logger.info("STAGE7_HUM_USER_OVERRIDE",
                    f"User forced hum_freq={detected_freq}Hz, skipping auto-detect")
    else:
        # Auto-detect path
        try:
            sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
            from audio_utils import detect_hum_frequency
            hum_result = detect_hum_frequency(audio_path, config=params)
            if hum_result.get("skip_bandreject", False):
                logger.info("STAGE7_HUM_SKIP", "Hum energy below threshold, skipping bandreject")
                # FIX-R7-S7-3a: explicit WARNING when no low-frequency content is
                # detected (hum_ratio==0.0) per v7.32 §3.10.22 — hum_freq stays
                # null (auto), bandreject skipped, hum removal deferred to afftdn.
                if float(hum_result.get("hum_ratio", 0.0)) == 0.0:
                    logger.warning("STAGE7_HUM_NO_LOWFREQ",
                                   "No low-frequency content detected; hum_freq stays "
                                   "null (auto), bandreject skipped per v7.32 §3.10.22")
                detected_freq = None
            else:
                detected_freq = hum_result.get("primary_freq", 50)
                # FIX-P7-9: Per v7.32 §3.10.22: save is_dual_hum from detection
                is_dual_hum = hum_result.get("is_dual_hum", False)
        except Exception as e:
            # FIX-R7-S7-3a (N+1 轮核验): Per round_7 fix doc item 3, a detection
            # FAILURE (exception) must NOT default to 50Hz and apply bandreject --
            # guessing a frequency on a crashed analysis risks cutting legitimate
            # low-freq content. Keep hum_freq null (auto-detect inconclusive),
            # skip bandreject, defer hum removal to afftdn general denoise. This
            # mirrors the skip_bandreject path (L137-141) for internal consistency.
            # (Spec §3.10.22's 50Hz fallback applies to the successful-detection
            # "no low-freq content" case, not to a detection exception.)
            detected_freq = None
            logger.warning("STAGE7_HUM_DETECT_FAIL",
                           f"Hum frequency detection failed: {e}. "
                           "hum_freq stays null (auto), bandreject skipped per v7.32 §3.10.22")
    
    if detected_freq and ffmpeg_version_ok:
        # Fundamental + 2nd + 3rd harmonic, all width=2Hz
        for f in [detected_freq, detected_freq * 2, detected_freq * 3]:
            audio_filters.append(f"bandreject=f={f}:width_type=h:width=2")
        
        # Dual hum support
        # FIX-P7-9: Per v7.32 §3.10.22: dual hum must be detected AND user-enabled.
        # Default: only process primary freq group, log WARNING if dual detected.
        # Only when hum_dual=true AND is_dual_hum=true: process both groups with CRITICAL.
        if hum_dual and is_dual_hum:
            # User explicitly enabled + actual dual hum detected
            other = 60 if detected_freq == 50 else 50
            for f in [other, other * 2, other * 3]:
                audio_filters.append(f"bandreject=f={f}:width_type=h:width=2")
            logger.critical("STAGE7_DUAL_HUM_RISK",
                            "Processing BOTH 50Hz+60Hz hum groups per "
                            "hum_dual=true. Audio quality risk — dual "
                            "filtering may over-process.",
                            primary=detected_freq, secondary=other)
        elif is_dual_hum and not hum_dual:
            logger.warning("STAGE7_DUAL_HUM_DEFAULT",
                           f"Dual hum detected but processing primary "
                           f"{detected_freq}Hz only. Set hum_dual=true to "
                           f"process both groups (audio quality risk).")
    elif detected_freq and not ffmpeg_version_ok:
        # FIX-P7-1: Per v6.32 §7.12 / v7.32 §3.10.22:
        # FFmpeg <4.4: arnndn replaces bandreject for hum removal
        # ORIGINAL: only logged, arnndn filter not inserted into audio_filters
        # FIX-R7-S7-A2: resolve arnndn path via audio_utils._get_arnndn_model_path
        # (single source of truth) instead of bare params.get("arnndn_model").
        # Falls back to the raw config value if resolution returns None, matching
        # the audio_utils.process_audio pattern (L487-489).
        from audio_utils import _get_arnndn_model_path, filter_option_path as _fpath
        arnndn_model = _get_arnndn_model_path(params, detected_freq or 0)
        if not arnndn_model:
            arnndn_model = params.get("arnndn_model", "")
        if arnndn_model and Path(arnndn_model).exists():
            if audio_layout is None:
                audio_layout = _get_audio_layout(audio_path)
            num_ch = audio_layout.get("channels", 2)
            if num_ch > 2:
                # Multi-channel: arnndn needs mono, skip for hum removal
                logger.warning("STAGE7_HUM_ARNNDN_MULTICH",
                               f"FFmpeg <4.4, arnndn requires mono input, cannot "
                               f"remove {detected_freq}Hz hum from {num_ch}ch. "
                               f"afftdn will provide general noise reduction only.")
            else:
                # R2-10：立体声同样不能用 mono 回绕，否则左右被下混成同一份。
                audio_filters.extend(_arnndn_filters(
                    arnndn_model, audio_layout, _fpath, logger))
            logger.info("STAGE7_HUM_ARNNDN",
                        f"FFmpeg <4.4, using arnndn for {detected_freq}Hz hum removal",
                        model=arnndn_model)
        else:
            logger.warning("STAGE7_HUM_NO_ARNNDN",
                           f"FFmpeg <4.4 and arnndn model not found at '{arnndn_model}', "
                           f"hum at {detected_freq}Hz will NOT be removed. "
                           f"afftdn will provide general noise reduction only. "
                           f"Download arnndn models from https://github.com/richardpl/arnndn-models (std.rnnn)")

    # R2-10: 消费真正合并后的降噪选择，而不是"默认 True + 看模型在不在"。
    choice, choice_source = _resolve_audio_denoise(params)
    if logger:
        logger.info("STAGE7_DENOISE_CHOICE",
                    f"audio denoise resolved to {choice!r}",
                    source=choice_source,
                    audio_denoise=params.get("audio_denoise", "<unset>"),
                    noise_reduction=params.get("noise_reduction", "<unset>"))

    if choice == "off":
        # 明确关闭降噪：后面 compand / tempo / asetpts 仍然要照常走。
        if logger:
            logger.info("STAGE7_DENOISE_OFF",
                        "audio denoise disabled by effective config; no "
                        "arnndn/afftdn will be added", source=choice_source)
    else:
        from audio_utils import _get_arnndn_model_path, filter_option_path as _fpath
        arnndn_model = _get_arnndn_model_path(params, detected_freq or 0)
        if not arnndn_model:
            arnndn_model = params.get("arnndn_model", "")
        if audio_layout is None:
            audio_layout = _get_audio_layout(audio_path)
        model_exists = bool(arnndn_model) and Path(arnndn_model).exists()

        if choice == "afftdn":
            # 用户明确指定 afftdn：模型存在也不许换成 arnndn。
            audio_filters.append(f"afftdn=nr={params.get('afftdn_nr', 12)}:"
                                 f"nf={params.get('afftdn_nf', -50)}")
            if logger:
                logger.info("STAGE7_DENOISE_AFFTDN",
                            "afftdn applied as explicitly requested")
        elif choice == "arnndn":
            if model_exists:
                # FIX-R3-08 补完：用户**明确要求** arnndn。>8ch 时是否允许
                # 降级由 allow_audio_degrade 决定（默认 False = 不许降级，
                # 未授权则抛 AudioDegradeNotAllowed → 阶段失败）。
                try:
                    audio_filters.extend(_arnndn_filters(
                        arnndn_model, audio_layout, _fpath, logger,
                        allow_degrade=bool(params.get("allow_audio_degrade",
                                                      False))))
                except AudioDegradeNotAllowed as exc:
                    if logger:
                        logger.error("STAGE7_ARNNDN_CHANNELS_UNAVAILABLE",
                                     "arnndn requested but per-channel "
                                     "processing is unavailable for this "
                                     "channel count and no degradation is "
                                     "authorized; failing the stage",
                                     detail=str(exc))
                    return "__DENOISE_UNAVAILABLE__"
            elif params.get("allow_audio_degrade", False):
                audio_filters.append("afftdn=nr=12:nf=-50")
                if logger:
                    logger.warning("STAGE7_DENOISE_DEGRADED",
                                   "arnndn requested but model not found; "
                                   "degraded to afftdn because "
                                   "allow_audio_degrade is set",
                                   model=arnndn_model)
            else:
                # R3-08 / B12：明确要求的算法不可用且**不许降级** ——
                # 只记日志等于没失败（实测：日志出现 STAGE7_DENOISE_UNAVAILABLE
                # 之后照样 STAGE7_DONE、rc=0）。必须把结果传回主流程。
                if logger:
                    logger.error("STAGE7_DENOISE_UNAVAILABLE",
                                 "arnndn requested but model not found and no "
                                 "degradation is authorized; failing the stage",
                                 model=arnndn_model)
                return "__DENOISE_UNAVAILABLE__"
        else:  # "auto"
            if model_exists:
                # auto 的语义本就是"有 arnndn 就用、没有就降级"，
                # 因此这里始终授权降级（与明确要求 arnndn 的分支不同）。
                audio_filters.extend(_arnndn_filters(arnndn_model, audio_layout,
                                                     _fpath, logger,
                                                     allow_degrade=True))
            else:
                audio_filters.append("afftdn=nr=12:nf=-50")
                if logger:
                    logger.info("STAGE7_DENOISE_AFFTDN",
                                "no arnndn model available; using afftdn")

    # V-P-053 fix: compand dynamic range compression
    # FIX-P7-8: Default changed to False per v7.32 §3.10.9 (not mandatory)
    # ORIGINAL: audio_compand defaulted to True
    use_compand = params.get("audio_compand", False)
    if use_compand:
        audio_filters.append("compand=0.3:1:-90/-900|-70/-70|-30/-9|0/-3:6:0:0:0.5")

    # Scene 2: tempo adjustment if video duration changed
    if tempo_ratio != 1.0:
        # atempo valid range [0.5, 100.0]; chain if >2.0
        if 0.5 <= tempo_ratio <= 100.0:
            remaining = tempo_ratio
            atempo_filters = []
            while remaining > 2.0:
                atempo_filters.append("atempo=2.0")
                remaining /= 2.0
            atempo_filters.append(f"atempo={remaining:.4f}")
            audio_filters.extend(atempo_filters)
            logger.info("STAGE7_ATEMPO",
                       f"Scene 2: audio tempo adjusted by ratio {tempo_ratio:.4f}",
                       tempo_ratio=tempo_ratio)
            # CRITICAL log if outside safe 0.8-1.2 range
            if tempo_ratio < 0.8 or tempo_ratio > 1.2:
                logger.critical("STAGE7_ATEMPO_RISK",
                               f"Tempo ratio {tempo_ratio:.4f} outside safe range 0.8-1.2; audio quality risk")
        else:
            logger.error("STAGE7_ATEMPO_FAIL",
                        f"Tempo ratio {tempo_ratio:.4f} outside atempo valid range [0.5, 100.0]")
            return "__INVALID_TEMPO__"

    # Reset PTS for A/V sync (v6.3 §3.10.9)
    audio_filters.append("asetpts=PTS")

    return ",".join(audio_filters) if audio_filters else "anull"


def _probe_packet_span(path: str, stream_type: str = "v") -> float:
    """用 packet 时间戳求流跨度（NEW-RV-07）。

    MKV/WebM 不填 `stream=duration`，而本管线的默认产物正是 `.mkv`
    （`{stem}_restored.mkv`），所以只靠 `stream=duration` 会让 A/V 同步门
    **对所有产物 100% 静默跳过**（真机实测：ffprobe 对 clip_restored.mkv
    的 `stream=duration` 一律返回 `N/A`，而容器时长 2.071s 是读得到的）。
    退路：读 packet 的 pts_time + duration_time，取 [min_start, max_end) 跨度。
    只读包头、不解码，代价随文件线性增长，因此仅在 stream=duration 缺失时调用。
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", f"{stream_type}:0",
             "-show_entries", "packet=pts_time,duration_time",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, errors="replace", timeout=300, shell=False
        )
        if r.returncode != 0:
            return 0.0
    except Exception:
        return 0.0

    start = None
    end = None
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        pts_s = parts[0].strip() if len(parts) > 0 else ""
        dur_s = parts[1].strip() if len(parts) > 1 else ""
        try:
            pts = float(pts_s)
        except (TypeError, ValueError):
            continue
        try:
            dur = float(dur_s)
        except (TypeError, ValueError):
            dur = 0.0
        if start is None or pts < start:
            start = pts
        e = pts + (dur if dur > 0 else 0.0)
        if end is None or e > end:
            end = e
    if start is None or end is None or end <= start:
        return 0.0
    return end - start


def _get_stream_duration(path: str, stream_type: str = "v") -> float:
    """Get video or audio stream duration using ffprobe.

    三级取值（NEW-RV-07）：
      1. `stream=duration` —— mp4/mov 有，MKV 基本为 N/A；
      2. packet 时间戳跨度 —— MKV/WebM 的可靠退路；
      3. 都没有 → 0.0（调用方按"无法确定"处理，不再假装通过）。
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", f"{stream_type}:0",
             "-show_entries", "stream=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False
        )
        if r.returncode == 0:
            dur = r.stdout.strip()
            if dur and dur != "N/A":
                return float(dur)
    except Exception:
        pass
    return _probe_packet_span(path, stream_type)


def _detect_black_bars_stage7(video_path: str, logger=None) -> dict:
    """Detect black bars on the stage7 input for crop/restore around color
    conversion per v7.32 §3.10.3 ("色域转换前应先自动检测并裁剪黑边,转换完成后
    还原黑边"). Bit-depth-aware cropdetect limit (limited-range black level
    scales with bit depth: 16 * 2^(bd-8)).
    """
    info = {"left": 0, "right": 0, "top": 0, "bottom": 0,
            "cropped_w": 0, "cropped_h": 0, "orig_w": 0, "orig_h": 0,
            "has_bars": False}
    try:
        pr = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,pix_fmt,bits_per_raw_sample",
             "-of", "json", video_path],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False)
        ow = oh = 0
        bd = 8
        if pr.returncode == 0:
            data = json.loads(pr.stdout)
            streams = data.get("streams", [])
            if streams:
                ow = int(streams[0].get("width", 0) or 0)
                oh = int(streams[0].get("height", 0) or 0)
                bpr = streams[0].get("bits_per_raw_sample", 0)
                pf = streams[0].get("pix_fmt", "")
                try:
                    bd = int(bpr) if bpr else 8
                except (ValueError, TypeError):
                    bd = 8
                if bd < 8:
                    bd = 8
                if not bpr:
                    if "p16" in pf or "16le" in pf or "16be" in pf:
                        bd = 16
                    elif "p10" in pf:
                        bd = 10
        info["orig_w"] = ow
        info["orig_h"] = oh
        if ow == 0 or oh == 0:
            return info
        # Bit-depth-aware cropdetect limit (limited-range black = 16 * 2^(bd-8))
        limit = 16 * (2 ** (bd - 8))
        cmd = [
            "ffmpeg", "-hide_banner", "-v", "info",
            "-i", video_path,
            "-vf", f"cropdetect=limit={limit}:round=2:reset=0",
            "-frames:v", "100", "-f", "null", "-"
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                           timeout=120, shell=False)
        crop_match = None
        for line in r.stderr.split("\n"):
            m = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", line)
            if m:
                crop_match = m
        if crop_match:
            cw, ch, cx, cy = map(int, crop_match.groups())
            if cw < ow or ch < oh:
                info.update({
                    "left": cx, "right": ow - cx - cw,
                    "top": cy, "bottom": oh - cy - ch,
                    "cropped_w": cw, "cropped_h": ch,
                    "has_bars": True,
                })
    except Exception as e:
        if logger:
            logger.warning("STAGE7_CROP_FAIL", f"cropdetect failed: {e}")
    return info


_ZSCALE_CAP_CACHE: Dict[str, bool] = {}
#: R2-12：探测结论（用于日志区分"不支持 / 超时 / 工具缺失"）
_ZSCALE_CAP_REASON: Dict[str, str] = {}
#: R2-12：探测必须**限制帧数**。lavfi 的 color/testsrc 源是无限的，
#: 不限制的话 ffmpeg 会一直跑到 timeout —— 实测同一被支持的选项要 15.031s
#: 才返回 False（超时被当成"不支持"），加 -frames:v 1 后约 0.063s 返回 0。
_ZSCALE_PROBE_FRAMES = "1"


def _zscale_probe(option_expr: str) -> str:
    """Probe an option and report *why*: supported / unsupported / timeout / missing."""
    if option_expr in _ZSCALE_CAP_REASON:
        return _ZSCALE_CAP_REASON[option_expr]
    reason = "unsupported"
    try:
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
             "color=c=black:s=64x64", "-frames:v", _ZSCALE_PROBE_FRAMES,
             "-vf", f"zscale={option_expr}", "-f", "null", "-"],
            capture_output=True, shell=False, timeout=15)
        if probe.returncode == 0:
            reason = "supported"
    except subprocess.TimeoutExpired:
        reason = "timeout"
    except FileNotFoundError:
        reason = "missing"
    except Exception:
        reason = "unsupported"
    _ZSCALE_CAP_REASON[option_expr] = reason
    return reason


def _zscale_has_option(option_expr: str) -> bool:
    """NEW-RV-05 / R2-12: probe whether the installed ffmpeg's zscale accepts an option.

    真机复现（ffmpeg 2023-11-13 gyan 构建）：``zscale=...:format=yuv444p10``
    直接失败 —— "Error applying option 'format' to filter 'zscale': Option not
    found"，stage7 因此 rc=84。该构建的 zscale 只有 w/h/size/dither/filter/
    out_range… 没有 format。
    选项是否可用必须**运行时探测**，不能写死（不同 ffmpeg/zimg 版本差异很大）。

    R2-12：探测必须限帧，否则无限 lavfi 源会跑到超时，
    "超时"被误记成"不支持"（实测 15.031s 判 False，限帧后 0.063s 判 True）。
    """
    if option_expr in _ZSCALE_CAP_CACHE:
        return _ZSCALE_CAP_CACHE[option_expr]
    available = (_zscale_probe(option_expr) == "supported")
    _ZSCALE_CAP_CACHE[option_expr] = available
    return available


def _zscale_dither_filters(dither: str, logger=None) -> List[str]:
    """Return the 16→10 bit filter chain that this ffmpeg can actually run."""
    filters: List[str] = ["format=yuv444p16"]
    if _zscale_has_option("format=yuv444p10"):
        filters.append(f"zscale=rangein=limited:range=limited:"
                       f"dither={dither}:format=yuv444p10")
        if logger:
            logger.info("STAGE7_DITHER_MODE",
                        "zscale performs the 16→10 bit reduction with the "
                        "requested dither", dither=dither)
    else:
        # 该构建没有 zscale 的 format 选项：降位深由后面的 format 滤镜完成。
        #
        # R2-12 修正：第二轮曾在这里断言"抖动不作用于降位深"，**这是错的**。
        # 实测（本机 ffmpeg，16-bit 梯度 → 10-bit，与 dither=none 逐字节比对）：
        #   ordered          → 1750 字节不同
        #   error_diffusion  → 1069 字节不同
        # 即所选抖动**确实改变了最终像素**。能确定的是像素层面可观察到差异；
        # 其内部作用点（zscale 内还是相邻 format 滤镜协商后）未进一步拆解，
        # 不作为结论。日志只陈述事实，不再声称"无效"。
        # R3-12 措辞修正：第二轮写的是"dither 不作用于降位深"（错），
        # 第三轮改成了"降位深由 zscale 之后的 format 滤镜完成"（也不严谨 ——
        # zscale 的输出格式由**相邻滤镜协商**决定，它会按协商结果执行转换，
        # 见 FFmpeg zscale 文档）。这里只陈述**实测事实**，不再断言
        # "转换发生在哪个滤镜内部"。
        reason = _zscale_probe("format=yuv444p10")
        filters.append(f"zscale=rangein=limited:range=limited:dither={dither}")
        filters.append("format=yuv444p10")
        if logger:
            logger.warning(
                "STAGE7_DITHER_FALLBACK_CHAIN",
                "This ffmpeg's zscale has no 'format' option "
                f"(probe: {reason}); the 16→10 bit reduction happens on the "
                "format negotiated between zscale and the following format "
                "filter. Measured: the requested dither DOES change the output "
                "pixels (ordered/error_diffusion differ from none), so dither "
                "is effective here; which filter performs the reduction "
                "internally is not asserted.", dither=dither,
                zscale_format_probe=reason)
    filters.append("format=yuv420p10le")
    return filters


def _get_source_color_info(video_path: str) -> tuple:
    """Probe source color_range and color_primaries via ffprobe.

    Returns (source_color_range, color_standard, range_known, primaries_known)
    where:
    - source_color_range: 'limited' or 'full'
    - color_standard: 'PAL', 'NTSC', or 'BT709'
    - range_known: True if ffprobe reported color_range
    - primaries_known: True if ffprobe reported recognizable primaries

    FIX-P7-2/P7-7: Added BT.709 recognition per v7.32 §3.10.3.
    FIX-R7-S7-A1: return range_known/primaries_known flags so the caller can
    log WARNING for unknown range/primaries per v6.32 §7.19 ("若无法读取源 range
    属性,默认假设老影像为 limited range,但记录 WARNING").
    ORIGINAL: only recognized PAL/NTSC, defaulted unknown to NTSC silently.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=color_range,color_primaries",
             "-of", "json", video_path],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            streams = data.get("streams", [])
            if streams:
                cr = streams[0].get("color_range", "")
                cp = streams[0].get("color_primaries", "")
                range_known = bool(cr)
                source_range = "full" if cr == "pc" else "limited"
                primaries_known = False
                # FIX-P7-2/P7-7: Add BT.709 recognition per v7.32 §3.10.3
                if cp in ("bt709", "1", "9"):
                    color_std = "BT709"
                    primaries_known = True
                elif cp in ("bt470bg", "5"):
                    color_std = "PAL"
                    primaries_known = True
                elif cp in ("smpte170m", "bt470m", "6", "4"):
                    color_std = "NTSC"
                    primaries_known = True
                else:
                    # Unknown primaries — default NTSC for old footage
                    color_std = "NTSC"
                return source_range, color_std, range_known, primaries_known
    except Exception:
        pass
    return "limited", "NTSC", False, False


def _probe_encoder_priming(path: str, stream_type: str = "a") -> float:
    """Return the encoder's declared priming (seconds, >= 0) for a stream.

    第三轮实测拆解 66/86ms 残差：AAC 编码器插入
    ``initial_padding = 1024`` 样本。44.1kHz 下 = **23.22ms**，ffprobe 因此把
    音频流的 ``start_time`` 报成 **-0.023** —— 这是编码器 priming，
    播放器靠 edit list / skip_samples 跳过它，**听感上并不偏移**。
    若不归一，一个真实的内容偏移会被这段 priming 掩盖（交接明确禁止
    "把所有起点强制归零来隐藏内容偏移"，这里是按编码器声明值归一，
    不是强制归零，且 end_diff / 漂移两个独立判据仍然保留）。
    """
    if stream_type != "a":
        return 0.0
    try:
        # 必须按**字段名**取值：`nokey=1` 的输出顺序不保证等于请求顺序。
        # 第一版按位置读，把 (sample_rate, initial_padding) 读成了
        # (padding, rate) = (44100, 1024)，算出 43.07 秒的"priming"，
        # 于是同步门把正常产物判成 43 秒错位（自测暴露）。
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=initial_padding,sample_rate",
             "-of", "json", path],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False)
        if r.returncode != 0:
            return 0.0
        streams = (json.loads(r.stdout or "{}").get("streams") or [])
        if not streams:
            return 0.0
        stream = streams[0]
        padding = stream.get("initial_padding")
        rate = stream.get("sample_rate")
        if padding is None or rate is None:
            return 0.0
        padding = float(padding)
        rate = float(rate)
        if rate <= 0:
            return 0.0
        return max(0.0, padding / rate)
    except Exception:
        return 0.0


def _source_has_audio_stream(path: str) -> bool:
    """R3-08 / B11: does the **source** file have an audio stream?

    "temp 里没有中间音轨"不能等同于"原片没有音轨"。B11 实测：源片明确含
    PCM 音轨，只因 temp 里没有 `.mka`，stage7 就记 "Source has no audio
    stream" 并加 `-an`，产物是静音视频且 rc=0。
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False)
        return r.returncode == 0 and bool((r.stdout or "").strip())
    except Exception:
        return False


def _extract_audio_from_source(source: str, dest: str, logger=None) -> Optional[str]:
    """R3-08 / B11: extract the audio track straight from the source file."""
    try:
        import os as _os
        _os.makedirs(str(Path(dest).parent), exist_ok=True)
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", source, "-vn",
             "-c:a", "pcm_s16le", dest],
            capture_output=True, text=True, errors="replace", timeout=120, shell=False)
        if r.returncode != 0 or not _os.path.isfile(dest) or \
                _os.path.getsize(dest) == 0:
            return None
        return dest
    except Exception as e:
        if logger:
            logger.warning("STAGE7_AUDIO_EXTRACT_FAIL",
                           f"Could not extract audio from the source: {e}")
        return None


def _probe_stream_start(path: str, stream_type: str = "v") -> Optional[float]:
    """Return a stream's **raw** start time in seconds, or None.

    R2-11：同步不能只比"时长"。MKV 的 `stream=start_time` 能真实反映
    `-itsoffset` 造成的偏移（实测：video 0.000000 / audio 1.000000），
    比逐 packet 便宜得多，因此作为首选；拿不到时再退回 packet 首时间戳。

    R3-10：这里返回的是**原始**起点（含编码器 priming 的负值）。
    第三轮曾在这里直接把 priming 加进去，导致后面 `end = start + span`
    把末端也整体平移 —— 原始 packet 区间与"有效内容区间"混在一起。
    现在由调用方（:func:`_validate_av_sync`）显式区分两者。
    """

    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", f"{stream_type}:0",
             "-show_entries", "stream=start_time",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False)
        if r.returncode == 0:
            txt = r.stdout.strip()
            if txt and txt != "N/A":
                return float(txt)
    except Exception:
        pass
    # 退回：packet 的最小 pts
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", f"{stream_type}:0",
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", path],
            capture_output=True, text=True, errors="replace", timeout=300, shell=False)
        if r.returncode == 0:
            vals = []
            for line in r.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    vals.append(float(line))
                except ValueError:
                    continue
            if vals:
                return min(vals)
    except Exception:
        pass
    return None


def _probe_audio_presence(path: str) -> str:
    """三态：``"present"`` / ``"absent"`` / ``"unknown"``。

    必须区分这三种，否则会把"文件不存在 / ffprobe 失败"误判成"本来就没音轨"，
    于是探测失败又变成静默通过（A07 就是这条路径）。
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False)
    except Exception:
        return "unknown"
    if r.returncode != 0:
        return "unknown"
    return "present" if r.stdout.strip() else "absent"


def _validate_av_sync(muxed_path: str, logger) -> int:
    """Validate A/V sync on the final muxed output file.

    Per v6.3 §3.10.9 / v7.32 §3.10.9:
    - diff > 100ms: WARNING
    - diff > 300ms: sync processing needed
    - diff > 500ms: ERROR/FAILED (return 86)

    R2-11：旧实现只比 `视频时长` 与 `音频时长` 的**差值**，因此"两边一样长、
    但整条音轨晚 1 秒"这种固定偏移会被判为同步（实测 A16：video [0,2] /
    audio [1,3]，旧门返回 0）。而且取不到时长时记一条 SKIP 就返回 0，
    等于"测不出来就算通过"（A07）。

    现在同时检查三件事：
      1. 起点差 start_diff —— 固定偏移只能靠它发现；
      2. 终点差 end_diff；
      3. 持续时长漂移 dur_diff —— 随时间线性漂移只能靠它发现。
    三者取最大值按阈值判定。取不到**任何**一侧区间时返回 86（未验收），
    不再静默放行；容器**根本没有音轨**是另一种情形，由输入契约决定是否允许，
    这里记录后返回 0，不与"探测失败"共用同一个 0 语义。
    """
    presence = _probe_audio_presence(muxed_path)
    if presence == "unknown":
        # 探测本身就失败了 —— 这**不是**"没有音轨"，不能放行。
        logger.error("STAGE7_SYNC_UNMEASURABLE",
                     "Cannot probe the muxed output; refusing to claim the "
                     "output is in sync (not a pass)", path=muxed_path)
        return 86
    if presence == "absent":
        logger.warning("STAGE7_SYNC_NO_AUDIO",
                       "No audio stream in the muxed output; sync check not "
                       "applicable (allowed only if the input contract says the "
                       "source has no audio)")
        return 0

    # 编码器 priming（AAC 常见 1024 样本）：说明"容器时长膨胀"中有多少是
    # 编码固有、不是内容错位。start 用它归一，end/跨度用原始区间，两者分开。
    priming_ms = _probe_encoder_priming(muxed_path, "a") * 1000

    intervals = {}
    for st in ("v", "a"):
        raw_start = _probe_stream_start(muxed_path, st)
        span = _get_stream_duration(muxed_path, st)
        if raw_start is None or not span:
            intervals[st] = None
        else:
            # R3-10：明确区分两种区间，不再把 priming 混进末端。
            #   * raw 区间：[raw_start, raw_start + span] —— 用于末端与跨度判定；
            #   * content 起点：raw_start + priming —— 用于起点判定
            #     （编码器声明的 priming 会被播放器丢弃，不是内容偏移）。
            # 第三轮把 priming 加进 start 后又用它拼 end，末端被整体平移了。
            priming = _probe_encoder_priming(muxed_path, st)
            intervals[st] = {
                "raw_start": raw_start,
                "raw_end": raw_start + span,
                "raw_span": span,
                "content_start": raw_start + priming,
                "priming": priming,
            }

    if intervals.get("v") is None or intervals.get("a") is None:
        logger.error("STAGE7_SYNC_UNMEASURABLE",
                     "Cannot determine one or both stream intervals; refusing "
                     "to claim the output is in sync (not a pass)",
                     video_interval=intervals.get("v"),
                     audio_interval=intervals.get("a"))
        return 86

    v = intervals["v"]
    a = intervals["a"]
    v_start, a_start = v["content_start"], a["content_start"]
    v_end, a_end = v["raw_end"], a["raw_end"]
    start_diff = abs(v_start - a_start)
    end_diff = abs(v_end - a_end)
    dur_diff = abs(v["raw_span"] - a["raw_span"])
    diff_ms = max(start_diff, end_diff, dur_diff) * 1000

    def _detail():
        return dict(video_content_start=v_start, video_raw_end=v_end,
                    audio_content_start=a_start, audio_raw_end=a_end,
                    video_raw_span=v["raw_span"], audio_raw_span=a["raw_span"],
                    start_diff_ms=start_diff * 1000,
                    end_diff_ms=end_diff * 1000,
                    duration_drift_ms=dur_diff * 1000,
                    encoder_priming_ms=priming_ms)

    if diff_ms > 500:
        logger.error("STAGE7_SYNC_FAIL",
                     f"AV sync difference {diff_ms:.0f}ms exceeds 500ms threshold "
                     f"(start {start_diff * 1000:.0f}ms / end {end_diff * 1000:.0f}ms "
                     f"/ drift {dur_diff * 1000:.0f}ms)", **_detail())
        return 86
    elif diff_ms > 300:
        logger.warning("STAGE7_SYNC_WARN",
                       f"AV sync difference {diff_ms:.0f}ms exceeds 300ms threshold, "
                       "sync processing recommended", **_detail())
        return 0  # Continue but warn
    elif diff_ms > 100:
        # FIX-R7-S7-11: >100ms must be WARNING per v7.32 §3.10.9.
        logger.warning("STAGE7_SYNC_WARN_100",
                       f"AV sync difference {diff_ms:.0f}ms exceeds 100ms threshold",
                       **_detail())
        return 0  # Continue

    logger.info("STAGE7_SYNC_OK",
                f"AV sync within tolerance ({diff_ms:.0f}ms)", **_detail())
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--params")
    # FIX-R6-07: main.py 统一传 `--params-file`；stage7 原来只认 `--params`
    # → argparse 退 2，批处理下 stage7 从未被执行过（实测 2026-10-18）。
    parser.add_argument("--params-file")
    parser.add_argument("--config")
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()

    if getattr(args, "params_file", None):
        try:
            with open(args.params_file, "r", encoding="utf-8") as f:
                params = json.load(f)
        except Exception:
            params = {}
    else:
        params = json.loads(args.params) if args.params else {}
    log_dir = params.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, job_id=args.job_id)

    # NEW-V-06: 注册取消守卫（video main.py 认 rc=88 = EXIT_INTERRUPTED，
    # manifest 记 cancelled；stage7 是最终成品，尤其不能留半截）。
    if _cancel_guard is not None:
        _cancel_guard.install_cancel_guard(stage=7, logger=logger)

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    # FIX-P7-10: Use .stem (without extension) to match stage0 naming
    # ORIGINAL: Path(args.input).name included extension, mismatch with stage0
    input_name = Path(args.input).stem

    # FIX-P7-9: Use pathlib.Path instead of os.path per CLAUDE.md §4
    # Find input - deterministic by mtime
    input_source = None
    for pat in ["step6_grain*", "step5_face*", "step4_postprocess*",
                "step3_upscale*", "step3_forced_upscale*",
                "step2_traditional*", "step1_preprocess*"]:
        pattern = f"{job_id}_{Path(input_name).stem}_{pat}"
        # ORIGINAL: glob.glob(os.path.join(temp_dir, pattern))
        # R3-01：只接受媒体产物，排除同前缀的身份/契约/状态侧车（B10）
        candidates = select_media_candidates(Path(temp_dir).glob(pattern))
        if candidates:
            input_source = str(candidates[0])
            break

    if not input_source:
        logger.error("STAGE7_NO_INPUT", "No input found")
        return 82

    output_path = params.get("output_path", "")
    if not output_path:
        # FIX-P7-9: Use pathlib.Path per CLAUDE.md §4
        output_dir = Path(params.get("output_dir", "./output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{Path(input_name).stem}_restored.mkv")

    # FIX-R5-07: output_path 无论来自 manifest 还是上面兜底生成，都必须
    # 1) 转绝对路径 —— 相对路径按**进程 cwd** 解析，而 manifest 是持久产物：
    #    扫描时 cwd=A、执行时 cwd=B，产物就会落到 B 下（find_final_output 的
    #    os.path.exists(explicit) 同样按 cwd 解析，resume 直接找不到成品）。
    # 2) 建父目录 —— 原来只有兜底分支 mkdir，manifest 显式路径分支不建。
    #    实测（真实素材验收）：manifest 的 output_path 是 config 里的相对
    #    路径 "./output"（manifest_generate 从 config 读，且忽略 --output），
    #    cwd 又是仓库根 -> ffmpeg 报
    #    "Error opening output output\real_seg_restored.mkv:
    #     No such file or directory" -> exit 84。整链 stage0→6 全绿、
    #    零 temp 零日志，只死在最后一步落盘，极难排查。
    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # WSL2 redirect: if running in WSL and Windows tools need Windows paths
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    platform = params.get("platform", "auto")

    # FIX-P7-13: Also redirect temp_dir per v7.32 §3.10.10 / CLAUDE.md §4
    # ORIGINAL: only output_path was redirected, temp_dir on /mnt/ caused slow I/O
    try:
        redirected_temp, _, _, _ = wsl_redirect_path(
            temp_dir, no_wsl_redirect=no_wsl_redirect, target_platform=platform
        )
        if redirected_temp != temp_dir:
            logger.info("STAGE7_WSL_TEMP_REDIRECT",
                       f"WSL temp redirect: {temp_dir} -> {redirected_temp}")
            temp_dir = redirected_temp
    except Exception:
        pass  # wsl_redirect_path not available or not in WSL

    redirected_path, original_output_path, _, _ = wsl_redirect_path(
        output_path, no_wsl_redirect=no_wsl_redirect, target_platform=platform
    )
    if original_output_path:
        logger.info("STAGE7_WSL_REDIRECT",
                   f"WSL redirect: {output_path} -> {redirected_path}")
        output_path = redirected_path

    # NEW-V-06: 编码开始前登记最终产物（WSL 重定向后的真实写入目标）——
    # 取消时删掉半截成品并写 FAILED 记录，否则"取消"在文件系统层面
    # 与"成功"无法区分（半截文件往往还是非空的）。
    if _cancel_guard is not None:
        _cancel_guard.watch_outputs(output_path, logger=logger, stage=7)

    logger.info("STAGE7_START", "Final encoding",
                input=input_source, output=output_path)

    # SVT-AV1 parameters
    crf = params.get("crf", 24)
    preset = params.get("preset", 4)
    pix_fmt = params.get("pix_fmt", "yuv420p10le")
    color_range = params.get("color_range", "limited")

    # FIX-P7-5: Preset adaptive per v6.32 §7.10:
    # On machines with ≥16 CPU cores, lower preset for better quality
    if params.get("preset_max_quality", False):
        import multiprocessing
        try:
            cpu_count = multiprocessing.cpu_count()
            if cpu_count >= 16:
                original_preset = preset
                preset = min(preset, 3)  # Try preset 2-3 for better quality
                logger.info("STAGE7_PRESET_ADAPTIVE",
                            f"CPU cores={cpu_count} >= 16, "
                            f"preset {original_preset} -> {preset}")
        except NotImplementedError:
            pass  # cpu_count not available on this platform

    # FIX-P7-6: CRF adaptive logic per v6.32 §7.10 / v7.32 §3.10.20
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "json", input_source],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False
        )
        if r.returncode == 0:
            info = json.loads(r.stdout)
            stream = info["streams"][0]
            vid_w = stream.get("width", 1920)
            vid_h = stream.get("height", 1080)

            # Base CRF by resolution
            if vid_w > 1920 or vid_h > 1080:
                crf = 26
            else:
                crf = 24  # 480p/576p/720p unified default

            # Grain adaptive (if upstream stages passed grain info)
            # FIX-P7-5: Per v7.32 §3.10.20, grain_strength formula is
            # std(Y) / (1 + 0.1 × motion_index), NOT noise_score * 10.
            # noise_score*10 is an approximation only.
            grain_strength = params.get("detected_grain_strength", None)
            if grain_strength is None and "noise_score" in params:
                grain_strength = params["noise_score"] * 10
                logger.warning("STAGE7_GRAIN_APPROX",
                              "Using noise_score*10 as grain_strength "
                              "approximation. Spec v7.32 §3.10.20 requires "
                              "std(Y)/(1+0.1*motion).")
            if grain_strength is not None:
                if grain_strength > 5.0:
                    crf -= 1
                    logger.info("STAGE7_CRF_GRAIN", f"High grain {grain_strength:.2f}, CRF -> {crf}")
                elif grain_strength < 1.0:
                    crf += 1
                    logger.info("STAGE7_CRF_CLEAN", f"Low grain {grain_strength:.2f}, CRF -> {crf}")

            detected_class = params.get("detected_class", "").lower()
            if "animation" in detected_class or "anime" in detected_class:
                crf += 2
                logger.info("STAGE7_ANIMATION_CRF", f"Animation detected, CRF adjusted to {crf}")
            logger.info("STAGE7_ADAPTIVE",
                        f"Adaptive encoding: {vid_w}x{vid_h} -> preset={preset}, crf={crf}")
    except Exception as e:
        logger.warning("STAGE7_ADAPTIVE_SKIP", f"Adaptive preset/CRF failed: {e}")

    # v6.3: tile settings
    tile_rows = params.get("svt_tile_rows", 2)
    tile_cols = params.get("svt_tile_cols", 2)

    # Video encoding with audio sync flags
    # BUG-041 fix: defer svtav1-params construction until all parameters are known
    # FIX-R7-S7-A4: -fflags +genpts is an OUTPUT option ("封装时使用 -fflags +genpts"
    # per v7.32 §3.10.9). ORIGINAL placed it before -i (input side). It is now
    # added on the output side below, before the output path.
    # FIX-R7-S7-13: support libx265 (HEVC) as an alternative encoder per v6.32
    # Part 3 ("libsvtav1 或 libx265"). config video_encoder selects the codec;
    # libsvtav1 remains the default. libx265 uses -preset slow -crf 18.
    # NEW-RV-04（真机 ffmpeg 复现）：**输入段与输出段必须分开拼**。
    # ORIGINAL 把 -vf / -c:v 等**输出侧**选项插在第一个 -i 之后、第二个 -i 之前，
    # ffmpeg 于是把它们当成"下一个输入文件的输入选项"：
    #   "Option vf ... cannot be applied to input url jobV1_clip.mka --
    #    you are trying to apply an input option to an output file or vice versa."
    # 结果：只要素材有音轨，stage7 必然 rc=84。
    # 修法：video_cmd 只放输入（所有 -i），output_opts 放全部输出侧选项，
    # 最后先 extend 输出选项再写输出路径。
    video_encoder = params.get("video_encoder", "libsvtav1")
    common_prefix = ["ffmpeg", "-hide_banner", "-v", "error", "-y"]
    video_cmd = common_prefix + ["-i", input_source]
    output_opts: List[str] = []
    if video_encoder == "libx265":
        # x265 CRF 18 ≈ visually lossless; preset slow per v6.32 Part 3.
        output_opts.extend(["-c:v", "libx265",
                            "-crf", str(params.get("libx265_crf", 18)),
                            "-preset", str(params.get("libx265_preset", "slow")),
                            "-pix_fmt", pix_fmt])
    else:
        output_opts.extend(["-c:v", video_encoder, "-crf", str(crf),
                            "-preset", str(preset), "-pix_fmt", pix_fmt])

    # FIX-P7-1: BT.601→BT.709 color space conversion with range handling
    # FIX-P7-3: Probe stage output (input_source), not original input (args.input)
    # per v7.32 §3.10.3: conversion should be based on the video being encoded
    # ORIGINAL: _get_source_color_info(args.input) probed original input
    source_color_range, color_standard, range_known, primaries_known = _get_source_color_info(input_source)

    # FIX-R7-S7-A1: log WARNING for unknown source range/primaries per v6.32 §7.19
    if not range_known:
        logger.warning("STAGE7_COLOR_RANGE_UNKNOWN",
                      "Cannot read source color_range from ffprobe, assuming "
                      "limited range per v6.32 §7.19")
    if not primaries_known:
        logger.warning("STAGE7_COLOR_PRIMARIES_UNKNOWN",
                      "Cannot recognize source color_primaries from ffprobe, "
                      "defaulting to NTSC (BT.601) per v7.32 §3.10.3")

    # FIX-R7-S7-16: detect black bars to wrap color conversion with crop/restore
    # per v7.32 §3.10.3 ("色域转换前应先自动检测并裁剪黑边,转换完成后还原黑边").
    # ORIGINAL vf_parts only had zscale+dither, no crop/restore — black bars got
    # color-converted and ended up at the wrong black level.
    bar_info = _detect_black_bars_stage7(input_source, logger)
    _crop_filter = None
    _pad_filter = None
    if bar_info["has_bars"]:
        cw = bar_info["cropped_w"]
        ch = bar_info["cropped_h"]
        ow = bar_info["orig_w"]
        oh = bar_info["orig_h"]
        # §3.10.3 / §7.26: effective area <50% original OR either dim <240 -> FAILED
        if (cw * ch) < (ow * oh) * 0.5 or cw < 240 or ch < 240:
            logger.error("STAGE7_CROP_FAILED",
                        f"Black-bar crop {cw}x{ch} invalid (orig {ow}x{oh}); "
                        f"effective area <50% or dim <240px per v7.32 §3.10.3")
            return 86
        _crop_filter = f"crop={cw}:{ch}:{bar_info['left']}:{bar_info['top']}"
        # pad restores black bars at the OUTPUT format's limited-range black level
        # (ffmpeg pad default 'black' = 16/64/4096 for 8/10/16bit limited YUV).
        _pad_filter = f"pad={ow}:{oh}:{bar_info['left']}:{bar_info['top']}:black"
        logger.info("STAGE7_CROP_RESTORE",
                   f"Black bars detected: crop {cw}x{ch}, restore to {ow}x{oh}")

    # FIX-P7-2: Skip colorspace conversion if source is already BT.709
    # per v7.32 §3.10.3: only BT.601 sources need conversion
    # ORIGINAL: always applied zscale, causing BT.709 sources to get color shift
    vf_parts = []
    # FIX-R7-S7-16: crop black bars BEFORE color conversion
    if _crop_filter:
        vf_parts.append(_crop_filter)
    if color_standard == "BT709":
        logger.info("STAGE7_COLORSPACE_SKIP",
                    "Source already BT.709, skipping colorspace conversion")
    elif not params.get("colorspace_converted", False):
        matrix_in = "bt470bg" if color_standard == "PAL" else "smpte170m"
        rangein = "full" if source_color_range == "full" else "limited"
        vf_parts.append(f"zscale=matrixin={matrix_in}:matrix=bt709:rangein={rangein}:range=limited")
        logger.info("STAGE7_COLORSPACE",
                   f"BT.601({color_standard})->BT.709 zscale: matrixin={matrix_in}, rangein={rangein}")

    # FIX-P7-6: 16bit→10bit error diffusion dithering per v6.32 §阶段3
    # Per fix doc review: zscale error_diffusion requires zimg ≥3.0;
    # Ubuntu 22.04's zimg may not support it, so check availability first.
    # ORIGINAL: used simple scale without proper dithering
    input_depth = params.get("input_bit_depth", 16)
    if input_depth > 10:
        # FIX-R7-S7-18: 16bit limited range (black=4096/white=60160) is industry
        # convention, NOT ITU standard — must log WARNING per v7.32 §3.10.26.
        logger.warning("STAGE7_16BIT_QUANT",
                       "16bit limited range (black=4096/white=60160) is industry "
                       "convention, NOT ITU standard (per v7.32 §3.10.26)")
        # Check if zscale supports error_diffusion dithering.
        # R2-12：这里原来是**无限** lavfi color 源 + timeout=10 ——
        # 实测要跑满 10 秒才把它判成"不支持"。改用统一限帧探测函数。
        _ed_probe = _zscale_probe("dither=error_diffusion")
        _zscale_ed_available = (_ed_probe == "supported")
        if _ed_probe in ("timeout", "missing"):
            logger.warning("STAGE7_ZSCALE_PROBE_INCONCLUSIVE",
                           "error_diffusion probe was inconclusive; treating as "
                           "unavailable", reason=_ed_probe)

        if _zscale_ed_available:
            # Per v6.32 §阶段3: error_diffusion dithering in YUV444 space
            vf_parts.extend(_zscale_dither_filters("error_diffusion", logger))
            logger.info("STAGE7_DITHER", "Using error_diffusion dithering (16bit->10bit)")
        else:
            # Fallback: ordered dithering (available on all zimg versions)
            vf_parts.extend(_zscale_dither_filters("ordered", logger))
            logger.warning("STAGE7_DITHER_FALLBACK",
                          "error_diffusion not available in zscale, "
                          "using ordered dithering (16bit->10bit)")

    # FIX-R7-S7-16: restore black bars AFTER color conversion + dither so they
    # use the target bit-depth/range black level per v7.32 §3.10.3.
    if _pad_filter:
        vf_parts.append(_pad_filter)

    if vf_parts:
        # NEW-RV-04: -vf 是**输出侧**选项，必须在所有 -i 之后，否则 ffmpeg
        # 会把它当成"下一个输入文件的输入选项"并直接报错退出。
        output_opts.extend(["-vf", ",".join(vf_parts)])

    # Scene detection for audio handling (v6.3 §3.10.9)
    # Scene 1: duration unchanged -> asetpts=PTS only
    # Scene 2: duration changed -> atempo adjustment needed
    original_duration = params.get("duration", 0)
    processed_duration = _get_stream_duration(input_source, "v")
    tempo_ratio = 1.0
    if original_duration > 0 and processed_duration > 0:
        diff_ms = abs(original_duration - processed_duration) * 1000
        if diff_ms > 300:
            tempo_ratio = original_duration / processed_duration
            logger.info("STAGE7_SCENE2",
                       f"Detected Scene 2: video duration changed by {diff_ms:.0f}ms (ratio={tempo_ratio:.4f})")
        else:
            logger.info("STAGE7_SCENE1",
                       f"Detected Scene 1: duration diff {diff_ms:.0f}ms <= 300ms")
    else:
        logger.info("STAGE7_SCENE1", "Duration info unavailable, defaulting to Scene 1")

    # Audio (extract from original if available)
    # V-P-049 fix: match stage0 output naming convention
    # FIX-P7-9: Use pathlib.Path per CLAUDE.md §4
    # FIX-R7-S0-OBS: stage0 now writes .mka (Matroska Audio) instead of the
    # non-standard .audio extension. Check .mka first, then fall back to the
    # legacy .audio name for intermediates produced by older stage0 runs.
    audio_path_obj = Path(temp_dir) / f"{job_id}_{input_name}.mka"
    if not audio_path_obj.exists() or audio_path_obj.stat().st_size == 0:
        audio_path_obj = Path(temp_dir) / f"{job_id}_{input_name}.audio"
    # Fallback: also check the old naming for backward compatibility
    if not audio_path_obj.exists() or audio_path_obj.stat().st_size == 0:
        audio_path_fallback = Path(temp_dir) / f"{job_id}_{input_name}_audio_raw.wav"
        if audio_path_fallback.exists() and audio_path_fallback.stat().st_size > 0:
            audio_path_obj = audio_path_fallback
    audio_path = str(audio_path_obj)
    has_audio = audio_path_obj.exists() and audio_path_obj.stat().st_size > 0
    if has_audio:
        # FIX-P7-2: multichannel preserve + explicit layout restore in filter chain
        audio_layout = _get_audio_layout(audio_path)
        channels = audio_layout.get("channels", 2)
        audio_filter = _build_audio_filter(audio_path, params, logger,
                                             tempo_ratio=tempo_ratio,
                                             audio_layout=audio_layout)
        # Invalid tempo check
        if audio_filter == "__INVALID_TEMPO__":
            logger.error("STAGE7_AUDIO_FAIL", "Invalid tempo ratio, cannot process audio")
            return 86
        # R3-08 / B12：明确要求的降噪算法不可用且不许降级 -> 阻断
        if audio_filter == "__DENOISE_UNAVAILABLE__":
            logger.error("STAGE7_DENOISE_FAIL",
                         "Required denoise algorithm is unavailable and "
                         "degradation is not authorized")
            return 86

        multichannel_preserve = params.get("multichannel_preserve", True)
        if multichannel_preserve and channels > 2:
            audio_codec = "flac"
            audio_bitrate = None
            logger.info("STAGE7_MULTICHANNEL",
                       f"Detected {channels} channels, using FLAC to preserve layout")
        else:
            audio_codec = params.get("audio_codec", params.get("codec", "aac"))
            audio_bitrate = params.get("audio_bitrate", params.get("bitrate", "320k"))

        # NEW-RV-04: 第二个输入必须紧跟在其它 -i 之后，输出侧选项全部后移
        video_cmd.extend(["-i", audio_path])
        output_opts.extend([
            "-af", audio_filter,
            "-map", "0:v:0",  # Map video from first input
            "-map", "1:a:0",  # Map audio from second input
        ])
        # FIX-P7-2: Layout restored inside audio filter chain via aformat;
        # -ac fallback removed to avoid overriding explicit layout

        # Select codec per config (v6.3 §3.10.9)
        if audio_codec == "aac":
            output_opts.extend(["-c:a", "aac", "-b:a", audio_bitrate])
        elif audio_codec == "flac":
            output_opts.extend(["-c:a", "flac"])
        else:
            output_opts.extend(["-c:a", audio_codec])
            if audio_bitrate:
                output_opts.extend(["-b:a", audio_bitrate])
    else:
        # R3-08 / B11：**缺临时音轨 ≠ 原片没有音轨**。
        # 旧实现只检查 temp 里的中间音轨，缺了就记"Source has no audio stream"
        # 并加 `-an` —— 实测源片明确含 PCM 音轨，产物却被 ffprobe 读成
        # `streams=[]`（静音视频），stage7 仍 rc=0。
        # 现在先核对源片本身：源片有声 → 直接从源片提取音轨；
        # 提取也失败 → 按 allow_silent_output 决定，默认**不允许**静默丢音轨。
        source_has_audio = _source_has_audio_stream(args.input)
        if source_has_audio:
            logger.warning("STAGE7_AUDIO_INTERMEDIATE_MISSING",
                           "No intermediate audio track in temp, but the "
                           "source has an audio stream; extracting it from "
                           "the source instead of dropping audio",
                           source=args.input)
            extracted = _extract_audio_from_source(
                args.input, str(Path(temp_dir) / f"{job_id}_{input_name}.mka"),
                logger)
            if extracted:
                audio_path = extracted
                audio_layout = _get_audio_layout(audio_path)
                audio_filter = _build_audio_filter(
                    audio_path, params, logger, tempo_ratio=tempo_ratio,
                    audio_layout=audio_layout)
                if audio_filter == "__INVALID_TEMPO__":
                    logger.error("STAGE7_AUDIO_FAIL",
                                 "Invalid tempo ratio, cannot process audio")
                    return 86
                if audio_filter == "__DENOISE_UNAVAILABLE__":
                    logger.error("STAGE7_DENOISE_FAIL",
                                 "Required denoise algorithm is unavailable "
                                 "and degradation is not authorized")
                    return 86
                video_cmd.extend(["-i", audio_path])
                output_opts.extend([
                    "-af", audio_filter,
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                ])
                output_opts.extend(["-c:a", params.get("audio_codec", "aac"),
                                    "-b:a",
                                    params.get("audio_bitrate", "320k")])
            elif not params.get("allow_silent_output", False):
                logger.error("STAGE7_AUDIO_REQUIRED_FAIL",
                             "The source has audio but no usable audio track "
                             "could be prepared; refusing to publish a silent "
                             "video (set allow_silent_output=true to accept)")
                return 86
            else:
                logger.warning("STAGE7_SILENT_OUTPUT_ALLOWED",
                               "Publishing without audio because "
                               "allow_silent_output=true")
                output_opts.append("-an")
        else:
            logger.info("STAGE7_NO_AUDIO",
                        "Source has no audio stream, outputting video only")
            output_opts.append("-an")

    # Color metadata (v6.3: double marking)
    # SVT-AV1 uses numeric values, not string names
    color_primaries = params.get("color_primaries", "bt709")
    color_trc = params.get("color_trc", "bt709")
    colorspace = params.get("colorspace", "bt709")

    # FIX-P7-9: FFmpeg -color_range uses 'tv'/'pc', not 'limited'/'full'
    ffmpeg_range = "tv" if color_range == "limited" else "pc"

    # Map color names to SVT-AV1 numeric values
    color_map = {
        "bt709": "1", "bt2020": "9", "bt470bg": "5", "smpte170m": "6",
        "bt601": "6", "bt470m": "4"
    }
    svt_primaries = color_map.get(color_primaries, "1")
    svt_trc = color_map.get(color_trc, "1")
    svt_matrix = color_map.get(colorspace, "1")
    svt_range = "0" if color_range == "limited" else "1"

    # FFmpeg global color metadata (string names)
    output_opts.extend([
        "-color_primaries", color_primaries,
        "-color_trc", color_trc,
        "-colorspace", colorspace,
        "-color_range", ffmpeg_range,
    ])
    
    # BUG-041 fix: build svtav1-params once after all parameters are known
    # FIX-R7-S7-13: -svtav1-params is libsvtav1-specific; skip for libx265
    # (x265 has no equivalent private-options key; color metadata is already
    # set via -color_primaries/-color_trc/-colorspace above).
    if video_encoder != "libx265":
        svt_params = f"tile-rows={tile_rows}:tile-columns={tile_cols}"
        svt_params += f":color-primaries={svt_primaries}:transfer-characteristics={svt_trc}"
        svt_params += f":matrix-coefficients={svt_matrix}:color-range={svt_range}"
        output_opts.extend(["-svtav1-params", svt_params])

    # Metadata and output（NEW-RV-04: 输出侧选项统一在输出路径之前落地）
    video_cmd.extend(output_opts)
    video_cmd.extend([
        "-metadata", "restored=true",
        # FIX-P7-12: Use proper datetime import per CLAUDE.md §4
        # ORIGINAL: __import__('datetime').datetime.now().isoformat()
        "-metadata", f"restored_at={datetime.datetime.now().isoformat()}",
        # FIX-R7-S7-A4: -fflags +genpts is an OUTPUT option ("封装时使用 -fflags
        # +genpts" per v7.32 §3.10.9) — placed here before the output path so
        # FFmpeg regenerates container PTS at mux time without modifying samples.
        "-fflags", "+genpts",
        "-y", output_path,
    ])

    try:
        # NEW-V-06/RV-12: 受管子进程——取消时 ffmpeg 一起收掉，
        # 否则守卫删掉的半截成品会被孤儿 ffmpeg 重新写回来。
        if _cancel_guard is not None:
            r = _cancel_guard.run_managed(video_cmd, capture_output=True,
                                          text=True, timeout=3600*2,
                                          shell=False, logger=logger)
        else:  # pragma: no cover - 守卫缺失时退回原行为
            r = subprocess.run(video_cmd, capture_output=True, text=True, errors="replace",
                               timeout=3600*2, shell=False)
        # FIX-P7-9: Use pathlib.Path per CLAUDE.md §4
        if r.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            # FIX-P7-4: Validate A/V sync on final muxed output (not separate files)
            if has_audio:
                sync_result = _validate_av_sync(output_path, logger)
                if sync_result != 0:
                    return sync_result
            
            # WSL redirect: copy back to original Linux path
            if original_output_path:
                if wsl_copy_back(output_path, original_output_path):
                    logger.info("STAGE7_WSL_COPYBACK",
                               f"Copied result back to {original_output_path}")
                    output_path = original_output_path
                else:
                    logger.error("STAGE7_WSL_COPYBACK_FAIL",
                                f"Failed to copy {output_path} back to {original_output_path}")
                    return 82
            
            logger.info("STAGE7_DONE", "Encoding complete", output=output_path)
            # NEW-V-06: 产物已完整落盘（含 WSL 回拷），取消守卫不再有权
            # 删它（RV-12：发布是不可逆的成功事件）。
            if _cancel_guard is not None:
                _cancel_guard.release_outputs()
            return 0
        else:
            logger.error("STAGE7_FAILED", f"Encoding failed: {r.stderr[:300]}")
            return 84
    except subprocess.TimeoutExpired:
        logger.error("STAGE7_TIMEOUT", "Encoding timed out")
        return 87
    except Exception as e:
        logger.error("STAGE7_ERROR", str(e))
        return 82


if __name__ == "__main__":
    sys.exit(main() or 0)
