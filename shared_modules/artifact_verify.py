"""产物核对（第七轮）：让"整体偏移一位"这类问题第一次跑就被拦住。

**为什么需要它**

2026-10-19 那个 60 帧 → 59 帧的 bug，根因只是 `global_idx = ... - 1`
多减了 1，导致帧号变成 -1..58，`-i frame_%08d.png` 从 0 读到 58 就停 ——
**源视频第一帧被整帧丢弃**。

我为此猜了三轮（concat 末帧 duration → 末帧重复 → 默认 vsync），
每次都"看起来有证据"，但都没打中。如果跑完就有一次机械核对，
第一次就能拦住。

**为什么不能简单地"比哈希"**

* 分辨率变了（1080p → 4K），逐像素哈希必然不同；
* 去隔行可能让帧数**翻倍**（bob），也可能不变（QTGMC 单倍率）；
* 编码是有损/无损都有可能，像素值会有微小差异。

**分层判据（从硬到软）**

| 层 | 指标 | 判据 | 能拦住什么 |
|---|---|---|---|
| 1 | 时长 | 相对误差 ≤ 3% | 整段截断/拼接错误 |
| 2 | 帧数 | 落在**允许集合**内 | 丢帧/多帧（本次 bug） |
| 3 | 首帧感知哈希 | 与源首帧汉明距离 ≤ 阈值 | **整批偏移**（本次 bug） |
| 4 | 末帧感知哈希 | 同上 | 末尾截断 |
| 5 | 画面非黑 | 均值 > 阈值、标准差 > 阈值 | 黑图/全白 |

第 2 层的"允许集合"是关键：不能硬编码"必须相等"。`expected_frame_counts()`
按去隔行/IVTC 配置推算可能的值。**推算不出来就降级为"只查时长 + 哈希"**，
并在结论里标明"帧数未校验"，不假装查过。

**感知哈希为什么能跨分辨率**

缩到 32×32 灰度 → DCT → 取左上 8×8 低频 → 与中位数比较得 64 bit。
低频分量描述的是**画面结构**，与分辨率、编码损失基本无关。
"""

from __future__ import annotations

import time

import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Set

__all__ = [
    "probe_media_facts",
    "perceptual_hash",
    "expected_frame_counts",
    "verify_output",
    "render_verdict",
    "hash_distance",
    "verify_frame_alignment",
    "normalized_correlation",
    "render_alignment",
]

#: 时长相对误差上限。放 3% 是因为容器时长与源可能差 1-2 帧。
DURATION_TOLERANCE = 0.03
#: 感知哈希汉明距离上限（64 bit 里最多差这么多位）。
PHASH_MAX_DISTANCE = 12
#: 对齐校验里"错位"的判定线：抽样点中偏向邻居的比例超过它就算错位。
#:
#: 实测：正常素材 40%、整体偏移一位 100%。取 70% 落在中间，
#: 两边都有余量。**这不是逐帧保证**，只是"整体错位"的证据。
ALIGNMENT_FLIP_RATIO = 0.7
#: 源相邻帧的"差异"低于它，这个采样点就没有判别力（两帧太像）。
#: 用 1-NCC 度量；0.01 大致对应"肉眼几乎看不出区别"。
MIN_SRC_FRAME_DIFF = 0.01
#: 有判别力的采样点少于这个数，就不下结论（记未校验）。
MIN_ALIGNMENT_SAMPLES = 3
#: 画面"不是黑图"的下限（0-255）。
MIN_MEAN = 8.0
MIN_STD = 3.0


def _run(cmd: List[str], timeout: int = 120) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, errors="replace")
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def _probe(path: str, entries: str, extra: Optional[List[str]] = None,
           count: bool = False) -> Optional[str]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count:
        cmd.append("-count_frames")
    cmd += ["-show_entries", entries, "-of", "default=nw=1:nokey=1", path]
    if extra:
        cmd = cmd[:-1] + extra + [path]
    out = _run(cmd)
    if out is None:
        return None
    return out.strip()


def _probe_all_streams(path: str, entries: str) -> Optional[str]:
    """不筛选流的探针 —— 用于判断"有没有音轨"这类全局事实。

    `_probe()` 写死了 `-select_streams v:0`，用它查 codec_type 永远只拿到
    "video"，`has_audio` 会恒为 False（那样 R16-06 的轨构成判断就失效）。
    """
    out = _run(["ffprobe", "-v", "error", "-show_entries", entries,
                "-of", "default=nw=1:nokey=1", path])
    if out is None:
        return None
    return out.strip()


def probe_media_facts(path: str) -> Dict[str, Any]:
    """一个媒体文件的硬事实：帧数、时长、分辨率、像素格式、首末帧哈希。

    帧数用 `-count_frames`（真解码计数）。**不要**用 `nb_read_packets`
    或"时长 × 帧率" —— 2026-10-19 那个丢帧 bug 就是靠"时长差不多"
    骗过了所有直觉判断（修复前 2.026s 反而比源 2.002s 更长）。
    """
    facts: Dict[str, Any] = {
        "path": path, "exists": os.path.exists(path),
        "frames": None, "duration": None, "width": None, "height": None,
        "pix_fmt": None, "first_frame_hash": None, "last_frame_hash": None,
        "mean": None, "std": None, "error": None,
    }
    if not facts["exists"]:
        facts["error"] = "file not found"
        return facts

    f = _probe(path, "stream=nb_read_frames", count=True)
    if f:
        try:
            facts["frames"] = int(f.splitlines()[0])
        except (ValueError, IndexError):
            pass

    d = _probe(path, "format=duration")
    if d:
        try:
            facts["duration"] = float(d.splitlines()[0])
        except (ValueError, IndexError):
            pass

    # R16-06：**容器时长 ≠ 视频时长**。容器取所有轨的最大值；成品带音轨、
    # 中间产物常常只有视频轨，两边直接比 `format.duration` 会得出假的
    # "时长差 3.6%"（实测：源视频 15.1307 / 源音频 15.7020 /
    # 中间产物 15.172 / 成品容器 15.723）。
    facts["video_duration"] = None
    facts["has_audio"] = None
    vd = _probe(path, "stream=duration")
    if vd:
        try:
            facts["video_duration"] = float(vd.splitlines()[0])
        except (ValueError, IndexError):
            pass
    ct = _probe_all_streams(path, "stream=codec_type")
    if ct:
        types = [x.strip() for x in ct.splitlines() if x.strip()]
        facts["has_audio"] = ("audio" in types)

    for key, entry in (("width", "stream=width"), ("height", "stream=height"),
                       ("pix_fmt", "stream=pix_fmt")):
        v = _probe(path, entry)
        if v:
            try:
                facts[key] = int(v.splitlines()[0]) if key != "pix_fmt" \
                    else v.splitlines()[0]
            except (ValueError, IndexError):
                pass

    # 首末帧哈希 + 亮度统计（一次解码拿首末两帧，避免解整片）
    hashes, stats = _first_last_frame_probe(path)
    facts["first_frame_hash"] = hashes.get("first")
    facts["last_frame_hash"] = hashes.get("last")
    stats_all = stats.get("all") or {}
    facts["mean"] = stats_all.get("mean")
    facts["std"] = stats_all.get("std")

    # GATE-02 补漏：原先只有"文件不存在"才设 error。ffprobe 超时 /
    # 非零退出 / 输出解析失败时，facts 会**全是 None 但 error 仍为 None**，
    # 于是 verify_output 里所有判据都变成 ok=None，最终可能 ok=True ——
    # "什么都没量到"反而通过。
    # 这里只兜最保守的一档：连尺寸都拿不到，说明探测彻底失败。
    # （照片是单帧、frames 可能合法地为 None，所以不能拿 frames 判。）
    if facts.get("width") is None and facts.get("height") is None:
        facts["error"] = facts.get("error") or "probe produced no usable facts"
    return facts


def _first_last_frame_probe(path: str) -> tuple:
    """解出首帧和末帧，算感知哈希 + 亮度统计。

    实现：抽 1 帧（首）+ 用 `-sseof -1` 从末尾 1 秒里抽 1 帧（末）。
    两次单独调用，避免为了末帧解码整片。
    """
    import tempfile

    hashes: Dict[str, str] = {}
    stats: Dict[str, Any] = {"all": {}}
    with tempfile.TemporaryDirectory() as td:
        for which, args in (
            ("first", []),
            ("last", ["-sseof", "-1"]),
        ):
            out = os.path.join(td, f"{which}.png")
            cmd = ["ffmpeg", "-v", "error"] + args + ["-i", path,
                   "-frames:v", "1", "-y", out]
            if _run(cmd, timeout=180) is None or not os.path.exists(out):
                continue
            h = perceptual_hash(out)
            if h:
                hashes[which] = h
            s = _frame_stats(out)
            if s:
                stats[which] = s
        # 用首帧统计作为"整片是否黑图"的代表（整片都黑时首帧必然黑）
        if "first" in stats and stats["first"]:
            stats["all"] = stats["first"]
    return hashes, stats


def _frame_stats(img_path: str) -> Optional[Dict[str, float]]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    a = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if a is None:
        return None
    if a.dtype != np.uint8:
        # 16bit → 8bit，按最大值缩放（不要直接 astype）
        scale = 65535.0 if a.dtype == np.uint16 else float(a.max() or 1)
        a = (a.astype(np.float32) / scale * 255.0).clip(0, 255).astype(np.uint8)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    return {"mean": round(float(a.mean()), 2),
            "std": round(float(a.std()), 2)}


def perceptual_hash(img_path: str, hash_size: int = 8) -> Optional[str]:
    """64-bit 感知哈希（DCT pHash）的十六进制串。

    缩到 32×32 灰度 → DCT → 取左上 8×8 低频 → 与中位数比较。
    低频描述画面结构，因此**跨分辨率**仍可比（1080p vs 4K）。
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    a = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if a is None:
        return None
    if a.dtype != np.uint8:
        scale = 65535.0 if a.dtype == np.uint16 else float(a.max() or 1)
        a = (a.astype(np.float32) / scale * 255.0).clip(0, 255).astype(np.uint8)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(a, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, :])          # 跳过 DC 分量
    bits = (low > med).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(bool(b))
    return f"{val:016x}"


def hash_distance(h1: Optional[str], h2: Optional[str]) -> Optional[int]:
    """两个十六进制感知哈希的汉明距离。任一为空返回 None。"""
    if not h1 or not h2:
        return None
    try:
        a, b = int(h1, 16), int(h2, 16)
    except ValueError:
        return None
    return bin(a ^ b).count("1")


def expected_frame_counts(src_frames: Optional[int],
                          params: Optional[Dict[str, Any]]) -> Optional[Set[int]]:
    """按配置推算**允许**的成品帧数集合。

    推算规则（来自 stage1 的实际行为）：

    * 不做去隔行/IVTC → 帧数不变；
    * QTGMC + `qtgmc_fps_divisor=1`（默认）→ 单倍率，帧数不变；
    * QTGMC + `qtgmc_fps_divisor=2` → bob 后减半，帧数**不变**（工业界的坑）；
    * bob（每场一帧）→ 帧数 × 2；
    * IVTC（`vdecimate_cycle=5`）→ 帧数 × 4/5。

    无法推算（缺参数或源帧数未知）时返回 None —— 调用方必须把
    "帧数未校验"写进结论，不能假装查过。
    """
    if not src_frames or src_frames <= 0:
        return None
    p = params or {}
    allowed: Set[int] = {src_frames}

    deint = str(p.get("deinterlace") or "").upper()
    ivtc = bool(p.get("ivtc"))
    divisor = p.get("qtgmc_fps_divisor")

    if ivtc:
        cycle = int(p.get("vdecimate_cycle") or 5)
        allowed.add(int(round(src_frames * (cycle - 1) / cycle)))
    if deint in ("BOB", "YADIF", "YADIFMOD", "BOBWEAVE"):
        # 每场一帧
        allowed.add(src_frames * 2)
    elif deint == "QTGMC":
        # QTGMC 默认单倍率；显式 divisor=2 时仍是"输入帧率"，不翻倍
        try:
            d = int(divisor) if divisor is not None else 1
        except (TypeError, ValueError):
            d = 1
        if d == 1:
            allowed.add(src_frames)
        else:
            allowed.add(src_frames)

    # **不加 ±1 容差**。
    #
    # 第一版给了 `{n-1, n, n+1}`，本意是容忍真实编码器的尾部处理。
    # 但实测证明这恰好放过了 2026-10-19 那个 60→59 的 bug ——
    # 允许集合变成 {59,60,61}，59 在里面，核对直接通过。
    #
    # 尾部 1 帧差异确实存在，但它和"整批丢一帧"在帧数上**无法区分**。
    # 所以这里返回精确集合，由 `verify_output` 把"差 1 帧"单独标成
    # `FRAME_COUNT_OFF_BY_ONE`，明确提示"两种可能，需人工确认"——
    # 既不默默放过，也不一刀切阻断。
    return {n for n in allowed if n > 0}


def verify_output(source_path: str, output_path: str,
                  params: Optional[Dict[str, Any]] = None,
                  *, strict: bool = False) -> Dict[str, Any]:
    """核对成品与源。返回结构化结论，不抛异常（核对本身不该弄坏产物）。"""
    src = probe_media_facts(source_path)
    out = probe_media_facts(output_path)
    issues: List[Dict[str, Any]] = []
    checks: Dict[str, Any] = {}

    if not out["exists"]:
        # 与 SOURCE_UNREADABLE 同理：这是提前 return，不经过尾部汇总，
        # 必须自带顶层 error / hard_error，否则全链门收不到信号。
        return {"ok": False,
                "issues": [{"level": "error",
                            "code": "OUTPUT_MISSING",
                            "message": "成品不存在"}],
                "source": src, "output": out, "checks": {},
                "error": "OUTPUT_MISSING",
                "hard_error": "OUTPUT_MISSING"}

    # FIX-2026-10-22：源不可读时必须**明确失败**，不能静默放行。
    #
    # 原来只判了 `out["exists"]`，从没判过 `src["exists"]`。于是源丢失
    # （或被删、或路径写错）时，下面 5 项判据全部退化成 `ok: None`
    # （"未校验"），issues 里一个 error 都没有 → 返回 `ok=True`。
    #
    # 实测（2026-10-22，E 盘清理把 real_run/ 下的测试样本删掉后）：
    #   verify_output(缺失的源, 59 帧成品) → ok=True
    # 明明一项都没核对，结论却是通过 —— 与 R16「全黑图 rc=0」、
    # manifest「0 jobs RC=0」同一族：**指标正常、结论错误**。
    #
    # 核对是发布前的最后一道防线，它自己绝不能在"没核对"时报通过。
    if not src["exists"] or src.get("error"):
        # GATE-02：这条是**提前 return**，必须同样带上顶层 error /
        # hard_error，否则 stage7 的 _fullchain_blockers 收不到信号，
        # 在 frames_only 模式下仍会放行（它只看 checks 与 error）。
        return {"ok": False,
                "issues": [{"level": "error",
                            "code": "SOURCE_UNREADABLE",
                            "message": f"源不可读（{src.get('error') or 'file not found'}）："
                                       f"帧数/时长/首末帧/黑图全部无法核对，"
                                       f"拒绝判为通过"}],
                "source": src, "output": out, "checks": {},
                "error": "SOURCE_UNREADABLE",
                "hard_error": "SOURCE_UNREADABLE"}

    # 1) 时长 —— **按帧粒度判断，不用固定百分比**
    #
    # 第一版用 3% 相对误差，结果在 1 秒 25 帧的短素材上误报：1 帧的 PTS
    # 舍入就是 1/25 = 4%。百分比阈值对短素材天然苛刻。
    #
    # 改成"差了几帧"：帧数本身已经严格校验（第 2 项），时长只是辅助，
    # 所以只要差不超过 max(2 帧, 3%) 就算一致。
    # ---- R16-06：先决定**用哪个时长**比 ----
    # 两边都有视频轨时长 → 用它（同一口径）。
    # 否则，若一边有音轨、另一边没有，容器时长本来就不可比 → 不判失败。
    _sv, _ov = src.get("video_duration"), out.get("video_duration")
    _cmp_src = _cmp_out = None
    _basis = ""
    if _sv and _ov:
        _cmp_src, _cmp_out = _sv, _ov
        _basis = "video_stream"
    elif (src.get("has_audio") is not None
          and out.get("has_audio") is not None
          and src.get("has_audio") != out.get("has_audio")):
        checks["duration"] = {
            "ok": None,
            "note": ("两边的轨构成不同（一边有音轨、一边没有），"
                     "容器时长不可比；本次不按时长判定"),
            "src_container": src.get("duration"),
            "out_container": out.get("duration"),
            "src_has_audio": src.get("has_audio"),
            "out_has_audio": out.get("has_audio"),
        }
        issues.append({"level": "warning",
                       "code": "DURATION_NOT_COMPARABLE",
                       "message": ("成品与中间产物的轨构成不同，容器时长"
                                   "不可比（源 "
                                   f"{src.get('duration')}s / 成品 "
                                   f"{out.get('duration')}s）——未按时长判定")})
    else:
        _cmp_src, _cmp_out = src.get("duration"), out.get("duration")
        _basis = "container"

    if _cmp_src and _cmp_out:
        # R25-02/R02-R03：**选定口径后必须一直用它**。
        #
        # 原实现先算出 `_cmp_src/_cmp_out`（优先视频流时长），
        # 紧接着又用 `src["duration"]`（容器时长）把 `rel` 覆盖掉，
        # 后面的 frame_dur 与 checks 里的 src/out 也全是容器时长 ——
        # 等于"先选了视频轨，又悄悄换成容器"。
        #
        # 审核用两个反例说明这有多严重：
        #   * 源视频 10s / 候选视频 6s / 两边容器都 10s  → 通过（漏掉 4s 损失）
        #   * 视频都 10s / 候选音轨把容器拉到 14s        → 失败（假报警）
        rel = abs(_cmp_out - _cmp_src) / _cmp_src
        diff_frames = None
        allowed_frames = None
        if src.get("frames") and _cmp_src > 0:
            frame_dur = _cmp_src / src["frames"]
            if frame_dur > 0:
                diff_frames = abs(_cmp_out - _cmp_src) / frame_dur
                allowed_frames = max(2.0, src["frames"] * DURATION_TOLERANCE)
        if diff_frames is not None:
            ok = diff_frames <= allowed_frames
            checks["duration"] = {
                "src": _cmp_src, "out": _cmp_out,
                "rel_error": round(rel, 4),
                "diff_frames": round(diff_frames, 2),
                "allowed_frames": round(allowed_frames, 2),
                "ok": ok,
                # 口径写清楚：这一条比的是**视频流**时长（不是容器），
                # 音频覆盖由 audio 项单独判。
                "basis": _basis,
            }
        else:
            ok = rel <= DURATION_TOLERANCE
            checks["duration"] = {"src": _cmp_src,
                                  "out": _cmp_out,
                                  "rel_error": round(rel, 4), "ok": ok,
                                  "basis": _basis,
                                  "note": "源帧数未知，退回百分比判据"}
        if not ok:
            issues.append({"level": "error", "code": "DURATION_MISMATCH",
                           "message": f"时长差 {rel:.1%}"
                                      f"（源 {_cmp_src:.3f}s → "
                                      f"成品 {_cmp_out:.3f}s，"
                                      f"口径={_basis}）"})
    else:
        checks["duration"] = {"ok": None, "note": "时长未知，未校验"}

    # 2) 帧数
    allowed = expected_frame_counts(src.get("frames"), params)
    if allowed and out.get("frames"):
        ok = out["frames"] in allowed
        checks["frames"] = {"src": src["frames"], "out": out["frames"],
                            "allowed": sorted(allowed), "ok": ok}
        if not ok:
            diff = out["frames"] - src["frames"]
            if abs(diff) == 1:
                # 差 1 帧有两种可能，实测无法自动区分：
                #   a) 编码器尾部处理（无害）
                #   b) 整批偏移一位（丢首帧/末帧，有害）
                # 2026-10-19 的 bug 就是 b)，而第一版的 ±1 容差把它放过了。
                issues.append({
                    "level": "error", "code": "FRAME_COUNT_OFF_BY_ONE",
                    "message": f"帧数差 1：源 {src['frames']} → 成品 "
                               f"{out['frames']}。可能是编码器尾部处理"
                               "（无害），也可能是整批偏移一位（丢首/末帧，"
                               "有害）—— **必须人工确认**：对比首末帧画面，"
                               "或看中间帧文件名是否从 0 开始连续。",
                })
            else:
                issues.append({"level": "error",
                               "code": "FRAME_COUNT_MISMATCH",
                               "message": f"帧数不符：源 {src['frames']} → "
                                          f"成品 {out['frames']}"
                                          f"（差 {diff:+d}），"
                                          f"允许 {sorted(allowed)}"})
    else:
        checks["frames"] = {"ok": None, "note": "无法推算允许帧数，未校验"}

    # 3) 首帧 / 4) 末帧 感知哈希
    for label, key in (("首帧", "first_frame_hash"),
                       ("末帧", "last_frame_hash")):
        d = hash_distance(src.get(key), out.get(key))
        if d is None:
            checks[key] = {"ok": None, "note": f"{label}哈希不可用，未校验"}
            continue
        ok = d <= PHASH_MAX_DISTANCE
        checks[key] = {"distance": d, "ok": ok}
        if not ok:
            # **只报 warning，不当 error**。
            #
            # 实测（2026-10-19，testsrc2 素材，64bit pHash）：
            #   同源相邻帧          17 / 14
            #   源 vs 成品同帧      14 / 12      ← 正常差异（去隔行+2x超分）
            #   源 vs 成品偏移一位  17 / 27      ← 与正常差异**区间重叠**
            #
            # 相邻帧本身就很像的素材上，pHash 分不出"偏移一位"。
            # 它能可靠拦住的是"画面完全不对"（距离 25-32）和黑图。
            # 拿它当偏移检测器会误报，也会漏报 —— 不能这么用。
            issues.append({
                "level": "warning",
                "code": "FRAME_CONTENT_SUSPECT",
                "message": f"{label}画面与源差异较大（汉明距离 {d} > "
                           f"{PHASH_MAX_DISTANCE}）。注意：pHash 对"
                           "相邻帧相似的素材区分度有限，这只是提示，"
                           "不代表一定有问题。",
            })

    # 5) 非黑图 —— **必须和源比，不能只看成品**
    #
    # 第一版只查"成品首帧是否纯色"，结果在一个首帧本来就是黑的测试素材上
    # 误报（源 mean=0.0，成品 mean=0.0，判定"黑图产物"）。源本来就黑时，
    # 成品黑是**正确的**。
    #
    # 现在的判据是"一致性"：
    #   源非黑 + 成品黑   → error（真问题：内容丢了）
    #   源黑     + 成品黑 → OK（一致）
    #   源黑     + 成品非黑 → warning（可能调了亮度，值得看一眼）
    s_mean, s_std = src.get("mean"), src.get("std")
    o_mean, o_std = out.get("mean"), out.get("std")
    if None in (o_mean, o_std):
        checks["not_black"] = {"ok": None, "note": "亮度未知，未校验"}
    elif None in (s_mean, s_std):
        ok = o_mean >= MIN_MEAN and o_std >= MIN_STD
        checks["not_black"] = {"mean": o_mean, "std": o_std, "ok": ok,
                               "note": "源亮度未知，只查成品"}
        if not ok:
            issues.append({"level": "error", "code": "BLANK_FRAMES",
                           "message": f"成品首帧接近纯色（mean={o_mean}, "
                                      f"std={o_std}），且源亮度无法确认"})
    else:
        src_blank = s_mean < MIN_MEAN or s_std < MIN_STD
        out_blank = o_mean < MIN_MEAN or o_std < MIN_STD
        checks["not_black"] = {
            "src_mean": s_mean, "out_mean": o_mean,
            "ok": (not out_blank) if not src_blank else out_blank,
            "note": ("源首帧本身接近纯色，成品同为纯色属一致"
                     if src_blank else ""),
        }
        if src_blank and not out_blank:
            issues.append({"level": "warning", "code": "BRIGHTNESS_CHANGED",
                           "message": f"源首帧接近纯色（mean={s_mean}），"
                                      f"成品却有内容（mean={o_mean}）—— "
                                      "亮度可能被改动过，值得看一眼"})
        elif not src_blank and out_blank:
            issues.append({"level": "error", "code": "BLANK_FRAMES",
                           "message": f"源有内容（mean={s_mean}）但成品首帧"
                                      f"接近纯色（mean={o_mean}）—— "
                                      "内容丢失"})

    # GATE-02 补漏：探测失败本身必须是 error，而不是"所有判据都 None"。
    # 源/成品任一侧拿不到可用事实，就不能声称核对通过。
    _probe_err = src.get("error") or out.get("error")
    if _probe_err:
        issues.append({"level": "error", "code": "PROBE_UNUSABLE",
                       "message": f"媒体事实探测失败：{_probe_err}"})

    errors = [i for i in issues if i["level"] == "error"]
    if strict:
        errors += [i for i in issues if i["level"] == "warning"]

    # 硬错误（内容丢失 / 探测失败）与模式无关，都要拦：
    # 全链门按模式只放宽"时长"，不该放宽"黑图"这类跨模式硬错误。
    _hard = ""
    for _i in errors:
        if _i.get("code") in ("BLANK_FRAMES", "SOURCE_UNREADABLE",
                              "PROBE_UNUSABLE"):
            _hard = _i["code"]
            break

    return {"ok": not errors, "issues": issues, "source": src,
            "output": out, "checks": checks,
            "error": _probe_err or _hard or None,
            "hard_error": _hard or None}


def render_verdict(result: Dict[str, Any]) -> str:
    """人能直接读的核对结论。"""
    lines: List[str] = []
    src, out = result.get("source") or {}, result.get("output") or {}
    lines.append(f"源:   {src.get('frames')} 帧 / {src.get('duration')}s / "
                 f"{src.get('width')}x{src.get('height')}")
    lines.append(f"成品: {out.get('frames')} 帧 / {out.get('duration')}s / "
                 f"{out.get('width')}x{out.get('height')} "
                 f"{out.get('pix_fmt')}")
    lines.append("")
    for name, c in (result.get("checks") or {}).items():
        if not isinstance(c, dict):
            continue
        state = "OK" if c.get("ok") else ("未校验" if c.get("ok") is None
                                          else "FAIL")
        detail = c.get("note") or ", ".join(
            f"{k}={v}" for k, v in c.items() if k not in ("ok", "note"))
        lines.append(f"  [{state:6s}] {name}: {detail}")
    if result.get("issues"):
        lines.append("")
        for i in result["issues"]:
            lines.append(f"  ! {i['code']}: {i['message']}")
    lines.append("")
    lines.append("结论: " + ("通过" if result.get("ok") else "**不通过**"))
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="核对成品与源：帧数/时长/首末帧/非黑图")
    ap.add_argument("--source", "-s", required=True, help="源文件")
    ap.add_argument("--output", "-o", required=True, help="成品文件")
    ap.add_argument("--deinterlace", default="", help="去隔行方式（QTGMC/BOB/...）")
    ap.add_argument("--qtgmc-fps-divisor", type=int, default=1)
    ap.add_argument("--ivtc", action="store_true", help="是否 IVTC")
    ap.add_argument("--strict", action="store_true", help="warning 也算失败")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    a = ap.parse_args()

    params = {"deinterlace": a.deinterlace,
              "qtgmc_fps_divisor": a.qtgmc_fps_divisor,
              "ivtc": a.ivtc}
    result = verify_output(a.source, a.output, params, strict=a.strict)
    if a.json:
        import json as _json
        print(_json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render_verdict(result))
    sys.exit(0 if result["ok"] else 86)


# ===========================================================================
# 逐帧对齐校验：抓"帧数对、内容错位"
# ===========================================================================
#
# **为什么光看 pHash 不够**
#
# 2026-10-19 实测：`testsrc2` 素材上，源与成品的正常差异距离是 14，
# 而"偏移一位"的距离是 17 —— 区间重叠，分不开。根因是相邻帧本身太像。
#
# **换个问法就分得开了**
#
# 不要问"src[k] 和 out[k] 像不像"（绝对相似度，受素材影响大），
# 而要问"src[k] 更像 out[k]，还是更像 out[k+1]"（相对比较）。
#
#   src[k] vs out[k]   应该**最像**
#   src[k] vs out[k±1] 应该**明显更不像**
#
# 即使相邻帧很像，只要"自己那帧"始终比"邻居"更像，顺序就是对的。
# 这个判据对"整体相似度高低"不敏感，只对**排序**敏感 —— 正是我们要的。
#
# 度量用归一化互相关（NCC），对亮度/对比度变化比 SSIM 更宽容，
# 且 cv2 一行就能算，不必额外依赖 scikit-image。
#
# **适用边界（实测，2026-10-19）——这条很重要**
#
# | 场景 | 结果 |
# |---|---|
# | 纯重编码（帧序不变） | 5/5 点 `对应=1.0000`、邻居 0.65-0.96 → **可靠** |
# | 全链含 QTGMC 去隔行 | 8/8 点 flipped → **不可用** |
#
# 后者不是阈值问题，是**物理上不对应**：QTGMC 用时间邻域重建每一帧，
# 输出的第 k 帧不等于输入的第 k 帧时刻。拿"源第 k 帧"去比"成品第 k 帧"
# 本来就不该相等。IVTC（4/5 抽帧）、补帧同理。
#
# 所以这个函数**只该用在帧序不变、时间基准不变的两端**：
# 典型是 stage7 的「输入（step6 产物）vs 输出（成品）」——
# 那里是纯编码，实测对应帧相关系数 1.0000，判别力极强。
# 不要拿它去比"原始源 vs 全链成品"。


def _extract_frames_batch(path: str, idxs, out_dir: str, prefix: str,
                          width: int = 320, logger=None) -> dict:
    """一次解码抽出**多帧**，返回 ``{帧号: png 路径}``。

    ## 为什么不复用 `_extract_frame`

    它每次都 `-i path` 从头解码到目标帧（**没有 `-ss` 定位**）。长片实测：
    `verify_frame_alignment(samples=4)` 要 4 个点 × 3 帧 = **12 次启动**，
    平均每次解 4000 帧 ≈ 把 2.7 GB 源**解码了 6 遍** → 26 分钟，
    占整条链 1h54m 的 23%。

    一次 `select` 只解一遍，同样素材降到 2~3 分钟。判据一个字没改。

    ## 一个必须防的错位

    `-vsync 0` 下输出文件数 = **实际选中的帧数**。如果请求里混进了超出范围的
    帧号，输出会少一个，于是"第 j 个文件对应第 j 个帧号"的映射**整体错位**
    —— 那会拿错误的帧去比，得出看似合理但完全错误的结论。
    所以：数量对不上就返回空 dict，让调用方按"抽帧失败"处理，**绝不猜**。
    """
    import glob as _glob
    want = sorted({int(i) for i in idxs if int(i) >= 0})
    if not want:
        return {}
    sel = "+".join(f"eq(n\\,{i})" for i in want)
    pattern = os.path.join(out_dir, f"{prefix}_%05d.png")
    for _old in _glob.glob(os.path.join(out_dir, f"{prefix}_*.png")):
        try:
            os.remove(_old)
        except OSError:
            pass
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-vf", f"select='{sel}',scale={width}:-2",
           "-vsync", "0", "-y", pattern]

    # ---- 不用固定总时长，用"输出还在长"判活 ----
    # 我第一版写的是 `_run(cmd, timeout=1800)` —— 同一个文件里的
    # `retime_to_source_pts` 已经明确记过这个教训（"4K 16 位中间件单件
    # 250 GB，30 分钟根本不够；换个更长的素材又得再调一次"），我还是照抄了。
    # 硬编码总时长的本质错误：**把"跑多久"当成了可以预先知道的量**，
    # 而它取决于素材大小/码率/磁盘速度。
    # 抽帧的**真实进展**是输出 PNG 的总字节在长；不长才是卡住。
    import subprocess as _sp
    def _bytes_watch():
        try:
            return sum(os.path.getsize(f) for f in
                       _glob.glob(os.path.join(out_dir, f"{prefix}_*.png")))
        except Exception:
            return 0
    try:
        _p = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.PIPE, shell=False)
    except Exception:
        return {}
    try:
        import cancel_guard as _cg
        _cg.collect_outputs_while_waiting(
            [_p], heartbeat=_bytes_watch,
            inactivity_timeout=getattr(_cg, "INACTIVITY_TIMEOUT_S", 300),
            logger=logger, label="extract-frames")
    except Exception:
        try:
            _p.kill(); _p.wait()
        except Exception:
            pass
        return {}
    files = sorted(_glob.glob(os.path.join(out_dir, f"{prefix}_*.png")))
    if len(files) != len(want):
        return {}
    return {k: files[j] for j, k in enumerate(want)}


# 抽帧是否优先用 seek。**默认 False —— 实测 `-ss` 抽帧不可靠**。
#
# 完整体测矩阵（240 帧素材，与"顺序解码"逐像素对拍）：
#
# | seek 方式                    | ffv1        | h264        | 1fps 素材 |
# |------------------------------|-------------|-------------|-----------|
# | 单段 `-ss t -i f`            | 差 32764 ✗  | 差 201 ✗    | —         |
# | 两段 `-ss t-2 -i f -ss 2`    | 差 0 ✓      | 差 0 ✓      | **抽到 k+1 ✗** |
# | `-ss` 放在 `-i` 之后（精确） | 对，但等于全片解码 | 同左 | 同左 |
#
# 两段式看似能用，但它第二段是**相对位移**，落点取决于第一段落在哪个
# 关键帧 —— 于是结果随素材帧率/GOP 结构漂移。而这条判据判的正是
# "是否错位 1 帧"，**任何不稳定都等于把要检测的误差引入测量**。
#
# 所以：宁可慢（批量 20.9 分钟），不用不可靠的快。
# 函数 `_extract_frames_seek` 保留，供**人工诊断**用（速度确实快：
# 长片上游件 4 帧 6.5 秒），但**不进判据链路**。
_SEEK_EXTRACT_ENABLED = False


def _extract_frames_seek(path: str, idxs, out_dir: str, prefix: str,
                         width: int = 320, logger=None) -> dict:
    """用**快 seek + 精 seek** 抽指定帧，只解 1 个 GOP；精度不可靠时返回空。

    ## 为什么需要它

    `_extract_frames_batch` 一次解码抽完，省掉了"12 次启动"的重复，
    但**省不掉"至少读完一遍"**：长片上游件 16.9 GB，实测仍要 20.9 分钟。
    真正的大头是数据量。

    ## 怎么做到又快又准

    `-ss T` 放在 `-i` **之前**是快速 seek（落到最近关键帧），精度不保证；
    放在 `-i` **之后**是精确 seek，但要从头解到 T。

    组合用法：先 `-ss (T-2)` 快 seek 到目标前 2 秒的关键帧，再用
    `-ss 2`（此时是**相对**前移，且是在已 seek 的流上做精确解码）
    走到 T。只解约 2 秒的内容（50 帧），而不是整片。

    ## 为什么需要 `probe_pts` 而不是按帧号除以帧率

    VFR 源上 `k / fps` 是错的。`probe_pts` 只读容器索引（长片 8160 帧约 9 秒），
    拿到的是每帧真实 PTS。

    ## 精度

    对 intra-only（ffv1）源，`-ss` 落点是"pts ≤ T 的最后一帧"，T 精确时即第 k 帧。
    对带 GOP 的源，`-ss (T-2) -i` 落在此前最近关键帧，`-ss 2` 精确前移。
    两种情况下都应落在第 k 帧；**调用方仍需抽样对拍确认**（见测试）。
    """
    import glob as _glob
    import subprocess as _sp
    want = sorted({int(i) for i in idxs if int(i) >= 0})
    if not want:
        return {}
    # 输出目录必须自己建：调用方可能传一个还不存在的路径（我就是这么用的），
    # 而 ffmpeg 不会替你建目录 —— 它会静默失败，于是本函数返回空，
    # 看上去就像"这个文件不支持 seek"。我第一次测长片时正是栽在这里，
    # 差点把一条**可用**的优化当成"无效"永久关掉。
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        return {}
    # ---- 先判 CFR：CFR 下时间戳就是 k/fps，**不需要**全片读 PTS ----
    # 这一步至关重要：`probe_pts` 对 16.9 GB 的 **ffv1** 上游件实测
    # **120 秒超时、读到 0 帧**（ffv1 没有帧索引，只能全片扫）。
    # 而 ffprobe 的 `r_frame_rate` / `avg_frame_rate` 只要几百毫秒。
    # 两者相等即 CFR —— 此时 t = k / fps 就是精确时间戳。
    _times = {}
    try:
        import json as _json
        _r = _sp.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate,avg_frame_rate",
             "-of", "json", path],
            capture_output=True, text=True, errors="replace",
            timeout=30, shell=False)
        _st = ((_json.loads(_r.stdout or "{}").get("streams") or [{}])[0] or {})
        _rf = str(_st.get("r_frame_rate") or "")
        _af = str(_st.get("avg_frame_rate") or "")
        if _rf and _rf == _af and "/" in _rf:
            _n, _, _d = _rf.partition("/")
            _fps = float(_n) / float(_d or 1)
            if _fps > 0:
                # **帧中点**，不是帧起始。
                # 用 `k / fps` 会让 seek 时间正好落在第 k 帧的边界上，
                # ffmpeg 取哪一帧取决于内部舍入 —— 实测在 1fps 素材上
                # 抽到的是 **k+1 帧**（`test_r7_artifact_verify.py::
                # test_alignment_passes_on_reencode` 抓到的：
                # 对应=0.9972 而邻居 k+1=1.0，正好反了）。
                # 加半个帧长，落在帧内部，边界歧义消失。
                _times = {k: (k + 0.5) / _fps for k in want}
    except Exception:
        _times = {}
    if not _times:
        # 非 CFR（或探测失败）才去读逐帧 PTS —— 它可能很慢，读不全就放弃，
        # 让调用方回退到"全片解码"的批量实现（正确性优先）。
        try:
            from vfr_timecodes import probe_pts as _probe
        except Exception:
            return {}
        pts = _probe(path)
        if len(pts) <= max(want):
            return {}
        _times = {k: float(pts[k]) for k in want}
    for _old in _glob.glob(os.path.join(out_dir, f"{prefix}_*.png")):
        try:
            os.remove(_old)
        except OSError:
            pass
    got = {}
    for j, k in enumerate(want):
        t = float(_times[k])
        lead = 2.0 if t > 2.0 else t
        dest = os.path.join(out_dir, f"{prefix}_{j:05d}.png")
        cmd = ["ffmpeg", "-v", "error",
               "-ss", f"{max(0.0, t - lead):.6f}", "-i", path,
               "-ss", f"{lead:.6f}",
               "-frames:v", "1", "-vf", f"scale={width}:-2",
               "-y", dest]
        try:
            _sp.run(cmd, capture_output=True, timeout=180, shell=False)
        except Exception:
            return {}
        if not os.path.exists(dest):
            return {}
        got[k] = dest
    return got


def _gray_float(img_path: str):
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    a = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if a is None:
        return None
    if a.dtype != np.uint8:
        scale = 65535.0 if a.dtype == np.uint16 else float(a.max() or 1)
        a = (a.astype(np.float32) / scale * 255.0).clip(0, 255).astype(np.uint8)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    return a.astype(np.float32)


def normalized_correlation(a, b) -> Optional[float]:
    """两张灰度图的归一化互相关（-1..1，1 表示完全一致）。"""
    try:
        import numpy as np
    except ImportError:
        return None
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        try:
            import cv2
            b = cv2.resize(b, (a.shape[1], a.shape[0]),
                           interpolation=cv2.INTER_AREA)
        except Exception:
            return None
    af, bf = a.ravel(), b.ravel()
    af = af - af.mean()
    bf = bf - bf.mean()
    denom = (float((af ** 2).sum()) ** 0.5) * (float((bf ** 2).sum()) ** 0.5)
    if denom <= 1e-9:
        return None
    return float((af * bf).sum() / denom)


def _frames_or_count(path: str) -> Optional[int]:
    """帧数：先读容器元数据，读不到就**真数一遍**。

    ## 为什么必须兜底

    mkv 的 `nb_frames` 经常不写。长片实测（2026-11-14，8160 帧 4.4 GB）：
    `probe_media_facts().frames is None` → `verify_frame_alignment` 直接返回
    `"帧数未知，无法做对齐校验"` → `frame_order = UNVERIFIED` → 整代被判
    "published but NOT verified"（exit 89），一次 48 分钟的跑批拿不到 PASS。

    "读不到元数据"和"无法校验"是两件事：前者可以多花一次解码解决。
    代价只在这条兜底路径上付（且只在发布前一次），不影响常规探测。

    `-count_frames` 走的是完整解码，对长片要几十秒；这是可接受的代价 ——
    比"整代未验证"轻得多。
    """
    facts = probe_media_facts(path)
    n = facts.get("frames")
    if n:
        return int(n)
    try:
        from frame_conservation import _count_frames as _cf
        m = _cf(path)
        return int(m) if m else None
    except Exception:                        # pragma: no cover - 部署异常
        return None


def verify_frame_alignment(source_path: str, output_path: str,
                           samples: int = 6,
                           margin: float = 0.02,
                           logger=None) -> Dict[str, Any]:
    """抽查若干位置，确认 src[k] 比 src[k±1] 更像 out[k]。

    Args:
        samples: 抽多少个位置（均匀分布）。
        margin: "自己那帧"要比"邻居"至少好这么多，避免噪声抖动误判。

    Returns:
        `{"ok": bool|None, "points": [...], "flipped": int,
          "note": str, "error": str|None}`

    `ok=None` 表示无法判定（抽帧失败/依赖缺失），调用方必须标"未校验"。
    """
    import tempfile

    out: Dict[str, Any] = {"ok": None, "points": [], "flipped": 0,
                           "skipped": 0, "note": "", "error": None}
    src_frames = _frames_or_count(source_path)
    out_frames = _frames_or_count(output_path)
    if not src_frames or not out_frames:
        out["error"] = "帧数未知，无法做对齐校验"
        return out
    n = min(src_frames, out_frames)
    if n < 4:
        out["error"] = f"帧数太少（{n}），对齐校验无意义"
        return out

    # 均匀取样，避开首末（首末可能有容器差异）
    idxs = sorted({max(1, min(n - 2, int(round(n * (i + 0.5) / samples))))
                   for i in range(samples)})

    try:
        with tempfile.TemporaryDirectory() as td:
            # ---- 一次解码抽完所有需要的帧 ----
            # 旧实现逐帧调 `_extract_frame`，每次都从头解到目标帧：4 个采样点
            # × 3 帧 = 12 次启动 ≈ 把源解码 6 遍，长片 26 分钟。
            # 这里每个文件只解一遍。**判据一个字没改**，只是取帧方式变了。
            need_src = sorted({k for k in idxs}
                              | {k + 1 for k in idxs if k + 1 < n})
            need_out = sorted({k for k in idxs}
                              | {k + 1 for k in idxs if k + 1 < n}
                              | {k - 1 for k in idxs if k - 1 >= 0})
            # ---- 先试 seek（只解 1 个 GOP），失败再回退全片解码 ----
            # 长片上游件 16.9 GB，全片解码 20.9 分钟；seek 只解目标前 2 秒。
            # 精度已验证：h264 源 / h264 成品 / ffv1 三种编码上，
            # seek 抽帧与顺序解码抽到的**是同一帧**（最大像素差 0，
            # 含首帧与末帧）。见 tests/test_p6_seek_extract_precision.py。
            _t0 = time.time()
            # ---- 默认走批量；seek 只对**小/有索引**的文件有意义 ----
            # 实测（2026-11-16）：对 16.9 GB 的 ffv1 上游件，
            #   * `probe_pts` 120 秒超时、读到 0 帧（ffv1 无帧索引）；
            #   * `-ss` 组合 seek 跑 600 秒仍未返回。
            # 也就是说 seek **救不了**"上游件过大"这个硬成本。
            # 而 h264 / 小 ffv1 上 seek 是精确且快的（见
            # tests/test_p6_seek_extract_precision.py），所以保留它，
            # 但**只在显式要求时**用（默认 False，避免长片掉进慢路径）。
            _use_seek = bool((globals().get("_SEEK_EXTRACT_ENABLED") or False))
            src_frames = (_extract_frames_seek(source_path, need_src, td, "s",
                                               logger=logger)
                          if _use_seek else {})
            _src_mode = "seek" if src_frames else "batch"
            if not src_frames:
                src_frames = _extract_frames_batch(source_path, need_src,
                                                   td, "s", logger=logger)
            if not src_frames:
                out["error"] = ("源抽帧失败（seek 与批量都取不到完整帧集），"
                                "拒绝在缺帧的情况下下结论")
                return out
            out_frames = (_extract_frames_seek(output_path, need_out, td, "c",
                                                logger=logger)
                          if _use_seek else {})
            _out_mode = "seek" if out_frames else "batch"
            if not out_frames:
                out_frames = _extract_frames_batch(output_path, need_out,
                                                   td, "c", logger=logger)
            if not out_frames:
                out["error"] = "成品抽帧失败（seek 与批量都取不到完整帧集）"
                return out
            out["extract_mode"] = {"source": _src_mode, "output": _out_mode,
                                   "seconds": round(time.time() - _t0, 1)}
            for k in idxs:
                s_png = src_frames.get(k)
                c_png = out_frames.get(k)
                if not s_png or not c_png:
                    out["error"] = f"第 {k} 帧缺失（批量抽帧未覆盖）"
                    return out
                self_sim = normalized_correlation(_gray_float(s_png),
                                                  _gray_float(c_png))
                # 邻居：往后一帧（错位最常表现为整体前移/后移一位）
                nbr = []
                p_png = out_frames.get(k + 1)
                if k + 1 < n and p_png:
                    nbr.append(("k+1", normalized_correlation(
                        _gray_float(s_png), _gray_float(p_png))))
                m_png = out_frames.get(k - 1)
                if k - 1 >= 0 and m_png:
                    nbr.append(("k-1", normalized_correlation(
                        _gray_float(s_png), _gray_float(m_png))))
                nbr = [(t, v) for t, v in nbr if v is not None]
                if self_sim is None or not nbr:
                    out["error"] = f"第 {k} 帧相似度计算失败"
                    return out
                best_nbr = max(v for _, v in nbr)
                # 两种判"对齐"：
                #   a) 几乎像素级一致（>= 0.999）—— 不可能错位，直接通过。
                #      实测：纯重编码时对应帧是 1.0000，而邻居 0.9972，
                #      只差 0.0028 < margin 0.02，会被误判成 flipped。
                #   b) 比最好的邻居至少好 margin（正常情况）。
                aligned = (self_sim >= 0.999
                           or self_sim >= best_nbr + margin)

                # **跳过"相邻帧本身几乎一样"的采样点**。
                #
                # 实测：合成素材上第 18 帧 对应=0.9765、邻居=0.9775，
                # 差 0.001 —— 这两帧肉眼几乎相同，任何相似度度量都分不出
                # 顺序。在这样的点上判"错位"是噪声，不是证据。
                #
                # 判据：源的第 k 帧与第 k+1 帧本身是否足够不同。
                # 太像就跳过（记 skipped），并让最终结论把"可判定点数"
                # 一起报出来 —— 不假装查过判不了的点。
                _s_next = src_frames.get(k + 1)
                src_self_diff = None
                if _s_next:
                    _a = _gray_float(s_png)
                    _b = _gray_float(_s_next)
                    _nc = normalized_correlation(_a, _b)
                    if _nc is not None:
                        src_self_diff = 1.0 - _nc      # 0 = 完全相同
                discriminative = (src_self_diff is None
                                  or src_self_diff >= MIN_SRC_FRAME_DIFF)
                point = {
                    "frame": k,
                    "self": round(self_sim, 4),
                    "neighbors": {t: round(v, 4) for t, v in nbr},
                    "best_neighbor": round(best_nbr, 4),
                    "src_frame_diff": (round(src_self_diff, 4)
                                       if src_self_diff is not None else None),
                    "aligned": aligned,
                    "discriminative": discriminative,
                }
                if not discriminative:
                    point["note"] = ("源相邻帧差异过小，此点无判别力，"
                                     "不计入统计")
                    out["skipped"] = out.get("skipped", 0) + 1
                else:
                    out["points"].append(point)
                    if not aligned:
                        out["flipped"] += 1
    except Exception as exc:            # 抽帧工具异常不该弄坏调用方
        out["error"] = str(exc)
        return out

    # 判据用**比例**，不用"每个点都必须过"。
    #
    # 实测（2026-10-19，60 帧 1080p→4K）：
    #   正常成品：2/5 个点被判 flipped（第 18 帧 对应 0.9765 vs 邻居
    #             0.9775，只差 0.001 —— 这两帧本身几乎一样）
    #   整体前移一位：5/5 全部 flipped
    #
    # 绝对区分很明显（40% vs 100%），但"全部通过"会让正常素材也报警。
    # 所以用 70% 作为错位判定线：正常素材即使有若干相似帧干扰，
    # 也很难有 70% 的点都偏向邻居；而真正错位时几乎必然接近 100%。
    total = len(out["points"])
    if total < MIN_ALIGNMENT_SAMPLES:
        out["ok"] = None
        out["flip_ratio"] = None
        out["error"] = (f"只有 {total} 个采样点有判别力"
                        f"（另 {out.get('skipped', 0)} 个因相邻帧过像被跳过），"
                        f"不足以判断帧序 —— 记为**未校验**")
        return out
    ratio = out["flipped"] / total
    out["flip_ratio"] = round(ratio, 3)
    out["ok"] = ratio < ALIGNMENT_FLIP_RATIO
    if out["ok"]:
        out["note"] = (f"{out['flipped']}/{total} 个抽样点偏向邻居"
                       f"（{ratio:.0%} < {ALIGNMENT_FLIP_RATIO:.0%}）"
                       "—— 帧序判定为正确")
    else:
        out["note"] = (f"{out['flipped']}/{total} 个抽样点偏向邻居"
                       f"（{ratio:.0%} ≥ {ALIGNMENT_FLIP_RATIO:.0%}）"
                       "—— **帧序可能整体错位**")
    if out["flipped"] and out["ok"]:
        out["note"] += ("；少量翻转点通常是相邻帧本身极像所致，"
                        "如需逐帧保证请人工复核这些位置")
    return out


def render_alignment(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    if result.get("ok") is None:
        return f"对齐校验: 未完成（{result.get('error')}）"
    for p in result.get("points") or []:
        mark = "OK  " if p["aligned"] else "FAIL"
        nbr = ", ".join(f"{k}={v}" for k, v in p["neighbors"].items())
        lines.append(f"  [{mark}] 第 {p['frame']:5d} 帧: 对应={p['self']:.4f}"
                     f"  邻居={nbr}")
    lines.append("  " + (result.get("note") or ""))
    return "\n".join(lines)
