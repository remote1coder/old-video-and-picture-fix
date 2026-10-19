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
        return {"ok": False, "issues": [{"level": "error",
                                         "code": "OUTPUT_MISSING",
                                         "message": "成品不存在"}],
                "source": src, "output": out, "checks": {}}

    # 1) 时长 —— **按帧粒度判断，不用固定百分比**
    #
    # 第一版用 3% 相对误差，结果在 1 秒 25 帧的短素材上误报：1 帧的 PTS
    # 舍入就是 1/25 = 4%。百分比阈值对短素材天然苛刻。
    #
    # 改成"差了几帧"：帧数本身已经严格校验（第 2 项），时长只是辅助，
    # 所以只要差不超过 max(2 帧, 3%) 就算一致。
    if src.get("duration") and out.get("duration"):
        rel = abs(out["duration"] - src["duration"]) / src["duration"]
        diff_frames = None
        allowed_frames = None
        if src.get("frames"):
            frame_dur = src["duration"] / src["frames"]
            if frame_dur > 0:
                diff_frames = abs(out["duration"] - src["duration"]) / frame_dur
                allowed_frames = max(2.0, src["frames"] * DURATION_TOLERANCE)
        if diff_frames is not None:
            ok = diff_frames <= allowed_frames
            checks["duration"] = {
                "src": src["duration"], "out": out["duration"],
                "rel_error": round(rel, 4),
                "diff_frames": round(diff_frames, 2),
                "allowed_frames": round(allowed_frames, 2),
                "ok": ok,
            }
        else:
            ok = rel <= DURATION_TOLERANCE
            checks["duration"] = {"src": src["duration"],
                                  "out": out["duration"],
                                  "rel_error": round(rel, 4), "ok": ok,
                                  "note": "源帧数未知，退回百分比判据"}
        if not ok:
            issues.append({"level": "error", "code": "DURATION_MISMATCH",
                           "message": f"时长差 {rel:.1%}"
                                      f"（源 {src['duration']:.3f}s → "
                                      f"成品 {out['duration']:.3f}s）"})
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

    errors = [i for i in issues if i["level"] == "error"]
    if strict:
        errors += [i for i in issues if i["level"] == "warning"]
    return {"ok": not errors, "issues": issues, "source": src,
            "output": out, "checks": checks}


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


def _extract_frame(path: str, index: int, out_png: str,
                   width: int = 320, timeout: int = 120) -> bool:
    """抽第 index 帧，缩到 width 宽（统一尺寸后可比）。"""
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-vf", f"select=eq(n\\,{index}),scale={width}:-2",
           "-frames:v", "1", "-y", out_png]
    return _run(cmd, timeout=timeout) is not None and os.path.exists(out_png)


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


def verify_frame_alignment(source_path: str, output_path: str,
                           samples: int = 6,
                           margin: float = 0.02) -> Dict[str, Any]:
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
    src_frames = probe_media_facts(source_path).get("frames")
    out_frames = probe_media_facts(output_path).get("frames")
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
            for k in idxs:
                s_png = os.path.join(td, f"s{k}.png")
                c_png = os.path.join(td, f"c{k}.png")
                p_png = os.path.join(td, f"p{k}.png")
                m_png = os.path.join(td, f"m{k}.png")
                if not _extract_frame(source_path, k, s_png):
                    out["error"] = f"源第 {k} 帧抽取失败"
                    return out
                if not _extract_frame(output_path, k, c_png):
                    out["error"] = f"成品第 {k} 帧抽取失败"
                    return out
                self_sim = normalized_correlation(_gray_float(s_png),
                                                  _gray_float(c_png))
                # 邻居：往后一帧（错位最常表现为整体前移/后移一位）
                nbr = []
                if k + 1 < n and _extract_frame(output_path, k + 1, p_png):
                    nbr.append(("k+1", normalized_correlation(
                        _gray_float(s_png), _gray_float(p_png))))
                if k - 1 >= 0 and _extract_frame(output_path, k - 1, m_png):
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
                _s_next = os.path.join(td, f"sn{k}.png")
                src_self_diff = None
                if _extract_frame(source_path, k + 1, _s_next):
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
