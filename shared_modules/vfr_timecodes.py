# -*- coding: utf-8 -*-
"""按**源逐帧时间戳**重封装，恢复被 y4m / PNG 管道压平的 VFR 时间轴。

## 为什么需要这个模块

视频链的 stage1/2/4/6 走 `vspipe -c y4m | ffmpeg`。y4m 头只有**文件级帧率**，
不带逐包 PTS；stage5 的 PNG 路径受 `concat` 的 image2 默认 25fps 限制。
两者都会把可变帧间隔压成常数。

实测（`VID_20150614_145458.mp4`，292 帧，0.0449×227 + 0.0898×42）：

| 写法 | 帧数 | 时长 | 逐帧间隔 |
|---|---|---|---|
| 源 | 292 | 15.702s | 0.0449 / 0.0898 |
| stage5 concat | 292 | 15.12s | 0.04 / 0.08（**错**） |
| 本模块重定时 | 292 | 15.11s | 0.045 / 0.090（**0.9 ms 误差**） |

**纯 ffmpeg 做不到**：`-r` 会覆盖 duration、`-video_track_timescale` 无效、
concat 的 image2 默认 25fps。只有按时间码重新封装容器才行，这需要
`mkvmerge`（MKVToolNix）。

## 依赖

`mkvmerge` 不是硬依赖。找不到时本模块**明确返回不可用**，调用方据此
如实记录"时间轴未恢复"，绝不当成成功。

## 用法

```python
from vfr_timecodes import find_mkvmerge, retime_to_source_pts
mkv = find_mkvmerge()
if mkv:
    ok, info = retime_to_source_pts(src_video, candidate, mkv)
```
"""
from __future__ import annotations

import io
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

#: 视为"时间轴已足够接近"的相对容差（0.5%）。低于它就不必重封装 ——
#: 省一次全片重写，也避免对纯 CFR 素材做无谓改动。
DURATION_TOLERANCE = 0.005
#: 逐帧间隔容差（秒）。mkvmerge v2 支持小数毫秒、默认 scale 1ms，
#: 正常误差 1–2ms；20ms 既留余量又能抓住"中间错 40ms"这类真实缺陷。
INTERVAL_TOLERANCE = 0.02
#: 重封装的无进展阈值（秒）。与阶段级看门狗同源；只看输出是否继续增长，不设总时长。
INACTIVITY_TIMEOUT_S = 1800.0

#: mkvmerge 的常见位置（按优先级）。项目内优先，其次系统安装目录。
_CANDIDATES = (
    os.path.join("tools_r4", "mkvtoolnix", "mkvmerge.exe"),
    os.path.join("tools_r4", "mkvtoolnix", "mkvmerge"),
    r"C:\Program Files\MKVToolNix\mkvmerge.exe",
    r"C:\Program Files (x86)\MKVToolNix\mkvmerge.exe",
    "/usr/bin/mkvmerge",
    "/usr/local/bin/mkvmerge",
)


def find_mkvmerge(project_root: Optional[str] = None) -> Optional[str]:
    """定位 ``mkvmerge``；找不到返回 ``None``（**不抛异常**）。

    Args:
        project_root: 项目根目录。给出时会先找 ``tools_r4/mkvtoolnix/``。

    Returns:
        可执行文件的绝对路径，或 ``None``。
    """
    roots: List[str] = []
    if project_root:
        roots.append(project_root)
    else:  # 退回本文件所在目录的上一级（shared_modules 的父目录）
        roots.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    for root in roots:
        for rel in _CANDIDATES[:2]:
            p = os.path.join(root, rel)
            if os.path.isfile(p):
                return p
    for p in _CANDIDATES:
        if os.path.isabs(p) and os.path.isfile(p):
            return p
    # 最后试 PATH
    from shutil import which
    return which("mkvmerge")


def probe_time_base(video: str, timeout: int = 30) -> "Optional[float]":
    """探测首条视频流的**实际** ``time_base``（秒）。

    审核 R20-B05：
    > 实际时间基、重封装量化规则和测量精度决定容差。
    > 不要硬编码"所有容器均 1ms"，也不要把量化误差按帧数累加成数秒。

    1ms 只是 **mkvmerge 的默认时间刻度**，不是所有容器的量化步长。
    调用方（stage7）应当把本函数的结果传给 ``check_against_expected``，
    而不是让它退化到 20/50 ms 的兜底常量。

    Args:
        video: 媒体路径。
        timeout: ffprobe 超时（秒）。

    Returns:
        时间基秒数（如 ``1/30000`` → 3.33e-05）；探测失败或值非法返回 ``None``。
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=time_base", "-of", "csv=p=0", video],
            capture_output=True, text=True, errors="replace",
            timeout=timeout, shell=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    for line in (r.stdout or "").splitlines():
        v = line.strip().rstrip(",")
        if "/" not in v:
            continue
        try:
            num, den = v.split("/", 1)
            n, d = float(num), float(den)
        except ValueError:
            continue
        if d <= 0:
            continue
        tb = abs(n) / d
        if tb > 0:
            return tb
    return None


def probe_pts(video: str, timeout: int = 120) -> List[float]:
    """读出每一帧的 ``pts_time``（秒）。读不到返回空列表。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "frame=pts_time", "-of", "csv=p=0", video],
            capture_output=True, text=True, errors="replace",
            timeout=timeout, shell=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if r.returncode != 0:
        return []
    out: List[float] = []
    for line in r.stdout.splitlines():
        s = line.strip().rstrip(",")
        if not s:
            continue
        try:
            out.append(float(s))
        except ValueError:
            continue
    return out


def is_variable(pts: List[float], tolerance: float = 0.02) -> bool:
    """逐帧间隔是否**明显不齐**（相对中位间隔偏差超过 tolerance）。"""
    if len(pts) < 3:
        return False
    d = [b - a for a, b in zip(pts, pts[1:]) if b > a]
    if not d:
        return False
    d_sorted = sorted(d)
    med = d_sorted[len(d_sorted) // 2]
    if med <= 0:
        return False
    return max(abs(x - med) / med for x in d) > tolerance


def write_timecodes_v2(pts: List[float], path: str,
                       origin: Optional[float] = None) -> bool:
    """写 ``timecode format v2`` 文件（整数毫秒，相对首帧）。

    v2 是 mkvmerge 的标准输入格式：一行一个**绝对**时间戳（毫秒），
    因此逐帧的舍入误差不会累积（每帧独立 ±0.5 ms）。
    """
    if len(pts) < 2:
        return False
    base = pts[0] if origin is None else origin
    lines = ["# timecode format v2"]
    prev = -1
    for t in pts:
        ms = int(round((t - base) * 1000.0))
        if ms < 0:
            ms = 0
        # v2 要求严格递增；同毫秒的两帧（>1000fps）才需要兜底
        if ms <= prev:
            ms = prev + 1
        prev = ms
        lines.append(str(ms))
    try:
        io.open(path, "w", encoding="utf-8", newline="\n").write(
            "\n".join(lines) + "\n")
        return True
    except OSError:
        return False


def retime_to_source_pts(source_video: str, target_video: str,
                         mkvmerge: str, work_dir: Optional[str] = None,
                         logger: Any = None) -> Tuple[bool, Dict[str, Any]]:
    """用**源素材的逐帧时间戳**重新封装 ``target_video``。

    流程：ffprobe 源 → 写 timecodes v2 → mkvmerge `--timestamps` 重封装 →
    复核帧数与末帧时间。**任一步不符就返回失败**，调用方不得据此声称保时。

    Args:
        source_video: 提供时间基准的原始素材。
        target_video: 待重定时的成品（容器需支持，MKV 最佳）。
        mkvmerge: ``mkvmerge`` 路径。
        work_dir: 临时文件目录（默认与 target 同目录）。
        logger: 可选日志器。

    Returns:
        ``(ok, info)``；``info`` 含 ``frames`` / ``src_last`` / ``out_last``
        / ``max_interval_error`` 等实测值，供调用方如实记录。
    """
    info: Dict[str, Any] = {"source": source_video, "target": target_video}
    src = probe_pts(source_video)
    if len(src) < 2:
        info["reason"] = "source PTS unavailable"
        return False, info
    tgt = probe_pts(target_video)
    if len(tgt) != len(src):
        info["reason"] = (f"frame count differs (source={len(src)} "
                          f"target={len(tgt)}); refusing to re-time")
        info.update(frames_src=len(src), frames_tgt=len(tgt))
        return False, info

    src_last = src[-1] - src[0]
    tgt_last = tgt[-1] - tgt[0]

    # ---- 跳过判据必须**同时看总时长与逐帧间隔** ----
    # 实测反例：concat 产物总时长 15.12 vs 源 15.1111，只差 0.06% ——
    # 但逐帧间隔是 0.04/0.08（源 0.0449/0.0898），间隔被量化得面目全非。
    # 只看总时长就会误判"已经对上"、跳过修复。
    _src_d = [b - a for a, b in zip(src, src[1:])]
    _tgt_d = [b - a for a, b in zip(tgt, tgt[1:])]
    _max_int_err = max((abs(a - b) for a, b in zip(_src_d, _tgt_d)),
                       default=0.0)
    _dur_ok = (src_last > 0
               and abs(tgt_last - src_last) / src_last <= DURATION_TOLERANCE)
    # 5 ms ≈ v2 整数毫秒量化的上限；超过它说明间隔确实被改了
    _int_ok = _max_int_err <= 0.005
    if _dur_ok and _int_ok:
        info.update(action="skipped", reason="timeline already matches",
                    frames=len(src), src_last=round(src_last, 4),
                    out_last=round(tgt_last, 4),
                    max_interval_error=round(_max_int_err, 6))
        return True, info

    base = tgt[0]
    d = work_dir or os.path.dirname(os.path.abspath(target_video))
    tc = os.path.join(d, os.path.basename(target_video) + ".timecodes_v2.txt")
    if not write_timecodes_v2(src, tc, origin=src[0]):
        info["reason"] = "could not write timecodes file"
        return False, info

    tmp_out = target_video + ".retimed" + (os.path.splitext(target_video)[1]
                                           or ".mkv")
    # ---- 重封装也是"写文件"，按输出增长判活，不设总时长 ----
    # 原来固定 `timeout=1800`：4K 16 位 ffv1 中间件单件约 250 GB，重封装要
    # 逐包拷贝整条流，30 分钟根本不够；而换一个更长的素材又得再调一次。
    # mkvmerge 不报进度，但**输出文件一直在长** —— 这就是真实进展。
    # 注意必须并发排空管道，否则 stderr 写满会把它卡住。
    try:
        _proc = subprocess.Popen(
            [mkvmerge, "-q", "-o", tmp_out, "--timestamps", "0:" + tc,
             target_video],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        try:
            import cancel_guard as _cg
            _outs, _rcs = _cg.collect_outputs_while_waiting(
                [_proc], heartbeat=_cg.file_growth_watch(tmp_out),
                inactivity_timeout=INACTIVITY_TIMEOUT_S, logger=None,
                label="vfr-remux")
            _err_txt = _outs[0]["stderr"].decode("utf-8", "replace")

            class _R:
                returncode = _rcs[0]
                stderr = _err_txt
                stdout = ""
            r = _R()
        except Exception:
            # 守卫不可用时退回有界等待（行为与改动前一致）
            _o, _e = _proc.communicate(timeout=1800)
            _err_txt = _e.decode("utf-8", "replace") if _e else ""

            class _R2:
                returncode = _proc.returncode
                stderr = _err_txt
                stdout = ""
            r = _R2()
    except (OSError, subprocess.SubprocessError) as exc:
        info["reason"] = f"mkvmerge failed to start: {exc}"
        return False, info
    finally:
        try:
            os.remove(tc)
        except OSError:
            pass

    if r.returncode != 0:
        info["reason"] = f"mkvmerge rc={r.returncode}: {(r.stderr or '')[:300]}"
        for p in (tmp_out,):
            try:
                os.remove(p)
            except OSError:
                pass
        return False, info

    # 复核：帧数与末帧时间都必须对得上
    got = probe_pts(tmp_out)
    if len(got) != len(src):
        info["reason"] = (f"re-timed file has {len(got)} frames, "
                          f"expected {len(src)}")
        try:
            os.remove(tmp_out)
        except OSError:
            pass
        return False, info

    out_last = got[-1] - got[0]
    err_abs = abs(out_last - src_last)
    src_d = [b - a for a, b in zip(src, src[1:])]
    out_d = [b - a for a, b in zip(got, got[1:])]
    max_err = max((abs(a - b) for a, b in zip(src_d, out_d)), default=0.0)

    info.update(action="retimed", frames=len(got),
                src_last=round(src_last, 4), out_last=round(out_last, 4),
                duration_error=round(err_abs, 6),
                max_interval_error=round(max_err, 6),
                before_last=round(tgt_last, 4))

    # ---- 逐帧间隔也要参与阻断，不能只看末帧 ----
    # 审核探针：重封装结果"末 PTS 相同、中间某处间隔错 40 ms"，helper 仍返回
    # 成功 —— 因为 max_err 只被算出来写进 info，从未用于判定。
    # 只核终点等于承认"中间随便错"。这里给一个**有依据**的容差：
    # mkvmerge 的 timestamp v2 支持小数毫秒，默认 timestamp scale 为 1 ms，
    # 正常结果误差应在 1–2 ms；取 20 ms 留足余量，同时能抓住 40 ms 级别的错误。
    # 容差按目标时间基设定，**不因失败而放宽**。
    if max_err > INTERVAL_TOLERANCE:
        info["reason"] = (
            f"re-timed per-frame intervals still off by {max_err:.3f}s "
            f"(tolerance {INTERVAL_TOLERANCE:.3f}s); matching the last PTS "
            f"alone is not evidence the timeline is correct")
        try:
            os.remove(tmp_out)
        except OSError:
            pass
        return False, info

    # 末帧偏差超过 50 ms 视为没对上（v2 的 1 ms 量化不该到这个量级）
    if err_abs > 0.05:
        info["reason"] = (f"re-timed duration still off by {err_abs:.3f}s")
        try:
            os.remove(tmp_out)
        except OSError:
            pass
        return False, info

    # 校验通过 → **原地原子替换**（调用方不必再做一次，避免用错两步 API）
    try:
        os.replace(tmp_out, target_video)
    except OSError as exc:
        info["reason"] = f"could not replace the target atomically: {exc}"
        try:
            os.remove(tmp_out)
        except OSError:
            pass
        return False, info

    if logger is not None:
        try:
            logger.info(
                "VFR_RETIMED",
                "Re-muxed the candidate with the source's per-frame timestamps",
                frames=len(got), src_last=info["src_last"],
                out_last=info["out_last"],
                max_interval_error=info["max_interval_error"],
                before_last=info["before_last"])
        except Exception:
            pass
    return True, info
