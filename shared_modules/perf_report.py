"""性能重算（第七轮）：用**实测墙钟 + 实际帧数**重估，而不是拍脑袋。

**为什么需要工具**

第六轮给出过"关 Stab 5.6 小时 / 开 Stab 18 小时"的估算，但那份估算
内部就矛盾：仅按它自己写的 14420 帧、AI 约 6 秒/帧算，AI 单项就是

```
14420 × 6 / 3600 ≈ 24.03 小时
```

推不出 5.6h。任务书要求「撤回未经一致计算的 5.6h/18h……以实际输出
帧数与 stage 墙钟时间重估」。

**这个工具做什么**

从 job 日志里解析每个 stage 的 `STAGE_START` / `STAGE_DONE` /
`STAGE_SKIP`，从产物里 ffprobe 出实际帧数，算出：

* 每个 stage 的墙钟时间（秒）；
* 该 stage 处理的**实际帧数**（不是"输入帧数"，也不是"batch 数"）；
* 吞吐（帧/秒）；
* 按目标帧数外推的总耗时，并**明确标注**这是线性外推、不是置信区间。

**明确不做什么**

* 不测 CPU/GPU 占用、显存、磁盘 IO —— 那需要系统级采样，本工具不做，
  也不假装做了；
* 不把"输入帧数"和"输出帧数"混为一谈（去隔行会改变帧数）；
* 不外推成"置信区间"—— 单次样本只能给点估计。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "parse_stage_timings",
    "probe_frame_count",
    "build_perf_report",
    "render_perf_report",
    "extrapolate",
]

_STAGE_RE = re.compile(r"Stage\s+(\d+)")


def _parse_ts(ts: str) -> Optional[float]:
    """ISO8601 → epoch 秒（容忍时区后缀）。"""
    import datetime as _dt
    try:
        return _dt.datetime.fromisoformat(ts).timestamp()
    except Exception:
        return None


def parse_stage_timings(log_path: str,
                        run_start: Optional[str] = None) -> List[Dict[str, Any]]:
    """从 job 日志里解析**一次**运行的各 stage 起止时间。

    日志可能是多次运行累积的（同一个 job 重跑），所以按 `Stage 0`
    重新开始切分；`run_start` 给定时只取该时刻之后的那一次。

    返回：`[{"stage":1, "status":"done"|"skipped"|"running",
            "start":epoch, "end":epoch|None, "wall_s":float|None}]`
    """
    events: List[Tuple[float, str, int]] = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                ev = str(d.get("event") or "")
                if ev not in ("STAGE_START", "STAGE_DONE", "STAGE_SKIP"):
                    continue
                msg = str(d.get("message") or "")
                m = _STAGE_RE.search(msg)
                if not m:
                    continue
                ts = _parse_ts(str(d.get("timestamp") or ""))
                if ts is None:
                    continue
                events.append((ts, ev, int(m.group(1))))
    except OSError:
        return []

    events.sort()
    if run_start:
        cut = _parse_ts(run_start)
        if cut is not None:
            events = [e for e in events if e[0] >= cut]

    # 切分：遇到 stage 0 的 STAGE_START 就开始新的一次
    runs: List[List[Tuple[float, str, int]]] = []
    cur: List[Tuple[float, str, int]] = []
    for ev in events:
        if ev[1] == "STAGE_START" and ev[2] == 0 and cur:
            runs.append(cur)
            cur = []
        cur.append(ev)
    if cur:
        runs.append(cur)
    if not runs:
        return []
    run = runs[-1] if not run_start else runs[0]

    out: List[Dict[str, Any]] = []
    open_stage: Dict[int, float] = {}
    for ts, ev, st in run:
        if ev == "STAGE_START":
            open_stage[st] = ts
        elif ev in ("STAGE_DONE", "STAGE_SKIP"):
            start = open_stage.pop(st, None)
            out.append({
                "stage": st,
                "status": "done" if ev == "STAGE_DONE" else "skipped",
                "start": start,
                "end": ts,
                "wall_s": (ts - start) if start else None,
            })
    for st, start in open_stage.items():
        out.append({"stage": st, "status": "running", "start": start,
                    "end": None, "wall_s": None})
    out.sort(key=lambda r: r["stage"])
    return out


def probe_frame_count(path: str) -> Optional[int]:
    """数一个媒体文件的实际帧数（不是时长 × 帧率）。"""
    if not path or not os.path.exists(path):
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=120)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    try:
        return int((r.stdout or "").strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def build_perf_report(timings: List[Dict[str, Any]],
                      artifacts: Dict[int, str],
                      logger=None) -> Dict[str, Any]:
    """把时间与帧数合起来，算每个 stage 的吞吐。"""
    rows: List[Dict[str, Any]] = []
    for t in timings:
        st = t["stage"]
        art = artifacts.get(st)
        frames = probe_frame_count(art) if art else None
        wall = t.get("wall_s")
        fps = (frames / wall) if (frames and wall and wall > 0) else None
        rows.append({
            "stage": st,
            "status": t.get("status"),
            "wall_s": round(wall, 2) if wall is not None else None,
            "frames": frames,
            "artifact": art,
            "fps": round(fps, 3) if fps else None,
            "note": ("no artifact" if not art
                     else ("no frame count" if frames is None else "")),
        })
    return {"rows": rows,
            "total_wall_s": round(sum(r["wall_s"] or 0 for r in rows), 2)}


def extrapolate(report: Dict[str, Any], target_frames: int) -> List[Dict[str, Any]]:
    """按每个 stage 的实测吞吐，线性外推到目标帧数。

    **这是线性外推，不是置信区间。** 它假设吞吐与帧数无关 —— 对长片
    通常偏乐观（磁盘缓存、显存碎片、热降频都会让长跑变慢）。报告里
    必须把这句话写出来。
    """
    out: List[Dict[str, Any]] = []
    for r in report.get("rows") or []:
        if not r.get("fps"):
            out.append({"stage": r["stage"], "hours": None,
                        "basis": r.get("note") or "no throughput"})
            continue
        hours = target_frames / float(r["fps"]) / 3600.0
        out.append({"stage": r["stage"], "hours": round(hours, 3),
                    "basis": f"{r['fps']} fps from {r['frames']} frames "
                             f"in {r['wall_s']}s"})
    return out


def render_perf_report(report: Dict[str, Any],
                       target_frames: Optional[int] = None) -> str:
    lines: List[str] = []
    lines.append("stage | status  | wall(s) | frames | fps   | artifact")
    lines.append("------+---------+---------+--------+-------+---------")
    for r in report.get("rows") or []:
        lines.append(
            f"{r['stage']:5d} | {str(r['status']):7s} | "
            f"{str(r['wall_s']):7s} | {str(r['frames']):6s} | "
            f"{str(r['fps']):5s} | {os.path.basename(r['artifact'] or '')[:36]}")
    lines.append(f"\ntotal wall: {report.get('total_wall_s')}s")

    if target_frames:
        lines.append(f"\n线性外推到 {target_frames} 帧"
                     "（**不是置信区间**，假设吞吐与帧数无关）：")
        for e in extrapolate(report, target_frames):
            if e["hours"] is None:
                lines.append(f"  stage {e['stage']}: — ({e['basis']})")
            else:
                lines.append(f"  stage {e['stage']}: {e['hours']:.2f} h  "
                             f"[{e['basis']}]")
        known = [e["hours"] for e in extrapolate(report, target_frames)
                 if e["hours"] is not None]
        if known:
            lines.append(f"  合计（仅可比 stage）: {sum(known):.2f} h")
            lines.append("  注意：这是下界估计 —— 冷启动、显存碎片、"
                         "热降频、磁盘 IO 都会让实际更慢。")
    return "\n".join(lines)
