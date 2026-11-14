# -*- coding: utf-8 -*-
"""P6 / G11：把一次跑批的"写入 / 耗时 / 帧吞吐 / 中间媒体"摊成一张表。

## 为什么单独写脚本而不是只在日志里打

G11 要求的是**可比较**的数字：同一份素材在 `--chain none` 与
`--chain stream_full` 下，各写了多少、花了多久。日志里那行
`WRITE_LEDGER` 是"这一次"的快照，不方便并排比。这个脚本读两边的
日志 + 目录，输出一张 markdown 表，直接贴进证据文件。

## 口径（必须写清，否则数字会被误读）

* `present`   —— 跑完之后**还在**的文件字节数（分六类，见 write_ledger）
* `purged`    —— 中间产物在收尾时被清掉的字节数（用"运行中见过的峰值"估算）
* `wall_s`    —— 从日志的 `JOB_START` 到 `JOB_DONE`（或最后一条）的墙钟
* `fps`       —— 成品帧数 / wall_s（**端到端**吞吐，不是单 stage）

**不报"SSD 寿命"**：应用层写入 ≠ 物理写入（pagefile/日志/NAND 磨损），
本脚本只统计应用层。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


def _rows(log_path: str) -> List[Dict[str, Any]]:
    out = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def _parse_ts(ts: str) -> Optional[float]:
    try:
        from datetime import datetime
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return None


def summarise(run_dir: str) -> Dict[str, Any]:
    """扫一个跑批目录，给出可比较的数字。"""
    run_dir = os.path.abspath(run_dir)
    logs_dir = os.path.join(run_dir, "logs")
    log_path = ""
    if os.path.isdir(logs_dir):
        cands = sorted(f for f in os.listdir(logs_dir) if f.endswith(".log"))
        if cands:
            log_path = os.path.join(logs_dir, cands[-1])
    rows = _rows(log_path)

    t0 = t1 = None
    events: List[str] = []
    receipt: Dict[str, Any] = {}
    for r in rows:
        ev = str(r.get("event") or "")
        events.append(ev)
        ts = _parse_ts(str(r.get("timestamp") or ""))
        if ev == "JOB_START" and t0 is None:
            t0 = ts
        if ev in ("JOB_DONE", "COMPLETION_VERIFIED", "JOB_UNVERIFIED") and ts:
            t1 = ts
    wall = round((t1 - t0), 1) if (t0 and t1) else None

    # 成品帧数（以 ffprobe 为准，不信日志自报）
    out_dir = os.path.join(run_dir, "out")
    frames = None
    out_bytes = 0
    out_name = ""
    if os.path.isdir(out_dir):
        for f in sorted(os.listdir(out_dir)):
            if f.endswith((".mkv", ".mp4")):
                p = os.path.join(out_dir, f)
                out_name = f
                out_bytes = os.path.getsize(p)
                try:
                    import subprocess
                    r = subprocess.run(
                        ["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-count_frames", "-show_entries",
                         "stream=nb_read_frames", "-of",
                         "default=nw=1:nk=1", p],
                        capture_output=True, text=True, timeout=600)
                    frames = int((r.stdout or "0").strip() or 0) or None
                except Exception:
                    frames = None
                break

    # 现存文件按 write_ledger 的六类统计
    scan = {}
    try:
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "shared_modules"))
    except Exception:
        pass
    try:
        import write_ledger as WL
        scan = WL.scan([os.path.join(run_dir, "temp"), out_dir],
                       output_path=os.path.join(out_dir, out_name)
                       if out_name else "")
    except Exception as exc:                    # pragma: no cover
        scan = {"error": str(exc)}

    def _has(ev: str) -> bool:
        return ev in events

    return {
        "run_dir": run_dir,
        "log": log_path,
        "wall_s": wall,
        "output": out_name,
        "output_bytes": out_bytes,
        "output_frames": frames,
        "end_to_end_fps": (round(frames / wall, 3)
                           if (frames and wall) else None),
        "chain": ("stream_full" if _has("STREAM_S1_SCRIPT_READY")
                  else ("stage67" if _has("STAGE6_FUSED_EMIT_ONLY")
                        else "classic")),
        "fused_s3s4": _has("STAGE3_FUSE34_ENABLED"),
        "fused_s6s7": _has("STAGE6_FUSED_EMIT_ONLY"),
        "two_pass_ok": _has("STAGE3_TWO_PASS_OK"),
        "audio_from_source": _has("STAGE7_AUDIO_FROM_SOURCE"),
        "completion": ("verified" if _has("COMPLETION_VERIFIED")
                       else ("unverified" if _has("JOB_UNVERIFIED")
                             else "unknown")),
        "write_ledger": scan,
    }


def render(rows: List[Dict[str, Any]]) -> str:
    head = ("| 跑批 | 链 | 墙钟(s) | 帧 | 端到端 fps | 成品(MB) | "
            "中间媒体(MB) | 报告(KB) | 完成态 |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in rows:
        wl = ((r.get("write_ledger") or {}).get("by_category") or {})
        med = (wl.get("intermediate_media") or {}).get("bytes") or 0
        rep = (wl.get("index_report") or {}).get("bytes") or 0
        lines.append(
            "| {name} | {chain}{f34}{f67} | {wall} | {frames} | {fps} | "
            "{mb:.1f} | {med:.1f} | {rep:.0f} | {comp} |".format(
                name=os.path.basename(r.get("run_dir") or "?"),
                chain=r.get("chain"),
                f34="+S3S4" if r.get("fused_s3s4") else "",
                f67="+S6S7" if r.get("fused_s6s7") else "",
                wall=r.get("wall_s"),
                frames=r.get("output_frames"),
                fps=r.get("end_to_end_fps"),
                mb=(r.get("output_bytes") or 0) / 1048576.0,
                med=med / 1048576.0,
                rep=rep / 1024.0,
                comp=r.get("completion")))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="跑批工作目录（含 logs/ 与 out/）")
    ap.add_argument("--json", action="store_true", help="输出 JSON 而不是表")
    a = ap.parse_args(argv)
    rows = [summarise(d) for d in a.run_dirs]
    if a.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(render(rows))
        print("\n口径：present=跑完还在的文件；墙钟=JOB_START→完成事件；"
              "fps=成品帧数/墙钟（端到端，不是单 stage）。\n"
              "不报 SSD 寿命：应用层写入 ≠ 物理写入。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
