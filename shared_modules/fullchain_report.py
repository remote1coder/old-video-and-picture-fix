# -*- coding: utf-8 -*-
"""全链核对结果**筛查报告**（体检工具）。

用途
----
`full_chain_verify_mode: absolute` 在老素材上失败率很高（实测 204 个手机
素材约 31%），而这些漂移大多是**源片自带的**，不是流程引入的。所以
absolute 档的主要用法不是"拒绝出片"，而是：

    配合 `allow_verify_fail_publish: true` 跑一遍 → 用本报告筛出
    "哪些素材有漂移、漂移多大" → 再决定要不要处理。

只看逐个日志太费劲，这里把 `*.fullchain.json`（stage7 发布时产出）
汇总成一张按漂移排序的表。

用法
----
    python shared_modules/fullchain_report.py <目录> [--csv out.csv]

输出列：素材 / 帧数(源→成品) / 时长(源→成品) / 成品内部 A/V 漂移 /
        全链核对结论。
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "video_restoration_pipeline", "stages"))

try:
    from stage7_encode import _measure_av_drift  # type: ignore
except Exception:                                # pragma: no cover
    _measure_av_drift = None


def _probe(path: str, args: list) -> str:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0"]
                       + args + ["-of", "csv=p=0", path],
                       capture_output=True, text=True, errors="replace",
                       timeout=180)
    return (r.stdout or "").strip()


def _duration(path: str):
    r = subprocess.run(["ffprobe", "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "csv=p=0", path],
                       capture_output=True, text=True, errors="replace",
                       timeout=60)
    try:
        return float((r.stdout or "").strip())
    except (TypeError, ValueError):
        return None


def collect(json_files: list, outputs_dir: str) -> list:
    """读取每份 fullchain.json，并量出对应成品的内部 A/V 漂移。"""
    rows = []
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            rows.append({"name": os.path.basename(jf), "error": str(e)[:60]})
            continue

        checks = (d or {}).get("checks") or {}
        src = (d or {}).get("source") or {}
        out = (d or {}).get("output") or {}

        # 成品本体：fullchain.json 名字是 <成品>.fullchain.json
        stem = os.path.basename(jf)
        stem = stem[:-len(".fullchain.json")] if stem.endswith(
            ".fullchain.json") else stem
        cand = [os.path.join(outputs_dir, n) for n in os.listdir(outputs_dir)
                if n.startswith(stem) and not n.endswith(".json")]
        artifact = cand[0] if cand else None

        drift_ms = None
        if artifact and _measure_av_drift is not None:
            try:
                v = _measure_av_drift(artifact)
                if v is not None:
                    drift_ms = abs(v) * 1000.0
            except Exception:
                drift_ms = None

        rows.append({
            "name": stem,
            "src_frames": src.get("frames"),
            "out_frames": out.get("frames"),
            "src_duration": (checks.get("duration") or {}).get("src"),
            "out_duration": (checks.get("duration") or {}).get("out"),
            "artifact_duration": _duration(artifact) if artifact else None,
            "av_drift_ms": drift_ms,
            "frames_ok": (checks.get("frames") or {}).get("ok"),
            "duration_ok": (checks.get("duration") or {}).get("ok"),
            "ok": (d or {}).get("ok"),
        })
    return rows


def main(argv=None) -> int:
    # 注意：不能用 `argv or sys.argv[1:]` —— 空列表是 falsy，
    # 会把显式的「无参数」误当成未传入，转而去读宿主的命令行。
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    csv_out = None
    if "--csv" in argv:
        i = argv.index("--csv")
        csv_out = argv[i + 1] if i + 1 < len(argv) else "fullchain_report.csv"
        del argv[i:i + 2]
    if not argv:
        sys.stderr.write(__doc__ + "\n")
        return 81

    d = argv[0]
    if not os.path.isdir(d):
        sys.stderr.write(f"不是目录: {d}\n")
        return 82

    files = sorted(os.path.join(d, n) for n in os.listdir(d)
                   if n.endswith(".fullchain.json"))
    if not files:
        sys.stderr.write(f"{d} 下没有 *.fullchain.json\n")
        return 0

    rows = collect(files, d)
    rows.sort(key=lambda r: (r.get("av_drift_ms") is None,
                             -(r.get("av_drift_ms") or 0)))

    print("%-30s %14s %18s %12s %8s" % (
        "素材", "帧数 源→成品", "时长 源→成品", "AV漂移ms", "结论"))
    print("-" * 88)
    for r in rows:
        if "error" in r:
            print("%-30s 读取失败: %s" % (r["name"][:28], r["error"]))
            continue
        fr = "%s→%s" % (r.get("src_frames"), r.get("out_frames"))
        du = "%s→%s" % (
            ("%.2f" % r["src_duration"]) if r.get("src_duration") else "?",
            ("%.2f" % r["out_duration"]) if r.get("out_duration") else "?")
        dv = ("%.0f" % r["av_drift_ms"]) if r.get(
            "av_drift_ms") is not None else "-"
        verdict = "OK" if r.get("ok") else "FAIL"
        print("%-30s %14s %18s %12s %8s" % (r["name"][:28], fr, du, dv, verdict))

    measured = [r for r in rows if r.get("av_drift_ms") is not None]
    if measured:
        over = [r for r in measured if r["av_drift_ms"] > 185]
        print("-" * 88)
        print("可测 %d 份；漂移超过 ITU-R BT.1359-1 可接受线(185ms) 的: %d 份"
              % (len(measured), len(over)))

    if csv_out:
        with open(csv_out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=[
                "name", "src_frames", "out_frames", "src_duration",
                "out_duration", "av_drift_ms", "frames_ok", "duration_ok", "ok"],
                extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("\nCSV 已写出: %s" % csv_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
