# -*- coding: utf-8 -*-
"""R22 §5.4 / N22 / A10 · 重复性与资源记录（同配置重复 + 故障重跑）。

## 审核要求

> 重复次数、每次差异、耗时、峰值 RAM/VRAM、临时磁盘都记录；范围写清。

第十八轮的"1 线程确定、4.78 倍慢"**只作历史样本**，不能当本轮结论 ——
那是 stage1 的 mthreads 竞态，与"整条链重复性"不是同一件事。这里量的是
**同一配置连跑 N 次，成品字节是否相同**，以及每次的耗时/峰值显存/临时磁盘。

## 量什么

| 项 | 怎么量 |
|---|---|
| 成品身份 | `sha256(成品)` + 逐帧 PSNR（与第一次比）|
| 耗时 | `time.monotonic()` 包住 batch 调用 |
| 峰值 VRAM | 后台线程 `nvidia-smi --query-gpu=memory.used` 每 0.5 s 采样取 max |
| 临时磁盘 | 跑完统计 `temp/` 目录总字节 |
| 故障重跑 | 跑到一半 `taskkill` 强杀，再原样重跑，要求仍然成功 |

## 用法

    python tests/probes/r22_repeatability.py [--runs 3] [--mode 444]

工作根：``E:\\codex\\_media_r22_20261111T2330\\s5_repeat\\<时间戳>``
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import r22_444_classic_path as s3probe  # noqa: E402

ROOT = s3probe.ROOT
PY = s3probe.PY
MAIN = s3probe.MAIN
S5_ROOT = r"E:\codex\_media_r22_20261111T2330\s5_repeat"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_bytes(path):
    total = 0
    for base, _dirs, files in os.walk(path):
        for n in files:
            try:
                total += os.path.getsize(os.path.join(base, n))
            except OSError:
                pass
    return total


class _VramSampler:
    """后台采 GPU 显存峰值（没有 nvidia-smi 就如实记 None）。"""

    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak = None
        self._stop = threading.Event()
        self._th = None
        self.available = shutil.which("nvidia-smi") is not None

    def _loop(self):
        while not self._stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10, shell=False)
                vals = [int(x) for x in (r.stdout or "").split()
                        if x.strip().isdigit()]
                if vals:
                    m = max(vals)
                    self.peak = m if self.peak is None else max(self.peak, m)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        if not self.available:
            return
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=5)


def _psnr_rows(a, b):
    r = s3probe._run(["ffmpeg", "-hide_banner", "-v", "info", "-i", a,
                      "-i", b, "-lavfi", "[0:v][1:v]psnr=stats_file=-",
                      "-f", "null", "-"], timeout=600)
    return [ln for ln in (r.stdout or "").splitlines() if ln.startswith("n:")]


def _one_run(root, idx, chroma_mode, encoder, pix_fmt, frames, size):
    work = os.path.join(root, f"run_{idx:02d}")
    for sub in ("input", "output", "temp", "logs"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    src = os.path.join(work, "input", "probe_src.mkv")
    # 小样本：帧数与尺寸都可调，重复性的重点在"是否相同"，不在画质
    s3probe.W, s3probe.H = size
    s3probe.FRAMES = frames
    s3probe._make_source(src)
    cfg = s3probe._config_for(work, chroma_mode, encoder, pix_fmt)
    manifest = os.path.join(work, "manifest.yaml")

    ev = {"run": idx, "work": work}
    scan = s3probe._run([PY, "-B", MAIN, "--config", cfg, "--input",
                         os.path.join(work, "input"), "--output",
                         os.path.join(work, "output"), "--no-wsl-redirect",
                         "--scan", "--output-manifest", manifest], cwd=work)
    ev["scan_rc"] = scan.returncode
    if scan.returncode != 0:
        ev["error"] = (scan.stderr or "")[-600:]
        return ev
    s3probe._unskip(manifest)

    sampler = _VramSampler()
    sampler.start()
    t0 = time.monotonic()
    run = s3probe._run([PY, "-B", MAIN, "--config", cfg, "--input",
                        os.path.join(work, "input"), "--output",
                        os.path.join(work, "output"), "--no-wsl-redirect",
                        "--manifest", manifest, "--workers", "1"],
                       cwd=work, timeout=3600)
    wall = time.monotonic() - t0
    sampler.stop()

    ev["rc"] = run.returncode
    ev["wall_s"] = round(wall, 2)
    ev["peak_vram_mb"] = sampler.peak
    ev["vram_available"] = sampler.available
    ev["temp_bytes"] = _dir_bytes(os.path.join(work, "temp"))
    out = os.path.join(work, "output")
    finals = sorted(os.path.join(out, n) for n in os.listdir(out)
                    if n.endswith((".mkv", ".mp4", ".avif", ".mov"))) \
        if os.path.isdir(out) else []
    ev["final"] = [{"file": os.path.basename(p), "sha256": _sha256(p),
                    "bytes": os.path.getsize(p),
                    "pix_fmt": (s3probe._probe_video(p) or {}).get("pix_fmt")}
                   for p in finals]
    ev["stderr_tail"] = (run.stderr or "")[-600:] if run.returncode else ""
    return ev


def _fault_rerun(root, chroma_mode, encoder, pix_fmt, frames, size):
    """故障重跑：跑到一半强杀，再原样重跑 —— 要求最终仍然成功且可证完成。

    强杀的是**整棵进程树**：main 是父进程，stage 是子进程，只杀父会留下
    孤儿继续写文件（项目里 RV-12 就是这个形态）。用 `taskkill /T /F`。
    """
    work = os.path.join(root, "run_fault")
    for sub in ("input", "output", "temp", "logs"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    src = os.path.join(work, "input", "probe_src.mkv")
    s3probe.W, s3probe.H = size
    s3probe.FRAMES = frames
    s3probe._make_source(src)
    cfg = s3probe._config_for(work, chroma_mode, encoder, pix_fmt)
    manifest = os.path.join(work, "manifest.yaml")
    scan = s3probe._run([PY, "-B", MAIN, "--config", cfg, "--input",
                         os.path.join(work, "input"), "--output",
                         os.path.join(work, "output"), "--no-wsl-redirect",
                         "--scan", "--output-manifest", manifest], cwd=work)
    ev = {"work": work, "scan_rc": scan.returncode}
    if scan.returncode != 0:
        ev["error"] = (scan.stderr or "")[-600:]
        return ev
    s3probe._unskip(manifest)

    cmd = [PY, "-B", MAIN, "--config", cfg, "--input",
           os.path.join(work, "input"), "--output",
           os.path.join(work, "output"), "--no-wsl-redirect",
           "--manifest", manifest, "--workers", "1"]
    e = os.environ.copy()
    e.update(s3probe.ENV_EXTRA)
    p = subprocess.Popen(cmd, cwd=work, env=e,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 等到出现第一个 step 产物再杀，确保"死在中间"而不是死在启动阶段
    killed_at = None
    deadline = time.monotonic() + 900
    temp = os.path.join(work, "temp")
    while time.monotonic() < deadline:
        if p.poll() is not None:
            break
        names = os.listdir(temp) if os.path.isdir(temp) else []
        if any("step1" in n or "step3" in n for n in names):
            killed_at = sorted(names)[:3]
            break
        time.sleep(1.0)
    if p.poll() is None:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)],
                       capture_output=True, shell=False)
        try:
            p.wait(timeout=60)
        except Exception:
            pass
    ev["killed_at"] = killed_at
    ev["killed_rc"] = p.returncode
    # 等孤儿收尾（守卫应该已经杀掉子进程，这里只等文件句柄释放）
    time.sleep(2.0)

    rerun = s3probe._run(cmd, cwd=work, timeout=3600)
    ev["rerun_rc"] = rerun.returncode
    ev["rerun_stderr_tail"] = (rerun.stderr or "")[-800:]
    out = os.path.join(work, "output")
    finals = sorted(os.path.join(out, n) for n in os.listdir(out)
                    if n.endswith((".mkv", ".mp4", ".avif", ".mov"))) \
        if os.path.isdir(out) else []
    ev["final"] = [{"file": os.path.basename(p2), "sha256": _sha256(p2),
                    "bytes": os.path.getsize(p2),
                    "pix_fmt": (s3probe._probe_video(p2) or {}).get("pix_fmt")}
                   for p2 in finals]
    # 完成证明的旁证（stage7 私有 result + marker 都在成品旁边）
    ev["sidecars"] = sorted(n for n in os.listdir(out)
                            if n.endswith(".json")) \
        if os.path.isdir(out) else []
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--mode", default="444", choices=["420", "444"])
    ap.add_argument("--frames", type=int, default=8)
    # 320x240 是下限：stage1 的黑边裁剪校验要求 >= 240px，
    # 160x120 会以 STAGE1_CROP_TOO_SMALL 直接 rc=81（实测踩过）。
    ap.add_argument("--size", default="320x240")
    ap.add_argument("--root", default=S5_ROOT)
    ap.add_argument("--skip-fault", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(PY):
        raise SystemExit(f"缺少视频链解释器: {PY}")

    size = tuple(int(x) for x in args.size.lower().split("x"))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.root, f"{args.mode}_{stamp}")
    os.makedirs(root, exist_ok=True)
    enc, fmt = ("libx265", "yuv444p10le") if args.mode == "444" \
        else ("libsvtav1", "yuv420p10le")
    print(f"[s5] root={root} mode={args.mode} encoder={enc} runs={args.runs} "
          f"sample={size[0]}x{size[1]}x{args.frames}f "
          f"vram={'yes' if shutil.which('nvidia-smi') else 'no'}")

    runs = []
    for i in range(args.runs):
        print(f"[s5] --- run {i} ---")
        ev = _one_run(root, i, args.mode, enc, fmt, args.frames, size)
        runs.append(ev)
        print(f"      rc={ev.get('rc')} wall={ev.get('wall_s')}s "
              f"vram={ev.get('peak_vram_mb')}MB "
              f"temp={ev.get('temp_bytes')}B "
              f"sha={[f['sha256'][:12] for f in ev.get('final') or []]}")

    fault = None
    if not args.skip_fault:
        print("[s5] --- fault rerun ---")
        fault = _fault_rerun(root, args.mode, enc, fmt, args.frames, size)
        print(f"      killed_rc={fault.get('killed_rc')} "
              f"killed_at={(fault.get('killed_at') or [])[:2]} "
              f"rerun_rc={fault.get('rerun_rc')} "
              f"sidecars={len(fault.get('sidecars') or [])}")

    # ---- 判定 ----
    ok = True
    shas = [tuple(f["sha256"] for f in ev.get("final") or [])
            for ev in runs]
    for ev in runs:
        if ev.get("rc") != 0:
            print(f"[FAIL] run {ev['run']} rc={ev.get('rc')}")
            ok = False
    base = shas[0] if shas else ()
    for i, s in enumerate(shas[1:], start=1):
        if s != base:
            print(f"[FAIL] run {i} 成品字节与 run 0 不同: {s} vs {base}")
            ok = False
    # 逐帧 PSNR（字节不同时要知道差在哪）
    psnrs = []
    if len(runs) > 1 and runs[0].get("final") and runs[1].get("final"):
        pa = os.path.join(runs[0]["work"], "output", runs[0]["final"][0]["file"])
        pb = os.path.join(runs[1]["work"], "output", runs[1]["final"][0]["file"])
        if os.path.exists(pa) and os.path.exists(pb):
            rows = _psnr_rows(pa, pb)
            identical = bool(rows) and all(
                "psnr_y:inf" in ln and "psnr_u:inf" in ln
                and "psnr_v:inf" in ln for ln in rows)
            psnrs.append({"run": "0-vs-1", "frames": len(rows),
                          "identical": identical,
                          "first_row": rows[0] if rows else None})
            print(f"[s5] run0 vs run1: {len(rows)} 帧, 逐像素相同={identical}")
    if fault is not None:
        if fault.get("rerun_rc") != 0:
            print(f"[FAIL] 故障重跑未成功: rc={fault.get('rerun_rc')}")
            ok = False
        if not fault.get("final"):
            print("[FAIL] 故障重跑没有产出成品")
            ok = False

    report = {"root": root, "mode": args.mode, "encoder": enc,
              "pix_fmt": fmt, "runs_requested": args.runs,
              "sample": {"w": size[0], "h": size[1], "frames": args.frames},
              "runs": runs, "psnr": psnrs, "fault": fault, "ok": ok,
              # 范围写清（审核要求）
              "scope": ("单进程 --workers 1；同机同配置；样本为合成静态帧；"
                        "VRAM 峰值由 nvidia-smi 0.5s 采样，非驱动级精确值；"
                        "本轮**不**做线程数消融（第十八轮 4.78× 只作历史样本）")}
    out = os.path.join(root, "repeatability.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[s5] report -> {out}")
    print(f"[s5] VERDICT = {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
