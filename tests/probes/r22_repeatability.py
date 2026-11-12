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



def _force_deterministic(cfg_path, on):
    """把 `vs.deterministic` 写进探针配置。

    为什么要两组：**默认（多线程）下 stage1 的 mvtools 运动估计是竞态的**
    （第十八轮实测：同 vpy 连跑两次 60 帧里 30 帧不同，本底 PSNR
    Y≈48.7/U≈51.5/V≈39.6）。所以"字节不同"在多线程下是**已知性质**，
    不是本轮缺陷；只有 `vs.deterministic: true`（core.num_threads=1）
    才应当字节完全相同。
    """
    if not on:
        return
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    doc.setdefault("vs", {})["deterministic"] = True
    # 光有单线程还不够：**stage6 的颗粒是随机数**（`haf.GrainFactory3`
    # 默认 seed=-1）。实测：单线程下 step1–step4 逐像素相同，step6 的
    # Y 平面仍不同（psnr_y 51.9，U/V 相同），成品跟着不同（53 dB）。
    # 所以"可重复"这一组必须把 grain_seed 也固定住。
    doc.setdefault("global", {})["grain_seed"] = 20261112
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)



def _det_suffix(on):
    return "_deterministic" if on else "_default"

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
    """返回 ``(rc, rows)`` —— **rc 不能丢**（R23-B12）。

    审核反例：工具失败（rc≠0）却留下 1 行 `inf`，旧实现只看那行，
    于是"比较失败"被当成"完全相同"。
    """
    r = s3probe._run(["ffmpeg", "-hide_banner", "-v", "info", "-i", a,
                      "-i", b, "-lavfi", "[0:v][1:v]psnr=stats_file=-",
                      "-f", "null", "-"], timeout=600)
    rows = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("n:")]
    return r.returncode, rows


def _comparison_complete(rc, rows, expected_frames):
    """一次逐帧比较是否"跑完且覆盖全部帧"（R23-B12）。

    三条缺一不可：工具成功、有结果行、行数等于预期帧数。
    审核反例：工具 rc≠0 但留下一行 `inf` → 旧实现当成"完全相同"；
    8 帧的样本只比了 1 帧也一样。
    """
    try:
        n = int(expected_frames)
    except (TypeError, ValueError):
        return False
    return (int(rc) == 0 and isinstance(rows, list)
            and len(rows) == n and n > 0)


def _one_run(root, idx, chroma_mode, encoder, pix_fmt, frames, size,
             deterministic=False):
    work = os.path.join(root, f"run_{idx:02d}")
    for sub in ("input", "output", "temp", "logs"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    src = os.path.join(work, "input", "probe_src.mkv")
    # 小样本：帧数与尺寸都可调，重复性的重点在"是否相同"，不在画质
    s3probe.W, s3probe.H = size
    s3probe.FRAMES = frames
    s3probe._make_source(src)
    cfg = s3probe._config_for(work, chroma_mode, encoder, pix_fmt)
    _force_deterministic(cfg, deterministic)
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
    # **必须保留中间产物**：两遍不一样时要知道"从哪个边界开始不一样"。
    # 不留中间件的跑法只能看到成品不同，定位不了原因（本轮踩过）。
    run = s3probe._run([PY, "-B", MAIN, "--config", cfg, "--input",
                        os.path.join(work, "input"), "--output",
                        os.path.join(work, "output"), "--no-wsl-redirect",
                        "--manifest", manifest, "--keep-intermediates",
                        "--workers", "1"],
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


def _fault_rerun(root, chroma_mode, encoder, pix_fmt, frames, size,
                 deterministic=False):
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
    _force_deterministic(cfg, deterministic)
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
    ap.add_argument("--deterministic", action="store_true",
                    help="置 vs.deterministic=true（core.num_threads=1）；"
                         "只有这一组才要求字节完全相同")
    args = ap.parse_args()
    if not os.path.exists(PY):
        raise SystemExit(f"缺少视频链解释器: {PY}")

    size = tuple(int(x) for x in args.size.lower().split("x"))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.root,
                        f"{args.mode}"
                        f"{_det_suffix(args.deterministic)}_"
                        f"{stamp}")
    os.makedirs(root, exist_ok=True)
    enc, fmt = ("libx265", "yuv444p10le") if args.mode == "444" \
        else ("libsvtav1", "yuv420p10le")
    print(f"[s5] root={root} mode={args.mode} encoder={enc} runs={args.runs} "
          f"sample={size[0]}x{size[1]}x{args.frames}f "
          f"vram={'yes' if shutil.which('nvidia-smi') else 'no'}")

    runs = []
    for i in range(args.runs):
        print(f"[s5] --- run {i} ---")
        ev = _one_run(root, i, args.mode, enc, fmt, args.frames, size,
                      deterministic=args.deterministic)
        runs.append(ev)
        print(f"      rc={ev.get('rc')} wall={ev.get('wall_s')}s "
              f"vram={ev.get('peak_vram_mb')}MB "
              f"temp={ev.get('temp_bytes')}B "
              f"sha={[f['sha256'][:12] for f in ev.get('final') or []]}")

    fault = None
    if not args.skip_fault:
        print("[s5] --- fault rerun ---")
        fault = _fault_rerun(root, args.mode, enc, fmt, args.frames, size,
                             deterministic=args.deterministic)
        print(f"      killed_rc={fault.get('killed_rc')} "
              f"killed_at={(fault.get('killed_at') or [])[:2]} "
              f"rerun_rc={fault.get('rerun_rc')} "
              f"sidecars={len(fault.get('sidecars') or [])}")

    # ---- 判定 ----
    psnrs = []
    if len(runs) > 1 and runs[0].get("final"):
        pa = os.path.join(runs[0]["work"], "output", runs[0]["final"][0]["file"])
        for k in range(1, len(runs)):
            if not runs[k].get("final"):
                continue
            pb = os.path.join(runs[k]["work"], "output",
                              runs[k]["final"][0]["file"])
            if not (os.path.exists(pa) and os.path.exists(pb)):
                continue
            rc, rows = _psnr_rows(pa, pb)
            identical = bool(rows) and all(
                "psnr_y:inf" in ln and "psnr_u:inf" in ln
                and "psnr_v:inf" in ln for ln in rows)
            # R23-B12：比较必须**跑完且覆盖全部帧**。1 行 inf 不是 8 帧。
            complete = _comparison_complete(rc, rows, args.frames)
            psnrs.append({"run": f"0-vs-{k}", "frames": len(rows),
                          "rc": rc, "expected_frames": int(args.frames),
                          "complete": complete,
                          "identical": identical,
                          "rows": rows,
                          "first_row": rows[0] if rows else None})
            print(f"[s5] run0 vs run{k}: rc={rc} {len(rows)}/"
                  f"{args.frames} 帧, 逐像素相同={identical}")
    ok = True
    shas = [tuple(f["sha256"] for f in ev.get("final") or [])
            for ev in runs]
    for ev in runs:
        if ev.get("rc") != 0:
            print(f"[FAIL] run {ev['run']} rc={ev.get('rc')}")
            ok = False
    base = shas[0] if shas else ()
    byte_identical = bool(shas) and all(x == base for x in shas)
    # ---- 字节相同是不是"必须"取决于这一组开没开确定性 ----
    # 默认（多线程）下 stage1 的 mvtools 运动估计有竞态，**字节不同是已知
    # 性质**（第十八轮实测：60 帧里 30 帧不同，本底 PSNR Y≈48.7/U≈51.5/V≈39.6）。
    # 把它当失败会得出"项目一直坏着"的错误结论；反过来，在
    # `vs.deterministic: true` 下若字节不同，那才是真缺陷。
    # 字节相同不等于像素相同（mkv 的 SegmentUID/Date 每次都不同），
    # 所以"可重复"的判据是**逐像素**。实测：同一输入用 libx265 编两遍，
    # 8/8 帧 PSNR inf，但 md5 不同 —— 差异全部来自容器元数据。
    pixel_identical = None
    if psnrs:
        pixel_identical = all(bool(p.get("identical")) for p in psnrs)

    # ---- R23-B12：**没有可比对的东西**不能算通过 ----
    _incomplete = [p for p in psnrs if not p.get("complete")]
    if _incomplete:
        for p in _incomplete:
            print(f"[FAIL] {p['run']}: 比较不完整 rc={p.get('rc')} "
                  f"行数={p['frames']}/{p.get('expected_frames')}")
        ok = False
    if args.runs > 1:
        if not runs or any(not (ev.get("final") or []) for ev in runs):
            print("[FAIL] 有运行没有成品；重复性无从比较")
            ok = False
        elif not psnrs:
            print("[FAIL] 没有任何可比对的成品对")
            ok = False

    if args.deterministic:
        if pixel_identical is False:
            print("[FAIL] deterministic 组逐帧不相同 —— 上游仍有未固定的随机源")
            for p in psnrs:
                print(f"   {p['run']}: frames={p['frames']} "
                      f"identical={p['identical']} first={p.get('first_row')}")
            ok = False
        elif pixel_identical:
            print("[s5] deterministic 组：逐帧 PSNR 全 inf（像素完全一致）；"
                  f"文件字节不同={not byte_identical}（容器 UID/时间戳）")
        else:
            print("[s5] deterministic 组：没有可比较的成品对")
    else:
        if byte_identical:
            print("[s5] 默认组：成品字节完全一致（多线程下也一致）")
        else:
            print("[s5] 默认组：成品字节不同 —— 与多线程 stage1 的已知竞态一致；"
                  "要看数值量级请读 report 里的 psnr")
    # 逐帧 PSNR（字节不同时要知道差在哪）
    if fault is not None:
        if fault.get("rerun_rc") != 0:
            print(f"[FAIL] 故障重跑未成功: rc={fault.get('rerun_rc')}")
            ok = False
        if not fault.get("final"):
            print("[FAIL] 故障重跑没有产出成品")
            ok = False

    report = {"root": root, "mode": args.mode, "encoder": enc,
              "deterministic": bool(args.deterministic),
              "byte_identical": bool(byte_identical),
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
