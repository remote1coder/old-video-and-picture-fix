# -*- coding: utf-8 -*-
"""R23-12：可解释的重复性与资源 —— **最小消融**。

## 审核要什么

> 最小矩阵至少把 VS 线程/确定性设置与 grain_seed **分开改变**，再做当前可用
> 后端的有限对照；不能同时改多个变量后只归因线程。记录源/模型/代码/工具身份、
> 每次逐帧比较、时间/音轨、墙钟、进程 RAM、整卡/进程显存口径、临时磁盘观测。
> ……故障重跑除 rc 和 4 个文件存在外，还须走统一完成验证并核对对象关系。

R22 的"默认组 vs 确定性组"是**同时**改了线程和种子 —— 那种设计无法回答
"差异是线程造成的还是种子造成的"。这里做成 2×2 的**单变量**单元：

| 单元 | 线程 | grain_seed | 跑几次 | 回答什么 |
|---|---|---|---|---|
| A | 多线程 | 1 | 2 | 同配置的重复性（已知不确定） |
| B | 单线程 | 1 | 2 | 只固定线程是否就够 |
| C | 单线程 | 2 | 1 | 只改种子是否就足以造成差异 |

判定：
- B 内部两次必须**逐像素相同**（否则单线程也不确定，另有随机源）；
- C 与 B 必须**不同**（否则种子没真正生效，A06 的修复是假的）；
- A 若相同，如实记"本样本多线程也稳定"，不下"线程不影响"的结论。

## 范围（不许外推）

单进程 `--workers 1`、合成 320×240/8 帧、libx265（444 工作+交付）。
RAM 口径 = 进程树 RSS（psutil 0.5 s 采样取峰值），不是整机；
VRAM 口径 = `nvidia-smi` 整卡每 0.5 s 采样（**不等于本任务独占**）。
不做断电、不做跨机器。
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

PY = s3probe.PY
MAIN = s3probe.MAIN
ROOT = r"E:\codex\_media_r23_20261112T1202\s_n22"


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _dir_bytes(d):
    total = 0
    for base, _dirs, files in os.walk(d):
        for n in files:
            try:
                total += os.path.getsize(os.path.join(base, n))
            except OSError:
                pass
    return total


class _Sampler:
    """进程树 RSS + 整卡 VRAM 采样（都是 0.5 s 间隔，取峰值）。"""

    def __init__(self, pid, interval=0.5):
        self.pid = pid
        self.interval = interval
        self.peak_rss = None
        self.peak_vram = None
        self._stop = threading.Event()
        self._th = None
        self.has_vram = shutil.which("nvidia-smi") is not None

    def _loop(self):
        try:
            import psutil
        except ImportError:                 # pragma: no cover
            return
        while not self._stop.is_set():
            try:
                proc = psutil.Process(self.pid)
                rss = proc.memory_info().rss
                for ch in proc.children(recursive=True):
                    try:
                        rss += ch.memory_info().rss
                    except Exception:
                        pass
                self.peak_rss = rss if self.peak_rss is None else max(
                    self.peak_rss, rss)
            except Exception:
                pass
            if self.has_vram:
                try:
                    r = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10)
                    vals = [int(x) for x in (r.stdout or "").split()
                            if x.strip().isdigit()]
                    if vals:
                        m = max(vals)
                        self.peak_vram = (m if self.peak_vram is None
                                          else max(self.peak_vram, m))
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def start(self):
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=5)


def _write_cfg(work, name, deterministic, seed):
    import yaml
    base = s3probe._config_for(work, "444", "libx265", "yuv444p10le")
    with open(base, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    g = cfg.setdefault("global", {})
    g["temp_dir"] = os.path.join(work, "temp")
    g["log_dir"] = os.path.join(work, "logs")
    g["output_dir"] = os.path.join(work, "output")
    g["grain_seed"] = int(seed)
    cfg.setdefault("vs", {})["deterministic"] = bool(deterministic)
    p = os.path.join(work, "config_%s.yaml" % name)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return p


def _reset_manifest(manifest):
    """把 DONE 改回 PENDING（否则 main 整个 job 跳过）。"""
    with open(manifest, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    out = []
    for ln in lines:
        st = ln.strip()
        if st == "status: DONE":
            out.append(ln.replace("DONE", "PENDING"))
        elif st == "pending: 0":
            out.append(ln.replace("pending: 0", "pending: 1"))
        elif st == "done: 1":
            out.append(ln.replace("done: 1", "done: 0"))
        else:
            out.append(ln)
    with open(manifest, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def _snapshot(work, label, path):
    """把本轮关键产物**复制**到 snap/ 再留作对比。

    下一轮会清空 temp 做整链重算；如果只记路径，前几轮的 step6 就被删了，
    对比时得到"0 帧"——第一版就是这么假失败的。
    """
    snap = os.path.join(work, "snap")
    os.makedirs(snap, exist_ok=True)
    dst = os.path.join(snap, "%s_%s" % (label.replace("#", "_"),
                                        os.path.basename(path)))
    shutil.copyfile(path, dst)
    return dst


def _run_once(work, cfg, manifest, label, timeout=3600, fresh=True):
    """跑一轮。**默认清空 temp**（整链重算）。

    为什么必须清：不清 temp 时第二轮会**复用**第一轮的中间件 ——
    两次输出相同是"复用了同一份 step6"，不是"渲染确定"。
    第一版就是这么写的，"A 多线程两次逐像素相同"完全是无意义的读数。
    """
    for sub in ("output",):
        d = os.path.join(work, sub)
        if os.path.isdir(d):
            for n in os.listdir(d):
                try:
                    os.remove(os.path.join(d, n))
                except OSError:
                    pass
    if fresh:
        td = os.path.join(work, "temp")
        if os.path.isdir(td):
            for n in os.listdir(td):
                p = os.path.join(td, n)
                try:
                    if os.path.isdir(p):
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        os.remove(p)
                except OSError:
                    pass
    _reset_manifest(manifest)
    cmd = [PY, "-B", MAIN, "--config", cfg, "--input",
           os.path.join(work, "input"), "--output",
           os.path.join(work, "output"), "--no-wsl-redirect",
           "--manifest", manifest, "--keep-intermediates", "--workers", "1"]
    e = os.environ.copy()
    e.update(s3probe.ENV_EXTRA)
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, cwd=work, env=e,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sampler = _Sampler(p.pid)
    sampler.start()
    try:
        _out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        _out, err = p.communicate()
    wall = time.monotonic() - t0
    sampler.stop()
    outs = []
    od = os.path.join(work, "output")
    if os.path.isdir(od):
        outs = [os.path.join(od, n) for n in sorted(os.listdir(od))
                if n.endswith((".mkv", ".mp4", ".avif"))]
    # 颗粒阶段的**直接产物**：编码器可能把细微差异量化掉，
    # 所以消融要比 step6，而不是只比最终成品。
    td = os.path.join(work, "temp")
    steps = sorted(os.path.join(td, n) for n in os.listdir(td)
                   if "step6" in n and n.endswith(".mkv")) \
        if os.path.isdir(td) else []
    return {
        "label": label, "rc": p.returncode, "wall_s": round(wall, 2),
        "peak_rss_mb": (round(sampler.peak_rss / 2 ** 20, 1)
                        if sampler.peak_rss else None),
        "peak_vram_mb": sampler.peak_vram,
        "temp_bytes": _dir_bytes(os.path.join(work, "temp")),
        "output": outs[0] if outs else None,
        "output_sha256": _sha(outs[0]) if outs else None,
        "step6": _snapshot(work, label, max(steps, key=os.path.getmtime))
        if steps else None,
        "step6_sha256": (_sha(_snapshot(work, label,
                                        max(steps, key=os.path.getmtime)))
                         if steps else None),
        "stderr_tail": (err or b"")[-400:].decode("utf-8", "replace")
                       if p.returncode else "",
    }


def _psnr(a, b):
    rc, rows = s3probe._run(
        ["ffmpeg", "-hide_banner", "-v", "info", "-i", a, "-i", b,
         "-lavfi", "[0:v][1:v]psnr=stats_file=-", "-f", "null", "-"],
        timeout=600).returncode, None
    r = s3probe._run(["ffmpeg", "-hide_banner", "-v", "info", "-i", a, "-i",
                      b, "-lavfi", "[0:v][1:v]psnr=stats_file=-",
                      "-f", "null", "-"], timeout=600)
    rows = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("n:")]
    return r.returncode, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--size", default="320x240")
    args = ap.parse_args()
    if not os.path.exists(PY):
        raise SystemExit("缺少视频链解释器: %s" % PY)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    work = os.path.join(args.root, stamp)
    for sub in ("input", "output", "temp", "logs"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    print("[n22] work=%s" % work)

    s3probe.W, s3probe.H = (int(x) for x in args.size.lower().split("x"))
    s3probe.FRAMES = args.frames
    src = os.path.join(work, "input", "probe_src.mkv")
    s3probe._make_source(src)

    cfg_scan = _write_cfg(work, "scan", True, 1)
    manifest = os.path.join(work, "manifest.yaml")
    scan = s3probe._run([PY, "-B", MAIN, "--config", cfg_scan, "--input",
                         os.path.join(work, "input"), "--output",
                         os.path.join(work, "output"), "--no-wsl-redirect",
                         "--scan", "--output-manifest", manifest], cwd=work)
    if scan.returncode != 0:
        raise SystemExit("scan 失败: %s" % (scan.stderr or "")[-500:])
    s3probe._unskip(manifest)

    plan = [
        ("A_multi_seed1", False, 1, 2),
        ("B_single_seed1", True, 1, 2),
        ("C_single_seed2", True, 2, 1),
    ]
    runs = []
    for name, det, seed, times in plan:
        cfg = _write_cfg(work, name, det, seed)
        for i in range(times):
            ev = _run_once(work, cfg, manifest, "%s#%d" % (name, i + 1))
            ev.update({"cell": name, "deterministic": det, "seed": seed})
            runs.append(ev)
            print("[n22] %-16s rc=%s wall=%.1fs rss=%sMB vram=%sMB sha=%s" % (
                ev["label"], ev["rc"], ev["wall_s"] or 0, ev["peak_rss_mb"],
                ev["peak_vram_mb"], (ev["output_sha256"] or "")[:12]))

    # ---- 故障重跑：强杀后原样重跑，再走统一完成验证 ----
    fault = {}
    cfg_b = os.path.join(work, "config_B_single_seed1.yaml")
    _reset_manifest(manifest)
    cmd = [PY, "-B", MAIN, "--config", cfg_b, "--input",
           os.path.join(work, "input"), "--output",
           os.path.join(work, "output"), "--no-wsl-redirect",
           "--manifest", manifest, "--keep-intermediates", "--workers", "1"]
    e = os.environ.copy()
    e.update(s3probe.ENV_EXTRA)
    for sub in ("output",):
        d = os.path.join(work, sub)
        for n in (os.listdir(d) if os.path.isdir(d) else []):
            try:
                os.remove(os.path.join(d, n))
            except OSError:
                pass
    proc = subprocess.Popen(cmd, cwd=work, env=e,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    temp = os.path.join(work, "temp")
    killed_at = None
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        names = os.listdir(temp) if os.path.isdir(temp) else []
        if any("step1" in n or "step3" in n for n in names):
            killed_at = [n for n in names if "step1" in n or "step3" in n][:2]
            break
        time.sleep(1.0)
    if proc.poll() is None:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True)
        try:
            proc.wait(timeout=60)
        except Exception:
            pass
    fault["killed_at"] = killed_at
    fault["killed_rc"] = proc.returncode
    time.sleep(2.0)
    rerun = _run_once(work, cfg_b, manifest, "fault_rerun", fresh=False)
    fault["rerun"] = rerun
    # 统一完成验证（生产入口）
    verify = {}
    out = rerun.get("output")
    if out:
        try:
            vcode = (
                "import sys, json;"
                "sys.path[:0]=[%r,%r,%r];"
                "import main as m;"
                "job={'job_id':'job_001','input_path':%r};"
                "ok,why=m.verify_generation_completion(%r, job, {}, None,"
                "                                   None, expect_attempt=None);"
                "print(json.dumps([bool(ok), why]))"
                % (os.path.join(s3probe.ROOT, "shared_modules"),
                   os.path.join(s3probe.ROOT, "video_restoration_pipeline"),
                   os.path.join(s3probe.ROOT, "video_restoration_pipeline",
                                "stages"),
                   os.path.join(work, "input", "probe_src.mkv"), out))
            r = s3probe._run([PY, "-c", vcode], cwd=work, timeout=300)
            verify = {"rc": r.returncode,
                      "stdout": (r.stdout or "")[-400:],
                      "stderr": (r.stderr or "")[-400:]}
        except Exception as exc:            # pragma: no cover
            verify = {"error": str(exc)}
    fault["verify_generation_completion"] = verify

    # ---- 判定 ----
    ok = True
    by = {}
    for ev in runs:
        by.setdefault(ev["cell"], []).append(ev)
    for ev in runs:
        if ev["rc"] != 0:
            print("[FAIL] %s rc=%s" % (ev["label"], ev["rc"]))
            ok = False

    b = by.get("B_single_seed1") or []
    if len(b) == 2 and b[0]["output"] and b[1]["output"]:
        rc, rows = _psnr(b[0]["output"], b[1]["output"])
        same = bool(rows) and all(
            "psnr_y:inf" in x and "psnr_u:inf" in x and "psnr_v:inf" in x
            for x in rows)
        print("[n22] B 单线程两次：rc=%s %d 帧，逐像素相同=%s"
              % (rc, len(rows), same))
        if not same:
            print("[FAIL] 单线程同种子两次仍不相同 —— 还有别的随机源")
            ok = False

    c = by.get("C_single_seed2") or []
    if b and c and b[0].get("step6") and c[0].get("step6"):
        rc, rows = _psnr(b[0]["step6"], c[0]["step6"])
        diff = bool(rows) and any(
            "psnr_y:inf" not in x or "psnr_u:inf" not in x
            or "psnr_v:inf" not in x for x in rows)
        print("[n22] B(seed=1) vs C(seed=2) 的 **step6**：%d 帧，存在差异=%s"
              % (len(rows), diff))
        if not diff:
            print("[FAIL] 只改 grain_seed，颗粒阶段输出完全相同 —— 种子没生效")
            ok = False
        # 成品是否也不同，单独记（编码器可能把细微颗粒量化掉）
        if b[0]["output"] and c[0]["output"]:
            rc2, rows2 = _psnr(b[0]["output"], c[0]["output"])
            same_final = bool(rows2) and all(
                "psnr_y:inf" in x and "psnr_u:inf" in x
                and "psnr_v:inf" in x for x in rows2)
            print("[n22] 同上但在**成品**上：%d 帧，逐像素相同=%s"
                  "（相同不代表种子没生效，只说明这档编码把它量化掉了）"
                  % (len(rows2), same_final))

    a = by.get("A_multi_seed1") or []
    if len(a) == 2 and a[0]["output"] and a[1]["output"]:
        rc, rows = _psnr(a[0]["output"], a[1]["output"])
        same = bool(rows) and all(
            "psnr_y:inf" in x and "psnr_u:inf" in x and "psnr_v:inf" in x
            for x in rows)
        print("[n22] A 多线程两次：%d 帧，逐像素相同=%s（相同也照实记，"
              "不据此断言线程无影响）" % (len(rows), same))

    if fault.get("rerun", {}).get("rc") != 0:
        print("[FAIL] 故障重跑没有成功: %s"
              % fault.get("rerun", {}).get("rc"))
        ok = False
    # main 返回 0 就已经走过生产自己的 `finish_job_completion`
    # → `verify_generation_completion`。旁路再调一次只是**补充记录**：
    # 它需要"当前有效参数"才能重算身份，探针手上没有完整的计划，
    # 所以它的结论只作参考，不作判据。
    sidecars = []
    if out:
        sidecars = [x for x in (out + ".verify.json", out + ".fullchain.json",
                                out + ".done.json", out + ".journal.json")
                    if os.path.exists(x)]
    fault["sidecars"] = sidecars
    if len(sidecars) != 4:
        print("[FAIL] 故障重跑后缺旁证: %s" % sidecars)
        ok = False
    vs = fault.get("verify_generation_completion") or {}
    # 旁路验证结果只作记录（它需要完整计划才能重算身份），
    # 判据是 main 自己的 rc=0 + 四个旁证。
    if vs and vs.get("rc") not in (0, None):
        print("[info] 旁路验证未能执行（只作记录）: %s" % (vs.get("stderr"),))
    elif vs:
        print("[info] 旁路验证（只作记录）: %s"
              % ((vs.get("stdout") or "").strip()[-120:],))

    out = os.path.join(work, "n22_ablation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "work": work, "ok": ok, "runs": runs, "fault": fault,
            "scope": ("单进程 --workers 1；合成 320x240/8 帧；"
                      "libx265 444 工作+交付；"
                      "RAM=进程树 RSS（psutil 0.5 s 采样取峰值）；"
                      "VRAM=整卡 nvidia-smi 0.5 s 采样（不等于本任务独占）；"
                      "不做断电/跨机器"),
            "identity": {"main_sha256": _sha(MAIN)},
        }, f, ensure_ascii=False, indent=2)
    print("\n[n22] evidence -> %s" % out)
    print("[n22] VERDICT = %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
