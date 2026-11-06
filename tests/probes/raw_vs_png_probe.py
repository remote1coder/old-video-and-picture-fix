# -*- coding: utf-8 -*-
"""重测第十七轮的「raw 流式 vs PNG 中转」等价性。

## 为什么重测

第十七轮的结论是：

    数值差异：PSNR y:42.80 u:54.94 v:54.22 avg:44.42 min:32.19 max:52.35
    SSIM Y:0.9968 All:0.9973
    → 判为"数值等价"，差异量级"与 stage6 随机颗粒相符"

两个问题：

1. **那组数字是在多线程下测的**。本轮已查明 stage1 渲染本底噪声为
   Y≈48.7 / U≈51.5 / V≈39.6（min≈42.1 dB）。avg 44.42 落在本底内，
   单看 avg 无法判断。
2. **但 min 32.19 dB 低于本底** —— 说明确实存在真差异，不是纯噪声。
   而且当时对比的是 **stage3 输出**（AI 超分结果，150 帧 544×960→1088×1920），
   **不含 stage6 颗粒**，所以"与随机颗粒相符"这个解释站不住。

## 这次怎么做

在 `vs.deterministic: true` 下跑两次同一个素材：

    A: ai_stream_raw = True   （零落盘流式：ffmpeg→AI→ffmpeg 管道）
    B: ai_stream_raw = False  （PNG 中转）

对比 **step3**（AI 输出），把差异严格归属于"两条 AI 路径"，
不受 stage4/6 影响。

判据：`np.array_equal` 级 / 逐帧逐平面 PSNR。**不看 avg 单值**，
同时报 min 与有差异的帧数。

## 运行

    tools_r4\\vs_portable\\r72\\python.exe tests/probes/raw_vs_png_probe.py

（不需要 .venv_r4：跑批由 main.py 驱动，它会自己解析 AI 解释器。）
"""
import json
import os
import shutil
import subprocess
import sys

PROJ = r"E:\codex\code1\old-video-and-picture-fix"
BASE = r"E:\codex\_media_r17_20261031T0004\rawcmp2"
SRC = r"E:\codex\_media_r17_20261031T0004\d2probe\in\src_cfr.mp4"
PY = os.path.join(PROJ, "tools_r4", "vs_portable", "r72", "python.exe")
MAIN = os.path.join(PROJ, "video_restoration_pipeline", "main.py")
CFG = os.path.join(PROJ, "video_restoration_pipeline", "config.yaml")


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace", shell=False, **kw)


def probe(path):
    r = sh(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries",
            "stream=nb_read_frames,width,height,pix_fmt,r_frame_rate",
            "-of", "json", path])
    try:
        s = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    except Exception:
        s = {}
    return {k: s.get(k) for k in
            ("width", "height", "pix_fmt", "nb_read_frames", "r_frame_rate")}


def write_manifest(work, stream_raw):
    import yaml

    os.makedirs(os.path.join(work, "temp"), exist_ok=True)
    os.makedirs(os.path.join(work, "out"), exist_ok=True)
    os.makedirs(os.path.join(work, "logs"), exist_ok=True)
    info = probe(SRC)
    job = {
        "job_id": "rvp", "input_path": SRC,
        "detected_class": "HDV_AVCHD", "confidence": 0.98,
        "field_order": "progressive", "needs_ivtc": False, "is_psf": False,
        "field_segments": [], "status": "PENDING", "priority": 0,
        "ai_upscale_banned": False, "recommended_params": {},
        "user_overrides": {
            "ai_stream_raw": bool(stream_raw),
            "force_allow_banned_upscale": True,
            # 保留 step3/step4，否则跑完只剩 step6，无从对照
            "cleanup_intermediates": False,
            # ---- 本次的核心变量：确定性 ----
            # 不开这个，stage1 的抖动（min≈42.1 dB）会盖住真实差异
            "vs_deterministic": True,
        },
        "duration": 5.0, "original_width": 544, "original_height": 960,
        "file_size": os.path.getsize(SRC), "file_mtime": 0.0,
        "file_hash_prefix": "", "run_generation": 1,
        "temp_dir": os.path.join(work, "temp"),
        "log_path": os.path.join(work, "logs", "job.log"),
        "output_path": os.path.join(work, "out", "rvp_restored.mkv"),
    }
    path = os.path.join(work, "m.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"meta": {
            "generated_at": "2026-11-06T13:10:00+08:00",
            "source_dirs": [os.path.dirname(SRC)],
            "config_path": "", "total_files": 1, "pending": 1, "done": 0,
            "failed": 0, "skipped": 0, "run_generation": 1},
            "jobs": [job]}, f, allow_unicode=True, sort_keys=False)
    return path


def run(stream_raw, tag):
    work = os.path.join(BASE, tag)
    if os.path.isdir(work):
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    man = write_manifest(work, stream_raw)

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([
        os.path.join(PROJ, "tools_r4"),
        os.path.join(PROJ, "tools_r4", "gmic", "gmic-4.0.5-cli-win64"),
        os.path.join(PROJ, "tools_r4", "exiftool", "exiftool-13.59_64"),
        env.get("PATH", "")])

    print("\n  [%s] ai_stream_raw=%s" % (tag, stream_raw))
    r = sh([PY, MAIN, "--manifest", man, "--config", CFG,
            "--mode", "batch", "--no-wsl-redirect", "--ai-enabled",
            "--ai-upscale-factor", "2", "--force-allow-banned-upscale",
            "--workers", "1", "--allow-write-root", work], env=env)
    print("    rc=%d" % r.returncode)
    if r.returncode != 0:
        for ln in (r.stdout or "").splitlines()[-6:]:
            print("      " + ln)
        for ln in (r.stderr or "").splitlines()[-6:]:
            print("      " + ln)

    step3 = None
    td = os.path.join(work, "temp")
    if os.path.isdir(td):
        for n in sorted(os.listdir(td)):
            if "_step3_" in n and n.endswith(".mkv"):
                step3 = os.path.join(td, n)
    print("    step3: %s" % (os.path.basename(step3) if step3 else "(无)"))
    return {"rc": r.returncode, "step3": step3, "work": work}


def compare(a, b):
    r = sh(["ffmpeg", "-hide_banner", "-v", "info", "-i", a, "-i", b,
            "-lavfi", "psnr=stats_file=-", "-f", "null", "-"])
    rows = []
    for ln in (r.stdout or "").splitlines():
        if not ln.startswith("n:"):
            continue
        d = dict(x.split(":", 1) for x in ln.split() if ":" in x)
        rows.append(d)
    if not rows:
        return {"ok": None, "reason": "未解析到逐帧数据",
                "stderr_tail": (r.stderr or "").splitlines()[-3:]}

    def plane(key):
        """统计某个平面；**返回 present 以区分"没有数据"与"全部 inf"**。

        这是本项目已经栽过两次的坑：
        - 融合对照时 `stats_file=-` 写 stdout 我读了 stderr → 0 帧数据被
          判成"完全相同"；
        - 这次是 y/u/v 拿不到（因为两边格式不同，ffmpeg 转成 RGB 比较，
          报的是 r/g/b），`min is None` 又被当成"全 inf"。
        **没有数据 ≠ 完全相同。**
        """
        vals = []
        for d in rows:
            v = d.get(key)
            if v is None:
                continue
            vals.append(None if v.startswith("inf") else float(v))
        finite = [x for x in vals if x is not None]
        return {
            "present": bool(vals),
            "n_total": len(vals),
            "n_inf": sum(1 for x in vals if x is None),
            "min": min(finite) if finite else None,
            "avg": (sum(finite) / len(finite)) if finite else None,
        }

    return {
        "ok": True, "frames": len(rows),
        "y": plane("psnr_y"), "u": plane("psnr_u"), "v": plane("psnr_v"),
        "r": plane("psnr_r"), "g": plane("psnr_g"), "b": plane("psnr_b"),
        "summary": [l.strip()[:120] for l in (r.stderr or "").splitlines()
                    if "PSNR" in l],
    }


def main():
    if not os.path.isfile(SRC):
        print("  素材不存在: %s" % SRC)
        return 2
    os.makedirs(BASE, exist_ok=True)

    info = probe(SRC)
    print("  素材: %s" % SRC)
    print("    %sx%s  pix_fmt=%s  帧数=%s  fps=%s"
          % (info.get("width"), info.get("height"), info.get("pix_fmt"),
             info.get("nb_read_frames"), info.get("r_frame_rate")))

    a = run(True, "raw")
    b = run(False, "png")

    print("\n  === 对照（stage3 输出，确定性模式）===")
    if not (a["step3"] and b["step3"]):
        print("    缺少 step3 产物，无法对照")
        return 1
    pinfo = {}
    for tag, r in (("raw", a), ("png", b)):
        p = probe(r["step3"])
        pinfo[tag] = p
        print("    %-4s %sx%s %s 帧=%s"
              % (tag, p.get("width"), p.get("height"), p.get("pix_fmt"),
                 p.get("nb_read_frames")))

    fmt_raw = pinfo["raw"].get("pix_fmt")
    fmt_png = pinfo["png"].get("pix_fmt")
    fmt_same = (fmt_raw == fmt_png)
    print("    pix_fmt 一致: %s（raw=%s / png=%s）"
          % (fmt_same, fmt_raw, fmt_png))

    cmp = compare(a["step3"], b["step3"])
    result = {"src": SRC, "src_info": info,
              "raw": {"rc": a["rc"], "step3": a["step3"]},
              "png": {"rc": b["rc"], "step3": b["step3"]},
              "compare": cmp}
    print("    " + json.dumps(cmp, ensure_ascii=False)[:400])

    verdict = "UNKNOWN"
    if not cmp.get("ok"):
        print("\n  未取得逐帧数据，无法判定。")
    elif not fmt_same:
        # ---- 格式不同就是不等价的**结构性**证据，不需要再看 PSNR ----
        verdict = "DIFFERENT_PIX_FMT"
        print("\n  **两条路径的 step3 像素格式不同** —— 这本身就不等价。")
        print("    raw = %s（RGB 无损保存）" % fmt_raw)
        print("    png = %s（stage3 编码时已转 YUV）" % fmt_png)
        print("    下游 stage4 的 `format=yuv420p` 在两种输入上的转换前提不同，")
        print("    所以成品会有可见但很小的差异（见下方 PSNR）。")
        print("    这不是噪声 —— 是两条路径的实现分叉。")
    else:
        present = [k for k in ("y", "u", "v", "r", "g", "b")
                   if (cmp.get(k) or {}).get("present")]
        if not present:
            verdict = "UNKNOWN"
            print("\n  没有任何平面数据 —— 不能据此判相同。")
        else:
            worst = min([(cmp[k] or {}).get("min")
                         for k in present if (cmp[k] or {}).get("min")
                         is not None] or [999])
            all_inf = all((cmp[k] or {}).get("n_inf", 0)
                          == (cmp[k] or {}).get("n_total", -1) for k in present)
            if all_inf:
                verdict = "IDENTICAL"
                print("\n  **逐帧完全相同** —— 两条路径数值一致。")
            elif worst < 42.0:
                verdict = "REAL_DIFF"
                print("\n  **存在真实差异**：最差平面 %.2f dB（低于本底 42.1）"
                      % worst)
            else:
                verdict = "SMALL_DIFF"
                print("\n  差异很小但**不为零**：最差 %.2f dB" % worst)
                print("    （高于本底 42.1，所以不是噪声；但也不构成'等价'）")
    print("    verdict = %s" % verdict)
    result["verdict"] = verdict

    out = os.path.join(BASE, "result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("    报告: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
