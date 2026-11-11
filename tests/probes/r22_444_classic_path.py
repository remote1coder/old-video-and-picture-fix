# -*- coding: utf-8 -*-
"""R22 §5.2 · 手动 444 / 高位深**经典路径**的小样本贯通探针。

## 它证明什么

审核要求：

> 先贯通一条小样本经典 444/高位深路径：入口→AI/非AI→后处理→颗粒/面部
> 适用分支→编码。确保每个边界的实际格式、颜色和时间合同明确。
> ……更新参数依赖与检查点身份，切换 420/444 后相关产物失效，
> 不能复用旧 420 再换标签。

这个探针跑两遍**同一个小样本**：一遍 `chroma_mode: 420`（历史路径），
一遍 `chroma_mode: 444`，然后：

1. 逐 stage 读**实际落盘的** pix_fmt（不是"我们请求了什么"）；
2. 断言 444 那一遍从 stage1 起每一处都是 4:4:4 —— 包括 stage4 的掩模分支
   与 stage7 的成品；
3. 断言两遍的 `param_hash` 不同（切换表示 → 旧产物失效）；
4. 断言两遍成品**不是同一份文件**（内容不同），并给出逐帧 PSNR
   —— 若完全相同，说明 444 什么都没做（那就是"换标签"）。

## 用法

    python tests/probes/r22_444_classic_path.py [--keep]

`--keep` 保留工作目录（默认也保留，证据要能回看）。
工作根：``E:\\codex\\_media_r22_20261111T2330\\s3_444\\<时间戳>``
（R22 指令要求所有测试副本/媒体/缓存都在新隔离根下，不写 C 盘。）

## 为什么用 libx265

`ffmpeg -h encoder=libsvtav1` 在本机只列 `yuv420p yuv420p10le` ——
**SVT-AV1 不支持 4:4:4**。`work_format.resolve()` 会在动手前就把它判成
blocker 并 rc=86。要真跑 444 必须换编码器，探针因此用 libx265。
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
MAIN = os.path.join(ROOT, "video_restoration_pipeline", "main.py")
CONFIG = os.path.join(ROOT, "video_restoration_pipeline", "config.yaml")
PY = os.path.join(ROOT, "tools_r4", "vs_portable", "r72", "python.exe")
MEDIA_ROOT = r"E:\codex\_media_r22_20261111T2330\s3_444"

#: 小样本：24 帧 / 10fps / 320x240 —— 全链几秒级。
W, H, FPS, FRAMES = 320, 240, 10, 24

ENV_EXTRA = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUTF8": "1",
    # vspipe / 插件 / gmic / exiftool 全靠 PATH 解析。
    # 缺 vspipe → 环境自检 rc=80；缺 gmic → DEP_CHECK rc=80
    # （日志里写 "Missing external tools: - gmic"）。
    "PATH": os.pathsep.join([
        os.path.join(ROOT, "tools_r4", "vs_portable", "r72"),
        os.path.join(ROOT, "tools_r4"),
        os.path.join(ROOT, "tools_r4", "gmic", "gmic-4.0.5-cli-win64"),
        os.path.join(ROOT, "tools_r4", "exiftool", "exiftool-13.59_64"),
        os.environ.get("PATH", ""),
    ]),
}


def _run(cmd, *, cwd=None, env=None, timeout=1800):
    e = os.environ.copy()
    e.update(ENV_EXTRA)
    if env:
        e.update(env)
    e["TEMP"] = os.environ.get("TEMP") or r"E:\codex\_media_r22_20261111T2330\tmp"
    e["TMP"] = e["TEMP"]
    return subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True,
                          errors="replace", timeout=timeout, shell=False)


def _ff(*args, timeout=600):
    return _run(["ffmpeg", "-hide_banner", "-v", "error", "-y", *args],
                timeout=timeout)


def _probe_video(path):
    r = _run(["ffprobe", "-hide_banner", "-v", "error",
              "-select_streams", "v:0",
              "-show_entries",
              "stream=pix_fmt,width,height,r_frame_rate,nb_frames",
              "-show_entries", "format=duration",
              "-of", "json", str(path)], timeout=60)
    if r.returncode != 0:
        return {}
    try:
        doc = json.loads(r.stdout or "{}")
    except ValueError:
        return {}
    st = (doc.get("streams") or [{}])[0]
    fmt = doc.get("format") or {}
    st["duration"] = fmt.get("duration")
    if not str(st.get("nb_frames") or "").isdigit():
        # ffv1/mkv 的 nb_frames 常是 N/A，用解复用数包补（1 packet = 1 frame）
        rc = _run(["ffprobe", "-hide_banner", "-v", "error",
                   "-select_streams", "v:0", "-count_packets",
                   "-show_entries", "stream=nb_read_packets",
                   "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                  timeout=120)
        v = (rc.stdout or "").strip()
        st["nb_frames"] = v if v.isdigit() else None
    return st


def _make_source(path, pix_fmt="yuv444p10le"):
    """造一个**带真实色度细节**的源：静态画面 + 逐像素细密彩色条纹。

    两点考虑：
    * **必须是静态/无梳状的画面**：`testsrc2` 的滚动条纹会被 stage2 的
      逐行率判据判成 interlaced（实测 Progressive=58.3% < 95%），
      于是 stage2 rc=86，跟 444 完全无关。用静止图避开这个误判。
    * **色度必须有细节**：纯灰阶在 420 下的损失看不出来，
      "444 到底有没有生效"就无法判定。条纹是逐像素的，420 必然抹掉。
    """
    still = os.path.join(os.path.dirname(path), "_still.png")
    r = _ff("-f", "lavfi", "-i",
            f"testsrc2=size={W}x{H}:rate=1:duration=1", "-frames:v", "1",
            "-pix_fmt", "rgb48be", still)
    if r.returncode != 0:
        raise SystemExit(f"造静帧失败: {(r.stderr or '')[-400:]}")
    # 叠一层逐像素红/青条纹（chroma 高频）
    stripe = (
        f"color=c=red:size={W}x{H}:rate=1:duration=1,"
        f"geq=lum='if(mod(X,2),235,16)':cb=128:cr=128[st];"
        f"[0:v][st]blend=all_mode=multiply:all_opacity=0.45"
    )
    r = _ff("-loop", "1", "-i", still, "-filter_complex", stripe,
            "-frames:v", str(FRAMES), "-r", str(FPS),
            "-c:v", "ffv1", "-pix_fmt", pix_fmt, path)
    if r.returncode != 0:
        raise SystemExit(f"造源失败: {(r.stderr or '')[-500:]}")
    try:
        os.remove(still)
    except OSError:
        pass
    return path


def _config_for(work, chroma_mode, encoder, pix_fmt):
    """从项目 config 派生一份探针配置（只改必要的几项）。"""
    import yaml
    with open(CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    g = cfg.setdefault("global", {})
    g["temp_dir"] = os.path.join(work, "temp")
    g["log_dir"] = os.path.join(work, "logs")
    g["output_dir"] = os.path.join(work, "output")
    # 小样本要快：关掉会拖时间的项
    g["video_encoder"] = encoder
    g["pix_fmt"] = pix_fmt
    g["chroma_mode"] = chroma_mode
    g["preset"] = 8
    g["crf"] = 30
    g["preset_max_quality"] = False
    g["stage_inactivity_timeout"] = 300
    # AI **开**：审核要求"入口→AI/非AI→后处理→颗粒/面部适用分支→编码"
    # 每个边界都有实际格式。AI 关掉时 stage3/stage4 整段不存在，
    # 444 在"AI 输出→后处理"这个最容易分叉的边界上就没有证据。
    cfg.setdefault("ai", {})
    cfg["ai"]["enabled"] = True
    cfg["ai"]["upscale_enabled"] = True
    cfg["ai"]["face_restore"] = ""
    cfg["ai"]["model"] = "RealESRGAN_x2plus"
    g["ai_enabled"] = True
    g["upscale_scale"] = 2
    g["upscale_model"] = "RealESRGAN_x2plus"
    # 分类器会把低分辨率素材判成"不该超分"；这是探针，显式放行
    g["ai_force_allow_banned_upscale"] = True
    g["force_allow_banned_upscale"] = True
    g["face_restore"] = ""
    p = os.path.join(work, "config_probe.yaml")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return p


def _unskip(manifest_path):
    """扫描器把低置信度素材写成 SKIP，而 main 只处理 PENDING/FAILED。"""
    with open(manifest_path, "r", encoding="utf-8") as f:
        txt = f.read()
    txt = re.sub(r"^(\s*)status:\s*SKIP\s*$", r"\1status: PENDING", txt,
                 flags=re.M)
    txt = re.sub(r"^\s*skip_reason:.*\n", "", txt, flags=re.M)
    txt = re.sub(r"^(\s*)pending:\s*0\s*$", r"\1pending: 1", txt, flags=re.M)
    txt = re.sub(r"^(\s*)skipped:\s*1\s*$", r"\1skipped: 0", txt, flags=re.M)
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


def _find_stage_files(work, tag):
    """在 temp 目录里按 stage 找产物（返回 {stage: path}）。"""
    temp = os.path.join(work, "temp")
    out = {}
    pats = {
        "stage1": r"step1",
        "stage2": r"step2",
        "stage3": r"step3",
        "stage4": r"step4",
        "stage5": r"step5",
        "stage6": r"step6",
    }
    if not os.path.isdir(temp):
        return out
    for name in sorted(os.listdir(temp)):
        if not name.endswith((".mkv", ".mp4", ".mov")):
            continue
        for stage, rx in pats.items():
            if stage in out:
                continue
            if re.search(rx, name, re.I):
                out[stage] = os.path.join(temp, name)
    return out


def _run_one(work, label, chroma_mode, encoder, pix_fmt):
    """跑一遍完整经典链，返回该遍的证据 dict。"""
    os.makedirs(work, exist_ok=True)
    for sub in ("input", "output", "temp", "logs"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    src = os.path.join(work, "input", "probe_src.mkv")
    _make_source(src)
    cfg = _config_for(work, chroma_mode, encoder, pix_fmt)
    manifest = os.path.join(work, "manifest.yaml")

    ev = {"label": label, "work": work, "mode": chroma_mode,
          "encoder": encoder, "pix_fmt_requested": pix_fmt,
          "source": _probe_video(src), "steps": []}

    scan = _run([PY, "-B", MAIN, "--config", cfg, "--input",
                 os.path.join(work, "input"),
                 "--output", os.path.join(work, "output"),
                 "--no-wsl-redirect", "--scan",
                 "--output-manifest", manifest], cwd=work)
    ev["steps"].append({"step": "scan", "rc": scan.returncode,
                        "stderr_tail": (scan.stderr or "")[-400:]})
    if scan.returncode != 0:
        return ev
    _unskip(manifest)

    run = _run([PY, "-B", MAIN, "--config", cfg, "--input",
                os.path.join(work, "input"),
                "--output", os.path.join(work, "output"),
                "--no-wsl-redirect", "--manifest", manifest,
                "--keep-intermediates", "--workers", "1"], cwd=work,
               timeout=3600)
    ev["steps"].append({"step": "batch", "rc": run.returncode,
                        "stdout_tail": (run.stdout or "")[-800:],
                        "stderr_tail": (run.stderr or "")[-1500:]})
    ev["rc"] = run.returncode

    # 每个边界的**实际**格式（探测，不信请求值）
    files = _find_stage_files(work, label)
    ev["boundaries"] = {}
    for stage, path in sorted(files.items()):
        probed = _probe_video(path)
        ev["boundaries"][stage] = {
            "file": os.path.basename(path),
            "pix_fmt": probed.get("pix_fmt"),
            "width": probed.get("width"),
            "height": probed.get("height"),
            "frames": probed.get("nb_frames"),
            "duration": probed.get("duration"),
        }
    outs = [os.path.join(work, "output", n)
            for n in os.listdir(os.path.join(work, "output"))
            if n.endswith((".mkv", ".mp4", ".avif", ".mov"))] \
        if os.path.isdir(os.path.join(work, "output")) else []
    ev["final"] = []
    for o in sorted(outs):
        probed = _probe_video(o)
        ev["final"].append({"file": os.path.basename(o), **probed})
    # 参数身份：切换表示必须换摘要。直接从**中间产物文件名**里取
    # `_p<param_hash>` —— 那正是检查点复用的依据，比翻日志可靠。
    hashes = set()
    temp = os.path.join(work, "temp")
    if os.path.isdir(temp):
        for n in os.listdir(temp):
            for m in re.finditer(r"_p([0-9a-f]{12,})", n):
                hashes.add(m.group(1))
    ev["param_hashes"] = sorted(hashes)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=MEDIA_ROOT)
    args = ap.parse_args()
    if not os.path.exists(PY):
        raise SystemExit(f"缺少视频链解释器: {PY}")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.root, stamp)
    os.makedirs(root, exist_ok=True)
    print(f"[probe] root = {root}")

    runs = [
        ("420", "420", "libsvtav1", "yuv420p10le"),
        ("444", "444", "libx265", "yuv444p10le"),
        # 负例：想 444 但没换编码器。必须**在动手前**就停（rc=86），
        # 而不是让 libsvtav1 悄悄降回 420 然后被当成"全程 444"。
        ("444_svtav1", "444", "libsvtav1", "yuv444p10le"),
    ]
    results = []
    for label, mode, enc, fmt in runs:
        work = os.path.join(root, f"run_{label}")
        print(f"\n[probe] === {label}: chroma_mode={mode} encoder={enc} "
              f"pix_fmt={fmt} ===")
        ev = _run_one(work, label, mode, enc, fmt)
        results.append(ev)
        print(f"[probe] rc={ev.get('rc')}")
        for stage, b in sorted((ev.get("boundaries") or {}).items()):
            print(f"   {stage:8} {b['pix_fmt']:14} {b['width']}x{b['height']} "
                  f"frames={b['frames']} dur={b['duration']}")
        for f in ev.get("final") or []:
            print(f"   FINAL    {f.get('pix_fmt')} {f.get('width')}x"
                  f"{f.get('height')} frames={f.get('nb_frames')}")
        print(f"   param_hashes={ev.get('param_hashes')}")

    # ---- 判定 ----
    ok = True
    a, b = results[0], results[1]
    neg = results[2] if len(results) > 2 else {}
    for ev in results[:2]:
        if ev.get("rc") != 0:
            print(f"[FAIL] {ev['label']}: rc={ev.get('rc')}")
            ok = False
    # 负例：请求 444 却没换编码器 → 必须在动手之前失败（rc=86），
    # 且**不产生任何成品**。若它 rc=0，说明静默降级了。
    if neg:
        if neg.get("rc") != 86:
            print(f"[FAIL] 444+libsvtav1 应 rc=86，实际 {neg.get('rc')}")
            ok = False
        if neg.get("final"):
            print(f"[FAIL] 能力核验失败却产出了成品: {neg.get('final')}")
            ok = False
        blog = ""
        for st in neg.get("steps") or []:
            blog += (st.get("stdout_tail") or "") + (st.get("stderr_tail") or "")
        if "WORK_FORMAT_REFUSED" not in blog and neg.get("rc") == 86:
            # 日志写在 log 文件里，stdout 没有也算正常，只提示不判失败
            print("[info] 负例的拒绝原因写在 logs/*.log 的 WORK_FORMAT_REFUSED")

    def _fmt(ev, stage):
        return ((ev.get("boundaries") or {}).get(stage) or {}).get("pix_fmt")

    def _chroma_ok(fmt, want):
        """RGB/GBRP 是**逐像素全采样**，对色度无损 → 任何模式都算合格。

        这条不是放宽标准：`gbrp16le` 没有色度子采样这一说，
        所以"444 要求"在它上面天然满足；真正要盯的是 YUV 边界。
        """
        if not fmt:
            return False
        f = str(fmt).lower()
        if f.startswith(("gbrp", "rgb", "bgr")):
            return True
        return want in f

    if ok:
        # AI 开着，所以 step1/2/3/4/6 都应当真的存在
        for stage in ("stage1", "stage2", "stage3", "stage4", "stage6"):
            if not _fmt(a, stage):
                print(f"[FAIL] 420 基线缺少 {stage} 产物（AI 没跑起来？）")
                ok = False
            elif not _chroma_ok(_fmt(a, stage), "420"):
                print(f"[FAIL] 420 基线 {stage} 不是 420: {_fmt(a, stage)}")
                ok = False
            if not _fmt(b, stage):
                print(f"[FAIL] 444 路径缺少 {stage} 产物（AI 没跑起来？）")
                ok = False
            elif not _chroma_ok(_fmt(b, stage), "444"):
                print(f"[FAIL] 444 路径 {stage} 不是 444: {_fmt(b, stage)}")
                ok = False
        final_b = (b.get("final") or [{}])[0].get("pix_fmt")
        if final_b != "yuv444p10le":
            print(f"[FAIL] 444 成品格式 = {final_b}（应为 yuv444p10le）")
            ok = False
        if a.get("param_hashes") and b.get("param_hashes") \
                and a["param_hashes"] == b["param_hashes"]:
            print("[FAIL] 切换 420/444 后 param_hash 没变 —— 旧产物会被复用")
            ok = False
        # 两遍成品必须真的不同（否则"444"只是换了个标签）
        fa = (a.get("final") or [{}])[0].get("file")
        fb = (b.get("final") or [{}])[0].get("file")
        if fa and fb:
            pa = os.path.join(a["work"], "output", fa)
            pb = os.path.join(b["work"], "output", fb)
            if os.path.exists(pa) and os.path.exists(pb):
                r = _run(["ffmpeg", "-hide_banner", "-v", "info", "-i", pa,
                          "-i", pb, "-lavfi",
                          "[0:v][1:v]psnr=stats_file=-", "-f", "null", "-"],
                         timeout=600)
                rows = [ln for ln in (r.stdout or "").splitlines()
                        if ln.startswith("n:")]
                identical = bool(rows) and all(
                    "psnr_y:inf" in ln and "psnr_u:inf" in ln
                    and "psnr_v:inf" in ln for ln in rows)
                print(f"[probe] 两遍成品逐帧比: {len(rows)} 帧, "
                      f"完全相同={identical}")
                if identical:
                    print("[FAIL] 444 与 420 成品逐像素相同 —— 444 没有生效")
                    ok = False

    out = os.path.join(root, "evidence.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"root": root, "ok": ok, "runs": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n[probe] evidence -> {out}")
    print(f"[probe] VERDICT = {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
