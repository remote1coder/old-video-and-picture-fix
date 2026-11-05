# -*- coding: utf-8 -*-
"""G34 融合等价性实测：同一条 raw+CFR 路线，经典 vs 融合，逐帧对照。

用法：
    python tools_r4/g34_equiv.py

产出：控制台报告 + `evidence/G34_EQUIV.json`。
判据（按审核的"无损处理边界"要求）：
  1. 两条路径的 step4 帧数一致；
  2. 逐帧 PSNR 为 inf（即像素完全相同）—— 只要出现有限值就说明
     融合改变了样本，必须定位原因，不能放宽成"肉眼差不多"。
"""
import json
import os
import shutil
import subprocess
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(PROJ, "tools_r4", "vs_portable", "r72", "python.exe")
ROOT = os.path.join("E:" + os.sep, "codex", "_media_r17_20261031T0004")
BASE = os.path.join(ROOT, "g34cmp")


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace", shell=False, **kw)


def make_clip(dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    clip = os.path.join(dst_dir, "g34.mp4")
    if os.path.exists(clip) and os.path.getsize(clip) > 0:
        return clip
    r = sh(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "lavfi",
            "-i", "testsrc2=size=1920x1080:rate=30000/1001:duration=1",
            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
            "-frames:v", "20", clip])
    if r.returncode != 0:
        raise SystemExit("造片失败: " + r.stderr[-300:])
    return clip


def write_manifest(work, clip):
    import yaml
    os.makedirs(os.path.join(work, "temp"), exist_ok=True)
    os.makedirs(os.path.join(work, "out"), exist_ok=True)
    os.makedirs(os.path.join(work, "logs"), exist_ok=True)
    job = {
        "job_id": "g34", "input_path": clip,
        "detected_class": "HDV_AVCHD", "confidence": 0.98,
        "field_order": "progressive", "needs_ivtc": False, "is_psf": False,
        "field_segments": [], "status": "PENDING", "priority": 0,
        "ai_upscale_banned": False, "recommended_params": {},
        "user_overrides": {"ai_stream_raw": True,
                           "force_allow_banned_upscale": True,
                           "original_fps": "30000/1001",
                           # 关掉中间产物清理：否则跑完只剩 step6，
                           # step3/step4 已被 keep_steps 回收，无从对照。
                           "cleanup_intermediates": False,
                           # 本轮已定位色度根因（packed vs planar）并补了
                           # format=gbrp16le；显式承认"允许融合"以便实测，
                           # 否则闸门会把融合挡掉、两边都跑经典就无从对照。
                           # ---- 本次对照的关键 ----
                           # 不开这个，stage1 两次跑批的差异本底是
                           # Y~48.7 / U~51.5 / V~39.6（min 42.07），
                           # 会把"融合机制是否等价"完全淹没。
                           # 单线程后 stage1 逐像素可复现，
                           # 剩下的差异才真的归属于 stage3+4 融合。
                           "vs_deterministic": True},
        "duration": 0.7, "original_width": 1920, "original_height": 1080,
        "file_size": os.path.getsize(clip), "file_mtime": 0.0,
        "file_hash_prefix": "", "run_generation": 1,
        "temp_dir": os.path.join(work, "temp"),
        "log_path": os.path.join(work, "logs", "job.log"),
        "output_path": os.path.join(work, "out", "g34_restored.mkv"),
    }
    path = os.path.join(work, "m.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"meta": {"generated_at": "2026-11-05T14:00:00+08:00",
                                 "source_dirs": [os.path.join(work, "in")],
                                 "config_path": "", "total_files": 1,
                                 "pending": 1, "done": 0, "failed": 0,
                                 "skipped": 0, "run_generation": 1},
                        "jobs": [job]}, f, allow_unicode=True, sort_keys=False)
    return path


def run_mode(mode, tag, clip):
    work = os.path.join(BASE, tag)
    if os.path.isdir(work):
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    man = write_manifest(work, clip)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([
        os.path.join(PROJ, "tools_r4"),
        os.path.join(PROJ, "tools_r4", "gmic", "gmic-4.0.5-cli-win64"),
        os.path.join(PROJ, "tools_r4", "exiftool", "exiftool-13.59_64"),
        env.get("PATH", "")])
    r = sh([PY, os.path.join(PROJ, "video_restoration_pipeline", "main.py"),
            "--manifest", man,
            "--config", os.path.join(PROJ, "video_restoration_pipeline",
                                     "config.yaml"),
            "--mode", "batch", "--no-wsl-redirect", "--ai-enabled",
            "--ai-upscale-factor", "2", "--force-allow-banned-upscale",
            "--workers", "1", "--allow-write-root", work,
            "--chain", mode], env=env)
    steps = sorted(
        [f for f in os.listdir(os.path.join(work, "temp"))
         if f.endswith(".mkv") and "_step" in f]) \
        if os.path.isdir(os.path.join(work, "temp")) else []
    rcpts = sorted(
        [f for f in os.listdir(os.path.join(work, "temp"))
         if f.endswith(".fused_stage34.json")]) \
        if os.path.isdir(os.path.join(work, "temp")) else []
    return {"mode": mode, "tag": tag, "rc": r.returncode,
            "steps": steps, "receipts": rcpts,
            "stdout_tail": (r.stdout or "").splitlines()[-4:],
            "stderr_tail": (r.stderr or "").splitlines()[-6:],
            "work": work}


def find_step4(res):
    for name in res["steps"]:
        if "_step4_" in name:
            return os.path.join(res["work"], "temp", name)
    return None


def probe_frames(path):
    r = sh(["ffprobe", "-v", "error", "-count_frames", "-select_streams",
            "v:0", "-show_entries", "stream=nb_read_frames,width,height,pix_fmt",
            "-of", "default=nw=1", path])
    info = {}
    for line in (r.stdout or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v
    return info


def compare(a, b):
    """逐帧 PSNR：完全相同时每帧都是 inf。

    注意两件事（第一版都踩了）：
    1. `psnr=stats_file=-` 把逐帧统计写到 **stdout**，不是 stderr；
    2. 解析不到任何帧时**绝不能**当作"相同" —— 空比较被判"完全相同"
       是最危险的一类假通过。
    """
    r = sh(["ffmpeg", "-hide_banner", "-v", "info", "-i", a, "-i", b,
            "-lavfi", "psnr=stats_file=-", "-f", "null", "-"])
    per_frame = []
    for line in (r.stdout or "").splitlines():
        if line.startswith("n:") and "psnr_avg:" in line:
            d = dict(x.split(":", 1) for x in line.split() if ":" in x)
            per_frame.append((d.get("n"), d.get("psnr_avg")))
    summary = [l for l in (r.stderr or "").splitlines() if "PSNR" in l]
    return per_frame, summary


def main():
    clip = make_clip(os.path.join(BASE, "in"))
    print("  测试片: %s (%.2f MB)" % (clip, os.path.getsize(clip) / 1048576))
    print()
    print("=== 1) 经典执行（chain=none）===")
    classic = run_mode("none", "classic", clip)
    print("  rc=%s  step 产物: %s" % (classic["rc"], classic["steps"]))
    if classic["rc"] != 0:
        print("  stderr:", "\n         ".join(classic["stderr_tail"]))
    print()
    print("=== 2) 融合执行（chain=stage34）===")
    fused = run_mode("stage34", "fused", clip)
    print("  rc=%s  step 产物: %s" % (fused["rc"], fused["steps"]))
    print("  融合回执: %s" % fused["receipts"])
    if fused["rc"] != 0:
        print("  stderr:", "\n         ".join(fused["stderr_tail"]))

    out = {"classic": classic, "fused": fused}
    a4, b4 = find_step4(classic), find_step4(fused)
    print()
    print("=== 3) 对照 ===")
    print("  经典 step4: %s" % (os.path.basename(a4) if a4 else None))
    print("  融合 step4: %s" % (os.path.basename(b4) if b4 else None))
    print("  经典有 step3: %s" % any("_step3_" in s for s in classic["steps"]))
    print("  融合有 step3: %s（应为 False）"
          % any("_step3_" in s for s in fused["steps"]))
    out["classic_has_step3"] = any("_step3_" in s for s in classic["steps"])
    out["fused_has_step3"] = any("_step3_" in s for s in fused["steps"])

    if a4 and b4:
        ia, ib = probe_frames(a4), probe_frames(b4)
        print("  经典: %s" % ia)
        print("  融合: %s" % ib)
        out["classic_info"], out["fused_info"] = ia, ib
        per, summary = compare(a4, b4)
        finite = [(n, v) for n, v in per if v not in ("inf", "inf.0")]
        print("  逐帧 PSNR: 共 %d 帧，有限值 %d 帧" % (len(per), len(finite)))
        if summary:
            print("  汇总: %s" % summary[-1].strip()[:100])
        if finite:
            worst = min(float(v) for _, v in finite)
            print("  **不等价**：最差 %.2f dB，前几帧 %s"
                  % (worst, finite[:5]))
            out["verdict"] = "PIXEL_DIFF"
            out["worst_psnr_db"] = worst
            out["first_diffs"] = finite[:10]
        elif per:
            print("  **逐帧完全相同**（PSNR 全 inf）")
            out["verdict"] = "IDENTICAL"
        else:
            # 一帧都没解析到 —— 这是"没测到"，不是"相同"
            print("  **未测到逐帧数据**，不能据此判等价")
            out["verdict"] = "UNKNOWN"
        # 也记整体字节是否相同（封装决定，仅作附加信息）
        out["same_size"] = os.path.getsize(a4) == os.path.getsize(b4)
        print("  文件大小相同: %s" % out["same_size"])
    else:
        out["verdict"] = "MISSING_STEP4"

    ev = os.path.join(ROOT, "evidence")
    os.makedirs(ev, exist_ok=True)
    with open(os.path.join(ev, "G34_EQUIV.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print()
    print("  报告: %s" % os.path.join(ev, "G34_EQUIV.json"))
    print("  结论: %s" % out.get("verdict"))


if __name__ == "__main__":
    main()
