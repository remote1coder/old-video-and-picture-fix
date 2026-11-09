# -*- coding: utf-8 -*-
"""G34 融合 vs 经典：**严格**等价性对照（第十九轮重写）。

## 审核 R18-A08 指出的问题（逐条对应）

1. **项目根定位错误**：
   > 归档到 tests/probes 后，PROJ 仍用两次 dirname，得到的是 tests 目录。
   > 从项目根启动不能改变 `__file__` 的这个结果。

   旧代码 `PROJ = dirname(dirname(abspath(__file__)))`：
   `tests/probes/x.py` → `tests/probes` → `tests`。**少了最后一级**。
   修：`Path(__file__).resolve().parents[2]`（probes → tests → repo），
   并允许 `--repo` 显式覆盖。

2. **硬编码历史测试根 / 会删既有目录**：
   > 脚本还硬编码历史测试根，并包含删除既有 work 目录的操作。

   旧代码 `ROOT = E:\\codex\\_media_r17_20261031T0004` + `shutil.rmtree(work)`。
   修：`--work` 可选，默认在 repo 的**父目录**下 `mkdtemp`（一定是新目录）；
   调用方给了已存在的非空 `--work` 就**拒绝**，绝不 rmtree。

3. **判据不充分**：
   > 探针修了零帧假通过，但没有要求全部帧比较、比较进程成功、
   > 两次流水线成功或确实执行融合。

   受控输入下旧版会误报 IDENTICAL 的情形：
   - 两个文件声明 60 帧，只比较到 1 帧 inf → IDENTICAL
   - 两条流水线均 rc=86，仍留有可比较的 step4 → IDENTICAL
   - 两例都没有融合回执、fused 仍有 step3 → 仍不阻断

   修：见 `evaluate()`，**任一条不满足就不得报 IDENTICAL**。

4. **失败也退出 0**：现在失败返回非零（1=未通过，2=参数/环境错误）。

## 用法

    python tests/probes/g34_equiv_det.py --frames 20
    python tests/probes/g34_equiv_det.py --repo E:\\..\\repo --work E:\\..\\out2 --input x.mp4

## 判据（全部满足才算 IDENTICAL）

- 两条流水线 rc 均为 0；
- classic 有 step3、fused **没有** step3（否则说明融合没真正跑）；
- fused 有 `*.fused_stage34.json` 回执；
- 两边尺寸 / pix_fmt / 帧数一致，且等于 `--frames`；
- 逐帧 PSNR 覆盖**全部**帧（部分比较不是证据）；
- y/u/v 三个平面**都有**数据；
- 所有帧都有限且为 inf（完全相同）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _repo_root() -> str:
    """tests/probes/x.py → parents[0]=probes, [1]=tests, [2]=repo。"""
    return str(Path(__file__).resolve().parents[2])


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace", shell=False, **kw)


def probe(path):
    r = sh(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries",
            "stream=nb_read_frames,width,height,pix_fmt,r_frame_rate,time_base",
            "-of", "json", path])
    try:
        s = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    except Exception:
        s = {}
    return {k: s.get(k) for k in
            ("width", "height", "pix_fmt", "nb_read_frames", "r_frame_rate",
             "time_base")}


def make_clip(dst_dir, frames):
    """造测试片。只在**自己创建的**目录里替换自产素材。"""
    os.makedirs(dst_dir, exist_ok=True)
    clip = os.path.join(dst_dir, "src.mp4")
    if os.path.exists(clip) and os.path.getsize(clip) > 0:
        info = probe(clip)
        if int(info.get("nb_read_frames") or 0) == frames:
            return clip
        os.remove(clip)
    r = sh(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "lavfi",
            "-i", "testsrc2=size=1920x1080:rate=30000/1001:duration=10",
            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
            "-frames:v", str(frames), clip])
    if r.returncode != 0:
        raise SystemExit("造片失败: " + (r.stderr or "")[-300:])
    return clip


def write_manifest(work, clip, frames):
    import yaml
    for sub in ("temp", "out", "logs"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    job = {
        "job_id": "g34", "input_path": clip,
        "detected_class": "HDV_AVCHD", "confidence": 0.98,
        "field_order": "progressive", "needs_ivtc": False, "is_psf": False,
        "field_segments": [], "status": "PENDING", "priority": 0,
        "ai_upscale_banned": False, "recommended_params": {},
        "user_overrides": {
            "ai_stream_raw": True,
            "force_allow_banned_upscale": True,
            "original_fps": "30000/1001",
            "cleanup_intermediates": False,
            # 确定性模式：不开的话 stage1 的抖动会盖住真实差异
            "vs_deterministic": True,
        },
        "duration": frames / 30.0, "original_width": 1920,
        "original_height": 1080,
        "file_size": os.path.getsize(clip), "file_mtime": 0.0,
        "file_hash_prefix": "", "run_generation": 1,
        "temp_dir": os.path.join(work, "temp"),
        "log_path": os.path.join(work, "logs", "job.log"),
        "output_path": os.path.join(work, "out", "g34_restored.mkv"),
    }
    path = os.path.join(work, "m.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"meta": {"generated_at": "2026-11-09T22:00:00+08:00",
                                 "source_dirs": [os.path.dirname(clip)],
                                 "config_path": "", "total_files": 1,
                                 "pending": 1, "done": 0, "failed": 0,
                                 "skipped": 0, "run_generation": 1},
                        "jobs": [job]}, f, allow_unicode=True, sort_keys=False)
    return path


def run_mode(repo, work, clip, frames, mode):
    """跑一次；`work` 必须是**调用方已创建好的**新目录。"""
    os.makedirs(work, exist_ok=True)
    man = write_manifest(work, clip, frames)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([
        os.path.join(repo, "tools_r4"),
        os.path.join(repo, "tools_r4", "gmic", "gmic-4.0.5-cli-win64"),
        os.path.join(repo, "tools_r4", "exiftool", "exiftool-13.59_64"),
        env.get("PATH", "")])
    py = os.path.join(repo, "tools_r4", "vs_portable", "r72", "python.exe")
    r = sh([py, os.path.join(repo, "video_restoration_pipeline", "main.py"),
            "--manifest", man,
            "--config", os.path.join(repo, "video_restoration_pipeline",
                                     "config.yaml"),
            "--mode", "batch", "--no-wsl-redirect", "--ai-enabled",
            "--ai-upscale-factor", "2", "--force-allow-banned-upscale",
            "--workers", "1", "--allow-write-root", work,
            "--chain", mode], env=env)

    td = os.path.join(work, "temp")
    names = sorted(os.listdir(td)) if os.path.isdir(td) else []
    step3 = [n for n in names if "_step3_" in n and n.endswith(".mkv")]
    step4 = [n for n in names if "_step4_" in n and n.endswith(".mkv")]
    receipts = [n for n in names if n.endswith(".fused_stage34.json")]
    return {
        "mode": mode, "rc": r.returncode, "work": work,
        "step3": os.path.join(td, step3[-1]) if step3 else None,
        "step4": os.path.join(td, step4[-1]) if step4 else None,
        "receipts": receipts,
        "stdout_tail": (r.stdout or "").splitlines()[-4:],
        "stderr_tail": (r.stderr or "").splitlines()[-6:],
    }


def compare(a, b):
    """逐帧 PSNR。`frames` 是**实际比较到的帧数**。"""
    r = sh(["ffmpeg", "-hide_banner", "-v", "info", "-i", a, "-i", b,
            "-lavfi", "psnr=stats_file=-", "-f", "null", "-"])
    rows = []
    for ln in (r.stdout or "").splitlines():
        if ln.startswith("n:"):
            rows.append(dict(x.split(":", 1) for x in ln.split() if ":" in x))
    summary = [l.strip()[:120] for l in (r.stderr or "").splitlines()
               if "PSNR" in l]
    return {"returncode": r.returncode, "frames": len(rows),
            "rows": rows, "summary": summary,
            "stderr_tail": (r.stderr or "").splitlines()[-3:]}


def _plane(rows, key):
    vals = []
    for d in rows:
        v = d.get(key)
        if v is not None:
            vals.append(None if v.startswith("inf") else float(v))
    return {
        "present": bool(vals),
        "n": len(vals),
        "n_inf": sum(1 for x in vals if x is None),
        "min": min((x for x in vals if x is not None), default=None),
    }


def evaluate(classic, fused, info_c, info_f, cmp_result, frames):
    """严格判定。**任一条不满足就不能报 IDENTICAL。**

    Returns:
        ``(verdict, reasons)``
    """
    reasons = []

    # ---- 1) 两条流水线都必须成功 ----
    if classic["rc"] != 0:
        reasons.append(f"classic pipeline rc={classic['rc']} (must be 0)")
    if fused["rc"] != 0:
        reasons.append(f"fused pipeline rc={fused['rc']} (must be 0)")

    # ---- 2) 必须**确实执行了**融合 ----
    if classic["step3"] is None:
        reasons.append("classic produced no step3 (unexpected)")
    if fused["step3"] is not None:
        reasons.append("fused produced a step3 — fusion did not actually run")
    if not fused["receipts"]:
        reasons.append("fused produced no *.fused_stage34.json receipt")
    if classic["step4"] is None or fused["step4"] is None:
        reasons.append("missing step4 on one side")

    # ---- 3) 帧数：两边相等且等于请求值 ----
    try:
        n_c = int(info_c.get("nb_read_frames") or 0)
        n_f = int(info_f.get("nb_read_frames") or 0)
    except (TypeError, ValueError):
        n_c = n_f = 0
    if n_c != n_f:
        reasons.append(f"frame count differs: classic={n_c} fused={n_f}")
    if frames and n_c != frames:
        reasons.append(f"classic frame count {n_c} != requested {frames}")

    # ---- 4) 尺寸与像素格式 ----
    for k in ("width", "height", "pix_fmt"):
        if info_c.get(k) != info_f.get(k):
            reasons.append(f"{k} differs: {info_c.get(k)} vs {info_f.get(k)}")

    # ---- 5) 比较本身必须成功且覆盖全部帧 ----
    if cmp_result.get("returncode") != 0:
        reasons.append(f"psnr comparison rc={cmp_result.get('returncode')}")
    got = cmp_result.get("frames", 0)
    if got <= 0:
        reasons.append("psnr comparison produced no frames")
    elif n_c and got != n_c:
        reasons.append(
            f"psnr covered {got} of {n_c} frames — partial comparison "
            f"is not evidence")

    # ---- 6) 三个平面都要有数据，且全部 inf ----
    planes = {k: _plane(cmp_result.get("rows") or [], f"psnr_{k}")
              for k in ("y", "u", "v")}
    missing = [k for k, v in planes.items() if not v["present"]]
    if missing:
        reasons.append(f"planes missing from comparison: {missing}")
    else:
        finite = {k: v for k, v in planes.items() if v["n_inf"] != v["n"]}
        if finite:
            detail = ", ".join(
                f"{k}:{v['n'] - v['n_inf']} finite (min {v['min']})"
                for k, v in finite.items())
            reasons.append(f"frames are not identical — {detail}")

    if reasons:
        return "NOT_IDENTICAL", reasons
    return "IDENTICAL", []


def main():
    ap = argparse.ArgumentParser(description="G34 vs classic strict equivalence")
    ap.add_argument("--repo", default=None, help="项目根（默认按 __file__ 推断）")
    ap.add_argument("--work", default=None,
                    help="输出根（默认在 repo 父目录下 mkdtemp，绝不覆盖已有目录）")
    ap.add_argument("--input", default=None, help="用指定素材（跳过造片）")
    ap.add_argument("--frames", type=int, default=20)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo or _repo_root())
    if not os.path.isfile(os.path.join(repo, "video_restoration_pipeline",
                                       "main.py")):
        print("  [错误] --repo 下找不到 video_restoration_pipeline/main.py：%s"
              % repo)
        return 2

    created_here = False
    if args.work:
        work_root = os.path.abspath(args.work)
        if os.path.isdir(work_root) and os.listdir(work_root):
            print("  [错误] --work 目录已存在且非空，拒绝覆盖：%s" % work_root)
            return 2
        os.makedirs(work_root, exist_ok=True)
    else:
        parent = os.path.dirname(repo) or tempfile.gettempdir()
        work_root = tempfile.mkdtemp(prefix="r19_g34_", dir=parent)
        created_here = True

    print("  repo    : %s" % repo)
    print("  work    : %s%s" % (work_root,
                                "  (本轮新建)" if created_here else ""))
    print("  frames  : %d" % args.frames)

    if args.input:
        clip = os.path.abspath(args.input)
        if not os.path.isfile(clip):
            print("  [错误] --input 不存在：%s" % clip)
            return 2
        print("  input   : %s（外部素材）" % clip)
    else:
        clip = make_clip(os.path.join(work_root, "in"), args.frames)
        print("  input   : %s（本轮生成）" % clip)

    classic = run_mode(repo, os.path.join(work_root, "classic"), clip,
                       args.frames, "none")
    fused = run_mode(repo, os.path.join(work_root, "fused"), clip,
                     args.frames, "stage34")
    print("  classic rc=%d step3=%s" % (classic["rc"],
                                        bool(classic["step3"])))
    print("  fused   rc=%d step3=%s receipts=%s"
          % (fused["rc"], bool(fused["step3"]), fused["receipts"]))

    info_c = info_f = {}
    cmp_result = {"frames": 0, "rows": [], "returncode": None}
    if classic["step4"] and fused["step4"]:
        info_c = probe(classic["step4"])
        info_f = probe(fused["step4"])
        cmp_result = compare(classic["step4"], fused["step4"])
        verdict, reasons = evaluate(classic, fused, info_c, info_f,
                                    cmp_result, args.frames)
    else:
        verdict = "NOT_IDENTICAL"
        reasons = ["missing step4 on one side — nothing to compare"]

    print("\n  === 判定 ===")
    print("    classic step4: %s" % (classic["step4"] or "(无)"))
    print("    fused   step4: %s" % (fused["step4"] or "(无)"))
    print("    classic: %sx%s %s frames=%s"
          % (info_c.get("width"), info_c.get("height"),
             info_c.get("pix_fmt"), info_c.get("nb_read_frames")))
    print("    fused  : %sx%s %s frames=%s"
          % (info_f.get("width"), info_f.get("height"),
             info_f.get("pix_fmt"), info_f.get("nb_read_frames")))
    print("    psnr   : %d 帧" % cmp_result.get("frames", 0))
    print("    verdict = %s" % verdict)
    if reasons:
        print("    未通过原因：")
        for r in reasons:
            print("      - %s" % r)
    elif cmp_result.get("summary"):
        print("    %s" % cmp_result["summary"][-1])

    report = {
        "repo": repo, "work": work_root, "input": clip,
        "frames_requested": args.frames,
        "classic": {k: v for k, v in classic.items() if k != "stdout_tail"},
        "fused": {k: v for k, v in fused.items() if k != "stdout_tail"},
        "info_classic": info_c, "info_fused": info_f,
        "comparison": {"frames": cmp_result.get("frames"),
                       "returncode": cmp_result.get("returncode"),
                       "summary": cmp_result.get("summary")},
        "verdict": verdict, "reasons": reasons,
        "argv": sys.argv,
    }
    out = os.path.join(work_root, "result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("    报告: %s" % out)

    return 0 if verdict == "IDENTICAL" else 1


if __name__ == "__main__":
    sys.exit(main())
