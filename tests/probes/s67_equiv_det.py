# -*- coding: utf-8 -*-
"""S6→S7 执行组 vs 经典两阶段：**逐像素**等价性对照。

## 为什么必须做这个对照

融合路线唯一的合法性来源是这句话：

> 省掉的只是那份 lossless 中间文件，**像素一个都没变**。

这句话不能靠"看着差不多"成立。它的成立条件是两条路线跑的是
**同一个 vpy** + **同一段 ffmpeg 参数**，任何一边漂移（多一个 resize、
少一个 `-pix_fmt`、色彩参数没传过去）都不会体现在退出码上。

## 两条路线

    classic: stage6 → step6_grain.mkv（FFV1/16bit） → stage7 读盘编码
    fused  : stage6 → grain.vpy + 交接单         → stage7 vspipe 直连编码

## 判据（**全部**满足才允许报 IDENTICAL）

审核 R18-A08 对 G34 探针的批评在这里同样适用 —— 旧探针"只比 1 帧""两条
流水线都失败""fused 其实没融合"都能报 IDENTICAL。所以：

1. stage6 / stage7 两侧 rc 均为 0；
2. classic **确实**产出了 step6 媒体；fused **确实没有**（否则没融合）；
3. fused 有交接单 + 组回执，且回执 `intermediate_written` 为 False；
4. 两边帧数相同且等于 `--frames`；
5. 逐帧 PSNR 覆盖**全部**帧（部分比较不是证据）；
6. y/u/v 三个平面都有数据，且每帧都**有限且为 inf**（完全相同）。

失败返回非零（1=未通过，2=参数/环境错误）。

## 用法

    python tests/probes/s67_equiv_det.py --frames 24
    python tests/probes/s67_equiv_det.py --repo E:\\repo --work E:\\out

## 为什么要 `vs_deterministic: true` + `grain_seed`

`GrainFactory3` 默认每次运行随机（`seed=-1`）。不固定的情况下两次运行
**本来就应该不同**，那种比较没有意义。所以本探针强制：
`vs_deterministic: true`（VapourSynth 单线程，见第十八轮实测）
+ `grain_seed: 20261114`。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))          # 让 strict_compare 可导入
import strict_compare as SC                    # noqa: E402

SEED = 20261114


def _repo_root() -> str:
    """tests/probes/x.py → parents[0]=probes, [1]=tests, [2]=repo。"""
    return str(Path(__file__).resolve().parents[2])


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace", shell=False, **kw)


def vs_env(repo: str) -> dict:
    """把项目自带的 VapourSynth 目录放到 PATH 最前面。

    生产脚本 `run_video_real.sh` 也是这么做的；本探针直接调 stage CLI，
    没有那层包装，缺了这一步 `vspipe` 就是"找不到命令"。
    """
    env = dict(os.environ)
    cand = [os.path.join(repo, "tools_r4", "vs_portable", "r72"),
            os.path.join(os.path.dirname(repo), "tools_r4", "vs_portable",
                         "r72")]
    for c in cand:
        if os.path.isdir(c):
            env["PATH"] = c + os.pathsep + env.get("PATH", "")
            return env
    return env


def make_clip(dst_dir: str, frames: int):
    """造一段**有运动**的测试片（静态画面无法暴露帧序/时间轴问题）。"""
    os.makedirs(dst_dir, exist_ok=True)
    clip = os.path.join(dst_dir, "src.mp4")
    r = sh(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "lavfi",
            "-i", "testsrc2=size=480x270:rate=25:duration=30",
            "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
            "-frames:v", str(frames), clip])
    if r.returncode != 0:
        raise SystemExit("造片失败: " + (r.stderr or "")[-300:])
    return clip


def write_config(path: str) -> str:
    """stage6 需要的 `grain_factory3` 配置（颗粒参数只在 config 里解析）。"""
    text = (
        "grain_factory3:\n"
        "  enabled: true\n"
        "  analog_tape:\n"
        "    g1str: 1.0\n"
        "    g2str: 0.2\n"
        "    g3str: 0.2\n"
        "    temp_avg: 0.0\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def base_params(temp_dir: str, log_dir: str, output_path: str,
                clip: str, attempt: str, *, fused: dict | None = None,
                upstream: str | None = None) -> dict:
    p = {
        "temp_dir": temp_dir,
        "log_dir": log_dir,
        "output_path": output_path,
        "input_path": clip,
        "job_id": "jobS67",
        "attempt_id": attempt,
        "chain_mode": "stage67",
        "ai_enabled": True,
        "face_restore": "gfpgan",
        "vs_bit_depth": 16,
        "pix_fmt": "yuv420p10le",
        "color_range": "limited",
        "video_encoder": "libsvtav1",
        "crf": 30,
        "preset": 8,
        "svt_tile_rows": 1,
        "svt_tile_cols": 1,
        # 确定性：单线程渲染 + 固定颗粒种子（见模块文档）
        "vs_deterministic": True,
        "grain_seed": SEED,
        "grain_factory3": {"enabled": True},
        "stage_inactivity_timeout": 600.0,
        "full_chain_verify": False,
        "_frozen_param_digest": "s67probe",
    }
    if upstream:
        p["upstream_artifact"] = upstream
    if fused is not None:
        p["_fused_stage67"] = fused
    return p


def run_stage(repo: str, script: str, params: dict, params_path: str,
              config: str, clip: str, env: dict, timeout: int = 900):
    with open(params_path, "w", encoding="utf-8") as fh:
        json.dump(params, fh, ensure_ascii=False, indent=2)
    cmd = [sys.executable,
           os.path.join(repo, "video_restoration_pipeline", "stages", script),
           "--job-id", "jobS67", "--input", clip,
           "--params-file", params_path, "--config", config]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       errors="replace", shell=False, env=env,
                       timeout=timeout)
    return {"rc": r.returncode, "stdout": r.stdout or "",
            "stderr": (r.stderr or "")[-2000:]}


def find_step6(temp_dir: str):
    hits = sorted(glob.glob(os.path.join(temp_dir, "jobS67_*_step6_grain*.mkv")))
    return hits[0] if hits else ""


def evaluate(classic: dict, fused: dict, cmp_res: dict, frames: int,
             paths: dict) -> tuple[str, list]:
    """**全部**判据都满足才允许说 IDENTICAL。"""
    bad = []
    if classic["s6"]["rc"] != 0:
        bad.append(f"classic stage6 rc={classic['s6']['rc']}")
    if classic["s7"]["rc"] != 0:
        bad.append(f"classic stage7 rc={classic['s7']['rc']}")
    if fused["s6"]["rc"] != 0:
        bad.append(f"fused stage6 rc={fused['s6']['rc']}")
    if fused["s7"]["rc"] != 0:
        bad.append(f"fused stage7 rc={fused['s7']['rc']}")

    # 融合的**定义**就是"没有中间文件"。这一条不成立，后面比出来的
    # "相同"证明的是经典路线自己和自己相同。
    if not paths.get("classic_step6"):
        bad.append("classic did not produce a step6 intermediate")
    if paths.get("fused_step6"):
        bad.append("fused produced a step6 intermediate — it did not fuse")
    if not paths.get("fused_spec"):
        bad.append("fused left no hand-off spec")
    if not paths.get("fused_receipt"):
        bad.append("fused left no group receipt")
    else:
        try:
            with open(paths["fused_receipt"], "r", encoding="utf-8") as fh:
                rc = json.load(fh)
            if (rc.get("group_evidence") or {}).get("intermediate_written") \
                    is not False:
                bad.append("group receipt does not deny writing an intermediate")
            if list(rc.get("stages") or []) != [6, 7]:
                bad.append(f"group receipt stages={rc.get('stages')!r}")
        except (OSError, ValueError) as exc:
            bad.append(f"group receipt unreadable: {exc}")

    if not cmp_res.get("ok"):
        bad.extend([f"comparison: {x}" for x in
                    (cmp_res.get("problems") or [])[:4]])
    fr = cmp_res.get("frames") or {}
    if int(fr.get("compared") or 0) != int(frames):
        # 部分比较不是证据：漏一帧就可能正好漏掉"错位一位"
        bad.append(f"compared {fr.get('compared')} frames, expected {frames}")
    if not (cmp_res.get("psnr") or {}).get("all_inf"):
        bad.append("MSE is not zero; PSNR="
                   f"{(cmp_res.get('psnr') or {}).get('db')}")
    if not cmp_res.get("identical"):
        bad.append("strict_compare refuses to call the pair identical "
                   f"(max_abs_diff={cmp_res.get('max_abs_diff')}, "
                   f"differing={cmp_res.get('differing_frames')})")
    return ("IDENTICAL" if not bad else "NOT_IDENTICAL"), bad


def negative_control(repo: str, work: str, config: str, clip: str,
                     env: dict) -> dict:
    """**反向对照**：vpy 坏掉时 stage7 必须失败，不能照样出片。

    为什么必须有这一条：融合路线的最大风险不是"结果不同"，而是
    "管道根本没接上却照样 rc=0" —— 例如 vspipe 立刻退出、ffmpeg 读到
    0 帧仍然成功收尾。没有反向对照，"IDENTICAL"可能只是"两条都没干活"。

    做法：把 spec 指向一个**语法上能载入但滤镜不存在**的 vpy，
    跑一次 stage7，要求 rc != 0 且没有产出成品。
    """
    bad_dir = os.path.join(work, "negative")
    os.makedirs(bad_dir, exist_ok=True)
    bad_vpy = os.path.join(bad_dir, "bad.vpy")
    with open(bad_vpy, "w", encoding="utf-8") as fh:
        fh.write("import vapoursynth as vs\n"
                 "core = vs.core\n"
                 "clip = core.this_plugin_does_not_exist.NoSuchFilter()\n"
                 "clip.set_output()\n")
    spec_p = os.path.join(bad_dir, "spec.json")
    spec = {
        "schema": "stage67_group/1", "group": "stage67", "stages": [6, 7],
        "attempt_id": "s67neg", "chain_mode": "stage67",
        "vs_script": bad_vpy,
        "vs_script_sha256": _sha256(bad_vpy),
        "probe_source": clip,
        "grain": {"g1str": 1.0}, "vs_format": "vs.YUV420P16",
        "pix_fmt": "yuv420p16le", "deterministic": True,
    }
    with open(spec_p, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=2)
    out = os.path.join(bad_dir, "must_not_exist.mkv")
    p = base_params(bad_dir, os.path.join(work, "logs"), out, clip, "s67neg",
                    upstream=clip,
                    fused={"role": "consumer", "spec_path": spec_p})
    r = run_stage(repo, "stage7_encode.py", p,
                  os.path.join(bad_dir, "params.json"), config, clip, env)
    produced = os.path.exists(out) and os.path.getsize(out) > 0
    ok = (r["rc"] != 0) and (not produced)
    print("  negative: s7 rc=%d produced=%s -> %s"
          % (r["rc"], produced, "OK（拒绝出片）" if ok else "**未被拦住**"))
    return {"rc": r["rc"], "produced": produced, "ok": ok,
            "stderr": r["stderr"][-800:]}


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="S6→S7 vs classic equivalence")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--work", default=None,
                    help="输出根（已存在且非空则拒绝，绝不覆盖）")
    ap.add_argument("--frames", type=int, default=24)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo or _repo_root())
    if not os.path.isfile(os.path.join(repo, "video_restoration_pipeline",
                                       "stages", "stage7_encode.py")):
        print("  [错误] --repo 下找不到 video_restoration_pipeline：%s" % repo)
        return 2

    if args.work:
        work = os.path.abspath(args.work)
        if os.path.isdir(work) and os.listdir(work):
            print("  [错误] --work 已存在且非空，拒绝覆盖：%s" % work)
            return 2
        os.makedirs(work, exist_ok=True)
    else:
        work = tempfile.mkdtemp(prefix="s67_equiv_",
                                dir=os.path.dirname(repo) or None)

    env = vs_env(repo)
    config = write_config(os.path.join(work, "config.yaml"))
    clip = make_clip(os.path.join(work, "in"), args.frames)
    print("  repo    : %s" % repo)
    print("  work    : %s" % work)
    print("  input   : %s (%d 帧)" % (clip, args.frames))

    paths: dict = {}
    result = {"repo": repo, "work": work, "input": clip,
              "frames_requested": args.frames, "argv": sys.argv}

    # ---------------- classic ----------------
    ct = os.path.join(work, "classic_temp")
    os.makedirs(ct, exist_ok=True)
    c_out = os.path.join(work, "classic_out", "final.mkv")
    os.makedirs(os.path.dirname(c_out), exist_ok=True)
    pc = base_params(ct, os.path.join(work, "logs"), c_out, clip, "s67c",
                     upstream=clip)
    s6c = run_stage(repo, "stage6_grain_synthesis.py", pc,
                    os.path.join(work, "p_classic6.json"), config, clip, env)
    paths["classic_step6"] = find_step6(ct)
    pc7 = base_params(ct, os.path.join(work, "logs"), c_out, clip, "s67c",
                      upstream=paths["classic_step6"] or clip)
    s7c = run_stage(repo, "stage7_encode.py", pc7,
                    os.path.join(work, "p_classic7.json"), config, clip, env)
    print("  classic : s6 rc=%d (step6=%s) s7 rc=%d"
          % (s6c["rc"], bool(paths["classic_step6"]), s7c["rc"]))

    # ---------------- fused ----------------
    ft = os.path.join(work, "fused_temp")
    os.makedirs(ft, exist_ok=True)
    f_out = os.path.join(work, "fused_out", "final.mkv")
    os.makedirs(os.path.dirname(f_out), exist_ok=True)
    spec = os.path.join(ft, "jobS67_src_step67_group.json")
    pf6 = base_params(ft, os.path.join(work, "logs"), f_out, clip, "s67f",
                      upstream=clip,
                      fused={"role": "producer", "emit_only": True,
                             "spec_path": spec})
    s6f = run_stage(repo, "stage6_grain_synthesis.py", pf6,
                    os.path.join(work, "p_fused6.json"), config, clip, env)
    paths["fused_step6"] = find_step6(ft)
    paths["fused_spec"] = spec if os.path.exists(spec) else ""
    pf7 = base_params(ft, os.path.join(work, "logs"), f_out, clip, "s67f",
                      upstream=clip,
                      fused={"role": "consumer", "spec_path": spec})
    s7f = run_stage(repo, "stage7_encode.py", pf7,
                    os.path.join(work, "p_fused7.json"), config, clip, env)
    paths["fused_receipt"] = (f_out + ".fused_stage67.json"
                              if os.path.exists(f_out + ".fused_stage67.json")
                              else "")
    print("  fused   : s6 rc=%d (vpy+spec=%s) s7 rc=%d (receipt=%s)"
          % (s6f["rc"], bool(paths["fused_spec"]), s7f["rc"],
             bool(paths["fused_receipt"])))

    # ---------------- 比较 ----------------
    cmp_res = {}
    if os.path.exists(c_out) and os.path.exists(f_out):
        cmp_res = SC.compare_media(
            c_out, f_out, pix_fmt="yuv420p10le",
            expect_frames=args.frames, require_time=True,
            require_audio=False)
    else:
        cmp_res = {"ok": False, "problems": ["one side produced no output"],
                   "frames": {}, "identical": False}

    verdict, reasons = evaluate(
        {"s6": s6c, "s7": s7c}, {"s6": s6f, "s7": s7f}, cmp_res,
        args.frames, paths)

    neg = negative_control(repo, work, config, clip, env)
    if not neg["ok"]:
        reasons.append("negative control was NOT blocked: a broken vpy still "
                       f"produced output (rc={neg['rc']})")
        verdict = "NOT_IDENTICAL"

    print("\n  === 判定 ===")
    print("    classic: %s" % (c_out if os.path.exists(c_out) else "(无)"))
    print("    fused  : %s" % (f_out if os.path.exists(f_out) else "(无)"))
    print("    比较帧数: %s" % (cmp_res.get("frames") or {}).get("compared"))
    print("    最小 PSNR: %s" % (cmp_res.get("frames") or {}).get("min_psnr"))
    print("    verdict = %s" % verdict)
    for r in reasons:
        print("      - %s" % r)

    result.update({
        "classic": {"s6": s6c, "s7": s7c},
        "fused": {"s6": s6f, "s7": s7f},
        "paths": paths,
        "negative_control": neg,
        "comparison": {k: v for k, v in cmp_res.items() if k != "hashes"},
        "verdict": verdict, "reasons": reasons,
    })
    out = os.path.join(work, "result.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(SC.json_safe(result), fh, ensure_ascii=False, indent=2)
    print("    报告: %s" % out)
    return 0 if verdict == "IDENTICAL" else 1


if __name__ == "__main__":
    sys.exit(main())
