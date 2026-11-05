# -*- coding: utf-8 -*-
"""定位「VapourSynth 渲染非确定」的具体滤镜。

## 为什么查这个

整批 G34 对照（经典 vs 融合）得到 Y 48 / U 51 / V 42.6 dB，我一度归因为
融合机制不等价。但进一步对比发现 **stage1 两次跑批就已经不同**
（Y 49.40 / U 52.18 / V 40.25，min 42.16），而 step2 与 step1 逐帧相同
（直传）。所以残差来自 stage1，不是融合。

单独跑同一个 vpy 两次（60 帧 1080p）确认：**30 帧相同、30 帧不同**，
汇总 Y 48.71 / U 51.53 / V 39.63、min 42.07 —— 与 step1 对照同量级。

## 这个脚本做什么

按 stage1 的滤镜顺序**逐段累加**，每段跑两次，看差异从哪一步开始出现：

    v0  source + resize(Bicubic → YUV420P16)      ← 基线
    v1  + fvf.TemporalDegrain(sigma=16,bw=16,bh=16)
    v2  + core.deblock.Deblock
    v3  + haf.Stab(dxmax=10,dymax=10)
    v4  + resize.Bicubic(BT.601→BT.709)          ← 完整链路
    v4s 同 v4 但 core.num_threads = 1            ← 判断是否多线程竞态

判据：某段之后开始出现差异 → 该段是（第一个）非确定源。

注意
----
- 只替换**外部工具调用**，不替换被测逻辑（vpy 就是生产里生成的那一份）。
- 输出用 `-c:v ffv1` 且**不指定 pix_fmt**，让 ffmpeg 保持 y4m 的
  yuv420p16le；任何 `-pix_fmt` 都会引入额外转换，可能掩盖真实差异。
- PSNR 统计走 `stats_file=-`（stdout），**解析不到帧时判 UNKNOWN，不判相同**。
"""
import json
import os
import subprocess
import sys

PROJ = r"E:\codex\code1\old-video-and-picture-fix"
WORK = r"E:\codex\_media_r17_20261031T0004\det_probe2"
SRC = r"E:\codex\_media_r17_20261031T0004\g34cmp\in\g34.mp4"
VSPIPE = os.path.join(PROJ, "tools_r4", "vs_portable", "r72", "VSPipe.exe")

HEAD = """import vapoursynth as vs
import havsfunc as haf
import fvsfunc as fvf
core = vs.core
{nt}
clip = core.ffms2.Source(source=r'{src}')
clip = core.resize.Bicubic(clip, format=vs.YUV420P16)
"""

TAIL = """
clip.set_output()
"""

_TD = "clip = fvf.TemporalDegrain(clip, sigma=16, bw=16, bh=16)"
_DB = "clip = core.deblock.Deblock(clip)"
_ST = "clip = haf.Stab(clip, dxmax=10, dymax=10)"
_CC = ("clip = core.resize.Bicubic(clip, matrix_in_s='170m', matrix_s='709',"
       " range_in=vs.RANGE_LIMITED, range=vs.RANGE_LIMITED)")

# 累加式：每一档都在前一档之上再加一段，才能看出"差异从哪一步开始"
# 四元组：(名字, 该档累加的滤镜, num_threads=1?, OMP_NUM_THREADS=1?)
#
# 为什么测 OMP：mvtools（TemporalDegrain / Depan 都基于它）内部用 OpenMP，
# SAD 之类的累加是**浮点归约**，不满足结合律 —— 线程数/调度不同，
# 累加顺序就不同，运动向量可能差 1，补偿结果随之差极小的量。
# `core.num_threads = 1` 只限制 VapourSynth 的**帧级**并行，
# **不管**滤镜内部的 OpenMP，所以两者必须分开测。
VARIANTS = [
    # 基线：确认 source+resize 本身确定
    ("w0_base", [], False, False),
    # 完整链路、默认线程 —— 复现整批观察到的非确定
    ("w1_full_default", [_TD, _DB, _ST, _CC], False, False),
    # 完整链路、num_threads=1（**设在 filter 创建之前**）
    ("w2_full_nt1", [_TD, _DB, _ST, _CC], True, False),
    # 完整链路、num_threads=1 + OMP_NUM_THREADS=1
    ("w3_full_nt1_omp1", [_TD, _DB, _ST, _CC], True, True),
]


def build_vpy(path, body, single_thread, omp1=False):
    # `core.num_threads` 必须设在**任何 filter 创建之前**才有意义。
    # 第一版我把它写在脚本末尾（set_output 前）—— 那时 source 和 resize
    # 早就建好了，设置不生效，于是"单线程仍非确定"是个**假结论**。
    nt = "core.num_threads = 1\n" if single_thread else ""
    parts = [HEAD.format(src=SRC, nt=nt)]
    # 每条语句必须独占一行：`"".join()` 会把相邻两条拼成
    # `...bh=16)clip = core.deblock.Deblock(clip)` → SyntaxError
    parts.extend(line + "\n" for line in body)
    parts.append(TAIL)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def render(vpy, out, env=None):
    """vspipe → ffmpeg(ffv1，保持原 pix_fmt)。"""
    p1 = subprocess.Popen([VSPIPE, "-c", "y4m", vpy, "-"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env=env)
    p2 = subprocess.Popen(["ffmpeg", "-hide_banner", "-v", "error", "-y",
                           "-i", "-", "-c:v", "ffv1", out],
                          stdin=p1.stdout, stderr=subprocess.PIPE)
    p1.stdout.close()
    _, e2 = p2.communicate()
    e1 = p1.stderr.read()
    p1.wait()
    ok = (p2.returncode == 0 and p1.returncode == 0
          and os.path.exists(out) and os.path.getsize(out) > 1024)
    err = (e2 or b"").decode("utf-8", "replace")[-200:]
    err += " | " + (e1 or b"").decode("utf-8", "replace")[-300:]
    return ok, err


def compare(a, b):
    """返回 (是否完全相同, 汇总行)。解析不到帧 → None（不是"相同"）。"""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-v", "info",
                        "-i", a, "-i", b, "-lavfi", "psnr=stats_file=-",
                        "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace")
    rows = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("n:")]
    if not rows:
        return None, "未解析到逐帧数据"
    diff = 0
    for ln in rows:
        d = dict(x.split(":", 1) for x in ln.split() if ":" in x)
        if not all((d.get(k) or "").startswith("inf")
                   for k in ("psnr_y", "psnr_u", "psnr_v")):
            diff += 1
    summary = ""
    for ln in (r.stderr or "").splitlines():
        if "PSNR" in ln:
            summary = ln.split("]")[-1].strip()[:96]
    return diff == 0, ("%d/%d 帧不同 | %s" % (diff, len(rows), summary))


def main():
    os.makedirs(WORK, exist_ok=True)
    print("  工作目录: %s\n" % WORK)
    results = {}
    for name, body, single, omp1 in VARIANTS:
        vpy = os.path.join(WORK, name + ".vpy")
        build_vpy(vpy, body, single, omp1)
        env = dict(os.environ)
        if omp1:
            env["OMP_NUM_THREADS"] = "1"
        outs = []
        tags = []
        if single:
            tags.append("num_threads=1")
        if omp1:
            tags.append("OMP_NUM_THREADS=1")
        print("  [%s]%s" % (name, ("  (" + ", ".join(tags) + ")") if tags else ""))
        bad = False
        for i in (1, 2):
            out = os.path.join(WORK, "%s_run%d.mkv" % (name, i))
            ok, err = render(vpy, out, env=env)
            if not ok:
                print("    渲染失败: %s" % err)
                bad = True
                break
            outs.append(out)
        if bad:
            results[name] = "RENDER_FAILED"
            continue
        same, detail = compare(outs[0], outs[1])
        verdict = ("DETERMINISTIC" if same is True
                   else "NON_DETERMINISTIC" if same is False else "UNKNOWN")
        results[name] = {"verdict": verdict, "detail": detail}
        print("    -> %s  %s" % (verdict, detail))

    print("\n  === 汇总 ===")
    first_bad = None
    for name, _b, _s, _o in VARIANTS:
        v = results.get(name)
        if isinstance(v, dict):
            print("    %-20s %s" % (name, v["verdict"]))
            if v["verdict"] == "NON_DETERMINISTIC" and first_bad is None:
                first_bad = name
        else:
            print("    %-20s %s" % (name, v))

    print("\n  === 结论 ===")
    if first_bad == "v0_source_resize":
        print("    连 source+resize 都不确定 —— 问题在 ffms2 解码或 resize，")
        print("    不是某个滤镜。需另查（源文件读取/缓存竞态）。")
    elif first_bad:
        print("    第一个出现不确定的变体是 **%s** —— 该段新增的滤镜即根因。" % first_bad)
    else:
        print("    各段两次渲染都完全相同 —— 与整批跑批的观察矛盾，需重测。")

    with open(os.path.join(WORK, "result.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n  结果: %s" % os.path.join(WORK, "result.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
