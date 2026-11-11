# -*- coding: utf-8 -*-
"""R22 §5.3 · 业务链偏色按**边界**定位（不改最终标签）。

## 审核怎么说

> 在已知参考和用户授权真实素材上，按"入口解码/解释→模型输入→AI输出→
> 后处理→RGB/YUV与采样转换→最终封装/解码"逐点保存可对照样本。
> 一次只改变一项因素，固定上游输入；记录实际命令、位深、矩阵、范围、
> primaries、transfer、采样位置和模型配置。
>
> 若从某一步开始偏差明显，先固定其前一边界，验证该转换数值和标签是否
> 一致。**只改标签看起来正常不等于修好**。AI 模型本身引起的色调变化要
> 单独记录，不能当作色彩转换的无损误差。

## 本探针怎么量

对**同一帧**（默认第 12 帧）在每个边界取该帧的**通道均值**：

    ffmpeg -i <boundary> -vf "select=eq(n\\,12),scale=1:1:flags=area,format=rgb48le"
           -frames:v 1 -f rawvideo -

`scale=1:1:flags=area` 就是整幅面积平均，所以 1×1 的那个像素 = 全帧均值
（16-bit，避免 8-bit 取整把小于 1 的差异抹平）。

然后**相邻两个边界**相减，得到"偏差从哪一步开始"。这样：
- AI 之前的边界（stage1/2）若已经在漂 → 属于传统链/解码解释问题；
- stage2→stage3 大幅跳变 → 属于模型色调（必须单独标注，不算转换误差）；
- stage4/6 之后才漂 → 属于后处理/颗粒/封装。

## 用法

    python tests/probes/r22_color_boundary.py [--run <已有运行目录>] [--frame 12]

`--run` 指向 `r22_444_classic_path.py` 产生的一个 `run_*` 目录
（复用已跑好的中间产物，不重跑 GPU）。不给就跑一次新的小样本。

**它不做断言**：这一轮的目标是"把偏差定位到某一步"，不是"判它合格"。
定论要看记录下来的数字，不能看标签。
"""
import argparse
import json
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
S3_ROOT = r"E:\codex\_media_r22_20261111T2330\s3_444"

ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUTF8": "1",
    "TEMP": r"E:\codex\_media_r22_20261111T2330\tmp",
    "TMP": r"E:\codex\_media_r22_20261111T2330\tmp",
}


def _run(cmd, timeout=300):
    e = os.environ.copy()
    e.update(ENV)
    return subprocess.run(cmd, capture_output=True, env=e, timeout=timeout,
                          shell=False)


def _probe(path):
    r = _run(["ffprobe", "-hide_banner", "-v", "error",
              "-select_streams", "v:0",
              "-show_entries",
              "stream=pix_fmt,width,height,color_space,color_range,"
              "color_primaries,color_transfer,r_frame_rate",
              "-of", "json", str(path)])
    try:
        doc = json.loads((r.stdout or b"{}").decode("utf-8", "replace"))
    except ValueError:
        return {}
    return (doc.get("streams") or [{}])[0]


def _frame_mean_rgb(path, n):
    """第 n 帧的 RGB 通道均值（16-bit 精度）。返回 (r, g, b) 或 None。

    **滤镜顺序是关键**：必须先在**原分辨率**转成 RGB，再面积平均。
    反过来（先 scale 到 1×1 再转 RGB）在 yuv420p 上会得到退化的
    色度平面 —— 实测同一帧两种顺序：

        scale=1:1,format=rgb48le  → (0, 33536, 0)        ← 垃圾
        format=rgb48le,scale=1:1  → (6957, 33391, 4903)  ← 正确

    这不是精度问题，是**测量对象被换掉了**：先缩放等于在 YUV 空间
    把 160×120 的色度压成 1×1，再对这个退化结果做矩阵转换。
    """
    vf = (f"select=eq(n\\,{int(n)}),"
          f"format=rgb48le,scale=1:1:flags=area")
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-i", str(path),
              "-vf", vf, "-frames:v", "1", "-f", "rawvideo",
              "-pix_fmt", "rgb48le", "-"])
    data = r.stdout or b""
    if r.returncode != 0 or len(data) < 6:
        return None
    r16, g16, b16 = struct.unpack("<HHH", data[:6])
    return (r16 / 65535.0 * 255.0, g16 / 65535.0 * 255.0,
            b16 / 65535.0 * 255.0)


def _find_boundaries(work):
    """按 stage 找中间产物与成品。"""
    temp = os.path.join(work, "temp")
    out = os.path.join(work, "output")
    found = []
    order = [("source", os.path.join(work, "input", "probe_src.mkv"))]
    pats = [("stage1", r"step1"), ("stage2", r"step2"), ("stage3", r"step3"),
            ("stage4", r"step4"), ("stage5", r"step5"), ("stage6", r"step6")]
    got = {}
    if os.path.isdir(temp):
        for name in sorted(os.listdir(temp)):
            if not name.endswith((".mkv", ".mp4", ".mov")):
                continue
            for stage, rx in pats:
                if stage in got:
                    continue
                import re
                if re.search(rx, name, re.I):
                    got[stage] = os.path.join(temp, name)
    for stage, _rx in pats:
        if stage in got:
            found.append((stage, got[stage]))
    if os.path.isdir(out):
        finals = sorted(n for n in os.listdir(out)
                        if n.endswith((".mkv", ".mp4", ".avif", ".mov")))
        for n in finals:
            found.append(("final:" + n, os.path.join(out, n)))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    ap.add_argument("--frame", type=int, default=12)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    work = args.run
    if not work:
        if not os.path.isdir(S3_ROOT):
            raise SystemExit("没有可复用的运行目录；先跑 "
                             "tests/probes/r22_444_classic_path.py")
        cands = sorted(os.path.join(S3_ROOT, d) for d in os.listdir(S3_ROOT))
        cands = [c for c in cands if os.path.isdir(c)]
        if not cands:
            raise SystemExit("s3_444 下没有运行目录")
        work = cands[-1]
    run_dirs = [os.path.join(work, d) for d in sorted(os.listdir(work))
                if d.startswith("run_") and
                os.path.isdir(os.path.join(work, d))] \
        if os.path.isdir(work) else []
    if not run_dirs:
        # 允许直接传 run_xxx 目录
        run_dirs = [work]

    report = {"work": work, "frame": args.frame, "runs": []}
    for rd in run_dirs:
        bounds = _find_boundaries(rd)
        if not bounds:
            continue
        rows = []
        prev = None
        for stage, path in bounds:
            meta = _probe(path)
            mean = _frame_mean_rgb(path, args.frame)
            row = {
                "boundary": stage,
                "file": os.path.basename(path),
                "pix_fmt": meta.get("pix_fmt"),
                "width": meta.get("width"),
                "height": meta.get("height"),
                "matrix": meta.get("color_space"),
                "range": meta.get("color_range"),
                "primaries": meta.get("color_primaries"),
                "transfer": meta.get("color_transfer"),
                "mean_rgb": [round(v, 4) for v in mean] if mean else None,
            }
            if mean and prev:
                row["delta_from_prev"] = [round(mean[i] - prev[i], 4)
                                          for i in range(3)]
                row["max_abs_delta"] = round(
                    max(abs(mean[i] - prev[i]) for i in range(3)), 4)
            elif mean:
                row["delta_from_prev"] = None
                row["max_abs_delta"] = 0.0
            rows.append(row)
            prev = mean
        report["runs"].append({"run": os.path.basename(rd), "rows": rows})

    # 打印成表：每行 = 一个边界 + 相对上一步的偏差
    for run in report["runs"]:
        print(f"\n=== {run['run']} （第 {args.frame} 帧）===")
        print(f"{'boundary':16} {'pix_fmt':14} {'WxH':10} {'R':>8} {'G':>8} "
              f"{'B':>8} {'Δmax':>8}  matrix/range")
        for row in run["rows"]:
            mr = row.get("mean_rgb") or [float('nan')] * 3
            d = row.get("max_abs_delta")
            print(f"{row['boundary'][:16]:16} {str(row['pix_fmt'])[:14]:14} "
                  f"{str(row['width']) + 'x' + str(row['height']):10} "
                  f"{mr[0]:8.3f} {mr[1]:8.3f} {mr[2]:8.3f} "
                  f"{('' if d is None else f'{d:8.3f}')}  "
                  f"{row.get('matrix')}/{row.get('range')}")
        # 指出最大跳变发生的位置
        worst = None
        for row in run["rows"]:
            d = row.get("max_abs_delta")
            if d is None:
                continue
            if worst is None or d > worst[1]:
                worst = (row["boundary"], d, row.get("delta_from_prev"))
        if worst:
            print(f"→ 最大跳变: {worst[0]}  Δmax={worst[1]:.3f}  "
                  f"ΔRGB={worst[2]}")

    out = args.out or os.path.join(work, "color_boundary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[color-boundary] 明细 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
