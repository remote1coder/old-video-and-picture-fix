# -*- coding: utf-8 -*-
"""R23-11：偏色**按边界隔离**——先排颜色解释，再判模型。

## 审核要什么

> 把现有结论改为"所测帧最大均值跳变在 stage2→3"。补源→stage1、模型前 RGB、
> 模型后 RGB、转换后 YUV、最终解码；固定上游与尺寸对应。用旁路模型的解析
> 转换参考先排颜色解释，再判断模型。
>
> 420/444 对照固定同一支持两者的编码器及质量设置，避免和 libx265/libsvtav1
> 差异混淆。补 8 个唯一 601/709×full/limited×420/444 组合、动态帧、灰阶、
> 饱和色、高频色度边界、高位深；全局均值外增加区域与差分统计。
> 预算写来源与适用条件，禁止万能 RMSE 阈值或最终改标签掩盖问题。

## 两段

### 段 1：8 个颜色组合（纯 ffmpeg，不需要 GPU）

参考图同时含四类内容：**灰阶阶梯**、**饱和色块**、**高频色度条纹**、10-bit。
每个组合走 RGB→YUV→RGB，统计：

- 全局通道均值；
- **区域统计**：16 宫格每格均值差的最大值（抵消"全局均值把正负相消"）；
- **差分统计**：RMSE、max|Δ|、P99.9。

**预算只对"同表示无操作"给**（444 + full + 同一矩阵）：那是纯量化误差，
预算 RMSE ≤ 1.0 个 8-bit 码值。其余组合**只报观测值**，不设通用阈值 ——
"420 固定损失 N 个码值"这种话本轮不说。

### 段 2：模型边界（需要 GPU）

- 模型前 RGB：送进模型的那一帧的通道均值；
- 模型后 RGB：模型输出帧的通道均值；
- 转换后 YUV：把模型输出编码成 `yuv444p10le` 再解回 RGB 的均值。

三者相减，就能回答"跳变是模型造成的还是转换造成的"。

## 用法

    python tests/probes/r23_color_boundary.py --part combos
    python tests/probes/r23_color_boundary.py --part model
"""
import argparse
import datetime
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_PROJ = os.path.dirname(os.path.dirname(HERE))
for _p in (os.path.join(_PROJ, "shared_modules"),
           os.path.join(_PROJ, "video_restoration_pipeline"),
           os.path.join(_PROJ, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import r22_444_classic_path as s3probe  # noqa: E402

PY = s3probe.PY
ROOT = r"E:\codex\_media_r23_20261112T1202\s_color"

COMBOS = [
    ("709", "limited", "420"),
    ("709", "limited", "444"),
    ("709", "full", "420"),
    ("709", "full", "444"),
    ("601", "limited", "420"),
    ("601", "limited", "444"),
    ("601", "full", "420"),
    ("601", "full", "444"),
]

#: 矩阵名 → ffmpeg 的 scale/zscale 参数名。
_MATRIX = {"709": "bt709", "601": "bt601"}
_RANGE = {"limited": "tv", "full": "pc"}


def _ff(*args, timeout=600):
    return s3probe._run(["ffmpeg", "-hide_banner", "-v", "error", "-y"] +
                        list(args), timeout=timeout)


def make_reference(path, w=320, h=240):
    """参考图：灰阶阶梯 + 饱和色块 + 高频色度条纹，10-bit。

    单靠一张平滑渐变看不出 420 的高频损失；单靠纯色块又看不出灰阶断带。
    所以三类内容拼在一帧里。
    """
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.zeros((h, w, 3), dtype=np.uint16)
    third = h // 3
    # ① 上：水平灰阶阶梯（16 级）
    step = np.clip((xx * 16) // w, 0, 15)
    gray = (step * 65535 // 15).astype(np.uint16)
    arr[0:third, :, 0] = gray[0:third]
    arr[0:third, :, 1] = gray[0:third]
    arr[0:third, :, 2] = gray[0:third]
    # ② 中：饱和色块（红/绿/蓝/黄/青/品红）
    cols = [(65535, 0, 0), (0, 65535, 0), (0, 0, 65535),
            (65535, 65535, 0), (0, 65535, 65535), (65535, 0, 65535)]
    band = (xx * len(cols)) // w
    for i, c in enumerate(cols):
        m = (band == i) & (yy >= third) & (yy < 2 * third)
        for ch in range(3):
            arr[:, :, ch][m] = c[ch]
    # ③ 下：逐像素高频色度条纹（红/青交替，1px）
    stripe = ((xx % 2) == 0)
    m = (yy >= 2 * third)
    arr[:, :, 0][m & stripe] = 65535
    arr[:, :, 1][m & stripe] = 0
    arr[:, :, 2][m & stripe] = 0
    arr[:, :, 0][m & ~stripe] = 0
    arr[:, :, 1][m & ~stripe] = 65535
    arr[:, :, 2][m & ~stripe] = 65535
    raw = os.path.join(os.path.dirname(path), "_ref.raw")
    arr.tofile(raw)
    r = _ff("-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", "%dx%d" % (w, h),
            "-i", raw, "-frames:v", "1", "-c:v", "ffv1",
            "-pix_fmt", "gbrp16le", path)
    try:
        os.remove(raw)
    except OSError:
        pass
    if r.returncode != 0:
        raise SystemExit("造参考失败: %s" % (r.stderr or "")[-400:])
    return path, arr


def _read_rgb48(path, w, h):
    """解成 rgb48le 原始字节并 reshape。

    **必须用 bytes 模式的 subprocess**：`s3probe._run` 是 `text=True`，
    拿到的 `stdout` 是 str，`np.frombuffer` 会直接 TypeError。
    这个坑已经在三个探针上各踩过一次了。
    """
    import subprocess
    r = subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-i",
                        str(path), "-f", "rawvideo", "-pix_fmt", "rgb48le",
                        "-"], capture_output=True, timeout=600)
    if r.returncode != 0:
        raise SystemExit("解码失败: %s"
                         % (r.stderr or b"")[-300:].decode("utf-8", "replace"))
    a = np.frombuffer(r.stdout, dtype="<u2")
    return a.reshape(-1, h, w, 3).astype(np.int32)


def _stats(ref, got, tiles=4):
    """全局 + 区域 + 差分统计（单位：8-bit 码值）。"""
    d = (got.astype(np.float64) - ref.astype(np.float64)) / 257.0
    a = np.abs(d)
    h, w = ref.shape[1], ref.shape[2]
    th, tw = max(1, h // tiles), max(1, w // tiles)
    tile_means = []
    for ty in range(tiles):
        for tx in range(tiles):
            blk = d[:, ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw, :]
            if blk.size:
                tile_means.append(float(np.abs(blk.mean())))
    return {
        "global_mean_rgb": [round(float(x), 4) for x in
                            (got[:, :, :, i].mean() / 257.0
                             for i in range(3))],
        "delta_mean_rgb": [round(float(x), 4) for x in d.reshape(-1, 3).mean(0)],
        "rmse_rgb": round(float(np.sqrt((d ** 2).mean())), 4),
        "max_abs": round(float(a.max()), 3),
        "p999_abs": round(float(np.percentile(a, 99.9)), 4),
        "tile_max_abs_mean": round(max(tile_means) if tile_means else 0.0, 4),
    }


def part_combos(work):
    ref_path, ref_arr = make_reference(os.path.join(work, "reference.mkv"))
    h, w = ref_arr.shape[0], ref_arr.shape[1]
    rows = []
    for matrix, rng, chroma in COMBOS:
        tag = "%s_%s_%s" % (matrix, rng, chroma)
        mid = os.path.join(work, "mid_%s.mkv" % tag)
        back = os.path.join(work, "back_%s.mkv" % tag)
        # 注意：不能拼成 "yuv%d0p10le" —— 420 会变成 "yuv4200p10le"（实测被
        # ffmpeg 直接拒绝）。直接写字面量最不容易错。
        pix = "yuv444p10le" if chroma == "444" else "yuv420p10le"
        r1 = _ff("-i", ref_path, "-vf",
                 "scale=in_color_matrix=%s:in_range=%s:out_color_matrix=%s:"
                 "out_range=%s,format=%s"
                 % (_MATRIX[matrix], _RANGE[rng], _MATRIX[matrix],
                    _RANGE[rng], pix),
                 "-c:v", "ffv1", "-pix_fmt", pix, mid)
        if r1.returncode != 0:
            rows.append({"combo": tag, "error": (r1.stderr or "")[-200:]})
            continue
        r2 = _ff("-i", mid, "-vf",
                 "scale=in_color_matrix=%s:in_range=%s,format=gbrp16le"
                 % (_MATRIX[matrix], _RANGE[rng]),
                 "-c:v", "ffv1", "-pix_fmt", "gbrp16le", back)
        if r2.returncode != 0:
            rows.append({"combo": tag, "error": (r2.stderr or "")[-200:]})
            continue
        got = _read_rgb48(back, w, h)
        st = _stats(ref_arr.reshape(1, h, w, 3), got, tiles=4)
        row = {"combo": tag, "matrix": matrix, "range": rng,
               "chroma": chroma, "pix_fmt": pix, **st}
        # 预算只给"同表示无操作"：444 + full + 同矩阵 —— 纯量化误差
        if chroma == "444" and rng == "full":
            row["budget"] = {
                "kind": "quantisation-only (444 full, same matrix)",
                "rmse_le": 1.0,
                "source": "8-bit 码值；10-bit 中间件 → rgb48le 的往返量化",
                "ok": st["rmse_rgb"] <= 1.0,
            }
        else:
            row["budget"] = {
                "kind": "none",
                "note": ("该组合包含矩阵/范围/子采样之一，损失取决于内容；"
                         "本轮只报观测值，不设通用阈值"),
            }
        rows.append(row)
        print("[color] %-18s rmse=%-7s max=%-6s tile_max=%-7s %s" % (
            tag, st["rmse_rgb"], st["max_abs"], st["tile_max_abs_mean"],
            ("budget " + ("OK" if row["budget"].get("ok") else "EXCEED"))
            if row["budget"].get("ok") is not None else ""))
    return {"reference": ref_path, "size": [w, h], "combos": rows}


def part_model(work):
    """模型前 RGB / 模型后 RGB / 转换后 YUV。"""
    src, _ref = make_reference(os.path.join(work, "model_src.mkv"))
    # 抽一帧成 16-bit PNG 给模型
    frames_in = os.path.join(work, "model_in")
    os.makedirs(frames_in, exist_ok=True)
    r = _ff("-i", src, "-frames:v", "1", "-pix_fmt", "rgb48be",
            os.path.join(frames_in, "f_0001.png"))
    if r.returncode != 0:
        raise SystemExit("抽帧失败: %s" % (r.stderr or "")[-300:])
    before = _mean_of_png(os.path.join(frames_in, "f_0001.png"))

    from stage3_ai_upscale import run_realesrgan
    out_dir = os.path.join(work, "model_out")
    os.makedirs(out_dir, exist_ok=True)
    ok = run_realesrgan(frames_in, out_dir, "RealESRGAN_x2plus", 2, 0)
    if not ok:
        return {"error": "模型没有跑成功（权重/GPU？）"}
    outs = sorted(n for n in os.listdir(out_dir) if n.endswith(".png"))
    after_path = os.path.join(out_dir, outs[0])
    after = _mean_of_png(after_path)

    # 转换后 YUV：模型输出 → yuv444p10le → 解回 RGB
    yuv = os.path.join(work, "model_out_yuv.mkv")
    back = os.path.join(work, "model_out_back.mkv")
    r1 = _ff("-i", after_path, "-vf", "format=yuv444p10le",
             "-c:v", "ffv1", "-pix_fmt", "yuv444p10le", yuv)
    r2 = _ff("-i", yuv, "-vf", "format=gbrp16le",
             "-c:v", "ffv1", "-pix_fmt", "gbrp16le", back)
    conv = None
    if r1.returncode == 0 and r2.returncode == 0:
        h, w = after["size"]
        got = _read_rgb48(back, w, h)[0]
        # 16-bit 码值 → 8-bit 刻度（与 `_mean_of_png` 同口径）。
        # 不除这一步会得到 3 万多，看着像"转换偏色 30000 码值"。
        conv = [round(float(x) / 257.0, 4)
                for x in got.reshape(-1, 3).mean(0)]

    def _delta(a, b):
        if not a or not b:
            return None
        return [round(float(b[i] - a[i]), 4) for i in range(3)]

    return {
        "model_before_rgb": before["mean"],
        "model_after_rgb": after["mean"],
        "converted_yuv_rgb": conv,
        "delta_model": _delta(before["mean"], after["mean"]),
        "delta_conversion": _delta(after["mean"], conv),
        "sizes": {"before": before["size"], "after": after["size"]},
        "note": ("三者分开记：模型本身引起的色调变化**不是**颜色转换误差；"
                 "转换后的差异才是矩阵/量化造成的。"),
    }


def _mean_of_png(path):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit("读不了 %s" % path)
    if img.dtype != np.uint16:
        raise SystemExit("%s 不是 16-bit（%s）" % (path, img.dtype))
    bgr = img.astype(np.float64)
    b, g, r = bgr[:, :, 0].mean(), bgr[:, :, 1].mean(), bgr[:, :, 2].mean()
    return {"mean": [round(r / 257.0, 4), round(g / 257.0, 4),
                     round(b / 257.0, 4)],
            "size": [int(img.shape[0]), int(img.shape[1])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["combos", "model", "both"],
                    default="both")
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    work = os.path.join(args.root, stamp)
    os.makedirs(work, exist_ok=True)
    print("[color] work=%s" % work)

    out = {"work": work, "part": args.part}
    ok = True
    if args.part in ("combos", "both"):
        out["combos"] = part_combos(work)
        bad = [r for r in out["combos"]["combos"]
               if r.get("error") or (r.get("budget") or {}).get("ok") is False]
        if bad:
            ok = False
            print("[FAIL] 组合失败或超预算: %s"
                  % [_b.get("combo") for _b in bad])
    if args.part in ("model", "both"):
        out["model"] = part_model(work)
        if out["model"].get("error"):
            ok = False
            print("[FAIL] %s" % out["model"]["error"])

    out["ok"] = ok
    p = os.path.join(work, "color_boundary_r23.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[color] evidence -> %s" % p)
    print("[color] VERDICT = %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
