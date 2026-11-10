# -*- coding: utf-8 -*-
"""N21：4:2:0 / 4:4:4 × bt601 / bt709 × full / limited 的颜色往返误差。

## 这个探针回答什么

审核 R19 §4.2：

> 不要把 420 中间格式当成已定结论。

现行中间格式固定 `yuv420p16le`。这是一次**信息有损**的降采样，而规范里
从来没写过"420 到底损失了多少、在什么条件下可以接受"。本探针给出**数字**，
让"420 现状"和"验收结论"能分开写。

## 方法（不涉及 AI、不涉及业务链路，纯 ffmpeg）

1. 造参考：`smptebars` 渲成 **rgb24** FFV1（无子采样），这就是"真值"。
2. 对每个组合：
   - 真值 RGB → YUV（指定 matrix / range）→ 16 位 FFV1 中间件（420 或 444）
   - 中间件 → RGB（**同一组** matrix / range）
   - 与真值逐帧比 PSNR（在 RGB 域，三个平面各自报）
3. 另测**标签错配**：按 A 编码、按 B 解码 —— 这是"偏色"最常见的来源，
   量出它比子采样误差大多少。

## 判据

- `444 + 同参数往返` 必须是 `inf`（无损）—— 否则说明命令本身有问题，
  整轮结论作废；
- 420 的损失用 **RGB 域 PSNR** 表达（dB），并换算出 8-bit 码值当量；
- 标签错配的损失**必须显著大于**同参数 420 的损失，否则说明"标签"在这条
  路径上不起作用，得回去查 scale 参数是否真的生效。

## 用法

    python tests/probes/color_matrix_probe.py
    python tests/probes/color_matrix_probe.py --work E:\\..\\n21 --size 1280x720
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

#: 有损组合：chroma 子采样 × 矩阵 × 量化范围
COMBOS = [
    ("444", "bt709", "limited"),
    ("444", "bt709", "full"),
    ("444", "bt601", "limited"),
    ("420", "bt709", "limited"),
    ("420", "bt709", "full"),
    ("420", "bt601", "limited"),
]

#: 标签错配：按 (编码) 写、按 (解码) 读
MISMATCH = [
    ("bt709", "limited", "bt601", "limited"),
    ("bt601", "limited", "bt709", "limited"),
    ("bt709", "limited", "bt709", "full"),
]


def sh(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace", timeout=timeout, shell=False)


def _need_tool(name):
    try:
        r = sh([name, "-version"], timeout=30)
    except Exception:
        return False
    return r.returncode == 0


def make_reference(work, size, frames):
    """造一段**已知 RGB 真值**的 FFV1（rgb24，无子采样）。"""
    out = os.path.join(work, "ref_rgb.mkv")
    r = sh(["ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-f", "lavfi",
            "-i", f"smptebars=size={size}:rate=30:duration={frames / 30.0:.4f}",
            # 再叠一层渐变，让色度不是"只有几块纯色"——纯色块会掩盖子采样误差
            "-vf", "format=rgb24,geq=r='r(X,Y)':g='g(X,Y)*0.7+70*X/W':"
                   "b='b(X,Y)*0.7+70*Y/H'",
            "-c:v", "ffv1", "-pix_fmt", "rgb24", out])
    if r.returncode != 0:
        raise RuntimeError(f"造参考失败: {(r.stderr or '')[:300]}")
    return out


def to_yuv(src, dst, chroma, matrix, rng):
    """RGB 真值 → YUV 中间件（16 位 FFV1）。"""
    pix = "yuv420p16le" if chroma == "420" else "yuv444p16le"
    vf = (f"scale=out_color_matrix={matrix}:out_range={rng},"
          f"format={pix}")
    r = sh(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", src,
            "-vf", vf, "-c:v", "ffv1", "-pix_fmt", pix, dst])
    if r.returncode != 0:
        raise RuntimeError(f"{chroma}/{matrix}/{rng} 编码失败: "
                           f"{(r.stderr or '')[:300]}")
    return dst


def back_to_rgb(src, dst, matrix, rng):
    """YUV 中间件 → RGB（用**指定**参数解释）。"""
    vf = (f"scale=in_color_matrix={matrix}:in_range={rng},format=rgb24")
    r = sh(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", src,
            "-vf", vf, "-c:v", "ffv1", "-pix_fmt", "rgb24", dst])
    if r.returncode != 0:
        raise RuntimeError(f"{matrix}/{rng} 解码失败: {(r.stderr or '')[:300]}")
    return dst


def psnr_rgb(a, b):
    """RGB 域逐帧 PSNR（对 rgb24 输入，ffmpeg 报 r/g/b 三个平面）。"""
    r = sh(["ffmpeg", "-hide_banner", "-v", "info", "-i", a, "-i", b,
            "-lavfi", "[0:v]format=rgb24[x];[1:v]format=rgb24[y];"
                      "[x][y]psnr=stats_file=-", "-f", "null", "-"])
    rows = []
    for ln in (r.stdout or "").splitlines():
        if not ln.startswith("n:"):
            continue
        row = {}
        for tok in ln.split():
            if ":" in tok:
                k, v = tok.split(":", 1)
                row[k] = v
        rows.append(row)
    planes = {}
    for p in ("r", "g", "b"):
        # 注意统计行的键名是 `psnr_r` / `psnr_g` / `psnr_b`，
        # 不是裸的 `r`（第一版按裸键取，全部拿到 None → 误判 INVALID）
        key = f"psnr_{p}"
        vals = []
        for row in rows:
            v = row.get(key)
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue        # inf 会在这里被跳过，单独判断
        inf_all = bool(rows) and all(row.get(key) == "inf" for row in rows)
        if inf_all:
            planes[p] = {"psnr": "inf", "frames": len(rows)}
        else:
            planes[p] = {
                "psnr": (sum(vals) / len(vals)) if vals else None,
                "min": min(vals) if vals else None,
                "frames": len(vals),
            }
    return {"frames": len(rows), "planes": planes,
            "rc": r.returncode}


def db_to_8bit(db):
    """PSNR(dB, 8-bit 满量程) → 等效 RMSE 的 8-bit 码值。"""
    if db is None or db == "inf":
        return 0.0
    return round(255.0 * (10 ** (-db / 20.0)), 4)


def main():
    ap = argparse.ArgumentParser(description="N21 colour round-trip matrix")
    ap.add_argument("--work", default=None,
                    help="输出根（默认在系统临时目录下 mkdtemp，绝不覆盖已有目录）")
    ap.add_argument("--size", default="640x360")
    ap.add_argument("--frames", type=int, default=10)
    args = ap.parse_args()

    for t in ("ffmpeg", "ffprobe"):
        if not _need_tool(t):
            print(f"  [错误] 找不到 {t}")
            return 2

    if args.work:
        work = os.path.abspath(args.work)
        if os.path.isdir(work) and os.listdir(work):
            print(f"  [错误] --work 已存在且非空，拒绝覆盖：{work}")
            return 2
        os.makedirs(work, exist_ok=True)
    else:
        work = tempfile.mkdtemp(prefix="n21_colour_")
    print(f"  work  : {work}")
    print(f"  size  : {args.size}   frames: {args.frames}")

    ref = make_reference(work, args.size, args.frames)
    print(f"  ref   : {ref}")

    results = {"work": work, "reference": ref, "roundtrip": [],
               "mismatch": []}

    # ---- ① 同参数往返 ----
    for chroma, matrix, rng in COMBOS:
        tag = f"{chroma}_{matrix}_{rng}"
        mid = os.path.join(work, f"mid_{tag}.mkv")
        back = os.path.join(work, f"back_{tag}.mkv")
        try:
            to_yuv(ref, mid, chroma, matrix, rng)
            back_to_rgb(mid, back, matrix, rng)
            m = psnr_rgb(ref, back)
        except Exception as exc:
            print(f"  [失败] {tag}: {exc}")
            results["roundtrip"].append({"combo": tag, "error": str(exc)})
            continue
        row = {"combo": tag, "chroma": chroma, "matrix": matrix,
               "range": rng, **m,
               "rmse8": {p: db_to_8bit(m["planes"][p]["psnr"])
                         for p in ("r", "g", "b")}}
        results["roundtrip"].append(row)
        print("  {:<22} r={:<10} g={:<10} b={:<10}".format(
            tag,
            str(m["planes"]["r"]["psnr"])[:8],
            str(m["planes"]["g"]["psnr"])[:8],
            str(m["planes"]["b"]["psnr"])[:8]))

    # ---- ② 标签错配（按编码 / 按解码）----
    for enc_m, enc_r, dec_m, dec_r in MISMATCH:
        tag = f"enc{enc_m}_{enc_r}__dec{dec_m}_{dec_r}"
        mid = os.path.join(work, f"mid_mm_{tag}.mkv")
        back = os.path.join(work, f"back_mm_{tag}.mkv")
        try:
            to_yuv(ref, mid, "444", enc_m, enc_r)
            back_to_rgb(mid, back, dec_m, dec_r)
            m = psnr_rgb(ref, back)
        except Exception as exc:
            print(f"  [失败] {tag}: {exc}")
            results["mismatch"].append({"combo": tag, "error": str(exc)})
            continue
        row = {"combo": tag, "encode": [enc_m, enc_r],
               "decode": [dec_m, dec_r], **m,
               "rmse8": {p: db_to_8bit(m["planes"][p]["psnr"])
                         for p in ("r", "g", "b")}}
        results["mismatch"].append(row)
        print("  {:<34} r={:<10} g={:<10} b={:<10}".format(
            tag,
            str(m["planes"]["r"]["psnr"])[:8],
            str(m["planes"]["g"]["psnr"])[:8],
            str(m["planes"]["b"]["psnr"])[:8]))

    # ---- ③ 判据 ----
    print("\n  === 判定 ===")
    verdict, reasons = "OK", []
    for row in results["roundtrip"]:
        if row.get("error"):
            verdict = "INVALID"
            reasons.append(f"{row['combo']}: {row['error']}")
            continue
        if row["chroma"] == "444":
            for p in ("r", "g", "b"):
                if row["planes"][p]["psnr"] != "inf":
                    verdict = "INVALID"
                    reasons.append(
                        f"{row['combo']} 的 {p} 平面不是 inf（{row['planes'][p]['psnr']}）"
                        "—— 说明命令本身有问题，整轮结论作废")
    if not any(r["chroma"] == "420" for r in results["roundtrip"]
               if not r.get("error")):
        verdict = "INVALID"
        reasons.append("没有一组 420 结果，无法回答 N21 的问题")

    # 错配必须显著大于同参数 420
    base420 = None
    for r in results["roundtrip"]:
        if r.get("chroma") == "420" and r.get("matrix") == "bt709" \
                and r.get("range") == "limited" and not r.get("error"):
            base420 = r["planes"]["g"]["psnr"]
    for r in results["mismatch"]:
        if r.get("error"):
            continue
        g = r["planes"]["g"]["psnr"]
        if isinstance(base420, float) and isinstance(g, float) \
                and g > base420:
            verdict = "SUSPECT"
            reasons.append(
                f"{r['combo']} 的 G 平面 PSNR({g:.2f}) 反而高于同参数 420"
                f"({base420:.2f}) —— 标签可能没生效，需复查 scale 参数")

    print(f"    verdict = {verdict}")
    for x in reasons:
        print(f"      - {x}")

    out = os.path.join(work, "n21_result.json")
    results["verdict"] = verdict
    results["reasons"] = reasons
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"    json    = {out}")
    return 0 if verdict in ("OK", "SUSPECT") else 1


if __name__ == "__main__":
    sys.exit(main())
