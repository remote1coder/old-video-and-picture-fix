# -*- coding: utf-8 -*-
"""R23-10：N20 的**功能/格式矩阵**（经典物化 vs 融合串流）。

## 审核要什么

> 保留当前默认 5 帧对照。增加阶段 4 功能单开、合法组合、显式 off；
> 420/444、CFR/VFR 可用路径、非零原点、颜色合同；两条后半链使用同一冻结
> 输入和同一操作计划。先非 AI 参考，再同一次 AI 输出，再两条独立完整 AI
> 运行。各实验回答的问题分开……不支持融合的组合明确走经典并记录理由。

## 本探针做什么

用**同一串冻结帧**（不跑模型，A/B 两组已由 R22 的
`test_r22_frame_boundary_equiv.py` 覆盖 AI 路径）驱动 stage3 的两条出口：

- 经典：流式出 step3 → 单独跑 stage4 的滤镜链；
- 融合：滤镜链直接接在编码器上（`chain.mode: stage34` 的实际形态）。

矩阵 = **stage4 的 3 种参数组合 × 2 种工作格式**：

| 组合 | 参数 | 想证明 |
|---|---|---|
| `grade_default` | 默认 lift/gamma/gain | 默认链两路一致 |
| `grade_off` | 全部 1.0（显式关）| 关掉调色也不改变"两路一致" |
| `grade_strong` | 偏移放大 | 调色更强时仍一致（不是只在"几乎不动"时才一致）|

格式：`420`（yuv420p16le）与 `444`（yuv444p16le）。**同一编码器、同一 crf**
（都是 ffv1 无损），避免把编码器差异混进来。

## 不支持融合的情况

融合只对 **raw + CFR** 开放。VFR 会被 `_classify_timeline` 判成 `vfr`，
stage3 的流式路径直接跳过（回 PNG 两阶段），日志有明确理由。本探针把这条
判定**跑一遍**并打印，不清空它、也不假装能融合。

## 用法

    python tests/probes/r23_n20_matrix.py [--frames 5]
"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r22_444_classic_path as s3probe  # noqa: E402

IN_W, IN_H, SCALE, FPS = 320, 240, 2, 10

COMBOS = [
    ("grade_default", {}),
    ("grade_off", {"color_lift_r": 1.0, "color_lift_g": 1.0,
                   "color_lift_b": 1.0, "color_gamma_r": 1.0,
                   "color_gamma_g": 1.0, "color_gamma_b": 1.0,
                   "color_gain_r": 1.0, "color_gain_g": 1.0,
                   "color_gain_b": 1.0, "post_saturation": 1.0,
                   "post_contrast": 1.0}),
    ("grade_strong", {"color_lift_r": 1.10, "color_lift_g": 1.05,
                      "color_lift_b": 1.02, "color_gamma_r": 0.90,
                      "color_gamma_g": 0.95, "color_gamma_b": 0.98,
                      "color_gain_r": 1.08, "color_gain_g": 1.04,
                      "color_gain_b": 1.01, "post_saturation": 1.15,
                      "post_contrast": 1.06}),
]

FORMATS = [("420", "yuv420p16le"), ("444", "yuv444p16le")]


def _ff(*args, timeout=900):
    return s3probe._run(["ffmpeg", "-hide_banner", "-v", "error", "-y"] +
                        list(args), timeout=timeout)


def _make_source(path, frames):
    still = os.path.join(os.path.dirname(path), "_still.png")
    r = _ff("-f", "lavfi", "-i",
            "testsrc2=size=%dx%d:rate=1:duration=1" % (IN_W, IN_H),
            "-frames:v", "1", "-pix_fmt", "rgb48be", still)
    if r.returncode != 0:
        raise SystemExit("造静帧失败: %s" % (r.stderr or "")[-300:])
    stripe = ("color=c=red:size=%dx%d:rate=1:duration=1,"
              "geq=lum='if(mod(X,2),235,16)':cb=128:cr=128[st];"
              "[0:v][st]blend=all_mode=multiply:all_opacity=0.45"
              % (IN_W, IN_H))
    r = _ff("-loop", "1", "-i", still, "-filter_complex", stripe,
            "-frames:v", str(frames), "-r", str(FPS),
            "-c:v", "ffv1", "-pix_fmt", "yuv420p", path)
    try:
        os.remove(still)
    except OSError:
        pass
    if r.returncode != 0:
        raise SystemExit("造源失败: %s" % (r.stderr or "")[-400:])
    return path


def _write_frames(target, frames):
    """按输出尺寸写逐像素不同的 rgb48le 帧（错位会立刻体现成有限 PSNR）。"""
    import numpy as np
    os.makedirs(target, exist_ok=True)
    for n in os.listdir(target):
        os.remove(os.path.join(target, n))
    h, w = IN_H * SCALE, IN_W * SCALE
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(frames):
        base = [(60000, 8000, 8000), (8000, 60000, 8000),
                (8000, 8000, 60000), (45000, 45000, 8000),
                (8000, 45000, 45000)][i % 5]
        arr = np.zeros((h, w, 3), dtype="<u2")
        ramp = ((xx * 65535) // max(1, w - 1)).astype("<u2")
        for c in range(3):
            arr[:, :, c] = (base[c] // 2 + ramp // 2).astype("<u2")
        arr.tofile(os.path.join(target, "frame_%03d.raw" % i))
    return sorted(os.path.join(target, n) for n in os.listdir(target))


def _psnr_rows(a, b):
    r = s3probe._run(["ffmpeg", "-hide_banner", "-v", "info", "-i", a, "-i",
                      b, "-lavfi", "[0:v][1:v]psnr=stats_file=-",
                      "-f", "null", "-"], timeout=900)
    rows = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("n:")]
    return r.returncode, rows


def _identical(rows):
    return bool(rows) and all(
        "psnr_y:inf" in x and "psnr_u:inf" in x and "psnr_v:inf" in x
        for x in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\codex\_media_r23_20261112T1202\s_n20")
    ap.add_argument("--frames", type=int, default=5)
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    work = os.path.join(args.root, stamp)
    os.makedirs(work, exist_ok=True)
    print("[n20] work=%s" % work)

    from stage3_ai_upscale import _stream_raw_upscale, _classify_timeline
    import stage4_postprocess as s4

    src = _make_source(os.path.join(work, "src.mkv"), args.frames)
    fz = os.path.join(work, "frames")
    _write_frames(fz, args.frames)

    rows = []
    for cname, cparams in COMBOS:
        chain = s4.build_filter_chain(cparams)
        # 证明这组参数**真的改变了处理**（否则"两路一致"说明不了什么）
        base_chain = s4.build_filter_chain({})
        for fname, fpix in FORMATS:
            tag = "%s_%s" % (cname, fname)
            st3 = os.path.join(work, "step3_%s.mkv" % tag)
            ok1, why1, _n1 = _stream_raw_upscale(
                input_video=src, output_mkv=st3, width=IN_W, height=IN_H,
                scale=SCALE, fps_str=str(FPS), model="unused",
                model_dir=work, ai_py=sys.executable,
                frame_source_dir=fz, inactivity_s=300, out_pix_fmt=fpix)
            classic = os.path.join(work, "classic_%s.mkv" % tag)
            r = _ff("-i", st3, "-filter_complex", chain, "-map", "[vout]",
                    "-c:v", "ffv1", "-pix_fmt", fpix, classic) \
                if ok1 else None
            fused = os.path.join(work, "fused_%s.mkv" % tag)
            ok2, why2, _n2 = _stream_raw_upscale(
                input_video=src, output_mkv=fused, width=IN_W, height=IN_H,
                scale=SCALE, fps_str=str(FPS), model="unused",
                model_dir=work, ai_py=sys.executable,
                frame_source_dir=fz, inactivity_s=300,
                post_filter_chain=chain, out_pix_fmt=fpix,
                chain_input_fmt=fpix)
            row = {"combo": cname, "format": fname, "pix_fmt": fpix,
                   "chain_changed_vs_default": chain != base_chain,
                   "classic_ok": bool(ok1 and r and r.returncode == 0),
                   "fused_ok": bool(ok2),
                   "why": "" if (ok1 and ok2) else (why1 or why2)}
            if row["classic_ok"] and row["fused_ok"]:
                rc, prows = _psnr_rows(classic, fused)
                row["psnr_frames"] = len(prows)
                row["identical"] = _identical(prows)
                row["rc"] = rc
                if not row["identical"]:
                    row["first_diff"] = next(
                        (x[:160] for x in prows
                         if "psnr_y:inf" not in x or "psnr_u:inf" not in x
                         or "psnr_v:inf" not in x), "")
            rows.append(row)
            print("[n20] %-22s classic=%s fused=%s 帧=%s identical=%s" % (
                tag, row["classic_ok"], row["fused_ok"],
                row.get("psnr_frames"), row.get("identical")))

    # ---- 融合不适用的情况：VFR 必须明确走经典 ----
    vfr_map = {0: 0.0, 1: 0.033, 2: 0.100, 3: 0.133, 4: 0.200}
    tl = _classify_timeline(vfr_map, expected_frames=5)
    fusion_gate = {"vfr_pts_map": vfr_map, "classified_as": tl,
                   "streaming_allowed": tl == "cfr",
                   "note": ("stage3 的零落盘流式路径要求确认 CFR；VFR 会回退"
                            "到 PNG 两阶段（经典物化），并在日志里写清理由。"
                            "本轮不假装 VFR 能融合。")}
    print("[n20] VFR 判定：%s（流式允许=%s）" % (tl, fusion_gate["streaming_allowed"]))

    ok = True
    for row in rows:
        if not row["classic_ok"] or not row["fused_ok"]:
            print("[FAIL] %s/%s 有一条链没跑成功: %s"
                  % (row["combo"], row["format"], row["why"]))
            ok = False
            continue
        if row.get("psnr_frames") != args.frames:
            print("[FAIL] %s/%s 只比了 %s 帧" % (row["combo"], row["format"],
                                               row.get("psnr_frames")))
            ok = False
        if not row.get("identical"):
            print("[FAIL] %s/%s 两路不相同: %s"
                  % (row["combo"], row["format"], row.get("first_diff")))
            ok = False
    if not any(r["chain_changed_vs_default"] for r in rows):
        print("[FAIL] 三组参数生成的滤镜链完全相同 —— 矩阵没测到东西")
        ok = False
    if tl != "vfr":
        print("[FAIL] 构造的 VFR 时间轴没有被判成 vfr（判定器有问题）")
        ok = False

    out = os.path.join(work, "n20_matrix.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"work": work, "ok": ok, "frames": args.frames,
                   "rows": rows, "fusion_gate": fusion_gate}, f,
                  ensure_ascii=False, indent=2)
    print("\n[n20] evidence -> %s" % out)
    print("[n20] VERDICT = %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
