# -*- coding: utf-8 -*-
"""G34 色度差异根因定位：只测 3 帧，几十秒出结果。

## 待验证的假设

经典路径与融合路径进入 `build_filter_chain()` 时的**像素格式不同**：

| 路径 | 进入滤镜图 | 依据 |
|---|---|---|
| 经典 | `gbrp16le`（planar RGB 16） | 实测 step3 的 pix_fmt |
| 融合 | `rgb48le`（**packed** RGB 16） | stage3 编码器的 rawvideo 输入 |

滤镜图里有三处对格式敏感：
- `edgedetect` → 只支持 yuv/gray
- `format=gray` + `geq(lum=...)`
- `format=yuv420p` + `geq(lum/cb/cr)` ← **肤色掩模靠 cb/cr 判断**

16-bit **packed** RGB → 8-bit 4:2:0 与 16-bit **planar** RGB → 8-bit 4:2:0，
在 swscale 里不是同一条代码路径，色度下采样的相位/舍入可能不同。
若 chroma 相位差半个像素，高频色度会大幅走样 —— 这与实测
"Y 38.9 dB 但 U/V 只有 10 dB" 的特征完全吻合。

## 实验

同一份 3 帧 rgb48le 原始数据，三种喂法：

- **A**（模拟融合）：raw → filter_chain
- **B**（模拟经典）：raw → ffv1(gbrp16le) → filter_chain
- **C**（候选修法）：raw → `format=gbrp16le` → filter_chain

* 若 A ≠ B 而 C == B → 根因确认，修法是给融合路径补一次
  `format=gbrp16le`（packed→planar 重排，**数学无损**）。
* 若 A == B → 我的假设错了，得换方向查（比如 ffv1 编码本身）。
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = r"E:\codex\code1\old-video-and-picture-fix"
sys.path.insert(0, os.path.join(ROOT, "video_restoration_pipeline", "stages"))
sys.path.insert(0, os.path.join(ROOT, "shared_modules"))

import stage4_postprocess as s4  # noqa: E402

W, H, FRAMES = 1920, 1080, 3
CHAIN = s4.build_filter_chain({})


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                       shell=False, **kw)
    if r.returncode != 0:
        print("    [!] rc=%d: %s" % (r.returncode, (r.stderr or "")[-400:]))
    return r


def psnr(a, b):
    """返回 (y, u, v) 三平面的平均 PSNR（dB）。"""
    r = run(["ffmpeg", "-hide_banner", "-v", "info", "-i", a, "-i", b,
             "-lavfi", "psnr=stats_file=-", "-f", "null", "-"])
    vals = {}
    for line in (r.stdout or "").splitlines():
        if not line.startswith("n:"):
            continue
        d = dict(x.split(":", 1) for x in line.split() if ":" in x)
        for k in ("psnr_y", "psnr_u", "psnr_v"):
            v = d.get(k)
            if v is None:
                continue
            # inf 记为 999，便于求均值
            vals.setdefault(k, []).append(999.0 if v.startswith("inf") else float(v))
    if not vals:
        return None
    out = []
    for k in ("psnr_y", "psnr_u", "psnr_v"):
        xs = vals.get(k) or []
        out.append(round(sum(xs) / len(xs), 3) if xs else None)
    return tuple(out)


def main():
    tmp = tempfile.mkdtemp(prefix="g34probe_", dir=r"E:\codex\_media_r17_20261031T0004\tmp")
    print("  工作目录: %s" % tmp)
    raw = os.path.join(tmp, "src.rgb48")
    a_out = os.path.join(tmp, "A_fused.mkv")
    step3 = os.path.join(tmp, "step3_gbrp16.mkv")
    b_out = os.path.join(tmp, "B_classic.mkv")
    c_out = os.path.join(tmp, "C_forced_planar.mkv")

    # ---- 造 3 帧 rgb48le 原始数据（带真实纹理与肤色块，让掩模真的被触发）----
    print("\n  [0] 生成 %d 帧 rgb48le 测试数据" % FRAMES)
    src = os.path.join(tmp, "src.mkv")
    run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
         # 上半：彩色测试图（有边缘）；下半：一块肤色（触发 skinmask）
         "-f", "lavfi", "-i",
         "testsrc2=size=%dx%d:rate=30000/1001" % (W, H),
         "-f", "lavfi", "-i",
         "color=c=0xC08060:size=%dx%d:rate=30000/1001" % (W, H // 2),
         "-filter_complex",
         "[1:v]crop=%d:%d:0:0[skin];[0:v]crop=%d:%d:0:%d[top];"
         "[top][skin]vstack" % (W, H // 2, W, H // 2, H // 2),
         "-frames:v", str(FRAMES), "-pix_fmt", "rgb48le", "-c:v", "ffv1", src])
    run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", src,
         "-f", "rawvideo", "-pix_fmt", "rgb48le", raw])
    size = os.path.getsize(raw) if os.path.exists(raw) else 0
    print("      raw 大小 %.1f MB（期望约 %.1f）"
          % (size / 1e6, W * H * 6 * FRAMES / 1e6))

    base_in = ["-f", "rawvideo", "-pix_fmt", "rgb48le",
               "-s", "%dx%d" % (W, H), "-framerate", "30000/1001", "-i", raw]

    # ---- A：模拟融合（packed RGB 直接进链）----
    print("\n  [A] 模拟融合：raw(packed rgb48le) → filter_chain")
    run(["ffmpeg", "-hide_banner", "-v", "error", "-y"] + base_in +
        ["-filter_complex", CHAIN, "-map", "[vout]",
         "-pix_fmt", "yuv420p16le", "-c:v", "ffv1", a_out])

    # ---- B：模拟经典（先存 gbrp16le，再进链）----
    print("  [B] 模拟经典：raw → ffv1(gbrp16le) → filter_chain")
    run(["ffmpeg", "-hide_banner", "-v", "error", "-y"] + base_in +
        ["-pix_fmt", "gbrp16le", "-c:v", "ffv1", step3])
    run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", step3,
         "-filter_complex", CHAIN, "-map", "[vout]",
         "-pix_fmt", "yuv420p16le", "-c:v", "ffv1", b_out])

    # ---- C：候选修法（进链前显式转 planar）----
    print("  [C] 候选修法：raw → format=gbrp16le → filter_chain")
    run(["ffmpeg", "-hide_banner", "-v", "error", "-y"] + base_in +
        ["-filter_complex", CHAIN.replace("[0:v]", "[0:v]format=gbrp16le,", 1),
         "-map", "[vout]", "-pix_fmt", "yuv420p16le", "-c:v", "ffv1", c_out])

    print("\n  === 逐平面 PSNR（dB，999=完全相同）===")
    result = {}
    for label, other in (("A(融合) vs B(经典)", (a_out, b_out)),
                         ("C(强制planar) vs B(经典)", (c_out, b_out))):
        if not all(os.path.exists(p) for p in other):
            print("    %-28s 缺少产物，跳过" % label)
            continue
        got = psnr(other[0], other[1])
        result[label] = got
        if got is None:
            print("    %-28s 未解析到逐帧数据（不能据此判等价）" % label)
        else:
            print("    %-28s Y=%s  U=%s  V=%s" % (label, got[0], got[1], got[2]))

    print("\n  === 结论 ===")
    a = result.get("A(融合) vs B(经典)")
    c = result.get("C(强制planar) vs B(经典)")
    verdict = "UNKNOWN"
    if a and c:
        if a[1] is not None and a[1] > 40 and a[2] is not None and a[2] > 40:
            print("    A 与 B 的色度已经接近 —— 输入格式不是根因，假设被推翻")
            verdict = "HYPOTHESIS_REJECTED"
        elif a[1] is not None and a[1] < 25 and c[1] is not None and c[1] > 40:
            print("    根因确认：packed rgb48le 与 planar gbrp16le 在滤镜图里不等价。")
            print("    修法：融合路径进链前补一次 format=gbrp16le（数学无损重排）。")
            verdict = "ROOT_CAUSE_CONFIRMED"
        else:
            print("    数据不一致，需要逐段拆链继续定位")
            verdict = "INCONCLUSIVE"
    print("    verdict = %s" % verdict)
    print("    产物保留在: %s" % tmp)
    with open(os.path.join(tmp, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict, "psnr": result,
                   "chain": CHAIN}, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
