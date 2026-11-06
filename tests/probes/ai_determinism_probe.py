# -*- coding: utf-8 -*-
"""AI 超分（stage3）的 GPU 确定性探针。

## 要回答的问题

stage1 的 VapourSynth 渲染已被证明在多线程下非确定（见 vs_render.py）。
那么 **stage3 的 RealESRGAN 推理呢**？如果它也非确定，那即使 stage1 单线程，
整条链路仍然不可复现，"逐帧等价"类判据永远测不准。

项目当前**没有任何** cuDNN 确定性设置（`grep cudnn` 零命中），
PyTorch 的默认值是：

    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.benchmark     = False    # 好：不做 autotune
    torch.backends.cudnn.deterministic = False    # 可能选非确定算法

## 测什么

忠实复现生产路径（`shared_modules/stream_ai_template.py` 的构造）：

    RealESRGANer(scale=4, model_path=models/realesr-general-x4v3.pth,
                 model=SRVGGNetCompact(num_in_ch=3, num_out_ch=3,
                                       num_feat=64, num_conv=32,
                                       upscale=4, act_type='prelu'),
                 half=True, gpu_id=0)
    out, _ = upsampler.enhance(img_bgr, outscale=2)

三种配置各跑两次，比较**是否逐像素 bit-identical**：

| 配置 | 说明 |
|---|---|
| `default` | 现状（deterministic=False, half=True） |
| `det_fp16` | deterministic=True + benchmark=False, half=True |
| `det_fp32` | deterministic=True + benchmark=False, half=False |

## 判据

不看 PSNR（那会掩盖 1-LSB 级差异），直接比 `np.array_equal` 与
`max|Δ|`。**"差 0 个 bit"才是确定**。

## 运行

必须用 `.venv_r4`（AI 栈在这里；r72 没有 torch）：

    .venv_r4\\Scripts\\python.exe tests/probes/ai_determinism_probe.py
"""
import os
import sys

PROJ = r"E:\codex\code1\old-video-and-picture-fix"
MODEL_PATH = os.path.join(PROJ, "models", "realesr-general-x4v3.pth")
MODEL_NAME = "realesr-general-x4v3"
NATIVE_SCALE = 4
TARGET_SCALE = 2
SIZE = 256          # 小图足够判断确定性，且快


def make_input():
    """造一张有渐变 + 高频噪声的测试图（BGR uint8）。

    纯随机噪声对超分模型是极端输入；渐变 + 噪声更接近真实画面，
    同时保留足够的高频细节让卷积走满所有分支。
    """
    import numpy as np

    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    r = (xx / SIZE) * 180 + 30
    g = (yy / SIZE) * 180 + 30
    b = ((xx + yy) / (2 * SIZE)) * 160 + 40
    base = np.stack([b, g, r], axis=-1)          # BGR
    rng = np.random.default_rng(20261106)
    noise = rng.normal(0, 12, base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def build_upsampler(deterministic, half):
    import torch
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
    from realesrgan import RealESRGANer

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    if deterministic:
        # 关掉所有"可能非确定"的算法选择（有些算子没有确定实现会直接报错，
        # 那样也能立刻暴露问题，比静默不确定好）
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_conv=32, upscale=NATIVE_SCALE,
                            act_type="prelu")
    return RealESRGANer(scale=NATIVE_SCALE, model_path=MODEL_PATH,
                        model=model, half=bool(half), gpu_id=0)


def run_once(upsampler, img):
    out, _mode = upsampler.enhance(img.copy(), outscale=TARGET_SCALE)
    return out


def compare(a, b):
    import numpy as np

    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {"same": False, "reason": "shape %s vs %s" % (a.shape, b.shape)}
    same = bool(np.array_equal(a, b))
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return {
        "same": same,
        "shape": list(a.shape),
        "max_abs_diff": int(diff.max()),
        "n_diff_pixels": int((diff > 0).sum()),
        "n_pixels": int(diff.size),
    }


def _run_once_mode(out_path):
    """跨进程模式：只跑一次，把结果存成 .npy。

    为什么要跨进程
    --------------
    同一进程内跑两次，CUDA context 与 cuDNN 的算法选择是同一份；
    生产里每次跑批都是**新进程**。只有跨进程对比才能回答
    "两次跑批会不会得到不同结果"。
    """
    import numpy as np

    img = make_input()
    up = build_upsampler(False, True)      # 与生产完全一致（默认配置）
    out = run_once(up, img)
    np.save(out_path, np.asarray(out))
    print("  已保存: %s shape=%s dtype=%s"
          % (out_path, getattr(out, "shape", "?"),
             getattr(out, "dtype", "?")))
    return 0


def main():
    if not os.path.isfile(MODEL_PATH):
        print("  [跳过] 权重不存在: %s" % MODEL_PATH)
        return 2

    if len(sys.argv) >= 3 and sys.argv[1] == "--once":
        return _run_once_mode(sys.argv[2])

    img = make_input()
    print("  输入: %dx%d BGR uint8（渐变 + 高斯噪声）" % (img.shape[1], img.shape[0]))
    print("  模型: %s（native x%d → 目标 x%d）"
          % (MODEL_NAME, NATIVE_SCALE, TARGET_SCALE))

    configs = [
        ("default", False, True),
        ("det_fp16", True, True),
        ("det_fp32", True, False),
    ]
    results = {}
    for name, det, half in configs:
        print("\n  [%s] deterministic=%s half=%s" % (name, det, half))
        try:
            up = build_upsampler(det, half)
            a = run_once(up, img)
            b = run_once(up, img)
            info = compare(a, b)
            results[name] = info
            if info.get("same"):
                print("    -> 完全相同（0 个像素有差异）")
            elif info.get("reason"):
                print("    -> 形状不一致: %s" % info["reason"])
            else:
                print("    -> **不同**：%d / %d 像素有差异，最大差 %d"
                      % (info["n_diff_pixels"], info["n_pixels"],
                         info["max_abs_diff"]))
            del up
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception as exc:
            import traceback
            results[name] = {"error": "%s: %s" % (type(exc).__name__, exc)}
            print("    -> 失败: %s: %s" % (type(exc).__name__, str(exc)[:120]))
            traceback.print_exc(limit=2)

    print("\n  === 汇总 ===")
    for name, _d, _h in configs:
        r = results.get(name) or {}
        if r.get("error"):
            print("    %-10s ERROR  %s" % (name, r["error"][:70]))
        elif r.get("same"):
            print("    %-10s 完全相同" % name)
        else:
            print("    %-10s 不同（max|Δ|=%s，%s/%s 像素）"
                  % (name, r.get("max_abs_diff"), r.get("n_diff_pixels"),
                     r.get("n_pixels")))

    print("\n  === 结论 ===")
    d = results.get("default") or {}
    if d.get("same") is True:
        print("    AI 推理在默认设置下就是确定的 —— 不需要额外改动。")
    elif d.get("same") is False:
        df = (results.get("det_fp32") or {})
        d16 = (results.get("det_fp16") or {})
        if df.get("same") is True or d16.get("same") is True:
            print("    默认**非确定**，但加上 deterministic=True 后确定。")
            print("    建议：跟随 vs.deterministic 开关，在 AI 脚本里设置")
            print("    torch.backends.cudnn.deterministic = True。")
        else:
            print("    默认非确定，且 deterministic=True 也没救 —— 需要进一步")
            print("    排查（可能是 tile 边界、FP16 归约，或模型含非确定算子）。")
    else:
        print("    未能取得有效结果。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
