#!/usr/bin/env python3
"""
Stage 3: Photo AI Upscaling
Integer-scale AI super-resolution for photos.
Supports: Real-ESRGAN, DAT, GPEN models per v7.32 §3.1.

v6.3 Compliance:
- Target resolution must be <= 1080p (1920x1080) for AI upscale
- Original input pixels and intermediate pixels are BANNED from AI upscale
- Lanczos preprocessing to AI is BANNED
- GPU memory pre-check with CPU fallback
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from checkpoint import (compute_param_hash, build_intermediate_name,  # noqa: E402
                         gpu_memory_check, safe_input_stem)


#: FIX-2026-10-22：AI 推理解释器缓存。
#: 探针要起子进程 import torch（约 1~2 秒），1346 张图每张探一次就是几十分钟
#: 纯开销，所以进程内只解析一次。
_AI_PY_CACHE: dict = {}


def _ai_python() -> str:
    """返回"能跑 torch/realesrgan"的解释器；解析不到时退回 sys.executable。

    为什么必须有这一层（2026-10-22 真实素材实测）：
    编排解释器可能是 `tools_r4/vs_portable/r72/python.exe` —— 它只有
    VapourSynth，**没有 torch / realesrgan**。原来这里直接写
    `sys.executable`，于是三个 AI 后端依次失败：

        ModuleNotFoundError: No module named 'realesrgan'
        STAGE3_MODEL_FALLBACK realesr-gener-x4v3 → RealESRGAN_x2plus → DAT_x2
        STAGE3_AI_SKIP_TRADITIONAL  All AI backends failed

    结果：**以为跑了 AI 超分，实际交付的还是传统输出**（只是日志里写了
    一行降级）。stage4 早已走 ai_runtime 解析（日志有
    AI_INTERPRETER_RESOLVED），照片链 stage3 一直漏着这一层。
    """
    if "py" in _AI_PY_CACHE:
        return _AI_PY_CACHE["py"]
    py = sys.executable
    try:
        import ai_runtime as _ai_rt
        resolved = _ai_rt.resolve_ai_interpreter()
        if resolved:
            py = resolved
        else:
            sys.stderr.write(
                "stage3: 未找到具备 AI 推理栈的解释器，退回编排解释器 "
                f"({sys.executable})；AI 超分很可能全部失败\n")
    except Exception as e:
        sys.stderr.write(f"stage3: ai_runtime 解析失败({e})，退回 sys.executable\n")
    _AI_PY_CACHE["py"] = py
    return py


def _get_model_basename(model: str) -> str:
    """Extract model family from full model name.

    Supported families per v7.32 §3.1: RealESRGAN, DAT, GPEN.
    """
    # e.g. "RealESRGAN_x2plus" -> "RealESRGAN", "DAT_x2" -> "DAT",
    # "realesr-general-x4v3" -> "realesr-general"
    if model.startswith("realesr-general"):
        return "realesr-general"
    elif model.startswith("RealESRGAN"):
        return "RealESRGAN"
    elif model.startswith("DAT"):
        return "DAT"
    elif model.startswith("GPEN"):
        return "GPEN"
    return "RealESRGAN"


def _get_model_for_scale(model: str, scale: int) -> str:
    """Get correct model name for a given scale, preserving model family.

    Supported families per v7.32 §3.1: realesr-general, RealESRGAN, DAT, GPEN.
    FIX-P1-15: realesr-general-x4v3 is a 4× model; for 2× output it stays
    the same model and is Lanczos-downscaled to 2× (no separate 2× model).
    """
    family = _get_model_basename(model)
    if family == "realesr-general":
        return "realesr-general-x4v3"  # single 4× model, Lanczos handles scale
    elif family == "RealESRGAN":
        return f"RealESRGAN_x{scale}plus"
    elif family == "DAT":
        # DAT-2 is default; preserve DAT_2 prefix if present, else DAT
        if model.startswith("DAT_2"):
            return f"DAT_2_x{scale}"
        return f"DAT_x{scale}"
    elif family == "GPEN":
        # FIX-2026-10-20：GPEN 是**同分辨率人脸修复**模型（blind face
        # restoration），没有 x2/x4 变体。原来返回 GPEN_x{scale} →
        # 去找根本不存在的 GPEN_x2.pth。固定为官方推荐的 512 版。
        return "GPEN-BFR-512"
    return f"RealESRGAN_x{scale}plus"


def _resolve_input_source(temp_dir: str, job_id: str, input_name: str,
                          ban_original: bool, ban_intermediate: bool) -> tuple:
    """Resolve valid input source for AI upscale.

    Rules (v6.3 §5.3):
    - Original input pixels are banned
    - Intermediate pixels from prior AI stages are banned
    - Only traditional-processed intermediates are allowed

    Returns: (path, source_type) or (None, "") if no valid source
    """
    stem = safe_input_stem(input_name)
    # Stage 1 preprocess is the only allowed source for AI upscale
    pattern = f"{job_id}_{stem}_step1_preprocess*"
    candidates = glob.glob(os.path.join(temp_dir, pattern))
    if candidates:
        # Sort by mtime, pick most recent
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0], "traditional"
    return None, ""


def _run_realesrgan_vulkan(input_path: str, output_path: str, model: str,
                             scale: int, model_dir: str, use_gpu: bool) -> bool:
    """Run Real-ESRGAN via ncnn-vulkan CLI."""
    # FIX-P12-8: ncnn-vulkan uses .param/.bin, not .pth
    param_path = os.path.join(model_dir, f"{model}.param")
    bin_path = os.path.join(model_dir, f"{model}.bin")
    if not (os.path.exists(param_path) and os.path.exists(bin_path)):
        return False
    vulkan_bin = shutil.which("realesrgan-ncnn-vulkan")
    if not vulkan_bin:
        return False
    cmd = [
        vulkan_bin,
        "-i", input_path,
        "-o", output_path,
        "-n", model,
        "-s", str(scale),
    ]
    if not use_gpu:
        cmd.extend(["-g", "-1"])
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=1800, shell=False)
        return r.returncode == 0
    except Exception:
        return False


def _run_realesrgan_python(input_path: str, output_path: str, model: str,
                           scale: int, model_dir: str, use_gpu: bool,
                           bit_depth: int = 8) -> bool:
    """Run Real-ESRGAN via Python API fallback.

    FIX-P12-3: Added bit_depth parameter per v6.32 §4.2.5.
    When bit_depth=16, output is saved as 16-bit TIFF.
    """
    model_path = os.path.join(model_dir, f"{model}.pth")
    if not os.path.exists(model_path):
        return False
    try:
        import json
        # PHOTO-AUDIT-012: Use json.dumps for safe string quoting
        safe_model_path = json.dumps(model_path)
        safe_input_path = json.dumps(input_path)
        safe_output_path = json.dumps(output_path)
        # R4: 必须**显式构造架构**再传给 RealESRGANer。
        # 实测（realesrgan 0.3.0）：不传 model= 时自动推断对
        # realesr-general-x4v3 返回 None，随后
        # `None.load_state_dict(...)` -> AttributeError，
        # 而这里 `except Exception: return False` 会把它**完全吞掉**，
        # 表现为 primary 与所有 fallback 都"静默失败"，
        # 最终输出未经超分的传统结果（真实素材仍是 240x320）。
        # realesr-general 是 4× 的 SRVGGNetCompact，需按 netscale=4 推理
        # 再下采样到目标倍数（v7.32 §3.1 / FIX-P1-15）。
        family_now = _get_model_basename(model)
        if family_now == "realesr-general":
            netscale = 4
            arch_code = (
                "from realesrgan.archs.srvgg_arch import SRVGGNetCompact; "
                "net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, "
                "num_feat=64, num_conv=32, upscale=4, act_type='prelu')")
        else:
            netscale = scale
            arch_code = (
                "from basicsr.archs.rrdbnet_arch import RRDBNet; "
                f"net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, "
                f"num_block=23, num_grow_ch=32, scale={scale})")
        # 推理倍数与目标倍数不同时，用 Lanczos 下采样到目标尺寸
        if netscale != scale:
            resample_code = (
                f"if output.shape[0] and {netscale} != {scale}:\n"
                f"    from PIL import Image as _I; "
                f"_h, _w = output.shape[:2]; "
                f"output = np.array(_I.fromarray(output).resize("
                f"(_w * {scale} // {netscale}, _h * {scale} // {netscale}), "
                f"_I.LANCZOS))\n"
            )
        else:
            resample_code = ""
        # FIX-P12-3: 16-bit output save logic
        # FIX-P12-6: PIL I;16 only supports 2D arrays; for 3-channel 16-bit,
        # use tifffile to save as TIFF (PIL cannot save RGB uint16 directly).
        # Fallback: save as 8-bit PNG (quality loss) if tifffile unavailable.
        # FIX-NEW-V-18 / R21: 8-bit 分支也不能用裸 `Image.fromarray(output)
        # .save(path)` —— 对 `.tif` PIL 会默认选 **LZW**，而 tifffile 解 LZW
        # 需要未安装的 imagecodecs → 下游 stage5 读不回来 →
        # MediaPrecisionError → rc=86（写得出、读不回来的自造依赖）。
        # 改走 tifffile zlib（纯 Python 内置，无需额外依赖）。
        save_code_8 = (
            f"    try:\n"
            f"        import tifffile as _tf2; "
            f"_tf2.imwrite({safe_output_path}, output, compression='zlib')\n"
            f"    except ImportError:\n"
            f"        Image.fromarray(output).save({safe_output_path})\n"
        )
        save_code = (
            f"if {bit_depth} == 16:\n"
            f"    out_16 = (output.astype(np.float32) / 255.0 * 65535.0)"
            f".clip(0, 65535).astype(np.uint16)\n"
            f"    try:\n"
            f"        import tifffile; "
            f"tifffile.imwrite({safe_output_path}, out_16, compression='zlib')\n"
            f"    except ImportError:\n"
            f"        Image.fromarray((out_16 // 256).astype(np.uint8))"
            f".save({safe_output_path})\n"
            f"else:\n"
            + save_code_8
        )
        cmd = [
            _ai_python(), "-c",
            # R4：**必须用换行，不能用 `;` 串成一行**。
            # Python 语法不允许在分号后接复合语句（if / for / try）：
            # `x=1; if True: pass` 直接 SyntaxError。
            # 而下面的 resample / save 都含 if / try，于是整个子程序
            # 一直是语法错误 —— 这就是"AI 后端全部静默失败"的真正原因
            # （真实素材跑出来仍是没有超分的 240x320）。
            chr(10).join([
                "from realesrgan import RealESRGANer",
                "from basicsr.utils.download_util import load_file_from_url",
                "import torch",
                arch_code,
                f"model = RealESRGANer(scale={netscale}, "
                f"model_path={safe_model_path}, model=net, "
                f"pre_pad=10, "
                f"half={'False' if not use_gpu else 'not torch.cuda.is_available()'})",
                "from PIL import Image",
                "import numpy as np",
                # FIX-NEW-V-14 / R16: 不能用 Image.open(...).convert('RGB')。
                # G'MIC 写出的 uint16 TIFF 里装的是 8-bit 量级数据，PIL 走
                # uint8 TIFF 路径会读出**全 0**（实测 mean 0.0，原图 107.27）
                # → AI 超分一张黑图 → 照片链交付纯黑 AVIF 而 rc 仍为 0。
                # 改走 media_io.load_image_rgb_uint8（tifffile + 值域自适应）。
                "import sys as _sys",
                f"_sys.path.insert(0, {str(SCRIPT_DIR.parent.resolve() / 'shared_modules')!r})",
                "from media_io import load_image_rgb_uint8 as _ldr",
                f"_arr = _ldr({safe_input_path})",
                "print('IN_ARR', _arr.shape, _arr.dtype, int(_arr.max()))",
                "output, _ = model.enhance(_arr)",
                resample_code,
                save_code,
            ]),
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=1800, shell=False)
        if r.returncode != 0:
            # R4: 不能静默吞掉。子进程的报错是唯一能定位"AI 全部失败"的线索。
            # 取 **尾部**：Python traceback 在末尾，取开头只会被命令本身占满
            # （这次就是因此白跑了好几轮）。
            _err = (r.stderr or b"").decode("utf-8", "replace").strip()
            sys.stderr.write(
                "RealESRGAN python backend failed "
                f"(model={model}): {_err[-800:]}\n")
        return r.returncode == 0
    except Exception as e:
        sys.stderr.write(
            f"RealESRGAN python backend error (model={model}): {e}\n")
        return False


# FIX-R6-02: DAT 运行时已抽到 shared_modules/dat_runtime.py（照片/视频链
# 共用唯一事实源）。原 _DAT_MODEL_CONFIGS / _get_dat_config /
# _run_dat_pytorch 整体搬走；这里保留同名薄封装，既有调用点与单测不变。
# 修复台账：DAT-light embed_dim=60（旧值 180 全错）、补 DAT_S 条目、
# 构造必须传 split_size（默认 [2,4] 与官方权重全不符）、strict=False
# 配套 missing/unexpected 主动核对（杜绝随机权重垃圾算成功）。
from dat_runtime import (  # noqa: E402
    DAT_MODEL_CONFIGS as _DAT_MODEL_CONFIGS,
    get_dat_config as _get_dat_config,
    run_dat_file as _run_dat_file,
    is_dat_model as _is_dat_model,
)


def _run_dat_pytorch(input_path: str, output_path: str, model: str,
                     scale: int, model_dir: str, use_gpu: bool,
                     bit_depth: int = 8) -> bool:
    """Run DAT model via PyTorch backend (delegates to shared_modules).

    FIX-P12-3: bit_depth=16 → 16-bit TIFF output.
    """
    return _run_dat_file(input_path, output_path, model, scale,
                         model_dir, use_gpu, bit_depth)


def _run_gpen_pytorch(input_path: str, output_path: str, model: str,
                      scale: int, model_dir: str, use_gpu: bool,
                      bit_depth: int = 8) -> bool:
    """Run GPEN via PyTorch backend.

    GPEN = GAN Prior Embedded Network for **blind face restoration**
    （CVPR2021，上游 ``yangxy/GPEN``）。⚠️ **它不是超分模型**：
    输出与输入同分辨率，因此 `scale` 对 GPEN 没有意义
    （`_get_model_for_scale` 对它固定返回 ``GPEN-BFR-512``）。

    FIX-2026-10-20 重写 —— 原来这一路是"静默产出垃圾"：
      1. **架构错**：旧代码用 ``RRDBNet(num_block=23, scale=scale)`` 载权重，
         而 GPEN 的真架构是 ``FullGenerator``（StyleGAN2 生成器 + encoder）。
         实测用 RRDBNet 载一个结构不符的权重会得到
         missing=702 / unexpected=515，而 ``strict=False`` **不报错**，
         权重保持 PyTorch 默认随机初始化 → 输出是噪声，rc 仍为 0。
      2. **文件名错**：旧代码找 ``GPEN_x2.pth`` / ``GPEN_x4.pth``；
         上游权重实际叫 ``GPEN-BFR-512.pth``。
      3. **现在**：``shared_modules/gpen`` 的真架构 + 载入后强制
         ``missing == 0 and unexpected == 0``，否则返回 False（rc=83）。

    流程与照片链 stage4 的 CodeFormer 分支同构：facexlib 检测/对齐 →
    GPEN 修复每个 crop → 回贴。之所以不直接对整图跑：GPEN 是在
    **对齐后的 512×512 人脸 crop** 上训练的，整图跑不仅效果差，
    还会被 512 的分辨率上限卡死。

    产物布局：结果按同名写回 `output_path`，位深跟随 `bit_depth`
    （16 → TIFF zlib；8 → PNG）。
    """
    model_path = os.path.join(model_dir, "GPEN-BFR-512.pth")
    if not os.path.exists(model_path):
        # 兼容：用户可能按旧命名或别的变体放过权重
        for alt in (f"{model}.pth", "GPEN-BFR-256.pth"):
            cand = os.path.join(model_dir, alt)
            if os.path.exists(cand):
                model_path = cand
                break
        else:
            return False
    try:
        import torch
        import numpy as np
        import cv2
        import sys
        from media_io import load_image_rgb_uint8 as _ldr
        from gpen import FullGenerator

        device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        # ---- 载入：架构 + 权重必须严格配套，否则宁可不跑 ----
        gpen = FullGenerator(512, 512, 8, channel_multiplier=2, narrow=1,
                             device=device)
        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict) and ("params" in ckpt or "params_ema" in ckpt):
            inner = ckpt.get("params", ckpt.get("params_ema"))
        else:
            inner = ckpt
        missing, unexpected = gpen.load_state_dict(inner, strict=False)
        if missing or unexpected:
            # FIX-2026-10-20：这一条是本次修复的核心。旧实现用 strict=False
            # 且不看返回值，结构不符时权重全是随机初始化却照样推理。
            print(f"GPEN weights/arch mismatch: missing={len(missing)} "
                  f"unexpected={len(unexpected)} "
                  f"(first missing: {missing[:3]})", file=sys.stderr)
            return False
        gpen.eval().to(device)

        # ---- 输入：RGB（media_io 做值域自适应，绕开 PIL 读 uint16 TIFF 得全 0）----
        img_rgb = _ldr(input_path)

        # ---- 人脸检测 / 对齐 / 回贴（与 stage4 CodeFormer 同一套依赖）----
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        helper = FaceRestoreHelper(upscale_factor=1, face_size=512)
        # facexlib 的 read_image/paste 期望 BGR（cv2 约定）
        helper.read_image(img_rgb[:, :, ::-1].copy())
        helper.get_face_landmarks_5(only_center_face=False, resize=640,
                                    eye_dist_threshold=5)
        helper.align_warp_face()
        # ⚠️ 这个检查**必须在 align_warp_face() 之后**：cropped_faces 是
        # align_warp_face() 裁剪出来的产物，在它之前判断恒为空。
        # 同一个坑今天在 CodeFormer 上已经踩过一次（输出与输入逐像素相同、
        # MAE=0.000 而 rc=0），这里不能再犯。
        if not helper.cropped_faces:
            # 无人脸：GPEN 无从发挥。原样输出但不声称修复过 ——
            # 调用方（stage3）会照常记 DONE，这里至少不产垃圾。
            _write_gpen_result(img_rgb, output_path, bit_depth)
            return True

        for cropped in helper.cropped_faces:
            # cropped_faces 是 RGB uint8，官方归一化 (x/255 - 0.5)/0.5
            t = torch.from_numpy(np.asarray(cropped, dtype=np.float32) / 255.0)
            t = ((t - 0.5) / 0.5).permute(2, 0, 1).unsqueeze(0).to(device)
            try:
                with torch.no_grad():
                    out, _ = gpen(t)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as oom:
                # FIX-P1-1: release VRAM on CUDA OOM before fallback (see DAT).
                import time as _time
                try:
                    del gpen
                except Exception:
                    pass
                del t
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    _time.sleep(0.1)
                print(f"GPEN CUDA OOM for {model}: {oom}", file=sys.stderr)
                return False
            out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            restored = np.clip(out * 0.5 + 0.5, 0.0, 1.0)
            helper.add_restored_face(
                (restored * 255.0).clip(0, 255).astype(np.uint8))

        helper.get_inverse_affine(None)
        # paste 返回 BGR（与 read_image 的输入一致），转回 RGB 再落盘
        result_bgr = helper.paste_faces_to_input_image(
            upsample_img=img_rgb[:, :, ::-1].copy())
        _write_gpen_result(result_bgr[:, :, ::-1], output_path, bit_depth)
        return True
    except Exception as _ge:
        import sys
        print(f"GPEN inference failed for {model}: {_ge}", file=sys.stderr)
        return False


def _write_gpen_result(rgb: "np.ndarray", output_path: str,
                       bit_depth: int = 8) -> None:
    """把 RGB 数组按位深落盘（16 → TIFF zlib；8 → PNG）。

    FIX-P12-6 的延续：PIL 的 I;16 只支持 2D，3 通道 16-bit 必须走 tifffile；
    且 TIFF 不能用 LZW（tifffile 解 LZW 需要未装的 imagecodecs）。
    """
    import numpy as np
    if bit_depth == 16:
        out16 = np.clip(np.asarray(rgb, dtype=np.float64) * 65535.0 / 255.0,
                        0, 65535).astype(np.uint16)
        try:
            import tifffile
            tifffile.imwrite(output_path, out16, compression="zlib")
        except ImportError:
            out8 = (out16 // 257).astype(np.uint8)
            from PIL import Image
            Image.fromarray(out8).save(output_path)
    else:
        out8 = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(out8).save(output_path)


# ---------------------------------------------------------------------------
# FIX-C-12-02 / FIX-M-12-09: spawn subprocess isolation per v7.32 §3.10.18.
# DAT/GPEN PyTorch inference runs in a spawned child process so each image
# gets a fresh CUDA context (no VRAM/context accumulation across a long batch,
# and a hard child crash - segfault/CUDA corruption - is caught via exitcode
# instead of corrupting the parent). The OOM del/empty_cache path inside the
# child still runs first; spawn is the outer isolation layer. Module-level
# functions below are required for multiprocessing spawn (constraint 3).
# ---------------------------------------------------------------------------

# Default per-image spawn timeout (seconds). Model load + inference on a large
# image can take minutes on CPU; 1200s leaves headroom.
_SPAWN_INFERENCE_TIMEOUT = 1200


def _spawn_inference_target(fn, args, result_queue):
    """Spawn target: run a module-level inference function and return its bool
    result via the queue. Catches all exceptions so a hard failure in the child
    is reported as ('err', repr) rather than crashing the parent silently.
    """
    try:
        ok = fn(*args)
        result_queue.put(("ok", bool(ok)))
    except Exception as e:
        result_queue.put(("err", repr(e)))


def _run_in_spawned_process(fn, args, timeout: int = _SPAWN_INFERENCE_TIMEOUT) -> bool:
    """Run fn(*args) in a spawned child process, return its bool result.

    Returns False if the child crashes (non-zero exitcode, e.g. segfault from
    CUDA corruption), times out, or returns False/raises. This is the spawn
    isolation layer per v7.32 §3.10.18 - a corrupted CUDA context dies with the
    child instead of poisoning subsequent images.
    """
    import multiprocessing
    try:
        ctx = multiprocessing.get_context("spawn")
        q = ctx.Queue()
        proc = ctx.Process(target=_spawn_inference_target,
                           args=(fn, args, q))
        proc.start()
        proc.join(timeout=timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            return False
        # exitcode 0 = clean exit; None = still running (treat as fail); non-zero
        # = crash (segfault / CUDA error) - child died, parent is safe.
        if proc.exitcode not in (0, None):
            return False
        try:
            kind, val = q.get_nowait()
            return bool(val) if kind == "ok" else False
        except Exception:
            return False
    except Exception:
        # spawn infrastructure unavailable - fall back to in-process (caller
        # still has the OOM del/empty_cache path inside fn).
        return False


def run_realesrgan(input_path: str, output_path: str, model: str,
                   scale: int, model_dir: str = "", use_gpu: bool = True,
                   bit_depth: int = 8, use_spawn: bool = True) -> bool:
    """Run AI upscale on a single image.

    Supports Real-ESRGAN (ncnn-vulkan + Python fallback), DAT (PyTorch only),
    and GPEN (PyTorch only) per v7.32 §3.1.
    PHOTO-AUDIT-011: Added DAT PyTorch backend per v7.32 §3.1.
    FIX-P12-3: Added bit_depth parameter per v6.32 §4.2.5.
    FIX-P12-ext: Added GPEN PyTorch backend per v7.32 §3.1 / v6.32 §4.
    """
    if not model_dir:
        project_root = SCRIPT_DIR.parent.resolve()
        model_dir = str(project_root / "models")

    family = _get_model_basename(model)
    if family == "DAT":
        # DAT only supports PyTorch backend
        # FIX-C-12-02: run in a spawned child for CUDA context isolation per
        # v7.32 §3.10.18. Falls back to in-process if use_spawn=False.
        dat_args = (input_path, output_path, model, scale, model_dir, use_gpu)
        if use_spawn:
            return _run_in_spawned_process(
                _run_dat_pytorch, dat_args + (bit_depth,))
        return _run_dat_pytorch(*dat_args, bit_depth=bit_depth)
    elif family == "GPEN":
        # GPEN only supports PyTorch backend (full-body portrait model)
        # FIX-M-12-09: spawn isolation (same as DAT).
        gpen_args = (input_path, output_path, model, scale, model_dir, use_gpu)
        if use_spawn:
            return _run_in_spawned_process(
                _run_gpen_pytorch, gpen_args + (bit_depth,))
        return _run_gpen_pytorch(*gpen_args, bit_depth=bit_depth)

    # RealESRGAN: try ncnn-vulkan first, then Python fallback
    if _run_realesrgan_vulkan(input_path, output_path, model, scale,
                              model_dir, use_gpu):
        return True
    return _run_realesrgan_python(input_path, output_path, model, scale,
                                  model_dir, use_gpu, bit_depth=bit_depth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--params")
    parser.add_argument("--params-file")
    parser.add_argument("--config")
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()

    # BUG-057 support: --params-file fallback for long JSON
    if getattr(args, "params_file", None):
        with open(args.params_file, "r", encoding="utf-8") as f:
            params = json.load(f)
    else:
        params = json.loads(args.params) if args.params else {}
    log_dir = params.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, job_id=args.job_id)

    ai_enabled = params.get("ai_enabled", False)
    if not ai_enabled:
        logger.info("STAGE3_SKIP", "AI not enabled")
        return 3

    ai_banned = params.get("ai_upscale_banned", False)
    force_allow = params.get("force_allow_banned_upscale", False)
    if ai_banned and not force_allow:
        logger.info("STAGE3_BANNED", "AI upscale banned")
        return 3

    scale = params.get("upscale_scale", 2)
    max_scale = params.get("max_scale", 4)
    # v6.3: Only integer scales 2x and 4x are allowed
    if scale not in (2, 4):
        logger.error("STAGE3_INVALID",
                     f"Invalid scale: {scale}. Only integer 2x or 4x allowed.")
        return 81
    if scale > max_scale:
        logger.error("STAGE3_INVALID", f"Scale {scale} exceeds max_scale={max_scale}")
        return 81

    model = params.get("upscale_model", "realesr-general-x4v3")
    # v6.3: Anime model banned for real photos unless explicitly allowed
    if "anime" in model.lower():
        logger.error("STAGE3_ANIME_BANNED",
                     f"Anime model {model} is banned for real photos. "
                     "Use realesr-general-x4v3, RealESRGAN_x2plus/x4plus or DAT instead.")
        return 81

    # Read original dimensions from params or fallback to input file
    original_width = params.get("original_width", 0)
    original_height = params.get("original_height", 0)
    if original_width == 0 or original_height == 0:
        try:
            from PIL import Image
            with Image.open(args.input) as img:
                original_width, original_height = img.size
        except Exception:
            original_width, original_height = 0, 0

    # FIX-P12-9: v7.32 §3.10.2: Photo ban uses long_edge/short_edge (not video 1080p)
    # ORIGINAL: used target_w > 1920 or target_h > 1080 (video 1080p threshold)
    long_edge = max(original_width, original_height)
    short_edge = min(original_width, original_height)
    # Photo ban: original long_edge > 3840 or short_edge > 2160
    if long_edge > 3840 or short_edge > 2160:
        if force_allow:
            logger.critical("STAGE3_FORCE_ALLOW",
                           f"FORCE ALLOW: original {original_width}x{original_height} "
                           f"exceeds photo ban threshold (long_edge>{3840}/short_edge>{2160})")
            logger.warning("STAGE3_FORCE_ALLOW_QUALITY",
                          f"画质风险：原始分辨率 {original_width}x{original_height} 超过照片禁令阈值，"
                          f"AI 超分可能产生伪影")
        else:
            logger.warning("STAGE3_BANNED_SIZE",
                           f"Original {original_width}x{original_height} exceeds photo "
                           f"ban threshold (long_edge>{3840}/short_edge>{2160}), "
                           f"skipping AI upscale per v7.32 §3.10.2")
            return 3

    # v6.3 §5.3: Enforce traditional-first if configured
    force_traditional_first = params.get("force_traditional_first", True)
    if force_traditional_first:
        temp_dir = params.get("temp_dir", "./temp")
        job_id = args.job_id
        input_name = Path(args.input).name
        # Check stage 1 output exists
        stage1_pattern = f"{job_id}_{safe_input_stem(input_name)}_step1_preprocess*"
        stage1_candidates = glob.glob(os.path.join(temp_dir, stage1_pattern))
        if not stage1_candidates:
            logger.error("STAGE3_NO_TRADITIONAL",
                       "force_traditional_first=true but no stage1 output found")
            return 82

    # v6.3 §5.3: Ban original input pixels and intermediate pixels
    ban_original = params.get("ban_original_input_pixels", True)
    ban_intermediate = params.get("ban_intermediate_pixels", True)

    temp_dir = params.get("temp_dir", "./temp")
    # FIX-P12-3: WSL2 redirect for temp_dir per v7.32 §3.10.10
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    platform = params.get("platform", "auto")
    try:
        from checkpoint import wsl_redirect_path
        redirected_temp, _, _, _ = wsl_redirect_path(
            temp_dir, no_wsl_redirect=no_wsl_redirect,
            target_platform=platform
        )
        if redirected_temp != temp_dir:
            logger.info("STAGE3_WSL_REDIRECT",
                       f"WSL temp_dir redirect: {temp_dir} -> {redirected_temp}")
            temp_dir = redirected_temp
            params["temp_dir"] = temp_dir
    except Exception:
        pass
    job_id = args.job_id
    input_name = Path(args.input).name

    # Resolve valid input source
    stage1_output, source_type = _resolve_input_source(
        temp_dir, job_id, input_name, ban_original, ban_intermediate
    )
    if not stage1_output:
        logger.error("STAGE3_NO_INPUT", "No valid traditional-processed input found for AI upscale")
        return 82

    logger.info("STAGE3_SOURCE", f"Using {source_type} source", path=stage1_output)

    # FIX-P12-2: Supplementary ban check — intermediate file pixel size
    # Per v7.32 §3.10.2: intermediate dimensions also trigger ban
    # (photo: long_edge > 3840 or short_edge > 2160)
    try:
        from PIL import Image
        with Image.open(stage1_output) as img:
            inter_w, inter_h = img.size
    except Exception:
        inter_w, inter_h = original_width, original_height

    # FIX-P12-9: Check both long_edge and short_edge for consistency
    # with original ban threshold per v7.32 §3.10.2
    inter_long = max(inter_w, inter_h)
    inter_short = min(inter_w, inter_h)
    if inter_long > 3840 or inter_short > 2160:
        if force_allow:
            logger.critical("STAGE3_FORCE_ALLOW",
                           f"FORCE ALLOW: intermediate {inter_w}x{inter_h} "
                           f"exceeds photo ban threshold "
                           f"(long_edge>{3840}/short_edge>{2160})")
            # FIX-P12-7: Per v7.32 §3.10.2
            logger.warning("STAGE3_FORCE_ALLOW_QUALITY",
                          f"画质风险：中间文件分辨率 {inter_w}x{inter_h} 超过照片禁令阈值，"
                          f"AI 超分可能产生伪影")
        else:
            logger.warning("STAGE3_INTER_BANNED",
                           f"Intermediate {inter_w}x{inter_h} exceeds photo "
                           f"ban threshold (long_edge>{3840}/short_edge>{2160}), "
                           f"skipping AI upscale per v7.32 §3.10.2")
            return 3

    # v6.3 §5.3: Lanczos target must be "auto" (no pre-resize to AI)
    # FIX-P12-9: Changed from warning to error (exit 81) per v6.32 §4.2.5
    lanczos_target = params.get("lanczos_target", "auto")
    if lanczos_target and lanczos_target != "auto":
        logger.error("STAGE3_LANCZOS_BANNED",
                     "Lanczos pre-processing to AI is BANNED per v6.3 §5.3")
        return 81

    # Output path
    # Bug 11.14 fix: add _forced_upscale suffix when force_allow is true
    # FIX-P12-3: Use TIFF extension for 16-bit output per v6.32 §4.2.5
    bit_depth = params.get("photo_bit_depth", 8)
    out_ext = "tif" if bit_depth == 16 else "png"
    suffix = "_forced_upscale" if force_allow else ""
    param_hash = compute_param_hash(args.input, params)
    inter_name = build_intermediate_name(
        args.job_id, input_name, 3, f"upscale{suffix}", param_hash, out_ext
    )
    output_path = os.path.join(temp_dir, inter_name)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info("STAGE3_SKIP", "Already upscaled")
        return 3

    model = params.get("upscale_model", "realesr-general-x4v3")
    logger.info("STAGE3_START", "AI upscaling",
                model=model, scale=scale, input=stage1_output)

    # Upscale directory
    upscale_dir = os.path.join(temp_dir, f"{job_id}_upscaled")
    os.makedirs(upscale_dir, exist_ok=True)
    output_file = os.path.join(upscale_dir, f"{safe_input_stem(input_name)}_x{scale}.png")

    # GPU memory pre-check and fallback
    try:
        from PIL import Image
        with Image.open(stage1_output) as img:
            img_w, img_h = img.size
    except Exception:
        img_w, img_h = 1920, 1080  # fallback conservative estimate

    # BUG-056 fix: Include model weight bytes in VRAM estimation
    # Estimated VRAM: width * height * channels * 4 bytes * (2 + scale^2) + model_weight_bytes
    base_vram = img_w * img_h * 3 * 4 * (2 + scale * scale)
    model_weight_bytes = 0
    try:
        model_path = os.path.join(model_dir, f"{model}.pth")
        if os.path.exists(model_path):
            model_weight_bytes = os.path.getsize(model_path)
    except Exception:
        pass
    vram_required = int(base_vram + model_weight_bytes)
    # FIX-P0-2: gpu_memory_check expects MiB (converts available VRAM bytes
    # to MiB internally for comparison). Passing raw bytes inflated the
    # requirement ~1048576x, forcing every 1080p image to CPU. Convert here.
    # FIX-H12-03: Apply 1.2 safety margin per v7.32 §3.10.23 ("可用显存 <
    # 动态预估需求 × 1.2 触发分级降级"). Applied at the call site because
    # gpu_memory_check's contract is MiB (C-12-01 fix approach B) and
    # shared_modules/checkpoint.py is outside this fix scope.
    vram_required_mb = int(vram_required * 1.2) // (1024 * 1024)
    sufficient, backend = gpu_memory_check(vram_required_mb)

    use_gpu = True
    actual_scale = scale
    actual_model = model

    if not sufficient:
        logger.warning("STAGE3_VRAM_LOW",
                       f"GPU VRAM insufficient for {scale}x (need ~{vram_required_mb}MiB)")
        # Try downgrade, preserving model family
        if scale == 4:
            actual_scale = 2
            actual_model = _get_model_for_scale(model, 2)
            vram_required = int(img_w * img_h * 3 * 4 * (2 + actual_scale * actual_scale))
            # FIX-H12-03: 1.2 safety margin on the 4×→2× downgrade re-check too.
            vram_required_mb = int(vram_required * 1.2) // (1024 * 1024)
            sufficient, backend = gpu_memory_check(vram_required_mb)
            if sufficient:
                logger.warning("STAGE3_DOWNSCALE", f"Downgraded from 4x to 2x ({actual_model}) due to VRAM limit")
            else:
                logger.warning("STAGE3_DOWNSCALE", "4x->2x still insufficient, will try CPU")
                use_gpu = False
        else:
            use_gpu = False

    model_dir = params.get("model_dir", "")
    # FIX-P12-3: Read bit_depth from params per v6.32 §4.2.5
    bit_depth = params.get("photo_bit_depth", 8)
    # When 16-bit, use TIFF extension for output
    if bit_depth == 16:
        output_file = os.path.join(
            upscale_dir, f"{safe_input_stem(input_name)}_x{scale}.tif")

    # FIX-P12-ext: Model fallback chain per v6.32 §4 / v7.32 §3.1
    # Photo upscale priority: DAT (best for photos) -> Real-ESRGAN (general)
    # -> GPEN (full-body portrait) -> fail
    # The user's chosen model is tried first; on failure, fall through the chain.
    _PHOTO_MODEL_FALLBACK = {
        "DAT": ["RealESRGAN", "GPEN"],
        "RealESRGAN": ["DAT", "GPEN"],
        "GPEN": ["DAT", "RealESRGAN"],
    }

    # FIX-C-12-02 / M-12-09: spawn isolation toggle per v7.32 §3.10.18.
    # Default ON (user decision). Set ai_spawn_isolation=false in config to
    # fall back to in-process inference (e.g. for debugging or if spawn infra
    # is unavailable on the target platform).
    use_spawn = bool(params.get("ai_spawn_isolation", True))

    success = run_realesrgan(
        stage1_output, output_file, actual_model, actual_scale, model_dir,
        use_gpu=use_gpu, bit_depth=bit_depth, use_spawn=use_spawn
    )

    # Model fallback: if primary model fails, try alternatives in priority order
    if not success:
        primary_family = _get_model_basename(actual_model)
        fallback_families = _PHOTO_MODEL_FALLBACK.get(primary_family,
                                                       ["RealESRGAN", "DAT"])
        for alt_family in fallback_families:
            alt_model = _get_model_for_scale(
                f"{alt_family}_x{actual_scale}plus" if alt_family == "RealESRGAN"
                else f"{alt_family}_x{actual_scale}",
                actual_scale)
            logger.warning("STAGE3_MODEL_FALLBACK",
                          f"Primary model {actual_model} failed, "
                          f"trying fallback: {alt_model}")
            success = run_realesrgan(
                stage1_output, output_file, alt_model, actual_scale,
                model_dir, use_gpu=use_gpu, bit_depth=bit_depth,
                use_spawn=use_spawn
            )
            if success:
                actual_model = alt_model
                logger.info("STAGE3_FALLBACK_SUCCESS",
                           f"Fallback to {alt_model} succeeded")
                break

    if success and os.path.exists(output_file):
        # R4: 这里原来有一句**函数内** `import shutil`，它会让 shutil 成为本函数的
        # 局部变量并遮蔽文件顶部（第 19 行）的全局 import。于是当 AI 全部失败、
        # 走到"回退输出传统结果"分支（下面的 shutil.copyfile）时，
        # 该局部变量尚未绑定 → UnboundLocalError: cannot access local variable
        # 'shutil' where it is not associated with a value
        # （真实素材跑 AI 链必现，且掩盖了 AI 后端失败的真正原因）。
        # 顶部已有 import shutil，直接删掉局部 import 即可。
        # FIX-P12-10: Use copyfile (not copy2) per v7.32 §3.10.10
        shutil.copyfile(output_file, output_path)
        logger.info("STAGE3_DONE", "Upscale complete",
                    output=output_path, scale=actual_scale,
                    backend=backend if use_gpu else "cpu")

        # Bug 11.15 fix: generate preview comparison frames for forced upscale
        if force_allow:
            try:
                sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
                from preview import extract_preview_frames
                preview_dir = os.path.join(temp_dir, "previews")
                os.makedirs(preview_dir, exist_ok=True)
                preview_frames = extract_preview_frames(
                    stage1_output, preview_dir,
                    num_frames=2, force_compare_frames=True
                )
                if preview_frames:
                    logger.info("STAGE3_PREVIEW", f"Generated comparison preview: {preview_frames}")
            except Exception as e:
                logger.warning("STAGE3_PREVIEW_FAIL", f"Preview generation failed: {e}")
        return 0

    # FIX-M12-10: Per v7.32 §3.10.23, when all AI backends fail (including
    # CPU), skip AI and emit the traditional (stage1) result rather than
    # failing the job with exit 83. Return 3 (skip) so the main loop treats
    # this stage as skipped and downstream stages consume the traditional
    # output. Only fall back to 83 if even the traditional copy fails.
    logger.warning("STAGE3_AI_SKIP_TRADITIONAL",
                   "All AI backends failed (incl. CPU); emitting traditional "
                   "stage1 output and skipping AI per v7.32 §3.10.23")
    try:
        shutil.copyfile(stage1_output, output_path)
        return 3
    except Exception as e:
        logger.error("STAGE3_FAILED",
                     f"AI upscaling failed and traditional fallback copy "
                     f"failed: {e}")
        return 83


if __name__ == "__main__":
    # FIX-P11-5: main() must return explicit int exit code per v7.32 §3.5
    # ORIGINAL: sys.exit(main() or 0) — "or 0" masks None returns
    rc = main()
    if rc is None:
        rc = 86  # validation failure — main() must always return explicit exit code
    sys.exit(rc)
