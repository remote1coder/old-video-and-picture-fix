#!/usr/bin/env python3
"""
Stage 3: AI Upscaling
Performs integer-scale AI super-resolution (2x or 4x only).
GPU memory pre-check with automatic fallback (4x->2x->CPU).

v6.3 Compliance:
- 256-frame sliding window
- VFR (Variable Frame Rate) support via PTS
- AI upscale banned for 720x480 DV
- Proper Real-ESRGAN CLI invocation (ncnn-vulkan or Python API)

P8 Fixes Applied:
- P8-1: Original input dims in dual-basis ban per v7.32 §3.10.2
- P8-2: Dual-basis three-layer ban check per v7.32 §3.10.2
- P8-3: Force override WARNING log per v7.32 §3.10.2 L3
- P8-4: Exit code 1 -> 83 (GPU error, per v7.32 §3.5)
- P8-5: WSL2 temp_dir/preview_dir redirect per v6.32 §7.3
- P8-6: ncnn-vulkan tried first, .pth fallback only
- P8-7: OOM downgrade only on actual OOM (three-state return)
- P8-8: VFR PTS preservation + bitdepth self-check
- P8-9: Batch model loading for Python API path
- P8-10: pathlib.Path consistent usage
- P8-11: ncnn-vulkan OOM detection from stderr per v6.32 §7.14
- P8-12: vs_bit_depth in param_hash core_params per v7.32 §3.10.19
- P8-13: pts_time first (FFmpeg>=7.0), fallback pkt_pts_time
- P8-14: VFR seek uses actual PTS, not CFR assumption
- P8-15: Enforce checkpoint.compute_param_hash (no local fallback)
- P8-16: MPS unified memory check per v6.32 §7.14
"""

import os
import sys
import json
import argparse
import subprocess
import re
import shutil
import time
import errno
import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Union

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402

try:  # NEW-V-06: 取消/中断守卫（KeyboardInterrupt 继承 BaseException，
    # 本文件所有 `except Exception` 都抓不到它——不装守卫时 Ctrl+C 会把
    # 半截 mkv 留在 temp 里被下游 stage 复用）
    import cancel_guard as _cancel_guard
except Exception:  # pragma: no cover - 缺失时不阻断 stage，仅失去该保护
    _cancel_guard = None

# FIX-R6-02: AI 推理子进程解释器解析。视频链编排器是 Portable R72
# （无 torch/realesrgan/basicsr/gfpgan），AI stage 历来用 sys.executable
# 生成推理脚本——R72 下必然 ImportError，`--ai-enabled` 整条路从未跑通
# （2026-10-18 实测：依赖自检 exit 80，job 从未执行）。解析规则：
# sys.executable 有 torch 就用它（照片链场景），否则回退项目内
# .venv_r4（探针子进程验证），再不行返回 None → AI 显式失败。
try:
    import ai_runtime as _ai_runtime
except Exception:  # pragma: no cover
    _ai_runtime = None

_AI_PYTHON_CACHE = {}


def _resolve_ai_python(logger=None) -> Optional[str]:
    """解析 AI 推理子进程解释器（带缓存；None = AI 不可用）。"""
    if "py" in _AI_PYTHON_CACHE:
        return _AI_PYTHON_CACHE["py"]
    py = None
    if _ai_runtime is not None:
        try:
            py = _ai_runtime.resolve_ai_interpreter(logger=logger)
        except Exception:
            py = None
    _AI_PYTHON_CACHE["py"] = py
    return py


def estimate_gpu_memory(width: int, height: int, scale: int,
                        model_name: str) -> int:
    """Estimate GPU memory requirement in bytes."""
    base = width * height * 3 * 4 * 1 * (2 + scale * scale)
    model_weights = {
        "realesr-general-x4v3": 18 * 1024 * 1024,   # ncnn .bin ~10-18MB
        "RealESRGAN_x2plus": 18 * 1024 * 1024,       # ncnn .bin
        "RealESRGAN_x4plus": 18 * 1024 * 1024,       # ncnn .bin
        "DAT_x2": 18 * 1024 * 1024,                   # ncnn .bin
        "DAT_x4": 18 * 1024 * 1024,                   # ncnn .bin
        "DAT_2_x2": 18 * 1024 * 1024,
        "DAT_2_x4": 18 * 1024 * 1024,
    }
    weight = model_weights.get(model_name, 64 * 1024 * 1024)
    return base + weight


def _query_nvidia_smi_free_bytes(logger=None):
    """FIX-R8-F1: Three-tier nvidia-smi fallback per v7.32 §3.10.23 / v6.32 §7.14.

    Unit contract (baseline §G): nvidia-smi returns MiB → ×1024×1024 to bytes;
    pynvml returns bytes. Returns bytes (matching check_gpu_memory's byte
    contract). Never mix MiB/bytes — a mismatch inflates 1048576×.
      1) --query-gpu=memory.free --format=csv,noheader,nounits (MiB)
      2) Full-table MiB field parse
      3) pynvml library
    Returns None if all three fail.
    """
    # Level 1: --query-gpu
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, errors="replace", timeout=10, shell=False
        )
        if r.returncode == 0 and r.stdout.strip():
            free_mib = float(r.stdout.strip().split("\n")[0].strip())
            return int(free_mib * 1024 * 1024)  # MiB → bytes
    except Exception as e:
        if logger:
            logger.warning("STAGE3_NVSMI_QUERY_FAIL",
                           f"nvidia-smi --query-gpu failed: {e}")
    # Level 2: full-table MiB field
    try:
        r = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, errors="replace", timeout=10, shell=False
        )
        if r.returncode == 0:
            mib_match = re.search(r"(\d+)\s*MiB", r.stdout)
            if mib_match:
                free_mib = float(mib_match.group(1))
                if logger:
                    logger.warning("STAGE3_NVSMI_TABLE_FALLBACK",
                                   "Falling back to full table MiB parse")
                return int(free_mib * 1024 * 1024)
    except Exception as e:
        if logger:
            logger.warning("STAGE3_NVSMI_TABLE_FAIL",
                           f"nvidia-smi table parse failed: {e}")
    # Level 3: pynvml
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if logger:
            logger.warning("STAGE3_NVSMI_PYNVML_FALLBACK",
                           "Falling back to pynvml")
        return int(info.free)
    except Exception as e:
        if logger:
            logger.warning("STAGE3_NVSMI_PYNVML_FAIL",
                           f"pynvml failed: {e}")
    return None


def check_gpu_memory(required_bytes: int, logger=None) -> tuple:
    """Check GPU availability. Returns (sufficient: bool, backend: str).

    Backend priority: cuda > rocm > mps > cpu.
    FIX-R8-F1: three-tier nvidia-smi fallback (query-gpu → table MiB → pynvml).
    FIX-R8-F2: each backend downgrade logs WARNING per v7.32 §3.5 / §3.10.23.
    Bug 13.30/13.31 fix: Detect PyTorch version, ROCm, and MPS.
    """
    # 1. nvidia-smi (native CUDA) — three-tier fallback
    free_bytes = _query_nvidia_smi_free_bytes(logger)
    if free_bytes is not None:
        if free_bytes >= required_bytes * 1.2:
            return True, "cuda"
        if logger:
            logger.warning("STAGE3_BACKEND_DOWNGRADE",
                           "nvidia-smi reports insufficient VRAM, trying PyTorch backends",
                           target="torch.cuda")
    else:
        if logger:
            logger.warning("STAGE3_BACKEND_DOWNGRADE",
                           "nvidia-smi unavailable (all 3 tiers failed), trying PyTorch backends",
                           target="torch.cuda")

    # 2. PyTorch backends (cuda / rocm / mps)
    try:
        import torch
        torch_ver = torch.__version__

        # Parse version for >=1.10 check (P8-4)
        ver_parts = torch_ver.split("+")[0].split(".")
        ver_major = int(ver_parts[0])
        ver_minor = int(ver_parts[1]) if len(ver_parts) > 1 else 0
        has_mem_get_info = (ver_major > 1) or (ver_major == 1 and ver_minor >= 10)

        # CUDA
        if torch.cuda.is_available():
            if has_mem_get_info:
                try:
                    free, total = torch.cuda.mem_get_info()
                    if free >= required_bytes * 1.2:
                        return True, "cuda"
                    if logger:
                        logger.warning("STAGE3_BACKEND_DOWNGRADE",
                                       "torch.cuda mem_get_info reports insufficient VRAM, trying ROCm",
                                       target="rocm")
                except Exception as e:
                    if logger:
                        logger.warning("STAGE3_BACKEND_DOWNGRADE",
                                       f"torch.cuda mem_get_info failed ({e}), trying ROCm",
                                       target="rocm")
            else:
                if logger:
                    logger.warning("STAGE3_PYTORCH_OLD",
                                   f"PyTorch {torch_ver} < 1.10, "
                                   "skipping torch.cuda.mem_get_info()")

        # ROCm (AMD GPU)
        if getattr(torch.version, "hip", None) is not None:
            if has_mem_get_info:
                try:
                    free, total = torch.cuda.mem_get_info()
                    if free >= required_bytes * 1.2:
                        return True, "rocm"
                    if logger:
                        logger.warning("STAGE3_BACKEND_DOWNGRADE",
                                       "ROCm insufficient, trying MPS",
                                       target="mps")
                except Exception as e:
                    if logger:
                        logger.warning("STAGE3_BACKEND_DOWNGRADE",
                                       f"ROCm mem_get_info failed ({e}), trying MPS",
                                       target="mps")

        # Apple Metal (MPS)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # FIX-P8-11: Check unified memory availability for MPS
            # ORIGINAL: Assumed sufficient without any memory check
            # Per v6.32 §7.14: must verify available memory >= required * 1.2
            try:
                import os as _os
                if hasattr(_os, 'sysconf'):
                    total_mem = (_os.sysconf('SC_PAGE_SIZE')
                                 * _os.sysconf('SC_PHYS_PAGES'))
                    if total_mem * 0.7 >= required_bytes * 1.2:
                        return True, "mps"
                    if logger:
                        logger.warning("STAGE3_BACKEND_DOWNGRADE",
                                       "MPS unified memory insufficient, falling back to CPU",
                                       target="cpu")
                else:
                    # Windows/other: assume sufficient (no API available)
                    return True, "mps"
            except Exception as e:
                if logger:
                    logger.warning("STAGE3_BACKEND_DOWNGRADE",
                                   f"MPS check failed ({e}), falling back to CPU",
                                   target="cpu")
            if logger:
                logger.warning("STAGE3_BACKEND_DOWNGRADE",
                               "MPS unavailable or insufficient, falling back to CPU",
                               target="cpu")
            return False, "cpu"

        # CPU-only PyTorch detection
        if "cpu" in torch_ver.lower():
            if logger:
                logger.warning("STAGE3_BACKEND_DOWNGRADE",
                               "CPU-only PyTorch, no GPU backend available",
                               target="cpu")
            return False, "cpu"
    except Exception as e:
        if logger:
            logger.warning("STAGE3_BACKEND_DOWNGRADE",
                           f"PyTorch backend detection failed ({e}), falling back to CPU",
                           target="cpu")

    if logger:
        logger.warning("STAGE3_BACKEND_DOWNGRADE",
                       "No GPU backend available, using CPU",
                       target="cpu")
    return False, "cpu"


def _get_model_for_scale(model: str, scale: int) -> str:
    """Get correct model name for a given scale, preserving model family.

    FIX-P1-15: realesr-general-x4v3 is a 4× ncnn model; for 2× output it
    stays the same model (ncnn -s 2 downscales). DAT_2 is default.
    """
    if model.startswith("realesr-general"):
        return "realesr-general-x4v3"
    elif model.startswith("RealESRGAN"):
        return f"RealESRGAN_x{scale}plus"
    elif model.startswith("DAT_2"):
        return f"DAT_2_x{scale}"
    elif model.startswith("DAT"):
        return f"DAT_x{scale}"
    return "realesr-general-x4v3"


# FIX-P8-6+P8-1: Restructured run_realesrgan
# ORIGINAL: Checked .pth first (blocking ncnn-vulkan), returned bool,
#           had broken OOM cleanup in finally block (del model on string)
# DEPRECATED: Use run_realesrgan_batch instead.
# This single-window function is kept for unit testing only.
def run_realesrgan(input_path: str, output_dir: str, model: str,
                   scale: int, gpu_id: int = 0) -> Union[bool, str]:
    """Run Real-ESRGAN upscaling. Returns True/False/"OOM".

    FIX-P8-6: ncnn-vulkan tried first (doesn't need .pth), then Python API.
    FIX-P8-1: OOM detection via child process exit code 83 per v6.32 §7.14.
    FIX-R6-02: DAT 家族分流到 _run_dat_batch（RealESRGANer 加载不了 DAT
    架构；ncnn-vulkan 的 DAT 需要 .bin/.param，本项目只有 .pth）。
    """
    model_dir = SCRIPT_DIR.parent.resolve() / "models"

    # FIX-R6-02: DAT 家族（单窗口 = 单窗口批处理）
    try:
        from dat_runtime import is_dat_model as _is_dat_model
    except Exception:  # pragma: no cover
        _is_dat_model = lambda m: m.lower().startswith("dat")
    if _is_dat_model(model):
        return _run_dat_batch([(input_path, output_dir)], model, scale,
                              gpu_id=gpu_id)

    # FIX-P8-6: 1. Try realesrgan-ncnn-vulkan CLI first (doesn't need .pth)
    vulkan_bin = shutil.which("realesrgan-ncnn-vulkan")
    if vulkan_bin:
        cmd = [
            vulkan_bin,
            "-i", input_path,
            "-o", output_dir,
            "-n", model,
            "-s", str(scale),
            "-g", str(gpu_id),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                            timeout=3600, shell=False)
            if r.returncode == 0:
                return True
        except Exception:
            pass

    # FIX-P8-6: 2. Python API fallback - needs .pth
    model_path = str(model_dir / f"{model}.pth")
    if not Path(model_path).exists():
        return False

    # FIX-R6-02: 推理子进程解释器。R72 编排器无 torch——必须换 .venv_r4，
    # 否则这里 sys.executable 起的脚本第一行 import torch 就炸（实测：
    # --ai-enabled 在 R72 下从未跑通，依赖自检直接 exit 80）。
    ai_py = _resolve_ai_python()
    if ai_py is None:
        return False

    try:
        env = os.environ.copy()
        # 【第七轮·关键修复】**必须清掉编排器的 Python 环境变量**。
        #
        # 父进程是 Portable R72（VapourSynth 环境），它的 PYTHONPATH 指向
        # `tools_r4/vs_portable/r72/Lib/site-packages`。`os.environ.copy()`
        # 会把这一套原封不动传给 **另一个解释器** `.venv_r4` —— 于是
        # .venv_r4 的 Python 去 R72 的目录里找包，`import torch` 直接死锁：
        # 进程 CPU 0.16 秒、内存 11MB、`<no Python frame>`，几十分钟不动。
        #
        # 实测对照（同一个 .venv_r4）：
        #   干净环境            -> torch 2.11.0+cu128 cuda True，1.8 秒
        #   带 R72 的 PYTHONPATH-> 卡死（60 秒超时，无 Python 栈帧）
        #
        # 这就是"AI 子进程卡死、一次两小时"的真正原因 —— 不是模型慢、
        # 不是 CUDA 驱动，是环境变量串了。之前所有"并发抢 GPU"的推断
        # 都是表象。
        for _bad_env in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
                         "PYTHONEXECUTABLE", "PYTHONUSERBASE",
                         "PYTHONNOUSERSITE"):
            env.pop(_bad_env, None)
        if gpu_id < 0:
            env["CUDA_VISIBLE_DEVICES"] = ""
        # FIX-R6-03: 架构选择。本项目安装的 realesrgan 版本 **没有 model_name
        # 参数**：RealESRGANer(scale, model_path, dni_weight, model, tile,
        # tile_pad, pre_pad, half, device, gpu_id) 要求调用方传入已实例化的
        # nn.Module（内部 load_state_dict(..., strict=True)）。
        # 原脚本传 `prepad=10`（正确名是 pre_pad）且不给 model：
        # 子进程 1.6s 内 TypeError 退出 → 被当成 non-OOM 失败 → stage3 rc=83。
        # 2026-10-18 实测：realesr-general-x4v3 2× 矩阵首案即此崩法。
        if "general-x4v3" in model:
            # realesr-general-x4v3 / realesr-general-wdn-x4v3：SRVGGNetCompact 4×
            native_scale = 4
            arch_code = (
                "from realesrgan.archs.srvgg_arch import SRVGGNetCompact\n"
                "arch = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, "
                "num_feat=64, num_conv=32, upscale=4, act_type='prelu')\n")
        elif "x2plus" in model:
            native_scale = 2
            arch_code = (
                "from basicsr.archs.rrdbnet_arch import RRDBNet\n"
                "arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, "
                "num_block=23, num_grow_ch=32, scale=2)\n")
        else:  # RealESRGAN_x4plus 及其它 RRDB 4× 系
            native_scale = 4
            arch_code = (
                "from basicsr.archs.rrdbnet_arch import RRDBNet\n"
                "arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, "
                "num_block=23, num_grow_ch=32, scale=4)\n")
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                          delete=False) as f:
            script_path = f.name
            # FIX-P8-1: Temp script with OOM detection (exit code 83)
            f.write(
                "import sys\n"
                "from realesrgan import RealESRGANer\n"
                "import torch\n"
                "from PIL import Image\n"
                "import os\n"
                "import glob\n"
                "import numpy as np\n"
                # COL-03：16bit 帧必须用 cv2 读写（PIL 会静默降 8bit）
                "import cv2\n"
                "import gc\n"
                f"scale = {scale}\n"
                f"model_path = {repr(model_path)}\n"
                f"input_path = {repr(input_path)}\n"
                f"output_dir = {repr(output_dir)}\n"
                f"use_half = not torch.cuda.is_available() "
                f"if {gpu_id} >= 0 else False\n"
                "os.makedirs(output_dir, exist_ok=True)\n"
                f"{arch_code}"
                "try:\n"
                # outscale=scale：x4v3 是 4× 网络，出 2× 时由 RealESRGANer
                # 内部 LANCZOS4 缩放（与官方 inference 脚本一致）
                f"    upsampler = RealESRGANer(scale={native_scale}, "
                "model_path=model_path, model=arch, pre_pad=10, half=use_half)\n"
                "    for fp in sorted(glob.glob("
                "os.path.join(input_path, '*.png'))):\n"
                "        # COL-01（第七轮）：RealESRGANer 的官方约定是"
                "**输入 BGR、输出 BGR**——内部开头即 "
                "cv2.cvtColor(img, COLOR_BGR2RGB)，结尾用 "
                "img[[2,1,0]] 再反转回来。\n"
                "        # 旧代码把 PIL 的 RGB 直接传进去、又按 RGB "
                "保存返回的 BGR：两次错配恰好抵消，\n"
                "        # 最终颜色看着是对的，但**模型实际处理的是 R/B "
                "互换的图**。\n"
                # COL-03（第七轮）：**必须用 cv2 读 16bit**。
                # 旧代码 `Image.open(fp).convert('RGB')` 会把 rgb48 的
                # 16bit PNG **静默降成 uint8**（实测 PIL mode=RGB，
                # dtype=uint8），低位信息直接丢掉——而 stage3 抽帧用的是
                # `-pix_fmt rgb48`、回编码用的是 `yuv420p16le`，整条链
                # 本来是为 16bit 设计的。
                # cv2.IMREAD_UNCHANGED 保留 uint16，且 cv2 本身就是 **BGR**，
                # 与 RealESRGANer 的官方输入约定天然一致，不需要再换通道。
                "        _arr = cv2.imread(fp, cv2.IMREAD_UNCHANGED)\n"
                "        if _arr is None:\n"
                "            _arr = np.array(Image.open(fp).convert('RGB'))[:, :, ::-1].copy()\n"
                # enhance() **已经**把输出还原到输入量级并转好类型：
                #   16bit 输入 -> (output_img*65535).round().astype(uint16)
                #   8bit  输入 -> (output_img*255).round().astype(uint8)
                # 所以这里直接写回，绝不能再乘一次（实测乘 65535 会让
                # 数值爆掉、clip 后整帧只剩黑白两值）。
                "        output, _ = upsampler.enhance(_arr, outscale=scale)\n"
                "        cv2.imwrite(os.path.join(output_dir, os.path.basename(fp)), output)\n"
                "    del upsampler\n"
                "    gc.collect()\n"
                "    if torch.cuda.is_available():\n"
                "        torch.cuda.empty_cache()\n"
                # FIX-N+1-4 (N+1 轮核验): merge OOM except clauses into one
                # version-safe RuntimeError handler. ORIGINAL used
                # `except torch.cuda.OutOfMemoryError:` as a sibling clause; on
                # PyTorch<1.13 that class doesn't exist, so evaluating the except
                # expression raised AttributeError, losing the original OOM and
                # skipping the RuntimeError string-match fallback (the subprocess
                # then exited non-83, breaking OOM downgrade). Now a single
                # except RuntimeError (OutOfMemoryError is a RuntimeError
                # subclass) detects OOM via hasattr-guarded isinstance
                # (non-English OOM on >=1.13) OR English string match
                # (<1.13 fallback); non-OOM RuntimeErrors re-raise. Per baseline
                # §G, OOM must NOT be swallowed by `except Exception`.
                "except RuntimeError as _e:\n"
                "    _is_oom = (hasattr(torch.cuda, 'OutOfMemoryError')\n"
                "               and isinstance(_e, torch.cuda.OutOfMemoryError))\n"
                "    if (not _is_oom) and ('out of memory' in str(_e).lower()):\n"
                "        _is_oom = True\n"
                "    if _is_oom:\n"
                "        try:\n"
                "            del upsampler\n"
                "        except NameError:\n"
                "            pass\n"
                "        gc.collect()\n"
                "        if torch.cuda.is_available():\n"
                "            torch.cuda.empty_cache()\n"
                "        import time as _t; _t.sleep(0.1)\n"
                "        sys.exit(83)  # FIX-P8-1: GPU OOM per v6.32 §7.14\n"
                "    raise\n"
            )
        # FIX-R6-02: 用 AI 解释器（.venv_r4）而不是 sys.executable
        cmd = [ai_py, script_path]

        # B9-后续（第七轮）：**子进程必须受取消守卫管理**，且超时不能是
        # 固定的 1 小时。
        #
        # 实测踩到的三个问题：
        #   1. 旧代码用裸 `subprocess.run(timeout=3600)` —— 子进程一旦
        #      卡住（CUDA 初始化死锁、驱动 hang、模型加载阻塞），父进程
        #      要干等**整整 1 小时**才超时；超时后 main 若重试，又是 1 小时
        #      （实测观察到"卡了约 2 小时"正是两轮超时）。
        #   2. `subprocess.run` 不受取消守卫管辖：Ctrl+C/取消只杀父进程，
        #      AI 子进程继续占着 GPU 写文件，守卫删掉的半成品会被它重建
        #      （与 RV-12 同族）。其他 stage 都用了 run_managed，只有这里
        #      漏了。
        #   3. 超时抛出 `TimeoutExpired` 时没有留下任何说明 —— 日志里只
        #      看到"卡住"，不知道是超时还是崩了。
        #
        # 现在：走 run_managed（可取消、超时会 kill+reap 子进程），
        # 超时按**窗口总帧数**动态给（每帧 30 秒的保守上限，最少 300 秒），
        # 并在超时时写明确日志与 stderr 尾部。
        _total_frames = 0
        for _fd, _od in window_dirs:
            try:
                _total_frames += len([n for n in os.listdir(_fd)
                                      if n.endswith(".png")])
            except OSError:
                pass
        _ai_timeout = max(300, min(7200, _total_frames * 30))
        if logger:
            logger.info("STAGE3_AI_SUBPROC_START",
                        "Launching AI subprocess under cancel-guard",
                        frames=_total_frames, timeout_s=_ai_timeout,
                        interpreter=ai_py)
        try:
            if _cancel_guard is not None:
                r = _cancel_guard.run_managed(
                    cmd, capture_output=True, text=True, timeout=_ai_timeout,
                    shell=False, env=env, logger=logger)
            else:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   errors="replace", timeout=_ai_timeout,
                                   shell=False, env=env)
        except subprocess.TimeoutExpired as _te:
            if logger:
                logger.error(
                    "STAGE3_AI_SUBPROC_TIMEOUT",
                    f"AI subprocess exceeded {_ai_timeout}s "
                    f"({_total_frames} frames). This is NOT a silent hang: "
                    "the child was killed and reaped. Common causes: CUDA "
                    "init deadlock (several processes sharing one GPU), "
                    "driver hang, or a model that is far slower than "
                    "expected.",
                    frames=_total_frames, timeout_s=_ai_timeout,
                    stdout_tail=str(getattr(_te, "stdout", "") or "")[-800:],
                    stderr_tail=str(getattr(_te, "stderr", "") or "")[-800:])
            if _trace_fh is not None:
                try:
                    _trace_fh.close()
                except Exception:
                    pass
            return False
        # FIX-R6-03: 失败必须留证据。原实现无条件删脚本 + 只返回 False，
        # 子进程真实 traceback（参数非法/架构不匹配/import 失败）全部丢失
        # —— 2026-10-18 实测因此多绕两圈才定位到 prepad kwarg。
        if r.returncode == 83:
            try:
                Path(script_path).unlink(missing_ok=True)
            except Exception:
                pass
            return "OOM"
        if r.returncode != 0:
            if logger:
                logger.error("STAGE3_AI_SUBPROC_FAIL",
                             f"AI subprocess exited {r.returncode}; "
                             f"script kept at {script_path}",
                             stderr_tail=(r.stderr or "")[-1500:])
            return False
        try:
            # FIX-P8-10: Use Path.unlink instead of os.unlink
            Path(script_path).unlink(missing_ok=True)
        except Exception:
            pass
        return True
    except Exception as e:
        if logger:
            logger.error("STAGE3_AI_LAUNCH_FAIL", str(e))
        return False


# FIX-R6-02: DAT 系批处理（2026-10-18 全选项矩阵验收新增）。
# DAT 家族（DAT/DAT-2/DAT-S/DAT-light）的架构与官方配置表在
# shared_modules/dat_runtime.py（唯一事实源，照片链同一份）。
# 子进程用 AI 解释器（.venv_r4，有 torch+basicsr+timm+einops）；
# 编排器 R72 只负责拼参数、起进程、收三态结果。
def _run_dat_batch(window_dirs: List[Tuple[str, str]], model: str,
                   scale: int, gpu_id: int = 0,
                   logger=None) -> Union[bool, str]:
    """Run DAT upscaling on all windows. Returns True/False/"OOM".

    帧级循环放在子进程里（与 RealESRGAN 分支同样的 spawned-script 模式），
    模型每个窗口重复加载——DAT 参数量小（DAT-light 573K / DAT-2 11.2M），
    且窗口数通常个位数，加载开销可接受；换来的是 OOM 隔离与降级链统一。
    """
    ai_py = _resolve_ai_python(logger)
    if ai_py is None:
        if logger:
            logger.error("STAGE3_AI_INTERPRETER_MISSING",
                         "找不到具备 torch/realesrgan/basicsr 的 AI 解释器"
                         "（视频编排器 R72 不含这些包）")
        return False

    model_dir = str(SCRIPT_DIR.parent.resolve() / "models")
    shared_dir = str(SCRIPT_DIR.parent.resolve() / "shared_modules")
    window_repr = repr([(d, o) for d, o in window_dirs])

    try:
        env = os.environ.copy()
        # 【第七轮·关键修复】**必须清掉编排器的 Python 环境变量**。
        #
        # 父进程是 Portable R72（VapourSynth 环境），它的 PYTHONPATH 指向
        # `tools_r4/vs_portable/r72/Lib/site-packages`。`os.environ.copy()`
        # 会把这一套原封不动传给 **另一个解释器** `.venv_r4` —— 于是
        # .venv_r4 的 Python 去 R72 的目录里找包，`import torch` 直接死锁：
        # 进程 CPU 0.16 秒、内存 11MB、`<no Python frame>`，几十分钟不动。
        #
        # 实测对照（同一个 .venv_r4）：
        #   干净环境            -> torch 2.11.0+cu128 cuda True，1.8 秒
        #   带 R72 的 PYTHONPATH-> 卡死（60 秒超时，无 Python 栈帧）
        #
        # 这就是"AI 子进程卡死、一次两小时"的真正原因 —— 不是模型慢、
        # 不是 CUDA 驱动，是环境变量串了。之前所有"并发抢 GPU"的推断
        # 都是表象。
        for _bad_env in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
                         "PYTHONEXECUTABLE", "PYTHONUSERBASE",
                         "PYTHONNOUSERSITE"):
            env.pop(_bad_env, None)
        if gpu_id < 0:
            env["CUDA_VISIBLE_DEVICES"] = ""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                          delete=False) as f:
            script_path = f.name
            f.write(
                "import sys, os, glob\n"
                f"sys.path.insert(0, {repr(shared_dir)})\n"
                "from dat_runtime import run_dat_file\n"
                f"model = {repr(model)}\n"
                f"scale = {scale}\n"
                f"model_dir = {repr(model_dir)}\n"
                f"use_gpu = {gpu_id >= 0}\n"
                f"windows = {window_repr}\n"
                "rc = 0\n"
                "for frames_dir, output_dir in windows:\n"
                "    os.makedirs(output_dir, exist_ok=True)\n"
                "    for fp in sorted(glob.glob("
                "os.path.join(frames_dir, '*.png'))):\n"
                "        out = os.path.join(output_dir, os.path.basename(fp))\n"
                "        if not run_dat_file(fp, out, model, scale, "
                "model_dir, use_gpu, 8):\n"
                "            print('DAT_FRAME_FAIL ' + fp, flush=True)\n"
                "            rc = 1\n"
                "sys.exit(rc)\n"
            )
        # FIX-R6-02: AI 解释器（不是 sys.executable）
        cmd = [ai_py, script_path]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                            timeout=7200, shell=False, env=env)
        try:
            Path(script_path).unlink(missing_ok=True)
        except Exception:
            pass
        # run_dat_file 的 CUDA OOM 会在 stderr 打 "DAT CUDA OOM"
        # ——映射成三态契约的 "OOM"，让调用方的 4x→2x→CPU 降级链生效
        if r.returncode != 0 and "DAT CUDA OOM" in (r.stderr or ""):
            return "OOM"
        return r.returncode == 0
    except Exception:
        return False


# FIX-P8-9: Batch model loading for Python API path
# ORIGINAL: Per-window model loading via run_realesrgan (slow for Python API)
def _plan_frame_transport(window_dirs, params, logger=None):
    """探测首帧尺寸并按可用内存决定传输方式（3a：只记日志，不改变行为）。

    返回 plan dict（transport / buffer_frames / reason / ...）；
    探测失败时返回 png 决策，绝不抛异常——这只是观测，不能弄坏跑批。
    """
    try:
        import glob
        import os
        sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "shared_modules"))
        from frame_transport import plan_transport
    except Exception as _e:                                   # pragma: no cover
        if logger:
            logger.warning("STAGE3_FRAME_TRANSPORT_PROBE_FAIL", str(_e))
        return None

    # 取第一个窗口的第一张 PNG，读尺寸
    frame_bytes = 0
    probe_desc = "unknown"
    try:
        for frames_dir, _out in window_dirs:
            pngs = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
            if not pngs:
                continue
            import cv2
            arr = cv2.imread(pngs[0], cv2.IMREAD_UNCHANGED)
            if arr is None:
                continue
            h, w = arr.shape[0], arr.shape[1]
            ch = 1 if arr.ndim == 2 else arr.shape[2]
            frame_bytes = int(w) * int(h) * int(ch) * arr.dtype.itemsize
            probe_desc = f"{w}x{h}x{ch}@{arr.dtype.itemsize * 8}bit"
            break
    except Exception as _e:                                   # pragma: no cover
        if logger:
            logger.warning("STAGE3_FRAME_TRANSPORT_PROBE_FAIL", str(_e))
        return None

    plan = plan_transport(frame_bytes,
                          int(params.get("ai_frame_buffer_frames", 256) or 256),
                          params)
    if logger:
        logger.info("STAGE3_FRAME_TRANSPORT_PLAN",
                    f"frame transport decided: {plan['transport']}",
                    transport=plan["transport"],
                    buffer_frames=plan["buffer_frames"],
                    frame=probe_desc,
                    available_gb=(round(plan["available_bytes"] / 2**30, 2)
                                  if plan.get("available_bytes") else None),
                    reason=plan["reason"])
    return plan


def run_realesrgan_batch(window_dirs: List[Tuple[str, str]], model: str,
                         scale: int, gpu_id: int = 0,
                         logger=None, params: Optional[dict] = None
                         ) -> Union[bool, str]:
    """Run Real-ESRGAN on all windows in a single model-load session.

    FIX-P8-9: Loads model once for Python API path, processing all windows.
    ncnn-vulkan still per-window (loads fast natively).
    Returns True/False/"OOM".

    FIX-R6-02: DAT 系模型（--ai-model DAT_x2/DAT_2_x2/...）走独立分支
    _run_dat_batch——DAT 不是 RealESRGANer 能加载的架构（RRDBNet），
    且 ncnn-vulkan 的 DAT 支持需要 .bin/.param（本项目 models/ 只有
    .pth）。DAT 分支与 RealESRGAN 分支返回同一套三态契约
    （True/False/"OOM"），调用方的 OOM 降级链原样适用。
    """
    model_dir = SCRIPT_DIR.parent.resolve() / "models"

    # 3a：只探测并记录传输方式决策，暂不改变行为
    _plan_frame_transport(window_dirs, params or {}, logger)

    # FIX-R6-02: DAT 家族分流（DAT / DAT-2 / DAT-S / DAT-light）
    try:
        from dat_runtime import is_dat_model as _is_dat_model
    except Exception:  # pragma: no cover
        _is_dat_model = lambda m: m.lower().startswith("dat")
    if _is_dat_model(model):
        return _run_dat_batch(window_dirs, model, scale,
                              gpu_id=gpu_id, logger=logger)

    # ncnn-vulkan: per-window (ncnn loads fast)
    vulkan_bin = shutil.which("realesrgan-ncnn-vulkan")
    if vulkan_bin:
        all_ok = True
        got_oom = False
        for frames_dir, out_dir in window_dirs:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            cmd = [vulkan_bin, "-i", frames_dir, "-o", out_dir,
                   "-n", model, "-s", str(scale), "-g", str(gpu_id)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                                timeout=3600, shell=False)
                if r.returncode != 0:
                    all_ok = False
                    # FIX-P8-3: Detect ncnn-vulkan OOM from stderr
                    # ORIGINAL: Only returned bool, OOM downgrade never triggered
                    # Per v6.32 §7.14: OOM must trigger 4x→2x→CPU downgrade
                    stderr_lower = (r.stderr or "").lower()
                    if ("out of memory" in stderr_lower or
                        ("cuda" in stderr_lower and "error" in stderr_lower) or
                        ("vulkan" in stderr_lower and "memory" in stderr_lower)):
                        got_oom = True
            except Exception:
                all_ok = False
        if got_oom:
            return "OOM"
        return all_ok

    # Python API: single load, process all windows
    model_path = str(model_dir / f"{model}.pth")
    if not Path(model_path).exists():
        return False

    # FIX-R6-03: 这里必须自己解析 AI 解释器。原实现直接在 cmd 里用 `ai_py`，
    # 但该名字只在 run_realesrgan / _run_dat_batch 内定义 —— 本函数内
    # NameError，又被外层裸 `except Exception: return False` 吞掉，
    # **批量 AI 超分路径从未真正跑通过**（2026-10-18 加失败日志后秒现形）。
    ai_py = _resolve_ai_python(logger)
    if ai_py is None:
        if logger:
            logger.error("STAGE3_AI_INTERPRETER_MISSING",
                         "找不到具备 torch/realesrgan/basicsr 的 AI 解释器"
                         "（视频编排器 R72 不含这些包）")
        return False

    try:
        env = os.environ.copy()
        # 【第七轮·关键修复】**必须清掉编排器的 Python 环境变量**。
        #
        # 父进程是 Portable R72（VapourSynth 环境），它的 PYTHONPATH 指向
        # `tools_r4/vs_portable/r72/Lib/site-packages`。`os.environ.copy()`
        # 会把这一套原封不动传给 **另一个解释器** `.venv_r4` —— 于是
        # .venv_r4 的 Python 去 R72 的目录里找包，`import torch` 直接死锁：
        # 进程 CPU 0.16 秒、内存 11MB、`<no Python frame>`，几十分钟不动。
        #
        # 实测对照（同一个 .venv_r4）：
        #   干净环境            -> torch 2.11.0+cu128 cuda True，1.8 秒
        #   带 R72 的 PYTHONPATH-> 卡死（60 秒超时，无 Python 栈帧）
        #
        # 这就是"AI 子进程卡死、一次两小时"的真正原因 —— 不是模型慢、
        # 不是 CUDA 驱动，是环境变量串了。之前所有"并发抢 GPU"的推断
        # 都是表象。
        for _bad_env in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
                         "PYTHONEXECUTABLE", "PYTHONUSERBASE",
                         "PYTHONNOUSERSITE"):
            env.pop(_bad_env, None)
        if gpu_id < 0:
            env["CUDA_VISIBLE_DEVICES"] = ""
        # FIX-R6-03: 架构选择（与 run_realesrgan 同一规则）。该 realesrgan
        # 版本无 model_name 参数，RealESRGANer 必须收到已实例化的 nn.Module；
        # 原脚本的 `prepad=10` 是非法 kwarg（应为 pre_pad），子进程秒崩。
        if "general-x4v3" in model:
            native_scale = 4
            arch_code = (
                "from realesrgan.archs.srvgg_arch import SRVGGNetCompact\n"
                "arch = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, "
                "num_feat=64, num_conv=32, upscale=4, act_type='prelu')\n")
        elif "x2plus" in model:
            native_scale = 2
            arch_code = (
                "from basicsr.archs.rrdbnet_arch import RRDBNet\n"
                "arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, "
                "num_block=23, num_grow_ch=32, scale=2)\n")
        else:  # RealESRGAN_x4plus 及其它 RRDB 4× 系
            native_scale = 4
            arch_code = (
                "from basicsr.archs.rrdbnet_arch import RRDBNet\n"
                "arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, "
                "num_block=23, num_grow_ch=32, scale=4)\n")
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                          delete=False) as f:
            script_path = f.name
            window_repr = repr([(d, o) for d, o in window_dirs])
            f.write(
                "import sys\n"
                "from realesrgan import RealESRGANer\n"
                "import torch\n"
                "from PIL import Image\n"
                "import os\n"
                "import glob\n"
                "import numpy as np\n"
                # COL-03：16bit 帧必须用 cv2 读写（PIL 会静默降 8bit）
                "import cv2\n"
                "import gc\n"
                f"model_path = {repr(model_path)}\n"
                f"scale = {scale}\n"
                f"use_half = not torch.cuda.is_available() "
                f"if {gpu_id} >= 0 else False\n"
                f"windows = {window_repr}\n"
                f"{arch_code}"
                "# FIX-P8-9: Load model ONCE for all windows\n"
                "try:\n"
                f"    upsampler = RealESRGANer(scale={native_scale}, "
                "model_path=model_path, model=arch, pre_pad=10, half=use_half)\n"
                "    for frames_dir, output_dir in windows:\n"
                "        os.makedirs(output_dir, exist_ok=True)\n"
                "        for fp in sorted(glob.glob("
                "os.path.join(frames_dir, '*.png'))):\n"
                "            # COL-01：官方接口输入 BGR/输出 BGR"
                "（见单窗路径说明）\n"
                # COL-03：同单窗路径——cv2 保留 16bit，且是 BGR
                "            _arr = cv2.imread(fp, cv2.IMREAD_UNCHANGED)\n"
                "            if _arr is None:\n"
                "                _arr = np.array(Image.open(fp).convert('RGB'))[:, :, ::-1].copy()\n"
                # 同单窗：enhance() 已还原量级与类型，直接写回
                "            output, _ = upsampler.enhance(_arr, outscale=scale)\n"
                "            cv2.imwrite(os.path.join(output_dir, os.path.basename(fp)), output)\n"
                "    del upsampler\n"
                "    gc.collect()\n"
                "    if torch.cuda.is_available():\n"
                "        torch.cuda.empty_cache()\n"
                # FIX-N+1-4 (N+1 轮核验): merge OOM except clauses into one
                # version-safe RuntimeError handler (see run_realesrgan for
                # full rationale). ORIGINAL `except torch.cuda.OutOfMemoryError:`
                # raised AttributeError on PyTorch<1.13, losing the OOM and
                # skipping the RuntimeError fallback (subprocess exited non-83,
                # breaking OOM downgrade). Single except RuntimeError now
                # detects OOM via hasattr-guarded isinstance (>=1.13, non-English)
                # OR English string match (<1.13); non-OOM re-raises. Per §G.
                "except RuntimeError as _e:\n"
                "    _is_oom = (hasattr(torch.cuda, 'OutOfMemoryError')\n"
                "               and isinstance(_e, torch.cuda.OutOfMemoryError))\n"
                "    if (not _is_oom) and ('out of memory' in str(_e).lower()):\n"
                "        _is_oom = True\n"
                "    if _is_oom:\n"
                "        try:\n"
                "            del upsampler\n"
                "        except NameError:\n"
                "            pass\n"
                "        gc.collect()\n"
                "        if torch.cuda.is_available():\n"
                "            torch.cuda.empty_cache()\n"
                "        import time as _t; _t.sleep(0.1)\n"
                "        sys.exit(83)  # FIX-P8-1: GPU OOM exit code\n"
                "    raise\n"
            )
        # FIX-R6-02: 用 AI 解释器（.venv_r4）而不是 sys.executable
        #
        # `-u` 很关键：子进程卡住时，缓冲的 stdout 会让诊断日志**空白**，
        # 看起来就像"什么都没发生"。无缓冲后至少能看到它停在哪一行。
        cmd = [ai_py, "-u", script_path]

        # B9-后续（第七轮）：**批量路径原来固定 timeout=7200（两小时）**。
        # 这就是"卡了约 2 小时"的字面来源：子进程一旦不返回，父进程要
        # 等满 2 小时才动。改成按窗口总帧数动态给（每帧 30 秒上限，
        # 最少 300 秒），并走取消守卫（可 Ctrl+C、超时会 kill+reap）。
        _batch_frames = 0
        for _fd, _od in window_dirs:
            try:
                _batch_frames += len([n for n in os.listdir(_fd)
                                      if n.endswith(".png")])
            except OSError:
                pass
        if _batch_frames == 0 and logger:
            # 实测踩过：传了 Git Bash 风格路径（/e/codex/...），Windows 的
            # os.listdir 抛 OSError 被上面吞掉，frames 变 0，子进程空跑
            # 却返回 True —— 表面上"成功"，实际一个字节都没产出。
            # 这里必须喊出来。
            logger.warning(
                "STAGE3_NO_FRAMES",
                "No .png frames found in any window directory; the AI "
                "subprocess would do nothing and still exit 0. Check that "
                "paths are Windows-style (E:/...) and that extraction ran.",
                windows=[list(w) for w in window_dirs][:3])
        _batch_timeout = max(300, min(7200, _batch_frames * 30))
        if logger:
            logger.info("STAGE3_AI_SUBPROC_START",
                        "Launching AI subprocess under cancel-guard",
                        frames=_batch_frames, timeout_s=_batch_timeout,
                        interpreter=ai_py)
        # 诊断开关：把子进程的 stdout/stderr **实时**写到文件。
        # `capture_output=True` 要等进程结束才拿得到内容 —— 正好在最需要
        # 线索的"卡住"场景下什么都没有。这里 line-buffered 落盘。
        _trace_path = None
        # 注意：本函数签名**没有 params**（只有 window_dirs/model/scale/
        # gpu_id/logger）。第一版这里写了 `params.get(...)`，直接
        # NameError —— 而且被外层 `except Exception` 记成
        # "STAGE3_AI_LAUNCH_FAIL"，看起来像 AI 起不来。
        # 改成从窗口的帧目录推断输出位置。
        _trace_dir = "."
        if window_dirs:
            try:
                _trace_dir = os.path.dirname(
                    os.path.abspath(window_dirs[0][0])) or "."
            except Exception:
                _trace_dir = "."
        if True:
            try:
                _trace_path = os.path.join(
                    _trace_dir,
                    f"ai_subproc_{os.getpid()}_{os.path.basename(str(script_path))}.log")
                _trace_fh = open(_trace_path, "w", encoding="utf-8",
                                 buffering=1)
                if logger:
                    logger.info("STAGE3_AI_SUBPROC_TRACE",
                                "AI subprocess output is being streamed to "
                                "this file (so a hang is diagnosable)",
                                path=_trace_path)
            except Exception:
                _trace_fh = None
        else:
            _trace_fh = None

        try:
            if _trace_fh is not None:
                r = subprocess.run(cmd, stdout=_trace_fh,
                                   stderr=subprocess.STDOUT, text=True,
                                   timeout=_batch_timeout, shell=False, env=env)
            elif _cancel_guard is not None:
                r = _cancel_guard.run_managed(
                    cmd, capture_output=True, text=True, timeout=_batch_timeout,
                    shell=False, env=env, logger=logger)
            else:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   errors="replace",
                                   timeout=_batch_timeout, shell=False, env=env)
        except subprocess.TimeoutExpired as _te:
            if logger:
                logger.error(
                    "STAGE3_AI_SUBPROC_TIMEOUT",
                    f"AI subprocess exceeded {_batch_timeout}s "
                    f"({_batch_frames} frames); child killed and reaped. "
                    "Check that PYTHONHOME/PYTHONPATH were not inherited from "
                    "the orchestrator (that makes .venv_r4 hang with zero CPU).",
                    frames=_batch_frames, timeout_s=_batch_timeout,
                    stderr_tail=str(getattr(_te, "stderr", "") or "")[-800:])
            if _trace_fh is not None:
                try:
                    _trace_fh.close()
                except Exception:
                    pass
            return False
        # FIX-R6-03: 失败必须留证据（同 run_realesrgan；原实现静默 False）
        if r.returncode == 83:
            try:
                Path(script_path).unlink(missing_ok=True)
            except Exception:
                pass
            return "OOM"
        if r.returncode != 0:
            if logger:
                logger.error("STAGE3_AI_SUBPROC_FAIL",
                             f"AI subprocess exited {r.returncode}; "
                             f"script kept at {script_path}",
                             stderr_tail=(r.stderr or "")[-1500:])
            return False
        try:
            # FIX-P8-10: Use Path.unlink instead of os.unlink
            Path(script_path).unlink(missing_ok=True)
        except Exception:
            pass
        return True
    except Exception as e:
        if logger:
            logger.error("STAGE3_AI_LAUNCH_FAIL", str(e))
        return False


def _query_all_pts(video_path: str, logger=None) -> dict:
    """Query PTS for ALL frames in a video, one-time.

    FIX-N8-2: Extracted from extract_frames_with_pts to avoid
    repeated full-video ffprobe calls per window.
    Returns {frame_index: pts_seconds}.
    """
    pts_map: dict = {}
    try:
        pts_cmd = [
            "ffprobe", "-hide_banner", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "frame=pts_time,pkt_pts_time",
            "-of", "json",
            video_path
        ]
        pts_proc = subprocess.run(pts_cmd, capture_output=True, text=True, errors="replace",
                                 timeout=300, shell=False)
        if pts_proc.returncode == 0:
            data = json.loads(pts_proc.stdout)
            for i, frame_info in enumerate(data.get("frames", [])):
                # FIX-P8-6: pts_time (FFmpeg>=7.0) first,
                # fallback to pkt_pts_time (FFmpeg<7.0)
                # ORIGINAL: Only used pkt_pts_time, fails on FFmpeg 7.0+
                pts_str = (frame_info.get("pts_time")
                           or frame_info.get("pkt_pts_time", ""))
                if pts_str:
                    try:
                        pts_map[i] = float(pts_str)
                    except ValueError:
                        pts_map[i] = i / 25.0
    except Exception:
        if logger:
            logger.warning("STAGE3_PTS_QUERY_FAIL",
                           "Failed to query PTS, using frame-index fallback")
    return pts_map


def extract_frames_with_pts(video_path: str, output_dir: str,
                              win_start: int = 0, win_size: int = 256,
                              logger=None, fps: float = 30.0,
                              pts_map: Optional[dict] = None
                            ) -> List[Tuple[str, float]]:
    """Extract frames in a sliding window with PTS support for VFR.

    FIX-N8-2: Accepts pre-built pts_map to avoid per-window ffprobe calls.
    Returns list of (frame_path, pts_seconds) ordered by PTS.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # V-P-033 fix: extract 16bit PNG for preservation
    # Use ffmpeg with -vsync 0 to preserve original timestamps
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", video_path,
    ]
    if win_start > 0:
        # FIX-P8-7: Use actual PTS for VFR seek per v6.32 §4B
        # ORIGINAL: Used CFR assumption (win_start / fps), wrong for VFR
        if pts_map and win_start in pts_map:
            seek_time = pts_map[win_start]
        else:
            seek_time = win_start / fps  # CFR fallback
        cmd.extend(["-ss", str(seek_time)])
    if win_size > 0:
        cmd.extend(["-frames:v", str(win_size)])
    cmd.extend([
        "-vsync", "0",
        "-frame_pts", "1",
        "-pix_fmt", "rgb48",
        str(Path(output_dir) / "frame_%08d.png")
    ])

    try:
        r = subprocess.run(cmd, capture_output=True, timeout=600, shell=False)
        if r.returncode != 0:
            if logger:
                logger.error("STAGE3_EXTRACT_FAIL",
                             f"Frame extract: {r.stderr.decode()[:200]}")
            return []
    except Exception as e:
        if logger:
            logger.error("STAGE3_EXTRACT_EXCEPT", str(e))
        return []

    # FIX-N8-2: Use pre-built pts_map instead of per-window ffprobe
    # Build ordered list by actual frame number
    frames: List[Tuple[str, float]] = []
    frame_files = sorted(Path(output_dir).glob("frame_*.png"))
    for i, fp in enumerate(frame_files):
        # Map window-local index to global index
        global_idx = win_start + i
        if pts_map:
            pts = pts_map.get(global_idx, global_idx / 25.0)
        else:
            pts = global_idx / 25.0
        frames.append((str(fp), pts))

    return frames


def _count_video_frames(video_path: str) -> int:
    """Count total frames in video. 返回 0 表示**确定不了**（调用方必须报错）。

    FIX-2026-10-22：加 `-count_packets` 快速路径。

    实测（38 MB 的 1080p AV1 成品）：
        -count_frames   → 580 帧，**19.4 s**（要逐帧解码）
        -count_packets  → 580 帧，**0.30 s**（只解复用）
        nb_frames(元数据) → N/A
    也就是说 `-count_frames` 的耗时随**文件大小**线性增长，而本函数原来给它
    60 秒固定超时。stage1 产出的 1.6 GB 16-bit 无损 mkv 需要十几分钟，
    必然超时；随后 `duration × fps` 兜底又因为 vspipe 产出的 mkv
    `format=duration` 是 **N/A** 而失败 → 返回 0。

    后果（2026-10-22 实测，1080p/20s 强制 AI）：调用方落到写死的
    `total_frames = 30000`，于是对 599 帧的素材切出 **118 个窗口**，
    其中 **115 个是空的**，白烧约 25 分钟，还让 AI 子进程在空目录上打转。

    现在的顺序：元数据 → 解复用数包 → 逐帧解（超时按大小放宽）→ 时长×帧率。
    对本项目自产的 mkv/ffv1/av1，1 packet = 1 frame，第二步即为准确值。
    """
    def _probe(extra_args, timeout):
        try:
            r = subprocess.run(
                ["ffprobe", "-hide_banner", "-v", "error"] + extra_args
                + [video_path],
                capture_output=True, text=True, errors="replace",
                timeout=timeout, shell=False)
            if r.returncode == 0:
                for ln in r.stdout.strip().splitlines():
                    ln = ln.strip()
                    if ln:
                        return ln
        except Exception:
            pass
        return ""

    # 1) 容器元数据里的 nb_frames —— 瞬间，且权威
    v = _probe(["-select_streams", "v:0",
                "-show_entries", "stream=nb_frames",
                "-of", "default=noprint_wrappers=1:nokey=1"], 15)
    if v.isdigit() and int(v) > 0:
        return int(v)

    # 2) -count_packets —— 只解复用，快 60~70 倍。命中即返回，**不要**再跑
    #    慢的 -count_frames（本函数早期版本就是忘了提前 return，实测多花 19 s）。
    #    对本项目产出的 mkv/ffv1/h264/av1，1 packet 恒等于 1 frame。
    v = _probe(["-select_streams", "v:0", "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "default=noprint_wrappers=1:nokey=1"], 120)
    if v.isdigit() and int(v) > 0:
        return int(v)

    # 3) -count_frames —— 需要逐帧解码，超时按文件大小放宽（每 10 MB 给 1 s）
    try:
        size_mb = os.path.getsize(video_path) / (1024.0 * 1024.0)
    except OSError:
        size_mb = 0.0
    dec_timeout = max(60, min(3600, int(size_mb / 10.0)))
    v = _probe(["-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=nb_read_frames",
                "-of", "default=noprint_wrappers=1:nokey=1"], dec_timeout)
    if v.isdigit() and int(v) > 0:
        return int(v)

    # 4) 时长 × 帧率
    try:
        dur_s = _probe(["-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1"], 30)
        fps_s = _probe(["-select_streams", "v:0",
                        "-show_entries", "stream=r_frame_rate",
                        "-of", "default=noprint_wrappers=1:nokey=1"], 30)
        if dur_s and fps_s:
            duration = float(dur_s)
            parts = fps_s.split("/")
            fps = (float(parts[0]) / float(parts[1])) if len(parts) == 2 \
                else float(parts[0])
            if duration > 0 and fps > 0:
                return int(duration * fps)
    except Exception:
        pass

    # 5) 全部失败 → 返回 0，由调用方显式报错（不再猜 30000）
    return 0


def _load_params(args):
    if getattr(args, "params_file", None):
        with open(args.params_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(args.params) if args.params else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--params")
    parser.add_argument("--params-file")
    parser.add_argument("--config")
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()

    params = _load_params(args)
    log_dir = params.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, job_id=args.job_id)

    # NEW-V-06: 注册取消守卫（video main.py 认 rc=88 = EXIT_INTERRUPTED）。
    if _cancel_guard is not None:
        _cancel_guard.install_cancel_guard(stage=3, logger=logger)

    # Check AI settings
    ai_enabled = params.get("ai_enabled", False)
    if not ai_enabled:
        logger.info("STAGE3_SKIP", "AI not enabled")
        return 3

    scale = params.get("upscale_scale", 2)
    max_scale = params.get("max_scale", 4)

    # --- Layer 1 (hard): scale not 2/4 ---
    if scale not in (2, 4):
        logger.error("STAGE3_INVALID_SCALE",
                     f"Invalid scale: {scale}. Only integer 2x or 4x allowed.")
        return 81
    if scale > max_scale:
        logger.error("STAGE3_INVALID_SCALE",
                     f"Scale {scale} exceeds max_scale={max_scale}")
        return 81

    # Bug 13.30 fix: PyTorch version / backend check
    require_cuda = params.get("require_cuda", True)
    if require_cuda:
        try:
            import torch
            torch_ver = torch.__version__
            if "cpu" in torch_ver.lower():
                logger.error("STAGE3_CPU_TORCH",
                             f"PyTorch {torch_ver} is CPU-only "
                             "but require_cuda=True")
                # FIX-P8-4: Exit code 1 -> 83 (GPU error)
                # ORIGINAL: return 1  (outside 80-113 legal range)
                # Per v7.32 §3.5: 83 = GPU error
                return 83
        except Exception:
            pass

    # FIX-P8-5: WSL2 temp_dir redirect per v6.32 §7.3
    model = params.get("upscale_model", "realesr-general-x4v3")
    temp_dir = params.get("temp_dir", "./temp")
    try:
        from checkpoint import wsl_redirect_path
        # wsl_redirect_path returns (redirected_out, orig_out,
        #                            redirected_temp, orig_temp)
        _, _, temp_dir, _ = wsl_redirect_path("", temp_dir=temp_dir,
                                               logger=logger)
    except Exception:
        pass
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # Find stage2 output - deterministic by mtime (v6.32 §7.1)
    job_id = args.job_id
    input_name = Path(args.input).name
    pattern = f"{job_id}_{Path(input_name).stem}_step2_traditional*"
    # R3-01（关键修复）：`_step2_traditional*` 通配同时命中 sidecar
    # `<...>.mkv.identity.json`（身份登记写在 mkv 之后、mtime 更晚），裸 sort
    # 选中 .json → ffprobe 解析失败 → STAGE3_STAGE2_UNDECODABLE (82)，
    # 而真正的 mkv 就在旁边（2026-10-18 矩阵首跑实测复现）。
    # 按 checkpoint 的媒体后缀白名单过滤后再按 mtime 选最新。
    from checkpoint import select_media_candidates
    candidates = select_media_candidates(Path(temp_dir).glob(pattern))
    if not candidates:
        logger.error("STAGE3_NO_INPUT",
                     "Stage 2 traditional output not found")
        return 82

    stage2_output = str(candidates[0])

    # Validate stage2 output (v6.32 §7.1 mandatory)
    if Path(stage2_output).stat().st_size == 0:
        logger.error("STAGE3_STAGE2_EMPTY", "Stage 2 output is empty")
        return 82
    try:
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error", stage2_output],
            capture_output=True, timeout=30, shell=False
        )
        if r.returncode != 0:
            logger.error("STAGE3_STAGE2_UNDECODABLE",
                         "Stage 2 output undecodable")
            return 82
    except Exception as e:
        logger.error("STAGE3_STAGE2_CHECK_FAIL", str(e))
        return 82

    # Get intermediate dimensions for supplementary ban check
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "json", stage2_output],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False
        )
        info = json.loads(r.stdout)
        stream = info["streams"][0]
        width = stream["width"]
        height = stream["height"]
    except Exception:
        width, height = 720, 480

    target_w = width * scale
    target_h = height * scale

    # FIX-P8-2: AI Upscale Ban Check - Three-Layer Verification
    # Per v7.32 §3.10.2: dual-basis (manifest flag + intermediate dims)
    # ORIGINAL: Only checked ai_banned manifest flag, no supplementary
    #           ban on intermediate dimensions, no three-layer logic
    ai_banned = params.get("ai_upscale_banned", False)
    force_allow = params.get("force_allow_banned_upscale", False)

    # FIX-P8-1: Dual-basis ban per v7.32 §3.10.2
    # Basis 1: Original input file pixel dimensions (permanent basis)
    # ORIGINAL: Only checked manifest flag + intermediate dims,
    #           missing original input dimensions check.
    # Per v7.32 §3.10.2: original file is the permanent baseline;
    # if it meets ban threshold, AI is banned regardless of
    # intermediate resizing/cropping.
    orig_w = params.get("original_width", 0)
    orig_h = params.get("original_height", 0)
    original_banned = False
    if orig_w > 0 and orig_h > 0:
        # HDV 1080p or larger: original already meets ban threshold
        if orig_w >= 1920 and orig_h >= 1080:
            original_banned = True
            logger.info("STAGE3_ORIGINAL_BANNED",
                        f"Original input {orig_w}x{orig_h} meets "
                        "ban threshold (>=1920x1080)")

    # FIX-P8-2: Analog tape 4x hard ban per v6.32 §4B
    # v6.32 §4B: "RealESRGAN 4x...strictly prohibited for analog tape"
    detected_class = params.get("detected_class", "")
    if detected_class == "ANALOG_TAPE" and scale == 4:
        logger.error("STAGE3_ANALOG_4X_BANNED",
                    "Real-ESRGAN 4x is strictly prohibited for analog tape "
                    "(VHS/Hi8/Betamax) per v6.32 §4B. Maximum upscale for "
                    "analog tape is 2x to 960p. "
                    "Use force_allow_banned_upscale to override.")
        if not force_allow:
            return 81  # Parameter error

    # FIX-R8-W4: anime model ban per v6.32 Part 4 — anime models strictly
    # prohibited for non-anime video sources (real-person content).
    model_requested = params.get("upscale_model", "realesr-general-x4v3")
    is_anime_source = detected_class in ("ANIME", "ILLUSTRATION", "CELL_ANIMATION")
    if "anime" in model_requested.lower() and not is_anime_source:
        logger.error("STAGE3_ANIME_MODEL_BANNED",
                    f"Anime model '{model_requested}' is strictly prohibited "
                    "for non-anime video sources per v6.32 Part 4. "
                    "Use realesr-general-x4v3 (default) for real-person content; "
                    "RealESRGAN_x2plus/x4plus as optional backup.")
        return 81  # parameter error (non-retryable)

    # Dual-basis ban: manifest flag OR original dims OR intermediate dims
    # Per v7.32 §3.10.2: either original or intermediate triggers ban
    # FIX-R8-W1 (方案A): intermediate_banned uses the AI-stage INPUT file
    # pixel dims (width/height), NOT the upscaled target dims — per spec
    # "以AI超分步骤的输入中间文件的像素尺寸为基准". Analogue-tape 4x is already
    # hard-banned above; this catches a >1080p intermediate fed to AI.
    intermediate_banned = (width > 1920 or height > 1080)
    combined_banned = ai_banned or original_banned or intermediate_banned

    # Layer 2 (default guard): skip when banned and not forced
    if combined_banned and not force_allow:
        logger.error("STAGE3_BANNED",
                     f"AI upscale banned: manifest={ai_banned}, "
                     f"intermediate_res_banned={intermediate_banned} "
                     f"({target_w}x{target_h})")
        return 3

    # FIX-P8-3: Layer 3 (force override): CRITICAL + WARNING dual log
    # Per v7.32 §3.10.2 L3: must emit CRITICAL + WARNING
    # ORIGINAL: Only had CRITICAL log, missing WARNING for quality risk
    # FIX-R8-W3: explicit operation_time field per §3.10.2 L3 "永久记录操作时间"
    # FIX-R8-M3: permanent audit archive (independent file, survives job log rotation)
    #             per §3.10.2 L3 "所有强制覆盖操作日志永久归档"
    if combined_banned and force_allow:
        op_time = datetime.datetime.now().isoformat()
        logger.critical("STAGE3_FORCE",
                       "Force-allowing banned upscale per user override",
                       operation_time=op_time,
                       original_width=params.get("original_width"),
                       original_height=params.get("original_height"),
                       intermediate_width=width,
                       intermediate_height=height,
                       upscale_scale=scale,
                       job_id=job_id,
                       input_name=input_name)
        logger.warning("STAGE3_FORCE_RISK",
                      "Quality risk: forced upscale may produce AI "
                      "artifacts, plastic textures, temporal flickering, "
                      "or loss of historical authenticity")
        # R8-M3: permanent audit archive
        try:
            audit_dir = Path("logs/forced_upscale_audit")
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_record = {
                "operation_time": op_time,
                "job_id": job_id,
                "input_name": input_name,
                "detected_class": detected_class,
                "original_width": params.get("original_width"),
                "original_height": params.get("original_height"),
                "intermediate_width": width,
                "intermediate_height": height,
                "upscale_scale": scale,
                "upscale_model": model_requested,
                "combined_banned_reason": {
                    "ai_banned": ai_banned,
                    "original_banned": original_banned,
                    "intermediate_banned": intermediate_banned,
                },
            }
            audit_file = audit_dir / f"{job_id}_forced_upscale_audit.json"
            with open(audit_file, "w", encoding="utf-8") as af:
                json.dump(audit_record, af, ensure_ascii=False, indent=2)
            logger.info("STAGE3_FORCE_ARCHIVED",
                        f"Forced upscale audit archived to {audit_file}")
        except Exception as ae:
            logger.warning("STAGE3_FORCE_ARCHIVE_FAIL",
                           f"Failed to archive forced upscale audit: {ae}")

    # GPU memory check
    mem_required = estimate_gpu_memory(width, height, scale, model)
    sufficient, backend = check_gpu_memory(mem_required, logger)

    if not sufficient and backend == "cpu":
        # Try 2x if requested 4x, preserving model family
        if scale == 4:
            logger.warning("STAGE3_DOWNSCALE",
                           "GPU memory insufficient for 4x, trying 2x")
            scale = 2
            mem_required = estimate_gpu_memory(width, height, scale, model)
            model = _get_model_for_scale(model, 2)
            sufficient, backend = check_gpu_memory(mem_required, logger)

        if not sufficient:
            # FIX-R8-W5: Per v7.32 §3.10.23 "CPU 也不可用 -> 跳过 AI 阶段,直接
            # 输出传统修复结果". CPU inference is always possible WHEN torch is
            # importable; the only "CPU unavailable" case is torch entirely missing
            # (no GPU backend and no CPU PyTorch). require_cuda=True already exits
            # 83 upstream (L702-715); this branch catches require_cuda=False + no
            # torch. Mirror photo M-12-10: copy stage2 traditional output as stage3
            # output and return 3 (skip AI) so downstream stages consume traditional.
            try:
                import torch  # noqa: F401
                _has_cpu_torch = True
            except ImportError:
                _has_cpu_torch = False
            if not _has_cpu_torch:
                logger.warning("STAGE3_SKIP_AI_NO_TORCH",
                               "No GPU and no CPU PyTorch backend; skipping AI stage, "
                               "emitting traditional (stage2) output directly per "
                               "v7.32 §3.10.23")
                # FIX-R8-W5-REGRESSION (N+1 轮核验): 原代码 shutil.copyfile(stage2_output,
                # output_path) 中的 output_path 在此处未定义 (首次定义于 L1188, 依赖
                # L953 的 param_hash, 均在本分支之后), 必抛 NameError, 被 except 捕获
                # 后返回 83 而非预期的 3, 导致 "跳过 AI 输出传统结果" 降级路径失效。
                # 就地计算 param_hash 与 step3 输出路径 (镜像正常路径 L946-988 的
                # core_params 与 L1178-1188 的命名逻辑), 使副本文件名与下游 stage4
                # 期望的 param_hash 一致。
                try:
                    from checkpoint import compute_param_hash
                    _skip_core_params = {
                        "upscale_scale": scale,
                        "upscale_model": model,
                        "lanczos_target": params.get("lanczos_target", "auto"),
                        "ai_enabled": ai_enabled,
                        "vs_bit_depth": params.get("vs_bit_depth", 16),  # FIX-P8-5
                    }
                    _skip_hash = compute_param_hash(stage2_output, _skip_core_params)
                    _skip_inter = (f"{job_id}_{Path(input_name).stem}"
                                   f"_step3_upscale_p{_skip_hash}")
                    if combined_banned and force_allow:
                        _skip_inter = (f"{job_id}_{Path(input_name).stem}"
                                       f"_step3_forced_upscale_p{_skip_hash}")
                    _skip_output = str(Path(temp_dir) / f"{_skip_inter}.mkv")
                    shutil.copyfile(stage2_output, _skip_output)
                    logger.info("STAGE3_SKIP_AI_DONE",
                                f"Traditional output copied to {_skip_output}")
                    return 3
                except Exception as _e:
                    logger.error("STAGE3_SKIP_AI_COPY_FAIL",
                                 f"Traditional fallback copy failed: {_e}")
                    return 83
            logger.warning("STAGE3_CPU",
                           "GPU unavailable, using CPU (very slow)")
            backend = "cpu"

    # P8-6: Compute param_hash for intermediate filename
    # FIX-P8-8: Enforce checkpoint.compute_param_hash per v7.32 §3.10.19
    # ORIGINAL: Local fallback used SHA-256 without file content hash,
    #           silently degrading param_hash quality
    # checkpoint is a hard dependency - fail fast if not available
    from checkpoint import compute_param_hash

    core_params = {
        "upscale_scale": scale,
        "upscale_model": model,
        "lanczos_target": params.get("lanczos_target", "auto"),
        "ai_enabled": ai_enabled,
        "vs_bit_depth": params.get("vs_bit_depth", 16),  # FIX-P8-5
    }
    param_hash = compute_param_hash(stage2_output, core_params)

    logger.info("STAGE3_START", "AI upscaling",
                model=model, scale=scale, backend=backend,
                input_width=width, input_height=height)

    # 256-frame sliding window processing
    win_size = params.get("ai_window_size", 256)
    total_frames = _count_video_frames(stage2_output)
    if total_frames <= 0:
        # FIX-2026-10-22：原来这里静默 `total_frames = 30000`。
        #
        # 实测后果（1080p / 20s / 599 帧，强制 AI）：按 30000 切出 **118 个
        # 窗口**，其中 **115 个是空的**（每窗还要起一次 vspipe 抽取），
        # 白烧约 25 分钟，且 `STAGE3_WINDOW Processing frames ...` 的日志
        # 与实际帧数完全脱节，排查时极具误导性。
        #
        # 帧数确定不了属于输入/IO 异常（v7.32 §3.10.18：82 = IO 错误），
        # 必须显式失败 —— 猜一个数继续跑只会产出可疑结果。
        logger.error(
            "STAGE3_FRAME_COUNT_FAIL",
            f"Cannot determine the frame count of {stage2_output}; "
            "refusing to guess a window count "
            "(tried nb_frames / -count_packets / -count_frames / "
            "duration×fps).")
        return 82

    output_dir = str(Path(temp_dir) / f"{job_id}_upscaled")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get actual fps for frame extraction (v6.32 §4B)
    fps_cmd = [
        "ffprobe", "-hide_banner", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        stage2_output,
    ]
    actual_fps = 30.0
    try:
        r = subprocess.run(fps_cmd, capture_output=True, text=True, errors="replace",
                          timeout=10, shell=False)
        if r.returncode == 0:
            fps_str_val = r.stdout.strip()
            if "/" in fps_str_val:
                num, den = fps_str_val.split("/")
                actual_fps = int(num) / int(den)
            else:
                actual_fps = float(fps_str_val)
    except Exception:
        pass

    # FIX-N8-2: Query all PTS once before window loop (not per-window)
    # ORIGINAL: extract_frames_with_pts called ffprobe for every window
    all_pts_map = _query_all_pts(stage2_output, logger)

    # FIX-P8-9: Collect window dirs for batch processing
    # ORIGINAL: Per-window run_realesrgan (model loaded per window)
    # FIX-P8-8: Collect PTS for VFR preservation
    all_success = True
    gpu_oom_retry_done = False
    window_dirs: List[Tuple[str, str]] = []
    all_pts: List[float] = []  # FIX-P8-8: PTS for VFR
    all_frame_mappings: List[Tuple[int, int]] = []  # (win_start, count)
    # FIX-2026-10-22：统计空窗口。空窗口本身不算错（尾窗可能不满），
    # 但**成片空**说明 total_frames 估错或抽取失败——必须留痕。
    empty_windows = 0
    # FIX-2026-10-23：连续空窗口计数，用于提前终止（见循环内注释）。
    consecutive_empty = 0

    for win_start in range(0, total_frames, win_size):
        win_end = min(win_start + win_size, total_frames)
        actual_win_size = win_end - win_start

        logger.info("STAGE3_WINDOW",
                    f"Processing frames {win_start}-{win_end}")

        # Extract window frames
        frames_dir = str(Path(temp_dir) / f"{job_id}_frames_{win_start}")
        # FIX-N8-2: Pass pre-built pts_map to avoid per-window ffprobe
        frames = extract_frames_with_pts(stage2_output, frames_dir,
                                         win_start, actual_win_size,
                                         logger, actual_fps,
                                         pts_map=all_pts_map)
        if not frames:
            empty_windows += 1
            consecutive_empty += 1
            logger.warning("STAGE3_WINDOW_EMPTY",
                           f"No frames in window {win_start}",
                           empty_so_far=empty_windows,
                           expected_total=total_frames)
            # FIX-2026-10-23：连续空窗口要**提前终止**，而不是把剩下的
            # 上百个窗口挨个试一遍。
            #
            # 为什么会有连续空窗口：`_count_video_frames` 的快速路径用
            # `-count_packets`，它对**含负 PTS 参考帧**的文件会**高估**——
            # 实测 `-c copy` 剪切的片段里 651 个包只有 599 帧是"可显示"的，
            # 另 52 帧是负 PTS 的解码参考帧（ffmpeg 官方论坛的解释：
            # 参考帧可以给负 PTS「可解码但不显示」，或从 0 编号
            # 「那它们也会被显示，剪切就不精确了」）。
            # 所以包数 > 显示帧数时，尾部窗口必然空。
            #
            # 注意：**不能**用 `-avoid_negative_ts make_zero` 去"救回"那些帧
            # ——那会把不可见的参考帧变成可见，片头平白多出画面，
            # 属于**改内容**，与本项目「只改画面质量不改内容」的原则冲突。
            if consecutive_empty >= 3:
                logger.error(
                    "STAGE3_WINDOW_LOOP_EARLY_STOP",
                    f"Stopped after {consecutive_empty} consecutive empty "
                    f"windows at {win_start}; total_frames={total_frames} is "
                    f"likely an overcount (packet count vs displayable frames).",
                    total_frames=total_frames, stopped_at=win_start)
                # FIX-2026-10-23：终止必须视为**失败**，不能静默继续。
                # 连续 3 个窗口抽不出帧，既可能是"包数高估"（无害），
                # 也可能是"抽取连续失败"（有害）。两者无法从循环内区分，
                # 而本项目原则是"宁可失败，不可静默产出缺帧成品"
                # （与下方 partial-window-failure 的处理保持一致）。
                # 注：改用 -count_packets 后，包数与显示帧数通常很接近
                # （如 651 vs 599，尾窗仍非空），此分支基本不会触发。
                all_success = False
                break
            continue
        consecutive_empty = 0

        # FIX-P8-8: Collect PTS for VFR encoding
        for _, pts in frames:
            all_pts.append(pts)

        win_output_dir = str(Path(temp_dir) / f"{job_id}_upscaled_{win_start}")
        Path(win_output_dir).mkdir(parents=True, exist_ok=True)
        window_dirs.append((frames_dir, win_output_dir))
        all_frame_mappings.append((win_start, len(frames)))

    # ID-07：执行指纹容器（AI 分支里填，落盘处读；空表示没走到 AI）
    _exec_fp = None

    # FIX-P8-9: Batch process all windows with single model load
    if window_dirs:
        result = run_realesrgan_batch(window_dirs, model, scale,
                                       gpu_id=0, logger=logger,
                                       params=params)

        # FIX-P8-7+P8-1: OOM downgrade only on actual OOM
        # ORIGINAL: OOM downgrade triggered on any failure (not just OOM),
        #           and model cleanup code was broken (del model on string)
        if result == "OOM" and backend != "cpu" and not gpu_oom_retry_done:
            # Step 1: 4x -> 2x downgrade
            if scale == 4:
                logger.warning("STAGE3_OOM_DOWNSCALE",
                              "GPU OOM detected, downgrading 4x->2x")
                time.sleep(0.1)  # v6.32 §7.14 mandatory 100ms wait
                scale = 2
                model = _get_model_for_scale(model, 2)
                result = run_realesrgan_batch(window_dirs, model, scale,
                                               gpu_id=0, logger=logger)

            # Step 2: GPU -> CPU downgrade
            if result == "OOM" or (isinstance(result, bool) and not result):
                if backend != "cpu":
                    logger.warning("STAGE3_OOM_CPU",
                                  "GPU still failing, downgrading to CPU")
                    time.sleep(0.1)
                    result = run_realesrgan_batch(
                        window_dirs, model, scale,
                        gpu_id=-1, logger=logger)
                    if result is True:
                        backend = "cpu"

            gpu_oom_retry_done = True

        # ID-07（第七轮）：把**实际执行**记下来（后端/模型/倍数/权重指纹）。
        # 注意这里读的是循环后**已被降级改写**的 backend/model/scale ——
        # 配置里写的是"请求"，这里记的是"真的跑了什么"。
        try:
            from exec_fingerprint import build_exec_fingerprint
            # 必须在**真正的 AI 解释器**里探版本：编排器（Portable R72）
            # 没有 torch，问它只会得到 probe_error。实测未传时
            # torch/cuda/gpu 全是 None。
            _ai_py_for_fp = _resolve_ai_python(logger)
            _exec_fp = build_exec_fingerprint(
                model=model, scale=scale,
                ai_python=_ai_py_for_fp,
                # 注意：`model_dir` 是 run_realesrgan_batch() 的**局部变量**，
                # main() 里看不到（实测 NameError: model_dir）。这里按同一
                # 约定重算一次。
                model_path=str(SCRIPT_DIR.parent.resolve()
                               / "models" / f"{model}.pth"),
                backend=("cpu" if backend == "cpu" else "gpu"),
                fallback_reason=(params.get("_oom_fallback_reason") or None),
                logger=logger)
        except Exception as _fp_exc:      # 记录失败不能中断超分
            _exec_fp = None
            logger.warning("STAGE3_EXEC_FP_FAIL", str(_fp_exc))

        # Non-OOM failure
        if result is not True:
            if result != "OOM":
                logger.error("STAGE3_WINDOW_FAIL",
                            "AI upscale failed (non-OOM)")
            else:
                logger.error("STAGE3_WINDOW_OOM",
                            "GPU OOM after retry exhausted")
            all_success = False

    # FIX-2026-10-22：成片空窗口必须留痕。
    # 正常情况最多只有"尾窗不满"，不会出现大片空窗；出现就说明
    # total_frames 明显偏大（旧版猜 30000 时的典型症状：118 窗里 115 空）。
    _planned_windows = (total_frames + win_size - 1) // win_size if win_size else 0
    if empty_windows and (_planned_windows and
                          empty_windows >= max(3, _planned_windows // 4)):
        logger.error(
            "STAGE3_MOSTLY_EMPTY_WINDOWS",
            f"{empty_windows}/{_planned_windows} windows had no frames while "
            f"total_frames={total_frames}. The frame count is almost certainly "
            f"wrong (or extraction is failing); output will be short.",
            empty_windows=empty_windows, planned=_planned_windows,
            total_frames=total_frames)
        all_success = False

    # Collect upscaled frames from window outputs
    for win_start, frame_count in all_frame_mappings:
        win_output_dir = str(Path(temp_dir) / f"{job_id}_upscaled_{win_start}")
        upscaled = sorted(Path(win_output_dir).glob("*.png"),
                         key=lambda p: p.name)
        for src in upscaled:
            match = re.search(r'\d+', src.name)
            if not match:
                continue
            # **不要 -1**。
            #
            # 抽帧用 ffmpeg 的 `frame_%08d.png`，默认 `-start_number 0`，
            # 所以编号是 0..N-1。AI 子进程用 `os.path.basename(fp)` 原样
            # 保留文件名。这里再减 1 会让整批编号前移一位：
            #
            #   frame_00000000.png  (源第 1 帧) -> frame_-0000001.png
            #   frame_00000001.png  (源第 2 帧) -> frame_00000000.png
            #   ...
            #   frame_00000059.png  (源第 60 帧) -> frame_00000058.png
            #
            # 实测输出目录正好是 -1..58 共 60 个文件。而下游 ffmpeg 用
            # `-i frame_%08d.png` 从 0 开始读，读到 58 就停 —— 于是
            # **源视频的第一帧被整帧丢弃**，成品只有 59 帧。
            #
            # 这个 -1 大概是早期抽帧从 1 开始时的补偿，但抽帧早已改成 0 起点。
            global_idx = win_start + int(match.group())
            dst = Path(output_dir) / f"frame_{global_idx:08d}.png"
            # FIX-R8-M2: detect disk-full (errno 28 / ENOSPC) on frame copy and
            # return 87 (non-retryable resource exhausted) per v7.32 §3.10.18.
            # ORIGINAL: bare shutil.copyfile would raise OSError that fell through
            # to a generic 82 (IO error), which is_retryable_error treats as
            # retryable - retrying disk-full only re-fills the disk.
            try:
                shutil.copyfile(str(src), str(dst))
            except OSError as _e:
                if getattr(_e, "errno", None) == errno.ENOSPC:
                    logger.error("STAGE3_DISK_FULL",
                                 f"Disk full (errno 28) writing {dst}. "
                                 f"Non-retryable per v7.32 §3.10.18.")
                    return 87
                raise
            try:
                st = src.stat()
                os.utime(str(dst), (st.st_atime, st.st_mtime))
            except Exception:
                pass

        # Cleanup window frames to save disk space
        if params.get("cleanup_intermediates", True):
            frames_dir = str(Path(temp_dir) / f"{job_id}_frames_{win_start}")
            shutil.rmtree(frames_dir, ignore_errors=True)
            shutil.rmtree(win_output_dir, ignore_errors=True)

    if not all_success:
        # FIX-R8-W2: partial window failure (e.g. CPU OOM after downgrade)
        # must mark FAILED per v7.32 §3.10.18, not return 0 with incomplete
        # output. ORIGINAL: only logged error and continued to encode,
        # producing incomplete results treated as success.
        logger.error("STAGE3_PARTIAL_FAIL_FAILED",
                     "AI upscale produced incomplete output (some windows "
                     "failed/OOM after CPU downgrade). Marking FAILED per "
                     "v7.32 §3.10.18.")
        return 83  # GPU error (incl. OOM failure)

    # Verify we have output frames
    output_frames = sorted(Path(output_dir).glob("*.png"))
    if not output_frames:
        logger.error("STAGE3_NO_OUTPUT", "No upscaled frames produced")
        return 83

    # V-P-034 fix: Lanczos secondary scaling when target != AI output res
    lanczos_target = params.get("lanczos_target", "auto")
    if lanczos_target != "auto" and lanczos_target:
        try:
            if "x" in lanczos_target:
                target_lw, target_lh = map(int, lanczos_target.split("x"))
                if target_lw != target_w or target_lh != target_h:
                    logger.info("STAGE3_LANCZOS",
                               f"Scaling AI output {target_w}x{target_h} "
                               f"-> {target_lw}x{target_lh}")
                    lanczos_dir = str(Path(temp_dir) / f"{job_id}_lanczos")
                    Path(lanczos_dir).mkdir(parents=True, exist_ok=True)
                    for src in sorted(Path(output_dir).glob("*.png")):
                        lanczos_cmd = [
                            "ffmpeg", "-hide_banner", "-v", "error", "-y",
                            "-i", str(src),
                            "-vf",
                            f"zscale=w={target_lw}:h={target_lh}:f=lanczos",
                            "-pix_fmt", "rgb48",
                            str(Path(lanczos_dir) / src.name)
                        ]
                        subprocess.run(lanczos_cmd, capture_output=True,
                                      timeout=60, shell=False)
                    if list(Path(lanczos_dir).glob("*.png")):
                        shutil.rmtree(output_dir, ignore_errors=True)
                        # FIX-P8-10: Use shutil.move instead of os.rename
                        # os.rename fails across filesystems (OSError 18)
                        shutil.move(lanczos_dir, output_dir)
                        target_w, target_h = target_lw, target_lh
        except Exception as e:
            logger.warning("STAGE3_LANCZOS_FAIL",
                           f"Lanczos scaling failed: {e}")

    # V-P-035 fix: preserve original frame rate
    fps_cmd = [
        "ffprobe", "-hide_banner", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        stage2_output,
    ]
    fps_str = "25"
    try:
        r = subprocess.run(fps_cmd, capture_output=True, text=True, errors="replace",
                          timeout=10, shell=False)
        if r.returncode == 0:
            fps_str = r.stdout.strip()
    except Exception:
        pass

    # V-P-031 fix: force_allow suffix (P8-2 + N8-1 corrected)
    inter_name = f"{job_id}_{Path(input_name).stem}" \
                 f"_step3_upscale_p{param_hash}"
    # FIX-N8-1: Use combined_banned (not just ai_banned) per v7.32 §3.10.2 L3
    # ORIGINAL: only ai_banned triggered suffix; intermediate_banned also
    #           forces override and should add _forced_upscale suffix
    if combined_banned and force_allow:
        inter_name = (f"{job_id}_{Path(input_name).stem}"
                      f"_step3_forced_upscale_p{param_hash}")

    output_path = str(Path(temp_dir) / f"{inter_name}.mkv")

    # NEW-V-06: 编码开始前登记产物——取消时删掉半截 mkv，否则下游 stage
    # 的 exists()>0 缓存检查会把它当成功产物复用。
    if _cancel_guard is not None:
        _cancel_guard.watch_outputs(output_path, logger=logger, stage=3)

    # FIX-P8-8: VFR PTS preservation using concat demuxer
    # ORIGINAL: CFR encoding only (lost VFR timestamps)
    # Per v6.32 §4B: VFR support required
    # COL-04（第七轮）：**显式携带颜色表示**，不要让下游去猜。
    #
    # rgb48 -> yuv420p16le 是一次真正的矩阵转换。旧代码没有传
    # `-colorspace`，于是 ffmpeg 按分辨率启发式选矩阵（SD 走 601、
    # HD 走 709），而产物**不带任何颜色标签**（实测中间产物
    # matrix=None/range=None）。下游 stage7 再把原片标签"补"上去 ——
    # 一旦 ffmpeg 选的矩阵与原片不同，这个标签就是**错的**。
    #
    # 现在：先探测源的颜色描述，转换时显式指定同一套，并写进容器标签。
    _src_color = {}
    try:
        from media_io import probe_color as _probe_color
        # 注意：main() 里的输入是 `args.input`，**没有** `input_path`
        # 这个局部变量。第一版写成 `input_path` → NameError → 被
        # `except Exception` 吞成 "STAGE3_COLOR_PROBE_FAIL" → 颜色探测
        # 从未真正生效，中间产物依旧不带标签（COL-04 白做）。
        _src_color = _probe_color(args.input) or {}
    except Exception as _ce:
        logger.warning("STAGE3_COLOR_PROBE_FAIL", str(_ce))
    _color_opts: List[str] = []
    if _src_color.get("probed"):
        if _src_color.get("matrix"):
            _color_opts += ["-colorspace", str(_src_color["matrix"])]
        if _src_color.get("range"):
            _color_opts += ["-color_range",
                            "tv" if _src_color["range"] == "limited" else "pc"]
        if _src_color.get("primaries"):
            _color_opts += ["-color_primaries", str(_src_color["primaries"])]
        if _src_color.get("transfer"):
            _color_opts += ["-color_trc", str(_src_color["transfer"])]
        logger.info("STAGE3_COLOR_CARRIED",
                    "Encoding with the source's colour description "
                    "explicitly (so downstream need not inherit blindly)",
                    matrix=_src_color.get("matrix"),
                    range=_src_color.get("range"),
                    primaries=_src_color.get("primaries"),
                    transfer=_src_color.get("transfer"))
    else:
        logger.warning("STAGE3_COLOR_UNKNOWN",
                       "Source colour description unavailable; the "
                       "intermediate will have no colour tags "
                       "(downstream must not assume BT.601/709)")

    is_vfr = False
    if len(all_pts) >= 3:
        intervals = [all_pts[i + 1] - all_pts[i]
                     for i in range(len(all_pts) - 1)]
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            max_deviation = max(abs(iv - avg_interval) for iv in intervals)
            # If max deviation > 10% of average, treat as VFR
            is_vfr = max_deviation > avg_interval * 0.1

    # 诊断：60 帧进、59 帧出的问题就出在这条分岔上。先把它记清楚，
    # 不要再靠猜。
    try:
        _png_total = 0
        for _od in (w[1] for w in window_dirs):
            _png_total += len(list(Path(_od).glob("frame_*.png")))
    except Exception:
        _png_total = -1
    logger.info("STAGE3_ENCODE_MODE",
                "Encoding path selected",
                is_vfr=is_vfr,
                pts_count=len(all_pts),
                windows=len(window_dirs),
                upscaled_pngs=_png_total,
                actual_fps=actual_fps,
                path=("vfr-concat" if (is_vfr and all_pts) else "cfr-sequence"))

    concat_path: Optional[str] = None
    if is_vfr and all_pts:
        # Use concat demuxer to preserve VFR timestamps
        concat_path = str(Path(temp_dir) / f"{job_id}_concat.txt")
        with open(concat_path, "w") as f:
            f.write("ffconcat version 1.0\n")
            for i, pts in enumerate(all_pts):
                if i < len(all_pts) - 1:
                    duration = all_pts[i + 1] - pts
                else:
                    duration = 1.0 / actual_fps
                frame_file = Path(output_dir) / f"frame_{i:08d}.png"
                f.write(f"file '{frame_file}'\n")
                f.write(f"duration {duration:.6f}\n")
            # 最后一帧**不要再重复写一次**（旧代码和旧注释都错了）。
            #
            # 实测对照（60 个 PNG，`-count_frames` 真解码计数）：
            #
            # | concat 文件写法              | 编码参数            | 帧数 |
            # |------------------------------|---------------------|------|
            # | 每帧 file+duration，末尾重复 | 默认 vsync          |  52  |
            # | 每帧 file+duration，末尾重复 | fps_mode passthrough|  61  |
            # | 每帧 file+duration，不重复   | 默认 vsync          |  51  |
            # | **每帧 file+duration，不重复** | **passthrough**   | **60** |
            #
            # 结论：**每一帧（含末帧）都要写 duration，且不要重复末帧**，
            # 编码时必须 `-fps_mode passthrough`。三者缺一都会丢帧或
            # 多帧。旧代码是"末帧不写 duration"，60 帧素材出来 59 帧。
            #
            # 循环里已经给每一帧（含最后一帧）写过 duration，所以这里
            # 什么都不用补。

        encode_cmd = [
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_path,
            # **必须 passthrough**：默认 `-vsync auto` 会在时间戳不连续时
            # 丢帧/复制帧。实测 60 帧素材经 concat 后只剩 59 帧 —— 就是
            # 默认 vsync 在末帧上做的"对齐"。VFR 路径的全部意义就是保留
            # 原始时间戳，绝不能让它再做帧率重整。
            "-fps_mode", "passthrough",
            "-c:v", "ffv1",
            "-pix_fmt", "yuv420p16le",
            *_color_opts,
            "-an",
            output_path,
        ]
    else:
        # CFR encoding (original path)
        encode_cmd = [
            "ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-framerate", fps_str,
            "-i", str(Path(output_dir) / "frame_%08d.png"),
            "-c:v", "ffv1",
            "-pix_fmt", "yuv420p16le",
            *_color_opts,
            "-an",
            output_path,
        ]

    try:
        # NEW-V-06/RV-12: 受管子进程——取消时 ffmpeg 一起收掉，
        # 否则守卫删掉的半截 mkv 会被孤儿 ffmpeg 重新写回来。
        if _cancel_guard is not None:
            r = _cancel_guard.run_managed(encode_cmd, capture_output=True,
                                          text=True, timeout=3600,
                                          shell=False, logger=logger)
        else:  # pragma: no cover - 守卫缺失时退回原行为
            r = subprocess.run(encode_cmd, capture_output=True, text=True, errors="replace",
                               timeout=3600, shell=False)
        if r.returncode != 0 or not Path(output_path).exists():
            logger.error("STAGE3_ENCODE_FAIL",
                         f"Failed to encode MKV: {r.stderr[:300]}")
            return 84
    except Exception as e:
        # FIX-R8-M2: disk-full (errno 28) is non-retryable per v7.32 §3.10.18
        if isinstance(e, OSError) and e.errno == errno.ENOSPC:
            logger.error("STAGE3_DISK_FULL",
                         "Disk full (errno 28) during encode. Non-retryable "
                         "per v7.32 §3.10.18.")
            return 87  # resource exhausted (non-retryable)
        logger.error("STAGE3_ENCODE_ERROR", str(e))
        return 82

    # FIX-P8-8: Bitdepth self-check for intermediate file
    # Verify intermediate is actually 16bit per v6.32 processing space req
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=pix_fmt",
             "-of", "default=noprint_wrappers=1:nokey=1",
             output_path],
            capture_output=True, text=True, errors="replace", shell=False
        )
        pix_fmt_out = r.stdout.strip()
        if "p16" not in pix_fmt_out and "16" not in pix_fmt_out:
            logger.error("STAGE3_BITDEPTH_MISMATCH",
                        f"Intermediate file not 16bit: {pix_fmt_out}")
            return 86
    except Exception:
        pass

    # Generate forced upscale comparison preview
    # FIX-N8-1: Use combined_banned (not just ai_banned) to cover
    # intermediate_banned force override scenarios too
    if combined_banned and force_allow:
        preview_dir = params.get("preview_dir", "./preview")
        # FIX-P8-5: WSL2 redirect for preview_dir
        try:
            from checkpoint import wsl_redirect_path
            _, _, preview_dir, _ = wsl_redirect_path(
                "", temp_dir=preview_dir, logger=logger)
        except Exception:
            pass
        Path(preview_dir).mkdir(parents=True, exist_ok=True)
        try:
            # 之前这里 import 的是 `generate_before_after_comparison` ——
            # shared_modules/preview.py **没有这个名字**，只有
            # `generate_comparison_frames(original, processed, output_dir,
            # num_pairs)`。于是每次都被 `except Exception` 吞成
            # STAGE3_PREVIEW_FAIL，"生成强制超分对比图"从来没成功过。
            #
            # 签名也不同：这里原来传的是一个 .png 文件路径，而函数要的是
            # **目录**（它在目录下自己建 preview/ 并写多张图）。
            from preview import generate_comparison_frames
            _pairs = generate_comparison_frames(
                stage2_output, output_path,
                str(Path(preview_dir) / f"{job_id}_forced_upscale"),
                num_pairs=1,
            )
            logger.info("STAGE3_PREVIEW",
                        "Generated forced upscale comparison frames",
                        count=len(_pairs or []),
                        first=(_pairs or [None])[0])
        except Exception as e:
            logger.warning("STAGE3_PREVIEW_FAIL",
                           f"Failed to generate preview: {e}")

    # Cleanup concat file if created
    if concat_path:
        try:
            Path(concat_path).unlink(missing_ok=True)
        except Exception:
            pass

    # ID-07：执行指纹落到产物旁边（与身份侧车并列，互不覆盖）
    if _exec_fp:
        try:
            from exec_fingerprint import write_exec_sidecar
            if write_exec_sidecar(output_path, _exec_fp):
                logger.info("STAGE3_EXEC_FINGERPRINT",
                            "Recorded what actually ran (backend/model/"
                            "weights/tools) next to the artifact",
                            backend=_exec_fp.get("backend"),
                            model=_exec_fp.get("requested_model"),
                            weights_sha256=(_exec_fp.get("model_weights")
                                            or {}).get("sha256"))
            else:
                # FIX-2026-10-20：写失败原来是静默的（只有 True 分支有日志）。
                # 后果是 exec 指纹缺失 → resume 时上游摘要对不上 → 重算，
                # 方向是安全的，但**不可见**：看不出为什么每次都重算。
                logger.warning("STAGE3_EXEC_SIDECAR_EMPTY",
                               "Exec fingerprint sidecar was not written; "
                               "resume will fall back to recompute",
                               output=output_path)
        except Exception as _we:
            logger.warning("STAGE3_EXEC_SIDECAR_FAIL", str(_we))

    logger.info("STAGE3_DONE", "AI upscaling complete",
                total_frames=len(output_frames), output_path=output_path)
    # NEW-V-06: 产物已完整落盘，取消守卫不再有权删它（RV-12）。
    if _cancel_guard is not None:
        _cancel_guard.release_outputs()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
