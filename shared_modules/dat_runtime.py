"""DAT 系模型运行时（FIX-R6-01/R6-02，2026-10-18 全选项矩阵验收抽取）。

原实现散在 photo_restoration_pipeline/stages/stage3_ai_upscale.py 的
`_DAT_MODEL_CONFIGS` / `_get_dat_config` / `_run_dat_pytorch`。视频链
stage3 的 AI 子进程也要跑 DAT（--ai-model DAT_x2/DAT_2_x2/...），
两边各写一份必然漂移，故抽到 shared_modules 作为唯一事实源。

配置来源（2026-10-18 联网 + 权重实测双重核实）：
- 官方仓库 zhengchen1999/DAT：test_DAT_*.yml / train_DAT_*.yml
  （本地副本 temp1/DAT-1/DAT-1/options/ 缺 DAT-light，从 GitHub main 补核）
- 权重形状反推交叉验证（norm.weight=embed_dim、last relative_position_index
  = 窗口面积、ffn.fc2 = (embed, embed*exp/2)、upsample.0.weight 的 out 通道）
- 12 个权重（DAT/DAT-2/DAT-S/DAT-light × x2/x3/x4）全部实测
  miss=0/unexp=0/前向尺寸正确；真图 LR→SR 对 GT 的 PSNR 全部高于
  bicubic 基线（DAT_2 34.71 / DAT 34.68 / DAT_S 34.67 / DAT_light 34.24
  vs bicubic 30.92 dB，RealESRGAN_x2plus 26.05 dB 为 GAN 感知增强的正常表现）

架构文件要求：basicsr.archs.dat_arch 必须是**官方新版**（含
Adaptive_*Attention 与 upsampler='pixelshuffledirect'）。旧版
（ basicsr 1.4.2 自带缺 dat_arch；DAT 仓库 v1.3.5 的副本缺 upsampler）
跑不了 DAT-light。环境修复见 requirements.txt：timm/einops + 官方
dat_arch.py。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

# BUG-052 fix: DAT model architecture config mapping.
# FIX-P1-14: Corrected per official zhengchen1999/DAT source (train_DAT_*.yml).
# Previous config used depth=[6,6,6,6] (4 groups) with no split_size — wrong,
# caused state_dict mismatch on any real DAT weight. Official architecture:
#   DAT / DAT-2: depth=[6,6,6,6,6,6] (6 groups), embed_dim=180, num_heads=[6,6,6,6,6,6],
#                split_size=[8,32], img_size=64, img_range=1.0, resi_connection='1conv'.
#   DAT vs DAT-2 differ ONLY in expansion_factor (4 vs 2).
#   DAT-S:     split_size=[8,16], expansion_factor=2（test_DAT_S_x*.yml）
#   DAT-light: depth=[18]（单组 18 blocks）, embed_dim=60, num_heads=[6],
#              split_size=[8,32], expansion_factor=2, resi_connection='3conv',
#              upsampler='pixelshuffledirect'（train_DAT_light_x2.yml，联网核实）
#
# FIX-R6-01（2026-10-18 全选项矩阵验收实测，权重形状反推）:
#   - DAT-light 的 embed_dim 是 **60** 不是 180（norm.weight=(60,)、fc2=(60,60)）：
#     旧表写 180 导致每个 block 形状全错，load_state_dict(strict=False) 静默
#     跳过 → 输出是随机权重垃圾还返回 True（假成功）。已按权重实测值更正。
#   - DAT-S 在旧表里**没有条目**，_get_dat_config 落到 "DAT"（split_size=[8,32],
#     expansion_factor=4），而官方 DAT-S 是 split_size=[8,16] / expansion_factor=2
#     （last relative_position_index=(128,128)=8×16 窗口、fc2=(180,180)=exp2）。
#     同样静默错配。已补条目 + 解析函数补分支（顺序：light → S → 2 → 原版）。
#   - 构造调用原来**不传 split_size**（DAT 默认 [2,4]）——配置表里有也没用。
#   - strict=False 现在配套主动核对：missing 只允许非持久化 buffer
#     （attn_mask / relative_position_index / rpe_biases），unexpected 必须为 0，
#     否则判加载失败（返回 False 走降级链），杜绝"随机权重垃圾算成功"。
DAT_MODEL_CONFIGS: Dict[str, Dict] = {
    # DAT-2 (default): expansion_factor=2
    "DAT_2": {
        "depth": [6, 6, 6, 6, 6, 6],
        "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6],
        "split_size": [8, 32],
        "expansion_factor": 2,
        "resi_connection": "1conv",
        "img_size": 64,
        "img_range": 1.0,
    },
    # DAT (original): expansion_factor=4
    "DAT": {
        "depth": [6, 6, 6, 6, 6, 6],
        "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6],
        "split_size": [8, 32],
        "expansion_factor": 4,
        "resi_connection": "1conv",
        "img_size": 64,
        "img_range": 1.0,
    },
    # DAT-S: split_size=[8,16], expansion_factor=2 (per test_DAT_S_x*.yml)
    "DAT_S": {
        "depth": [6, 6, 6, 6, 6, 6],
        "embed_dim": 180,
        "num_heads": [6, 6, 6, 6, 6, 6],
        "split_size": [8, 16],
        "expansion_factor": 2,
        "resi_connection": "1conv",
        "img_size": 64,
        "img_range": 1.0,
    },
    # DAT-light（官方配置，从 GitHub main options/Train/train_DAT_light_x2.yml
    # 联网核实）：单组 18 blocks、embed_dim=60、3conv、pixelshuffledirect
    # （轻量直出，conv 输出 3×scale²=12 通道——DAT_light_x2.pth 里
    # upsample.0.weight=[12,60,3,3] 的来历；旧 arch 无此 upsampler）
    "DAT_light": {
        "depth": [18],
        "embed_dim": 60,
        "num_heads": [6],
        "split_size": [8, 32],
        "expansion_factor": 2,
        "resi_connection": "3conv",
        "upsampler": "pixelshuffledirect",
        "img_size": 64,
        "img_range": 1.0,
    },
}

# 加载后允许缺失的非持久化 buffer（架构内部按输入尺寸惰性生成）
_NON_PERSISTENT_BUFFER_KEYS = ("attn_mask", "relative_position_index",
                               "rpe_biases")


def get_dat_config(model_name: str) -> Dict:
    """Get DAT architecture config by model name.

    FIX-P1-14: Match by model filename. DAT_2 (default) vs DAT (original)
    differ only in expansion_factor (2 vs 4); DAT_light uses split_size=[8,16].
    Falls back to DAT_2 (the default per v7.32 §3.6).

    FIX-R6-01: added DAT_S branch (split_size=[8,16], expansion_factor=2);
    order matters — check DAT_light and DAT_S before DAT_2/DAT (substring overlap).
    """
    name = model_name.lower()
    # Order matters: check DAT_light before DAT_2/DAT (substring overlap).
    if "dat_light" in name or "dat-light" in name:
        return DAT_MODEL_CONFIGS["DAT_light"]
    if "dat_s" in name or "dat-s" in name:
        return DAT_MODEL_CONFIGS["DAT_S"]
    if "dat_2" in name or "dat2" in name:
        return DAT_MODEL_CONFIGS["DAT_2"]
    if "dat" in name:
        return DAT_MODEL_CONFIGS["DAT"]
    return DAT_MODEL_CONFIGS["DAT_2"]


def is_dat_model(model_name: str) -> bool:
    """模型名是否属于 DAT 家族（含 DAT-2 / DAT-S / DAT-light）。"""
    return model_name.lower().startswith("dat")


def load_dat_model(model: str, scale: int, model_dir: str,
                   device) -> "Optional[object]":
    """构造 DAT 架构并加载权重；任何不符都返回 None（调用方走降级链）。

    这是"唯一允许的 DAT 加载入口"：构造参数全部来自配置表，
    加载后主动核对 missing/unexpected，把 strict=False 的静默错配
    变成显式失败。
    """
    import torch  # noqa: F401  （延迟导入：无 torch 的解释器根本不该走到这）

    from basicsr.archs.dat_arch import DAT

    model_path = os.path.join(model_dir, f"{model}.pth")
    if not os.path.exists(model_path):
        return None
    try:
        state_dict = torch.load(model_path, map_location=device,
                                weights_only=False)
        cfg = get_dat_config(model)
        kwargs = dict(
            upscale=scale, in_chans=3,
            img_size=cfg["img_size"], img_range=cfg["img_range"],
            depth=cfg["depth"], embed_dim=cfg["embed_dim"],
            num_heads=cfg["num_heads"],
            # FIX-R6-01: split_size 必须从配置表传进去——DAT 默认值是 [2,4]，
            # 不传则窗口面积与官方权重（8×32/8×16）全不符，strict=False 会
            # 静默跳过所有形状不符的层（输出是随机权重垃圾还返回 True）。
            split_size=cfg["split_size"],
            expansion_factor=cfg["expansion_factor"],
            resi_connection=cfg["resi_connection"],
        )
        # FIX-R6-01: DAT-light 专用 pixelshuffledirect 上采样（官方新版 arch 才有）
        if cfg.get("upsampler"):
            kwargs["upsampler"] = cfg["upsampler"]
        model = DAT(**kwargs)
        sd = state_dict['params'] if 'params' in state_dict else state_dict
        missing, unexpected = model.load_state_dict(sd, strict=False)
        real_missing: List[str] = [
            k for k in missing
            if not any(t in k for t in _NON_PERSISTENT_BUFFER_KEYS)
        ]
        if unexpected or real_missing:
            import sys
            print(
                f"DAT weight/arch mismatch for {model_path}: "
                f"missing={real_missing[:3]} unexpected={unexpected[:3]}",
                file=sys.stderr)
            return None
        return model.eval().to(device)
    except Exception as e:
        import sys
        print(f"DAT load failed for {model_path}: {e}", file=sys.stderr)
        return None


def run_dat_file(input_path: str, output_path: str, model: str, scale: int,
                 model_dir: str, use_gpu: bool = True,
                 bit_depth: int = 8) -> bool:
    """对单个图像文件跑 DAT 推理（与 photo stage3 历史行为一致）。

    FIX-P12-3: bit_depth=16 时输出 16-bit TIFF（PIL 不能存 RGB uint16，
    用 tifffile；缺 tifffile 才退 8-bit PNG）。
    """
    import torch
    from PIL import Image
    import numpy as np

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available()
                          else "cpu")
    dat_model = load_dat_model(model, scale, model_dir, device)
    if dat_model is None:
        return False
    try:
        # FIX-NEW-V-14 / R16: 不用 Image.open(...).convert('RGB')——G'MIC 的
        # uint16 TIFF 里装的是 8-bit 量级数据，PIL 会读出全 0（黑图）。
        from media_io import load_image_rgb_uint8 as _ldr
        img_np = _ldr(input_path).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                output = dat_model(img_tensor)
        except RuntimeError as oom:
            # FIX-P1-1: CUDA OOM 释放显存后再判失败（OutOfMemoryError 是
            # RuntimeError 子类；旧代码被外层 except Exception 吞掉 → 静默
            # 模型降级且不释放显存）。
            _is_oom = (hasattr(torch.cuda, 'OutOfMemoryError')
                       and isinstance(oom, torch.cuda.OutOfMemoryError))
            if not _is_oom:
                _is_oom = 'out of memory' in str(oom).lower()
            try:
                del dat_model
            except Exception:
                pass
            del img_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import sys
            print(f"DAT CUDA OOM for {model}: {oom}", file=sys.stderr)
            return False
        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # FIX-P12-6: PIL I;16 only supports 2D arrays; for 3-channel 16-bit,
        # use tifffile to save as TIFF (PIL cannot save RGB uint16 directly).
        if bit_depth == 16:
            output_16 = np.clip(output * 65535.0, 0, 65535).astype(np.uint16)
            try:
                import tifffile
                tifffile.imwrite(output_path, output_16, compression="zlib")
            except ImportError:
                output_8 = (output_16 // 256).astype(np.uint8)
                Image.fromarray(output_8).save(output_path)
        else:
            output_8 = np.clip(output * 255.0, 0, 255).astype(np.uint8)
            # FIX-NEW-V-18 / R21: 不用裸 PIL `.save(path)`——PIL 对 .tif
            # 会默认选 **LZW** 压缩，而 tifffile 解 LZW 需要 `imagecodecs`
            #（本机未装）→ 下游 stage5 读不回来 → MediaPrecisionError
            # → rc=86。走 media_io.write_tiff（zlib，纯 Python 内置）。
            try:
                from media_io import write_tiff as _wt
                _wt(output_path, output_8)
            except Exception:
                Image.fromarray(output_8).save(output_path)
        return True
    except Exception:
        return False
    finally:
        try:
            del dat_model
        except Exception:
            pass
