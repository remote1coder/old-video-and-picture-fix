"""B9：AI 降级路径（4x→2x→CPU）的输出必须仍然正确。

本轮矩阵验了"架构/倍数对应关系"，但没验降级路径：
`stage3_ai_upscale.py` 的 OOM 三态（True/False/"OOM"）会触发
4x→2x→CPU 降级链（约 1308–1331 行）。需确认降级后：

1. **模型随倍数同步切换**（4x 模型配 2x 会形状不符）；
2. **降级终点（CPU）输出尺寸 = 输入 × 实际倍数**；
3. 帧数不丢。

实测（`run_realesrgan_batch` + `gpu_id=-1`）：
64×64 输入 × 2 → **128×128**，帧数 2/2，返回 True。

注：真实 OOM 需要显存不足才能触发，本文件测的是**降级链的终点**
（CPU 路径）与**模型切换规则**，这两者可确定性复现。
"""

import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_stage3():
    vstages = os.path.join(ROOT, "video_restoration_pipeline", "stages")
    if vstages not in sys.path:
        sys.path.insert(0, vstages)
    import stage3_ai_upscale as s3  # noqa: E402
    return s3


def test_model_switches_with_downgraded_scale():
    """4x->2x 降级时，模型必须同步切换（否则架构形状不符）。

    realesr-general-x4v3 是 4× 网络但可用 outscale 出 2×，故保持不变；
    x4plus / DAT_x4 则有独立的 2× 权重。
    """
    s3 = _load_stage3()
    assert s3._get_model_for_scale("RealESRGAN_x4plus", 2) == \
        "RealESRGAN_x2plus"
    assert s3._get_model_for_scale("DAT_x4", 2) == "DAT_x2"
    # 4x 网络用 outscale 出 2x，模型名不变（不是 bug）
    assert s3._get_model_for_scale("realesr-general-x4v3", 2) == \
        "realesr-general-x4v3"


def test_scale4_models_map_back_at_scale4():
    """不降级时不应改模型名。"""
    s3 = _load_stage3()
    assert s3._get_model_for_scale("RealESRGAN_x4plus", 4) == \
        "RealESRGAN_x4plus"


@pytest.mark.slow
def test_cpu_degrade_endpoint_output_size(tmp_path):
    """降级终点（CPU）：输出必须是 输入 × 实际倍数，且帧数不丢。"""
    s3 = _load_stage3()
    import numpy as np
    from PIL import Image

    wd = tmp_path / "w0"
    wd.mkdir()
    out = tmp_path / "w0_out"
    out.mkdir()
    arr = (np.random.RandomState(1).rand(64, 64, 3) * 255).astype("uint8")
    for i in range(2):
        Image.fromarray(arr).save(str(wd / f"frame_{i:08d}.png"))

    r = s3.run_realesrgan_batch([(str(wd), str(out))],
                                "realesr-general-x4v3", 2,
                                gpu_id=-1, logger=None)
    assert r is True, "CPU 降级路径应成功"
    pngs = sorted(out.glob("*.png"))
    assert len(pngs) == 2, f"帧数应保持一致，实际 {len(pngs)}"
    got = np.array(Image.open(str(pngs[0])))
    h, w = got.shape[:2]
    assert (w, h) == (128, 128), f"期望 128x128，实际 {w}x{h}"
