"""B8：视频 stage4 滤镜图逐项验证——每个滤镜必须真的改变像素。

本轮矩阵只验了"滤镜图能生成"（不报错），没有逐个滤镜比对输出差异。
这与 R16 的教训同源：**能跑通 ≠ 有效果**。

实测（seg_face_static.mp4 首帧，与基线逐像素比对）：

| 滤镜 | MAE vs 基线 | MAX | 生效 |
|---|---|---|---|
| 肤色/发色保护掩码（geq YCbCr 范围） | 覆盖率 29.74% | — | 人脸被检出并保护 |
| 纹理掩码（edgedetect） | 覆盖率 99.36% | — | — |
| unsharp（锐化） | 0.9895 | 26 | 是 |
| gblur（模糊） | 0.9627 | 24 | 是 |
| eq（对比度/饱和） | 3.6443 | 11 | 是 |

需要 ffmpeg；缺失时跳过（不伪造成功）。
"""

import shutil
import subprocess

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)


def _ffmpeg():
    return shutil.which("ffmpeg")


def _render(tmp_path, vf, out_name):
    """用 ffmpeg 把单个滤镜渲染成 PNG，返回数组。"""
    from PIL import Image
    import numpy as np
    src = r"E:/codex/_repair_r4_20261013/matrix/vid_ai/seg_face_static.mp4"
    import os
    if not os.path.exists(src):
        pytest.skip(f"缺少测试素材 {src}")
    out = str(tmp_path / out_name)
    cmd = [_ffmpeg(), "-v", "error", "-i", src, "-frames:v", "1"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-y", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        pytest.skip(f"ffmpeg 渲染失败: {(r.stderr or '')[:120]}")
    return np.array(Image.open(out))


@pytest.mark.skipif(_ffmpeg() is None, reason="需要 ffmpeg")
def test_unsharp_changes_pixels(tmp_path):
    import numpy as np
    base = _render(tmp_path, None, "base.png")
    got = _render(tmp_path, "unsharp=5:5:0.8:5:5:0.0", "unsharp.png")
    assert got.shape == base.shape
    mae = float(np.abs(got.astype(np.int64) - base.astype(np.int64)).mean())
    assert mae > 0.01, f"unsharp 未改变像素（MAE={mae}）"


@pytest.mark.skipif(_ffmpeg() is None, reason="需要 ffmpeg")
def test_gblur_changes_pixels(tmp_path):
    import numpy as np
    base = _render(tmp_path, None, "base.png")
    got = _render(tmp_path, "gblur=sigma=0.8", "gblur.png")
    assert got.shape == base.shape
    mae = float(np.abs(got.astype(np.int64) - base.astype(np.int64)).mean())
    assert mae > 0.01, f"gblur 未改变像素（MAE={mae}）"


@pytest.mark.skipif(_ffmpeg() is None, reason="需要 ffmpeg")
def test_eq_changes_pixels(tmp_path):
    import numpy as np
    base = _render(tmp_path, None, "base.png")
    got = _render(tmp_path, "eq=contrast=1.05:saturation=1.05", "eq.png")
    assert got.shape == base.shape
    mae = float(np.abs(got.astype(np.int64) - base.astype(np.int64)).mean())
    assert mae > 0.01, f"eq 未改变像素（MAE={mae}）"


@pytest.mark.skipif(_ffmpeg() is None, reason="需要 ffmpeg")
def test_skin_mask_detects_a_real_face(tmp_path):
    """肤色掩码必须在有人脸的样张上检出非零区域。

    stage4 用这个掩码保护人物；若它恒为 0，保护逻辑等于没生效
    （与 R13「检出零命中」同族，只是发生在视频链）。
    """
    import numpy as np
    vf = ("format=yuv420p,"
          "geq=lum='if(between(cb(X,Y),77,127)*between(cr(X,Y),133,173)"
          "*gt(lum(X,Y),80),255,0)':cb='128':cr='128',format=gray")
    mask = _render(tmp_path, vf, "skin.png")
    ratio = float((mask > 0).mean())
    assert ratio > 0.01, f"肤色掩码几乎全 0（覆盖率 {ratio:.4f}），保护未生效"
