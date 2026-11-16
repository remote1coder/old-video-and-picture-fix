# -*- coding: utf-8 -*-
"""照片全链端到端（真实模型、真实 GPU）。

这是"照片链能跑"的**验收测试**，不是单元测试：
S1 → S2 → S3（RealESRGAN）→ S4（GFPGAN）→ S5 全程内存，无中间 TIFF。

标 `slow`：一次约 6 秒（含模型加载 ~4 秒）。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared_modules"))

import media_io  # noqa: E402
import photo_chain as PC  # noqa: E402

pytest.importorskip("scipy")


def _has_torch_python():
    from photo_model_client import find_ai_python
    p = find_ai_python()
    return bool(p) and Path(p).is_file() and ".venv_r4" in str(p)


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _has_torch_python(),
                       reason="no .venv_r4 interpreter with torch"),
]


def _test_image(h=48, w=64):
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    base = ((y * x) * 65535).astype(np.uint16)
    return np.stack([base, base // 2, 65535 - base], axis=2)


#: 关掉 G'MIC 三条链（本测试只验"能串起来"，不验滤镜效果）
NO_GMIC = {"deblock_strength": 0, "scratch_remove_strength": 0,
           "deband_threshold": 0, "photo_bit_depth": 16,
           "upscale_model": "realesr-general-x4v3", "upscale_scale": 2,
           "ai_half": True, "face_upscale": 2}


def test_full_chain_runs_with_real_models():
    """S1→S2→S3→S4→S5 全部跑过，几何只受 S3 的 2x 影响。"""
    from photo_model_client import ModelWorkerClient

    arr = _test_image()
    png = media_io.encode_png(arr)
    with ModelWorkerClient() as client:
        r = PC.run_inproc_chain(png, dict(NO_GMIC), client=client,
                                stages=["s1", "s2", "s3", "s4", "s5"])
    assert r["ok"] is True, (r["error"], r["trace"][-2:])
    assert r["stages_run"] == ["s1", "s2", "s3", "s4", "s5"], r["stages_run"]

    out = media_io.read_png_bytes(r["png"])
    assert out.dtype == np.uint16, out.dtype
    assert (out.shape[1], out.shape[0]) == (arr.shape[1] * 2,
                                            arr.shape[0] * 2), (
        "几何应该是 2x（只有 S3 放大）；GFPGAN 的 upscale 是副作用，"
        "必须被缩回去，实测 %.0fx" % (out.shape[1] / arr.shape[1]))


def test_chain_without_client_refuses_model_stages():
    """没有 client 时请求 S3/S4 → 拒绝（不是"跳过继续"）。"""
    r = PC.run_inproc_chain(media_io.encode_png(_test_image()), dict(NO_GMIC),
                            stages=["s1", "s3"])
    assert r["ok"] is False
    assert "s3" in r["unsupported"]
    assert r["stages_run"] == []
