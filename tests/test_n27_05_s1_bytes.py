# -*- coding: utf-8 -*-
"""S1 的内存入口（N27-05 的第一块）。

与 `main()` 的分工：`main()` 处理色彩/ICC/EXIF/sidecar（需要文件元数据），
本函数只做"PNG 字节 → 规范化 RGB16 PNG 字节"（含 G'MIC 三条链）。
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared_modules"))

import gmic_stdio as G  # noqa: E402
import media_io  # noqa: E402
from conftest import load_stage  # noqa: E402

# 与 test_r26_a10 同理：按路径加载，不插 sys.path
s1 = load_stage("photo_restoration_pipeline", "stage1_preprocess")

pytestmark = pytest.mark.skipif(not G.find_gmic(),
                                reason="gmic executable not found")


def _png(arr):
    # **PIL 不支持多通道 16bit PNG**（`(h,w,3), '<u2'` → KeyError）。
    # 项目里的 `media_io.encode_png` 用 zlib 自己拼，才支持。
    return media_io.encode_png(arr)


def _rgb16(h=24, w=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65535, (h, w, 3), dtype=np.uint16)


#: 关掉两条可选链，只验证几何/位深规范化
NO_GMIC = {"deblock_strength": 0, "scratch_remove_strength": 0,
           "deband_threshold": 0}


def test_rgb16_passthrough_keeps_geometry_and_depth():
    arr = _rgb16()
    r = s1.preprocess_png_bytes(_png(arr), NO_GMIC)
    assert r["ok"] is True, r["error"]
    assert (r["width"], r["height"]) == (32, 24)
    assert r["bit_depth"] == 16 and r["channels"] == 3
    back = media_io.read_png_bytes(r["png"])
    assert back.shape == arr.shape and back.dtype == np.uint16


def test_8bit_input_is_lifted_to_16bit():
    arr = (np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3))
    r = s1.preprocess_png_bytes(_png(arr), NO_GMIC)
    assert r["ok"] is True, r["error"]
    assert r["bit_depth"] == 16
    assert "8bit->16bit" in r["steps"]


def test_gray_input_becomes_rgb():
    arr = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
    r = s1.preprocess_png_bytes(_png(arr), NO_GMIC)
    assert r["ok"] is True, r["error"]
    assert r["channels"] == 3
    assert "gray->rgb" in r["steps"]


def test_rgba_drops_alpha_and_declares_it():
    arr = np.dstack([_rgb16(), np.full((24, 32), 30000, np.uint16)])
    r = s1.preprocess_png_bytes(_png(arr), NO_GMIC)
    assert r["ok"] is True, r["error"]
    assert r["channels"] == 3
    assert any("alpha" in w for w in r["warnings"]), r["warnings"]


def test_deband_chain_runs_through_gmic():
    """deband 是必需链：它必须真的在内存里跑过 G'MIC。"""
    arr = _rgb16()
    r = s1.preprocess_png_bytes(_png(arr), {"deblock_strength": 0,
                                            "scratch_remove_strength": 0,
                                            "deband_threshold": 1.2})
    assert r["ok"] is True, r["error"]
    assert any("denoise_haar(deband)" in s for s in r["steps"]), r["steps"]
    back = media_io.read_png_bytes(r["png"])
    assert back.shape == arr.shape
    assert not np.array_equal(back, arr), "deband 应该真的改了像素"


def test_optional_chain_failure_is_a_warning_not_a_failure():
    """可选链（scratch/deblock）失败 → 记 warning，整张仍成功。"""
    arr = _rgb16()
    # 极端参数让 patchpca 更容易崩（4.0.6 已知会崩）；结果是 warning
    r = s1.preprocess_png_bytes(
        _png(arr), {"deblock_strength": 25, "scratch_remove_strength": 0,
                    "deband_threshold": 0})
    assert r["ok"] is True, r["error"]
    # 无论 patchpca 成功还是失败，都不能因此判整张失败
    assert r["png"]


def test_bad_input_is_rejected():
    r = s1.preprocess_png_bytes(b"not a png", NO_GMIC)
    assert r["ok"] is False
    assert "not a decodable PNG" in r["error"], r["error"]


def test_empty_input_is_rejected():
    r = s1.preprocess_png_bytes(b"", NO_GMIC)
    assert r["ok"] is False
    assert "empty" in r["error"]
