# -*- coding: utf-8 -*-
"""S5 的内存入口（N27-05）——必须复用同一个算法核心。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared_modules"))

import media_io  # noqa: E402
from conftest import load_stage  # noqa: E402

pytest.importorskip("scipy", reason="S5 的选择性处理依赖 scipy")

s5 = load_stage("photo_restoration_pipeline", "stage5_postprocess")


def _rgb16(seed=0, h=20, w=24):
    rng = np.random.default_rng(seed)
    a = rng.integers(8000, 60000, (h, w, 3), dtype=np.uint16)
    # 加点结构，让"边缘/纹理/平坦"三类区域都能被分出来
    a[5:10, 5:10, :] = 30000
    a[12:18, 14:22, :] = 12000
    return a


def test_geometry_and_depth_are_preserved():
    arr = _rgb16()
    r = s5.postprocess_bytes(media_io.encode_png(arr),
                             {"photo_bit_depth": 16})
    assert r["ok"] is True, r["error"]
    assert r["bit_depth"] == 16
    assert (r["width"], r["height"]) == (24, 20)
    back = media_io.read_png_bytes(r["png"])
    assert back.shape == arr.shape and back.dtype == np.uint16


def test_processing_actually_changes_pixels():
    """选择性处理应该真的动了像素（否则等于没跑）。"""
    arr = _rgb16()
    r = s5.postprocess_bytes(media_io.encode_png(arr),
                             {"photo_bit_depth": 16})
    assert r["ok"] is True, r["error"]
    back = media_io.read_png_bytes(r["png"])
    assert not np.array_equal(back, arr), "S5 没改任何像素"


def test_bad_input_is_rejected():
    r = s5.postprocess_bytes(b"nope", {"photo_bit_depth": 16})
    assert r["ok"] is False
    assert "not a decodable PNG" in r["error"]


def test_non_rgb16_is_rejected():
    arr = np.zeros((10, 10, 3), np.uint8)
    r = s5.postprocess_bytes(media_io.encode_png(arr), {})
    assert r["ok"] is False
    assert "expects RGB16" in r["error"], r["error"]
