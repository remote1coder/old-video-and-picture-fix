# -*- coding: utf-8 -*-
"""S2 的内存入口（N27-05）。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared_modules"))

import media_io  # noqa: E402
from conftest import load_stage  # noqa: E402

s2 = load_stage("photo_restoration_pipeline", "stage2_ai_pre")


def _rgb16(seed=0, h=16, w=16):
    rng = np.random.default_rng(seed)
    return rng.integers(20000, 45000, (h, w, 3), dtype=np.uint16)


def test_no_degradation_returns_input_unchanged():
    arr = _rgb16()
    png = media_io.encode_png(arr)
    r = s2.correct_channels_bytes(png, {})
    assert r["ok"] is True, r["error"]
    assert r["changed"] is False
    assert r["png"] == png, "没有退化时必须原样返回（不是重新编码）"


def test_degraded_channel_is_corrected():
    arr = _rgb16()
    # 人为压暗蓝通道，模拟"褪色"
    arr[:, :, 2] = (arr[:, :, 2].astype(np.float32) * 0.45).astype(np.uint16)
    png = media_io.encode_png(arr)
    r = s2.correct_channels_bytes(png, {}, degraded_channels=[2])
    assert r["ok"] is True, r["error"]
    assert r["changed"] is True
    assert r["corrected_channels"] == [2]
    back = media_io.read_png_bytes(r["png"])
    # 蓝通道应该被抬起来（与其它通道更接近）
    assert back[:, :, 2].mean() > arr[:, :, 2].mean() * 1.2


def test_dict_descriptor_is_accepted():
    """stage1 的 sidecar 用 `{"channel": 2, ...}`，裸整数也要接受。"""
    arr = _rgb16(1)
    arr[:, :, 0] = (arr[:, :, 0].astype(np.float32) * 0.5).astype(np.uint16)
    png = media_io.encode_png(arr)
    r = s2.correct_channels_bytes(
        png, {}, degraded_channels=[{"channel": 0, "mean": 1.0}])
    assert r["ok"] is True, r["error"]
    assert r["corrected_channels"] == [0]


def test_non_iec_formula_is_rejected():
    """与 file 模式一致：非 IEC 公式是参数错误，不静默退回 pow(2.2)。"""
    arr = _rgb16()
    png = media_io.encode_png(arr)
    r = s2.correct_channels_bytes(
        png, {"channel_correction_formula": "pow22"}, degraded_channels=[1])
    assert r["ok"] is False
    assert "iec61966-2-1" in r["error"], r["error"]


def test_bad_input_is_rejected():
    r = s2.correct_channels_bytes(b"not a png", {}, degraded_channels=[1])
    assert r["ok"] is False
    assert "not a decodable PNG" in r["error"]


def test_non_rgb16_is_rejected():
    arr = (np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3))
    r = s2.correct_channels_bytes(media_io.encode_png(arr), {},
                                  degraded_channels=[1])
    assert r["ok"] is False
    assert "expects RGB16" in r["error"], r["error"]
