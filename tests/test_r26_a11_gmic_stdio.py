# -*- coding: utf-8 -*-
"""G'MIC 的 stdin/stdout 端点（R26-A11 事实更正）。

上一轮把"G'MIC CLI 只吃文件"当成照片链不能全链流式的理由。复核纠正：
官方支持 `-.ext` 类型化 stdin / `-o -.ext` stdout，并在本机 4.0.6 上
用它处理过 uint16 合成图（rc=0、逐值相等、max_abs=0）。

本文件钉住 adapter 的**三重校验**：rc → 魔数 → 解码 + 几何/位深。
"""
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared_modules"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import gmic_stdio as G  # noqa: E402
import media_io  # noqa: E402

pytestmark = pytest.mark.skipif(not G.find_gmic(),
                                reason="gmic executable not found")


def _png16() -> bytes:
    arr = np.arange(11 * 7, dtype=np.uint16).reshape(7, 11) * 97
    buf = io.BytesIO()
    Image.fromarray(arr, mode="I;16").save(buf, format="PNG")
    return buf.getvalue()


def _png8() -> bytes:
    arr = np.arange(11 * 7, dtype=np.uint8).reshape(7, 11) * 3
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def test_identity_roundtrip_preserves_every_value():
    src = _png16()
    r = G.gmic_filter_png(src, [], expect_width=11, expect_height=7,
                          expect_channels=1, expect_bit_depth=16)
    assert r["ok"] is True, r["error"]
    assert r["bit_depth"] == 16
    back = media_io.read_png_bytes(r["png"])
    orig = media_io.read_png_bytes(src)
    assert np.array_equal(back.astype(np.int64), orig.astype(np.int64)), (
        "identity 滤镜必须逐值不变")


def test_filter_changes_pixels_but_keeps_geometry():
    src = _png16()
    r = G.gmic_filter_png(src, ["-blur", "2"], expect_width=11,
                          expect_height=7, expect_bit_depth=16)
    assert r["ok"] is True, r["error"]
    back = media_io.read_png_bytes(r["png"])
    orig = media_io.read_png_bytes(src)
    assert back.shape == orig.shape
    assert not np.array_equal(back, orig), "滤镜应该真的改了像素"


def test_bad_input_is_rejected_by_rc():
    r = G.gmic_filter_png(b"NOT A PNG AT ALL", [])
    assert r["ok"] is False
    assert r["error"], r
    # 即使某些版本在 stdout 留了帮助文本，也不能当成图
    assert r["png"] == b""


def test_declared_bit_depth_mismatch_is_rejected():
    """声明 16bit 但输入是 8bit → identity 不会升位深，必须被抓住。"""
    r = G.gmic_filter_png(_png8(), [], expect_bit_depth=16)
    assert r["ok"] is False, "位深与声明不符却通过了"
    assert "bit depth" in r["error"], r["error"]


def test_missing_gmic_is_an_explicit_error():
    r = G.gmic_filter_png(_png16(), [], gmic="Z://nope//gmic.exe")
    assert r["ok"] is False
    assert "not found" in r["error"] or "cannot run" in r["error"]


def test_empty_input_is_rejected():
    r = G.gmic_filter_png(b"", [])
    assert r["ok"] is False
    assert "empty" in r["error"]
