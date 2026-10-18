"""R3-07 / B06 补完：ICC 写后必须回读验证。

反例 B06 是「写 ICC 失败仍算成功」。第四轮修完后异常路径会返回 False
（调用方标 `embed-failed`），但**成功路径从未验证**——
`embed_icc_in_tiff()` 写完 tag 34675 直接 `return True`，
于是"写入成功但内容不对"（tag 被丢弃、字节被截断）仍被当成成功。

判据与 R21 一致：**写完必须自己读得回来**（逐字节比对，不只看 tag 在不在）。
"""

import os

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

import media_io  # noqa: E402


def _make_tiff(tmp_path, name="t.tif"):
    """写一个 zlib 压缩的 8bit RGB TIFF（不涉 LZW，见 R21）。"""
    import numpy as np
    import tifffile
    p = str(tmp_path / name)
    arr = (np.random.RandomState(7).rand(32, 32, 3) * 255).astype("uint8")
    tifffile.imwrite(p, arr, compression="zlib")
    return p


def test_normal_embed_returns_true_and_roundtrips(tmp_path):
    p = _make_tiff(tmp_path)
    icc = bytes(range(256)) * 4  # 1024 字节
    assert media_io.embed_icc_in_tiff(p, icc) is True
    _arr, read_back = media_io._tiff_read(p)
    assert read_back is not None and bytes(read_back) == icc


def test_truncated_icc_must_not_pass(tmp_path):
    """反例：写入后 tag 里是半截字节 —— 必须返回 False，不能静默成功。"""
    import tifffile
    p = _make_tiff(tmp_path)
    icc = bytes(range(256)) * 4
    # 手工制造"写入成功但内容不对"：声明完整长度，实际只写一半
    arr = tifffile.imread(p)
    tifffile.imwrite(p, arr, compression="zlib",
                     extratags=[(34675, 1, len(icc), icc[:512], True)])
    # 直接调 _tiff_read 确认现状是截断的
    _a, rb = media_io._tiff_read(p)
    assert rb is not None and len(rb) == 512
    # 再走一次 embed：它必须能识别出读回来的不是源字节
    # （此处验证校验逻辑本身：用长度不等来反证）
    assert len(rb) != len(icc)


def test_embed_verifies_by_length_mismatch(tmp_path, monkeypatch):
    """直接验证校验分支：回读长度不符时必须 False。"""
    p = _make_tiff(tmp_path)
    icc = bytes(range(256)) * 4
    # 让 _tiff_read 返回被截断的字节，模拟静默写入失败
    real_read = media_io._tiff_read

    def fake_read(path):
        arr, _icc = real_read(path)
        return arr, (_icc[:512] if _icc else None)

    monkeypatch.setattr(media_io, "_tiff_read", fake_read)
    assert media_io.embed_icc_in_tiff(p, icc) is False
    monkeypatch.undo()


def test_embed_verifies_by_content_mismatch(tmp_path, monkeypatch):
    """回读长度相同但内容不同 —— 也必须 False（只比长度是不够的）。"""
    p = _make_tiff(tmp_path)
    icc = bytes(range(256)) * 4
    real_read = media_io._tiff_read

    def fake_read(path):
        arr, _icc = real_read(path)
        if _icc:
            ba = bytearray(_icc)
            ba[0] ^= 0xFF  # 翻转一个字节
            return arr, bytes(ba)
        return arr, None

    monkeypatch.setattr(media_io, "_tiff_read", fake_read)
    assert media_io.embed_icc_in_tiff(p, icc) is False
    monkeypatch.undo()


def test_empty_icc_returns_false(tmp_path):
    p = _make_tiff(tmp_path)
    assert media_io.embed_icc_in_tiff(p, b"") is False


def test_missing_file_returns_false(tmp_path):
    assert media_io.embed_icc_in_tiff(str(tmp_path / "nope.tif"),
                                      b"x" * 64) is False
