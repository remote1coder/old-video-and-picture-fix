"""V01/V02/V05 regression: encoder input adaptation + bit-depth contract.

Targeted regressions only — no model inference, no GPU. Facts asserted here
come from the libavif manual and the Pillow behaviour observed on this machine.
"""

import os
import sys

import numpy as np
import pytest

import conftest  # noqa: F401  (sys.path bootstrap)
import media_io


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out")


@pytest.fixture(scope="module")
def outdir():
    os.makedirs(OUT, exist_ok=True)
    return OUT


def _grad16(h=8, w=6):
    """16-bit gradient with adjacent values that 8-bit quantisation merges."""
    arr = np.zeros((h, w, 3), dtype=np.uint16)
    arr[..., 0] = (np.arange(h * w) * 64).astype(np.uint16).reshape(h, w)
    arr[..., 1] = 1000
    arr[..., 2] = 65535
    return arr


# --- V01: --depth contract --------------------------------------------------

def test_v01_bare_depth_16_is_never_emitted():
    """avifenc --depth only accepts 8/10/12; 16 needs a documented extension."""
    cap = media_io.AvifencCapability(available=True, version=(1, 4, 0))
    depth_arg, effective, info = media_io.resolve_depth_spec(16, None, cap)
    assert depth_arg == "12,8", depth_arg
    assert effective == 16
    assert info["mode"] == "depth-extension"
    assert info["primary_depth"] == 12 and info["extension_depth"] == 8


def test_v01_supported_base_depths():
    cap = media_io.AvifencCapability(available=True, version=(1, 4, 0))
    for depth in (8, 10, 12):
        arg, effective, _ = media_io.resolve_depth_spec(depth, None, cap)
        assert arg == str(depth) and effective == depth


def test_v01_supported_extension_combinations():
    cap = media_io.AvifencCapability(available=True, version=(1, 4, 0))
    for combo in sorted(media_io.AVIFENC_DEPTH_EXTENSIONS):
        arg, effective, info = media_io.resolve_depth_spec(combo[0], combo[1], cap)
        assert arg == f"{combo[0]},{combo[1]}"
        assert effective == 16
        assert info["mode"] == "depth-extension"


def test_v01_16bit_refused_when_encoder_lacks_extension():
    """Must fail, not silently truncate 16 -> 12 while still saying 16-bit."""
    old = media_io.AvifencCapability(available=True, version=(1, 2, 0))
    with pytest.raises(media_io.UnsupportedAvifConfig):
        media_io.resolve_depth_spec(16, None, old)
    # explicit authorisation records the downgrade as 12-bit
    arg, effective, info = media_io.resolve_depth_spec(16, None, old,
                                                       allow_downgrade=True)
    assert (arg, effective) == ("12", 12)
    assert info["downgraded"] is True


def test_v01_command_is_validated_before_launch():
    cap = media_io.AvifencCapability(available=True, version=(1, 4, 0))
    cmd, contract = media_io.build_avifenc_cmd(
        "in.png", "out.avif", {"avif_rgb444_params": {"depth": 16}}, True, cap)
    assert "--depth" in cmd
    assert cmd[cmd.index("--depth") + 1] == "12,8"
    assert contract["effective_decode_depth"] == 16
    assert contract["depth_info"]["decode_requires"].startswith("avifdec")


def test_v01_invalid_choices_rejected():
    cap = media_io.AvifencCapability(available=True, version=(1, 4, 0))
    for bad in ({"yuv": "411"}, {"cicp": "1/13"}, {"range": "tv"},
                {"speed": 42}):
        with pytest.raises(media_io.UnsupportedAvifConfig):
            media_io.build_avifenc_cmd("in.png", "o.avif",
                                       {"avif_params": bad}, False, cap)


# --- V05: encoder input adaptation -----------------------------------------

def test_v05_tiff_input_is_not_accepted_by_avifenc():
    assert ".tif" not in media_io.AVIFENC_SUPPORTED_INPUT_EXTS
    assert ".tiff" not in media_io.AVIFENC_SUPPORTED_INPUT_EXTS
    cap = media_io.AvifencCapability(available=True, version=(1, 4, 0))
    with pytest.raises(media_io.UnsupportedAvifConfig):
        media_io.build_avifenc_cmd("intermediate.tif", "o.avif", {}, False, cap)


def test_v05_adapter_converts_tiff_to_png_without_precision_loss(outdir):
    src = os.path.join(outdir, "v05_source.tif")
    arr = _grad16()
    media_io.write_tiff(src, arr)

    res = media_io.prepare_encoder_input(src, os.path.join(outdir, "ad"))
    assert res["converted"] is True
    assert res["source_dtype"] == "uint16"
    assert res["path"].lower().endswith(".png")
    assert res["verification"].endswith(":exact"), res["verification"]

    # independent decoder confirms the 16-bit values survived
    try:
        import imagecodecs
    except Exception:  # pragma: no cover
        pytest.skip("imagecodecs not installed (independent PNG decoder)")
    back = imagecodecs.imread(res["path"])
    assert back.dtype == np.uint16
    assert np.array_equal(back, arr)


def test_v05_adapter_keeps_8bit_and_png_passthrough(outdir):
    from PIL import Image
    png = os.path.join(outdir, "v05_passthrough.png")
    Image.fromarray(np.full((4, 4, 3), 7, dtype=np.uint8), mode="RGB").save(png)
    res = media_io.prepare_encoder_input(png, os.path.join(outdir, "ad"))
    assert res["converted"] is False
    assert res["path"] == png


def test_v05_grayscale_and_alpha_roundtrip(outdir):
    for arr in (np.linspace(0, 65535, 12, dtype=np.uint16).reshape(3, 4),
                np.dstack((np.zeros((4, 4), dtype=np.uint16),
                           np.full((4, 4), 32768, dtype=np.uint16),
                           np.full((4, 4), 1000, dtype=np.uint16),
                           np.full((4, 4), 65535, dtype=np.uint16)))):
        p = os.path.join(outdir, f"v05_{arr.ndim}c.png")
        media_io.write_png(p, arr)
        back = media_io.read_png(p)
        assert back.shape == arr.shape, (back.shape, arr.shape)
        assert np.array_equal(back, arr)


# --- V02: high-bit-depth writing -------------------------------------------

def test_v02_tiff_master_keeps_real_precision(outdir):
    tifffile = pytest.importorskip("tifffile")
    p = os.path.join(outdir, "v02_master.tif")
    arr = _grad16()
    media_io.write_tiff(p, arr)
    back = tifffile.imread(p)
    assert back.dtype == np.uint16
    assert np.array_equal(back, arr)


def test_v02_internal_tiff_writer_is_readable_by_third_party(outdir, monkeypatch):
    """With tifffile unavailable the internal writer must still emit real 16-bit."""
    tifffile = pytest.importorskip("tifffile")
    monkeypatch.setitem(sys.modules, "tifffile", None)
    p = os.path.join(outdir, "v02_internal.tif")
    arr = _grad16()
    writer = media_io.write_tiff(p, arr)
    monkeypatch.undo()
    assert writer == "internal-tiff"
    back = tifffile.imread(p)
    assert back.dtype == np.uint16
    assert np.array_equal(back, arr)


def test_v02_original_defect_is_reproduced(outdir):
    """Documented evidence: Pillow 'RGB' -> TIFF yields 8-bit samples."""
    from PIL import Image
    p = os.path.join(outdir, "v02_pillow_rgb.tif")
    img = Image.new("RGB", (8, 8), (10, 200, 30))
    img.save(p, format="TIFF", compression="tiff_lzw")
    with Image.open(p) as r:
        assert r.tag_v2.get(258) == (8, 8, 8)

# ---------------------------------------------------------------------------
# NEW-V-02: PNG 反滤波必须逐字节进行（进位丢弃）
# ---------------------------------------------------------------------------

def _third_party_png_writer():
    """返回一个能用 libpng 写出 PNG 的 (name, write_func)，都没有则 None。"""
    try:
        import cv2  # type: ignore
        def _write(path, arr):
            # cv2 以 BGR/BGRA 语义读写，这里来回翻转抵消
            cv2.imwrite(path, arr[:, :, ::-1])
        return "cv2", _write
    except Exception:
        pass
    try:
        import imagecodecs  # type: ignore
        def _write(path, arr):
            imagecodecs.imwrite(path, arr)
        return "imagecodecs", _write
    except Exception:
        return None


def test_new_v02_reader_matches_libpng_on_filtered_16bit_rgb(outdir):
    """自家解码器必须与 libpng 逐像素一致（旧实现错把进位传进高字节）。"""
    tp = _third_party_png_writer()
    if tp is None:
        pytest.skip("需要 cv2 或 imagecodecs 才能产出 libpng 自适应滤波的 16-bit PNG")
    name, write = tp
    rng = np.random.default_rng(20261002)
    arr = rng.integers(0, 65536, (37, 53, 3)).astype(np.uint16)
    p = os.path.join(outdir, "new_v02_rgb16.png")
    write(p, arr)
    mine = media_io.read_png(p)
    assert mine.dtype == np.uint16, f"回读丢失精度: {mine.dtype}"
    assert np.array_equal(mine, arr), (
        f"独立解码器({name})与原数组不一致 —— PNG 反滤波不是逐字节执行"
    )


def test_new_v02_carry_must_not_leak_into_high_byte():
    """构造必然产生低字节进位的样本：旧实现会在高字节上多出 +256。"""
    tp = _third_party_png_writer()
    if tp is None:
        pytest.skip("需要 cv2 或 imagecodecs")
    name, write = tp
    # 相邻像素低字节和 >= 256、高字节不同 —— Sub 滤波必然触发进位
    arr = np.zeros((2, 4, 3), dtype=np.uint16)
    arr[0] = [[0x1234, 0x00ff, 0x0101], [0x1201, 0x0100, 0x0202],
              [0x00f0, 0x00ff, 0x0303], [0x1100, 0x0001, 0x0404]]
    arr[1] = arr[0][::-1].copy()
    p = os.path.join(os.path.dirname(__file__), "_out", "new_v02_carry.png")
    write(p, arr)
    mine = media_io.read_png(p)
    assert np.array_equal(mine, arr), "低字节进位被错误传播到高字节"
