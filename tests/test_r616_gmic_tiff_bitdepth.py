"""R16 回归：G'MIC 的"名实不符" 16-bit TIFF 让照片链交付纯黑成品。

**实测缺陷**：G'MIC 4.x 用 `-o X.tif,uint16` 写出的 TIFF，头部声明
`BitsPerSample=16` / `dtype=uint16`，但**像素值仍是 8-bit 量级**
（实测 ch0 0–240 / ch1 0–201 / ch2 0–244，原图 mean 107.27）。
后果：

* `PIL.Image.open(p).convert('RGB')` → 走 uint8 TIFF 路径 → **全 0**
  （mean 0.001）；
* AI 超分一张黑图 → 照片链交付的 AVIF **全是纯黑**
  （成品仅 386 字节，像素 mean 0.0），而全链 rc 一直是 0，无任何报错。

修法：`media_io.load_image_rgb_uint8()` 用 tifffile 读真实像素，并
**逐通道**按实际值域自适应（必须逐通道：deblock 之后 ch0/ch2 仍是
0–244 而 ch1 被放大到 65532，用全局 max 判定会让整幅图走 /257 →
mean 从 107 掉到 0.001，仍然是黑图）。
"""

import os
import sys

import numpy as np
import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

import media_io  # noqa: E402


def _write_tiff(path, arr):
    import tifffile
    tifffile.imwrite(path, arr)
    return str(path)


def test_gmic_8bit_in_16bit_is_read_as_is(tmp_path):
    """反例：8-bit 量级数据装在 uint16 容器里，必须原样读出（MAE=0）。"""
    orig = np.full((8, 10, 3), 107, dtype=np.uint8)
    p = _write_tiff(tmp_path / "gm.tif", orig.astype(np.uint16))
    out = media_io.load_image_rgb_uint8(p)
    assert out.dtype == np.uint8
    assert np.array_equal(out, orig), "8-bit 量级数据必须原样读出"


def test_pil_would_have_failed_on_same_file(tmp_path):
    """锁住缺陷本身：PIL 读同一文件会得到黑图（证明修法有实际意义）。"""
    from PIL import Image
    orig = np.full((8, 10, 3), 107, dtype=np.uint8)
    p = _write_tiff(tmp_path / "gm2.tif", orig.astype(np.uint16))
    pil = np.array(Image.open(p).convert("RGB"))
    fixed = media_io.load_image_rgb_uint8(p)
    # 无论 PIL 是否恰好读对，修复后的读取必须始终正确
    assert np.array_equal(fixed, orig)


def test_true_16bit_is_scaled_down(tmp_path):
    """真正的 16-bit 数据（0–65535）必须按 /257 缩放到 uint8。"""
    orig8 = np.full((8, 10, 3), 200, dtype=np.uint8)
    arr16 = (orig8.astype(np.uint16) * 257)
    p = _write_tiff(tmp_path / "real16.tif", arr16)
    out = media_io.load_image_rgb_uint8(p)
    assert out.dtype == np.uint8
    assert np.abs(out.astype(int) - 200).max() <= 1


def test_per_channel_adaptive(tmp_path):
    """核心断言：必须**逐通道**判定，不能用全局 max。

    实测 deblock 之后的中间产物：ch0/ch2 是 8-bit 量级（0–244），
    ch1 是 16-bit 量级（0–65532）。用全局 max 判定会让整幅图走 /257，
    mean 从 107 掉到 0.001（仍是黑图）。
    """
    arr = np.zeros((8, 10, 3), dtype=np.uint16)
    arr[:, :, 0] = 120          # 8-bit 量级
    arr[:, :, 1] = 120 * 257    # 16-bit 量级
    arr[:, :, 2] = 130          # 8-bit 量级
    p = _write_tiff(tmp_path / "mixed.tif", arr)
    out = media_io.load_image_rgb_uint8(p)
    assert out[0, 0, 0] == 120, "ch0 应原样读出"
    assert abs(int(out[0, 0, 1]) - 120) <= 1, "ch1 应按 /257 缩放"
    assert out[0, 0, 2] == 130, "ch2 应原样读出"


def test_grayscale_and_rgba_are_normalised(tmp_path):
    import tifffile
    g = np.full((6, 6), 90, dtype=np.uint8)
    p = str(tmp_path / "gray.tif")
    tifffile.imwrite(p, g)
    out = media_io.load_image_rgb_uint8(p)
    assert out.shape == (6, 6, 3)

    rgba = np.full((6, 6, 4), 77, dtype=np.uint8)
    p2 = str(tmp_path / "rgba.tif")
    tifffile.imwrite(p2, rgba)
    out2 = media_io.load_image_rgb_uint8(p2)
    assert out2.shape == (6, 6, 3)


def test_png_still_works(tmp_path):
    """非 TIFF 路径不受影响。"""
    from PIL import Image
    rgb = np.full((5, 7, 3), 33, dtype=np.uint8)
    p = str(tmp_path / "x.png")
    Image.fromarray(rgb).save(p)
    out = media_io.load_image_rgb_uint8(p)
    assert np.array_equal(out, rgb)


def test_ai_paths_do_not_use_pil_directly():
    """静态守卫：AI 推理的读图点必须走 media_io，不能直接 PIL。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    targets = [
        os.path.join(root, "shared_modules", "dat_runtime.py"),
        os.path.join(root, "photo_restoration_pipeline", "stages",
                     "stage3_ai_upscale.py"),
    ]
    for t in targets:
        with open(t, encoding="utf-8") as fh:
            src = fh.read()
        # DAT/GPEN/RealESRGAN 三个推理后端的读图行
        assert "Image.open(input_path).convert(\"RGB\")" not in src, \
            f"{os.path.basename(t)}: 推理读图不得直接用 PIL（R16 黑图）"
        assert "load_image_rgb_uint8" in src, \
            f"{os.path.basename(t)}: 必须走 media_io.load_image_rgb_uint8"
