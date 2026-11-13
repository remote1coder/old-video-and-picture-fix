# -*- coding: utf-8 -*-
"""P4：照片人脸混合**不得**把 ROI 外的 16bit 像素压成 8bit。

## 缺陷（到 R25 为止）

`stage4_ai_face.feather_blend()` 是这样写的：

```python
orig = Image.open(original_path).convert("RGB")     # ← 8bit per channel
...
blended_16 = (blended.astype(np.uint16) * 257)      # ← 再抬回 16bit
```

整张图先被压成 8bit 再抬回 16bit。ROI 外哪怕一个像素都没被模型碰过，
也会经历一次 `x → round(x/257) → ×257` 的量化：**16bit 链路在这一步断了**。
更糟的是另外两条分支：

* `mask_ratio < 0.01 → copyfile(restored)` —— 6000×4000 的照片里人脸占
  0.4% 是常态，这个分支直接把 8bit 修复图当成交付；
* 异常兜底同样 copy 修复图。

## 判据

1. ROI 外（离人脸 32px 以外）的像素必须与输入**逐位相同**；
2. 输出必须是 uint16 且**不全**是 257 的倍数（那只可能是 8bit 回环）；
3. 完全没有改动时，输出与输入**字节级相同**；
4. 8bit 路径不受影响（行为与改动前一致）。
"""
import json
import os

import numpy as np
import pytest

import conftest  # noqa: F401
import stage4_ai_face as s4


def _write_tiff16(path, arr):
    import tifffile
    tifffile.imwrite(str(path), arr, compression="zlib")
    return str(path)


def _read_tiff16(path):
    import tifffile
    return tifffile.imread(str(path))


def _scene(h=96, w=128):
    """16bit 原图：背景是**非 257 倍数**的值（一压 8bit 必然变）。"""
    rng = np.random.default_rng(11)
    arr = np.zeros((h, w, 3), dtype=np.uint16)
    # 背景：0x1200 附近的随机值，故意不是 257 的整数倍
    arr[:, :, :] = (0x1200 + rng.integers(0, 200, (h, w, 3))).astype(np.uint16)
    # "人脸"区：左上 24x24，给一个明显不同的亮块
    arr[8:32, 8:32, :] = 0xC000
    return arr


def _restored8(arr16, h=96, w=128):
    """伪造模型输出：只改人脸区，其余与原图相同（8bit 表示）。"""
    base = (arr16.astype(np.float32) / 257.0).round().clip(0, 255).astype(
        np.uint8)
    base[8:32, 8:32, :] = 40          # 模型把脸改暗了
    return base


def test_roi_outside_is_bit_exact(tmp_path):
    """核心判据：离人脸远的地方，16bit 值必须**逐位**不变。"""
    orig = _scene()
    op = _write_tiff16(tmp_path / "in.tif", orig)
    rp = str(tmp_path / "restored.png")
    from PIL import Image
    Image.fromarray(_restored8(orig)).save(rp)

    out = str(tmp_path / "blended.tif")
    assert s4.feather_blend(op, rp, out, feather_width=8, bit_depth=16)

    got = _read_tiff16(out)
    assert got.dtype == np.uint16
    assert got.shape == orig.shape

    # 右下角远离人脸（羽化半径 8px，取 32px 安全边距）
    far = got[64:, 64:, :]
    ref = orig[64:, 64:, :]
    assert np.array_equal(far, ref), (
        "ROI 外的 16bit 像素被改动了：最大差 "
        f"{int(np.abs(far.astype(int) - ref.astype(int)).max())}")

    # 变化只允许出现在"人脸区 + 羽化半径"之内。
    # 高斯模糊的支撑近似 3σ（σ=feather_width），人脸到 32，故上界 32+3×8=56。
    diff = np.abs(got.astype(np.int32) - orig.astype(np.int32)).max(axis=2)
    changed = diff > 0
    ys, xs = np.nonzero(changed)
    _limit = 32 + 3 * 8
    assert ys.max() <= _limit and xs.max() <= _limit, (
        f"变化扩散到 ({ys.max()},{xs.max()})，超出 人脸区+3σ={_limit}")


def test_output_is_not_an_8bit_roundtrip(tmp_path):
    """所有像素都落在 257 的倍数上 = 只可能是 8bit 回环。"""
    orig = _scene()
    op = _write_tiff16(tmp_path / "in.tif", orig)
    from PIL import Image
    rp = str(tmp_path / "restored.png")
    Image.fromarray(_restored8(orig)).save(rp)
    out = str(tmp_path / "blended.tif")
    assert s4.feather_blend(op, rp, out, bit_depth=16)

    got = _read_tiff16(out)
    multiples = (got.astype(np.int64) % 257) == 0
    assert not multiples.all(), "整幅都是 257 的倍数：16bit 精度已被压成 8bit"


def test_no_change_returns_the_original_bytes(tmp_path):
    """模型什么都没改时，交付的必须是**原图**，不是 8bit 的修复图。"""
    orig = _scene()
    op = _write_tiff16(tmp_path / "in.tif", orig)
    from PIL import Image
    same = (orig.astype(np.int32) // 257).astype(np.uint8)
    rp = str(tmp_path / "restored.png")
    Image.fromarray(same).save(rp)
    out = str(tmp_path / "blended.tif")
    assert s4.feather_blend(op, rp, out, bit_depth=16)
    with open(op, "rb") as a, open(out, "rb") as b:
        assert a.read() == b.read(), "无改动时输出与原图字节不同"


def test_tiny_face_still_preserves_background(tmp_path):
    """人脸 <1% 也不能触发"整幅 copy 修复图"那条老捷径。

    实测配置：6000×4000 里的人脸约 0.4%。这里用 200×200 图 + 12×12 人脸
    模拟同一种比例（0.36%）。
    """
    h = w = 200
    orig = np.full((h, w, 3), 0x2345, dtype=np.uint16)
    orig[90:102, 90:102, :] = 0xD000
    op = _write_tiff16(tmp_path / "in.tif", orig)
    from PIL import Image
    r8 = (orig.astype(np.float32) / 257.0).round().astype(np.uint8)
    r8[90:102, 90:102, :] = 30
    rp = str(tmp_path / "restored.png")
    Image.fromarray(r8).save(rp)

    out = str(tmp_path / "blended.tif")
    assert s4.feather_blend(op, rp, out, bit_depth=16)
    got = _read_tiff16(out)
    assert np.array_equal(got[:60, :60, :], orig[:60, :60, :]), \
        "小人脸场景下 ROI 外又被整体降位深了"


def test_8bit_path_unchanged(tmp_path):
    """8bit 输入输出行为与改动前一致（不能顺手改掉别人的路线）。"""
    from PIL import Image
    orig8 = np.zeros((64, 64, 3), dtype=np.uint8)
    orig8[:, :] = 120
    orig8[16:32, 16:32, :] = 200
    op = str(tmp_path / "in.png")
    Image.fromarray(orig8).save(op)
    r8 = orig8.copy()
    r8[16:32, 16:32, :] = 60
    rp = str(tmp_path / "restored.png")
    Image.fromarray(r8).save(rp)

    out = str(tmp_path / "blended.tif")
    assert s4.feather_blend(op, rp, out, bit_depth=8)
    got = np.array(Image.open(out).convert("RGB"))
    assert got.dtype == np.uint8
    assert got[40:, 40:, :].max() == got[40:, 40:, :].min() == 120


def test_16bit_requested_on_8bit_source_still_writes_uint16(tmp_path):
    """源是 8bit 但调用方要 16bit：容器仍须是 uint16（下游位深核对依赖它）。"""
    from PIL import Image
    a = np.full((32, 32, 3), 77, dtype=np.uint8)
    a[8:16, 8:16, :] = 200
    op = str(tmp_path / "in.png")
    Image.fromarray(a).save(op)
    r = a.copy()
    r[8:16, 8:16, :] = 10
    rp = str(tmp_path / "restored.png")
    Image.fromarray(r).save(rp)
    out = str(tmp_path / "blended.tif")
    assert s4.feather_blend(op, rp, out, bit_depth=16)
    got = _read_tiff16(out)
    assert got.dtype == np.uint16
    # 77 × 257 = 19789（8bit→16bit 容器的既有约定）
    assert int(got[24, 24, 0]) == 77 * 257


def test_save_uint16_uses_zlib_not_lzw(tmp_path):
    """写出必须是 tifffile+zlib：PIL 对 .tif 默认 LZW，读侧解不开。"""
    arr = np.full((8, 8, 3), 1234, dtype=np.float32)
    out = str(tmp_path / "x.tif")
    assert s4._save_uint16(arr, out)
    got = _read_tiff16(out)
    assert got.dtype == np.uint16 and int(got[0, 0, 0]) == 1234
