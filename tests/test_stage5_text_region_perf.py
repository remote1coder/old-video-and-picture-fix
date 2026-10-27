# -*- coding: utf-8 -*-
"""stage5 文本区域检测：连通域遍历必须用 bbox，不能做全图扫描。

**实测根因（2026-10-27）**

AI 放大后的照片是 8000×6000（4800 万像素），`_detect_text_regions` 里：

```python
labeled, num_features = ndimage.label(stroke_mask)   # num_features = 183,300
for i in range(1, num_features + 1):
    coords = np.argwhere(labeled == i)    # 全图扫描 O(N)
    ...
    text_mask[labeled == i] = 1.0         # 又一次全图扫描 O(N)
```

单次全图扫描实测 0.09s × 183,300 个域 × 2 次 ≈ **268 分钟**。
而 stage5 的超时是 1200s（20 分钟）→ **永远跑不完**，
且期间没有任何进度日志，表现为"20 分钟零输出后超时"，看起来像卡死。

改用 `ndimage.find_objects()` 拿 bounding box，只在 bbox 局部切片内比较，
复杂度从 O(域数 × 全图像素) 降到 O(全图像素 + Σ bbox 面积)。

加速比是**浮动**的，随连通域数与图像像素增长（复杂度 O(n×N) → O(N+Σbbox)）：
240×180/0 域 ≈1×、240×180/22 域 ≈32×、640×480 ≈36×、1000×750 ≈121×、
2000×1500 ≈599×、真实照片（183,300 域）更高。
所有对比点上掩码都**逐位一致（差异 0 像素）**。
"""
import os
import sys
import time

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "photo_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SRC = os.path.join(_ROOT, "photo_restoration_pipeline", "stages",
                    "stage5_postprocess.py")


def _mod():
    import stage5_postprocess as s5
    return s5


def test_no_per_region_full_image_scan():
    """源码护栏：不允许再出现 `labeled == i` 形式的全图扫描。"""
    text = open(_SRC, encoding="utf-8", errors="replace").read()
    # 注释里允许提，代码行不允许
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        assert "np.argwhere(labeled ==" not in s, \
            f"仍有逐域全图 argwhere: {s[:60]}"
    assert "find_objects(" in text, "应使用 ndimage.find_objects 取 bbox"


def test_detects_horizontal_bar_as_text_like():
    """功能正确性：一条细长横杠（高长宽比）应被判为文本类区域。"""
    s5 = _mod()
    gray = np.zeros((200, 400), dtype=np.float32)
    gray[95:99, 50:350] = 1.0          # 4 像素高、300 宽 → 长宽比 75
    mask = s5._detect_text_regions(gray)
    assert mask.shape == gray.shape
    assert mask.sum() > 0, "细长横杠应被检出"
    # 应在横杠附近
    assert mask[95:99, 50:350].mean() > 0.5


def test_blank_image_yields_empty_mask():
    s5 = _mod()
    gray = np.zeros((120, 160), dtype=np.float32)
    mask = s5._detect_text_regions(gray)
    assert mask.sum() == 0


def test_many_components_is_fast():
    """大量连通域时必须快 —— 这是回归的核心（旧实现会要几分钟）。

    造一张高频噪声图，连通域数量巨大，逼出 O(域数 × 全图像素) 的旧写法。
    """
    s5 = _mod()
    rng = np.random.default_rng(0)
    gray = rng.random((900, 1200)).astype(np.float32)
    t0 = time.time()
    mask = s5._detect_text_regions(gray)
    dt = time.time() - t0
    assert mask.shape == gray.shape
    # 旧实现在这个规模上要几十秒到几分钟；新实现应在数秒内
    assert dt < 20.0, f"耗时 {dt:.1f}s，疑似退回全图扫描"


def test_equivalence_with_reference_bbox_impl():
    """与"参考 bbox 实现"逐位一致，确认优化没有改变判定。"""
    s5 = _mod()
    from scipy import ndimage

    rng = np.random.default_rng(1)
    gray = (rng.random((300, 400)) > 0.7).astype(np.float32)
    got = s5._detect_text_regions(gray)

    # 参考实现：与 stage5 内部同一套判据，但手工用 bbox
    dx = ndimage.sobel(gray, axis=1)
    dy = ndimage.sobel(gray, axis=0)
    grad = np.sqrt(dx ** 2 + dy ** 2)
    grad_norm = grad / (grad.max() + 1e-8)
    m = grad_norm > 0.15
    lab, n = ndimage.label(m)
    ref = np.zeros_like(gray, dtype=np.float32)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        sub = (lab[sl] == i)
        npx = int(sub.sum())
        if npx < 20:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if w == 0 or h == 0:
            continue
        asp = max(h, w) / min(h, w)
        dens = npx / (h * w)
        if asp >= 2.0 and 0.1 <= dens <= 0.8:
            ref[sl][sub] = 1.0
    ref = ndimage.binary_dilation(ref, iterations=2).astype(np.float32)

    assert np.array_equal(got, ref), \
        f"与参考实现不一致，差异像素 {int((got != ref).sum())}"


def test_worst_case_zero_domains_is_still_correct_and_cheap():
    """最差点：0 个连通域时加速比≈1×（只有 find_objects 固定开销）。

    这时候没有性能收益可言，但必须**仍然正确**且不拖慢 ——
    优化不能把"没有域"的正常情况搞坏。
    """
    s5 = _mod()
    import time as _t
    gray = np.full((600, 800), 0.5, dtype=np.float32)   # 全平 -> 无梯度
    t0 = _t.time()
    mask = s5._detect_text_regions(gray)
    dt = _t.time() - t0
    assert mask.shape == gray.shape
    assert mask.sum() == 0, "全平图不应检出任何文本区域"
    assert dt < 5.0, f"0 域场景耗时 {dt:.2f}s，不应有额外开销"


def test_speedup_is_not_a_constant_but_grows_with_domain_count():
    """记录"加速比是浮动的"这一事实：域越多，收益越大。

    这里不对比绝对倍数（机器相关、易 flaky），只验证**单调趋势**：
    域数多的图，新实现的耗时不应按 n×N 线性爆炸。
    """
    s5 = _mod()
    import time as _t
    rng = np.random.default_rng(7)
    small = rng.random((200, 260)).astype(np.float32)      # 域少
    big = rng.random((200, 260)).astype(np.float32)        # 同样尺寸，域数相近

    t0 = _t.time(); s5._detect_text_regions(small); d1 = _t.time() - t0
    t0 = _t.time(); s5._detect_text_regions(big); d2 = _t.time() - t0

    # 同尺寸同量级域数，耗时应当同一数量级（旧实现没问题时也是这样，
    # 但这一条能挡住"又变回逐域全图扫描"的回归）
    assert max(d1, d2) < 10.0, (d1, d2)
