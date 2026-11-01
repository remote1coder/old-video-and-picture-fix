# -*- coding: utf-8 -*-
"""`_classify_timeline` 的容差必须**大于容器量化步长**。

实测（150 帧 CFR 30fps，从真实素材重编）：

| | 逐帧间隔 |
|---|---|
| 源 | 0.03333 × 149（全同） |
| stage1 产物 | **0.033 × 99 + 0.034 × 50** |

stage1 输出的时间基是整数毫秒，33.333ms 被舍成 33/34 交替。
原判定用 1% 相对容差 → 1ms 抖动 = 3% > 1% → **CFR 被判成 VFR**。

后果不只是统计难看：`source_timeline=vfr` 会让 stage3 的**零落盘流式
超分路径被永久跳过**（它要求"确认 CFR"），而 30 / 29.97 / 23.976
恰恰是最常见的帧率 —— 那条"默认开启"的路实际走不到。

修法：容差取 `max(2ms, 中位间隔 × 1%)`。真 VFR 的步长远不止 1ms
（实测 0.0449 vs 0.0898 差 45ms），依然能判出来。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _cls(intervals):
    """按给定的逐帧间隔构造 pts_map 并分类。"""
    s3 = _import()
    m = {0: 0.0}
    t = 0.0
    for i, d in enumerate(intervals, start=1):
        t += d
        m[i] = t
    return s3._classify_timeline(m, len(m))


def _import():
    import importlib
    return importlib.import_module("stage3_ai_upscale")


@pytest.mark.parametrize("label,intervals", [
    ("30fps 精确", [1 / 30] * 30),
    ("30fps 量化成 33/34", [0.033, 0.034] * 15),
    ("29.97 量化", [0.033, 0.034] * 15),
    ("25fps 精确 0.04", [0.04] * 30),
    ("23.976 量化 41/42", [0.041, 0.042] * 15),
    ("50fps 精确 0.02", [0.02] * 30),
    ("23.976 量化 42/41 反向", [0.042, 0.041] * 15),
])
def test_quantised_cfr_is_still_cfr(label, intervals):
    """毫秒量化造成 1ms 抖动，**不得**判成 VFR。"""
    assert _cls(intervals) == "cfr", label


@pytest.mark.parametrize("label,intervals", [
    ("真 VFR 0.0449/0.0898", [0.0449] * 20 + [0.0898] * 10),
    ("2 倍步长 0.033/0.066", [0.033] * 20 + [0.066] * 10),
    ("掉帧 0.033/0.1", [0.033] * 15 + [0.1] * 3 + [0.033] * 12),
])
def test_real_vfr_is_still_vfr(label, intervals):
    """真 VFR 的步长远大于 1ms —— 放宽容差不能把它们也归成 CFR。"""
    assert _cls(intervals) == "vfr", label


def test_zero_and_non_monotonic_are_unknown():
    s3 = _import()
    # 重复时间戳（d=0）
    assert s3._classify_timeline({0: 0.0, 1: 0.0, 2: 0.04}, 3) == "unknown"
    # 非单调
    assert s3._classify_timeline({0: 0.0, 1: 0.04, 2: 0.02}, 3) == "unknown"
    # 帧号不完整
    assert s3._classify_timeline({0: 0.0, 2: 0.04, 4: 0.08}, 5) == "unknown"


def test_tolerance_floor_is_documented_as_2ms():
    """源码守卫：容差必须显式含时间基下限（别改回纯相对百分比）。"""
    p = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                     "stage3_ai_upscale.py")
    src = open(p, encoding="utf-8", errors="replace").read()
    assert "_tol = max(0.002, 0.01 * med)" in src, \
        "容差必须有 2ms 绝对下限，否则毫秒量化会把 CFR 判成 VFR"
    assert "max(d) - min(d)) / med > 0.01" not in src, \
        "旧的纯相对判据不该保留"
