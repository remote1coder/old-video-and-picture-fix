# -*- coding: utf-8 -*-
"""R13-02：时间轴必须"一次采集、全路径共享"，且缺数据不得假装成 CFR。

**审核指出的三处漏洞**

1. `all_pts_map` 在 2162 被使用、2185 才赋值 → fps 探测失败时先抛
   `UnboundLocalError`，根本走不到 PTS 反推。（这是 R12-10 引入的）
2. 流式与 PNG 各自二次 probe 且默认 25 → 首次得到 30000/1001、第二次失败
   时，同一批帧按 25fps 编码：帧数对、时长错。
3. 非数字 PTS 被填成 `i/25`、缺失帧被填成 `global_idx/25`；
   分类器也不检查帧号覆盖 → 全部 N/A 或"规则但缺帧"的子集会被判 CFR。

本文件覆盖测试矩阵中可在函数层验证的部分（3、4 项与有理数保真）；
需要跑 `main()` 的 1、2 项由端到端实测覆盖，不在此假装。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage3_ai_upscale as s3  # noqa: E402


# ------------------------------------------------ 第 3 项：坏 PTS 必须 unknown

def test_all_pts_missing_is_unknown():
    """PTS 全部读不出 → 空 map → unknown（不能当成规则 25fps）。"""
    assert s3._classify_timeline({}, 300) == "unknown"
    assert s3._classify_timeline(None, 300) == "unknown"


def test_partial_pts_is_unknown_not_cfr():
    """部分缺失 → 帧号不连续 → unknown。

    旧实现会把缺失帧填成 `i/25`，于是"一半帧读不出"也能拼出完美 CFR。
    """
    partial = {i: i * 0.04 for i in range(0, 300, 2)}   # 只采到偶数帧
    assert s3._classify_timeline(partial, 300) == "unknown"


def test_regular_subset_is_still_unknown():
    """规则但缺帧的子集（0/2/4…）→ unknown。

    这是审核点名的例子：间隔看起来完全均匀，但帧根本不完整。
    """
    subset = {i: i * 0.08 for i in range(0, 10, 2)}     # 0,2,4,6,8
    assert s3._classify_timeline(subset, 10) == "unknown"
    assert s3._classify_timeline(subset) == "unknown"   # 不传期望帧数也一样


def test_non_monotonic_and_nan_are_unknown():
    """非单调 / NaN / Inf / 重复时间戳 → unknown。"""
    assert s3._classify_timeline(
        {0: 0.0, 1: 0.04, 2: 0.02, 3: 0.06}, 4) == "unknown"
    assert s3._classify_timeline(
        {0: 0.0, 1: float("nan"), 2: 0.08}, 3) == "unknown"
    assert s3._classify_timeline(
        {0: 0.0, 1: float("inf"), 2: 0.08}, 3) == "unknown"
    assert s3._classify_timeline(
        {0: 0.0, 1: 0.04, 2: 0.04, 3: 0.08}, 4) == "unknown"


# ------------------------------------------------ 第 4 项：正常的时间轴

@pytest.mark.parametrize("fps", [25.0, 30.0, 30000 / 1001])
def test_common_cfr_rates_are_cfr(fps):
    """25 / 30 / 30000÷1001 都应判 CFR（覆盖完整）。"""
    n = 120
    pts = {i: i / fps for i in range(n)}
    assert s3._classify_timeline(pts, n) == "cfr"


def test_real_vfr_is_vfr():
    """真实变帧率（40/42ms 交替）→ vfr。"""
    pts, t = {}, 0.0
    for i in range(120):
        pts[i] = t
        t += 0.040 if i % 2 == 0 else 0.042
    assert s3._classify_timeline(pts, 120) == "vfr"


def test_coverage_mismatch_is_unknown():
    """帧数与时间轴覆盖不一致（多了或少了）→ unknown。"""
    pts = {i: i * 0.04 for i in range(100)}
    assert s3._classify_timeline(pts, 120) == "unknown"    # 少了 20 帧
    assert s3._classify_timeline(pts, 80) == "unknown"     # 多了 20 帧
    assert s3._classify_timeline(pts, 100) == "cfr"        # 正好 → 通过


# ------------------------------------------------ 有理数保真

def test_rational_fps_is_not_rounded():
    """30000/1001 不能变成 29.97 —— 长片会累积可见误差。"""
    got = s3._fps_rational_str(30000 / 1001)
    assert got == "30000/1001", got
    num, den = (int(x) for x in got.split("/"))
    assert abs(num / den - 30000 / 1001) < 1e-12


def test_rational_fps_for_common_rates():
    assert s3._fps_rational_str(25.0) == "25/1"
    assert s3._fps_rational_str(30.0) == "30/1"
    assert s3._fps_rational_str(24000 / 1001) == "24000/1001"


# ------------------------------------------------ 源码护栏：不再制造时间戳

def test_no_fabricated_timestamps_in_source():
    """不得再用 `i/25` 之类的固定帧率伪造 PTS。

    这条守调用形态（真实效果由上面的分类器用例与端到端实测覆盖）：
    一旦有人把 `pts_map[i] = i / 25.0` 写回来，缺帧又会被伪装成 CFR。
    """
    src = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                       "stage3_ai_upscale.py")
    text = open(src, encoding="utf-8", errors="replace").read()
    code_lines = [ln for ln in text.splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    code = "\n".join(code_lines)
    assert "i / 25.0" not in code, "又在用 i/25 伪造 PTS"
    assert "global_idx / 25.0" not in code, "又在用帧号/25 伪造 PTS"
