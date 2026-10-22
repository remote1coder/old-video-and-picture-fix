# -*- coding: utf-8 -*-
"""stage 超时语义的回归测试。

两条契约（都是 2026-10-22 实测踩出来的，别再改回去）：

1. **硬超时是"最终保险"，不是判死线。**
   判死由看门狗负责（`stage_inactivity_timeout` 内 temp 目录没长）。
   旧实现 `min(按 duration×分辨率估算, 48h)` 让估算值继续判死，于是：
     - `int(20/60) == 0` → 20s/1080p 开 AI 只拿到 **600s**
       （分辨率缩放与 AI 开销全被乘零）；
     - 962s/1080p 只给 3.8h，而实测 stage7 单独就要 3h（27% 余量）。
   实测后果：1080p 隔行素材的 stage1 正常推进、正常写盘，第 600 秒被
   硬超时杀掉 —— 正是设计要避免的"慢的被杀"。

2. **cancel_guard 必须能在模块级拿到 subprocess。**
   旧代码在 `hard_timeout` 分支写 `raise subprocess.TimeoutExpired(...)`，
   但模块从未 `import subprocess`（只有函数内的 `import subprocess as _sp`），
   于是超时被伪装成 `NameError: name 'subprocess' is not defined`，
   真正的"超时"原因完全看不出来。
"""
import importlib.util
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "shared_modules"))
sys.path.insert(0, os.path.join(_ROOT, "video_restoration_pipeline"))


def _load_video_main():
    spec = importlib.util.spec_from_file_location(
        "video_main_under_test",
        os.path.join(_ROOT, "video_restoration_pipeline", "main.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def vmain():
    return _load_video_main()


# ---------------------------------------------------------------- 硬超时


def test_hard_timeout_is_final_safety_not_estimate(vmain):
    """无论时长/分辨率，硬超时都应等于"最终保险"值。"""
    hard = int(vmain.DEFAULT_HARD_TIMEOUT_HOURS * 3600)
    cases = [
        (0.5, True, 1920, 1080),    # 短视频 + AI（旧代码只给 600s）
        (0.5, False, 1920, 1080),
        (20, True, 1920, 1080),     # 短视频（int(20/60)==0 的经典坑）
        (60, True, 1280, 720),
        (962, False, 1920, 1080),   # 长片：旧代码只给 3.8h
        (3600, True, 3840, 2160),   # 超长 + 4K
    ]
    for dur, ai, w, h in cases:
        got = vmain._calculate_timeout(dur, ai, w, h)
        assert got == hard, (
            f"duration={dur}s {w}x{h} ai={ai}: 硬超时 {got}s 不等于保险值 "
            f"{hard}s —— 估算值又在判死了")


def test_hard_timeout_is_configurable(vmain):
    """`stage_hard_timeout_hours` 必须能改变保险值。"""
    got = vmain._calculate_timeout(20, True, 1920, 1080, hard_hours=6)
    assert got == 6 * 3600


def test_hard_timeout_never_follows_estimate_even_when_estimate_is_huge(vmain, capsys):
    """估算超过保险值时，硬超时**仍然是保险值**，只打告警。

    反例：1 小时 4K + AI 的估算约 84h（scale 被上限 12 截断后
    120*12+300*12=5040 s/分钟 × 60 分钟）。若用 max(estimate, hard)，
    估算就重新变成判死线，等于把"猜出来的数判死"这个毛病装回去。
    """
    got = vmain._calculate_timeout(3600, True, 3840, 2160, hard_hours=48)
    assert got == 48 * 3600, f"硬超时被估算带跑了: {got}s"
    err = capsys.readouterr().err
    assert "stage_hard_timeout_hours" in err, \
        "估算超过保险值时应在 stderr 提示调高 stage_hard_timeout_hours"


def test_estimate_below_hard_emits_no_warning(vmain, capsys):
    """估算小于保险值时不该有噪音告警。"""
    vmain._calculate_timeout(20, True, 1920, 1080, hard_hours=48)
    assert capsys.readouterr().err == ""


def test_short_video_estimate_not_zeroed(vmain):
    """短视频的估算不能被 `int(dur/60)` 截断成 base。

    估算是参考值，但它失真会让人误判"短视频不需要时间"。
    这里通过 hard_hours=0 让返回值退化为估算，从而观察到估算本身。
    """
    # hard_hours 极小 → return max(estimate, hard) == estimate
    est_20s = vmain._calculate_timeout(20, True, 1920, 1080, hard_hours=0.0001)
    est_60s = vmain._calculate_timeout(60, True, 1920, 1080, hard_hours=0.0001)
    # 20s 按"不满一分钟计一分钟"，应与 60s 同量级而不是只剩 base 600
    assert est_20s == est_60s, (
        f"20s 估算 {est_20s} 与 60s 估算 {est_60s} 不一致 —— "
        f"短视频被截断成 0 了")


# ---------------------------------------------------------------- cancel_guard


def test_cancel_guard_can_raise_timeout_expired():
    """cancel_guard 必须在模块级能构造 subprocess.TimeoutExpired。

    回归：旧代码 `raise subprocess.TimeoutExpired(...)` 抛的是 NameError，
    把"超时"伪装成"名字未定义"。
    """
    import cancel_guard

    assert hasattr(cancel_guard, "subprocess"), \
        "cancel_guard 模块级缺少 subprocess（超时会被伪装成 NameError）"
    # 真的能构造出来（不是 None / 未绑定）
    exc = cancel_guard.subprocess.TimeoutExpired(["x"], 1)
    assert isinstance(exc, Exception)


def test_cancel_guard_subprocess_is_the_real_module():
    """确认拿到的是真的 subprocess 模块，而不是别处传进来的同名对象。"""
    import cancel_guard
    import subprocess as real

    assert cancel_guard.subprocess is real
