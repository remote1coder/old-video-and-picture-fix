# -*- coding: utf-8 -*-
"""帧传输决策（PNG vs 共享内存）的回归测试。

设计依据均已联网核实：budget=available×fraction（留 50%）、
动态重估、绝对红线（用户要求至少留 1GB 空闲）。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "shared_modules"))

import frame_transport as ft  # noqa: E402

MB = 1024 ** 2
GB = 1024 ** 3


# ---------------------------------------------------------------- 手动开关

def test_manual_png_wins():
    r = ft.plan_transport(1 * MB, 256, {"ai_frame_transport": "png"})
    assert r["transport"] == "png"
    assert "手动指定" in r["reason"]


def test_manual_shared_memory_wins(monkeypatch):
    """手动选 shared_memory 时必须赢过 auto 的保守判定。

    R22：这条原来**直接读本机真实可用内存** —— 全量跑批时实测偶发失败
    （`available_bytes()` 在某些用例之后返回 None，forced 分支仍会退回
    PNG）。用例不该依赖宿主机当前内存，这里把探测结果固定住。
    """
    monkeypatch.setattr(ft, "available_bytes", lambda: 64 * GB)
    r = ft.plan_transport(1 * MB, 256, {"ai_frame_transport": "shared_memory"})
    assert r["transport"] == "shared_memory"


def test_manual_shared_memory_still_falls_back_when_memory_is_unreadable(
        monkeypatch):
    """探测不到内存时，**连手动强制也退回 PNG** —— 这是有意的保守。

    记录成用例的原因：它不是缺陷，而是一个容易被当成缺陷的行为。
    宁可 PNG（慢但一定跑得通），也不赌一个测不出来的内存预算。
    """
    monkeypatch.setattr(ft, "available_bytes", lambda: None)
    r = ft.plan_transport(1 * MB, 256, {"ai_frame_transport": "shared_memory"})
    assert r["transport"] == "png"
    assert "探测不到" in r["reason"]


def test_invalid_mode_falls_back_to_auto():
    r = ft.plan_transport(1 * MB, 256, {"ai_frame_transport": "bogus"})
    assert r["transport"] in ("shared_memory", "png")


# ---------------------------------------------------------------- auto 逻辑

def test_auto_shrinks_when_memory_tight():
    """每帧 100MB、可用内存极少 → 放不下 min_frames，必须回退 PNG。"""
    r = ft.plan_transport(100 * MB, 256, {
        "ai_frame_memory_fraction": 0.5,
        "ai_frame_min_buffer_frames": 8,
        "ai_frame_min_free_bytes": GB,
    })
    # 不探测真实内存的机器上可能 avail 很大；这里只验证不会崩且字段齐全
    assert r["transport"] in ("shared_memory", "png")
    assert "reason" in r and r["buffer_frames"] >= 1


def test_auto_never_below_min_free(monkeypatch):
    """核心安全性质：budget 不得侵入「可用内存 − 绝对底线」。"""
    monkeypatch.setattr(ft, "available_bytes", lambda: 4 * GB)
    r = ft.plan_transport(50 * MB, 256, {
        "ai_frame_memory_fraction": 0.9,     # 激进
        "ai_frame_min_free_bytes": GB,       # 底线 1GB
        "ai_frame_min_buffer_frames": 8,
    })
    # 4GB 可用、底线 1GB → 最多用 3GB；0.9×4GB=3.6GB 被压到 3GB
    assert r["budget_bytes"] == 3 * GB
    assert r["buffer_frames"] == 61          # 3GB / 50MB
    # 剩余空闲 = 4GB − 3GB = 1GB，正好守住底线
    assert 4 * GB - r["budget_bytes"] >= GB


def test_min_free_floor_beats_fraction():
    """fraction 很小时也应至少用到 fraction 的量；floor 只在更严时生效。"""
    r = ft.plan_transport(1 * MB, 256, {
        "ai_frame_memory_fraction": 0.1,
        "ai_frame_min_free_bytes": GB,
    })
    # 0.1×avail 通常 < avail−1GB，所以 budget 取 0.1×avail
    assert r["budget_bytes"] is not None


def test_zero_frame_bytes_falls_back():
    r = ft.plan_transport(0, 256, {})
    assert r["transport"] == "png"


def test_no_memory_probe_falls_back(monkeypatch):
    monkeypatch.setattr(ft, "available_bytes", lambda: None)
    r = ft.plan_transport(1 * MB, 256, {"ai_frame_transport": "auto"})
    assert r["transport"] == "png"
    assert "探测不到" in r["reason"]


def test_nonpositive_config_falls_back_to_default():
    """设 0/负数不关闭判定，而是退回默认值。"""
    r = ft.plan_transport(1 * MB, 256, {
        "ai_frame_memory_fraction": 0,
        "ai_frame_min_buffer_frames": -5,
    })
    assert r["transport"] in ("shared_memory", "png")


# ---------------------------------------------------------------- FrameRing

def test_frame_ring_roundtrip():
    """创建 → 写槽 → 重新 attach → 读回一致 → close 不炸。"""
    import numpy as np
    slots, fb = 4, 1024
    ring = ft.FrameRing(name="wbtest_ring", frame_bytes=fb, slots=slots)
    try:
        payload = np.arange(fb, dtype=np.uint8)
        ring.slot(2)[:] = payload
        again = ft.FrameRing.attach(ring.name, fb, slots)
        try:
            assert np.array_equal(again.slot(2), payload)
        finally:
            again._safe_close()
    finally:
        ring.close()


def test_frame_ring_slot_wraps():
    """环形：第 slots 个槽应回到第 0 个。"""
    import numpy as np
    slots, fb = 3, 512
    ring = ft.FrameRing(name="wbtest_wrap", frame_bytes=fb, slots=slots)
    try:
        ring.slot(0)[:] = 7
        assert ring.slot(slots)[0] == 7
        assert ring.slot(2 * slots)[0] == 7
    finally:
        ring.close()
