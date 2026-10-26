# -*- coding: utf-8 -*-
"""STR-06：共享内存预算必须按「输入 ring + 输出 ring」合计。

调用方实际创建两份 ring：
  - 输入 ring：frame_bytes / 帧
  - 输出 ring：frame_bytes × out_scale² / 帧（面积是输入的 scale² 倍）

所以每帧真实占用 = frame_bytes × (1 + out_scale²)：
  2× → 5 倍，4× → 17 倍。

旧实现 `fits = budget // frame_bytes` 只算输入 ring，于是"输入 ring 落在
预算内"的表象下，两份 ring 合计可能远超预留内存底线。回退路径（流式关闭
或失败）就会申请远超预期的内存。

注意：预算 = min(avail × fraction, avail − min_free)，且 `min_free` 传 0
会被当成假值退回默认（1 GiB），所以测试里显式把它算进 avail。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import frame_transport as ft  # noqa: E402

FB = 1 * 2 ** 20            # 每帧 1 MiB
FLOOR = 1 * 2 ** 30         # 默认 ai_frame_min_free_bytes = 1 GiB


def _plan(headroom_bytes, out_scale, fb=FB, req=10 ** 6):
    """用受控可用内存跑一次计划。

    Args:
        headroom_bytes: 期望拿到的**净预算**。avail 会加上 FLOOR 与
            fraction=1.0 的影响，使 budget 恰为 headroom_bytes。
    """
    avail = headroom_bytes + FLOOR
    orig = ft.available_bytes
    ft.available_bytes = lambda: avail
    try:
        return ft.plan_transport(
            fb, req,
            {"ai_frame_transport": "shared_memory",
             "ai_frame_memory_fraction": 1.0,
             "ai_frame_min_buffer_frames": 0},
            out_scale=out_scale)
    finally:
        ft.available_bytes = orig


def _frames(headroom, out_scale, fb=FB):
    p = _plan(headroom, out_scale, fb)
    assert p["transport"] == "shared_memory", p
    return p["buffer_frames"]


def test_out_scale_1_matches_old_behaviour():
    """out_scale=1 退化为只有输入 ring，与旧行为一致。"""
    assert _frames(50 * FB, 1) == 50


def test_2x_costs_five_times_per_frame():
    """2×：每帧 1+4=5 倍 —— 同样净预算能放的帧数是 1× 的 1/5。"""
    head = 50 * FB
    assert _frames(head, 1) == 50
    assert _frames(head, 2) == 10


def test_4x_costs_seventeen_times_per_frame():
    """4×：每帧 1+16=17 倍。"""
    head = 170 * FB
    assert _frames(head, 1) == 170
    assert _frames(head, 4) == 10


def test_budget_never_exceeds_available():
    """任意 scale 下，两份 ring 合计都不能超出净预算。

    余量要够大：帧数低于 ai_frame_min_buffer_frames（默认 8）时程序会
    **正确地**退回 PNG，那是另一条规则，不是预算算错。
    """
    head = 400 * FB          # s=4 时 per=17MiB → 23 帧，仍高于下限 8
    for s in (1, 2, 3, 4):
        per = FB * (1 + s * s) if s > 1 else FB
        assert _frames(head, s) * per <= head, s


def test_too_few_frames_still_falls_back_to_png():
    """帧数低于下限时仍退回 PNG（预算修好后这条规则不能丢）。"""
    p = _plan(8 * FB, 4)     # per=17MiB → 0 帧
    assert p["transport"] == "png", p


def test_invalid_out_scale_falls_back_to_input_only():
    """非法 scale 不能让预算变成 0，退化为只算输入 ring。"""
    for bad in (0, -3, None, "x"):
        assert _frames(50 * FB, bad) == 50, bad
