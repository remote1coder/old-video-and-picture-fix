# -*- coding: utf-8 -*-
"""R12-13 / R12-14：照片链的坐标与色彩输入契约。

**R12-13 浮点检测框**

stage4 原样保存 facexlib 的 **float** 坐标。stage5 用
`x1 = max(0, fx)` / `x2 = min(w, fx + fw)` 之后仍是 float，
`img[y1:y2, x1:x2]` 立刻抛 TypeError。

项目内**三次照片成功日志**都在同一行记录了这个错误（三次都检测到 20 张脸），
随后走人工/未匹配降级 —— 最终非黑图、rc=0，但**所请求的自动颜色匹配从未完成**。

**R12-14 手部肤色域**

`_hand_skin_roi_photo()` 的 docstring 写 "RGB float32 in [0,255]"，阈值用
0–255（r>95 / g>40 / b>20）；唯一调用方传的是归一化 **0–1** 的图。
0–1 的像素永远不可能 > 95 → 手部参考**恒为空**。
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "photo_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage5_postprocess as s5  # noqa: E402


# ------------------------------------------------------------ R12-13 框契约

def test_float_box_becomes_int_bounds():
    """浮点框必须返回 int 半开区间 —— 否则切片会 TypeError。"""
    got = s5._box_to_pixel_bounds((10.4, 20.6, 100.2, 80.9), 1000, 800)
    assert got is not None
    x1, y1, x2, y2 = got
    assert all(isinstance(v, int) for v in got), got
    assert (x1, y1) == (10, 20)                  # floor
    assert x2 == int(np.ceil(10.4 + 100.2))
    assert y2 == int(np.ceil(20.6 + 80.9))


def test_float_bounds_actually_slice_an_array():
    """回归：用返回的 bounds 真的去切片，不能抛 TypeError。"""
    img = np.zeros((100, 120, 3), dtype=np.float32)
    x1, y1, x2, y2 = s5._box_to_pixel_bounds(
        (11.7, 12.3, 40.9, 30.2), 120, 100)
    roi = img[y1:y2, x1:x2]          # 旧实现这里 TypeError
    assert roi.size > 0


def test_box_clamped_to_image():
    """越界框裁到图像内。"""
    x1, y1, x2, y2 = s5._box_to_pixel_bounds((-50, -50, 500, 500), 120, 100)
    assert (x1, y1, x2, y2) == (0, 0, 120, 100)


def test_invalid_boxes_return_none():
    """非有限值 / 零尺寸 / 完全在外 → None（由调用方降级，不静默乱切）。"""
    for box in ((float("nan"), 0, 10, 10),
                (0, float("inf"), 10, 10),
                (10, 10, 0, 10),
                (10, 10, 10, -5),
                ("a", "b", "c", "d"),
                (500, 500, 10, 10)):      # 完全在图外
        assert s5._box_to_pixel_bounds(box, 120, 100) is None, box


def test_face_box_from_stage4_json_is_accepted():
    """用 stage4 实际会写出的那种 JSON 值（float）跑一遍。"""
    import json
    raw = json.loads('[123.456, 78.9, 204.5, 198.25]')
    got = s5._box_to_pixel_bounds(tuple(raw), 1200, 900)
    assert got is not None
    assert all(isinstance(v, int) for v in got)


# ------------------------------------------------------------ R12-14 手部域

def _skin_image(h=200, w=200, value=(160, 100, 70)):
    """构造一块手部肤色区域（给定 0–255 值），返回 0–1 归一化图。"""
    arr = np.zeros((h, w, 3), dtype=np.float32)
    arr[:, :] = np.array(value, dtype=np.float32)
    return arr


def test_hand_roi_samples_normalised_image():
    """0–1 图像必须真的采到样本（旧实现恒为 0 个）。"""
    img01 = _skin_image() / 255.0
    got = s5._hand_skin_roi_photo(img01, (40.0, 20.0, 60.0, 60.0))
    assert got.shape[0] > 0, "归一化图像必须能采到肤色样本"
    assert got.max() <= 1.0, "返回值应仍在 0–1 域"


def test_hand_roi_rejects_wrong_domain():
    """把 0–255 的图传进来 → **必须被显式拒绝**，不是"碰巧采不到"。

    R13-09：本条原名与断言相反（名字说 reject，断言却是 `shape[0] > 0`）。
    那等于把"契约错误"记成"这张图没有手"。现在函数入口有 `_is_unit_range()`
    边界检查，传错域直接返回空并留痕，与"没有肤色像素"可区分。

    注意：这说明的是**输入契约被强制**，不是"R12-14 原缺陷仍在"。
    """
    img255 = _skin_image()          # 0–255 域
    assert s5._is_unit_range(img255) is False, "0–255 不应被判为单位区间"

    log = _Log()
    got = s5._hand_skin_roi_photo(img255, (40.0, 20.0, 60.0, 60.0),
                                  logger=log)
    assert got.shape[0] == 0, "错误数值域必须被拒绝"
    assert log.has("STAGE5_HAND_ROI_DOMAIN_INVALID"), \
        "拒绝时要留痕，不能静默返回空"


def test_is_unit_range_accepts_valid_domains():
    """[0,1] 的合法输入（含端点）必须通过；越界/NaN 必须拒绝。"""
    import numpy as np
    assert s5._is_unit_range(np.zeros((4, 4, 3), dtype=np.float32)) is True
    assert s5._is_unit_range(np.ones((4, 4, 3), dtype=np.float32)) is True
    assert s5._is_unit_range(
        np.full((4, 4, 3), 0.5, dtype=np.float32)) is True
    assert s5._is_unit_range(
        np.full((4, 4, 3), 1.5, dtype=np.float32)) is False
    assert s5._is_unit_range(
        np.full((4, 4, 3), -0.1, dtype=np.float32)) is False
    bad = np.zeros((4, 4, 3), dtype=np.float32)
    bad[0, 0, 0] = float("nan")
    assert s5._is_unit_range(bad) is False


class _Log:
    def __init__(self):
        self.events = []

    def _r(self, ev):
        self.events.append(ev)

    def info(self, ev, msg="", **k):
        self._r(ev)

    def warning(self, ev, msg="", **k):
        self._r(ev)

    def error(self, ev, msg="", **k):
        self._r(ev)

    def has(self, ev):
        return ev in self.events


def test_hand_roi_empty_when_no_skin():
    """纯灰（无肤色对比）→ 空样本，调用方可降级。"""
    grey = np.full((200, 200, 3), 0.5, dtype=np.float32) / 1.0
    grey[:] = 0.5
    got = s5._hand_skin_roi_photo(grey, (40.0, 20.0, 60.0, 60.0))
    assert got.shape[0] == 0


def test_hand_roi_equivalent_uint8_and_uint16():
    """同一内容的 8bit / 16bit 表达，归一化后应采到相同区域。"""
    v = (160, 100, 70)
    a8 = (np.full((200, 200, 3), v, dtype=np.float32) / 255.0)
    a16 = (np.full((200, 200, 3), v, dtype=np.float32) * 257.0 / 65535.0)
    n8 = s5._hand_skin_roi_photo(a8, (40.0, 20.0, 60.0, 60.0)).shape[0]
    n16 = s5._hand_skin_roi_photo(a16, (40.0, 20.0, 60.0, 60.0)).shape[0]
    assert n8 == n16 and n8 > 0, (n8, n16)
