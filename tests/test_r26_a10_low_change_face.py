# -*- coding: utf-8 -*-
"""有脸但变化很小，不能判成失败（R26-A10）。

复核原话："当前按8bit量纲差值大于12.75构造mask；有脸但mask为空就失败。
有效、细微的修复可能没有任何像素越过阈值，这不能证明模型没执行。"

所以 mask 必须由**已确认的人脸区域**决定，而不是由"变化够不够大"决定。
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared_modules"))
sys.path.insert(0, str(ROOT / "photo_restoration_pipeline" / "stages"))

import stage4_ai_face as s4  # noqa: E402


class _Log:
    def __init__(self):
        self.events = []

    def info(self, ev, msg, **kw): self.events.append(("I", ev))
    def warning(self, ev, msg, **kw): self.events.append(("W", ev))
    def error(self, ev, msg, **kw): self.events.append(("E", ev))

    def names(self):
        return [e[1] for e in self.events]


def _write16(path, value=1000, shape=(60, 60, 3)):
    tifffile.imwrite(str(path), np.full(shape, value, np.uint16),
                     compression="zlib")


def test_tiny_change_inside_face_is_a_success(tmp_path):
    """差异只有 2 个 8bit 量级（远低于 12.75）→ **必须成功**。"""
    orig = tmp_path / "o.tif"
    rest = tmp_path / "r.tif"
    out = tmp_path / "x.tif"
    _write16(orig, 1000)
    # 修复图 = 原值/257 + 2 → 差异约 2（旧阈值 12.75 会判"没改"）
    base = (1000 // 257)
    tifffile.imwrite(str(rest),
                     np.full((60, 60, 3), base + 2, np.uint8),
                     compression="zlib")
    log = _Log()
    # `feather_width=2` → pad=6，ROI 约 19..46；用默认 8（pad=24）会把
    # 60×60 的整幅都包进去，"ROI 外"就没有采样点了。
    ok = s4.feather_blend(str(orig), str(rest), str(out), bit_depth=16,
                          feather_width=2,
                          face_boxes=[(25, 25, 10, 10)], logger=log)
    assert ok is True, ("小幅但有效的修复被判成失败了：%s" % log.names())
    got = tifffile.imread(str(out))
    # ROI 内必须取修复值（base+2 抬到 16bit 量级），不是原值
    center = int(got[30, 30, 0])
    assert center != 1000, "ROI 内没有用修复值（mask 还是被差异筛空了）"
    # ROI 外必须逐位保留
    assert int(got[0, 0, 0]) == 1000, "ROI 外被改写了"


def test_zero_change_inside_face_is_still_not_a_failure(tmp_path):
    """模型原样返回（差异=0）：交付原图，但**不报失败**。

    这不是"修复成功"，也不是"失败" —— 它是"模型没改"。旧实现会在这里
    进 `not mask.any()` 分支并 `return False`，把"没改"当成"坏了"。
    """
    orig = tmp_path / "o2.tif"
    rest = tmp_path / "r2.tif"
    out = tmp_path / "x2.tif"
    # 用 257 的整数倍，保证 8bit↔16bit 往返无损（1000 不是 257 的倍数，
    # 会被量化成 771 —— 那是测试用例的问题，不是实现的问题）。
    _write16(orig, 1028)
    base = (1028 // 257)
    tifffile.imwrite(str(rest), np.full((60, 60, 3), base, np.uint8),
                     compression="zlib")
    log = _Log()
    # `feather_width=2` → pad=6，ROI 约 19..46；用默认 8（pad=24）会把
    # 60×60 的整幅都包进去，"ROI 外"就没有采样点了。
    ok = s4.feather_blend(str(orig), str(rest), str(out), bit_depth=16,
                          feather_width=2,
                          face_boxes=[(25, 25, 10, 10)], logger=log)
    assert ok is True, log.names()
    assert int(tifffile.imread(str(out))[30, 30, 0]) == 1028


def test_boxes_entirely_outside_the_frame_is_a_real_failure(tmp_path):
    """框坐标全在画外 → 这是真错误，不是"低变化"。"""
    orig = tmp_path / "o3.tif"
    rest = tmp_path / "r3.tif"
    out = tmp_path / "x3.tif"
    _write16(orig, 1000)
    tifffile.imwrite(str(rest), np.full((60, 60, 3), 50, np.uint8),
                     compression="zlib")
    log = _Log()
    ok = s4.feather_blend(str(orig), str(rest), str(out), bit_depth=16,
                          face_boxes=[(500, 500, 20, 20)], logger=log)
    assert ok is False
    assert "STAGE4_FACE_ROI_OUTSIDE_FRAME" in log.names(), log.names()


def test_no_boxes_still_delivers_the_original(tmp_path):
    """`face_boxes=[]`（确实无脸）→ 原样交付且成功。"""
    orig = tmp_path / "o4.tif"
    rest = tmp_path / "r4.tif"
    out = tmp_path / "x4.tif"
    _write16(orig, 1000)
    tifffile.imwrite(str(rest), np.full((60, 60, 3), 50, np.uint8),
                     compression="zlib")
    ok = s4.feather_blend(str(orig), str(rest), str(out), bit_depth=16,
                          face_boxes=[], logger=_Log())
    assert ok is True
    assert int(tifffile.imread(str(out))[0, 0, 0]) == 1000
