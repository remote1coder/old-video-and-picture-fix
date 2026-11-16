# -*- coding: utf-8 -*-
"""`evaluate_checks` 的判定不能依赖报告里字段的**顺序**（R26-A03）。

原来第二个循环用 `row`（上一个循环最后一次迭代的残留值）判断
"报告自称 required=False 吗"，于是这一条永远在检查**最后一项**。
审核实测：合法的必需 PASS 项和可选 PASS 项只互换顺序，一份被错拒、
一份通过 —— 确定的变量引用错误，不是媒体随机性。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared_modules"))

import pytest

import required_checks as rc


def _plan():
    """一份"必需 + 可选"混合的计划。"""
    return rc.derive(
        {"required": {"media": "m", "verify": "v", "marker": "k",
                      "fullchain": "f"}, "optional": {}},
        {"input_path": "a.mp4"},
        media_kind="video")


def _rows(**extra):
    base = {
        "mechanical": {"status": "PASS", "required": True},
        "final_timeline": {"status": "PASS", "required": True},
        "frame_order": {"status": "PASS", "required": False},   # 可选
        "fullchain": {"status": "PASS", "required": True},
    }
    base.update(extra)
    return base


def test_all_pass_accepts_in_either_order():
    """全 PASS 时，两种字段顺序都必须通过。"""
    plan = _plan()
    a = dict(list(_rows().items()))
    b = dict(list(_rows().items())[::-1])
    assert rc.evaluate_checks(a, plan) == []
    assert rc.evaluate_checks(b, plan) == []


def test_optional_before_required_and_after():
    """可选 PUT 在首位 vs 末位，结果必须一致。"""
    plan = _plan()
    rows = _rows()
    # 把可选的 frame_order 移到最前
    opt = rows.pop("frame_order")
    first = {"frame_order": opt, **rows}
    last = {**rows, "frame_order": opt}
    assert rc.evaluate_checks(first, plan) == rc.evaluate_checks(last, plan) == []


def test_relaxed_required_detected_regardless_of_position():
    """必需的 fullchain 自报 required=False：**无论排第几都要被抓**。"""
    plan = _plan()
    bad = {"status": "PASS", "required": False}

    # fullchain 放最后（旧实现只在这种情况下碰巧能抓到）
    rows_last = _rows()
    rows_last["fullchain"] = bad
    p_last = rc.evaluate_checks(rows_last, plan)

    # fullchain 放最前（旧实现会漏）
    items = list(_rows().items())
    items = [("fullchain", bad)] + [(k, v) for k, v in items if k != "fullchain"]
    p_first = rc.evaluate_checks(dict(items), plan)

    assert p_last, "末位时没抓到"
    assert p_first, "首位时没抓到（顺序依赖 bug）"
    assert len(p_last) == len(p_first) == 1
    assert "relax" in p_last[0]


def test_relaxing_a_different_check_does_not_trip_others():
    """放松的是**可选**项时不该报（计划没要求它）。"""
    plan = _plan()
    rows = _rows(frame_order={"status": "PASS", "required": False})
    assert rc.evaluate_checks(rows, plan) == []


def test_malformed_row_is_reported_once():
    """非 dict 行报"不是对象"，且不再重复报"缺失"。"""
    plan = _plan()
    rows = _rows()
    rows["mechanical"] = "PASS"          # 不是对象
    probs = rc.evaluate_checks(rows, plan)
    joined = " | ".join(probs)
    assert "is not an object" in joined
    # 不应同时报"plan requires the 'mechanical' check but ... not contain"
    assert "does not contain it" not in joined, probs


def test_missing_required_still_reported():
    """真的缺失（plan 要、报告没有）仍要报。"""
    plan = _plan()
    rows = _rows()
    rows.pop("fullchain")
    probs = rc.evaluate_checks(rows, plan)
    assert any("does not contain it" in p for p in probs), probs
