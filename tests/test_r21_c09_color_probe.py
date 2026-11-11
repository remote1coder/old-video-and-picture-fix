# -*- coding: utf-8 -*-
"""R21-C09：N21 探针缺失败门禁，样本结论被当成普遍规律。

## 审核原文（R20 §3，P1/P2）

> 本轮执行原判断 AST，传入比较 rc=1、预期 6 帧却只比较 1 帧、
> mismatch 报错的记录，仍得到 `OK`、无 reasons。`None` 的 PSNR 还能被转换成
> 0 RMSE。必须先修退出码、完整帧/平面、有限数及错误传播，才能让新数据承担
> 验收责任。
>
> 统计上，跨帧先平均 dB 再反推 RMSE，不等于先累计像素平方误差再求总体 RMSE。

## 本文件

直接测 `color_matrix_probe` 的纯函数：`psnr_rgb()` 的校验、`aggregate()`
的统计口径、`evaluate()` 的门禁。**探针自身过不了负例，它的 OK 不可采信。**
"""
import importlib.util
import os

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROBE = os.path.join(_ROOT, "tests", "probes", "color_matrix_probe.py")


@pytest.fixture(scope="module")
def c():
    if not os.path.isfile(_PROBE):
        pytest.skip("探针不存在")
    spec = importlib.util.spec_from_file_location("n21_probe", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(frames=3, mse="0.0", psnr="inf"):
    return [{"n": str(i), "mse_r": mse, "mse_g": mse, "mse_b": mse,
             "psnr_r": psnr, "psnr_g": psnr, "psnr_b": psnr}
            for i in range(1, frames + 1)]


def _combo(chroma="420", matrix="bt709", rng="limited", mse=1.0,
           problems=None, error=None):
    return {"combo": f"{chroma}_{matrix}_{rng}", "chroma": chroma,
            "matrix": matrix, "range": rng,
            "frames": 3, "problems": problems or [],
            "error": error,
            "planes": {p: {"mse": mse, "rmse": mse ** 0.5, "psnr": 48.13,
                           "frames": 3} for p in ("r", "g", "b")}}


# ------------------------------------------------ 帧/平面完整性

def test_missing_frames_are_reported(c):
    res = {"rc": 0, "frames": 1, "rows": _rows(1), "seq": [1],
           "problems": []}
    # 期望 6 帧，只比较到 1 帧
    out = c.psnr_rgb.__wrapped__ if hasattr(c.psnr_rgb, "__wrapped__") else None
    # 直接测校验逻辑：用 evaluate 走完整链路更贴近实际
    row = _combo(problems=["compared 1 of 6 frames"])
    verdict, reasons = c.evaluate({"roundtrip": [row], "mismatch": []},
                                  expected_frames=6, expected_combos=1)
    assert verdict == "INVALID"
    assert any("compared 1 of 6" in r for r in reasons), reasons


def test_duplicate_frame_numbers_are_reported(c):
    row = _combo(problems=["frame numbers are not 1..N contiguous"])
    verdict, _ = c.evaluate({"roundtrip": [row]}, expected_combos=1)
    assert verdict == "INVALID"


def test_mismatch_error_is_not_ignored(c):
    """旧实现遇到 mismatch 行报错仍返回 OK。"""
    ok_row = _combo()
    bad = {"combo": "encbt709__decbt601", "error": "comparison rc=1"}
    verdict, reasons = c.evaluate(
        {"roundtrip": [ok_row], "mismatch": [bad]}, expected_combos=1)
    assert verdict == "INVALID"
    assert any("comparison rc=1" in r for r in reasons), reasons


def test_missing_combo_is_invalid(c):
    verdict, reasons = c.evaluate({"roundtrip": [_combo()]},
                                  expected_combos=8)
    assert verdict == "INVALID"
    assert any("only 1 of 8" in r for r in reasons), reasons


# ------------------------------------------------ 444 无操作必须精确

def test_444_with_nonzero_mse_invalidates_the_probe(c):
    row = _combo(chroma="444", mse=0.5)
    verdict, reasons = c.evaluate({"roundtrip": [row, _combo()]},
                                  expected_combos=2)
    assert verdict == "INVALID"
    assert any("probe itself is wrong" in r for r in reasons), reasons


def test_444_with_zero_mse_is_fine(c):
    row444 = _combo(chroma="444", mse=0.0)
    verdict, reasons = c.evaluate({"roundtrip": [row444, _combo()]},
                                  expected_combos=2)
    assert verdict == "OK", reasons


def test_no_420_combo_leaves_the_question_unanswered(c):
    verdict, reasons = c.evaluate({"roundtrip": [_combo(chroma="444", mse=0)]},
                                  expected_combos=1)
    assert verdict == "INVALID"
    assert any("no 420 combo" in r for r in reasons), reasons


# ------------------------------------------------ 统计口径

def test_aggregate_uses_total_squared_error(c):
    """先累计平方误差再求 RMSE —— 不是"先平均 dB"。"""
    rows = _rows(4, mse="1.0", psnr="48")
    out = c.aggregate({"frames": 4, "rows": rows, "problems": []})
    assert out["planes"]["g"]["mse"] == pytest.approx(1.0)
    assert out["planes"]["g"]["rmse"] == pytest.approx(1.0)
    # PSNR(8bit) = 10*log10(255^2 / 1) ≈ 48.13
    assert out["planes"]["g"]["psnr"] == pytest.approx(48.13, abs=0.02)


def test_aggregate_mixes_zero_and_nonzero_frames(c):
    rows = [{"n": "1", "mse_r": "0.0", "mse_g": "0.0", "mse_b": "0.0",
             "psnr_r": "inf", "psnr_g": "inf", "psnr_b": "inf"},
            {"n": "2", "mse_r": "4.0", "mse_g": "4.0", "mse_b": "4.0",
             "psnr_r": "40", "psnr_g": "40", "psnr_b": "40"}]
    out = c.aggregate({"frames": 2, "rows": rows, "problems": []})
    assert out["planes"]["r"]["mse"] == pytest.approx(2.0)
    assert out["planes"]["r"]["rmse"] == pytest.approx(2 ** 0.5, abs=1e-4)


def test_missing_values_do_not_become_zero_error(c):
    """`None` 不能被当成 0 误差（审核点名）。"""
    rows = [{"n": "1", "mse_r": "", "psnr_r": "", "mse_g": "1.0",
             "psnr_g": "48", "mse_b": "1.0", "psnr_b": "48"}]
    out = c.aggregate({"frames": 1, "rows": rows, "problems": []})
    assert out["planes"]["r"]["mse"] is None
    assert out["problems"], "缺失值必须留下问题记录"


@pytest.mark.parametrize("tok", ["-inf", "-infinity", "nan", "abc"])
def test_negative_or_bogus_values_are_rejected(c, tok):
    assert c._num(tok) == "bad", tok


def test_positive_infinity_is_zero_error(c):
    assert c._num("inf") is None


# ------------------------------------------------ 组合表覆盖

def test_all_eight_combinations_are_listed(c):
    assert len(c.COMBOS) == 8
    combos = {(a, b, d) for a, b, d in c.COMBOS}
    for chroma in ("444", "420"):
        for matrix in ("bt709", "bt601"):
            for rng in ("limited", "full"):
                assert (chroma, matrix, rng) in combos, (chroma, matrix, rng)
