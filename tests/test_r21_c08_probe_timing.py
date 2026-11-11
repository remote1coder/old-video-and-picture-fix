# -*- coding: utf-8 -*-
"""R21-C08：N20 探针还会误报等价（时间轴 / EOF / 负 Inf / 缺回执）。

## 审核原文（R20 §3，P1）

> - evaluator 比较尺寸、格式、帧数，却未核对实际 PTS、尾帧覆盖和时间映射。
>   受控输入中，声明时间不同仍判 IDENTICAL。
> - `_INF_TOKENS` 接受 `-inf/-infinity`。负无穷不能证明零误差。
> - FFmpeg 比较没有明确建立适用的 EOF/补帧策略；仅数到 N 行不保证比较的是
>   两边原本完整对应的 N 帧。必须独立核对两边 PTS 与覆盖，不能让隐式重采样、
>   重复末帧制造一致。
> - 回执检查仍偏重文件名/数量，未复用完整内容验证。

## 本文件

直接测 `g34_equiv_det.evaluate()`（纯函数）与 `_plane()`（解析器），
每条判据单独钉住。**探针自身不过负例时，它的 IDENTICAL 不可采信** ——
所以这些用例必须先绿，再谈 N20 的结论。
"""
import importlib.util
import os

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROBE = os.path.join(_ROOT, "tests", "probes", "g34_equiv_det.py")


@pytest.fixture(scope="module")
def g():
    if not os.path.isfile(_PROBE):
        pytest.skip("探针不存在")
    spec = importlib.util.spec_from_file_location("g34_c08", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sides(frames=20, **info_over):
    classic = {"rc": 0, "step3": "/w/c/temp/a_step3.mkv",
               "step4": "/w/c/temp/a_step4.mkv", "receipts": []}
    fused = {"rc": 0, "step3": None, "step4": "/w/f/temp/b_step4.mkv",
             "receipts": ["b.fused_stage34.json"]}
    info = {"width": 3840, "height": 2160, "pix_fmt": "yuv420p16le",
            "nb_read_frames": frames, "frames": frames,
            "first_pts": 0.0, "last_pts": 0.033 * (frames - 1),
            "last_end": 0.033 * frames, "median_interval": 0.033,
            # R22-A07：判据逐帧比 PTS，合法样本必须给出逐帧证据
            "pts": [round(i * 0.033, 6) for i in range(frames)],
            "durations": [0.033] * frames}
    info.update(info_over)
    return classic, fused, info, dict(info)


def _rows(frames=20, plane="inf"):
    return [{"n": str(i), "psnr_y": plane, "psnr_u": plane, "psnr_v": plane}
            for i in range(1, frames + 1)]


def _cmp(frames=20, rows=None):
    return {"returncode": 0, "frames": frames,
            "rows": rows if rows is not None else _rows(frames),
            "seq": list(range(1, frames + 1)), "dup": [],
            "noncontiguous": False, "summary": []}


# ------------------------------------------------ 时间轴

def test_identical_requires_matching_timeline(g):
    verdict, reasons = g.evaluate(*_sides(), _cmp(), 20)
    assert verdict == "IDENTICAL", reasons


def test_different_frame_reinterpretation_is_rejected(g):
    """帧数/尺寸/像素都一样，但 PTS 不同 → 不是等价。"""
    classic, fused, info_c, info_f = _sides()
    info_f = dict(info_f, last_pts=0.066 * 19, last_end=0.066 * 20,
                  median_interval=0.066)
    verdict, reasons = g.evaluate(classic, fused, info_c, info_f, _cmp(), 20)
    assert verdict == "NOT_IDENTICAL"
    assert any("last_pts" in r or "last_end" in r for r in reasons), reasons


def test_last_frame_coverage_difference_is_rejected(g):
    """末帧覆盖不同 = 末帧时长不同，即使 PTS 全同也必须拒。"""
    classic, fused, info_c, info_f = _sides()
    info_f = dict(info_f, last_end=info_f["last_end"] + 0.02)
    verdict, reasons = g.evaluate(classic, fused, info_c, info_f, _cmp(), 20)
    assert verdict == "NOT_IDENTICAL"
    assert any("last_end" in r for r in reasons), reasons


def test_missing_timing_fields_are_not_evidence(g):
    """缺 PTS 观测 → 不能声称等价（旧实现只看帧数就过了）。"""
    classic, fused, info_c, info_f = _sides()
    info_c = {k: v for k, v in info_c.items()
              if k not in ("first_pts", "last_pts", "last_end")}
    verdict, reasons = g.evaluate(classic, fused, info_c, info_f, _cmp(), 20)
    assert verdict == "NOT_IDENTICAL"
    assert any("unproven" in r for r in reasons), reasons


# ------------------------------------------------ EOF / 补帧

def test_partial_comparison_is_rejected(g):
    """只比较 1/N 帧 → 不是证据。"""
    classic, fused, info_c, info_f = _sides()
    cmp_partial = _cmp(frames=1, rows=_rows(1))
    verdict, reasons = g.evaluate(classic, fused, info_c, info_f,
                                  cmp_partial, 20)
    assert verdict == "NOT_IDENTICAL"
    assert any("partial comparison" in r for r in reasons), reasons


def test_repeated_last_frame_in_the_psnr_log_is_rejected(g):
    """比较日志里末帧被重复计入（EOF 补帧）→ 帧号不连续。"""
    classic, fused, info_c, info_f = _sides()
    rows = _rows(20)
    rows.append(dict(rows[-1]))       # 末帧重复
    cmp_dup = {"returncode": 0, "frames": 21, "rows": rows,
               "seq": list(range(1, 21)) + [20], "dup": [20],
               "noncontiguous": True, "summary": []}
    verdict, reasons = g.evaluate(classic, fused, info_c, info_f, cmp_dup, 20)
    assert verdict == "NOT_IDENTICAL"
    assert any("repeats frame numbers" in r for r in reasons), reasons


# ------------------------------------------------ 负 Inf / 平面缺失

@pytest.mark.parametrize("bogus", ["-inf", "-infinity", "INFINITY_BAD"])
def test_negative_or_bogus_infinity_is_not_zero_error(g, bogus):
    """审核点名 `-inf`：负无穷不能证明零误差。"""
    rows = _rows(20, plane=bogus)
    got = g._plane(rows, "psnr_y", total=20)
    assert got["present"] is False, f"{bogus!r} 被当成了零误差"


def test_negative_infinity_blocks_identical(g):
    classic, fused, info_c, info_f = _sides()
    rows = _rows(20)
    rows[5]["psnr_u"] = "-inf"
    verdict, reasons = g.evaluate(classic, fused, info_c, info_f,
                                  _cmp(rows=rows), 20)
    assert verdict == "NOT_IDENTICAL"
    assert reasons


def test_missing_plane_blocks_identical(g):
    classic, fused, info_c, info_f = _sides()
    rows = _rows(20)
    for r in rows:
        r.pop("psnr_v")
    verdict, reasons = g.evaluate(classic, fused, info_c, info_f,
                                  _cmp(rows=rows), 20)
    assert verdict == "NOT_IDENTICAL"
    assert any("planes missing" in r for r in reasons), reasons


# ------------------------------------------------ 回执 / 模式

def test_missing_receipt_blocks_identical(g):
    classic, fused, info_c, info_f = _sides()
    fused = dict(fused, receipts=[])
    verdict, reasons = g.evaluate(classic, fused, info_c, info_f, _cmp(), 20)
    assert verdict == "NOT_IDENTICAL"
    assert any("no *.fused_stage34.json" in r for r in reasons), reasons


def test_fused_still_producing_step3_blocks_identical(g):
    classic, fused, info_c, info_f = _sides()
    fused = dict(fused, step3="/w/f/temp/b_step3.mkv")
    verdict, reasons = g.evaluate(classic, fused, info_c, info_f, _cmp(), 20)
    assert verdict == "NOT_IDENTICAL"
    assert any("fusion did not actually run" in r for r in reasons), reasons
