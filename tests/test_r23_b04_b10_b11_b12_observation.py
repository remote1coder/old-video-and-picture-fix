# -*- coding: utf-8 -*-
"""R23-B04 / B10 / B11 / B12：观测与比较必须"先证明自己跑对，再谈相等"。

## 审核给的反例（本轮先重放、再修）

| 位置 | 反例 | 修复前 |
|---|---|---|
| B04 `stage7` 最终时间门 | CSV `0,inf` | span=inf、coverage=observed → **PASS** |
| B04 同上 | PTS `0,0,0.04`、duration 全 0.04 | 3 帧、完整覆盖 → **PASS** |
| B10 `g34_equiv_det.evaluate` | 一侧中段 PTS=NaN | **IDENTICAL** |
| B10 同上 | 两侧 PTS 都为空但声称 5 帧 | **IDENTICAL** |
| B10 同上 | 两侧 durations 缺失 / 尾帧 duration=0 | **IDENTICAL** |
| B11 `color_matrix_probe` | `mse_r:1` + `psnr_r:inf` | 归零、`exact_zero=True` |
| B11 同上 | 8 份同名/零帧/空平面组合 | **OK** |
| B11 同上 | 两张 16-bit PPM 差 1 个码值 | rgb24 下判 **inf**（真差异被抹平）|
| B12 重复性探针 | rc≠0 但留 1 行 inf / 无成品 / 无比较对 | **PASS** |

## 共同病根

**把"我要比较的东西"当成"我已经比较了"**：数组存在 ≠ 有效；
一行 inf ≠ 全帧相同；8-bit 相同 ≠ 16-bit 相同；条目数量对 ≠ 身份对。
所以这里每条判据都先问"观测本身完整吗"，再问"两边相等吗"。
"""
import json
import os
import struct
import subprocess
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages"),
           os.path.join(_ROOT, "tests", "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage7_encode as s7  # noqa: E402
import g34_equiv_det as g34  # noqa: E402
import color_matrix_probe as cm  # noqa: E402
import r22_repeatability as rep  # noqa: E402


# ------------------------------------------------ B04：最终时间门

@pytest.mark.parametrize("value,ok", [
    (0.4, True), (1, True), (0.0001, True),
    (0, False), (-1, False), (float("inf"), False), (float("-inf"), False),
    (float("nan"), False), (True, False), ("0.4", False), (None, False),
])
def test_duration_is_usable(value, ok):
    """`inf` / `nan` / 布尔 / 字符串都不是可用时长。"""
    assert s7._duration_is_usable(value) is ok


def test_final_timeline_passes_only_with_all_three_facts():
    good = {"measured": True, "coverage_complete": True, "duration": 0.4,
            "basis": "video-packet-span"}
    assert s7.final_timeline_status(good)[0] == "PASS"
    for broken in (dict(good, duration=float("inf")),
                   dict(good, duration=float("nan")),
                   dict(good, duration=True),
                   dict(good, coverage_complete=False),
                   dict(good, measured=False)):
        status, why = s7.final_timeline_status(broken)
        assert status == "UNVERIFIED", (broken, status, why)
        assert why


def _fake_ffprobe(monkeypatch, csv_text):
    class _R:
        returncode = 0
        stdout = csv_text
        stderr = ""

    monkeypatch.setattr(s7.subprocess, "run", lambda *a, **k: _R())


def test_probe_rejects_non_finite_packets(monkeypatch):
    _fake_ffprobe(monkeypatch, "0.000000,inf\n")
    info = s7._probe_final_timeline("fake.mkv")
    assert info["ok"] is False
    assert "non-finite" in (info.get("problem") or "")


def test_probe_rejects_duplicate_display_timestamps(monkeypatch):
    _fake_ffprobe(monkeypatch, "0.000000,0.040000\n0.000000,0.040000\n"
                               "0.040000,0.040000\n")
    info = s7._probe_final_timeline("fake.mkv")
    assert info["ok"] is False
    assert "duplicate" in (info.get("problem") or "")


# ------------------------------------------------ B10：N20 判定

def _n20_info(**over):
    info = {"width": 10, "height": 10, "pix_fmt": "yuv420p16le",
            "frames": 5, "nb_read_frames": 5, "first_pts": 0.0,
            "last_pts": 0.16, "last_end": 0.2,
            "pts": [0.0, 0.04, 0.08, 0.12, 0.16],
            "durations": [0.04] * 5}
    info.update(over)
    return info


_N20_CLASSIC = {"rc": 0, "step3": "x", "receipts": [1], "step4": "y"}
_N20_FUSED = {"rc": 0, "step3": None, "receipts": [1], "step4": "y"}
_N20_CMP = {"returncode": 0, "frames": 5, "dup": 0, "noncontiguous": False,
            "rows": [{"psnr_y": "inf", "psnr_u": "inf", "psnr_v": "inf"}] * 5}


def _n20(c, f):
    return g34.evaluate(_N20_CLASSIC, _N20_FUSED, c, f, _N20_CMP, 5)


def test_n20_identical_needs_a_complete_observation():
    """正例：观测完整且全 inf → IDENTICAL（不许修成永远拒绝）。"""
    assert _n20(_n20_info(), _n20_info()) == ("IDENTICAL", [])


@pytest.mark.parametrize("tweak,fragment", [
    (lambda i: i["pts"].__setitem__(2, float("nan")), "non-finite"),
    (lambda i: i.update(pts=[]), "empty"),
    (lambda i: i.pop("durations"), "durations are missing"),
    (lambda i: i["durations"].__setitem__(-1, 0.0), "last frame duration"),
    (lambda i: i.update(pts=[0.0, 0.04]), "!= decoded frames"),
])
def test_n20_rejects_incomplete_timelines(tweak, fragment):
    c = _n20_info()
    f = _n20_info()
    tweak(f)
    if tweak.__code__.co_consts and "pts" in str(tweak.__code__.co_consts):
        pass
    verdict, reasons = _n20(c, f)
    assert verdict == "NOT_IDENTICAL", reasons
    assert any(fragment in r for r in reasons), reasons


# ------------------------------------------------ B11：N21 统计与组合

def test_contradictory_mse_and_psnr_is_rejected():
    res = {"frames": 1, "problems": [], "seq": [1],
           "rows": [{"n": "1", "mse_r": "1.00", "psnr_r": "inf",
                     "mse_g": "0.00", "psnr_g": "inf",
                     "mse_b": "0.00", "psnr_b": "inf"}]}
    agg = cm.aggregate(res)
    assert any("contradictory" in p for p in agg["problems"]), agg["problems"]
    assert agg["planes"]["r"]["exact_zero"] is False


def _combo_row(name, frames=6, mse=0.0, exact=False, exact16=None):
    return {"combo": name, "chroma": "420",
            "planes": {p: {"mse": mse, "frames": frames,
                           "exact_zero": exact} for p in ("r", "g", "b")},
            "exact16": exact16 if exact16 is not None else {"ok": True}}


def test_duplicate_combo_identities_are_rejected():
    rows = [_combo_row("same") for _ in range(8)]
    verdict, reasons = cm.evaluate({"roundtrip": rows, "mismatch": []},
                                   expected_frames=6, expected_combos=8)
    assert verdict == "INVALID"
    assert any("duplicate combo" in r for r in reasons), reasons


def test_zero_frame_combo_is_rejected():
    rows = [_combo_row(f"c{i}", frames=0) for i in range(8)]
    verdict, reasons = cm.evaluate({"roundtrip": rows, "mismatch": []},
                                   expected_frames=6, expected_combos=8)
    assert verdict == "INVALID"
    assert any("0 of 6 frames" in r for r in reasons), reasons


def test_exact_claim_needs_the_16bit_comparison():
    rows = [_combo_row("c1", exact=True,
                       exact16={"ok": False, "differing": 3,
                                "max_code_diff": 1})]
    verdict, reasons = cm.evaluate({"roundtrip": rows, "mismatch": []},
                                   expected_frames=6, expected_combos=1)
    assert verdict == "INVALID"
    assert any("16-bit comparison differs" in r for r in reasons), reasons


def _write_ppm16(path, pixels, w=16, h=16):
    """16-bit binary PPM（P6，maxval 65535，大端样本）。"""
    body = b"".join(struct.pack(">HHH", *px) for px in pixels)
    with open(path, "wb") as f:
        f.write(b"P6\n%d %d\n65535\n" % (w, h))
        f.write(body)
    return path


def test_a_one_code_16bit_difference_is_not_lost(tmp_path):
    """审核的真实反例：两张 16×16 的 16-bit PPM 只差**一个 R 码值**。

    `psnr` 滤镜把两边都 `format=rgb24` 之后得到全 `inf` —— 那只证明
    "在 8-bit 表示上相同"。16-bit 精确比较必须能看见这个差异。
    """
    w = h = 16
    base = [(30000, 20000, 10000)] * (w * h)
    other = list(base)
    other[5] = (30001, 20000, 10000)
    a = _write_ppm16(str(tmp_path / "a.ppm"), base, w, h)
    b = _write_ppm16(str(tmp_path / "b.ppm"), other, w, h)

    # 8-bit 比较：相同（这正是审核指出的"看起来相等"）
    res8 = cm.psnr_rgb(a, b, expected_frames=1)
    agg8 = cm.aggregate(res8)
    assert agg8["planes"]["r"]["exact_zero"] is True, agg8["planes"]["r"]

    # 16-bit 精确比较：必须看见 1 个码值的差异
    exact = cm.rgb48_exact_diff(a, b)
    assert exact["problems"] == [], exact
    assert exact["ok"] is False, "16-bit 单码值差异被当成完全相同"
    assert exact["differing"] >= 1
    assert exact["max_code_diff"] == 1


def test_identical_16bit_inputs_compare_equal(tmp_path):
    """反面对照：真的相同就要判相同（不许一律判不同）。"""
    px = [(30000, 20000, 10000)] * 256
    a = _write_ppm16(str(tmp_path / "a.ppm"), px)
    b = _write_ppm16(str(tmp_path / "b.ppm"), px)
    exact = cm.rgb48_exact_diff(a, b)
    assert exact["problems"] == [], exact
    assert exact["ok"] is True


# ------------------------------------------------ B12：重复性比较

@pytest.mark.parametrize("rc,rows,frames,ok", [
    (0, ["n:1 ..."] * 8, 8, True),
    (0, ["n:1 ..."], 8, False),          # 1 行不是 8 帧
    (1, ["n:1 ..."] * 8, 8, False),      # 工具失败
    (0, [], 8, False),
    (0, ["n:1 ..."] * 8, 0, False),
])
def test_comparison_completeness(rc, rows, frames, ok):
    assert rep._comparison_complete(rc, rows, frames) is ok


def test_psnr_rows_keeps_the_exit_code(tmp_path):
    """工具失败时 rc 不能被丢掉（旧实现只回 rows）。"""
    rc, rows = rep._psnr_rows(str(tmp_path / "nope-a.mkv"),
                              str(tmp_path / "nope-b.mkv"))
    assert rc != 0
    assert rows == []
