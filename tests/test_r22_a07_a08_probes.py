# -*- coding: utf-8 -*-
"""R22-A07/A08：探针不能把差异汇总成"相同"。

## A07（N20 等价性探针）

审核反例：

> 输入两组 PTS `0, 0.04, 0.08, 0.12, 0.16` 与 `0, 0.04, 0.10, 0.12, 0.16`，
> 逐帧显然不同，但两份**摘要**相同；用原 probe 读取受控 ffprobe 输出，
> 再交给原 evaluator，仍判 IDENTICAL。

所以：判据必须逐帧比 PTS，摘要不足以做证据。

## A08（N21 颜色探针）

审核的真实工具反例：

| 项 | 值 |
|---|---|
| 输入 | 两张 64×64 RGB PPM，仅一个像素 R 由 100 改 101 |
| 正确 R 通道 MSE | `1/4096 = 0.000244140625` |
| ffmpeg 实际返回 | `mse_r:0.00`、`psnr_r:84.25` |
| 生产 aggregate 旧输出 | `mse=0.0 / psnr=inf`，无问题 |

`mse` 只有两位小数，"0.00" 是**显示精度**下的近似，不是精确零。
另外 `mse_*=inf` 被按 PSNR 的"正无穷=零误差"规则转成 0 —— 必须分开解析。
"""
import importlib.util
import os

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_G34 = os.path.join(_ROOT, "tests", "probes", "g34_equiv_det.py")
_N21 = os.path.join(_ROOT, "tests", "probes", "color_matrix_probe.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def g():
    return _load(_G34, "g34_a07")


@pytest.fixture(scope="module")
def c():
    return _load(_N21, "n21_a08")


# ------------------------------------------------ A07：逐帧 PTS

def _sides(g, pts_c, pts_f, dur=None):
    dur = dur or [0.04] * len(pts_c)
    classic = {"rc": 0, "step3": "/w/c/a3.mkv", "step4": "/w/c/a4.mkv",
               "receipts": []}
    fused = {"rc": 0, "step3": None, "step4": "/w/f/b4.mkv",
             "receipts": ["r.json"]}
    info_c = {"width": 320, "height": 240, "pix_fmt": "yuv420p16le",
              "nb_read_frames": len(pts_c), "frames": len(pts_c),
              "first_pts": pts_c[0], "last_pts": pts_c[-1],
              "last_end": pts_c[-1] + dur[-1], "median_interval": 0.04,
              "last_frame_duration": dur[-1],
              "pts": list(pts_c), "durations": list(dur)}
    info_f = dict(info_c, pts=list(pts_f),
                  first_pts=pts_f[0], last_pts=pts_f[-1],
                  last_end=pts_f[-1] + dur[-1])
    rows = [{"n": str(i), "psnr_y": "inf", "psnr_u": "inf", "psnr_v": "inf"}
            for i in range(1, len(pts_c) + 1)]
    cmp_result = {"returncode": 0, "frames": len(pts_c), "rows": rows,
                  "seq": list(range(1, len(pts_c) + 1)), "dup": [],
                  "noncontiguous": False, "summary": []}
    return classic, fused, info_c, info_f, cmp_result


def test_middle_pts_shift_with_identical_summary_is_rejected(g):
    """审核原例：摘要相同（首尾/帧数/中位）但中段错位 → 必须拒绝。"""
    a = [0.0, 0.04, 0.08, 0.12, 0.16]
    b = [0.0, 0.04, 0.10, 0.12, 0.16]
    # 先确认"摘要"确实一样（否则这条测试就没意义）
    assert (len(a) == len(b) and a[0] == b[0] and a[-1] == b[-1])
    verdict, reasons = g.evaluate(*_sides(g, a, b), 5)
    assert verdict == "NOT_IDENTICAL"
    assert any("per-frame PTS differs at frame 2" in r for r in reasons), reasons


def test_identical_pts_still_passes(g):
    a = [0.0, 0.04, 0.08, 0.12, 0.16]
    verdict, reasons = g.evaluate(*_sides(g, a, list(a)), 5)
    assert verdict == "IDENTICAL", reasons


def test_missing_pts_arrays_are_not_evidence(g):
    """没有逐帧数组 → 判"未证明"，不能用摘要凑。"""
    a = [0.0, 0.04, 0.08]
    classic, fused, info_c, info_f, cmp_result = _sides(g, a, a)
    info_c.pop("pts")
    verdict, reasons = g.evaluate(classic, fused, info_c, info_f, cmp_result, 3)
    assert verdict == "NOT_IDENTICAL"
    assert any("PTS arrays are missing" in r for r in reasons), reasons


def test_last_frame_duration_differs_is_rejected(g):
    """末帧被重复填充会让末帧 duration 不同 —— 也要拒。"""
    a = [0.0, 0.04, 0.08]
    classic, fused, info_c, info_f, cmp_result = _sides(g, a, a)
    info_f["durations"] = [0.04, 0.04, 0.08]
    verdict, reasons = g.evaluate(classic, fused, info_c, info_f, cmp_result, 3)
    assert verdict == "NOT_IDENTICAL"
    assert any("last frame duration" in r for r in reasons), reasons


def test_probe_keeps_raw_file_order_for_b_frames(g, monkeypatch):
    """有 B 帧时文件顺序 ≠ 显示顺序：要排序取显示顺序，但保留原始顺序。"""
    class R:
        returncode = 0
        stdout = "0.0,0.04\n0.133,0.033\n0.066,0.033\n"
        stderr = ""

    monkeypatch.setattr(g.subprocess, "run", lambda *a, **k: R())
    out = g.probe_timing("x.mkv")
    assert out["pts"] == [0.0, 0.066, 0.133]        # 显示顺序
    assert out["pts_file_order"] == [0.0, 0.133, 0.066]
    assert out["reordered_by_pts"] is True


def test_probe_rejects_duplicate_presentation_timestamps(g, monkeypatch):
    class R:
        returncode = 0
        stdout = "0.0,0.04\n0.04,0.04\n0.04,0.04\n"
        stderr = ""

    monkeypatch.setattr(g.subprocess, "run", lambda *a, **k: R())
    out = g.probe_timing("x.mkv")
    assert "error" in out
    assert "duplicate presentation timestamp" in out["error"]


# ------------------------------------------------ A08：精确零 vs 舍入零

def test_single_pixel_difference_is_not_zero(c):
    """审核的真实反例：mse 显示 0.00、psnr 84.25 → MSE 必须是非零。"""
    rows = [{"n": "1", "mse_r": "0.00", "mse_g": "0.00", "mse_b": "0.00",
             "psnr_r": "84.25", "psnr_g": "inf", "psnr_b": "inf"}]
    agg = c.aggregate({"frames": 1, "rows": rows, "problems": []})
    r = agg["planes"]["r"]
    assert r["mse"] > 0, "把显示精度下的 0.00 当成了精确零"
    assert r["mse"] == pytest.approx(1 / 4096, rel=0.01), r
    assert r["exact_zero"] is False
    assert r["mse_from_psnr_frames"] == 1
    # 另外两个通道确实是零
    assert agg["planes"]["g"]["exact_zero"] is True


def test_exact_zero_requires_every_frame_to_be_inf(c):
    rows = [{"n": "1", "mse_r": "0", "psnr_r": "inf"},
            {"n": "2", "mse_r": "0.00", "psnr_r": "60"}]
    agg = c.aggregate({"frames": 2, "rows": rows, "problems": []})
    assert agg["planes"]["r"]["exact_zero"] is False, (
        "只要有一帧 PSNR 有限，就不能声称精确相等")
    assert agg["planes"]["r"]["mse"] > 0


def test_mse_inf_is_illegal_not_zero_error(c):
    """`mse=inf` 不能按 PSNR 的"正无穷=零误差"处理。"""
    rows = [{"n": "1", "mse_r": "inf", "psnr_r": "40"}]
    agg = c.aggregate({"frames": 1, "rows": rows, "problems": []})
    assert agg["planes"]["r"]["mse"] is None
    assert agg["problems"], "非法值必须留下问题记录"


@pytest.mark.parametrize("tok", ["-inf", "-infinity", "nan", "abc"])
def test_bogus_values_are_rejected(c, tok):
    assert c._num(tok) == "bad"


def test_psnr_inf_means_zero_for_psnr_only(c):
    assert c._num("inf") is None            # PSNR：正无穷 = 零误差
    assert c._num("inf", zero_means="bad") == "bad"   # MSE：正无穷 = 非法


def test_444_roundtrip_is_judged_by_budget_not_exactness(c):
    """444 往返允许小误差；超预算才判无效（R22-A08）。"""
    def combo(mse):
        return {"combo": "444_x", "chroma": "444", "matrix": "bt709",
                "range": "limited", "frames": 1, "problems": [],
                "planes": {p: {"mse": mse, "rmse": mse ** 0.5, "psnr": 48,
                               "frames": 1, "exact_zero": mse == 0}
                           for p in ("r", "g", "b")}}

    ok_row = {"combo": "420_x", "chroma": "420", "matrix": "bt709",
              "range": "limited", "frames": 1, "problems": [],
              "planes": {p: {"mse": 1.0, "rmse": 1.0, "psnr": 48,
                             "frames": 1, "exact_zero": False}
                         for p in ("r", "g", "b")}}
    v, why = c.evaluate({"roundtrip": [combo(0.5), ok_row]}, expected_combos=2)
    assert v == "OK", why
    v2, why2 = c.evaluate({"roundtrip": [combo(4.0), ok_row]},
                          expected_combos=2)
    assert v2 == "INVALID"
    assert any("error budget" in r for r in why2), why2
