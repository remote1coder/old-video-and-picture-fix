# -*- coding: utf-8 -*-
"""color_diagnose 的单元测试。

覆盖点（对应 2026-10-22 的真实素材教训）：
1. **只读**：诊断不得修改入参数组，且只写 json/txt，不产图片。
2. **三态判定**：ok / content_color / faded 三个分支都要有用例。
3. **关键反例**：红色婚礼照（暖色来自物体本色）不得被判成褪色。
   旧实现（stage2 的全局通道均值判据）在 150 张真实素材里误触发 23%，
   其中就包含这类画面。
4. **正例**：模拟青染料褪色（蓝通道量程塌缩 + 中性参考偏暖）必须报警。
"""
import json
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "shared_modules"))

import color_diagnose as cd  # noqa: E402


def _img(rgb, h=120, w=160):
    """构造纯色 float32 图像。"""
    a = np.zeros((h, w, 3), dtype=np.float32)
    a[:, :, 0], a[:, :, 1], a[:, :, 2] = rgb
    return a


def _write_rgb(path, f_float):
    """把 float32 RGB 数组写盘。

    **必须用 PIL**：`media_io.read_image_array` 返回的是 **RGB** 顺序
    （已用纯红图验证：PIL 写 RGB=255,0,0 → 读回 [255,0,0]），
    而 `cv2.imwrite` 按 **BGR** 解释数组。用 cv2 写 RGB 数组会静默
    交换红蓝通道，导致读回来 R/B 变成倒数——本文件早期就踩过这个坑。
    """
    from PIL import Image
    Image.fromarray((np.clip(f_float, 0, 1) * 255).astype(np.uint8)).save(path)


def _noisy_neutral(seed=0, h=160, w=200, level=0.45, spread=0.25):
    """中性灰噪点图：三通道统计一致，蓝通道量程正常。"""
    rng = np.random.default_rng(seed)
    base = rng.normal(level, spread, size=(h, w)).clip(0, 1).astype(np.float32)
    return np.stack([base, base, base], axis=2)


def _faded_like(seed=0, h=160, w=200):
    """模拟青染料褪色：蓝通道量程压到 ~35% 且整体压低 → 偏暖。"""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.62, 0.22, size=(h, w)).clip(0, 1)
    g = rng.normal(0.60, 0.22, size=(h, w)).clip(0, 1)
    b = (rng.normal(0.55, 0.22, size=(h, w)).clip(0, 1) * 0.35 + 0.03)
    return np.stack([r, g, b], axis=2).astype(np.float32)


def _warm_lit_like(seed=7, h=160, w=200):
    """模拟暖光源照明（如钨丝灯下的室内照）：整体偏暖但三通道量程健康。

    这是 content_color 分支的典型场景 —— 偏色来自**光照/物体本色**，
    不是染料褪色，校正它就是改变内容。
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.42, 0.16, size=(h, w)).clip(0.05, 0.85)
    return np.stack([base * 1.20, base, base * 0.90],
                    axis=2).clip(0, 1).astype(np.float32)


# ---------------------------------------------------------------- 只读性


def test_analyze_array_does_not_modify_input():
    """诊断**不得**修改入参数组 —— 这是"一个像素都不改"的底线。"""
    a = _noisy_neutral()
    before = a.copy()
    cd.analyze_array(a)
    assert np.array_equal(a, before), "analyze_array 修改了入参"


def test_diagnose_writes_only_json_and_txt(tmp_path):
    """批量诊断只允许产出 json/txt，不得写任何图片。"""
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    _write_rgb(str(src_dir / "a.png"), _noisy_neutral())
    _write_rgb(str(src_dir / "b.png"), _noisy_neutral(seed=1))

    out_dir = tmp_path / "out"
    rep = cd.diagnose_paths([str(src_dir)], str(out_dir))

    written = sorted(p.name for p in out_dir.iterdir())
    assert written == ["color_cast_report.json", "color_cast_summary.txt"], \
        f"诊断写入了意外文件: {written}"
    assert os.path.exists(rep["report_json"])
    assert os.path.exists(rep["summary_txt"])


# ---------------------------------------------------------------- 三态判定


def test_neutral_image_is_ok_and_not_recommended():
    """中性图 → ok，不建议校正。"""
    r = cd.analyze_array(_noisy_neutral())
    assert r["verdict"] == "ok"
    assert r["recommend_correction"] is False
    assert abs(r["neutral_ref_rb_ratio"] - 1.0) < 0.10


def test_faded_image_is_flagged():
    """模拟褪色 → faded，且建议校正，并给出逐通道拉伸量。"""
    r = cd.analyze_array(_faded_like())
    assert r["verdict"] == "faded"
    assert r["recommend_correction"] is True
    assert r["blue_range_deficit"] is True
    # 蓝通道需要更大的拉伸倍数（这正是"青塌缩"的可操作信号）
    stretch = r["suggested_per_channel_stretch"]
    assert stretch[2] > stretch[0], \
        f"蓝通道拉伸量应最大，实际 {stretch}"


def test_warm_tint_with_healthy_blue_range_is_content_color():
    """只有暖色偏、蓝通道量程正常 → content_color（物体本色，不要动）。

    这条对应真实教训：红色婚礼照被旧判据误判成"褪色"。旧实现用全局
    通道均值差判定，150 张真实素材里误触发 23%，其中就包含这类画面。
    """
    res = cd.analyze_array(_warm_lit_like())
    assert res["neutral_ref_rb_ratio"] > 1.10, "构造的暖偏强度不够"
    assert res["blue_range_deficit"] is False, "构造图不符：蓝量程不该塌缩"
    assert res["verdict"] == "content_color"
    assert res["recommend_correction"] is False
    assert any("物体本色" in x or "场景色温" in x for x in res["reasons"])


# ---------------------------------------------------------------- 边界


def test_unreadable_file_does_not_raise(tmp_path):
    """读不了的文件只标记 unreadable，不抛异常、不算建议校正。"""
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image at all")
    r = cd.diagnose_file(str(bad))
    assert r["verdict"] == "unreadable"
    assert r["recommend_correction"] is False


def test_grayscale_and_rgba_are_normalized():
    """2 通道（灰度）与 4 通道（带 alpha）都要能处理。"""
    gray = np.full((64, 64), 128, dtype=np.uint8)
    r1 = cd.analyze_array(cd._to_float_rgb(gray))
    assert abs(r1["neutral_ref_rb_ratio"] - 1.0) < 0.05

    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[:, :, :3] = 128
    rgba[:, :, 3] = 255
    r2 = cd.analyze_array(cd._to_float_rgb(rgba))
    assert abs(r2["neutral_ref_rb_ratio"] - 1.0) < 0.05


def test_uint16_is_scaled_not_clipped():
    """16-bit 输入必须按 65535 归一，否则会被当成全白。"""
    a = np.full((40, 40, 3), 32768, dtype=np.uint16)
    f = cd._to_float_rgb(a)
    assert 0.49 < float(f.mean()) < 0.51, f"16-bit 归一化错误: {f.mean()}"


def test_summary_counts_are_consistent(tmp_path):
    """summary 里的计数必须与 images 里的一致。"""
    d = tmp_path / "in"
    d.mkdir()
    _write_rgb(str(d / "ok.png"), _noisy_neutral())
    _write_rgb(str(d / "faded.png"), _faded_like())

    rep = cd.diagnose_paths([str(d)], str(tmp_path / "out"))
    s = rep["summary"]
    assert s["total"] == 2
    assert sum(s["by_verdict"].values()) == 2
    assert s["recommend_count"] == len(s["recommend_files"])
    assert any("faded" in p for p in s["recommend_files"])


def test_rgb_order_contract():
    """documented contract：media_io.read_image_array 返回 **RGB**。

    如果哪天它改成 BGR，本模块的 R/B 会静默变成倒数——用纯色图钉住。
    """
    from PIL import Image
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "pure_red.png")
        Image.fromarray(np.array([[[255, 0, 0]]], dtype=np.uint8)).resize(
            (16, 16), Image.NEAREST).save(p)
        f = cd._read_rgb_float(p)
        assert f is not None
        px = f[0, 0]
        assert px[0] > 0.9 and px[1] < 0.1 and px[2] < 0.1, \
            f"read_image_array 通道顺序不是 RGB: {px}"
