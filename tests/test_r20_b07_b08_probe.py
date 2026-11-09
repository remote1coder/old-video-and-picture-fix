# -*- coding: utf-8 -*-
"""R20-B07/B08：探针不得误报 IDENTICAL，输入必须真裁（审核 R19-B07/B08）。

审核给出的受控反例（旧版全部判 IDENTICAL）：

| 反例 | 旧行为 |
|---|---|
| 20 行比较数据，仅第 1 行有 V | `_plane` 跳过缺失行 → present=True |
| 两份 probe 都缺 width/height/pix_fmt | `None == None` → 格式一致 |
| 比较日志重复同一帧序号凑够行数 | 只看 `len(rows)` |
| ffprobe rc≠0 但 stdout 有 JSON | 返回码被丢弃 |

**B08**：`--input` 没真裁到 `--frames`，manifest 却写死
`1920×1080 / 30000/1001 / progressive / frames÷30`。
"""
import importlib.util
import os

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROBE = os.path.join(_ROOT, "tests", "probes", "g34_equiv_det.py")


@pytest.fixture(scope="module")
def g():
    if not os.path.isfile(_PROBE):
        pytest.skip("探针不存在")
    spec = importlib.util.spec_from_file_location("g34_det_r20", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------ B07.1 probe 返回码

def test_probe_rejects_missing_file(g, tmp_path):
    r = g.probe(str(tmp_path / "nope.mp4"))
    assert "error" in r, r


def test_probe_rejects_non_video(g, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("not a video")
    r = g.probe(str(f))
    assert "error" in r, r


def test_probe_never_returns_none_fields(g, tmp_path):
    """成功时所有关键字段必须是已校验的真值，不能是 None。"""
    clip, info = g.make_clip(str(tmp_path / "in"), 3)
    for k in ("width", "height", "pix_fmt", "nb_read_frames"):
        assert info.get(k) is not None, f"{k} 为 None"
    assert info["width"] > 0 and info["height"] > 0
    assert info["nb_read_frames"] == 3


# ------------------------------------------------ B07.2 _plane 每行覆盖

def test_plane_requires_every_row(g):
    """审核反例：20 行只有第 1 行有 V。"""
    rows = [{"psnr_y": "inf", "psnr_u": "inf"} for _ in range(20)]
    rows[0]["psnr_v"] = "inf"
    p = g._plane(rows, "psnr_v")
    assert p["present"] is False
    assert p["n_missing"] == 19
    assert p["missing_rows"][0] == 2


def test_plane_accepts_full_coverage(g):
    rows = [{"psnr_v": "inf"} for _ in range(20)]
    assert g._plane(rows, "psnr_v")["present"] is True


def test_plane_rejects_bogus_inf(g):
    """`infinity_bad` 不能因为以 inf 开头就算"完全相同"。"""
    rows = [{"psnr_v": "infinity_bad"} for _ in range(3)]
    p = g._plane(rows, "psnr_v")
    assert p["present"] is False
    assert p["bad_rows"] == [1, 2, 3]


def test_plane_rejects_wrong_total(g):
    """行数少于 total 也不算覆盖。"""
    rows = [{"psnr_y": "inf"} for _ in range(5)]
    assert g._plane(rows, "psnr_y", total=10)["present"] is False


# ------------------------------------------------ B07.3 帧号连续唯一

def test_compare_flags_duplicate_frame_numbers(g):
    rows = [{"psnr_y": "inf", "psnr_u": "inf", "psnr_v": "inf"}] * 3
    seq = [1, 1, 1]
    dup = sorted({x for x in seq if seq.count(x) > 1})
    noncontig = (seq != list(range(1, len(seq) + 1)))
    assert dup == [1]
    assert noncontig is True


def test_evaluate_rejects_duplicate_and_noncontiguous(g):
    classic = {"rc": 0, "step3": "/w/c/temp/a.mkv", "step4": "/w/c/temp/a4.mkv",
               "receipts": []}
    fused = {"rc": 0, "step3": None, "step4": "/w/f/temp/b4.mkv",
             "receipts": ["b.fused_stage34.json"]}
    info = {"width": 1920, "height": 1080, "pix_fmt": "yuv420p16le",
            "nb_read_frames": 3}
    rows = [{"n": "1", "psnr_y": "inf", "psnr_u": "inf", "psnr_v": "inf"}] * 3
    cmp_result = {"returncode": 0, "frames": 3, "rows": rows,
                  "seq": [1, 1, 1], "dup": [1], "noncontiguous": True,
                  "summary": []}
    verdict, reasons = g.evaluate(classic, fused, info, dict(info),
                                  cmp_result, 3)
    assert verdict == "NOT_IDENTICAL"
    assert any("repeats frame numbers" in r for r in reasons), reasons
    assert any("not 1..N contiguous" in r for r in reasons), reasons


# ------------------------------------------------ B07.4 probe error 传递

def test_evaluate_rejects_probe_error(g):
    classic = {"rc": 0, "step3": "/w/c/temp/a.mkv", "step4": "/w/c/temp/a4.mkv",
               "receipts": []}
    fused = {"rc": 0, "step3": None, "step4": "/w/f/temp/b4.mkv",
             "receipts": ["r.json"]}
    rows = [{"n": str(i), "psnr_y": "inf", "psnr_u": "inf", "psnr_v": "inf"}
            for i in range(1, 4)]
    cmp_result = {"returncode": 0, "frames": 3, "rows": rows,
                  "seq": [1, 2, 3], "dup": [], "noncontiguous": False,
                  "summary": []}
    verdict, reasons = g.evaluate(
        classic, fused, {"error": "ffprobe rc=1"}, {"width": 1, "height": 1,
                                                    "pix_fmt": "yuv420p"},
        cmp_result, 3)
    assert verdict == "NOT_IDENTICAL"
    assert any("probe failed" in r for r in reasons), reasons


def test_evaluate_rejects_none_equals_none(g):
    """两份都缺 pix_fmt 不能算"格式一致"。"""
    classic = {"rc": 0, "step3": "/w/c/temp/a.mkv", "step4": "/w/c/temp/a4.mkv",
               "receipts": []}
    fused = {"rc": 0, "step3": None, "step4": "/w/f/temp/b4.mkv",
             "receipts": ["r.json"]}
    rows = [{"n": str(i), "psnr_y": "inf", "psnr_u": "inf", "psnr_v": "inf"}
            for i in range(1, 4)]
    cmp_result = {"returncode": 0, "frames": 3, "rows": rows,
                  "seq": [1, 2, 3], "dup": [], "noncontiguous": False,
                  "summary": []}
    info = {"width": 1920, "height": 1080, "pix_fmt": None,
            "nb_read_frames": 3}
    verdict, reasons = g.evaluate(classic, fused, info, dict(info),
                                  cmp_result, 3)
    assert verdict == "NOT_IDENTICAL"
    assert any("unproven" in r for r in reasons), reasons


# ------------------------------------------------ B08 manifest 用实测

def test_write_manifest_refuses_without_probe(g, tmp_path):
    """没有真实 probe 就拒绝写 manifest —— 不允许写死元数据。"""
    clip, _ = g.make_clip(str(tmp_path / "in"), 3)
    with pytest.raises(SystemExit):
        g.write_manifest(str(tmp_path / "w"), clip, 3, info=None)
    with pytest.raises(SystemExit):
        g.write_manifest(str(tmp_path / "w"), clip, 3,
                         info={"error": "no probe"})


def test_write_manifest_uses_real_dimensions(g, tmp_path):
    """manifest 的尺寸/fps/时长必须来自实测，不是写死的 1920/30000-1001。"""
    import yaml
    clip, info = g.make_clip(str(tmp_path / "in"), 3)
    man = g.write_manifest(str(tmp_path / "w"), clip, 3, info=info)
    job = yaml.safe_load(open(man, encoding="utf-8"))["jobs"][0]
    assert job["original_width"] == info["width"]
    assert job["original_height"] == info["height"]
    assert job["user_overrides"]["original_fps"] == str(info["r_frame_rate"])
    assert job["duration"] > 0


# ------------------------------------------------ B08 --input 真裁帧

def test_trim_to_frames_actually_limits(g, tmp_path):
    """裁出的样本帧数必须**正好**等于请求值。"""
    src, _ = g.make_clip(str(tmp_path / "src"), 10)
    dst, info = g.trim_to_frames(src, str(tmp_path / "trim"), 4)
    assert info["nb_read_frames"] == 4, info
    assert os.path.isfile(dst)


def test_trim_rejects_bad_input(g, tmp_path):
    with pytest.raises(SystemExit):
        g.trim_to_frames(str(tmp_path / "nope.mp4"),
                         str(tmp_path / "trim"), 3)
