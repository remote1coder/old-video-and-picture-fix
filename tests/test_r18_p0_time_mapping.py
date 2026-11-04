# -*- coding: utf-8 -*-
"""第十八轮 P0-C / R17-03：预期输出映射。

审核点名的两个症状：

1. **同帧数变速被抹掉**：stage7 在 VFR 重封装时按**源 PTS** 回填，没有考虑
   用户声明的变速/重解释 —— 合法的 2× 结果被改回原速度。
2. **倍帧被拒**：QTGMC 倍帧后帧数与源不同，`retime_to_source_pts` 直接拒绝
   （`frame count differs`），合法结果发不出去。
3. **只核末帧**：重封装后只比对最后一个 PTS；"末 PTS 相同、中间某处间隔错
   40ms"仍然返回成功 —— `max_interval_error` 算了却从不参与阻断。

本轮修法（最小可用）：
- `_has_timing_transform()` 声明了变换时**不回填**源 PTS，改由后面的
  `build_transform_verdict` 按声明裁定；
- `vfr_timecodes` 增加 `INTERVAL_TOLERANCE`，逐帧间隔参与阻断。

**完整预期输出映射（逐帧 PTS）仍未实现** —— 那需要把每个变换的映射显式
建模，属于 P0-C 的后续工作。本文件只锁住"不要把合法变换抹掉/拒掉"和
"不能只看终点"这两条底线。
"""
import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vfr_timecodes as vt  # noqa: E402


def _import_stage7():
    import importlib
    return importlib.import_module("stage7_encode")


# ------------------------------------------------ 变换判定

@pytest.mark.parametrize("params,expected", [
    ({}, False),
    ({"expected_speed": None}, False),
    ({"expected_speed": 1}, False),
    ({"expected_speed": 1.0}, False),
    ({"expected_speed": "1.0"}, False),
    ({"expected_speed": 2.0}, True),
    ({"expected_speed": "0.5"}, True),
    ({"expected_tempo": 1.5}, True),
    ({"expected_tempo": 1.0}, False),
    ({"fps_interpret": "25"}, True),
    ({"fps_interpret": ""}, False),
])
def test_R17_03_has_timing_transform(params, expected):
    """声明了变速/重解释才允许跳过源 PTS 回填。"""
    s7 = _import_stage7()
    assert s7._has_timing_transform(params) is expected


def test_R17_03_speed_two_is_not_overwritten_by_source_pts():
    """2 倍速（同帧数）必须被识别为变换 → 不回填源 PTS。

    审核探针：修复前合法的 2× 结果被改回源 PTS，耗时翻倍且用户看不出。
    """
    s7 = _import_stage7()
    assert s7._has_timing_transform({"expected_speed": 2.0}) is True


def test_R17_03_frame_doubling_is_not_rejected_as_vfr_mismatch():
    """倍帧（帧数与源不同）同样属于变换 —— 不该走"回填源 PTS"那条死路。

    审核探针：修复前 `frame count differs` 直接把合法结果判失败。
    """
    s7 = _import_stage7()
    assert s7._has_timing_transform({"fps_interpret": "50"}) is True


# ------------------------------------------------ 逐帧间隔阻断

def _fake_pts(n, step=1 / 30.0):
    return [i * step for i in range(n)]


def test_R17_03_interval_tolerance_is_documented_and_tight():
    """容差必须存在、且有依据（≤ 50ms，能抓住 40ms 级错误）。"""
    assert hasattr(vt, "INTERVAL_TOLERANCE")
    assert 0 < vt.INTERVAL_TOLERANCE <= 0.05


def test_T13_middle_interval_error_blocks_even_if_last_pts_matches(
        tmp_path, monkeypatch):
    """T13：末点相同但中间某处间隔错 40ms → **必须**失败。

    修复前 `max_interval_error` 只写进 info，不参与判定。
    """
    src_pts = _fake_pts(30)
    # 目标：总长与末帧完全一致，但第 10 个间隔被拉长 40ms（后面压缩回来）
    bad = list(src_pts)
    bad[11:] = [p + 0.04 for p in bad[11:]]
    # 让它末点回到一致：末尾减回来
    bad[-1] = src_pts[-1]

    src = tmp_path / "src.mkv"
    tgt = tmp_path / "tgt.mkv"
    src.write_bytes(b"src")
    tgt.write_bytes(b"tgt")

    calls = {"n": 0}

    def fake_probe(path, timeout=120):
        # 第 1 次：源；第 2 次：候选；第 3 次起：重封装后的结果
        calls["n"] += 1
        if calls["n"] == 1:
            return list(src_pts)
        if calls["n"] == 2:
            # 候选的时间轴**确实不对**（间隔 0.04 而非 0.0333），
            # 这样才会真正走到重封装那一步 —— 否则会被"已匹配"提前返回。
            return _fake_pts(30, 0.04)
        return list(bad)              # 重封装"之后"出现了 40ms 偏差

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(vt, "probe_pts", fake_probe)
    monkeypatch.setattr(vt, "write_timecodes_v2",
                        lambda pts, path, origin=0: (open(path, "w").close()
                                                     or True))
    monkeypatch.setattr(vt.subprocess, "run", lambda *a, **k: _R())

    ok, info = vt.retime_to_source_pts(str(src), str(tgt), "mkvmerge",
                                       work_dir=str(tmp_path))
    assert ok is False, f"中间间隔错 40ms 必须失败，实际 {info}"
    assert "interval" in (info.get("reason") or "").lower(), info
    assert not os.path.exists(str(tgt) + ".retimed.mkv"), "失败时必须清掉半成品"


def test_T13_clean_retime_still_succeeds(tmp_path, monkeypatch):
    """回归保护：真的对上了就不能被新判据误杀。"""
    src_pts = _fake_pts(30)
    shifted = [p + 0.002 for p in src_pts]   # 仅 v2 量化级偏差

    src = tmp_path / "src.mkv"
    tgt = tmp_path / "tgt.mkv"
    src.write_bytes(b"src")
    tgt.write_bytes(b"tgt")

    calls = {"n": 0}

    def fake_probe(path, timeout=120):
        calls["n"] += 1
        return list(src_pts) if calls["n"] == 1 else list(shifted)

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(vt, "probe_pts", fake_probe)
    monkeypatch.setattr(vt, "write_timecodes_v2",
                        lambda pts, path, origin=0: (open(path, "w").close()
                                                     or True))
    monkeypatch.setattr(vt.subprocess, "run", lambda *a, **k: _R())

    ok, info = vt.retime_to_source_pts(str(src), str(tgt), "mkvmerge",
                                       work_dir=str(tmp_path))
    assert ok is True, info
