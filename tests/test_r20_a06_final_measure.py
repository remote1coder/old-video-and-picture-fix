# -*- coding: utf-8 -*-
"""R20-A06：最后一次 mux 之后必须**重新测量**最终候选。

审核 R19-A06 原文：

> 最后一次 mux/重封装后重测最终候选，旧缓存失效，再生成报告与发布决定。

**缺陷**

`processed_duration` / `_timing_basis` 是在**编码之前**用中间产物
`input_source` 算出来的（`_probe_packet_span(input_source)`）。之后的
`retime_to_source_pts()` 会 `os.replace()` 掉候选文件 —— 新容器的时间轴
**按设计就是要变的**。报告却继续用编码前那个数字，描述的是一个**已经不存在
的文件**。

**第二个缺陷（同一病族）**

`build_transform_verdict()` 把"有没有测到"压缩成
`timing_basis == "packet-span"` 这个**字符串相等**判断。换一种同样可信的
测量方式（容器时长、重定时助手的实测）就会被误判成"没测"。

**第三个缺陷（量化步长）**

`is_variable()` 只用相对容差（中位间隔的 2%）。ffprobe 在 MKV 里的
`pts_time` 是**毫秒量化**，30fps 的 33.333 ms 被舍成 33/34 交替 ——
相对偏差 3% > 2%，CFR 被判成 VFR，于是白跑一次全片重封装。
与 R19 已修的 `_classify_timeline` 是同一病族：**容差必须大于量化步长**。

**第四个缺陷（探针异常不算问题）**

`probe` 异常 / 空 PTS / 时长未知 以前只落成一条问题列表，
`timeline_mapping_strict=false` 时打一条 CRITICAL 就放行 ——
"测不出来"于是等于"没问题"。现在它是一条正式 `CheckResult`。
"""
import ast
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE7 = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                       "stage7_encode.py")
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage7_encode as s7  # noqa: E402
import timeline_plan as tp  # noqa: E402
import vfr_timecodes as vt  # noqa: E402


def _src():
    return open(_STAGE7, encoding="utf-8", errors="replace").read()


class _P:
    def __init__(self, rc, out=""):
        self.returncode = rc
        self.stdout = out
        self.stderr = ""


class _FakeSubprocess:
    """只实现 `run()` 的 subprocess 替身，按 ffprobe 的查询串分派。"""

    def __init__(self, stream_dur=None, fmt_dur=None, packets=None,
                 raise_on=None):
        self.stream_dur = stream_dur
        self.fmt_dur = fmt_dur
        self.packets = packets
        self.raise_on = raise_on or ()

    def run(self, cmd, **kw):
        joined = " ".join(str(c) for c in cmd)
        for k in self.raise_on:
            if k in joined:
                raise OSError(f"boom: {k}")
        if "packet=pts_time,duration_time" in joined:
            if self.packets is None:
                return _P(1, "")
            return _P(0, self.packets)
        if "stream=duration" in joined:
            return _P(0, "" if self.stream_dur is None else str(self.stream_dur))
        if "format=duration" in joined:
            return _P(0, "" if self.fmt_dur is None else str(self.fmt_dur))
        return _P(1, "")


def _packets(intervals, start=0.0, dur=0.033):
    """把间隔序列写成 ffprobe 的 csv：`pts,duration` 每行一条。"""
    lines = []
    t = start
    for iv in intervals:
        lines.append(f"{t:.6f},{dur:.6f}")
        t += iv
    return "\n".join(lines) + "\n"


# ------------------------------------------------ ① 取值顺序与代价

def test_remeasure_prefers_the_video_stream_duration(monkeypatch):
    """有视频流时长时用它 —— 它**含末帧时长**、且不受音轨影响。

    R21-C05 改了优先级：`retime_info["out_last"]` 是"首帧→末帧"，
    不含末帧自己的时长，不能当完整覆盖；所以它退到最后兜底，
    只在没有其它证据时才用，并且要标注不完整。
    """
    monkeypatch.setattr(
        s7, "subprocess",
        _FakeSubprocess(stream_dur=15.5, fmt_dur=99.0, packets="1.0,0.03\n"))
    monkeypatch.setattr(s7, "_count_frames_quietly", lambda *a, **k: 0)
    # 没有重定时 → 视频流时长直接可用，不需要全扫
    info = s7.measure_final_candidate(
        "cand.mkv", None, previous_duration=12.0, previous_basis="packet-span")
    assert info["measured"] is True
    assert info["duration"] == pytest.approx(15.5)
    assert info["basis"] == "stream-duration"
    assert info["coverage_complete"] is True
    assert info["attempts"][0]["source"] == "stream-duration"


def test_remeasure_uses_stream_duration_without_scanning(monkeypatch):
    """没有重定时 → 用视频流时长；**不得**再跑一次 packet 全扫。"""
    monkeypatch.setattr(
        s7, "subprocess",
        _FakeSubprocess(stream_dur=15.4, fmt_dur=15.6, packets=_packets([0.04] * 5)))
    monkeypatch.setattr(s7, "_count_frames_quietly", lambda *a, **k: 385)
    info = s7.measure_final_candidate("cand.mkv", None, previous_duration=15.4)
    assert info["measured"] is True
    assert info["duration"] == pytest.approx(15.4)
    assert info["basis"] == "stream-duration"
    assert [a["source"] for a in info["attempts"]] == ["stream-duration"]
    assert info["frames"] == 385


def test_remeasure_falls_back_to_container_duration(monkeypatch):
    monkeypatch.setattr(s7, "subprocess",
                        _FakeSubprocess(stream_dur=None, fmt_dur=9.5))
    monkeypatch.setattr(s7, "_count_frames_quietly", lambda *a, **k: 0)
    info = s7.measure_final_candidate("cand.mkv", None)
    assert info["measured"] is True
    assert info["basis"] == "container-duration"
    assert info["duration"] == pytest.approx(9.5)


def test_remeasure_rescans_only_when_nothing_cheaper_is_available(monkeypatch):
    """容器/流时长都取不到时才付全扫的代价，并算出是否变帧率。"""
    monkeypatch.setattr(
        s7, "subprocess",
        _FakeSubprocess(packets=_packets([0.04] * 9 + [0.08] * 9)))
    monkeypatch.setattr(s7, "_count_frames_quietly", lambda *a, **k: 0)
    info = s7.measure_final_candidate("cand.mkv", None)
    assert info["measured"] is True
    # R21-C05：basis 改名为 video-packet-span（强调"视频流、含末帧时长"）
    assert info["basis"] == "video-packet-span"
    assert info["frames"] == 18
    assert info["varies"] is True
    assert info["coverage_complete"] is True
    assert [a["source"] for a in info["attempts"]] == ["packet-scan"]


def test_remeasure_reports_unmeasured_instead_of_reusing_the_old_value(monkeypatch):
    """全部取不到 → measured=False 且 duration=0（旧值不得冒充已验证）。"""
    monkeypatch.setattr(s7, "subprocess", _FakeSubprocess())
    monkeypatch.setattr(s7, "_count_frames_quietly", lambda *a, **k: 0)
    info = s7.measure_final_candidate("cand.mkv", None,
                                      previous_duration=12.0,
                                      previous_basis="packet-span")
    assert info["measured"] is False
    assert info["duration"] == 0.0
    assert info["problems"], "取不到时长必须留下问题记录"
    assert info["previous_duration"] == pytest.approx(12.0)


def test_remeasure_survives_a_probe_exception(monkeypatch):
    monkeypatch.setattr(s7, "subprocess",
                        _FakeSubprocess(raise_on=("ffprobe",)))
    monkeypatch.setattr(s7, "_count_frames_quietly", lambda *a, **k: 0)
    info = s7.measure_final_candidate("cand.mkv", None, previous_duration=12.0)
    assert info["measured"] is False
    assert info["duration"] == 0.0
    assert info["problems"]


# ------------------------------------------------ ② 时长变化要留痕

def test_changed_flag_uses_a_relative_tolerance(monkeypatch):
    monkeypatch.setattr(s7, "subprocess", _FakeSubprocess(stream_dur=10.4))
    monkeypatch.setattr(s7, "_count_frames_quietly", lambda *a, **k: 0)
    near = s7.measure_final_candidate("c.mkv", None, previous_duration=10.0)
    assert near["changed"] is False, "4% 的变化在默认 5% 容差之内"
    assert near["delta_ms"] == pytest.approx(400.0, abs=1.0)

    monkeypatch.setattr(s7, "subprocess", _FakeSubprocess(stream_dur=10.6))
    far = s7.measure_final_candidate("c.mkv", None, previous_duration=10.0)
    assert far["changed"] is True
    assert far["delta_ms"] == pytest.approx(600.0, abs=1.0)


# ------------------------------------------------ ③ 量化步长（病族）

def test_is_variable_tolerates_millisecond_quantisation():
    """30fps 的 33.333ms 被毫秒量化成 33/34 交替 —— 这是 CFR，不是 VFR。"""
    pts = []
    t = 0.0
    for i in range(60):
        pts.append(round(t, 3))
        # 真实的 1/30 s 量化到毫秒，天然会产生 33/34 的交替
        t += 1.0 / 30.0
    assert vt.is_variable(pts) is False, (
        "毫秒量化造成的 1ms 抖动被误判成了变帧率；"
        "容差必须大于量化步长")


def test_is_variable_still_detects_real_vfr():
    pts = []
    t = 0.0
    for i in range(20):
        pts.append(round(t, 3))
        t += 0.04 if i < 10 else 0.12
    assert vt.is_variable(pts) is True


def test_is_variable_keeps_the_relative_part_for_large_intervals():
    """0.5s 的中位间隔下，100ms 的偏差仍必须判为变帧率（绝对下限不会盖住它）。"""
    pts = []
    t = 0.0
    for i in range(10):
        pts.append(round(t, 3))
        t += 0.5
    pts.append(round(t + 0.1, 3))
    assert vt.is_variable(pts) is True


def test_interval_floor_is_documented_and_shared():
    """下限必须是一个具名常量 —— 两处（stage7 兜底 / 这里）共用一个值。"""
    assert hasattr(vt, "INTERVAL_ABS_FLOOR_S")
    assert vt.INTERVAL_ABS_FLOOR_S == pytest.approx(0.002)


# ------------------------------------------------ ④ 测量有效性不再靠字符串

def test_transform_verdict_accepts_an_explicit_valid_measurement():
    """同样可信的测量方式（容器时长）不能被字符串相等判成"没测"。"""
    v = tp.build_transform_verdict(
        declared_speed=2.0, original_duration=20.0, processed_duration=10.0,
        timing_state="adjusted", tempo_applied=2.0,
        timing_basis="container-duration", measurement_valid=True)
    assert v["status"] == "PASS", v["reason"]
    assert v["observed_ratio"] == pytest.approx(2.0)


def test_transform_verdict_rejects_when_measurement_is_invalid():
    """反过来：字符串写着 packet-span，但调用方说没测到 → 仍是 UNVERIFIED。"""
    v = tp.build_transform_verdict(
        declared_speed=2.0, original_duration=20.0, processed_duration=10.0,
        timing_state="adjusted", tempo_applied=2.0,
        timing_basis="packet-span", measurement_valid=False)
    assert v["status"] == "UNVERIFIED"


def test_transform_verdict_defaults_to_the_old_string_behaviour():
    """不传新形参时行为不变（向后兼容，老调用点不动）。"""
    v = tp.build_transform_verdict(
        declared_speed=2.0, original_duration=20.0, processed_duration=10.0,
        timing_state="adjusted", tempo_applied=2.0, timing_basis="packet-span")
    assert v["status"] == "PASS"
    v2 = tp.build_transform_verdict(
        declared_speed=2.0, original_duration=20.0, processed_duration=10.0,
        timing_state="adjusted", tempo_applied=2.0,
        timing_basis="frame-ratio (estimated)")
    assert v2["status"] == "UNVERIFIED"


# ------------------------------------------------ ⑤ 接线：位置与判定门

def test_remeasure_runs_after_the_candidate_is_frozen():
    """必须在候选**冻结之后**测，否则测的又是另一批字节。"""
    src = _src()
    frozen = src.index("_cand_digest = _sha256_file(_publish_tmp)")
    measured = src.index("_final_tl = measure_final_candidate(")
    published = src.index("from publisher import")
    assert frozen < measured < published, (
        "重新测量必须落在『摘要算完』与『开始提交』之间："
        f"frozen={frozen} measured={measured} published={published}")


def test_remeasure_hands_the_retime_measurement_to_the_helper():
    src = _src()
    assert "retime_info=_retime_info" in src


def test_unmeasured_discards_the_pre_encode_number():
    """测不到时 processed_duration 必须清零，不能拿编码前的值顶包。"""
    src = _src()
    i = src.index("_final_tl = measure_final_candidate(")
    tail = src[i:i + 2600]
    assert "processed_duration = 0.0" in tail
    assert 'unverified (post-mux probe failed)' in tail


def test_final_timeline_is_a_formal_check_result():
    """探针异常/时长未知必须进决策门，不能只当"问题列表"。"""
    src = _src()
    assert '"final_timeline"' in src
    i = src.index('_checks["final_timeline"]')
    tail = src[i:i + 2200]
    assert '"UNVERIFIED"' in tail
    assert 'force_key' in tail


def test_measurement_valid_is_passed_to_the_verdict():
    src = _src()
    assert "measurement_valid=bool(_final_tl.get(\"measured\"))" in src


def test_measure_final_candidate_is_a_module_level_function():
    tree = ast.parse(_src())
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert "measure_final_candidate" in names
    assert "_probe_container_duration" in names
    assert "_probe_final_timeline" in names
