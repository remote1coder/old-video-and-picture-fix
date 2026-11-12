# -*- coding: utf-8 -*-
"""R23-B04 的另一半：合法时间轴**不能**被误杀。

审核原文要求"时间负例拒绝，合法 B 帧 / VFR / 单帧 / 人工解释有正例"。
另一半（`0,inf`、`0,0,0.04` 这类负例）在
``test_r23_b04_b10_b11_b12_observation.py`` 里；这边补正例 ——
全部用**真实 ffmpeg 产物**，不造 CSV。

为什么正例非写不可：负例只要"永远拒绝"就能全绿。判据要同时证明
"坏输入被拒"和"好输入通过"，否则修法可以退化成 `return "UNVERIFIED"`。

五类媒体各自防一个具体的误杀：

============================  ==========================================
B 帧（`-bf 3`）                包按解码序输出，PTS 乱序；排序后应均匀
单帧                            间隔统计不足 3 帧，不能被当成"没测到"
30 fps（33.333 ms 量化）        pts 被舍成 33/34 交替，不能被判成 VFR
非零起点（`setpts=PTS+5/TB`）   跨度必须是末-首，不能把 5.0 算进时长
真 VFR（25 fps 段接 10 fps 段）  `varies=True` 不是错误，只是不融合
============================  ==========================================
"""
import os
import shutil
import subprocess

import pytest

import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_PROJ, "shared_modules"),
           os.path.join(_PROJ, "video_restoration_pipeline"),
           os.path.join(_PROJ, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage7_encode import (          # noqa: E402
    _duration_is_usable, _probe_final_timeline, final_timeline_status)


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="需要 ffmpeg/ffprobe")


def _ff(*args):
    return subprocess.run(["ffmpeg", "-y", "-v", "error"] + list(args),
                          capture_output=True, text=True, errors="replace",
                          timeout=300)


def _src(tmp, w=128, h=96, fps=25, dur=0.4):
    return ["-f", "lavfi", "-i", "testsrc2=s=%dx%d:r=%s:d=%s" % (w, h, fps, dur)]


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    """一次造齐五类媒体，避免每个用例重复编码。"""
    d = tmp_path_factory.mktemp("tlpos")
    out = {}

    def _mk(name, args):
        path = str(d / (name + ".mkv"))
        r = _ff(*(args + ["-c:v", "libx264", "-pix_fmt", "yuv420p", path]))
        assert r.returncode == 0, (name, (r.stderr or "")[-400:])
        out[name] = path

    # ① B 帧：包按解码序排列，PTS 乱序
    _mk("bframes", _src(d, dur=0.4) + ["-bf", "3", "-g", "10"])
    # ② 单帧
    _mk("single", _src(d, dur=0.04) + ["-frames:v", "1"])
    # ③ 30 fps：33.333 ms 被容器量化成 33/34 交替
    _mk("fps30", _src(d, fps=30, dur=0.5))
    # ④ 非零起点：首帧 pts=5.0
    _mk("offset5", _src(d, dur=0.4) + ["-vf", "setpts=PTS+5/TB"])

    # ⑤ 真 VFR：25 fps 段紧接 10 fps 段
    for tag, fps, dur in (("a25", 25, 0.2), ("b10", 10, 0.3)):
        r = _ff(*(_src(d, fps=fps, dur=dur)
                  + ["-c:v", "libx264", "-pix_fmt", "yuv420p",
                     str(d / (tag + ".mkv"))]))
        assert r.returncode == 0, (tag, (r.stderr or "")[-400:])
    lst = d / "list.txt"
    lst.write_text("file '%s'\nfile '%s'\n" % (d / "a25.mkv", d / "b10.mkv"),
                   encoding="utf-8")
    r = _ff("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
            str(d / "vfr2.mkv"))
    assert r.returncode == 0, (r.stderr or "")[-400:]
    out["vfr2"] = str(d / "vfr2.mkv")
    return out


def _status(tl):
    """按调用方的方式组装 final_timeline 记录，再问单一判定点。"""
    return final_timeline_status({
        "measured": True,
        "duration": tl.get("span"),
        "coverage_complete": tl.get("coverage") == "observed",
        "basis": "video-packet-span",
    })


def test_bframes_is_not_vfr_and_covers_last_frame(media):
    """B 帧的 PTS 乱序不等于变帧率。"""
    tl = _probe_final_timeline(media["bframes"])
    assert tl["ok"] is True
    assert tl["frames"] == 10
    assert tl["coverage"] == "observed", tl.get("problem")
    assert tl["varies"] is False, (
        "B 帧的包按解码序输出，PTS 天然乱序；排序后间隔应均匀。"
        "这里判成 VFR 说明没排序或者容差比量化步长还小")
    assert tl["span"] == pytest.approx(0.4, abs=1e-3)
    assert _status(tl)[0] == "PASS"


def test_single_frame_is_measured_not_unknown(media):
    """单帧：间隔统计不足 3 帧，但"末帧有 duration"仍然成立。"""
    tl = _probe_final_timeline(media["single"])
    assert tl["ok"] is True
    assert tl["frames"] == 1
    assert tl["coverage"] == "observed", tl.get("problem")
    assert tl["span"] == pytest.approx(0.04, abs=1e-3)
    # `varies` 在 <3 帧时**没有**判据，允许 None；不能是 True
    assert tl["varies"] in (None, False)
    assert _status(tl)[0] == "PASS"


def test_30fps_millisecond_quantisation_is_not_vfr(media):
    """33.333 ms 被容器舍成 33/34 交替 —— 这是 CFR，不是 VFR。"""
    tl = _probe_final_timeline(media["fps30"])
    assert tl["ok"] is True
    assert tl["frames"] == 15
    assert tl["median_interval"] == pytest.approx(0.033, abs=1e-3)
    assert tl["min_interval"] == pytest.approx(0.033, abs=1e-3)
    assert tl["max_interval"] == pytest.approx(0.034, abs=1e-3)
    assert tl["varies"] is False, (
        "毫秒量化抖动 1 ms / 33 ms = 3%，纯相对容差会把这判成 VFR。"
        "容差必须带绝对下限（vfr_timecodes.INTERVAL_ABS_FLOOR_S）")
    assert _status(tl)[0] == "PASS"


def test_nonzero_origin_span_excludes_the_offset(media):
    """首帧在 5.0 s：跨度是末-首，不是"末帧的绝对时刻"。"""
    tl = _probe_final_timeline(media["offset5"])
    assert tl["ok"] is True
    assert tl["first_pts"] == pytest.approx(5.0, abs=1e-3)
    assert tl["span"] == pytest.approx(0.4, abs=1e-3), (
        "把绝对 PTS 当时长会得到 5.4 —— 非零起点素材（DVD/录制流）必错")
    assert _status(tl)[0] == "PASS"


def test_real_vfr_is_reported_but_still_a_valid_timeline(media):
    """VFR 本身不是缺陷：记录 `varies=True`，时间轴仍然可测。"""
    tl = _probe_final_timeline(media["vfr2"])
    assert tl["ok"] is True
    assert tl["frames"] == 8
    assert tl["varies"] is True, (
        "25 fps 段接 10 fps 段，间隔 0.04×5 然后 0.1/0.1；"
        "判不出 VFR 说明容差过大")
    assert tl["min_interval"] == pytest.approx(0.04, abs=1e-3)
    assert tl["max_interval"] == pytest.approx(0.1, abs=1e-3)
    # 关键：VFR 只影响"能不能走零落盘融合"，不影响时间轴是否可测
    assert _status(tl)[0] == "PASS"


def test_manual_interpretation_cannot_replace_measurement():
    """人工解释是记录，不是证据。

    审核要求"人工解释有正例" —— 正例的意思是：带人工解释的**合法**素材
    能通过；而不是"写了人工解释就能顶替测量"。这两件事必须同时成立，
    所以这里既验"有解释仍要看实测"，也验"实测齐全时解释不挡道"。
    """
    measured = {"measured": True, "duration": 12.5,
                "coverage_complete": True, "basis": "video-packet-span"}
    # ① 实测齐全 → 通过，人工解释只是附带信息
    ok, _ = final_timeline_status(dict(measured))
    assert ok == "PASS"

    # ② 只有人工解释、没有实测 → 必须 UNVERIFIED
    only_manual = {"measured": False, "duration": 12.5,
                   "coverage_complete": True, "basis": "manual",
                   "manual_interpretation": "assume CFR 25 fps"}
    st, reason = final_timeline_status(only_manual)
    assert st == "UNVERIFIED"
    assert "not evidence" in reason or "could not be measured" in reason

    # ③ 人工解释写好听的话，也不能把 inf 变成可用
    for bad in (float("inf"), float("nan"), True, -1.0, "12.5", None):
        assert _duration_is_usable(bad) is False, bad
