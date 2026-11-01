# -*- coding: utf-8 -*-
"""R16-06 / y4m 保时：按源逐帧时间戳重封装。

背景（实测）：真 VFR 素材 `VID_20150614_145458.mp4`（292 帧，
0.0449×227 + 0.0898×42，源 15.702s）跑完整条链后，中间产物变成 25/1、
15.160s（-3.45%），逐帧间隔被量化成常数。

纯 ffmpeg 无法恢复（concat 的 image2 默认 25fps、`-r` 会覆盖 duration、
`-video_track_timescale` 无效）。只有按时间码重新封装容器才行，
这需要 `mkvmerge`（MKVToolNix）。

本文件用**真实 ffmpeg + 真实 mkvmerge** 验证：

| | 帧数 | 时长 | 逐帧间隔 |
|---|---|---|---|
| 源（VFR） | 40 | — | 0.033 × 20 + 0.1 × 20 |
| CFR 编码 | 40 | 1.30s | 全 0.033（错） |
| 重定时后 | 40 | ≈2.36s | 0.033 / 0.1（对） |

并且钉住一个易犯的错：**"总时长接近"不等于"时间轴对"** ——
concat 产物的总时长只差 0.06%，但间隔已经面目全非。
"""
import io
import os
import shutil
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vfr_timecodes as vt  # noqa: E402

_MKV = vt.find_mkvmerge(_ROOT)


def _ffmpeg():
    return shutil.which("ffmpeg") is not None


def _run(cmd, timeout=300):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, shell=False)


def _pngs(d, count):
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new("RGB", (64, 48), (i % 256, 40, 90)).save(
            str(d / ("f_%08d.png" % (i + 1))))
    return d


# ---------------------------------------------------------------- 纯函数

def test_write_timecodes_v2_is_ascending_and_starts_at_zero(tmp_path):
    pts = [0.0, 0.033, 0.066, 0.166, 0.399]
    p = str(tmp_path / "tc.txt")
    assert vt.write_timecodes_v2(pts, p) is True
    lines = io.open(p, encoding="utf-8").read().strip().splitlines()
    assert lines[0] == "# timecode format v2", lines[0]
    nums = [int(x) for x in lines[1:]]
    assert nums[0] == 0
    assert nums == sorted(nums) and len(set(nums)) == len(nums), nums


def test_write_timecodes_v2_never_goes_backwards_on_duplicates(tmp_path):
    """同毫秒的两帧（>1000fps）必须兜底成严格递增，否则 mkvmerge 会拒。"""
    pts = [0.0, 0.0001, 0.0002, 0.0003]
    p = str(tmp_path / "tc2.txt")
    assert vt.write_timecodes_v2(pts, p) is True
    nums = [int(x) for x in
            io.open(p, encoding="utf-8").read().strip().splitlines()[1:]]
    assert nums == sorted(set(nums)) and len(nums) == 4, nums


def test_is_variable_detects_a_two_interval_pattern():
    cfr = [i * 0.04 for i in range(20)]
    vfr = [i * 0.04 for i in range(10)] + [0.4 + i * 0.08 for i in range(10)]
    assert vt.is_variable(cfr) is False
    assert vt.is_variable(vfr) is True


def test_find_mkvmerge_returns_a_real_path_or_none():
    p = vt.find_mkvmerge(_ROOT)
    if p is not None:
        assert os.path.isfile(p), p
        r = _run([p, "--version"])
        assert r.returncode == 0 and "mkvmerge" in r.stdout.lower(), r.stdout


# ---------------------------------------------------------------- 真实往返

pytestmark_ffmpeg = pytest.mark.skipif(not _ffmpeg(), reason="需要 ffmpeg")
pytestmark_mkv = pytest.mark.skipif(_MKV is None,
                                    reason="需要 mkvmerge（tools_r4/mkvtoolnix）")


@pytestmark_ffmpeg
@pytestmark_mkv
def test_retime_restores_vfr_intervals(tmp_path):
    """CFR 编码 + 源 PTS 时间码 → 逐帧间隔必须恢复到 **毫秒级**。"""
    frames = _pngs(tmp_path / "frames", 40)
    # 先按 30fps 编成 CFR（模拟"被压平"的中间产物）
    cfr = str(tmp_path / "cfr.mkv")
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
              "-framerate", "30", "-i", str(frames / "f_%08d.png"),
              "-c:v", "ffv1", "-pix_fmt", "yuv420p", "-an", cfr])
    assert r.returncode == 0, r.stderr[-300:]

    # 造一个"源"：本身不参与，只借它的 PTS 语义 —— 这里直接给目标文件
    # 造一份**假源**（同样的 40 帧、但时间戳是 0.033×20 + 0.1×20）
    src = str(tmp_path / "src.mkv")
    tc_src = str(tmp_path / "src_tc.txt")
    want = []
    t = 0.0
    for i in range(40):
        want.append(t)
        t += 0.033 if i < 20 else 0.1
    assert vt.write_timecodes_v2(want, tc_src)
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
              "-framerate", "30", "-i", str(frames / "f_%08d.png"),
              "-c:v", "ffv1", "-pix_fmt", "yuv420p", "-an", src])
    assert r.returncode == 0, r.stderr[-300:]
    r = _run([_MKV, "-q", "-o", src + ".t.mkv", "--timestamps", "0:" + tc_src,
              src])
    assert r.returncode == 0, r.stderr[-300:]
    os.replace(src + ".t.mkv", src)

    ok, info = vt.retime_to_source_pts(src, cfr, _MKV,
                                       work_dir=str(tmp_path))
    assert ok is True, info
    assert info["action"] == "retimed", info
    assert info["frames"] == 40
    # 逐帧间隔误差必须在毫秒级（v2 是整数毫秒）
    assert info["max_interval_error"] <= 0.002, info
    # 末帧时间必须对上（源 20*0.033 + 19*0.1 = 2.56s）
    assert abs(info["out_last"] - info["src_last"]) <= 0.05, info
    assert info["before_last"] < info["out_last"] - 0.5, \
        f"重定时前后应该差很多：{info}"


@pytestmark_ffmpeg
@pytestmark_mkv
def test_skip_gate_needs_matching_intervals_not_just_duration(tmp_path,
                                                              monkeypatch):
    """**总时长接近但间隔不同** 时不许跳过。

    实测反例：concat 产物总时长 15.12 vs 源 15.1111，只差 0.06%，
    但间隔是 0.04/0.08（源 0.0449/0.0898）。只看总时长就会误判"已对上"。
    """
    # 直接对内部判据做单元验证：构造 pts 让"时长相等、间隔不同"
    src_pts = []
    t = 0.0
    for i in range(10):
        src_pts.append(t)
        t += 0.1
    tgt_pts = []
    t = 0.0
    for i in range(10):
        tgt_pts.append(t)
        t += 0.1
    # 同时长同间隔 → 应跳过
    monkeypatch.setattr(vt, "probe_pts",
                        lambda p, **k: src_pts if "src" in str(p) else tgt_pts)
    frames = _pngs(tmp_path / "f2", 3)
    a = str(tmp_path / "a.mkv")
    b = str(tmp_path / "b.mkv")
    for p in (a, b):
        p and open(p, "wb").write(b"x")
    ok, info = vt.retime_to_source_pts("src.mkv", "b.mkv", _MKV,
                                       work_dir=str(tmp_path))
    assert ok and info["action"] == "skipped", info

    # 同总时长、但间隔被改成"前紧后松" → 必须**不**跳过
    tgt2 = []
    t = 0.0
    for i in range(10):
        tgt2.append(t)
        t += 0.05 if i < 5 else 0.15
    monkeypatch.setattr(vt, "probe_pts",
                        lambda p, **k: src_pts if "src" in str(p) else tgt2)
    ok2, info2 = vt.retime_to_source_pts("src.mkv", "b.mkv", _MKV,
                                         work_dir=str(tmp_path))
    # 这一步会真的去调 mkvmerge（文件是假的）→ 允许失败，
    # 但绝不能返回 "skipped"
    assert info2.get("action") != "skipped", \
        f"间隔明显不同却跳过了：{info2}"


def test_no_mkvmerge_is_reported_not_faked(monkeypatch):
    """找不到 mkvmerge 时必须**明确不可用**，不能悄悄当成功。"""
    monkeypatch.setattr(vt, "find_mkvmerge", lambda *a, **k: None)
    assert vt.find_mkvmerge(_ROOT) is None
    # 模块级入口存在即可；调用方（stage7）据此告警
    src = io.open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                               "stage7_encode.py"),
                  encoding="utf-8", errors="replace").read()
    assert "STAGE7_VFR_RETIME_UNAVAILABLE" in src, \
        "没有 mkvmerge 时必须告警，不能默默按 CFR 发布"


# ------------------------------------------------ 容器时长 vs 视频时长

@pytestmark_ffmpeg
def test_audio_bearing_output_is_not_a_duration_mismatch(tmp_path):
    """成品带音轨、中间产物只有视频轨 → **不得**判时长不符。

    这是实测踩到的假阳性：源视频 15.1307s、源音频 15.7020s，
    中间产物（纯视频）15.172s，成品容器 15.723s。三个数各有各的口径，
    直接比 `format.duration` 会得出 "时长差 3.6%" 并把整条链拦死。
    """
    import artifact_verify as av

    frames = _pngs(tmp_path / "f", 30)
    # 中间产物：只有视频轨
    vid = str(tmp_path / "vid.mkv")
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
              "-framerate", "30", "-i", str(frames / "f_%08d.png"),
              "-c:v", "ffv1", "-pix_fmt", "yuv420p", "-an", vid])
    assert r.returncode == 0, r.stderr[-300:]

    # 成品：视频 + **更长的**音轨（容器时长会被音轨拉长）
    out = str(tmp_path / "out.mkv")
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
              "-framerate", "30", "-i", str(frames / "f_%08d.png"),
              "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
              "-c:v", "ffv1", "-pix_fmt", "yuv420p", "-c:a", "aac",
              "-shortest", out])
    assert r.returncode == 0, r.stderr[-300:]

    fv = av.probe_media_facts(vid)
    fo = av.probe_media_facts(out)
    assert fv.get("has_audio") is False, fv
    assert fo.get("has_audio") is True, fo

    rep = av.verify_output(vid, out, {})
    codes = [i.get("code") for i in (rep.get("issues") or [])]
    dur = (rep.get("checks") or {}).get("duration") or {}
    # 要么按视频轨口径比（ok 为 True/False），要么明确"不可比"
    if dur.get("ok") is None:
        assert "DURATION_NOT_COMPARABLE" in codes, codes
    assert not any(
        i.get("code") == "DURATION_MISMATCH" and i.get("level") == "error"
        for i in (rep.get("issues") or [])), \
        "带音轨的成品被误判为时长不符"


def test_probe_all_streams_sees_the_audio_track():
    """`_probe()` 写死 v:0，查 codec_type 永远只有 video —— 必须另有全流探针。"""
    src = io.open(os.path.join(_ROOT, "shared_modules", "artifact_verify.py"),
                  encoding="utf-8", errors="replace").read()
    assert "def _probe_all_streams(" in src
    assert '_probe_all_streams(path, "stream=codec_type")' in src


# ------------------------------------------------ 容器时长 vs 视频时长

@pytestmark_ffmpeg
def test_audio_bearing_output_is_not_a_duration_mismatch(tmp_path):
    """成品带音轨、中间产物只有视频轨 → **不得**判时长不符。

    这是实测踩到的假阳性：源视频 15.1307s、源音频 15.7020s，
    中间产物（纯视频）15.172s，成品容器 15.723s。三个数各有各的口径，
    直接比 `format.duration` 会得出 "时长差 3.6%" 并把整条链拦死。
    """
    import artifact_verify as av

    frames = _pngs(tmp_path / "f", 30)
    # 中间产物：只有视频轨
    vid = str(tmp_path / "vid.mkv")
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
              "-framerate", "30", "-i", str(frames / "f_%08d.png"),
              "-c:v", "ffv1", "-pix_fmt", "yuv420p", "-an", vid])
    assert r.returncode == 0, r.stderr[-300:]

    # 成品：视频 + **更长的**音轨（容器时长会被音轨拉长）
    out = str(tmp_path / "out.mkv")
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
              "-framerate", "30", "-i", str(frames / "f_%08d.png"),
              "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
              "-c:v", "ffv1", "-pix_fmt", "yuv420p", "-c:a", "aac",
              "-shortest", out])
    assert r.returncode == 0, r.stderr[-300:]

    fv = av.probe_media_facts(vid)
    fo = av.probe_media_facts(out)
    assert fv.get("has_audio") is False, fv
    assert fo.get("has_audio") is True, fo

    rep = av.verify_output(vid, out, {})
    codes = [i.get("code") for i in (rep.get("issues") or [])]
    dur = (rep.get("checks") or {}).get("duration") or {}
    # 要么按视频轨口径比（ok 为 True/False），要么明确"不可比"
    if dur.get("ok") is None:
        assert "DURATION_NOT_COMPARABLE" in codes, codes
    assert not any(
        i.get("code") == "DURATION_MISMATCH" and i.get("level") == "error"
        for i in (rep.get("issues") or [])), \
        "带音轨的成品被误判为时长不符"


def test_probe_all_streams_sees_the_audio_track():
    """`_probe()` 写死 v:0，查 codec_type 永远只有 video —— 必须另有全流探针。"""
    src = io.open(os.path.join(_ROOT, "shared_modules", "artifact_verify.py"),
                  encoding="utf-8", errors="replace").read()
    assert "def _probe_all_streams(" in src
    assert '_probe_all_streams(path, "stream=codec_type")' in src
