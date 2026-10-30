# -*- coding: utf-8 -*-
"""R16-06：stage5 的 PNG 重编码不许把 VFR 压成常数间隔。

**实测背景**（真素材 `VID_20150614_145458.mp4`，292 帧：0.0449×227 + 0.0898×42）

| 写法 | 帧数 | 时长 | 逐帧间隔 |
|---|---|---|---|
| 源 | 292 | 15.702s | 0.0449 / 0.0898 |
| 旧：单个 `-framerate 89/4` | 292 | **13.124s（-16.4%）** | 全 0.0449 |
| 新：concat + 逐帧 duration + passthrough | 292 | **15.12s** | 0.04 / 0.08（量化） |

结论（不夸大）：
* 旧写法**连总时长都保不住**；新写法保住了**帧数与总时长**。
* 逐帧间隔仍被 muxer 量化到 1/25 s —— 纯 ffmpeg 无法消除（concat 的
  image2 默认 25fps；`-r` 会覆盖 duration、`-video_track_timescale` 无效）。
* 因此 stage5 现在会**量出残差**并记 `STAGE5_VFR_INTERVAL_QUANTIZED`，
  不允许下游据此声称"精确保时"。
"""
import io
import os
import shutil
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_S5 = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                   "stage5_ai_face.py")


def _ffmpeg():
    return shutil.which("ffmpeg") is not None


def _run(cmd, timeout=300):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          shell=False)


def _make_pngs(d, count):
    """用 PIL 造 count 张 64x48 的 PNG（内容各不相同，便于按帧核对）。"""
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = Image.new("RGB", (64, 48), (i % 256, (i * 7) % 256, 40))
        img.save(str(d / ("f_%08d.png" % (i + 1))))
    return d


def _make_pngs(d, count):
    """用 PIL 造 count 张 64x48 的 PNG（内容各不相同，便于按帧核对）。"""
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = Image.new("RGB", (64, 48), (i % 256, (i * 7) % 256, 40))
        img.save(str(d / ("f_%08d.png" % (i + 1))))
    return d


def _pts(path):
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "frame=pts_time", "-of", "csv=p=0", path])
    return [float(x.strip().rstrip(",")) for x in r.stdout.splitlines()
            if x.strip()]


# ---------------------------------------------------------------- 源码守卫

def test_stage5_reencode_uses_per_frame_durations():
    """重编码必须走 concat + 逐帧 duration + passthrough，而不是单个帧率。"""
    src = io.open(_S5, encoding="utf-8", errors="replace").read()
    assert "concat_vfr.txt" in src, "必须写逐帧时长的 concat 列表"
    assert '"duration %.6f"' in src
    assert '"-fps_mode", "passthrough"' in src, "必须 passthrough 防止再定时"
    assert "STAGE5_VFR_PRESERVED" in src
    # 保留回退路径（PTS 与帧数不符时），但必须告警
    assert "STAGE5_VFR_FALLBACK" in src


def test_stage5_measures_and_reports_interval_residual():
    """必须量出间隔残差并如实记录，不能只说"保住了"。"""
    src = io.open(_S5, encoding="utf-8", errors="replace").read()
    assert "def _measure_interval_error(" in src
    assert "STAGE5_VFR_INTERVAL_QUANTIZED" in src, \
        "量化超出容差时必须告警"
    assert "STAGE5_VFR_INTERVALS_OK" in src, "在容差内也要留痕"
    assert "vfr_interval_error" in src, "STAGE5_DONE 要带上残差"


# ---------------------------------------------------------------- 真实往返

@pytest.mark.skipif(not _ffmpeg(), reason="ffmpeg 不在 PATH 上")
def test_concat_passthrough_preserves_frames_and_total_duration(tmp_path):
    """真实往返：帧数与总时长必须保住（这是新写法达到的保证）。"""
    # 造 60 帧、两种间隔（0.033 / 0.066）的 VFR 序列
    frames = _make_pngs(tmp_path / "frames", 60)

    durs = [0.033 if i % 2 == 0 else 0.066 for i in range(60)]
    concat = str(tmp_path / "c.txt")
    with open(concat, "w", encoding="utf-8") as f:
        for i in range(60):
            f.write("file '%s'\n" % (frames / f"f_{i + 1:08d}.png").as_posix())
            f.write("duration %.6f\n" % durs[i])
    expected = sum(durs)

    out = str(tmp_path / "out.mkv")
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
              "-f", "concat", "-safe", "0", "-i", concat,
              "-fps_mode", "passthrough", "-c:v", "ffv1",
              "-pix_fmt", "yuv420p16le", "-an", out])
    assert r.returncode == 0, r.stderr[-300:]

    ts = _pts(out)
    assert len(ts) == 60, f"帧数必须保住，实际 {len(ts)}"
    # 总时长在 5% 以内（量化误差会累积，但不应像旧写法那样掉 16%）
    assert abs(ts[-1] - (expected - durs[-1])) / expected < 0.05, \
        f"总时长偏差过大：末 PTS={ts[-1]:.4f}，期望≈{expected - durs[-1]:.4f}"


@pytest.mark.skipif(not _ffmpeg(), reason="ffmpeg 不在 PATH 上")
def test_single_framerate_loses_duration_but_concat_does_not(tmp_path):
    """对照：单个 `-framerate` 会丢时长，concat 不会。

    这条把"为什么必须换成 concat"钉住，防止回退。
    """
    frames = _make_pngs(tmp_path / "frames", 40)

    # 一半 0.033、一半 0.1 → 真实平均远低于 30fps
    durs = [0.033 if i < 20 else 0.1 for i in range(40)]
    concat = str(tmp_path / "c.txt")
    with open(concat, "w", encoding="utf-8") as f:
        for i in range(40):
            f.write("file '%s'\n" % (frames / f"f_{i + 1:08d}.png").as_posix())
            f.write("duration %.6f\n" % durs[i])
    expected = sum(durs)

    old = str(tmp_path / "old.mkv")
    _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
          "-framerate", "30", "-i", str(frames / "f_%08d.png"),
          "-c:v", "ffv1", "-pix_fmt", "yuv420p16le", "-an", old])
    new = str(tmp_path / "new.mkv")
    _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
          "-f", "concat", "-safe", "0", "-i", concat,
          "-fps_mode", "passthrough", "-c:v", "ffv1",
          "-pix_fmt", "yuv420p16le", "-an", new])

    old_last = _pts(old)[-1]
    new_last = _pts(new)[-1]
    # 旧写法按 30fps 排 → 40 帧只有 1.3s 左右，远小于真实 2.66s
    assert abs(old_last - expected) / expected > 0.3, \
        f"旧写法居然没丢时长（{old_last:.3f} vs {expected:.3f}），对照失效"
    assert abs(new_last - expected) / expected < 0.05, \
        f"新写法总时长偏差过大（{new_last:.3f} vs {expected:.3f}）"
