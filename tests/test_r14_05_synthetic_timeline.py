# -*- coding: utf-8 -*-
"""R14-05 / T06：用**真实 ffmpeg/ffprobe** 生成的合成素材验证时间轴判定。

**为什么要有这一层**

前面的时间轴用例全是纯函数（喂 dict）。审核 §6 明确要求：
"先调用实际 main 依赖替换/command-builder，再用**合成** CFR/VFR、
末帧、非零起点和原始音画偏移素材验证输出 PTS 及音轨"。

本文件只做**能快速跑**的那部分：生成小素材 → 调生产 `_query_all_pts` /
`_classify_timeline` → 断言判定。不跑模型，不跑全链。

**隔离**：素材写在 `R16_MEDIA_DIR`（默认 `E:/codex/_media_r16_*/tmp/synthetic`）。
没有该目录时退回 pytest 的 tmp_path（配合 `--basetemp` 放 E 盘）。
"""
import os
import shutil
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage3_ai_upscale as s3  # noqa: E402


def _media_root(tmp_path):
    env = os.environ.get("R16_MEDIA_DIR")
    if env and os.path.isdir(env):
        d = os.path.join(env, "synthetic")
        os.makedirs(d, exist_ok=True)
        return d
    return str(tmp_path)


def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None


pytestmark = pytest.mark.skipif(
    not _ffmpeg_available(), reason="ffmpeg 不在 PATH 上")


def _run(cmd, timeout=60):
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, shell=False)


def _make_cfr(path, rate=30, dur=2, size="160x120"):
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y",
              "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}:duration={dur}",
              "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
    assert r.returncode == 0, r.stderr[-400:]
    return str(path)


def _make_vfr(path, tmpdir, size="160x120"):
    """用 setpts 在第 60 帧后插一个 50ms 间隙 → 真实的变帧率（单调）。

    原来用 `-f concat -c copy` 拼 30fps + 24fps 两段，结果 PTS **非单调**
    （copy 拼接不同帧率会重叠），被分类器正确判成 unknown —— 那不是
    "分类器太严"，是我的素材不对。这里改成确定的间隙：
    间隔从 1/30 变成 1/30+0.05 一次，之后再回到 1/30。
    """
    base = _make_cfr(os.path.join(tmpdir, "vfrbase.mp4"), rate=30, dur=4,
                     size=size)
    # **必须** `-fps_mode vfr`：libx264 默认按 CFR 重新定时，
    # 会把 setpts 造出来的间隙抹平（实测：不加时全部间隔都是 0.0333）。
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", base,
              "-vf", "setpts=PTS+(gte(N\\,60))*0.05/TB",
              "-fps_mode", "vfr",
              "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)])
    assert r.returncode == 0, r.stderr[-400:]
    return str(path)


# ---------------------------------------------------------------- CFR

def test_real_cfr_is_classified_cfr(tmp_path):
    """真实 30fps 素材 → 覆盖完整 → cfr。"""
    d = _media_root(tmp_path)
    clip = _make_cfr(os.path.join(d, "cfr30.mp4"))

    pts = s3._query_all_pts(clip)
    n = len(pts)
    assert n >= 50, f"帧数太少，素材没生成好：{n}"

    # 覆盖完整时判 cfr
    assert s3._classify_timeline(pts, n) == "cfr"

    # 帧号必须是 0..n-1
    assert sorted(pts) == list(range(n))


def test_real_cfr_intervals_are_uniform(tmp_path):
    """真实 CFR 的帧间隔应接近 1/30。"""
    d = _media_root(tmp_path)
    clip = _make_cfr(os.path.join(d, "cfr30b.mp4"))
    pts = s3._query_all_pts(clip)
    ts = [pts[k] for k in sorted(pts)]
    deltas = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    assert deltas
    med = sorted(deltas)[len(deltas) // 2]
    assert abs(med - 1 / 30) < 0.002, f"帧间隔异常：{med}"


# ---------------------------------------------------------------- VFR

def test_real_vfr_is_classified_vfr(tmp_path):
    """真实变帧率素材 → vfr（不是 cfr，也不是 unknown）。"""
    d = _media_root(tmp_path)
    clip = _make_vfr(os.path.join(d, "vfr.mp4"), d)

    pts = s3._query_all_pts(clip)
    n = len(pts)
    assert n >= 50, f"帧数太少：{n}"

    got = s3._classify_timeline(pts, n)
    assert got == "vfr", f"应变帧率被误判为 {got}"

    # 帧间隔确实不均匀
    ts = [pts[k] for k in sorted(pts)]
    deltas = [round(b - a, 4) for a, b in zip(ts, ts[1:]) if b > a]
    assert len(set(deltas)) > 1, "间隔全相同，说明 VFR 没生成成功"


# ---------------------------------------------------------------- 覆盖度

def test_real_partial_coverage_is_unknown(tmp_path):
    """只取一半帧 → 覆盖不完整 → unknown（不能凭规则子集判 cfr）。"""
    d = _media_root(tmp_path)
    clip = _make_cfr(os.path.join(d, "cfr30c.mp4"))
    pts = s3._query_all_pts(clip)
    n = len(pts)
    half = {k: v for k, v in list(sorted(pts.items()))[::2]}
    assert s3._classify_timeline(half, n) == "unknown"


def test_real_expected_frames_mismatch_is_unknown(tmp_path):
    """给的应有帧数与实际不符 → unknown。"""
    d = _media_root(tmp_path)
    clip = _make_cfr(os.path.join(d, "cfr30d.mp4"))
    pts = s3._query_all_pts(clip)
    n = len(pts)
    assert s3._classify_timeline(pts, n + 10) == "unknown"
    assert s3._classify_timeline(pts, n) == "cfr"


# ---------------------------------------------------------------- 非零起点

def test_real_nonzero_start_keeps_cfr(tmp_path):
    """整体时间轴平移（非法 0 起点）不影响 CFR 判定。

    判定看的是**间隔**与**覆盖**，不是绝对起点。
    """
    d = _media_root(tmp_path)
    base = _make_cfr(os.path.join(d, "base.mp4"))
    shifted = os.path.join(d, "shifted.mp4")
    r = _run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", base,
              "-vf", "setpts=PTS+2/TB", "-c:v", "libx264", "-pix_fmt",
              "yuv420p", shifted])
    assert r.returncode == 0, r.stderr[-400:]

    pts = s3._query_all_pts(shifted)
    n = len(pts)
    assert n >= 50
    ts = [pts[k] for k in sorted(pts)]
    assert ts[0] > 1.0, f"起点没有平移：{ts[0]}"
    assert s3._classify_timeline(pts, n) == "cfr"


# ---------------------------------------------------------------- 探测器本身

def test_query_all_pts_does_not_fabricate(tmp_path):
    """不可解析的输入 → 返回空 map，**不得**造假时间戳。"""
    bogus = tmp_path / "notavideo.mp4"
    bogus.write_bytes(b"not a video")
    pts = s3._query_all_pts(str(bogus))
    assert pts == {}, pts
