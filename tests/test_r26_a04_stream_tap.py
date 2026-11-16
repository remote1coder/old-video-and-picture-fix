# -*- coding: utf-8 -*-
"""`StreamTap`：在真实数据流上观测，且**任何不完整都不能算成功**。

对应 R26-A04。复核的四个反例：
  ① 同帧数、摘要完全不同 → 旧实现只比帧数，绑定成功；
  ② 任意一端 0 帧 → 也通过；
  ③ 消费侧车缺失 → 退回额外重放，再次通过；
  ④ 短帧/截断 → 不报。

本文件钉住 tap 自己的行为（①的"内容敏感"、②③④的"不完整必须拒"）。
"两遍绑定改用它" 是下一步，见 `stage3._two_pass_check`。
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared_modules"))

from stream_tap import StreamTap, parse_y4m_header  # noqa: E402


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


pytestmark = pytest.mark.skipif(not _have_ffmpeg(),
                                reason="ffmpeg not available")


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    """64x48 yuv420p 10fps 2s 的合成片。"""
    p = tmp_path_factory.mktemp("tap") / "src.mp4"
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=64x48:rate=10:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)],
        capture_output=True, timeout=120)
    assert r.returncode == 0, r.stderr[:300]
    return str(p)


def _prod(src, *extra):
    return ["ffmpeg", "-v", "error", "-i", src, *extra,
            "-f", "yuv4mpegpipe", "-"]


def _cons(*extra):
    return ["ffmpeg", "-v", "error", "-f", "yuv4mpegpipe", "-i", "-",
            *extra, "-f", "null", "-"]


def test_header_parsing_handles_mpeg2_chroma_suffix():
    """`C420mpeg2` 不能被当成"未知色度"。

    这是实测踩到的 bug：`"420mpeg2".split("p")[0]` 得到 `"420m"`，
    匹配不上任何分支 → 退回 RGB 口径（每像素 3 字节）→ `frame_bytes`
    算成 1.5 倍 → 读帧越界 → 下一帧的 `FRAME` 标记错位。
    """
    m = parse_y4m_header(b"YUV4MPEG2 W64 H48 F10:1 Ip A1:1 C420mpeg2\n")
    assert m["ok"] is True, m
    assert m["width"] == 64 and m["height"] == 48
    assert m["frame_bytes"] == 64 * 48 * 3 // 2, m

    m10 = parse_y4m_header(b"YUV4MPEG2 W64 H48 F10:1 Ip C420p10\n")
    assert m10["ok"] is True and m10["frame_bytes"] == 64 * 48 * 3 // 2 * 2


def test_normal_passthrough_is_ok_and_digest_is_stable(src):
    tap = StreamTap()
    a = tap.run(_prod(src), _cons())
    b = tap.run(_prod(src), _cons())
    assert a["ok"] is True, a
    assert a["frames"] == 20, a["frames"]
    assert a["producer_rc"] == 0 and a["consumer_rc"] == 0
    assert a["digest"] == b["digest"], "同一源两次摘要必须相同"


def test_zero_frames_is_rejected(src):
    """0 帧 → 不是"有效的一代"，不能 ok。"""
    r = StreamTap().run(_prod(src, "-t", "0"), _cons())
    assert r["ok"] is False
    assert r["frames"] == 0


def test_short_frame_is_rejected(src, tmp_path):
    """截断的数据流：`short_frame=True` 且 ok=False（不补零、不当正常结束）。"""
    y4m = tmp_path / "full.y4m"
    r0 = subprocess.run(_prod(src) + [""], capture_output=True)
    # 用 ffmpeg 直接写 y4m 文件，再截掉尾部
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", src, "-f", "yuv4mpegpipe",
         str(y4m)], capture_output=True, timeout=120)
    data = y4m.read_bytes()
    cut = data[: len(data) - 200]        # 去掉最后一行的一部分
    (tmp_path / "cut.y4m").write_bytes(cut)
    # producer 用 cat 吐截断内容
    r = StreamTap().run(["cmd", "/c", "type", str(tmp_path / "cut.y4m")],
                        _cons())
    assert r["ok"] is False, r
    assert r["short_frame"] is True, r


def test_digest_changes_when_content_changes(src, tmp_path):
    """**内容敏感**：换一个像素，摘要必须变（旧实现只比帧数抓不到）。"""
    alt = tmp_path / "alt.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", src,
         "-vf", "drawbox=x=0:y=0:w=8:h=8:color=red@1:t=fill",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(alt)],
        capture_output=True, timeout=120)
    tap = StreamTap()
    a = tap.run(_prod(src), _cons())
    b = tap.run(_prod(str(alt)), _cons())
    assert a["frames"] == b["frames"], "帧数应相同（这正是旧实现唯一比的东西）"
    assert a["digest"] != b["digest"], "同帧数不同内容必须得到不同摘要"
