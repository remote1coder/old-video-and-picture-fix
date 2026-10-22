# -*- coding: utf-8 -*-
"""stage7 编码命令中 ffmpeg 选项**位置**的回归测试。

背景（2026-10-23）：追查「599 帧进 -> 580 帧出」的落差点，定位在
stage7 的最终编码。根因两处：

1) `-fflags` 是 **input** 选项，必须写在 `-i` 之前才生效。老注释
   （FIX-R7-S7-A4）声称它是输出侧选项并把 `-fflags +genpts` 移到
   输出路径之前；按 ffmpeg 文档，`-fflags` 只对紧随其后的输入文件
   生效，写在 `-i` 之后等于没写。
   → 本意「封装时重建一份干净的容器 PTS」从未真正执行。

2) 编码未显式指定 `-fps_mode`，走默认 `auto`。按文档：
       passthrough (0) 每帧按原时间戳直传；
       cfr (1)  **复制并丢弃**帧以凑出恒定帧率；
       vfr (2)  时间戳直传，但**丢弃时间戳重复的帧**；
       auto (-1) 按 muxer 能力在 cfr/vfr 间选（默认）。
   → 默认路径下 ffmpeg 会为维持单调 DTS 而静默丢帧。

（时间轴被摊薄的根因是测试素材用了 `-ss ... -c copy`，
 留下负起始 PTS；入口校验另案处理。）
"""
import io
import os
import re

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE7 = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                                 "stage7_encode.py")


@pytest.fixture(scope="module")
def src():
    with io.open(_STAGE7, encoding="utf-8", errors="replace") as f:
        return f.read()


def _input_line(text):
    """取 `video_cmd = common_prefix + [...]` 这一整段。"""
    m = re.search(r"video_cmd\s*=\s*common_prefix\s*\+\s*\[.*?\]",
                  text, re.DOTALL)
    assert m, "找不到 video_cmd 的输入侧组装语句"
    return m.group(0)


def test_genpts_is_before_input(src):
    """`-fflags +genpts` 必须在 `-i` 之前 —— 它是 input 选项。"""
    line = _input_line(src)
    assert '"-fflags"' in line, \
        "输入侧没有 -fflags，genpts 会被当成输出标志而失效"
    assert line.index('"-fflags"') < line.index('"-i"'), \
        "-fflags 写在了 -i 之后，不会生效"


def test_fps_mode_is_explicitly_passthrough(src):
    """输出侧必须显式 passthrough；默认 auto 可能复制/丢弃帧。"""
    assert '"-fps_mode", "passthrough"' in src
