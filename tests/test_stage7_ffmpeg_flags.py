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

## 第二轮（S6→S7 融合）之后为什么还要改这个测试

融合把输入从「一个文件」换成「vspipe 的 y4m 管道」，于是出现了**两条**
组装语句。原来的正则 `video_cmd\\s*=\\s*common_prefix...` 只取第一处，
改完后第一处正是管道那条 —— 于是"`-fflags` 在 `-i` 之前"这条**真实存在
的保证**被误报成失败。

判据改成按**语义**取：

* 读文件的那条（含 `"-i", input_source`）→ 必须有 `-fflags +genpts`；
* 读管道的那条（含 `"pipe:0"`）→ 必须有 `-f yuv4mpegpipe`，且
  **不该**有 `-fflags`（y4m 没有容器时间戳，重建 PTS 无意义）；
* 两条都必须只有**一个** `-i`（多写一个会把后面的选项当成下一个输入）。
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


def _input_lines(text):
    """所有 `video_cmd = common_prefix + [...]` 的输入侧组装语句。"""
    return re.findall(r"video_cmd\s*=\s*common_prefix\s*\+\s*\[.*?\]",
                      text, re.DOTALL)


def _file_branch(text):
    for line in _input_lines(text):
        if '"-i", input_source' in line.replace("'", '"'):
            return line
    assert False, "找不到读文件那条输入侧组装语句"


def _pipe_branch(text):
    for line in _input_lines(text):
        if "pipe:0" in line:
            return line
    assert False, "找不到融合（管道）那条输入侧组装语句"


def test_exactly_two_input_branches(src):
    """只允许「读文件」与「读管道」两条；多出来的一条必须显式审查。"""
    lines = _input_lines(src)
    assert len(lines) == 2, (
        "输入侧组装语句数量变了（%d 条）。新增的输入方式必须单独审查 "
        "ffmpeg 选项位置，不能悄悄加：%r" % (len(lines), lines))


def test_genpts_is_before_input(src):
    """（经典路线）`-fflags +genpts` 必须在 `-i` 之前 —— 它是 input 选项。"""
    line = _file_branch(src)
    assert '"-fflags"' in line, \
        "输入侧没有 -fflags，genpts 会被当成输出标志而失效"
    assert line.index('"-fflags"') < line.index('"-i"'), \
        "-fflags 写在了 -i 之后，不会生效"


def test_fused_branch_reads_y4m_from_stdin(src):
    """（融合路线）必须是 yuv4mpegpipe + stdin，且 `-f` 在 `-i` 之前。"""
    line = _pipe_branch(src)
    assert '"-f", "yuv4mpegpipe"' in line, line
    assert line.index('"-f"') < line.index('"-i"'), \
        "-f 是输入侧格式声明，写在 -i 之后不生效"
    assert "-fflags" not in line, (
        "融合路线不该加 -fflags：y4m 没有容器时间戳，"
        "genpts 无从重建，加了只是让两条路线的参数看起来一致（假一致）")


@pytest.mark.parametrize("which", ["file", "pipe"])
def test_each_branch_has_exactly_one_input(src, which):
    line = _file_branch(src) if which == "file" else _pipe_branch(src)
    assert line.count('"-i"') == 1, f"{which} 分支的 -i 数量不是 1：{line}"


def test_fps_mode_is_explicitly_passthrough(src):
    """输出侧必须显式 passthrough；默认 auto 可能复制/丢弃帧。"""
    assert '"-fps_mode", "passthrough"' in src
