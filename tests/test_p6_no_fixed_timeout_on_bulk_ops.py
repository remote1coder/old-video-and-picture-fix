# -*- coding: utf-8 -*-
"""守卫：**处理整片/整段**的调用不许用固定总时长判超时。

## 为什么单独立一条

这个错在本项目里犯了至少 4 次（见 `R25_MY_ERRORS.md` 第 3 条）：

| 位置 | 固定值 | 后果 |
|---|---|---|
| `retime_to_source_pts` | `timeout=1800` | 前人已修，注释里记着教训 |
| stage stall | 300s 无进展 | 长片被误杀 5 次 |
| `_extract_frames_batch` | `timeout=1800` | 我**照抄了上面那条教训** |
| 整条音轨转码 | `timeout=300` | 长片音频必然超 |

根本错误是**把"跑多久"当成可以预先知道的量** —— 它取决于素材大小/码率/磁盘。
正确做法是"真实进展信号 + 无进展超时"（`file_growth_watch` /
`collect_outputs_while_waiting`）。

## 这条测试的边界（避免变成假守卫）

它**不**去判断"哪个调用算处理级"（做不到可靠）。只钉住**已经踩过坑的两处**：
一旦有人把它们改回固定超时，立刻报错。新加的处理级调用仍需要人工 review ——
这条测试只是防止**已修的问题复发**，不是全面的静态分析。
"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SM = os.path.join(ROOT, "shared_modules")


def _read(name):
    return io.open(os.path.join(SM, name), encoding="utf-8",
                   errors="replace").read()


def test_batch_frame_extraction_uses_growth_watch_not_fixed_timeout():
    src = _read("artifact_verify.py")
    # 抽帧必须走"输出增长"判活
    assert "collect_outputs_while_waiting" in src, \
        "批量抽帧没有接入增长判活"
    # 不许再出现 `_extract_frames_batch(..., timeout=` 这种固定时长
    assert not re.search(r"_extract_frames_batch\([^)]*timeout\s*=", src), \
        "_extract_frames_batch 又出现了固定超时参数"
    # 那个"从头解码"的低效函数不该回来
    assert "def _extract_frame(" not in src, \
        "低效的单帧抽帧函数又出现了（它会每次都从头解码）"


def test_audio_processing_uses_growth_watch_not_fixed_timeout():
    src = _read("audio_utils.py")
    # 整条音轨转码那一段必须用判活
    assert "collect_outputs_while_waiting" in src, \
        "音频处理没有接入增长判活"
    # 那个具体的 300 秒是给"整段音频转码"写死的，不许回来
    assert "subprocess.run(cmd, capture_output=True, timeout=300" not in src, \
        "整条音轨转码又回到固定 300 秒超时"
