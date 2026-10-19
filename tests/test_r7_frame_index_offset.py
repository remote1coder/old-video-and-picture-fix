"""60 帧进、59 帧出：根因是**帧号整体偏移一位**，不是 concat 的问题。

**表象误导了我很久**

一开始怀疑 ffmpeg 的 `concat demuxer`（末帧要不要写 duration、
要不要重复末帧），做了 2×2 对照，也确实发现默认 `-vsync` 会丢帧。
但改完 concat 之后 pipeline **仍然出 59 帧** —— 说明找错了地方。

**真正的原因**

```python
global_idx = win_start + int(match.group()) - 1     # ← 这个 -1
dst = Path(output_dir) / f"frame_{global_idx:08d}.png"
```

抽帧用 `frame_%08d.png`（ffmpeg 默认 `-start_number 0`），AI 子进程用
`os.path.basename(fp)` 原样保留文件名。所以源编号是 **0..59**。
再减 1 之后：

| 源 | 目标 |
|---|---|
| `frame_00000000.png`（第 1 帧） | `frame_-0000001.png` |
| `frame_00000001.png` | `frame_00000000.png` |
| ... | ... |
| `frame_00000059.png`（第 60 帧） | `frame_00000058.png` |

下游 `-i frame_%08d.png` 从 **0** 开始读，读到 58 就停 ——
**源视频的第一帧被整帧丢弃**，成品只剩 59 帧。

**为什么以前没发现**

`-count_frames` 才看得出来。看 `nb_read_packets` 或"时长差不多"
都会被骗过去（时长 2.026s 反而比源 2.002s 长，很像"帧数口径问题"）。
"""

import os
import re
import subprocess
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VSTAGES = os.path.join(ROOT, "video_restoration_pipeline", "stages")
if VSTAGES not in sys.path:
    sys.path.insert(0, VSTAGES)

import stage3_ai_upscale as s3  # noqa: E402


def test_no_minus_one_offset_in_index_math():
    """源码里不能再出现 `win_start + idx - 1`。"""
    src = open(s3.__file__, encoding="utf-8").read()
    assert "int(match.group()) - 1" not in src, (
        "帧号仍然在做 -1 偏移 —— 会把源视频第一帧挤成负数编号而丢失")
    assert "global_idx = win_start + int(match.group())" in src


def test_index_math_reproduces_zero_based_numbering():
    """把修复后的公式跑一遍：0..59 必须映射到 0..59。"""
    win_start = 0
    for idx in range(60):
        name = f"frame_{idx:08d}.png"
        m = re.search(r"\d+", name)
        assert m, name
        global_idx = win_start + int(m.group())      # 修复后的公式
        assert global_idx == idx, f"{name} -> frame_{global_idx:08d}.png"


def test_old_formula_would_produce_negative_name():
    """反例：旧的 -1 公式会把第一帧写成负数文件名。

    这条不是测"希望它错"，而是证明 `-1` 确实是错的 ——
    `frame_-0000001.png` 这种名字 ffmpeg 的 `%08d` 序列读不到。
    """
    win_start = 0
    m = re.search(r"\d+", "frame_00000000.png")
    old_idx = win_start + int(m.group()) - 1
    assert old_idx == -1
    assert f"frame_{old_idx:08d}.png" == "frame_-0000001.png"


# 说明：这里原本还想用 ffmpeg 实跑一条"负数编号不参与 %08d 序列"的
# 模拟测试，但那个测试装置本身不稳定（合成 PNG 序列时 ffmpeg 只读到 1 帧，
# 与待验证的假设无关）。删掉它，不拿一个不可靠的模拟当证据。
#
# 真正的证据是 pipeline 实跑：修复后 stage3 收集目录编号为 0..59，
# 成品 `-count_frames` = 60（源也是 60）。
