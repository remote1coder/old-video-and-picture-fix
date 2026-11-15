# -*- coding: utf-8 -*-
"""`get_audio_info` 必须**按 key** 解析 ffprobe 输出，不能按行序。

## 那个 bug

```python
["ffprobe", ..., "-show_entries", "stream=sample_rate,channels,codec_name",
 "-of", "default=noprint_wrappers=1:nokey=1"]
sr = lines[0]; ch = lines[1]; codec = lines[2]
```

假设 ffprobe 按**请求顺序**输出。它实际按**内部字段序**——对 audio stream
是 `codec_name` 在前。实测 `-of csv=p=0` 输出 `aac,48000,2`，于是
`channels` 读成 48000、`codec` 读成 `"2"`。

日志里的 `Detected 48000ch (stereo)` 就是它；下游
`num_channels > 2` 成立 → 走进多声道分支 → ffmpeg 失败。

## 为什么这条测试不能只做字符串断言

字符串断言（"必须出现 `-of json`"）挡不住"把 json 结果又按行序取"这类
错误。这里直接**核对结构契约**：`channels` 必须是 2 位数量级、`sample_rate`
必须是 5 位数量级 —— 两者互换会立刻被数值范围发现。
"""
import os
import subprocess
import sys

import pytest

import conftest  # noqa: F401

import audio_utils


def _make_tone(tmp_path, channels=2, rate=48000):
    """生成一段极短音频作为被测对象。"""
    out = tmp_path / f"tone_{channels}_{rate}.m4a"
    cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y",
           "-f", "lavfi", "-i",
           f"anullsrc=channel_layout={'stereo' if channels == 2 else 'mono'}"
           f":sample_rate={rate}",
           "-t", "0.2", "-c:a", "aac", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    except Exception:
        pytest.skip("cannot synthesize test audio")
    return str(out)


@pytest.mark.parametrize("ch,rate", [(2, 48000), (2, 44100)])
def test_channels_and_sample_rate_are_not_swapped(tmp_path, ch, rate):
    src = _make_tone(tmp_path, ch, rate)
    info = audio_utils.get_audio_info(src)
    assert info["channels"] == ch, (
        f"channels 读成了 {info['channels']!r}（期望 {ch}）—— 疑似按行序解析，"
        f"把 sample_rate 当成了 channels")
    assert info["sample_rate"] == rate, (
        f"sample_rate 读成了 {info['sample_rate']!r}（期望 {rate}）")
    # 数值范围断言：互换会让 channels 变成 4~5 位
    assert 1 <= int(info["channels"]) <= 64, "channels 数量级异常"
    assert 8000 <= int(info["sample_rate"]) <= 768000, "sample_rate 数量级异常"


def test_codec_is_a_name_not_a_number(tmp_path):
    src = _make_tone(tmp_path)
    info = audio_utils.get_audio_info(src)
    assert info["codec"] and not str(info["codec"]).isdigit(), (
        f"codec 读成了 {info['codec']!r} —— 按行序解析时它会拿到 channels 的数值")
