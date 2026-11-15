# -*- coding: utf-8 -*-
"""`process_audio` 端到端必须跑通（它曾因两个 bug 恒返回 False）。

## Bug 1：ffprobe 按行序解析 → channels 读成 48000

见 `test_p6_audio_info_parse.py`。

## Bug 2：filtergraph 里同一个标签被消费多次

逐声道去噪那一段原来写：

```
[aout]pan=mono|c0=c0[dch0]
[aout]pan=mono|c0=c1[dch1]      ← 同一个 [aout] 又消费一次
```

ffmpeg 不允许（一个标签只能被消费一次，要么先 `split/asplit`）：

```
Stream specifier 'aout' in filtergraph description ... matches no streams.
Error initializing complex filters: Invalid argument
```

这条用**端到端**测：跑一遍 `process_audio`，要求返回 True 且输出非空。
两个 bug 任何一个复发都会让它返回 False。

## 为什么不做"只查字符串"的测试

字符串断言（"必须有 asplit"）挡不住"asplit 加了但标签名写错"。
端到端跑一遍 8 秒音频只要 1 秒，代价可接受。
"""
import os
import subprocess

import pytest

import conftest  # noqa: F401

import audio_utils


def _make_audio(tmp_path, channels=2):
    src = tmp_path / "in.m4a"
    layout = "stereo" if channels == 2 else "mono"
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-y",
             "-f", "lavfi", "-i",
             f"anoisesrc=d=1.0:c=pink:r=48000:a=0.05,"
             f"aformat=channel_layouts={layout}",
             "-c:a", "aac", str(src)],
            check=True, capture_output=True, timeout=60)
    except Exception:
        pytest.skip("cannot synthesize test audio")
    return str(src)


@pytest.mark.parametrize("ch", [2])
def test_process_audio_returns_true_and_writes_output(tmp_path, ch):
    src = _make_audio(tmp_path, ch)
    out = str(tmp_path / "out.m4a")
    cfg = {
        "audio": {
            "enabled": True, "denoise": "afftdn", "hum_freq": None,
            "hum_harmonics": 3, "hum_bandwidth": 2, "hum_width_type": "h",
            "hum_auto_detect_seconds": 1, "hum_energy_threshold": 0.005,
            "hum_dual": False, "codec": "aac", "bitrate": "128k",
            "sample_rate": "auto", "multichannel_preserve": True,
        }
    }
    ok = audio_utils.process_audio(src, out, cfg, None)
    assert ok is True, (
        "process_audio 返回 False —— 两个已知 bug 之一复发："
        "① ffprobe 按行序解析（channels 会变 48000）；"
        "② filtergraph 标签被重复消费（缺 asplit）")
    assert os.path.exists(out) and os.path.getsize(out) > 1000, \
        "process_audio 声称成功但输出为空"
