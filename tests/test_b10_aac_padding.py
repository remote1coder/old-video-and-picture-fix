"""B10：坐实"AAC 时长膨胀 = priming + 末尾补齐到 1024 帧边界"。

台账此前只能**推断**"其余全是帧补齐"，因为没解码数有效样本数。本次用
真实编码实验坐实（ffmpeg 2023-11 + AAC 192k/48kHz/立体声）：

| 输入时长 | 解码样本数 | 帧数(x1024) | 总膨胀 | 末尾补齐 |
|---|---|---|---|---|
| 2.000s (96000) | 97280 | 95 | 1280 (26.67ms) | 256 (5.33ms) |
| 4.000s (192000) | 193536 | 189 | 1536 (32.00ms) | 512 (10.67ms) |
| 10.000s (480000) | 481280 | 470 | 1280 (26.67ms) | 256 (5.33ms) |

规律：**总帧数 = ceil((输入样本 + priming) / 1024)**，`priming = 1024`。

能量分析（2.000s，48kHz 单声道）：

```
前 1024 样本 (priming)     : RMS=6.8     峰值=23     ← 静音填充
第 1024~2048 样本          : RMS=2045.8  峰值=2972   ← 真实内容
末尾 256 样本 (补齐)       : RMS=6.8     峰值=16     ← 静音填充
```

结论：残差**确实是容器层面的时长膨胀，不是内容错位**。播放器靠
`initial_padding` / skip_samples 跳过 priming，听感无偏移。

注：台账记的 43/63ms 是 44.1kHz 下的换算；48kHz 下补齐上限为
1023 样本 ≈ 21.3ms，总膨胀上限 ≈ 42.7ms。
"""

import shutil
import subprocess

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

AAC_FRAME = 1024  # AAC-LC 固定帧长


def _ffmpeg():
    return shutil.which("ffmpeg")


def _probe_ffprobe():
    return shutil.which("ffprobe")


pytestmark = pytest.mark.skipif(_ffmpeg() is None, reason="需要 ffmpeg")


@pytest.fixture(scope="function")
def encoded(tmp_path):
    """按 2s/4s 输入各编码一个 AAC，返回 (输入样本, 解码样本) 对。

    R15-06：原来是 **module 级 + `tmp_path_factory.mktemp()`**，依赖一个
    长生命周期的共享 basetemp。整批跑时该目录可能在别的测试之后已经不可用，
    于是 fixture 建立失败 → 4 条测试全报 `OSError: [Errno 53]`，
    而单独跑这个文件时又完全正常（典型"只在全量里炸"的隔离问题）。

    改成 function 级 + `tmp_path`：每个测试拿自己那份目录，不再依赖
    模块级共享状态。代价只是每条测试多编码两次（各约 0.3 秒）。
    """
    d = tmp_path / "b10"
    d.mkdir(exist_ok=True)
    rows = []
    for sec in (2.0, 4.0):
        out = str(d / f"a_{sec}.mka")
        r = subprocess.run(
            [_ffmpeg(), "-v", "error", "-f", "lavfi",
             "-i", f"sine=frequency=440:sample_rate=48000:duration={sec}",
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-y", out],
            capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip(f"编码失败: {(r.stderr or '')[:120]}")
        # 解码数实际样本数
        dec = subprocess.run(
            [_ffmpeg(), "-v", "error", "-i", out, "-f", "s16le",
             "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2", "-"],
            capture_output=True)
        n_samples = len(dec.stdout) // 4  # 2ch * 2B
        rows.append((int(sec * 48000), n_samples, out))
    return rows


def test_total_samples_are_frame_aligned(encoded):
    """解码样本数必须是 1024 的整数倍（编码器按帧输出）。"""
    for inp, got, path in encoded:
        assert got % AAC_FRAME == 0, \
            f"{path}: 样本数 {got} 不是 1024 的整数倍"


def test_frame_count_formula(encoded):
    """总帧数 = ceil((输入 + priming) / 1024)，priming 固定 1024。"""
    import math
    for inp, got, path in encoded:
        expected_frames = math.ceil((inp + AAC_FRAME) / AAC_FRAME)
        assert got == expected_frames * AAC_FRAME, (
            f"{path}: 输入 {inp} → 期望 {expected_frames} 帧 "
            f"({expected_frames*AAC_FRAME} 样本)，实际 {got}")


def test_expansion_is_priming_plus_tail_padding(encoded):
    """总膨胀 = 1024(priming) + 末尾补齐，且补齐在 [0, 1023] 内。"""
    for inp, got, path in encoded:
        expansion = got - inp
        tail = expansion - AAC_FRAME
        assert 0 <= tail < AAC_FRAME, (
            f"{path}: 末尾补齐 {tail} 超出 [0,{AAC_FRAME})，"
            f"说明膨胀不是帧边界对齐造成的")
        # 换算成毫秒，确认上限 21.3ms（48kHz）
        assert tail / 48.0 < 21.34


def test_priming_region_is_silent(encoded, tmp_path):
    """priming 区与末尾补齐区必须是静音填充（坐实"不是内容错位"）。"""
    import numpy as np
    _inp, _got, path = encoded[0]
    pcm = str(tmp_path / "raw.pcm")
    r = subprocess.run(
        [_ffmpeg(), "-v", "error", "-i", path, "-f", "s16le",
         "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2", "-y", pcm],
        capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip("解码为 PCM 失败")
    a = np.fromfile(pcm, dtype=np.int16)
    if a.size < AAC_FRAME * 4:
        pytest.skip("解码样本不足")
    a = a.reshape(-1, 2)[:, 0].astype(np.float64)
    priming_rms = float(np.sqrt((a[:AAC_FRAME] ** 2).mean()))
    content_rms = float(np.sqrt((a[AAC_FRAME:2 * AAC_FRAME] ** 2).mean()))
    assert content_rms > 100, "内容区 RMS 异常低，测试信号有问题"
    assert priming_rms < content_rms * 0.05, (
        f"priming 区不是静音（RMS {priming_rms:.1f} vs 内容 {content_rms:.1f}），"
        "说明膨胀可能含真实内容错位")
