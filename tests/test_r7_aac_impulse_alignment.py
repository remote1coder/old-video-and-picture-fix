"""AAC 同步：用**脉冲位置**核验，而不是数静音样本。

第六轮的 B10 结论是"低能量首尾 + 样本数补齐 ⇒ priming 1024 + 帧边界"。
任务书指出这个推论过头：

> 低能量首尾与填充解释相符，但单凭它既不能判定全部多余采样的来源，
> 也不能证明内容在播放时间轴的位置正确。

真正要回答的是：**解码出来的内容，在时间轴上是不是还在原来的位置**。
所以这里放已知时刻的短脉冲，解码后找到脉冲的峰值位置，与原始位置比对：

* 固定偏移 → priming 没被正确跳过（或容器没有 edit list）；
* 随时长增长的偏移 → 采样率/时基换算有累积误差；
* 尾部多出来的样本 → remainder padding（无害）。

只有第一种会破坏音画同步。
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

pytestmark = pytest.mark.skipif(
    not __import__("shutil").which("ffmpeg"), reason="需要 ffmpeg")


#: 脉冲时刻（秒）。刻意用非整秒、非周期，避免"恰好对齐"的假通过。
PULSES = [0.35, 1.10, 2.40, 3.05]
SR = 48000


def _make_impulse_wav(tmp_path):
    """生成 4 秒静音 + 4 个 10ms 短脉冲的 WAV。"""
    n = int(4.0 * SR)
    x = np.zeros(n, dtype=np.float32)
    for t in PULSES:
        i = int(t * SR)
        w = int(0.010 * SR)
        x[i:i + w] = 0.9 * np.hanning(w).astype(np.float32)
    path = str(tmp_path / "impulse.wav")
    # 手写 WAV，避免依赖 scipy
    import struct
    data = (x * 32767).astype("<i2").tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)
    return path


def _encode_aac(wav, out, bitrate="128k"):
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", wav, "-c:a", "aac",
         "-b:a", bitrate, "-y", out], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"AAC 编码失败：{r.stderr[:200]}")
    return out


def _decode_positions(path, tmp_path):
    """解码成 WAV 并返回每个脉冲的峰值时刻（秒）。"""
    out = str(tmp_path / "decoded.wav")
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
         "-c:a", "pcm_s16le", "-y", out], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"解码失败：{r.stderr[:200]}")
    import wave
    with wave.open(out, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    a = np.abs(a)
    if not a.size:
        return []
    # 阈值取峰值的 30%，在每段 0.5s 窗口内找峰值
    thr = a.max() * 0.3
    found = []
    win = int(0.5 * SR)
    i = 0
    while i < len(a):
        seg = a[i:i + win]
        if seg.size and seg.max() > thr:
            found.append((i + int(np.argmax(seg))) / SR)
            i += win
        else:
            i += win
    return found


def test_aac_impulse_positions_are_preserved(tmp_path):
    """核心：AAC 往返后，脉冲仍在原来的时刻（±20ms）。"""
    wav = _make_impulse_wav(tmp_path)
    m4a = _encode_aac(wav, str(tmp_path / "a.m4a"))
    got = _decode_positions(m4a, tmp_path)

    assert len(got) >= len(PULSES), (
        f"只找到 {len(got)} 个脉冲，期望 {len(PULSES)}：{got}")
    # 只比较前 len(PULSES) 个（尾部可能有编码残留）
    worst = 0.0
    pairs = []
    for want, actual in zip(PULSES, got[:len(PULSES)]):
        d = abs(actual - want)
        pairs.append(f"want={want:.3f}s got={actual:.3f}s d={d*1000:.1f}ms")
        worst = max(worst, d)
    assert worst <= 0.020, (
        "脉冲位置偏移超过 20ms —— 音画会不同步\n" + "\n".join(pairs))


def test_offset_does_not_grow_with_time(tmp_path):
    """偏移不能随时长增长（那是采样率/时基换算错误）。"""
    wav = _make_impulse_wav(tmp_path)
    m4a = _encode_aac(wav, str(tmp_path / "b.m4a"))
    got = _decode_positions(m4a, tmp_path)
    assert len(got) >= len(PULSES)

    deltas = [actual - want for want, actual in zip(PULSES, got[:len(PULSES)])]
    # 首尾两个脉冲的偏移差异就是"漂移"
    drift = abs(deltas[-1] - deltas[0])
    assert drift <= 0.020, (
        f"偏移随时长漂移了 {drift*1000:.1f}ms：{['%.1fms' % (d*1000) for d in deltas]}"
        " —— 这是时基换算问题，不是固定 priming")


def test_tail_padding_does_not_shift_content(tmp_path):
    """尾部多出样本（remainder padding）不应影响内容位置。"""
    wav = _make_impulse_wav(tmp_path)
    m4a = _encode_aac(wav, str(tmp_path / "c.m4a"))
    # 解码后的总长度可能略长于原始（padding），但脉冲位置已在上面的
    # 测试里核对过；这里只确认"更长"不会导致内容整体前/后移。
    got = _decode_positions(m4a, tmp_path)
    assert got[0] == pytest.approx(PULSES[0], abs=0.020), \
        "首个脉冲位置不对 —— 内容整体发生了平移"


def test_encoder_and_container_are_recorded(tmp_path):
    """记录实际编码器/容器信息（任务书要求），不是只报"成功"。"""
    wav = _make_impulse_wav(tmp_path)
    m4a = _encode_aac(wav, str(tmp_path / "d.m4a"))
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,sample_rate,time_base,"
         "initial_padding,duration_ts",
         "-of", "default=noprint_wrappers=1", m4a],
        capture_output=True, text=True)
    info = r.stdout
    assert "codec_name=aac" in info
    assert "sample_rate=48000" in info
    assert "time_base=" in info
