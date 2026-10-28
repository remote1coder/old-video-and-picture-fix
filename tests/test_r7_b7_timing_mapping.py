"""B7（第七轮）：时间映射必须"有据可依"，不能用 atempo 掩盖损坏。

**旧逻辑**

```python
if abs(original_duration - processed_duration) * 1000 > 300:
    tempo_ratio = original_duration / processed_duration   # 无条件拉音频
```

时长变化至少有四种原因：

| 原因 | 应该怎么做 |
|---|---|
| (a) 明确要求的变速 | 按映射做 tempo |
| (b) 去隔行/抽帧改变帧率但内容完整 | 时长应不变；变了就是异常 |
| (c) 丢帧 / 截断 | **报警，不能拉音频** |
| (d) 错误的 PTS / 时基 | **报警，不能拉音频** |

旧实现把 (c)(d) 也当成 (a)，用一个 atempo 把音频拉到和损坏视频一样长 ——
表面"同步正常"，实际把缺陷藏了起来。任务书明确禁止。

**新规则**

* 上游显式声明 `expected_speed` / `expected_tempo` 才做 tempo，
  且声明值要与实测比例互相印证（差 >2% 视为声明不可信）；
* 无声明但时长变了 → 记 `STAGE7_TIMING_UNEXPLAINED`，**不改音频**，
  产物 metadata 标 `timing_state=unexplained`；
* 时长信息缺失 → `timing_state=unverified`，不宣称同步已验证。
"""

import json
import os
import subprocess
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VSTAGES = os.path.join(ROOT, "video_restoration_pipeline", "stages")
if VSTAGES not in sys.path:
    sys.path.insert(0, VSTAGES)

import stage7_encode as s7  # noqa: E402


def test_source_no_longer_uses_bare_duration_diff():
    """静态守卫：不能再出现「时长差 > 300ms 就 atempo」的无条件逻辑。"""
    src = open(s7.__file__, encoding="utf-8").read()
    assert "STAGE7_TIMING_UNEXPLAINED" in src
    assert "timing_unexplained" in src
    assert "STAGE7_TIMING_DECLARATION_MISMATCH" in src
    # 旧的直接赋值必须是过去式
    old = ("        if diff_ms > 300:\n"
           "            tempo_ratio = original_duration / processed_duration\n")
    assert old not in src, "仍在无条件按时长差推导 tempo"


def test_declared_mapping_is_required_for_tempo():
    """只有**实测且显式声明**时才改音频 tempo。

    原来是源码结构断言（依赖 `_declared_tempo` 这个局部变量名）；
    R13-03 把决策逻辑抽成 `_decide_timing()` 纯函数之后，名不副实了。
    改成直接调用生产函数验证行为 —— 更强，也不再绑实现细节。
    """
    # 未声明 + 时长变了 → 不许改音频
    assert s7._decide_timing(10.0, 7.0, "packet-span", None, None) ==         (1.0, "unexplained")
    # 声明且实测吻合 → 允许
    ratio, state = s7._decide_timing(10.0, 5.0, "packet-span", 2.0, None)
    assert state == "adjusted" and abs(ratio - 2.0) < 1e-9
    # 声明吻合但时间轴是**估算** → 不许改音频（R13-03）
    assert s7._decide_timing(10.0, 5.0, "frame-ratio (estimated)",
                             2.0, None) == (1.0, "unverified")


def test_timing_state_is_written_into_metadata():
    """判定结果要跟着文件走，不能只躺在日志里。"""
    src = open(s7.__file__, encoding="utf-8").read()
    assert '"timing_state={_timing_state}"' in src


def test_three_states_are_distinguishable():
    """verified / adjusted / unverified / unexplained 四种状态都要有落点。"""
    src = open(s7.__file__, encoding="utf-8").read()
    for state in ("verified", "adjusted", "unverified", "unexplained"):
        assert f'"{state}"' in src, f"缺少状态 {state}"


def test_real_stage7_unexplained_duration_change_keeps_audio(tmp_path):
    """端到端：视频被判短（模拟截断），音频**不得**被 atempo 拉伸。

    构造：源 2 秒带音轨；step6 替身只有 1 秒（模拟截断）。
    期望 rc=0，但日志必须出现 STAGE7_TIMING_UNEXPLAINED，
    且不得出现 STAGE7_SCENE2（那是"做了 tempo"的标志）。
    """
    src = tmp_path / "clip.mp4"
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=s=160x120:d=2:r=25",
         "-f", "lavfi", "-i", "sine=f=440:d=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", "-y", str(src)], capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"合成失败：{r.stderr[:200]}")

    temp = tmp_path / "temp"
    logs = tmp_path / "logs"
    out = tmp_path / "out"
    for d in (temp, logs, out):
        d.mkdir()

    # 截断版"上游产物"：只有 1 秒
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-t", "1",
         "-c", "copy", "-y", str(temp / "jobB7_clip_step6_grain.mkv")],
        capture_output=True)
    # 音轨（stage0 形态）
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-vn", "-c:a", "copy",
         "-y", str(temp / "jobB7_clip.mka")], capture_output=True)

    params = json.dumps({
        "temp_dir": str(temp), "log_dir": str(logs), "output_dir": str(out),
        "duration": 2.0,          # 原始素材 2 秒
    })
    rc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "video_restoration_pipeline",
                                      "stages", "stage7_encode.py"),
         "--input", str(src), "--job-id", "jobB7", "--params", params],
        capture_output=True, text=True, cwd=str(tmp_path))

    if rc.returncode == 80:
        pytest.skip("依赖预检未通过")

    log_text = ""
    for f in logs.glob("*.log"):
        log_text += f.read_text(encoding="utf-8", errors="replace")

    # rc=86 是 `_validate_av_sync` 的正确拦截（音视频差 1s > 500ms），
    # 不是"失败"——重点是**没有用 atempo 把差异抹平**。
    assert rc.returncode == 86, (
        f"期望同步校验拦下（86），实际 {rc.returncode}\n{rc.stderr[-400:]}")
    assert "STAGE7_TIMING_UNEXPLAINED" in log_text, \
        "时长变化未被标记为 unexplained —— 可能仍在静默拉音频"
    assert "STAGE7_SCENE2" not in log_text, \
        "没有声明映射却做了 atempo —— 这正是「掩盖损坏」"
    assert "STAGE7_SYNC_FAIL" in log_text, "同步校验没有如实失败"


def test_real_stage7_declared_speed_applies_tempo(tmp_path):
    """端到端正例：显式声明 2x 变速时应做 tempo 并记 adjusted。"""
    src = tmp_path / "clip.mp4"
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=s=160x120:d=2:r=25",
         "-f", "lavfi", "-i", "sine=f=440:d=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", "-y", str(src)], capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"合成失败：{r.stderr[:200]}")

    temp = tmp_path / "temp"; logs = tmp_path / "logs"; out = tmp_path / "out"
    for d in (temp, logs, out):
        d.mkdir()
    # 上游是 2x 速度（1 秒）
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-vf", "setpts=0.5*PTS",
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y",
         str(temp / "jobB7b_clip_step6_grain.mkv")], capture_output=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-vn", "-c:a", "copy",
         "-y", str(temp / "jobB7b_clip.mka")], capture_output=True)

    params = json.dumps({
        "temp_dir": str(temp), "log_dir": str(logs), "output_dir": str(out),
        "duration": 2.0,
        "expected_speed": 2.0,        # ← 明确声明
    })
    rc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "video_restoration_pipeline",
                                      "stages", "stage7_encode.py"),
         "--input", str(src), "--job-id", "jobB7b", "--params", params],
        capture_output=True, text=True, cwd=str(tmp_path))
    if rc.returncode == 80:
        pytest.skip("依赖预检未通过")

    log_text = "".join(f.read_text(encoding="utf-8", errors="replace")
                       for f in logs.glob("*.log"))
    # 声明了 2x → 必须**尝试**应用 tempo（SCENE2），或者因为声明与实测
    # 不符而明确拒绝（DECLARATION_MISMATCH）。两者都算正确：
    # 前者承认映射，后者拒绝盲信。绝不允许"沉默地按裸时长差拉音频"。
    applied = "STAGE7_SCENE2" in log_text
    refused = "STAGE7_TIMING_DECLARATION_MISMATCH" in log_text
    assert applied or refused, (
        "既没应用声明的映射，也没记录拒绝 —— 行为不可解释\n"
        + log_text[-600:])
    assert "STAGE7_TIMING_UNEXPLAINED" not in log_text, \
        "已声明映射却仍被判 unexplained"
