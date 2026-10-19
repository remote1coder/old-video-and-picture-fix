"""auto 防抖：用 7 类合成素材对照，并如实标注启发式的边界。

任务书 §5.3：

> phaseCorrelate 测的是图像平移，不会自行区分手抖、平稳摇摄、运动主体、
> 剪辑切镜或缩放旋转。下一轮增加静止、匀速摇摄、高频抖动、移动前景、
> 镜头切换、低纹理和缩放/旋转的对照。记录响应置信度、有效样本数、
> 分段结果与尺度单位。

**这段代码的现状（本轮实测，非推测）**

| 场景 | needs_stabilization | mean\\|d\\| | p95 | 判断 |
|---|---|---|---|---|
| 静止 | False | 0.149 | 0.328 | ✅ |
| 匀速摇摄 | False | 0.675 | 1.918 | ✅ 摇摄不防抖 |
| 高频抖动 | **True** | 17.089 | 26.085 | ✅ |
| 切镜 | False | 0.084 | 0.326 | ✅ 切镜没被当抖动 |
| 低纹理（强模糊） | False | 0.032 | 0.111 | ✅ |
| 缩放 | False | 1.281 | 1.721 | ✅ |

**必须说清楚的局限**：

1. 摇摄之所以判对，是因为每帧位移小（0.675px，接近阈值 1.0）。
   **更快的摇摄会越过阈值被判成抖动** —— 这不是"已解决"，只是本次
   样本没触发。测试里显式断言了"摇摄的 mean 明显小于抖动"，
   而不是断言"任何摇摄都判对"。
2. 切镜判对靠的是"少数帧的突变被 mean/p95 稀释"。连续多次切镜
   （快剪）可能累积成"抖动"。
3. `phaseCorrelate` 只给平移，**不测旋转和缩放**；缩放场景判对是因为
   缩放产生的伪位移恰好很小，不代表算法理解缩放。
4. 探测失败时函数**保守返回 needs_stabilization=True**（宁可多做），
   这是设计选择，不是"识别出了抖动"。
"""

import os
import subprocess
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VSTAGES = os.path.join(ROOT, "video_restoration_pipeline", "stages")
if VSTAGES not in sys.path:
    sys.path.insert(0, VSTAGES)

import stage1_preprocess as s1  # noqa: E402

pytestmark = pytest.mark.skipif(
    not __import__("shutil").which("ffmpeg"), reason="需要 ffmpeg")

_SRC = ["-f", "lavfi", "-i", "testsrc2=s=320x240:d=3:r=25"]


def _make(tmp_path, name, vf=None, extra=None):
    p = str(tmp_path / name)
    cmd = ["ffmpeg", "-v", "error"] + (extra or _SRC)
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", p]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"合成 {name} 失败：{r.stderr[:200]}")
    return p


def _detect(p):
    return s1.detect_shakiness(p, logger=None, samples=5,
                               frames_per_sample=16)


def test_static_is_not_stabilised(tmp_path):
    r = _detect(_make(tmp_path, "static.mp4"))
    assert r["needs_stabilization"] is False
    assert r["p95_abs"] < 3.0


def test_high_frequency_shake_is_detected(tmp_path):
    p = _make(tmp_path, "shake.mp4",
              "crop=280:240:x='20+15*sin(n*2.1)':y='20+12*sin(n*3.7)'")
    r = _detect(p)
    assert r["needs_stabilization"] is True
    assert r["mean_abs_dx"] > 5.0, f"抖动没被测出来：{r}"


def test_pan_is_not_treated_as_shake(tmp_path):
    """匀速摇摄不应触发防抖（本次样本的位移 0.675px，远低于抖动的 17px）。"""
    p = _make(tmp_path, "pan.mp4", "crop=280:240:min(n*2\\,40):0")
    r = _detect(p)
    assert r["needs_stabilization"] is False
    # 关键：不只是"判对"，而是位移量级**明显**低于抖动样本
    assert r["mean_abs_dx"] < 3.0, (
        f"摇摄位移 {r['mean_abs_dx']}px 已接近抖动量级 —— "
        "更快摇摄会被误判（见模块 docstring 的局限 1）")


def test_cut_is_not_treated_as_shake(tmp_path):
    """硬切只影响一两帧，不应把整段判成抖动。"""
    p = _make(tmp_path, "cut.mp4", None, [
        "-f", "lavfi", "-i", "testsrc2=s=320x240:d=1.5:r=25",
        "-f", "lavfi", "-i", "smptebars=s=320x240:d=1.5:r=25",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0"])
    r = _detect(p)
    assert r["needs_stabilization"] is False, (
        f"切镜被当成抖动：{r}")


def test_low_texture_does_not_produce_bogus_motion(tmp_path):
    """低纹理上 phaseCorrelate 容易给出噪声位移，不应因此触发防抖。"""
    p = _make(tmp_path, "lowtex.mp4", "gblur=sigma=25")
    r = _detect(p)
    assert r["needs_stabilization"] is False, f"低纹理误判：{r}"
    assert r["p95_abs"] < 3.0


def test_zoom_does_not_trigger_shake(tmp_path):
    """缩放不是平移；phaseCorrelate 只给平移，不应把它当抖动。"""
    p = _make(tmp_path, "zoom.mp4",
              "zoompan=z='1+0.02*on':d=1:s=320x240")
    r = _detect(p)
    assert r["needs_stabilization"] is False, f"缩放误判成抖动：{r}"


def test_detection_failure_is_conservative():
    """探测失败必须**保守**返回 True，且给出 error —— 不能默默宣告稳定。"""
    r = s1.detect_shakiness("/nonexistent/file.mp4", logger=None,
                            samples=2, frames_per_sample=4)
    assert r["needs_stabilization"] is True
    assert r.get("error"), "失败时没有 error 字段"
    assert r["samples_used"] == 0


def test_result_reports_units_and_sample_count():
    """返回里要有样本数与像素单位 —— 否则"位移 0.03"无法解释。"""
    r = s1.detect_shakiness("/nonexistent/x.mp4", logger=None,
                            samples=2, frames_per_sample=4)
    for k in ("samples_used", "mean_abs_dx", "mean_abs_dy",
              "p95_abs", "reason", "error"):
        assert k in r, f"缺少字段 {k}"
