"""COL-04（第七轮）：每一代都携带实际颜色表示，继承必须有证据。

**实测证据**

```
源   real2s.mov   pix=yuv420p     matrix=bt709   range=limited  prim=bt709
中间 step6_grain  pix=yuv420p16le matrix=None    range=None     prim=None
```

中间产物**没有任何颜色标签**。stage3 的 `rgb48 -> yuv420p16le` 是一次
真正的矩阵转换，但旧代码没传 `-colorspace`，矩阵由 ffmpeg 按分辨率
启发式选择（SD 走 601、HD 走 709）。stage7 再把原片标签"补"上去 ——
**一旦 ffmpeg 的选择与原片不同，这个标签就是错的**。

修法：
1. stage3 探测源颜色，编码时显式 `-colorspace/-color_range/
   -color_primaries/-color_trc`（让中间产物自带正确标签）；
2. stage7 的继承改为**看证据**：中间产物与原片像素族一致（没发生
   rgb↔yuv 转换）才继承，否则拒绝并留痕；
3. 身份侧车记录 `color`（字段分开，探测失败时 probed=False）。
"""

import os
import subprocess
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "shared_modules"),
           os.path.join(ROOT, "video_restoration_pipeline"),
           os.path.join(ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import media_io as M  # noqa: E402


def _make(tmp_path, name, extra):
    out = str(tmp_path / name)
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=s=64x64:d=1:r=5", "-c:v", "libx264",
         "-pix_fmt", "yuv420p"] + extra + ["-y", out],
        capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"ffmpeg 失败：{r.stderr[:200]}")
    return out


def test_probe_reports_all_fields_separately(tmp_path):
    f = _make(tmp_path, "a.mp4",
              ["-colorspace", "bt709", "-color_primaries", "bt709",
               "-color_trc", "bt709", "-color_range", "tv"])
    d = M.probe_color(f)
    assert d["probed"] is True
    assert d["matrix"] == "bt709"
    assert d["primaries"] == "bt709"
    assert d["transfer"] == "bt709"
    assert d["range"] == "limited"


def test_bt2020_is_reported_as_bt2020(tmp_path):
    f = _make(tmp_path, "b.mp4",
              ["-colorspace", "bt2020nc", "-color_primaries", "bt2020",
               "-color_trc", "bt2020-10"])
    d = M.probe_color(f)
    assert d["matrix"] == "bt2020nc"
    assert d["primaries"] == "bt2020"
    assert d["transfer"] == "bt2020-10"


def test_untagged_file_reports_none_not_default(tmp_path):
    """无标签必须报 None，不能填 601/709 冒充已知。"""
    f = _make(tmp_path, "c.mp4", [])
    d = M.probe_color(f)
    assert d["probed"] is True
    assert d["matrix"] is None
    assert d["primaries"] is None
    assert d["range"] is None


def test_probe_failure_is_explicit(tmp_path):
    d = M.probe_color(str(tmp_path / "nope.mkv"))
    assert d["probed"] is False
    assert d.get("error")


def test_color_descriptions_equal_detects_difference(tmp_path):
    a = M.probe_color(_make(tmp_path, "x.mp4",
                            ["-colorspace", "bt709", "-color_primaries",
                             "bt709", "-color_trc", "bt709"]))
    b = M.probe_color(_make(tmp_path, "y.mp4",
                            ["-colorspace", "bt470bg", "-color_primaries",
                             "bt470bg", "-color_trc", "bt709"]))
    assert M.color_descriptions_equal(a, a) is True
    assert M.color_descriptions_equal(a, b) is False


def test_stage3_carries_colour_into_intermediate():
    """stage3 必须显式传颜色参数，而不是让 ffmpeg 猜。"""
    src = open(os.path.join(ROOT, "video_restoration_pipeline", "stages",
                            "stage3_ai_upscale.py"), encoding="utf-8").read()
    assert "STAGE3_COLOR_CARRIED" in src
    assert '"-colorspace"' in src
    assert '"-color_range"' in src
    assert "*_color_opts" in src, "颜色参数没接进 encode_cmd"


def test_stage7_refuses_blind_inheritance():
    """stage7 继承前必须检查证据。"""
    src = open(os.path.join(ROOT, "video_restoration_pipeline", "stages",
                            "stage7_encode.py"), encoding="utf-8").read()
    assert "STAGE7_COLOR_INHERIT_REFUSED" in src
    assert "pixel-family changed" in src
    assert "evidence=" in src


def test_identity_sidecar_records_colour():
    src = open(os.path.join(ROOT, "video_restoration_pipeline", "main.py"),
               encoding="utf-8").read()
    assert '"color": _read_color_description(artifact)' in src
