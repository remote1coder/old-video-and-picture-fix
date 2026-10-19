"""COL-02（第七轮）：色彩描述四项必须分开解析，枚举 9 是 BT2020。

**旧实现的三个错误**（都在 `_get_source_color_info`）：

1. `if cp in ("bt709", "1", "9")` —— FFmpeg 的 AVColorPrimaries 里
   **9 是 bt2020**，不是 bt709。BT2020 素材因此被判成"已经是 709"，
   走到"跳过转换"分支，宽色域像素被打上 bt709 标签。
2. 用 **primaries** 推断要不要做矩阵转换。决定矩阵的是
   `color_space`（matrix），primaries 只是色域原色。
3. `range_known = bool(cr)` —— 字符串非空就算已知，垃圾值也放行。

任务书 §4.3：「HDR/BT2020 不支持时明确拒绝或标未支持，不能默认塞进
601/709」。本项目对 BT2020 选择**阻断**（rc=82），需显式
`color_standard_override` 才继续，并警告颜色仍不正确。
"""

import os
import shutil
import subprocess
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VSTAGES = os.path.join(ROOT, "video_restoration_pipeline", "stages")
if VSTAGES not in sys.path:
    sys.path.insert(0, VSTAGES)

import stage7_encode as s7  # noqa: E402


def _make(tmp_path, name, vf_args):
    out = str(tmp_path / name)
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi",
           "-i", "testsrc2=s=64x64:d=1:r=5", "-c:v", "libx264",
           "-pix_fmt", "yuv420p10le"] + vf_args + ["-y", out]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"ffmpeg 合成失败：{r.stderr[:200]}")
    return out


def test_bt709_source_is_recognised(tmp_path):
    f = _make(tmp_path, "a.mp4",
              ["-colorspace", "bt709", "-color_primaries", "bt709",
               "-color_trc", "bt709"])
    rng, std, rk, pk = s7._get_source_color_info(f, None)
    assert std == "BT709" and rk and pk
    assert rng == "limited"


def test_bt2020_source_is_not_treated_as_bt709(tmp_path):
    """核心反例：枚举 9 / bt2020 绝不能变成 BT709。"""
    f = _make(tmp_path, "b.mp4",
              ["-colorspace", "bt2020nc", "-color_primaries", "bt2020",
               "-color_trc", "bt2020-10"])
    rng, std, rk, pk = s7._get_source_color_info(f, None)
    assert std == "BT2020", f"BT2020 被误判成 {std}"
    assert std != "BT709"


def test_pal_source_is_recognised(tmp_path):
    f = _make(tmp_path, "c.mp4",
              ["-colorspace", "bt470bg", "-color_primaries", "bt470bg",
               "-color_trc", "bt709"])
    _, std, _, pk = s7._get_source_color_info(f, None)
    assert std == "PAL" and pk


def test_ntsc_source_is_recognised(tmp_path):
    f = _make(tmp_path, "d.mp4",
              ["-colorspace", "smpte170m", "-color_primaries", "smpte170m",
               "-color_trc", "bt709"])
    _, std, _, pk = s7._get_source_color_info(f, None)
    assert std == "NTSC" and pk


def test_bt2020_is_blocked_by_default():
    """源码守卫：BT2020 必须有明确的阻断分支 + 显式 override。"""
    src = open(s7.__file__, encoding="utf-8").read()
    assert "STAGE7_COLORSPACE_UNSUPPORTED" in src
    assert "color_standard_override" in src
    # 阻断必须发生在"跳过转换"之前
    i_block = src.index("STAGE7_COLORSPACE_UNSUPPORTED")
    i_skip = src.index('logger.info("STAGE7_COLORSPACE_SKIP"')
    assert i_block < i_skip, "BT2020 判断在'跳过转换'之后 —— 会被误跳过"


def test_all_four_colour_fields_are_requested():
    """probe 必须同时取 range/primaries/space(transfer)，不能只取两个。"""
    src = open(s7.__file__, encoding="utf-8").read()
    assert "stream=color_range,color_primaries,color_space,color_transfer" in src
    assert "STAGE7_COLOR_PROBE" in src, "缺少分开记录四项的日志"


def test_numeric_enum_map_does_not_alias_9_to_709():
    """静态守卫：数字 9 只能映射到 bt2020。"""
    src = open(s7.__file__, encoding="utf-8").read()
    assert '"9": "bt2020"' in src
    assert '"9": "bt709"' not in src
    assert '("bt709", "1", "9")' not in src
