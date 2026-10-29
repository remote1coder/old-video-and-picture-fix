# -*- coding: utf-8 -*-
"""R14-08：人工帧率的 `30000/1001` 被 `float()` 拒掉。

**缺陷**

`stage3` 原来：

```python
_manual_fps = float(_manual_fps)          # "30000/1001" → ValueError
except (TypeError, ValueError):
    _manual_fps = 0.0                     # → 当成"用户没指定"
```

于是一个用户填了**最标准的 NTSC 帧率**，反而被判成"没给帧率"，
直接 `STAGE3_FPS_UNAVAILABLE` → 退出 86。

而 `float("23.976")` 再反推有理数会累积误差（23.976 ≠ 24000/1001）。

**修法**：`shared_modules/rate_spec.py` 用 `fractions.Fraction` 精确表示，
全程不经过浮点；非法值给出**具体原因**而不是笼统失败。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rate_spec as rs  # noqa: E402


# ---------------------------------------------------------------- 合法输入

@pytest.mark.parametrize("spec,num,den", [
    ("30000/1001", 30000, 1001),
    ("24000/1001", 24000, 1001),
    ("25", 25, 1),
    ("30", 30, 1),
    ("  30  ", 30, 1),
    ("23.976", 2997, 125),      # 小数按精确值保留，不"猜"标准帧率
    ("29.97", 2997, 100),
])
def test_parses_declared_rates(spec, num, den):
    r = rs.parse_rate(spec)
    assert (r.num, r.den) == (num, den), (r.num, r.den)
    assert r.as_ffmpeg() == f"{num}/{den}"


def test_fraction_is_exact_not_float_roundtrip():
    """30000/1001 必须是**精确**的分数，不能先转 float 再反推。"""
    r = rs.parse_rate("30000/1001")
    assert r.num * 1001 == r.den * 30000          # 交叉相乘精确相等
    # 与浮点近似比较：误差应小于 1e-9
    assert abs(r.value - 30000 / 1001) < 1e-9


def test_equality_uses_cross_multiplication():
    """同值不同写法应相等（25 与 50/2）。"""
    assert rs.parse_rate("25") == rs.parse_rate("50/2")
    assert rs.parse_rate("30000/1001") == rs.parse_rate("30000/1001")


def test_source_is_carried():
    """来源要跟着走，报告里才能说清"听谁的"。"""
    assert rs.parse_rate("25", source="user").source == "user"
    assert rs.parse_rate("25", source="manifest").source == "manifest"


# ---------------------------------------------------------------- 非法输入

@pytest.mark.parametrize("spec,needle", [
    ("", "empty"),
    (None, "empty"),
    ("   ", "empty"),
    ("0/0", "zero denominator"),
    ("abc", "not a number"),
    ("25/0", "zero denominator"),
    ("-25", "positive"),
    ("0", "positive"),
    ("nan", "NaN"),
    ("inf", "Infinity"),
    ("1000", "outside the supported range"),
    ("0.5", "outside the supported range"),
])
def test_rejects_with_specific_reason(spec, needle):
    """非法值必须给出**具体**原因，不能只抛一个 ValueError。"""
    with pytest.raises(rs.RateSpecError) as ei:
        rs.parse_rate(spec)
    assert needle.lower() in ei.value.reason.lower(), ei.value.reason


def test_range_is_configurable():
    """范围可调：某些特殊素材可能需要更高帧率。"""
    r = rs.parse_rate("500", max_rate=1000)
    assert r.as_ffmpeg() == "500/1"
    with pytest.raises(rs.RateSpecError):
        rs.parse_rate("500")


def test_parse_rate_or_none_swallows_reason():
    """可选覆盖场景用 or_none，但**不要**用它代替告诉用户原因。"""
    assert rs.parse_rate_or_none("30000/1001") is not None
    assert rs.parse_rate_or_none("garbage") is None
    assert rs.parse_rate_or_none(None) is None


# ---------------------------------------------------------------- 生产接入

def test_stage3_uses_rate_spec_not_float():
    """生产代码必须走有理数解析，不能再 `float(...)` 后当 0。

    这条守调用形态；行为由上面的解析器用例覆盖。
    `float("30000/1001")` 会抛 ValueError，是本缺陷的直接来源。
    """
    src = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                       "stage3_ai_upscale.py")
    text = open(src, encoding="utf-8", errors="replace").read()
    assert "from rate_spec import parse_rate" in text, \
        "stage3 必须使用 rate_spec 解析人工帧率"
    assert "_manual_fps = float(_manual_fps)" not in text, \
        "不得再用 float() 解析人工帧率"


def test_stage3_marks_manifest_source_separately():
    """`job.fps` 来自 manifest/probe，**不等于**用户明确选择。

    审核 R14-05 指出：main 会从 job.fps 注入 original_fps，
    stage3 随后把它称为 user-assumed。来源必须如实区分。
    """
    src = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                       "stage3_ai_upscale.py")
    text = open(src, encoding="utf-8", errors="replace").read()
    assert "manifest-assumed" in text, \
        "必须能区分 manifest 来源与用户明确指定"
    assert "_fps_from_user" in text, \
        "需要有显式标记来判断是否为用户指定"
