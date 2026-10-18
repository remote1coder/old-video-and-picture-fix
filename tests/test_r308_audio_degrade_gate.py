"""R3-08 / B12 补完：>8 声道 arnndn 回退必须守 allow_audio_degrade。

原实现：声道数 > `_MAX_PER_CHANNEL_ARNNDN`(8) 时**无条件**退回 afftdn，
只记一条 WARNING。同一文件里"arnndn 模型缺失"的分支已经正确遵守
`allow_audio_degrade`（未授权则 `return "__DENOISE_UNAVAILABLE__"`），
唯独这条 >8ch 路径绕过了开关——用户明确要求 arnndn 且未授权降级时，
仍会被静默换成 afftdn，rc 照样 0。

判据（与 R3-08/B12 一致）：**明确要求的算法不可用必须失败，不能降级**。
`auto` 模式的语义是"有就用、没有就降级"，因此始终授权降级。
"""

import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_stage7():
    """加载 video stage7 模块（与 test_rv_round2.py 同一写法）。"""
    vstages = os.path.join(ROOT, "video_restoration_pipeline", "stages")
    if vstages not in sys.path:
        sys.path.insert(0, vstages)
    import stage7_encode as s7  # noqa: E402
    return s7


s7 = _load_stage7()


def _esc(x):
    return x


def test_over_8ch_without_authorization_raises():
    """反例：10ch + 明确要求 arnndn + 未授权 → 必须抛异常，不能静默降级。"""
    lay = {"layout": "7.1", "channels": 10}
    with pytest.raises(s7.AudioDegradeNotAllowed):
        s7._arnndn_filters("m.rnnn", lay, _esc, None, allow_degrade=False)


def test_over_8ch_with_authorization_degrades_to_afftdn():
    lay = {"layout": "7.1", "channels": 10}
    out = s7._arnndn_filters("m.rnnn", lay, _esc, None, allow_degrade=True)
    assert out == ["afftdn=nr=12:nf=-50"]


def test_at_limit_8ch_is_not_affected():
    """边界：正好 8ch 属于支持范围，不受降级开关影响。"""
    lay = {"layout": "7.1", "channels": 8}
    out = s7._arnndn_filters("m.rnnn", lay, _esc, None, allow_degrade=False)
    assert "arnndn" in out[0]
    assert out != ["afftdn=nr=12:nf=-50"]


def test_stereo_unaffected():
    lay = {"layout": "stereo", "channels": 2}
    out = s7._arnndn_filters("m.rnnn", lay, _esc, None, allow_degrade=False)
    assert "channelsplit" in out[0] and "arnndn" in out[0]


def test_default_is_degrade_allowed_for_auto_callers():
    """默认参数保持向后兼容（auto 调用方不传参时行为不变）。"""
    lay = {"layout": "7.1", "channels": 10}
    assert s7._arnndn_filters("m.rnnn", lay, _esc, None) == \
        ["afftdn=nr=12:nf=-50"]
