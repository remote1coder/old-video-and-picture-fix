# -*- coding: utf-8 -*-
"""stage7 全链核对（源片 vs 成品）三档模式的回归测试。

背景（2026-10-24）：stage7 原有的两道核对都只看局部——
    * verify_output(input_source, output) → 只比"编码这一跳"（有意设计）
    * _validate_av_sync()                 → 只比成品**内部**的音视频
而"源片 → 成品"的帧数丢失/时长漂移，自动流程里**从未核对过**
（artifact_verify 只以独立 CLI 形式存在，main.py 里没有任何调用）。

补上后按三档模式决定拦哪些判据。判据逻辑提取成纯函数
`_fullchain_blockers()` 以便单测——否则它埋在 stage7 的 main() 里，
几百行控制流，测不到。
"""
import io
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE7 = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                                 "stage7_encode.py")
_CONFIG = os.path.join(_ROOT, "video_restoration_pipeline", "config.yaml")


def _load_stage7():
    import importlib.util
    spec = importlib.util.spec_from_file_location("s7_under_test", _STAGE7)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def s7():
    return _load_stage7()


def _ck(frames=None, duration=None):
    return {"frames": {"ok": frames}, "duration": {"ok": duration}}


# ---------------------------------------------------------------- 三档语义

def test_frames_only_blocks_frame_loss(s7):
    """frames_only：帧数不符→拦。"""
    assert s7._fullchain_blockers(_ck(frames=False), "frames_only") == ["FRAMES"]


def test_frames_only_ignores_duration_drift(s7):
    """frames_only：时长漂移**只记录不拦**（这是它误杀最低的原因）。"""
    assert s7._fullchain_blockers(_ck(duration=False), "frames_only") == []


def test_relative_also_blocks_duration(s7):
    """relative：时长也拦。"""
    got = s7._fullchain_blockers(_ck(frames=True, duration=False), "relative")
    assert got == ["DURATION"]


def test_absolute_adds_absolute_sync(s7):
    """absolute：再加"成品自身绝对漂移"。"""
    got = s7._fullchain_blockers(_ck(frames=True, duration=True),
                                 "absolute", sync_abs_rc=86)
    assert got == ["SYNC_ABSOLUTE"]


def test_absolute_without_sync_failure_is_clean(s7):
    """absolute 但绝对漂移未超限 → 放行。"""
    got = s7._fullchain_blockers(_ck(frames=True, duration=True),
                                 "absolute", sync_abs_rc=0)
    assert got == []


def test_all_good_blocks_nothing(s7):
    for mode in ("frames_only", "relative", "absolute"):
        assert s7._fullchain_blockers(_ck(frames=True, duration=True),
                                      mode) == []


def test_unknown_mode_falls_back_to_frames_only(s7):
    """未知模式按最保守的 frames_only 处理，不因拼错而放行一切。"""
    assert s7._fullchain_blockers(_ck(frames=False), "typo_mode") == ["FRAMES"]
    assert s7._fullchain_blockers(_ck(duration=False), "typo_mode") == []


def test_mode_is_case_insensitive(s7):
    assert s7._fullchain_blockers(_ck(duration=False), "RELATIVE") == ["DURATION"]


def test_handles_missing_or_none_checks(s7):
    """checks 缺失/为 None 时不炸，按"没核对到"处理（不拦）。"""
    assert s7._fullchain_blockers({}, "frames_only") == []
    assert s7._fullchain_blockers(None, "absolute") == []


def test_multiple_blockers_accumulate(s7):
    got = s7._fullchain_blockers(_ck(frames=False, duration=False),
                                 "absolute", sync_abs_rc=86)
    assert got == ["FRAMES", "DURATION", "SYNC_ABSOLUTE"]


# ---------------------------------------------------------------- 配置契约

def test_config_declares_the_switches():
    """三个开关必须在 config 里显式声明，且默认值是安全的那一侧。"""
    import yaml
    cfg = yaml.safe_load(io.open(_CONFIG, encoding="utf-8"))
    top = cfg.get("global", cfg)
    assert top.get("full_chain_verify", True) is True, "总开关应默认开启"
    assert top.get("full_chain_verify_mode", "frames_only") == "frames_only", \
        "默认必须是误杀最低的那档"
    assert top.get("allow_verify_fail_publish", False) is False, \
        "强制出片必须默认关闭"
    assert top.get("allow_sync_fail_publish", False) is False
    assert top.get("stage_hard_timeout_hours", 48) == 48
    # 阈值可手动调整
    assert top.get("sync_fail_ms_lag", 400) == 400
    assert top.get("sync_fail_ms_lead", 200) == 200
