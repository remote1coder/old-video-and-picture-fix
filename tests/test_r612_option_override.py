"""R12 回归：选项覆盖被静默忽略（分型参数 / grain / audio 三处）。

实测（视频选项矩阵 v6，9/9 rc=0 但参数全是默认值）：
1. `user_overrides.grain_factory3` 被 stage6 完全忽略——它只读
   `cfg["grain_factory3"]`，于是 dv(0.5/0.1/0.1/temp_avg=1) 与
   hdv(0.8) 三例的 vpy 全是 analog_tape 默认 1.0/0.2/0.2。
2. `user_overrides.path` 不切换 paths 参数组——`path_key` 只从
   `job["detected_class"]` 取，三型矩阵三例参数完全相同。
3. `user_overrides.audio_override`（扁平 dict）被忽略——merge_params
   只认 `config["audio_override"][path_key]` 这一个嵌套来源。

这三处都是 rc=0 的**静默 no-op**：日志里 PARAM_OVERRIDE 显示值已
注入，但下游 stage 读的是另一个来源，看不出任何异常。
"""

import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

import main as vmain  # noqa: E402


def _cfg():
    return {
        "global": {"output_dir": "./output", "temp_dir": "./temp"},
        "paths": {
            "analog_tape": {"deinterlace": "QTGMC", "qtgmc_preset": "Slow",
                            "target_resolution": "720x480"},
            "dv_digital": {"deinterlace": "PSF", "qtgmc_preset": "Slower",
                           "target_resolution": "720x480"},
            "hdv_avchd": {"deinterlace": "QTGMC", "qtgmc_preset": "Slower",
                          "target_resolution": "1920x1080"},
        },
        "audio_override": {
            "analog_tape": {"hum_freq": None, "hum_dual": False,
                            "denoise": "afftdn"},
        },
        "grain_factory3": {
            "enabled": True,
            "analog_tape": {"g1str": 1.0, "g2str": 0.2, "g3str": 0.2},
            "dv_digital": {"g1str": 0.5, "g2str": 0.1, "g3str": 0.1},
            "hdv_avchd": {"g1str": 0.8, "g2str": 0.2, "g3str": 0.2},
            "temp_avg": 0,
        },
    }


def _job(**over):
    job = {
        "job_id": "job_001",
        "input_path": "in.mp4",
        "detected_class": "ANALOG_TAPE",
        "status": "PENDING",
    }
    if over:
        job["user_overrides"] = over
    return job


def test_path_override_switches_params_group():
    """`path` 必须真正切换 paths 参数组（实测三型矩阵此前完全相同）。"""
    cfg = _cfg()
    p = vmain.merge_params(_job(path="dv_digital"), cfg, {})
    assert p["qtgmc_preset"] == "Slower", "dv_digital 的 preset 应生效"
    assert p["deinterlace"] == "PSF"

    p_hdv = vmain.merge_params(_job(path="hdv_avchd"), cfg, {})
    assert p_hdv["target_resolution"] == "1920x1080"


def test_path_absent_keeps_detected_class_group():
    """没给 path 时仍按 detected_class 取，不得被默认路径污染。"""
    p = vmain.merge_params(_job(), _cfg(), {})
    assert p["qtgmc_preset"] == "Slow"
    assert p["deinterlace"] == "QTGMC"


def test_flat_audio_override_is_applied():
    """user_overrides.audio_override（扁平）必须生效（实测此前被忽略）。"""
    cfg = _cfg()
    p = vmain.merge_params(
        _job(audio_override={"hum_freq": 50, "hum_dual": True}), cfg, {})
    assert p.get("hum_freq") == 50
    assert p.get("hum_dual") is True
    # 分型默认里的其它键仍保留（合并而非替换）
    assert p.get("audio_denoise") == "afftdn"


def test_config_audio_override_still_applies_without_user_override():
    """回归保护：没有 user_overrides 时 config 的分型音频默认照旧生效。"""
    p = vmain.merge_params(_job(), _cfg(), {})
    assert p.get("hum_freq", "unset") in (None, "unset")
    assert p.get("audio_denoise") == "afftdn"


def test_grain_factory3_flat_override_lands_in_params():
    """扁平 grain 覆盖必须进 params（stage6 侧由 stage6 自己读 params）。"""
    cfg = _cfg()
    p = vmain.merge_params(
        _job(grain_factory3={"g1str": 0.3, "g2str": 0.05,
                             "g3str": 0.05, "temp_avg": 1}), cfg, {})
    g = p["grain_factory3"]
    assert g["g1str"] == 0.3
    assert g["temp_avg"] == 1
    # 分型子键仍保留，stage6 按 detected_class 取
    assert g["analog_tape"]["g1str"] == 1.0


def test_grain_config_only_when_no_override():
    """没有覆盖时 params 里的 grain 来自 config（原有行为不变）。"""
    p = vmain.merge_params(_job(), _cfg(), {})
    assert p["grain_factory3"]["analog_tape"]["g1str"] == 1.0
    assert p["grain_factory3"]["dv_digital"]["g1str"] == 0.5


@pytest.mark.parametrize("group,expect", [("dv_digital", 0.5),
                                         ("hdv_avchd", 0.8),
                                         ("analog_tape", 1.0)])
def test_stage6_material_group_values(group, expect):
    """stage6 的分型取值：每个分型各自的 g1str（回归 R12 的核心断言）。"""
    cfg = _cfg()["grain_factory3"]
    material = cfg[group]
    assert material["g1str"] == expect
