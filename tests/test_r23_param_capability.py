# -*- coding: utf-8 -*-
"""R23-13：逐参数能力表 + "声明/读取/身份"三方一致性的**行为**验证。

## 为什么不能只靠 grep

审核：

> 参数进入 hash 但没有真实读取、代码读取但没有依赖声明，都应被发现；
> **仍要有行为测试，不能只靠 grep**。

所以这个文件分两层：

1. **一致性**：静态扫（`tests/probes/param_capability_matrix.py`）必须为 0 问题，
   且"已知未实现"的键不能偷偷跑回 hash / stage_deps。
2. **行为**：本轮认定的 5 个"读了但没进 hash"和 3 个"写了但没读"的键，
   逐个用**真实函数调用**证明改它确实改变输出（或确实不再造成失效）。

## 本轮的三类处置

| 类 | 例子 | 处置 |
|---|---|---|
| 读了但不在 hash | `stabilize`、`temp_avg`、`arnndn_model` | **加进** `CORE_QUALITY_PARAMS` |
| 只控制资源/门禁 | `ai_frame_*`、`stage_inactivity_timeout` | 明确登记为**非语义**（读，但不该重算媒体）|
| 声明了但没人读 | `target_fps`、`psf_handling_alt`、`deblock_alt` | **移出** hash/deps，登记到 `DECLARED_NOT_IMPLEMENTED` |
"""
import io
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages"),
           os.path.join(_ROOT, "tests", "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import checkpoint as cp  # noqa: E402
import stage_deps as sd  # noqa: E402
import audio_utils as au  # noqa: E402
import stage3_ai_upscale as s3  # noqa: E402
import param_capability_matrix as matrix  # noqa: E402


# ------------------------------------------------ 1. 一致性

def test_the_capability_matrix_has_no_inconsistency():
    """静态扫描必须为 0 问题（三类：hash 无读取 / 读了没声明 / 依赖无读取）。"""
    _rows, problems = matrix.scan()
    assert problems == [], "\n".join(
        f"{p['kind']}: {p['key']} — {p['detail']}" for p in problems)


def test_declared_but_unimplemented_keys_are_out_of_the_identity_sets():
    """登记为"未实现"的键不能再造成无谓失效。"""
    for k in sd.DECLARED_NOT_IMPLEMENTED:
        assert k not in cp.CORE_QUALITY_PARAMS, (
            f"{k} 没人读，却还在 CORE_QUALITY_PARAMS 里 —— 改它会让中间产物"
            f"白重算一遍")
        for stage, keys in sd._STAGE_KEYS.items():
            assert k not in keys, f"{k} 还在 stage {stage} 的依赖表里"


def test_resource_and_gate_keys_are_non_semantic_not_silently_missing():
    """资源/门禁类键要**明确**登记为非语义，不能靠"谁都没声明"混过去。"""
    for k in ("stage_inactivity_timeout", "ai_frame_transport",
              "color_drift_threshold", "full_chain_verify"):
        assert k in sd.NON_SEMANTIC_PARAMS, k
        assert k not in cp.CORE_QUALITY_PARAMS, k


# ------------------------------------------------ 2. 行为：hum 参数

def test_hum_harmonics_changes_the_filter_count():
    """`hum_harmonics` 此前写死 3 —— 改它只改 hash，滤镜不变。"""
    assert len(au.build_hum_filters(50)) == 3          # 默认不变
    assert len(au.build_hum_filters(50, config={"hum_harmonics": 1})) == 1
    assert len(au.build_hum_filters(50, config={"hum_harmonics": 5})) == 5
    # 超出范围要夹住，而不是造出 100 个滤波器
    assert len(au.build_hum_filters(50, config={"hum_harmonics": 99})) == 8


def test_hum_bandwidth_changes_the_filter_width():
    f = au.build_hum_filters(60, config={"hum_bandwidth": 7.5})
    assert all("width=7.5" in x for x in f), f
    default = au.build_hum_filters(60)
    assert all("width=2" in x for x in default), default


def test_hum_width_type_is_honoured_and_validated():
    f = au.build_hum_filters(50, config={"hum_width_type": "q"})
    assert all("width_type=q" in x for x in f), f
    bad = au.build_hum_filters(50, config={"hum_width_type": "nonsense"})
    assert all("width_type=h" in x for x in bad), bad


def test_dual_hum_still_works_with_custom_harmonics():
    f = au.build_hum_filters(50, True, {"hum_dual": True,
                                        "hum_harmonics": 2})
    assert len(f) == 4, f        # 50/100 + 60/120


# ------------------------------------------------ 3. 行为：GPU 回退开关

@pytest.mark.parametrize("value,allowed", [
    # R25-05：缺省**不再**授权降质 —— None 从 True 改为 False
    (None, False), (True, True), ("true", True), (1, True),
    (False, False), ("false", False), (0, False), ("no", False),
])
def test_gpu_fallback_flag_is_parsed_strictly(value, allowed):
    assert s3.gpu_fallback_allowed({"gpu_memory_fallback": value}) is allowed


# ------------------------------------------------ 4. 行为：身份真的变了

@pytest.fixture(scope="module")
def identity_file(tmp_path_factory):
    """**同一个**输入文件给所有比较用。

    注意：`compute_param_hash` 的身份里含 `file_mtime`。每次新建临时文件
    会让两次调用天然不同 —— 那是测试自己的错，不是"键影响 hash"的证据。
    （我第一次就是这么写的，两条用例假失败。）
    """
    p = str(tmp_path_factory.mktemp("identity") / "fixture.bin")
    with open(p, "wb") as f:
        f.write(b"identity-fixture")
    os.utime(p, (1700000000, 1700000000))
    return p


def _h(identity_file, **kw):
    return cp.compute_param_hash(identity_file, kw)


@pytest.mark.parametrize("key,lo,hi", [
    ("stabilize", False, True),
    ("temp_avg", 0, 2),
    ("arnndn_model", "", "models/x.rnnn"),
    ("hum_width_type", "h", "q"),
    ("multichannel_preserve", False, True),
    ("progressive_percent", 80.0, 95.0),
    ("shake_probe_threshold_px", 1.0, 3.0),
])
def test_quality_keys_now_change_the_param_hash(identity_file, key, lo, hi):
    """这些键会被读取并改变输出；不同取值必须产生不同身份。"""
    assert _h(identity_file, **{key: lo}) != _h(identity_file, **{key: hi}), (
        f"{key} 改了但 param_hash 没变 —— 旧中间产物会被错误复用")


def test_removed_dead_keys_no_longer_change_the_hash(identity_file):
    """没人读的键不再造成无谓失效（改它不该触发重算）。"""
    assert _h(identity_file, qtgmc_input_type_rule="a") == \
        _h(identity_file, qtgmc_input_type_rule="b")
    assert _h(identity_file, psf_handling="a") == \
        _h(identity_file, psf_handling="b")
    assert _h(identity_file, target_fps=24) == \
        _h(identity_file, target_fps=30)


def test_non_semantic_keys_do_not_change_the_hash(identity_file):
    assert _h(identity_file, stage_inactivity_timeout=60) == \
        _h(identity_file, stage_inactivity_timeout=600)
    assert _h(identity_file, full_chain_verify=True) == \
        _h(identity_file, full_chain_verify=False)
