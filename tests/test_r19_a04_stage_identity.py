# -*- coding: utf-8 -*-
"""第十九轮 A4：stage 身份与内部键（对应 R18-A04）。

审核原文：

> 先构建 reuse_plan，后加入 `_run_identity`、chain_mode、vs_deterministic。
> 局部 stage_identity 会把这些未知键算作影响所有阶段，下次比较时却尚未
> 加入这些键。内存比较确认 stage1 身份不同；`_run_identity` 中的全局参数
> 摘要还会让仅改 CRF 的操作污染前级身份。

根因（比审核描述更具体）：**视频链的 `param_affects_stage` 缺少 `_` 前缀
跳过**，而照片链的 `photo_param_affects_stage` 有：

    if k.startswith("_"): return False

于是 `_run_identity` / `_shake_probe` / `_stabilize_effective` 这些纯内部键
全部落到"无人认领 → 影响所有阶段（保守 True）"分支，污染每一个 stage 的身份。

本文件锁住三件事：

1. 内部键（`_` 前缀）不得进入任何 stage 的身份；
2. `vs_deterministic` / `chain_mode` 必须**只**影响真正受影响的阶段；
3. 保守策略（真正未知的业务参数 → 影响所有阶段）不能因此被削弱。
"""
import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage_deps as sd  # noqa: E402

ALL_STAGES = (0, 1, 2, 3, 4, 5, 6, 7)


# ------------------------------------------------ 内部键必须被忽略

@pytest.mark.parametrize("key", [
    "_run_identity", "_shake_probe", "_stabilize_effective",
    "_stabilize_auto_note", "_stage", "_mode", "_quick",
])
def test_internal_keys_never_affect_any_stage(key):
    """`_` 前缀是内部/回执键，不是内容语义。

    审核：`_run_identity` 里的全局参数摘要会让仅改 CRF 的操作污染前级身份。
    """
    for s in ALL_STAGES:
        assert sd.param_affects_stage(key, s) is False, (
            f"{key} 不该影响 stage{s} 的身份")


def test_run_identity_does_not_change_stage1_identity():
    """加/不加 `_run_identity`，stage1 的身份必须完全相同。"""
    base = {"crf": 20, "vs_bit_depth": 16}
    with_id = dict(base, _run_identity={"input": {"digest": "abc"},
                                        "params": "def"})
    assert sd.stage_identity(base, 1) == sd.stage_identity(with_id, 1)


def test_changing_crf_does_not_touch_upstream_identity():
    """改 CRF 只能让 stage7 的身份变，前级必须不变（防止全链重算）。"""
    a = {"crf": 20, "vs_bit_depth": 16, "upscale_model": "m"}
    b = dict(a, crf=28)
    assert sd.stage_identity(a, 7) != sd.stage_identity(b, 7), "stage7 应变"
    for s in (0, 1, 2, 3, 4, 5, 6):
        assert sd.stage_identity(a, s) == sd.stage_identity(b, s), (
            f"改 CRF 不该改变 stage{s} 的身份")


# ------------------------------------------------ 新增键必须有明确认领

def test_vs_deterministic_only_affects_ideally_affected_stages():
    """`vs.deterministic` 只改变 stage1 / stage6 的渲染输出。

    stage1 用 TemporalDegrain/Stab，stage6 用 GrainFactory3 —— 这两个
    才读 `core.num_threads = 1`。其他 stage 不受影响。
    """
    affects = {s for s in ALL_STAGES if sd.param_affects_stage("vs_deterministic", s)}
    assert affects == {1, 6}, f"vs_deterministic 影响范围不对：{affects}"


def test_chain_mode_only_affects_the_fused_stages():
    """`chain.mode` 只改变 stage3+4 的执行方式（是否融合），其他阶段无关。"""
    affects = {s for s in ALL_STAGES if sd.param_affects_stage("chain_mode", s)}
    assert affects == {3, 4}, f"chain_mode 影响范围不对：{affects}"


# ------------------------------------------------ 保守策略不能被削弱

def test_unknown_business_key_still_affects_every_stage():
    """真正未知的业务参数仍要保守处理（宁可多跑，不可误复用）。

    这条是防止"修内部键"顺手把保守策略也删了。
    """
    for s in ALL_STAGES:
        assert sd.param_affects_stage("some_brand_new_knob", s) is True, s


def test_photo_chain_already_had_the_fix():
    """照片链本来就有 `_` 前缀跳过 —— 视频链要和它对齐。"""
    for s in range(1, 7):
        assert sd.photo_param_affects_stage("_run_identity", s) is False


# ------------------------------------------------ 冻结顺序（结构守卫）

def test_execution_mode_is_frozen_before_the_reuse_plan():
    """结构守卫：`chain_mode` 必须在 `build_reuse_plan()` **之前**赋值。

    审核 R18-A04：原来复用计划在前、注入在后，导致"计划时的 params"
    与"执行时的 params"不是同一份 —— 局部身份对不上，每阶段都被判失效。
    """
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "main.py"),
               encoding="utf-8", errors="replace").read()
    i_mode = src.find('params["chain_mode"] = str(')
    i_plan = src.find("_reuse_plan = build_reuse_plan(")
    assert i_mode > 0, "找不到 chain_mode 赋值"
    assert i_plan > 0, "找不到 build_reuse_plan 调用"
    assert i_mode < i_plan, (
        "chain_mode 必须在构建复用计划之前冻结（否则计划与执行用的不是同一份 params）")


def test_vs_deterministic_is_frozen_before_the_reuse_plan():
    """同上：`vs_deterministic` 也要先冻。"""
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "main.py"),
               encoding="utf-8", errors="replace").read()
    i_vs = src.find('params["vs_deterministic"] = bool(')
    i_plan = src.find("_reuse_plan = build_reuse_plan(")
    assert i_vs > 0 and i_plan > 0
    assert i_vs < i_plan, "vs_deterministic 必须在构建复用计划之前冻结"
