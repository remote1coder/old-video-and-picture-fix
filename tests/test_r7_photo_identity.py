"""照片链身份记录：同一套 schema，但**不做** plan/裁剪。

**为什么两条链的策略不同**

| | 视频 | 照片 |
|---|---|---|
| 处理量 | 数万帧 | 单张 |
| 最小重算收益 | 小时级 | 秒级 |
| 是否需要 plan/裁剪 | 需要 | **不需要** |

但**身份记录必须一致**：侧车都要有 `stage_identity` + `stage_scope`，
否则"这张图是哪套参数产出的"就追溯不了。

**照片链的作用域与视频链不同**

照片只有 6 个 stage，且有 `stage2_ai_pre`（视频链没有）。
参数命名也不同：`deband_threshold` / `scratch_remove` /
`grain_synthesis` / `avif_params` 等。
"""

import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "shared_modules") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "shared_modules"))

import stage_deps as sd  # noqa: E402


@pytest.mark.parametrize("key,stage,expected", [
    # 编码参数只影响 stage6
    ("crf", 6, True), ("crf", 3, False), ("crf", 1, False),
    ("avif_params", 6, True), ("avif_params", 5, False),
    ("archival_mode", 6, True), ("archival_mode", 3, False),
    # AI 超分只影响 stage3
    ("upscale_model", 3, True), ("upscale_model", 1, False),
    # 人脸只影响 stage4
    ("face_restore", 4, True), ("face_restore", 1, False),
    ("gfpgan_model", 4, True), ("gfpgan_model", 6, False),
    # 前置修复只影响 stage1
    ("deband_threshold", 1, True), ("deband_threshold", 5, False),
    ("scratch_remove", 1, True), ("scratch_remove", 4, False),
    # 后处理只影响 stage5
    ("grain_synthesis", 5, True), ("grain_synthesis", 6, False),
    ("unsharp_strength", 5, True), ("unsharp_strength", 3, False),
    # 日志不影响任何 stage
    ("log_level", 3, False), ("log_dir", 6, False),
    # 认不出的键保守为 True
    ("some_unknown_photo_key", 4, True),
])
def test_photo_param_scope(key, stage, expected):
    assert sd.photo_param_affects_stage(key, stage) is expected


def test_photo_identity_ignores_other_stage_params():
    base = {"crf": 18, "upscale_model": "x4plus", "deband_threshold": 3}
    changed = dict(base, crf=12)
    assert sd.photo_stage_identity(base, 3) == \
        sd.photo_stage_identity(changed, 3)
    assert sd.photo_stage_identity(base, 6) != \
        sd.photo_stage_identity(changed, 6)


def test_photo_identity_ignores_log_level():
    base = {"crf": 18, "log_level": "INFO"}
    for s in range(1, 7):
        assert sd.photo_stage_identity(base, s) == \
            sd.photo_stage_identity(dict(base, log_level="DEBUG"), s)


def test_photo_stage_names_cover_all_six():
    assert set(sd.PHOTO_STAGE_NAMES) == set(range(1, 7))


def test_photo_main_records_identity():
    """照片 main 必须在 stage 成功后记录身份。"""
    src = open(os.path.join(ROOT, "photo_restoration_pipeline", "main.py"),
               encoding="utf-8").read()
    assert "PHOTO_STAGE_IDENTITY" in src
    assert "photo_stage_identity" in src
    assert "photo_stage_scope_keys" in src
    assert "_stage_identities.json" in src
    # 记录失败不能把成功的 stage 判死
    assert "PHOTO_STAGE_IDENTITY_FAIL" in src


def test_photo_chain_does_not_claim_minimal_rerun():
    """照片链**不应该**有 plan/裁剪 —— 这是有意的设计选择，不是遗漏。"""
    src = open(os.path.join(ROOT, "photo_restoration_pipeline", "main.py"),
               encoding="utf-8").read()
    assert "build_reuse_plan" not in src, (
        "照片链是单张秒级处理，不需要最小重算；如果有，说明有人误加了")
