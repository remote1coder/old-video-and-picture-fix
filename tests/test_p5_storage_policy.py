# -*- coding: utf-8 -*-
"""P5：显式中间存储策略（G10 的两半）。

## 为什么要有这一层

中间产物的物化规则原来散在各 stage 里：一律无损、放 temp、按 keep_steps 清。
没有一处能回答"这批活允许有损吗""允许的话哪一步会掉精度""这份缓存还能不能用"。

P5 把决策收到 `shared_modules/storage_policy.py`，并且**有损必须显式授权**。

## G10 的通过条件是两半

1. **没有显式许可时**（默认）：任何路径都不得启用有损 —— 包括 OOM、
   低磁盘、编码失败这些"看起来不得不降级"的时刻；
2. **显式许可后**：codec / 采样 / 位深 / 质量 / 代数必须与**运行预览**一致；
   同计划可复用、改质量失效、损坏缓存拒绝。

本文件逐条钉住这两半，尤其是"拒绝"的方向 —— 一条只会在放开时生效的
守卫等于没有。
"""
import os

import pytest

import conftest  # noqa: F401

import storage_policy as SP


# --------------------------------------------------------------------------
# 1) 默认 = 旧行为（G10：旧配置仍走旧行为）
# --------------------------------------------------------------------------
def test_defaults_are_lossless_and_match_old_behaviour():
    pol, problems = SP.resolve({}, {})
    assert problems == []
    assert pol["mode"] == "lossless"
    assert pol["lossy_allowed"] is False
    assert pol["max_lossy_generations"] == 0
    assert SP.storage_block(pol)["mode"] == "lossless"
    assert SP.lossy_generations(pol) == 0
    # 预览必须明确说"无损、与改动前一致"，不能沉默
    assert "lossless" in SP.preview_lines(pol)[0]


def test_missing_config_section_is_still_lossless():
    pol, problems = SP.resolve({}, {"global": {}})
    assert problems == [] and pol["mode"] == "lossless"


# --------------------------------------------------------------------------
# 2) 无许可不得启用有损（**这是 G10 的负半**）
# --------------------------------------------------------------------------
def test_lossy_without_permission_is_refused():
    pol, problems = SP.resolve(
        {"intermediate_storage": {"mode": "lossy", "codec": "libx264",
                                 "quality": 20,
                                 "max_lossy_generations": 1}},
        {})
    assert problems, "没有 lossy_allowed 却放行了有损模式"
    assert any("lossy_allowed" in p for p in problems), problems


def test_lossy_allowed_as_string_false_is_still_refused():
    """`"false"` 不能被当成真 —— 这是 R19-B06 的老毛病。"""
    _pol, problems = SP.resolve(
        {"intermediate_storage": {"mode": "lossy", "lossy_allowed": "false",
                                 "codec": "libx264", "quality": 20,
                                 "max_lossy_generations": 1}}, {})
    assert problems


def test_lossy_without_quality_is_refused():
    _pol, problems = SP.resolve(
        {"intermediate_storage": {"mode": "lossy", "lossy_allowed": True,
                                 "codec": "libx264",
                                 "max_lossy_generations": 1}}, {})
    assert any("quality" in p for p in problems), problems


def test_lossy_without_generation_budget_is_refused():
    _pol, problems = SP.resolve(
        {"intermediate_storage": {"mode": "lossy", "lossy_allowed": True,
                                 "codec": "libx264", "quality": 20,
                                 "max_lossy_generations": 0}}, {})
    assert any("max_lossy_generations" in p for p in problems), problems


def test_lossless_with_nonzero_generations_is_refused():
    _pol, problems = SP.resolve(
        {"intermediate_storage": {"mode": "lossless",
                                 "max_lossy_generations": 2}}, {})
    assert problems


def test_lossless_with_a_lossy_codec_is_refused():
    """无损模式里写 libx264 = 自相矛盾，不能"按 codec 执行"就放过。"""
    _pol, problems = SP.resolve(
        {"intermediate_storage": {"mode": "lossless", "codec": "libx264"}}, {})
    assert problems


def test_unknown_mode_is_refused_not_defaulted():
    _pol, problems = SP.resolve({"intermediate_storage": {"mode": "lösy"}}, {})
    assert problems


# --------------------------------------------------------------------------
# 3) 显式许可后：预览必须逐次列出转换
# --------------------------------------------------------------------------
def test_preview_lists_every_lossy_conversion():
    pol, problems = SP.resolve(
        {"intermediate_storage": {"mode": "lossy", "lossy_allowed": True,
                                 "codec": "libx264", "quality": 20,
                                 "pix_fmt": "yuv420p10le",
                                 "max_lossy_generations": 2}}, {})
    assert problems == []
    lines = SP.preview_lines(pol)
    assert "LOSSY" in lines[0]
    # 代数 2 → 必须列出 #1 与 #2
    assert any("#1" in ln for ln in lines)
    assert any("#2" in ln for ln in lines)
    # 必须写明"无损结论不可复用"
    assert any("CANNOT be reused" in ln for ln in lines), lines


def test_lossy_generations_is_the_declared_budget():
    pol, _ = SP.resolve(
        {"intermediate_storage": {"mode": "lossy", "lossy_allowed": True,
                                 "codec": "libx264", "quality": 20,
                                 "max_lossy_generations": 3}}, {})
    assert SP.lossy_generations(pol) == 3


# --------------------------------------------------------------------------
# 4) 缓存身份：有损/无损结论不可互换；改质量必须失效
# --------------------------------------------------------------------------
def _policy(**kw):
    base = {"intermediate_storage": {"mode": "lossless"}}
    base["intermediate_storage"].update(kw)
    return SP.resolve(base, {})[0]


def test_legacy_sidecar_without_storage_block_is_reusable_only_losslessly():
    """旧侧车（没有 storage 块）只能是历史的无损产物。"""
    assert SP.cache_mismatch(None, _policy()) == ""
    why = SP.cache_mismatch(None, _policy(mode="lossy", lossy_allowed=True,
                                          codec="libx264", quality=20,
                                          max_lossy_generations=1))
    assert why and "no storage record" in why, why


def test_lossy_artifact_cannot_be_reused_by_a_lossless_plan():
    lossy = _policy(mode="lossy", lossy_allowed=True, codec="libx264",
                    quality=20, max_lossy_generations=1)
    rec = SP.storage_block(lossy)
    why = SP.cache_mismatch(rec, _policy())
    assert why and "mode" in why, why


def test_lossless_artifact_cannot_be_reused_by_a_lossy_plan():
    rec = SP.storage_block(_policy())
    lossy = _policy(mode="lossy", lossy_allowed=True, codec="libx264",
                    quality=20, max_lossy_generations=1)
    why = SP.cache_mismatch(rec, lossy)
    assert why and "mode" in why, why


def test_changing_quality_invalidates_the_cache():
    p20 = _policy(mode="lossy", lossy_allowed=True, codec="libx264",
                  quality=20, max_lossy_generations=1)
    p28 = _policy(mode="lossy", lossy_allowed=True, codec="libx264",
                  quality=28, max_lossy_generations=1)
    why = SP.cache_mismatch(SP.storage_block(p20), p28)
    assert why and "quality" in why, why


def test_same_lossy_plan_reuses():
    p = _policy(mode="lossy", lossy_allowed=True, codec="libx264",
                quality=20, max_lossy_generations=1)
    assert SP.cache_mismatch(SP.storage_block(p), p) == ""


def test_different_generation_count_invalidates():
    p1 = _policy(mode="lossy", lossy_allowed=True, codec="libx264",
                 quality=20, max_lossy_generations=1)
    p2 = _policy(mode="lossy", lossy_allowed=True, codec="libx264",
                 quality=20, max_lossy_generations=2)
    why = SP.cache_mismatch(SP.storage_block(p1), p2)
    assert why and "lossy_generations" in why, why


# --------------------------------------------------------------------------
# 5) 接进真实入口：策略不成立时**开跑前**就拒绝
# --------------------------------------------------------------------------
def test_main_refuses_lossy_without_permission(tmp_path, monkeypatch):
    import main as vmain

    started = []
    monkeypatch.setattr(vmain, "run_stage",
                        lambda *a, **k: (started.append(a[0] if a else ""),
                                         (0, ""))[1])
    src = tmp_path / "src.mov"
    src.write_bytes(b"SRC" * 64)
    job = {"job_id": "jobSP", "input_path": str(src),
           "output_path": str(tmp_path / "out.mkv")}

    class _Log:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
        def critical(self, *a, **k): pass
        def debug(self, *a, **k): pass

    # 必须先过 work_format 那道门（它要求声明编码器），否则测不到存储策略
    cfg = {"global": {"video_encoder": "libsvtav1"}, "ai": {},
           "intermediate_storage": {"mode": "lossy", "codec": "libx264",
                                    "quality": 20,
                                    "max_lossy_generations": 1}}
    rc, err = vmain.process_job(job, cfg, {"chain": None, "stage": None}, _Log())
    assert rc == 85, (rc, err)
    assert "storage policy" in err, err
    assert started == [], f"拒绝之前不该跑任何 stage：{started}"


def test_main_accepts_lossless_default(tmp_path, monkeypatch):
    import main as vmain

    started = []

    def _stage(*a, **k):
        started.append(a[0] if a else "")
        return 0, ""

    monkeypatch.setattr(vmain, "run_stage", _stage)
    src = tmp_path / "src.mov"
    src.write_bytes(b"SRC" * 64)
    job = {"job_id": "jobSP2", "input_path": str(src),
           "output_path": str(tmp_path / "out.mkv")}

    class _Log:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
        def critical(self, *a, **k): pass
        def debug(self, *a, **k): pass

    cfg = {"global": {"video_encoder": "libsvtav1"}, "ai": {}}
    rc, err = vmain.process_job(job, cfg, {"chain": None, "stage": None}, _Log())
    assert rc != 85, (rc, err)
    assert "storage policy" not in (err or ""), err
