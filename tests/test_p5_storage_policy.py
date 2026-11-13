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


# --------------------------------------------------------------------------
# 6) 代码级守卫：**没有第二条路**能打开有损
# --------------------------------------------------------------------------
def test_no_stage_can_turn_on_lossy_by_itself():
    """有损只能由 `storage_policy` 判定，stage 里不许出现自己的开关。

    G10 的负半是"无显式许可时……不会启用有损"。光有 `resolve()` 的拒绝
    还不够 —— 如果某个 stage 自己读 `intermediate_storage` 或写死
    `mode: lossy`，那它就能绕过守卫。这条用**静态扫描**把这个口子钉住：
    除 `main.py`（守卫本身）与 `storage_policy.py`（判定实现）之外，
    任何 stage 文件都不许出现这些关键字。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    allowed = {
        os.path.join(root, "shared_modules", "storage_policy.py"),
        os.path.join(root, "video_restoration_pipeline", "main.py"),
        os.path.join(root, "photo_restoration_pipeline", "main.py"),
        os.path.join(root, "shared_modules", "write_ledger.py"),
        # `stage_deps` 只是把这些键**声明**为非语义（不进参数身份），
        # 它不解析策略、更不能开启有损 —— 真正的决策点仍只有
        # `storage_policy` 与两个 main。把它排除在外是准确的，不是放水。
        os.path.join(root, "shared_modules", "stage_deps.py"),
    }
    offenders = []
    for sub in ("video_restoration_pipeline", "photo_restoration_pipeline",
                "shared_modules"):
        base = os.path.join(root, sub)
        for dirpath, _dirs, files in os.walk(base):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                p = os.path.join(dirpath, fn)
                if os.path.abspath(p) in {os.path.abspath(a) for a in allowed}:
                    continue
                try:
                    src = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                for kw in ("intermediate_storage", "lossy_allowed",
                           "max_lossy_generations"):
                    if kw in src:
                        offenders.append(f"{os.path.relpath(p, root)}: {kw}")
    assert not offenders, (
        "这些文件自己碰了存储策略配置 —— 有损必须只有一个决策点：\n  "
        + "\n  ".join(offenders))


# --------------------------------------------------------------------------
# 7) 低磁盘不得成为"启用有损"的理由（G10 的负半，行为级）
# --------------------------------------------------------------------------
def test_low_disk_refuses_instead_of_switching_to_lossy(tmp_path, monkeypatch):
    """磁盘不够时只能**停下**（87），不能自己换成有损中间件继续跑。

    G10：「无显式许可时模拟OOM/低磁盘/编码失败，确认不会启用有损或物化」。
    这条把"低磁盘"这条真实路径走一遍：磁盘门在策略之前，返回 87，
    而且**一个 stage 都没启动**、策略仍是 lossless。
    """
    import main as vmain

    monkeypatch.setattr(vmain, "check_wsl_space", lambda *a, **k: False)
    started = []
    monkeypatch.setattr(vmain, "run_stage",
                        lambda *a, **k: (started.append(a[0] if a else ""),
                                         (0, ""))[1])

    src = tmp_path / "src.mov"
    src.write_bytes(b"SRC" * 64)
    job = {"job_id": "jobDisk", "input_path": str(src),
           "output_path": str(tmp_path / "out.mkv")}

    class _Log:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
        def critical(self, *a, **k): pass
        def debug(self, *a, **k): pass

    cfg = {"global": {"video_encoder": "libsvtav1"}, "ai": {}}
    rc, err = vmain.process_job(job, cfg, {"chain": None, "stage": None}, _Log())
    assert rc == 87, (rc, err)
    assert "disk" in (err or "").lower(), err
    assert started == [], f"低磁盘时还在跑 stage：{started}"
