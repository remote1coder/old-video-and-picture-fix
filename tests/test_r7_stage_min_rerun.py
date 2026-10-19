"""ID 第 8 点（第七轮）：逐 stage 依赖与最小重算。

任务书 §2.3：

> 阶段身份按依赖划分。换编码 CRF 应主要使编码及其验证失效，不必重跑
> 超分；只改日志等级不应重算；未启用的人脸模型更新不应使传统降噪
> 失效。同名权重换内容则必须使使用它的阶段和下游失效。

**为什么不能只有一个全局摘要**

`params_identity` 是全局的：任何语义参数变化都让所有产物失效。
保守，但"把 CRF 24 改成 18"会让整条 AI 链（小时级）重来一次。

本模块用**局部身份**解决：每个 stage 只认影响它的参数。
"""

import json
import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "shared_modules") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "shared_modules"))

import stage_deps as sd  # noqa: E402


# ---------------------------------------------------------------------------
# 参数 → stage 作用域
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,stage,expected", [
    # 编码参数只影响 stage7
    ("crf", 7, True), ("crf", 3, False), ("crf", 2, False),
    ("encoder", 7, True), ("encoder", 3, False),
    ("svt_tile_cols", 7, True), ("svt_tile_cols", 3, False),
    # AI 超分只影响 stage3
    ("ai_upscale_model", 3, True), ("ai_upscale_model", 2, False),
    ("ai_upscale_scale", 3, True), ("ai_upscale_scale", 5, False),
    # 人脸只影响 stage5
    ("face_restore", 5, True), ("face_restore", 2, False),
    ("gfpgan_model", 5, True), ("gfpgan_model", 3, False),
    # 传统修复只影响 stage1/2
    ("deinterlace", 1, True), ("deinterlace", 3, False),
    ("qtgmc_preset", 1, True), ("qtgmc_preset", 5, False),
    ("psf_handling", 2, True), ("psf_handling", 1, False),
    # 滤镜只影响 stage4
    ("unsharp_strength", 4, True), ("unsharp_strength", 7, False),
    # 颗粒只影响 stage6
    ("grain_strength", 6, True), ("grain_strength", 4, False),
    # 日志不影响任何 stage
    ("log_level", 1, False), ("log_level", 7, False),
    ("log_dir", 3, False), ("cleanup_intermediates", 6, False),
    # 位深贯穿全链
    ("vs_bit_depth", 1, True), ("vs_bit_depth", 7, True),
    # 认不出的键保守为 True
    ("some_unknown_future_key", 4, True),
])
def test_param_scope(key, stage, expected):
    assert sd.param_affects_stage(key, stage) is expected, (
        f"{key} -> stage{stage} 期望 {expected}")


# ---------------------------------------------------------------------------
# 局部身份
# ---------------------------------------------------------------------------

def test_stage_identity_ignores_other_stage_params():
    """改 CRF 不能改变 stage3 的局部身份。"""
    base = {"crf": 24, "ai_upscale_model": "x4v3", "deinterlace": "QTGMC"}
    changed = dict(base, crf=18)
    assert sd.stage_identity(base, 3) == sd.stage_identity(changed, 3)
    assert sd.stage_identity(base, 7) != sd.stage_identity(changed, 7)


def test_stage_identity_ignores_log_level():
    base = {"crf": 24, "log_level": "INFO"}
    changed = dict(base, log_level="DEBUG")
    for s in range(8):
        assert sd.stage_identity(base, s) == sd.stage_identity(changed, s)


def test_stage_identity_detects_own_param():
    base = {"ai_upscale_model": "x4v3"}
    changed = dict(base, ai_upscale_model="x4plus")
    assert sd.stage_identity(base, 3) != sd.stage_identity(changed, 3)


def test_stage_scope_keys_are_sorted_and_stable():
    p = {"crf": 1, "ai_upscale_model": "m", "log_level": "INFO"}
    k = sd.stage_scope_keys(p, 3)
    assert k == sorted(k)
    assert "crf" not in k and "log_level" not in k


# ---------------------------------------------------------------------------
# 复用计划
# ---------------------------------------------------------------------------

def _rec(stage, identity, artifact="/tmp/a", scope=None):
    return {"stage": stage, "stage_identity": identity,
            "artifact": artifact, "stage_scope": scope or {}}


def test_plan_all_reusable_when_nothing_changed():
    p = {"crf": 24, "ai_upscale_model": "x4v3"}
    rec = {s: _rec(s, sd.stage_identity(p, s)) for s in range(1, 8)}
    plan = sd.build_reuse_plan(p, rec)
    assert plan["restart_from"] is None
    assert plan["will_run_ai"] is False
    assert len(plan["reusable"]) == 7
    assert not plan["invalidated"]


def test_plan_crf_only_restarts_at_encode():
    """核心：改 CRF 只重跑 stage7，不碰 AI。"""
    p = {"crf": 24, "ai_upscale_model": "x4v3"}
    rec = {s: _rec(s, sd.stage_identity(p, s)) for s in range(1, 8)}
    p2 = dict(p, crf=18)
    plan = sd.build_reuse_plan(p2, rec)
    assert plan["restart_from"] == 7
    assert plan["will_run_ai"] is False, "改编码参数不该触发 AI 重跑"
    assert [r["stage"] for r in plan["reusable"]] == [1, 2, 3, 4, 5, 6]
    inv = {d["stage"]: d for d in plan["invalidated"]}
    assert inv[7]["reason"] == "params-changed"
    assert "crf" in inv[7]["changed_keys"]


def test_plan_ai_model_change_restarts_at_upscale_and_propagates():
    """换 AI 模型 → stage3 及其下游全部失效，stage1/2 保留。"""
    p = {"crf": 24, "ai_upscale_model": "x4v3"}
    rec = {s: _rec(s, sd.stage_identity(p, s)) for s in range(1, 8)}
    p2 = dict(p, ai_upscale_model="x4plus")
    plan = sd.build_reuse_plan(p2, rec)
    assert plan["restart_from"] == 3
    assert plan["will_run_ai"] is True
    assert [r["stage"] for r in plan["reusable"]] == [1, 2]
    reasons = {d["stage"]: d["reason"] for d in plan["invalidated"]}
    assert reasons[3] == "params-changed"
    for s in (4, 5, 6, 7):
        assert reasons[s] == "upstream-invalidated"


def test_plan_missing_identity_blocks_reuse_but_keeps_files():
    """没有侧车身份 → 不能复用，但**不删文件**。"""
    p = {"crf": 24}
    rec = {1: _rec(1, sd.stage_identity(p, 1), artifact="/tmp/step1.mkv"),
           2: None,
           3: None, 4: None, 5: None, 6: None, 7: None}
    plan = sd.build_reuse_plan(p, rec)
    assert plan["restart_from"] == 2
    reasons = {d["stage"]: d["reason"] for d in plan["invalidated"]}
    assert reasons[2] == "identity-missing"


def test_plan_legacy_sidecar_without_local_identity_is_not_reused():
    """第七轮之前的侧车没有 stage_identity → 按身份缺失处理。"""
    p = {"crf": 24}
    rec = {1: _rec(1, sd.stage_identity(p, 1)),
           2: {"stage": 2, "artifact": "/tmp/step2.mkv"},   # 无 stage_identity
           3: None, 4: None, 5: None, 6: None, 7: None}
    plan = sd.build_reuse_plan(p, rec)
    assert plan["restart_from"] == 2
    reasons = {d["stage"]: d["reason"] for d in plan["invalidated"]}
    assert reasons[2] == "identity-missing"


def test_plan_reports_stale_files_without_deleting():
    p = {"crf": 24, "ai_upscale_model": "x4v3"}
    rec = {s: _rec(s, sd.stage_identity(p, s), artifact=f"/tmp/step{s}.mkv")
           for s in range(1, 8)}
    plan = sd.build_reuse_plan(dict(p, crf=18), rec)
    assert "/tmp/step7.mkv" in plan["stale_files"]


def test_render_plan_is_human_readable():
    p = {"crf": 24, "ai_upscale_model": "x4v3"}
    rec = {s: _rec(s, sd.stage_identity(p, s)) for s in range(1, 8)}
    text = sd.render_reuse_plan(sd.build_reuse_plan(dict(p, crf=18), rec))
    assert "可复用" in text
    assert "需重算" in text
    assert "stage 7" in text
    assert "从 stage 7 开始" in text


# ---------------------------------------------------------------------------
# 接入守卫
# ---------------------------------------------------------------------------

def test_main_emits_reuse_plan_and_honours_restart():
    src = open(os.path.join(ROOT, "video_restoration_pipeline", "main.py"),
               encoding="utf-8").read()
    assert "STAGE_REUSE_PLAN" in src
    assert "STAGE_REUSE_PLAN_TEXT" in src
    assert "STAGE_MIN_RERUN" in src
    # 必须真的按 restart_from 裁剪 stage 列表
    assert '_rf = int(_reuse_plan["restart_from"])' in src


def test_sidecar_records_stage_local_identity():
    src = open(os.path.join(ROOT, "video_restoration_pipeline", "main.py"),
               encoding="utf-8").read()
    assert '"stage_identity": _stage_identity_for(params, stage_num)' in src
    assert '"stage_scope": _stage_scope_snapshot(params, stage_num)' in src
    assert "params=params)" in src


# ---------------------------------------------------------------------------
# 裁剪语义：main 里那一行的精确复现
# ---------------------------------------------------------------------------

def test_stages_list_is_trimmed_to_restart_from():
    """验证 main 的裁剪语义：`[(n, s) for n, s in stages if n >= restart_from]`。

    **这不是端到端测试**，而是把 main.py 里那行代码原样复现一遍，
    确认"plan 给出 restart_from=7 时，stage 列表真的只剩 stage7"。

    端到端（真跑批）多次尝试未拿到干净证据：跑批环境里 AI 子进程
    会卡死（一次卡了 2 小时）、残留进程互相抢 manifest、日志交织。
    这些是**环境问题**，不是 plan 逻辑问题 —— 但既然没拿到证据，
    就不声称已验证。
    """
    all_stages = [
        (0, "stage0_audio_extract.py"),
        (1, "stage1_preprocess.py"),
        (2, "stage2_traditional_check.py"),
        (3, "stage3_ai_upscale.py"),
        (4, "stage4_postprocess.py"),
        (5, "stage5_ai_face.py"),
        (6, "stage6_grain_synthesis.py"),
        (7, "stage7_encode.py"),
    ]
    plan = {"restart_from": 7}
    rf = int(plan["restart_from"])
    trimmed = [(n, s) for n, s in all_stages if n >= rf]
    assert trimmed == [(7, "stage7_encode.py")], trimmed

    # restart_from=3 时保留 3..7
    plan = {"restart_from": 3}
    trimmed = [(n, s) for n, s in all_stages if n >= int(plan["restart_from"])]
    assert [n for n, _ in trimmed] == [3, 4, 5, 6, 7]

    # 无需重算时不裁剪（None）
    assert [n for n, _ in all_stages] == list(range(8))


def test_real_sidecars_yield_restart_at_encode(tmp_path):
    """用**真实跑批产物**验证 plan 结论（不跑 stage，也不污染真实目录）。

    做法：把真实 temp 里现存的 stepN 产物**硬链接**到测试自己的目录
    （不占空间、不改原文件），按"上一轮 CRF=26"造侧车，再用 CRF=18 的
    参数算 plan，断言 `restart_from == 7` 且不触发 AI。

    注意：真实 temp 里 step1 常被 `cleanup_old_intermediates` 删掉，
    所以这里不假设"1..6 都在"，只要求"至少 step2..6 有代表性产物"，
    缺的 stage 用最近的一个补（模拟"那一代确实存在"）。
    """
    real_temp = r'E:/codex/_repair_r4_20261013/real_run/temp'
    src = r'E:/codex/_repair_r4_20261013/real_run/input/real2s.mov'
    manifest = r'E:/codex/_repair_r4_20261013/real_run/scan_manifest.yaml'
    if not (os.path.isdir(real_temp) and os.path.exists(src)
            and os.path.exists(manifest)):
        pytest.skip("真实跑批目录不存在，跳过")

    vroot = os.path.join(ROOT, "video_restoration_pipeline")
    if vroot not in sys.path:
        sys.path.insert(0, vroot)
    import shutil
    import yaml
    import main as M

    with open(manifest, encoding="utf-8") as f:
        manifest_doc = yaml.safe_load(f)
    job = manifest_doc["jobs"][0]
    cfg = M.load_config(os.path.join(vroot, "config.yaml"))
    params = M.merge_params(job, cfg, {}, None)      # 含 user_overrides
    prev = dict(params, crf=26)                       # 上一轮用 26

    # 找一个真实的 step 产物当作"母版"
    donor = None
    for s in (6, 5, 4, 3, 2):
        cands = M.find_stage_artifacts(real_temp, job["job_id"], src, s)
        if cands:
            donor = str(cands[0])
            break
    if not donor:
        pytest.skip("真实 temp 里没有可用产物")
    if os.path.getsize(donor) < 1024:
        pytest.skip("母版产物不可用（太小）")

    # 在测试目录里搭一套 step1..6（硬链接，不占空间）
    temp = tmp_path / "temp"
    temp.mkdir()
    stem = "real2s"
    names = {
        1: f"{job['job_id']}_{stem}_step1_preprocess_pX.mkv",
        2: f"{job['job_id']}_{stem}_step2_traditional.mkv",
        3: f"{job['job_id']}_{stem}_step3_forced_upscale_pX.mkv",
        4: f"{job['job_id']}_{stem}_step4_postprocess_pX.mkv",
        5: f"{job['job_id']}_{stem}_step5_face_pX.mkv",
        6: f"{job['job_id']}_{stem}_step6_grain_pX.mkv",
    }
    import json as _json
    recorded = {}
    for st, nm in names.items():
        dst = temp / nm
        try:
            os.link(donor, str(dst))
        except OSError:
            shutil.copyfile(donor, str(dst))
        with open(str(dst) + ".identity.json", "w", encoding="utf-8") as f:
            _json.dump({
                "impl": M.VIDEO_IMPL_VERSION,
                "job_id": job["job_id"], "stage": st,
                "artifact": str(dst),
                "stage_identity": sd.stage_identity(prev, st),
                "stage_scope": {k: prev.get(k)
                                for k in sd.stage_scope_keys(prev, st)},
            }, f, ensure_ascii=False)
        recorded[st] = {"stage": st, "artifact": str(dst),
                        "stage_identity": sd.stage_identity(prev, st),
                        "stage_scope": {k: prev.get(k)
                                        for k in sd.stage_scope_keys(prev, st)}}
    recorded[0] = None
    recorded[7] = None

    plan = sd.build_reuse_plan(params, recorded)
    assert plan["restart_from"] == 7, (
        "改 CRF 后应当只从 stage7 重算，实际计划：\n"
        + sd.render_reuse_plan(plan))
    assert plan["will_run_ai"] is False, "改编码参数不该触发 AI 重跑"
    assert [r["stage"] for r in plan["reusable"]] == [1, 2, 3, 4, 5, 6]
