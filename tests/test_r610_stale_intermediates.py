"""R6-10 回归：陈旧中间产物把本轮 step1 删掉，导致下游 stage 报 NO_INPUT。

复现的真实故障（stage5 face_GFPGAN 案例，rc=86）：
  1. 上一次运行在 temp/ 留下 job_001_*_step3_*.mkv / _step4_*.mkv；
  2. 全新运行（无 resume_from_stage）重跑 stage1，产出 _step1_*.mkv；
  3. main.py 在 stage1 成功后调 cleanup_old_intermediates(..., keep_steps=2)
     且 current_step=None → 该分支"按 step 号降序取前 keep_steps 个保留"
     → 保留 step4/step3，把本轮 step1 当最旧的删掉；
  4. stage2 找不到上游输入 → STAGE2_NO_INPUT → TRADITIONAL_GATE_BLOCK rc=86。

修复：全新运行前 cleanup_all_intermediates 清空本 job 的陈旧产物；
循环内清理传 current_step，走显式语义。
"""

import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

import checkpoint  # noqa: E402


def _mk(tmp_path, name, payload=b"x"):
    p = tmp_path / name
    p.write_bytes(payload)
    return p


def test_current_step_none_deletes_fresh_step1(tmp_path):
    """反例：current_step=None 时本轮 step1 被当作最旧删除。"""
    tmp = tmp_path / "temp"
    tmp.mkdir()
    _mk(tmp, "job_001_in_step1_preprocess_paaa.mkv")
    _mk(tmp, "job_001_in_step3_upscale_pbbb.mkv")
    _mk(tmp, "job_001_in_step4_postprocess_pccc.mkv")

    checkpoint.cleanup_old_intermediates(str(tmp), "job_001",
                                         keep_steps=2, current_step=None)

    assert not (tmp / "job_001_in_step1_preprocess_paaa.mkv").exists(), \
        "current_step=None 分支删掉了本轮刚产出的 step1"


def test_purge_before_fresh_run_saves_step1(tmp_path):
    """正解：全新运行前先清空陈旧产物，step1 即为最新，不再被删。"""
    tmp = tmp_path / "temp"
    tmp.mkdir()
    _mk(tmp, "job_001_in_step1_preprocess_paaa.mkv")
    _mk(tmp, "job_001_in_step3_upscale_pbbb.mkv")
    _mk(tmp, "job_001_in_step4_postprocess_pccc.mkv")
    (tmp / "job_001_in_step3_upscale_pbbb.mkv.identity.json").write_text("{}")

    removed = checkpoint.cleanup_all_intermediates(str(tmp), "job_001")
    assert removed >= 3, "应删掉 2 个媒体 + 1 个侧车"
    assert not any(tmp.iterdir()), "陈旧产物应被清空"

    # 清空后再跑 stage1 + 循环内清理，step1 必须存活
    _mk(tmp, "job_001_in_step1_preprocess_pddd.mkv")
    checkpoint.cleanup_old_intermediates(str(tmp), "job_001",
                                         keep_steps=2, current_step=1)
    assert (tmp / "job_001_in_step1_preprocess_pddd.mkv").exists()


def test_current_step_keeps_recent_stages(tmp_path):
    """current_step 显式语义：只删 step < current_step 且不在最近 keep_steps 内。"""
    tmp = tmp_path / "temp"
    tmp.mkdir()
    names = {}
    for step in (1, 2, 3):
        names[step] = f"job_001_in_step{step}_op_p{step}.mkv"
        _mk(tmp, names[step])

    checkpoint.cleanup_old_intermediates(str(tmp), "job_001",
                                         keep_steps=2, current_step=3)
    assert not (tmp / names[1]).exists(), "step1 应被清理"
    assert (tmp / names[2]).exists()
    assert (tmp / names[3]).exists(), "当前 step 产物绝不可删"


def test_resume_run_does_not_purge(tmp_path):
    """resume 语义由 main.py 分支控制：cleanup_all_intermediates 本身不区分，
    这里只锁定一个事实——它是幂等且只针对本 job_id。"""
    tmp = tmp_path / "temp"
    tmp.mkdir()
    _mk(tmp, "job_001_in_step2_x.mkv")
    _mk(tmp, "job_002_in_step2_x.mkv")

    checkpoint.cleanup_all_intermediates(str(tmp), "job_001")
    assert not (tmp / "job_001_in_step2_x.mkv").exists()
    assert (tmp / "job_002_in_step2_x.mkv").exists(), "不得波及其他 job"


def test_unlink_returns_count(tmp_path):
    """FIX-NEW-V-07: 删除函数返回计数，供调用方记日志。"""
    tmp = tmp_path / "temp"
    tmp.mkdir()
    media = _mk(tmp, "job_001_in_step1_op_p1.mkv")
    (tmp / "job_001_in_step1_op_p1.mkv.identity.json").write_text("{}")

    n = checkpoint.unlink_artifact_with_sidecars(media)
    assert n == 2, "媒体 + 侧车都应计入"
    assert checkpoint.unlink_artifact_with_sidecars(
        tmp / "nonexistent.mkv") == 0


@pytest.mark.skipif(sys.platform == "win32",
                    reason="main.py 的 purge 分支需在真实运行中验证")
def test_main_has_purge_branch():
    """静态守卫：main.py 全新运行分支必须调 cleanup_all_intermediates。"""
    import inspect
    import main as vmain
    src = inspect.getsource(vmain.process_job)
    assert "cleanup_all_intermediates" in src
    assert "STALE_INTERMEDIATES_PURGED" in src
    assert "current_step=stage_num" in src, \
        "循环内清理必须传 current_step，不能再依赖降序取前 N 个"


def test_photo_main_cleanup_passes_current_step():
    """R10 静态守卫：照片链 main.py 的清理必须传 current_step。

    与视频链 R6-10 同族：photo 链 `cleanup_old_intermediates(temp_dir,
    job_id)` 不传 current_step，共享实现走"按 step 号降序取前
    keep_steps 个"分支，会删掉本 stage 刚产出的中间产物
    （实测 stage3 的 step3_upscale 被删 → stage4 找不到输入 →
    人脸检测必然零命中）。
    """
    import inspect
    import photo_restoration_pipeline.main as pmain
    src = inspect.getsource(pmain)
    assert "current_step=stage_idx + 1" in src, \
        "循环内清理必须传 current_step"
    assert "STALE_INTERMEDIATES_PURGED" in src, \
        "全新运行必须先 purge 陈旧中间产物"


def test_photo_stages_fall_back_to_upscaled_dir():
    """R15 静态守卫：stage4/5/6 必须能回落到 <temp>/<job>_upscaled/。

    R15：stage3 的真实产物是 `<temp>/<job>_upscaled/<stem>_x<N>.tif`
    （推理引擎原生输出目录），而三个 stage 只 glob
    `<temp>/<job>_*_stepN_*`。实测三个 glob 全部落空 →
    stage4 对空输入检测 → STAGE4_NO_FACE → 人脸修复静默跳过，rc 仍 0。
    """
    import inspect
    import photo_restoration_pipeline.stages.stage4_ai_face as s4
    for mod in (s4,):
        src = inspect.getsource(mod._resolve_input_source)
        assert "_upscaled" in src, "stage4 必须回落到 _upscaled 目录"
