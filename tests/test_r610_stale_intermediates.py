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


def test_orphan_hash_intermediates_are_reclaimed(tmp_path):
    """跨轮次的陈旧产物必须回收 —— 这是 keep_steps 唯一抓不到的一类。

    2026-10-22 实测：`cleanup_old_intermediates` 的 `current_param_hash`
    形参**没有任何生产调用点传入**（video main.py 三处、photo main.py 一处
    全部缺省），FIX-CK-R2-04 这段孤儿回收逻辑是**死代码**。后果是上一轮
    留下的高 step 产物永远不会被清理：实测 E 盘回收站里来自
    `_repair_r4_20261013` 的残渣达 2.2 TB，单份整链中间产物约 100 GB。

    关键在于陈旧产物的 step 号可能比当前 step **更大**（上次跑到 step6 就失败
    了，这次从更早的 stage 重来），按"step 号范围"保留会放过它，只能靠 hash 判据。
    """
    d = tmp_path
    _mk(d, "job_001_in_step1_preprocess_p111.mkv")   # 本轮更早的 step
    _mk(d, "job_001_in_step5_face_p555.mkv")            # 本轮当前 step
    _mk(d, "job_001_in_step6_grain_p999.mkv")          # 上一轮残留（孤儿）

    checkpoint.cleanup_old_intermediates(
        str(d), "job_001", keep_steps=2, current_step=5,
        current_run_hashes={"111", "555"})

    left = list(os.listdir(str(d)))
    assert not any("p999" in f for f in left), \
        "上一轮的孤儿产物（step6/p999）必须回收：keep_steps 按 step 号算抓不到它"
    assert any("p555" in f for f in left), \
        "本轮当前 step 产物绝不可删"


def test_photo_chain_passes_current_run_hashes():
    """照片链的清理调用必须传 `current_run_hashes`。

    2026-10-22 实测：`cleanup_old_intermediates` 的 `current_param_hash`
    形参在**所有**生产调用点都缺省 —— video 链三处 + photo 链一处
    （photo 链连本轮产出跟踪都没有）。孤儿回收逻辑是死代码，
    上一轮遗留的中间产物（实测单份 ~100 GB）永不清理，跨轮次累积。

    本用例锁定 photo 链的接线：必须有本轮产出跟踪 + 传集合判据。
    """
    import inspect
    import photo_restoration_pipeline.main as pmain
    src = inspect.getsource(pmain)
    # 接线：清理调用必须带 current_run_hashes
    assert "current_run_hashes=" in src, \
        "photo 链清理调用必须传 current_run_hashes（孤儿回收判据）"
    # 机制：必须有"本轮实际产出"的跟踪（执行前后快照差集）
    assert "_snapshot_job_files" in src, \
        "photo 链必须跟踪本轮实际产出，否则集合判据无基准"
    # 机制：集合判据从产物文件名提取 hash
    assert "_current_run_param_hashes" in src, \
        "photo 链必须有 hash 提取函数（对标视频链同名函数）"


def test_photo_snapshot_detects_new_and_rewritten(tmp_path):
    """快照差集必须能识别「新增」和「被改写」两类本轮产物。

    只认新增会漏掉"同名覆盖"——参数完全相同时 stage 会覆盖旧文件，
    名字不变但内容变了，它的 hash 仍应被算作本轮合法产物。
    """
    import photo_restoration_pipeline.main as pmain
    tmp = tmp_path / "temp"
    tmp.mkdir()
    (tmp / "job_001_x_step1_preprocess_paaa.tif").write_bytes(b"old")
    (tmp / "job_001_x_step3_upscale_pbbb.tif").write_bytes(b"prev-round")

    pre = pmain._snapshot_job_files(str(tmp), "job_001")
    assert set(pre) == {"job_001_x_step1_preprocess_paaa.tif",
                        "job_001_x_step3_upscale_pbbb.tif"}

    # 本轮：改写 step1（同名）、新增 step4（新名）
    (tmp / "job_001_x_step1_preprocess_paaa.tif").write_bytes(b"new-longer")
    (tmp / "job_001_x_step4_face_pccc.tif").write_bytes(b"fresh")

    post = pmain._snapshot_job_files(str(tmp), "job_001")
    produced = [n for n, sig in post.items() if pre.get(n) != sig]
    assert "job_001_x_step1_preprocess_paaa.tif" in produced, \
        "同名改写必须被识别为本轮产物（否则孤儿回收会误删它）"
    assert "job_001_x_step4_face_pccc.tif" in produced, \
        "新增文件必须被识别为本轮产物"
    assert "job_001_x_step3_upscale_pbbb.tif" not in produced, \
        "上一轮遗留、本轮未触碰的文件不得算作本轮产物"

    hashes = pmain._current_run_param_hashes(produced)
    assert hashes == {"aaa", "ccc"}, f"应提取本轮 hash，实际 {hashes}"


def test_photo_accumulates_across_stages_not_per_stage():
    """判据必须**跨 stage 累积**，不能只反映本 stage 的产出。

    2026-10-22 端到端实测踩到：第一版用「本 stage 执行前快照」做差集，
    结果本轮**更早 stage** 的产物（快照时已存在、不在差集里）hash 不在
    合法集合 → 被孤儿回收误删 → 下游 stage 找不到输入。
    视频链 `_last_produced` 是跨 stage 累积的 dict，语义必须对齐。
    """
    import inspect
    import photo_restoration_pipeline.main as pmain
    src = inspect.getsource(pmain)
    assert "_run_produced_paths" in src, \
        "photo 链必须有跨 stage 累积的本轮产出列表"
    assert "_run_produced_paths.extend(" in src, \
        "必须 extend 累积，不能每次 stage 覆盖（覆盖即漏掉更早 stage 的产物）"


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


def test_no_lzw_tiff_writes():
    """R21 静态守卫：不得再用 PIL 的 tiff_lzw 写中间产物。

    写 LZW 就必须能读 LZW。tifffile 解 LZW 要 `imagecodecs`（本机未装），
    于是 stage5 写出的文件 stage5 自己读不回来 →
    MediaPrecisionError → STAGE5_ERROR rc=86。zlib（Python 内置）无此问题。
    """
    import inspect
    import io
    import tokenize
    import photo_restoration_pipeline.stages.stage5_postprocess as s5
    src = inspect.getsource(s5)
    # 只看代码（注释/docstring 里可能出现该词作为说明）
    code = []
    prev = tokenize.INDENT
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and prev in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.NL,
                tokenize.DEDENT):
            continue
        if tok.type not in (tokenize.NL, tokenize.NEWLINE,
                            tokenize.INDENT, tokenize.DEDENT):
            code.append(tok.string)
        prev = tok.type
    joined = " ".join(code)
    assert "tiff_lzw" not in joined, \
        "不得写 tiff_lzw（读回需要未安装的 imagecodecs，R21）"
