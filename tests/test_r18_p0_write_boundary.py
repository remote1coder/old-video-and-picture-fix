# -*- coding: utf-8 -*-
"""第十八轮 P0-A / R17-06：写入边界。

审核点名三处：
1. `run_stage` 的超长参数用裸 `mkstemp()` —— **不传 dir**，落到系统临时目录
   （Windows 上是 C 盘用户 temp），既不在授权写根内，也违反"项目内优先"。
2. bootstrap 日志目录取配置值（默认 `./logs`）→ 相对**当前工作目录**，
   同一个 run 换个 cwd 启动就写到别处，甚至写到项目外。
3. manifest 领取（= 改写 manifest 文件）早于路径校验 → "先写再验"。

修法分别是：把目录钉在 `params["temp_dir"]`（兜底项目内 `temp/`）；
把相对日志目录锚到**项目根**；领取前先判 manifest 是否在授权写根内。
"""
import os
import sys

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import main as vmain  # noqa: E402


def test_params_file_goes_into_planned_temp_dir(tmp_path):
    """超长参数文件必须落在 `temp_dir` 里，不能去系统临时目录。"""
    temp_dir = tmp_path / "job_temp"
    params = {"temp_dir": str(temp_dir), "x": "y"}
    payload = '{"big": "' + ("a" * 40000) + '"}'

    path = vmain._write_stage_params_file(params, payload)
    try:
        assert os.path.isfile(path)
        assert os.path.dirname(os.path.abspath(path)) == \
            os.path.abspath(str(temp_dir)), \
            f"参数文件跑到了 {os.path.dirname(path)}，应在 {temp_dir}"
        with open(path, encoding="utf-8") as fh:
            assert fh.read() == payload
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_params_file_falls_back_inside_project(tmp_path):
    """没有 temp_dir 时，兜底也必须留在**项目内**，不能写 C 盘。"""
    path = vmain._write_stage_params_file({}, "{}")
    try:
        proj = os.path.abspath(str(vmain.SCRIPT_DIR.parent))
        assert os.path.abspath(path).startswith(proj), \
            f"兜底路径 {path} 跑到了项目外"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_log_dir_relative_is_resolved_against_project_root():
    """相对日志目录必须锚到项目根 —— 与 cwd 无关。"""
    resolved = vmain._resolve_log_dir("./logs")
    proj = os.path.abspath(str(vmain.SCRIPT_DIR.parent))
    assert os.path.isabs(resolved)
    assert os.path.abspath(resolved) == os.path.join(proj, "logs")


def test_log_dir_absolute_is_respected(tmp_path):
    """显式配置的绝对路径原样尊重（那是用户的选择）。"""
    target = str(tmp_path / "mylogs")
    assert vmain._resolve_log_dir(target) == target


def test_log_dir_default_is_not_cwd_relative():
    """默认值（空/None）也不能退化成 cwd 相对。"""
    for raw in (None, "", "./logs"):
        resolved = vmain._resolve_log_dir(raw)
        assert os.path.isabs(resolved), f"{raw!r} 解析成了相对路径"
        assert os.path.abspath(resolved).startswith(
            os.path.abspath(str(vmain.SCRIPT_DIR.parent)))


def test_manifest_claim_happens_after_the_batch_plan_check():
    """结构守卫：claim（改写 manifest）必须排在**批次计划校验之后**。

    审核 R17-06 的原话是"manifest ownership 写入早于完整计划检查"——
    要的是**顺序**，不是给 manifest 加"必须在写根内"的约束。
    （第一版我加过那种约束，结果把"manifest 在根外、产物在根内"的合法批次
      也拒掉了，既有 5 条测试由 rc=82 变成静默 rc=0。已回退。）
    """
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "main.py"),
               encoding="utf-8", errors="replace").read()
    i_plan = src.find("BATCH_PLAN_CONFLICT")
    i_claim = src.find("run_generation = acquire_run_ownership(")
    assert i_plan > 0, "缺少批次计划冲突检查"
    assert i_claim > 0
    assert i_claim > i_plan, \
        "claim 必须在批次计划校验之后（否则计划不成立时已经改写了 manifest）"
