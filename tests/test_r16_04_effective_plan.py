# -*- coding: utf-8 -*-
"""R16-04：有效计划必须与执行一致，授权根必须贯通。

审核确认三处接入遗漏：

1. 批次 `merge_params` 传空 CLI 覆盖，`resolve_run_plan` 没传
   `relocate_output`，授权根也没用与 job 执行相同的回退。
   → 两个旧输出不同但文件同名、重定位到同一目录后会碰撞，批次检查却看不见。
2. 并行 worker 的参数字典漏了 `allow_write_root` / `relocate_output`。
3. 重定向改写 `params.temp_dir` 后，二次预检仍传原 job；resolver 优先
   `job.temp_dir` → 批准旧路径、实际使用另一个路径。

本文件先用**可精确定位**的用例覆盖 1、3 与"共享目录不算冲突"，
再用真实 `main.py --workers 2` 覆盖 2。
"""
import io
import os
import subprocess
import sys

import pytest
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_plan as rp  # noqa: E402
from main import (  # noqa: E402
    _build_cli_overrides, _plan_write_roots, _resolve_plan_for_job)


def _job(jid, inp, out=None, temp=None):
    return {"job_id": jid, "input_path": inp,
            "output_path": out or "", "temp_dir": temp or "",
            "detected_class": "HDV_AVCHD", "confidence": 0.9,
            "status": "PENDING", "recommended_params": {},
            "user_overrides": {}}


class _Args:
    """模拟 argparse.Namespace 与 dict 两种形态（worker 收到的是 dict）。"""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ---------------------------------------------------------------- 重定向覆盖

def test_redirect_temp_outranks_job_temp(tmp_path):
    """重定向后的实际 temp 必须压过 job.temp_dir（审核第 3 项）。"""
    # 真实存在的文件：否则"输入不存在"会污染 ok，看不出 temp 的效果
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    job_temp = tmp_path / "job_temp"
    job_temp.mkdir()
    redirected = tmp_path / "redirected_temp"

    job = _job("j1", str(src), out=str(tmp_path / "a.mkv"), temp=str(job_temp))
    # log_dir 必须也在根内：默认 ./logs 会额外触发越界，掩盖 temp 的效果
    params = {"temp_dir": str(redirected), "log_dir": str(tmp_path / "logs")}

    # 旧行为：不传覆盖 → job.temp_dir 胜出
    p_old = rp.resolve_run_plan(job, params, write_roots=None)
    assert p_old.get("temp_dir") == rp.resolve_path(str(job_temp))
    assert p_old.source_of("temp_dir") == rp.SOURCE_JOB

    # 新行为：传重定向覆盖 → 实际路径胜出
    p_new = rp.resolve_run_plan(job, params, write_roots=None,
                                redirect_overrides={
                                    "temp_dir": str(redirected)})
    assert p_new.get("temp_dir") == rp.resolve_path(str(redirected))
    assert p_new.source_of("temp_dir") == rp.SOURCE_REDIRECT


def test_redirect_temp_outside_root_is_rejected_even_if_job_temp_inside(tmp_path):
    """job.temp 在根内、实际 temp 在根外 → **必须拒绝**。

    这条正是"检查旧路径后放行"的反例：只按 job.temp_dir 判定会通过，
    但真正写入的是根外的另一个目录。
    """
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "job_temp"
    inside.mkdir()
    outside = tmp_path / "outside_temp"

    job = _job("j1", str(src), out=str(root / "a.mkv"), temp=str(inside))
    params = {"temp_dir": str(outside), "log_dir": str(root / "logs")}

    # 不传覆盖：只看根内的 job.temp_dir → **假通过**
    p_old = rp.resolve_run_plan(job, params, write_roots=[str(root)])
    assert p_old.ok is True, \
        f"旧路径（根内）会通过，这正是问题所在：{p_old.errors}"

    # 传覆盖：实际路径在根外 → 必须拒绝
    p_new = rp.resolve_run_plan(job, params, write_roots=[str(root)],
                                redirect_overrides={"temp_dir": str(outside)})
    assert p_new.ok is False, p_new.as_dict()
    assert p_new.errors, "拒绝了却没有给出原因"


def test_redirect_output_outranks_relocate_and_job(tmp_path):
    """重定向后的实际输出也最强（WSL 重定向会改写它）。"""
    job = _job("j1", "/src/a.mp4", out="/job_out/a.mkv")
    params = {"output_dir": "/cli_out"}
    p = rp.resolve_run_plan(
        job, params, write_roots=None,
        relocate_output="/relocated",
        redirect_overrides={"output_path": "/redirected/a.mkv"})
    assert p.get("output_path") == rp.resolve_path("/redirected/a.mkv")
    assert p.source_of("output_path") == rp.SOURCE_REDIRECT


# ---------------------------------------------------------------- 授权根

def test_write_roots_from_config_when_cli_absent():
    """CLI 未给授权根时，批次与单 job 必须取 config 的同一集合。"""
    cfg = {"allowed_write_roots": ["/a", "/b"]}
    roots_cli_absent = _plan_write_roots(_Args(), cfg)
    assert roots_cli_absent == ["/a", "/b"]

    # 同一份 config 下，无论用 Namespace 还是 dict，结果一致
    roots_dict = _plan_write_roots({"allow_write_root": None}, cfg)
    assert roots_dict == roots_cli_absent


def test_cli_root_outranks_config():
    cfg = {"allowed_write_roots": ["/from_config"]}
    assert _plan_write_roots(_Args(allow_write_root="/from_cli"), cfg) == \
        ["/from_cli"]


def test_no_roots_anywhere_means_unrestricted_none():
    assert _plan_write_roots(_Args(), {}) is None


# ---------------------------------------------------------------- 共享 helper

def test_cli_overrides_are_identical_for_namespace_and_dict():
    """worker 收到 dict、父进程是 Namespace —— 两者必须得到同一份覆盖。"""
    ns = _Args(output="/o", no_wsl_redirect=True, keep_intermediates=True,
               gpu="cuda", ai_enabled=True, ai_upscale_factor=2,
               face_restore="gfpgan", ai_model="x4plus",
               force_allow_banned_upscale=True)
    d = {"output": "/o", "no_wsl_redirect": True, "keep_intermediates": True,
         "gpu": "cuda", "ai_enabled": True, "ai_upscale_factor": 2,
         "face_restore": "gfpgan", "ai_model": "x4plus",
         "force_allow_banned_upscale": True}
    assert _build_cli_overrides(ns) == _build_cli_overrides(d)


def test_batch_and_single_job_resolve_the_same_plan(tmp_path):
    """批次预检与单 job 执行走**同一个** helper → 同一份计划。"""
    root = tmp_path / "root"
    root.mkdir()
    job = _job("j1", "/src/a.mp4", out=str(root / "a.mkv"),
                temp=str(root / "t"))

    class _Log:
        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

    args = _Args(output=str(root), allow_write_root=str(root),
                 config="x.yaml")
    cfg = {"allowed_write_roots": [str(root)]}

    p1_params, p1, ok1 = _resolve_plan_for_job(job, cfg, args, _Log())
    p2_params, p2, ok2 = _resolve_plan_for_job(job, cfg, args, _Log())
    assert ok1 and ok2 and p1 is not None and p2 is not None
    assert p1.get("output_path") == p2.get("output_path")
    assert p1.get("temp_dir") == p2.get("temp_dir")
    assert p1.write_roots == p2.write_roots == [str(root)]


# ---------------------------------------------------------------- 冲突判定

def test_same_basename_from_different_dirs_relocated_to_one_dir_conflicts(tmp_path):
    """不同源目录的同名输出重定位到同一目录 → 派发前必须拒绝。

    这是审核第 1 项的具体反例：不传 relocate_output 时两个输出路径不同，
    看起来没冲突；一旦重定位就撞名。
    """
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    j1 = _job("j1", str(d1 / "clip.mp4"), out=str(d1 / "clip.mp4"))
    j2 = _job("j2", str(d2 / "clip.mp4"), out=str(d2 / "clip.mp4"))
    params = {}

    # 不重定位：输出路径不同 → 无冲突
    p1 = rp.resolve_run_plan(j1, params, write_roots=None)
    p2 = rp.resolve_run_plan(j2, params, write_roots=None)
    assert rp.detect_cross_job_conflicts([p1, p2]) == []

    # 重定位到同一目录：同 basename 撞名 → 必须有冲突
    r1 = rp.resolve_run_plan(j1, params, write_roots=None,
                             relocate_output=str(tmp_path / "reloc"))
    r2 = rp.resolve_run_plan(j2, params, write_roots=None,
                             relocate_output=str(tmp_path / "reloc"))
    conflicts = rp.detect_cross_job_conflicts([r1, r2])
    assert conflicts, "重定位后撞名却没检测出来 = 审核第 1 项仍然存在"


def test_shared_temp_and_log_dirs_are_not_conflicts(tmp_path):
    """共享 temp/log 容器目录可以共用；独占文件才需要隔离（不许误杀）。"""
    shared_tmp = tmp_path / "shared_temp"
    shared_log = tmp_path / "shared_log"
    shared_tmp.mkdir()
    shared_log.mkdir()
    j1 = _job("j1", "/src/a.mp4", out=str(tmp_path / "o" / "a.mkv"),
              temp=str(shared_tmp))
    j2 = _job("j2", "/src/b.mp4", out=str(tmp_path / "o" / "b.mkv"),
              temp=str(shared_tmp))
    params = {"log_dir": str(shared_log)}
    p1 = rp.resolve_run_plan(j1, params, write_roots=None)
    p2 = rp.resolve_run_plan(j2, params, write_roots=None)
    assert rp.detect_cross_job_conflicts([p1, p2]) == []


# ------------------------------------------------- R16-05 手动 FPS 贯通

def test_fps_interpret_cli_reaches_cli_overrides():
    """`--fps-interpret` 必须进 cli_overrides，并标 `_fps_from_user`。

    审核确认：`_fps_from_user` 在 stage3 被读取，但 main 从来没有设置入口，
    于是"用户要求重解释时间轴"根本传不下去。
    """
    args = _Args(manifest="x.yaml", fps_interpret="30000/1001")
    ov = _build_cli_overrides(args)
    assert ov.get("fps_interpret") == "30000/1001"
    assert ov.get("_fps_from_user") is True


def test_fps_interpret_absent_by_default():
    ov = _build_cli_overrides(_Args(manifest="x.yaml"))
    assert "fps_interpret" not in ov
    assert "_fps_from_user" not in ov


def test_fps_interpret_survives_worker_arg_dict():
    """worker 收到的是 dict，父进程是 Namespace —— 两者必须等价。"""
    ns = _Args(fps_interpret="25", output="/o")
    d = {"fps_interpret": "25", "output": "/o"}
    assert _build_cli_overrides(ns) == _build_cli_overrides(d)


def test_argparse_exposes_fps_interpret():
    """真实 argparse 必须认这个参数（防止只改了 override 没加开关）。"""
    import sys as _sys
    _sys.path.insert(0, os.path.join(_ROOT, "video_restoration_pipeline"))
    import main as _m
    _old = _sys.argv
    try:
        _sys.argv = ["main.py", "--manifest", "x.yaml",
                     "--fps-interpret", "30000/1001"]
        a = _m.parse_args()
        assert getattr(a, "fps_interpret", None) == "30000/1001"
    finally:
        _sys.argv = _old


def test_stage3_reinterpret_overrides_probe():
    """stage3 必须让用户的**重解释**压过 probe，并如实标 user-assumed。"""
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage3_ai_upscale.py"),
               encoding="utf-8", errors="replace").read()
    assert "STAGE3_FPS_USER_REINTERPRET" in src, \
        "重解释必须有自己的事件（否则无法与 probe 结果区分）"
    assert 'params.get("fps_interpret")' in src
    assert '_fps_source = "user-assumed"' in src
    # 反例：这段必须在 `if not _fps_ok:` 之外（否则 probe 成功就永远走不到）
    _i_re = src.find('params.get("fps_interpret")')
    _i_blk = src.find("if not _fps_ok:")
    assert _i_re > _i_blk, "重解释逻辑落在了 probe 失败分支里"
