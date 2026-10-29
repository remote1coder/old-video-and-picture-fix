# -*- coding: utf-8 -*-
"""R15-01/02/03：路径计划必须贯穿执行、发现、标记、复用。

**复核报告指出我上轮引入的三项接入回归**

| 编号 | 问题 |
|---|---|
| R15-01 | `run_plan` 把 `job.log_path`（**文件**）当 `log_dir`（目录）→ logger 拿 `.log` 去 mkdir |
| R15-02 | `find_final_output` 仍优先 `job.output_path`，不看 relocate → 执行写 B、完成标记写 A |
| R15-03 | 预检在 `JOB_START` 与 `merge_params` 之后 → "任何落盘之前"不成立；且 `check_wsl_space` 会在校验后改 temp |

本文件用**真实生产入口**（`process_job`、`find_final_output`、`get_logger`）
验证，不用内存假 logger 假装隔离。
"""
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_plan as rp  # noqa: E402
import main as vmain  # noqa: E402


def _touch(p, data=b"fixture"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return str(p)


def _plan(tmp_path, **jobkw):
    job = {"job_id": "j1", "input_path": _touch(tmp_path / "in" / "clip.mp4"),
           "output_path": str(tmp_path / "media" / "clip_restored.mkv")}
    job.update(jobkw)
    params = {"temp_dir": str(tmp_path / "tmp"),
              "log_dir": str(tmp_path / "logs")}
    return rp.resolve_run_plan(job, params, write_roots=[str(tmp_path)]), job, params


# ---------------------------------------------------------------- PATH-1

def test_path1_existing_log_file_is_not_treated_as_directory(tmp_path):
    """PATH-1：`logs/job_x.log` 已存在时，log_dir 仍是**目录**，logger 建得起来。"""
    logdir = tmp_path / "logs"
    logdir.mkdir()
    existing = logdir / "job_j1.log"
    existing.write_text('{"event": "old"}\n', encoding="utf-8")

    plan, _job, _params = _plan(
        tmp_path, log_path=str(existing))

    assert plan.ok is True, plan.errors
    assert rp.path_key(plan.get("log_dir")) == rp.path_key(str(logdir)), \
        "log_dir 必须是目录，不能是那个 .log 文件"
    assert rp.path_key(plan.get("log_file")) == rp.path_key(str(existing))

    # 真的建一次 logger：文件已存在也不应冲突
    from logger import get_logger
    lg = get_logger(log_dir=str(logdir), job_id="j1")
    lg.info("PATH1_OK", "logger created next to an existing log file")
    assert logdir.is_dir(), "log_dir 必须仍是目录"


def test_path1_log_dir_pointing_at_a_file_is_rejected(tmp_path):
    """反过来：如果上游把文件当目录传进来，预检要拦。"""
    logfile = tmp_path / "logs" / "job_j1.log"
    logfile.parent.mkdir(parents=True)
    logfile.write_text("x", encoding="utf-8")

    job = {"job_id": "j1", "input_path": _touch(tmp_path / "in" / "clip.mp4"),
           "output_path": str(tmp_path / "media" / "o.mkv")}
    plan = rp.resolve_run_plan(
        job, {"temp_dir": str(tmp_path / "tmp"), "log_dir": str(logfile)},
        write_roots=[str(tmp_path)])
    assert plan.ok is False
    assert any("log_dir 指向的是文件" in e for e in plan.errors), plan.errors


# ---------------------------------------------------------------- PATH-2

def test_path2_relocate_moves_discovery_too(tmp_path):
    """PATH-2：执行写 B 之后，`find_final_output` 必须发现 B，不能返回 A。"""
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    old_media = _touch(old_dir / "clip_restored.mkv", b"OLD")
    new_media = _touch(new_dir / "clip_restored.mkv", b"NEW")

    job = {"job_id": "j1", "input_path": _touch(tmp_path / "in" / "clip.mp4"),
           "output_path": str(old_media)}
    args = {"output": None, "relocate_output": str(new_dir)}

    plan = rp.resolve_run_plan(
        job, {"temp_dir": str(tmp_path / "tmp"),
              "log_dir": str(tmp_path / "logs")},
        write_roots=[str(tmp_path)], relocate_output=str(new_dir))
    assert plan.ok is True, plan.errors

    got = vmain.find_final_output(job, {}, args, plan=plan)
    assert got, "必须至少发现一个候选"
    assert rp.path_key(got[0]) == rp.path_key(new_media), \
        f"应优先返回搬迁后的 {new_media}，实际 {got[0]}"
    assert rp.path_key(old_media) not in [rp.path_key(x) for x in got[:1]]


def test_path2_without_plan_still_prefers_relocated_target(tmp_path):
    """即使没传 plan（例如 resume 扫描），也要按同一套逻辑推导出新位置。"""
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    new_media = _touch(new_dir / "clip_restored.mkv", b"NEW")
    old_media = _touch(tmp_path / "old" / "clip_restored.mkv", b"OLD")

    job = {"job_id": "j1", "input_path": _touch(tmp_path / "in" / "clip.mp4"),
           "output_path": str(old_media)}
    args = {"output": None, "relocate_output": str(new_dir)}

    got = vmain.find_final_output(job, {}, args)
    assert got and rp.path_key(got[0]) == rp.path_key(new_media), got


# ---------------------------------------------------------------- PATH-3

def test_path3_default_naming_matches_writer(tmp_path):
    """PATH-3：无显式输出时，计划里的默认名必须与 stage7 写的一致。

    stage7 的兜底是 `{Path(input).stem}_restored.mkv`。
    用 `{job_id}_restored.mkv` 会让执行与发现分叉（这正是 R15-02）。
    """
    inp = _touch(tmp_path / "in" / "clip.mp4")
    job = {"job_id": "j1"}
    raw, source = rp.resolve_final_media_path(
        {"job_id": "j1", "input_path": inp}, output_dir=str(tmp_path / "out"))
    assert os.path.basename(raw) == "clip_restored.mkv", raw
    assert source == rp.SOURCE_CLI


def test_path3_default_discovery_uses_same_name(tmp_path):
    """默认命名的发现结果与计划一致。"""
    inp = _touch(tmp_path / "in" / "clip.mp4")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    made = _touch(out_dir / "clip_restored.mkv", b"X")

    job = {"job_id": "j1", "input_path": inp}
    args = {"output": str(out_dir)}
    got = vmain.find_final_output(job, {}, args)
    assert got and rp.path_key(got[0]) == rp.path_key(made), got


# ---------------------------------------------------------------- PATH-5

def test_path5_shared_container_dirs_are_not_conflicts(tmp_path):
    """PATH-5：两个 job 共用 temp/log 目录**是正常的**，不算冲突。"""
    a = rp.resolve_run_plan(
        {"job_id": "a", "input_path": _touch(tmp_path / "a.mp4"),
         "output_path": str(tmp_path / "out" / "a_restored.mkv")},
        {"temp_dir": str(tmp_path / "shared_tmp"),
         "log_dir": str(tmp_path / "shared_logs")},
        write_roots=[str(tmp_path)])
    b = rp.resolve_run_plan(
        {"job_id": "b", "input_path": _touch(tmp_path / "b.mp4"),
         "output_path": str(tmp_path / "out" / "b_restored.mkv")},
        {"temp_dir": str(tmp_path / "shared_tmp"),
         "log_dir": str(tmp_path / "shared_logs")},
        write_roots=[str(tmp_path)])
    assert a.ok and b.ok
    assert rp.detect_cross_job_conflicts([a, b]) == [], \
        "共享 temp/log 目录不应被当成冲突"


def test_path5_same_media_is_a_conflict(tmp_path):
    """同一媒体被两个 job 写 → 必须报冲突。"""
    same = str(tmp_path / "out" / "same.mkv")
    a = rp.resolve_run_plan(
        {"job_id": "a", "input_path": _touch(tmp_path / "a.mp4"),
         "output_path": same},
        {"temp_dir": str(tmp_path / "t1"), "log_dir": str(tmp_path / "l1")},
        write_roots=[str(tmp_path)])
    b = rp.resolve_run_plan(
        {"job_id": "b", "input_path": _touch(tmp_path / "b.mp4"),
         "output_path": same},
        {"temp_dir": str(tmp_path / "t2"), "log_dir": str(tmp_path / "l2")},
        write_roots=[str(tmp_path)])
    conflicts = rp.detect_cross_job_conflicts([a, b])
    assert conflicts, "同一媒体必须报冲突"


def test_path5_output_overwriting_another_input_is_a_conflict(tmp_path):
    """A 的输出正好是 B 的输入 → 必须报冲突（会毁掉 B 的素材）。"""
    b_in = _touch(tmp_path / "b_in.mp4", b"B-INPUT")
    a = rp.resolve_run_plan(
        {"job_id": "a", "input_path": _touch(tmp_path / "a.mp4"),
         "output_path": b_in},
        {"temp_dir": str(tmp_path / "t1"), "log_dir": str(tmp_path / "l1")},
        write_roots=[str(tmp_path)])
    b = rp.resolve_run_plan(
        {"job_id": "b", "input_path": b_in,
         "output_path": str(tmp_path / "out" / "b_restored.mkv")},
        {"temp_dir": str(tmp_path / "t2"), "log_dir": str(tmp_path / "l2")},
        write_roots=[str(tmp_path)])
    conflicts = rp.detect_cross_job_conflicts([a, b])
    assert any("覆盖" in c for c in conflicts), conflicts


# ---------------------------------------------------------------- PATH-6

def test_path6_write_roots_requested_requires_module(tmp_path, monkeypatch):
    """PATH-6：要求限制写根时，路径模块缺失必须**硬失败**，不能静默放行。

    这里通过把 `resolve_run_plan` 的导入目标从 sys.modules 中移除来模拟
    模块缺失（真实部署里就是文件被删/路径不对）。
    """
    inp = _touch(tmp_path / "in" / "clip.mp4")
    job = {"job_id": "j1", "input_path": str(inp),
           "output_path": str(tmp_path / "out" / "o.mkv")}
    args = {"output": None, "allow_write_root": [str(tmp_path)],
            "relocate_output": None, "no_wsl_redirect": True}

    import builtins
    real_import = builtins.__import__

    def _fake_import(name, *a, **kw):
        if name == "run_plan":
            raise ImportError("simulated missing module")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.delitem(sys.modules, "run_plan", raising=False)

    rc, msg = vmain.process_job(job, {}, args, None)
    assert rc == 82, f"模块缺失时必须失败，实际 rc={rc}"
    assert "run-plan" in msg.lower() or "module missing" in msg.lower(), msg
