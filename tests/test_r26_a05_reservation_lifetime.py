# -*- coding: utf-8 -*-
"""资源预约必须**覆盖执行**，不能"借完立刻还"（R26-A05）。

复核的观察：受控执行到启动子进程之前，`live` 与 `ram_used` 已经为 0 ——
准入闸门只证明"开工前有一瞬间余量够"，从不约束真正的工作，
多 job 并发时不构成任何合计上限。

本测试把 `_run_stage_inner` 换成探针：在它执行的那一刻检查 governor
的状态，断言额度**是占着的**，并在结束后断言已释放。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared_modules"))
sys.path.insert(0, str(ROOT / "video_restoration_pipeline"))

import main as vmain  # noqa: E402
from governor import ResourceGovernor  # noqa: E402


def test_reservation_is_held_during_the_stage(monkeypatch):
    gov = ResourceGovernor(1000, 0, name="t")
    seen = {}

    def _probe(script_name, job, params, config_path):
        seen["live"] = len(gov._live)
        seen["ram_used"] = gov.stats()["ram_used"]
        seen["proof"] = list(gov._live.keys())
        return (0, "")

    monkeypatch.setattr(vmain, "_run_stage_inner", _probe)
    rc, err = vmain.run_stage(
        "stage1_preprocess.py", {"job_id": "j1", "input_path": "x"},
        {"_governor": gov, "_governor_stage_ram": 100,
         "_governor_stage_vram": 0, "_governor_wait_s": 1.0},
        "")

    assert rc == 0, err
    assert seen["live"] == 1, (
        "执行期间预约必须还在；live=0 说明又是'借完立刻还'")
    assert seen["ram_used"] == 100, seen
    assert len(gov._live) == 0, "结束后必须释放"
    assert gov.stats()["balanced"] is True


def test_refusal_is_explicit_when_budget_is_exhausted(monkeypatch):
    """余量不足 → rc=85（不降质、不硬跑）。"""
    gov = ResourceGovernor(100, 0, name="tight")
    holder = gov.reserve("other", 100, 0, wait=0)      # 已被别人占满

    def _probe(*a, **kw):
        raise AssertionError("不该跑到执行体：准入就该拒绝")

    monkeypatch.setattr(vmain, "_run_stage_inner", _probe)
    rc, err = vmain.run_stage(
        "stage1_preprocess.py", {"job_id": "j1", "input_path": "x"},
        {"_governor": gov, "_governor_stage_ram": 50,
         "_governor_wait_s": 0.05}, "")
    assert rc == 85, (rc, err)
    assert "admission refused" in err
    gov.release(holder)


def test_no_governor_means_no_gate(monkeypatch):
    """没注入 governor 时行为不变（不能因为加了闸门就要求必须注入）。"""
    monkeypatch.setattr(vmain, "_run_stage_inner",
                        lambda *a, **kw: (0, "ok"))
    rc, err = vmain.run_stage("stage2_traditional_check.py",
                              {"job_id": "j1", "input_path": "x"}, {}, "")
    assert (rc, err) == (0, "ok")
