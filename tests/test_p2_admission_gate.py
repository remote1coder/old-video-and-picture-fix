# -*- coding: utf-8 -*-
"""P2：`run_stage` 的**可选**资源准入闸门。

实施包 P2 原文：「保留现有 FrameRing 兼容入口，但让新执行组经同一
governor 准入。」

三条判据：

1. **不注入 governor 时零行为改变** —— 现有链路完全不受影响；
2. **余量不足** → 明确返回 85，且**不启动子进程**（不是"起来再失败"）；
3. **余量充足** → 正常放行（这里用假 run_managed 拦截，不真跑 stage）。
"""
import os
import sys

import pytest

import conftest  # noqa: F401

import main as vmain  # noqa: E402

from governor import ResourceGovernor  # noqa: E402

MB = 1 << 20


def _job(tmp_path):
    src = tmp_path / "src.mov"
    src.write_bytes(b"SRC")
    return {"job_id": "jobGate", "input_path": str(src),
            "output_path": str(tmp_path / "out.mkv")}


def test_no_governor_means_no_change(tmp_path):
    """不注入 `_governor`：仍然先走"脚本不存在 → 80"，闸门没有抢先介入。"""
    rc, err = vmain.run_stage("stage_does_not_exist.py", _job(tmp_path),
                              {}, str(tmp_path / "cfg.yaml"))
    assert rc == 80, (rc, err)
    assert "Script not found" in err


def test_gate_refuses_when_budget_short(tmp_path):
    """余量不足：返回 85，且**没有**启动子进程。"""
    gov = ResourceGovernor(ram_budget=8 * MB, vram_budget=1 * MB)
    starts = []
    real_popen = None

    params = {
        "_governor": gov,
        # 申请 64MB，远超 8MB 预算 → 立即拒（不是等到超时）
        "_governor_stage_ram": 64 * MB,
        "_governor_stage_vram": 0,
        "_governor_wait_s": 0.1,
    }
    rc, err = vmain.run_stage("stage1_preprocess.py", _job(tmp_path),
                              params, str(tmp_path / "cfg.yaml"))
    assert rc == 85, (rc, err)
    assert "resource admission refused" in err, err
    # 关键：拒绝发生在**起进程之前**
    st = gov.stats()
    assert st["rejected"] >= 1, st
    assert st["live"] == 0, st


def test_gate_passes_and_holds_the_reservation(tmp_path, monkeypatch):
    """余量充足：放行，且预约**覆盖整个 stage 执行**（R26-A05）。

    旧语义是"借了立刻还"：受控执行到启动子进程之前 `live` 就已经归 0，
    等于从未约束过真正的工作。现在预约持有到 stage 结束。
    """
    import cancel_guard

    gov = ResourceGovernor(ram_budget=128 * MB, vram_budget=8 * MB,
                           min_forward_ram=1 * MB)
    seen = {}

    def fake_run(cmd, **kwargs):
        class _R:
            returncode = 0
            stdout = "ok"
            stderr = ""
        seen["called"] = True
        # R26-A05：执行期间预约**必须还在占着**（旧断言是 == 0）
        seen["live_during_run"] = gov.stats()["live"]
        seen["ram_during_run"] = gov.stats()["ram_used"]
        return _R()

    monkeypatch.setattr(cancel_guard, "run_managed", fake_run)
    params = {
        "_governor": gov,
        "_governor_stage_ram": 32 * MB,
        "_governor_stage_vram": 2 * MB,
        "_governor_wait_s": 1.0,
    }
    rc, err = vmain.run_stage("stage1_preprocess.py", _job(tmp_path),
                              params, str(tmp_path / "cfg.yaml"))
    assert seen.get("called") is True, "闸门放行后应当真的去起子进程"
    assert seen.get("live_during_run") == 1, (
        "执行期间预约必须还在；live=0 说明又是'借完立刻还'")
    assert seen.get("ram_during_run") == 32 * MB, seen
    assert rc == 0, (rc, err)
    st = gov.stats()
    assert st["live"] == 0 and st["balanced"], st


def test_gate_respects_minimum_forward_unit(tmp_path):
    """最小可前进单元：申请量低于它 → 属于"不该开工"，同样 85 拒绝。"""
    gov = ResourceGovernor(ram_budget=128 * MB, vram_budget=8 * MB,
                           min_forward_ram=16 * MB)
    params = {
        "_governor": gov,
        "_governor_stage_ram": 1 * MB,      # 低于 16MB 的下限
        "_governor_stage_vram": 0,
        "_governor_wait_s": 0.1,
    }
    rc, err = vmain.run_stage("stage1_preprocess.py", _job(tmp_path),
                              params, str(tmp_path / "cfg.yaml"))
    assert rc == 85, (rc, err)
    assert "resource admission refused" in err
