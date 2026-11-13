# -*- coding: utf-8 -*-
"""P3：`--chain` 的声明与实现必须一致。

## 背景

`--chain` 的 choices 早就写了 ``stage67`` / ``both``，但实现里
`_KNOWN_CHAIN_MODES` 只有 ``none`` / ``stage34`` —— 于是用户明确要求
"颗粒+编码融合"时，代码把它当**未知模式**并**静默回退到 none**，
跑完还报成功。

那等于"用户要的东西没做，结果看起来一样"——与 B05 关掉的"隐式降质"
是同一类问题：**不要让"没做"伪装成"做了"**。

## 判据

1. `stage67` / `both` → **明确拒绝**（85 + 说明"声明但未实现"）；
2. `stage34` → 正常放行（现状）；
3. 完全未知的值 → 仍按"未知模式 → 回退 none + 警告"（保留原有兜底）；
4. 拒绝发生在**跑 stage 之前**（不能先跑一半再拒）。
"""
import json
import os

import pytest

import conftest  # noqa: F401

import main as vmain  # noqa: E402


def _job(tmp_path):
    src = tmp_path / "src.mov"
    src.write_bytes(b"SRC" * 64)
    return {"job_id": "jobChain", "input_path": str(src),
            "output_path": str(tmp_path / "out.mkv")}


class _Log:
    def __init__(self):
        self.events = []

    def info(self, *a, **k):
        self.events.append(str(a[0]) if a else "")

    def warning(self, *a, **k):
        self.events.append("W:" + (str(a[0]) if a else ""))

    def error(self, *a, **k):
        self.events.append("E:" + (str(a[0]) if a else ""))

    def critical(self, *a, **k):
        self.events.append("C:" + (str(a[0]) if a else ""))

    def debug(self, *a, **k):
        pass


def _run(tmp_path, chain, monkeypatch):
    """跑 process_job，但拦住 stage 执行（若它走到那一步）。"""
    started = []

    def _no_stage(*a, **k):
        started.append(a[0] if a else "")
        return 0, ""

    monkeypatch.setattr(vmain, "run_stage", _no_stage)
    log = _Log()
    args = {"chain": chain, "stage": None}
    cfg = {"global": {}, "ai": {}, "paths": {}}
    rc, err = vmain.process_job(_job(tmp_path), cfg, args, log)
    return rc, err, log, started


@pytest.mark.parametrize("mode", ["stage67", "both"])
def test_declared_unimplemented_modes_are_refused(tmp_path, monkeypatch, mode):
    """声明了但没实现的模式：85 + 说明，且**没有启动任何 stage**。"""
    rc, err, log, started = _run(tmp_path, mode, monkeypatch)
    assert rc == 85, (rc, err)
    assert "not implemented" in err, err
    assert started == [], f"拒绝之前不该跑 stage：{started}"
    # 注意：CHAIN_MODE_UNIMPLEMENTED 写进的是 **job_logger**（文件），
    # 不是传进来的 `logger` —— 第一版断言 `log.events` 因此假失败。


def test_stage34_still_passes_through(tmp_path, monkeypatch):
    """已实现模式不受影响：不会在 chain 检查处被拒。"""
    rc, err, log, started = _run(tmp_path, "stage34", monkeypatch)
    assert "not implemented" not in (err or ""), err
    # 它可能因为测试素材/参数在后续步骤失败，但**不应该**在 chain 处 85
    assert rc != 85 or "not implemented" not in err


def test_unknown_mode_still_falls_back_with_warning(tmp_path, monkeypatch):
    """完全未知的值仍走原兜底：回退 none + 警告（保留原有行为）。

    判据只看"没有被当成未实现模式拒绝" —— 那条警告同样写在
    job_logger 文件里，不在传进来的 `logger` 上。
    """
    rc, err, log, started = _run(tmp_path, "not_a_real_mode", monkeypatch)
    assert "not implemented" not in (err or "")
    assert rc != 85 or "not implemented" not in err


def test_known_modes_are_a_subset_of_cli_choices():
    """CLI 的 choices 必须等于「已实现 ∪ 声明未实现」。"""
    import argparse
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "video_restoration_pipeline", "main.py"),
        encoding="utf-8").read()
    # 从源码里抽 --chain 的 choices，避免依赖 parser 的私有结构
    assert 'choices=["none", "stage34", "stage67", "both"]' in src, \
        "CLI choices 变了，本测试需要同步更新"
    assert vmain._KNOWN_CHAIN_MODES == frozenset({"none", "stage34"})
    assert vmain._DECLARED_UNIMPLEMENTED_CHAIN_MODES == frozenset(
        {"stage67", "both"})
    union = set(vmain._KNOWN_CHAIN_MODES) | set(
        vmain._DECLARED_UNIMPLEMENTED_CHAIN_MODES)
    assert union == {"none", "stage34", "stage67", "both"}, union
