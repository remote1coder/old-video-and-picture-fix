# -*- coding: utf-8 -*-
"""P3：`--chain` 的声明与实现必须一致。

## 背景（两段）

**第一段**：`--chain` 的 choices 早就写了 ``stage67`` / ``both``，但实现里
`_KNOWN_CHAIN_MODES` 只有 ``none`` / ``stage34`` —— 用户明确要求
"颗粒+编码融合"时，代码把它当**未知模式**并**静默回退到 none**，
跑完还报成功。那等于"用户要的东西没做，结果看起来一样"。

**第二段（本轮）**：S6→S7 执行组真的接上了，于是 ``stage67`` / ``both``
不再是"声明但未实现"，而是**真有实现**。本文件随之改写：

1. 四种模式全部在 `_KNOWN_CHAIN_MODES`，且等于 CLI choices；
2. `stage67` 不再被 85 拒绝；
3. `_chain67_on()` / `_chain_mode_on()` 的真值表；
4. 完全未知的值仍走"回退 none + 警告"（保留原有兜底）；
5. **融合组没有交接单时，stage7 必须停**（不能回退成经典编码，
   那会在有残留 step6 文件时静默跳过颗粒）。
"""
import os

import pytest

import conftest  # noqa: F401

import main as vmain  # noqa: E402

MODES = ("none", "stage34", "stage67", "both")


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
def test_s67_is_no_longer_refused_as_unimplemented(tmp_path, monkeypatch, mode):
    """`stage67` / `both` 现在有实现，不该再被 85 拒绝。"""
    rc, err, _log, _started = _run(tmp_path, mode, monkeypatch)
    assert rc != 85, (rc, err)
    assert "not implemented" not in (err or ""), err


def test_stage34_still_passes_through(tmp_path, monkeypatch):
    """已实现模式不受影响：不会在 chain 检查处被拒。"""
    rc, err, _log, _started = _run(tmp_path, "stage34", monkeypatch)
    assert "not implemented" not in (err or ""), err
    assert rc != 85 or "not implemented" not in err


def test_unknown_mode_still_falls_back_with_warning(tmp_path, monkeypatch):
    """完全未知的值仍走原兜底：回退 none + 警告（保留原有行为）。

    判据只看"没有被当成未实现模式拒绝" —— 那条警告同样写在
    job_logger 文件里，不在传进来的 `logger` 上。
    """
    rc, err, _log, _started = _run(tmp_path, "not_a_real_mode", monkeypatch)
    assert "not implemented" not in (err or "")
    assert rc != 85 or "not implemented" not in err


def test_known_modes_equal_cli_choices():
    """`_KNOWN_CHAIN_MODES` 必须**等于** CLI 的 choices。

    两者一旦分叉就出现两种病：choices 里有而实现不认（静默回退成 none），
    或实现认得而 choices 不让选（功能存在但用户用不到）。
    """
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "video_restoration_pipeline", "main.py"),
        encoding="utf-8").read()
    # 从源码里抽 --chain 的 choices，避免依赖 parser 的私有结构
    assert 'choices=["none", "stage34", "stage67", "both"]' in src, \
        "CLI choices 变了，本测试需要同步更新"
    assert vmain._KNOWN_CHAIN_MODES == frozenset(MODES), \
        vmain._KNOWN_CHAIN_MODES
    # 旧的"_声明但未实现"清单必须消失：留着它就是留一条死代码，
    # 而任何"未实现"都必须由 `_KNOWN_CHAIN_MODES` 之外的路径表达。
    assert not hasattr(vmain, "_DECLARED_UNIMPLEMENTED_CHAIN_MODES"), \
        "未实现模式清单应随 S6→S7 落地一并删除"


@pytest.mark.parametrize("mode,g34,s67", [
    ("none", False, False),
    ("stage34", True, False),
    ("stage67", False, True),
    ("both", True, True),
])
def test_group_switches_truth_table(mode, g34, s67):
    """两个执行组各自只看自己的开关，`both` 是并集而不是互斥。"""
    p = {"chain_mode": mode}
    assert vmain._chain_mode_on(p) is g34
    assert vmain._chain67_on(p) is s67


def test_group_switches_default_off_and_tolerate_garbage():
    """缺省/None/奇怪类型都不能把执行组**打开**（默认必须是经典路线）。"""
    for p in ({}, {"chain_mode": None}, {"chain_mode": ""}, None):
        assert vmain._chain_mode_on(p) is False
        assert vmain._chain67_on(p) is False


def test_stage7_refuses_when_hand_off_is_missing(tmp_path, monkeypatch):
    """融合组没有交接单时 stage7 必须停（82），不能退回经典编码。

    反例（必须被拦住）：temp 里还留着上一轮的 `step6_grain*.mkv`。
    经典 stage7 会 glob 到它并**成功**编码 —— 但本轮选的颗粒参数根本没
    生效，成品看起来完全正常。这是"静默降质"，比失败危险得多。
    """
    started = []
    calls = []

    def _stage(script, *a, **k):
        calls.append(script)
        started.append(script)
        if script == "stage6_grain_synthesis.py":
            # 模拟 stage6 没写交接单（例如被外部杀掉后重入）
            return 0, ""
        return 0, ""

    monkeypatch.setattr(vmain, "run_stage", _stage)

    # 直接构造"stage7 将要跑，但 params 里没有 consumer 角色"的状态，
    # 通过 process_job 跑不出这个状态（main 自己会设置），所以这里
    # 直接检查代码里的守卫存在且在当前 params 下会触发。
    import re
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "video_restoration_pipeline", "main.py"),
        encoding="utf-8").read()
    assert "STAGE7_FUSED_STATE_MISSING" in src
    # 守卫必须在 stage 启动**之前**（`STAGE_START` 之前）判定
    i_guard = src.index("STAGE7_FUSED_STATE_MISSING")
    i_start = src.index('job_logger.info("STAGE_START"', i_guard - 4000)
    assert i_guard < i_start, "守卫必须在 stage 启动之前"
    # 而且必须真的 return 非零
    blk = src[i_guard:i_guard + 1200]
    assert re.search(r"return \(82,", blk), blk[-400:]
