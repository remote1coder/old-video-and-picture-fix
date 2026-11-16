# -*- coding: utf-8 -*-
"""未验证的一代**不许覆盖正式代**（R26-A01 / 原始 D01）。

复核明确指出：只校验字典不算修好。所以这里调 **`stage7` 的
`export_diagnostic_generation()` 真实实现**，造一整代旧文件 + 一个新候选，
断言旧代五个对象**一个字节都不变**。

原始 D01 的关键是 `verified=False, forced=False` —— 上一版把它改成
"只有 forced 才走诊断"，等于重新定义了关闭条件。
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared_modules"))
sys.path.insert(0, str(ROOT / "video_restoration_pipeline" / "stages"))

import stage7_encode as s7


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class _Log:
    def __init__(self):
        self.events = []

    def _rec(self, lvl, ev, msg, **kw):
        self.events.append((lvl, ev, kw))

    def info(self, ev, msg, **kw): self._rec("INFO", ev, msg, **kw)
    def warning(self, ev, msg, **kw): self._rec("WARN", ev, msg, **kw)
    def error(self, ev, msg, **kw): self._rec("ERR", ev, msg, **kw)
    def critical(self, ev, msg, **kw): self._rec("CRIT", ev, msg, **kw)

    def names(self):
        return [e[1] for e in self.events]


@pytest.fixture()
def generation(tmp_path):
    """一整代旧文件（媒体 + 四个侧车），返回路径与 hash。"""
    out = tmp_path / "out" / "clip_restored.mkv"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"OLD-VERIFIED-MEDIA")
    for suffix in (".verify.json", ".fullchain.json", ".done.json",
                   ".journal.json"):
        Path(str(out) + suffix).write_text(
            json.dumps({"old": True, "suffix": suffix}), encoding="utf-8")
    paths = [str(out)] + [str(out) + s for s in
                          (".verify.json", ".fullchain.json", ".done.json",
                           ".journal.json")]
    return {"out": str(out), "paths": paths,
            "hashes": {p: _sha(p) for p in paths}}


def _new_candidate(tmp_path, name="cand.mkv", data=b"NEW-UNVERIFIED"):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_unverified_forced_false_does_not_touch_official(generation, tmp_path):
    """**原始 D01**：`verified=False, forced=False` → 五个对象 hash 全不变。"""
    cand = _new_candidate(tmp_path)
    log = _Log()
    ver = {"media_sha256": "x", "checks": {}}
    rc, why = s7.export_diagnostic_generation(
        output_path=generation["out"], attempt_id="att-new",
        ver_doc=ver, publish_tmp=cand,
        fc_pending=None,
        decision={"publish": True, "verified": False, "forced": False,
                  "blocked_by": ["mechanical"], "reasons": {}},
        logger=log)

    assert rc == 89, why
    # 1) 正式代五个对象一个字节都没变
    for p in generation["paths"]:
        assert os.path.exists(p), "正式对象被删了: %s" % p
        assert _sha(p) == generation["hashes"][p], (
            "正式对象被改写: %s" % p)
    # 2) 诊断目录有媒体 + verify + 状态清单
    diag = Path(generation["out"]).parent / ".diagnostic-att-new"
    assert (diag / "clip_restored.mkv").exists()
    assert (diag / "clip_restored.mkv.verify.json").exists()
    state = json.loads((diag / "_diagnostic.json").read_text(encoding="utf-8"))
    assert state["complete"] is True
    assert state["attempt_id"] == "att-new"
    assert state["missing"] == []
    # 3) 候选没有被留在原处
    assert not os.path.exists(cand), "候选应已被移入诊断目录"


def test_forced_true_also_does_not_touch_official(generation, tmp_path):
    """`forced=True` 同样不覆盖（强制不改变"这不是正式成果"）."""
    cand = _new_candidate(tmp_path, "c2.mkv")
    rc, _ = s7.export_diagnostic_generation(
        output_path=generation["out"], attempt_id="att-f",
        ver_doc={}, publish_tmp=cand, fc_pending=None,
        decision={"publish": True, "verified": False, "forced": True,
                  "blocked_by": [], "reasons": {}},
        logger=_Log())
    assert rc == 89
    for p in generation["paths"]:
        assert _sha(p) == generation["hashes"][p]


def test_fullchain_move_failure_is_incomplete_not_89(generation, tmp_path):
    """fullchain 搬不过去 → `complete=false` + rc=82（不能伪装成完整代）."""
    cand = _new_candidate(tmp_path, "c3.mkv")
    # 造一个不存在的 fc_pending：os.replace 会抛 OSError
    ghost = str(tmp_path / "ghost.fullchain.json")
    log = _Log()
    rc, why = s7.export_diagnostic_generation(
        output_path=generation["out"], attempt_id="att-inc",
        ver_doc={}, publish_tmp=cand,
        fc_pending=(ghost, str(generation["out"]) + ".fullchain.json"),
        decision={"publish": True, "verified": False, "forced": False,
                  "blocked_by": [], "reasons": {}},
        logger=log)

    assert rc == 82, why
    assert "STAGE7_DIAGNOSTIC_INCOMPLETE" in log.names(), log.names()
    diag = Path(generation["out"]).parent / ".diagnostic-att-inc"
    state = json.loads((diag / "_diagnostic.json").read_text(encoding="utf-8"))
    assert state["complete"] is False
    assert any("fullchain" in m for m in state["missing"]), state
    # 正式代仍然没被动
    for p in generation["paths"]:
        assert _sha(p) == generation["hashes"][p]


def test_diagnostic_dir_is_not_reused_as_official(generation, tmp_path):
    """诊断目录名不可能撞上正式路径（`.diagnostic-<attempt>` 前缀）."""
    cand = _new_candidate(tmp_path, "c4.mkv")
    s7.export_diagnostic_generation(
        output_path=generation["out"], attempt_id="att-x",
        ver_doc={}, publish_tmp=cand, fc_pending=None,
        decision={"publish": True, "verified": False, "forced": False,
                  "blocked_by": [], "reasons": {}},
        logger=_Log())
    diag = Path(generation["out"]).parent / ".diagnostic-att-x"
    assert diag.is_dir()
    assert diag.name.startswith(".diagnostic-")
