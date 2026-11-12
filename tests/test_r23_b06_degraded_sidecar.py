# -*- coding: utf-8 -*-
"""R23-08：降级侧车必须**被消费**，而不是只写不读。

## 审核的判词

> 降级侧车仍未被检查点/最终报告消费，交接对此已经承认。下一轮应连同
> 侧车写失败、旧侧车遗留、正常重跑一起收口。

修复前：`<artifact>.degraded.json` 由 stage6 写入，`output_sha256` 也有，
但**没有任何代码读它** —— 检查点不知道这份中间产物是降级货，
最终报告也不提，"换了插件、强度减半"只留在会轮转的日志里。

修复后四条：

| 场景 | 行为 |
|---|---|
| 侧车描述当前字节 | 复用**允许**，但复用结论里标 `degraded: True` 并带插件名 |
| 侧车描述别的字节 | **不复用**（`stale-degraded-sidecar`），并列出该文件为 stale |
| 侧车写失败 | 阶段**失败**（rc=84），不再只告警 |
| 正常重跑（无降级） | 删掉上一代遗留的侧车 |
"""
import hashlib
import json
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage_deps as sd  # noqa: E402
import stage6_grain_synthesis as s6  # noqa: E402


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _artifact(tmp_path, name="step6_grain.mkv", data=b"GRAIN-BYTES"):
    p = str(tmp_path / name)
    with open(p, "wb") as f:
        f.write(data)
    return p


def _note(artifact, digest=None, plugin="AddGrainC"):
    p = artifact + ".degraded.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"schema": "stage6-degraded/1",
                   "output": os.path.abspath(artifact),
                   "output_sha256": digest if digest is not None else _sha(artifact),
                   "degraded": True,
                   "plugin_used": plugin}, f)
    return p


def _records(artifact):
    """造 1..6 六个"身份匹配"的记录。

    为什么不能只给 stage 6：`build_reuse_plan` 遇到**缺记录**的 stage 会判
    身份缺失，失效向下游传播，stage 6 会被连带剔除 —— 那是它的正确行为，
    不是本用例要测的东西。
    """
    params = {"vs_bit_depth": 16}
    return {st: {"stage_identity": sd.stage_identity(params, st),
                 "artifact": artifact,
                 "stage_scope": {"vs_bit_depth": 16}}
            for st in range(1, 7)}


# ------------------------------------------------ 读取与判定

def test_missing_note_is_none(tmp_path):
    a = _artifact(tmp_path)
    assert sd.read_degraded_note(a) is None


def test_a_matching_note_is_current(tmp_path):
    a = _artifact(tmp_path)
    _note(a)
    ok, why = sd.degraded_note_is_current(a)
    assert ok is True, why


def test_a_note_for_other_bytes_is_not_current(tmp_path):
    a = _artifact(tmp_path, data=b"NEW-CONTENT")
    _note(a, digest="0" * 64)          # 侧车记的是别的字节
    ok, why = sd.degraded_note_is_current(a)
    assert ok is False
    assert "other bytes" in why, why


def test_a_note_without_a_digest_is_not_current(tmp_path):
    a = _artifact(tmp_path)
    _note(a, digest="")
    ok, why = sd.degraded_note_is_current(a)
    assert ok is False and "no content digest" in why, why


# ------------------------------------------------ 复用计划消费侧车

def test_reusing_a_degraded_artifact_is_marked(tmp_path):
    a = _artifact(tmp_path)
    _note(a, plugin="AddGrainC")
    plan = sd.build_reuse_plan({"vs_bit_depth": 16},
                               _records(a), ai_stages=())
    stages = {r["stage"]: r for r in plan["reusable"]}
    assert 6 in stages, plan
    assert stages[6].get("degraded") is True
    assert stages[6].get("degraded_plugin") == "AddGrainC"
    assert any("DEGRADED" in n for n in plan.get("notes") or []), plan.get("notes")


def test_a_stale_sidecar_blocks_reuse(tmp_path):
    a = _artifact(tmp_path, data=b"CHANGED-AFTER-NOTE")
    _note(a, digest="f" * 64)          # 侧车描述的是更早的字节
    plan = sd.build_reuse_plan({"vs_bit_depth": 16},
                               _records(a), ai_stages=())
    assert not [r for r in plan["reusable"] if r["stage"] == 6], plan
    reasons = [d["reason"] for d in plan["invalidated"]]
    assert "stale-degraded-sidecar" in reasons, reasons


def test_without_a_sidecar_nothing_changes(tmp_path):
    a = _artifact(tmp_path)
    plan = sd.build_reuse_plan({"vs_bit_depth": 16},
                               _records(a), ai_stages=())
    stages = {r["stage"]: r for r in plan["reusable"]}
    assert 6 in stages
    assert not stages[6].get("degraded")


# ------------------------------------------------ 侧车生命周期

class _Log:
    def __init__(self):
        self.events = []

    def _rec(self, ev, msg="", **kw):
        self.events.append((ev, msg, kw))

    info = warning = error = critical = _rec


def test_writing_the_note_returns_true(tmp_path):
    a = _artifact(tmp_path)
    first = s6.GrainPipelineResult(ok=False, vspipe_rc=1, ffmpeg_rc=1,
                                   failed_stage="ambigUous",
                                   vspipe_tail="grainfactory not found")
    second = s6.GrainPipelineResult(ok=True, vspipe_rc=0, ffmpeg_rc=0)
    ok = s6._write_degraded_note(a, first, second, 1.0, 0.2, 0.2, 0.5,
                                 _Log())
    assert ok is True
    doc = sd.read_degraded_note(a)
    assert doc and doc["output_sha256"] == _sha(a)


def test_a_write_failure_returns_false_and_logs_an_error(tmp_path):
    """写不出来 → 返回 False（调用方据此让阶段失败），且必须 error 级。"""
    bad = str(tmp_path / "no-such-dir" / "x.mkv")
    log = _Log()
    first = s6.GrainPipelineResult(ok=False, vspipe_rc=1, ffmpeg_rc=1)
    second = s6.GrainPipelineResult(ok=True, vspipe_rc=0, ffmpeg_rc=0)
    ok = s6._write_degraded_note(bad, first, second, 1.0, 0.2, 0.2, 0.5, log)
    assert ok is False
    assert any(e[0] == "STAGE6_DEGRADED_NOTE_FAIL" for e in log.events)


def test_the_note_can_be_cleared_for_a_clean_rerun(tmp_path):
    a = _artifact(tmp_path)
    _note(a)
    assert os.path.exists(a + ".degraded.json")
    s6._clear_degraded_note(a, _Log())
    assert not os.path.exists(a + ".degraded.json")
    # 再删一次也不该报错
    s6._clear_degraded_note(a, _Log())


# ------------------------------------------------ 报告可见性

def test_fullchain_report_exposes_degraded_inputs(tmp_path):
    import fullchain_report as fr
    outdir = tmp_path / "out"
    outdir.mkdir()
    (outdir / "clip.mkv").write_bytes(b"MEDIA")
    doc = {"ok": True,
           "checks": {},
           "degraded_inputs": [{"stage": 6, "plugin": "AddGrainC"}]}
    jf = str(outdir / "clip.mkv.fullchain.json")
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    rows = fr.collect([jf], str(outdir))
    assert rows and rows[0]["degraded"] is True
    assert rows[0]["degraded_inputs"][0]["plugin"] == "AddGrainC"
