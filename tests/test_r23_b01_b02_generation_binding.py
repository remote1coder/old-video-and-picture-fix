# -*- coding: utf-8 -*-
"""R23-B01 / B02：完成状态由**唯一的整代验证**决定（跨对象绑定）。

## 审核给的反例（本轮先重放、再修）

审核 R22-B01/B02 用**生产** marker 生成器与 publisher 发布自产小文件，
再用生产完成入口判定，八类坏状态全部 `rc=0 / JOB_DONE`：

| 改动 | 修复前 |
|---|---|
| `generation_receipt=None`、`publish_result=None` | rc=0 |
| `ok="false"`、`schema=true`、`terminal_pending=0` | rc=0 |
| 换成另一份输入/计划/实现的 RunIdentity（自洽可重算）| rc=0 |
| 私有 result 改 att2，marker/journal 仍 att1 | rc=0 |
| 清单删掉 fullchain 并删文件（计划要求它）| rc=0 |
| verify 改成 `{}` | rc=0 |
| journal 换 attempt、`targets=[]`、`resolved="false"` | rc=0 |
| 完成锁依赖导入失败 | rc=0 |

## 本文件锁住什么

关系（不是字段齐不齐）：

```
计划必需角色 == marker.payload.objects == journal.targets ⊆ receipt.required
result.attempt == marker.attempt == journal.attempt == 本次 attempt（正常）
receipt.run_identity == 当前计划的 RunIdentity 摘要
receipt.marker_digest == 当前 marker 文件的摘要
每个必需角色的磁盘摘要 == marker 记录 == receipt 记录
```

**同时**必须保留正常路径：完整一代要通过；resume 允许跨 attempt。
修成"永远拒绝"与修成"永远通过"一样错。
"""
import builtins
import json
import os
import sys

import pytest
from unittest.mock import patch

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import main as vmain  # noqa: E402
import publish_journal as J  # noqa: E402
import stage7_encode as s7  # noqa: E402
import test_r22_a01_completion_path as base  # noqa: E402


def _read(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _write(p, doc):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, default=str)


def _mut_result(ctx, **updates):
    doc = _read(ctx["result_path"])
    doc.update(updates)
    _write(ctx["result_path"], doc)
    return doc


def _mut_journal(ctx, **updates):
    p = ctx["out"] + ".journal.json"
    doc = J.read_journal(p)
    doc.update(updates)
    assert J.write_journal(p, doc)
    return doc


def _mut_marker(ctx, **updates):
    p = vmain.completion_marker_path(ctx["out"])
    doc = _read(p)
    doc.update(updates)
    _write(p, doc)
    return doc


# ------------------------------------------------ 正例（不许修成永远拒绝）

def test_a_complete_generation_is_accepted(tmp_path):
    ctx = base.build(tmp_path)
    ok, why = base.verify(ctx)
    assert ok is True, why


def test_resume_accepts_the_same_generation_without_the_private_result(tmp_path):
    """resume 允许没有本次 attempt 的 result —— 但仍要整代事实成立。"""
    ctx = base.build(tmp_path)
    os.remove(ctx["result_path"])
    ok, why = vmain.resume_artifact_matches(ctx["out"], ctx["job"], {},
                                            None, params=ctx["params"])
    assert ok is True, why


def test_a_disabled_report_may_be_absent(tmp_path):
    """计划关闭 fullchain 时，缺失它是**合法**的（不是缺陷）。"""
    ctx = base.build(tmp_path, with_fullchain=False,
                     params_over={"full_chain_verify": False})
    ok, why = base.verify(ctx)
    assert ok is True, why


# ------------------------------------------------ B01：私有结果的严格绑定

def test_missing_generation_receipt_is_rejected(tmp_path):
    ctx = base.build(tmp_path)
    _mut_result(ctx, generation_receipt=None)
    ok, why = base.verify(ctx)
    assert ok is False and "generation receipt" in why, why


def test_string_ok_is_rejected(tmp_path):
    """`ok="false"` 是真值 —— 旧实现就这么放过了。"""
    ctx = base.build(tmp_path)
    _mut_result(ctx, ok="false")
    ok, why = base.verify(ctx)
    assert ok is False and "ok=" in why, why


def test_boolean_schema_is_rejected(tmp_path):
    """`schema=True` 与 1 相等 —— 必须是**整数** 1。"""
    ctx = base.build(tmp_path)
    _mut_result(ctx, schema=True)
    ok, why = base.verify(ctx)
    assert ok is False and "schema" in why, why


def test_terminal_pending_string_is_rejected(tmp_path):
    ctx = base.build(tmp_path)
    _mut_result(ctx, terminal_pending="false")
    ok, why = base.verify(ctx)
    assert ok is False and "terminal_pending" in why, why


def test_self_consistent_foreign_identity_is_rejected(tmp_path):
    """摘要能自洽重算，但与**当前计划**无关 → 必须拒绝。"""
    ctx = base.build(tmp_path)
    foreign = {"schema": "totally-wrong-schema", "input_digest": "other-input",
               "plan_digest": "other-plan", "impl_fingerprint": "other-tool"}
    foreign["digest"] = vmain._recompute_run_identity_digest(foreign)
    _mut_result(ctx, run_identity=foreign)
    ok, why = base.verify(ctx)
    assert ok is False, "自洽但与当前计划无关的身份被接受了"
    assert "does not match the current plan" in why, why


def test_result_attempt_must_match_the_published_marker(tmp_path):
    """私有 result 换 attempt、marker/journal 不动 → 不是同一次执行。"""
    ctx = base.build(tmp_path)
    ctx["params"]["attempt_id"] = "att2"
    _mut_result(ctx, attempt_id="att2")
    ok, why = base.verify(ctx, expect_attempt="att2")
    assert ok is False, "result 与 marker 的 attempt 不同却通过了"
    assert "attempt" in why, why


def test_receipt_bound_to_another_marker_is_rejected(tmp_path):
    """把 marker 换成另一份（摘要变了）→ 回执绑定的不是这一代。"""
    ctx = base.build(tmp_path)
    m = _read(vmain.completion_marker_path(ctx["out"]))
    m["completed_at"] = "1999-01-01T00:00:00+00:00"
    _write(vmain.completion_marker_path(ctx["out"]), m)
    ok, why = base.verify(ctx)
    assert ok is False, "marker 被换掉后回执仍然通过"
    assert "marker" in why, why


def test_publish_result_missing_is_rejected(tmp_path):
    ctx = base.build(tmp_path)
    _mut_result(ctx, publish_result=None)
    ok, why = base.verify(ctx)
    assert ok is False and "publish result" in why, why


def test_publish_result_phase_must_be_committed(tmp_path):
    ctx = base.build(tmp_path)
    doc = _read(ctx["result_path"])
    pr = dict(doc["publish_result"], phase="COMMITTING")
    _mut_result(ctx, publish_result=pr)
    ok, why = base.verify(ctx)
    assert ok is False and "COMMITTING" in why, why


# ------------------------------------------------ B02：计划决定的必需集合

def test_enabled_fullchain_missing_is_rejected(tmp_path):
    """计划要求 fullchain（默认开启）→ 清单删掉 + 文件删掉都不能通过。"""
    ctx = base.build(tmp_path)
    m = _read(vmain.completion_marker_path(ctx["out"]))
    m["payload_manifest"]["objects"] = [
        o for o in m["payload_manifest"]["objects"] if o["name"] != "fullchain"]
    m["payload_manifest_digest"] = vmain._plain_sha256_text(
        json.dumps(m["payload_manifest"], sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")))
    _write(vmain.completion_marker_path(ctx["out"]), m)
    os.remove(ctx["out"] + ".fullchain.json")
    ok, why = base.verify(ctx)
    assert ok is False and "missing planned object" in why, why


def test_empty_verify_report_is_rejected(tmp_path):
    ctx = base.build(tmp_path)
    _write(ctx["out"] + ".verify.json", {})
    # 清单里的摘要也要跟着改，否则会被"对象变了"先拦住
    m = _read(vmain.completion_marker_path(ctx["out"]))
    for row in m["payload_manifest"]["objects"]:
        if row["name"] == "verify":
            row["sha256"] = base._sha(ctx["out"] + ".verify.json")
    m["payload_manifest_digest"] = vmain._plain_sha256_text(
        json.dumps(m["payload_manifest"], sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")))
    _write(vmain.completion_marker_path(ctx["out"]), m)
    ok, why = base.verify(ctx)
    assert ok is False, "空 verify 报告被接受"
    # 改 marker 内容会同时改掉它的摘要 → 回执先报"不是同一个 marker"。
    # 两条都算检出（都是对"改写已发布对象"的拒绝），但必须说清是哪一条。
    assert ("verify report" in why) or ("different marker" in why), why


def test_journal_without_targets_is_rejected(tmp_path):
    ctx = base.build(tmp_path)
    _mut_journal(ctx, targets=[])
    ok, why = base.verify(ctx)
    assert ok is False, "journal 没有 targets 却通过"
    assert "target" in why, why


def test_journal_string_resolved_is_rejected(tmp_path):
    ctx = base.build(tmp_path)
    _mut_journal(ctx, resolved="false")
    ok, why = base.verify(ctx)
    assert ok is False and "unresolved" in why, why


def test_journal_attempt_must_match_the_marker(tmp_path):
    ctx = base.build(tmp_path)
    _mut_journal(ctx, attempt_id="unrelated-attempt")
    ok, why = base.verify(ctx)
    assert ok is False and "attempt" in why, why


def test_lock_dependency_failure_is_not_a_success(tmp_path):
    """锁不可用时"照跑照过"等于以为持锁、实际没持。"""
    ctx = base.build(tmp_path)
    real_import = builtins.__import__

    def fail(name, *a, **k):
        if name == "publish_lock":
            raise ImportError("controlled missing completion lock dependency")
        return real_import(name, *a, **k)

    with patch("builtins.__import__", side_effect=fail):
        ok, why = base.verify(ctx)
        assert ok is False, "锁依赖导入失败仍判完成"
        assert "lock is unavailable" in why, why
        log = base._Log()
        rc, _w = vmain.finish_job_completion(ctx["job"], {}, {}, ctx["params"],
                                             log, log)
        assert rc == 82, (rc, log.names())
        assert "JOB_DONE" not in log.names()


# ------------------------------------------------ 回执往返

def test_generation_receipt_round_trips_through_the_production_parser(tmp_path):
    """生产者写出的回执必须能被消费者读回来（`kind` 由 to_dict 自带）。"""
    import receipts as R
    ctx = base.build(tmp_path)
    raw = _read(ctx["result_path"])["generation_receipt"]
    assert raw.get("kind") == "GenerationReceipt", raw.keys()
    obj = R.load_receipt(raw)
    assert obj is not None
    ok, why = obj.is_complete()
    assert ok is True, why
