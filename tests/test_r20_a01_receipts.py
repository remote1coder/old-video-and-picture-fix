# -*- coding: utf-8 -*-
"""R20-A01：整代回执对象 + terminal_pending 不能算完成（审核 R19-A01）。

审核原文：

> `publisher.py:458–474` 在 COMMITTED 写失败时设置 terminal_pending，
> 但仍返回 ok=True；as_dict 未包含 terminal_pending。
> **媒体已经晋升与任务已经可靠完成是两种状态。**

本文件锁住：

1. `PublishResult.as_dict()` 必须带 `terminal_pending`；
2. `GenerationReceipt.is_complete()` 在 `terminal_persisted=False` 时**必须**为 False；
3. `CheckpointReceipt` 只认**同一 RunIdentity**（跨身份不可复用），
   且允许跨 attempt 复用（`reused_from` 显式声明）；
4. `load_receipt` 拒绝旧 schema / 缺字段（按 legacy 处理，不自动复用）。
"""
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "shared_modules") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "shared_modules"))

import publisher as pub  # noqa: E402
import receipts as R  # noqa: E402


# ------------------------------------------------ ① as_dict 完整性

def test_publish_result_as_dict_includes_terminal_pending():
    r = pub.PublishResult("att-1", "/x/j.json")
    assert "terminal_pending" in r.as_dict()
    assert r.as_dict()["terminal_pending"] is False

    r.terminal_pending = True
    assert r.as_dict()["terminal_pending"] is True


def test_publish_result_as_dict_includes_generation_receipt():
    r = pub.PublishResult("att-1", "/x/j.json")
    assert "generation_receipt" in r.as_dict()
    assert r.as_dict()["generation_receipt"] is None

    r.generation_receipt = R.GenerationReceipt(
        generation=1, attempt_id="att-1", run_identity="a" * 64,
        required_objects=["/x/a.mkv"], object_digests={"/x/a.mkv": "b" * 64},
        marker_digest="c" * 64, terminal_persisted=True)
    d = r.as_dict()["generation_receipt"]
    assert isinstance(d, dict) and d["generation"] == 1


# ------------------------------------------------ ② 未持久化 ≠ 完成

def test_generation_not_complete_when_terminal_not_persisted():
    g = R.GenerationReceipt(
        generation=1, attempt_id="att-1", run_identity="a" * 64,
        required_objects=["/x/a.mkv"], object_digests={"/x/a.mkv": "b" * 64},
        marker_digest="c" * 64, terminal_persisted=False)
    ok, why = g.is_complete()
    assert ok is False
    assert "terminal_pending" in why or "not persisted" in why


def test_generation_complete_when_everything_present():
    # R21-C02：`is_complete()` 现在还要求 journal_phase == COMMITTED
    # （停在 PREPARED/COMMITTING 的一代不算完成）。生产 publisher 在
    # 提交成功时就是写 COMMITTED，这里照同一形状。
    g = R.GenerationReceipt(
        generation=1, attempt_id="att-1", run_identity="a" * 64,
        required_objects=["/x/a.mkv"], object_digests={"/x/a.mkv": "b" * 64},
        marker_digest="c" * 64, terminal_persisted=True,
        journal_phase="COMMITTED")
    assert g.is_complete() == (True, "")


@pytest.mark.parametrize("missing", ["required_objects", "object_digests",
                                     "marker_digest", "run_identity"])
def test_generation_incomplete_when_any_part_missing(missing):
    kw = dict(generation=1, attempt_id="att-1", run_identity="a" * 64,
              required_objects=["/x/a.mkv"],
              object_digests={"/x/a.mkv": "b" * 64},
              marker_digest="c" * 64, terminal_persisted=True,
              journal_phase="COMMITTED")
    kw[missing] = [] if missing == "required_objects" else (
        {} if missing == "object_digests" else "")
    g = R.GenerationReceipt(**kw)
    ok, _why = g.is_complete()
    assert ok is False, f"{missing} 缺失却被判完成"


def test_generation_incomplete_when_object_lacks_digest():
    g = R.GenerationReceipt(
        generation=1, attempt_id="att-1", run_identity="a" * 64,
        required_objects=["/x/a.mkv", "/x/a.json"],
        object_digests={"/x/a.mkv": "b" * 64},   # json 没摘要
        marker_digest="c" * 64, terminal_persisted=True,
        journal_phase="COMMITTED")
    ok, why = g.is_complete()
    assert ok is False and "digest" in why


# ------------------------------------------------ ③ 检查点身份

def _ident(tag="a"):
    return R.RunIdentity(input_digest=tag * 64, plan_digest="p" * 64,
                         impl_fingerprint="i" * 16)


def test_checkpoint_usable_only_for_same_identity():
    i = _ident()
    c = R.CheckpointReceipt(run_identity=i.digest(), attempt_id="att-1",
                            covered_stages=["stage3", "stage4"],
                            artifacts={"/x/a4.mkv": "d" * 64})
    assert c.is_usable_for(i.digest()) == (True, "")
    ok, why = c.is_usable_for(_ident("b").digest())
    assert ok is False and "different run identity" in why


def test_checkpoint_allows_cross_attempt_reuse_if_declared():
    """检查点**允许**跨 attempt 复用（resume），但必须声明来源。"""
    i = _ident()
    c = R.CheckpointReceipt(run_identity=i.digest(), attempt_id="att-OLD",
                            covered_stages=["stage3"],
                            artifacts={"/x/a3.mkv": "d" * 64},
                            reused_from="att-OLD")
    ok, _ = c.is_usable_for(i.digest())
    assert ok is True
    assert c.attempt_id != "att-NEW"      # 身份不依赖 attempt 相等
    assert c.to_dict()["reused_from"] == "att-OLD"


def test_checkpoint_rejects_no_artifacts():
    i = _ident()
    c = R.CheckpointReceipt(run_identity=i.digest(), attempt_id="att-1",
                            covered_stages=["stage3"], artifacts={})
    ok, why = c.is_usable_for(i.digest())
    assert ok is False and "no artifacts" in why


def test_checkpoint_rejects_artifact_without_digest():
    i = _ident()
    c = R.CheckpointReceipt(run_identity=i.digest(), attempt_id="att-1",
                            covered_stages=["stage3"],
                            artifacts={"/x/a3.mkv": ""})
    ok, why = c.is_usable_for(i.digest())
    assert ok is False and "no digest" in why


# ------------------------------------------------ ④ schema / 反序列化

def test_load_receipt_rejects_old_schema():
    """旧 schema 按 legacy/unproven 处理 —— 返回 None，不自动复用。"""
    for raw in ({"schema": 1, "kind": "GenerationReceipt"},
                {"kind": "GenerationReceipt"},          # 没 schema
                {"schema": R.RECEIPT_SCHEMA, "kind": "NoSuchKind"},
                {"schema": R.RECEIPT_SCHEMA, "kind": "RunIdentity"},  # 缺字段
                "not-a-dict"):
        assert R.load_receipt(raw) is None


def test_roundtrip_preserves_digest():
    i = _ident()
    raw = i.to_dict()
    raw["kind"] = "RunIdentity"
    back = R.load_receipt(raw)
    assert isinstance(back, R.RunIdentity)
    assert back.digest() == i.digest()


def test_labels_do_not_change_identity_digest():
    """`labels` 只是注释，不参与摘要（否则改个名字就让检查点失效）。"""
    a = R.RunIdentity(input_digest="a" * 64, plan_digest="p" * 64,
                      impl_fingerprint="i" * 16, labels={"src": "X"})
    b = R.RunIdentity(input_digest="a" * 64, plan_digest="p" * 64,
                      impl_fingerprint="i" * 16, labels={"src": "Y"})
    assert a.digest() == b.digest()


# ------------------------------------------------ ⑤ 生产：publish 产出回执

def test_publish_emits_generation_receipt(tmp_path):
    out = tmp_path / "out"
    backup = tmp_path / "temp" / "publish_backup"
    out.mkdir()
    backup.mkdir(parents=True)
    staging = tmp_path / "s.bin"
    staging.write_bytes(b"CONTENT")
    final = out / "media.bin"

    res = pub.publish(
        [pub.Artifact("media", str(staging), str(final))],
        str(out / "media.bin.journal.json"), str(backup), "att-1",
        allowed_output_roots=[str(out)],
        run_identity="r" * 64)

    assert res.ok is True, res.errors
    gr = res.generation_receipt
    assert gr is not None, "成功发布必须产出整代回执"
    assert gr.run_identity == "r" * 64
    assert gr.attempt_id == "att-1"
    assert str(final) in gr.required_objects
    assert gr.object_digests.get(str(final))
    # 终态已持久化（本次 COMMITTED 写成功）
    assert gr.terminal_persisted is True
    # 但**没传 marker_digest** → 仍不算"完整代"。
    # 审核 §3.2：marker 必须列入同一事务的必需对象，缺它就不能声称整代提交。
    ok, why = gr.is_complete()
    assert ok is False, "没 marker 却判完成"
    assert "marker" in why.lower()


def test_publish_complete_when_marker_supplied(tmp_path):
    """传了 marker_digest → 回执可判完成。"""
    out = tmp_path / "out"
    backup = tmp_path / "temp" / "publish_backup"
    out.mkdir()
    backup.mkdir(parents=True)
    staging = tmp_path / "s.bin"
    staging.write_bytes(b"CONTENT")
    final = out / "media.bin"

    res = pub.publish(
        [pub.Artifact("media", str(staging), str(final))],
        str(out / "media.bin.journal.json"), str(backup), "att-1",
        allowed_output_roots=[str(out)],
        run_identity="r" * 64,
        marker_digest="m" * 64)

    assert res.ok is True, res.errors
    gr = res.generation_receipt
    assert gr is not None and gr.marker_digest == "m" * 64
    ok, why = gr.is_complete()
    assert ok is True, why
