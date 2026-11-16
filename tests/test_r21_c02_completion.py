# -*- coding: utf-8 -*-
"""R21-C02：完成判定 —— JOB_DONE 仍在成功之前，resume 只看媒体。

## 审核原文（R20 §3，P1）

> 另一方面，main 先记录 `JOB_DONE`，再调用 `write_completion_marker`，
> 没有检查它的 False 返回值。原 main 尾段的受控调用确认：即使标记写入
> 明确失败，仍返回 `(0, "")`。
>
> resume 检查仍主要围绕 `.done.json` 和媒体；受控路径没有读取所需报告和
> journal，不能证明缺失报告的一代可继续被当作完整完成。

## 本文件锁住什么

1. `finish_job_completion()` 只有在**全部证据成立**时才返回 `(0, "")`；
   找不到成品 / 标记写失败 / 标记抛异常 → `82`，且**不**打 `JOB_DONE`。
2. `verify_generation_objects()` 把"整代在不在"判成唯一结论：
   媒体 + 必需报告 + journal 终态，缺一不可。
3. `resume_artifact_matches()` 复用同一个验证器 —— 报告被删的一代，
   即使 marker 完好也不许当"已完成"复用。
"""
import json
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline"),
           # R22-A01：夹具改用**真实生产者** build_marker_candidate，
           # 它在 stages/ 下
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import main as vmain  # noqa: E402
import publish_journal as J  # noqa: E402


class _Log:
    def __init__(self):
        self.events = []

    def _rec(self, ev, msg="", **kw):
        self.events.append((ev, msg, kw))

    info = warning = error = critical = _rec

    def names(self):
        return [e[0] for e in self.events]


def _write(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(path, mode, **({} if mode == "wb" else {"encoding": "utf-8"})) as f:
        f.write(data)
    return path


def _gen(tmp_path, *, with_reports=True, with_journal=True,
         journal_phase="COMMITTED", resolved=True, media=b"MEDIA-BYTES"):
    final = str(tmp_path / "out" / "clip_restored.mkv")
    _write(final, media)
    if with_reports:
        # R24-02：完成验证现在读报告的**结论**（必需检查 + 发布决定）。
        # 旧夹具只写 media_sha256 —— 那不是真实 stage7 会产出的报告，
        # 也会让"报告说失败却判完成"这条判据在夹具上永远测不出来。
        _write(final + ".verify.json", json.dumps({
            # B02：真实 stage7 的 verify 报告带执行编号与 RunIdentity；
            # 缺这两项的报告不能被绑定到某次执行（C02/C03）。
            "attempt_id": "att1",
            "run_identity": {"schema": "run-identity/1",
                             "digest": "0" * 64},
            "media_sha256": vmain._plain_sha256(final),
            "checks": {
                "mechanical": {"name": "mechanical", "status": "PASS",
                               "required": True},
                "frame_order": {"name": "frame_order", "status": "PASS",
                                "required": True},
                "final_timeline": {"name": "final_timeline", "status": "PASS",
                                   "required": True},
                "fullchain": {"name": "fullchain", "status": "PASS",
                              "required": True},
            },
            "publish_decision": {"publish": True, "verified": True,
                                 "blocked_by": [], "forced": False,
                                 "reasons": {}},
        }))
        _write(final + ".fullchain.json", json.dumps({"ok": True, "attempt_id": "att1", "checks": {"frames": {"ok": True}, "duration": {"ok": True}, "not_black": {"ok": True}}}))
    if with_journal:
        # R22-A01：journal 的 targets 必须覆盖**同一代**的全部对象。
        # 真实 publisher 就是 `sorted(set(finals))`；旧夹具只写媒体，
        # 于是新加的"targets 必须覆盖对象集合"检查会（正确地）拒绝它。
        _targets = [final]
        if with_reports:
            _targets += [final + ".verify.json", final + ".fullchain.json"]
        # R23-B01：journal 必须写明 attempt，且与 marker 是同一代
        J.write_journal(final + ".journal.json",
                        {"phase": journal_phase, "job_id": "j1",
                         "attempt_id": "att1",
                         "targets": _targets, "records": {},
                         "resolved": resolved})
    return final


# ------------------------------------------------ 收尾顺序

def test_no_finished_output_is_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(vmain, "find_final_output", lambda *a, **k: [])
    log = _Log()
    rc, why = vmain.finish_job_completion(
        {"job_id": "j1", "input_path": str(tmp_path / "in.mp4")},
        {}, None, {}, log, log)
    assert rc == 82, (rc, why)
    assert "JOB_DONE" not in log.names()
    assert "JOB_COMPLETION_UNPROVEN" in log.names()


def test_marker_failure_blocks_done(tmp_path, monkeypatch):
    final = _gen(tmp_path)
    monkeypatch.setattr(vmain, "find_final_output", lambda *a, **k: [final])
    monkeypatch.setattr(vmain, "write_completion_marker",
                        lambda *a, **k: False)
    log = _Log()
    rc, why = vmain.finish_job_completion(
        {"job_id": "j1", "input_path": str(tmp_path / "in.mp4")},
        {}, None, {}, log, log)
    assert rc == 82
    assert "JOB_DONE" not in log.names(), "标记失败仍然报了 JOB_DONE"


def test_marker_exception_blocks_done(tmp_path, monkeypatch):
    final = _gen(tmp_path)
    monkeypatch.setattr(vmain, "find_final_output", lambda *a, **k: [final])

    def _boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(vmain, "write_completion_marker", _boom)
    log = _Log()
    rc, why = vmain.finish_job_completion(
        {"job_id": "j1", "input_path": str(tmp_path / "in.mp4")},
        {}, None, {}, log, log)
    assert rc == 82
    assert "JOB_DONE" not in log.names()
    assert "disk full" in why


def test_success_logs_done_exactly_once(tmp_path, monkeypatch):
    final = _gen(tmp_path)
    monkeypatch.setattr(vmain, "find_final_output", lambda *a, **k: [final])
    monkeypatch.setattr(vmain, "write_completion_marker", lambda *a, **k: True)
    log = _Log()
    rc, why = vmain.finish_job_completion(
        {"job_id": "j1", "input_path": str(tmp_path / "in.mp4")},
        {}, None, {}, log, log)
    assert (rc, why) == (0, "")
    assert log.names().count("JOB_DONE") == 1


def test_done_is_logged_after_the_marker_not_before(tmp_path, monkeypatch):
    """顺序断言：写标记必须发生在 JOB_DONE **之前**。"""
    order = []
    final = _gen(tmp_path)
    monkeypatch.setattr(vmain, "find_final_output", lambda *a, **k: [final])
    monkeypatch.setattr(vmain, "write_completion_marker",
                        lambda *a, **k: order.append("marker") or True)

    log = _Log()

    class _L(_Log):
        def _rec(self, ev, msg="", **kw):
            if ev == "JOB_DONE":
                order.append("done")
            super()._rec(ev, msg, **kw)

        # 必须重新绑定：`_Log` 里 info/error 是**函数对象**的别名，
        # 子类只覆盖 `_rec` 不会改变 `info` 指向。
        info = warning = error = critical = _rec

    vmain.finish_job_completion(
        {"job_id": "j1", "input_path": str(tmp_path / "in.mp4")},
        {}, None, {}, _L(), log)
    assert order == ["marker", "done"], order


# ------------------------------------------------ 整代验证器

def test_complete_generation_passes(tmp_path):
    final = _gen(tmp_path)
    ok, why = vmain.verify_generation_objects(final, {"job_id": "j1"})
    assert ok is True, why


@pytest.mark.parametrize("kwargs,fragment", [
    ({"media": b""}, "empty"),
    ({"with_reports": False}, "missing"),
    ({"with_journal": False}, "journal"),
    ({"journal_phase": "PREPARED"}, "not COMMITTED"),
    ({"resolved": False}, "unresolved"),
])
def test_incomplete_generation_is_rejected(tmp_path, kwargs, fragment):
    final = _gen(tmp_path, **kwargs)
    ok, why = vmain.verify_generation_objects(final, {"job_id": "j1"})
    assert ok is False, why
    assert fragment in why, why


def test_media_bytes_changed_after_the_report_is_rejected(tmp_path):
    final = _gen(tmp_path)
    _write(final, b"DIFFERENT-BYTES")
    ok, why = vmain.verify_generation_objects(final, {"job_id": "j1"})
    assert ok is False
    assert "different media bytes" in why


def test_missing_media_is_rejected(tmp_path):
    final = _gen(tmp_path)
    os.remove(final)
    ok, why = vmain.verify_generation_objects(final, {"job_id": "j1"})
    assert ok is False and "missing" in why


# ------------------------------------------------ resume 复用同一验证器

def _make_marker(final, job, config, log, objects=None):
    """用**真实生产者**造一份带 payload 的 marker（R22-A01 起唯一合法形状）。

    旧版这里手写 marker JSON（无 payload），R21 时 `verify_payload_manifest`
    对"没有 payload"直接返回 True，于是"同大小替换媒体"也能过。
    现在旧格式一律判 legacy/unproven，所以夹具必须走生产函数。
    """
    import stage7_encode as s7
    if objects is None:
        objects = []
        for role, path in (("media", final),
                           ("verify", final + ".verify.json"),
                           ("fullchain", final + ".fullchain.json")):
            if os.path.exists(path):
                objects.append((role, path, vmain._plain_sha256(path)))
    seed = {
        "impl": vmain.VIDEO_IMPL_VERSION,
        "job_id": job.get("job_id", ""),
        "input_path": job.get("input_path", ""),
        "input_identity": vmain.build_input_identity(job.get("input_path", "")),
        "params_identity": vmain.build_params_identity(config, job, log),
    }
    _mk, _stage, _dig, _pm = s7.build_marker_candidate(
        seed=seed, pub_digests=objects, attempt_id="att1",
        job_id=job.get("job_id", "j1"), output_path=final,
        media_digest=objects[0][2], media_size=os.path.getsize(final))
    os.replace(_stage, vmain.completion_marker_path(final))
    return _mk


def test_resume_accepts_a_fully_proven_generation(tmp_path):
    job = {"job_id": "j1", "input_path": str(tmp_path / "in.mp4")}
    _write(job["input_path"], b"SOURCE")
    final = _gen(tmp_path)
    log = _Log()
    _make_marker(final, job, {"global": {}}, log)
    ok, why = vmain.resume_artifact_matches(final, job, {"global": {}}, log)
    assert ok is True, why


def test_resume_refuses_when_a_required_report_is_gone(tmp_path):
    """marker 完好、媒体完好，但 fullchain 被删 → 不是完整代。"""
    job = {"job_id": "j1", "input_path": str(tmp_path / "in.mp4")}
    _write(job["input_path"], b"SOURCE")
    final = _gen(tmp_path)
    log = _Log()
    _make_marker(final, job, {"global": {}}, log)
    os.remove(final + ".fullchain.json")

    ok, why = vmain.resume_artifact_matches(final, job, {"global": {}}, log)
    assert ok is False, why
    assert "missing" in why and "fullchain" in why, why


def test_resume_refuses_when_the_journal_is_unresolved(tmp_path):
    job = {"job_id": "j1", "input_path": str(tmp_path / "in.mp4")}
    _write(job["input_path"], b"SOURCE")
    final = _gen(tmp_path, journal_phase="PREPARED", resolved=False)
    log = _Log()
    _make_marker(final, job, {"global": {}}, log)

    ok, why = vmain.resume_artifact_matches(final, job, {"global": {}}, log)
    assert ok is False, why
    assert "COMMITTED" in why and "PREPARED" in why, why


# ------------------------------------------------ 回执强类型（C02 第三点）

def _receipt(**kw):
    import receipts as R
    base = dict(generation=1, attempt_id="att-1", run_identity="a" * 64,
                required_objects=["/x/a.mkv"],
                object_digests={"/x/a.mkv": "b" * 64},
                marker_digest="c" * 64, terminal_persisted=True,
                journal_phase="COMMITTED")
    base.update(kw)
    return R.GenerationReceipt(**base)


def test_receipt_requires_committed_journal_phase():
    ok, why = _receipt(journal_phase="PREPARED").is_complete()
    assert ok is False and "COMMITTED" in why, why


def test_receipt_rejects_an_empty_attempt_id():
    ok, why = _receipt(attempt_id="").is_complete()
    assert ok is False and "attempt" in why, why


def test_receipt_string_false_is_not_a_persisted_terminal():
    """`"false"` 不是布尔 True —— 旧实现 `bool("false")` 会读成真。"""
    import receipts as R
    raw = dict(_receipt().to_dict(), kind="GenerationReceipt")
    raw["terminal_persisted"] = "false"
    assert R.load_receipt(raw) is None, "字符串 'false' 被当成了真"


def test_receipt_with_a_wrong_declared_digest_is_rejected():
    import receipts as R
    raw = dict(_receipt().to_dict(), kind="GenerationReceipt")
    raw["digest"] = "0" * 64
    assert R.load_receipt(raw) is None


def test_receipt_with_a_matching_declared_digest_round_trips():
    import receipts as R
    raw = dict(_receipt().to_dict(), kind="GenerationReceipt")
    obj = R.load_receipt(raw)
    assert obj is not None
    ok, why = obj.is_complete()
    assert ok is True, why
