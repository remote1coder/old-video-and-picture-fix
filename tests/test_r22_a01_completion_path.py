# -*- coding: utf-8 -*-
"""R22-A01：完成证明必须只有一条可信路径。

## 审核给的反例（本轮先复现、再修）

> 用生产 `build_marker_candidate` 和 `publisher.publish` 提交完整四件套；
> 生产 `write_journal` 将状态改为摘要合法的 PREPARED；给不同 attempt 及
> 缺失私有 result；生产 `finish_job_completion` 仍输出 JOB_DONE。

复现结果（修复前）：

```
rc = 0, why = ""
events = COMPLETION_MARKER_VERIFIED_PUBLISHED, JOB_DONE
strict_check = (False, "publish journal is not COMMITTED (phase='PREPARED')")
```

同一个对象，两套判据给出相反结论。

另一个更宽的洞：旧格式 marker（无 payload）+ 媒体同大小替换，
`verify_published_marker` 返回 True —— 因为 `verify_payload_manifest`
对"没有 payload"直接 `return True`。

## 本文件锁住什么

`verify_generation_completion()` 是**唯一**入口，顺序固定、无提前返回：

1. 锁集合内
2. 私有 result（正常收尾必须）：schema / attempt / 最终路径 / 终态 / RunIdentity
3. marker payload：三态 + 角色 + **media 路径必须等于当前成品**
4. 整代磁盘事实：媒体 + 计划确定的报告 + journal COMMITTED + targets 覆盖

resume 与正常收尾共用它，唯一区别是 `expect_attempt=None`。
"""
import hashlib
import json
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import main as vmain  # noqa: E402
import publisher as P  # noqa: E402
import publish_journal as J  # noqa: E402
import stage7_encode as s7  # noqa: E402


class _Log:
    def __init__(self):
        self.events = []

    def _rec(self, ev, msg="", **kw):
        self.events.append((ev, msg, kw))

    info = warning = error = critical = _rec

    def names(self):
        return [e[0] for e in self.events]


def _w(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(path, mode, **({} if mode == "wb" else {"encoding": "utf-8"})) as f:
        f.write(data)
    return path


def _sha(path):
    if not os.path.exists(path):
        return "<absent>"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _snapshot(*paths):
    return {os.path.basename(p): _sha(p) for p in paths}


def build(tmp_path, *, with_fullchain=True, media=b"GEN-BYTES",
          attempt="att1", with_result=True, result_over=None,
          journal_phase=None):
    """用**生产函数**组装一整代并发布；可选把 journal 改成未决态。

    Returns:
        ctx：路径、job、params（含 attempt 与私有 result）。
    """
    out = str(tmp_path / "out" / "clip_restored.mkv")
    stage = str(tmp_path / "staging")
    src = str(tmp_path / "src.mp4")
    _w(src, b"SOURCE")
    job = {"job_id": "j1", "input_path": src, "output_path": out}

    _w(os.path.join(stage, "media.bin"), media)
    _w(os.path.join(stage, "verify.json"),
       json.dumps({"media_sha256": _sha(os.path.join(stage, "media.bin"))}))
    arts = [
        P.Artifact("media", os.path.join(stage, "media.bin"), out),
        P.Artifact("verify", os.path.join(stage, "verify.json"),
                   out + ".verify.json"),
    ]
    if with_fullchain:
        _w(os.path.join(stage, "fullchain.json"), json.dumps({"ok": True}))
        arts.append(P.Artifact("fullchain", os.path.join(stage, "fullchain.json"),
                               out + ".fullchain.json"))

    digests = [(a.name, a.final, _sha(a.staging)) for a in arts]
    eff = vmain.merge_params(job, {}, {}, None)
    eff["attempt_id"] = attempt
    seed = {
        "impl": vmain.VIDEO_IMPL_VERSION,
        "job_id": "j1",
        "input_path": src,
        "input_identity": vmain.build_input_identity(src),
        "params_identity": vmain.build_params_identity({}, job, None,
                                                       params=eff),
    }
    _mk, mk_stage, mk_dig, _pm = s7.build_marker_candidate(
        seed=seed, pub_digests=digests, attempt_id=attempt, job_id="j1",
        output_path=out, media_digest=digests[0][2],
        media_size=os.path.getsize(arts[0].staging))
    arts.append(P.Artifact("marker", mk_stage, out + ".done.json"))

    res = P.publish(arts, out + ".journal.json", str(tmp_path / "backup"),
                    attempt_id=attempt, decision={"publish": True},
                    allowed_output_roots=[str(tmp_path / "out")],
                    required_objects=[a.final for a in arts],
                    marker_digest=mk_dig)
    assert res.ok, res.errors

    if journal_phase:
        doc = J.read_journal(out + ".journal.json")
        doc["phase"] = journal_phase
        assert J.write_journal(out + ".journal.json", doc)

    result_path = str(tmp_path / "stage7_result.json")
    eff["_stage7_result_path"] = result_path
    if with_result:
        payload = {
            "schema": 1,
            "attempt_id": attempt,
            "ok": True,
            "terminal_pending": False,
            "journal_phase": "COMMITTED",
            "journal_path": out + ".journal.json",
            "generation_receipt": (res.generation_receipt.to_dict()
                                   if res.generation_receipt else None),
            "final_output_path": out,
            "publish_result": res.as_dict(),
            "run_identity": vmain.build_run_identity({}, job, None,
                                                     params=eff),
        }
        payload.update(result_over or {})
        _w(result_path, json.dumps(payload, ensure_ascii=False, default=str))

    return {"out": out, "job": job, "params": eff, "res": res,
            "arts": arts, "result_path": result_path, "seed": seed}


def verify(ctx, *, expect_attempt="att1"):
    return vmain.verify_generation_completion(
        ctx["out"], ctx["job"], {}, ctx["params"], _Log(),
        expect_attempt=expect_attempt)


# ------------------------------------------------ 正常路径

def test_complete_generation_is_proven(tmp_path):
    ctx = build(tmp_path)
    ok, why = verify(ctx)
    assert ok is True, why


def test_optional_fullchain_is_not_required_when_the_plan_omits_it(tmp_path):
    """`full_chain_verify=false`：发布端只交 media+verify，验证端**不能**
    反过来硬要 fullchain（审核点名的"一端可选、一端固定"）。"""
    ctx = build(tmp_path, with_fullchain=False)
    assert not os.path.exists(ctx["out"] + ".fullchain.json")
    ok, why = verify(ctx)
    assert ok is True, why


def test_resume_shares_the_core_but_allows_another_attempt(tmp_path):
    """resume 跳过私有 result，其余判据相同。"""
    ctx = build(tmp_path, attempt="att1")
    log = _Log()
    ok, why = vmain.resume_artifact_matches(ctx["out"], ctx["job"], {},
                                            log, params=ctx["params"])
    assert ok is True, why
    # 而"正常收尾"拿错 attempt 时必须拒绝
    ctx["params"]["attempt_id"] = "att2"
    ok2, why2 = verify(ctx, expect_attempt="att2")
    assert ok2 is False
    assert "result" in why2, why2


# ------------------------------------------------ 私有 result

def test_missing_private_result_is_rejected(tmp_path):
    ctx = build(tmp_path, with_result=False)
    ok, why = verify(ctx)
    assert ok is False
    assert "result" in why and "missing" in why, why


def test_wrong_attempt_is_rejected(tmp_path):
    ctx = build(tmp_path, attempt="att1")
    ok, why = vmain.verify_generation_completion(
        ctx["out"], ctx["job"], {}, ctx["params"], _Log(),
        expect_attempt="att2")
    assert ok is False
    assert "attempt" in why, why


def test_corrupt_result_is_rejected(tmp_path):
    ctx = build(tmp_path, with_result=False)
    _w(ctx["result_path"], "{not json")
    ok, why = verify(ctx)
    assert ok is False
    assert "unreadable" in why, why


@pytest.mark.parametrize("over,fragment", [
    ({"schema": "1"}, "schema"),
    ({"terminal_pending": True}, "terminal"),
    ({"ok": False}, "failed publish"),
    ({"final_output_path": "/somewhere/else.mkv"}, "artifact under test"),
])
def test_result_must_match_this_generation(tmp_path, over, fragment):
    ctx = build(tmp_path)
    doc = json.load(open(ctx["result_path"], encoding="utf-8"))
    doc.update(over)
    _w(ctx["result_path"], json.dumps(doc, ensure_ascii=False))
    ok, why = verify(ctx)
    assert ok is False, why
    assert fragment in why, why


def test_run_identity_must_be_recomputable(tmp_path):
    """把 run_identity 的 digest 改掉 → 重算对不上 → 拒绝（R22-A02）。"""
    ctx = build(tmp_path)
    doc = json.load(open(ctx["result_path"], encoding="utf-8"))
    doc["run_identity"]["digest"] = "0" * 64
    _w(ctx["result_path"], json.dumps(doc, ensure_ascii=False))
    ok, why = verify(ctx)
    assert ok is False
    assert "cannot be reproduced" in why, why


def test_run_identity_must_be_present(tmp_path):
    ctx = build(tmp_path)
    doc = json.load(open(ctx["result_path"], encoding="utf-8"))
    doc.pop("run_identity", None)
    _w(ctx["result_path"], json.dumps(doc, ensure_ascii=False))
    ok, why = verify(ctx)
    assert ok is False
    assert "run_identity" in why, why


# ------------------------------------------------ journal 终态

@pytest.mark.parametrize("phase", ["PREPARED", "COMMITTING"])
def test_unresolved_journal_is_rejected(tmp_path, phase):
    """审核的原始复现：PREPARED（摘要合法）也必须拒绝。"""
    ctx = build(tmp_path, journal_phase=phase)
    before = _snapshot(ctx["out"], ctx["out"] + ".verify.json",
                       ctx["out"] + ".done.json")
    ok, why = verify(ctx)
    assert ok is False, why
    assert "COMMITTED" in why, why
    # 拒绝时**不能动任何文件**
    after = _snapshot(ctx["out"], ctx["out"] + ".verify.json",
                      ctx["out"] + ".done.json")
    assert after == before


def test_journal_without_commit_blocks_even_with_a_good_marker(tmp_path):
    """有 marker 不能提前绕过 journal 终态（这是 A01 的核心）。"""
    ctx = build(tmp_path, journal_phase="PREPARED")
    log = _Log()
    rc, why = vmain.finish_job_completion(
        ctx["job"], {}, {"stage": None}, ctx["params"], log, log)
    assert rc == 82, (rc, why)
    assert "JOB_COMPLETION_UNPROVEN" in log.names()
    assert "JOB_DONE" not in log.names(), "未提交的代被记成了成功"


# ------------------------------------------------ marker payload

def test_legacy_marker_without_payload_is_unproven(tmp_path):
    """旧格式 marker 不能作为"新式事务 marker"通过（含同大小替换）。"""
    ctx = build(tmp_path)
    mp = ctx["out"] + ".done.json"
    doc = json.load(open(mp, encoding="utf-8"))
    doc.pop("payload_manifest", None)
    doc.pop("payload_manifest_digest", None)
    _w(mp, json.dumps(doc, ensure_ascii=False))
    _w(ctx["out"], b"OLD-BYTES")            # 同大小替换（都是 9 字节）
    ok, why = verify(ctx)
    assert ok is False
    assert "legacy" in why, why


def test_same_size_media_replacement_is_rejected(tmp_path):
    ctx = build(tmp_path)
    _w(ctx["out"], b"OLD-BYTES")            # 与 GEN-BYTES 同为 9 字节
    ok, why = verify(ctx)
    assert ok is False
    assert "changed since it was published" in why, why


def test_marker_media_must_be_the_artifact_under_test(tmp_path):
    """marker 里 media 的路径不是当前成品 → 拒绝（防止拿别处正确文件冒充）。"""
    other = tmp_path / "other" / "x.mkv"
    _w(str(other), b"GEN-BYTES")
    ctx = build(tmp_path)
    mp = ctx["out"] + ".done.json"
    doc = json.load(open(mp, encoding="utf-8"))
    for o in doc["payload_manifest"]["objects"]:
        if o["name"] == "media":
            o["path"] = str(other)
            o["sha256"] = _sha(str(other))
    doc["payload_manifest_digest"] = vmain._plain_sha256_text(
        json.dumps(doc["payload_manifest"], sort_keys=True,
                   ensure_ascii=False, separators=(",", ":")))
    _w(mp, json.dumps(doc, ensure_ascii=False))
    ok, why = verify(ctx)
    assert ok is False
    assert "not the artifact under test" in why, why


def test_marker_with_duplicate_roles_is_rejected(tmp_path):
    ctx = build(tmp_path)
    mp = ctx["out"] + ".done.json"
    doc = json.load(open(mp, encoding="utf-8"))
    objs = doc["payload_manifest"]["objects"]
    objs.append(dict(objs[1]))              # 重复 verify 角色
    doc["payload_manifest_digest"] = vmain._plain_sha256_text(
        json.dumps(doc["payload_manifest"], sort_keys=True,
                   ensure_ascii=False, separators=(",", ":")))
    _w(mp, json.dumps(doc, ensure_ascii=False))
    ok, why = verify(ctx)
    assert ok is False
    assert "twice" in why, why


# ------------------------------------------------ 报告完整性

def test_missing_report_is_rejected(tmp_path):
    ctx = build(tmp_path)
    os.remove(ctx["out"] + ".verify.json")
    ok, why = verify(ctx)
    assert ok is False
    assert "missing" in why, why


def test_tampered_report_is_rejected(tmp_path):
    """改报告内容 → payload 摘要先对不上（比"报告描述别的媒体"更早发现）。"""
    ctx = build(tmp_path)
    _w(ctx["out"] + ".verify.json",
       json.dumps({"media_sha256": "f" * 64}))
    ok, why = verify(ctx)
    assert ok is False
    assert "changed since it was published" in why, why


def test_marker_from_another_generation_is_rejected(tmp_path):
    """把 A 代的 marker 放到 B 代成品旁边 → payload 指向 A，拒绝。

    这是真正的"串代"：A、B 各自自洽，但拼不到一起。
    """
    a = build(tmp_path / "a")
    b = build(tmp_path / "b")
    with open(a["out"] + ".done.json", "r", encoding="utf-8") as f:
        a_marker = f.read()
    _w(b["out"] + ".done.json", a_marker)
    ok, why = verify(b)
    assert ok is False
    assert "not the artifact under test" in why or \
        "changed since it was published" in why, why


def test_no_marker_at_all_is_rejected(tmp_path):
    ctx = build(tmp_path)
    os.remove(ctx["out"] + ".done.json")
    ok, why = verify(ctx)
    assert ok is False
    assert "no published completion marker" in why, why


# ------------------------------------------------ 收尾：正常场景仍能 DONE

def test_finish_job_completion_logs_done_only_when_proven(tmp_path, monkeypatch):
    ctx = build(tmp_path)
    log = _Log()
    monkeypatch.setattr(vmain, "find_final_output",
                        lambda *a, **k: [ctx["out"]])
    rc, why = vmain.finish_job_completion(
        ctx["job"], {}, {"stage": None}, ctx["params"], log, log)
    assert (rc, why) == (0, ""), (rc, why)
    assert log.names().count("JOB_DONE") == 1
    assert "COMPLETION_VERIFIED" in log.names()


def test_partial_stage_run_does_not_claim_a_completed_generation(tmp_path):
    """`--stage N` 是单步调试：可以返回成功，但不是"整代完成"。"""
    ctx = build(tmp_path)
    log = _Log()
    rc, why = vmain.finish_job_completion(
        ctx["job"], {}, {"stage": 3}, ctx["params"], log, log)
    assert (rc, why) == (0, "")
    assert "JOB_PARTIAL_STAGE_RUN" in log.names()
    assert "JOB_DONE" not in log.names()
