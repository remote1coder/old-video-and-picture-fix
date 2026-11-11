# -*- coding: utf-8 -*-
"""R21-C07：copyback 的本地两目录协议（redirected/ → final/）。

## 审核原文（R20 §3，P1）

> copyback 的 artifacts 指向 `original_output_path`，`allowed_output_roots`
> 却来自重定向后的 `output_path`。首次无旧 journal 时不一定立即暴露，
> 恢复时会出现根授权与目标不匹配。与此同时，私有结果在 copyback 之前写出，
> 证明的是中转代；copyback 仅检查 `_wres.ok`，没有同样处理 `terminal_pending`。

## 本文件的范围（诚实标注）

**本地两目录**：`E:` 盘本轮隔离根下的 `redirected/` 与 `final/`。
**不是**真实跨卷、也不是真实 WSL —— 平台差异另列未验。
复制步骤调用**生产函数** `stage7.stage_generation_copy()`，
不是测试里另写一遍。

## 锁什么

1. 源侧复制使用冻结摘要，源被换就中止（不拷混代）
2. 目标侧走 attempt 私有暂存 + 统一事务
3. 目标侧有自己的 marker（不是把中转代的 marker 拷过去）
4. 两端都能被 `verify_generation_objects()` 判成完整代
5. 目标侧 COMMITTED 失败 ⇒ `terminal_pending`，不得当完成
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
import stage7_encode as s7  # noqa: E402


def _w(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(path, mode, **({} if mode == "wb" else {"encoding": "utf-8"})) as f:
        f.write(data)
    return path


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _build_and_publish(root, job, media=b"GENERATION-BYTES", attempt="att1"):
    """在 `root` 下造一整代并发布（真实 publisher + 真实 marker 生产者）。"""
    out = str(root / "clip_restored.mkv")
    stage = root / "staging"
    _w(str(stage / "media.bin"), media)
    _w(str(stage / "verify.json"),
       json.dumps({"media_sha256": _digest(str(stage / "media.bin"))}))
    _w(str(stage / "fullchain.json"), json.dumps({"ok": True}))
    arts = [
        P.Artifact("media", str(stage / "media.bin"), out),
        P.Artifact("verify", str(stage / "verify.json"), out + ".verify.json"),
        P.Artifact("fullchain", str(stage / "fullchain.json"),
                   out + ".fullchain.json"),
    ]
    digests = [(a.name, a.final, _digest(a.staging)) for a in arts]
    seed = {
        "impl": vmain.VIDEO_IMPL_VERSION,
        "job_id": job["job_id"],
        "input_path": job["input_path"],
        "input_identity": vmain.build_input_identity(job["input_path"]),
        "params_identity": vmain.build_params_identity({}, job, None),
    }
    _mk, mk_stage, mk_dig, _pm = s7.build_marker_candidate(
        seed=seed, pub_digests=digests, attempt_id=attempt,
        job_id=job["job_id"], output_path=out,
        media_digest=digests[0][2], media_size=os.path.getsize(arts[0].staging))
    arts.append(P.Artifact("marker", mk_stage, out + ".done.json"))
    res = P.publish(arts, out + ".journal.json", str(root / "backup"),
                    attempt_id=attempt, decision={"publish": True},
                    allowed_output_roots=[str(root)],
                    required_objects=[a.final for a in arts],
                    marker_digest=mk_dig)
    assert res.ok is True, res.errors
    return {"out": out, "digests": digests, "seed": seed}


def _copyback(redirected, final_root, job, attempt="att1"):
    """用生产函数把整代复制到 final/ 并作为**新的一代**提交。"""
    src_entries = [(n, f, d) for (n, f, d) in redirected["digests"]]
    staged, digests = s7.stage_generation_copy(
        src_entries, str(final_root / "clip_restored.mkv"), attempt, tag="wsl")
    arts = [P.Artifact(n, st, fin) for (n, st, fin) in staged]

    media_final, media_dig = digests[0][1], digests[0][2]
    # 大小取**暂存**文件 —— 目标路径此刻还不存在（发布前它不该存在）
    media_size = os.path.getsize(staged[0][1])
    _mk, mk_stage, mk_dig, _pm = s7.build_marker_candidate(
        seed=redirected["seed"], pub_digests=digests,
        attempt_id=f"{attempt}-wsl", job_id=job["job_id"],
        output_path=media_final, media_digest=media_dig,
        media_size=media_size)
    arts.append(P.Artifact("marker", mk_stage, media_final + ".done.json"))

    res = P.publish(arts, media_final + ".journal.json",
                    str(final_root / "backup"), attempt_id=f"{attempt}-wsl",
                    decision={"publish": True},
                    # 授权根来自**最终目标**（R21-C07 的核心修正）
                    allowed_output_roots=[str(final_root)],
                    required_objects=[a.final for a in arts],
                    marker_digest=mk_dig)
    return res, media_final


# ------------------------------------------------ 正常路径

def test_copyback_publishes_a_complete_generation_on_both_sides(tmp_path):
    job = {"job_id": "j1", "input_path": str(tmp_path / "src.mp4")}
    _w(job["input_path"], b"SOURCE")
    red = tmp_path / "redirected"
    fin = tmp_path / "final"
    red.mkdir()
    fin.mkdir()

    src = _build_and_publish(red, job)
    res, final_media = _copyback(src, fin, job)
    assert res.ok is True, res.errors

    # 两端都是完整代（同一条验证器）
    ok_r, why_r = vmain.verify_generation_objects(src["out"], job)
    assert ok_r is True, why_r
    ok_f, why_f = vmain.verify_generation_objects(final_media, job)
    assert ok_f is True, why_f

    # 目标侧有**自己的** marker，且引用的是目标侧路径
    m = json.loads(open(final_media + ".done.json", encoding="utf-8").read())
    paths = [o["path"] for o in m["payload_manifest"]["objects"]]
    assert all(str(fin) in p for p in paths), paths
    # 中转代的 marker 仍然引用中转路径（没有被改写）
    m0 = json.loads(open(src["out"] + ".done.json", encoding="utf-8").read())
    paths0 = [o["path"] for o in m0["payload_manifest"]["objects"]]
    assert all(str(red) in p for p in paths0), paths0


def test_copyback_bytes_match_the_redirected_generation(tmp_path):
    job = {"job_id": "j1", "input_path": str(tmp_path / "src.mp4")}
    _w(job["input_path"], b"SOURCE")
    (tmp_path / "redirected").mkdir()
    (tmp_path / "final").mkdir()
    src = _build_and_publish(tmp_path / "redirected", job)
    res, final_media = _copyback(src, tmp_path / "final", job)
    assert res.ok is True
    assert _digest(src["out"]) == _digest(final_media)
    assert (_digest(src["out"] + ".verify.json")
            == _digest(final_media + ".verify.json"))


def test_final_target_marker_is_verifiable_by_main(tmp_path):
    job = {"job_id": "j1", "input_path": str(tmp_path / "src.mp4")}
    _w(job["input_path"], b"SOURCE")
    (tmp_path / "redirected").mkdir()
    (tmp_path / "final").mkdir()
    src = _build_and_publish(tmp_path / "redirected", job)
    _res, final_media = _copyback(src, tmp_path / "final", job)

    ok, why = vmain.verify_published_marker(final_media, job, {}, None, None)
    assert ok is True, why
    ok2, why2 = vmain.resume_artifact_matches(final_media, job, {}, None)
    assert ok2 is True, why2


# ------------------------------------------------ 反例

def test_source_changed_between_freeze_and_copy_aborts(tmp_path):
    """复制前核对冻结摘要：源被换 → 中止，不产生混代。"""
    job = {"job_id": "j1", "input_path": str(tmp_path / "src.mp4")}
    _w(job["input_path"], b"SOURCE")
    (tmp_path / "redirected").mkdir()
    (tmp_path / "final").mkdir()
    src = _build_and_publish(tmp_path / "redirected", job)

    # 冻结后再改源（模拟下一代替换）
    _w(src["out"], b"OTHER-BYTES")

    with pytest.raises(OSError) as ei:
        s7.stage_generation_copy(
            src["digests"], str(tmp_path / "final" / "clip_restored.mkv"),
            "att1", tag="wsl")
    assert "source changed underneath" in str(ei.value)


def test_missing_report_in_the_source_aborts_the_copy(tmp_path):
    job = {"job_id": "j1", "input_path": str(tmp_path / "src.mp4")}
    _w(job["input_path"], b"SOURCE")
    (tmp_path / "redirected").mkdir()
    (tmp_path / "final").mkdir()
    src = _build_and_publish(tmp_path / "redirected", job)
    os.remove(src["out"] + ".fullchain.json")

    with pytest.raises(OSError) as ei:
        s7.stage_generation_copy(
            src["digests"], str(tmp_path / "final" / "clip_restored.mkv"),
            "att1", tag="wsl")
    assert "fullchain" in str(ei.value) or "source changed" in str(ei.value)


def test_target_side_commit_failure_does_not_claim_completion(tmp_path):
    """目标侧 COMMITTED 写失败 → terminal_pending → 不得当完成。"""
    job = {"job_id": "j1", "input_path": str(tmp_path / "src.mp4")}
    _w(job["input_path"], b"SOURCE")
    (tmp_path / "redirected").mkdir()
    (tmp_path / "final").mkdir()
    src = _build_and_publish(tmp_path / "redirected", job)

    real_write = P.write_journal if hasattr(P, "write_journal") else None

    # 让目标侧的 COMMITTED 写失败：第二次 publish 的 journal 写入返回 False
    import publish_journal as J
    calls = {"n": 0}

    def _flaky(path, payload):
        calls["n"] += 1
        if payload.get("phase") == "COMMITTED" and "final" in str(path):
            return False
        return real_write(path, payload) if real_write else J.write_journal(
            path, payload)

    import publisher
    publisher.write_journal = _flaky
    try:
        res, final_media = _copyback(src, tmp_path / "final", job)
    finally:
        publisher.write_journal = real_write or J.write_journal

    assert res.ok is True, res.errors
    assert getattr(res, "terminal_pending", False) is True, (
        "目标侧 COMMITTED 写失败必须标 terminal_pending")
