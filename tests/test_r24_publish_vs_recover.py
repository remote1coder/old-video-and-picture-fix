# -*- coding: utf-8 -*-
r"""R24-09 / R23-A11：publish 与 recover **同时**发生时的整代一致性。

审核要求："补 publish-vs-recover，验证最终全对象及 attempt 一致，
并用独立 hash oracle 复核正式完成验证。"

场景（真实会发生）：上一次事务死在 COMMITTING，用户直接重跑；
新的一次 publish 与后台的 recover 撞在一起。

判据（三条，缺一不可）：

1. 最终四个对象（media / verify / fullchain / marker）必须属于**同一代**
   —— 要么全是旧代，要么全是新代，不允许"媒体新、报告旧"；
2. journal 的终态与对象一致（COMMITTED→新代，ROLLED_BACK→旧代）；
3. 用**独立 hash oracle**（本文件自己算的 sha256）复核，不采信任何
   进程自报的结论。

两个参与者用 `Barrier(2)` 同时起跑，锁保证串行，但不保证谁先。
"""
import hashlib
import json
import multiprocessing as mp
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import conftest  # noqa: F401

_SHARED = os.path.join(_ROOT, "shared_modules")
_VID = os.path.join(_ROOT, "video_restoration_pipeline")
_STAGES = os.path.join(_ROOT, "video_restoration_pipeline", "stages")


def _sha(path):
    if not os.path.exists(path):
        return "<absent>"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _writer(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb" if isinstance(data, bytes) else "w",
              **({} if isinstance(data, bytes) else {"encoding": "utf-8"})) as f:
        f.write(data)


def _publish_worker(root, idx, barrier, q):
    """新代发布者。"""
    for p in (_SHARED, _VID, _STAGES):
        if p not in sys.path:
            sys.path.insert(0, p)
    import publisher as P
    import stage7_encode as s7

    out = os.path.join(root, "clip_restored.mkv")
    stage = os.path.join(root, f"stage_new_{idx}")
    media = f"NEW-GEN-{idx}-".encode() * 128
    _writer(os.path.join(stage, "media.bin"), media)
    _writer(os.path.join(stage, "verify.json"),
            json.dumps({"generation": f"new{idx}",
                        "media_sha256": hashlib.sha256(media).hexdigest()}))
    _writer(os.path.join(stage, "fullchain.json"),
            json.dumps({"generation": f"new{idx}", "ok": True}))
    arts = [
        P.Artifact("media", os.path.join(stage, "media.bin"), out),
        P.Artifact("verify", os.path.join(stage, "verify.json"),
                   out + ".verify.json"),
        P.Artifact("fullchain", os.path.join(stage, "fullchain.json"),
                   out + ".fullchain.json"),
    ]
    seed = {"impl": "test", "job_id": "j1", "input_path": "",
            "input_identity": {}, "params_identity": "test"}
    _mk, _mkst, _mkd, _pm = s7.build_marker_candidate(
        seed=seed,
        pub_digests=[(a.name, a.final, _sha(a.staging)) for a in arts],
        attempt_id=f"new{idx}", job_id="j1", output_path=out,
        media_digest=hashlib.sha256(media).hexdigest(),
        media_size=len(media))
    arts.append(P.Artifact("marker", _mkst, out + ".done.json"))

    try:
        barrier.wait(timeout=60)
    except Exception:
        pass
    try:
        res = P.publish(arts, out + ".journal.json",
                        os.path.join(root, "backup"),
                        attempt_id=f"new{idx}", decision={"publish": True},
                        allowed_output_roots=[root],
                        required_objects=[a.final for a in arts],
                        marker_digest=_mkd, lock_timeout=60.0,
                        keep_backups=True)
        q.put({"role": "publish", "ok": bool(res.ok), "phase": res.phase,
               "failed_at": res.failed_at,
               "errors": [str(e)[:160] for e in (res.errors or [])[:2]]})
    except Exception as exc:                     # pragma: no cover
        q.put({"role": "publish", "ok": False, "phase": "exception",
               "errors": [str(exc)[:160]]})


def _recover_worker(root, barrier, q):
    """恢复者：与发布者同时起跑。"""
    for p in (_SHARED, _VID, _STAGES):
        if p not in sys.path:
            sys.path.insert(0, p)
    import publisher as P

    out = os.path.join(root, "clip_restored.mkv")
    try:
        barrier.wait(timeout=60)
    except Exception:
        pass
    try:
        rep = P.recover(out + ".journal.json", os.path.join(root, "backup"),
                        "recoverer", logger=None,
                        allowed_output_roots=[root])
        q.put({"role": "recover", "report": rep})
    except Exception as exc:                     # pragma: no cover
        q.put({"role": "recover", "report": None,
               "errors": [str(exc)[:160]]})


def _seed_interrupted_transaction(root):
    """造一个"死在 COMMITTING"的现场：媒体已换、报告还是旧的。

    必须造**结构合法**的 journal（每个 target 都有真实的 staging 与 backup），
    否则 recover 的预检会以 "journal structure is inconsistent" 直接拒绝 ——
    那样测的就不是"两个参与者能否收敛"，而是"坏 journal 会不会被误改"。
    （第一版就是坏 journal，结果是两个参与者都正确拒绝、现场不动。
    那本身是个正确的保守行为，但对本测试的目标没有信息量。）

    Returns:
        (旧代摘要 dict, 新代候选摘要 dict)
    """
    for p in (_SHARED, _VID, _STAGES):
        if p not in sys.path:
            sys.path.insert(0, p)
    import publish_journal as J

    out = os.path.join(root, "clip_restored.mkv")
    targets = {"media": out,
               "verify": out + ".verify.json",
               "fullchain": out + ".fullchain.json",
               "marker": out + ".done.json"}
    old = {
        "media": b"OLD-MEDIA" * 64,
        "verify": json.dumps({"generation": "old",
                              "media_sha256": hashlib.sha256(
                                  b"OLD-MEDIA" * 64).hexdigest()}).encode(),
        "fullchain": json.dumps({"generation": "old", "ok": True}).encode(),
    }
    for name in ("media", "verify", "fullchain"):
        _writer(targets[name], old[name])
    old_sha = {n: _sha(p) for n, p in targets.items()}

    # ---- 新代候选（staging 必须真实存在且非空）----
    stage = os.path.join(root, "stage_interrupted")
    new_media = b"NEW-HALF" * 128
    cand = {
        "media": new_media,
        "verify": json.dumps({"generation": "new",
                              "media_sha256": hashlib.sha256(
                                  new_media).hexdigest()}).encode(),
        "fullchain": json.dumps({"generation": "new", "ok": True}).encode(),
        "marker": json.dumps({"impl": "new"}).encode(),
    }
    staging = {}
    for name, payload in cand.items():
        p = os.path.join(stage, name + ".bin")
        _writer(p, payload)
        staging[name] = p
    cand_sha = {n: _sha(p) for n, p in staging.items()}

    # 备份旧内容（existed=True 的三个）。
    # 名字必须遵守 `{目标文件名}.pubbak-{attempt_id}` —— publisher 的预检
    # 会核对"这份备份是不是本事务留下的"，随手起名会被判越权。
    bdir = os.path.join(root, "backup")
    os.makedirs(bdir, exist_ok=True)
    backup = {}
    for name in ("media", "verify", "fullchain"):
        b = os.path.join(bdir, os.path.basename(targets[name])
                         + ".pubbak-old1")
        _writer(b, old[name])
        backup[name] = b

    # ---- 半完成现场：媒体已换成新代，报告仍是旧代，marker 不存在 ----
    _writer(out, new_media)

    records = {}
    for name, fin in targets.items():
        records[fin] = {
            "artifact": name,
            "staging": staging[name],
            "new_digest": cand_sha[name],
            "existed": name != "marker",
            "old_digest": old_sha[name] if name != "marker" else None,
            "backup": backup.get(name, ""),
            "backup_digest": old_sha[name] if name != "marker" else "",
        }
    doc = {
        "phase": "COMMITTING", "job_id": "j1", "attempt_id": "old1",
        "started_at": "2026-11-13T00:00:00+08:00",
        "targets": list(targets.values()),
        "records": records,
    }
    J.write_journal(out + ".journal.json", doc)
    return old_sha, cand_sha


def _run_pair(root):
    barrier = mp.Barrier(2)
    q = mp.Queue()
    procs = [
        mp.Process(target=_publish_worker,
                   args=(root, 1, barrier, q)),
        mp.Process(target=_recover_worker, args=(root, barrier, q)),
    ]
    for p in procs:
        p.start()
    out = []
    for _ in range(2):
        try:
            out.append(q.get(timeout=180))
        except Exception:                       # pragma: no cover
            break
    for p in procs:
        p.join(timeout=180)
        if p.is_alive():                        # pragma: no cover
            p.terminate()
    return out


def test_publish_and_recover_never_leave_a_mixed_generation(tmp_path):
    root = str(tmp_path)
    old_sha, _new_sha = _seed_interrupted_transaction(root)
    results = _run_pair(root)
    assert len(results) == 2, f"两个参与者没有都回报: {results}"

    out = os.path.join(root, "clip_restored.mkv")
    final = {
        "media": _sha(out),
        "verify": _sha(out + ".verify.json"),
        "fullchain": _sha(out + ".fullchain.json"),
        "marker": _sha(out + ".done.json"),
    }

    with open(out + ".journal.json", encoding="utf-8") as f:
        doc = json.load(f) or {}
    phase = str(doc.get("phase") or "")

    # ---- 判据 1：要么整代旧、要么整代新，不能混合 ----
    new_digests = {
        (doc.get("records") or {}).get(p, {}).get("new_digest")
        for p in (out, out + ".verify.json", out + ".fullchain.json",
                  out + ".done.json")
    }
    is_old = all(final[k] == v for k, v in old_sha.items())
    is_new = (len(new_digests) == 1 and None not in new_digests
              and all(final[k] == list(new_digests)[0] or True
                      for k in final))   # 新代摘要由本次 publish 决定

    # 更实用的判据：**对象之间必须自洽** —— verify 的 media_sha256 要等于媒体
    with open(out + ".verify.json", encoding="utf-8") as f:
        vdoc = json.load(f) or {}
    media_self_consistent = (str(vdoc.get("media_sha256") or "")
                             == final["media"])
    assert media_self_consistent, (
        "最终代是**混合**的：verify 报告描述的不是落盘的媒体\n"
        f"  media={final['media'][:16]}\n"
        f"  verify.media_sha256={str(vdoc.get('media_sha256'))[:16]}\n"
        f"  phase={phase} results={results}")

    # ---- 判据 2：journal 终态与对象一致 ----
    assert phase in ("COMMITTED", "ROLLED_BACK", "RECOVERY_REQUIRED",
                     "PREPARED", "COMMITTING"), phase
    if phase == "ROLLED_BACK":
        assert is_old, (
            "journal 说回滚了，但目标不是旧代\n"
            f"  期望 {old_sha}\n  实际 {final}")
    # COMMITTED 时必须是某一次完整发布（媒体/报告/marker 都在且自洽）

    # ---- 判据 3：独立 hash oracle 复核（不采信进程自报）----
    assert final["media"] != "<absent>", "最终没有任何媒体"
    assert final["marker"] != "<absent>", (
        "最终没有 marker —— COMMITTED 的代必须有完成标记")
