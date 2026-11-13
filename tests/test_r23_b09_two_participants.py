# -*- coding: utf-8 -*-
"""R23-09：**两个真实锁参与者**（barrier 同步），不是 monkeypatch 假并发。

## 审核的判词

> 所谓并发测试实际使用同步 monkeypatch，未验证两个真实锁参与者。
> ……应补真实同步事件、精确旧/新代期望值、完整既有 marker 替换及首次发布
> 两种矩阵；区分进程崩溃验证与断电持久性。

所以这里：

- **两个独立进程**（`multiprocessing` spawn），各自调用生产 `publisher.publish`；
- 用 `Barrier(2)` 让它们**同时**冲进临界区，不靠 sleep 撞时间；
- 两种矩阵：**首次发布**（目标不存在）与**既有旧代 + 旧 marker**（替换）；
- 断言的是"最终只有一整代"：媒体摘要、verify 摘要、journal targets 三者
  必须来自**同一个** participant，不能 A 的媒体配 B 的报告。

## 边界（不许外推）

这是**进程级**并发 + 文件锁验证，**不是**断电/掉盘持久性测试；
也不承诺"防住任何不遵守协议的写入"（那条由 `_restore_locked` 的
复制后再核目标来兜，见 `test_r23_b09_restore_window.py`）。
"""
import hashlib
import json
import multiprocessing as mp
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARED = os.path.join(_ROOT, "shared_modules")
_STAGES = os.path.join(_ROOT, "video_restoration_pipeline", "stages")
for _p in (_ROOT, _SHARED, _STAGES):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


# --------------------------------------------------------------------------
# 子进程侧：必须是**模块级**函数（spawn 要能 pickle 到它）
# --------------------------------------------------------------------------

def _worker(shared_dir, root, idx, barrier, q, timeout):
    """一个参与者：造自己那一代的候选件，等着同时开跑，然后 publish。"""
    import sys as _sys
    if shared_dir not in _sys.path:
        _sys.path.insert(0, shared_dir)
    import publisher as P

    out = os.path.join(root, "clip_restored.mkv")
    stage = os.path.join(root, f"staging_{idx}")
    os.makedirs(stage, exist_ok=True)

    media = f"MEDIA-GENERATION-{idx}-".encode() * 64
    verify = json.dumps({"generation": idx,
                         "media_sha256": _sha_bytes(media)}).encode()
    full = json.dumps({"generation": idx}).encode()
    for name, data in (("media.bin", media), ("verify.json", verify),
                       ("fullchain.json", full)):
        with open(os.path.join(stage, name), "wb") as f:
            f.write(data)

    arts = [
        P.Artifact("media", os.path.join(stage, "media.bin"), out),
        P.Artifact("verify", os.path.join(stage, "verify.json"),
                   out + ".verify.json"),
        P.Artifact("fullchain", os.path.join(stage, "fullchain.json"),
                   out + ".fullchain.json"),
    ]
    # **必须包含 marker**：上一代的 journal targets 里有 `.done.json`，
    # 这一代若只锁三件，publisher 会（正确地）判"有目标不在锁集合内"并拒绝 ——
    # 否则旧 marker 会留下来描述新媒体。
    import stage7_encode as _s7
    _seed = {"impl": "test", "job_id": "j1", "input_path": "",
             "input_identity": {}, "params_identity": "test"}
    _mk, _mkst, _mkd, _pm = _s7.build_marker_candidate(
        seed=_seed,
        pub_digests=[(a.name, a.final, _sha_file(a.staging)) for a in arts],
        attempt_id=f"att{idx}", job_id="j1", output_path=out,
        media_digest=_sha_bytes(media), media_size=len(media))
    arts.append(P.Artifact("marker", _mkst, out + ".done.json"))
    try:
        barrier.wait(timeout=60)
    except Exception:
        pass
    _t0 = __import__("time").monotonic()
    # R24-09：让 publisher 记下"进入锁内/离开锁内"两个事件。
    # 只有这两个点才能证明互斥；`t0/t1` 是**调用区间**（含等锁），
    # 重叠只能证明"有争用"。
    _trace = {}
    try:
        res = P.publish(
            arts, out + ".journal.json", os.path.join(root, "backup"),
            attempt_id=f"att{idx}", decision={"publish": True},
            allowed_output_roots=[root],
            required_objects=[a.final for a in arts],
            marker_digest=_mkd,
            lock_timeout=float(timeout), keep_backups=True,
            trace=_trace)
        q.put({"idx": idx, "ok": bool(res.ok), "phase": res.phase,
               "failed_at": res.failed_at,
               "t0": _t0, "t1": __import__("time").monotonic(),
               "lock_acquired_at": _trace.get("lock_acquired_at"),
               "lock_released_at": _trace.get("lock_released_at"),
               "errors": [str(e)[:200] for e in (res.errors or [])[:2]]})
    except Exception as exc:            # pragma: no cover - 子进程兜底
        q.put({"idx": idx, "ok": False, "phase": "exception",
               "failed_at": "exception", "errors": [str(exc)[:200]]})


def _run_pair(root, timeout=45):
    """起两个真实进程，用 Barrier 同时开跑，收集两边结论。"""
    barrier = mp.Barrier(2)
    q = mp.Queue()
    procs = [mp.Process(target=_worker,
                        args=(_SHARED, root, i, barrier, q, timeout))
             for i in (1, 2)]
    for p in procs:
        p.start()
    out = []
    for _ in range(2):
        try:
            out.append(q.get(timeout=timeout + 60))
        except Exception:               # pragma: no cover - 真挂了
            break
    for p in procs:
        p.join(timeout=timeout + 60)
        if p.is_alive():                # pragma: no cover
            p.terminate()
    return out


def _assert_overlapped(results):
    """两个参与者的临界区必须**在时间上重叠**。

    否则"并发测试"可能只是两个进程先后跑完 —— 那验证不到锁。
    """
    a, b = results[0], results[1]
    overlap = min(a["t1"], b["t1"]) - max(a["t0"], b["t0"])
    assert overlap > 0, (
        f"两个参与者没有重叠（a={a['t0']:.3f}..{a['t1']:.3f}, "
        f"b={b['t0']:.3f}..{b['t1']:.3f}）—— 这不是并发")


def _assert_lock_sections_disjoint(results):
    """**锁内**区间不能重叠（R24-09 / R23-A11）。

    审核原话：旧测试的时间重叠是整个 publish 调用区间、含等锁 ——
    那是**争用证据**，不是锁内临界区重叠。正确互斥要求锁内区间不重叠。

    这里用 publisher 打的 `lock_acquired_at` / `lock_released_at`：
    两个参与者的 `[acquired, released]` 必须没有交集。
    """
    a, b = results[0], results[1]
    for r in (a, b):
        assert r.get("lock_acquired_at") is not None, (
            "publisher 没有回报进入锁内的时间；互斥无法证明")
        assert r.get("lock_released_at") is not None, (
            "publisher 没有回报离开锁内的时间；互斥无法证明")
    overlap = min(a["lock_released_at"], b["lock_released_at"]) - max(
        a["lock_acquired_at"], b["lock_acquired_at"])
    assert overlap <= 0, (
        "两个参与者的**锁内区间**重叠了 %.4f 秒 —— 互斥失效\n"
        "a: 锁内 %.4f..%.4f / 调用 %.4f..%.4f\n"
        "b: 锁内 %.4f..%.4f / 调用 %.4f..%.4f"
        % (overlap,
           a["lock_acquired_at"], a["lock_released_at"], a["t0"], a["t1"],
           b["lock_acquired_at"], b["lock_released_at"], b["t0"], b["t1"]))
    # 同时确认"锁内区间"确实是调用区间的子集（打点没装错位置）
    for r in (a, b):
        assert r["t0"] <= r["lock_acquired_at"] <= r["lock_released_at"] \
            <= r["t1"] + 1e-6, (
            "锁内打点不在调用区间内，说明 trace 接错了位置: %s" % (r,))


def _assert_one_complete_generation(root):
    """最终必须**只有一整代**：媒体/verify/journal 同源。"""
    out = os.path.join(root, "clip_restored.mkv")
    assert os.path.exists(out), "没有任何一代落地"
    media_digest = _sha_file(out)

    with open(out + ".verify.json", "r", encoding="utf-8") as f:
        verify = json.load(f)
    assert verify.get("media_sha256") == media_digest, (
        "verify 报告描述的媒体与落盘的媒体不是同一份（跨代混合）")

    with open(out + ".journal.json", "r", encoding="utf-8") as f:
        journal = json.load(f)
    assert journal.get("phase") == "COMMITTED", journal.get("phase")
    targets = {os.path.abspath(x) for x in (journal.get("targets") or [])}
    for must in (out, out + ".verify.json", out + ".fullchain.json"):
        assert os.path.abspath(must) in targets, (
            f"journal targets 缺少 {os.path.basename(must)}")
    assert journal.get("resolved") in (True, None)

    with open(out + ".fullchain.json", "r", encoding="utf-8") as f:
        full = json.load(f)
    assert full.get("generation") == verify.get("generation"), (
        "fullchain 与 verify 来自不同代")
    return verify.get("generation")


# ------------------------------------------------ 矩阵一：首次发布

def test_two_real_participants_first_publish(tmp_path):
    root = str(tmp_path / "out")
    os.makedirs(root, exist_ok=True)
    results = _run_pair(root)
    assert len(results) == 2, f"没有两个参与者都返回：{results}"
    assert any(r["ok"] for r in results), f"两方都没成功：{results}"
    _assert_overlapped(results)
    _assert_lock_sections_disjoint(results)
    winners = [r for r in results if r["ok"]]
    losers = [r for r in results if not r["ok"]]
    for r in losers:
        # 输了必须是**明确的**锁/事务结论，不是"莫名其妙失败"
        assert r["failed_at"] in ("lock", "precheck", "decision",
                                  "journal", None), r
    gen = _assert_one_complete_generation(root)
    assert gen in (1, 2), gen
    # 最终落地的那一代必须由**某个成功参与者**写的。
    # 注意"两个都成功"是合法的：锁是串行的，后一个会完整替换前一代 ——
    # 关键不是"只能有一个成功"，而是"最后只剩一整代、不混"。
    assert gen in [w["idx"] for w in winners], (winners, gen)


# ------------------------------------------------ 矩阵二：既有旧代 + 旧 marker

def _seed_old_generation(root):
    """用生产 publisher 先提交一整代，充当"旧代 + 旧 marker"。"""
    import publisher as P
    import stage7_encode as s7

    out = os.path.join(root, "clip_restored.mkv")
    stage = os.path.join(root, "staging_old")
    os.makedirs(stage, exist_ok=True)
    media = b"OLD-GENERATION-" * 64
    verify = json.dumps({"generation": 0,
                         "media_sha256": _sha_bytes(media)}).encode()
    full = json.dumps({"generation": 0}).encode()
    for name, data in (("media.bin", media), ("verify.json", verify),
                       ("fullchain.json", full)):
        with open(os.path.join(stage, name), "wb") as f:
            f.write(data)
    arts = [
        P.Artifact("media", os.path.join(stage, "media.bin"), out),
        P.Artifact("verify", os.path.join(stage, "verify.json"),
                   out + ".verify.json"),
        P.Artifact("fullchain", os.path.join(stage, "fullchain.json"),
                   out + ".fullchain.json"),
    ]
    seed = {"impl": "test", "job_id": "j1", "input_path": "",
            "input_identity": {}, "params_identity": "test"}
    _mk, mk_stage, mk_dig, _pm = s7.build_marker_candidate(
        seed=seed,
        pub_digests=[(a.name, a.final, _sha_file(a.staging)) for a in arts],
        attempt_id="att0", job_id="j1", output_path=out,
        media_digest=_sha_bytes(media), media_size=len(media))
    arts.append(P.Artifact("marker", mk_stage, out + ".done.json"))
    res = P.publish(arts, out + ".journal.json", os.path.join(root, "backup"),
                    attempt_id="att0", decision={"publish": True},
                    allowed_output_roots=[root],
                    required_objects=[a.final for a in arts],
                    marker_digest=mk_dig, keep_backups=True)
    assert res.ok, res.errors
    return out


def test_two_real_participants_replace_an_existing_generation(tmp_path):
    root = str(tmp_path / "out")
    os.makedirs(root, exist_ok=True)
    out = _seed_old_generation(root)
    old_digest = _sha_file(out)

    results = _run_pair(root)
    assert len(results) == 2, results
    assert any(r["ok"] for r in results), results
    _assert_overlapped(results)
    _assert_lock_sections_disjoint(results)
    gen = _assert_one_complete_generation(root)
    assert gen in (1, 2), gen

    # 旧代要么被完整替换，要么（两方都没成时）完整保留 —— 但两方至少一方
    # 成功，所以这里必须**已经换了**，而且换成了赢家那一代。
    new_digest = _sha_file(out)
    assert new_digest != old_digest, "赢了却还是旧代的内容"
    assert new_digest == _sha_bytes(
        (f"MEDIA-GENERATION-{gen}-".encode() * 64)), "落盘内容不是赢家那一代"

    # 旧 marker 必须被**替换**（不能留着描述旧媒体）
    with open(out + ".done.json", "r", encoding="utf-8") as f:
        marker = json.load(f)
    assert marker.get("output_sha256") == new_digest, (
        "marker 仍然描述旧代（没有随事务替换）")
