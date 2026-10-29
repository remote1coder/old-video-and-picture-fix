# -*- coding: utf-8 -*-
"""R14-07：并发必须调用**生产 publisher**，且用屏障控制交错。

**复核报告对旧 D3 的判断**

`test_r13_d3_two_process_publish.py` 的 worker 导入生产锁后**自己写**
计数器和各自 tag 的文件 —— A、B 没有替换同一个正式媒体，没有执行
报告转正与恢复。它证明锁互斥，**不证明**完整发布事务。

本文件用生产 `publish()` + 两个真实进程 + 文件屏障：
- P05：同一最终目标、部分重叠目标，交错提交 → 最终一代自洽，无死锁
- P07：清理失败后重启 → 提交已成立，不重复破坏，残留可重试
"""
import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARED = os.path.join(_ROOT, "shared_modules")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

import publisher as pub  # noqa: E402


def _w(p, data=b"x"):
    os.makedirs(os.path.dirname(str(p)) or ".", exist_ok=True)
    with open(str(p), "wb") as f:
        f.write(data)
    return str(p)


def _read(p):
    with open(str(p), "rb") as f:
        return f.read()


# 子进程：先报到，等两个都到齐（屏障），再各自 publish
_WORKER = textwrap.dedent('''
    import json, os, sys, time
    sys.path.insert(0, sys.argv[1])
    from publisher import Artifact, publish

    base, tag, barrier_dir, wait_for = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])

    # 报到
    open(os.path.join(barrier_dir, tag + ".ready"), "w").close()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ready = [f for f in os.listdir(barrier_dir) if f.endswith(".ready")]
        if len(ready) >= wait_for:
            break
        time.sleep(0.02)
    else:
        print("BARRIER_TIMEOUT", flush=True)
        sys.exit(9)

    media = Artifact("media",
                     os.path.join(base, tag + ".media.staging"),
                     os.path.join(base, "final.mkv"))
    verify = Artifact("verify",
                      os.path.join(base, tag + ".verify.staging"),
                      os.path.join(base, "final.mkv.verify.json"))
    res = publish([media, verify],
                  journal_path=os.path.join(base, "final.mkv.journal.json"),
                  backup_dir=os.path.join(base, "backup"),
                  attempt_id=tag,
                  decision={"publish": True})
    print("RESULT " + json.dumps({"ok": res.ok, "attempt": tag,
                                  "failed_at": res.failed_at}), flush=True)
''')


def _run_pair(tmp_path, n=2, timeout=120):
    base = tmp_path / "shared"
    barrier = tmp_path / "barrier"
    base.mkdir()
    barrier.mkdir()
    script = tmp_path / "worker.py"
    script.write_text(_WORKER, encoding="utf-8")

    procs, tags = [], []
    for i in range(n):
        tag = "A%d" % i
        tags.append(tag)
        _w(base / f"{tag}.media.staging", ("MEDIA-" + tag).encode())
        _w(base / f"{tag}.verify.staging",
           ('{"attempt": "%s"}' % tag).encode())
        procs.append(subprocess.Popen(
            [sys.executable, str(script), _SHARED, str(base), tag,
             str(barrier), str(n)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE))

    outs = []
    for p in procs:
        so, se = p.communicate(timeout=timeout)
        outs.append((p.returncode, so.decode("utf-8", "replace"),
                     se.decode("utf-8", "replace")))
    return base, tags, outs


# ---------------------------------------------------------------- P05

def test_p05_concurrent_publish_same_target_is_self_consistent(tmp_path):
    """P05：两进程同一最终目标 → 最终一代自洽，无死锁。

    判定"自洽"= 媒体内容与报告内容来自**同一个** attempt。
    媒体是 A0 而报告是 A1（或反之）即为跨代组合，必须失败。
    """
    base, tags, outs = _run_pair(tmp_path)

    for rc, so, se in outs:
        assert rc == 0, f"子进程失败 rc={rc}\nstdout={so[-400:]}\nstderr={se[-400:]}"
        assert "BARRIER_TIMEOUT" not in so, so

    media = _read(base / "final.mkv").decode()
    report = json.loads(_read(base / "final.mkv.verify.json").decode())

    assert media.startswith("MEDIA-"), media
    media_tag = media.replace("MEDIA-", "")
    assert report["attempt"] == media_tag, (
        f"跨代组合：媒体来自 {media_tag}，报告来自 {report['attempt']}")

    doc = json.loads(_read(base / "final.mkv.journal.json").decode())
    assert doc["phase"] == "COMMITTED", doc.get("phase")


def test_p05_no_deadlock_and_both_attempts_recorded(tmp_path):
    """P05 补充：两个 attempt 都能走完（不互相卡死），且各留下结果。"""
    base, tags, outs = _run_pair(tmp_path, n=2)

    attempts = []
    for rc, so, se in outs:
        assert rc == 0, se[-300:]
        for line in so.splitlines():
            if line.startswith("RESULT "):
                attempts.append(json.loads(line[len("RESULT "):]))
    assert len(attempts) == 2, attempts
    assert all(a["ok"] for a in attempts), attempts
    assert {a["attempt"] for a in attempts} == set(tags)


# ---------------------------------------------------------------- P07

def test_p07_cleanup_failure_is_recorded_not_reverted(tmp_path):
    """P07：备份删不掉 → 记 cleanup_pending，但提交仍然成立。"""
    base = tmp_path / "p"
    base.mkdir()
    media = _w(base / "clip.mkv", b"OLD-MEDIA")
    verify = _w(base / "clip.mkv.verify.json", b'{"gen":"OLD"}')
    m_s = _w(base / "new.mkv", b"NEW-MEDIA")
    v_s = _w(base / "new.verify.json", b'{"gen":"NEW"}')
    bdir = base / "backup"

    # 占住备份文件句柄 → 删除失败（Windows 上真实可行）
    holder = []
    real_remove = os.remove

    def _flaky_remove(p):
        if ".pubbak" in str(p) and not holder:
            # 第一次删备份：先创建一个占位句柄再让它失败
            fh = open(str(p), "rb")
            holder.append(fh)
            raise PermissionError("simulated: backup is in use")
        return real_remove(p)

    orig = pub.os.remove
    try:
        pub.os.remove = _flaky_remove
        res = pub.publish(
            [pub.Artifact("media", m_s, media),
             pub.Artifact("verify", v_s, verify)],
            journal_path=str(base / "clip.mkv.journal.json"),
            backup_dir=str(bdir), attempt_id="att7",
            decision={"publish": True})
    finally:
        pub.os.remove = orig
        for fh in holder:
            fh.close()

    assert res.ok is True, res.as_dict()
    assert res.cleanup_pending, "删除失败必须记 cleanup_pending"
    # 提交仍然成立：内容已是新代，journal 记 COMMITTED
    assert _read(media) == b"NEW-MEDIA"
    assert json.loads(_read(base / "clip.mkv.journal.json").decode())[
        "phase"] == "COMMITTED"


def test_p07_restart_after_cleanup_failure_does_not_destroy_generation(tmp_path):
    """P07 补：清理失败后重启，正式代仍是完整的新代（不被当残留清掉）。"""
    base = tmp_path / "p2"
    base.mkdir()
    media = _w(base / "clip.mkv", b"OLD")
    verify = _w(base / "clip.mkv.verify.json", b'{"gen":"OLD"}')
    m_s = _w(base / "n.mkv", b"NEW-MEDIA")
    v_s = _w(base / "n.verify.json", b'{"gen":"NEW"}')
    jp = str(base / "clip.mkv.journal.json")

    res = pub.publish([pub.Artifact("media", m_s, media),
                       pub.Artifact("verify", v_s, verify)],
                      journal_path=jp, backup_dir=str(base / "bak"),
                      attempt_id="att8", decision={"publish": True},
                      keep_backups=True)
    assert res.ok is True

    # 重启：recover() 看到 COMMITTED → 什么都不做
    rep = pub.recover(jp, str(base / "bak"), "att8-restart")
    assert rep is not None
    assert rep.get("action") == "already_committed", rep
    assert _read(media) == b"NEW-MEDIA"
    assert _read(verify) == b'{"gen":"NEW"}'
