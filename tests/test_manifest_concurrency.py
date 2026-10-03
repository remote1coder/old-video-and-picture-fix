"""V08 regression: concurrent manifest updates must not lose completions."""

import json
import multiprocessing
import os

import pytest

import conftest  # noqa: F401
from manifest_loader import (load_manifest, save_manifest,
                             save_manifest_job_result, update_job_status,
                             update_manifest, _unique_tmp_path,
                             ManifestLockError)


def _write(path, jobs):
    with open(path, "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")


def _jobs(n=2):
    return [{"job_id": f"job{i}", "input_path": f"/in/clip{i}.mp4",
             "detected_class": "dv_digital", "status": "PENDING",
             "priority": 0} for i in range(n)]


def _statuses(path):
    m = load_manifest(path)
    return {j["job_id"]: j["status"] for j in m["jobs"]}


def test_v08_concurrent_single_job_updates_keep_every_result(tmp_path):
    path = str(tmp_path / "m.jsonl")
    _write(path, _jobs(3))
    # Two "workers" each hold a stale snapshot, then report their own job.
    _stale_a = load_manifest(path)
    _stale_b = load_manifest(path)
    save_manifest_job_result(path, "job0", "DONE", attempt=0,
                             attempt_group="g1")
    save_manifest_job_result(path, "job1", "DONE", attempt=0,
                             attempt_group="g1")
    assert _statuses(path) == {"job0": "DONE", "job1": "DONE",
                               "job2": "PENDING"}


def test_v08_old_whole_snapshot_write_still_loses_updates(tmp_path):
    """Documents the defect: the legacy API is unsafe for multi-writer use."""
    path = str(tmp_path / "m.jsonl")
    _write(path, _jobs(2))
    stale = load_manifest(path)          # snapshot taken before job0 finished
    save_manifest_job_result(path, "job0", "DONE", attempt=0,
                             attempt_group="g1")
    update_job_status(stale, "job1", "DONE")   # other worker finishes
    save_manifest(stale, path)                 # writes its stale snapshot
    assert _statuses(path)["job0"] == "PENDING", (
        "the legacy whole-snapshot write regressed job0; this is why callers "
        "must use save_manifest_job_result")


def test_v08_stale_attempt_cannot_overwrite_newer_result(tmp_path):
    path = str(tmp_path / "m.jsonl")
    _write(path, _jobs(1))
    assert save_manifest_job_result(path, "job0", "DONE", attempt=2,
                                    attempt_group="g1")
    # a late result from an older attempt must be rejected
    assert not save_manifest_job_result(path, "job0", "FAILED", attempt=1,
                                        attempt_group="g1")
    assert _statuses(path)["job0"] == "DONE"
    # a new pipeline run (new group) is allowed to re-evaluate the job
    assert save_manifest_job_result(path, "job0", "FAILED", attempt=0,
                                    attempt_group="g2")
    assert _statuses(path)["job0"] == "FAILED"


def test_v08_corrupt_manifest_is_not_wiped(tmp_path):
    path = str(tmp_path / "m.jsonl")
    _write(path, _jobs(2))
    with open(path, "a", encoding="utf-8") as f:
        f.write("{not json\n")
    before = open(path, "rb").read()   # corrupt-but-present content
    assert update_manifest(path, lambda m: True) is False
    assert open(path, "rb").read() == before, "corrupt read must not rewrite"


def test_v08_temp_files_are_unique():
    from pathlib import Path
    a = _unique_tmp_path(Path("/tmp/m.jsonl"))
    b = _unique_tmp_path(Path("/tmp/m.jsonl"))
    assert a != b


def _mp_worker(path, job_ids, group):
    """Child process: report its own jobs through the locked API."""
    import conftest  # noqa: F401  (path bootstrap inside the spawned process)
    from manifest_loader import save_manifest_job_result
    for jid in job_ids:
        save_manifest_job_result(path, jid, "DONE", attempt=0,
                                 attempt_group=group)


def test_v08_real_multiprocess_writers(tmp_path):
    """4 real processes × 10 jobs each on one manifest file."""
    path = str(tmp_path / "m.jsonl")
    jobs = _jobs(40)
    _write(path, jobs)
    groups = [f"g{i}" for i in range(4)]
    ctx = multiprocessing.get_context("spawn")
    procs = []
    for i in range(4):
        ids = [j["job_id"] for j in jobs[i * 10:(i + 1) * 10]]
        p = ctx.Process(target=_mp_worker, args=(path, ids, groups[i]))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0, f"worker failed with {p.exitcode}"

    final = load_manifest(path)
    assert len(final["jobs"]) == 40
    assert all(j["status"] == "DONE" for j in final["jobs"]), \
        [j for j in final["jobs"] if j["status"] != "DONE"][:3]
    assert final["meta"]["done"] == 40
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert not leftovers, leftovers


def test_v08_lock_timeout_is_reported(tmp_path):
    path = str(tmp_path / "locked.jsonl")
    _write(path, _jobs(1))
    lock = tmp_path / "locked.jsonl.lock"
    with open(lock, "ab"):
        pass
    try:
        import msvcrt
        holder = open(lock, "a+b")
        msvcrt.locking(holder.fileno(), msvcrt.LK_NBLCK, 1)
    except Exception:
        pytest.skip("cannot hold an exclusive lock in this environment")
    try:
        with pytest.raises(ManifestLockError):
            save_manifest_job_result(path, "job0", "DONE", timeout=0.3)
    finally:
        try:
            msvcrt.locking(holder.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass
        holder.close()


def test_new_v07_lock_file_open_denied_is_reported_not_crashed(tmp_path,
                                                               monkeypatch):
    """NEW-V-07: 打开锁文件被拒绝必须报"锁超时"，不能崩成 PermissionError。

    实测触发条件（Windows）：另一个进程持有 msvcrt 锁区间时，
    `open(lock_path, "a+b")` 会抛 `PermissionError: [Errno 13]`。
    修复前它不在重试循环内 —— 4 进程并发时有 worker 直接崩（exitcode=1），
    既没等到锁也没报超时。这里用 monkeypatch 稳定复现"open 被拒绝"。
    """
    import manifest_loader as ml

    path = str(tmp_path / "m.jsonl")
    _write(path, _jobs(1))

    def denied(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(ml, "open", denied, raising=False)

    with pytest.raises(ml.ManifestLockError) as excinfo:
        ml.save_manifest_job_result(path, "job0", "DONE", timeout=0.3)

    assert "lock file" in str(excinfo.value), str(excinfo.value)


def test_new_v07_lock_file_open_denied_is_reported_not_crashed(tmp_path,
                                                               monkeypatch):
    """NEW-V-07: 打开锁文件被拒绝必须报"锁超时"，不能崩成 PermissionError。

    实测触发条件（Windows）：另一个进程持有 msvcrt 锁区间时，
    `open(lock_path, "a+b")` 会抛 `PermissionError: [Errno 13]`。
    修复前它不在重试循环内 —— 4 进程并发时有 worker 直接崩（exitcode=1），
    既没等到锁也没报超时。这里用 monkeypatch 稳定复现"open 被拒绝"。
    """
    import manifest_loader as ml

    path = str(tmp_path / "m.jsonl")
    _write(path, _jobs(1))

    def denied(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(ml, "open", denied, raising=False)

    with pytest.raises(ml.ManifestLockError) as excinfo:
        ml.save_manifest_job_result(path, "job0", "DONE", timeout=0.3)

    assert "lock file" in str(excinfo.value), str(excinfo.value)
