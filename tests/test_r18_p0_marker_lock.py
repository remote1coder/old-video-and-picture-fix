# -*- coding: utf-8 -*-
"""第十八轮 P0-B / R17-01：完成标记的锁与归属。

## 缺陷一：锁从未被真正获取（已修）

`write_completion_marker` 原来这样用锁：

```python
_lock = acquire_target_locks([final_path], timeout=60, logger=logger)
try:
    ... 读 verify.json、算摘要、比对 ...
finally:
    _lock.__exit__(None, None, None)      # ← 从未 __enter__
```

`acquire_target_locks` 是 **generator 型**上下文管理器（`@contextmanager`）：
不 `next()` 就一行都不执行。后果有两个：

1. **核验在无锁状态下进行** —— 审核探针确认"核验 verify 时未持锁"。
   这意味着核验与 B 的替换之间没有任何互斥，A 完全可能读到"核验时是 A、
   写标记时是 B"的错配。
2. `finally` 里那句 `__exit__` 反而把生成器**启动**了：锁被获取后永不释放
   （`contextlib` 对"生成器没停"抛 `RuntimeError`，又被 `except` 吞掉）。

## 缺陷二：还没有跨代归属证据（本轮未修完，标记为 TODO）

即使锁用对了，只要 B 把**媒体和 verify.json 一起**换掉，两者仍然彼此自洽，
A 无法察觉"这份报告已经不是我那一代的了"。

彻底修法（审核要求）：publisher 把本次 attempt 的**冻结回执**传给 marker，
核对 attempt / 完整成果集合 / 媒体身份。当前 verify.json 里有 `attempt_id`
与 `media_sha256`，但**没有输入身份与参数身份**，所以 A 的输入仍可能被绑到
B 的媒体上。本文件末尾的 `test_T01_*_known_gap` 用 xfail 记录了这一点。
"""
import contextlib
import json
import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import main as vmain  # noqa: E402


class _Log:
    def __init__(self):
        self.events = []

    def info(self, e, m="", **k):
        self.events.append(e)

    def warning(self, e, m="", **k):
        self.events.append(e)

    def error(self, e, m="", **k):
        self.events.append(e)


def _job(inp):
    return {"job_id": "r18lock", "input_path": inp,
            "detected_class": "HDV_AVCHD", "recommended_params": {},
            "user_overrides": {}}


def _prepare(tmp_path, media_bytes=b"GEN-A", verify_sha=None):
    """造一份"成品 + verify 报告"。verify_sha=None 表示用成品真实摘要。"""
    media = tmp_path / "clip_restored.mkv"
    media.write_bytes(media_bytes)
    inp = tmp_path / "in.mp4"
    inp.write_bytes(b"src")
    sha = verify_sha if verify_sha is not None else vmain._plain_sha256(str(media))
    (tmp_path / "clip_restored.mkv.verify.json").write_text(
        json.dumps({"media_sha256": sha, "attempt_id": "A-1",
                    "run_identity": {
                        "input": vmain.build_input_identity(str(inp))}}),
        encoding="utf-8")
    return media, inp


def _publish_full(tmp_path, media_bytes=b"GEN-A", input_bytes=b"src"):
    """用**生产函数**发布一整代（媒体 + verify + fullchain + journal + marker
    + 私有 result），返回 ``(media_path, params)``。

    R22-A01：`write_completion_marker` 现在只验证、不写标记；标记由生产者在
    事务里生成。所以"正常路径"的夹具必须是**真实发布**的一整代 ——
    手造一个 marker 文件不再能代表"可证明的完成"。
    """
    _vstages = os.path.join(_ROOT, "video_restoration_pipeline", "stages")
    if _vstages not in sys.path:
        sys.path.insert(0, _vstages)
    import publisher as _P
    import stage7_encode as _s7

    out = str(tmp_path / "clip_restored.mkv")
    stage = str(tmp_path / "staging")
    inp = str(tmp_path / "in.mp4")
    if not os.path.exists(inp):
        with open(inp, "wb") as f:
            f.write(input_bytes)
    job = _job(inp)

    os.makedirs(stage, exist_ok=True)
    with open(os.path.join(stage, "media.bin"), "wb") as f:
        f.write(media_bytes)
    with open(os.path.join(stage, "verify.json"), "w", encoding="utf-8") as f:
        json.dump({"media_sha256": vmain._plain_sha256(
            os.path.join(stage, "media.bin"))}, f)
    with open(os.path.join(stage, "fullchain.json"), "w",
              encoding="utf-8") as f:
        json.dump({"ok": True}, f)

    arts = [
        _P.Artifact("media", os.path.join(stage, "media.bin"), out),
        _P.Artifact("verify", os.path.join(stage, "verify.json"),
                    out + ".verify.json"),
        _P.Artifact("fullchain", os.path.join(stage, "fullchain.json"),
                    out + ".fullchain.json"),
    ]
    digests = [(a.name, a.final, vmain._plain_sha256(a.staging)) for a in arts]
    eff = vmain.merge_params(job, {}, {}, None)
    eff["attempt_id"] = "att1"
    seed = {
        "impl": vmain.VIDEO_IMPL_VERSION,
        "job_id": job["job_id"],
        "input_path": inp,
        "input_identity": vmain.build_input_identity(inp),
        "params_identity": vmain.build_params_identity({}, job, None,
                                                       params=eff),
    }
    _mk, mk_stage, mk_dig, _pm = _s7.build_marker_candidate(
        seed=seed, pub_digests=digests, attempt_id="att1",
        job_id=job["job_id"], output_path=out, media_digest=digests[0][2],
        media_size=os.path.getsize(arts[0].staging))
    arts.append(_P.Artifact("marker", mk_stage, out + ".done.json"))
    res = _P.publish(arts, out + ".journal.json", str(tmp_path / "backup"),
                     attempt_id="att1", decision={"publish": True},
                     allowed_output_roots=[str(tmp_path)],
                     required_objects=[a.final for a in arts],
                     marker_digest=mk_dig)
    assert res.ok, res.errors

    result_path = str(tmp_path / "stage7_result.json")
    eff["_stage7_result_path"] = result_path
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "schema": 1, "attempt_id": "att1", "ok": True,
            "terminal_pending": False,
            "journal_phase": "COMMITTED",
            "journal_path": out + ".journal.json",
            "generation_receipt": (res.generation_receipt.to_dict()
                                   if res.generation_receipt else None),
            "final_output_path": out,
            "publish_result": res.as_dict(),
            "run_identity": vmain.build_run_identity({}, job, None,
                                                     params=eff),
        }, f, ensure_ascii=False, default=str)
    return out, eff


def test_R17_01_verification_runs_inside_the_lock(tmp_path, monkeypatch):
    """核验必须发生在**锁内**：事件顺序应为 enter → digest → exit。"""
    import publish_lock

    media, params = _publish_full(tmp_path)
    job = _job(str(tmp_path / "in.mp4"))
    order = []

    @contextlib.contextmanager
    def fake_locks(targets, timeout=30.0, poll_interval=0.05, logger=None):
        order.append("enter")
        try:
            yield [str(t) for t in targets]
        finally:
            order.append("exit")

    monkeypatch.setattr(publish_lock, "acquire_target_locks", fake_locks)

    real_plain = vmain._plain_sha256

    def spy_plain(path):
        order.append("digest")
        return real_plain(path)

    monkeypatch.setattr(vmain, "_plain_sha256", spy_plain)

    log = _Log()
    ok = vmain.write_completion_marker(str(media), job, {}, params=params,
                                       logger=log)

    assert ok is True, log.events
    # R22-A01：验证会核对**整代**（media/verify/... 各一次摘要），
    # 所以不能假设"只算一个摘要"。要断言的是：所有摘要都夹在锁内。
    assert order[0] == "enter", f"第一次动作必须是取锁，实际 {order}"
    assert order[-1] == "exit", f"最后一次动作必须是放锁，实际 {order}"
    assert order.count("digest") >= 1, order
    assert order.count("enter") == 1 and order.count("exit") == 1
    assert all(x == "digest" for x in order[1:-1]), \
        f"核验必须全部发生在锁内，实际顺序 {order}"


def test_R17_01_lock_released_when_verification_raises(tmp_path, monkeypatch):
    """核验抛异常时锁也必须释放 —— 不能像旧实现那样泄漏。"""
    import publish_lock

    media, params = _publish_full(tmp_path)
    job = _job(str(tmp_path / "in.mp4"))
    state = {"entered": 0, "exited": 0}

    @contextlib.contextmanager
    def fake_locks(targets, timeout=30.0, poll_interval=0.05, logger=None):
        state["entered"] += 1
        try:
            yield [str(t) for t in targets]
        finally:
            state["exited"] += 1

    monkeypatch.setattr(publish_lock, "acquire_target_locks", fake_locks)
    monkeypatch.setattr(vmain, "_plain_sha256",
                        lambda p: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        vmain.write_completion_marker(str(media), job, {}, params=params,
                                      logger=_Log())

    assert state["entered"] == 1
    assert state["exited"] == 1, "异常路径也必须释放锁"


def test_R17_01_marker_written_when_digest_matches(tmp_path, monkeypatch):
    """正常路径：摘要一致 → 写出标记（防止修复引入回归）。"""
    import publish_lock

    media, params = _publish_full(tmp_path)
    job = _job(str(tmp_path / "in.mp4"))

    @contextlib.contextmanager
    def fake_locks(targets, timeout=30.0, poll_interval=0.05, logger=None):
        yield [str(t) for t in targets]

    monkeypatch.setattr(publish_lock, "acquire_target_locks", fake_locks)

    log = _Log()
    assert vmain.write_completion_marker(str(media), job, {}, params=params,
                                         logger=log) is True
    # R22-A01：main 不再写标记 —— 它验证的是**生产者已经发布**的那一份
    assert os.path.exists(vmain.completion_marker_path(str(media)))


def test_R17_01_refuses_when_media_changed(tmp_path, monkeypatch):
    """媒体已换代（摘要对不上）→ 拒绝写标记。"""
    import publish_lock

    # R22-A01：先发布**完整**一整代，再替换媒体 —— 这样拒绝点落在
    # 摘要比对（而不是"根本没有 marker"）。
    media, params = _publish_full(tmp_path, media_bytes=b"GEN-A")
    job = _job(str(tmp_path / "in.mp4"))
    with open(media, "wb") as f:
        f.write(b"GEN-B")            # 同样 5 字节，内容不同

    @contextlib.contextmanager
    def fake_locks(targets, timeout=30.0, poll_interval=0.05, logger=None):
        yield [str(t) for t in targets]

    monkeypatch.setattr(publish_lock, "acquire_target_locks", fake_locks)

    log = _Log()
    assert vmain.write_completion_marker(str(media), job, {}, params=params,
                                         logger=log) is False
    assert any(e.startswith("COMPLETION_") for e in log.events), log.events


def test_T01_input_identity_cannot_bind_to_another_generation(tmp_path,
                                                              monkeypatch):
    """T01：A 的输入不得被绑到 B 的产物上。

    R22-A01：现在用**真实发布的一整代**做 B，再用 A 的 job 去验证 ——
    marker 里的 `input_identity` 是 B 的，A 必须被拒。
    """
    import publish_lock

    media, params = _publish_full(tmp_path, input_bytes=b"input-B-different")

    # A 的输入：内容与 B 不同
    inp_a = tmp_path / "a.mp4"
    inp_a.write_bytes(b"input-A")

    @contextlib.contextmanager
    def fake_locks(targets, timeout=30.0, poll_interval=0.05, logger=None):
        yield [str(t) for t in targets]

    monkeypatch.setattr(publish_lock, "acquire_target_locks", fake_locks)

    log = _Log()
    ok = vmain.write_completion_marker(str(media), _job(str(inp_a)), {},
                                       params=params, logger=log)

    assert ok is False, "A 的输入被绑到了 B 的产物上"
    assert any(e.startswith("COMPLETION_") for e in log.events), log.events
    # R22-A01：marker 是**发布时**就存在的；验证失败不该动它（保留证据，
    # 更不能"覆盖成一份新的"）。所以断言的是"没被改写"，不是"不存在"。
    assert os.path.exists(vmain.completion_marker_path(str(media))), \
        "验证失败不能删掉生产者已经发布的 marker"
