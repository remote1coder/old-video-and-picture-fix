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


def test_R17_01_verification_runs_inside_the_lock(tmp_path, monkeypatch):
    """核验必须发生在**锁内**：事件顺序应为 enter → digest → exit。"""
    import publish_lock

    media, inp = _prepare(tmp_path)
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
    ok = vmain.write_completion_marker(str(media), _job(str(inp)), {}, log)

    assert ok is True, log.events
    assert order[:3] == ["enter", "digest", "exit"], \
        f"核验必须夹在 enter/exit 之间，实际顺序 {order}"
    assert order.count("enter") == 1 and order.count("exit") == 1


def test_R17_01_lock_released_when_verification_raises(tmp_path, monkeypatch):
    """核验抛异常时锁也必须释放 —— 不能像旧实现那样泄漏。"""
    import publish_lock

    media, inp = _prepare(tmp_path)
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
        vmain.write_completion_marker(str(media), _job(str(inp)), {}, _Log())

    assert state["entered"] == 1
    assert state["exited"] == 1, "异常路径也必须释放锁"


def test_R17_01_marker_written_when_digest_matches(tmp_path, monkeypatch):
    """正常路径：摘要一致 → 写出标记（防止修复引入回归）。"""
    import publish_lock

    media, inp = _prepare(tmp_path)

    @contextlib.contextmanager
    def fake_locks(targets, timeout=30.0, poll_interval=0.05, logger=None):
        yield [str(t) for t in targets]

    monkeypatch.setattr(publish_lock, "acquire_target_locks", fake_locks)

    log = _Log()
    assert vmain.write_completion_marker(str(media), _job(str(inp)), {},
                                         log) is True
    assert os.path.exists(vmain.completion_marker_path(str(media)))


def test_R17_01_refuses_when_media_changed(tmp_path, monkeypatch):
    """媒体已换代（摘要对不上）→ 拒绝写标记。"""
    import publish_lock

    other = tmp_path / "other.bin"
    other.write_bytes(b"GEN-B")
    media, inp = _prepare(tmp_path, verify_sha=vmain._plain_sha256(str(other)))

    @contextlib.contextmanager
    def fake_locks(targets, timeout=30.0, poll_interval=0.05, logger=None):
        yield [str(t) for t in targets]

    monkeypatch.setattr(publish_lock, "acquire_target_locks", fake_locks)

    log = _Log()
    assert vmain.write_completion_marker(str(media), _job(str(inp)), {},
                                         log) is False
    assert "COMPLETION_MARKER_GENERATION_CHANGED" in log.events
    assert not os.path.exists(vmain.completion_marker_path(str(media)))


def test_T01_input_identity_cannot_bind_to_another_generation(tmp_path,
                                                              monkeypatch):
    """T01：A 的输入不得被绑到 B 的媒体上。

    B 用自己的输入跑出另一份媒体，并把媒体与 verify.json 一起替换
    （两者互相自洽）。此时 A 再来写标记，**应当**发现"这份报告不是我的"。
    """
    import publish_lock

    # A 的输入
    inp_a = tmp_path / "a.mp4"
    inp_a.write_bytes(b"input-A")
    # B 的输入（不同内容）
    inp_b = tmp_path / "b.mp4"
    inp_b.write_bytes(b"input-B-different")

    media = tmp_path / "clip_restored.mkv"
    media.write_bytes(b"GEN-B-RESULT")

    # B 的 verify 报告：媒体摘要与**B 自己的输入身份**都自洽
    (tmp_path / "clip_restored.mkv.verify.json").write_text(
        json.dumps({"media_sha256": vmain._plain_sha256(str(media)),
                    "attempt_id": "B-1",
                    "run_identity": {
                        "input": vmain.build_input_identity(str(inp_b))}}),
        encoding="utf-8")

    @contextlib.contextmanager
    def fake_locks(targets, timeout=30.0, poll_interval=0.05, logger=None):
        yield [str(t) for t in targets]

    monkeypatch.setattr(publish_lock, "acquire_target_locks", fake_locks)

    log = _Log()
    ok = vmain.write_completion_marker(str(media), _job(str(inp_a)), {}, log)

    # 期望：拒绝（因为报告属于 B，不是 A）
    assert ok is False, "A 的输入被绑到了 B 的媒体上"
    assert "COMPLETION_MARKER_FOREIGN_GENERATION" in log.events, log.events
    assert not os.path.exists(vmain.completion_marker_path(str(media)))
