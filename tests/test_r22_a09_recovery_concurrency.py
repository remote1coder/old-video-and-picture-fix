# -*- coding: utf-8 -*-
"""R22-A09：恢复的并发合同 —— 用**事件点**控制，不用 sleep。

## 审核要求（原文摘）

> 事务指纹应覆盖影响恢复语义的 records、授权/对象绑定，而不只是 record 键集合。
> 锁内重读的新 journal 必须重新验证；同 targets 但 old/new 摘要、backup 或状态
> 变更不能继续使用旧恢复计划。
>
> 用 barrier/事件控制：预读 journal 后、获得完整锁集合后、实际写删前、
> 备份复制中/结束。…… 不要用 sleep 控制竞态，不只断言 rc 非零。

## 本文件的四个"事件点"

| 点 | 怎么卡住 | 期望 |
|---|---|---|
| ① 预读后、取锁前 | hook `acquire_target_locks`，进入时改写 journal | 锁内重读发现指纹变了 → 重试/拒绝 |
| ② 获得锁后、构造计划中 | hook `_recover_locked` 之后的 `_restore_locked` 入口 | 写删前最后一次核对发现变了 → 拒绝 |
| ③ 目标在预检后被改 | hook `_restore_locked`，执行前改目标文件 | 分类重验 → 保留现场 |
| ④ 指纹本身 | 直接改 records 字段 | 指纹必须变（已由 A05 文件覆盖） |

**这些测试不依赖 GPU / 用户素材**，全部在临时目录里跑真实小文件。
"""
import json
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import publisher as P  # noqa: E402
import publish_journal as J  # noqa: E402


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


def _sha(p):
    return J.file_digest(p)


def _unresolved(tmp_path, *, media=b"NEW", old=b"OLD"):
    """造一个**混合代**未决事务：媒体已是新代，报告仍是旧代。

    为什么要两个目标：只有一个目标时，磁盘要么全旧（补记回滚）、
    要么全新（补记提交）—— 两者都不会真正"回滚"。
    要测回滚路径，必须造 `mixed`。
    """
    out = str(tmp_path / "clip_restored.mkv")
    rep_p = out + ".verify.json"
    bk = str(tmp_path / "backup"
             / (os.path.basename(out) + ".pubbak-att1"))
    bk_rep = str(tmp_path / "backup"
                 / (os.path.basename(rep_p) + ".pubbak-att1"))
    _w(bk, old)
    _w(bk_rep, b"OLD-REPORT")
    _w(out, media)                 # 媒体：新代
    _w(rep_p, b"OLD-REPORT")       # 报告：旧代 → 整组 mixed

    jp = out + ".journal.json"
    doc = {
        "phase": "PREPARED",
        "attempt_id": "att1",
        "targets": [out, rep_p],
        "records": {
            out: {
                "artifact": "media",
                "staging": str(tmp_path / "staging" / "m.bin"),
                "existed": True,
                "old_digest": _sha(bk),
                "new_digest": _sha(out),
                "backup": bk,
                "backup_digest": _sha(bk),
            },
            rep_p: {
                "artifact": "verify",
                "staging": str(tmp_path / "staging" / "v.json"),
                "existed": True,
                "old_digest": _sha(bk_rep),
                "new_digest": "f" * 64,     # 新代从未落地
                "backup": bk_rep,
                "backup_digest": _sha(bk_rep),
            },
        },
    }
    assert J.write_journal(jp, doc)
    return {"out": out, "rep": rep_p, "jp": jp, "bk": bk, "doc": doc}


def _recover(ctx, logger=None):
    return P.recover(ctx["jp"], str(os.path.dirname(ctx["bk"])), "att2",
                     logger=logger or _Log(),
                     allowed_output_roots=[str(os.path.dirname(ctx["jp"]))])


# ------------------------------------------------ ① 预读后、取锁前

def test_journal_change_while_taking_the_lock_is_detected(tmp_path, monkeypatch):
    """事件点：`acquire_target_locks` 一进入就改写 journal。

    这是"预读 journal → 取锁"之间的窗口。锁内重读必须发现变化，
    不能拿着 A 的计划去操作被改成 B 的目标集合。
    """
    ctx = _unresolved(tmp_path)
    calls = {"n": 0}

    # 注意：publisher 是 **模块级** `from publish_lock import acquire_target_locks`，
    # 所以必须 patch `publisher` 上的那个名字；patch `publish_lock` 不生效。
    real = P.acquire_target_locks

    def _hook(targets, timeout=30.0, poll_interval=0.05, logger=None):
        # 第一次进锁前改一次（模拟另一个同协议进程先提交了下一代）
        calls["n"] += 1
        if calls["n"] <= 2:
            doc = J.read_journal(ctx["jp"])
            doc["attempt_id"] = f"other{calls['n']}"
            doc["phase"] = "COMMITTING"
            J.write_journal(ctx["jp"], doc)
        return real(targets, timeout=timeout, poll_interval=poll_interval,
                    logger=logger)

    monkeypatch.setattr(P, "acquire_target_locks", _hook)
    rep = _recover(ctx)
    # 有界重试：要么最终拒绝，要么**基于新 journal**完成 —— 但绝不能
    # 拿着旧计划（旧 targets/旧摘要）动手。
    assert rep is not None
    assert calls["n"] >= 1, "根本没取锁"
    # 两条正确出路：① 拒绝（结构/归属已经不成立）；
    # ② 重试后基于**新** journal 完成。唯一不可接受的是"按旧计划动手"。
    if rep.get("ok") is False:
        assert "changed" in str(rep.get("reason", "")) \
            or "ownership" in str(rep.get("reason", "")) \
            or "not COMMITTED" in str(rep.get("reason", "")), rep
    if rep.get("ok") is True:
        # 成功的话，用的必须是改写后的那份 journal
        final = J.read_journal(ctx["jp"])
        assert str(final.get("attempt_id", "")).startswith("other"), (
            f"重试后仍按旧 journal 操作：{final.get('attempt_id')!r}")


# ------------------------------------------------ ② 写删前最后一次核对

def test_journal_change_before_rollback_blocks_the_restore(tmp_path,
                                                           monkeypatch):
    """事件点：恢复计划已构造完成、即将真正写删。

    审核："同 targets 但 old/new 摘要、backup 或状态变更不能继续使用旧恢复计划。"
    """
    ctx = _unresolved(tmp_path)
    before = open(ctx["out"], "rb").read()

    # 在 `_recover_locked` **构造恢复计划期间**（分类那一步）把 journal 改掉。
    # 这是"检查 → 动手"之间的窗口：写删前那次核对必须发现它。
    real_classify = P.classify_generation

    def _hook(records):
        doc = J.read_journal(ctx["jp"])
        doc["records"][ctx["out"]]["new_digest"] = "0" * 64
        J.write_journal(ctx["jp"], doc)
        return real_classify(records)

    monkeypatch.setattr(P, "classify_generation", _hook)
    rep = _recover(ctx)

    assert rep.get("ok") is False, rep
    assert "changed before the rollback" in str(rep.get("reason")), rep
    # **一个字节都不能动**
    assert open(ctx["out"], "rb").read() == before


# ------------------------------------------------ ③ 目标在预检后被改

def test_target_changed_between_precheck_and_execution_is_kept(tmp_path,
                                                               monkeypatch):
    """事件点：`_restore_locked` 预检通过、执行前，目标被外部改动。

    期望：分类重验发现"现在这份不是我该动的"，保留现场。
    """
    ctx = _unresolved(tmp_path)
    real_restore = P._restore_locked

    def _hook(snapshot, attempt_id, logger=None):
        # 预检还没跑，先准备好"被外部改过"的目标：这里改的是**执行前**
        _w(ctx["out"], b"FOREIGN-BYTES")
        return real_restore(snapshot, attempt_id, logger)

    monkeypatch.setattr(P, "_restore_locked", _hook)
    rep = _recover(ctx)

    # journal 的 new_digest 是旧的，目标现在是外来内容 → 必须整代保留
    assert rep.get("ok") is False, rep
    assert open(ctx["out"], "rb").read() == b"FOREIGN-BYTES"


# ------------------------------------------------ ④ 正常路径仍然能恢复

def test_clean_unresolved_generation_still_rolls_back(tmp_path):
    """没有并发干扰时，未决事务必须能正常回滚 —— 收紧不能把功能弄坏。"""
    ctx = _unresolved(tmp_path)
    rep = _recover(ctx)
    assert rep.get("ok") is True, rep
    assert open(ctx["out"], "rb").read() == b"OLD", "媒体没回滚"
    assert open(ctx["rep"], "rb").read() == b"OLD-REPORT"
    assert J.read_journal(ctx["jp"]).get("phase") == "ROLLED_BACK"
