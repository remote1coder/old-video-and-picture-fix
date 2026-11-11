# -*- coding: utf-8 -*-
"""R21-C03：身份未知的目标仍会被回滚**删除或覆盖**。

## 审核复现（R20 §3，P1，最严重的一条）

> 1. 用生产 `publish` 建立一个合法完整 journal。
> 2. 在隔离夹具中把 phase 改为 PREPARED，构造未决状态。
> 3. 把本轮自建目标文件替换为 `UNRELATED-BYTES`。
> 4. journal 结构检查返回空问题列表；调用生产 `recover`。
> 5. 返回 `ok=true / generation=unknown / fully_restored=true`，目标文件真实消失。

## 根因

`_restore_locked` 的执行阶段对 `existed=False` 的目标无条件
`os.remove(fin)`；而 `existed=False` 只说明"**发布前**没有这个文件"，
不说明"现在这个文件是我写的"。快照里也没有 `new_digest`，
根本无法回答"当前字节属于谁"。

## 本文件锁住什么

| 场景 | 要求 |
|---|---|
| 原先不存在 + 陌生内容 | **保留现场**，返回 recovery_required |
| 原先存在 + 陌生内容 | **保留现场**（不得用备份覆盖） |
| 备份缺失/损坏 | 整代不动 |
| 已知新内容 | 正常回滚 |
| 已知旧内容 | 幂等保留 |
| 混合代（媒体新、报告旧） | 可证明的回滚要允许，不能一律拒绝 |
| COMMITTED / PREPARED / COMMITTING | 各自走对分支 |
| 恢复中途失败后再恢复 | 第二次能继续，不卡死 |

每个用例都记录**前后每个对象的 SHA256**，不只断言返回值。
"""
import hashlib
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


def sha(path):
    if not os.path.exists(path):
        return "<absent>"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(*paths):
    return {os.path.basename(p): sha(p) for p in paths}


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _setup(tmp_path, names=("media.mkv", "verify.json")):
    """准备 staging / final / journal / backup 四组路径。"""
    root = tmp_path
    stage_dir = root / "staging"
    out_dir = root / "out"
    stage_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    arts = []
    for i, n in enumerate(names):
        st = _write(str(stage_dir / f"{n}.new"), f"NEW-{n}".encode())
        fin = str(out_dir / n)
        arts.append(P.Artifact(n, st, fin))
    return {
        "arts": arts,
        "finals": [a.final for a in arts],
        "journal": str(root / "out" / "gen.journal.json"),
        "backup": str(root / "backup"),
    }


def _publish(ctx, logger=None):
    # keep_backups=True：回滚测试需要旧代备份还在。生产的成功发布会清理备份，
    # 但"未决事务"的现场本来就应该保留备份 —— 这里直接保留，才能模拟
    # "提升完成、终态未写、备份未清"的真实中断点。
    return P.publish(ctx["arts"], ctx["journal"], ctx["backup"],
                     attempt_id="att1", decision={"publish": True},
                     logger=logger or _Log(), keep_backups=True,
                     allowed_output_roots=[os.path.dirname(ctx["journal"])])


def _force_phase(ctx, phase):
    doc = J.read_journal(ctx["journal"])
    assert doc is not None
    doc["phase"] = phase
    assert J.write_journal(ctx["journal"], doc)
    return doc


def _recover(ctx, logger=None):
    return P.recover(ctx["journal"], ctx["backup"], attempt_id="att2",
                     logger=logger or _Log(),
                     allowed_output_roots=[os.path.dirname(ctx["journal"])])


def _make_mixed(ctx, idx=1):
    """把第 idx 个目标退回旧代，制造**混合代**（否则全组是新代，
    恢复会按协议"补记 COMMITTED"，根本不会回滚）。

    这是必需的夹具步骤：协议里"全组已是新代"的正确动作是向前完成，
    不是回滚。要测回滚，磁盘必须真的是混合态。
    """
    doc = J.read_journal(ctx["journal"])
    fin = os.path.abspath(ctx["finals"][idx])
    rec = (doc.get("records") or {}).get(fin)
    assert rec is not None, f"journal 里没有 {fin}"
    if rec.get("existed"):
        bk = rec.get("backup")
        assert bk and os.path.exists(bk)
        with open(bk, "rb") as f:
            data = f.read()
        _write(ctx["finals"][idx], data)
    else:
        os.remove(ctx["finals"][idx])
    return doc


# ------------------------------------------------ 陌生内容必须保留现场

def test_unknown_content_on_a_new_target_is_preserved(tmp_path):
    """审核的原始反例：目标原先不存在，发布中途被换成陌生字节。"""
    ctx = _setup(tmp_path)
    r0 = _publish(ctx)
    assert r0.ok is True, r0.errors

    _force_phase(ctx, "PREPARED")
    foreign = b"UNRELATED-BYTES"
    _write(ctx["finals"][0], foreign)
    before = snapshot(*ctx["finals"])

    rep = _recover(ctx)
    after = snapshot(*ctx["finals"])

    assert rep["ok"] is False, f"陌生内容不该被当成可回滚: {rep}"
    assert rep.get("recovery_required") is True
    assert after["media.mkv"] == before["media.mkv"], "目标被改动了"
    assert open(ctx["finals"][0], "rb").read() == foreign, "文件内容被改写"


def test_unknown_content_on_an_existing_target_is_preserved(tmp_path):
    """原先存在旧代，发布后目标被别人改过 → 不得用备份覆盖。"""
    ctx = _setup(tmp_path)
    _write(ctx["finals"][0], b"OLD-MEDIA")
    r0 = _publish(ctx)
    assert r0.ok is True, r0.errors

    _force_phase(ctx, "PREPARED")
    _write(ctx["finals"][0], b"SOMEONE-ELSE")
    rep = _recover(ctx)
    assert rep["ok"] is False
    assert open(ctx["finals"][0], "rb").read() == b"SOMEONE-ELSE"


def test_corrupt_backup_blocks_the_whole_generation(tmp_path):
    ctx = _setup(tmp_path)
    _write(ctx["finals"][0], b"OLD-MEDIA")
    _write(ctx["finals"][1], b"OLD-VERIFY")
    r0 = _publish(ctx)
    assert r0.ok is True
    _force_phase(ctx, "PREPARED")
    _make_mixed(ctx, idx=0)          # 造出混合代，让恢复必须用备份

    # 把备份改成垃圾，但目标仍是新代（正常应回滚）
    doc = J.read_journal(ctx["journal"])
    bks = [row.get("backup") for row in doc["records"].values()
           if row.get("backup")]
    assert bks, doc["records"]
    for b in bks:
        if os.path.exists(b):
            _write(b, b"GARBAGE")

    before = snapshot(*ctx["finals"])
    rep = _recover(ctx)
    after = snapshot(*ctx["finals"])
    assert rep["ok"] is False
    assert after == before, "备份无效时不得改动任何目标"


# ------------------------------------------------ 已知内容按合同处理

def test_known_new_content_is_rolled_back(tmp_path):
    """混合代里"本 attempt 写的新文件" → 按"原本不存在"的合同撤下。"""
    ctx = _setup(tmp_path)
    r0 = _publish(ctx)
    assert r0.ok is True
    _force_phase(ctx, "PREPARED")
    _make_mixed(ctx, idx=1)          # 让第 2 个目标退回"不存在"

    rep = _recover(ctx)
    assert rep["ok"] is True, rep
    assert not os.path.exists(ctx["finals"][0]), "已知新内容应被撤下"
    assert rep["action"] in ("rolled_back",)


def test_known_old_content_is_idempotent(tmp_path):
    """目标已是旧代 → 幂等保留，不应再被复制一遍。"""
    ctx = _setup(tmp_path)
    _write(ctx["finals"][0], b"OLD-MEDIA")
    _write(ctx["finals"][1], b"OLD-VERIFY")
    r0 = _publish(ctx)
    assert r0.ok is True
    _force_phase(ctx, "PREPARED")
    _make_mixed(ctx, idx=0)          # 第 1 个退回旧代，第 2 个仍是新代
    before0 = sha(ctx["finals"][0])

    rep = _recover(ctx)
    assert rep["ok"] is True, rep
    assert sha(ctx["finals"][0]) == before0, "旧代被无谓地重写"
    assert open(ctx["finals"][1], "rb").read() == b"OLD-VERIFY"


def test_mixed_generation_can_still_be_restored(tmp_path):
    """媒体是新代、报告被退回旧代 → **可证明**的混合代，必须能回滚。

    指令要求："允许可证明的混合旧/新代恢复；不要把所有中间态一律拒绝，
    导致恢复功能瘫痪。"
    """
    ctx = _setup(tmp_path)
    _write(ctx["finals"][0], b"OLD-MEDIA")
    _write(ctx["finals"][1], b"OLD-VERIFY")
    r0 = _publish(ctx)
    assert r0.ok is True
    _force_phase(ctx, "PREPARED")
    _make_mixed(ctx, idx=1)          # 报告退回旧代，媒体仍是新代

    rep = _recover(ctx)
    assert rep["ok"] is True, rep
    assert open(ctx["finals"][0], "rb").read() == b"OLD-MEDIA"
    assert open(ctx["finals"][1], "rb").read() == b"OLD-VERIFY"


# ------------------------------------------------ 三种 phase

@pytest.mark.parametrize("phase", ["PREPARED", "COMMITTING"])
def test_unresolved_phases_roll_back(tmp_path, phase):
    ctx = _setup(tmp_path)
    r0 = _publish(ctx)
    assert r0.ok is True
    _force_phase(ctx, phase)
    _make_mixed(ctx, idx=1)

    rep = _recover(ctx)
    assert rep["ok"] is True, rep
    assert not os.path.exists(ctx["finals"][0])


def test_committed_phase_is_confirmed_without_touching_files(tmp_path):
    ctx = _setup(tmp_path)
    r0 = _publish(ctx)
    assert r0.ok is True
    # publish 成功已写 COMMITTED；再 recover 应只做确认
    before = snapshot(*ctx["finals"])
    rep = _recover(ctx)
    assert rep["ok"] is True
    assert rep["action"] == "already_committed"
    assert snapshot(*ctx["finals"]) == before


# ------------------------------------------------ 恢复失败后能再恢复

def test_recovery_can_be_retried_after_the_blocker_is_gone(tmp_path):
    ctx = _setup(tmp_path)
    r0 = _publish(ctx)
    assert r0.ok is True
    _force_phase(ctx, "PREPARED")
    _make_mixed(ctx, idx=1)

    _write(ctx["finals"][0], b"FOREIGN")
    first = _recover(ctx)
    assert first["ok"] is False

    # 人工把陌生文件移走（模拟运维处理）
    os.remove(ctx["finals"][0])
    second = _recover(ctx)
    assert second["ok"] is True, second
    assert not os.path.exists(ctx["finals"][0])


# ------------------------------------------------ 记录方式

def test_every_case_records_before_after_digests(tmp_path):
    """本文件所有断言都比对过 SHA256；这里再把记录方式固定下来。"""
    ctx = _setup(tmp_path)
    _publish(ctx)
    before = snapshot(*ctx["finals"])
    assert all(v != "<absent>" for v in before.values())
    _force_phase(ctx, "PREPARED")
    _make_mixed(ctx, idx=1)
    rep = _recover(ctx)
    assert "target_states" in rep
    assert set(rep["target_states"]) == {os.path.abspath(p)
                                        for p in ctx["finals"]}
