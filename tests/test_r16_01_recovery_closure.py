# -*- coding: utf-8 -*-
"""R16-01：失败 → 完整回滚 → **下一次仍能正常发布**的闭环。

审核复现的两条生产控制流：

1. 已有完整 A，发布 B 时媒体替换成功、报告替换失败。代码能恢复完整 A，
   但随后把 journal 保留为 ``RECOVERY_REQUIRED`` 并**删除备份**。
   下次 recover 不先判断正式文件是否已完整等于 A，而是再次要求备份存在
   → 拒绝恢复 → 下一次 publish 也无法继续。
2. 写 ``COMMITTED`` 失败。代码仍返回 ``ok=True``，只记
   ``cleanup_pending=['journal_committed']``，随后清理备份。磁盘 journal
   留在 ``COMMITTING``，重启后又因备份缺失无法恢复。

本文件用**生产 publisher**（真实文件系统，只 monkeypatch 外部失败点）
覆盖 B01 / B03 / B04 / B07，以及新增的 ``ROLLED_BACK`` 终态语义。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import publisher as pub  # noqa: E402
from publish_journal import (  # noqa: E402
    PHASE_COMMITTED, PHASE_COMMITTING, PHASE_RECOVERY_REQUIRED,
    PHASE_ROLLED_BACK, read_journal)
from test_r14_01_publisher import (  # noqa: E402
    _arts, _backup_dir, _gen_a, _journal, _Log, _read, _stage_b,
    _write)


def _digest(p):
    return pub.file_digest(p)


# ---------------------------------------------------------------- B01

def test_b01_rollback_then_next_publish_succeeds(tmp_path, monkeypatch):
    """B01：B 报告晋升失败 → 完整回到 A；**不手删 journal**；C 正常成功。

    这是审核复现的第一条：原来回滚成功后 journal 停在
    RECOVERY_REQUIRED 且备份已删，下一次 publish 被自己挡住。
    """
    d, media, verify, full = _gen_a(tmp_path)
    a_media, a_verify = _read(media), _read(verify)

    # 发 B：让 verify 的 replace 失败
    _s, m, v, f = _stage_b(tmp_path)
    arts_b = _arts(media, verify, full, m, v, f)
    real_replace = os.replace

    def _boom(src, dst):
        if str(dst).endswith(".verify.json"):
            raise PermissionError("simulated: verify promote denied")
        return real_replace(src, dst)

    monkeypatch.setattr(pub.os, "replace", _boom)
    jp = _journal(tmp_path)
    log = _Log()
    res_b = pub.publish(arts_b, jp, _backup_dir(tmp_path), "att-b",
                        decision={"publish": True}, logger=log)
    monkeypatch.undo()

    # 当次：失败 + 恢复完整 A
    assert res_b.ok is False
    assert res_b.restored is True
    assert _read(media) == a_media
    assert _read(verify) == a_verify

    # 关键：终态必须是 ROLLED_BACK，而不是 RECOVERY_REQUIRED
    doc = read_journal(jp)
    assert doc["phase"] == PHASE_ROLLED_BACK, \
        f"回滚成功后 journal 应落成 ROLLED_BACK，实际 {doc['phase']}"

    # 不手删 journal 的前提下，下一次 recover 应能确认已回滚
    rep = pub.recover(jp, _backup_dir(tmp_path), "att-c")
    assert rep is not None and rep.get("ok") is True, rep
    assert rep.get("action") in ("already_rolled_back", "confirmed_rolled_back"), rep

    # 然后 C 可以正常发布成功
    _s2, m2, v2, f2 = _stage_b(tmp_path)
    arts_c = _arts(media, verify, full, m2, v2, f2)
    res_c = pub.publish(arts_c, jp, _backup_dir(tmp_path), "att-c",
                        decision={"publish": True}, logger=_Log())
    assert res_c.ok is True, res_c.as_dict()
    assert res_c.phase == PHASE_COMMITTED
    assert _read(media) != a_media, "C 的内容应当真的上去"
    assert _read(verify) != a_verify


# ---------------------------------------------------------------- B03

def test_b03_committed_write_failure_keeps_recovery_material(
        tmp_path, monkeypatch):
    """B03：仅 COMMITTED 持久化失败 → 状态诚实，**保留备份**，不被永久挡住。

    原来：返回 ok=True、清备份、journal 停在 COMMITTING，下次无法恢复。
    """
    d, media, verify, full = _gen_a(tmp_path)
    _s, m, v, f = _stage_b(tmp_path)
    arts = _arts(media, verify, full, m, v, f)

    real_write = pub.write_journal
    calls = {"n": 0}

    def _fail_on_committed(path, doc, **kw):
        if doc.get("phase") == PHASE_COMMITTED:
            calls["n"] += 1
            return False
        return real_write(path, doc, **kw)

    monkeypatch.setattr(pub, "write_journal", _fail_on_committed)
    jp = _journal(tmp_path)
    bdir = _backup_dir(tmp_path)
    res = pub.publish(arts, jp, bdir, "att-1", decision={"publish": True},
                      logger=_Log())
    monkeypatch.undo()

    assert calls["n"] == 1, "应该真的尝试过写 COMMITTED"
    # 新代已生效（不能谎报失败）
    assert res.ok is True
    assert res.phase == PHASE_COMMITTED
    # 但**状态必须诚实**：终态没持久化要有标记
    assert res.terminal_pending is True, res.as_dict()
    assert any("journal_committed" in c for c in res.cleanup_pending), \
        res.cleanup_pending

    # 关键：备份必须还在（它是下一次恢复/补记的依据）
    leftovers = [n for n in os.listdir(bdir) if ".pubbak" in n]
    assert leftovers, "终态未持久化时不能清备份"

    # 磁盘事实 = 全新 → 下一次 recover 应能按协议补记 COMMITTED
    doc = read_journal(jp)
    assert doc["phase"] != PHASE_COMMITTED
    rep = pub.recover(jp, bdir, "att-2")
    assert rep is not None and rep.get("ok") is True, rep
    assert rep.get("action") == "recorded_committed", rep
    final = read_journal(jp)
    if final is None:
        raw = ""
        try:
            with open(jp, encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as exc:
            raw = f"<unreadable: {exc}>"
        pytest.fail(f"recover 之后 journal 不可读。原始内容:\n{raw[:600]}")
    assert final["phase"] == PHASE_COMMITTED, final


# ---------------------------------------------------------------- B04

def test_b04_prepared_write_failure_leaves_old_generation_intact(
        tmp_path, monkeypatch):
    """B04：PREPARED 阶段写失败 → 旧代不变，下一次可继续。"""
    d, media, verify, full = _gen_a(tmp_path)
    a_media, a_verify = _read(media), _read(verify)
    _s, m, v, f = _stage_b(tmp_path)
    arts = _arts(media, verify, full, m, v, f)

    real_write = pub.write_journal

    def _fail_on_prepared(path, doc, **kw):
        if doc.get("phase") == pub.PHASE_PREPARED:
            return False
        return real_write(path, doc, **kw)

    monkeypatch.setattr(pub, "write_journal", _fail_on_prepared)
    jp = _journal(tmp_path)
    res = pub.publish(arts, jp, _backup_dir(tmp_path), "att-1",
                      decision={"publish": True}, logger=_Log())
    monkeypatch.undo()

    assert res.ok is False
    assert res.failed_at == "journal_prepared", res.failed_at
    # 一个正式文件都没动
    assert _read(media) == a_media
    assert _read(verify) == a_verify

    # 下一次能继续（没有陈旧事务挡住）
    _s2, m2, v2, f2 = _stage_b(tmp_path)
    res2 = pub.publish(_arts(media, verify, full, m2, v2, f2), jp,
                       _backup_dir(tmp_path), "att-2",
                       decision={"publish": True}, logger=_Log())
    assert res2.ok is True, res2.as_dict()


# ---------------------------------------------------------------- B07

def test_b07_committed_but_group_broken_is_not_accepted(tmp_path):
    """B07：journal 说 COMMITTED，但一份正式报告不符 → **不认可完整**。

    也不能销毁未解决记录。
    """
    d, media, verify, full = _gen_a(tmp_path)
    _s, m, v, f = _stage_b(tmp_path)
    jp = _journal(tmp_path)
    res = pub.publish(_arts(media, verify, full, m, v, f), jp,
                      _backup_dir(tmp_path), "att-1",
                      decision={"publish": True}, logger=_Log())
    assert res.ok is True
    final = read_journal(jp)
    if final is None:
        raw = ""
        try:
            with open(jp, encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as exc:
            raw = f"<unreadable: {exc}>"
        pytest.fail(f"recover 之后 journal 不可读。原始内容:\n{raw[:600]}")
    assert final["phase"] == PHASE_COMMITTED, final

    # 人为把一份报告改坏（模拟"成果坏"）
    _write(verify, b"CORRUPTED")

    doc = read_journal(jp)
    assert doc["phase"] == PHASE_COMMITTED   # 字段仍这么说
    rep = pub.recover(jp, _backup_dir(tmp_path), "att-2")
    assert rep is not None
    assert rep.get("ok") is False, \
        "仅凭 phase 字段就认可完整 = 审核 B07 的问题依然存在"
    assert rep.get("generation") in ("mixed", "unknown"), rep


# ------------------------------------------------- classify_generation

def test_classify_generation_four_states(tmp_path):
    """``classify_generation`` 必须能分出全旧/全新/混合/未知。"""
    media = str(tmp_path / "m.mkv")
    verify = str(tmp_path / "m.mkv.verify.json")
    _write(media, b"OLD-MEDIA")
    _write(verify, b"OLD-VERIFY")
    old = {"m": pub.file_digest(media), "v": pub.file_digest(verify)}

    _write(media, b"NEW-MEDIA")
    _write(verify, b"NEW-VERIFY")
    new = {"m": pub.file_digest(media), "v": pub.file_digest(verify)}

    recs = {
        media: {"existed": True, "old_digest": old["m"], "new_digest": new["m"]},
        verify: {"existed": True, "old_digest": old["v"],
                 "new_digest": new["v"]},
    }

    # 全新
    assert pub.classify_generation(recs) == pub.GENERATION_ALL_NEW

    # 全旧
    _write(media, b"OLD-MEDIA")
    _write(verify, b"OLD-VERIFY")
    assert pub.classify_generation(recs) == pub.GENERATION_ALL_OLD

    # 混合
    _write(media, b"OLD-MEDIA")
    _write(verify, b"NEW-VERIFY")
    assert pub.classify_generation(recs) == pub.GENERATION_MIXED

    # 未知：既不是旧代也不是新代
    _write(media, b"SOMETHING-ELSE")
    _write(verify, b"NEW-VERIFY")
    assert pub.classify_generation(recs) == pub.GENERATION_UNKNOWN

    # 未知：目标是目录（不能等同于不存在）
    os.remove(media)
    os.makedirs(media, exist_ok=True)
    assert pub.classify_generation(recs) == pub.GENERATION_UNKNOWN


def test_classify_generation_absent_targets(tmp_path):
    """目标不存在：只有"旧代本来就不存在"才算符合旧代。"""
    p = str(tmp_path / "gone.mkv")
    # 旧代本来就不存在 → absent 符合旧代
    recs = {p: {"existed": False, "old_digest": None, "new_digest": "x"}}
    assert pub.classify_generation(recs) == pub.GENERATION_ALL_OLD

    # 旧代本来存在却没了 → 不是"全旧"，应判 unknown
    recs2 = {p: {"existed": True, "old_digest": "old", "new_digest": "new"}}
    assert pub.classify_generation(recs2) == pub.GENERATION_UNKNOWN


# ------------------------------------------------- write_journal digest

def test_journal_update_is_readable_after_write(tmp_path):
    """读出来 → 改 phase → 写回 → **必须还能读回来**。

    这是本轮发现的一个**既有**严重缺陷：`write_journal` 计算 digest 时把
    payload 里带来的**旧 digest 卷进了哈希**，于是任何"读→改→写"的更新都会
    产出一个 `read_journal` 校验失败（返回 None）的 journal。
    后果不只是"读不出来"——下一步的恢复/补记全都无从下手。
    """
    from publish_journal import write_journal

    jp = str(tmp_path / "j.json")
    doc = {"attempt_id": "att-1", "phase": "PREPARED", "records": {}}
    assert write_journal(jp, doc) is True

    first = read_journal(jp)
    assert first is not None and first["phase"] == "PREPARED"

    # 关键：把**读回来的对象**（里面带着 digest）改一版再写回
    updated = dict(first, phase="COMMITTING")
    assert write_journal(jp, updated) is True

    second = read_journal(jp)
    assert second is not None, "更新后的 journal 读不回来 —— digest 被旧值污染"
    assert second["phase"] == "COMMITTING"

    # 再改一次：必须仍然可读（不是只修好了第一次）
    third_in = dict(second, phase=PHASE_COMMITTED)
    assert write_journal(jp, third_in) is True
    third = read_journal(jp)
    assert third is not None and third["phase"] == PHASE_COMMITTED


# ------------------------------------------------- R16-02 必需成果集合

def test_stage7_does_not_swallow_fullchain_write_failure():
    """R16-02：fullchain 候选写失败**不许**被 `except: pass` 吞掉。

    审核确认：原来写报告失败会静默，`_fc_pending` 留空，publisher 只提交
    "媒体 + verify"，根本不知道本代缺 fullchain；若目标已有 A 的 fullchain，
    就留下 **B 视频 + A 报告**。
    """
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage7_encode.py"),
               encoding="utf-8", errors="replace").read()

    # 记录位 + 报错事件必须在
    assert "_fc_write_error" in src, "写失败必须有记录位"
    assert "STAGE7_FULLCHAIN_WRITE_FAIL" in src

    # 必需清单 + 发布前硬失败必须在
    assert 'STAGE7_REQUIRED_ARTIFACT_MISSING' in src, \
        "缺少必需报告时必须在发布前失败"
    assert '_required = ["media", "verify"]' in src

    # 禁用时必须有明确的"不适用"报告，避免旧报告被当成新证据
    assert '"status": "NOT_APPLICABLE"' in src, \
        "禁用 fullchain 时必须写一份 NOT_APPLICABLE 替换旧报告"

    # 具体反例：fullchain 候选写入的那段 try 不能再是 `except: pass`
    import re
    m = re.search(r"_fc_pending = \(_fc_tmp, _fc_final\)\s*\n\s*(except[^\n]*)\n(\s*)pass\b",
                  src)
    assert m is None, "fullchain 候选写入仍是 `except: pass`（静默吞掉）"


# ---------------------------------------------------------------- B02

def test_b02_rollback_terminal_write_failure_keeps_material(
        tmp_path, monkeypatch):
    """B02：回滚成功但**终态日志写失败** → 保留备份，重启仍能确认。

    审核要求："终态日志写入失败时，下一次仍有明确、可验证的恢复办法。"
    """
    d, media, verify, full = _gen_a(tmp_path)
    a_media, a_verify = _read(media), _read(verify)
    _s, m, v, f = _stage_b(tmp_path)
    arts = _arts(media, verify, full, m, v, f)

    real_replace = os.replace
    real_write = pub.write_journal

    def _boom_replace(src, dst):
        if str(dst).endswith(".verify.json"):
            raise PermissionError("simulated: verify promote denied")
        return real_replace(src, dst)

    def _fail_rolled_back(path, doc, **kw):
        if doc.get("phase") == PHASE_ROLLED_BACK:
            return False
        return real_write(path, doc, **kw)

    monkeypatch.setattr(pub.os, "replace", _boom_replace)
    monkeypatch.setattr(pub, "write_journal", _fail_rolled_back)
    jp = _journal(tmp_path)
    bdir = _backup_dir(tmp_path)
    res = pub.publish(arts, jp, bdir, "att-1", decision={"publish": True},
                      logger=_Log())
    monkeypatch.undo()

    # 当次：恢复完整 A，但终态没写成
    assert res.restored is True
    assert res.terminal_pending is True, res.as_dict()
    assert any("journal_rolled_back" in c for c in res.cleanup_pending), \
        res.cleanup_pending
    assert _read(media) == a_media and _read(verify) == a_verify

    # **备份必须还在**（下一次恢复的唯一依据）
    assert [n for n in os.listdir(bdir) if ".pubbak" in n], \
        "终态未持久化时不能删备份"

    # 重启后：按磁盘事实应能确认"已是完整旧代"，不需要人工
    rep = pub.recover(jp, bdir, "att-2")
    assert rep is not None and rep.get("ok") is True, rep
    assert rep.get("action") == "confirmed_rolled_back", rep
    assert read_journal(jp)["phase"] == PHASE_ROLLED_BACK


# ---------------------------------------------------------------- B06

def test_b06_journal_without_records_changes_nothing(tmp_path):
    """B06：journal 结构不全（没有 records）→ 一个正式文件都不改。"""
    d, media, verify, full = _gen_a(tmp_path)
    a_media, a_verify = _read(media), _read(verify)
    jp = _journal(tmp_path)
    # 手写一个"有 schema 有 targets 但没 records"的 journal
    from publish_journal import write_journal
    assert write_journal(jp, {"attempt_id": "att-x", "phase": "PREPARING",
                              "targets": [media, verify]})

    rep = pub.recover(jp, _backup_dir(tmp_path), "att-1")
    assert rep is not None and rep.get("ok") is False, rep
    assert _read(media) == a_media and _read(verify) == a_verify


def test_b06_directory_journal_is_not_treated_as_absent(tmp_path):
    """B06：journal 是**目录**时不能等同于"没有事务"。"""
    d, media, verify, full = _gen_a(tmp_path)
    jp = _journal(tmp_path)
    os.makedirs(jp, exist_ok=True)      # 同名目录
    rep = pub.recover(jp, _backup_dir(tmp_path), "att-1")
    # 目录 → read_journal 返回 None → recover 返回 ok=False（不能当"没事"）
    assert rep is None or rep.get("ok") is False, rep


def test_b06_recovery_never_trusts_records_over_targets(tmp_path):
    """B06：records 里的目标若不在 targets 里，恢复不得为之加锁后操作。

    审核："独立 recover 根据 doc.targets 取锁，却按 records 操作"。
    """
    d, media, verify, full = _gen_a(tmp_path)
    a_media = _read(media)
    outsider = str(tmp_path / "outsider.bin")
    _write(outsider, b"OUTSIDER")

    from publish_journal import write_journal
    jp = _journal(tmp_path)
    assert write_journal(jp, {
        "attempt_id": "att-x", "phase": "COMMITTING",
        "targets": [media],                     # 只锁 media
        "records": {                            # 却想动 outsider
            outsider: {"existed": True, "old_digest": "deadbeef",
                       "backup": None, "backup_digest": None},
        },
    })
    rep = pub.recover(jp, _backup_dir(tmp_path), "att-1")
    # 不管结果是失败还是拒绝，**都不许**在没锁 outsider 的情况下改它
    assert _read(outsider) == b"OUTSIDER", "越界目标被改了"
    assert _read(media) == a_media
    assert rep is None or rep.get("ok") is False, rep
