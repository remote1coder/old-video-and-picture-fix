# -*- coding: utf-8 -*-
"""R14-01：媒体与报告必须作为**一代**提交，失败可恢复。

**审核复现的缺陷**（本文件逐条验证）

正式 A 媒体 + A 报告；B 媒体替换成功，B 的 verify 报告替换抛
PermissionError → 结果是 **B 媒体 + A 报告**，`_generation_incomplete=True`。
返回非零纠正了"假成功"，却**没有保住上一完整代**；进程若在两次替换之间
消失，同样没有恢复依据。

修法：`shared_modules/publisher.py` —— 候选冻结 → 锁内备份 → PREPARED →
COMMITTING → 逐个 replace → COMMITTED；任一步失败按备份恢复**完整旧代**。

本文件用极小字节 fixture 直接调用生产 `publish()` / `recover()`，
不需要 GPU 或真实媒体。
"""
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import publisher as pub  # noqa: E402
from publish_journal import read_journal  # noqa: E402


def _write(p, data: bytes):
    os.makedirs(os.path.dirname(str(p)) or ".", exist_ok=True)
    with open(str(p), "wb") as f:
        f.write(data)
    return str(p)


def _read(p):
    with open(str(p), "rb") as f:
        return f.read()


class _Log:
    def __init__(self):
        self.events = []

    def _r(self, lv, ev):
        self.events.append((lv, ev))

    def info(self, ev, m="", **k):
        self._r("info", ev)

    def warning(self, ev, m="", **k):
        self._r("warn", ev)

    def error(self, ev, m="", **k):
        self._r("error", ev)

    def debug(self, *a, **k):
        pass

    def has(self, ev):
        return any(e == ev for _l, e in self.events)


def _gen_a(tmp_path):
    """旧完整代 A：媒体 + 两份报告。"""
    d = tmp_path / "out"
    media = _write(d / "clip_restored.mkv", b"OLD-MEDIA")
    verify = _write(d / "clip_restored.mkv.verify.json", b'{"gen":"A-verify"}')
    full = _write(d / "clip_restored.mkv.fullchain.json",
                  b'{"gen":"A-fullchain"}')
    return d, media, verify, full


def _stage_b(tmp_path):
    """新代 B 的候选件（attempt 私有）。"""
    s = tmp_path / "staging"
    m = _write(s / "b.mkv", b"NEW-MEDIA")
    v = _write(s / "b.verify.json", b'{"gen":"B-verify"}')
    f = _write(s / "b.fullchain.json", b'{"gen":"B-fullchain"}')
    return s, m, v, f


def _arts(media, verify, full, m, v, f):
    return [pub.Artifact("media", m, media),
            pub.Artifact("verify", v, verify),
            pub.Artifact("fullchain", f, full)]


def _journal(tmp_path):
    return str(tmp_path / "out" / "clip_restored.mkv.journal.json")


def _backup_dir(tmp_path):
    return str(tmp_path / "backup")


# ---------------------------------------------------------------- P01

def test_p01_report_failure_restores_the_whole_old_generation(tmp_path,
                                                             monkeypatch):
    """P01：媒体替换成功、verify 替换失败 → 必须恢复完整 A。

    绝不允许停在 **B 媒体 + A 报告**（那就是审核复现的状态）。
    """
    d, media, verify, full = _gen_a(tmp_path)
    _s, m, v, f = _stage_b(tmp_path)
    arts = _arts(media, verify, full, m, v, f)

    real_replace = os.replace

    def _boom(src, dst):
        if str(dst).endswith(".verify.json"):
            raise PermissionError("simulated: verify promote denied")
        return real_replace(src, dst)

    monkeypatch.setattr(pub.os, "replace", _boom)
    log = _Log()
    res = pub.publish(arts, _journal(tmp_path), _backup_dir(tmp_path),
                      "att-1", decision={"publish": True}, logger=log)

    assert res.ok is False
    assert res.failed_at == "replace:verify", res.failed_at
    assert res.restored is True, "失败后必须恢复完整旧代"

    # 三者都回到 A，不能是"B 媒体 + A 报告"
    assert _read(media) == b"OLD-MEDIA"
    assert _read(verify) == b'{"gen":"A-verify"}'
    assert _read(full) == b'{"gen":"A-fullchain"}'
    assert log.has("PUBLISH_ROLLED_BACK"), log.events


def test_p01_media_and_reports_never_mix_generations(tmp_path, monkeypatch):
    """P01 加强：失败后不允许出现任何跨代组合。"""
    d, media, verify, full = _gen_a(tmp_path)
    _s, m, v, f = _stage_b(tmp_path)

    real_replace = os.replace
    monkeypatch.setattr(
        pub.os, "replace",
        lambda s, t: (_ for _ in ()).throw(PermissionError("denied"))
        if str(t).endswith(".fullchain.json") else real_replace(s, t))

    res = pub.publish(_arts(media, verify, full, m, v, f), _journal(tmp_path),
                      _backup_dir(tmp_path), "att-2",
                      decision={"publish": True})
    assert res.ok is False and res.restored is True

    gen = {_read(media), _read(verify), _read(full)}
    assert gen == {b"OLD-MEDIA", b'{"gen":"A-verify"}',
                   b'{"gen":"A-fullchain"}'}, f"出现了跨代组合：{gen}"


# ---------------------------------------------------------------- P02

def test_p02_first_publish_all_or_nothing(tmp_path, monkeypatch):
    """P02：首次发布（无旧代）中途失败 → 撤下已出现的部分目标。"""
    d = tmp_path / "out"
    d.mkdir()
    media = str(d / "clip_restored.mkv")
    verify = str(d / "clip_restored.mkv.verify.json")
    full = str(d / "clip_restored.mkv.fullchain.json")
    _s, m, v, f = _stage_b(tmp_path)

    real_replace = os.replace
    monkeypatch.setattr(
        pub.os, "replace",
        lambda s, t: (_ for _ in ()).throw(PermissionError("denied"))
        if str(t).endswith(".fullchain.json") else real_replace(s, t))

    res = pub.publish(_arts(media, verify, full, m, v, f), _journal(tmp_path),
                      _backup_dir(tmp_path), "att-3",
                      decision={"publish": True})

    assert res.ok is False
    assert res.restored is True, "旧代本来不存在 → 撤下部分目标即恢复"
    assert not os.path.exists(media), "首次发布失败不能留下媒体"
    assert not os.path.exists(verify)
    assert not os.path.exists(full)


# ---------------------------------------------------------------- P03

def test_p03_crash_after_prepared_recovers_cleanly(tmp_path):
    """P03a：写完 PREPARED 就"消失"（什么都没替换）→ recover 不动正式文件。"""
    d, media, verify, full = _gen_a(tmp_path)
    _s, m, v, f = _stage_b(tmp_path)

    # 直接手工造一个 PREPARED 状态：媒体/报告都还是 A
    from publish_journal import write_journal
    jp = _journal(tmp_path)
    doc = {"phase": "PREPARED", "attempt_id": "att-crash",
           "targets": [media, verify, full], "records": {}}
    for fin in (media, verify, full):
        doc["records"][fin] = {"existed": True, "old_digest": None,
                               "backup": None, "backup_digest": None}
    assert write_journal(jp, doc)

    rep = pub.recover(jp, _backup_dir(tmp_path), "att-crash")
    assert rep is not None
    # 没有备份 → 预检失败，但**一个字节都不能动**
    assert rep.get("ok") is False
    assert rep.get("precheck_failed") is True
    assert _read(media) == b"OLD-MEDIA"
    assert _read(verify) == b'{"gen":"A-verify"}'


def test_p03_crash_mid_commit_rolls_back_on_recover(tmp_path):
    """P03b：媒体已换、报告未换就"消失" → recover 把媒体退回 A。"""
    d, media, verify, full = _gen_a(tmp_path)
    _s, m, v, f = _stage_b(tmp_path)

    # 造一个 COMMITTING 状态：媒体已是 B，且已有 A 的备份
    from publish_journal import file_digest, write_journal
    bdir = _backup_dir(tmp_path)
    os.makedirs(bdir, exist_ok=True)
    import shutil
    bk_media = os.path.join(bdir, "media.pubbak")
    shutil.copyfile(media, bk_media)
    bk_verify = os.path.join(bdir, "verify.pubbak")
    shutil.copyfile(verify, bk_verify)
    bk_full = os.path.join(bdir, "full.pubbak")
    shutil.copyfile(full, bk_full)

    old_m, old_v, old_f = _read(media), _read(verify), _read(full)
    _write(media, b"NEW-MEDIA")          # 只换了媒体

    jp = _journal(tmp_path)
    doc = {"phase": "COMMITTING", "attempt_id": "att-mid",
           "targets": [media, verify, full], "records": {
               media: {"existed": True, "old_digest": file_digest(bk_media),
                       "backup": bk_media,
                       "backup_digest": file_digest(bk_media)},
               verify: {"existed": True, "old_digest": file_digest(bk_verify),
                        "backup": bk_verify,
                        "backup_digest": file_digest(bk_verify)},
               full: {"existed": True, "old_digest": file_digest(bk_full),
                      "backup": bk_full,
                      "backup_digest": file_digest(bk_full)},
           }}
    assert write_journal(jp, doc)

    rep = pub.recover(jp, bdir, "att-mid")
    assert rep.get("ok") is True, rep
    assert rep.get("action") == "rolled_back"
    assert _read(media) == old_m, "媒体必须退回旧代"
    assert _read(verify) == old_v
    assert _read(full) == old_f


# ---------------------------------------------------------------- 决策/前置

def test_decision_denied_touches_nothing(tmp_path):
    """决策不允许 → 一个字节都不动，也不写 journal。"""
    d, media, verify, full = _gen_a(tmp_path)
    _s, m, v, f = _stage_b(tmp_path)
    jp = _journal(tmp_path)

    res = pub.publish(_arts(media, verify, full, m, v, f), jp,
                      _backup_dir(tmp_path), "att-deny",
                      decision={"publish": False, "blocked_by": ["sync"]})

    assert res.ok is False and res.failed_at == "decision"
    assert _read(media) == b"OLD-MEDIA"
    assert not os.path.exists(jp), "拒绝发布不应留下 journal"


def test_decision_error_is_not_a_pass(tmp_path):
    """汇总构造失败（含 error 字段）→ 同样拒绝。"""
    d, media, verify, full = _gen_a(tmp_path)
    _s, m, v, f = _stage_b(tmp_path)
    res = pub.publish(_arts(media, verify, full, m, v, f),
                      _journal(tmp_path), _backup_dir(tmp_path), "att-err",
                      decision={"error": "boom"})
    assert res.ok is False and res.failed_at == "decision"
    assert _read(media) == b"OLD-MEDIA"


def test_unreadable_staging_blocks_before_any_change(tmp_path):
    """候选件不可读 → 准备阶段就停，旧代不受影响。"""
    d, media, verify, full = _gen_a(tmp_path)
    _s, m, v, f = _stage_b(tmp_path)
    os.remove(v)                                  # 候选报告不见了

    res = pub.publish(_arts(media, verify, full, m, v, f), _journal(tmp_path),
                      _backup_dir(tmp_path), "att-miss",
                      decision={"publish": True})
    assert res.ok is False and res.failed_at == "prepare"
    assert _read(media) == b"OLD-MEDIA"
    assert _read(verify) == b'{"gen":"A-verify"}'


def test_existing_target_unreadable_is_refused(tmp_path):
    """已有目标存在但读不出 → 拒绝，不去碰它。"""
    d, media, verify, full = _gen_a(tmp_path)
    _s, m, v, f = _stage_b(tmp_path)
    # 把 verify 换成一个目录（存在但不可读/非普通文件）
    os.remove(verify)
    os.makedirs(verify)

    res = pub.publish(_arts(media, verify, full, m, v, f), _journal(tmp_path),
                      _backup_dir(tmp_path), "att-unread",
                      decision={"publish": True})
    assert res.ok is False and res.failed_at == "backup"
    assert _read(media) == b"OLD-MEDIA"
    assert os.path.isdir(verify), "不可读目标必须原样保留"


# ---------------------------------------------------------------- 成功路径

def test_success_publishes_all_and_cleans_backups(tmp_path):
    """正常提交：三者都换成 B，备份在成功后清掉，journal 记 COMMITTED。"""
    d, media, verify, full = _gen_a(tmp_path)
    _s, m, v, f = _stage_b(tmp_path)
    jp = _journal(tmp_path)

    res = pub.publish(_arts(media, verify, full, m, v, f), jp,
                      _backup_dir(tmp_path), "att-ok",
                      decision={"publish": True})

    assert res.ok is True, res.as_dict()
    assert res.phase == "COMMITTED"
    assert _read(media) == b"NEW-MEDIA"
    assert _read(verify) == b'{"gen":"B-verify"}'
    assert _read(full) == b'{"gen":"B-fullchain"}'
    assert res.cleanup_pending == []

    doc = read_journal(jp)
    assert doc["phase"] == "COMMITTED"
    # 备份已清
    leftovers = [n for n in os.listdir(_backup_dir(tmp_path))
                 if ".pubbak" in n]
    assert leftovers == [], leftovers


def test_success_with_keep_backups(tmp_path):
    """显式要求保留备份时不清。"""
    d, media, verify, full = _gen_a(tmp_path)
    _s, m, v, f = _stage_b(tmp_path)
    res = pub.publish(_arts(media, verify, full, m, v, f), _journal(tmp_path),
                      _backup_dir(tmp_path), "att-keep",
                      decision={"publish": True}, keep_backups=True)
    assert res.ok is True
    assert any(".pubbak" in n for n in os.listdir(_backup_dir(tmp_path)))


# ---------------------------------------------------------------- WSL 回拷

def test_wsl_copyback_commits_media_and_reports_together(tmp_path):
    """WSL 最终路径也走同一事务：媒体 + 报告一起，不能只拷媒体。

    原来 `wsl_copy_back()` 只拷媒体、用共享 `<original>.tmp`；报告留在
    重定向区。现在 stage7 先在**目标侧 attempt 私有暂存**复制，再调用
    同一个 `publish()` —— 本条验证该协议本身。
    """
    redirect_dir = tmp_path / "redirected"
    wsl_dir = tmp_path / "wsl_original"
    wsl_dir.mkdir()

    # 重定向区已发布的成果
    media = _write(redirect_dir / "clip_restored.mkv", b"NEW-MEDIA")
    verify = _write(redirect_dir / "clip_restored.mkv.verify.json",
                    b'{"gen":"NEW-verify"}')
    full = _write(redirect_dir / "clip_restored.mkv.fullchain.json",
                  b'{"gen":"NEW-fullchain"}')

    # 目标侧私有暂存（模拟 stage7 的 copyfile）
    import shutil
    stage_dir = wsl_dir
    m_s = str(stage_dir / "clip_restored.mkv.wsl-att9.mkv")
    v_s = str(stage_dir / "clip_restored.mkv.verify.json.wsl-att9")
    f_s = str(stage_dir / "clip_restored.mkv.fullchain.json.wsl-att9")
    shutil.copyfile(media, m_s)
    shutil.copyfile(verify, v_s)
    shutil.copyfile(full, f_s)

    tgt_m = str(wsl_dir / "clip_restored.mkv")
    tgt_v = str(wsl_dir / "clip_restored.mkv.verify.json")
    tgt_f = str(wsl_dir / "clip_restored.mkv.fullchain.json")

    res = pub.publish(
        [pub.Artifact("media", m_s, tgt_m),
         pub.Artifact("verify", v_s, tgt_v),
         pub.Artifact("fullchain", f_s, tgt_f)],
        journal_path=str(wsl_dir / "clip_restored.mkv.journal.json"),
        backup_dir=str(tmp_path / "wsl_backup"),
        attempt_id="att9", decision={"publish": True})

    assert res.ok is True, res.as_dict()
    assert _read(tgt_m) == b"NEW-MEDIA"
    assert _read(tgt_v) == b'{"gen":"NEW-verify"}', "报告必须一起过去"
    assert _read(tgt_f) == b'{"gen":"NEW-fullchain"}'


def test_wsl_staging_is_attempt_private_not_shared_tmp(tmp_path):
    """暂存名必须带 attempt，不能是共享的 `<original>.tmp`。"""
    import inspect
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage7_encode.py"),
               encoding="utf-8", errors="replace").read()
    assert ".wsl-{_attempt_id}" in src, "WSL 暂存必须带 attempt 标识"
    # 只在**非注释行**里查调用（注释里提到旧函数名是正常的）
    code = [ln for ln in src.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    assert not any("wsl_copy_back(" in ln for ln in code), \
        "不得再调用只拷媒体、使用共享 .tmp 的旧实现"
    assert "import" not in "".join(
        ln for ln in code if "wsl_copy_back" in ln), \
        "也不得再导入它"
