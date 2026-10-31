# -*- coding: utf-8 -*-
"""B15：提交过程中**真实终止子进程**，新进程能否安全恢复。

审核明确要求：
  "B15 必须是测试控制器终止它自己启动的子进程，只能作用于本轮隔离测试，
   不杀用户程序。不能用手写 journal 后调用 recover 冒充硬退出。"

做法：子进程跑**真实 `publisher.publish`**，但在第 N 次 `os.replace` 时
`os.kill(os.getpid(), SIGKILL)` —— 进程瞬间消失，没有任何 finally 能跑，
等价于断电/强杀。父进程随后调 `recover()`，断言：
  ① 要么完整旧代、要么完整新代，**绝不混代**；
  ② 恢复结论与磁盘事实一致。
"""
import json
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import publisher as pub  # noqa: E402
from publish_journal import read_journal  # noqa: E402

_CHILD = r"""
import json, os, shutil, sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {shared!r})
import publisher as pub

BASE = {base!r}
KILL_AT = {kill_at!r}          # 在第几次 os.replace 时硬退出（1-based）
MODE = {mode!r}                # "replace" 或 "journal"

def _mk(name, data):
    p = os.path.join(BASE, name)
    with open(p, "wb") as f:
        f.write(data)
    return p

# ---- 准备旧代 A 与候选 B ----
old_media = _mk("out.mkv", b"OLD-MEDIA")
old_verify = _mk("out.mkv.verify.json", b"OLD-VERIFY")
staged_media = _mk("stage_media", b"NEW-MEDIA")
staged_verify = _mk("stage_verify", b"NEW-VERIFY")

arts = [
    pub.Artifact("media", staged_media, old_media),
    pub.Artifact("verify", staged_verify, old_verify),
]

real_replace = os.replace
_n = {{"c": 0}}

def _replace(src, dst):
    _n["c"] += 1
    if _n["c"] == KILL_AT:
        os.kill(os.getpid(), 9)   # Windows: TerminateProcess，无 finally
    return real_replace(src, dst)

pub.os.replace = _replace

real_write = pub.write_journal
_calls = {{"c": 0}}

def _write(path, doc, **kw):
    if doc.get("phase") == "COMMITTED":
        _calls["c"] += 1
        if MODE == "journal" and _calls["c"] == KILL_AT:
            os.kill(os.getpid(), 9)
    return real_write(path, doc, **kw)

pub.write_journal = _write

res = pub.publish(arts, os.path.join(BASE, "out.mkv.journal.json"),
                  os.path.join(BASE, "backup"), "att-child",
                  decision={{"publish": True}})
print("SURVIVED", res.ok)
"""


def _run_child(base, kill_at, mode="replace"):
    code = _CHILD.format(root=_ROOT, shared=os.path.join(_ROOT, "shared_modules"),
                         base=str(base), kill_at=kill_at, mode=mode)
    p = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True, timeout=180)
    return p


def _seed_a(base):
    (base / "out.mkv").write_bytes(b"OLD-MEDIA")
    (base / "out.mkv.verify.json").write_bytes(b"OLD-VERIFY")


# 3 次 os.replace 用于 journal（PREPARED/COMMITTING/COMMITTED 前后），
# 第 4/5 次才是**晋升**两个产物 —— 这两处才是要测的临界点。
@pytest.mark.parametrize("kill_at", [4, 5])
def test_b15_hard_exit_during_promote_leaves_a_consistent_group(tmp_path,
                                                                kill_at):
    """在第 1/2 次晋升（os.replace）时硬杀子进程 → 绝不混代。"""
    base = tmp_path / "g"
    base.mkdir()
    _seed_a(base)

    proc = _run_child(base, kill_at)
    # SIGKILL → 负返回码（Windows 上可能被映射成大正数，两者都算"被杀"）
    assert proc.returncode != 0, f"子进程居然正常退出：{proc.stdout}"
    assert "SURVIVED" not in proc.stdout, proc.stdout

    media = str(base / "out.mkv")
    verify = str(base / "out.mkv.verify.json")

    # 崩溃瞬间：进程已死、没人恢复，磁盘上**允许**停在半代
    # （这正是"必须先写 journal 再动文件"的意义：事后能恢复）。
    m = open(media, "rb").read()
    v = open(verify, "rb").read()
    assert m in (b"OLD-MEDIA", b"NEW-MEDIA"), m
    assert v in (b"OLD-VERIFY", b"NEW-VERIFY"), v

    # **核心要求：恢复之后必须是同代**
    jp = str(base / "out.mkv.journal.json")
    rep = pub.recover(jp, str(base / "backup"), "att-parent")
    m2 = open(media, "rb").read()
    v2 = open(verify, "rb").read()
    assert (m2, v2) in ((b"OLD-MEDIA", b"OLD-VERIFY"),
                        (b"NEW-MEDIA", b"NEW-VERIFY")), (m2, v2)

    # 恢复报告必须与磁盘事实一致（不能谎报已回滚）
    if rep is not None and rep.get("ok"):
        doc = read_journal(jp) or {}
        gen = pub.classify_generation(doc.get("records") or {})
        assert gen in (pub.GENERATION_ALL_OLD, pub.GENERATION_ALL_NEW), (gen, rep)
        act = rep.get("action")
        if act in ("confirmed_rolled_back", "already_rolled_back"):
            assert (m2, v2) == (b"OLD-MEDIA", b"OLD-VERIFY"), (act, m2, v2)
        if act == "already_committed":
            assert (m2, v2) == (b"NEW-MEDIA", b"NEW-VERIFY"), (act, m2, v2)


def test_b15_hard_exit_after_commit_write(tmp_path):
    """在**写 COMMITTED 时**硬杀 → 重启后仍能按磁盘事实确认/补记。"""
    base = tmp_path / "g2"
    base.mkdir()
    _seed_a(base)

    proc = _run_child(base, 1, mode="journal")
    assert proc.returncode != 0, proc.stdout

    media = str(base / "out.mkv")
    verify = str(base / "out.mkv.verify.json")
    m = open(media, "rb").read()
    v = open(verify, "rb").read()
    assert (m, v) in ((b"OLD-MEDIA", b"OLD-VERIFY"),
                      (b"NEW-MEDIA", b"NEW-VERIFY")), (m, v)

    jp = str(base / "out.mkv.journal.json")
    exists = os.path.exists(jp)
    rep = pub.recover(jp, str(base / "backup"), "att-parent")
    if not exists:
        # 连 journal 都没写出来 → 什么都没发生，磁盘必须是完整旧代
        assert rep is None, rep
    elif rep is not None and rep.get("ok") is False:
        # 允许失败，但**不允许**把现场销毁
        assert os.path.exists(jp), "恢复失败却把 journal 删了"
    # 无论结果如何，两份仍必须同代
    m2 = open(media, "rb").read()
    v2 = open(verify, "rb").read()
    assert (m2, v2) in ((b"OLD-MEDIA", b"OLD-VERIFY"),
                        (b"NEW-MEDIA", b"NEW-VERIFY")), (m2, v2)
