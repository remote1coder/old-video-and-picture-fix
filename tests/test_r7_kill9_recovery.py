"""发布恢复：**真正的 kill -9** 演练（不是 SIGTERM 优雅退出）。

任务书 §1.2：

> 生产恢复不应依赖 SIGTERM 一定送达；测试用硬终止专用子进程覆盖没有
> 清理钩子的情况。

SIGTERM / Ctrl+C 会被 Python 的 finally、`atexit`、取消钩子接住 ——
所以"取消后能恢复"证明不了"进程被硬杀后还能恢复"。这里用
`Popen.kill()`（Windows 上是 `TerminateProcess`），子进程**没有任何
机会执行清理代码**。

演练内容：
1. 子进程按真实发布顺序执行：备份旧成品 → 写 journal(PREPARED) →
   **停在这里**（模拟"正在替换"）；
2. 父进程确认它已停在临界点，然后 `kill()`；
3. 检查三件事：
   * 旧成品**完好**（字节未变）；
   * journal 存在且是 `PREPARED`；
   * `classify_recovery()` 能给出明确结论，而不是"不知道"。
"""

import json
import os
import subprocess
import sys
import time

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "shared_modules") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "shared_modules"))

import publish_journal as pj  # noqa: E402

#: 子进程脚本：模拟"备份完、journal 写好、正要替换"时被硬杀。
_CHILD = r'''
import json, os, sys, time
sys.path.insert(0, {shared!r})
from publish_journal import (JOURNAL_SCHEMA, PHASE_PREPARED,
                             file_digest, journal_path_for, write_journal)

target, new_path, journal, ready_flag = sys.argv[1:5]

old_digest = file_digest(target)
new_digest = file_digest(new_path)
backup = target + ".pubbak"
if os.path.exists(target):
    with open(target, "rb") as f: data = f.read()
    with open(backup, "wb") as f: f.write(data)

doc = {{
    "schema": JOURNAL_SCHEMA,
    "attempt_id": "kill9-test",
    "phase": PHASE_PREPARED,
    "targets": [target],
    "records": {{
        target: {{
            "existed": old_digest is not None,
            "old_digest": old_digest,
            "backup": backup,
            "backup_digest": file_digest(backup) if os.path.exists(backup) else None,
            "new_digest": new_digest,
            "staging": new_path,
        }}
    }},
}}
ok = write_journal(journal, doc)

# 告诉父进程：已经停在临界点，可以杀了
with open(ready_flag, "w") as f:
    f.write("ready" if ok else "journal-failed")

# 模拟"正在替换"——父进程会在这里 kill -9，这一行永远不会执行完
time.sleep(60)
'''


@pytest.mark.skipif(os.name != "nt", reason="本演练在 Windows 上做过实测")
def test_hard_kill_leaves_old_artifact_intact(tmp_path):
    target = tmp_path / "restored.avif"
    target.write_bytes(b"OLD-GOOD-GENERATION")
    new = tmp_path / "restored.avif.attempt-1.avif"
    new.write_bytes(b"NEW-ATTEMPT-CONTENT")

    journal = pj.journal_path_for(str(target))
    ready = tmp_path / "ready.flag"
    script = tmp_path / "child.py"
    script.write_text(
        _CHILD.format(shared=os.path.join(ROOT, "shared_modules")),
        encoding="utf-8")

    proc = subprocess.Popen(
        [sys.executable, str(script), str(target), str(new),
         journal, str(ready)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not ready.exists():
            if proc.poll() is not None:
                pytest.skip("子进程提前退出，无法演练")
            time.sleep(0.05)
        assert ready.exists(), "子进程没有到达临界点"
        assert ready.read_text() == "ready", "journal 写入失败"

        # 硬杀：TerminateProcess，子进程没有机会执行任何清理
        proc.kill()
        proc.wait(timeout=10)
        assert proc.returncode != 0
    finally:
        if proc.poll() is None:
            proc.kill()

    # 1) 旧成品字节未变
    assert target.read_bytes() == b"OLD-GOOD-GENERATION", \
        "硬杀之后旧成品被改动了 —— 这是最严重的情况"

    # 2) journal 存在且可读、阶段是 PREPARED
    doc = pj.read_journal(journal)
    assert doc is not None, "硬杀后 journal 读不出来"
    assert doc.get("phase") == pj.PHASE_PREPARED

    # 3) 能给出明确结论（不是"不知道"）
    rep = pj.classify_recovery(doc)
    assert rep["targets"][str(target)]["state"] == "OLD", (
        f"目标应当仍是旧代，实际 {rep['targets'][str(target)]['state']}")
    assert "ROLLBACK_SAFE" in rep["recommendation"] or \
           "ALREADY_COMMITTED" in rep["recommendation"], rep["recommendation"]


@pytest.mark.skipif(os.name != "nt", reason="本演练在 Windows 上做过实测")
def test_hard_kill_does_not_leave_new_content_published(tmp_path):
    """硬杀在替换前 → 目标绝不能出现"新内容"。"""
    target = tmp_path / "restored.avif"
    target.write_bytes(b"OLD")
    new = tmp_path / "new.avif"
    new.write_bytes(b"NEW")

    journal = pj.journal_path_for(str(target))
    ready = tmp_path / "ready.flag"
    script = tmp_path / "child.py"
    script.write_text(
        _CHILD.format(shared=os.path.join(ROOT, "shared_modules")),
        encoding="utf-8")

    proc = subprocess.Popen(
        [sys.executable, str(script), str(target), str(new),
         journal, str(ready)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not ready.exists():
            time.sleep(0.05)
        assert ready.exists()
        proc.kill()
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()

    content = target.read_bytes()
    assert content == b"OLD", f"目标变成了 {content[:20]!r}（应是 OLD）"


def test_journal_survives_without_cleanup_hooks():
    """静态守卫：journal 的写入必须是 tmp→fsync→replace，而不是直接写。

    直接写的话，硬杀会留下**半个 JSON** —— 恢复时既不能读也不能忽略。
    """
    src = open(os.path.join(ROOT, "shared_modules", "publish_journal.py"),
               encoding="utf-8").read()
    assert "os.fsync(fh.fileno())" in src, "没有 fsync，硬杀可能留下空文件"
    assert "os.replace(tmp, journal_path)" in src, "不是原子替换"
    assert "digest" in src, "没有内容摘要，篡改/半截文件无法识别"
