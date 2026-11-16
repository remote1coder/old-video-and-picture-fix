# -*- coding: utf-8 -*-
"""跨 stage 管道编排（N27-04）。

三条硬约束（AGENTS §3.2）：整组可回收、不能中途 resume、stderr 各自落文件。
成功判据：**所有** rc == 0 —— 下游碰巧产出一帧不算成功（PA03 同一原则）。
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared_modules"))

import stream_chain as SC  # noqa: E402

PY = sys.executable


def test_two_process_chain_succeeds(tmp_path):
    cmds = [[PY, "-c", "print('hello')"],
            [PY, "-c", "import sys; d=sys.stdin.read(); assert 'hello' in d"]]
    r = SC.run_chain(cmds, tmpd=str(tmp_path), label="ok2")
    assert r["ok"] is True, r
    assert r["rcs"] == [0, 0], r
    assert (tmp_path / "ok2-0.err").exists()
    assert (tmp_path / "ok2-1.err").exists()


def test_middle_failure_fails_the_whole_chain(tmp_path):
    """**中间进程失败**必须让整组失败，哪怕最后一个正常退出。"""
    cmds = [[PY, "-c", "print('x')"],
            [PY, "-c", "import sys; sys.stdout.write('y'); sys.exit(3)"],
            [PY, "-c", "import sys; sys.stdin.read()"]]
    r = SC.run_chain(cmds, tmpd=str(tmp_path), label="bad")
    assert r["ok"] is False, r
    assert any(rc == 3 for rc in r["rcs"]), r
    assert r["errors"], r


def test_timeout_kills_the_whole_group(tmp_path):
    cmds = [[PY, "-c", "import time; time.sleep(30)"],
            [PY, "-c", "import sys; sys.stdin.read()"]]
    r = SC.run_chain(cmds, tmpd=str(tmp_path), label="slow", timeout_s=0.6)
    assert r["ok"] is False
    assert r["timeout"] is True
    assert r["killed"] is True
    assert r["elapsed"] < 10, ("整组没被杀干净，耗时 %.1fs" % r["elapsed"])


def test_stderr_goes_to_files_not_pipes(tmp_path):
    """每个进程的 stderr 落到自己的文件（挂 PIPE 不读会阻塞整条链）。"""
    cmds = [[PY, "-c",
             "import sys; sys.stderr.write('E'*5000); print('x')"],
            [PY, "-c", "import sys; sys.stdin.read()"]]
    r = SC.run_chain(cmds, tmpd=str(tmp_path), label="err")
    assert r["ok"] is True, r
    size = (tmp_path / "err-0.err").stat().st_size
    assert size >= 5000, size


def test_single_command_is_rejected(tmp_path):
    r = SC.run_chain([[PY, "-c", "print(1)"]], tmpd=str(tmp_path))
    assert r["ok"] is False
    assert "at least two" in r["errors"][0]


def test_empty_argv_is_rejected(tmp_path):
    r = SC.run_chain([[PY, "-c", "print(1)"], []], tmpd=str(tmp_path))
    assert r["ok"] is False
    assert any("empty command" in e for e in r["errors"]), r
