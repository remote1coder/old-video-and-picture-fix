"""看门狗（inactivity watchdog）的回归测试。

FIX-2026-10-21：项目原用固定 ``subprocess.run(timeout=N)`` 判断"卡死"，
而 N 是按 ``duration × 系数`` 猜出来的。16 分钟 1080p 实测暴露问题：
纯传统链 stage7 编码要 3 小时（旧公式给 42 分钟），开了 AI 后 4 小时
上限更不够；而真正的卡死反而要等到上限才被发现。

新语义：**产出不再增长**才是卡死的证据。这组测试锁住三条：
1. 完全没进展 → 在 inactivity_timeout 附近抛 StageStalled；
2. 慢但持续推进 → 不被误杀；
3. 抛异常时子进程被真正杀掉（不留孤儿）。
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared_modules"))

from cancel_guard import StageStalled, run_managed  # noqa: E402


def _dir_marker(root: str):
    def _marker() -> int:
        total = 0
        for r, _d, files in os.walk(root):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(r, fn))
                except OSError:
                    pass
        return total
    return _marker


def test_watchdog_kills_when_no_progress(tmp_path):
    """写一次就挂住 → 必须在阈值附近被判卡死。"""
    target = str(tmp_path / "once.bin").replace("\\", "\\\\")
    code = (f'import time; open(r"{target}", "wb").write(b"x" * 2000); '
            'time.sleep(40)')
    started = time.monotonic()
    with pytest.raises(StageStalled):
        run_managed([sys.executable, "-c", code], timeout=120,
                    progress_watch=_dir_marker(str(tmp_path)),
                    inactivity_timeout=3, poll_interval=1)
    elapsed = time.monotonic() - started
    # 3s 阈值 + 1s 采样粒度 → 应在 3~10 秒之间结束，绝不能拖到 40 秒
    assert 2.0 <= elapsed <= 12.0, f"detected after {elapsed:.1f}s"


def test_watchdog_does_not_kill_a_slow_but_progressing_child(tmp_path):
    """每 0.5 秒追加数据 —— 慢，但一直在推进，不该被杀。"""
    target = str(tmp_path / "slow.bin").replace("\\", "\\\\")
    code = (f'import time\n'
            f'f = open(r"{target}", "ab")\n'
            'for _ in range(12):\n'
            '    f.write(b"x" * 20000); f.flush(); time.sleep(0.5)\n')
    started = time.monotonic()
    r = run_managed([sys.executable, "-c", code], timeout=120,
                    progress_watch=_dir_marker(str(tmp_path)),
                    inactivity_timeout=3, poll_interval=1)
    elapsed = time.monotonic() - started
    assert r.returncode == 0
    assert elapsed >= 5.0, f"finished suspiciously early ({elapsed:.1f}s)"


def test_watchdog_reaps_the_child(tmp_path):
    """判卡死后必须真的杀掉子进程，否则孤儿会继续写半成品（RV-12 同族）。"""
    pid_file = str(tmp_path / "pid.txt").replace("\\", "\\\\")
    code = (f'import os, time\n'
            f'open(r"{pid_file}", "w").write(str(os.getpid()))\n'
            'time.sleep(60)\n')
    with pytest.raises(StageStalled):
        run_managed([sys.executable, "-c", code], timeout=120,
                    progress_watch=lambda: 0,
                    inactivity_timeout=2, poll_interval=1)
    time.sleep(1.0)
    pid = int(Path(pid_file).read_text().strip())
    alive = True
    try:
        os.kill(pid, 0)
    except OSError:
        alive = False
    except PermissionError:
        alive = True
    assert not alive, f"child {pid} still alive after stall detection"


def test_watchdog_disabled_keeps_old_behaviour(tmp_path):
    """不传 progress_watch → 行为与改动前完全一致（不被看门狗干扰）。"""
    code = 'print("done")'
    r = run_managed([sys.executable, "-c", code], timeout=60,
                    capture_output=True, text=True)
    assert r.returncode == 0
    assert "done" in r.stdout
