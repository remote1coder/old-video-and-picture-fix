# -*- coding: utf-8 -*-
"""看门狗的"进程树 CPU"分量：纯计算段落的存活信号。

## 为什么需要它

`progress_watch` 只看"写了多少字节 / 日志涨了多少"。凡是**纯计算、不写文件、
也不打日志**的段落，这两个信号都静止 —— 长片实测被误杀 5 次
（stage2 摘要、stage3 正式遍摘要、stage5 人脸推理、stage7 fullchain、
stage7 pre-publish）。之前只能一处一处补进度文件/进度日志。

进程树 CPU 是操作系统计数器，不受 GIL、日志锁、缓冲影响。真死锁时它也不涨，
所以**不掩盖**停滞。

## 一个容易搞错的前提

`GetProcessTimes` / `psutil.cpu_times()` 只算**该进程自己的线程**。stage 脚本
大部分时间在等子进程，实测：正在编码的 main.py 自身 11.48s、进程树 52.08s。
只读自身必然误判"没在干活"。所以**必须递归子进程**。

## 一段踩坑史（写在这里以免重复）

第一版我测 `python -c "while True: pass"`，读到 0.0156 → 0.0156 不涨，
就下了"本机 CPU 计数器不可用"的结论，还把这条写进了文档。
后来用**脚本文件**起同样的死循环，`Get-Process` 读到 1.52 → 4.52
（3 秒墙钟涨 3.00 秒）—— **计数器一直是好的，是我的测试脚本错了**
（`-c` 带空格被参数拆分，进程启动即退出，我读到的是 launcher 的 pid）。

教训：**测试工具本身也要被验证**。用"对照组 + 脚本文件"而不是内联带空格的 `-c`。
"""
import os
import subprocess
import sys
import time

import pytest

import conftest  # noqa: F401

import cancel_guard as CG


def test_unknown_pid_returns_zero():
    assert CG._tree_cpu_seconds(999999) == 0.0


def test_reads_growing_cpu_from_a_real_child(tmp_path):
    """起一个真吃 CPU 的子进程（脚本文件，避免 -c 参数被拆），读到增长。"""
    if sys.platform != "win32" and not os.path.exists("/proc/self/stat"):
        pytest.skip("no way to read process cpu time")
    script = tmp_path / "burn.py"
    script.write_text(
        "import time\n"
        "t = time.time()\n"
        "while time.time() - t < 20:\n"
        "    pass\n", encoding="ascii")
    p = subprocess.Popen([sys.executable, str(script)])
    try:
        time.sleep(1.5)
        a = CG._tree_cpu_seconds(p.pid)
        time.sleep(2.0)
        b = CG._tree_cpu_seconds(p.pid)
    finally:
        p.kill(); p.wait()
    assert a > 0.0, f"读不到 CPU 时间（a={a}）"
    assert b - a > 1.0, f"2 秒墙钟只涨了 {b - a:.2f}s CPU，进程树读数不可靠"


def test_tree_cpu_counts_descendants(tmp_path):
    """父进程只 sleep，子进程吃 CPU —— 树读数应远大于父进程自身。"""
    child = tmp_path / "burn.py"
    child.write_text(
        "import time\n"
        "t = time.time()\n"
        "while time.time() - t < 20:\n"
        "    pass\n", encoding="ascii")
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, r'{child}'])\n"
        "time.sleep(20)\n", encoding="ascii")
    p = subprocess.Popen([sys.executable, str(parent)])
    try:
        time.sleep(2.5)
        try:
            import psutil
        except Exception:
            pytest.skip("psutil unavailable")
        self_s = psutil.Process(p.pid).cpu_times()
        self_s = self_s.user + self_s.system
        tree_s = CG._tree_cpu_seconds(p.pid)
    finally:
        p.kill()
        try:
            import psutil
            for c in psutil.Process(p.pid).children(recursive=True):
                try: c.kill()
                except Exception: pass
        except Exception:
            pass
        p.wait()
    assert tree_s > self_s + 1.0, (
        f"进程树 {tree_s:.2f}s 未包含后代（父自身 {self_s:.2f}s）")


def test_marker_is_a_tuple_of_bytes_and_cpu():
    """结构守卫：marker 必须是二元组，否则 CPU 分量等于没接入。"""
    import io
    src = io.open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "shared_modules", "cancel_guard.py"),
        encoding="utf-8", errors="replace").read()
    assert "return (_b, round(_tree_cpu_seconds(proc.pid), 1))" in src
    # 两个调用点：初始化 `last_mark = _mark()` 与每次采样 `mark = _mark()`
    assert "last_mark = _mark()" in src, "缺少初始化"
    assert "\n                    mark = _mark()" in src, "采样点未接入 CPU 分量"
