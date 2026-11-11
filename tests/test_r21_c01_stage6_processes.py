# -*- coding: utf-8 -*-
"""R21-C01：stage6 管道的**真实轻量子进程**行为。

## 为什么要这一组

审核 R20 §3.1：

> 进程回收顺序还有静态风险：停止子进程前先等待日志线程，停止后没有完整
> 再次收束。应以轻量进程验证背压、大日志、部分启动和取消；本轮不宣称已复现
> Windows 死锁。

以及执行指令 §3.2：

> 用实际轻量子进程覆盖：生产者先失败、消费者先失败、第二个进程启动失败、
> 消费者背压、大量 stderr、无产出但持续日志的假心跳、超时、取消、线程排空。
> 断言所有自建子进程已结束、输出状态正确、无孤儿；真实长编码另列。

## 做法

用 `sys.executable -c "<inline>"` 当生产者和消费者 —— 是**真的进程**，
但不需要 vspipe/ffmpeg/GPU。每个子进程把自己的 PID 写进文件，
测试结束时逐个核对"还活着吗"，以此断言没有孤儿。

## 边界

真实 vspipe + ffmpeg 的长时间编码**不在本文件范围**（需真实素材与数十分钟），
在关闭表里单列为"长编码另列"。
"""
import os
import subprocess
import sys
import time

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage6_grain_synthesis as s6  # noqa: E402

PY = sys.executable


class _Log:
    def __init__(self):
        self.events = []

    def _rec(self, ev, msg="", **kw):
        self.events.append((ev, msg, kw))

    info = warning = error = critical = _rec

    def names(self):
        return [e[0] for e in self.events]


def _pid_alive(pid):
    """子进程是否还活着（Windows 用 OpenProcess，避免 os.kill 语义陷阱）。

    注意：**不能用 `os.kill(pid, 0)`** —— Windows 上它的含义是
    TerminateProcess，会直接把进程杀掉，测试就失去意义了。
    """
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        SYNCHRONIZE = 0x00100000
        h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
        if not h:
            return False
        try:
            # WAIT_TIMEOUT(258) = 仍存活；WAIT_OBJECT_0(0) = 已退出
            return ctypes.windll.kernel32.WaitForSingleObject(h, 0) == 258
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class _Case:
    """一次管道运行的上下文：PID 文件 + 输出路径。"""

    def __init__(self, tmp_path, name):
        self.dir = tmp_path / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.out = str(self.dir / "out.mkv")
        self.vs_pid = str(self.dir / "vs.pid")
        self.enc_pid = str(self.dir / "enc.pid")

    def pids(self):
        got = []
        for f in (self.vs_pid, self.enc_pid):
            try:
                got.append(int(open(f).read().strip()))
            except (OSError, ValueError):
                pass
        return got


def _py(code):
    return [PY, "-c", code]


def producer(kind, case):
    """生产者：往 stdout 写"帧"，可选往 stderr 写日志。"""
    pid = case.vs_pid.replace("\\", "\\\\")
    if kind == "ok":
        return _py(
            f"import os,sys;open(r'{pid}','w').write(str(os.getpid()));"
            f"sys.stdout.buffer.write(b'F'*1024);sys.stdout.flush()")
    if kind == "fail":
        return _py(
            f"import os,sys;open(r'{pid}','w').write(str(os.getpid()));"
            f"sys.stderr.write(\"AttributeError: No attribute with the name "
            f"grainfactory exists\");sys.stderr.flush();sys.exit(3)")
    if kind == "chatty":
        # 200 KB stderr —— 管道缓冲区默认 64 KB，不并发排空就会卡死
        return _py(
            f"import os,sys;open(r'{pid}','w').write(str(os.getpid()));"
            f"sys.stderr.write('x'*200000);sys.stderr.flush();"
            f"sys.stdout.buffer.write(b'F'*1024);sys.stdout.flush()")
    if kind == "bulk":
        # 8 MB，制造背压
        return _py(
            f"import os,sys;open(r'{pid}','w').write(str(os.getpid()));"
            f"chunk=b'F'*65536\n"
            f"for _ in range(128):\n"
            f"    sys.stdout.buffer.write(chunk)\n"
            f"sys.stdout.flush()")
    if kind == "silent_busy":
        # 没有任何 stdout 产出，但 stderr 一直在响 —— 假心跳
        return _py(
            f"import os,sys,time;open(r'{pid}','w').write(str(os.getpid()));"
            f"t=time.time()\n"
            f"while time.time()-t < 12:\n"
            f"    sys.stderr.write('still working\\n');sys.stderr.flush()\n"
            f"    time.sleep(0.05)")
    raise AssertionError(kind)


def consumer(kind, case):
    pid = case.enc_pid.replace("\\", "\\\\")
    out = case.out.replace("\\", "\\\\")
    if kind == "ok":
        return _py(
            f"import os,sys;open(r'{pid}','w').write(str(os.getpid()));"
            f"d=sys.stdin.buffer.read();"
            f"open(r'{out}','wb').write(d);print(len(d))")
    if kind == "fail":
        return _py(
            f"import os,sys;open(r'{pid}','w').write(str(os.getpid()));"
            f"sys.stderr.write('encoder exploded');sys.stderr.flush();"
            f"sys.exit(7)")
    if kind == "slow":
        return _py(
            f"import os,sys,time;open(r'{pid}','w').write(str(os.getpid()));"
            f"n=0\n"
            f"while True:\n"
            f"    c=sys.stdin.buffer.read(65536)\n"
            f"    if not c: break\n"
            f"    n+=len(c)\n"
            f"    open(r'{out}','ab').write(c)\n"
            f"    time.sleep(0.02)\n"
            f"print(n)")
    if kind == "wait":
        return _py(
            f"import os,sys;open(r'{pid}','w').write(str(os.getpid()));"
            f"d=sys.stdin.buffer.read();"
            f"open(r'{out}','wb').write(d)")
    raise AssertionError(kind)


def _run(case, prod, cons, inactivity_s=60.0, logger=None):
    return s6._run_grain_pipeline(
        "print('script')", str(case.dir / "grain.vpy"),
        producer(prod, case), consumer(cons, case),
        case.out, logger or _Log(), inactivity_s=inactivity_s)


def _assert_no_orphans(case):
    time.sleep(0.4)          # 给操作系统回收句柄的时间
    alive = [p for p in case.pids() if _pid_alive(p)]
    assert not alive, f"这些子进程还活着（孤儿）: {alive}"


# ------------------------------------------------ 首发失败归因

def test_producer_fails_first_is_attributed_to_the_producer(tmp_path):
    case = _Case(tmp_path, "p_first")
    r = _run(case, "fail", "ok")
    assert r.ok is False
    assert r.failed_stage == "vspipe", r.detail
    assert r.vspipe_rc == 3
    assert not os.path.exists(case.out), "失败必须清掉半成品"
    assert "grainfactory" in (r.vspipe_tail or "")
    _assert_no_orphans(case)


def test_consumer_fails_first_is_attributed_to_the_encoder(tmp_path):
    """消费者先死 → 生产者随后 EPIPE。不能把它算成插件失败，
    否则会误触发 AddGrain 回退，把磁盘/编码错误掩盖成"换个插件就好了"。"""
    case = _Case(tmp_path, "c_first")
    r = _run(case, "bulk", "fail")
    assert r.ok is False
    assert r.failed_stage == "ffmpeg", f"归因错了: {r.failed_stage} / {r.detail}"
    assert r.ffmpeg_rc == 7
    _assert_no_orphans(case)


def test_second_process_start_failure_reaps_the_first(tmp_path):
    """第二个进程起不来（命令不存在）→ 第一个也不能留成孤儿。"""
    case = _Case(tmp_path, "start_fail")
    r = s6._run_grain_pipeline(
        "print('script')", str(case.dir / "grain.vpy"),
        producer("silent_busy", case),
        ["definitely-not-a-real-binary-r21"],
        case.out, _Log(), inactivity_s=30.0)
    assert r.ok is False
    assert r.failed_stage == "start", r.detail
    _assert_no_orphans(case)


# ------------------------------------------------ 管道压力

def test_large_stderr_does_not_deadlock(tmp_path):
    """200 KB stderr：不并发排空的话，子进程会卡在写管道上永远不退出。"""
    case = _Case(tmp_path, "chatty")
    t0 = time.time()
    r = _run(case, "chatty", "ok", inactivity_s=60.0)
    assert r.ok is True, r.detail
    assert time.time() - t0 < 55
    assert os.path.getsize(case.out) == 1024
    _assert_no_orphans(case)


def test_backpressure_with_a_slow_consumer_still_completes(tmp_path):
    case = _Case(tmp_path, "backpressure")
    r = _run(case, "bulk", "slow", inactivity_s=60.0)
    assert r.ok is True, r.detail
    assert os.path.getsize(case.out) == 8 * 1024 * 1024
    _assert_no_orphans(case)


def test_fake_heartbeat_is_caught_by_inactivity(tmp_path):
    """stderr 一直在响、却没有任何产出 → 必须按"无进展"判卡死。

    心跳绑定的是**真实产出**（输出文件大小），不是一个空转的定时器；
    否则主循环卡死了照样"心跳正常"。
    """
    case = _Case(tmp_path, "fake_beat")
    t0 = time.time()
    r = _run(case, "silent_busy", "wait", inactivity_s=1.0)
    dt = time.time() - t0
    assert r.ok is False, "没有产出却被判成功"
    assert dt < 11, f"没有在 inactivity 内收手（{dt:.1f}s）"
    assert not os.path.exists(case.out)
    _assert_no_orphans(case)


# ------------------------------------------------ 取消

class _CancellingGuard:
    """取消守卫替身：等待时抛 KeyboardInterrupt（= 用户 Ctrl+C）。"""

    def watch_process(self, p):
        pass

    def unwatch_process(self, p):
        pass

    def wait_with_heartbeat(self, procs, heartbeat, inactivity_timeout,
                            poll_interval=None, logger=None, label=""):
        raise KeyboardInterrupt("user cancelled")


def test_cancel_stops_children_and_reports_cancelled(tmp_path, monkeypatch):
    monkeypatch.setattr(s6, "_cancel_guard", _CancellingGuard())
    case = _Case(tmp_path, "cancel")
    r = _run(case, "silent_busy", "wait", inactivity_s=60.0)
    assert r.cancelled is True, "取消必须被标记成 cancelled，不能当成普通失败去回退"
    assert r.ok is False
    assert not os.path.exists(case.out)
    _assert_no_orphans(case)


def test_cancel_is_not_turned_into_success(tmp_path, monkeypatch):
    monkeypatch.setattr(s6, "_cancel_guard", _CancellingGuard())
    case = _Case(tmp_path, "cancel2")
    r = _run(case, "silent_busy", "wait", inactivity_s=60.0)
    rc, degraded = s6.dispatch_result(r, True, _Log())
    assert rc == 88
    assert degraded is None
