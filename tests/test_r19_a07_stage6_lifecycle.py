# -*- coding: utf-8 -*-
"""第十九轮 A3：stage6 管道生命周期（对应 R18-A07）。

审核指出的三个问题（原话）：

> 第二次 Popen 失败发生在登记与收尾之前；外层 except 只返回错误，
> 已经启动的 vspipe 未被回收。正常返回只看 ffmpeg 的 returncode，
> 生产者 rc=9、编码器 rc=0 的受控例返回 `(0, '')`。
> stderr 虽被并发排空，却持续保存在列表，长任务的日志内存没有上限。

本文件用**真实子进程**验证，不用 mock 子进程对象：

- 用 `python -c` 当假 vspipe（先写 pid 文件证明已启动，再 sleep，最后写 marker）；
- 用不存在的可执行文件当假 ffmpeg，让第二次 Popen 真的抛错；
- 如果第一个进程没被回收，它 sleep 完会写出 marker —— 断言 marker **不存在**。
"""
import os
import subprocess
import sys
import time

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage6_grain_synthesis as s6  # noqa: E402


class _Log:
    def __init__(self):
        self.events = []

    def _rec(self, lvl, evt, msg="", **kw):
        self.events.append((lvl, evt, str(msg)[:200]))

    def info(self, evt, msg="", **kw):
        self._rec("INFO", evt, msg, **kw)

    def warning(self, evt, msg="", **kw):
        self._rec("WARN", evt, msg, **kw)

    def error(self, evt, msg="", **kw):
        self._rec("ERR", evt, msg, **kw)

    def events_of(self, prefix):
        return [e for e in self.events if e[1].startswith(prefix)]


def _py(code):
    return [sys.executable, "-c", code]


# ------------------------------------------------ N17：部分启动必须回收

def test_partial_start_reaps_the_first_process(tmp_path):
    """**核心缺口**：第二个 Popen 失败时，第一个进程必须被回收。

    旧实现：vspipe 已 Popen 成功 → ffmpeg 的 Popen 抛错 → 外层 except
    直接 return，vspipe 继续跑（本例里它 sleep 完会写出 marker）。
    """
    marker = tmp_path / "still_alive.txt"
    out = tmp_path / "out.mkv"
    vs_path = tmp_path / "grain.vpy"

    # 假 vspipe：等 2 秒后写 marker。如果被正确回收，marker 永远不会出现。
    #
    # 注意：**不能**用"子进程写出 pidfile"来证明它启动过 —— 新实现会在
    # 第二个 Popen 失败后立刻 terminate 整组，子进程往往还没跑到写文件
    # 那一步就被杀掉了（这恰恰说明回收生效）。改为看 detail 里的步骤标记。
    vspipe_cmd = _py(
        "import sys, time, pathlib\n"
        "sys.stdout.buffer.write(b'x' * 4096)\n"
        "sys.stdout.buffer.flush()\n"
        "time.sleep(2.0)\n"
        f"pathlib.Path(r'{marker}').write_text('alive')\n")

    # 假 ffmpeg：路径不存在 → 第二次 Popen 必然抛 FileNotFoundError
    ffmpeg_cmd = [str(tmp_path / "no-such-ffmpeg-binary"), "-f", "null", "-"]

    log = _Log()
    status, detail = s6._run_grain_pipeline(
        "clip = None", str(vs_path), vspipe_cmd, ffmpeg_cmd,
        str(out), log)

    assert status != 0, f"启动失败必须返回非 0，实际 {status!r} {detail!r}"
    # detail 必须指出**失败发生在哪一步**：这里是第二步 ffmpeg，
    # 说明第一步 vspipe 已经起来了 —— 所以紧接着的回收断言才有意义。
    assert "at ffmpeg" in detail, f"失败步骤未标注清楚：{detail!r}"

    # 等到 marker 本该出现的时刻之后，断言它不存在
    # （旧实现不回收第一个进程 → 它 sleep 完会写出 marker）
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        if marker.exists():
            break
        time.sleep(0.1)
    assert not marker.exists(), (
        "第一个进程没有被回收：它活到了 sleep 结束（会继续写输出文件）")


# ------------------------------------------------ N17：只看 encoder 会假成功

def test_producer_failure_is_not_reported_as_success(tmp_path):
    """生产者 rc=9、编码器 rc=0 → **绝不能**返回成功。

    旧实现 `if ffmpeg_proc.returncode == 0: return (0, "")`
    —— 上游写失败、下游正常收尾，就报成功。
    """
    out = tmp_path / "out.bin"
    vs_path = tmp_path / "grain.vpy"

    # 假 vspipe：写一点数据然后以 9 退出
    vspipe_cmd = _py(
        "import sys\n"
        "sys.stdout.buffer.write(b'D' * 8192)\n"
        "sys.stdout.buffer.flush()\n"
        "sys.exit(9)\n")

    # 假 ffmpeg：读完 stdin、写出文件、以 0 退出（"正常收尾"）
    ffmpeg_cmd = _py(
        "import sys, pathlib\n"
        "data = sys.stdin.buffer.read()\n"
        f"pathlib.Path(r'{out}').write_bytes(data)\n"
        "sys.exit(0)\n")

    log = _Log()
    status, detail = s6._run_grain_pipeline(
        "clip = None", str(vs_path), vspipe_cmd, ffmpeg_cmd,
        str(out), log)

    assert status != 0, (
        f"生产者 rc=9 却报了成功：status={status!r} detail={detail!r}")
    assert "9" in str(detail) or "producer" in str(detail).lower(), detail
    # 下游那个"成功"的文件是半成品，不该留下
    assert not out.exists(), "失败时不得留下半成品"


def test_clean_run_returns_zero(tmp_path):
    """正常路径必须仍然返回成功（防止修过头）。"""
    out = tmp_path / "out.bin"
    vs_path = tmp_path / "grain.vpy"

    vspipe_cmd = _py("import sys; sys.stdout.buffer.write(b'D' * 4096)\n"
                     "sys.stdout.buffer.flush()\n")
    ffmpeg_cmd = _py(
        "import sys, pathlib\n"
        "data = sys.stdin.buffer.read()\n"
        f"pathlib.Path(r'{out}').write_bytes(data)\n")

    log = _Log()
    status, detail = s6._run_grain_pipeline(
        "clip = None", str(vs_path), vspipe_cmd, ffmpeg_cmd,
        str(out), log)
    assert status == 0, (status, detail)
    assert out.exists() and out.stat().st_size == 4096


def test_cancel_guard_registration_is_released(tmp_path):
    """所有进程都必须被 unwatch —— 否则后续 stage 的取消会误伤。"""
    out = tmp_path / "out.bin"
    vs_path = tmp_path / "grain.vpy"

    class Guard:
        def __init__(self):
            self.watched = []
            self.unwatched = []

        def watch_process(self, p):
            self.watched.append(p)

        def unwatch_process(self, p):
            self.unwatched.append(p)

        # 不用 wait_with_heartbeat → 走有界等待分支
        def __getattr__(self, name):
            raise AttributeError(name)

    guard = Guard()
    old = s6._cancel_guard
    s6._cancel_guard = guard
    try:
        vspipe_cmd = _py("import sys; sys.stdout.buffer.write(b'D' * 1024)\n"
                         "sys.stdout.buffer.flush()\n")
        ffmpeg_cmd = _py(
            "import sys, pathlib\n"
            "data = sys.stdin.buffer.read()\n"
            f"pathlib.Path(r'{out}').write_bytes(data)\n")
        s6._run_grain_pipeline("clip = None", str(vs_path), vspipe_cmd,
                               ffmpeg_cmd, str(out), _Log())
    finally:
        s6._cancel_guard = old

    assert len(guard.watched) == 2, guard.watched
    assert len(guard.unwatched) == 2, guard.unwatched


# ------------------------------------------------ N18：stderr 内存必须有界

def test_tail_log_keeps_memory_bounded(tmp_path):
    """长任务的 stderr 不能无限累加。

    旧实现 `_sinks = {"enc": [], "vs": []}` 把**全部**输出留在内存里；
    4K 长片 + 详细日志可以到几百 MB。
    """
    assert hasattr(s6, "_TailLog"), "缺少有界日志容器 _TailLog"
    path = tmp_path / "enc.log"
    buf = s6._TailLog(str(path), limit=4096)
    try:
        for _ in range(200):
            buf.write(b"A" * 1024)          # 共 200 KiB
    finally:
        buf.close()

    tail = buf.tail_bytes()
    assert len(tail) <= 4096, f"内存尾部未受限：{len(tail)}"
    assert tail.endswith(b"A" * 1024), "尾部内容应是最后写入的块"
    # 全文必须落盘（不能因为内存有界就丢日志）
    assert path.stat().st_size == 200 * 1024, path.stat().st_size
