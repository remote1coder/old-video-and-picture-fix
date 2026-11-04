# -*- coding: utf-8 -*-
"""第十八轮 P0-A / T08：`run_managed` 的输出收集类型必须与 Popen 的模式一致。

## 缺陷（审核 R17-05 指出，本次实测确认机制）

`run_managed` 里 `text` 被 **pop 掉**（不传给 Popen），但 `errors` 留在 kwargs：

```python
text = kwargs.pop("text", False)          # 被丢掉
capture = kwargs.pop("capture_output", False)
...
proc = subprocess.Popen(cmd, **kwargs)    # errors='replace' 还在
```

CPython 的判定是：

```python
self.text_mode = encoding or errors or text or universal_newlines
```

**`errors='replace'` 单独存在就会让 Popen 进入文本模式**（实测：
`Popen(..., errors='replace').communicate()` 返回 `str 'hi\\n'`，
不带 errors 时返回 `bytes b'hi\\r\\n'`）。

而 `_supervise_with_watchdog` 里的收集器固定用 bytes 拼接：

```python
out = b"".join(chunks["out"])     # chunks 里是 str → TypeError
```

## 为什么长期没暴露

只有 **带 `progress_watch`** 时才走 `_supervise_with_watchdog`；不带时走
`communicate()`（类型正确）。而且子进程通常很安静（ffmpeg `-v error`），
chunks 为空时 `b"".join([])` 恰好返回 `b""`，不报错。

**一旦子进程往 stdout/stderr 写了一个字节**，收集就抛 TypeError。
`main.run_stage` 正是 `text=True + errors="replace" + progress_watch` 的组合，
于是任何 stage 只要输出一行日志，`run_stage` 就崩成
`except Exception → return 82, str(e)`，**真实失败原因被"IO error"掩盖**。

## 验收要求

与 `subprocess.run` 语义一致：
- 文本模式（`text=True` / `encoding` / `errors` 任一存在）→ 返回 `str`
- 否则 → 返回 `bytes`
"""
import os
import subprocess
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

import cancel_guard as cg


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_stage3():
    """导入生产模块（不做任何 monkeypatch）。"""
    p = os.path.join(_ROOT, "video_restoration_pipeline", "stages")
    if p not in sys.path:
        sys.path.insert(0, p)
    import importlib
    return importlib.import_module("stage3_ai_upscale")


def _hb_dir(tmp_path):
    """一个永远在增长的心跳：让调用走 _supervise_with_watchdog。"""
    marker = tmp_path / "hb"
    marker.write_bytes(b"0")

    def hb():
        return os.path.getsize(str(marker))

    return hb


def test_T08_text_mode_with_output_does_not_raise(tmp_path):
    """**核心缺陷**：子进程有输出 + text + errors + progress_watch。

    修复前：`TypeError: sequence item 0: expected a bytes-like object,
    str found`。
    """
    r = cg.run_managed(
        [sys.executable, "-c",
         "import sys; print('hello-stdout'); "
         "sys.stderr.write('hello-stderr\\n')"],
        capture_output=True, text=True, errors="replace",
        timeout=30,
        progress_watch=_hb_dir(tmp_path),
        inactivity_timeout=10,
    )
    assert r.returncode == 0
    assert isinstance(r.stdout, str), f"文本模式应返回 str，实际 {type(r.stdout)}"
    assert isinstance(r.stderr, str), f"文本模式应返回 str，实际 {type(r.stderr)}"
    assert "hello-stdout" in r.stdout
    assert "hello-stderr" in r.stderr


def test_T08_errors_alone_implies_text_mode(tmp_path):
    """`errors` 单独存在即文本模式 —— 与 `subprocess.run` 一致。"""
    r = cg.run_managed(
        [sys.executable, "-c", "print('x')"],
        capture_output=True, errors="replace", timeout=30,
        progress_watch=_hb_dir(tmp_path), inactivity_timeout=10,
    )
    assert r.returncode == 0
    assert isinstance(r.stdout, str)


def test_T08_binary_mode_still_returns_bytes(tmp_path):
    """没要求文本模式时必须仍是 bytes —— 不能因为修 bug 把二进制搞坏。"""
    r = cg.run_managed(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'raw\\x00bytes')"],
        capture_output=True, timeout=30,
        progress_watch=_hb_dir(tmp_path), inactivity_timeout=10,
    )
    assert r.returncode == 0
    assert isinstance(r.stdout, bytes), f"二进制模式应返回 bytes，实际 {type(r.stdout)}"
    assert b"raw\x00bytes" in r.stdout


def test_T08_matches_subprocess_run_semantics(tmp_path):
    """同一命令交给 `subprocess.run` 与 `run_managed`，类型必须一致。"""
    cmd = [sys.executable, "-c",
           "import sys; print('o'); sys.stderr.write('e\\n')"]

    ref = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    got = cg.run_managed(cmd, capture_output=True, text=True, errors="replace",
                         timeout=30, progress_watch=_hb_dir(tmp_path),
                         inactivity_timeout=10)

    assert type(got.stdout) is type(ref.stdout), \
        f"stdout 类型不一致: run_managed={type(got.stdout)} subprocess.run={type(ref.stdout)}"
    assert type(got.stderr) is type(ref.stderr), \
        f"stderr 类型不一致: run_managed={type(got.stderr)} subprocess.run={type(ref.stderr)}"
    assert got.stdout == ref.stdout
    assert got.stderr == ref.stderr


def test_T08_no_progress_watch_path_unaffected(tmp_path):
    """不带 progress_watch 的老路径（communicate）本来是对的，不能被改坏。"""
    r = cg.run_managed(
        [sys.executable, "-c", "print('plain')"],
        capture_output=True, text=True, errors="replace", timeout=30,
    )
    assert r.returncode == 0
    assert isinstance(r.stdout, str)
    assert "plain" in r.stdout


# ------------------------------------------------ T09 整组回收

def _spawn_sleeper(seconds=60):
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_T09_reap_kills_whole_group():
    """T09：三个都还活着 → 全部回收，不留孤儿。"""
    s3 = _import_stage3()
    procs = [_spawn_sleeper() for _ in range(3)]
    try:
        assert all(p.poll() is None for p in procs), "前置：三个都应还活着"
        s3._reap_stream_pipeline(procs, logger=None, label="t09")
        for i, p in enumerate(procs):
            assert p.returncode is not None, f"第 {i} 个进程未被回收"
            assert p.poll() is not None, f"第 {i} 个进程仍存活"
    finally:
        for p in procs:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass


def test_T09_reap_tolerates_already_exited_and_none():
    """T09：已退出的、以及 None 占位（第 2/3 个没启动成功）都不能炸。"""
    s3 = _import_stage3()
    done = subprocess.Popen([sys.executable, "-c", "pass"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    done.wait(timeout=30)
    alive = _spawn_sleeper()
    try:
        s3._reap_stream_pipeline([done, None, alive], logger=None, label="t09b")
        assert done.returncode == 0, "已退出的进程不应被改变返回码"
        assert alive.poll() is not None, "活着的必须被回收"
    finally:
        try:
            if alive.poll() is None:
                alive.kill()
        except Exception:
            pass


def test_T09_reap_closes_pipe_handles():
    """T09：管道句柄必须关掉，否则父进程会漏 fd。"""
    s3 = _import_stage3()
    p = _spawn_sleeper(30)
    # 拿住引用再做断言：收尾后 .closed 应为 True
    stdout_handle, stderr_handle = p.stdout, p.stderr
    try:
        s3._reap_stream_pipeline([p], logger=None, label="t09c")
        assert stdout_handle.closed, "stdout 未关闭"
        assert stderr_handle.closed, "stderr 未关闭"
    finally:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass


def test_T09_reap_never_raises_on_broken_process_object():
    """T09：收尾阶段再抛异常只会掩盖真因 —— 必须静默容忍。"""
    s3 = _import_stage3()

    class Broken:
        pid = -1
        stdout = stderr = stdin = None

        def poll(self):
            raise RuntimeError("boom")

        def terminate(self):
            raise RuntimeError("boom")

    s3._reap_stream_pipeline([Broken()], logger=None, label="t09d")  # 不应抛


def test_T09_stream_pipeline_has_single_cleanup_path():
    """结构守卫：非 Stall 的异常不得在 except 里 raise 掉、绕过公共收尾。

    这正是 c3e5349 引入、审核 R17-05 点名的回归。
    """
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage3_ai_upscale.py"),
               encoding="utf-8", errors="replace").read()
    assert "_reap_stream_pipeline([dec, ai, enc]" in src, "必须有统一收尾"
    # 旧的"raise 后再也到不了下一个 except"写法必须消失
    assert "except Exception as _stall_exc:" not in src


# ------------------------------------------------ R17-05 进程组归属

def test_R17_05_may_killpg_refuses_own_group():
    """**核心安全约束**：子进程与调用方同组时，绝不能允许 killpg。

    原实现无条件 `os.killpg(os.getpgid(pid), SIGKILL)`。POSIX 下若子进程
    没建独立 session（原 spawn 在 POSIX 上 AttributeError 被吞掉），
    `getpgid(pid)` 就等于**调用方自己的组** —— 一条 killpg 会把调用方
    和它的兄弟进程一起杀掉。
    """
    from unittest import mock

    # 子进程与调用方同属 1000 组
    with mock.patch.object(cg, "_safe_getpgid",
                           lambda pid: 1000, create=True):
        assert cg._may_killpg(4242) is False


def test_R17_05_may_killpg_allows_owned_leader():
    """子进程是独立组组长（pgid == pid 且 != 我方组）才允许 killpg。"""
    from unittest import mock

    with mock.patch.object(
            cg, "_safe_getpgid",
            lambda pid: 4242 if pid else 1000, create=True):
        assert cg._may_killpg(4242) is True


def test_R17_05_may_killpg_refuses_when_unknown():
    """取不到进程组（Win32/权限不足）→ 保守退化为单杀。"""
    from unittest import mock

    with mock.patch.object(cg, "_safe_getpgid", lambda pid: None,
                           create=True):
        assert cg._may_killpg(4242) is False


def test_R17_05_may_killpg_refuses_non_leader():
    """子进程不是组长（pgid != pid）→ 不该按组杀。"""
    from unittest import mock

    with mock.patch.object(
            cg, "_safe_getpgid",
            lambda pid: 7777 if pid else 1000, create=True):
        assert cg._may_killpg(4242) is False


def test_R17_05_spawn_creates_own_group_on_posix():
    """POSIX 必须传 `start_new_session=True` —— 否则组归属判定根本不成立。"""
    src = open(os.path.join(_ROOT, "shared_modules", "cancel_guard.py"),
               encoding="utf-8", errors="replace").read()
    assert 'kwargs.setdefault("start_new_session", True)' in src, \
        "POSIX 上必须自建 session/进程组"
    # 不能再无条件 killpg
    assert "os.killpg(os.getpgid(pid), signal.SIGKILL)" not in src
