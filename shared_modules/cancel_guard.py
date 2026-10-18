"""NEW-V-06 · 取消/中断守卫（cancel guard）。

问题（本机实测，非推断）
----------------------
`Ctrl+C` / `SIGTERM` 触发的是 `KeyboardInterrupt`，它继承 `BaseException`
而**不是** `Exception`。本项目所有 stage 的兜底出口写的都是
`except Exception` —— 抓不到中断。实测 16 个 stage 脚本里
`KeyboardInterrupt` / `BaseException` / `signal.*` 的命中数为 **0**。

后果（三条，逐条对应交接的验收要求"故障和取消不会显示完成"）
----------------------------------------------------------
1. **半成品留在最终产物路径上**，且往往是非空的 —— 与"成功产物"在文件系统层面
   无法区分（`V07` 修好后靠 sidecar 才能区分，但 sidecar 此时还没写）。
2. **没有 status sidecar** —— 与"成功但缺 sidecar"、"失败残留"三者混淆。
   下游 stage 用 glob 选产物时可能选中这半截文件。
3. **审计与调度器看不到这个 job 被取消过** —— 既没有 DONE 也没有 FAILED，
   任务状态悬空。

修复
----
在 stage 主入口注册信号处理器。中断发生时：
  1. 删除已登记的半成品（最终产物 + 缓存/契约 sidecar）；
  2. 写 `status` sidecar = `FAILED`，`failed_operation = "cancelled"`，
     并记下具体信号名；
  3. 以退出码 **88（interrupted）** 结束进程 —— 落在项目约定的
     80–113 自定义退出码区间内（AGENTS.md §4.4），不与既有 80–87 冲突。

显式不做的事
------------
* 不追求"优雅地跑完当前步骤"：取消就是取消，宁可留 FAILED 也不留半成品。
* 不把取消包装成 `DEGRADED` —— 那会让"降级完成"这个词失去意义。
* 不吞掉 `SystemExit`：那是程序主动退出，不是取消。

用法
----
```python
from cancel_guard import install_cancel_guard, watch_outputs
install_cancel_guard(stage=5, logger=logger)
...
watch_outputs(output_path, extra_paths=[cache_meta_path], logger=logger)
```

Author: repair agent (2026-10-03) — see 修复清单.md NEW-V-06.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "EXIT_INTERRUPTED",
    "CANCELLED_OPERATION",
    "CancelGuard",
    "install_cancel_guard",
    "watch_outputs",
    "watch_process",
    "unwatch_process",
    "release_outputs",
    "run_managed",
    "guard",
]

# 退出码 88：被信号中断。项目自定义区间为 80–113（AGENTS.md §4.4），
# 既有 80 依赖缺失 / 81 参数 / 82 I/O / 83 GPU / 84 编码 / 85 权限 /
# 86 校验 / 87 资源耗尽；88 此前未使用。
EXIT_INTERRUPTED = 88

# sidecar 里的 failed_operation 取值；与"处理失败""校验失败"区分开
CANCELLED_OPERATION = "cancelled"

# 需要挂处理器的信号；SIGBREAK 仅 Windows 有
_SIGNAL_NAMES: List[str] = ["SIGINT", "SIGTERM", "SIGBREAK"]


def _signal_number(name: str) -> Optional[int]:
    """Return the numeric signal for `name`, or None when unsupported."""
    return getattr(signal, name, None)


class CancelGuard:
    """Register signal handlers that leave an honest FAILED record behind.

    The guard is deliberately a singleton per process: a stage script has one
    final artifact set, and nested handlers would race on cleanup.
    """

    def __init__(self) -> None:
        self._installed: bool = False
        self._fired: bool = False
        self._previous: Dict[int, Any] = {}
        self._output_path: Optional[str] = None
        self._extra_paths: List[str] = []
        self._logger: Any = None
        self._stage: Optional[int] = None
        # RV-12: 发布路径与"本轮临时产物"必须分开 —— 取消时只能删本轮的临时
        # 文件，绝不能顺手删掉先前已经发布过的有效成品。
        self._status_path: Optional[str] = None
        self._delete_output: bool = True
        # RV-12: 外部编码子进程。取消/超时必须把它一起收掉，否则会出现
        # "stage 已经退出，avifenc 还在往回写"的孤儿进程。
        self._processes: List[Any] = []
        self._released: bool = False
        # R3-03: 取消时必须先跑的收尾（例如成套发布回滚）
        self._cancel_hooks: List[Any] = []

    # ------------------------------------------------------------------ setup
    def install(self, stage: Optional[int] = None, logger: Any = None) -> bool:
        """Install signal handlers. Idempotent; safe to call more than once.

        Args:
            stage: Stage number recorded in the sidecar (e.g. ``5``).
            logger: Optional project logger used for a ``*_CANCELLED`` event.

        Returns:
            True when at least one handler was installed.
        """
        if logger is not None:
            self._logger = logger
        if stage is not None:
            self._stage = stage
        if self._installed:
            return True

        installed = 0
        for name in _SIGNAL_NAMES:
            signum = _signal_number(name)
            if signum is None:
                continue
            try:
                self._previous[signum] = signal.signal(signum, self._fire)
                installed += 1
            except (ValueError, OSError, RuntimeError):
                # 非主线程 / 平台不支持：属于可接受的降级，不阻断 stage
                continue
        self._installed = installed > 0
        return self._installed

    def uninstall(self) -> None:
        """Restore previous handlers. Intended for tests."""
        for signum, handler in self._previous.items():
            if handler is not None:
                try:
                    signal.signal(signum, handler)
                except (ValueError, OSError, RuntimeError):
                    pass
        self._previous.clear()
        self._installed = False

    def watch(
        self,
        output_path: str,
        extra_paths: Optional[List[str]] = None,
        logger: Any = None,
        stage: Optional[int] = None,
        delete_output: bool = True,
        status_path: Optional[str] = None,
    ) -> None:
        """Register the artifacts that must be discarded on cancellation.

        Call this as soon as the final artifact path is known — before any
        bytes are written — otherwise a first-run cancellation has nothing
        to clean up.

        RV-12 additions:
        * ``delete_output=False`` keeps the published artifact (it may hold a
          previously completed result) while still deleting the temp files
          listed in ``extra_paths``.
        * ``status_path`` writes the FAILED record next to the *published*
          artifact even when publishing happens through a temp file.
        """
        self._output_path = str(output_path)
        self._extra_paths = [str(p) for p in (extra_paths or [])]
        self._delete_output = bool(delete_output)
        self._status_path = str(status_path) if status_path else None
        self._released = False
        if logger is not None:
            self._logger = logger
        if stage is not None:
            self._stage = stage

    def watch_process(self, proc: Any) -> None:
        """RV-12: register a child process that must be killed on cancellation.

        Encoders (avifenc/ffmpeg) are separate processes; killing only the
        Python stage leaves them running and able to keep writing files.
        """
        if proc is not None:
            self._processes.append(proc)

    def unwatch_process(self, proc: Any) -> None:
        """Stop tracking a finished child process."""
        try:
            self._processes.remove(proc)
        except (ValueError, BaseException):  # noqa: BLE001
            pass

    def release(self) -> None:
        """RV-12: artifacts are published successfully — stop guarding them.

        Called after a transactional publish so a later interrupt cannot delete
        a finished product.
        """
        self._released = True
        self._extra_paths = []
        self._processes = []
        # R3-03: 提交已完成，取消钩子（发布回滚）不再需要。
        self._cancel_hooks = []

    # ------------------------------------------------------- cancel hooks
    def add_cancel_hook(self, hook) -> None:
        """R3-03: register a callback that must run **before** cleanup.

        发布临界区（AVIF 已替换、TIFF 还没换完）被中断时，
        单纯"删掉半成品"会留下 **新 AVIF + 旧 TIFF** 的混合代（实测 B14）。
        这类收尾必须能执行**成套回滚**，因此挂在取消路径上，
        且在 `_fire` 的 `os._exit` 之前运行（普通 try/finally 绕不过它）。
        """
        if callable(hook):
            self._cancel_hooks.append(hook)

    def _run_cancel_hooks(self) -> None:
        for hook in list(self._cancel_hooks):
            try:
                hook()
            except BaseException:  # noqa: BLE001 - 处理器内不允许再抛出
                continue

    # ------------------------------------------------------------- behaviour
    def _fire(self, signum: int, frame: Any) -> None:
        """Signal handler: record the cancellation, then leave the process.

        本机实测（Windows）：**在信号处理器内部做文件 I/O 会再抛出一个
        `KeyboardInterrupt`**（待处理的中断标志在 I/O 释放 GIL 时被处理）。
        它是 `BaseException`，`except Exception` 抓不到 —— 结果是清理半途而废、
        进程以退出码 1 结束，而不是我们登记的 88。
        所以这里：清理用 `except BaseException` 兜住，退出码无条件生效。
        """
        if self._fired:
            return
        self._fired = True
        try:
            self.cleanup_and_record(signum)
        except BaseException:  # noqa: BLE001 - 处理器内不允许再抛出
            pass
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except BaseException:  # noqa: BLE001 - 管道可能已断开
            pass
        os._exit(EXIT_INTERRUPTED)

    def cleanup_and_record(self, signum: Optional[int] = None) -> Dict[str, Any]:
        """Discard partial artifacts and write a FAILED sidecar.

        Separated from :meth:`_fire` so tests can exercise the logic without
        killing the interpreter. Every step is individually guarded: a
        cancellation must never leave the process in a worse state than the
        interruption itself.

        Args:
            signum: Signal number, used only for the recorded signal name.

        Returns:
            A dict describing what was removed and which sidecar was written.
        """
        removed: List[str] = []
        status_path: Optional[str] = None

        # R2-01 / A13：已发布（release）之后，任何清理都不得再动产物。
        # 旧实现 release 只清 _extra_paths/_processes，_output_path 仍在 targets
        # 里 —— 实测 `watch → release → cleanup_and_record` 之后已完成的文件
        # 被删除（A13）。发布是不可逆的成功事件，后续取消/清理必须直接退出。
        if self._released:
            if self._logger is not None:
                try:
                    self._logger.info(
                        f"STAGE{self._stage if self._stage is not None else ''}"
                        "_CLEANUP_SKIPPED",
                        "Artifacts were already published (release called); "
                        "cancellation cleanup skipped",
                        output=self._output_path or "")
                except BaseException:  # noqa: BLE001
                    pass
            return {"removed": [], "status_path": None, "released": True}

        # RV-12: 先收子进程再删文件 —— 否则被杀掉的 stage 之外，编码器子进程
        # 可能在我们清理完之后继续往产物路径写回。
        for proc in list(self._processes):
            try:
                poll = getattr(proc, "poll", None)
                if poll is not None and poll() is None:
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except BaseException:  # noqa: BLE001
                        pass
            except BaseException:  # noqa: BLE001
                continue
        self._processes = []

        # R3-03 / B14：取消发生在发布临界区时，**先成套回滚**再清理，
        # 否则会留下"新 AVIF + 旧 TIFF"的混合代。钩子必须在
        # `os._exit` 之前跑，所以放在这里而不是普通 try/finally。
        if not self._released:
            self._run_cancel_hooks()

        targets: List[str] = []
        if self._output_path and self._delete_output:
            targets.append(self._output_path)
        targets.extend(self._extra_paths)

        for path in targets:
            if not path:
                continue
            try:
                if os.path.exists(path):
                    os.remove(path)
                    removed.append(path)
            except BaseException:  # noqa: BLE001 - 处理器内 I/O 会抛 KBInterrupt
                continue

        status_path = self._status_path or (
            f"{self._output_path}.status.json" if self._output_path else None)
        if status_path:
            payload = {
                "status": "FAILED",
                "failed_operation": CANCELLED_OPERATION,
                "stage": self._stage,
                "reason": (
                    f"interrupted by {self._signal_name(signum)}"
                    if signum is not None
                    else "interrupted"
                ),
                "exit_code": EXIT_INTERRUPTED,
            }
            self._write_json(status_path, payload)

        if self._logger is not None:
            try:
                self._logger.error(
                    f"STAGE{self._stage if self._stage is not None else ''}_CANCELLED",
                    "Stage cancelled by user/system signal; partial output "
                    "discarded and a FAILED record written",
                    output=self._output_path or "",
                    removed=",".join(removed),
                )
            except BaseException:  # noqa: BLE001 - 日志失败不影响退出码
                pass

        return {
            "removed": removed,
            "status_path": status_path,
            "exit_code": EXIT_INTERRUPTED,
        }

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _signal_name(signum: Optional[int]) -> str:
        """Best-effort signal name for the record."""
        if signum is None:
            return "unknown"
        for name in _SIGNAL_NAMES:
            if _signal_number(name) == signum:
                return name
        return str(signum)

    @staticmethod
    def _write_json(path: str, payload: Dict[str, Any]) -> bool:
        """Write JSON atomically-ish (temp file + replace). Never raises.

        用 `BaseException` 兜底而不是 `Exception`：信号处理器内的 I/O 会再抛
        `KeyboardInterrupt`（见 :meth:`_fire` 的实测说明）。
        """
        tmp = f"{path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False,
                          sort_keys=True, indent=2)
            os.replace(tmp, path)
            return True
        except BaseException:  # noqa: BLE001 - 处理器内不允许再抛出
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except BaseException:  # noqa: BLE001
                pass
            return False


#: Process-wide guard instance.
guard = CancelGuard()  # spawn-safe: 构造仅初始化属性，无信号注册/线程/IO


def install_cancel_guard(stage: Optional[int] = None,
                         logger: Any = None) -> bool:
    """Install the process-wide cancel guard. See :meth:`CancelGuard.install`."""
    return guard.install(stage=stage, logger=logger)


def watch_outputs(output_path: str,
                  extra_paths: Optional[List[str]] = None,
                  logger: Any = None,
                  stage: Optional[int] = None,
                  delete_output: bool = True,
                  status_path: Optional[str] = None) -> None:
    """Register artifacts for the process-wide guard. See :meth:`CancelGuard.watch`."""
    guard.watch(output_path, extra_paths=extra_paths, logger=logger,
                stage=stage, delete_output=delete_output,
                status_path=status_path)


def watch_process(proc: Any) -> None:
    """RV-12: register a child process to be killed when cancellation fires."""
    guard.watch_process(proc)


def unwatch_process(proc: Any) -> None:
    """Stop tracking a child process that finished normally."""
    guard.unwatch_process(proc)


def add_cancel_hook(hook) -> None:
    """R3-03: 注册"取消时必须先跑"的收尾（见 :meth:`CancelGuard.add_cancel_hook`）。

    发布临界区被中断时要成套回滚，而取消处理器最后会 `os._exit`，
    普通 try/finally 跑不到 —— 只能通过这里注册。
    """
    guard.add_cancel_hook(hook)


def clear_cancel_hooks() -> None:
    """R3-03: 提交成功后清除回滚钩子（产物已发布，不能再回滚）。"""
    guard._cancel_hooks = []


def release_outputs() -> None:
    """RV-12: artifacts are published; stop guarding them."""
    guard.release()


def _kill_quietly(proc: Any) -> None:
    """Kill a child process and reap it; never raises."""
    try:
        poll = getattr(proc, "poll", None)
        if poll is not None and poll() is None:
            proc.kill()
    except BaseException:  # noqa: BLE001
        pass
    try:
        proc.wait(timeout=10)
    except BaseException:  # noqa: BLE001
        pass


def run_managed(cmd, timeout=None, logger=None, **kwargs):
    """Run a child process under cancel-guard supervision.

    ``subprocess.run`` cannot be interrupted: a cancellation kills only the
    Python stage while the child (ffmpeg/vspipe/avifenc...) keeps writing —
    the guard would delete the partial artifact and the orphan would recreate
    it (RV-12). Here the child is registered with the guard, so cancellation
    kills it *before* the artifact cleanup runs.

    Semantics deliberately mirror ``subprocess.run`` so existing call sites
    behave identically:

    * success  -> ``CompletedProcess`` with decoded stdout/stderr when
      ``text=True`` was passed;
    * timeout  -> child killed and reaped, then ``TimeoutExpired`` is
      **re-raised** (call sites' ``except subprocess.TimeoutExpired`` keeps
      working);
    * ``BaseException`` (e.g. ``KeyboardInterrupt`` when the guard is not
      installed) -> child killed and reaped, then re-raised.
    """
    import subprocess

    capture = kwargs.pop("capture_output", False)
    text = kwargs.pop("text", False)
    if capture:
        kwargs.setdefault("stdout", subprocess.PIPE)
        kwargs.setdefault("stderr", subprocess.PIPE)
    kwargs.setdefault("shell", False)
    try:  # Windows: 独立进程组，便于 CTRL_BREAK_EVENT
        kwargs.setdefault("creationflags",
                          subprocess.CREATE_NEW_PROCESS_GROUP)
    except Exception:  # pragma: no cover - 非 Windows 平台无此属性
        pass

    proc = subprocess.Popen(cmd, **kwargs)
    guard.watch_process(proc)
    try:
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_quietly(proc)
            raise
        returncode = proc.returncode
    except BaseException:
        # 取消/中断/超时都已先收掉子进程，再按原语义向上传递
        _kill_quietly(proc)
        raise
    finally:
        guard.unwatch_process(proc)

    if text:
        out = (out or b"").decode("utf-8", "replace")
        err = (err or b"").decode("utf-8", "replace")
    return subprocess.CompletedProcess(cmd, returncode, out, err)
