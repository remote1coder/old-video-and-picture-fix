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
    ) -> None:
        """Register the artifacts that must be discarded on cancellation.

        Call this as soon as the final artifact path is known — before any
        bytes are written — otherwise a first-run cancellation has nothing
        to clean up.
        """
        self._output_path = str(output_path)
        self._extra_paths = [str(p) for p in (extra_paths or [])]
        if logger is not None:
            self._logger = logger
        if stage is not None:
            self._stage = stage

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

        for path in [self._output_path] + list(self._extra_paths):
            if not path:
                continue
            try:
                if os.path.exists(path):
                    os.remove(path)
                    removed.append(path)
            except BaseException:  # noqa: BLE001 - 处理器内 I/O 会抛 KBInterrupt
                continue

        if self._output_path:
            status_path = f"{self._output_path}.status.json"
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
guard = CancelGuard()


def install_cancel_guard(stage: Optional[int] = None,
                         logger: Any = None) -> bool:
    """Install the process-wide cancel guard. See :meth:`CancelGuard.install`."""
    return guard.install(stage=stage, logger=logger)


def watch_outputs(output_path: str,
                  extra_paths: Optional[List[str]] = None,
                  logger: Any = None,
                  stage: Optional[int] = None) -> None:
    """Register artifacts for the process-wide guard."""
    guard.watch(output_path, extra_paths=extra_paths,
                logger=logger, stage=stage)
