# -*- coding: utf-8 -*-
"""把多个 stage 进程串成一条管道，并**整组**负责它们的生死。

## 为什么需要（N27-04）

视频链现在是一段一段跑的：stage3 写 `step4_*.mkv`，stage5 读它、
再写 `step5_*.mkv`，stage6 读它。每一段都落一份**整片**中间媒体。

要"不落盘"，就得让 stage3 的 stdout 直接接 stage5 的 stdin、
stage5 的 stdout 接 stage6 —— 也就是**跨 stage 管道**。
`AGENTS.md` §3.2 明确允许（那是要求，不是"今天代码没这么用"）。

## 三条硬约束（来自 §3.2）

1. **整组必须能被整体回收**：失败/取消时 terminate → 有界等待 → kill →
   关句柄。**不能只杀第一个**，否则下游会读到半截数据还"成功退出"。
2. **不能中途 resume**：管道对没有"断点"，要么整组重启，要么放弃。
3 **stderr 各自落文件**：挂 `PIPE` 不读会在 64 KiB 处阻塞整条链。

## 成功判据

**所有**进程 `rc == 0`。任一非 0 → 整组失败（即便下游写出了字节）。
这与 PA03 是同一原则：`producer` 挂了、`consumer` 碰巧产出一帧，不算成功。
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import Any, Dict, List, Optional

__all__ = ["run_chain"]


def _terminate_group(procs: List[subprocess.Popen], *,
                     grace: float = 5.0) -> List[int]:
    """terminate → 有界等待 → kill。返回每组的最终 rc（尽力而为）。"""
    rcs: List[int] = []
    for p in procs:
        if p is None:
            rcs.append(-1)
            continue
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:
            pass
    deadline = time.monotonic() + grace
    for p in procs:
        if p is None:
            continue
        left = max(0.0, deadline - time.monotonic())
        try:
            p.wait(timeout=left)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
            try:
                p.wait(timeout=3.0)
            except Exception:
                pass
    for p in procs:
        try:
            rcs.append(p.returncode if p is not None else -1)
        except Exception:
            rcs.append(-1)
    # 关句柄（Windows 上不关会漏句柄，长跑会耗光）
    for p in procs:
        for stream in ("stdout", "stdin", "stderr"):
            try:
                s = getattr(p, stream, None)
                if s is not None:
                    s.close()
            except Exception:
                pass
    return rcs


def run_chain(cmds: List[List[str]], *, tmpd: str,
              logger=None, cancel_guard=None,
              timeout_s: Optional[float] = None,
              label: str = "chain") -> Dict[str, Any]:
    """把 `cmds` 串成一条管道：cmds[0] → cmds[1] → ... → cmds[-1]。

    Args:
        cmds: 每个元素是一个 argv 列表。第一个的 stdout 接第二个的 stdin，
            以此类推。**最后**一个的 stdout 归父进程（通常是 `None`，
            即写到哪里由它自己决定）。
        tmpd: stderr 落盘目录（每个进程一个 `{label}-{i}.err`）。
        timeout_s: 整组超时；超时后**整组**被杀。
        cancel_guard: 若提供，会把这些进程登记进去（以便外部取消）。

    Returns:
        ``{"ok", "rcs", "errors", "elapsed", "timeout", "killed"}``。

        ``ok`` 只有在**每个** rc == 0、且没超时、且进程数不少于 2 时为真。
    """
    out: Dict[str, Any] = {"ok": False, "rcs": [], "errors": [],
                           "elapsed": 0.0, "timeout": False,
                           "killed": False, "label": str(label)}
    if not isinstance(cmds, (list, tuple)) or len(cmds) < 2:
        out["errors"].append("need at least two commands to form a chain")
        return out

    t0 = time.monotonic()
    procs: List[Optional[subprocess.Popen]] = []
    err_files: List[Any] = []
    try:
        prev_stdout = None
        for i, argv in enumerate(cmds):
            if not argv:
                out["errors"].append("empty command at index %d" % i)
                raise RuntimeError("empty command")
            errp = os.path.join(tmpd, "%s-%d.err" % (label, i))
            fh = open(errp, "wb")
            err_files.append(fh)
            p = subprocess.Popen(
                [str(a) for a in argv],
                stdin=(prev_stdout if prev_stdout is not None
                       else subprocess.DEVNULL),
                stdout=subprocess.PIPE,
                stderr=fh,
                bufsize=10 ** 8,
                shell=False)
            # 让上一个进程的 stdout 与这个的 stdin 直接连通，父进程松手
            if prev_stdout is not None:
                try:
                    prev_stdout.close()
                except Exception:
                    pass
            procs.append(p)
            prev_stdout = p.stdout

        # 最后一个的 stdout 归父进程：除非调用方要，否则关掉（写到哪里
        # 由那条命令自己决定，通常是它自己的参数）。
        if prev_stdout is not None:
            try:
                prev_stdout.close()
            except Exception:
                pass

        if cancel_guard is not None:
            try:
                cancel_guard.register(
                    [p for p in procs if p is not None],
                    label=str(label))
            except Exception:
                pass

        # ---- 等待全部 ----
        rcs: List[int] = []
        for i, p in enumerate(procs):
            left = None
            if timeout_s:
                left = max(0.0, float(timeout_s) - (time.monotonic() - t0))
            try:
                p.wait(timeout=left if left is not None else None)
            except subprocess.TimeoutExpired:
                out["timeout"] = True
                out["killed"] = True
                _terminate_group([x for x in procs if x is not None])
                out["errors"].append(
                    "chain timed out after %.0fs; the whole group was killed"
                    % float(timeout_s or 0))
                out["rcs"] = [x.returncode for x in procs if x is not None]
                out["elapsed"] = round(time.monotonic() - t0, 3)
                return out
            rcs.append(p.returncode)

        out["rcs"] = rcs
        bad = [(i, rc) for i, rc in enumerate(rcs) if rc != 0]
        if bad:
            # 有失败 → 整组已经结束（管道会自然收尾），但仍要确认没有
            # 残留进程在写
            out["killed"] = True
            _terminate_group([x for x in procs if x is not None])
            out["errors"].append(
                "non-zero exit in the chain: %s"
                % ", ".join("%d:%s" % (i, rc) for i, rc in bad))
        out["ok"] = (not bad and not out["timeout"]
                     and len(procs) == len(cmds))
        return out
    except Exception as exc:
        out["errors"].append("%s: %s" % (type(exc).__name__, exc))
        out["killed"] = True
        _terminate_group([x for x in procs if x is not None])
        out["rcs"] = [x.returncode for x in procs if x is not None]
        return out
    finally:
        out["elapsed"] = round(time.monotonic() - t0, 3)
        for fh in err_files:
            try:
                fh.close()
            except Exception:
                pass
        if logger is not None:
            try:
                logger.info(
                    "STREAM_CHAIN_DONE",
                    "multi-stage pipe finished",
                    label=label, ok=out["ok"], rcs=out["rcs"],
                    timeout=out["timeout"], elapsed=out["elapsed"],
                    errors=out["errors"][:2] or None)
            except Exception:
                pass
