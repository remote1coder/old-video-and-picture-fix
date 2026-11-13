# -*- coding: utf-8 -*-
"""结构化阶段事件（B08 / R25 A09 的"attempt/PID 注入"前置）。

## 为什么需要它

审核指出：现有故障注入靠"temp 里出现 step1/step3 文件"触发，
而那种触发**无法区分本次执行与上一轮残留** —— 旧文件还在，
注入器就会以为"到阶段了"，于是：

* 可能杀早了（本阶段还没真开始）；
* 可能杀错进程（PID 复用）；
* 重跑时被旧残留再次触发，测出来的"故障恢复"是假的。

审核要求：事件至少含 **job / plan / attempt、PID 及创建时间、
阶段/边界、事件序号**，注入器必须等**本次**事件。

## 设计

`emit(logger, stage, phase, params, **extra)` 写一条 `STAGE_EVENT`：

```json
{"event": "STAGE_EVENT", "stage": 3, "phase": "start",
 "attempt_id": "3e6d943a…", "pid": 12345, "process_started_at": 1699999999.12,
 "seq": 1, "job_id": "job_001", ...}
```

* `pid` 由 logger 统一注入（见 `logger.log`）；
* `process_started_at` 取本进程的创建时间（`psutil` 不可用时退回
  模块导入时刻 + `/proc` 不可用时用 `time.time() - uptime` 的近似），
  用来识别 PID 复用；
* `seq` 是本进程内递增序号，配合 attempt 就能唯一确定"第几个事件"。

**不改变任何业务逻辑** —— 只是多写几行日志。
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

#: 本进程的创建时刻（近似）。用于识别 PID 复用。
_PROCESS_STARTED_AT = time.time()

_SEQ = 0


def process_started_at() -> float:
    """本进程的创建时刻（尽可能精确）。

    Windows 上退化为"模块导入时刻"，因为标准库没有直接接口；
    这足以识别"同一个 PID 但明显是后来的进程"（差值远大于模块加载时间）。
    """
    global _PROCESS_STARTED_AT
    try:                                    # Linux / WSL
        with open("/proc/self/stat", "r", encoding="utf-8") as f:
            fields = f.read().split()
        # field 22（1-based）= starttime，单位 clock ticks
        _ticks = int(fields[21])
        _hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        with open("/proc/uptime", "r", encoding="utf-8") as f:
            _uptime = float(f.read().split()[0])
        _PROCESS_STARTED_AT = time.time() - _uptime + _ticks / float(_hz)
    except Exception:
        pass
    return _PROCESS_STARTED_AT


def emit(logger, stage: int, phase: str,
         params: Optional[Dict[str, Any]] = None,
         **extra: Any) -> None:
    """写一条 `STAGE_EVENT`。**永不抛异常**（日志不能弄坏主流程）。"""
    global _SEQ
    try:
        _SEQ += 1
        p = params or {}
        if logger is None:
            return
        logger.info(
            "STAGE_EVENT",
            "stage %s %s" % (stage, phase),
            stage=int(stage),
            phase=str(phase),
            attempt_id=str(p.get("attempt_id") or ""),
            job_id=str(p.get("job_id") or ""),
            process_started_at=round(process_started_at(), 3),
            seq=_SEQ,
            **extra)
    except Exception:
        pass


def seq() -> int:
    """当前进程内已发出的事件数（诊断用）。"""
    return _SEQ
