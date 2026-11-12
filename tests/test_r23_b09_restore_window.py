# -*- coding: utf-8 -*-
"""R23-B09（恢复窗口）：复制备份期间目标被外部写入 → 停手保留现场。

## 审核给的反例

在"备份复制结束、`os.replace` 之前"注入陌生内容，`recover` 随后把它
覆盖成 OLD，并报告 `rolled_back`。

修复后：复制完**再核一次目标**，不一致就删掉临时件、保留现场、
整代判 `recovery_required`。

**边界声明**：这不是"断电测试"，也不承诺"不存在跨进程任意写入"。
遵守协议的两个进程本来就被目标锁互斥；这里防的是**不遵守协议的外部写入**，
并把它变成"停下 + 留证据"而不是"覆盖 + 假成功"。
"""
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import publisher as P  # noqa: E402
import test_r22_a09_recovery_concurrency as CT  # noqa: E402


def test_a_foreign_write_during_the_restore_copy_is_preserved(tmp_path):
    ctx = CT._unresolved(tmp_path)
    real = P.shutil.copyfile
    hits = []

    def copying(src, dst, *a, **kw):
        r = real(src, dst, *a, **kw)
        if str(src) == ctx["bk"]:
            # 备份已经复制完，目标被"别人"改了
            Path(ctx["out"]).write_bytes(b"FOREIGN-WRITER-CONTENT")
            hits.append(str(dst))
        return r

    with patch.object(P.shutil, "copyfile", side_effect=copying):
        rep = CT._recover(ctx)

    assert hits, "注入点没有触发，这条用例没有测到目标场景"
    assert Path(ctx["out"]).read_text() == "FOREIGN-WRITER-CONTENT", (
        "陌生内容被覆盖了 —— 复制窗口仍然存在")
    assert rep["recovery_required"] is True
    assert rep["fully_restored"] is False
    reasons = [f.get("reason", "") for f in rep.get("failed") or []]
    assert any("during the restore copy" in r for r in reasons), reasons


def test_a_normal_restore_still_works(tmp_path):
    """反面对照：没人捣乱时必须照常恢复（不能修成永远 recovery_required）。"""
    ctx = CT._unresolved(tmp_path)
    rep = CT._recover(ctx)
    assert Path(ctx["out"]).read_text() == "OLD"
    assert rep["fully_restored"] is True
    assert rep["recovery_required"] is False
