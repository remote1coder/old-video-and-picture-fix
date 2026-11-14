# -*- coding: utf-8 -*-
"""整片级操作必须走进度助手，否则会被卡死看门狗误杀。

## 这条测试防的是什么

看门狗（`run_stage` 的 `progress_watch`）按"temp 字节 + 日志大小"判进展。
`verify_output()` / `check_declared_timeline_mapping()` / `_probe_pts()`
都会**整片解码**，几分钟内既不写文件也不打日志 —— 明明在干活却被判 stall
（rc=87）。长片实测踩了 5 次，其中两次都在 stage7 的发布段：

* `STAGE7_FULLCHAIN_VERIFY` 之后的那次整片比对；
* **发布前** `_vo_pre(...)` —— 同一对文件第二次整片解码，5 分 28 秒后被杀。

所以规则是：stage7 里这些调用必须包在 `_run_with_progress(...)` 里。
这不是"加心跳"，是让耗时段落自报进度；真卡住照样会被杀
（判据从"300 秒没字节"变成"300 秒没日志"）。

## 为什么用窗口判断而不是 AST

调用常写成 `lambda: f(...)` 或跨行的 `_run_with_progress(`，AST 也能做，
但这里要的是"便宜且不会被重构搞坏"。窗口判定足够，且失败信息直接给行号。
"""
import io
import os

import conftest  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_S7 = os.path.join(ROOT, "video_restoration_pipeline", "stages",
                   "stage7_encode.py")

#: 这些函数会整片解码/扫描，必须走进度助手
_HEAVY = ("verify_output(", "check_declared_timeline_mapping(", "_probe_pts(")
#: 出现这些字样的行不算调用（定义、import、注释、文档）
_SKIP_HINTS = ("def ", "import ", "from ", "#", '"""', "`")


def _lines():
    with io.open(_S7, encoding="utf-8", errors="replace") as f:
        return f.read().split("\n")


def _wrapped_function_span(lines):
    """`check_declared_timeline_mapping` 的函数体范围（含首尾）。

    它内部会调 `_probe_pts()`，但**所有调用点**都已经包在
    `_run_with_progress` 里了（那才是看门狗能看到的地方）。
    函数体内的调用不需要再单独包一层，否则就是同一个操作报两次进度。
    """
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("def check_declared_timeline_mapping("):
            start = i
            continue
        if start is not None and ln.startswith("def "):
            return (start, i - 1)
    return (start, len(lines) - 1) if start is not None else (0, -1)


def test_heavy_calls_in_stage7_are_wrapped_with_progress():
    lines = _lines()
    _lo, _hi = _wrapped_function_span(lines)
    offenders = []
    for i, ln in enumerate(lines):
        if _lo <= i <= _hi:
            continue          # 见 _wrapped_function_span 的说明
        for fn in _HEAVY:
            if fn not in ln:
                continue
            stripped = ln.strip()
            if any(h in stripped for h in _SKIP_HINTS):
                continue
            # 允许 4 行前到本行的窗口里出现助手调用（lambda 常跨行）
            window = "\n".join(lines[max(0, i - 4):i + 2])
            if "_run_with_progress(" in window:
                continue
            offenders.append((i + 1, stripped[:80]))
    assert not offenders, (
        "stage7 里这些整片级调用没有走进度助手，长片会被 300s 判 stall：\n"
        + "\n".join(f"  line {n}: {t}" for n, t in offenders))


def test_the_helper_exists_and_logs_while_running():
    """助手必须真的在"还在跑"时打日志，而不是跑完才打。"""
    src = "\n".join(_lines())
    i = src.find("def _run_with_progress(")
    assert i > 0, "缺少 _run_with_progress"
    body = src[i:i + 2000]
    assert "while t.is_alive()" in body, "必须轮询存活状态"
    assert "STAGE7_LONG_OP_PROGRESS" in body, "必须有可被看门狗看到的事件名"
