# -*- coding: utf-8 -*-
"""G11：把一次运行的**磁盘写入按类型分开记账**。

## 为什么不能只报一个总数

审核与任务书都要求分清这几类，因为它们**不是一回事**：

| 类型 | 含义 | 能不能省 |
|---|---|---|
| `source_copy` | 复制/预切源素材 | 主路线要求"只读 source"，这一项应当接近 0 |
| `intermediate_media` | stage 之间的像素/编码媒体 | 融合能省一部分；省不掉的是"真的改了像素" |
| `final_candidate` | 本次 attempt 的暂存成品 | 必须写（原子发布的前提） |
| `old_generation_backup` | 发布事务为旧代做的备份 | 必须写，且发布后应回收 |
| `index_report` | 清单/报告/回执/侧车 | 小，但数量多，容易失控 |
| `tool_cache` | 工具自己的缓存（ffindex / 模型 cache） | 不该混进"我们的写入" |

一个"总写入 3 GB"的数字，既可能是"源被复制了一遍"，也可能是"六代中间产物
都没清" —— 两者要修的地方完全不同。所以必须分类。

## 两种取数方式（都要有，因为各自有盲区）

1. **`scan()`**：运行结束后扫目录，按名字/扩展名归类**现存**文件。
   盲区：已经被清理掉的中间产物看不到。
2. **`record()`**：调用方在自己知道的时刻记账（清理前、发布后）。
   盲区：需要有人在正确的地方调用。

两者相加才是"这次到底写了多少"。本模块把两个数**分开呈现**，不合并成
一个容易误读的数 —— 合并会让人以为"扫描到的不见了 = 没写过"。

## 边界

只统计**本任务拥有的目录**（temp / output）。不去扫全机、不读未授权目录。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

#: 分类名（顺序固定，便于报告排版）。
CATEGORIES = (
    "source_copy",
    "intermediate_media",
    "final_candidate",
    "old_generation_backup",
    "index_report",
    "tool_cache",
)

#: 纯报告的扩展名 —— 数量多但单文件小。
_REPORT_EXT = frozenset({
    ".json", ".jsonl", ".log", ".txt", ".yaml", ".yml", ".md", ".csv",
})

#: 工具缓存的特征（不属于"我们的写入"）。
_TOOL_CACHE_HINTS = (".ffindex", ".cache", ".tmp.partial")


def _ext(path: str) -> str:
    return os.path.splitext(str(path))[1].lower()


def classify(path: str, *, output_path: str = "",
             temp_dir: str = "") -> str:
    """把一个路径归到某类。**宁可归到 intermediate，也不归到 report**。"""
    name = os.path.basename(str(path))
    ext = _ext(path)
    low = name.lower()
    if any(low.endswith(h) or h in low for h in _TOOL_CACHE_HINTS):
        return "tool_cache"
    if ext in _REPORT_EXT:
        # 侧车（`<media>.identity.json` / `.verify.json` / `.done.json` …）
        # 都归 report：它们不是媒体，但**必须**存在。
        return "index_report"
    if output_path and os.path.abspath(str(path)) == os.path.abspath(
            str(output_path)):
        return "final_candidate"
    if "attempt-" in low and "_restored" in low:
        return "final_candidate"
    if "publish_backup" in str(path).replace("\\", "/").lower() or \
            ".bak" in low:
        return "old_generation_backup"
    if low.endswith((".mkv", ".mp4", ".mov", ".avi", ".mxf", ".vob",
                     ".m2ts", ".ts", ".webm",
                     ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".avif",
                     ".webp", ".bmp", ".dpx", ".exr")):
        return "intermediate_media"
    # 兜底方向很重要：归到 `index_report` 会把"未知的媒体字节"洗成"小报告"，
    # 于是"我们到底写了多少媒体"被系统性低估 —— 而那正是要盯的数。
    # 反过来归到 media 只会让报告类偏大，容易发现、不影响判断。
    return "intermediate_media"


def scan(paths: Iterable[str], *, output_path: str = "",
         temp_dir: str = "") -> Dict[str, Any]:
    """按类统计**现存**文件。

    Returns:
        ``{"by_category": {cat: {"files": n, "bytes": n}}, "total": {...},
        "unreadable": n}``
    """
    by: Dict[str, Dict[str, int]] = {
        c: {"files": 0, "bytes": 0} for c in CATEGORIES}
    unreadable = 0
    seen = set()
    for root in paths:
        if not root or not os.path.exists(str(root)):
            continue
        stack: List[str] = [str(root)]
        while stack:
            cur = stack.pop()
            try:
                entries = list(os.scandir(cur))
            except OSError:
                unreadable += 1
                continue
            for e in entries:
                try:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                        continue
                    real = os.path.abspath(e.path)
                    if real in seen:
                        continue
                    seen.add(real)
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    unreadable += 1
                    continue
                cat = classify(e.path, output_path=output_path,
                               temp_dir=temp_dir)
                by[cat]["files"] += 1
                by[cat]["bytes"] += int(st.st_size)
    total = {"files": sum(v["files"] for v in by.values()),
             "bytes": sum(v["bytes"] for v in by.values())}
    return {"by_category": by, "total": total, "unreadable": unreadable}


class WriteLedger:
    """运行期累加器：调用方在**自己知道**的时刻记账。

    `record()` 记的是"发生了的写入"；`scan()` 记的是"还剩下的"。两者
    分开报，不合并。
    """

    def __init__(self) -> None:
        self._by: Dict[str, Dict[str, int]] = {
            c: {"events": 0, "bytes": 0} for c in CATEGORIES}
        self._unknown: List[str] = []
        #: 按对象角色的小计（media/verify/marker/...）
        self._by_role: Dict[str, Dict[str, int]] = {}
        #: 覆盖写次数（同一次写入不计两次存量，但覆盖是事实）
        self._overwrites = 0
        #: 目录快照看到的**存量增量**（只用于泄漏检查，不是写入量）
        self._present_delta = 0

    def record(self, category: str, nbytes: Optional[int] = None,
               *, path: str = "") -> None:
        if category not in self._by:
            self._unknown.append(f"{category}:{path or nbytes}")
            return
        if nbytes is None:
            try:
                nbytes = os.path.getsize(path)
            except OSError:
                nbytes = 0
        self._by[category]["events"] += 1
        self._by[category]["bytes"] += int(nbytes or 0)

    def record_path(self, path: str, *, output_path: str = "") -> str:
        cat = classify(path, output_path=output_path)
        self.record(cat, path=path)
        return cat

    def record_write(self, category: str, nbytes: int, *,
                     role: str = "", path: str = "",
                     overwrote: bool = False) -> None:
        """记一次**真实发生的成功写入**（R26-A07）。

        与 `record_path()` 的区别：`record_path` 会去 `os.path.getsize`
        问"现在多大"，而"现在多大"回答不了这五个问题 —— 复核给的精确反例：

        | 行为 | 实际写入 | 目录差法 |
        |---|---:|---:|
        | 新建并保留 | 1024 | 1024 ✓ |
        | 同一 stage 内新建后删除 | 1024 | **0** |
        | 仅删除旧文件 | 0 | **1024** |
        | 等长覆盖 | 1024 | **0** |
        | 截断并写 512 | 512 | **0** |

        所以**写入量必须由 writer 在写成功的那一刻上报**，不能事后从
        目录状态推断。删除只影响"存量"，不增加 `written`（这里不做任何
        减法：删除不该调用本方法）。

        Args:
            category: 六类之一（见 `CATEGORIES`）。
            nbytes: **本次成功写出的字节数**。覆盖写要传新写的量，
                不是"文件最终大小"。
            role: 对象角色（`media`/`verify`/`marker`/...），供按 attempt
                归属统计。
            path: 目标路径（只作审计，不参与计量）。
            overwrote: 是否覆盖了已有对象（同一次写入不重复计"存量"）。
        """
        if category not in self._by:
            self._unknown.append("%s:%s" % (category, path or nbytes))
            return
        n = int(nbytes or 0)
        if n < 0:
            self._unknown.append("%s:negative(%s)" % (category, n))
            return
        self._by[category]["events"] += 1
        self._by[category]["bytes"] += n
        # 逐 role 的小计：便于回答"最终成品写了多少 / 备份写了多少"
        if role:
            _r = self._by_role.setdefault(role, {"events": 0, "bytes": 0})
            _r["events"] += 1
            _r["bytes"] += n
        if overwrote:
            self._overwrites += 1

    def observe_present_delta(self, nbytes: int) -> None:
        """记一次**目录存量增量**（不是写入量）。

        R26-A07：目录快照回答不了"写过多少"（5 种行为错 4 种），
        但它仍有一个正当用途：**发现泄漏**（中间件该被清理却还在）。
        所以它单独记，**不混进 `by_category` 的 `bytes`**，避免
        "看起来写了很多"和"实际剩了很多"互相污染。
        """
        n = int(nbytes or 0)
        self._present_delta += max(0, n)

    def summary(self) -> Dict[str, Any]:
        return {
            "by_category": {k: dict(v) for k, v in self._by.items()},
            "by_role": {k: dict(v) for k, v in self._by_role.items()},
            "overwrites": int(self._overwrites),
            "present_delta": int(self._present_delta),
            "unknown_categories": list(self._unknown),
            # 语义说明：这里是**累计成功写入**，不是目录存量。
            # 存量另由 `scan()` 给，两者不能互相替代。
            "semantics": "cumulative successful writes (not directory state)",
        }


def render(scan_result: Dict[str, Any],
           ledger_result: Optional[Dict[str, Any]] = None) -> List[str]:
    """把两个数摊成一张可读的表（日志与报告共用一份排版）。"""
    lines = ["write ledger (bytes)"]
    by = (scan_result or {}).get("by_category") or {}
    led = ((ledger_result or {}).get("by_category") or {})
    for cat in CATEGORIES:
        s = by.get(cat) or {}
        l = led.get(cat) or {}
        lines.append(
            "  %-22s present=%12d B (%d files)   written=%12d B (%d events)"
            % (cat, int(s.get("bytes") or 0), int(s.get("files") or 0),
               int(l.get("bytes") or 0), int(l.get("events") or 0)))
    tot = (scan_result or {}).get("total") or {}
    lines.append("  %-22s present=%12d B (%d files)"
                 % ("TOTAL", int(tot.get("bytes") or 0),
                    int(tot.get("files") or 0)))
    unread = int((scan_result or {}).get("unreadable") or 0)
    if unread:
        lines.append(f"  unreadable entries: {unread} (reported, not hidden)")
    unk = (ledger_result or {}).get("unknown_categories") or []
    if unk:
        lines.append(f"  unknown categories dropped: {unk[:5]} "
                     "(a bucket we do not know is NOT merged into another)")
    return lines
