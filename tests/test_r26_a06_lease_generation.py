# -*- coding: utf-8 -*-
"""槽位代际与隔离（R26-A06）。

复核的复现：新 lease 写入 `NEW!` 后，**旧 lease 保留的 bytearray 仍能把它
改成 `BAD!`**，统计依旧 `balanced`。

## 本文件要钉住的三件事

1. **代际**让旧 lease 可被识别为 stale（`is_current() == False`）；
2. **隔离**让"无法确认旧 owner 已停止写"的槽**不再分配**；
3. **诚实承认**：Python 的裸 `bytearray` 引用**无法被撤销** ——
   第 3 条测试**故意**用旧句柄写新帧并断言它**成功**了。
   这不是"没修好"，而是把局限写成可执行的证据，防止有人把
   "加了 seq 校验"当成"内存安全"。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared_modules"))

from lease import LeasePool  # noqa: E402


def test_generation_invalidates_the_old_lease():
    p = LeasePool(capacity=1, slot_bytes=8, name="a06")
    old = p.acquire("old", 1)
    assert p.is_current(old) is True
    p.release(old.lease_id, 1)
    new = p.acquire("new", 2)
    assert new.generation > old.generation
    assert p.is_current(new) is True
    assert p.is_current(old) is False, "旧 lease 必须被判过期"


def test_quarantine_retires_the_slot():
    """隔离后的槽不再分配 —— 异常路径不能把槽交给新帧。"""
    p = LeasePool(capacity=3, slot_bytes=8, name="q")
    a = p.acquire("x", 1)
    assert p.quarantine(a.lease_id, "owner may still be writing") is True
    assert p.quarantined_slots() == [a.slot]
    b = p.acquire("y", 2)
    assert b.slot != a.slot, "隔离槽被复用了"
    # 隔离不影响另一个槽
    c = p.acquire("z", 3)
    assert c.slot not in (a.slot, b.slot)


def test_bare_reference_still_writes_the_new_frame():
    """**局限的显式证据**：纯 Python 挡不住裸引用。

    旧 lease 的 `buffer` 与新 lease 的是**同一个对象**（预分配复用），
    所以用旧句柄写 `BAD!` 仍然会改写新帧。代际号只能让**遵守 API 的
    调用方**发现"我过期了"，不能阻止裸写。

    接口和文档必须承认这一点（指令 §1.3："纯 Python 裸 buffer 引用
    不可被撤销，API 和隔离边界必须承认这一点"）。
    真正隔离要靠跨进程共享槽位描述符。
    """
    p = LeasePool(capacity=1, slot_bytes=4, name="bare")
    old = p.acquire("old", 1)
    old.buffer[:] = b"OLD!"
    p.release(old.lease_id, 1)

    new = p.acquire("new", 2)
    new.buffer[:] = b"NEW!"
    assert p.is_current(old) is False, "代际已判旧句柄过期"

    # 但裸写仍然生效 —— 这就是局限
    old.buffer[:] = b"BAD!"
    assert bytes(new.buffer) == b"BAD!", (
        "预期（并接受）裸引用能改写新帧；若这里失败说明底层不再复用同一块 "
        "内存 —— 那是更强的保证，请更新本测试与 docstring")


def test_pool_reports_balanced_even_when_bare_reference_corrupted():
    """统计平衡**不能**当作"内存安全"的证据。"""
    p = LeasePool(capacity=1, slot_bytes=4, name="bal")
    old = p.acquire("old", 1)
    p.release(old.lease_id, 1)
    new = p.acquire("new", 2)
    old.buffer[:] = b"XXX"
    st = p.stats()
    assert st["balanced"] is True, st
    assert bytes(new.buffer) == b"XXX", "账面平衡而内容已被旧句柄改写"
