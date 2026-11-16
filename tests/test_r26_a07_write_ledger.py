# -*- coding: utf-8 -*-
"""累计写入必须由 writer 上报，不能从目录状态推断（R26-A07）。

复核给了精确的五行表：

| 行为 | 实际写入 | 目录差法 |
|---|---:|---:|
| 新建并保留 | 1024 | 1024 ✓ |
| 同一 stage 内新建后删除 | 1024 | **0** |
| 仅删除旧文件 | 0 | **1024** |
| 等长覆盖 | 1024 | **0** |
| 截断并写 512 | 512 | **0** |

本文件把五行都跑一遍：一边用 `record_write`（writer 上报），
一边用目录快照（旧实现），对照两者的差异。
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared_modules"))

import write_ledger as WL  # noqa: E402


def _snapshot(root) -> dict:
    out = {}
    for dirpath, _d, files in os.walk(str(root)):
        for fn in files:
            fp = os.path.join(dirpath, fn)
            try:
                out[fp] = os.path.getsize(fp)
            except OSError:
                pass
    return out


def _delta_bytes(pre: dict, post: dict) -> int:
    """旧实现的口径：新增/变大记增量，变小不记，删掉的按快照大小记。

    这就是当前 `run_stage` 包装器的算法。
    """
    total = 0
    for fp, sz in post.items():
        old = pre.get(fp, 0)
        if sz > old:
            total += sz - old
    for fp, sz in pre.items():
        if fp not in post:
            total += sz
    return total


@pytest.mark.parametrize("behavior,expected", [
    ("create_keep", 1024),
    ("create_then_delete", 1024),
    ("delete_only", 0),
    ("overwrite_same_size", 1024),
    ("truncate_write_512", 512),
])
def test_writer_reported_matches_truth(tmp_path, behavior, expected):
    """`record_write` 必须给出**真实写入量**。"""
    led = WL.WriteLedger()
    cat = "intermediate_media"

    if behavior == "create_keep":
        p = tmp_path / "a.bin"
        p.write_bytes(b"A" * 1024)
        led.record_write(cat, 1024, role="media", path=str(p))

    elif behavior == "create_then_delete":
        p = tmp_path / "b.bin"
        p.write_bytes(b"B" * 1024)
        led.record_write(cat, 1024, role="media", path=str(p))
        p.unlink()                      # 删掉**不**减 written

    elif behavior == "delete_only":
        p = tmp_path / "c.bin"
        p.write_bytes(b"C" * 1024)
        p.unlink()
        # 删除不调 record_write（它不写字节）

    elif behavior == "overwrite_same_size":
        p = tmp_path / "d.bin"
        p.write_bytes(b"D" * 1024)
        p.write_bytes(b"E" * 1024)      # 等长覆盖
        led.record_write(cat, 1024, role="media", path=str(p),
                         overwrote=True)

    elif behavior == "truncate_write_512":
        p = tmp_path / "e.bin"
        p.write_bytes(b"F" * 2048)
        with open(p, "wb") as f:
            f.write(b"G" * 512)         # 截断并写 512
        led.record_write(cat, 512, role="media", path=str(p),
                         overwrote=True)

    got = led.summary()["by_category"][cat]["bytes"]
    assert got == expected, (
        "writer 上报口径不对：%s 期望 %d 实得 %d" % (behavior, expected, got))


def test_directory_delta_misreports_four_of_five(tmp_path):
    """**对照**：目录差法在 5 种行为里错 4 种（复核的那张表）。"""
    results = {}

    for behavior in ("create_keep", "create_then_delete", "delete_only",
                     "overwrite_same_size", "truncate_write_512"):
        d = tmp_path / behavior
        d.mkdir()
        pre = _snapshot(d)

        if behavior == "create_keep":
            (d / "a.bin").write_bytes(b"A" * 1024)
        elif behavior == "create_then_delete":
            p = d / "b.bin"
            p.write_bytes(b"B" * 1024)
            p.unlink()
        elif behavior == "delete_only":
            p = d / "c.bin"
            p.write_bytes(b"C" * 1024)
            pre = _snapshot(d)          # 快照在**写入之后**（模拟"旧文件"）
            p.unlink()
        elif behavior == "overwrite_same_size":
            p = d / "d.bin"
            p.write_bytes(b"D" * 1024)
            pre = _snapshot(d)
            p.write_bytes(b"E" * 1024)
        elif behavior == "truncate_write_512":
            p = d / "e.bin"
            p.write_bytes(b"F" * 2048)
            pre = _snapshot(d)
            with open(p, "wb") as f:
                f.write(b"G" * 512)

        post = _snapshot(d)
        results[behavior] = _delta_bytes(pre, post)

    truth = {"create_keep": 1024, "create_then_delete": 1024,
             "delete_only": 0, "overwrite_same_size": 1024,
             "truncate_write_512": 512}
    wrong = [k for k in truth if results[k] != truth[k]]
    assert results["create_keep"] == 1024
    # 这四种旧口径都算错（新建后删=0、纯删=1024、等长覆盖=0、截断=0）
    assert set(wrong) == {"create_then_delete", "delete_only",
                          "overwrite_same_size", "truncate_write_512"}, results


def test_summary_declares_semantics():
    """summary 必须自报"这是累计写入，不是目录存量"。"""
    s = WL.WriteLedger().summary()
    assert "cumulative" in s["semantics"]
    assert "by_role" in s and "overwrites" in s


def test_zero_write_is_visible_not_hidden():
    """没有任何写入时，`written` 全 0 —— 但这是"真的没写"，不是"没观测"。"""
    led = WL.WriteLedger()
    s = led.summary()
    assert all(v["bytes"] == 0 for v in s["by_category"].values())
    assert s["unknown_categories"] == []
