# -*- coding: utf-8 -*-
"""第十八轮 P0-B / R17-02：恢复分类与 journal 结构。

## 缺陷一：old==new 的对象被当成了"新代"（已修）

`classify_generation` 原来这样判：

```python
if new_d and dig == new_d:
    n_new += 1
elif old_d is not None and dig == old_d:
    n_old += 1
```

当 `old_digest == new_digest`（这一次发布写出的字节与上一代**完全相同**），
两个条件同时成立，而代码**先匹配 new**。于是"媒体已回滚到旧代、但新旧字节
恰好一样"被记成 1 个新代对象 —— 与组内其他仍为旧代的对象混在一起，
整组被判 `mixed`，尽管磁盘其实完整等于旧代（审核探针复现）。

正确语义：此时"等于旧代"和"等于新代"是**同一件事**，它既不支持也不反对
任何一方，因此是**中性证据**，不能参与"混合"判定。

## 缺陷二：journal 是目录被当成"没有事务"（已修）

`if not os.path.isfile(journal_path): return None` 把"存在但不是普通文件"
和"不存在"混为一谈。目录/设备/被替换的情况下返回 None，调用方会当成
"无事务"直接继续发布 —— 把一份**未知状态**当成干净起点。
审核要求：目录/不可访问是**未知或错误**，不是无事务。
"""
import json
import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import publisher as pub  # noqa: E402
from publish_journal import write_journal  # noqa: E402


def _write(path, data):
    with open(path, "wb") as f:
        f.write(data)


def _digest(path):
    return pub.file_digest(path)


# ------------------------------------------------ 中性对象

def test_R17_02_identical_digests_are_neutral(tmp_path):
    """整组新旧字节相同 → identical（不是 all_new）。"""
    a = str(tmp_path / "a.mkv")
    _write(a, b"SAME-BYTES")
    d = _digest(a)

    recs = {a: {"existed": True, "old_digest": d, "new_digest": d}}
    assert pub.classify_generation(recs) == pub.GENERATION_IDENTICAL


def test_R17_02_identical_plus_old_is_all_old(tmp_path):
    """中性对象 + 明确的旧代对象 → all_old（不能被中性对象拉成 mixed）。"""
    a = str(tmp_path / "a.mkv")
    b = str(tmp_path / "b.json")
    _write(a, b"SAME")
    _write(b, b"OLD-B")
    da = _digest(a)

    recs = {
        a: {"existed": True, "old_digest": da, "new_digest": da},   # 中性
        b: {"existed": True, "old_digest": _digest(b),
            "new_digest": "0" * 64},                                 # 明确旧代
    }
    assert pub.classify_generation(recs) == pub.GENERATION_ALL_OLD


def test_R17_02_identical_plus_new_is_all_new(tmp_path):
    """中性对象 + 明确的新代对象 → all_new。"""
    a = str(tmp_path / "a.mkv")
    b = str(tmp_path / "b.json")
    _write(a, b"SAME")
    _write(b, b"NEW-B")
    da = _digest(a)

    recs = {
        a: {"existed": True, "old_digest": da, "new_digest": da},
        b: {"existed": True, "old_digest": "1" * 64,
            "new_digest": _digest(b)},
    }
    assert pub.classify_generation(recs) == pub.GENERATION_ALL_NEW


def test_R17_02_real_mixed_still_mixed(tmp_path):
    """真正的新旧混杂仍必须判 mixed —— 修复不能把这条掩盖掉。"""
    a = str(tmp_path / "a.mkv")
    b = str(tmp_path / "b.json")
    _write(a, b"OLD-A")
    _write(b, b"NEW-B")

    recs = {
        a: {"existed": True, "old_digest": _digest(a),
            "new_digest": "2" * 64},
        b: {"existed": True, "old_digest": "3" * 64,
            "new_digest": _digest(b)},
    }
    assert pub.classify_generation(recs) == pub.GENERATION_MIXED


# ------------------------------------------------ 回滚后新旧相同

def test_R17_02_rolled_back_with_identical_bytes_recovers(tmp_path):
    """T02 核心：回滚后媒体为旧代、报告新旧字节相同 → 应识别为完整旧代。

    修复前：被判 all_new/mixed → 恢复拒绝（`journal says ROLLED_BACK but
    the group on disk is ...`），下一次发布也发不出去。
    """
    media = str(tmp_path / "m.mkv")
    _write(media, b"IDENTICAL-CONTENT")
    d = _digest(media)

    jp = str(tmp_path / "journal.json")
    doc = {
        "attempt_id": "att-A",
        "phase": "ROLLED_BACK",
        "targets": [media],
        "records": {
            media: {"existed": True, "old_digest": d, "new_digest": d},
        },
    }
    assert write_journal(jp, doc) is True

    res = pub.recover(jp, str(tmp_path / "bak"), "att-B")
    assert res is not None, "有 journal 就不该当作无事务"
    assert res.get("ok") is True, res
    assert res.get("action") == "already_rolled_back", res


def test_R17_02_committed_with_identical_bytes_recovers(tmp_path):
    """对称情形：journal 说 COMMITTED 且新旧相同 → 按提交收尾。"""
    media = str(tmp_path / "m.mkv")
    _write(media, b"IDENTICAL")
    d = _digest(media)

    jp = str(tmp_path / "journal.json")
    doc = {
        "attempt_id": "att-C",
        "phase": "COMMITTED",
        "targets": [media],
        "records": {
            media: {"existed": True, "old_digest": d, "new_digest": d},
        },
    }
    assert write_journal(jp, doc) is True

    res = pub.recover(jp, str(tmp_path / "bak"), "att-D")
    assert res is not None
    assert res.get("ok") is True, res
    assert res.get("action") == "already_committed", res


# ------------------------------------------------ journal 类型

def test_R17_02_journal_directory_is_unknown_not_absent(tmp_path):
    """journal 路径是目录 → 必须报"未知"，绝不能返回 None（= 无事务）。"""
    jp = tmp_path / "journal.json"
    jp.mkdir()

    res = pub.recover(str(jp), str(tmp_path / "bak"), "att-E")
    assert res is not None, "目录形式的 journal 不能被当成'没有事务'"
    assert res.get("ok") is False, res


def test_R17_02_missing_journal_still_returns_none(tmp_path):
    """真的不存在时仍返回 None —— 修复不能把'无事务'也弄坏。"""
    res = pub.recover(str(tmp_path / "nope.json"), str(tmp_path / "bak"),
                      "att-F")
    assert res is None


# ------------------------------------------------ T03 journal 结构

def test_T03_validate_structure_flags_records_outside_targets():
    """records 里有未列入 targets 的路径 → 必须报结构问题。

    审核受控探针：这种 journal 会让恢复对"目标集合之外"的对象执行删除计划。
    """
    from publish_journal import validate_journal_structure as vjs

    doc = {
        # R19：结构校验现在**要求** attempt_id（没有它无法说"这是哪一次"）
        "attempt_id": "att-T03",
        "phase": "PREPARED",
        "targets": ["/allowed/a.mkv"],
        "records": {
            "/allowed/a.mkv": {"existed": True},
            "/elsewhere/b.mkv": {"existed": True},   # ← 越界
        },
    }
    problems = vjs(doc)
    assert problems, "records 超出 targets 必须被报出来"
    assert any(("not in targets" in p) or ("not declared in targets" in p)
               for p in problems), problems


def test_T03_validate_structure_accepts_a_consistent_journal():
    """一致的结构不能误报（防止修过头把正常 journal 也拒了）。"""
    from publish_journal import validate_journal_structure as vjs

    doc = {
        "attempt_id": "att-T03",
        "phase": "PREPARED",
        "targets": ["/allowed/a.mkv", "/allowed/a.json"],
        "records": {
            "/allowed/a.mkv": {"existed": True},
            "/allowed/a.json": {"existed": True},
        },
    }
    assert vjs(doc) == []


def test_T03_validate_structure_flags_missing_targets():
    from publish_journal import validate_journal_structure as vjs
    assert vjs({"attempt_id": "a", "phase": "PREPARED",
                "records": {"/x": {}}})
    assert vjs({"attempt_id": "a", "phase": "PREPARED",
                "targets": ["/x"], "records": {}})


def test_T03_recover_refuses_inconsistent_journal_and_touches_nothing(tmp_path):
    """**核心**：结构不成立 → 恢复拒绝，且越界路径**一个字节都不动**。"""
    inside = tmp_path / "a.mkv"
    inside.write_bytes(b"INSIDE")
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.bin"
    victim.write_bytes(b"MUST-NOT-BE-TOUCHED")

    jp = str(tmp_path / "journal.json")
    doc = {
        "attempt_id": "att-X",
        "phase": "PREPARED",
        "targets": [str(inside)],
        "records": {
            str(inside): {"existed": True,
                          "old_digest": "0" * 64, "new_digest": "1" * 64},
            str(victim): {"existed": True,
                          "old_digest": "2" * 64, "new_digest": "3" * 64},
        },
    }
    assert write_journal(jp, doc) is True

    res = pub.recover(jp, str(tmp_path / "bak"), "att-Y")
    assert res is not None
    assert res.get("ok") is False, res
    assert "structure" in (res.get("reason") or "").lower(), res
    # 越界对象必须原封不动
    assert victim.exists() and victim.read_bytes() == b"MUST-NOT-BE-TOUCHED"
