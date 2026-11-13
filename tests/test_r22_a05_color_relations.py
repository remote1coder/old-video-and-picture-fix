# -*- coding: utf-8 -*-
"""R22-A05：颜色关系必须由**事实 + 转换计划**推导，不接受回执自报。

## 审核的两个反例

1. 源 `color_source.probed=False` 时，旧实现**无问题通过** ——
   把"源没探测过"当成了"源本来没标签"。
2. 源 matrix=bt601、输出=bt709，但 `color_changed_tags=[]` 时也通过 ——
   消费者直接相信回执自报的空列表，从不根据字段重新推导。

另外审核指出反向的坑：只要 changed 非空就拒绝，会让**合法**的
BT.601→BT.709 转换永远无法通过。所以"变了"要由**声明过的转换**支撑。

## 本文件锁住什么

| 场景 | 期望 |
|---|---|
| 源探测失败 | 拒绝（unverified） |
| 源有 bt601、输出 bt709、自报 changed=[] | 拒绝（自相矛盾 + 无声明转换） |
| 同上但有 `color_conversion.applied + contract` | 通过 |
| 只改标签（applied 有、contract 空） | 拒绝 |
| 源本来就没标签、输出也没有 | 通过（"没有"不是"丢了"） |
| 源有、输出丢 | 拒绝 |
"""
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import generation_evidence as ge  # noqa: E402


def _probed(**kw):
    base = {"pix_fmt": "yuv420p16le", "range": "limited",
            "matrix": "bt709", "primaries": "bt709", "transfer": "bt709",
            "probed": True, "complete": True, "unknown_fields": []}
    base.update(kw)
    return base


def _receipt(src=None, out=None, lost=None, changed=None, conv=None):
    d = {
        "color": out if out is not None else _probed(),
        "color_source": src if src is not None else _probed(),
        "color_lost_tags": lost if lost is not None else [],
        "color_changed_tags": changed if changed is not None else [],
        "bounds": {"ok": True, "performed": True, "reason": ""},
        # R24-06：合同要绑到**真实**对象上，回执必须记真实输入/输出身份。
        # 夹具缺这两个字段时，"合同绑到了错误的输入输出"永远测不出来。
        "input_digest": "a" * 64,
        "output_digest": "b" * 64,
    }
    if conv is not None:
        d["color_conversion"] = conv
    return d


# ------------------------------------------------ 源侧证据

def test_unprobed_source_is_unverified_not_untagged():
    """审核反例 1：源探测失败不能当"源本来没标签"。"""
    problems = ge.receipt_evidence_problems(
        _receipt(src=_probed(probed=False)))
    assert problems, "源没探测过却通过了"
    assert any("never probed" in p for p in problems), problems


def test_missing_source_evidence_is_rejected():
    d = _receipt()
    d.pop("color_source")
    problems = ge.receipt_evidence_problems(d)
    assert any("no source color evidence" in p for p in problems), problems


def test_untagged_source_and_output_is_fine():
    """源本来就没标签、输出也没标签 —— "没有"不是"丢了"。"""
    blank = {"pix_fmt": "yuv420p16le", "range": "limited",
             "matrix": None, "primaries": None, "transfer": None,
             "probed": True, "complete": False,
             "unknown_fields": ["matrix", "primaries", "transfer"]}
    assert ge.receipt_evidence_problems(_receipt(src=blank, out=blank)) == []


def test_lost_tag_is_rejected():
    src = _probed(matrix="bt601")
    out = _probed(matrix=None, complete=False, unknown_fields=["matrix"])
    problems = ge.receipt_evidence_problems(
        _receipt(src=src, out=out, lost=["matrix"]))
    assert any("lost between input and output" in p for p in problems), problems


# ------------------------------------------------ 关系由字段推导

def test_empty_changed_list_cannot_hide_a_real_change():
    """审核反例 2：源 bt601、输出 bt709，自报 changed=[] → 必须拒。"""
    problems = ge.receipt_evidence_problems(
        _receipt(src=_probed(matrix="bt601", primaries="bt601"),
                 out=_probed(matrix="bt709", primaries="bt709"),
                 changed=[]))
    assert problems
    assert any("disagrees with its own fields" in p for p in problems), problems
    assert any("without a declared conversion" in p for p in problems), problems


def test_relations_are_derived_not_trusted():
    """直接把 derive_color_relations 的结论钉住。"""
    src = {"matrix": "bt601", "range": "limited", "primaries": "bt601",
           "transfer": "bt709"}
    out = {"matrix": "bt709", "range": "limited", "primaries": "bt709",
           "transfer": "bt709"}
    lost, changed = ge.derive_color_relations(src, out)
    assert lost == []
    assert [c["field"] for c in changed] == ["matrix", "primaries"]
    # 源无值 → 不产生任何关系
    assert ge.derive_color_relations({"matrix": None}, {"matrix": "bt709"}) \
        == ([], [])


# ------------------------------------------------ 合法转换

def test_declared_conversion_passes():
    """有 applied + **结构化且绑定内容**的 contract 才合法（R23-B07）。

    R22 这里用的是一句自由文字 `"bt601 -> bt709 (declared)"` —— 审核正是
    以它为例证明"任意非空文字都能自证"。合同现在必须写清源/目标四元组、
    操作、实现和输入输出摘要。
    """
    problems = ge.receipt_evidence_problems(
        _receipt(src=_probed(matrix="bt601", primaries="bt601"),
                 out=_probed(matrix="bt709", primaries="bt709"),
                 changed=[{"field": "matrix", "source": "bt601",
                           "output": "bt709"},
                          {"field": "primaries", "source": "bt601",
                           "output": "bt709"}],
                 conv={"applied": ["matrix", "primaries"],
                       "contract": {
                           "schema": "color-conversion/1",
                           "from": {"matrix": "bt601", "range": "limited",
                                    "primaries": "bt601", "transfer": "bt709"},
                           "to": {"matrix": "bt709", "range": "limited",
                                  "primaries": "bt709", "transfer": "bt709"},
                           "operations": ["matrix", "primaries"],
                           "implementation": "ffmpeg zscale 2023-11-13",
                           "binding": {"input_digest": "a" * 64,
                                       "output_digest": "b" * 64},
                       }}))
    assert problems == [], problems


def test_relabelling_without_a_contract_is_rejected():
    """只声明"应用了转换"但没有任何合同 = 只改了标签。"""
    problems = ge.receipt_evidence_problems(
        _receipt(src=_probed(matrix="bt601"),
                 out=_probed(matrix="bt709"),
                 changed=[{"field": "matrix", "source": "bt601",
                           "output": "bt709"}],
                 conv={"applied": ["matrix"], "contract": ""}))
    # R23-B07：空字符串也走"不是合同"这条；措辞随实现变，判据不变。
    assert any(("no conversion contract" in p) or ("bare string" in p)
               for p in problems), problems


def test_conversion_must_cover_every_changed_field():
    """声明只覆盖 matrix，但 primaries 也变了 → 仍要拒。"""
    problems = ge.receipt_evidence_problems(
        _receipt(src=_probed(matrix="bt601", primaries="bt601"),
                 out=_probed(matrix="bt709", primaries="bt709"),
                 changed=[{"field": "matrix", "source": "bt601",
                           "output": "bt709"},
                          {"field": "primaries", "source": "bt601",
                           "output": "bt709"}],
                 conv={"applied": ["matrix"],
                       "contract": "bt601 -> bt709 (matrix only)"}))
    assert any("primaries" in p for p in problems), problems


# ------------------------------------------------ 全一致

def test_identical_colour_passes():
    assert ge.receipt_evidence_problems(_receipt()) == []


# ------------------------------------------------ 事务指纹（R22-A09 前置）

def test_transaction_fingerprint_covers_record_contents():
    """指纹必须覆盖 old/new 摘要与备份 —— 只数键集合会漏。

    审核："事务指纹只包含 attempt/phase/targets/record 键，
    records 里的 old/new 摘要和 backup 改变不会改变指纹。"
    """
    import copy
    import publisher as P

    base = {
        "attempt_id": "att1", "phase": "PREPARED",
        "targets": ["/x/a.mkv"],
        "records": {"/x/a.mkv": {
            "existed": False, "old_digest": None, "new_digest": "n1",
            "backup": None, "backup_digest": None}},
    }
    fp0 = P._txn_fingerprint(base)
    for field, value in (("new_digest", "n2"),
                         ("old_digest", "o1"),
                         ("backup", "/x/bk"),
                         ("backup_digest", "bd"),
                         ("existed", True)):
        d = copy.deepcopy(base)
        d["records"]["/x/a.mkv"][field] = value
        assert P._txn_fingerprint(d) != fp0, f"{field} 改了但指纹没变"
    # 同一份内容重复计算必须稳定
    assert P._txn_fingerprint(copy.deepcopy(base)) == fp0
