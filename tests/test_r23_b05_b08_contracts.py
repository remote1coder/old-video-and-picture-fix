# -*- coding: utf-8 -*-
"""R23-B05 / B08：颜色转换与帧边界合同必须**结构化且绑定实际数据**。

## 审核给的反例

| 位置 | 反例 | 修复前 |
|---|---|---|
| 颜色 B05 | `applied=["matrix"]` + `contract="not-a-conversion-contract"` | 问题列表**为空** |
| 颜色 B05 | `contract={"nonsense": true}` | 问题列表**为空** |
| 帧 B08 | `frames_digest_proven=True` 但摘要为 None | 问题列表**为空** |
| 帧 B08 | 同摘要，域/时间/颜色/模型都不同 | `compare_contracts` 为空 |
| 帧 B08 | 模拟 probe 失败，写入器回填请求的尺寸/格式 | 写了 gaps 但 `contract_problems` 仍空 |

## 判据的共同点

"我声明了"不等于"能证明"。颜色合同要写出**哪一侧是什么、做了什么操作、
谁做的、绑定了哪两份字节**；帧合同要把**请求值/观测值/未知**分开，
并且时间引用必须有内容身份。
"""
import json
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import generation_evidence as GE  # noqa: E402
import frame_contract as fc  # noqa: E402


# ------------------------------------------------ B05：颜色转换合同

def _color_receipt(contract):
    return {
        "color_source": {"probed": True, "matrix": "bt601",
                         "range": "limited", "primaries": "bt709",
                         "transfer": "bt709"},
        "color": {"probed": True, "matrix": "bt709", "range": "limited",
                  "primaries": "bt709", "transfer": "bt709"},
        "color_changed_tags": [{"field": "matrix", "source": "bt601",
                                "output": "bt709"}],
        "color_lost_tags": [],
        "color_conversion": {"applied": ["matrix"], "contract": contract},
        "bounds": {"performed": True, "ok": True},
    }


def _good_contract():
    return {
        "schema": "color-conversion/1",
        "from": {"matrix": "bt601", "range": "limited",
                 "primaries": "bt709", "transfer": "bt709"},
        "to": {"matrix": "bt709", "range": "limited",
               "primaries": "bt709", "transfer": "bt709"},
        "operations": ["matrix"],
        "implementation": "ffmpeg zscale 2023-11-13",
        "binding": {"input_digest": "a" * 64, "output_digest": "b" * 64},
    }


def test_a_bare_string_contract_is_not_evidence():
    probs = GE.receipt_evidence_problems(_color_receipt("not-a-contract"))
    assert probs, "任意非空文字被当成了转换证据"
    assert any("bare string" in p for p in probs), probs


def test_an_unrelated_dict_is_not_a_contract():
    probs = GE.receipt_evidence_problems(_color_receipt({"nonsense": True}))
    assert probs
    assert any("schema" in p for p in probs), probs
    assert any("from description" in p for p in probs), probs


def test_a_missing_contract_is_rejected():
    doc = _color_receipt(None)
    probs = GE.receipt_evidence_problems(doc)
    assert probs and any("no contract" in p for p in probs), probs


def test_a_properly_bound_contract_is_accepted():
    """合法转换不能一律拒绝（BT.601→BT.709 是真实需求）。"""
    assert GE.receipt_evidence_problems(_color_receipt(_good_contract())) == []


def test_a_contract_describing_a_different_conversion_is_rejected():
    c = _good_contract()
    c["from"]["matrix"] = "bt470bg"        # 与观测到的源值不符
    probs = GE.receipt_evidence_problems(_color_receipt(c))
    assert any("but the source recorded" in p for p in probs), probs


def test_a_contract_without_content_binding_is_rejected():
    c = _good_contract()
    c["binding"] = {}
    probs = GE.receipt_evidence_problems(_color_receipt(c))
    assert any("not bound" in p for p in probs), probs


def test_a_contract_missing_an_operation_is_rejected():
    c = _good_contract()
    c["operations"] = ["range"]
    probs = GE.receipt_evidence_problems(_color_receipt(c))
    assert any("does not list the applied operation" in p for p in probs), probs


def test_the_producer_does_not_fabricate_a_conversion(tmp_path):
    """`collect` 自己不执行转换，就不许声明 `applied`（B05 后半段）。"""
    p = str(tmp_path / "a.mkv")
    with open(p, "wb") as f:
        f.write(b"\x00" * 32)
    ev = GE.collect(p, p, "stage34", {}, None)
    conv = ev.get("color_conversion") or {}
    assert conv.get("applied") == []
    assert conv.get("contract") is None
    assert conv.get("evidence_state")


# ------------------------------------------------ B08：帧边界合同

def _contract(**over):
    base = dict(frames=5, width=640, height=480, pix_fmt="yuv420p16le",
                timeline={"plan_path": "/t/plan.json"},
                color={"probed": True, "pix_fmt": "yuv420p16le"},
                ai={"impl": "realesrgan", "model": "x2plus"},
                frames_digest="a" * 64,
                frames_digest_domain="raw-rgb48le")
    base.update(over)
    return fc.build_contract(**base)


def test_a_complete_contract_has_no_problems():
    assert fc.contract_problems(_contract()) == []


def test_proven_digest_without_a_digest_is_reported():
    c = _contract(frames_digest=None, frames_digest_proven=True)
    probs = fc.contract_problems(c)
    assert any("proven but no digest" in p for p in probs), probs


def test_a_backfilled_pixel_format_is_not_a_measurement():
    """probe 失败时回填请求值 —— 必须标成"没测到"，不能当事实。"""
    c = _contract(pix_fmt_proven=False)
    probs = fc.contract_problems(c)
    assert any("pixel format was not probed" in p for p in probs), probs


def test_a_timeline_reference_needs_a_content_identity():
    """只写 fps + 帧数不算引用（审核："不能只留文件名、fps和计数"）。"""
    c = _contract(timeline={"fps": "25", "frames": 5})
    probs = fc.contract_problems(c)
    assert any("no content identity" in p for p in probs), probs


def test_recorded_gaps_are_problems_not_notes():
    c = _contract()
    c["gaps"] = ["the AI subprocess did not report FRAMES_DIGEST"]
    probs = fc.contract_problems(c)
    assert any("recorded gap" in p for p in probs), probs


def test_declaration_differences_are_compared_but_labelled():
    """时间/颜色/模型不同必须被比较，且要标明"这是声明层，不是像素"。"""
    a = _contract()
    b = _contract(timeline={"plan_path": "/t/other.json"},
                  color={"probed": True, "pix_fmt": "yuv420p16le",
                         "matrix": "bt709"},
                  ai={"impl": "realesrgan", "model": "x4plus"})
    probs = fc.compare_contracts(a, b)
    assert any("timeline reference differs" in p for p in probs), probs
    assert any("colour contract differs" in p for p in probs), probs
    assert any("fingerprint differs" in p for p in probs), probs
    assert all("not pixels" in p for p in probs), probs


def test_execution_differences_are_not_frame_identity():
    """经典 vs 融合只是"怎么跑"，不改变"这串帧是谁产的"。"""
    a = _contract(execution={"fused_stage4": False})
    b = _contract(execution={"fused_stage4": True})
    assert fc.compare_contracts(a, b) == []


def test_same_contract_compares_clean():
    assert fc.compare_contracts(_contract(), _contract()) == []
