# -*- coding: utf-8 -*-
"""R20-A03：回执必须绑定**颜色成组描述**与**物理边界判定**。

审核 R19-A03 指出的两个缺口：

1. 回执只有 `out_pix_fmt` 一个字符串。颜色是**一组**属性
   （matrix / range / primaries / transfer），单看一个既不能证明"没被改色"，
   也不能证明"改对了色"。
2. **边界判定根本没有**。经典路线由 `stage4_postprocess.py` 自己调
   `frame_conservation.check_conservation()`；融合路线把 stage4 吞进 stage3
   之后，这一步从来没跑过 —— 优化路径绕过了唯一的边界门。

本文件锁住 `shared_modules/generation_evidence.py` 的语义，尤其是
**"没查过" ≠ "查过了没问题"**。
"""
import hashlib
import json
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import generation_evidence as ge  # noqa: E402
import fused_receipt as fr  # noqa: E402


# ------------------------------------------------ 颜色：成组，不塌缩

def _probe(monkeypatch, **kw):
    import media_io
    doc = {"pix_fmt": "yuv420p16le", "range": "limited", "matrix": "bt709",
           "primaries": "bt709", "transfer": "bt709", "probed": True}
    doc.update(kw)
    monkeypatch.setattr(media_io, "probe_color", lambda p, *a, **k: doc)


def test_color_group_is_complete_only_when_all_fields_are_known(monkeypatch):
    _probe(monkeypatch)
    got = ge.color_group("x.mkv")
    assert got["probed"] is True
    assert got["complete"] is True
    assert got["unknown_fields"] == []
    assert got["matrix"] == "bt709" and got["range"] == "limited"


def test_color_group_lists_the_unknown_fields(monkeypatch):
    """字段缺失要**点名**，不能悄悄当"没问题"。"""
    _probe(monkeypatch, matrix=None)
    got = ge.color_group("x.mkv")
    assert got["probed"] is True
    assert got["complete"] is False
    assert got["unknown_fields"] == ["matrix"]


def test_color_group_reports_a_failed_probe(monkeypatch):
    import media_io
    monkeypatch.setattr(media_io, "probe_color",
                        lambda p, *a, **k: {"probed": False, "error": "boom"})
    got = ge.color_group("x.mkv")
    assert got["probed"] is False
    assert got["complete"] is False
    assert "boom" in got["error"]


# ------------------------------------------------ 边界：没查过 ≠ 通过

def _conservation(monkeypatch, ok=True, reasons=None):
    import frame_conservation
    import rate_spec
    monkeypatch.setattr(rate_spec, "parse_rate_or_none",
                        lambda v: None, raising=False)
    monkeypatch.setattr(
        frame_conservation, "check_conservation",
        lambda *a, **k: (ok, {"stage": "stage34", "ok": ok,
                              "reasons": list(reasons or [])}), raising=False)


def test_bounds_report_records_a_pass(monkeypatch):
    _conservation(monkeypatch, ok=True)
    got = ge.bounds_report("a.mkv", "b.mkv", "stage34")
    assert got["performed"] is True and got["ok"] is True
    assert got["report"]["stage"] == "stage34"


def test_bounds_report_surfaces_the_reasons_of_a_failure(monkeypatch):
    _conservation(monkeypatch, ok=False, reasons=["r_frame_rate changed 24 -> 25"])
    got = ge.bounds_report("a.mkv", "b.mkv", "stage34")
    assert got["performed"] is True and got["ok"] is False
    assert "r_frame_rate" in got["reason"]


def test_bounds_report_marks_a_raised_check_as_not_performed(monkeypatch):
    """核对抛异常 = 没查，不是通过。"""
    import frame_conservation
    import rate_spec

    monkeypatch.setattr(rate_spec, "parse_rate_or_none", lambda v: None,
                        raising=False)

    def _boom(*a, **k):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(frame_conservation, "check_conservation", _boom,
                        raising=False)
    got = ge.bounds_report("a.mkv", "b.mkv", "stage34")
    assert got["performed"] is False
    assert got["ok"] is False
    assert "probe exploded" in got["reason"]


def test_bounds_report_marks_a_missing_helper_as_not_performed(monkeypatch):
    """模块缺失时**不能**假装查过（stage4 旧行为只是 warn，这里必须留痕）。"""
    monkeypatch.setitem(sys.modules, "frame_conservation", None)
    got = ge.bounds_report("a.mkv", "b.mkv", "stage34")
    assert got["performed"] is False
    assert got["ok"] is False
    assert got["reason"]


# ------------------------------------------------ 证据齐全性判定

def _good_receipt():
    return {
        "color": {"pix_fmt": "yuv420p16le", "range": "limited",
                  "matrix": "bt709", "primaries": "bt709", "transfer": "bt709",
                  "probed": True, "complete": True, "unknown_fields": []},
        "color_source": {"pix_fmt": "yuv420p16le", "range": "limited",
                         "matrix": "bt709", "primaries": "bt709",
                         "transfer": "bt709", "probed": True, "complete": True,
                         "unknown_fields": []},
        "color_lost_tags": [],
        "bounds": {"ok": True, "performed": True, "reason": ""},
    }


def test_complete_evidence_has_no_problems():
    assert ge.receipt_evidence_problems(_good_receipt()) == []


def test_untagged_output_is_not_a_problem_when_the_source_is_untagged_too():
    """N20 实测反例：合成素材本身没有 matrix/primaries/transfer 标签。

    产物同样没有标签是**正确**的 —— 逼它"字段齐全"等于逼实现编造一个矩阵。
    """
    doc = _good_receipt()
    doc["color"] = {"pix_fmt": "yuv420p16le", "range": "limited",
                    "matrix": None, "primaries": None, "transfer": None,
                    "probed": True, "complete": False,
                    "unknown_fields": ["matrix", "primaries", "transfer"]}
    doc["color_source"] = dict(doc["color"])
    doc["color_lost_tags"] = []
    assert ge.receipt_evidence_problems(doc) == []


def test_losing_a_tag_the_source_had_is_a_problem():
    doc = _good_receipt()
    doc["color"].update(matrix=None, complete=False,
                        unknown_fields=["matrix"])
    doc["color_lost_tags"] = ["matrix"]
    problems = ge.receipt_evidence_problems(doc)
    assert any("colour tags lost" in p for p in problems), problems


def test_lost_tags_is_relative_to_the_source():
    src = {"pix_fmt": "yuv420p16le", "range": "limited", "matrix": "bt709",
           "primaries": None, "transfer": None}
    out = {"pix_fmt": "yuv420p16le", "range": "limited", "matrix": None,
           "primaries": None, "transfer": None}
    assert ge._lost_tags(src, out) == ["matrix"]
    assert ge._lost_tags({"matrix": None}, {"matrix": None}) == []
    assert ge._lost_tags({"matrix": "bt709"}, {"matrix": "bt709"}) == []


@pytest.mark.parametrize("mutate,fragment", [
    (lambda d: d.pop("color"), "no color evidence"),
    (lambda d: d.pop("bounds"), "no bounds evidence"),
    (lambda d: d["color"].update(probed=False), "never probed"),
    (lambda d: d.update(color_lost_tags=["matrix"]), "colour tags lost"),
    (lambda d: d["bounds"].update(performed=False, ok=False),
     "was not performed"),
    (lambda d: d["bounds"].update(performed=True, ok=False), "bounds check failed"),
])
def test_every_missing_or_failed_evidence_is_reported(mutate, fragment):
    doc = _good_receipt()
    mutate(doc)
    problems = ge.receipt_evidence_problems(doc)
    assert problems, "证据不全必须留下问题记录"
    assert any(fragment in p for p in problems), problems


# ------------------------------------------------ 与共享验证器接线

def test_verify_fused_receipt_requires_the_evidence_blocks(tmp_path):
    """走到证据层的端到端：身份都对，但证据块缺一块就得拒。"""
    src = tmp_path / "src.mp4"
    src.write_bytes(b"S" * 64)
    out = tmp_path / "o.mkv"
    out.write_bytes(b"X" * 8)
    rcpt = tmp_path / "o.mkv.fused_stage34.json"
    # 源身份用**空 params** 的摘要（跨 stage 稳定），参数用冻结值
    dig = fr.source_digest(str(src))

    def _write(doc):
        base = {"attempt_id": "a1", "input_digest": dig,
                "input_digest_kind": "source-only", "param_digest": "",
                "output": str(out), "output_size": out.stat().st_size,
                # R21-C06：回执必须带内容摘要，否则按 legacy 拒
                "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
        base.update(doc)
        rcpt.write_text(json.dumps(base), encoding="utf-8")

    def _verify():
        return fr.verify_fused_receipt(str(rcpt), {}, expected_attempt="a1",
                                       src_path=str(src))

    _write(_good_receipt())
    ok, why, _ = _verify()
    assert ok is True, why

    doc = _good_receipt()
    doc.pop("bounds")
    _write(doc)
    ok, why, _ = _verify()
    assert ok is False and "no bounds evidence" in why, why

    doc = _good_receipt()
    doc["color_lost_tags"] = ["matrix"]
    doc["color"]["matrix"] = None
    _write(doc)
    ok, why, _ = _verify()
    assert ok is False and "colour tags lost" in why, why


# ------------------------------------------------ 摘要语义（N20 实测反例）

def test_source_digest_ignores_params(tmp_path):
    """源身份必须**跨 stage 稳定**：同一文件、不同 params → 同一个值。"""
    src = tmp_path / "s.mp4"
    src.write_bytes(b"Z" * 128)
    a = fr.source_digest(str(src))
    b = fr.source_digest(str(src))
    assert a and a == b
    # 换内容 → 必须变
    src.write_bytes(b"Z" * 129)
    assert fr.source_digest(str(src)) != a


def test_receipt_with_a_stage_dependent_digest_is_rejected(tmp_path):
    """反例固化：用 `compute_param_hash(源, params)` 当 input_digest 的回执
    必须被拒 —— 那个值随 params 逐 stage 变化，消费方重算必然"变了"。"""
    from checkpoint import compute_param_hash

    src = tmp_path / "s.mp4"
    src.write_bytes(b"S" * 32)
    out = tmp_path / "o.mkv"
    out.write_bytes(b"X" * 4)
    rcpt = tmp_path / "o.mkv.fused_stage34.json"
    doc = _good_receipt()
    doc.update({"attempt_id": "a1", "output": str(out),
                "output_size": 4,
                "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                # 旧写法：把"源+参数"的摘要塞进 input_digest，且不声明 kind
                "input_digest": compute_param_hash(str(src), {"crf": 20})})
    rcpt.write_text(json.dumps(doc), encoding="utf-8")
    ok, why, _ = fr.verify_fused_receipt(str(rcpt), {}, expected_attempt="a1",
                                        src_path=str(src))
    assert ok is False
    assert "source-only" in why or "no input_digest" in why, why


def test_param_digest_is_compared_against_the_frozen_value(tmp_path):
    """参数身份只认 main 冻结的值；对不上就拒。"""
    src = tmp_path / "s.mp4"
    src.write_bytes(b"S" * 32)
    out = tmp_path / "o.mkv"
    out.write_bytes(b"X" * 4)
    rcpt = tmp_path / "o.mkv.fused_stage34.json"
    doc = _good_receipt()
    doc.update({"attempt_id": "a1", "output": str(out), "output_size": 4,
                "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "input_digest": fr.source_digest(str(src)),
                "input_digest_kind": "source-only",
                "param_digest": "deadbeef"})
    rcpt.write_text(json.dumps(doc), encoding="utf-8")

    ok, why, _ = fr.verify_fused_receipt(
        str(rcpt), {"_frozen_param_digest": "cafebabe"},
        expected_attempt="a1", src_path=str(src))
    assert ok is False and "quality params changed" in why, why

    ok, why, _ = fr.verify_fused_receipt(
        str(rcpt), {"_frozen_param_digest": "deadbeef"},
        expected_attempt="a1", src_path=str(src))
    assert ok is True, why


def test_frozen_param_digest_absent_skips_the_param_check(tmp_path):
    """手工跑（没有 main 冻结值）时不阻断，但这是已知的弱校验路径。"""
    src = tmp_path / "s.mp4"
    src.write_bytes(b"S" * 32)
    out = tmp_path / "o.mkv"
    out.write_bytes(b"X" * 4)
    rcpt = tmp_path / "o.mkv.fused_stage34.json"
    doc = _good_receipt()
    doc.update({"attempt_id": "a1", "output": str(out), "output_size": 4,
                "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "input_digest": fr.source_digest(str(src)),
                "input_digest_kind": "source-only", "param_digest": ""})
    rcpt.write_text(json.dumps(doc), encoding="utf-8")
    ok, why, _ = fr.verify_fused_receipt(str(rcpt), {}, expected_attempt="a1",
                                        src_path=str(src))
    assert ok is True, why


def test_main_freezes_the_param_digest_once():
    """摘要必须在 main 的计划阶段算一次并下传，不能让每个 stage 各算一遍。"""
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "main.py"),
               encoding="utf-8", errors="replace").read()
    assert '_frozen_param_digest' in src
    i = src.index('params["_frozen_param_digest"]')
    assert i > 0


def test_stage4_refuses_the_step2_fallback_under_fusion():
    """融合模式下缺 step3 是**预期**，静默退到 step2 会产出未超分的成品。"""
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage4_postprocess.py"),
               encoding="utf-8", errors="replace").read()
    assert "STAGE4_FUSED_INPUT_MISSING" in src
    i = src.index("STAGE4_FUSED_INPUT_MISSING")
    # 必须出现在 STAGE4_FALLBACK 之前（先拦住，再考虑回退）
    assert src.index("STAGE4_FALLBACK") > i
