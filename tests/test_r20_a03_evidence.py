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
        "bounds": {"ok": True, "performed": True, "reason": ""},
    }


def test_complete_evidence_has_no_problems():
    assert ge.receipt_evidence_problems(_good_receipt()) == []


@pytest.mark.parametrize("mutate,fragment", [
    (lambda d: d.pop("color"), "no color evidence"),
    (lambda d: d.pop("bounds"), "no bounds evidence"),
    (lambda d: d["color"].update(probed=False), "never probed"),
    (lambda d: d["color"].update(complete=False, unknown_fields=["matrix"]),
     "color fields unknown"),
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
    from checkpoint import compute_param_hash

    src = tmp_path / "src.mp4"
    src.write_bytes(b"S" * 64)
    out = tmp_path / "o.mkv"
    out.write_bytes(b"X" * 8)
    rcpt = tmp_path / "o.mkv.fused_stage34.json"
    dig = compute_param_hash(str(src), {})

    def _write(doc):
        base = {"attempt_id": "a1", "input_digest": dig,
                "output": str(out), "output_size": out.stat().st_size}
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
    doc["color"]["matrix"] = None
    doc["color"]["complete"] = False
    doc["color"]["unknown_fields"] = ["matrix"]
    _write(doc)
    ok, why, _ = _verify()
    assert ok is False and "color fields unknown" in why, why
