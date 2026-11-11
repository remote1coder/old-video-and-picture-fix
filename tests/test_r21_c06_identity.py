# -*- coding: utf-8 -*-
"""R21-C06：融合回执能接受被换过的产物 / 颜色证据传递不完整 / 身份冻结太早。

## 审核原文（R20 §3，P1）

> 当前 output 校验主要是存在与可选文件大小，没有内容摘要。同大小替换仍可通过；
> 没有 `_frozen_param_digest` 时直接跳过参数一致性比较；记录的实现指纹和
> filterchain 没有得到足够对照。
>
> 颜色采集函数产生 `color_source`、`color_lost_tags` 等信息，但 stage3 回执
> 只写部分 `color` 和 `bounds`，漏掉源状态与丢标签结果。验证器也允许这些字段缺失。
>
> `_frozen_param_digest` 当前早于 resolve_run_plan 和 chain_mode 落定。

## 本文件锁住什么

1. 回执必须带 `output_sha256`；**同大小替换**要被发现；缺摘要按 legacy 拒
2. `color_source` / `color_lost_tags` / `color_changed_tags`（known→另一 known）
   都要进回执，并参与"证据是否完整"的判定
3. `_frozen_param_digest` 必须在 `resolve_run_plan` 与执行模式落定**之后**算；
   算不出来要非零退出，不能退化成空串跳过校验
"""
import hashlib
import json
import os
import re
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fused_receipt as fr  # noqa: E402
import generation_evidence as ge  # noqa: E402

_MAIN = os.path.join(_ROOT, "video_restoration_pipeline", "main.py")
_STAGE3 = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                       "stage3_ai_upscale.py")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _receipt(tmp_path, **over):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"S" * 32)
    out = tmp_path / "o.mkv"
    out.write_bytes(b"X" * 8)
    doc = {
        "attempt_id": "a1",
        "input_digest": fr.source_digest(str(src)),
        "input_digest_kind": "source-only",
        "param_digest": "",
        "output": str(out),
        "output_size": 8,
        "output_sha256": _sha(str(out)),
        "color": {"pix_fmt": "yuv420p16le", "range": "limited",
                  "matrix": None, "primaries": None, "transfer": None,
                  "probed": True, "complete": False,
                  "unknown_fields": ["matrix", "primaries", "transfer"]},
        "color_source": {"pix_fmt": "yuv420p16le", "range": "limited",
                         "matrix": None, "primaries": None, "transfer": None,
                         "probed": True, "complete": False,
                         "unknown_fields": ["matrix", "primaries", "transfer"]},
        "color_lost_tags": [],
        "color_changed_tags": [],
        "bounds": {"ok": True, "performed": True, "reason": ""},
    }
    doc.update(over)
    p = tmp_path / "o.mkv.fused_stage34.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p, src, out


def _verify(path, src):
    return fr.verify_fused_receipt(str(path), {}, expected_attempt="a1",
                                   src_path=str(src))


# ------------------------------------------------ ① 内容摘要

def test_receipt_without_output_digest_is_rejected_as_legacy(tmp_path):
    p, src, _out = _receipt(tmp_path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc.pop("output_sha256")
    p.write_text(json.dumps(doc), encoding="utf-8")
    ok, why, _ = _verify(p, src)
    assert ok is False
    assert "no output digest" in why, why


def test_same_size_replacement_is_detected(tmp_path):
    """审核点名的场景：字节换了但大小一样。"""
    p, src, out = _receipt(tmp_path)
    out.write_bytes(b"Y" * 8)          # 同样 8 字节
    ok, why, _ = _verify(p, src)
    assert ok is False
    assert "bytes changed" in why, why


def test_untouched_output_still_passes(tmp_path):
    p, src, _out = _receipt(tmp_path)
    ok, why, _ = _verify(p, src)
    assert ok is True, why


# ------------------------------------------------ ② 颜色证据

def test_changed_colour_tags_are_reported():
    problems = ge.receipt_evidence_problems({
        "color": {"probed": True, "matrix": "bt709", "range": "limited",
                  "primaries": "bt709", "transfer": "bt709",
                  "complete": True, "unknown_fields": []},
        "color_source": {"probed": True, "matrix": "bt601",
                         "range": "limited", "primaries": "bt601",
                         "transfer": "bt709", "complete": True,
                         "unknown_fields": []},
        "color_lost_tags": [],
        "color_changed_tags": [{"field": "matrix", "source": "bt601",
                                "output": "bt709"}],
        "bounds": {"ok": True, "performed": True},
    })
    assert any("changed" in p and "matrix" in p for p in problems), problems


def test_changed_tags_helper_only_flags_known_to_known():
    src = {"matrix": "bt601", "range": "limited", "primaries": None}
    out = {"matrix": "bt709", "range": "limited", "primaries": "bt709"}
    got = ge._changed_tags(src, out)
    assert got == [{"field": "matrix", "source": "bt601", "output": "bt709"}], got
    assert ge._changed_tags({"matrix": None}, {"matrix": "bt709"}) == []


def test_stage3_writes_the_full_colour_evidence():
    src = open(_STAGE3, encoding="utf-8", errors="replace").read()
    i = src.index('".fused_stage34.json"')
    block = src[max(0, i - 4000):i]
    for key in ('"color_source"', '"color_lost_tags"', '"color_changed_tags"',
                '"output_sha256"'):
        assert key in block, f"回执缺少 {key}"


# ------------------------------------------------ ③ 身份冻结顺序

def test_frozen_digest_is_computed_after_the_plan_and_mode():
    src = open(_MAIN, encoding="utf-8", errors="replace").read()
    i_plan = src.index("_plan = _resolve_run_plan(")
    i_mode = src.index('params["chain_mode"]')
    i_frozen = src.index('params["_frozen_param_digest"] = _fz_digest')
    assert i_plan < i_frozen, "冻结早于计划解析"
    assert i_mode < i_frozen, "冻结早于执行模式落定"


def test_frozen_digest_failure_is_fatal_not_empty():
    src = open(_MAIN, encoding="utf-8", errors="replace").read()
    i = src.index('params["_frozen_param_digest"] = _fz_digest')
    block = src[i:i + 1600]
    assert "return 82" in block, "冻结失败必须非零退出"
    assert 'params["_frozen_param_digest"] = ""' not in block, (
        "冻结失败不得退化成空串（那会让校验被跳过）")


def test_frozen_digest_is_frozen_once():
    """全文只能有**一处**赋值 —— 否则会出现"冻结后又被改"。"""
    src = open(_MAIN, encoding="utf-8", errors="replace").read()
    n = len(re.findall(r'params\["_frozen_param_digest"\]\s*=', src))
    assert n == 1, f"_frozen_param_digest 有 {n} 处赋值"


# ------------------------------------------------ N15：人工 fps 解释

import stage3_ai_upscale as s3  # noqa: E402


def test_fps_interpret_alone_is_usable_when_the_probe_fails():
    """只给 `fps_interpret` + 源 probe 失败 → 必须能继续，且标成人工解释。

    旧实现只看 `original_fps` / `fps`，于是这种输入被判成"没给帧率"→ rc=86，
    把最需要人工解释的场景堵死。
    """
    rate, src = s3.resolve_manual_fps({"fps_interpret": "25"})
    assert rate is not None, "只给 fps_interpret 时应当可用"
    assert rate.value == pytest.approx(25.0)
    assert src.endswith("-interpretation"), src


def test_explicit_fps_still_wins_over_interpretation():
    rate, src = s3.resolve_manual_fps({"original_fps": "30000/1001",
                                       "fps_interpret": "25"})
    assert src == "manifest"
    assert rate.value == pytest.approx(30000 / 1001)


def test_user_flag_changes_the_recorded_source():
    _rate, src = s3.resolve_manual_fps({"fps": "24", "_fps_from_user": True})
    assert src == "user"


def test_invalid_interpretation_is_rejected_not_guessed():
    rate, src = s3.resolve_manual_fps({"fps_interpret": "not-a-rate"})
    assert rate is None
    assert src == ""


def test_no_spec_at_all_returns_nothing():
    assert s3.resolve_manual_fps({}) == (None, "")


def test_every_exit_writes_the_timeline_plan():
    """四个出口（AI-off / 流式 / PNG / 原样返回）都必须写计划。"""
    src = open(_STAGE3, encoding="utf-8", errors="replace").read()
    calls = src.count("write_job_timeline_plan(") - 1   # 减去定义本身
    assert calls >= 4, f"只有 {calls} 个出口写了 TimelinePlan"
