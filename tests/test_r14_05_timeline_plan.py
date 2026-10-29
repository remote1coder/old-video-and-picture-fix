# -*- coding: utf-8 -*-
"""R14-05：时间轴的三个结论必须分开，且跨阶段传递。

**审核要求（§6）**

跨阶段保存至少**三个不同**结论：

1. 源时间轴是否已验证；
2. 用户请求的变换是否执行正确；
3. 输出时间轴是否测得。

**特别是**：③ 成立**不能**让 ① 变成已验证。
原实现的 `_fps_source` 只写日志、不跨阶段传递，于是 stage7 测出
"生成物可测"后可能把局部 timing 标成 verified —— 而"生成物能测量"
不等于"未知的源时间轴已被保留"。

本文件验证 `shared_modules/timeline_plan.py` 的判定与往返。
"""
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import timeline_plan as tp  # noqa: E402


# ---------------------------------------------------------------- 构造

def test_cfr_with_full_coverage_is_verified():
    plan = tp.build_plan("j1", "cfr", fps_value="30000/1001",
                         fps_source="probe", pts_usable=300,
                         expected_frames=300)
    assert plan["source_verified"] is True
    assert plan["pts_coverage"]["complete"] is True
    assert plan["manual_interpretation"] is False
    assert plan["assumptions"] == []


def test_unknown_is_not_verified():
    plan = tp.build_plan("j1", "unknown", fps_value="25",
                         fps_source="user-assumed", pts_usable=0,
                         expected_frames=300)
    assert plan["source_verified"] is False
    assert plan["manual_interpretation"] is True
    assert plan["assumptions"], "人工假设必须留下痕迹"


def test_manifest_source_is_not_user_choice():
    """manifest 注入的 fps 不是"用户明确选择"——来源要如实区分。"""
    plan = tp.build_plan("j1", "cfr", fps_value="25",
                         fps_source="manifest-assumed", pts_usable=300,
                         expected_frames=300)
    assert plan["fps"]["source"] == "manifest-assumed"
    assert plan["manual_interpretation"] is True


def test_vfr_is_verified_when_coverage_complete():
    plan = tp.build_plan("j1", "vfr", fps_value=None, fps_source=None,
                         pts_usable=300, expected_frames=300)
    assert plan["source_verified"] is True
    assert plan["fps"] is None, "VFR 没有单一帧率"


# ---------------------------------------------------------------- 唯一判定

def test_output_measured_cannot_upgrade_unknown_source():
    """**核心**：输出测得（③）不能把源未知（①）洗成已验证。"""
    plan = tp.build_plan("j1", "unknown", fps_value="25",
                         fps_source="user-assumed")
    # stage7 测得 packet span，timing_state=verified
    assert tp.claim_source_verified(plan, "verified") is False, \
        "源时间轴未知时，无论输出能否测量，都不能宣称源时序已验证"


def test_claim_true_only_when_both_sides_hold():
    good = tp.build_plan("j1", "cfr", fps_value="25", fps_source="probe",
                         pts_usable=300, expected_frames=300)
    assert tp.claim_source_verified(good, "verified") is True
    # 源已验证，但本次判定不是 verified → 不宣称
    assert tp.claim_source_verified(good, "unverified") is False
    assert tp.claim_source_verified(good, "adjusted") is False
    assert tp.claim_source_verified(good, "unexplained") is False


def test_claim_false_without_plan():
    """没有上游计划 → 无从证明，返回 False（不猜）。"""
    assert tp.claim_source_verified(None, "verified") is False
    assert tp.claim_source_verified({}, "verified") is False
    assert tp.claim_source_verified({"source_verified": "yes"}, "verified") \
        is False, "字符串 'yes' 不是 True"


# ---------------------------------------------------------------- 往返

def test_plan_roundtrip(tmp_path):
    plan = tp.build_plan("job7", "cfr", fps_value="24000/1001",
                         fps_source="probe", pts_usable=120,
                         expected_frames=120)
    path = tp.write_plan(str(tmp_path), plan)
    assert path and os.path.exists(path)

    back = tp.read_plan(str(tmp_path), "job7")
    assert back is not None
    assert back["fps"]["value"] == "24000/1001"
    assert back["source_verified"] is True
    assert tp.claim_source_verified(back, "verified") is True


def test_read_plan_returns_none_for_missing_or_bad(tmp_path):
    assert tp.read_plan(str(tmp_path), "nope") is None

    bad = tmp_path / "jobbad_timeline_plan.json"
    bad.write_text("{not json", encoding="utf-8")
    assert tp.read_plan(str(tmp_path), "jobbad") is None

    wrong_schema = tmp_path / "jobws_timeline_plan.json"
    wrong_schema.write_text(json.dumps({"schema": "other/9"}), encoding="utf-8")
    assert tp.read_plan(str(tmp_path), "jobws") is None, \
        "schema 不符不能当成有效计划"


def test_write_plan_failure_returns_none(tmp_path, monkeypatch):
    """写不出去返回 None，由调用方告警（不静默当成功）。"""
    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(tp, "open", _boom, raising=False)
    plan = tp.build_plan("j", "cfr", fps_value="25", fps_source="probe")
    assert tp.write_plan(str(tmp_path), plan) is None


# ---------------------------------------------------------------- 生产接入

def test_stage3_and_stage7_use_the_shared_module():
    """source guard：写/读两侧都用同一模块，不能各写一份实现。"""
    s3 = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                           "stage3_ai_upscale.py"),
              encoding="utf-8", errors="replace").read()
    s7 = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                           "stage7_encode.py"),
              encoding="utf-8", errors="replace").read()
    assert "from timeline_plan import" in s3
    assert "from timeline_plan import" in s7
    assert "claim_source_verified" in s7, \
        "stage7 必须用共享判定，不能自己写 source_verified and ..."


# ---------------------------------------------------------------- 结论②：变换是否执行

def test_no_transform_request_is_not_applicable():
    """没要求变换 → NOT_APPLICABLE，但**字段齐全**（不是 None）。"""
    v = tp.build_transform_verdict()
    assert v["status"] == "NOT_APPLICABLE"
    assert v["requested"] is False
    assert v["kind"] == "none"
    for k in ("requested", "kind", "expected_ratio", "observed_ratio",
              "tempo_applied", "status", "reason"):
        assert k in v, f"缺字段 {k}"
    assert v["reason"]


def test_declared_and_matching_is_pass():
    v = tp.build_transform_verdict(declared_speed=2.0,
                                   original_duration=10.0,
                                   processed_duration=5.0,
                                   timing_state="verified")
    assert v["status"] == "PASS", v
    assert v["observed_ratio"] == pytest.approx(2.0)
    assert "matches" in v["reason"]


def test_declared_but_output_differs_is_fail():
    """声明 2x 但成品只跑了 1.5x → FAIL（这才是"请求没兑现"）。"""
    v = tp.build_transform_verdict(declared_speed=2.0,
                                   original_duration=10.0,
                                   processed_duration=6.6667,
                                   timing_state="verified")
    assert v["status"] == "FAIL", v
    assert "2.0" in v["reason"] and "off" in v["reason"]


def test_timing_unverified_means_transform_unverified():
    """时序本身没验证 → 变换也不能算已验证（不能凭空认账）。"""
    for st in ("unverified", "unexplained"):
        v = tp.build_transform_verdict(declared_tempo=1.5,
                                       original_duration=10.0,
                                       processed_duration=6.6667,
                                       timing_state=st)
        assert v["status"] == "UNVERIFIED", (st, v)
        assert st in v["reason"]


def test_missing_durations_is_unverified_not_pass():
    v = tp.build_transform_verdict(declared_tempo=1.5,
                                   original_duration=0,
                                   processed_duration=0,
                                   timing_state="verified")
    assert v["status"] == "UNVERIFIED", v
    assert v["observed_ratio"] is None


def test_both_kinds_are_reported():
    v = tp.build_transform_verdict(declared_speed=2.0, declared_tempo=2.0,
                                   original_duration=10.0,
                                   processed_duration=5.0,
                                   timing_state="verified")
    assert "declared_tempo" in v["kind"] and "declared_speed" in v["kind"]


def test_invalid_declared_values_are_ignored():
    for bad in (None, "", "abc", 0, -1):
        v = tp.build_transform_verdict(declared_speed=bad,
                                       original_duration=10.0,
                                       processed_duration=10.0,
                                       timing_state="verified")
        assert v["status"] == "NOT_APPLICABLE", (bad, v)


def test_plan_no_longer_has_a_permanently_empty_field():
    """`build_plan()` 不许再写"声明了永远是 None"的字段。"""
    plan = tp.build_plan("j1", "cfr", fps_value="25", fps_source="probe")
    assert "transform_applied_verified" not in plan, \
        "这个字段必须由 stage7 用 build_transform_verdict 填，不能留 None 占位"
    assert "transform_request" in plan
    assert plan["transform_request"]["fps_reinterpreted"] is False


def test_stage7_fills_the_verdict():
    """source guard：stage7 必须真的调用裁定并写进报告。"""
    s7 = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                           "stage7_encode.py"),
              encoding="utf-8", errors="replace").read()
    assert "build_transform_verdict" in s7, \
        "stage7 必须用共享裁定，不能自己拼"
    assert '"transform_applied_verified": _tv' in s7, \
        "stage7 必须把裁定写进报告的 timeline 段"
    assert "transform_applied_verified\": None" not in s7, \
        "不得再留 None 占位"
