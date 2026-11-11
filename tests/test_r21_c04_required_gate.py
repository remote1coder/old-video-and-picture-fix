# -*- coding: utf-8 -*-
"""R21-C04：必需时间检查没有真正阻断发布。

## 审核原文（R20 §3，P1）

> 映射函数在缺 PTS、数据不足或 probe 异常时可以返回 True，同时把
> `mapping_ok` 留为 None。已知映射失败在 `strict=false` 时也可能仅记录日志，
> 没有作为必需检查进入统一发布门。新加入的 `final_timeline` 主要看时长是否
> 大于零，这不能证明变速、帧数、原点和映射正确。
>
> 本轮直接调用生产 `publish_decision`，仅给 `final_timeline=UNVERIFIED`，
> 得到 `publish=true / verified=false / forced=false`。

## 根因

`publish_decision` 把 `UNVERIFIED` 一律当作"不阻断"—— 这条规则对**可选**
的质量提示是合理的（静态画面让帧序方法失效，不该判整个作业失败），
但对**必需**的完整性/时间门就是"测不出来等于通过"。

## 修法

给 `CheckResult` 加 `required`。必需项的 `UNVERIFIED` 与 `FAIL` 同样阻断
（可被该项自己的 `force_key` 强制，强制后 `verified=False` 并留痕）。
可选质量项保持原行为。
"""
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

import check_result as CR  # noqa: E402


def _r(name, status, required=False, force_key=None, severity="info"):
    return CR.CheckResult(name, status, reason=f"{name} {status}",
                          method="test", severity=severity,
                          force_key=force_key, required=required)


# ------------------------------------------------ 必需项必须阻断

def test_required_unverified_blocks_publish():
    """审核的原反例：只有 final_timeline=UNVERIFIED → 必须 publish=False。"""
    d = CR.publish_decision([_r("final_timeline", CR.UNVERIFIED, required=True)],
                            enabled_forces=set())
    assert d["publish"] is False, d
    assert "final_timeline" in d["blocked_by"]
    assert d["verified"] is False


def test_required_unverified_can_be_forced_with_its_own_key():
    d = CR.publish_decision(
        [_r("final_timeline", CR.UNVERIFIED, required=True,
            force_key="allow_verify_fail_publish")],
        enabled_forces={"allow_verify_fail_publish"})
    assert d["publish"] is True, d
    assert d["verified"] is False
    assert d["forced"] is True
    assert "final_timeline" in d["forced_checks"]


def test_required_unverified_ignores_an_unrelated_force_key():
    """别的检查的钥匙不能放行这一项。"""
    d = CR.publish_decision(
        [_r("final_timeline", CR.UNVERIFIED, required=True,
            force_key="allow_verify_fail_publish")],
        enabled_forces={"allow_sync_fail_publish"})
    assert d["publish"] is False, d


def test_optional_unverified_still_publishes_but_not_verified():
    """可选质量项保持原行为：不阻断，但 verified=False。"""
    d = CR.publish_decision(
        [_r("frame_order", CR.UNVERIFIED, required=False, severity="warning"),
         _r("mechanical", CR.PASS)],
        enabled_forces=set())
    assert d["publish"] is True, d
    assert d["verified"] is False


def test_all_pass_is_verified():
    d = CR.publish_decision([_r("mechanical", CR.PASS),
                             _r("final_timeline", CR.PASS, required=True)],
                            enabled_forces=set())
    assert d["publish"] is True and d["verified"] is True


def test_required_flag_survives_as_dict():
    d = _r("final_timeline", CR.PASS, required=True).as_dict()
    assert d.get("required") is True
    d2 = _r("frame_order", CR.PASS).as_dict()
    assert "required" not in d2


# ------------------------------------------------ stage7 接线

def test_stage7_marks_the_time_check_required():
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage7_encode.py"),
               encoding="utf-8", errors="replace").read()
    i = src.index('_checks["final_timeline"]')
    block = src[i:i + 2200]
    assert '"required": True' in block, "final_timeline 必须是必需项"
    # 两个分支（PASS / UNVERIFIED）都要带
    assert block.count('"required": True') >= 2, block[:600]


def test_stage7_passes_required_into_check_result():
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage7_encode.py"),
               encoding="utf-8", errors="replace").read()
    assert "required=bool(_d.get(\"required\"))" in src, (
        "决策门拿不到 required，必需性就白标了")


def test_transform_check_is_required_only_when_requested():
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage7_encode.py"),
               encoding="utf-8", errors="replace").read()
    assert '"required": bool(_tv.get("requested"))' in src


# ------------------------------------------------ 映射校验的四态（C04 尾）

import stage7_encode as s7  # noqa: E402


def _patch_pts(monkeypatch, src=None, act=None, tb=0.001, raises=False):
    import vfr_timecodes as vt
    if raises:
        def _boom(path):
            raise RuntimeError("probe exploded")
        monkeypatch.setattr(vt, "probe_pts", lambda p: (_ for _ in ()).throw(
            RuntimeError("probe exploded")))
    else:
        def _pts(path):
            return list(src if "src" in str(path) else (act or []))
        monkeypatch.setattr(vt, "probe_pts", _pts)
    monkeypatch.setattr(vt, "probe_time_base", lambda p: tb)


def test_identity_transform_is_not_applicable(monkeypatch):
    _patch_pts(monkeypatch, src=[0.0, 0.04], act=[0.0, 0.04])
    state, info = s7.check_declared_timeline_mapping({}, "src.mkv", "cand.mkv")
    assert state == "not_applicable"
    assert info["mapping_ok"] is None


def test_declared_transform_without_source_pts_is_unverified(monkeypatch):
    _patch_pts(monkeypatch, src=[], act=[0.0, 0.04])
    state, info = s7.check_declared_timeline_mapping(
        {"expected_speed": 2.0}, "src.mkv", "cand.mkv")
    assert state == "unverified", info
    assert info["mapping_ok"] is None


def test_declared_transform_without_candidate_pts_is_unverified(monkeypatch):
    _patch_pts(monkeypatch, src=[0.0, 0.04, 0.08], act=[])
    state, info = s7.check_declared_timeline_mapping(
        {"expected_speed": 2.0}, "src.mkv", "cand.mkv")
    assert state == "unverified", info


def test_probe_exception_is_unverified_not_pass(monkeypatch):
    _patch_pts(monkeypatch, raises=True)
    state, info = s7.check_declared_timeline_mapping(
        {"expected_speed": 2.0}, "src.mkv", "cand.mkv")
    assert state == "unverified", info
    assert "raised" in info.get("mapping_reason", "")


@pytest.mark.parametrize("state,status,required,force_key", [
    ("pass", "PASS", True, None),
    ("fail", "FAIL", True, "allow_verify_fail_publish"),
    ("unverified", "UNVERIFIED", True, "allow_verify_fail_publish"),
    ("not_applicable", "NOT_APPLICABLE", False, None),
])
def test_mapping_entry_maps_each_state(state, status, required, force_key):
    e = s7.mapping_check_entry(state, {"mapping_problems": ["x"]})
    assert e["status"] == status
    assert e["required"] is required
    assert e.get("force_key") == force_key


def test_fail_or_unverified_mapping_blocks_the_gate():
    for state in ("fail", "unverified"):
        e = s7.mapping_check_entry(state, {})
        r = CR.CheckResult(e["name"], e["status"], reason="x",
                           required=e["required"],
                           force_key=e.get("force_key"),
                           severity=e.get("severity", "info"))
        d = CR.publish_decision([r], enabled_forces=set())
        assert d["publish"] is False, (state, d)


def test_stage7_writes_the_mapping_check_into_the_gate():
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage7_encode.py"),
               encoding="utf-8", errors="replace").read()
    assert src.count('_checks["timeline_mapping"] = mapping_check_entry(') == 2, (
        "两个分支都必须把映射结论写进统一发布门")
    assert "_map_ok, _map_info" not in src, "仍在用旧的布尔返回"
