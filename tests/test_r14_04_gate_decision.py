# -*- coding: utf-8 -*-
"""R14-04：检查结果必须**真的**决定是否发布。

**审核复现的三个矛盾**（本文件逐一验证）

1. **WARN 与 FAIL 冲突**：同步起点增量 −120ms、终点 0、跨度 +120ms 时，
   生产 `_validate_av_sync` 返回 **0（允许出片）**，但同一结果被记成
   `CheckResult.FAIL`，汇总得到 `publish=false`。同一次运行两个相反结论。
2. **unknown 早退不填结果**：`presence == "unknown"` 直接 `return 86`，
   `out_check` 仍为空 —— 结果集里少一项，汇总"看不见"它。
3. **授权串用**：sync 的 `allow_sync_fail_publish` 与 verify 的
   `allow_verify_fail_publish` 被当成同一个开关。

**修法**：`status` 与 `severity` 分开；force 按检查项自己的 `force_key`
匹配；所有出口都必须填结果。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check_result as cr  # noqa: E402
import stage7_encode as s7  # noqa: E402


class _Log:
    def __init__(self):
        self.events = []

    def _r(self, lv, ev):
        self.events.append((lv, ev))

    def info(self, ev, msg="", **k):
        self._r("info", ev)

    def warning(self, ev, msg="", **k):
        self._r("warn", ev)

    def error(self, ev, msg="", **k):
        self._r("error", ev)

    def debug(self, *a, **k):
        pass

    def has(self, ev):
        return any(e == ev for _l, e in self.events)


def _patch(monkeypatch, video, audio, priming=(0.0, 0.0), presence="present"):
    """video/audio = (raw_start, span)。"""
    iv = {"v": video, "a": audio}
    pv = {"v": priming[0], "a": priming[1]}
    monkeypatch.setattr(s7, "_probe_audio_presence", lambda p: presence,
                        raising=False)
    monkeypatch.setattr(s7, "_probe_stream_start",
                        lambda p, st: iv[st][0], raising=False)
    monkeypatch.setattr(s7, "_get_stream_duration",
                        lambda p, st: iv[st][1], raising=False)
    monkeypatch.setattr(s7, "_probe_encoder_priming",
                        lambda p, st: pv[st], raising=False)
    monkeypatch.setattr(s7, "_measure_av_drift_vec", lambda p: None,
                        raising=False)


# ---------------------------------------------------------------- G01

def test_g01_warn_band_allows_publish_and_keeps_warning(monkeypatch):
    """审核的 G01 反例：−120ms 漂移落在 WARN 档。

    源 [0,10]/[0,10]，候选视频 [0,10]、音频 [−0.12,10]，priming=0。
    要求：**返回码、sync 记录、summary、发布**四者一致。
    """
    _patch(monkeypatch, video=(0.0, 10.0), audio=(-0.12, 10.0))
    log = _Log()
    checks = {}

    rc = s7._validate_av_sync("out.mkv", log, {}, out_check=checks)

    # 1) 返回码：WARN 档允许出片
    assert rc == 0, f"−120ms 在 WARN 档，应返回 0，实际 {rc}"
    assert log.has("STAGE7_SYNC_WARN"), [e for _l, e in log.events]

    # 2) 记录必须与返回码同向（原来这里记 FAIL）
    rec = checks["sync"]
    assert rec["status"] == cr.PASS, \
        f"返回 0 却记 {rec['status']} —— 这正是审核复现的矛盾"
    assert rec["severity"] == "warning", rec
    assert rec["reason"], "警告必须带原因"

    # 3) 汇总：允许发布，警告跟着走，不算失败
    d = cr.publish_decision([cr.CheckResult(
        "sync", rec["status"], severity=rec["severity"])])
    assert d["publish"] is True
    assert "sync" in d["warnings"], d
    assert d["blocked_by"] == []


def test_g01_level3_is_fail_and_uses_its_own_force_key(monkeypatch):
    """超出失败线（+250ms）→ FAIL，force_key 是 sync 自己的那把。"""
    _patch(monkeypatch, video=(0.0, 10.0), audio=(-0.25, 10.0))
    log = _Log()
    checks = {}
    rc = s7._validate_av_sync("out.mkv", log, {}, out_check=checks)

    assert rc == 86
    rec = checks["sync"]
    assert rec["status"] == cr.FAIL
    assert rec["severity"] == "error"
    assert rec.get("force_key") == "allow_sync_fail_publish", rec


# ---------------------------------------------------------------- 授权不串用

def test_sync_force_does_not_unlock_verify_fail():
    """**sync 的授权不能替代 verify 的授权**（反之亦然）。"""
    sync_fail = cr.CheckResult("sync", cr.FAIL, reason="audio leads 250ms",
                               force_key="allow_sync_fail_publish")
    verify_fail = cr.CheckResult("mechanical", cr.FAIL, reason="duration +3.7%",
                                 force_key="allow_verify_fail_publish")

    # 只开 sync 的钥匙 → sync 可过，verify 仍挡
    d = cr.publish_decision([sync_fail, verify_fail],
                            enabled_forces={"allow_sync_fail_publish"})
    assert d["publish"] is False
    assert "mechanical" in d["blocked_by"], d

    # 只开 verify 的钥匙 → 反过来
    d2 = cr.publish_decision([sync_fail, verify_fail],
                             enabled_forces={"allow_verify_fail_publish"})
    assert d2["publish"] is False
    assert "sync" in d2["blocked_by"], d2

    # 两把都开 → 才全过，且记录 forced_checks
    d3 = cr.publish_decision(
        [sync_fail, verify_fail],
        enabled_forces={"allow_sync_fail_publish",
                        "allow_verify_fail_publish"})
    assert d3["publish"] is True
    assert d3["forced"] is True
    assert set(d3["forced_checks"]) == {"sync", "mechanical"}
    assert d3["verified"] is False, "强制放行不等于验证通过"


# ---------------------------------------------------------------- 缺失结果

def test_sync_unknown_presence_fills_result(monkeypatch):
    """`presence == "unknown"` 这个早退**必须**填结果，不能消失。"""
    _patch(monkeypatch, video=(0.0, 10.0), audio=(0.0, 10.0),
           presence="unknown")
    log = _Log()
    checks = {}
    rc = s7._validate_av_sync("out.mkv", log, {}, out_check=checks)

    assert rc == 86
    assert "sync" in checks, "unknown 早退没有填 out_check（审核复现项）"
    assert checks["sync"]["status"] == cr.UNVERIFIED
    assert checks["sync"]["severity"] == "error"


def test_sync_absent_audio_is_not_applicable(monkeypatch):
    """无音轨 → NOT_APPLICABLE，既不是 PASS 也不是 FAIL。"""
    _patch(monkeypatch, video=(0.0, 10.0), audio=(0.0, 10.0),
           presence="absent")
    log = _Log()
    checks = {}
    rc = s7._validate_av_sync("out.mkv", log, {}, out_check=checks)
    assert rc == 0
    assert checks["sync"]["status"] == cr.NOT_APPLICABLE


def test_unmeasurable_intervals_fills_unverified(monkeypatch):
    """流区间测不出 → UNVERIFIED，且不是 FAIL。"""
    _patch(monkeypatch, video=(0.0, 0.0), audio=(0.0, 10.0))
    log = _Log()
    checks = {}
    rc = s7._validate_av_sync("out.mkv", log, {}, out_check=checks)
    assert rc == 86
    assert checks["sync"]["status"] == cr.UNVERIFIED


# ---------------------------------------------------------------- 严重度语义

def test_severity_is_independent_of_status():
    """status 说"结论"，severity 说"有多严重"，两者不能混。"""
    warn_pass = cr.CheckResult("sync", cr.PASS, severity="warning",
                               reason="120ms within warn band")
    assert warn_pass.is_pass is True
    assert warn_pass.status == cr.PASS

    err_fail = cr.CheckResult("sync", cr.FAIL, severity="error")
    assert err_fail.is_pass is False

    with pytest.raises(ValueError):
        cr.CheckResult("x", cr.PASS, severity="fatal")


def test_all_pass_with_warning_stays_verified():
    """全 PASS（含 warning）→ 可以称 verified，但把 warning 留在报告里。"""
    rs = [cr.CheckResult("mechanical", cr.PASS),
          cr.CheckResult("sync", cr.PASS, severity="warning",
                         reason="120ms")]
    d = cr.publish_decision(rs)
    assert d["publish"] is True
    assert d["verified"] is True
    assert d["warnings"] == ["sync"]


# ---------------------------------------------------------------- 完整结果集

def test_disabled_fullchain_is_not_applicable_not_missing():
    """禁用的必需检查要有**明确表示**，不能从汇总里消失。"""
    rs = [cr.CheckResult("mechanical", cr.PASS),
          cr.CheckResult("sync", cr.PASS),
          cr.CheckResult("frame_order", cr.PASS),
          cr.CheckResult("fullchain", cr.NOT_APPLICABLE,
                         reason="full_chain_verify is disabled")]
    d = cr.publish_decision(rs)
    assert d["publish"] is True
    assert d["verified"] is True, "禁用的是可选检查，不该拖垮 verified"


def test_fullchain_verifier_crash_is_unverified_not_pass():
    """校验器崩了 → UNVERIFIED，不是通过。"""
    r = cr.CheckResult("fullchain", cr.UNVERIFIED,
                       reason="the full-chain verifier raised",
                       severity="error")
    d = cr.publish_decision([r])
    assert d["verified"] is False


def test_fullchain_fail_uses_the_verify_force_key():
    """全链 FAIL 走 `allow_verify_fail_publish`（与机械门同一把钥匙）。"""
    r = cr.CheckResult("fullchain", cr.FAIL, reason="blocked: DURATION",
                       severity="error", force_key="allow_verify_fail_publish")
    assert cr.publish_decision([r])["publish"] is False
    assert cr.publish_decision(
        [r], enabled_forces={"allow_sync_fail_publish"})["publish"] is False
    d = cr.publish_decision(
        [r], enabled_forces={"allow_verify_fail_publish"})
    assert d["publish"] is True and d["forced"] is True


def test_color_without_enforce_is_pass_plus_warning():
    """颜色飘移但未开 enforce → PASS+warning（记录，不阻断）。"""
    r = cr.CheckResult("color", cr.PASS, severity="warning",
                       reason="delta 3.2 > threshold 3.0 (enforce off)")
    d = cr.publish_decision([r])
    assert d["publish"] is True
    assert "color" in d["warnings"]


def test_color_probe_failure_is_unverified():
    """颜色探不出 → UNVERIFIED。"""
    r = cr.CheckResult("color", cr.UNVERIFIED,
                       reason="RGB means could not be probed")
    assert cr.publish_decision([r])["verified"] is False


def test_stage7_records_color_and_fullchain():
    """source guard：stage7 必须把颜色与全链写进 `_checks`。

    行为由上面的四态用例证明；这里守"有没有真的接进去"。
    """
    src = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                       "stage7_encode.py")
    text = open(src, encoding="utf-8", errors="replace").read()
    assert '_checks["fullchain"]' in text, "全链结论未进结果集"
    assert '_color_checks["color"]' in text, "颜色结论未进结果集"
    assert "_checks.update(_color_checks or {})" in text, \
        "颜色结论没有并入 _checks"
