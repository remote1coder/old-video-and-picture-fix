# -*- coding: utf-8 -*-
"""manifest 编辑器「人工确认」入口的回归测试。

背景（2026-10-24）：古董手机素材被判成 ANALOG_TAPE 时置信度上不去
（分类器兜底分支固定给 0.70，interlaced 还会被 `min(conf, fo+0.1)` 再压），
批量处理时一半素材会被 main.py 的 0.80 门禁 SKIP。

而"人工确认"这条路原本是**断的**，两处：
    1. `needs_confirmation` 只有分类器会写，全库没有任何地方清除它；
    2. main.py 的门禁是两个条件的**或**：
           if needs_confirmation or confidence < 0.80:
       所以光清 flag 不够，0.55 的 job 照样被拦。

`confirm_job()` 一次修掉两处：清 flag + 把置信度抬到过线。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "shared_modules"))

import manifest_editor as me  # noqa: E402
from manifest_loader import save_manifest  # noqa: E402


def _mk_manifest(path):
    m = {"meta": {"total_jobs": 3, "status_counts": {}},
         "jobs": [
             {"job_id": "job_001", "status": "DONE", "detected_class": "HDV_AVCHD",
              "confidence": 0.90, "needs_confirmation": False},
             {"job_id": "job_002", "status": "SKIP", "detected_class": "ANALOG_TAPE",
              "confidence": 0.55, "needs_confirmation": True},
             {"job_id": "job_003", "status": "SKIP", "detected_class": "ANALOG_TAPE",
              "confidence": 0.70, "needs_confirmation": True},
         ]}
    save_manifest(m, str(path))
    return str(path)


@pytest.fixture()
def mpath(tmp_path):
    return _mk_manifest(tmp_path / "manifest.yaml")


def _gate_blocked(job):
    """复现 main.py 的门禁判定。"""
    conf = job.get("confidence", 0.0)
    return bool(job.get("needs_confirmation", False)) or conf < 0.80


def test_confirm_clears_flag_and_raises_confidence(mpath):
    n = me.confirm_job(mpath, "job_002")
    assert n == 1
    from manifest_loader import load_manifest
    j = [x for x in load_manifest(mpath)["jobs"]
         if x["job_id"] == "job_002"][0]
    assert j["needs_confirmation"] is False
    assert j["confidence"] >= 0.80
    assert j["manual_confirmed"] is True
    assert j["confirmed_at"]
    # 门禁必须放行
    assert _gate_blocked(j) is False


def test_confirm_all_low_confidence(mpath):
    n = me.confirm_job(mpath, all_low_confidence=True)
    assert n == 2   # 只有 job_002 / job_003 低于门禁
    from manifest_loader import load_manifest
    jobs = load_manifest(mpath)["jobs"]
    assert [j["job_id"] for j in jobs if _gate_blocked(j)] == []


def test_high_confidence_job_is_not_touched(mpath):
    """已经过线的 job 不该被 --all-low-confidence 选中。"""
    n = me.confirm_job(mpath, all_low_confidence=True)
    assert n == 2
    from manifest_loader import load_manifest
    j = [x for x in load_manifest(mpath)["jobs"]
         if x["job_id"] == "job_001"][0]
    assert j.get("manual_confirmed") is None     # 未被动过
    assert j["confidence"] == 0.90               # 未被抬高


def test_confirm_unknown_job_returns_zero(mpath):
    assert me.confirm_job(mpath, "nope") == 0


def test_confirm_without_target_returns_zero(mpath):
    assert me.confirm_job(mpath) == 0


def test_confidence_override(mpath):
    me.confirm_job(mpath, "job_002", confidence=0.95)
    from manifest_loader import load_manifest
    j = [x for x in load_manifest(mpath)["jobs"]
         if x["job_id"] == "job_002"][0]
    assert j["confidence"] == 0.95


def test_confidence_override_clamped(mpath):
    """越界值要夹到 [0,1]，不能写进非法置信度。"""
    me.confirm_job(mpath, "job_002", confidence=5.0)
    from manifest_loader import load_manifest
    j = [x for x in load_manifest(mpath)["jobs"]
         if x["job_id"] == "job_002"][0]
    assert j["confidence"] == 1.0


def test_confirm_resets_skip_to_pending(mpath):
    """确认必须把 SKIP 复位成 PENDING，否则批次照样跳过、白确认一场。

    scan 对低置信度 job 直接标 SKIP，confirm 只改 confidence/flag，
    status 仍是 SKIP → "Processing 0 video jobs"。实测连踩三次。
    """
    from manifest_loader import load_manifest
    # 先把两个低置信度 job 标成 SKIP（模拟 scan 的行为）
    m = load_manifest(mpath)
    for j in m["jobs"]:
        if j["job_id"] in ("job_002", "job_003"):
            j["status"] = "SKIP"
    save_manifest(m, mpath)

    assert [j["status"] for j in load_manifest(mpath)["jobs"]
            if j["job_id"] == "job_002"] == ["SKIP"]

    n = me.confirm_job(mpath, all_low_confidence=True)
    assert n == 2

    m2 = load_manifest(mpath)
    for j in m2["jobs"]:
        if j["job_id"] in ("job_002", "job_003"):
            assert j["status"] == "PENDING", \
                f"{j['job_id']} 确认后仍是 {j['status']}"
            assert _gate_blocked(j) is False
    # 已过线的 job 状态不该被动
    assert [j["status"] for j in m2["jobs"]
            if j["job_id"] == "job_001"] == ["DONE"]


def test_confirm_all_low_confidence_is_idempotent(mpath):
    """连跑两次 confirm，第二次也必须把 SKIP 复位。

    第一轮把 confidence 抬过线后，`--all-low-confidence` 若只按
    "当前低于门禁" 选目标，第二轮就选不中任何 job，SKIP→PENDING 复位失效
    （实测：第二次 "Confirmed 0"，status 仍是 SKIP，批次 "Processing 0 jobs"）。
    """
    from manifest_loader import load_manifest
    # 真实场景：**上一轮已经确认过**（confidence 抬过线、manual_confirmed=True），
    # 但 status 仍是 SKIP（例如那次 confirm 被 SIGPIPE 打断，或跑在修复之前）。
    m = load_manifest(mpath)
    for j in m["jobs"]:
        if j["job_id"] in ("job_002", "job_003"):
            j["status"] = "SKIP"
            j["confidence"] = 0.80
            j["needs_confirmation"] = False
            j["manual_confirmed"] = True
    save_manifest(m, mpath)

    # 此时它们已不过门禁，但 status 还卡在 SKIP —— confirm 必须仍能救出来
    assert me.confirm_job(mpath, all_low_confidence=True) == 2

    m2 = load_manifest(mpath)
    for j in m2["jobs"]:
        if j["job_id"] in ("job_002", "job_003"):
            assert j["status"] == "PENDING"
