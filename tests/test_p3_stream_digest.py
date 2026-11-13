# -*- coding: utf-8 -*-
"""两遍绑定的像素摘要（设计 §5.1 / G8）。

> 正式遍：从原源重算同一 stage1……累计**同边界有序像素、时间和颜色摘要**，
> 验证与检查遍一致，再复用内容绑定的检查结论。**只比帧数、计划和尺寸不够。**

所以这一层要能回答三件事：
1. 两遍比的是**同一个域**（尺寸/像素格式/时间基），不是两串数字；
2. 某一遍没有摘要时 → **不可比较**，绝不能当成"一致"；
3. 非确定性渲染下，摘要相同只能说"这一次一致"，不能证明"可复现"。
"""
import os

import pytest

import conftest  # noqa: F401

import stream_digest as SD


def test_missing_script_is_not_a_digest():
    r = SD.digest_vpy("E:/nope/none.vpy")
    assert r["ok"] is False
    assert r["digest"] is None, "不能返回空串冒充摘要"
    assert "gone" in r["error"]


def test_compare_rejects_missing_digest():
    c = {"ok": False, "digest": None, "frames": 0}
    f = {"ok": True, "digest": "abc", "frames": 10}
    r = SD.compare(c, f, deterministic=True)
    assert r["match"] is None and r["comparable"] is False
    assert "no digest" in r["reason"]


def test_compare_detects_frame_count_difference():
    c = {"ok": True, "digest": "abc", "frames": 10}
    f = {"ok": True, "digest": "abc", "frames": 11}
    r = SD.compare(c, f, deterministic=True)
    assert r["match"] is False and r["comparable"] is True
    assert "frame count" in r["reason"]


def test_compare_equal_digests_are_only_proof_when_deterministic():
    c = {"ok": True, "digest": "same", "frames": 10}
    f = {"ok": True, "digest": "same", "frames": 10}
    det = SD.compare(c, f, deterministic=True)
    assert det["match"] is True and "not reproducible" not in det["reason"]

    nondet = SD.compare(c, f, deterministic=False)
    assert nondet["match"] is True
    assert "not reproducible" in nondet["reason"], nondet["reason"]


def test_compare_reports_different_pixels():
    c = {"ok": True, "digest": "aaa", "frames": 10}
    f = {"ok": True, "digest": "bbb", "frames": 10}
    r = SD.compare(c, f, deterministic=True)
    assert r["match"] is False
    assert "differ" in r["reason"]


def test_stage2_stream_records_binding_before_returning():
    """检查遍必须在 return 之前写绑定侧车，写不出来就失败（不能静默跳过）。"""
    import io
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    s = io.open(os.path.join(root, "video_restoration_pipeline", "stages",
                             "stage2_traditional_check.py"),
                encoding="utf-8", errors="replace").read()
    i = s.index("if _vpy:")
    i_copy = s.index("compute_param_hash(stage1_output, params)")
    block = s[i:i_copy]
    assert "stream_s1_check/1" in block, "没有写绑定侧车"
    assert "STAGE2_STREAM_BINDING_FAIL" in block, "写不出来时必须失败"
    assert "return 84" in block
