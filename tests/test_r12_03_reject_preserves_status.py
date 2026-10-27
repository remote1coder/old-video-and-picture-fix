# -*- coding: utf-8 -*-
"""R12-03：提交前的拒绝不得覆盖旧代的正式 status。

**缺陷**

旧事务 needs_human、本次颜色守卫拒绝、备份失败这些"尚未提交"的分支，
调用 `_fail()` 时没有 `preserve_valid_status=True`。默认分支写正式
`status.json`，于是旧媒体保留了，**正式状态却变成当前尝试的 FAILED**。

后果有两层：
1. 一份仍然有效的旧成品被标成失败 → 下次运行可能误判要重跑，甚至清理掉它；
2. needs_human 场景下，未解决 journal 仍保留原 status 摘要 —— 改写 status
   等于毁掉一条本可用于恢复的证据。

**修法**：提交前失败一律 `preserve_valid_status=True`，诊断写 attempt 级侧车。
"""
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "photo_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage6_encode as s6  # noqa: E402


def test_preserve_valid_status_does_not_touch_fixed_status(tmp_path):
    """preserve=True：正式 status 原样保留，诊断写到 attempt 侧车。"""
    status = tmp_path / "out.avif.status.json"
    original = {"status": "SUCCESS", "stage": 6, "published": True}
    status.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    before = status.read_bytes()

    s6._fail(None, str(status), "color_drift", "delta too large",
             paths=[], preserve_valid_status=True)

    assert status.read_bytes() == before, "正式 status 被覆盖了"

    attempts = [p for p in os.listdir(str(tmp_path))
                if ".attempt-" in p and p.endswith(".json")]
    assert attempts, "应留下 attempt 级诊断侧车"
    doc = json.loads((tmp_path / attempts[0]).read_text(encoding="utf-8"))
    assert doc["status"] == "FAILED"
    assert doc["failed_operation"] == "color_drift"


def test_without_preserve_overwrites_status(tmp_path):
    """反例：不传 preserve 时会覆盖（证明这条开关确实起作用）。"""
    status = tmp_path / "out.avif.status.json"
    status.write_text(json.dumps({"status": "SUCCESS"}), encoding="utf-8")

    s6._fail(None, str(status), "commit", "boom", paths=[])

    doc = json.loads(status.read_text(encoding="utf-8"))
    assert doc["status"] == "FAILED"


def test_recovery_required_still_overwrites(tmp_path):
    """recovery_required 是例外：磁盘处于未知混合代，必须标记阻止误用。"""
    status = tmp_path / "out.avif.status.json"
    status.write_text(json.dumps({"status": "SUCCESS"}), encoding="utf-8")

    s6._fail(None, str(status), "rollback_failed", "partial restore",
             paths=[], recovery_required=True)

    doc = json.loads(status.read_text(encoding="utf-8"))
    assert doc["status"] == "RECOVERY_REQUIRED"


def test_precommit_rejections_use_preserve():
    """源码护栏：提交前四类拒绝都必须带 preserve_valid_status=True。

    （行为测试需要跑完整个 main()，这里锁住调用形态；
    上面的 test_preserve_valid_status_does_not_touch_fixed_status
    已用生产函数验证了该开关的真实效果。）
    """
    src = os.path.join(_ROOT, "photo_restoration_pipeline", "stages",
                       "stage6_encode.py")
    text = open(src, encoding="utf-8", errors="replace").read()
    for op in ("prior_publish_recovery", "color_drift",
               "publish_backup", "publish_staging"):
        i = text.find(f'_fail(logger, status_path, "{op}"')
        assert i != -1, f"找不到 {op} 的 _fail 调用"
        window = text[i:i + 700]
        assert "preserve_valid_status=True" in window, \
            f"{op} 的拒绝分支缺少 preserve_valid_status=True"
