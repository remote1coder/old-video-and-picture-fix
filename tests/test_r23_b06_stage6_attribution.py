# -*- coding: utf-8 -*-
"""R23-B06：stage6 双失败归因 —— 独立原因优先，顺序只作观测。

## 审核给的反例

双进程都失败、退出顺序不可辨；vspipe 留
`havsfunc.GrainFactory3 / BrokenPipeError`，ffmpeg 留
`No space left on device`。

修复前：`failed_stage = "vspipe"`（因为生产者留下了日志）→ 插件能力失败 →
调用一次**救不了磁盘**的 AddGrainC 回退 → dispatcher 返回 0（看起来成功）。

修复后要求：

1. 顺序不可辨时**不许**把 `failed_stage` 写成 `vspipe`（那是观察，不是归因）；
2. 编码器侧的独立原因（磁盘满/权限/只读/OOM/IO 错误）**优先**，
   一票否决"插件缺失"；
3. 真正的插件缺失级联仍要能回退 —— 不能因为收紧而把正常能力判死。
"""
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage6_grain_synthesis as s6  # noqa: E402


def _result(*, failed_stage, vspipe_tail="", encoder_tail="", first_exit="",
            vspipe_rc=1, ffmpeg_rc=1, ok=False, cancelled=False):
    return s6.GrainPipelineResult(
        ok=ok, vspipe_rc=vspipe_rc, ffmpeg_rc=ffmpeg_rc,
        failed_stage=failed_stage, first_exit=first_exit,
        vspipe_tail=vspipe_tail, encoder_tail=encoder_tail,
        cancelled=cancelled)


def test_disk_full_with_a_plugin_looking_producer_trace_is_not_a_plugin_issue():
    """审核原例：生产者有 GrainFactory3 堆栈，编码器说磁盘满 → 不是插件问题。"""
    r = _result(failed_stage="ambiguous",
                vspipe_tail="Traceback in havsfunc.GrainFactory3\n"
                            "BrokenPipeError: broken pipe",
                encoder_tail="No space left on device")
    assert r.encoder_failure_is_independent() is True
    assert r.is_plugin_capability_failure() is False

    calls = []

    def fallback():
        calls.append("called")
        return s6.GrainPipelineResult(ok=True, vspipe_rc=0, ffmpeg_rc=0)

    rc, _second = s6.dispatch_result(r, True, None, run_fallback=fallback)
    assert calls == [], "磁盘满不该触发插件回退"
    assert rc == 84, "独立失败必须原样返回非零，而不是被回退掩盖"


@pytest.mark.parametrize("encoder_tail", [
    "No space left on device",
    "Read-only file system",
    "Permission denied",
    "Cannot allocate memory",
    "Input/output error",
])
def test_every_independent_cause_wins(encoder_tail):
    r = _result(failed_stage="ambiguous",
                vspipe_tail="grainfactory not found in the plugin",
                encoder_tail=encoder_tail)
    assert r.is_plugin_capability_failure() is False


def test_a_genuine_plugin_cascade_still_falls_back():
    """反面对照：真正的插件缺失级联仍要能回退（不能修成永远拒绝）。"""
    r = _result(failed_stage="ambiguous",
                vspipe_tail="AttributeError: No attribute with the name "
                            "grainfactory exists.",
                encoder_tail="Invalid data found when processing input")
    assert r.is_plugin_capability_failure() is True

    calls = []

    def fallback():
        calls.append("called")
        return s6.GrainPipelineResult(ok=True, vspipe_rc=0, ffmpeg_rc=0)

    rc, _ = s6.dispatch_result(r, True, None, run_fallback=fallback)
    assert calls == ["called"]
    assert rc == 0


def test_ordering_alone_is_not_evidence_of_plugin_failure():
    """`first_exit` 只是观测：编码器先退出 + 生产者有插件措辞 → 不算插件问题。"""
    r = _result(failed_stage="ffmpeg", first_exit="ffmpeg",
                vspipe_tail="grainfactory not found in the plugin",
                encoder_tail="Invalid data found when processing input")
    assert r.is_plugin_capability_failure() is False

    r2 = _result(failed_stage="ffmpeg", first_exit="vspipe",
                 vspipe_tail="grainfactory not found in the plugin",
                 encoder_tail="Invalid data found when processing input")
    assert r2.is_plugin_capability_failure() is True


def test_a_bare_traceback_word_is_not_plugin_evidence():
    """裸 `no attribute` / `importerror` 太泛，不算证据（R22-A06 的收紧保留）。"""
    r = _result(failed_stage="ambiguous",
                vspipe_tail="AttributeError: 'NoneType' object has no attribute "
                             "'x'",
                encoder_tail="Invalid data found when processing input")
    assert r.is_plugin_capability_failure() is False


def test_cancellation_is_never_a_plugin_failure():
    r = _result(failed_stage="ambiguous",
                vspipe_tail="grainfactory not found in the plugin",
                encoder_tail="", cancelled=True)
    assert r.is_plugin_capability_failure() is False
