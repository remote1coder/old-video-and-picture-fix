# -*- coding: utf-8 -*-
"""S6→S7 融合：stage6 必须先**预检**脚本能出帧，再交给 stage7。

## 为什么需要

经典路线里 stage6 自己跑 vspipe，插件缺失立刻暴露；融合路线把 vspipe 挪到
stage7，缺失要到 stage7 已把编码器拉起来才暴露 —— 更晚、更难归因，而且
经典路线的 **AddGrainC 回退**在这里就用不上了。
同一个配置换个 `chain.mode` 就少一条能力，这不叫"只是换个执行方式"。

所以融合模式下 stage6 先出 1 帧（`vspipe --start 0 --end 0`）：
* 能出帧 → 写交接单，交给 stage7；
* 出不了帧且判为插件能力问题 → 换 AddGrainC 脚本重试一次（与经典同能力）；
* 还不行 → rc=84，**不把坏脚本交出去**。

## 判据

1. rc=0 且**有输出**才算通过（rc=0 但零字节不是"能出帧"）；
2. rc!=0 → 失败，并把 stderr 尾部带回去；
3. vspipe 起不来（OSError）→ 失败，不能当成插件问题。
"""
import os
import sys

import pytest

import conftest  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_stage6():
    vstages = os.path.join(ROOT, "video_restoration_pipeline", "stages")
    if vstages not in sys.path:
        sys.path.insert(0, vstages)
    import stage6_grain_synthesis as s6          # noqa: E402
    return s6


class _R:
    def __init__(self, rc, out=b"", err=b""):
        self.returncode = rc
        self.stdout = out
        self.stderr = err


def test_preflight_ok_requires_actual_bytes(tmp_path, monkeypatch):
    s6 = _load_stage6()
    vpy = tmp_path / "g.vpy"
    vpy.write_text("clip.set_output()\n", encoding="utf-8")

    monkeypatch.setattr(s6.subprocess, "run",
                        lambda *a, **k: _R(0, b"YUV4MPEG2 W2 H2 F25:1\n"))
    assert s6._fused_preflight(str(vpy), {}) == (True, "")


def test_preflight_rejects_zero_output_even_with_rc0(tmp_path, monkeypatch):
    """rc=0 但一帧都没出来 —— 这不是"能出帧"。"""
    s6 = _load_stage6()
    vpy = tmp_path / "g.vpy"
    vpy.write_text("clip.set_output()\n", encoding="utf-8")
    monkeypatch.setattr(s6.subprocess, "run", lambda *a, **k: _R(0, b""))
    ok, err = s6._fused_preflight(str(vpy), {})
    assert ok is False


def test_preflight_carries_stderr_tail(tmp_path, monkeypatch):
    s6 = _load_stage6()
    vpy = tmp_path / "g.vpy"
    vpy.write_text("clip.set_output()\n", encoding="utf-8")
    msg = (b"AttributeError: No attribute with the name grainfactory exists.")
    monkeypatch.setattr(s6.subprocess, "run", lambda *a, **k: _R(1, b"", msg))
    ok, err = s6._fused_preflight(str(vpy), {})
    assert ok is False
    assert "grainfactory" in err.lower()


def test_preflight_treats_launch_failure_as_not_a_plugin_issue(tmp_path,
                                                               monkeypatch):
    s6 = _load_stage6()
    vpy = tmp_path / "g.vpy"
    vpy.write_text("clip.set_output()\n", encoding="utf-8")

    def _boom(*a, **k):
        raise OSError("vspipe not found")

    monkeypatch.setattr(s6.subprocess, "run", _boom)
    ok, err = s6._fused_preflight(str(vpy), {})
    assert ok is False
    assert "could not start" in err


def test_gf3_missing_message_is_classified_as_plugin_failure():
    """预检失败后要用**同一套**插件判据，不能自己另写一份。"""
    s6 = _load_stage6()
    res = s6.GrainPipelineResult(
        ok=False, failed_stage="vspipe", vspipe_rc=1, ffmpeg_rc=0,
        vspipe_tail=("AttributeError: No attribute with the name "
                     "grainfactory exists."))
    assert res.is_plugin_capability_failure() is True


def test_disk_full_is_not_a_plugin_failure():
    """磁盘满不能触发 AddGrainC 回退（换插件救不了磁盘）。"""
    s6 = _load_stage6()
    res = s6.GrainPipelineResult(
        ok=False, failed_stage="ffmpeg", vspipe_rc=1, ffmpeg_rc=1,
        vspipe_tail="No space left on device")
    assert res.is_plugin_capability_failure() is False
