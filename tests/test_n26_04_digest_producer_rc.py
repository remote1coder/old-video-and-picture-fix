# -*- coding: utf-8 -*-
"""`digest_vpy` 必须合取**所有生产者**的退出码（R25-06 / PA03）。

复核反例：vspipe 退出 17（脚本求值失败），下游 ffmpeg 退出 0 并产出一帧
→ 旧实现 `ok=True`，把"一帧"当成"整片摘要"。

为什么这在生产上危险：两遍绑定拿这个摘要去比。截断的摘要帧数少，
如果另一侧也恰好少（比如同一脚本同样失败），两边"相同"就成立 ——
而那正是"两边都没跑完"。
"""
import io
import os
import subprocess
import sys
import tempfile

import pytest

import conftest  # noqa: F401

import stream_digest as SD


class _FakeVP:
    def __init__(self, rc):
        self.returncode = rc
        self.stdout = io.BytesIO(b"")

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


def test_vspipe_failure_is_not_a_valid_digest(monkeypatch, tmp_path):
    """vspipe 退出 17 + ffmpeg 成功 → 必须 ok=False。"""
    vpy = tmp_path / "x.vpy"
    vpy.write_text("x", encoding="utf-8")

    class _FakeRun:
        returncode = 0
        stdout = ("#format: frame checksums\n"
                  "#dimensions 0: 64x48\n"
                  "#tb 0: 1/10\n"
                  "0, 0, 0, 1, 4608, 00\n")
        stderr = ""

    def _fake_popen(*a, **kw):
        return _FakeVP(17)

    monkeypatch.setattr(SD.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(SD.subprocess, "run", lambda *a, **kw: _FakeRun())

    out = SD.digest_vpy(str(vpy), pix_fmt="yuv420p16le")
    assert out["ok"] is False, (
        "vspipe 退出 17 却返回 ok=True —— 截断的流被当成整片摘要")
    assert "17" in (out["error"] or ""), out["error"]


def test_both_producers_ok_still_works(monkeypatch, tmp_path):
    """两边都 0 → 正常出摘要（防过度收紧）。"""
    vpy = tmp_path / "x.vpy"
    vpy.write_text("x", encoding="utf-8")

    class _FakeRun:
        returncode = 0
        stdout = ("#format: frame checksums\n"
                  "#dimensions 0: 64x48\n"
                  "#tb 0: 1/10\n"
                  "0, 0, 0, 1, 4608, aa\n"
                  "0, 1, 1, 1, 4608, bb\n")
        stderr = ""

    monkeypatch.setattr(SD.subprocess, "Popen", lambda *a, **kw: _FakeVP(0))
    monkeypatch.setattr(SD.subprocess, "run", lambda *a, **kw: _FakeRun())

    out = SD.digest_vpy(str(vpy), pix_fmt="yuv420p16le")
    assert out["ok"] is True, out.get("error")
    assert out["frames"] == 2
