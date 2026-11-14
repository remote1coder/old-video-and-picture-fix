# -*- coding: utf-8 -*-
"""两遍绑定：正式遍必须拿**自己的**像素摘要与检查遍比。

《设计 v1》§5.1：「正式遍……累计同边界有序像素、时间和颜色摘要，验证与
检查遍一致，再复用内容绑定的检查结论。只比帧数、计划和尺寸不够。」

这里钉住四种情形（都不需要 GPU）：
1. 关掉开关 → 明确记"这一代没做比对"，不能静默；
2. 没有检查遍侧车 → 拒绝，不能"没有反例就当通过"；
3. 侧车跑的是**另一份脚本** → 拒绝（身份不同，结论不可携带）；
4. 拿得到但**摘要不同** → 拒绝（这就是 G8 要的那个反例）。
"""
import json
import os
import sys

import pytest

import conftest  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "video_restoration_pipeline", "stages"),
           os.path.join(ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage3_ai_upscale as S3   # noqa: E402
import stream_digest as REAL_SD  # noqa: E402  （必须先抓住真模块，
# 否则 monkeypatch 掉 sys.modules 之后 import 回来的还是假的那个）


class _Log:
    def __init__(self):
        self.rows = []

    def _add(self, level, ev, msg="", **kw):
        self.rows.append((level, ev, msg))

    def info(self, ev, msg="", **kw): self._add("INFO", ev, msg)
    def warning(self, ev, msg="", **kw): self._add("WARN", ev, msg)
    def error(self, ev, msg="", **kw): self._add("ERROR", ev, msg)

    def has(self, ev):
        return any(r[1] == ev for r in self.rows)


def _vpy(tmp_path):
    p = tmp_path / "a.vpy"
    p.write_text("clip.set_output()\n", encoding="utf-8")
    return str(p)


def _side(tmp_path, vpy, digest="a" * 64):
    d = tmp_path / "temp"
    d.mkdir(exist_ok=True)
    doc = {"schema": "stream_s1_check/1",
           "vs_script": os.path.abspath(vpy),
           "pixels": {"ok": True, "digest": digest, "frames": 3,
                      "pix_fmt": "yuv420p16le"}}
    f = d / "jobX_clip_step1_check.json"
    f.write_text(json.dumps(doc), encoding="utf-8")
    return str(d)


def test_disabled_is_loud_not_silent(tmp_path):
    log = _Log()
    ok, why = S3._two_pass_check(str(tmp_path), "jobX", "clip.mp4",
                                 _vpy(tmp_path),
                                 {"stream_two_pass_check": False}, log)
    assert ok is True and "skipped" in why
    assert log.has("STAGE3_TWO_PASS_SKIPPED"), "关掉时必须留痕"


def test_missing_sidecar_is_refused(tmp_path):
    log = _Log()
    ok, why = S3._two_pass_check(str(tmp_path), "jobX", "clip.mp4",
                                 _vpy(tmp_path), {}, log)
    assert ok is False
    assert log.has("STAGE3_TWO_PASS_NO_CHECK")


def test_other_script_is_refused(tmp_path):
    vpy = _vpy(tmp_path)
    tdir = _side(tmp_path, vpy)
    log = _Log()
    ok, why = S3._two_pass_check(tdir, "jobX", "clip.mp4",
                                 str(tmp_path / "other.vpy"), {}, log)
    assert ok is False
    assert log.has("STAGE3_TWO_PASS_OTHER_SCRIPT")


def test_digest_mismatch_blocks_publication(tmp_path, monkeypatch):
    """G8 的反例：故意让第二遍像素不同，必须拒绝，不能只比帧数就放行。"""
    vpy = _vpy(tmp_path)
    tdir = _side(tmp_path, vpy, digest="b" * 64)

    class _SD:
        @staticmethod
        def digest_vpy(*a, **k):
            return {"ok": True, "digest": "c" * 64, "frames": 3,
                    "pix_fmt": "yuv420p16le", "error": ""}

        compare = staticmethod(REAL_SD.compare)

    monkeypatch.setitem(sys.modules, "stream_digest", _SD)
    log = _Log()
    ok, why = S3._two_pass_check(tdir, "jobX", "clip.mp4", vpy, {}, log)
    assert ok is False, "摘要不同却放行了"
    assert log.has("STAGE3_TWO_PASS_MISMATCH")


def test_digest_match_passes(tmp_path, monkeypatch):
    vpy = _vpy(tmp_path)
    tdir = _side(tmp_path, vpy, digest="d" * 64)

    class _SD:
        @staticmethod
        def digest_vpy(*a, **k):
            return {"ok": True, "digest": "d" * 64, "frames": 3,
                    "pix_fmt": "yuv420p16le", "error": ""}

        compare = staticmethod(REAL_SD.compare)

    monkeypatch.setitem(sys.modules, "stream_digest", _SD)
    log = _Log()
    ok, why = S3._two_pass_check(tdir, "jobX", "clip.mp4", vpy, {}, log)
    assert ok is True, why
    assert log.has("STAGE3_TWO_PASS_OK")
