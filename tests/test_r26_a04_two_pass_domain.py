# -*- coding: utf-8 -*-
"""两遍绑定必须比**同一域的实际流**（R26-A04）。

复核的四个反例：
  ① 同 10 帧、摘要完全不同 → 旧实现只比帧数，绑定成功；
  ② 任意一端 0 帧 → 也通过；
  ③ 消费侧车缺失 → 退回额外重放，再次通过；
  ④ 短帧/截断 → 不报。

现在：检查遍（stage2）与正式遍（stage3）各在 **y4m** 上过 tap，
`_two_pass_check` 直接比这两份同域摘要。
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared_modules"))
sys.path.insert(0, str(ROOT / "video_restoration_pipeline" / "stages"))

import stage3_ai_upscale as s3


class _Log:
    def __init__(self):
        self.events = []

    def info(self, ev, msg, **kw): self.events.append(("I", ev))
    def warning(self, ev, msg, **kw): self.events.append(("W", ev))
    def error(self, ev, msg, **kw): self.events.append(("E", ev))
    def critical(self, ev, msg, **kw): self.events.append(("C", ev))

    def names(self):
        return [e[1] for e in self.events]


@pytest.fixture()
def scene(tmp_path):
    """造出 `_two_pass_check` 需要的最小现场。"""
    temp = tmp_path / "temp"
    temp.mkdir()
    vpy = str(tmp_path / "x.vpy")
    Path(vpy).write_text("x", encoding="utf-8")
    out = str(tmp_path / "out" / "clip.mkv")
    Path(out).parent.mkdir()

    # stage1 的检查侧车（`_two_pass_check` 先核 vs_script）
    chk_doc = {"vs_script": os.path.abspath(vpy),
               "pixels": {"digest": "a" * 32, "frames": 20,
                          "pix_fmt": "yuv420p16le"}}
    Path(temp / "j1_clip_step1_check.json").write_text(
        json.dumps(chk_doc), encoding="utf-8")
    return {"temp": str(temp), "vpy": vpy, "out": out}


def _write_tap(scene, *, c_digest="b" * 64, c_frames=20,
               f_digest="b" * 64, f_frames=20, c_bytes=92160, f_bytes=92160):
    Path(scene["temp"], "j1_clip_step2_check_tap.json").write_text(
        json.dumps({"schema": "check-pass-tap/1", "digest": c_digest,
                    "frames": c_frames, "bytes": c_bytes,
                    "producer_rc": 0, "consumer_rc": 0,
                    "short_frame": False}),
        encoding="utf-8")
    Path(scene["out"] + ".formal_tap.json").write_text(
        json.dumps({"ok": True, "digest": f_digest, "frames": f_frames,
                    "bytes": f_bytes, "producer_rc": 0, "consumer_rc": 0,
                    "short_frame": False}),
        encoding="utf-8")


def _run(scene, log=None):
    return s3._two_pass_check(scene["temp"], "j1", "clip.mp4", scene["vpy"],
                              {"stream_two_pass_check": True}, log or _Log(),
                              output_mkv=scene["out"])


def test_same_domain_equal_accepts(scene):
    _write_tap(scene)
    ok, why = _run(scene)
    assert ok is True, why
    assert "same-domain" in why


def test_same_frame_count_but_different_content_rejects(scene):
    """① 同帧数、摘要不同 → 必须拒（这正是旧实现放过的）。"""
    _write_tap(scene, f_digest="c" * 64)
    log = _Log()
    ok, why = _run(scene, log)
    assert ok is False, why
    assert "STAGE3_TWO_PASS_TAP_MISMATCH" in log.names()


def test_zero_frames_rejects(scene):
    """② 0 帧 → 不是有效证据。"""
    _write_tap(scene, c_frames=0, f_frames=0, c_bytes=0, f_bytes=0)
    ok, why = _run(scene)
    assert ok is False, why
    assert "incomplete" in why.lower() or "0-frame" in why.lower()


def test_missing_formal_tap_falls_back_not_auto_pass(scene):
    """③ 正式遍 tap 缺失 → 不能"因为 0 对比就算通过"。"""
    Path(scene["temp"], "j1_clip_step2_check_tap.json").write_text(
        json.dumps({"digest": "b" * 64, "frames": 20}), encoding="utf-8")
    log = _Log()
    ok, why = _run(scene, log)
    # 没 formal tap 时会走旧退路（AI 输入摘要或旁路重放）；这里两者都没有，
    # 所以必须**失败**，不能因为"没东西可比"就放过。
    assert ok is False, why


def test_short_frame_flag_rejects(scene):
    """④ 短帧标记 → 拒（半截的相同比"不同"更危险）。"""
    Path(scene["temp"], "j1_clip_step2_check_tap.json").write_text(
        json.dumps({"digest": "b" * 64, "frames": 20,
                    "short_frame": True}), encoding="utf-8")
    Path(scene["out"] + ".formal_tap.json").write_text(
        json.dumps({"ok": False, "digest": "b" * 64, "frames": 20,
                    "short_frame": True}), encoding="utf-8")
    log = _Log()
    ok, why = _run(scene, log)
    assert ok is False, why
