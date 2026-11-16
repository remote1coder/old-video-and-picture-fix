# -*- coding: utf-8 -*-
"""照片链内存编排（N27-05）。

最重要的一条：**请求里包含 S3/S4 时必须明确报"能力不足"**，
不能静默跳过它们然后说"全链跑完"。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared_modules"))

import media_io  # noqa: E402
import photo_chain as PC  # noqa: E402

pytest.importorskip("scipy")


def _png(seed=0, h=20, w=24):
    rng = np.random.default_rng(seed)
    a = rng.integers(10000, 60000, (h, w, 3), dtype=np.uint16)
    a[6:12, 6:12, :] = 28000
    return media_io.encode_png(a)


#: 关掉 G'MIC 三条链，只验证编排本身
NO_GMIC = {"deblock_strength": 0, "scratch_remove_strength": 0,
           "deband_threshold": 0}


def test_s1_only_runs():
    r = PC.run_inproc_chain(_png(), dict(NO_GMIC), stages=["s1"])
    assert r["ok"] is True, r["error"]
    assert r["stages_run"] == ["s1"]
    assert r["png"]


def test_s1_s2_s5_chain_runs():
    r = PC.run_inproc_chain(_png(), dict(NO_GMIC, photo_bit_depth=16),
                            stages=["s1", "s2", "s5"])
    assert r["ok"] is True, r["error"]
    assert r["stages_run"] == ["s1", "s2", "s5"]
    back = media_io.read_png_bytes(r["png"])
    assert back.dtype == np.uint16 and back.ndim == 3


def test_s3_is_reported_as_unsupported_not_skipped():
    """请求 S3 → 明确拒绝。静默跳过等于交付未超分的成品还说过全链。"""
    r = PC.run_inproc_chain(_png(), dict(NO_GMIC),
                            stages=["s1", "s2", "s3", "s5"])
    assert r["ok"] is False
    assert r["unsupported"] == ["s3"], r
    assert "model worker client" in r["error"], r["error"]
    # 拒绝发生**在动手之前**：一 stage 都没跑
    assert r["stages_run"] == []


def test_s4_is_reported_as_unsupported_too():
    r = PC.run_inproc_chain(_png(), dict(NO_GMIC), stages=["s1", "s4"])
    assert r["ok"] is False
    assert r["unsupported"] == ["s4"]


def test_default_stage_list_is_the_full_six_and_therefore_refused():
    """默认顺序包含 S3/S4 → 默认调用现在**必须**被拒（诚实优先）。"""
    r = PC.run_inproc_chain(_png(), dict(NO_GMIC))
    assert r["ok"] is False
    assert r["unsupported"] == ["s3", "s4"], r


def test_capabilities_are_self_describing():
    c = PC.capabilities()
    assert c["memory_stages"] == ["s1", "s2", "s5"]
    assert c["file_only_stages"] == ["s3", "s4"]
    assert "S6" in c["note"] or "final candidate" in c["note"]


def test_stage_failure_aborts_the_chain():
    """某一 stage 失败 → 整链失败（不继续用坏数据往下跑）。"""
    r = PC.run_inproc_chain(b"not a png", dict(NO_GMIC), stages=["s1", "s2"])
    assert r["ok"] is False
    assert "s1 failed" in r["error"], r["error"]
    assert r["stages_run"] == []
