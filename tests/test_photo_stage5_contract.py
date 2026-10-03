"""V06/V07 regression: photo stage5 failure propagation + cache identity.

Uses stubs (no models, no GPU): the point is control flow and cache decisions,
not image quality. Every assertion below failed against the pre-fix code.
"""

import json
import os
import shutil

import numpy as np
import pytest
from PIL import Image

import conftest  # noqa: F401
import stage5_postprocess as s5


BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out", "stage5")


def _seed(tmp_path, job_id="job1", stem="clip", seed=7):
    """Create a stage3 artifact + a matching input file, return (input, outdir)."""
    temp = tmp_path / "temp"
    temp.mkdir(parents=True, exist_ok=True)
    art = temp / f"{job_id}_{stem}_step3_upscale.png"
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(art)
    inp = tmp_path / f"{stem}.mp4"
    inp.write_bytes(b"not-a-real-video")
    return str(inp), str(temp)


def _run(monkeypatch, tmp_path, params, job_id, sel=None, grain=None, seed=7):
    inp, temp = _seed(tmp_path, job_id=job_id, seed=seed)
    p = dict(params)
    p.setdefault("temp_dir", temp)
    p.setdefault("log_dir", str(tmp_path / "logs"))
    p.setdefault("photo_bit_depth", 8)
    if sel is not None:
        monkeypatch.setattr(s5, "_apply_selective_processing", sel)
    if grain is not None:
        monkeypatch.setattr(s5, "_add_grain", grain)
    monkeypatch.setattr("sys.argv",
                        ["stage5", "--input", inp, "--job-id", job_id,
                         "--params", json.dumps(p)])
    rc = s5.main()
    return rc, os.path.join(temp, f"{job_id}_clip_step5_postprocess.png"), p


def _log_text(tmp_path, job_id):
    path = tmp_path / "logs" / f"job_{job_id}.log"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _status(output_path):
    p = output_path + ".status.json"
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# --- V06 --------------------------------------------------------------------

def test_v06_processing_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    def boom(img, params, logger):
        raise RuntimeError("injected processing failure")

    rc, out, _ = _run(monkeypatch, tmp_path, {}, "job_v06_a", sel=boom)
    assert rc != 0, "a failed stage must not return success"
    assert rc == 86, rc
    assert not os.path.exists(out), "no artifact may occupy the success path"
    st = _status(out)
    assert st and st["status"] == "FAILED"
    # diagnostic copy keeps a distinct name, it is NOT the stage result
    assert os.path.exists(out + ".failed_input_copy")
    log = _log_text(tmp_path, "job_v06_a")
    assert "STAGE5_DONE" not in log, "the old code logged DONE on failure"
    assert "STAGE5_ERROR" in log


def test_v06_failure_exit_code_depends_on_exception_type(tmp_path, monkeypatch):
    def io_boom(img, params, logger):
        raise OSError("injected io failure")

    rc, out, _ = _run(monkeypatch, tmp_path, {}, "job_v06_b", sel=io_boom)
    assert rc == 82, rc  # I/O error → retryable per v7.32 §3.7
    assert not os.path.exists(out)


def test_v06_grain_failure_requires_explicit_authorisation(tmp_path, monkeypatch):
    def dead_grain(src, dst, strength, size, rough, bit_depth=8):
        return False

    rc, out, _ = _run(monkeypatch, tmp_path, {}, "job_v06_c", grain=dead_grain)
    assert rc == 86, rc
    assert not os.path.exists(out)
    st = _status(out)
    assert st and st["failed_operation"] == "grain_synthesis"

    # same failure, but the caller explicitly accepts a degraded result
    rc2, out2, _ = _run(monkeypatch, tmp_path,
                        {"allow_stage5_degrade": True}, "job_v06_d",
                        grain=dead_grain)
    assert rc2 == 0, rc2
    assert os.path.exists(out2)
    st2 = _status(out2)
    assert st2 and st2["status"] == "DEGRADED"
    assert st2["skipped_operations"] == ["grain_synthesis"]
    # degraded output must not be cached as if it were complete
    assert not os.path.exists(out2 + ".cache.json")


# --- V07 --------------------------------------------------------------------

def _ok_sel(img, params, logger):
    return img


def _make_grain(counter):
    def grain(src, dst, strength, size, rough, bit_depth=8):
        counter["n"] += 1
        shutil.copyfile(src, dst)
        return True
    return grain


def test_v07_cache_hit_then_invalidation(tmp_path, monkeypatch):
    counter = {"n": 0}
    grain = _make_grain(counter)

    rc1, out, p = _run(monkeypatch, tmp_path,
                       {"grain_strength": 0.3}, "job_v07_a",
                       sel=_ok_sel, grain=grain)
    assert rc1 == 0, rc1
    assert os.path.exists(out + ".cache.json")
    assert counter["n"] == 1

    # identical params + identical input → reuse, processing must not rerun
    rc2, out2, _ = _run(monkeypatch, tmp_path,
                        {"grain_strength": 0.3}, "job_v07_a",
                        sel=_ok_sel, grain=grain)
    assert rc2 == 3, rc2
    assert counter["n"] == 1, "cache hit must skip the work"

    # a param that changes the output must invalidate the cache
    rc3, out3, _ = _run(monkeypatch, tmp_path,
                        {"grain_strength": 0.9}, "job_v07_a",
                        sel=_ok_sel, grain=grain)
    assert rc3 == 0, rc3
    assert counter["n"] == 2, "changed grain_strength must force a recompute"


def test_v07_unrelated_param_does_not_invalidate(tmp_path, monkeypatch):
    counter = {"n": 0}
    grain = _make_grain(counter)
    _run(monkeypatch, tmp_path, {"grain_strength": 0.3}, "job_v07_b",
         sel=_ok_sel, grain=grain)
    rc, _, _ = _run(monkeypatch, tmp_path,
                    {"grain_strength": 0.3, "ui_preview_theme": "dark"},
                    "job_v07_b", sel=_ok_sel, grain=grain)
    assert rc == 3, rc
    assert counter["n"] == 1


def test_v07_legacy_cache_without_identity_is_not_trusted(tmp_path, monkeypatch):
    counter = {"n": 0}
    grain = _make_grain(counter)
    rc1, out, _ = _run(monkeypatch, tmp_path, {"grain_strength": 0.3},
                       "job_v07_c", sel=_ok_sel, grain=grain)
    assert rc1 == 0
    os.remove(out + ".cache.json")  # simulate a pre-fix / failed-run artifact
    rc2, _, _ = _run(monkeypatch, tmp_path, {"grain_strength": 0.3},
                     "job_v07_c", sel=_ok_sel, grain=grain)
    assert rc2 == 0, "existing file without identity must not be reused"
    assert counter["n"] == 2


def test_v07_upstream_change_invalidates(tmp_path, monkeypatch):
    counter = {"n": 0}
    grain = _make_grain(counter)
    _run(monkeypatch, tmp_path, {"grain_strength": 0.3}, "job_v07_d",
         sel=_ok_sel, grain=grain)
    # replace the upstream artifact content (same path, different bytes)
    rc, _, _ = _run(monkeypatch, tmp_path, {"grain_strength": 0.3},
                    "job_v07_d", sel=_ok_sel, grain=grain, seed=99)
    assert rc == 0, rc
    assert counter["n"] == 2


def test_v07_impl_version_is_part_of_identity(tmp_path):
    a = s5.compute_stage5_cache_identity(__file__, {"grain_strength": 0.3}, 16)
    b = s5.compute_stage5_cache_identity(__file__, {"grain_strength": 0.3}, 8)
    assert a["cache_id"] != b["cache_id"]
    assert a["impl"] == s5.STAGE5_IMPL_VERSION
