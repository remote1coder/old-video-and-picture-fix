"""V03 regression: resume scanning must use the same artifact identity as the
stages that write the artifacts."""

import os
from pathlib import Path

import conftest  # noqa: F401
import main as vmain


def _seed(tmp_path, job_id, filename, step):
    temp = tmp_path / "temp"
    temp.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem.replace(" ", "_")
    p = temp / f"{job_id}_{stem}_step{step}_traditional_pABC.mkv"
    p.write_bytes(b"\x00" * 16)
    return str(temp), str(p)


def test_v03_safe_stem_matches_generator_contract():
    from checkpoint import build_intermediate_name
    name = build_intermediate_name("job1", "/in/clip.final.mp4", 2,
                                   "traditional", "ABC", "mkv")
    stem = vmain._safe_input_stem("/in/clip.final.mp4")
    assert name.startswith(f"job1_{stem}_step2_"), (name, stem)


def test_v03_resume_finds_artifacts_that_the_old_pattern_missed(tmp_path):
    import glob
    temp, created = _seed(tmp_path, "job1", "clip.final.mp4", 2)
    found = vmain.find_stage_artifacts(temp, "job1", "/in/clip.final.mp4", 2)
    assert found and os.path.abspath(found[0]) == os.path.abspath(created)
    # the pre-fix pattern (full filename incl. extension) never matched:
    legacy = glob.glob(str(Path(temp) / "job1_clip.final.mp4_step2*"))
    assert legacy == []


def test_v03_unicode_and_multi_dot_names(tmp_path):
    temp, created = _seed(tmp_path, "job2", "老照片 01.final.mov", 3)
    found = vmain.find_stage_artifacts(temp, "job2", "/in/老照片 01.final.mov", 3)
    assert len(found) == 1


def test_v03_final_output_is_found_outside_temp(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    final = out / "clip_restored.mkv"
    final.write_bytes(b"x")
    job = {"job_id": "job3", "input_path": "/in/clip.mp4"}
    cands = vmain.find_final_output(job, {}, {"output": str(out)})
    assert any(os.path.abspath(c) == os.path.abspath(str(final))
               for c in cands)


def test_v03_explicit_output_path_is_honoured(tmp_path):
    explicit = tmp_path / "custom.mkv"
    explicit.write_bytes(b"x")
    job = {"job_id": "job4", "input_path": "/in/clip.mp4",
           "output_path": str(explicit)}
    cands = vmain.find_final_output(job, {}, {"output": str(tmp_path / "out")})
    assert str(explicit) in cands


def test_v03_no_cross_job_bleed(tmp_path):
    temp, _ = _seed(tmp_path, "jobA", "clip.mp4", 2)
    _seed(tmp_path, "jobB", "clip.mp4", 2)
    found = vmain.find_stage_artifacts(temp, "jobA", "/in/clip.mp4", 2)
    assert len(found) == 1 and "jobA" in os.path.basename(found[0])
