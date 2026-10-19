"""A5 身份（ID-01 ~ ID-06，第七轮）。

逐条对应任务书 §2.1 的表格。每条都尽量做成**可运行的**判定，
而不只是断言源码里有某个字符串。
"""

import hashlib
import json
import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "video_restoration_pipeline"),
           os.path.join(ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import main as M  # noqa: E402


# ---------------------------------------------------------------------------
# 公共装置
# ---------------------------------------------------------------------------

def _fake_input(tmp_path, content=b"SRC-DATA"):
    p = tmp_path / "src.mov"
    p.write_bytes(content)
    return str(p)


def _job(tmp_path, content=b"SRC-DATA", **over):
    job = {"job_id": "jobT", "input_path": _fake_input(tmp_path, content)}
    job.update(over)
    return job


def _config():
    return {
        "global": {"crf": 24, "encoder": "svtav1"},
        "ai": {"upscale_model": "realesr-general-x4v3", "upscale_scale": 2},
        "paths": {},
    }


def _write_marker(final_path, **fields):
    payload = dict(fields)
    payload.setdefault("impl", M.VIDEO_IMPL_VERSION)
    with open(M.completion_marker_path(final_path), "w", encoding="utf-8") as f:
        json.dump(payload, f)


# ---------------------------------------------------------------------------
# ID-01：缺 params_identity 必须拒绝（两条路径都要拒绝）
# ---------------------------------------------------------------------------

def test_id01_final_marker_without_params_identity_is_rejected(tmp_path):
    out = tmp_path / "clip_restored.mkv"
    out.write_bytes(b"RESULT")
    job = _job(tmp_path)
    _write_marker(str(out),
                  job_id="jobT",
                  input_identity=M.build_input_identity(job["input_path"]),
                  params_identity="",            # 空身份
                  output_size=out.stat().st_size,
                  output_sha256=M._chunked_sha256(str(out)))
    ok, why = M.resume_artifact_matches(str(out), job, _config(), None)
    assert ok is False
    assert "no params identity" in why
    assert out.exists(), "拒绝复用不等于删除文件"


def test_id01_final_marker_missing_key_is_rejected(tmp_path):
    """键完全缺失（不是空串）同样拒绝。"""
    out = tmp_path / "clip_restored.mkv"
    out.write_bytes(b"RESULT")
    job = _job(tmp_path)
    _write_marker(str(out), job_id="jobT",
                  input_identity=M.build_input_identity(job["input_path"]),
                  output_size=out.stat().st_size,
                  output_sha256=M._chunked_sha256(str(out)))
    ok, why = M.resume_artifact_matches(str(out), job, _config(), None)
    assert ok is False and "no params identity" in why


def test_id01_intermediate_without_params_identity_is_rejected(tmp_path):
    art = tmp_path / "jobT_src_step2_traditional.mkv"
    art.write_bytes(b"INTERMEDIATE")
    job = _job(tmp_path)
    side = str(art) + ".identity.json"
    with open(side, "w", encoding="utf-8") as f:
        json.dump({
            "impl": M.VIDEO_IMPL_VERSION,      # 侧车 schema 的必填项
            "job_id": "jobT",
            "stage": 2,
            "source_identity": M.build_input_identity(job["input_path"]),
            "params_identity": "",              # 空身份
            "artifact_digest": M._artifact_digest(str(art)),
            "artifact_size": art.stat().st_size,
        }, f)
    ok, why = M.intermediate_identity_matches(str(art), job, _config(), None)
    assert ok is False
    assert "no params identity" in why
    assert art.exists()


# ---------------------------------------------------------------------------
# ID-02：身份必须来自"实际执行的那一份参数"（含 CLI 覆盖）
# ---------------------------------------------------------------------------

def test_id02_cli_override_changes_identity(tmp_path):
    job = _job(tmp_path)
    cfg = _config()
    base = M.build_params_identity(cfg, job, None)

    job2 = dict(job)
    job2["_effective_params"] = M.merge_params(
        job2, cfg, {"ai_upscale_model": "RealESRGAN_x4plus"}, None)
    with_cli = M.build_params_identity(cfg, job2, None)

    assert base != with_cli, \
        "改了 --ai-model 身份却没变 —— 旧产物会被误认为同一身份"


def test_id02_positional_fields_do_not_change_identity(tmp_path):
    """只改输出目录/临时目录不该触发重算。"""
    job = _job(tmp_path)
    cfg = _config()
    p1 = M.merge_params(job, cfg, {}, None)
    p2 = dict(p1, temp_dir="D:/other/temp", log_dir="D:/other/logs",
              output_dir="D:/other/out")
    id1 = M.build_params_identity(cfg, dict(job, _effective_params=p1), None)
    id2 = M.build_params_identity(cfg, dict(job, _effective_params=p2), None)
    assert id1 == id2, "纯位置字段影响了身份 —— 换个目录就全量重算"


def test_id02_semantic_field_does_change_identity(tmp_path):
    job = _job(tmp_path)
    cfg = _config()
    p1 = M.merge_params(job, cfg, {}, None)
    p2 = dict(p1, crf=18)
    id1 = M.build_params_identity(cfg, dict(job, _effective_params=p1), None)
    id2 = M.build_params_identity(cfg, dict(job, _effective_params=p2), None)
    assert id1 != id2


def test_id02_identity_semantics_strips_internal_keys(tmp_path):
    view = M._identity_semantics({"crf": 20, "_shake_probe": {"x": 1},
                                  "temp_dir": "t", "log_dir": "l"})
    assert view == {"crf": 20}


# ---------------------------------------------------------------------------
# ID-03：成品内容摘要
# ---------------------------------------------------------------------------

def test_id03_equal_length_replacement_is_detected(tmp_path):
    out = tmp_path / "clip_restored.mkv"
    out.write_bytes(b"AAAA")
    job = _job(tmp_path)
    _write_marker(str(out), job_id="jobT",
                  input_identity=M.build_input_identity(job["input_path"]),
                  params_identity=M.build_params_identity(_config(), job, None),
                  output_size=4,
                  output_sha256=M._chunked_sha256(str(out)))
    ok, _ = M.resume_artifact_matches(str(out), job, _config(), None)
    assert ok is True

    out.write_bytes(b"BBBB")      # 等长替换
    ok2, why2 = M.resume_artifact_matches(str(out), job, _config(), None)
    assert ok2 is False
    assert "content differs" in why2


def test_id03_legacy_marker_without_digest_is_not_auto_reused(tmp_path):
    out = tmp_path / "clip_restored.mkv"
    out.write_bytes(b"RESULT")
    job = _job(tmp_path)
    _write_marker(str(out), job_id="jobT",
                  input_identity=M.build_input_identity(job["input_path"]),
                  params_identity=M.build_params_identity(_config(), job, None),
                  output_size=out.stat().st_size)   # 无 output_sha256
    ok, why = M.resume_artifact_matches(str(out), job, _config(), None)
    assert ok is False
    assert "legacy/unproven" in why


# ---------------------------------------------------------------------------
# ID-04 / ID-05：精确产物传递 与 只登记本次产出
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stage_file,marker", [
    ("stage4_postprocess.py", "STAGE4_INPUT_PINNED"),
    ("stage5_ai_face.py", "STAGE5_INPUT_PINNED"),
    ("stage6_grain_synthesis.py", "STAGE6_INPUT_PINNED"),
    ("stage7_encode.py", "STAGE7_INPUT_PINNED"),
])
def test_id04_stages_honor_pinned_upstream(stage_file, marker):
    """三个消费 stage 都要支持 params['upstream_artifact'] 精确优先。"""
    src = open(os.path.join(ROOT, "video_restoration_pipeline", "stages",
                            stage_file), encoding="utf-8").read()
    assert "upstream_artifact" in src, f"{stage_file} 不支持精确上游"
    assert marker in src, f"{stage_file} 缺少 PINNED 日志"
    assert "_PINNED_INPUT_MISSING" in src, \
        f"{stage_file} 在点名路径缺失时会静默 glob 另一个文件"


def test_id04_main_pins_upstream_between_stages():
    """main 必须在 stage 之间传递本次产出的精确路径。"""
    src = open(os.path.join(ROOT, "video_restoration_pipeline", "main.py"),
               encoding="utf-8").read()
    assert "STAGE_UPSTREAM_PINNED" in src
    assert 'params["upstream_artifact"] = _pinned_upstream' in src
    assert "params.pop(\"upstream_artifact\", None)" in src


def test_id05_only_newly_produced_artifacts_get_identity():
    """登记身份必须基于 stage 前后的指纹差，而不是整个 glob 结果。"""
    src = open(os.path.join(ROOT, "video_restoration_pipeline", "main.py"),
               encoding="utf-8").read()
    assert "_snapshot_stage_files" in src
    assert "_produced = [" in src
    assert "STAGE_STALE_ARTIFACTS_PRESENT" in src, \
        "缺少「陈旧产物未登记」的显式告警"


# ---------------------------------------------------------------------------
# ID-06：身份不符 ≠ 损坏，不得删除
# ---------------------------------------------------------------------------

def test_id06_identity_mismatch_does_not_delete(tmp_path):
    """main 的清理分支只能删"真的解不开"的文件。"""
    src = open(os.path.join(ROOT, "video_restoration_pipeline", "main.py"),
               encoding="utf-8").read()
    # 旧写法：无条件 os.remove + 记成 corrupt
    assert 'Removed corrupt intermediate: {fpath}' not in src, \
        "仍是旧的无条件删除文案"
    assert "RESUME_ARTIFACTS_KEPT_UNPROVEN" in src, \
        "缺少「保留但标 unproven」的分支"
    assert "verify_ffprobe_decodeable(fpath)" in src, \
        "没有用「能否解码」来区分损坏与身份不符"
