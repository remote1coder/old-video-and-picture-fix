"""ID-07 / ID-08（第七轮）：实际执行指纹 与 上游摘要链。

**ID-07**：配置写 `upscale_model: realesr-general-x4v3` 说明不了执行事实
（同名权重换内容、后端不同、OOM 降级都看不出来）。改为把"真的跑了什么"
记在产物旁边的 `.exec.json`，并由 main 并入身份侧车。

**ID-08**：侧车记了 `stage` 却从不比对；也没有"这份中间产物来自哪一份
上游产物"。于是把 step3 的文件塞进 step5 的位置、或上游换了一版，
都察觉不到。
"""

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

import exec_fingerprint as ef  # noqa: E402
import main as M  # noqa: E402


# ---------------------------------------------------------------------------
# ID-07
# ---------------------------------------------------------------------------

def test_weight_fingerprint_uses_content_not_just_size(tmp_path):
    w = tmp_path / "model.pth"
    w.write_bytes(b"AAAA")
    fp1 = ef.file_fingerprint(str(w))
    w.write_bytes(b"BBBB")            # 同名等长替换
    fp2 = ef.file_fingerprint(str(w))
    assert fp1["size"] == fp2["size"]
    assert fp1["sha256"] != fp2["sha256"], \
        "同名等长替换没被识别 —— 不能用大小/路径当权重身份"


def test_missing_weight_is_reported_not_guessed(tmp_path):
    fp = ef.file_fingerprint(str(tmp_path / "nope.pth"))
    assert fp["exists"] is False and fp["sha256"] is None


def test_probe_reports_real_interpreter(tmp_path):
    """探测必须针对**真正的 AI 解释器**，而不是编排器 Python。"""
    ai_py = os.path.join(ROOT, ".venv_r4", "Scripts", "python.exe")
    if not os.path.exists(ai_py):
        pytest.skip("本机没有 .venv_r4，无法验证 AI 解释器探测")
    tools = ef.probe_ai_backend(ai_py)
    assert "probe_error" not in tools, tools
    assert tools.get("interpreter") == ai_py
    assert tools.get("torch"), "没读到 torch 版本"


def test_probe_of_missing_interpreter_is_honest(tmp_path):
    tools = ef.probe_ai_backend(str(tmp_path / "no-python.exe"))
    assert "probe_error" in tools, "探测失败却返回了看似正常的版本号"


def test_exec_sidecar_roundtrip(tmp_path):
    art = tmp_path / "step3_upscale.mkv"
    art.write_bytes(b"ART")
    payload = ef.build_exec_fingerprint(
        "m", None, ai_python=None, backend="gpu", scale=2)
    assert ef.write_exec_sidecar(str(art), payload) is True
    got = ef.read_exec_sidecar(str(art))
    assert got is not None
    assert got["backend"] == "gpu"
    assert got["schema"] == ef.EXEC_SCHEMA


def test_exec_sidecar_does_not_overwrite_identity_sidecar(tmp_path):
    """两个侧车是不同文件，互不覆盖。"""
    art = tmp_path / "a.mkv"
    art.write_bytes(b"ART")
    ident = str(art) + ".identity.json"
    open(ident, "w").write('{"stage": 3}')
    ef.write_exec_sidecar(str(art), {"schema": ef.EXEC_SCHEMA,
                                     "backend": "gpu"})
    assert json.loads(open(ident).read())["stage"] == 3


def test_stage3_records_exec_fingerprint():
    src = open(os.path.join(ROOT, "video_restoration_pipeline", "stages",
                            "stage3_ai_upscale.py"), encoding="utf-8").read()
    assert "STAGE3_EXEC_FINGERPRINT" in src
    assert "build_exec_fingerprint" in src
    # 必须记录降级后的实际值，而不是请求值
    assert '"cpu" if backend == "cpu" else "gpu"' in src


def test_main_merges_exec_into_identity_sidecar():
    src = open(os.path.join(ROOT, "video_restoration_pipeline", "main.py"),
               encoding="utf-8").read()
    assert "_read_exec_fingerprint" in src
    assert '"exec": _read_exec_fingerprint(artifact)' in src


# ---------------------------------------------------------------------------
# ID-08
# ---------------------------------------------------------------------------

def _sidecar(art, **fields):
    base = {
        "impl": M.VIDEO_IMPL_VERSION,
        "job_id": "jobT",
        "artifact": str(art),
        "artifact_size": art.stat().st_size,
        "artifact_digest": M._artifact_digest(str(art)),
    }
    base.update(fields)
    with open(str(art) + ".identity.json", "w", encoding="utf-8") as f:
        json.dump(base, f)


def _job_cfg(tmp_path):
    src = tmp_path / "src.mov"
    src.write_bytes(b"SRC")
    job = {"job_id": "jobT", "input_path": str(src)}
    cfg = {"global": {"crf": 24}, "ai": {}, "paths": {}}
    job["_effective_params"] = M.merge_params(job, cfg, {}, None)
    return job, cfg


def test_id08_stage_mismatch_is_rejected(tmp_path):
    """把 stage3 的产物当成 stage5 的来用 → 必须拒绝。"""
    job, cfg = _job_cfg(tmp_path)
    art = tmp_path / "jobT_src_step3_upscale.mkv"
    art.write_bytes(b"ART")
    _sidecar(art, stage=3,
             source_identity=M.build_input_identity(job["input_path"]),
             params_identity=M.build_params_identity(cfg, job, None))
    ok, why = M.intermediate_identity_matches(art, job, cfg, None,
                                              expected_stage=5)
    assert ok is False
    assert "stage 3" in why and "expected stage 5" in why


def test_id08_matching_stage_passes(tmp_path):
    job, cfg = _job_cfg(tmp_path)
    art = tmp_path / "jobT_src_step5_face.mkv"
    art.write_bytes(b"ART")
    _sidecar(art, stage=5,
             source_identity=M.build_input_identity(job["input_path"]),
             params_identity=M.build_params_identity(cfg, job, None))
    ok, why = M.intermediate_identity_matches(art, job, cfg, None,
                                              expected_stage=5)
    assert ok is True, why


def test_id08_sidecar_without_stage_is_rejected(tmp_path):
    job, cfg = _job_cfg(tmp_path)
    art = tmp_path / "jobT_src_step5_face.mkv"
    art.write_bytes(b"ART")
    _sidecar(art,  # 不写 stage
             source_identity=M.build_input_identity(job["input_path"]),
             params_identity=M.build_params_identity(cfg, job, None))
    ok, why = M.intermediate_identity_matches(art, job, cfg, None,
                                              expected_stage=5)
    assert ok is False
    assert "does not record which stage" in why


def test_id08_upstream_change_invalidates(tmp_path):
    """上游换了一版（同输入同参数）→ 中间产物必须失效。"""
    job, cfg = _job_cfg(tmp_path)
    up = tmp_path / "up.mkv"
    up.write_bytes(b"UPSTREAM-V1")
    art = tmp_path / "jobT_src_step5_face.mkv"
    art.write_bytes(b"ART")
    _sidecar(art, stage=5,
             source_identity=M.build_input_identity(job["input_path"]),
             params_identity=M.build_params_identity(cfg, job, None),
             upstream_artifact=str(up),
             upstream_digest=M._artifact_digest(str(up)))

    ok, _ = M.intermediate_identity_matches(art, job, cfg, None,
                                            expected_stage=5,
                                            expected_upstream=str(up))
    assert ok is True

    up.write_bytes(b"UPSTREAM-V2")     # 上游变了
    ok2, why2 = M.intermediate_identity_matches(art, job, cfg, None,
                                                expected_stage=5,
                                                expected_upstream=str(up))
    assert ok2 is False
    assert "upstream artifact changed" in why2


def test_id08_main_records_upstream_digest():
    src = open(os.path.join(ROOT, "video_restoration_pipeline", "main.py"),
               encoding="utf-8").read()
    assert '"upstream_digest"' in src
    assert "upstream_artifact=params.get(\"upstream_artifact\")" in src
    assert "expected_upstream" in src
