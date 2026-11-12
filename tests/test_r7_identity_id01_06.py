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

def _complete_generation(final_path, job=None, config=None):
    """补齐"一整代"：verify + fullchain + COMMITTED journal + **事务 marker**。

    R21-C02：resume 复用同一个整代验证器，只造"媒体 + marker"不算完整代。
    R22-A01：marker **只能由生产者生成**（`build_marker_candidate`）——
    手写的旧格式 marker 现在一律判 legacy/unproven，这是审核要求的收紧。
    """
    import publish_journal as _pj
    _vstages = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "video_restoration_pipeline", "stages")
    if _vstages not in sys.path:
        sys.path.insert(0, _vstages)
    import stage7_encode as _s7

    p = str(final_path)
    with open(p + ".verify.json", "w", encoding="utf-8") as f:
        json.dump({"media_sha256": M._plain_sha256(p)}, f)
    with open(p + ".fullchain.json", "w", encoding="utf-8") as f:
        json.dump({"ok": True}, f)
    # R23-B01：journal 必须写明**是哪一次执行**（生产 publisher 就是这么写的）。
    # 缺 attempt_id 时，marker 与 journal 无法证明是同一代 → 拒绝。
    _pj.write_journal(p + ".journal.json",
                      {"phase": "COMMITTED", "job_id": "jobT",
                       "attempt_id": "att1",
                       "targets": [p, p + ".verify.json",
                                   p + ".fullchain.json"],
                       "records": {}})
    objects = [("media", p, M._plain_sha256(p)),
               ("verify", p + ".verify.json",
                M._plain_sha256(p + ".verify.json")),
               ("fullchain", p + ".fullchain.json",
                M._plain_sha256(p + ".fullchain.json"))]
    _job = job or {"job_id": "jobT", "input_path": ""}
    _seed = {
        "impl": M.VIDEO_IMPL_VERSION,
        "job_id": _job.get("job_id", "jobT"),
        "input_path": _job.get("input_path", ""),
        "input_identity": M.build_input_identity(_job.get("input_path", "")),
        # R22-A01：marker 必须有 params identity，且与验证端可重算
        "params_identity": M.build_params_identity(config or {}, _job, None),
    }
    _mk, _stage, _dig, _pm = _s7.build_marker_candidate(
        seed=_seed, pub_digests=objects, attempt_id="att1", job_id="jobT",
        output_path=p, media_digest=objects[0][2],
        media_size=os.path.getsize(p))
    os.replace(_stage, M.completion_marker_path(p))


def test_id03_equal_length_replacement_is_detected(tmp_path):
    out = tmp_path / "clip_restored.mkv"
    out.write_bytes(b"AAAA")
    job = _job(tmp_path)
    _write_marker(str(out), job_id="jobT",
                  input_identity=M.build_input_identity(job["input_path"]),
                  params_identity=M.build_params_identity(_config(), job, None),
                  output_size=4,
                  output_sha256=M._chunked_sha256(str(out)))
    _complete_generation(out, job, _config())
    ok, why = M.resume_artifact_matches(str(out), job, _config(), None)
    assert ok is True, why

    out.write_bytes(b"BBBB")      # 等长替换
    ok2, why2 = M.resume_artifact_matches(str(out), job, _config(), None)
    assert ok2 is False
    # R22-A01：拒绝点前移到 marker payload 的摘要比对（更早、更精确）
    assert "changed since it was published" in why2, why2


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
    assert "legacy" in why, why


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


# ---------------------------------------------------------------------------
# 回归守卫：新增代码用到的模块必须已导入（静态字符串检查抓不到这类问题）
# ---------------------------------------------------------------------------

import ast


@pytest.mark.parametrize("stage_file", [
    "stage4_postprocess.py",
    "stage5_ai_face.py",
    "stage6_grain_synthesis.py",
    "stage7_encode.py",
])
def test_stage_imports_os_when_using_os(stage_file):
    """ID-04 的 pinned 分支用了 `os.path.exists`。

    实测踩坑：stage4 原本**没有** `import os`，加完 pinned 逻辑后
    真实运行时 `NameError: name 'os' is not defined` → stage4 rc=1，
    而我的静态测试（只查字符串）全部通过。这条用 AST 兜住。
    """
    path = os.path.join(ROOT, "video_restoration_pipeline", "stages",
                        stage_file)
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
    # 真的用到 os. 属性吗
    uses_os = any(isinstance(n, ast.Attribute)
                  and isinstance(n.value, ast.Name)
                  and n.value.id == "os"
                  for n in ast.walk(tree))
    if uses_os:
        assert "os" in imported, \
            f"{stage_file} 用了 os.* 但没有 import os（运行会 NameError）"


@pytest.mark.parametrize("stage_file", [
    "stage4_postprocess.py",
    "stage5_ai_face.py",
    "stage6_grain_synthesis.py",
    "stage7_encode.py",
])
def test_stage_compiles_and_has_no_obvious_name_errors(stage_file):
    """AST 级别的引号/缩进守卫（真实运行前的最低成本拦截）。"""
    path = os.path.join(ROOT, "video_restoration_pipeline", "stages",
                        stage_file)
    src = open(path, encoding="utf-8").read()
    compile(src, path, "exec")      # 语法/缩进问题直接抛


def test_pin_skips_artifacts_removed_by_cleanup():
    """pin 前必须检查路径仍在磁盘上。

    实测踩坑：stage5/stage6 被跳过时没有新产物，而 stage4 的产物在
    stage6 结束时被 `cleanup_old_intermediates(keep_steps=2)` 删掉。
    旧逻辑直接把 `_last_produced[4]` 当作 pin 下发 → stage7 拿到不存在
    的路径 → `STAGE7_PINNED_INPUT_MISSING` → rc=82 → 整链重试。
    """
    src = open(os.path.join(ROOT, "video_restoration_pipeline", "main.py"),
               encoding="utf-8").read()
    assert "STAGE_PIN_CANDIDATE_GONE" in src, "缺少 pin 失效告警"
    assert "_alive = [_c for _c in _cands if os.path.exists(_c)]" in src, \
        "pin 没有过滤已被清理的候选"


def test_cleanup_skipped_when_stage_produced_nothing():
    """跳过的 stage 不能执行中间产物清理。

    实测：stage5/stage6 被 SKIP 时没有产物，但 cleanup 仍按
    `keep_steps=2, current_step=6` 保留 {step5, step6}（空集）→ 删掉
    step4，而 step7 正需要它 → rc=82 → 整链重试。
    """
    src = open(os.path.join(ROOT, "video_restoration_pipeline", "main.py"),
               encoding="utf-8").read()
    assert "_produced_this_stage = bool(_last_produced.get(stage_num))" in src
    assert "_produced_this_stage or stage_num == 7" in src
