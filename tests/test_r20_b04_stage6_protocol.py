# -*- coding: utf-8 -*-
"""R20-B04：stage6 结果协议（审核 R19-B04）。

审核原文：

> helper 失败现在返回 `(84, detail)`；调用方仍当作两段字符串接收，再做
> `"GrainFactory3" in err_str`。`err_str` 实为整数 84，触发
> `TypeError: argument of type 'int' is not iterable`。
> 原本识别 GrainFactory3 错误并回退 AddGrainC 的路径也受影响。
> **补 main 层用例，不能只测 helper。**

本文件锁住三件事：

1. `GrainPipelineResult` 的语义：取消 ≠ 插件失败，编码器失败 ≠ 插件失败。
2. **AST 守卫**：调用方不得再解包 tuple 或用 `result[0]`。
3. 真实子进程路径下，失败结果是结构化的，且 `failed_stage` 指向真实步骤。
"""
import ast
import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE6 = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                       "stage6_grain_synthesis.py")


def _load_s6():
    sys.path.insert(0, os.path.join(_ROOT, "video_restoration_pipeline", "stages"))
    sys.path.insert(0, os.path.join(_ROOT, "shared_modules"))
    import stage6_grain_synthesis as s6
    return s6


@pytest.fixture(scope="module")
def s6():
    return _load_s6()


# ------------------------------------------------ 结果语义

def test_success_result_is_ok(s6):
    r = s6.GrainPipelineResult(ok=True, vspipe_rc=0, ffmpeg_rc=0)
    assert r.ok is True
    assert r.is_plugin_capability_failure() is False


def test_plugin_failure_on_producer_is_fallbackable(s6):
    """vspipe 求值 .vpy 失败且 stderr 提到 GrainFactory3 → 可回退。"""
    r = s6.GrainPipelineResult(
        ok=False, vspipe_rc=1, ffmpeg_rc=0, failed_stage="vspipe",
        stderr_tail="Script evaluation failed: name 'GrainFactory3' is not defined")
    assert r.is_plugin_capability_failure() is True


def test_encoder_failure_is_not_fallbackable(s6):
    """**编码器失败不能靠换插件掩盖**（审核明确要求）。"""
    r = s6.GrainPipelineResult(
        ok=False, vspipe_rc=0, ffmpeg_rc=1, failed_stage="ffmpeg",
        stderr_tail="GrainFactory3 ok but ffmpeg: No space left on device")
    assert r.is_plugin_capability_failure() is False


def test_cancel_is_never_fallbackable(s6):
    """**取消绝不能当插件失败去回退**（否则用户取消后还在干活）。"""
    r = s6.GrainPipelineResult(
        ok=False, vspipe_rc=None, ffmpeg_rc=None, failed_stage="vspipe",
        cancelled=True,
        stderr_tail="GrainFactory3 interrupted by user")
    assert r.is_plugin_capability_failure() is False


def test_both_fail_is_not_fallbackable(s6):
    """生产者 + 编码器都失败时，不是纯插件问题。"""
    r = s6.GrainPipelineResult(
        ok=False, vspipe_rc=1, ffmpeg_rc=1, failed_stage="vspipe",
        stderr_tail="GrainFactory3 failed")
    assert r.is_plugin_capability_failure() is False


def test_start_failure_is_not_fallbackable(s6):
    """第二个进程压根没起来 → 不是插件问题。"""
    r = s6.GrainPipelineResult(
        ok=False, vspipe_rc=None, ffmpeg_rc=None, failed_stage="start",
        stderr_tail="GrainFactory3")
    assert r.is_plugin_capability_failure() is False


def test_unrelated_stderr_is_not_fallbackable(s6):
    """stderr 里没有插件关键字 → 不回退。"""
    r = s6.GrainPipelineResult(
        ok=False, vspipe_rc=1, ffmpeg_rc=0, failed_stage="vspipe",
        stderr_tail="SyntaxError: invalid syntax")
    assert r.is_plugin_capability_failure() is False


# ------------------------------------------------ AST 守卫（main 层协议）

def _stage6_tree():
    with open(_STAGE6, encoding="utf-8", errors="replace") as f:
        return ast.parse(f.read())


def test_caller_never_unpacks_result_tuple():
    """调用方不得再写 `x, y = _run_grain_pipeline(...)`。

    审核：这正是 TypeError 的直接来源。
    """
    tree = _stage6_tree()
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        # 目标是 Tuple（解包）
        if not any(isinstance(t, ast.Tuple) for t in node.targets):
            continue
        # 右值里出现 _run_grain_pipeline
        src = ast.dump(node.value)
        if "_run_grain_pipeline" in src:
            bad.append(node.lineno)
    assert not bad, f"仍存在解包 _run_grain_pipeline 返回值：行 {bad}"


def test_caller_never_subscripts_result():
    """调用方不得再用 `result[0]` / `result2[0]`。"""
    tree = _stage6_tree()
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            v = node.value
            if isinstance(v, ast.Name) and v.id in ("result", "result2"):
                bad.append(node.lineno)
    assert not bad, f"仍存在 result[N] 下标访问：行 {bad}"


def test_plugin_result_class_exists_and_has_fields():
    """结果对象必须带齐审核要求的字段。"""
    s6 = _load_s6()
    r = s6.GrainPipelineResult()
    for f in ("ok", "vspipe_rc", "ffmpeg_rc", "failed_stage",
              "cancelled", "log_paths", "stderr_tail", "detail"):
        assert hasattr(r, f), f"缺少字段 {f}"


# ------------------------------------------------ 真实子进程路径

def test_real_failure_reports_stage_and_rc(tmp_path):
    """真实子进程：生产者非零 → 结构化结果带真实 rc 与步骤。"""
    s6 = _load_s6()
    log = type("L", (), {
        "error": lambda *a, **k: None,
        "warning": lambda *a, **k: None,
        "info": lambda *a, **k: None,
        "critical": lambda *a, **k: None,
    })()

    py = sys.executable
    vspipe = [py, "-c",
              "import sys; sys.stderr.write('GrainFactory3 boom\\n'); "
              "sys.stdout.write('x' * 100); sys.exit(9)"]
    ffmpeg = [py, "-c",
              "import sys; d=sys.stdin.buffer.read(); sys.exit(0)"]

    res = s6._run_grain_pipeline(
        "clip = None", str(tmp_path / "g.vpy"), vspipe, ffmpeg,
        str(tmp_path / "out.mkv"), log)

    assert res.ok is False
    assert res.vspipe_rc == 9, res.vspipe_rc
    assert res.failed_stage == "vspipe", res.failed_stage
    # stderr 里确实有插件名 + rc 非零 + 编码器正常 → 可回退
    assert res.is_plugin_capability_failure() is True
    assert "GrainFactory3" in (res.stderr_tail or "")
