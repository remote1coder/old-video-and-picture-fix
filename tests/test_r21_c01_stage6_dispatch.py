# -*- coding: utf-8 -*-
"""R21-C01：stage6 失败可以返回成功 / AddGrain 回退执行丢失。

## 审核原文（R20 审核 §3，P1）

> `result.ok` 返回 0，取消返回 88，插件能力失败且禁用 AddGrain 返回 84。
> 但以下两条路径会直接走到 `main()` 末尾：**普通编码失败**；
> **插件能力失败且允许 AddGrain**。原先真正执行回退的代码没有保留下来，
> 末尾也没有一般失败返回值。
>
> 本轮提取原 `main` 的结果处理尾段，传入真实 `GrainPipelineResult`：
> 上述两个失败输入均得到 `None`。入口使用 `sys.exit(main())`，
> 而 `None` 等价于成功退出。

## 本文件锁住什么

1. `dispatch_result()` 覆盖**全部**返回路径，且没有任何"走到末尾"的分支；
2. 回退**真的被调用**（不是把第一次失败改写成成功），且强度减半；
3. 回退失败要报 84，不能算成功；
4. 插件能力失败的判定看**首发原因**（生产者侧尾部），
   ffmpeg 的连带非零不能否掉它；反之编码/磁盘错误不得被关键词吞掉；
5. 取消**绝不**触发回退。
"""
import ast
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE6 = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                       "stage6_grain_synthesis.py")
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage6_grain_synthesis as s6  # noqa: E402

GF3_ERR = ('Traceback (most recent call last):\n'
           '  File "grain.vpy", line 41, in <module>\n'
           '    clip = core.grainfactory.GrainFactory3(clip)\n'
           'AttributeError: No attribute with the name grainfactory exists')


class _Log:
    def __init__(self):
        self.events = []

    def _rec(self, ev, msg="", **kw):
        self.events.append((ev, msg, kw))

    info = warning = error = critical = _rec

    def names(self):
        return [e[0] for e in self.events]


def _ok():
    return s6.GrainPipelineResult(ok=True, vspipe_rc=0, ffmpeg_rc=0)


def _plugin_fail(ffmpeg_rc=1):
    return s6.GrainPipelineResult(
        ok=False, vspipe_rc=1, ffmpeg_rc=ffmpeg_rc, failed_stage="vspipe",
        stderr_tail=GF3_ERR, vspipe_tail=GF3_ERR, encoder_tail="",
        detail="producer (vspipe) failed rc=1")


def _encoder_fail():
    return s6.GrainPipelineResult(
        ok=False, vspipe_rc=0, ffmpeg_rc=1, failed_stage="ffmpeg",
        stderr_tail="disk quota exceeded", vspipe_tail="",
        encoder_tail="disk quota exceeded", detail="encoder failed rc=1")


# ------------------------------------------------ 返回路径全覆盖

def test_success_returns_zero():
    rc, degraded = s6.dispatch_result(_ok(), True, _Log())
    assert rc == 0 and degraded is None


def test_cancel_returns_88_and_never_falls_back():
    called = []
    r = s6.GrainPipelineResult(ok=False, cancelled=True, failed_stage="wait")
    rc, degraded = s6.dispatch_result(
        r, True, _Log(), run_fallback=lambda: called.append(1) or _ok())
    assert rc == 88
    assert degraded is None
    assert called == [], "取消绝不能触发插件回退"


def test_plugin_failure_with_fallback_disabled_returns_84():
    log = _Log()
    rc, degraded = s6.dispatch_result(_plugin_fail(), False, log,
                                      run_fallback=lambda: _ok())
    assert rc == 84
    assert degraded is None
    assert "STAGE6_GF3_FAILED" in log.names()


def test_plugin_failure_with_fallback_enabled_really_runs_it():
    """**核心**：允许回退时必须真的调用回退，并返回降级结果。"""
    calls = []

    def _fb():
        calls.append(1)
        return _ok()

    log = _Log()
    rc, degraded = s6.dispatch_result(_plugin_fail(), True, log,
                                      run_fallback=_fb)
    assert rc == 0
    assert calls == [1], "回退没有被真正执行 —— 正是 R20 丢掉的代码"
    assert degraded is not None and degraded.ok is True
    assert "STAGE6_FALLBACK" in log.names()


def test_plugin_failure_and_fallback_failure_returns_84():
    log = _Log()
    rc, degraded = s6.dispatch_result(
        _plugin_fail(), True, log,
        run_fallback=lambda: s6.GrainPipelineResult(
            ok=False, failed_stage="vspipe", vspipe_rc=1, ffmpeg_rc=1,
            detail="fallback also died"))
    assert rc == 84
    assert degraded is None
    assert "STAGE6_FAIL" in log.names()


def test_plain_encoder_failure_returns_84_not_none():
    """普通编码失败 —— R20 里这条会走到函数末尾返回 None（= 成功）。"""
    log = _Log()
    rc, degraded = s6.dispatch_result(_encoder_fail(), True, log)
    assert rc == 84, "普通失败必须非零；None/0 等于把失败当成功"
    assert degraded is None
    assert "STAGE6_FAIL" in log.names()


def test_missing_fallback_runner_is_not_success():
    rc, degraded = s6.dispatch_result(_plugin_fail(), True, _Log(),
                                      run_fallback=None)
    assert rc == 84 and degraded is None


@pytest.mark.parametrize("builder,expected", [
    (lambda: s6.GrainPipelineResult(ok=True), 0),
    (lambda: s6.GrainPipelineResult(ok=False, cancelled=True), 88),
    (lambda: s6.GrainPipelineResult(
        ok=False, failed_stage="vspipe", vspipe_rc=1, vspipe_tail=GF3_ERR), 84),
    (lambda: s6.GrainPipelineResult(
        ok=False, failed_stage="ffmpeg", vspipe_rc=0, ffmpeg_rc=1), 84),
    (lambda: s6.GrainPipelineResult(
        ok=False, failed_stage="exception", detail="boom"), 84),
    (lambda: s6.GrainPipelineResult(
        ok=False, failed_stage="start", detail="FileNotFoundError: vspipe"), 84),
])
def test_every_shape_returns_an_int_never_none(builder, expected):
    rc, _ = s6.dispatch_result(builder(), False, _Log())
    assert isinstance(rc, int), f"返回了 {rc!r}；None 会被 sys.exit 当成成功"
    assert rc == expected


# ------------------------------------------------ 首发原因判定

def test_cascading_ffmpeg_failure_does_not_mask_plugin_cause():
    """插件缺失 → vspipe rc=1 → ffmpeg 也没输入而 rc=1。

    旧判定要求 `ffmpeg_rc in (None, 0)`，于是**真实的插件缺失永远走不到回退**。
    """
    r = _plugin_fail(ffmpeg_rc=1)
    assert r.is_plugin_capability_failure() is True
    assert r.is_cascading_encoder_failure() is False


def test_encoder_failure_is_not_plugin_capability():
    assert _encoder_fail().is_plugin_capability_failure() is False


def test_producer_failure_without_plugin_words_is_not_plugin_capability():
    """vspipe 挂但报错是磁盘/语法 —— 不许回退，否则会把真问题盖住。"""
    r = s6.GrainPipelineResult(
        ok=False, failed_stage="vspipe", vspipe_rc=1, ffmpeg_rc=0,
        vspipe_tail="OSError: [Errno 28] No space left on device")
    assert r.is_plugin_capability_failure() is False


def test_cancelled_is_never_plugin_capability():
    r = s6.GrainPipelineResult(ok=False, cancelled=True,
                               failed_stage="vspipe", vspipe_rc=1,
                               vspipe_tail=GF3_ERR)
    assert r.is_plugin_capability_failure() is False


@pytest.mark.parametrize("tail", [
    # 本机实测的真实报错（2026-11-11，r72）：没有 "grainfactory3" 这个词
    "AttributeError: No attribute with the name grainfactory exists. "
    "Did you mistype a plugin namespace or forget to install a plugin?",
    # havsfunc 版本太老
    "AttributeError: module 'havsfunc' has no attribute 'GrainFactory3'",
    # 模块整个没有
    "ModuleNotFoundError: No module named 'havsfunc'",
    # 脚本里 haf 未定义（注意旧表里的 "haf." 匹配不到这句）
    "NameError: name 'haf' is not defined",
])
def test_real_plugin_missing_messages_are_recognised(tail):
    """旧关键词表只认 `grainfactory3`/`haf.`/`havsfunc`，前两条与第四条
    在真实环境里**永远匹配不上**（实测 `core.grainfactory` 的报错原文
    里没有 `grainfactory3`）。"""
    r = s6.GrainPipelineResult(ok=False, failed_stage="vspipe",
                               vspipe_rc=1, ffmpeg_rc=0, vspipe_tail=tail)
    assert r.is_plugin_capability_failure() is True


def test_a_disk_error_in_the_encoder_tail_does_not_trigger_fallback():
    """插件词只出现在**编码器**尾部里不能算插件缺失 —— 那是编码器自己的事。"""
    r = s6.GrainPipelineResult(
        ok=False, failed_stage="vspipe", vspipe_rc=1, ffmpeg_rc=1,
        vspipe_tail="Error: failed to open script",
        encoder_tail="ImportError: something unrelated")
    assert r.is_plugin_capability_failure() is False


def test_cascading_helper_reports_the_truth_when_both_fail():
    r = s6.GrainPipelineResult(
        ok=False, failed_stage="ffmpeg", vspipe_rc=1, ffmpeg_rc=1,
        vspipe_tail=GF3_ERR)
    assert r.is_cascading_encoder_failure() is True
    assert r.is_plugin_capability_failure() is True


# ------------------------------------------------ 结构守卫

def _tree():
    return ast.parse(open(_STAGE6, encoding="utf-8").read())


def test_main_never_falls_through_to_none():
    """`main()` 末尾必须是有值 return，且内部没有裸 `return`。

    这正是审核复现的那条路径：函数末尾没有 return → None → exit 0。
    """
    fn = next(n for n in _tree().body
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    last = fn.body[-1]
    assert isinstance(last, ast.Return) and last.value is not None, (
        f"main() 最后一条语句是 {type(last).__name__}，会隐式返回 None")
    for node in ast.walk(fn):
        if isinstance(node, ast.Return):
            assert node.value is not None, f"第 {node.lineno} 行有裸 return"


def test_main_delegates_to_the_testable_dispatcher():
    src = open(_STAGE6, encoding="utf-8").read()
    assert "dispatch_result(" in src
    i = src.index("rc, _degraded = dispatch_result(")
    assert i > src.index("def main(")


def test_fallback_uses_halved_intensity_and_the_addgrain_builder():
    """回退强度规则沿用 v6.32 §7.17 / FIX-P9-9：三类强度乘 0.5。"""
    src = open(_STAGE6, encoding="utf-8").read()
    i = src.index("def _run_fallback")
    block = src[i:i + 900]
    for tok in ("g1str * 0.5", "g2str * 0.5", "g3str * 0.5",
                "use_addgrainc=True", 'fallback_label="AddGrainC"'):
        assert tok in block, f"回退块缺少 {tok}"


def test_degraded_note_is_written_only_on_fallback_success(tmp_path):
    out = tmp_path / "step6_grain_p.mkv"
    out.write_bytes(b"X" * 16)
    s6._write_degraded_note(str(out), _plugin_fail(), _ok(),
                            8.0, 6.0, 6.0, 0.2, _Log())
    note = tmp_path / "step6_grain_p.mkv.degraded.json"
    assert note.exists(), "降级必须落侧车，否则读报告的人会以为用的 GF3"
    import json
    doc = json.loads(note.read_text(encoding="utf-8"))
    assert doc["degraded"] is True
    assert doc["plugin_used"] == "AddGrainC"
    assert doc["intensity"]["applied_scale"] == 0.5
