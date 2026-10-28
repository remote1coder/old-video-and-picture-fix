# -*- coding: utf-8 -*-
"""R13-01：stage7 全零同步分量会让整个阶段崩溃。

**缺陷**

`_validate_av_sync()` 里 `_worst` 的初值是
`{"level": 0, "mag": 0.0, "thr": None, "lag": None}`，
而循环内的更新条件是"等级更高 或 同等级且幅值更大"。

当 start / end / span **三个分量都恰好为 0** 时：
- 每个分量算出 `_lv=0, _a=0`
- `0 > 0` 与 `0 > 0.0` 都不成立 → `_worst` **永远不会被更新**
- `thr` 保持 `None`

随后一句（而且是**死变量**，算完从不使用）

```python
_worst_txt = f"... threshold {_worst['thr']:.0f}ms)"
```

对 `None` 做 `:.0f` → `TypeError` → 外层返回 82。

**为什么危险**：源片与成品都同步（最常见的好情况！），
或两者保持同一个非零原始偏移（老素材，也是好情况），都会命中。
即"越正常越容易崩"。

测试全部调用**真实** `_validate_av_sync()`，只打桩探测函数。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage7_encode as s7  # noqa: E402


class _Log:
    def __init__(self):
        self.events = []

    def _rec(self, lv, ev, msg="", **kw):
        self.events.append((lv, ev, msg))

    def info(self, ev, msg="", **kw):
        self._rec("info", ev, msg)

    def warning(self, ev, msg="", **kw):
        self._rec("warning", ev, msg)

    def error(self, ev, msg="", **kw):
        self._rec("error", ev, msg)

    def debug(self, *a, **k):
        pass

    def has(self, ev):
        return any(e == ev for _lv, e, _m in self.events)


def _patch(monkeypatch, video, audio, priming=(0.0, 0.0), ref_vec=None):
    """video/audio = (raw_start, span)；priming = (视频, 音频)。"""
    iv = {"v": video, "a": audio}
    pv = {"v": priming[0], "a": priming[1]}
    monkeypatch.setattr(s7, "_probe_audio_presence", lambda p: "present",
                        raising=False)
    monkeypatch.setattr(s7, "_probe_stream_start",
                        lambda p, st: iv[st][0], raising=False)
    monkeypatch.setattr(s7, "_get_stream_duration",
                        lambda p, st: iv[st][1], raising=False)
    monkeypatch.setattr(s7, "_probe_encoder_priming",
                        lambda p, st: pv[st], raising=False)
    # 参考向量：给了就走相对判据，没给走绝对判据
    monkeypatch.setattr(
        s7, "_measure_av_drift_vec",
        (lambda p: ref_vec) if ref_vec is not None else (lambda p: None),
        raising=False)


# ------------------------------------------------ 触发崩溃的那一大类

def test_all_zero_components_pass_instead_of_crash(monkeypatch):
    """三个分量全为 0（完全同步）→ 必须 PASS，不能 TypeError。

    这是最正常的情况：源片和成品都没漂。
    """
    _patch(monkeypatch, video=(0.0, 10.0), audio=(0.0, 10.0))
    log = _Log()
    rc = s7._validate_av_sync("out.mkv", log, {})
    assert rc == 0, f"完全同步应返回 0，实际 {rc}"
    assert log.has("STAGE7_SYNC_OK"), [e for _l, e, _m in log.events]


def test_identical_nonzero_offset_passes(monkeypatch):
    """源片与成品保持**同一个非零原始偏移** → 相对差为 0 → PASS。

    老素材常自带 300–500ms 偏移；这是"保留原始偏移"，不是流程引入的错误。
    """
    # 成品自身 start 差 +0.5s、end 差 +0.5s、span 差 0
    _patch(monkeypatch, video=(0.0, 10.0), audio=(0.5, 10.0),
           ref_vec=(0.5, 0.5, 0.0))     # 源片也是 +0.5
    log = _Log()
    rc = s7._validate_av_sync("out.mkv", log, {}, reference_path="src.mkv")
    assert rc == 0, f"原始偏移应被保留，实际 {rc}"


def test_zero_delta_after_priming(monkeypatch):
    """priming 抵消后增量为 0 → PASS。

    起点用 content_start（raw + priming），终点用 raw_end。
    两端 priming 相同 → 起点差不变、终点差不变 → 三分量都是 0。
    """
    _patch(monkeypatch, video=(0.0, 10.0), audio=(0.0, 10.0),
           priming=(0.1, 0.1))
    log = _Log()
    rc = s7._validate_av_sync("out.mkv", log, {})
    assert rc == 0, f"priming 相同应无净差，实际 {rc}"


# ------------------------------------------------ 边界与阈值

def test_threshold_equality_is_not_a_failure(monkeypatch):
    """恰好等于失败线（200ms 超前）→ 判据是 `>` 不是 `>=`，不应失败。"""
    _patch(monkeypatch, video=(0.0, 10.0), audio=(-0.2, 10.0))
    log = _Log()
    rc = s7._validate_av_sync("out.mkv", log, {})
    assert rc == 0, f"200ms 恰好等于超前失败线，不应失败，实际 {rc}"


def test_slight_drift_in_both_directions(monkeypatch):
    """轻微正负漂移都在 100ms 噪声地板内 → 不失败。"""
    for off in (0.05, -0.05):
        _patch(monkeypatch, video=(0.0, 10.0), audio=(off, 10.0))
        log = _Log()
        rc = s7._validate_av_sync("out.mkv", log, {})
        assert rc == 0, f"{off*1000:.0f}ms 漂移不应失败，实际 {rc}"


# ------------------------------------------------ 回归：上一轮的反例仍要拦住

def test_regression_report_counterexample_still_fails(monkeypatch):
    """审核的 -250/+50/+300ms 反例：起点超前 250ms 超 200ms 线 → 必须 86。"""
    _patch(monkeypatch, video=(0.0, 10.0), audio=(-0.25, 10.30),
           ref_vec=(0.0, 0.0, 0.0))
    log = _Log()
    rc = s7._validate_av_sync("out.mkv", log, {}, reference_path="src.mkv")
    assert rc == 86, f"起点超前 250ms 应失败，实际 {rc}"
    assert log.has("STAGE7_SYNC_FAIL")


def test_no_dead_formatting_of_none():
    """源码护栏：不得再对 _worst['thr'] 做无条件格式化。

    这条守的是"死变量被删掉/不再对 None 格式化"，
    真实效果由上面几条行为用例覆盖。
    """
    src = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                       "stage7_encode.py")
    text = open(src, encoding="utf-8", errors="replace").read()
    assert "_worst_txt" not in text, \
        "_worst_txt 是死变量，且曾对 None 做 :.0f 导致崩溃，应删除"
