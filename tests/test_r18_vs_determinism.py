# -*- coding: utf-8 -*-
"""第十八轮追加：VapourSynth 渲染确定性开关。

## 背景（2026-11-06 实测）

同一个 vpy、同一个源，**连跑两次结果不同**：

```
完全相同 30 帧 / 有差异 30 帧
汇总 PSNR  y:48.71 u:51.53 v:39.63 average:45.48 min:42.07
```

逐段定位（累加滤镜，每档跑两次）：

| 变体 | 差异帧 | PSNR min | 说明 |
|---|---|---|---|
| source + resize | 0/60 | inf | 确定 |
| + TemporalDegrain | 10/60 | 131.67 | **源头**（0.005 LSB） |
| + Deblock | 8/60 | 144.10 | 未放大 |
| + Stab（Depan） | **33/60** | **41.12** | **放大 400 倍** |
| + 颜色矩阵（完整） | 27/60 | 42.07 | 1.8 LSB |

线程数扫描：默认 4.95s/35 帧不同 → 8:5.72s/27 → 4:7.39s/27 →
2:12.25s/13 → **1:23.47s/0（完全确定）**。

根因：VapourSynth 帧级并行下，mvtools 的运动估计从"相邻块已确定的运动
向量"取 predictor，平局时选哪个取决于评价顺序。

## 本文件守什么

1. `wants_deterministic()` 的解析（坏配置绝不能把生产变慢 4.78 倍）
2. **`num_threads` 必须在任何 filter 创建之前** —— 这条我踩过：
   写在脚本末尾时设置不生效，于是得出"单线程仍非确定"的假结论。
3. 默认关闭（生产要速度）
4. stage1 / stage6 的生成结果真的带上了这一行，且位置正确
"""
import os
import re
import sys

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import vs_render as vr  # noqa: E402


# ------------------------------------------------ 开关解析

def test_default_is_off():
    """默认必须关闭 —— 4.78 倍代价不能默认承担。"""
    assert vr.wants_deterministic({}) is False
    assert vr.wants_deterministic(None) is False
    assert vr.wants_deterministic({"vs_deterministic": False}) is False


def test_truthy_forms_are_accepted():
    for v in (True, 1, 1.0, "true", "True", "1", "yes", "on"):
        assert vr.wants_deterministic({"vs_deterministic": v}) is True, v


def test_falsy_and_garbage_never_enable_it():
    """坏配置绝不能把它打开（否则生产莫名慢 4.78 倍）。"""
    for v in (False, 0, 0.0, "", "false", "no", "off", [], {}, None, "maybe"):
        assert vr.wants_deterministic({"vs_deterministic": v}) is False, v


def test_broken_params_container_does_not_raise():
    class Broken:
        def get(self, *a, **k):
            raise RuntimeError("boom")

    assert vr.wants_deterministic(Broken()) is False


# ------------------------------------------------ 生成的代码

def test_preamble_is_empty_when_disabled():
    assert vr.determinism_preamble({}) == []


def test_preamble_sets_single_thread():
    lines = vr.determinism_preamble({"vs_deterministic": True})
    joined = "\n".join(lines)
    assert "core.num_threads = 1" in joined
    # 必须能看出为什么 —— 否则下一个人只会看到一行魔法
    assert "4.78" in joined or "非确定" in joined


# ------------------------------------------------ 位置守卫（核心）

def test_preamble_comes_before_any_filter_in_stage1():
    """**核心守卫**：`num_threads` 必须出现在 `core = vs.core` 之后、
    任何 `core.<plugin>.` 调用之前。

    这条不是洁癖：VapourSynth 里 `num_threads` 只对**之后创建**的 filter
    生效。我第一版写在了脚本末尾，于是 source/resize 全都按多线程建好了，
    设置形同虚设 —— 直接导致"单线程仍非确定"的错误结论。
    """
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage1_preprocess.py"),
               encoding="utf-8", errors="replace").read()
    i_core = src.find('lines.append("core = vs.core")')
    assert i_core > 0, "找不到 vpy 的 core 初始化"
    i_det = src.find("determinism_preamble", i_core)
    assert i_det > i_core, \
        "确定性 preamble 必须紧跟在 core = vs.core 之后"
    i_source = src.find("core.ffms2.Source", i_core)
    assert i_source > i_det, \
        "preamble 必须在 source 创建（第一个 filter）之前"


def test_stage1_script_contains_single_thread_when_enabled():
    """跑真实生成函数，确认输出文本里确实带上了这一行且位置正确。"""
    import stage1_preprocess as s1

    params = {
        "denoise": "",
        "deblock": "",
        "stabilize": "",
        "denoise_strength": 16,
        "denoise_bs": 16,
        "deblock_quant": 24,
        "deblock_auto": False,
        "field_order": "progressive",
        "vs_bit_depth": 16,
        "color_matrix_in": "170m",
        "color_matrix_out": "709",
        "crop_auto": False,
        "ivtc_auto": False,
        "vs_deterministic": True,
    }
    script = s1.generate_vs_script("E:/fake/in.mp4", params, "job1")

    assert "core.num_threads = 1" in script, "开启后必须写入单线程设置"
    i_core = script.find("core = vs.core")
    i_nt = script.find("core.num_threads = 1")
    i_src = script.find("core.ffms2.Source")
    assert 0 < i_core < i_nt, "num_threads 必须在 core 初始化之后"
    assert i_nt < i_src, "num_threads 必须在第一个 filter 之前"


def _base_params(**kw):
    p = {
        "denoise": "", "deblock": "", "stabilize": "",
        "denoise_strength": 16, "denoise_bs": 16,
        "deblock_quant": 24, "deblock_auto": False,
        "field_order": "progressive", "vs_bit_depth": 16,
        "color_matrix_in": "170m", "color_matrix_out": "709",
        "crop_auto": False, "ivtc_auto": False,
    }
    p.update(kw)
    return p


def test_stage1_script_does_not_single_thread_by_default():
    """默认**不得**因为"确定性"把 VS 压成单线程（那会慢 4.78 倍）。

    R25 起默认改为按资源上限（整机 80%）限制线程数 —— 这是用户明确要求的
    "任何一项不超过 80%，慢一点可以接受"。所以判据从"不得出现
    core.num_threads"改成"不得等于 1，且必须等于资源上限"：
    只要它还是多线程，就不存在"无端慢 4.78 倍"这条风险。
    """
    import os as _os
    import re as _re
    import stage1_preprocess as s1

    script = s1.generate_vs_script("E:/fake/in.mp4", _base_params(), "job1")
    m = _re.search(r"core\.num_threads = (\d+)", script)
    assert m, "默认应当按资源上限写入线程数（见 resource_limits）"
    n = int(m.group(1))
    assert n >= 2, f"默认不得单线程化（实测会是 4.78 倍慢）：{n}"
    expect = max(1, int((_os.cpu_count() or 4) * 0.8))
    assert n == expect, f"线程数应等于整机 80% 上限：{n} != {expect}"


def test_stage1_script_has_no_thread_override_when_limits_disabled():
    """关掉资源限幅 = 回到旧行为（完全不写 num_threads）。"""
    import stage1_preprocess as s1

    script = s1.generate_vs_script(
        "E:/fake/in.mp4",
        _base_params(resource_limits={"enabled": False}), "job1")
    assert "core.num_threads" not in script


def test_stage6_determinism_block_toggles():
    import stage6_grain_synthesis as s6

    on = s6._determinism_block(True)
    off = s6._determinism_block(False)
    assert "core.num_threads = 1" in on
    assert off == ""


# ------------------------------------------------ 配置默认

def test_config_defaults_to_false():
    import io
    cfg = io.open(os.path.join(_ROOT, "video_restoration_pipeline",
                               "config.yaml"), encoding="utf-8").read()
    assert "vs:" in cfg
    assert re.search(r"^\s*deterministic:\s*false\s*$", cfg, re.M), \
        "vs.deterministic 默认必须是 false"


def test_noise_constant_is_documented():
    """本底噪声要能被判据引用，而不是各处手抄数字。"""
    n = vr.NONDETERMINISM_NOISE
    assert n["psnr_min"] == 42.07
    assert n["psnr_y"] > n["psnr_min"]
    assert "本底噪声" in n["note"]
