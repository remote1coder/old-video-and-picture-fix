# -*- coding: utf-8 -*-
"""资源上限：任何一项都不超过整机 80%（用户明确要求，宁可慢）。

## 为什么要有这一层

长片跑批会把整机吃满：`vspipe` 按逻辑核开线程、`libsvtav1` 占满逻辑核、
ffmpeg 解码/滤镜各自再开。多者叠加实际并发达标 100% 以上，轻则拖慢同机
其它工作，重则把内存/显存推到临界点让 40 分钟的跑批中途崩。

## 判据

1. 默认**开启**，`max_fraction=0.80`；线程数 = floor(cpu×0.8) − reserve；
2. `vspipe -t` **不是**线程数（那是 `--timecodes`），所以线程上限只能写进
   vpy 的 `core.num_threads`；这条必须被测试钉住，否则很容易「看起来限了」；
3. `core.num_threads` 必须插在**任何 filter 之前**（写在末尾无效）；
4. 开确定性时以 `num_threads=1` 为准，不再重复写；
5. 关掉限幅 = 回到旧行为（进程可以用满），但要走显式配置。
"""
import os
import re

import pytest

import conftest  # noqa: F401

import resource_limits as RL
import vs_render


def test_default_caps_threads_at_eighty_percent():
    plan = RL.resolve({}, {})
    assert plan["enabled"] is True
    assert plan["max_fraction"] == 0.8
    cpu = plan["cpu_count"]
    assert plan["threads"] == max(1, int(cpu * 0.8))
    assert plan["threads"] < cpu, "默认必须真的低于整机核数"


def test_explicit_config_wins():
    plan = RL.resolve({}, {"resource_limits": {"max_fraction": 0.5,
                                               "reserve_cores": 2}})
    # 参数侧优先：这里只给了 config
    assert plan["max_fraction"] == 0.5
    assert plan["threads"] <= max(1, int(plan["cpu_count"] * 0.5))


def test_can_be_disabled_explicitly():
    plan = RL.resolve({"resource_limits": {"enabled": False}}, {})
    assert plan["enabled"] is False
    assert RL.num_threads({"resource_limits": {"enabled": False}}) == (
        os.cpu_count() or 4)


def test_fraction_is_clamped_to_a_sane_range():
    """0.0 会让所有进程单线程或 0 线程；>1 等于没限。"""
    assert RL.resolve({"resource_limits": {"max_fraction": 0.0}},
                      {})["max_fraction"] >= 0.05
    assert RL.resolve({"resource_limits": {"max_fraction": 9}},
                      {})["max_fraction"] <= 1.0


def test_threads_never_below_one():
    plan = RL.resolve({"resource_limits": {"max_fraction": 0.05,
                                           "reserve_cores": 999}}, {})
    assert plan["threads"] == 1


def test_vspipe_dash_t_is_not_a_thread_flag():
    """`vspipe -t` 是 `--timecodes`，拿它限线程会让 vspipe 把数字当文件名。

    这条是回归：我第一版就是这么写的，读了 `vspipe --help` 才发现。
    现在模块里**不该**再有任何生成 `-t` 的函数。
    """
    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "shared_modules", "resource_limits.py"), encoding="utf-8").read()
    assert '"-t"' not in src, "resource_limits 里又出现了 -t（那是 --timecodes）"
    assert not hasattr(RL, "vspipe_args"), "vspipe_args 已被移除，别再引回来"


def test_thread_preamble_emits_core_num_threads():
    lines = vs_render.thread_preamble({"resource_limits": {"enabled": True,
                                                           "max_fraction": 0.5}})
    text = "\n".join(lines)
    assert "core.num_threads" in text, lines
    n = int(re.search(r"core\.num_threads = (\d+)", text).group(1))
    assert 1 <= n < (os.cpu_count() or 4)


def test_thread_preamble_yields_to_determinism():
    """确定性要的是 num_threads=1；再写一遍上限只会覆盖掉它。"""
    p = {"vs_deterministic": True,
         "resource_limits": {"enabled": True, "max_fraction": 0.8}}
    assert vs_render.thread_preamble(p) == []


def test_thread_preamble_empty_when_not_limiting():
    assert vs_render.thread_preamble({"resource_limits": {"enabled": False}}) == []


def test_stage6_puts_the_thread_block_before_any_filter():
    """`core.num_threads` 必须出现在第一个 filter 之前，否则不生效。

    这条不是样式问题：写在脚本末尾等于没写（R20 曾据此得出"单线程仍非
    确定"的假结论）。
    """
    import sys
    stages = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "video_restoration_pipeline", "stages")
    if stages not in sys.path:
        sys.path.insert(0, stages)
    import stage6_grain_synthesis as s6
    script = s6._build_vs_script(
        "in.mkv", "vs.YUV420P16", 1.0, 0.2, 0.2, 0.2,
        use_addgrainc=False, deterministic=False, seed=1, threads=2)
    i_core = script.index("core = vs.core")
    i_thr = script.index("core.num_threads = 2")
    i_first_filter = script.index("core.ffms2.Source")
    assert i_core < i_thr < i_first_filter, script[:300]


def test_stage6_thread_block_skipped_under_determinism():
    import sys
    stages = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "video_restoration_pipeline", "stages")
    if stages not in sys.path:
        sys.path.insert(0, stages)
    import stage6_grain_synthesis as s6
    script = s6._build_vs_script(
        "in.mkv", "vs.YUV420P16", 1.0, 0.2, 0.2, 0.2,
        use_addgrainc=False, deterministic=True, seed=1, threads=2)
    assert "core.num_threads = 2" not in script
