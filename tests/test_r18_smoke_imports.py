# -*- coding: utf-8 -*-
"""导入冒烟：本轮动过的模块都必须能**真正导入**。

为什么需要它
------------
`py_compile` 只查语法，**不查名字**。本轮就踩过一次：

- `stage4_postprocess` 里新加了 `build_filter_chain(params: Dict[str, Any])`，
  但模块里没有 `from typing import Any, Dict` → 语法通过、**导入就 NameError**；
- `stage3_ai_upscale` 里写了 `VIDEO_IMPL_VERSION`（只在 main 里定义），
  又包在 `except Exception` 里 → 运行期静默返回 86，"融合永远失败"却没人知道。

这类错误只有在**导入 + 调用**时才会露头，所以这里把动过的模块逐个导入，
并对新增的公共函数做一次最小调用。
"""
import importlib
import os
import sys

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def test_shared_modules_import_cleanly():
    for name in ("cancel_guard", "checkpoint", "publisher", "publish_journal",
                 "timeline_plan", "vfr_timecodes", "time_mapping",
                 "run_plan", "publish_lock"):
        mod = importlib.import_module(name)
        assert mod is not None, f"{name} 导入失败"


def test_video_stages_import_cleanly():
    for name in ("stage1_preprocess", "stage3_ai_upscale",
                 "stage4_postprocess", "stage5_ai_face",
                 "stage6_grain_synthesis", "stage7_encode"):
        mod = importlib.import_module(name)
        assert mod is not None, f"{name} 导入失败"


def test_stage4_filter_chain_is_callable():
    """`build_filter_chain` 必须真的能调用（不只是能导入）。"""
    s4 = importlib.import_module("stage4_postprocess")
    chain = s4.build_filter_chain({})
    assert isinstance(chain, str) and chain.startswith("[0:v]"), chain[:40]
    assert chain.endswith("[vout]"), chain[-20:]
    # 带饱和度/对比度时必须更长（说明可选段真的生效）
    longer = s4.build_filter_chain({"post_saturation": 1.1, "post_contrast": 1.05})
    assert len(longer) > len(chain)


def test_stage3_helpers_callable():
    s3 = importlib.import_module("stage3_ai_upscale")
    assert s3._chain_mode({}) == "none"
    assert s3._chain_mode({"chain_mode": "stage34"}) == "stage34"
    assert s3._chain_mode({"chain_mode": {"mode": "both"}}) == "both"
    assert len(s3._sha256_text("abc")) == 64
