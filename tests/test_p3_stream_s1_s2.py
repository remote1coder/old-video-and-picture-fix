# -*- coding: utf-8 -*-
"""P3 / G8：S1 只出 vpy、S2 只出结论 —— 不写 step1 / step2 媒体。

## 文档依据

实施包 §5.2：S2「变成检查消费者/结构化报告；**不创建 step2 媒体**」；
设计 §5.1：检查遍 = 原源 → **同一 stage1 图** → 检查 → null sink。
G8 通过条件：「不出现 stage1 全片临时视频」。

## 这些断言为什么不是形式检查

真正要防的是"看起来走了流式、其实还是落了盘"：
* stage1 若在 emit_only 分支之后**继续往下走**，就会写 step1；
* stage2 若在 stream 分支之后继续 `compute_param_hash(stage1_output)`，
  会拿空路径算哈希并尝试拷贝（实测报 `Failed to copy stage1 to step2`）。

所以判据是**分支必须 return**，且 stream 分支不得引用文件路径。
"""
import io
import os
import re

import pytest

import conftest  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_S1 = os.path.join(ROOT, "video_restoration_pipeline", "stages",
                   "stage1_preprocess.py")
_S2 = os.path.join(ROOT, "video_restoration_pipeline", "stages",
                   "stage2_traditional_check.py")


def _src(p):
    with io.open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


def test_stage1_stream_branch_returns_before_encoding():
    """emit_only 分支必须在 vspipe→ffmpeg 编码之前 return 0。"""
    s = _src(_S1)
    i_branch = s.index('if bool(_stream_cfg.get("emit_only")):')
    i_render = s.index("# Render with vspipe -> ffmpeg", i_branch)
    block = s[i_branch:i_render]
    assert "return 0" in block, "emit_only 分支没有 return，会继续写 step1"
    assert "intermediate_written" in block, "没有声明没有落盘"
    assert "vspipe_cmd" not in block, "emit_only 分支里不该出现 vspipe 渲染"


def test_stage1_stream_writes_script_identity():
    """交接单必须带 vpy 摘要，否则下游无法证明"同一张图"。"""
    s = _src(_S1)
    assert '"vs_script_sha256"' in s
    assert '"schema": "stream_s1/1"' in s


def test_stage2_stream_branch_skips_the_artifact_copy():
    """stream 模式必须在拷贝 step2 **之前** return，且日志说明不落盘。"""
    s = _src(_S2)
    i_guard = s.index("if _vpy:")
    i_copy = s.index("compute_param_hash(stage1_output, params)")
    # 取 return 0 附近那段（guard 之后、copy 之前）
    block = s[i_guard:i_copy]
    assert "return 0" in block, "stream 分支没 return，会继续拷 step2 媒体"
    assert "STAGE2_STREAM_NO_ARTIFACT" in block


def test_stage2_idet_on_vpy_drains_vspipe_stderr():
    """vspipe 的 stderr 必须落文件/DEVNULL，不能挂在管道上。

    stage1 踩过：PIPE 不读 → 64KB 满 → vspipe 阻塞 → ffmpeg 读到无效数据。
    这条用源码断言钉住，避免以后"顺手"改成 PIPE。
    """
    s = _src(_S2)
    i = s.index("def _run_idet_on_vpy")
    block = s[i:i + 2000]
    assert "subprocess.DEVNULL" in block
    assert 'stderr=subprocess.PIPE' not in block.replace(" ", "")


def test_stage2_stream_handles_missing_script():
    s = _src(_S2)
    assert "STAGE2_STREAM_VPY_MISSING" in s
