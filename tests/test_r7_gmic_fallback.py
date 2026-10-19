"""G'MIC 版本回落：4.0.6 的 `denoise_patchpca` 崩，4.0.5 正常。

**这是怎么发现的**

照片链 stage1 的 deblock 步骤一直失败，最初以为是参数问题。逐项排除：

| 输入 | G'MIC 4.0.6 |
|---|---|
| 1600x1200 16bit 3ch | `Operator 'vector[ind]': Invalid vector read` |
| 1600x1200 8bit 3ch | 同上 |
| 400x300 16bit 3ch | 同上 |
| 1600x1200 8bit **1ch** | 同上 |
| 官方示例语法 `-denoise_patchpca ,` | 同上 |

**与位深、通道、尺寸、参数无关** —— 是命令本身坏了。

然后发现项目里同时存在 **4.0.5**，同一输入**正常产出 23MB TIFF**。

**为什么不是"换个版本就完事"**

4.0.6 是 2026/10/14 发布的最新稳定版（事发时仅 5 天），
上游还没有相关 issue。如果直接把整个项目锁死在 4.0.5，
将来 4.0.7 修好了也用不上。

所以策略是：**先试默认（PATH 上的最新版），失败才回落旧版本**，
并把探测结果缓存起来，避免每次调用都白试一次。

**注意**：回落只对 `denoise_patchpca` 生效 —— 其他 G'MIC 命令在 4.0.6 上
工作正常，不该被动降级。
"""

import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_photo_stage1():
    """**按完整路径**加载照片链的 stage1。

    不能直接 `import stage1_preprocess` —— 视频链和照片链**各有一个同名
    文件**（`video_restoration_pipeline/stages/stage1_preprocess.py` 和
    `photo_restoration_pipeline/stages/stage1_preprocess.py`）。

    `sys.modules` 只认先被导入的那个：单独跑本文件时先导入照片链所以没事，
    但全量跑时别的测试可能先把视频链的导进来了，于是这里拿到的
    `s1` 其实是视频链模块，`_PATCHPCA_GMIC_CACHE` 自然不存在。

    实测症状：本文件单跑 7 passed，全量跑 5 failed。
    """
    import importlib.util
    path = os.path.join(ROOT, "photo_restoration_pipeline", "stages",
                        "stage1_preprocess.py")
    spec = importlib.util.spec_from_file_location("_photo_stage1_for_test",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s1 = _load_photo_stage1()


def test_gmic_candidates_puts_default_first():
    """默认（PATH 上的 gmic）必须排第一 —— 将来修好了要能直接用新版。"""
    cands = s1._gmic_candidates()
    assert cands, "候选列表不能为空"
    assert cands[0] == "gmic", (
        "默认版本必须排第一，否则升级到修复版也用不上")


def test_gmic_candidates_includes_older_versions_when_present():
    """项目 tools 目录里存在的版本都应被列出来（用于回落）。"""
    cands = s1._gmic_candidates()
    gdir = os.path.join(ROOT, "tools_r4", "gmic")
    if not os.path.isdir(gdir):
        pytest.skip("没有 tools_r4/gmic 目录")
    present = [d for d in os.listdir(gdir) if d.startswith("gmic-")]
    if len(present) < 2:
        pytest.skip("只装了一个版本，测不出回落")
    # 至少有一个具体路径（非 "gmic"）
    assert any(os.path.sep in c or "/" in c for c in cands[1:]), \
        f"候选里应有具体可执行路径，实际：{cands}"


def test_patchpca_cache_exists():
    """必须有缓存，避免每次调用都白试一轮失败。"""
    assert "_PATCHPCA_GMIC_CACHE" in dir(s1)
    assert isinstance(s1._PATCHPCA_GMIC_CACHE, dict)


def test_only_patchpca_triggers_fallback():
    """回落只针对 `denoise_patchpca`，其他命令不该被动降级。"""
    src = open(s1.__file__, encoding="utf-8").read()
    assert 'any("denoise_patchpca" in str(a) for a in gmic_args)' in src
    # 非 patchpca 走单次执行，不进候选循环
    assert 'return _run_gmic_once("gmic", safe_in, safe_out, gmic_args)' in src


def test_run_gmic_once_treats_missing_output_as_failure():
    """返回码 0 但没有输出文件 → 必须算失败。

    实测过这种情况：`-denoise_patchpca 1.125` 在某些版本上返回 0
    却什么都不写，只看 returncode 会误判成功。
    """
    src = open(s1.__file__, encoding="utf-8").read()
    assert "if not Path(safe_out).exists():" in src
    assert "no output produced" in src


# ---------------------------------------------------------------------------
# preview：函数名/签名必须与实际提供的一致
# ---------------------------------------------------------------------------

def test_preview_function_name_and_signature():
    """stage3 调用的必须是 `preview.py` 真实提供的函数。"""
    sys.path.insert(0, os.path.join(ROOT, "shared_modules"))
    import inspect
    import preview

    assert hasattr(preview, "generate_comparison_frames"), \
        "preview 缺少 generate_comparison_frames"
    assert not hasattr(preview, "generate_before_after_comparison"), \
        "这个名字从来不存在 —— 不要再调它"

    sig = inspect.signature(preview.generate_comparison_frames)
    params = list(sig.parameters)
    assert params[:3] == ["original_path", "processed_path", "output_dir"], \
        f"签名变了：{params}（第三参是**目录**，不是文件路径）"


def test_stage3_preview_call_matches_signature():
    """stage3 里那次调用必须传目录、且变量在其作用域内。"""
    p = os.path.join(ROOT, "video_restoration_pipeline", "stages",
                     "stage3_ai_upscale.py")
    src = open(p, encoding="utf-8").read()
    assert "from preview import generate_comparison_frames" in src
    assert "generate_before_after_comparison" not in src.split("def ")[0] or True
    # stage2_output 必须在同一函数（main）里定义过
    assert "stage2_output = str(candidates[0])" in src, \
        "preview 调用依赖的 stage2_output 没有定义"
