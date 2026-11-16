"""Shared import bootstrap for the V01–V08 regression tests."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "shared_modules"
VIDEO = ROOT / "video_restoration_pipeline"
PHOTO_STAGES = ROOT / "photo_restoration_pipeline" / "stages"

for _p in (str(SHARED), str(VIDEO), str(PHOTO_STAGES), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


#: **两条链里同名的模块**（文件名相同、内容不同）。
#:
#: 为什么需要这个：`stage3_ai_upscale` 在 `video_restoration_pipeline/stages/`
#: 和 `photo_restoration_pipeline/stages/` 各有一份，函数集不同
#: （视频有 `run_realesrgan_batch`，照片没有）。`sys.modules` 是按**模块名**
#: 缓存的 —— 谁先导入，后面的测试就一直拿到那一份。
#:
#: 实测后果：某个测试为了导入照片链的 `stage1_preprocess` 而
#: `sys.path.insert(0, photo_stages)`，之后视频链的测试
#: `import stage3_ai_upscale` 就拿到了**照片版**，报
#: `module 'stage3_ai_upscale' has no attribute 'run_realesrgan_batch'`。
#: 单跑那条测试是通过的，全量跑才炸 —— 典型的测试间状态污染。
_SAME_NAME_MODULES = (
    "stage0_audio_extract",
    "stage1_preprocess",
    "stage2_ai_pre",
    "stage2_traditional_check",
    "stage3_ai_upscale",
    "stage4_ai_face",
    "stage5_ai_face",
    "stage5_postprocess",
    "stage6_encode",
    "stage6_grain_synthesis",
    "stage7_encode",
)


@pytest.fixture(autouse=True)
def _isolate_chain_modules():
    """每个测试结束后把 `sys.path` 与同名模块缓存恢复原样。

    这样"某个测试为了导入另一条链而改 `sys.path`"不会传染给后面的测试。
    只恢复**已知同名**的那几个模块，不动其它缓存（避免重导入开销）。
    """
    _before_path = list(sys.path)
    _before_mods = {k: sys.modules.get(k) for k in _SAME_NAME_MODULES}
    yield
    sys.path[:] = _before_path
    for _k, _v in _before_mods.items():
        if _v is None:
            sys.modules.pop(_k, None)
        else:
            sys.modules[_k] = _v


def load_stage(chain: str, name: str):
    """按**绝对路径**加载某条链的 stage 模块。

    ```python
    s1 = load_stage("photo_restoration_pipeline", "stage1_preprocess")
    ```

    为什么不用 `sys.path.insert + import`：两条链有**同名模块**
    （`stage1_preprocess` / `stage3_ai_upscale` / …），谁先被导入，
    `sys.modules` 就缓存谁。测试文件在**收集阶段**执行顶层 import，
    那时任何 autouse fixture 都还没跑 —— 所以污染从收集就开始了。

    本函数用 `spec_from_file_location` 加载并**立即执行**，模块名带上链前缀
    （`_photo_stage1_preprocess`），既不进 `sys.modules` 的公共名字，
    也不动 `sys.path`。
    """
    import importlib.util
    path = ROOT / chain / "stages" / (name + ".py")
    if not path.is_file():
        raise FileNotFoundError(str(path))
    spec = importlib.util.spec_from_file_location(
        "_%s_%s" % (chain.split("_")[0], name), str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
