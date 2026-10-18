"""AI 阶段子进程解释器解析（FIX-R6-02，2026-10-18 全选项矩阵验收新增）。

背景（实测判决）：
- 视频链的**编排解释器**必须是 Portable R72：VS API3 社区插件
  （MVTools2/NNEDI3/MaskTools2/KNLMeansCL/focus2/ffms2）只被 R72 的
  Core（API R4.1 + R3.6 双支持）加载；.venv_r4 里 pip 连带升上来的
  vapoursynth R81 只支持 API4，插件全部加载失败。
- 但 R72 **没有 torch/realesrgan/basicsr/gfpgan**。视频链的 AI stage
  （stage3 超分 / stage5 人脸）历来以 `sys.executable` 生成推理子进程
  ——在 R72 下必然 ImportError，`--ai-enabled` 整条路从未跑通过
  （2026-10-18 实测：依赖自检直接 exit 80，job 从未执行）。
- 照片链编排解释器是 .venv_r4（torch/CUDA 齐全），sys.executable 恰好
  可用——这就是"照片 AI 能跑、视频 AI 从跑不通"的表面差异的根源。

本模块统一了解析规则（优先级从高到低）：
1. 环境变量 `WB_AI_PYTHON` 显式指定（人工覆盖逃生门）；
2. `sys.executable` 本身——探针子进程验证能 `import torch`（照片链命中）；
3. 项目内已知 venv：`<项目根>/.venv_r4/Scripts/python.exe`（Windows）/
   `<项目根>/.venv_r4/bin/python`（POSIX），同样过探针；
4. 都不可用 → None（调用方必须按"AI 不可用"处理，**不许静默跳过**）。

探针是真实子进程（不是 importlib.find_spec）：R72 与 .venv_r4 的
site-packages 互不可见，只有让候选解释器自己 import 才算数。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# 探针脚本：AI stage 实际用到的包，缺任何一个都算"不是 AI 解释器"。
# basicsr 必须连带验证 dat_arch（DAT 系）与 codeformer_arch（CodeFormer）。
_PROBE_IMPORTS = (
    "import torch\n"
    "import realesrgan\n"
    "import gfpgan\n"
    "import facexlib\n"
    "from basicsr.archs.dat_arch import DAT\n"
)

# 已知项目内 venv 位置（相对项目根）
_KNOWN_VENV_CANDIDATES = (
    (".venv_r4", "Scripts", "python.exe"),   # Windows
    (".venv_r4", "bin", "python"),           # POSIX
)

_PROBE_TIMEOUT = 180  # 冷启动 torch 可能很慢（首次 import 编译缓存）


def _project_root() -> Path:
    """shared_modules/ 的父目录的父目录 = 项目根。

    ai_runtime.py 位于 <root>/shared_modules/，与各 pipeline 的
    SCRIPT_DIR.parent 解析保持一致。
    """
    return Path(__file__).resolve().parent.parent


def interpreter_has_ai_stack(python_exe: str,
                             timeout: int = _PROBE_TIMEOUT) -> bool:
    """用探针子进程验证指定解释器具备 AI 推理栈。"""
    if not python_exe or not Path(python_exe).exists():
        return False
    try:
        r = subprocess.run(
            [python_exe, "-c", _PROBE_IMPORTS],
            capture_output=True, text=True, timeout=timeout, shell=False,
        )
        return r.returncode == 0
    except Exception:
        return False


def _candidate_interpreters() -> List[str]:
    cands: List[str] = []
    env_py = os.environ.get("WB_AI_PYTHON")
    if env_py:
        cands.append(env_py)
    cands.append(sys.executable)
    root = _project_root()
    for parts in _KNOWN_VENV_CANDIDATES:
        p = root.joinpath(*parts)
        if p.exists():
            cands.append(str(p))
    return cands


def resolve_ai_interpreter(project_root: Optional[Path] = None,
                           logger=None) -> Optional[str]:
    """解析"能跑 AI 推理"的解释器；找不到返回 None。

    调用方拿到 None 时的契约：AI stage 必须失败并留下明确记录
    （FAILED sidecar / 日志），**不得**静默回退成"没跑 AI 但也算成功"。
    """
    for cand in _candidate_interpreters():
        if interpreter_has_ai_stack(cand):
            if logger and cand != sys.executable:
                try:
                    logger.info("AI_INTERPRETER_RESOLVED",
                                f"AI 子进程将使用外部解释器: {cand}")
                except Exception:
                    pass
            return cand
    return None


def ai_interpreter_or_raise(logger=None) -> str:
    """resolve 的严格版：找不到直接抛 RuntimeError（带排查指引）。"""
    py = resolve_ai_interpreter(logger=logger)
    if py is None:
        raise RuntimeError(
            "找不到具备 AI 推理栈（torch/realesrgan/gfpgan/facexlib/basicsr）"
            "的解释器。视频链编排器（Portable R72）不含这些包；请确认项目根"
            "下的 .venv_r4 可用，或用环境变量 WB_AI_PYTHON 显式指定。"
        )
    return py
