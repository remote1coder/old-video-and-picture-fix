"""GPEN（GAN Prior Embedded Network）推理架构，取自上游 ``yangxy/GPEN``。

**为什么单独放一个包**

上游用 ``from face_model.op import ...`` 这种**以 ``face_model`` 为顶层**
的绝对 import。直接并进 ``shared_modules/`` 会和项目自己的模块名冲突，
也会让 ``face_model`` 这个名字污染 sys.path。所以整体搬进子包，
由本文件把包目录插入 sys.path 后再转发，调用方只看到
``from gpen import FullGenerator``。

**为什么能在 Windows 上免编译运行**

上游的两个自定义算子（``fused_act.py`` / ``upfirdn2d.py``）都带
**纯 PyTorch 回退**，走 fused CUDA 内核的条件是
``platform.system() == 'Linux' and torch.cuda.is_available()``。
Windows 上该条件为假，自动落到
``scale * F.leaky_relu(...)`` 与 ``upfirdn2d_native(...)``，
**不需要** nvcc 编译 ``.cpp/.cu``。因此这里只保留 .py，
不复制 CUDA 源码（复制了也不会被用到）。

**权重**

``models/GPEN-BFR-512.pth``（284MB，HuggingFace ``akhaliq/GPEN-BFR-512``）。
实测 ``FullGenerator(512, 512, 8, channel_multiplier=2, narrow=1)``
载入它是 **missing=0 / unexpected=0** —— 这个 0/0 是校验权重与架构
是否配套的依据，见 photo stage3 的 ``_run_gpen_pytorch``。
"""
import sys as _sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

from face_model.gpen_model import FullGenerator  # noqa: E402,F401

__all__ = ["FullGenerator"]
