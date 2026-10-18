"""R17 回归：Deblock_QED 传了不存在的 `quant` 参数 → dv_digital 整条路径失败。

**实测故障**（选项矩阵 v8，9 例中唯一失败）：
```
CASE path_dv_digital RC=84 NO_ARTIFACT
vspipe: TypeError: Deblock_QED() got an unexpected keyword argument 'quant'
        clip = haf.Deblock_QED(clip, quant=25)
stage1 STAGE1_FAILED rc=84
```
havsfunc r33 的真实签名是
`Deblock_QED(clp, quant1=24, quant2=26, aOff1=1, bOff1=2, aOff2=1,
bOff2=2, uv=3)` —— **没有 `quant`**。

后果：`config.yaml` 里 `paths.dv_digital.deblock = "Deblock_QED"`
这一条配置**从未真正执行过**，且 dv_digital 素材一进来就在 stage1 崩。

与 R6-08 / R4 的 `pre_pad` / `python -c` 分号属同一族：
**照文档/直觉猜第三方 API 的参数名**，而没有对签名。
"""

import inspect
import os
import re

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

STAGE1 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "video_restoration_pipeline", "stages", "stage1_preprocess.py")


def _source():
    with open(STAGE1, encoding="utf-8") as fh:
        return fh.read()


def test_no_bare_quant_kwarg():
    """反例：vpy 生成代码里不得再出现 `Deblock_QED(clip, quant=`。"""
    src = _source()
    assert not re.search(r"Deblock_QED\(\s*clip\s*,\s*quant\s*=", src), \
        "Deblock_QED 不得使用不存在的 quant 参数（R17）"


def test_uses_quant1_quant2():
    src = _source()
    assert "quant1=" in src and "quant2=" in src, \
        "应使用 havsfunc r33 真实的 quant1/quant2"


@pytest.mark.skipif(os.name == "nt" and not os.path.isdir(
    os.path.join(os.path.dirname(STAGE1), "..", "..", "tools_r4")),
    reason="需要项目自带的 Portable R72 才能导入 havsfunc")
def test_signature_binds_with_real_havsfunc():
    """用真实 havsfunc 验证生成的调用能被接受。"""
    import subprocess
    root = os.path.dirname(STAGE1)
    py = os.path.join(root, "..", "..", "tools_r4", "vs_portable", "r72",
                      "python.exe")
    py = os.path.normpath(py)
    if not os.path.exists(py):
        pytest.skip("Portable R72 不存在")
    code = (
        "import inspect, havsfunc as haf\n"
        "sig = inspect.signature(haf.Deblock_QED)\n"
        "sig.bind(None, quant1=24, quant2=24)\n"
        "try:\n"
        "    sig.bind(None, quant=25)\n"
        "    print('QUANT_ACCEPTED')\n"
        "except TypeError:\n"
        "    print('QUANT_REJECTED')\n"
    )
    r = subprocess.run([py, "-c", code], capture_output=True, text=True,
                       cwd=os.path.normpath(os.path.join(root, "..", "..")))
    out = r.stdout or ""
    assert "QUANT_ACCEPTED" not in out, \
        "havsfunc 不应接受 quant；若接受说明环境变了，需重新评估 R17"
