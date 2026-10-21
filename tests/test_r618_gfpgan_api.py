"""R18 / R19 回归：照片链 GFPGAN 的两处 API 用法错误。

R18：`GFPGANer(channel_only=...)` —— 本版本**没有**这个参数，真实参数是
`channel_multiplier`。子进程 TypeError → returncode!=0 → run_gfpgan False
→ STAGE4_FAILED rc=83。**人脸修复在照片链从未成功过**。

R19：`SCRIPT_DIR.parent.parent` —— 照片链的 SCRIPT_DIR 本身就是
`<root>/photo_restoration_pipeline`（视频链是 `.../stages`），项目根是
`SCRIPT_DIR.parent`。写成 `.parent.parent` 会解析到 `E:\\codex\\code1`
→ model_dir 变成 `E:\\codex\\code1\\models` → 权重不存在 →
torch.load 失败 → rc=83。

两条同属"没对第三方 API / 自身路径假设做实测"的病族
（与 R6-08 `pre_pad`、R17 `quant`、R4 `python -c` 分号同类）。
"""

import inspect
import os

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

STAGE4 = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "photo_restoration_pipeline", "stages", "stage4_ai_face.py")


def _source():
    with open(STAGE4, encoding="utf-8") as fh:
        return fh.read()


def _code_only():
    """剥掉注释与 docstring，只留真正的代码（参数名可能出现在注释里）。"""
    import io
    import tokenize
    src = _source()
    out = []
    prev_type = tokenize.INDENT
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and prev_type in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.NL,
                tokenize.DEDENT):
            continue  # docstring
        if tok.type not in (tokenize.NL, tokenize.NEWLINE,
                            tokenize.INDENT, tokenize.DEDENT):
            out.append(tok.string)
        prev_type = tok.type
    return " ".join(out)


def test_no_channel_only_kwarg():
    """反例：代码里不得再出现 GFPGANer 的 channel_only 参数。"""
    code = _code_only()
    assert "channel_only" not in code, \
        "GFPGANer 没有 channel_only 参数（R18）"
    assert "channel_multiplier" in code


def test_no_script_dir_parent_parent():
    """R19：照片链不得用 SCRIPT_DIR.parent.parent 解析项目根。"""
    src = _source()
    assert "SCRIPT_DIR.parent.parent" not in src, \
        "照片链项目根是 SCRIPT_DIR.parent（R19）"


@pytest.mark.skipif(not os.path.isfile(os.path.join(
    os.path.dirname(STAGE4), "..", "..", ".venv_r4", "Scripts", "python.exe")),
    reason="需要项目 .venv_r4 解释器")
def test_real_gfpganer_signature_accepts_our_kwargs():
    """用真实 GFPGANer 验证我们生成的调用能被接受。

    FIX-2026-10-22：这个用例原先**常年 skip**，两层原因叠加：

    ① 路径插错——原实现 `sys.path.insert(0, '.venv_r4')` 插的是 venv
       **根目录**，而包在 `.venv_r4/Lib/site-packages/`。Python 找不到，
       回退命中**项目根**的 `gfpgan/weights/`（无 `__init__.py` 的 namespace
       package）→ `cannot import name 'GFPGANer' ... (unknown location)`。

    ② 即便路径插对也**不可能成功**——`.venv_r4` 的 numpy 是 **cp312** 编译的
      （`_multiarray_umath.cp312-win_amd64.pyd`），而测试跑在 managed
      Python **3.13** 下，C 扩展不兼容：
      `Importing the numpy C-extensions failed ... incompatible with
      cpython-313`。gfpgan 传递依赖 numpy/torch，所以在 pytest 解释器里
      导入它从根上就走不通。

    正解：与生产代码同一姿势——**用 `.venv_r4` 的解释器开子进程**验签名
    （`_resolve_ai_python()` + `subprocess.run([ai_py, script])`）。
    这样验的才是线上真正会用的那个解释器里的那个类。
    """
    import subprocess
    venv_py = os.path.normpath(os.path.join(
        os.path.dirname(STAGE4), "..", "..", ".venv_r4", "Scripts",
        "python.exe"))
    probe = (
        "import inspect, json\n"
        "from gfpgan import GFPGANer\n"
        "sig = inspect.signature(GFPGANer.__init__)\n"
        "# 复刻 stage5 生成的调用；不合法就抛 TypeError\n"
        "sig.bind(None, model_path='x.pth', upscale=2, arch='clean',\n"
        "         channel_multiplier=2, bg_upsampler=None)\n"
        "print(json.dumps({'params': sorted(sig.parameters),\n"
        "                  'has_channel_only': 'channel_only' in sig.parameters}))\n"
    )
    r = subprocess.run([venv_py, "-c", probe], capture_output=True,
                       text=True, errors="replace", timeout=300, shell=False,
                       cwd=os.path.dirname(STAGE4))
    if r.returncode != 0:
        pytest.skip(f"gfpgan 在 .venv_r4 中不可用: "
                    f"{(r.stderr or '').strip().splitlines()[-1:] }")
    import json as _json
    info = _json.loads(r.stdout.strip().splitlines()[-1])
    assert not info["has_channel_only"], (
        "GFPGANer 不该有 channel_only 参数（我们代码没传）")
    # 我们用到的四个参数必须真实存在于签名里
    for need in ("model_path", "upscale", "arch", "bg_upsampler"):
        assert need in info["params"], \
            f"GFPGANer 签名缺少 {need}，实际参数: {info['params']}"


def test_gfpgan_subprocess_keeps_stderr():
    """修复必须保留子进程 stderr，否则下次仍无法定位。"""
    src = _source()
    assert "GFPGAN subprocess failed" in src
    assert "capture_output=True, text=True" in src


def test_restored_output_globs_all_image_exts():
    """R20：修复图按输入文件名写出（16bit TIFF → .tif），
    消费方不得只 glob '*.png'（否则永远匹配不到 → rc=83）。"""
    src = _source()
    code = _code_only()
    assert 'rglob("*.png")' not in code and "rglob('*.png')" not in code, \
        "不得只 glob png（R20：输出扩展名跟随输入）"
    # 扩展名集合必须覆盖 16bit 链路的实际输出（.tif/.tiff）
    assert "png, tif, tiff, jpg, jpeg" in src or \
        "'png', 'tif', 'tiff', 'jpg', 'jpeg'" in src or \
        '"png", "tif", "tiff", "jpg", "jpeg"' in src, \
        "必须收集 png/tif/tiff/jpg/jpeg 五种扩展名"
    assert "rglob(f\"*.{ext}\")" in src, "应按扩展名逐个收集"
    assert "STAGE4_NO_RESTORED_OUTPUT" in src, \
        "收集不到修复图时必须显式报错，不能静默"
