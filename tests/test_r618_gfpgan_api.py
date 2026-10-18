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


@pytest.mark.skipif(not os.path.isdir(os.path.join(
    os.path.dirname(STAGE4), "..", "..", ".venv_r4")),
    reason="需要项目 .venv_r4 才能导入 gfpgan")
def test_real_gfpganer_signature_accepts_our_kwargs():
    """用真实 GFPGANer 验证我们生成的调用能被接受。"""
    import sys
    venv = os.path.normpath(os.path.join(
        os.path.dirname(STAGE4), "..", "..", ".venv_r4"))
    sys.path.insert(0, venv)
    try:
        from gfpgan import GFPGANer
    except Exception as e:  # pragma: no cover
        pytest.skip(f"gfpgan 不可导入: {e}")
    sig = inspect.signature(GFPGANer.__init__)
    # 我们代码里用到的参数必须全部合法
    sig.bind(None, model_path="x.pth", upscale=2, arch="clean",
             channel_multiplier=2, bg_upsampler=None)
    assert "channel_only" not in sig.parameters


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
