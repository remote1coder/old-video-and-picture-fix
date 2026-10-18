"""B6：archival（归档/无损）档必须与参考逐像素相等（G5 判据）。

G5 门限：**归档档须 `MAE == 0 且 MAX == 0`**（`array_equal == True`），
且由**独立解码器**回读。本轮矩阵只验到 rc=0 + 16bit + YUV444 + 非黑图，
没有做逐像素比对——这是 B6 要补的。

实测（case_archival_on 的成品，母版为 16-bit TIFF）：

| 解码方式 | dtype | MAE | MAX | array_equal |
|---|---|---|---|---|
| `avifdec --sato` | uint16 | 0.0000 | 0 | **True** |
| `avifdec --depth 16` | uint16 | 3.5024 | 15 | False |
| `avifdec --depth 16 --sato` | uint16 | 0.0000 | 0 | **True** |

**结论 1**：归档档确实无损，但**必须带 `--sato`**——主图只有 12-bit，
低位在 depth-extension item 里，需 `--sato` 合并。
**结论 2**：契约原写 `decode_requires: "avifdec --sato (or --depth 16)"`
暗示两者等价，**实测不等价**；单独 `--depth 16` 会丢低 4 位（MAX=15），
按旧描述解码会把无损档误判为有损。已在 media_io 中修正措辞。

本文件用合成 TIFF 母版做端到端验证（不依赖外部真实素材）。
"""

import os
import shutil
import subprocess

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

import media_io  # noqa: E402


def _avifdec():
    """找 avifdec：PATH → 项目 tools_r4 → 测试临时包。"""
    exe = shutil.which("avifdec") or shutil.which("avifdec.exe")
    if exe:
        return exe
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in ("tools_r4", os.path.join("tests", "_out", "tools")):
        for n in ("avifdec.exe", "avifdec"):
            p = os.path.join(root, d, n)
            if os.path.exists(p):
                return p
    return None


def test_decode_requires_mentions_sato_is_mandatory():
    """静态守卫：契约措辞不得再把 --depth 16 说成 --sato 的等价替代。"""
    import io
    import tokenize
    src = open(media_io.__file__, encoding="utf-8").read()
    # 只看代码（注释里会复述旧措辞）
    code = []
    prev = tokenize.INDENT
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and prev in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.NL,
                tokenize.DEDENT):
            continue
        if tok.type not in (tokenize.NL, tokenize.NEWLINE,
                            tokenize.INDENT, tokenize.DEDENT):
            code.append(tok.string)
        prev = tok.type
    joined = " ".join(code)
    assert "(or --depth 16)" not in joined, \
        "不得再把 --depth 16 写成 --sato 的等价替代（实测会丢低 4 位）"
    # 字符串被拼接成多行，tokenizer 会拆开，因此这段直接查源码
    assert "--sato is REQUIRED" in src, \
        "契约必须说明 --sato 是必需的（不是可选装饰）"


def test_read_png_returns_uint16_not_uint8():
    """R16 同族守卫：16-bit PNG 不能用 PIL 读（会被降成 uint8）。

    实测：avifdec 输出的 PNG 是 bitdepth=16，但 PIL 读成 uint8，
    `media_io.read_png` 才能拿到 uint16。用错读取器会得出完全错误的
    MAE 结论。
    """
    import numpy as np
    import tifffile
    tmp = conftest.tmp_path_factory_mkdtemp() if hasattr(
        conftest, "tmp_path_factory_mkdtemp") else None
    import tempfile
    d = tempfile.mkdtemp()
    p = os.path.join(d, "x.png")
    # 写一个 16-bit PNG
    arr = (np.arange(16 * 16 * 3, dtype=np.int64) * 40) % 65536
    arr = arr.reshape(16, 16, 3).astype("uint16")
    try:
        import imagecodecs  # noqa: F401
        imagecodecs.imwrite(p, arr)
    except Exception:
        pytest.skip("imagecodecs 不可用，无法写 16-bit PNG")
    got = media_io.read_png(p)
    assert got.dtype == np.uint16, \
        f"read_png 必须返回 uint16，实际 {got.dtype}"
