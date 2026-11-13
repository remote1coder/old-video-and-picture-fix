# -*- coding: utf-8 -*-
r"""R24-10 / 审核 §4.4：硬链接的**别名生命周期**。

审核对 stage2 硬链接的态度是"有效，但不能称零风险"：

> stage2 硬链接有效，但同一数据的多个名字共享写入；不能称"零风险、比复制更安全"。
> 应保证已发布中间件不可原地重写，候选完成后替换路径；校验别名生命周期和清理。

这条不是"撤销硬链接"，而是把它隐含的前提**测出来**：

1. **原地写会穿透别名** —— `open(dst, "w")` 改的是同一 inode，src 也变。
   所以任何"改写中间件"的操作都必须走"写新文件 + `os.replace`"。
2. `os.replace(dst, ...)` 只换名字，src 不受影响 —— 这是正确写法。
3. 删掉一个名字不会毁掉另一个（硬链接的正常语义）。
4. 清理一个名字不影响另一个；两个都删才释放。

前两条是"必须遵守的规则"，后两条是"允许的操作"。把它们写成测试，
比在文档里写一句"注意别名"有用。
"""
import os

import pytest

import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage2_traditional_check as s2  # noqa: E402

OLD = b"OLD-CONTENT" * 64
NEW = b"NEW-CONTENT" * 64


def _pair(tmp_path, payload=OLD):
    src = tmp_path / "step1.mkv"
    src.write_bytes(payload)
    dst = tmp_path / "step2.mkv"
    how = s2._materialise_step2(str(src), str(dst))
    return src, dst, how


def test_in_place_write_punches_through_the_alias(tmp_path):
    """**规则**：原地写 dst 会同时改掉 src —— 所以中间件绝不能原地重写。

    这不是缺陷，是硬链接的定义。写出来是为了让"不能原地覆盖"这条规则
    有可执行的依据，而不是只写在注释里。
    """
    src, dst, how = _pair(tmp_path)
    if how != "linked":
        pytest.skip("本机/本卷不支持硬链接，这条规则无从验证")

    with open(str(dst), "r+b") as f:      # 原地改，不换 inode
        f.write(b"XXXX")

    assert src.read_bytes()[:4] == b"XXXX", (
        "同 inode 的两个名字应当共享写入；这里没穿透说明根本不是硬链接")


def test_replacing_the_name_does_not_touch_the_source(tmp_path):
    """**正确写法**：候选写完再 `os.replace` 换名字，源文件不动。"""
    src, dst, how = _pair(tmp_path)
    before = src.read_bytes()

    candidate = tmp_path / "step2.candidate"
    candidate.write_bytes(NEW)
    os.replace(str(candidate), str(dst))

    assert dst.read_bytes() == NEW
    assert src.read_bytes() == before, (
        "换名字不该改到源；改了说明走了原地写")


def test_removing_one_name_keeps_the_other_readable(tmp_path):
    """删掉一个名字不影响另一个（允许的操作）。"""
    src, dst, how = _pair(tmp_path)
    os.unlink(str(src))
    assert dst.read_bytes() == OLD


def test_cleaning_one_name_leaves_no_stale_alias(tmp_path):
    """误清理一个名字后，另一个仍是**完整内容**，不是半截。"""
    src, dst, how = _pair(tmp_path)
    os.unlink(str(dst))
    assert src.read_bytes() == OLD
    assert os.path.exists(str(src)) and not os.path.exists(str(dst))


def test_both_names_gone_releases_the_object(tmp_path):
    """两个名字都删掉才算真正释放（这里只验证两个都不存在）。"""
    src, dst, how = _pair(tmp_path)
    os.unlink(str(src))
    os.unlink(str(dst))
    assert not os.path.exists(str(src)) and not os.path.exists(str(dst))


def test_fallback_copy_is_not_an_alias(tmp_path, monkeypatch):
    """回退路径产出的是**独立副本**：改它不会影响源。

    这条把"链接 / 复制"两种形态的差别钉死 —— 消费者不能假设
    "step2 一定是别名"，也不能假设"一定不是"。
    """

    def no_link(s, d):
        raise OSError(0, "not supported", None, 1)

    monkeypatch.setattr(s2.os, "link", no_link)
    src, dst, how = _pair(tmp_path, NEW)
    assert how == "copied"

    with open(str(dst), "r+b") as f:
        f.write(b"ZZZZ")

    assert src.read_bytes()[:4] == b"NEW-", (
        "回退复制后两边应当是独立对象；这里穿透了，说明其实是硬链接")
    assert dst.read_bytes()[:4] == b"ZZZZ"
