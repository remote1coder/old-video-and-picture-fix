# -*- coding: utf-8 -*-
"""PHOTO-02：恢复前必须核对备份摘要，不能只看"文件存在"。

原实现 `stage6_encode.py` 里：
    bk = r.get("backup")
    if bk and os.path.exists(bk):
        snapshot[t] = {"state": BACKUP_OK, "backup": bk}

只判断存在性，从不核对 journal 里记录的 `backup_digest`。于是异常后
被截断的备份会被判成 BACKUP_OK，然后**用它覆盖仍然完好的旧成品** ——
比不恢复更糟。

修法：存在之外还要 `file_digest(bk) == backup_digest`，不一致就判
BACKUP_FAILED 并记 STAGE6_BACKUP_DIGEST_MISMATCH。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from publish_journal import file_digest  # noqa: E402


def test_digest_detects_truncation(tmp_path):
    """截断后的文件摘要必须与原文件不同 —— 这是修复依赖的机制。"""
    p = tmp_path / "good.avif"
    p.write_bytes(b"RIFF" + b"payload" * 500)
    good = file_digest(p)
    assert good, "file_digest 应算出摘要"

    q = tmp_path / "truncated.avif"
    q.write_bytes(p.read_bytes()[: len(p.read_bytes()) // 2])
    assert file_digest(q) != good, "截断必须能被摘要发现"


def test_digest_detects_corruption(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"0123456789" * 100)
    good = file_digest(p)
    p.write_bytes(b"0123456789" * 99 + b"X" * 10)
    assert file_digest(p) != good


def test_missing_file_digest_is_none():
    assert file_digest(os.path.join("no", "such", "file")) is None


def test_stage6_verifies_digest_before_restore():
    """回归护栏：恢复前核对摘要这段逻辑不能被悄悄删掉。

    这段是内联在 stage6 的大函数里，不便直接调用；这里锁住源码里
    确实存在摘要核对（防止以后有人又退回只查 exists）。
    """
    src = os.path.join(_ROOT, "photo_restoration_pipeline", "stages",
                       "stage6_encode.py")
    text = open(src, encoding="utf-8", errors="replace").read()
    assert "backup_digest" in text, "应引用 journal 里的 backup_digest"
    assert "BACKUP_FAILED" in text
    # 存在性检查之外必须有摘要比对
    assert "file_digest" in text and "_actual != bk_digest" in text, \
        "必须比对磁盘摘要与记录摘要，不能只查 exists"
