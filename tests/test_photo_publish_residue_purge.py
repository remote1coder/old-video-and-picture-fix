# -*- coding: utf-8 -*-
"""照片链发布成功后的 `.pubbak` 备份清理（2026-10-27 实测缺陷）。

**现象**

正常跑完一张照片后，输出目录里残留一个 `.pubbak` 备份目录，
含上一代的完整文件（实测单张 **238 MB**：avif + tif + encode_contract + status）。

**根因**

`_purge_publish_residue()` 只在两条**异常恢复**路径被调用：
- 上次已完整提交（`ALREADY_COMMITTED`）
- 回滚成功

成功提交（`PHASE_COMMITTED` 写入之后）直接 `return 0`，**从不清理**。
于是稳态下恒留一份完整上一代备份。

**修法**

成功路径也清理，但要 `keep_journal=True`：
- 备份删掉（省 238 MB、目录干净）
- journal 保留（它同时是输出目录里的审计文件，且下次启动要靠它
  识别 `ALREADY_COMMITTED`；删掉会导致重复发布）
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "photo_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage6_encode as s6  # noqa: E402


def _mk_backup(tmp_path, name="0_a.avif"):
    d = tmp_path / "out.avif.photo_1.pubbak"
    d.mkdir()
    f = d / name
    f.write_bytes(b"old generation bytes")
    return d, f


def test_purge_removes_backups_and_journal_by_default(tmp_path):
    d, f = _mk_backup(tmp_path)
    journal = tmp_path / "out.avif.journal.json"
    journal.write_text("{}", encoding="utf-8")
    rep = {"targets": {str(tmp_path / "out.avif"): {"backup": str(f)}}}

    s6._purge_publish_residue(str(journal), rep, None)

    assert not f.exists(), "备份文件应被删除"
    assert not d.exists(), "空掉的 .pubbak 目录应被移除"
    assert not journal.exists(), "默认也应删除 journal"


def test_purge_keeps_journal_when_asked(tmp_path):
    """成功提交后的清理：备份删掉，journal 必须留着。"""
    d, f = _mk_backup(tmp_path)
    journal = tmp_path / "out.avif.journal.json"
    journal.write_text("{}", encoding="utf-8")
    rep = {"targets": {str(tmp_path / "out.avif"): {"backup": str(f)}}}

    s6._purge_publish_residue(str(journal), rep, None, keep_journal=True)

    assert not f.exists(), "备份文件仍应删除"
    assert not d.exists(), "空 .pubbak 目录仍应移除"
    assert journal.exists(), "keep_journal=True 时 journal 必须保留"


def test_purge_leaves_nonempty_dir_alone(tmp_path):
    """目录里还有 journal 没记录到的文件时，不能用 rmtree 盲删。"""
    d, f = _mk_backup(tmp_path)
    stray = d / "unknown_leftover.bin"
    stray.write_bytes(b"not in journal")
    journal = tmp_path / "out.avif.journal.json"
    journal.write_text("{}", encoding="utf-8")
    rep = {"targets": {str(tmp_path / "out.avif"): {"backup": str(f)}}}

    s6._purge_publish_residue(str(journal), rep, None, keep_journal=True)

    assert not f.exists()
    assert stray.exists(), "未记录的文件不能删"
    assert d.exists(), "非空目录必须保留（rmdir 只在空目录成功）"


def test_purge_tolerates_missing_backup(tmp_path):
    """备份已不存在时应静默跳过，不抛。"""
    journal = tmp_path / "j.json"
    journal.write_text("{}", encoding="utf-8")
    rep = {"targets": {str(tmp_path / "x.avif"):
                       {"backup": str(tmp_path / "nope.pubbak" / "0_x")}}}
    s6._purge_publish_residue(str(journal), rep, None, keep_journal=True)
    assert journal.exists()


def test_success_path_calls_purge():
    """源码护栏：成功提交路径（STAGE6_DONE 之前）必须调用清理。"""
    src = os.path.join(_ROOT, "photo_restoration_pipeline", "stages",
                       "stage6_encode.py")
    text = open(src, encoding="utf-8", errors="replace").read()
    i_purge = text.rfind("_purge_publish_residue(")
    i_done = text.find('"STAGE6_DONE"')
    assert i_purge != -1 and i_done != -1
    assert i_purge < i_done, \
        "成功路径的清理必须在 STAGE6_DONE 之前"
    assert "keep_journal=True" in text, "成功路径清理必须保留 journal"
