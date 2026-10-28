# -*- coding: utf-8 -*-
"""R13-05：首次发布的半成品被判成"完整提交"。

**缺陷**

`classify_recovery()` 里 `ABSENT_OK` 分支（`cur is None` 且
`records[t]["existed"] is False`）**两个标志都不动**：

```python
elif cur is None and rec.get("existed") is False:
    state = "ABSENT_OK"      # 本来就该不存在
```

首次发布前所有目标都不存在。若只替换了 AVIF 就被硬终止：

| 目标 | 磁盘状态 | 判定 | 对 all_new 的影响 |
|---|---|---|---|
| AVIF | 已是新内容 | NEW | 不变（仍 True） |
| TIFF / contract / status | 不存在（本该被新建） | ABSENT_OK | **不变（仍 True）** |

于是 `all_new=True` → 推荐 `ALREADY_COMMITTED` → **journal 被清掉**，
一个只有 1/4 文件的半成品被当成完整一代，后续再也不会被恢复。

**根因**：`all_old` 与 `all_new` 被当成互斥的两个方向，而实际上
"旧代本来不存在"只满足**旧代**侧；**新代要求把这个文件创建出来**。

修法：`ABSENT_OK` 必须置 `all_new = False`。
"""
import hashlib
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import publish_journal as pj  # noqa: E402


def _digest_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_publish_doc(paths, new_contents):
    """构造"首次发布"的 journal：所有目标本来都不存在。"""
    records = {}
    for p, content in zip(paths, new_contents):
        records[p] = {
            "existed": False,
            "old_digest": None,
            "new_digest": hashlib.sha256(content).hexdigest(),
            "backup": None,
            "backup_digest": None,
            "staging": None,
            "state": None,
        }
    return {"phase": "PREPARED", "attempt_id": "t", "targets": paths,
            "records": records}


def test_half_first_publish_is_not_already_committed(tmp_path):
    """只替换了 1/4 个目标 → **不能**是 ALREADY_COMMITTED。"""
    names = ["a.avif", "a.tif", "a.contract.json", "a.status.json"]
    paths = [str(tmp_path / n) for n in names]
    contents = [b"NEW-AVIF", b"NEW-TIF", b"NEW-CON", b"NEW-STA"]
    doc = _first_publish_doc(paths, contents)

    # 只把第一个目标写出来（模拟"替换 AVIF 后硬终止"）
    with open(paths[0], "wb") as f:
        f.write(contents[0])

    rep = pj.classify_recovery(doc)
    assert rep["targets"][paths[0]]["state"] == "NEW"
    for p in paths[1:]:
        assert rep["targets"][p]["state"] == "ABSENT_OK"

    assert rep["all_new"] is False, \
        "半成品不能算完整新代（这是 R13-05 的核心）"
    assert rep["recommendation"].startswith("MIXED"), \
        f"应为混合代，实际 {rep['recommendation']}"


def test_complete_first_publish_is_already_committed(tmp_path):
    """4/4 都写出来 → 才是 ALREADY_COMMITTED（对照，确保没有过度收紧）。"""
    names = ["a.avif", "a.tif", "a.contract.json", "a.status.json"]
    paths = [str(tmp_path / n) for n in names]
    contents = [b"NEW-AVIF", b"NEW-TIF", b"NEW-CON", b"NEW-STA"]
    doc = _first_publish_doc(paths, contents)

    for p, c in zip(paths, contents):
        with open(p, "wb") as f:
            f.write(c)

    rep = pj.classify_recovery(doc)
    assert all(rep["targets"][p]["state"] == "NEW" for p in paths)
    assert rep["all_new"] is True
    assert rep["recommendation"].startswith("ALREADY_COMMITTED")


def test_none_written_first_publish_is_not_new(tmp_path):
    """一个都没写出来（刚写完 PREPARED 就崩）→ 不能算新代。"""
    names = ["a.avif", "a.tif"]
    paths = [str(tmp_path / n) for n in names]
    doc = _first_publish_doc(paths, [b"NEW-A", b"NEW-B"])

    rep = pj.classify_recovery(doc)
    assert rep["all_new"] is False
    assert rep["all_old"] is True, "全部不存在 = 完整旧代（旧代本来就该没有）"
    assert rep["recommendation"].startswith("ROLLBACK_SAFE")


def test_overwrite_publish_still_detects_mixed(tmp_path):
    """覆盖式发布（旧代存在）半途终止 → 仍是 MIXED。"""
    a = str(tmp_path / "a.avif")
    b = str(tmp_path / "b.avif")
    with open(a, "wb") as f:
        f.write(b"NEW-A")
    with open(b, "wb") as f:
        f.write(b"OLD-B")
    doc = {
        "phase": "PREPARED", "attempt_id": "t", "targets": [a, b],
        "records": {
            a: {"existed": True, "old_digest": hashlib.sha256(b"OLD-A").hexdigest(),
                "new_digest": hashlib.sha256(b"NEW-A").hexdigest(),
                "backup": None, "backup_digest": None, "staging": None},
            b: {"existed": True, "old_digest": hashlib.sha256(b"OLD-B").hexdigest(),
                "new_digest": hashlib.sha256(b"NEW-B").hexdigest(),
                "backup": None, "backup_digest": None, "staging": None},
        },
    }
    rep = pj.classify_recovery(doc)
    assert rep["targets"][a]["state"] == "NEW"
    assert rep["targets"][b]["state"] == "OLD"
    assert rep["all_new"] is False and rep["all_old"] is False
    assert rep["recommendation"].startswith("MIXED")
