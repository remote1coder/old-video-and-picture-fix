# -*- coding: utf-8 -*-
"""第十八轮 P0-B / T07：照片链取消时的回滚结果**必须可持久追溯**。

审核 R17 指出：

> journal 写入返回 False 的路径仍可能只记回滚结果、不记持久记录失败。
> 探针确认该差别。

原实现直接调 `write_journal(...)` 并**忽略返回值**：

```python
write_journal(_jp, _doc)      # 只在抛异常时才会走到 except
```

而 `write_journal` 写失败时是**返回 False**（不抛）。于是"取消时的回滚结果是
否真的落盘"这件事被静默吞掉 —— 取消处理器随后 `os._exit`，现场就没了。

修法：抽出 `_persist_cancel_rollback()`，返回 False 视为持久化失败抛出，
由调用方记 `STAGE6_CANCEL_ROLLBACK_UNRECORDED`。

> 注：本项目有两个同名 `stage6_encode`（视频/照片）。这里按**文件路径**加载
> 照片链那一个，避免 `import stage6_encode` 拿到视频链的模块。
"""
import importlib.util
import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PHOTO_STAGE6 = os.path.join(_ROOT, "photo_restoration_pipeline", "stages",
                             "stage6_encode.py")


def _load_photo_stage6():
    spec = importlib.util.spec_from_file_location("photo_stage6_encode",
                                                  _PHOTO_STAGE6)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def s6():
    if not os.path.isfile(_PHOTO_STAGE6):
        pytest.skip("照片链 stage6 不存在")
    return _load_photo_stage6()


def test_T07_write_journal_false_is_treated_as_failure(s6, monkeypatch):
    """**核心缺陷**：write_journal 返回 False 不能被当成写成功。"""
    monkeypatch.setattr(s6, "journal_path_for", lambda p: "j.json")
    monkeypatch.setattr(s6, "read_journal",
                        lambda p: {"phase": "PREPARED", "records": {}})
    monkeypatch.setattr(s6, "write_journal", lambda p, d: False)

    with pytest.raises(OSError) as ei:
        s6._persist_cancel_rollback("out.avif", {"fully_restored": False,
                                                 "failed": ["x"]})
    assert "False" in str(ei.value) or "persist" in str(ei.value).lower()


def test_T07_missing_journal_is_also_reported(s6, monkeypatch):
    """没有可读 journal 同样是"无处记录"，不能当无事发生。"""
    monkeypatch.setattr(s6, "journal_path_for", lambda p: "j.json")
    monkeypatch.setattr(s6, "read_journal", lambda p: None)

    with pytest.raises(OSError):
        s6._persist_cancel_rollback("out.avif", {"fully_restored": True})


def test_T07_successful_persist_records_the_result(s6, monkeypatch):
    """正常路径：结果写进 journal，且返回成功（防止修过头把好的也判错）。"""
    captured = {}

    def fake_write(path, doc):
        captured.update(doc)
        return True

    monkeypatch.setattr(s6, "journal_path_for", lambda p: "j.json")
    monkeypatch.setattr(s6, "read_journal",
                        lambda p: {"phase": "PREPARED", "records": {}})
    monkeypatch.setattr(s6, "write_journal", fake_write)

    assert s6._persist_cancel_rollback(
        "out.avif", {"fully_restored": True}) is True
    assert captured["cancel_rollback"]["fully_restored"] is True
    assert "at" in captured["cancel_rollback"]
    assert captured["phase"] == "PREPARED", "其余字段必须保留"


def test_T07_no_longer_calls_write_journal_ignoring_result(s6):
    """结构守卫：取消钩子里不得再有"忽略返回值"的裸 write_journal 调用。"""
    src = open(_PHOTO_STAGE6, encoding="utf-8", errors="replace").read()
    assert "_persist_cancel_rollback(" in src
    i = src.find("def _cancel_rollback")
    assert i > 0
    hook = src[i:i + 3000]
    assert "write_journal(_jp, _doc)" not in hook, \
        "取消钩子里仍有忽略返回值的 write_journal 调用"
