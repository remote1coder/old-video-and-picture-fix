# -*- coding: utf-8 -*-
"""R20-B01：发布允许根不能靠 backup_dir 的父目录猜（审核 R19-B01）。

审核的真实小文件反例：

    temp/publish_backup  与  out/media.bin  分开
      第一次发布 → COMMITTED
      第二次发布 → 失败于 recover_prior，「正式目标不在允许根内」

根因：`_recover_locked` 在 `allowed_roots` 为空时用
`dirname(backup_dir)`（= `temp/`），而成果在 `out/`。

本文件锁住两件事：

1. 显式传 `allowed_output_roots=[out]` 时，**两次发布都成功**；
2. 不传时**必须有 `PUBLISH_ALLOWED_ROOTS_INFERRED` 警告**，
   不能静默地做这个猜测（否则下一个人还会踩）。
"""
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "shared_modules") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "shared_modules"))

from publisher import Artifact, publish  # noqa: E402


class _Cap:
    """最小 logger：记录 (level, event, kwargs)。"""

    def __init__(self):
        self.rows = []

    def _add(self, level, event, msg="", **kw):
        self.rows.append({"level": level, "event": event, "msg": msg, **kw})

    def info(self, e, m="", **k): self._add("info", e, m, **k)
    def warning(self, e, m="", **k): self._add("warning", e, m, **k)
    def error(self, e, m="", **k): self._add("error", e, m, **k)
    def critical(self, e, m="", **k): self._add("critical", e, m, **k)
    def debug(self, e, m="", **k): self._add("debug", e, m, **k)

    def events(self, level=None):
        return [r["event"] for r in self.rows
                if level is None or r["level"] == level]


def _layout(tmp_path):
    """temp/ 与 out/ **不同父**的布局（审核反例的形态）。"""
    out = tmp_path / "out"
    temp = tmp_path / "temp"
    backup = temp / "publish_backup"
    out.mkdir()
    backup.mkdir(parents=True)
    return out, temp, backup


def _artifact(tmp_path, gen, final):
    staging = tmp_path / f"staging_{gen}.bin"
    staging.write_bytes(b"GEN" + bytes([gen]) * 200)
    return Artifact("media", str(staging), str(final))


def test_second_publish_succeeds_with_explicit_root(tmp_path):
    """核心回归：显式传 allowed_output_roots 时二次发布必须成功。"""
    out, _temp, backup = _layout(tmp_path)
    final = out / "media.bin"
    jp = str(out / "media.bin.journal.json")
    log = _Cap()

    for gen in (1, 2):
        res = publish([_artifact(tmp_path, gen, final)], jp, str(backup),
                      f"att{gen}", logger=log,
                      allowed_output_roots=[str(out)])
        assert res.ok is True, f"第 {gen} 次发布失败：{res.reason}"
        assert final.exists()
        assert final.read_bytes().startswith(b"GEN")
        assert final.read_bytes()[3] == gen, f"第 {gen} 次内容不对"

    # 两次都成功，且不该出现"推断根"的警告
    assert "PUBLISH_ALLOWED_ROOTS_INFERRED" not in log.events("warning")


def test_missing_root_is_loud_not_silent(tmp_path):
    """不传 allowed_output_roots 时必须告警，不能静默猜测。"""
    out, _temp, backup = _layout(tmp_path)
    final = out / "media.bin"
    jp = str(out / "media.bin.journal.json")
    log = _Cap()

    # 第一次没有 prior journal，不会走到推断分支也没关系；
    # 关键是**只要有推断就留痕**。
    publish([_artifact(tmp_path, 1, final)], jp, str(backup), "att1", logger=log)
    publish([_artifact(tmp_path, 2, final)], jp, str(backup), "att2", logger=log)

    assert "PUBLISH_ALLOWED_ROOTS_INFERRED" in log.events("warning"), (
        "推断允许根却没有告警；日志事件：%r" % (log.events(),))


def test_publish_and_recover_accept_the_parameter():
    """接口层：`publish` / `recover` 都必须接受 `allowed_output_roots`。"""
    import inspect
    from publisher import recover
    p_sig = inspect.signature(publish)
    r_sig = inspect.signature(recover)
    assert "allowed_output_roots" in p_sig.parameters
    assert "allowed_output_roots" in r_sig.parameters


def test_stage7_passes_allowed_output_roots():
    """生产调用点必须真的传（否则日志里会一直有 INFERRED 警告）。"""
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage7_encode.py"),
               encoding="utf-8", errors="replace").read()
    assert src.count("allowed_output_roots=") >= 2, (
        "stage7 的两处 _publish 都应传 allowed_output_roots")
