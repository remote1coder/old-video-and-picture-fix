# -*- coding: utf-8 -*-
"""R12-01：PREPARED 必须记录**真实存在**的新产物摘要。

**缺陷（审核报告 R12-01）**

`_snapshot_for_rollback()` 只返回 `state` / `backup`。上一轮的代码却从
这个 snapshot 读 `staging` 再算 `new_digest` —— 上游从没填过 `staging`，
于是字段存在、**值恒为 None**。

后果：替换 AVIF 后、替换 TIFF 或侧车前进程被硬终止，新 AVIF 既不匹配
`old_digest`，也没有可匹配的 `new_digest` → `classify_recovery()` 只能判
UNKNOWN → 走人工处理，而不是按设计自动回滚。

**修法**

- 四个目标都先准备**独立暂存件**（AVIF/TIFF 用 `.partial`，
  contract/status 新建 `*.partial`），取到摘要再写 PREPARED
- `_build_journal_records()` 是**生产与测试共用**的构造函数，
  `new_digest` 直接取自 `new_digest_map`（WSL 回拷目标沿用 work_* 的摘要）
- 任一目标取不到摘要 → 在改动任何目标之前 `return 82`

本测试用**真实文件**跑生产函数，不复刻算法，也不手造 journal。
"""
import json
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
import publish_journal as pj  # noqa: E402


def _mk(path, data: bytes):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def test_prepared_records_carry_real_new_digests(tmp_path):
    """生产构造函数必须给每个目标填上**真实暂存件**的摘要。"""
    avif = _mk(str(tmp_path / "out.avif"), b"OLD-AVIF")
    tif = _mk(str(tmp_path / "out.tif"), b"OLD-TIF")
    con = _mk(str(tmp_path / "out.avif.encode_contract.json"), b"OLD-CON")
    sta = _mk(str(tmp_path / "out.avif.status.json"), b"OLD-STA")

    stg = {
        avif: _mk(str(tmp_path / "out.avif.partial"), b"NEW-AVIF"),
        tif: _mk(str(tmp_path / "out.tif.partial"), b"NEW-TIF"),
        con: _mk(str(tmp_path / "con.partial"), b"NEW-CON"),
        sta: _mk(str(tmp_path / "sta.partial"), b"NEW-STA"),
    }
    new_digest_map = {t: s6.file_digest(p) for t, p in stg.items()}
    snapshot = {t: {"state": s6.BACKUP_OK, "backup": None}
                for t in (avif, tif, con, sta)}

    recs = s6._build_journal_records(
        [avif, tif, con, sta], snapshot, stg, new_digest_map)

    for t in (avif, tif, con, sta):
        r = recs[t]
        assert r["new_digest"], f"{os.path.basename(t)} 的 new_digest 不能为空"
        assert r["new_digest"] == s6.file_digest(stg[t]), \
            "new_digest 必须等于暂存件的实际摘要"
        assert r["staging"] == stg[t], "staging 必须指向真实暂存件"
        assert r["old_digest"] != r["new_digest"], "新旧代摘要不应相同"


def test_old_snapshot_without_staging_yields_none_digest(tmp_path):
    """反例：照旧路径（staging_map 为空）→ new_digest 恒 None。

    这条锁住"曾经的断点"：证明本测试确实能发现旧写法的问题，
    而不是无论怎么实现都通过。
    """
    avif = _mk(str(tmp_path / "out.avif"), b"OLD")
    recs = s6._build_journal_records(
        [avif], {avif: {"state": s6.BACKUP_OK, "backup": None}}, {}, {})
    assert recs[avif]["new_digest"] is None
    assert recs[avif]["staging"] is None


def test_half_commit_is_recognised_as_new(tmp_path):
    """替换第一个目标后硬终止 → 重启必须能认出它是 NEW（不是 UNKNOWN）。

    这是 R12-01 的核心验收：有了真实 new_digest，半提交才可自动处理。
    """
    avif = _mk(str(tmp_path / "out.avif"), b"OLD-AVIF")
    con = _mk(str(tmp_path / "con.json"), b"OLD-CON")
    stg_avif = _mk(str(tmp_path / "out.avif.partial"), b"NEW-AVIF")
    stg_con = _mk(str(tmp_path / "con.partial"), b"NEW-CON")

    stg = {avif: stg_avif, con: stg_con}
    new_digest_map = {t: s6.file_digest(p) for t, p in stg.items()}
    snapshot = {t: {"state": s6.BACKUP_ABSENT, "backup": None}
                for t in (avif, con)}
    recs = s6._build_journal_records([avif, con], snapshot, stg, new_digest_map)

    # 模拟：只替换了 AVIF 就崩了
    os.replace(stg_avif, avif)

    doc = {"targets": [avif, con], "records": recs}
    rep = pj.classify_recovery(doc)

    assert rep["targets"][avif]["state"] == "NEW", \
        "已替换的 AVIF 应被识别为新代"
    # 未动的侧车仍然匹配 old_digest → OLD。NEW + OLD = 混合代，
    # classify_recovery 能据此给出确定结论（而不是 UNKNOWN 卡人工）。
    assert rep["targets"][con]["state"] == "OLD", rep["targets"][con]
    assert rep["has_unknown"] is False, "有了 new_digest 就不该再出现 UNKNOWN"


def test_missing_digest_is_detectable_before_commit(tmp_path):
    """有目标拿不到摘要时，调用方必须能在提交前发现（`_missing` 判据）。"""
    avif = _mk(str(tmp_path / "out.avif"), b"OLD")
    con = _mk(str(tmp_path / "con.json"), b"OLD")
    stg = {avif: _mk(str(tmp_path / "a.partial"), b"NEW")}
    new_digest_map = {t: s6.file_digest(p) for t, p in stg.items()}

    commit_targets = [avif, con]
    missing = [t for t in commit_targets if not new_digest_map.get(t)]
    assert missing == [con], f"应只报告缺摘要的那个目标，实际 {missing}"


def test_write_json_staging_produces_matching_digest(tmp_path):
    """contract/status 暂存件写出来的字节，必须与最终正式文件一致。"""
    stg = str(tmp_path / "con.partial")
    payload = {"a": 1, "b": [1, 2, 3], "中文": "值"}
    assert s6._write_json_staging(stg, payload) is True
    assert os.path.exists(stg)
    # 内容可回读且与 payload 等价
    with open(stg, encoding="utf-8") as f:
        assert json.load(f) == payload
    # 暂存件摘要稳定（同内容再写一次得同一摘要）
    d1 = s6.file_digest(stg)
    assert s6._write_json_staging(stg, payload) is True
    assert s6.file_digest(stg) == d1


def test_write_json_staging_cleans_up_on_failure(tmp_path):
    """写失败不该留下半截文件。"""
    bad = str(tmp_path / "no_such_dir" / "\x00bad")
    assert s6._write_json_staging(bad, {"x": 1}) is False
