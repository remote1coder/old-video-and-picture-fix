# -*- coding: utf-8 -*-
"""G11：磁盘写入必须**按类型**记账，不能只报一个总数。

## 为什么

"总写入 3 GB"既可能是"源被复制了一遍"，也可能是"六代中间产物没清"——
要修的地方完全不同。任务书把写入分成六类：
source_copy / intermediate_media / final_candidate / old_generation_backup /
index_report / tool_cache。

## 判据

1. 分类按**语义**（是不是最终成品、是不是侧车、是不是工具缓存），不按大小；
2. 拿不准的宁可归 `intermediate_media`，不能归 `index_report`（会低估媒体）；
3. 无法识别的类别**单列**，不合并进别的桶（合并等于把未知洗成已知）；
4. 统计不到的文件要报 `unreadable`，不能静默少算；
5. 同一文件被扫两次只算一次（output 目录可能就是 temp 的父目录）。
"""
import os

import pytest

import conftest  # noqa: F401

import write_ledger as WL


def _mk(path, n=10):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(str(path), "wb") as fh:
        fh.write(b"x" * n)
    return str(path)


def test_classify_final_vs_intermediate_vs_report(tmp_path):
    out = str(tmp_path / "out" / "final.mkv")
    assert WL.classify(out, output_path=out) == "final_candidate"
    assert WL.classify(str(tmp_path / "temp" / "job_x_step1_pre_p1.mkv"),
                       output_path=out) == "intermediate_media"
    assert WL.classify(str(tmp_path / "temp" / "job_x_step1.mkv.identity.json"),
                       output_path=out) == "index_report"
    assert WL.classify(str(tmp_path / "temp" / "a.ffindex"),
                       output_path=out) == "tool_cache"


def test_attempt_staging_counts_as_final_candidate(tmp_path):
    p = str(tmp_path / "out" / "clip_restored.attempt-abc.mkv")
    assert WL.classify(p, output_path=str(tmp_path / "out" / "clip_restored.mkv")) \
        == "final_candidate"


def test_unknown_extension_falls_back_to_media_not_report():
    """拿不准的扩展名必须归 media，不能归 report。

    归 report 会把"未知的媒体字节"洗成"小报告"，于是"写了多少媒体"被
    系统性低估 —— 而那正是要盯的数。归 media 只会让报告类偏大，看得见。
    """
    assert WL.classify("x.weirdext") == "intermediate_media"
    assert WL.classify("x.mkv") == "intermediate_media"
    # 已知的报告扩展名仍然归 report（不能反过来把侧车算成媒体）
    assert WL.classify("x.json") == "index_report"


def test_scan_counts_bytes_and_files_per_category(tmp_path):
    out = str(tmp_path / "out" / "final.mkv")
    _mk(out, 100)
    _mk(tmp_path / "temp" / "a_step1.mkv", 200)
    _mk(tmp_path / "temp" / "a_step1.mkv.identity.json", 30)
    _mk(tmp_path / "temp" / "x.ffindex", 5)

    r = WL.scan([str(tmp_path)], output_path=out)
    by = r["by_category"]
    assert by["final_candidate"]["bytes"] == 100
    assert by["intermediate_media"]["files"] == 1
    assert by["intermediate_media"]["bytes"] == 200
    assert by["index_report"]["bytes"] == 30
    assert by["tool_cache"]["bytes"] == 5
    assert r["total"]["files"] == 4


def test_scan_dedupes_when_output_dir_nested_in_temp(tmp_path):
    out = str(tmp_path / "out" / "final.mkv")
    _mk(out, 100)
    r = WL.scan([str(tmp_path), str(tmp_path / "out")], output_path=out)
    assert r["by_category"]["final_candidate"]["files"] == 1, \
        "同一个文件被扫了两次，字节数会虚高"


def test_missing_root_is_ignored_not_fatal(tmp_path):
    r = WL.scan([str(tmp_path / "nope")])
    assert r["total"]["files"] == 0


def test_ledger_keeps_unknown_bucket_separate():
    led = WL.WriteLedger()
    led.record("intermediate_media", 10)
    led.record("weird_bucket", 99)
    s = led.summary()
    assert s["by_category"]["intermediate_media"]["bytes"] == 10
    assert s["unknown_categories"], "未知类别必须单列，不能悄悄丢掉或并入他桶"


def test_ledger_record_path_uses_the_same_classifier(tmp_path):
    out = str(tmp_path / "out" / "final.mkv")
    p = _mk(out, 42)
    led = WL.WriteLedger()
    cat = led.record_path(p, output_path=out)
    assert cat == "final_candidate"
    assert led.summary()["by_category"]["final_candidate"]["bytes"] == 42


def test_render_lists_present_and_written_separately(tmp_path):
    out = str(tmp_path / "out" / "final.mkv")
    _mk(out, 10)
    scan = WL.scan([str(tmp_path)], output_path=out)
    led = WL.WriteLedger()
    led.record("intermediate_media", 1234)
    text = "\n".join(WL.render(scan, led.summary()))
    assert "present=" in text and "written=" in text
    assert "intermediate_media" in text


def test_render_surfaces_unreadable_entries(tmp_path):
    scan = {"by_category": {}, "total": {"files": 0, "bytes": 0},
            "unreadable": 3}
    text = "\n".join(WL.render(scan))
    assert "unreadable" in text and "3" in text
