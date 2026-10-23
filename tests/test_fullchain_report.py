# -*- coding: utf-8 -*-
"""全链核对筛查报告（体检工具）的回归测试。

`full_chain_verify_mode: absolute` 在老素材上失败率很高（实测 204 个手机
素材约 31%），而这些漂移大多是**源片自带的**。所以 absolute 档主要当作
**体检工具**：配合 `allow_verify_fail_publish: true` 跑一遍，再用本报告
筛出"哪些素材有漂移、漂移多大"，而不是拿它拒绝出片。
"""
import io
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "shared_modules"))

import fullchain_report as fr  # noqa: E402


@pytest.fixture()
def outdir(tmp_path):
    """造一个含两份 fullchain.json 的输出目录。"""
    good = {"ok": True,
            "source": {"frames": 458, "duration": 15.19},
            "output": {"frames": 458, "duration": 15.21},
            "checks": {"frames": {"src": 458, "out": 458, "ok": True},
                       "duration": {"src": 15.19, "out": 15.21,
                                    "ok": True}}}
    bad = {"ok": False,
           "source": {"frames": 556, "duration": 28.44},
           "output": {"frames": 500, "duration": 28.46},
           "checks": {"frames": {"src": 556, "out": 500, "ok": False},
                      "duration": {"src": 28.44, "out": 28.46,
                                   "ok": True}}}
    for name, payload in (("clip_good_restored.mkv", good),
                          ("clip_bad_restored.mkv", bad)):
        with io.open(str(tmp_path / (name + ".fullchain.json")), "w",
                     encoding="utf-8") as f:
            json.dump(payload, f)
    return str(tmp_path)


def test_collect_reads_all_sidecars(outdir):
    files = sorted(os.path.join(outdir, n) for n in os.listdir(outdir)
                   if n.endswith(".fullchain.json"))
    rows = fr.collect(files, outdir)
    assert len(rows) == 2
    names = {r["name"] for r in rows}
    assert names == {"clip_good_restored.mkv", "clip_bad_restored.mkv"}


def test_collect_reports_frame_mismatch(outdir):
    files = [os.path.join(outdir, "clip_bad_restored.mkv.fullchain.json")]
    rows = fr.collect(files, outdir)
    r = rows[0]
    assert r["src_frames"] == 556 and r["out_frames"] == 500
    assert r["frames_ok"] is False
    assert r["ok"] is False


def test_collect_keeps_duration_ok(outdir):
    files = [os.path.join(outdir, "clip_bad_restored.mkv.fullchain.json")]
    r = fr.collect(files, outdir)[0]
    # 帧数不符但时长一致 —— 报告要把两者分开呈现
    assert r["frames_ok"] is False
    assert r["duration_ok"] is True


def test_rows_sorted_by_drift_desc(outdir):
    files = sorted(os.path.join(outdir, n) for n in os.listdir(outdir)
                   if n.endswith(".fullchain.json"))
    rows = fr.collect(files, outdir)
    rows.sort(key=lambda r: (r.get("av_drift_ms") is None,
                             -(r.get("av_drift_ms") or 0)))
    # 没有成品本体可量漂移 → av_drift_ms 全为 None，排序应稳定不炸
    assert all(r["av_drift_ms"] is None for r in rows)


def test_main_requires_a_directory():
    assert fr.main([]) == 81


def test_main_rejects_non_directory(tmp_path):
    p = str(tmp_path / "nope")
    assert fr.main([p]) == 82


def test_main_writes_csv(outdir, tmp_path):
    csvp = str(tmp_path / "rep.csv")
    assert fr.main([outdir, "--csv", csvp]) == 0
    assert os.path.isfile(csvp)
    with io.open(csvp, encoding="utf-8-sig") as f:
        head = f.readline()
    assert "src_frames" in head and "av_drift_ms" in head


def test_main_empty_dir_is_ok(tmp_path):
    assert fr.main([str(tmp_path)]) == 0
