# -*- coding: utf-8 -*-
"""stage3 必须清掉同一 job_id 上一轮残留在 temp 里的帧。

2026-10-26 实测事故
------------------
两次跑批都叫 job_001、共用 temp/。上一轮（585 帧素材）的
`temp/job_001_upscaled/frame_*.png` 没清，本轮（300 帧素材）直接
glob("*.png") 把它们当自己的输出 → src 300 / out 585 →
全链核对判 FRAMES 失败（rc=86）。
因为它表现为"帧数多了"而不是"损坏"，非常隐蔽。

本测试锁住：清的是**本 job 专属目录**的 PNG，别的东西一律不动。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "shared_modules"))
sys.path.insert(0, os.path.join(_ROOT, "video_restoration_pipeline", "stages"))

import stage3_ai_upscale as s3  # noqa: E402


@pytest.fixture()
def temp_tree(tmp_path):
    t = tmp_path / "temp"
    t.mkdir()
    # 本 job 的两类帧目录（含上一轮残留）
    for d in ("job_001_upscaled", "job_001_upscaled_256"):
        (t / d).mkdir()
        for i in range(5):
            (t / d / ("frame_%08d.png" % i)).write_bytes(b"x")
    # 本 job 目录里的非 PNG（侧车）必须保留
    (t / "job_001_upscaled" / "identity.json").write_bytes(b"{}")
    # 别的 job 的目录必须保留
    (t / "job_002_upscaled").mkdir()
    (t / "job_002_upscaled" / "frame_00000000.png").write_bytes(b"y")
    return str(t)


def test_stale_pngs_of_same_job_are_removed(temp_tree):
    removed = s3._clear_stale_frame_dirs(temp_tree, "job_001")
    assert removed == 10, f"应删 10 个 PNG，实际 {removed}"
    assert not list((__import__("pathlib").Path(temp_tree) / "job_001_upscaled")
                    .glob("*.png"))
    assert not list((__import__("pathlib").Path(temp_tree) / "job_001_upscaled_256")
                    .glob("*.png"))


def test_other_jobs_untouched(temp_tree):
    s3._clear_stale_frame_dirs(temp_tree, "job_001")
    from pathlib import Path
    other = Path(temp_tree) / "job_002_upscaled" / "frame_00000000.png"
    assert other.exists(), "误删了其他 job 的帧！"
    assert other.read_bytes() == b"y"


def test_sidecar_files_untouched(temp_tree):
    s3._clear_stale_frame_dirs(temp_tree, "job_001")
    from pathlib import Path
    assert (Path(temp_tree) / "job_001_upscaled" / "identity.json").exists(), \
        "误删了非 PNG 的侧车文件"


def test_no_job_prefix_does_not_overmatch(tmp_path):
    """job_1 不能把 job_10 的目录当成自己的删掉。"""
    t = tmp_path / "temp"
    t.mkdir()
    (t / "job_10_upscaled").mkdir()
    (t / "job_10_upscaled" / "f.png").write_bytes(b"z")
    removed = s3._clear_stale_frame_dirs(str(t), "job_1")
    # glob 的 `job_1_upscaled*` 不会匹配 `job_10_upscaled`（因为要求 `_upscaled`）
    assert removed == 0, f"job_1 误删了 job_10 的帧（removed={removed}）"
    assert (t / "job_10_upscaled" / "f.png").exists()


def test_missing_dir_is_noop(tmp_path):
    assert s3._clear_stale_frame_dirs(str(tmp_path), "job_999") == 0
