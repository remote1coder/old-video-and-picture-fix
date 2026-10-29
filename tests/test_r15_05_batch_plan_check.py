# -*- coding: utf-8 -*-
"""R15-05：批次在 dispatch 前一次性检查所有计划。

**为什么需要它**

`detect_cross_job_conflicts()` 早就写好了（且已区分"共享容器目录"与
"独占文件"），但**从来没有调用点** —— 于是两个 job 写到同一份媒体/侧车、
或一个 job 的输出覆盖另一个 job 的输入，都要等整批跑完才发现。

本文件用**真实 `main.py` 进程**验证：碰撞必须在任何子进程启动之前就拦下
（退出 82，且输出目录为空）。
"""
import io
import os
import shutil
import subprocess
import sys

import pytest
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG = os.path.join(_ROOT, "video_restoration_pipeline", "config.yaml")
_PY = os.path.join(_ROOT, "tools_r4", "vs_portable", "r72", "python.exe")


def _ffmpeg():
    return shutil.which("ffmpeg") is not None


def _make_clip(path, size="160x120", rate=30, dur=1, src="testsrc"):
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y",
         "-f", "lavfi", "-i",
         f"{src}=size={size}:rate={rate}:duration={dur}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    return str(path)


def _job(jid, inp, out, temp, log):
    return {
        "job_id": jid, "input_path": inp, "output_path": out,
        "temp_dir": temp, "log_path": log,
        "detected_class": "HDV_AVCHD", "confidence": 0.9,
        "field_order": "progressive", "needs_ivtc": False, "is_psf": False,
        "field_segments": [], "status": "PENDING", "priority": 0,
        "ai_upscale_banned": False,
        "recommended_params": {"ivtc": "auto", "vfm_mode": 1,
                               "vdecimate_cycle": 5,
                               "deinterlace": "QTGMC",
                               "psf_handling": "SeparateFields_Weave"},
        "user_overrides": {}, "duration": 1.0,
        "original_width": 160, "original_height": 120,
        "file_size": os.path.getsize(inp),
        "file_mtime": 0.0, "file_hash_prefix": "", "run_generation": 1,
    }


def _write_manifest(path, jobs, source_dirs):
    doc = {"meta": {"generated_at": "2026-10-30T02:00:00+08:00",
                    "source_dirs": source_dirs, "config_path": "",
                    "total_files": len(jobs), "pending": len(jobs),
                    "done": 0, "failed": 0, "skipped": 0,
                    "run_generation": 1},
           "jobs": jobs}
    with io.open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
    return str(path)


def _run_main(manifest, root):
    env = dict(os.environ)
    tools = os.path.join(_ROOT, "tools_r4")
    env["PATH"] = os.pathsep.join(
        [tools,
         os.path.join(tools, "gmic", "gmic-4.0.5-cli-win64"),
         os.path.join(tools, "exiftool", "exiftool-13.59_64"),
         env.get("PATH", "")])
    return subprocess.run(
        [_PY, os.path.join(_ROOT, "video_restoration_pipeline", "main.py"),
         "--manifest", manifest, "--config", _CONFIG, "--mode", "batch",
         "--no-wsl-redirect", "--workers", "1",
         "--allow-write-root", root],
        capture_output=True, text=True, env=env, timeout=300)


@pytest.mark.skipif(not _ffmpeg(), reason="ffmpeg 不在 PATH 上")
@pytest.mark.skipif(not os.path.exists(_PY), reason="视频链解释器缺失")
def test_same_output_is_caught_before_dispatch(tmp_path):
    """两个 job 写同一份媒体 → 派发前拦下，输出目录为空。"""
    inp_a = _make_clip(tmp_path / "a.mp4")
    inp_b = _make_clip(tmp_path / "b.mp4")
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)

    jobs = [_job("j1", inp_a, os.path.join(out, "same.mkv"),
                 str(tmp_path / "temp"), str(tmp_path / "logs" / "j1.log")),
            _job("j2", inp_b, os.path.join(out, "same.mkv"),
                 str(tmp_path / "temp"), str(tmp_path / "logs" / "j2.log"))]
    man = _write_manifest(str(tmp_path / "m.yaml"), jobs, [str(tmp_path)])

    r = _run_main(man, str(tmp_path))
    assert r.returncode == 82, f"rc={r.returncode} out={r.stdout[-300:]}"
    assert "would collide" in (r.stdout + r.stderr), \
        (r.stdout + r.stderr)[-400:]
    assert os.listdir(out) == [], "派发前拦截，输出目录必须为空"


@pytest.mark.skipif(not _ffmpeg(), reason="ffmpeg 不在 PATH 上")
@pytest.mark.skipif(not os.path.exists(_PY), reason="视频链解释器缺失")
def test_output_overwriting_another_input_is_caught(tmp_path):
    """A 的输出正好是 B 的输入 → 派发前拦下（会毁掉 B 的素材）。"""
    inp_a = _make_clip(tmp_path / "a.mp4")
    # B 的输入就是 A 将要写的文件
    a_out = str(tmp_path / "aout")
    os.makedirs(a_out, exist_ok=True)
    b_in = os.path.join(a_out, "a_restored.mkv")
    _make_clip(b_in)

    jobs = [_job("j1", inp_a, b_in, str(tmp_path / "temp"),
                 str(tmp_path / "logs" / "j1.log")),
            _job("j2", b_in, str(tmp_path / "out2" / "b_restored.mkv"),
                 str(tmp_path / "temp"), str(tmp_path / "logs" / "j2.log"))]
    man = _write_manifest(str(tmp_path / "m.yaml"), jobs, [str(tmp_path)])

    r = _run_main(man, str(tmp_path))
    assert r.returncode == 82, f"rc={r.returncode}"
    assert "覆盖" in (r.stdout + r.stderr) or "collide" in (r.stdout + r.stderr), \
        (r.stdout + r.stderr)[-400:]


@pytest.mark.skipif(not _ffmpeg(), reason="ffmpeg 不在 PATH 上")
@pytest.mark.skipif(not os.path.exists(_PY), reason="视频链解释器缺失")
def test_shared_temp_and_log_dirs_are_not_a_conflict(tmp_path):
    """**共享 temp/log 目录不是冲突** —— 不能误杀。

    这条走完整预检（不真正跑编码，靠"没被拦下"来证明没有误判）：
    若被误判，rc 会是 82 且提示 collide。
    """
    inp_a = _make_clip(tmp_path / "a.mp4")
    inp_b = _make_clip(tmp_path / "b.mp4")
    out = str(tmp_path / "out")
    os.makedirs(out, exist_ok=True)

    jobs = [_job("j1", inp_a, os.path.join(out, "a_restored.mkv"),
                 str(tmp_path / "shared_temp"), str(tmp_path / "shared_logs" / "j1.log")),
            _job("j2", inp_b, os.path.join(out, "b_restored.mkv"),
                 str(tmp_path / "shared_temp"), str(tmp_path / "shared_logs" / "j2.log"))]
    man = _write_manifest(str(tmp_path / "m.yaml"), jobs, [str(tmp_path)])

    r = _run_main(man, str(tmp_path))
    # 关键：不能因为"共享目录"被判冲突
    assert "would collide" not in (r.stdout + r.stderr), \
        f"共享 temp/log 被误判为冲突：{(r.stdout + r.stderr)[-400:]}"
    # 若因碰撞被拦，rc 会是 82；这里只要求不是"collide"这一原因
    if r.returncode == 82:
        assert "collide" not in (r.stdout + r.stderr)
