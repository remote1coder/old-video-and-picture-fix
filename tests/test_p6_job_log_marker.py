# -*- coding: utf-8 -*-
"""看门狗的"日志增长"分量必须真的能定位到日志文件。

## 这条测试防的是一个**已经踩过两次**的坑

第一次：marker 只算 temp/out 的字节数 → 纯计算段落（stage2/stage3 摘要、
stage5 人脸推理、stage7 校验）在它眼里完全静止 → 长片被 300s 判 stall。

第二次：我加了"日志大小"进 marker，但只信 `params["log_path"]`，
而 **manifest 里的路径与 logger 实际落盘的文件名不一致**：

    manifest:  logs/job_001.log
    实际文件:  logs/job_job_001.log     ← logger 自己又加了一次 `job_` 前缀

于是 `os.path.exists()` 为假 → 日志分量被**静默跳过** → 依然被判 stall
（`last marker=16443759457`，而日志每 30 秒都在涨）。

所以：宁可多试几种拼法，也不能"找不到就当 0"。
"""
import os

import conftest  # noqa: F401

import main as M   # video_restoration_pipeline/main.py


def _mk(tmp_path, name, n=1234):
    p = tmp_path / name
    p.write_bytes(b"x" * n)
    return str(p)


def test_finds_the_double_prefixed_log(tmp_path):
    """manifest 写 `job_001.log` 不存在，真实文件是 `job_job_001.log`。"""
    logs = tmp_path
    real = _mk(logs, "job_job_001.log", 777)
    params = {"log_path": str(logs / "job_001.log"),   # 不存在
              "log_dir": str(logs)}
    got = M._job_log_bytes(params, {"job_id": "job_001"})
    assert got == 777, f"应回退到真实文件 {real}，实际 {got}"


def test_falls_back_when_only_log_dir_is_known(tmp_path):
    _mk(tmp_path, "job_job_009.log", 42)
    got = M._job_log_bytes({"log_dir": str(tmp_path)}, {"job_id": "job_009"})
    assert got == 42


def test_prefers_existing_log_path(tmp_path):
    good = _mk(tmp_path, "whatever.log", 10)
    assert M._job_log_bytes({"log_path": good}, {}) == 10


def test_returns_zero_when_nothing_found(tmp_path):
    assert M._job_log_bytes({"log_dir": str(tmp_path)}, {"job_id": "nope"}) == 0


def test_marker_scales_log_bytes_by_1000():
    """日志涨几百字节也要让整数 marker 变化（16 GB 基数下否则会被淹没）。"""
    import io
    src = io.open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "video_restoration_pipeline", "main.py"),
        encoding="utf-8", errors="replace").read()
    assert "total += _job_log_bytes(params, job) * 1000" in src, \
        "marker 里必须把日志字节数放大后再累加"
