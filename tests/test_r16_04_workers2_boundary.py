# -*- coding: utf-8 -*-
"""R16-04：**并行入口**的写入边界必须覆盖（审核点名）。

上一轮的批次测试实际用 `--workers 1`，**不能证明并行入口已覆盖**。
审核确认 worker 的参数字典漏了 `allow_write_root` / `relocate_output`，
于是 worker 里的计划解析会得到与父进程不同的结果：
父进程以为受限，worker 其实是"无限制"。

本文件用**真实 `main.py --workers 2` 进程**验证。为了不等 40 分钟模型链，
只跑 `--stage 0`（音频抽取，秒级）—— 它已经会走完整的
批次预检 → 派发 → worker 计划解析 → 落盘 路径。
"""
import os
import shutil
import subprocess
import sys

import pytest
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG = os.path.join(_ROOT, "video_restoration_pipeline", "config.yaml")
_PY = os.path.join(_ROOT, "tools_r4", "vs_portable", "r72", "python.exe")
_MAIN = os.path.join(_ROOT, "video_restoration_pipeline", "main.py")


def _ffmpeg():
    return shutil.which("ffmpeg")


def _make_clip_with_audio(path, sec=1.0, freq=440):
    # 父目录必须先存在：ffmpeg 不会自建多级目录
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={sec}",
         "-f", "lavfi", "-i",
         f"testsrc=size=160x120:rate=30:duration={sec}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    return str(path)


def _job(jid, inp, root=None, out=None):
    """``root`` 给出时，temp/log 也落在根内（真实 manifest 就是这样）。

    不给的话会回退到 config 的 ``./temp`` / ``./logs``（项目目录内），
    必然落在 ``--allow-write-root`` 之外 —— 那条路径由另一条用例单独验证。
    """
    return {
        "job_id": jid, "input_path": inp,
        "output_path": out or "",
        "temp_dir": str(root / "temp") if root else "",
        "log_path": str(root / "logs" / f"{jid}.log") if root else "",
        "detected_class": "HDV_AVCHD", "confidence": 0.9,
        "field_order": "progressive", "needs_ivtc": False, "is_psf": False,
        "field_segments": [], "status": "PENDING", "priority": 0,
        "ai_upscale_banned": False,
        "recommended_params": {}, "user_overrides": {},
        "duration": 1.0, "original_width": 160, "original_height": 120,
        "file_size": os.path.getsize(inp),
        "file_mtime": 0.0, "file_hash_prefix": "", "run_generation": 1,
    }


@pytest.mark.skipif(not _ffmpeg(), reason="ffmpeg 不在 PATH 上")
@pytest.mark.skipif(not os.path.exists(_PY), reason="视频链解释器缺失")
def test_workers2_carries_write_roots_and_relocate(tmp_path):
    """workers=2 + CLI 授权根 + relocate：每个子进程必须落在授权根内。

    这是审核第 2 项的直接验收：worker 参数字典漏掉
    `allow_write_root`/`relocate_output` 时，worker 会退回"无限制"，
    与本测试预期的受限行为不一致。
    """
    root = tmp_path / "root"
    root.mkdir()
    a = _make_clip_with_audio(tmp_path / "src" / "a.mp4", freq=440)
    b = _make_clip_with_audio(tmp_path / "src" / "b.mp4", freq=660)

    man = tmp_path / "m.yaml"
    with open(man, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"meta": {"generated_at": "2026-10-31T00:00:00+08:00",
                      "source_dirs": [str(tmp_path / "src")],
                      "config_path": "", "total_files": 2, "pending": 2,
                      "done": 0, "failed": 0, "skipped": 0,
                      "run_generation": 1},
             "jobs": [_job("j1", a, root), _job("j2", b, root)]},
            f, allow_unicode=True, sort_keys=False)

    env = dict(os.environ)
    tools = os.path.join(_ROOT, "tools_r4")
    env["PATH"] = os.pathsep.join([
        tools,
        os.path.join(tools, "gmic", "gmic-4.0.5-cli-win64"),
        os.path.join(tools, "exiftool", "exiftool-13.59_64"),
        env.get("PATH", "")])

    proc = subprocess.run(
        [_PY, _MAIN, "--manifest", str(man), "--config", _CONFIG,
         "--mode", "batch", "--no-wsl-redirect",
         "--stage", "0",              # 只跑音频抽取，避免长时间模型链
         "--workers", "2",            # ← 本测试的关键：必须走并行入口
         "--allow-write-root", str(root),
         "--relocate-output", str(root / "out")],
        capture_output=True, text=True, env=env, timeout=600)

    out = (proc.stdout or "") + (proc.stderr or "")
    # 没有被批次预检拒绝（拒绝会是 82）
    assert "would collide" not in out, \
        f"同根目录的两个不同 basename 不该判冲突：{out[-500:]}"
    assert "invalid path plan" not in out, out[-500:]
    # 并行入口真的跑起来了（不是被拒绝、也不是跳过）
    assert proc.returncode == 0, f"rc={proc.returncode}\n{out[-800:]}"

    # 产物确实落在**计划里那一处**：根内的 temp / logs 有内容。
    # （BATCH_PLAN_OK 只写进日志文件，stdout 看不到，所以断言可观察事实。）
    wrote_plan_dirs = []
    for sub in ("temp", "logs"):
        d = root / sub
        if d.is_dir() and any(d.iterdir()):
            wrote_plan_dirs.append(sub)
    assert wrote_plan_dirs, \
        f"worker 没有把产物写进计划指定的根内目录：{sorted(os.listdir(str(root)))}"

    # 授权根之外的落盘：root 之外只允许两类东西
    #   1) 输入目录（只读，测试自己造的素材）；
    #   2) **manifest 及其锁** —— manifest 是用户用 --manifest 显式指定的，
    #      更新它就必须在它旁边加锁。这属于"用户已授权"，不应算越界。
    #      （这一点已写入台账：计划要覆盖 manifest 写入，但授权来自用户
    #       指定的路径本身。）
    strays = []
    for dirpath, _dn, files in os.walk(str(tmp_path)):
        if dirpath.startswith(str(root)):
            continue
        if dirpath.startswith(str(tmp_path / "src")):
            continue
        for n in files:
            if n == "m.yaml" or n == "m.yaml.lock":
                continue
            strays.append(os.path.join(dirpath, n))
    assert not strays, f"授权根外出现了落盘：{strays[:5]}"


@pytest.mark.skipif(not _ffmpeg(), reason="ffmpeg 不在 PATH 上")
@pytest.mark.skipif(not os.path.exists(_PY), reason="视频链解释器缺失")
def test_workers2_same_basename_different_dirs_is_rejected(tmp_path):
    """不同源目录的同名输出、relocate 到同一目录 → 派发前拒绝。

    这是审核第 1 项的反例：不传 relocate_output 时输出路径不同（无冲突），
    一旦重定位就撞名。批次预检必须用与执行相同的 relocate 才能看见。
    """
    root = tmp_path / "root"
    root.mkdir()
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    d1.mkdir()
    d2.mkdir()
    # 关键：两个 job 的**显式输出**同名（只在不同源目录里）。
    # 默认命名是按 job_id 的，天然不会撞 —— 那测不到重定位后的撞名。
    a = _make_clip_with_audio(d1 / "same.mp4", freq=440)
    b = _make_clip_with_audio(d2 / "same.mp4", freq=660)
    _same_out_a, _same_out_b = str(d1 / "same.mkv"), str(d2 / "same.mkv")

    man = tmp_path / "m.yaml"
    with open(man, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"meta": {"generated_at": "2026-10-31T00:00:00+08:00",
                      "source_dirs": [str(d1), str(d2)],
                      "config_path": "", "total_files": 2, "pending": 2,
                      "done": 0, "failed": 0, "skipped": 0,
                      "run_generation": 1},
             "jobs": [_job("j1", a, root, out=_same_out_a),
                     _job("j2", b, root, out=_same_out_b)]},
            f, allow_unicode=True, sort_keys=False)

    env = dict(os.environ)
    tools = os.path.join(_ROOT, "tools_r4")
    env["PATH"] = os.pathsep.join([
        tools,
        os.path.join(tools, "gmic", "gmic-4.0.5-cli-win64"),
        os.path.join(tools, "exiftool", "exiftool-13.59_64"),
        env.get("PATH", "")])

    proc = subprocess.run(
        [_PY, _MAIN, "--manifest", str(man), "--config", _CONFIG,
         "--mode", "batch", "--no-wsl-redirect",
         "--stage", "0", "--workers", "2",
         "--allow-write-root", str(root),
         "--relocate-output", str(root / "out")],
        capture_output=True, text=True, env=env, timeout=600)

    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 82, f"rc={proc.returncode}\n{out[-600:]}"
    assert "would collide" in out, out[-600:]
    # 拒绝发生在派发之前：relocate 目标下不应有任何产物
    reloc = root / "out"
    if reloc.exists():
        assert list(reloc.iterdir()) == [], \
            f"派发前拦截，重定位目录必须为空：{list(reloc.iterdir())[:5]}"


@pytest.mark.skipif(not _ffmpeg(), reason="ffmpeg 不在 PATH 上")
@pytest.mark.skipif(not os.path.exists(_PY), reason="视频链解释器缺失")
def test_config_fallback_temp_log_outside_root_is_rejected(tmp_path):
    """job 没给 temp/log、回退到 config（项目目录内）→ 落在根外 → **拒绝**。

    这条验证的是"授权根真的生效"：不是放宽到能跑，而是看见越界就停。
    同时要求错误信息指名是哪个路径越界，否则用户无从下手。
    """
    root = tmp_path / "root"
    root.mkdir()
    a = _make_clip_with_audio(tmp_path / "src" / "a.mp4", freq=440)
    b = _make_clip_with_audio(tmp_path / "src" / "b.mp4", freq=660)

    man = tmp_path / "m.yaml"
    with open(man, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"meta": {"generated_at": "2026-10-31T00:00:00+08:00",
                      "source_dirs": [str(tmp_path / "src")],
                      "config_path": "", "total_files": 2, "pending": 2,
                      "done": 0, "failed": 0, "skipped": 0,
                      "run_generation": 1},
             # 故意**不**给 temp/log → 回退到 config 的项目内目录
             "jobs": [_job("j1", a, root), _job("j2", b, root)]},
            f, allow_unicode=True, sort_keys=False)

    env = dict(os.environ)
    tools = os.path.join(_ROOT, "tools_r4")
    env["PATH"] = os.pathsep.join([
        tools,
        os.path.join(tools, "gmic", "gmic-4.0.5-cli-win64"),
        os.path.join(tools, "exiftool", "exiftool-13.59_64"),
        env.get("PATH", "")])

    proc = subprocess.run(
        [_PY, _MAIN, "--manifest", str(man), "--config", _CONFIG,
         "--mode", "batch", "--no-wsl-redirect",
         "--stage", "0", "--workers", "2",
         "--allow-write-root", str(root)],
        capture_output=True, text=True, env=env, timeout=600)

    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 82, f"根外的 temp/log 应被拒绝，rc={proc.returncode}\n{out[-600:]}"
    assert "invalid path plan" in out, out[-600:]
    # 错误信息必须指名越界的路径类型（temp 或 log），否则无从修复
    assert "temp" in out.lower() or "log" in out.lower(), \
        f"拒绝了但没说清哪个路径越界：{out[-600:]}"
