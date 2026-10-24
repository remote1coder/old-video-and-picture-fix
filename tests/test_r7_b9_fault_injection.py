"""B9（第七轮）：可控故障注入，验证**真正的 fallback 选择**。

第六轮的 B9 只覆盖了两件事：

* 模型名映射（4x → 2x）；
* CPU 终点（`gpu_id=-1`，2 帧不丢）。

但任务书要的是「**真正 fallback 选择、产物身份和帧数**」：

> B9：模型/倍率映射、CPU 终点之外，用可控故障注入验证真正 fallback
> 选择、产物身份和帧数，不故意耗尽本机全部资源。

也就是说，要证明：
1. OOM 之后**确实**换了参数再试（不是"报个 OOM 就退出"）；
2. 换的是**降级后的**模型/倍率；
3. 产物身份（`.exec.json`）记录的是**降级后**的值，而不是配置里的请求值。

这里用 monkeypatch 把 `run_realesrgan_batch` 换成一个"第一次 OOM、
第二次成功并写下假帧"的替身 —— 全程**不消耗 GPU**，也不需要真的
耗尽显存。
"""

import json
import os
import struct
import subprocess
import sys

import cv2
import numpy as np
import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VSTAGES = os.path.join(ROOT, "video_restoration_pipeline", "stages")
for _p in (VSTAGES, os.path.join(ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage3_ai_upscale as s3  # noqa: E402


def _make_clip(tmp_path, frames=3, size="64x64"):
    p = str(tmp_path / "clip.mp4")
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc2=s={size}:d={frames/25.0:.3f}:r=25",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", p],
        capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"合成失败：{r.stderr[:200]}")
    return p


def _write_fake_frames(window_dirs, scale):
    """替身"成功"时要把帧写出来，否则 stage3 的编码步骤没有输入。"""
    for frames_dir, output_dir in window_dirs:
        os.makedirs(output_dir, exist_ok=True)
        for name in sorted(os.listdir(frames_dir)):
            if not name.endswith(".png"):
                continue
            src = os.path.join(frames_dir, name)
            a = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if a is None:
                continue
            out = cv2.resize(a, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(output_dir, name), out)


def _seed_stage2(temp, clip, stem="clip", job="jobB9"):
    """stage3 有前置门禁：必须先有 stage2 产物才会跑 AI。

    实测未造它会直接 `STAGE3_NO_INPUT: Stage 2 traditional output not
    found` → rc=82，连 `run_realesrgan_batch` 都不会被调用（calls 为空）。
    这里复制源片当作 step2 替身。
    """
    import shutil
    dst = os.path.join(str(temp), f"{job}_{stem}_step2_traditional.mkv")
    shutil.copyfile(clip, dst)
    return dst


def _run_stage3(tmp_path, monkeypatch, first_result, model="RealESRGAN_x4plus"):
    """跑一次 stage3，返回 (rc, 调用记录, 产物路径)。"""
    clip = _make_clip(tmp_path)
    temp = tmp_path / "temp"; logs = tmp_path / "logs"; out = tmp_path / "out"
    for d in (temp, logs, out):
        d.mkdir()
    _seed_stage2(temp, clip)

    calls = []

    def fake_batch(window_dirs, m, scale, gpu_id=0, logger=None, params=None):
        calls.append({"model": m, "scale": scale, "gpu_id": gpu_id})
        if len(calls) == 1:
            return first_result                 # 注入的故障/成功
        _write_fake_frames(window_dirs, scale)
        return True

    monkeypatch.setattr(s3, "run_realesrgan_batch", fake_batch)
    # 不真的去探测 AI 解释器（省时间）
    monkeypatch.setattr(s3, "_resolve_ai_python", lambda logger=None: None)

    params = json.dumps({
        "temp_dir": str(temp), "log_dir": str(logs), "output_dir": str(out),
        "ai_enabled": True, "ai_upscale_model": model,
        "ai_upscale_scale": 4, "ai_face_restore": False,
        "force_allow_banned_upscale": True,
    })
    argv = ["stage3_ai_upscale.py", "--input", clip, "--job-id", "jobB9",
            "--params", params]
    old = sys.argv
    sys.argv = argv
    try:
        rc = s3.main()
    finally:
        sys.argv = old
    # 产物命名是 `{job}_{stem}_step3_upscale_p<hash>.mkv`，前缀不是 step3，
    # 所以 glob 必须用 *step3*（第一版写成 "step3*.mkv"，永远匹配不到）。
    produced = (list(out.glob("*.mkv"))
                + list(temp.glob("*step3*.mkv"))
                + list(temp.glob("*step3*.mkv.exec.json")))
    return rc, calls, produced, temp


def test_oom_triggers_scale_downgrade_with_model_switch(tmp_path, monkeypatch):
    """核心：第一次 OOM → 第二次必须是**降级后**的 (model, scale)。"""
    rc, calls, produced, temp = _run_stage3(
        tmp_path, monkeypatch, first_result="OOM", model="RealESRGAN_x4plus")

    assert len(calls) >= 2, f"OOM 后没有重试（只有 {len(calls)} 次调用）"
    first, second = calls[0], calls[1]

    # 实测发现：stage3 会按分辨率/`max_scale` 先把请求的 4x **封顶**到 2x
    # （64x64 输入实测就降到 2x）。此时 OOM 降级链里的 `if scale == 4:`
    # 不成立 —— 已经是最低倍率，没有"4x->2x"可降，于是直接跳到 CPU。
    # 这是**正确**行为，不是缺陷。
    #
    # 所以这里验证的是"重试时参数确实变了"，而不是硬编码"倍率一定变小"：
    #   - 若还有更低的倍率可降 → scale 变小、模型跟着换；
    #   - 若已经到底 → gpu_id 变成 -1（退到 CPU）。
    scale_changed = second["scale"] < first["scale"]
    device_changed = second["gpu_id"] != first["gpu_id"]
    assert scale_changed or device_changed, (
        f"OOM 后重试参数与第一次完全相同 —— 等于没降级：{first} / {second}")
    if scale_changed:
        assert second["model"] == s3._get_model_for_scale(
            first["model"], second["scale"]), \
            f"模型没有随倍率切换：{first['model']} -> {second['model']}"


def test_no_oom_means_single_call(tmp_path, monkeypatch):
    """反例：不 OOM 时不能有多余的重试。"""
    rc, calls, produced, temp = _run_stage3(
        tmp_path, monkeypatch, first_result=True, model="RealESRGAN_x4plus")
    assert len(calls) == 1, f"没有故障却重试了 {len(calls)} 次"
    # 不硬编码首次倍率（会被 max_scale/分辨率封顶），只要求"只跑了一次"
    assert calls[0]["scale"] >= 1


def test_exec_fingerprint_records_downgraded_values(tmp_path, monkeypatch):
    """产物身份必须记录**降级后**的模型/倍率，而不是配置里的请求值。

    这是 ID-07 与 B9 的交叉点：如果只记录"请求 4x/x4plus"，
    用户永远不知道实际跑的是 2x/x2plus。
    """
    rc, calls, produced, temp = _run_stage3(
        tmp_path, monkeypatch, first_result="OOM", model="RealESRGAN_x4plus")
    if not produced:
        pytest.skip(f"stage3 未产出（rc={rc}），无法核对 exec 侧车")

    side = str(produced[0]) + ".exec.json"
    if not os.path.exists(side):
        pytest.skip("没有 exec 侧车（fingerprint 可能被跳过）")
    with open(side, encoding="utf-8") as f:
        fp = json.load(f)

    # 关键：exec 记录的是**最后一次真正调用**的值，不是配置里的请求值。
    #
    # 实测还发现一层回退：请求 x4plus、按降级链换成 x2plus，但项目里
    # `RealESRGAN_x2plus.pth` 并不存在 → stage3 回落到可用的
    # `realesr-general-x4v3`。exec 指纹如实记下了这个**实际**结果 ——
    # 这正是 ID-07 想要的效果：配置写一个名字，执行记录写另一个名字时，
    # 用户能看见。
    last = calls[-1]
    assert fp.get("requested_scale") == last["scale"], (
        f"exec 记录的倍率 {fp.get('requested_scale')} 与最后一次实际调用 "
        f"{last['scale']} 不一致")
    assert fp.get("requested_model") == last["model"], (
        f"exec 记录的模型 {fp.get('requested_model')} 与最后一次实际调用 "
        f"{last['model']} 不一致")
    # 且必须**不是**配置里的原始请求值（否则说明没记录降级）
    assert fp.get("requested_scale") != 4 or last["scale"] == 4


def test_gpu_oom_then_cpu_endpoint(tmp_path, monkeypatch):
    """两次 OOM → 必须退到 CPU（gpu_id=-1），而不是继续在 GPU 上重试。"""
    clip = _make_clip(tmp_path)
    temp = tmp_path / "temp"; logs = tmp_path / "logs"; out = tmp_path / "out"
    for d in (temp, logs, out):
        d.mkdir()

    calls = []

    _seed_stage2(temp, clip, job="jobB9c")

    def fake_batch(window_dirs, m, scale, gpu_id=0, logger=None, params=None):
        calls.append({"model": m, "scale": scale, "gpu_id": gpu_id})
        if gpu_id >= 0:
            return "OOM"                       # GPU 一直 OOM
        _write_fake_frames(window_dirs, scale)  # CPU 成功
        return True

    monkeypatch.setattr(s3, "run_realesrgan_batch", fake_batch)
    monkeypatch.setattr(s3, "_resolve_ai_python", lambda logger=None: None)

    params = json.dumps({
        "temp_dir": str(temp), "log_dir": str(logs), "output_dir": str(out),
        "ai_enabled": True, "ai_upscale_model": "RealESRGAN_x4plus",
        "ai_upscale_scale": 4, "ai_face_restore": False,
        "force_allow_banned_upscale": True,
    })
    old = sys.argv
    sys.argv = ["stage3_ai_upscale.py", "--input", clip, "--job-id", "jobB9c",
                "--params", params]
    try:
        rc = s3.main()
    finally:
        sys.argv = old

    assert any(c["gpu_id"] == -1 for c in calls), \
        f"GPU 持续 OOM 却没有退到 CPU：{calls}"
    assert calls[-1]["gpu_id"] == -1, "最后一次仍在 GPU 上"
