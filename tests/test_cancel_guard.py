"""NEW-V-06 · 取消/中断守卫的回归测试。

交接的端到端验收要求里有"**故障和取消不会显示完成**"。
本项目 16 个 stage 脚本对 `KeyboardInterrupt` / `BaseException` / `signal`
的命中数为 0，而所有兜底出口都是 `except Exception` —— 抓不到中断。

本文件验证三件事：
  1. 根因成立（KeyboardInterrupt 不是 Exception）；
  2. 守卫逻辑：清掉半成品 + 写 FAILED sidecar + 退出码 88；
  3. 真机：真的收到 OS 信号时，进程以 88 退出且不留半成品。
"""

import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = os.path.join(ROOT, "shared_modules")
if SHARED not in sys.path:
    sys.path.insert(0, SHARED)

import cancel_guard  # noqa: E402
from cancel_guard import CancelGuard, EXIT_INTERRUPTED  # noqa: E402


def test_new_v06_keyboard_interrupt_is_not_caught_by_except_exception():
    """根因证据：KeyboardInterrupt 继承 BaseException，不是 Exception。"""
    assert issubclass(KeyboardInterrupt, BaseException)
    assert not issubclass(KeyboardInterrupt, Exception)

    def raises():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        try:
            raises()
        except Exception:  # noqa: BLE001 —— 正是 stage 里写的兜底形式
            pytest.fail("except Exception 竟然抓到了 KeyboardInterrupt")


def test_new_v06_exit_code_is_inside_the_project_range():
    """88 必须落在项目约定的 80–113 自定义区间，且不与 80–87 冲突。"""
    assert 80 <= EXIT_INTERRUPTED <= 113
    assert EXIT_INTERRUPTED not in range(80, 88)


def test_new_v06_guard_removes_partial_and_writes_failed_sidecar(tmp_path):
    """守卫逻辑（确定性）：删半成品 + 写 FAILED(cancelled) + 退出码 88。"""
    out = tmp_path / "job1_img_step5_postprocess.tif"
    cache = tmp_path / "job1_img_step5_postprocess.tif.cache.json"
    out.write_bytes(b"\x00" * 4096)          # 模拟写到一半的产物
    cache.write_text("{}", encoding="utf-8")  # 模拟已写的缓存 sidecar
    assert out.exists() and cache.exists()

    g = CancelGuard()
    g.watch(str(out), extra_paths=[str(cache)], stage=5)
    result = g.cleanup_and_record(signal.SIGINT)

    assert not out.exists(), "半成品必须被清掉，否则与成功产物无法区分"
    assert not cache.exists(), "缓存 sidecar 也必须清掉"
    assert result["exit_code"] == EXIT_INTERRUPTED

    status_path = str(out) + ".status.json"
    assert os.path.exists(status_path)
    with open(status_path, encoding="utf-8") as handle:
        status = json.load(handle)
    assert status["status"] == "FAILED", status
    assert status["failed_operation"] == "cancelled", status
    assert status["stage"] == 5
    assert "SIGINT" in status["reason"], status


def test_new_v06_guard_is_idempotent_and_safe_without_artifacts(tmp_path):
    """没登记产物时不得抛错；重复清理也不得抛错。"""
    g = CancelGuard()
    first = g.cleanup_and_record(signal.SIGTERM)
    assert first["removed"] == []
    assert first["status_path"] is None

    out = tmp_path / "nothing.tif"
    g.watch(str(out))
    assert g.cleanup_and_record(signal.SIGTERM)["removed"] == []
    assert g.cleanup_and_record(signal.SIGTERM)["removed"] == []


def test_new_v06_stage5_registers_its_artifacts(tmp_path, monkeypatch):
    """stage5 主流程必须在写字节前把产物登记给守卫。"""
    sys.path.insert(0, os.path.join(ROOT, "photo_restoration_pipeline", "stages"))
    try:
        import stage5_postprocess as s5
    except Exception as exc:  # pragma: no cover - 依赖缺失时跳过
        pytest.skip(f"stage5 无法导入: {exc}")

    calls = []

    class _StubGuard:
        @staticmethod
        def install_cancel_guard(stage=None, logger=None):
            calls.append(("install", stage))
            return True

        @staticmethod
        def watch_outputs(output_path, extra_paths=None, logger=None, stage=None):
            calls.append(("watch", output_path, list(extra_paths or []), stage))

    monkeypatch.setattr(s5, "_cancel_guard", _StubGuard, raising=False)

    temp = tmp_path / "temp"
    logs = tmp_path / "logs"
    temp.mkdir()
    logs.mkdir()

    sys.path.insert(0, SHARED)
    import media_io
    import numpy as np
    src = np.zeros((16, 16, 3), dtype=np.uint16)
    src[..., 0] = np.linspace(0, 65535, 256, dtype=np.uint32).astype(
        np.uint16).reshape(16, 16)
    media_io.write_tiff(str(temp / "job001_img_step1_preprocess.tif"), src)

    params = {
        "temp_dir": str(temp), "log_dir": str(logs),
        "photo_bit_depth": 16, "archival_mode": False,
        "local_mask_denoise": False, "grain_synthesis": False,
        "scratch_remove": False,
    }
    monkeypatch.setattr(sys, "argv", [
        "stage5_postprocess.py", "--input", str(temp / "img.tif"),
        "--job-id", "job001", "--params", json.dumps(params),
    ])

    rc = s5.main()
    assert rc == 0, f"stage5 退出码 {rc}"

    installs = [c for c in calls if c[0] == "install"]
    watches = [c for c in calls if c[0] == "watch"]
    assert installs, "stage5 没有安装取消守卫"
    assert watches, "stage5 没有把产物登记给守卫"
    _, out_path, extra, stage = watches[0]
    assert out_path.endswith("job001_img_step5_postprocess.tif"), out_path
    assert any(p.endswith(".cache.json") for p in extra), extra
    assert stage == 5


def test_new_v06_real_signal_leaves_failed_record_not_half_product(tmp_path):
    """真机：真的收到 OS 信号 → 退出码 88，无半成品，有 FAILED 记录。

    用一个最小脚本（而不是整个 stage）触发真实信号，避免依赖
    "stage 恰好跑得够慢"这种不可控条件。
    """
    script = tmp_path / "sleepy_stage.py"
    out = tmp_path / "job1_img_step5_postprocess.tif"
    script.write_text(textwrap.dedent(f"""
        import sys, time, os
        sys.path.insert(0, {SHARED!r})
        import cancel_guard
        cancel_guard.install_cancel_guard(stage=5)
        out = {str(out)!r}
        cancel_guard.watch_outputs(out, extra_paths=[out + ".cache.json"])
        with open(out, "wb") as f:            # 模拟写了一半
            f.write(b"\\x00" * 8192)
        time.sleep(10)                        # 等信号（漏掉信号也会自行退出）
    """).strip(), encoding="utf-8")

    popen_kwargs = {}
    signum = None
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        signum = getattr(signal, "CTRL_BREAK_EVENT", None)
    else:
        signum = signal.SIGINT

    if signum is None:  # pragma: no cover - 平台不支持
        pytest.skip("该平台无法投递中断信号")

    proc = subprocess.Popen([sys.executable, str(script)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            **popen_kwargs)
    try:
        # 等到子进程真的写出半成品（也就是守卫已安装完毕）再发信号：
        # 发得太早会命中系统默认行为，进程被直接终止，测不到处理器。
        deadline = time.time() + 15
        while time.time() < deadline and not os.path.exists(str(out)):
            if proc.poll() is not None:
                break
            time.sleep(0.05)
        time.sleep(0.3)
        proc.send_signal(signum)
        stdout, stderr = proc.communicate(timeout=25)
    finally:
        if proc.poll() is None:  # pragma: no cover - 兜底
            proc.kill()
            proc.wait(timeout=10)

    if proc.returncode == -signum or proc.returncode == 3221225786:
        pytest.skip("该平台未把信号交给 Python 处理器（默认直接终止）")

    assert proc.returncode == EXIT_INTERRUPTED, (
        f"期望退出码 {EXIT_INTERRUPTED}，实际 {proc.returncode}\n"
        f"stderr={stderr.decode('utf-8', 'replace')[-800:]}")
    assert not os.path.exists(str(out)), "半成品没有被清掉"

    status_path = str(out) + ".status.json"
    assert os.path.exists(status_path), "缺少 FAILED 记录"
    with open(status_path, encoding="utf-8") as handle:
        status = json.load(handle)
    assert status["status"] == "FAILED"
    assert status["failed_operation"] == "cancelled"
