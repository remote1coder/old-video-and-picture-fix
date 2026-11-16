# -*- coding: utf-8 -*-
"""从**真实 main** 跑照片内存链（N27-05 的验收）。

不是单元测试：起 `photo_restoration_pipeline/main.py` 子进程，
走分类器 → inproc 链（S1–S5）→ S6 编码，断言：
  * rc=0 且产出 AVIF；
  * 几何 = 输入 × S3 的 2 倍；
  * **temp 里没有中间件**（inproc 的意义就在这里）。
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "shared_modules"))

pytestmark = pytest.mark.slow


def _tools_dir() -> str:
    return str(ROOT / "tools_r4")


def _make_scan_like(path: Path, w=600, h=450):
    """造一张**会被分类器判成 SCAN_PRINT** 的图。

    判据里 DPI ≥ 1200 是 SCAN_PRINT 的强信号（实测 conf=0.80，
    刚好过 0.80 的批量阈值）。纯渐变 + 一个色块，避免"全平"。
    """
    y = np.linspace(0, 1, h)[:, None, None]
    x = np.linspace(0, 1, w)[None, :, None]
    base = 0.3 + 0.4 * y + 0.2 * x
    arr = np.clip(np.concatenate([base, base * 0.8, 1 - base * 0.7],
                                  axis=2) * 255, 0, 255).astype(np.uint8)
    arr[h // 3:2 * h // 3, w // 4:3 * w // 4] = [200, 160, 120]
    Image.fromarray(arr).save(str(path), dpi=(1200, 1200))


@pytest.fixture()
def env(tmp_path):
    if not (ROOT / "tools_r4" / "avifenc.exe").exists():
        pytest.skip("avifenc not found in tools_r4")
    if not (ROOT / ".venv_r4" / "Scripts" / "python.exe").exists():
        pytest.skip("no .venv_r4 interpreter")
    e = os.environ.copy()
    e["PATH"] = _tools_dir() + os.pathsep + e.get("PATH", "")
    (tmp_path / "temp").mkdir()
    (tmp_path / "out").mkdir()
    (tmp_path / "logs").mkdir()
    return e


def test_main_inproc_produces_avif_without_intermediates(tmp_path, env):
    src = tmp_path / "scan.png"
    _make_scan_like(src)

    cmd = [sys.executable, str(ROOT / "photo_restoration_pipeline" / "main.py"),
           "--config", str(ROOT / "photo_restoration_pipeline" / "config.yaml"),
           "--input", str(src), "--job-id", "e2e",
           "--temp-dir", str(tmp_path / "temp"),
           "--log-dir", str(tmp_path / "logs"),
           "--output", str(tmp_path / "out"),
           "--photo-chain-mode", "inproc"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                       env=env, cwd=str(tmp_path))
    assert r.returncode == 0, (
        "main 失败 rc=%s\nstdout尾部=%s\nstderr尾部=%s"
        % (r.returncode, (r.stdout or "")[-600:], (r.stderr or "")[-600:]))

    outs = list((tmp_path / "out").glob("*.avif"))
    assert outs, "没有产出 AVIF: %r" % list((tmp_path / "out").iterdir())

    # 几何：输入 ×2（S3 的 upscale_scale=2）
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(outs[0])], capture_output=True, text=True, timeout=120)
    assert probe.returncode == 0, probe.stderr[:300]
    w, h = [int(x) for x in probe.stdout.strip().split(",")[:2]]
    assert (w, h) == (600 * 2, 450 * 2), (w, h)

    # **没有中间件**：temp 里不该有 step*.png / *.tif
    leftovers = [p.name for p in (tmp_path / "temp").iterdir()
                 if p.suffix.lower() in (".png", ".tif", ".tiff")]
    assert leftovers == [], "inproc 模式留下了中间件: %r" % leftovers


def test_inproc_refuses_without_model_worker(tmp_path, env):
    """没有 .venv_r4 时不做模型阶段（本机有，所以只验证"能跑"的分支）。"""
    # 这条只确认 `photo_chain.capabilities()` 的语义没漂
    import photo_chain as PC
    assert "s3" not in PC.capabilities()["memory_stages"]
    assert "s3" in PC.capabilities(with_model_client=True)["memory_stages"]
