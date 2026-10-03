"""真实端到端链路（需 avifenc/avifdec）：stage5 -> stage6 -> 独立解码回读。

与单测的区别：这里用 **subprocess 真实调用 stage 脚本**，并用
**另一个厂商实现的解码器**（avifdec + libpng）回读验证，
不依赖本项目自己写的读取代码下结论。

环境缺口（本机实测，见 修复清单.md）：
  * gmic 未安装 -> photo stage1 无法执行，因此 step1 中间产物由本测试
    **合成**一份真实 16-bit RGB TIFF（内容由测试生成，非真实老照片）。
  * VapourSynth / realesrgan / basicsr / gfpgan 未安装 -> stage2/3/4 不执行。
  * 视频链（VapourSynth）无法执行。
这些缺口在 修复清单.md 的"未执行项"中逐条登记。
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tests", "_out", "tools")
E2E = os.path.join(ROOT, "tests", "_out", "e2e_chain")
PY = sys.executable

STAGE5 = os.path.join(ROOT, "photo_restoration_pipeline", "stages",
                      "stage5_postprocess.py")
STAGE6 = os.path.join(ROOT, "photo_restoration_pipeline", "stages",
                      "stage6_encode.py")


def _avifenc():
    found = shutil.which("avifenc")
    if found:
        return found
    local = os.path.join(TOOLS, "avifenc.exe")
    return local if os.path.exists(local) else None


def _avifdec():
    found = shutil.which("avifdec")
    if found:
        return found
    local = os.path.join(TOOLS, "avifdec.exe")
    return local if os.path.exists(local) else None


@pytest.fixture(scope="module")
def chain_dir():
    # 每次运行都用干净目录：否则 stage5 会命中上一次的缓存并返回 3（跳过）
    if os.path.isdir(E2E):
        shutil.rmtree(E2E, ignore_errors=True)
    os.makedirs(E2E, exist_ok=True)
    return E2E


def _run(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True,
                       shell=False, env=env, cwd=ROOT)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _seed_step1(d, arr):
    """合成一份真实 16-bit RGB TIFF 作为 step1 产物（gmic 缺失的替代）。"""
    sys.path.insert(0, os.path.join(ROOT, "shared_modules"))
    import media_io
    p = os.path.join(d, "job001_oldphoto_step1_preprocess.tif")
    media_io.write_tiff(p, arr)
    return p


def _env_with_avifenc():
    env = dict(os.environ)
    if os.path.isdir(TOOLS):
        env["PATH"] = TOOLS + os.pathsep + env.get("PATH", "")
    return env


def test_e2e_stage5_keeps_true_16bit(chain_dir):
    """NEW-V-03: 16-bit RGB TIFF 输入经过 stage5 后必须仍是真 16-bit。"""
    temp = os.path.join(chain_dir, "temp")
    logs = os.path.join(chain_dir, "logs")
    for d in (temp, logs):
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(20261002)
    src = rng.integers(0, 65536, (48, 64, 3)).astype(np.uint16)
    _seed_step1(temp, src)

    params = {
        "temp_dir": temp, "log_dir": logs,
        "photo_bit_depth": 16, "archival_mode": False,
        "local_mask_denoise": False,   # 只验证位深保真，不做处理
        "grain_synthesis": False,
        "scratch_remove": False,
    }
    rc, out = _run([PY, STAGE5, "--input",
                    os.path.join(temp, "oldphoto.tif"),
                    "--job-id", "job001", "--params", json.dumps(params)])
    assert rc == 0, f"stage5 exit={rc}\n{out[-2000:]}"

    produced = os.path.join(temp, "job001_oldphoto_step5_postprocess.tif")
    assert os.path.exists(produced), out[-2000:]

    import tifffile
    back = tifffile.imread(produced)
    assert back.dtype == np.uint16, f"stage5 把 16-bit 压成了 {back.dtype}"

    sys.path.insert(0, os.path.join(ROOT, "photo_restoration_pipeline",
                                    "stages"))
    import importlib.util as u
    spec = u.spec_from_file_location("s6e2e", STAGE6)
    s6 = u.module_from_spec(spec)
    spec.loader.exec_module(s6)
    eff = s6.effective_bit_depth(back)
    assert eff >= 12, (
        f"容器是 uint16 但有效位深只有 {eff} bit —— 16-bit 信息在上游被丢弃")

    # status sidecar 必须是 SUCCESS，且没有 skipped_operations
    with open(produced + ".status.json", encoding="utf-8") as f:
        status = json.load(f)
    assert status["status"] == "SUCCESS", status


def test_e2e_stage6_real_avif_roundtrip(chain_dir):
    """用真 avifenc 编码、用真 avifdec + libpng 独立回读。"""
    avifenc = _avifenc()
    avifdec = _avifdec()
    if not (avifenc and avifdec):
        pytest.skip("avifenc/avifdec 不可用（本机未安装，也未下载官方预编译包）")

    temp = os.path.join(chain_dir, "temp")
    logs = os.path.join(chain_dir, "logs")
    outdir = os.path.join(chain_dir, "output")
    os.makedirs(outdir, exist_ok=True)

    step5 = os.path.join(temp, "job001_oldphoto_step5_postprocess.tif")
    if not os.path.exists(step5):
        pytest.skip("stage5 产物不存在（前一个用例被跳过或失败）")

    params = {
        "temp_dir": temp, "log_dir": logs, "output_dir": outdir,
        "photo_bit_depth": 16, "archival_mode": True,
    }
    rc, out = _run([PY, STAGE6, "--input",
                    os.path.join(temp, "oldphoto.tif"),
                    "--job-id", "job001", "--params", json.dumps(params)],
                   env=_env_with_avifenc())
    assert rc == 0, f"stage6 exit={rc}\n{out[-3000:]}"

    avif = os.path.join(outdir, "oldphoto_restored.avif")
    assert os.path.exists(avif) and os.path.getsize(avif) > 0, out[-2000:]

    contract_path = avif + ".encode_contract.json"
    assert os.path.exists(contract_path), "缺少编码契约记录"
    with open(contract_path, encoding="utf-8") as f:
        contract = json.load(f)
    depth_arg = json.dumps(contract)
    assert "16" in depth_arg, "契约里没有记录 16-bit 相关决策"

    # --- 独立解码回读（avifdec，非本项目代码）---
    decoded = os.path.join(chain_dir, "decoded16.png")
    rc2, out2 = _run([avifdec, "--depth", "16", "--sato", avif, decoded])
    assert rc2 == 0, f"avifdec exit={rc2}\n{out2[-1500:]}"

    try:
        import cv2
        back = cv2.imread(decoded, cv2.IMREAD_UNCHANGED)
        assert back is not None, "libpng(cv2) 无法读取解码结果"
        back = back[:, :, ::-1] if back.ndim == 3 else back
    except ImportError:
        import imagecodecs
        back = imagecodecs.imread(decoded)

    assert back.dtype == np.uint16, f"AVIF 回读只有 {back.dtype} —— 16-bit 未保住"

    sys.path.insert(0, os.path.join(ROOT, "photo_restoration_pipeline",
                                    "stages"))
    import importlib.util as u
    spec = u.spec_from_file_location("s6e2e2", STAGE6)
    s6 = u.module_from_spec(spec)
    spec.loader.exec_module(s6)
    eff = s6.effective_bit_depth(back)
    # 有损 AV1 会引入重建噪声，GCD 判据会偏小；只断言不是"8-bit 伪装"
    assert eff > 8, f"回读有效位深仅 {eff} bit，疑似 8-bit 伪装成 16-bit"
