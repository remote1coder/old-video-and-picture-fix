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


def _source_array():
    """固定种子的合成 16-bit RGB 素材（gmic 缺失，无法跑真实 stage1）。"""
    rng = np.random.default_rng(20261002)
    return rng.integers(0, 65536, (48, 64, 3)).astype(np.uint16)


def _ensure_step5(chain_dir):
    """确保 step5 产物存在（不存在就自己跑一遍 stage5）。

    RV-交付要求：每条 e2e 用例必须自带输入，不能依赖前一个用例的产物 ——
    单独运行 stage6 用例时不应该 skip。
    """
    temp = os.path.join(chain_dir, "temp")
    logs = os.path.join(chain_dir, "logs")
    os.makedirs(temp, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    step5 = os.path.join(temp, "job001_oldphoto_step5_postprocess.tif")
    if os.path.exists(step5):
        return step5

    _seed_step1(temp, _source_array())
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
    return step5


def _load_stage6_module(alias):
    import importlib.util as u
    spec = u.spec_from_file_location(alias, STAGE6)
    mod = u.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _env_with_avifenc():
    env = dict(os.environ)
    if os.path.isdir(TOOLS):
        env["PATH"] = TOOLS + os.pathsep + env.get("PATH", "")
    return env


def test_e2e_stage5_keeps_true_16bit(chain_dir):
    """NEW-V-03: 16-bit RGB TIFF 输入经过 stage5 后必须仍是真 16-bit。"""
    temp = os.path.join(chain_dir, "temp")
    os.makedirs(temp, exist_ok=True)
    src = _source_array()
    _seed_step1(temp, src)
    step5 = _ensure_step5(chain_dir)

    import tifffile
    back = tifffile.imread(step5)
    assert back.dtype == np.uint16, f"stage5 把 16-bit 压成了 {back.dtype}"

    s6 = _load_stage6_module("s6e2e")
    # RV-14: 量化网格诊断只作参考，真正的无损判据是**逐像素比较**
    assert np.array_equal(back, src), \
        "stage5 在无处理分支上改动了像素（应逐像素相等）"

    # status sidecar 必须是 SUCCESS，且没有 skipped_operations
    with open(step5 + ".status.json", encoding="utf-8") as f:
        status = json.load(f)
    assert status["status"] == "SUCCESS", status
    assert status["skipped_operations"] == [], status


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

    # 自带输入：不依赖前一个用例（单独运行本用例也必须能跑）
    step5 = _ensure_step5(chain_dir)

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
    # 不能只查"JSON 里出现过字符串 16"—— 必须核对真正决定位深的字段
    assert contract["depth_arg"] == "12,8", contract
    assert contract["effective_decode_depth"] == 16, contract
    assert contract["cicp"] == "2/2/0", contract  # 未知来源 → 未指定标签
    assert contract["lossless"] is True

    # RV-12: 成功必须留下 SUCCESS 状态侧车
    with open(avif + ".status.json", encoding="utf-8") as f:
        st = json.load(f)
    assert st["status"] == "SUCCESS", st

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

    # 归档档是 lossless：必须与 stage5 输出**逐像素相等**（不是"差不多"）
    import tifffile
    reference = tifffile.imread(step5)
    assert np.array_equal(back, reference), (
        "无损归档分支出现像素差异："
        f"MAE={np.abs(back.astype(np.int64) - reference.astype(np.int64)).mean()}")


def test_e2e_lossless_assertion_can_actually_fail(chain_dir):
    """负向对照：注入 1-LSB 错误后，上面的无损判据必须失败。

    没有这条，"array_equal 通过"可能只是因为两边都被写成了同一个空数组。
    """
    step5 = _ensure_step5(chain_dir)
    import tifffile
    ref = tifffile.imread(step5)
    assert ref.size > 0

    corrupted = ref.copy()
    corrupted[0, 0, 0] = (int(corrupted[0, 0, 0]) + 1) & 0xFFFF
    assert not np.array_equal(corrupted, ref), "1-LSB 错误竟然被判为相等"

    swapped = ref[:, :, ::-1].copy()
    assert not np.array_equal(swapped, ref), "通道交换竟然被判为相等"
