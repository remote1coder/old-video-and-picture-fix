# -*- coding: utf-8 -*-
"""第十九轮 C：探针判据必须严格（对应 R18-A08）。

审核给出的三个**旧版会误报 IDENTICAL** 的受控输入：

| 受控输入 | 旧结论 |
|---|---|
| 两个文件声明 60 帧，只比较到 1 帧 inf | IDENTICAL |
| 两条流水线均 rc=86，仍留有可比较的 step4 | IDENTICAL |
| 上述两例都没有融合回执，fused 仍有 step3 | 仍不阻断 IDENTICAL |

本文件直接测 `g34_equiv_det.evaluate()`（纯函数），把每一条判据单独钉住。
另外测**路径推断**：旧版 `dirname(dirname(__file__))` 得到的是 `tests/`。
"""
import importlib.util
import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROBE = os.path.join(_ROOT, "tests", "probes", "g34_equiv_det.py")


def _load():
    spec = importlib.util.spec_from_file_location("g34_det", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe():
    if not os.path.isfile(_PROBE):
        pytest.skip("探针不存在")
    return _load()


# ------------------------------------------------ 路径推断

def test_repo_root_points_at_the_repository(probe):
    """**核心**：`tests/probes/x.py` 往上三级才是 repo 根。

    旧版 `dirname(dirname(abspath(__file__)))` 只上两级，得到 `tests/`，
    从项目根启动也改不了这个结果。
    """
    got = os.path.abspath(probe._repo_root())
    assert os.path.isfile(os.path.join(got, "video_restoration_pipeline",
                                       "main.py")), (
        f"_repo_root() 没有指向仓库根：{got}")
    assert os.path.basename(got) == "old-video-and-picture-fix", got


# ------------------------------------------------ 判据：每一条都要能单独阻断

def _ok_sides():
    classic = {"rc": 0, "step3": "/w/c/temp/a_step3.mkv",
               "step4": "/w/c/temp/a_step4.mkv", "receipts": []}
    fused = {"rc": 0, "step3": None,
             "step4": "/w/f/temp/b_step4.mkv",
             "receipts": ["b.fused_stage34.json"]}
    # R21-C08：判据新增"时间轴必须独立核对"，所以**合法样本**也要带上
    # PTS / 末帧覆盖 —— 否则它证明不了"两边时间轴相同"，被拒绝是对的。
    info_c = {"width": 3840, "height": 2160, "pix_fmt": "yuv420p16le",
              "nb_read_frames": "20", "frames": 20,
              "first_pts": 0.0, "last_pts": 0.633367,
              "last_end": 0.6667, "median_interval": 0.033367,
              # R22-A07：判据逐帧比 PTS，合法样本必须给出逐帧证据
              "pts": [round(i * 0.033367, 6) for i in range(20)],
              "durations": [0.033367] * 20}
    info_f = dict(info_c)
    rows = [{"n": str(i), "psnr_y": "inf", "psnr_u": "inf", "psnr_v": "inf"}
            for i in range(1, 21)]
    cmp_result = {"returncode": 0, "frames": 20, "rows": rows, "summary": []}
    return classic, fused, info_c, info_f, cmp_result


def test_all_good_is_identical(probe):
    """全满足才算 IDENTICAL（防止把判定改成永远失败）。"""
    verdict, reasons = probe.evaluate(*_ok_sides(), 20)
    assert verdict == "IDENTICAL", reasons


def test_nonzero_rc_blocks_identical(probe):
    """**审核的受控例**：两条流水线 rc=86，仍留有可比较的 step4。"""
    classic, fused, ic, iff, cmp_result = _ok_sides()
    classic["rc"] = 86
    fused["rc"] = 86
    verdict, reasons = probe.evaluate(classic, fused, ic, iff, cmp_result, 20)
    assert verdict == "NOT_IDENTICAL"
    assert any("rc=86" in r for r in reasons), reasons


def test_fused_step3_blocks_identical(probe):
    """融合如果还产出 step3，说明它根本没生效 —— 不能算等价。"""
    classic, fused, ic, iff, cmp_result = _ok_sides()
    fused["step3"] = "/w/f/temp/b_step3.mkv"
    verdict, reasons = probe.evaluate(classic, fused, ic, iff, cmp_result, 20)
    assert verdict == "NOT_IDENTICAL"
    assert any("fusion did not actually run" in r for r in reasons), reasons


def test_missing_receipt_blocks_identical(probe):
    """没有融合回执 → 无法证明走的是融合路径。"""
    classic, fused, ic, iff, cmp_result = _ok_sides()
    fused["receipts"] = []
    verdict, reasons = probe.evaluate(classic, fused, ic, iff, cmp_result, 20)
    assert verdict == "NOT_IDENTICAL"
    assert any("receipt" in r for r in reasons), reasons


def test_partial_comparison_blocks_identical(probe):
    """**审核的受控例**：声明 60 帧但只比较到 1 帧 inf。"""
    classic, fused, ic, iff, cmp_result = _ok_sides()
    ic["nb_read_frames"] = "60"
    iff["nb_read_frames"] = "60"
    cmp_result["rows"] = cmp_result["rows"][:1]
    cmp_result["frames"] = 1
    verdict, reasons = probe.evaluate(classic, fused, ic, iff, cmp_result, 60)
    assert verdict == "NOT_IDENTICAL"
    assert any("partial comparison" in r for r in reasons), reasons


def test_zero_frames_blocks_identical(probe):
    """零帧比较绝不能通过。"""
    classic, fused, ic, iff, cmp_result = _ok_sides()
    cmp_result["rows"] = []
    cmp_result["frames"] = 0
    verdict, reasons = probe.evaluate(classic, fused, ic, iff, cmp_result, 20)
    assert verdict == "NOT_IDENTICAL"
    assert any("no frames" in r for r in reasons), reasons


def test_missing_plane_blocks_identical(probe):
    """缺一个平面 → 不得 IDENTICAL。"""
    classic, fused, ic, iff, cmp_result = _ok_sides()
    for row in cmp_result["rows"]:
        row.pop("psnr_v", None)
    verdict, reasons = probe.evaluate(classic, fused, ic, iff, cmp_result, 20)
    assert verdict == "NOT_IDENTICAL"
    assert any("planes missing" in r for r in reasons), reasons


def test_finite_psnr_blocks_identical(probe):
    """只要有一帧不是 inf，就不是"逐像素相同"。"""
    classic, fused, ic, iff, cmp_result = _ok_sides()
    cmp_result["rows"][7]["psnr_v"] = "42.5"
    verdict, reasons = probe.evaluate(classic, fused, ic, iff, cmp_result, 20)
    assert verdict == "NOT_IDENTICAL"
    assert any("not identical" in r for r in reasons), reasons


def test_frame_count_mismatch_blocks_identical(probe):
    classic, fused, ic, iff, cmp_result = _ok_sides()
    iff["nb_read_frames"] = "19"
    verdict, reasons = probe.evaluate(classic, fused, ic, iff, cmp_result, 20)
    assert verdict == "NOT_IDENTICAL"
    assert any("frame count differs" in r for r in reasons), reasons


def test_pix_fmt_mismatch_blocks_identical(probe):
    classic, fused, ic, iff, cmp_result = _ok_sides()
    iff["pix_fmt"] = "gbrp16le"
    verdict, reasons = probe.evaluate(classic, fused, ic, iff, cmp_result, 20)
    assert verdict == "NOT_IDENTICAL"
    assert any("pix_fmt differs" in r for r in reasons), reasons


# ------------------------------------------------ 不删别人的目录

def test_probe_does_not_import_shutil_for_deletion(probe):
    """探针**不得**具备删除目录的能力（旧版会 `shutil.rmtree(work)`）。

    注意：不能只在全文里搜 "rmtree" —— 文档字符串里合法地写了
    "绝不 rmtree" 来解释历史。所以检查的是**能力**：没有 `import shutil`
    就不可能调用 rmtree。
    """
    src = open(_PROBE, encoding="utf-8", errors="replace").read()
    assert "import shutil" not in src, "探针不应 import shutil（旧版用它删目录）"


def test_probe_refuses_a_non_empty_work_dir(tmp_path):
    """`--work` 指向已存在且非空的目录时必须**拒绝**（返回 2）。

    旧版无条件 `shutil.rmtree(work)`，会删掉别人的目录。
    """
    import subprocess
    d = tmp_path / "occupied"
    d.mkdir()
    (d / "someone-elses-file.txt").write_text("keep me")

    r = subprocess.run(
        [sys.executable, _PROBE, "--repo", _ROOT, "--work", str(d)],
        capture_output=True, text=True, errors="replace", timeout=120)
    assert r.returncode == 2, (
        f"非空 --work 应被拒绝，实际 rc={r.returncode}: "
        f"{(r.stdout or '')[-200:]}{r.stderr or ''}"[-300:])
    assert (d / "someone-elses-file.txt").exists(), "别人的文件被删了"
