# -*- coding: utf-8 -*-
"""R20-A06：时间检查必须独立于 mkvmerge（审核 R19-A06）。

审核原文：

> 纯时间检查独立于 mkvmerge。工具是否存在只决定能否执行该重封装方案，
> 不能决定是否检查时间。

旧实现把 `check_against_expected` 那段**嵌在 `_has_timing_transform` 分支里**：

- 没声明变换 → 根本不校验；
- 没装 mkvmerge → 走 `elif not _mkv` 分支，**也不校验**，
  只打一条 warning 就继续发布。

本文件锁住：`check_declared_timeline_mapping()` 是独立函数，
**两个分支都调用它**，且 strict 模式下不通过就拒绝发布。
"""
import ast
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE7 = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                       "stage7_encode.py")

if os.path.join(_ROOT, "shared_modules") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "shared_modules"))
if os.path.join(_ROOT, "video_restoration_pipeline", "stages") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "video_restoration_pipeline",
                                    "stages"))


def _src():
    return open(_STAGE7, encoding="utf-8", errors="replace").read()


# ------------------------------------------------ ① 校验函数存在且独立

def test_mapping_check_is_a_module_level_function():
    """必须是模块级函数 —— 才能被"没有 mkvmerge"的分支复用。"""
    tree = ast.parse(_src())
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert "check_declared_timeline_mapping" in names


def test_both_branches_call_the_mapping_check():
    """`not _mkv` 分支与 transform 分支**都要**调用。"""
    src = _src()
    # 除定义外至少两处调用
    calls = src.count("check_declared_timeline_mapping(")
    assert calls >= 3, f"调用点只有 {calls - 1} 处（含定义共 {calls}）"


def test_mkvmerge_missing_branch_still_checks_timeline():
    """`no mkvmerge` 分支的代码块里必须出现时间校验调用。"""
    src = _src()
    i = src.find('"reason": "no mkvmerge"')
    assert i > 0, "找不到 no-mkvmerge 分支"
    # 取该分支之后的 1200 字符，看有没有校验调用
    tail = src[i:i + 1200]
    assert "check_declared_timeline_mapping(" in tail, (
        "mkvmerge 缺失分支没有调用时间校验 —— 审核明确不允许跳过检查")


def test_strict_mode_refuses_to_publish_on_mapping_fail():
    """strict（默认）下校验不通过 → 返回 86，不发布。"""
    src = _src()
    # 两处调用之后都应有 "timeline_mapping_strict" + return 86
    assert src.count("timeline_mapping_strict") >= 2, (
        "至少两处分支都要遵守 strict 开关")
    assert src.count("STAGE7_EXPECTED_MAPPING_FAIL_NONSTRICT") >= 2, (
        "非 strict 时必须留 CRITICAL，标注未验证")


# ------------------------------------------------ ② 函数行为

def _stage7():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_s7_r20a06", _STAGE7)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def s7():
    return _stage7()


def test_returns_ok_when_no_transform_declared(s7, tmp_path):
    """没声明变换 → 不算失败（mapping_ok=None，不是 False）。"""
    src = tmp_path / "s.mkv"
    src.write_bytes(b"x")
    cand = tmp_path / "c.mkv"
    cand.write_bytes(b"y")
    ok, info = s7.check_declared_timeline_mapping(
        {}, str(src), str(cand), None)
    assert ok is True
    assert info.get("mapping_ok") is None
    assert "no declared transform" in info.get("mapping_reason", "")


def test_returns_ok_when_source_pts_unreadable(s7, tmp_path):
    """源 PTS 读不出来（不存在/非视频）→ 不阻断，但要标 mapping_ok=None。"""
    ok, info = s7.check_declared_timeline_mapping(
        {"fps_interpret": "25"}, str(tmp_path / "nope.mp4"),
        str(tmp_path / "nope2.mp4"), None)
    assert ok is True
    assert info.get("mapping_ok") is None


def test_speed_transform_mismatch_is_reported(s7, tmp_path):
    """真实素材：声明 2× 变速但候选没变速 → 必须报 fail。"""
    import shutil
    ff = shutil.which("ffmpeg")
    if not ff:
        pytest.skip("ffmpeg 不在 PATH")
    src = tmp_path / "s.mp4"
    os.system(f'"{ff}" -v error -y -f lavfi -i '
              f'"testsrc2=size=64x48:rate=10:duration=2" '
              f'-c:v libx264 -pix_fmt yuv420p "{src}"')
    if not src.exists():
        pytest.skip("造片失败")
    # 候选 = 原样（没变速），但声明 speed=2 → 期望时长应减半
    ok, info = s7.check_declared_timeline_mapping(
        {"expected_speed": 2.0}, str(src), str(src), None)
    assert ok is False, info
    assert info.get("mapping_ok") is False
    assert info.get("mapping_problems")
