# -*- coding: utf-8 -*-
"""R22 §5.2：工作表示（色度/位深）选择的单元测试。

管的是"选什么表示"这件事本身：
  * 四态解析与理由（`auto` 默认不变，只给建议）
  * **能力前置核验**（libsvtav1 不支持 444 —— 实测）
  * 冲突检测（`chroma_mode: 444` + `pix_fmt: yuv420p10le`）
  * 位深改写（`set_bit_depth` 不能把 rgb48le 改成 rgb10le 这种胡说）
  * 端到端边界表（stage1…stage7 每个边界都有明确格式）

真实链路的贯通证据在 `tests/probes/r22_444_classic_path.py`
（三遍：420 基线 / 444 / 负例），本文件只做快速回归。
"""
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARED = os.path.join(_ROOT, "shared_modules")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

import work_format as wf  # noqa: E402


# ------------------------------------------------ 像素格式解析

@pytest.mark.parametrize("fmt,chroma", [
    ("yuv444p10le", "444"),
    ("yuv420p16le", "420"),
    ("yuv422p", "422"),
    ("gbrp16le", "rgb-or-gray"),
    ("rgb48le", "rgb-or-gray"),
    ("gray", "rgb-or-gray"),
    ("", "unknown"),
    ("something-weird", "unknown"),
])
def test_chroma_is_read_not_guessed(fmt, chroma):
    assert wf.chroma_of_pix_fmt(fmt) == chroma


def test_unknown_format_is_unknown_not_420():
    """读不出来 ≠ 是 420。猜成 420 会把"未知"当成"已知"。"""
    assert wf.chroma_of_pix_fmt("mystery") == "unknown"
    assert wf.chroma_of_pix_fmt(None) == "unknown"


@pytest.mark.parametrize("fmt,depth,expect", [
    ("yuv420p16le", 10, "yuv420p10le"),
    ("yuv444p16le", 10, "yuv444p10le"),
    ("yuv444p10le", 16, "yuv444p16le"),
    ("yuv444p10le", 8, "yuv444p"),
    ("gbrp16le", 12, "gbrp12le"),
    ("yuv420p16be", 10, "yuv420p10be"),
])
def test_set_bit_depth(fmt, depth, expect):
    assert wf.set_bit_depth(fmt, depth) == expect


@pytest.mark.parametrize("fmt", ["rgb48le", "gray", "weird"])
def test_set_bit_depth_leaves_unknown_names_alone(fmt):
    """认不出就原样返回 —— 不要造出 `rgb10le` 这种不存在的格式。"""
    assert wf.set_bit_depth(fmt, 10) == fmt


def test_bit_depth_of():
    assert wf.bit_depth_of("yuv444p10le") == 10
    assert wf.bit_depth_of("yuv444p") == 8
    assert wf.bit_depth_of("rgb48le") == 16
    assert wf.bit_depth_of("rgb24") == 8
    assert wf.bit_depth_of("weird") is None


@pytest.mark.parametrize("fmt,vs", [
    ("yuv444p16le", "vs.YUV444P16"),
    ("yuv420p10le", "vs.YUV420P10"),
    ("gray", ""),
    ("rgb48le", ""),
])
def test_vs_format_of(fmt, vs):
    assert wf.vs_format_of(fmt) == vs


# ------------------------------------------------ 能力核验（实测编码器）

def test_encoder_formats_are_read_from_ffmpeg():
    fmts = wf.ffmpeg_encoder_formats("libx265")
    if not fmts:
        pytest.skip("本机没有 libx265，无法核验")
    assert "yuv444p10le" in fmts


def test_svtav1_really_cannot_do_444():
    """这条是**实测事实**，不是查表：libsvtav1 只列 420 两种。

    若哪天真支持了，这条会失败 —— 那时才是改默认值的时机。
    """
    fmts = wf.ffmpeg_encoder_formats("libsvtav1")
    if not fmts:
        pytest.skip("本机没有 libsvtav1，无法核验")
    assert not any("444" in f for f in fmts), sorted(fmts)


def test_unknown_encoder_yields_empty_set():
    """读不到就是空集：调用方必须把它当"没能证明支持"。"""
    assert wf.ffmpeg_encoder_formats("") == set()
    assert wf.ffmpeg_encoder_formats("no-such-encoder") == set()


# ------------------------------------------------ 四态解析

def test_auto_keeps_the_compatible_path_and_only_recommends():
    """`auto` 不改变历史行为；源是 444 时只给建议，不替用户切换。"""
    p = wf.resolve({"chroma_mode": "auto", "video_encoder": "libx265"},
                   source_pix_fmt="yuv444p10le")
    assert p["resolved"] == "420"
    assert p["recommendation"] == "444"
    assert "4:4:4" in p["reason"]
    assert p["ok"] is True


def test_auto_without_mode_key_is_auto():
    p = wf.resolve({}, source_pix_fmt="yuv420p10le")
    assert p["mode"] == "auto" and p["resolved"] == "420"


def test_declared_444_pix_fmt_pulls_the_whole_chain():
    """用户把成品写成 444 → 整链跟着走。

    只在最后一步换标签是最典型的"静默转换后仍称 444"：前面全是 420，
    最后转成 444 落盘，细节一点没回来。
    """
    p = wf.resolve({"pix_fmt": "yuv444p10le", "video_encoder": "libx265"},
                   source_pix_fmt="yuv420p10le")
    assert p["resolved"] == "444"
    assert "whole chain follows" in p["reason"]
    assert p["boundaries"]["stage1_render"] == "yuv444p16le"


def test_preserve_follows_the_source():
    p = wf.resolve({"chroma_mode": "preserve", "video_encoder": "libx265"},
                   source_pix_fmt="yuv444p10le")
    assert p["resolved"] == "444" and p["ok"] is True
    p2 = wf.resolve({"chroma_mode": "preserve", "video_encoder": "libx265"},
                    source_pix_fmt="yuv420p10le")
    assert p2["resolved"] == "420" and p2["ok"] is True


def test_preserve_with_unknown_source_fails_instead_of_assuming_420():
    """源色度读不出来时，`preserve` 必须失败。

    猜 420 就是"把一个未知当已知"，而且会真的毁掉源里的色度细节。
    """
    p = wf.resolve({"chroma_mode": "preserve", "video_encoder": "libx265"})
    assert p["ok"] is False
    assert any(b["kind"] == "source_chroma_unknown" for b in p["blockers"])


def test_444_with_an_incapable_encoder_is_blocked():
    p = wf.resolve({"chroma_mode": "444", "video_encoder": "libsvtav1"},
                   source_pix_fmt="yuv444p10le")
    fmts = wf.ffmpeg_encoder_formats("libsvtav1")
    if not fmts:
        pytest.skip("本机没有 libsvtav1")
    assert p["ok"] is False
    kinds = [b["kind"] for b in p["blockers"]]
    assert "encoder_unsupported" in kinds
    # 必须给出可行替代，而不是只说"不行"
    sug = [b.get("suggest") for b in p["blockers"] if b.get("suggest")]
    assert sug and sug[0], p["blockers"]


def test_444_with_a_capable_encoder_passes():
    if "yuv444p10le" not in wf.ffmpeg_encoder_formats("libx265"):
        pytest.skip("本机没有支持 444 的 libx265")
    p = wf.resolve({"chroma_mode": "444", "video_encoder": "libx265"})
    assert p["ok"] is True and p["blockers"] == []


def test_accepted_degradation_produces_a_concrete_plan():
    """接受降级 ≠ 只把 ok 翻成 True（R23-B09）。

    审核："即使传真正的 True，也只是把 ok 改成 True，resolved/boundaries
    仍要求不支持的444，没有生成实际可执行的降级方案。"
    现在必须给出**具体的交付格式替换**，并且计划里保留原因。
    """
    p = wf.resolve({"chroma_mode": "444", "pix_fmt": "yuv444p10le",
                    "video_encoder": "libsvtav1",
                    "chroma_mode_accept_degraded": True},
                   source_pix_fmt="yuv444p10le")
    if not wf.ffmpeg_encoder_formats("libsvtav1"):
        pytest.skip("本机没有 libsvtav1")
    assert p["ok"] is True
    assert p["accepted_degraded"] is True
    assert p["degraded"], "接受降级必须给出具体替代（不是只改 ok）"
    assert p["degraded"]["from"] == "yuv444p10le"
    assert p["degraded"]["to"] == p["delivery"]["pix_fmt"]
    assert p["degraded"]["to"] in wf.ffmpeg_encoder_formats("libsvtav1")


def test_accept_degraded_as_the_string_false_is_not_accepted():
    """`"false"` 被 `bool()` 读成 True —— 用户写"不接受"，程序读成"接受"。"""
    p = wf.resolve({"chroma_mode": "444", "pix_fmt": "yuv444p10le",
                    "video_encoder": "libsvtav1",
                    "chroma_mode_accept_degraded": "false"},
                   source_pix_fmt="yuv444p10le")
    if not wf.ffmpeg_encoder_formats("libsvtav1"):
        pytest.skip("本机没有 libsvtav1")
    assert p["accepted_degraded"] is False
    assert p["ok"] is False


def test_444_working_with_420_delivery_is_allowed():
    """**444 工作 + 420 交付是合法组合**（R23-B09）。

    审核："`chroma_mode=444`、最终 `pix_fmt=yuv420p10le` 被判冲突；
    处理中保持 444、最后才 420 交付是合理且应保留的组合。"
    计划要把它写成"末尾一次明确下采样"，并保留损失说明。
    """
    p = wf.resolve({"chroma_mode": "444", "pix_fmt": "yuv420p10le",
                    "video_encoder": "libx265"})
    assert p["ok"] is True, p["blockers"]
    assert p["working"]["chroma"] == "444"
    assert p["delivery"]["pix_fmt"] == "yuv420p10le"
    assert p["conversion"]["from"] == "444"
    assert p["conversion"]["to"] == "420"
    assert "down-sample" in p["conversion"]["operation"]
    # 工作表示仍然全程 444（stage1–stage6），只有交付是 420
    assert p["boundaries"]["stage1_render"] == "yuv444p16le"
    assert p["boundaries"]["stage6_out"] == "yuv444p10le"
    assert p["boundaries"]["stage7_final"] == "yuv420p10le"


def test_invalid_mode_is_rejected_instead_of_silently_auto():
    p = wf.resolve({"chroma_mode": "bogus", "video_encoder": "libx265"})
    assert p["ok"] is False
    assert any(b["kind"] == "invalid_mode" for b in p["blockers"])
    assert p["mode_error"]


def test_preserve_distinguishes_known_unsupported_from_unknown():
    """4:2:2 是**已知**格式，不能说成"源未知"（审核 B09）。"""
    p422 = wf.resolve({"chroma_mode": "preserve", "video_encoder": "libx265"},
                      source_pix_fmt="yuv422p10le")
    kinds422 = [b["kind"] for b in p422["blockers"]]
    assert "source_chroma_unsupported" in kinds422, kinds422
    assert "source_chroma_unknown" not in kinds422
    # 给出"最接近、不再降低色度分辨率"的候选
    assert p422["working"]["chroma"] == "444"

    p_unknown = wf.resolve({"chroma_mode": "preserve",
                            "video_encoder": "libx265"})
    assert any(b["kind"] == "source_chroma_unknown"
               for b in p_unknown["blockers"])


def test_rgb_source_preserves_without_subsampling():
    p = wf.resolve({"chroma_mode": "preserve", "video_encoder": "libx265"},
                   source_pix_fmt="rgb48le")
    assert p["ok"] is True, p["blockers"]
    assert p["working"]["chroma"] == "444"


@pytest.mark.parametrize("fmt,expect", [
    ("yuv444p", "vs.YUV444P8"),
    ("yuv420p", "vs.YUV420P8"),
    ("yuv444p16le", "vs.YUV444P16"),
])
def test_vs_format_understands_8bit_planar(fmt, expect):
    """8-bit 平面格式没有数字位深后缀，旧正则因此认不出（B09 分支反例）。"""
    assert wf.vs_format_of(fmt) == expect


# ------------------------------------------------ 边界表

@pytest.mark.parametrize("mode,chroma", [("420", "420"), ("444", "444")])
def test_every_boundary_is_declared(mode, chroma):
    p = wf.resolve({"chroma_mode": mode, "video_encoder": "libx265"})
    b = p["boundaries"]
    for name in ("stage1_render", "stage2_step2", "stage3_step3",
                 "stage3_fused_in", "stage4_mask", "stage4_out",
                 "stage5_out", "stage6_out", "stage7_work", "stage7_final"):
        assert name in b, f"边界 {name} 没有声明格式"
    # 除了 stage7_work（抖动固定在高色度空间做），其余都要跟模式一致
    for name, fmt in b.items():
        if name == "stage7_work":
            assert "444" in fmt
            continue
        assert chroma in fmt, f"{name}={fmt} 与 chroma_mode={mode} 不一致"


def test_boundary_lookup_goes_through_the_plan():
    params = {"chroma_mode": "444", "video_encoder": "libx265"}
    assert wf.boundary_pix_fmt(params, "stage3_step3") == "yuv444p16le"
    assert wf.boundary_pix_fmt(params, "no-such-boundary",
                               default="x") == "x"


def test_frozen_plan_wins_over_recompute():
    """计划一旦冻结，运行期改 params 不得改变边界格式。"""
    params = {"chroma_mode": "444", "video_encoder": "libx265"}
    frozen = wf.resolve(params)
    params["_work_format"] = frozen
    params["chroma_mode"] = "420"          # 运行期被改（模拟注入）
    assert wf.boundary_pix_fmt(params, "stage1_render") == "yuv444p16le"


def test_param_keys_are_identity_bearing():
    """切换表示必须改变计划身份 —— 键不能是下划线内部键。"""
    keys = wf.param_keys()
    assert "chroma_mode" in keys
    for k in keys:
        assert not k.startswith("_"), f"{k} 以下划线开头，不会进身份摘要"
