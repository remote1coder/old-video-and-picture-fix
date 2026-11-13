# -*- coding: utf-8 -*-
"""R22 §5.1：帧边界合同（纯函数层）。

合同的作用：证明"经典物化"与"融合串流"吃进去的是**同一串帧**。
所以它必须能回答三个问题：

1. 帧数/尺寸/格式是什么（会直接决定像素）
2. 帧内容按**顺序**是什么（换顺序 = 不同的一串，即使帧数尺寸全同）
3. 这份摘要是不是**真算过**（流式管道拿不到就写"未证明"，不拿替代物冒充）

第 3 条是审核反复强调的："不能仅给一个 FPS 就丢掉 VFR 信息"、
"没有可核对的内容"不等于"核对通过"。
"""
import json
import os
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import frame_contract as fc  # noqa: E402


def _w(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


# ------------------------------------------------ 帧序身份

def test_frame_order_is_part_of_the_digest(tmp_path):
    """同样的帧换个顺序 = 不同的一串（这正是"错位"的典型形态）。"""
    a = _w(str(tmp_path / "1.raw"), b"AAAA")
    b = _w(str(tmp_path / "2.raw"), b"BBBB")
    d_ab = fc.digest_ordered_frames([a, b])
    d_ba = fc.digest_ordered_frames([b, a])
    assert d_ab and d_ba and d_ab != d_ba


def test_frame_count_is_bound_into_the_digest(tmp_path):
    """两帧 `A`+`B` 与一帧 `AB` 不能撞同一个摘要。"""
    a = _w(str(tmp_path / "1.raw"), b"AA")
    b = _w(str(tmp_path / "2.raw"), b"BB")
    one = _w(str(tmp_path / "one.raw"), b"AABB")
    assert fc.digest_ordered_frames([a, b]) != fc.digest_ordered_frames([one])


def test_unreadable_frame_makes_the_digest_unproven(tmp_path):
    """任取一帧读不出来就返回 None —— 不跳过、不补零。"""
    a = _w(str(tmp_path / "1.raw"), b"AA")
    assert fc.digest_ordered_frames([a, str(tmp_path / "missing.raw")]) is None
    assert fc.digest_ordered_frames([]) is None


def test_same_frames_same_digest(tmp_path):
    a = _w(str(tmp_path / "1.raw"), b"AA")
    b = _w(str(tmp_path / "2.raw"), b"BB")
    assert fc.digest_ordered_frames([a, b]) == fc.digest_ordered_frames([a, b])


# ------------------------------------------------ 合同字段

def test_pix_fmt_maps_to_bit_depth_and_order():
    assert fc.pix_fmt_info("rgb48le") == (16, "rgb")
    assert fc.pix_fmt_info("gbrp16le") == (16, "rgb")
    assert fc.pix_fmt_info("yuv420p16le") == (16, None)
    # 未知格式：如实返回 None，不当成错误
    assert fc.pix_fmt_info("weird420") == (None, None)


def test_contract_carries_every_required_field():
    c = fc.build_contract(frames=5, width=640, height=480,
                          pix_fmt="yuv420p16le",
                          timeline={"path": "/t/plan.json", "digest": "d" * 64},
                          color={"range": "limited"},
                          ai={"model": "x4v3", "scale": 2},
                          frames_digest="a" * 64,
                          frames_digest_domain="raw-rgb48le",
                          source="model")
    for k in ("frames", "width", "height", "pix_fmt", "bit_depth",
              "channel_order", "frames_digest", "frames_digest_proven",
              "frames_digest_domain", "timeline", "color", "ai", "source"):
        assert k in c, k
    assert c["bit_depth"] == 16
    assert c["frames_digest_proven"] is True
    assert fc.contract_problems(c) == []


def test_unproven_digest_is_reported_not_hidden():
    """流式管道拿不到逐帧摘要 → 必须显式标注，且合同不能当证据。"""
    c = fc.build_contract(frames=5, width=640, height=480,
                          pix_fmt="rgb48le", source="stream")
    assert c["frames_digest_proven"] is False
    problems = fc.contract_problems(c)
    assert any("not proven" in p for p in problems), problems


def test_unknown_pix_fmt_is_reported():
    c = fc.build_contract(frames=1, width=2, height=2, pix_fmt="weird",
                          frames_digest="a" * 64)
    problems = fc.contract_problems(c)
    assert any("bit_depth is unknown" in p for p in problems), problems


@pytest.mark.parametrize("bad,fragment", [
    ({"frames": 0}, "frames must be positive"),
    ({"width": -1}, "width must be positive"),
    ({"pix_fmt": ""}, "pix_fmt is empty"),
])
def test_incomplete_contract_is_reported(bad, fragment):
    kw = dict(frames=5, width=640, height=480, pix_fmt="yuv420p16le",
              frames_digest="a" * 64)
    kw.update(bad)
    problems = fc.contract_problems(fc.build_contract(**kw))
    assert any(fragment in p for p in problems), problems


# ------------------------------------------------ 两份合同的比较

def test_same_contracts_compare_clean():
    a = fc.build_contract(frames=5, width=640, height=480,
                          pix_fmt="yuv420p16le", frames_digest="a" * 64,
                          frames_digest_domain="raw-rgb48le")
    assert fc.compare_contracts(a, dict(a)) == []


def test_frame_count_difference_is_reported():
    a = fc.build_contract(frames=5, width=640, height=480,
                          pix_fmt="yuv420p16le", frames_digest="a" * 64,
                          frames_digest_domain="raw-rgb48le")
    b = dict(a, frames=4)
    assert any("frames differs" in p for p in fc.compare_contracts(a, b))


def test_digest_difference_is_reported():
    a = fc.build_contract(frames=5, width=640, height=480,
                          pix_fmt="yuv420p16le", frames_digest="a" * 64,
                          frames_digest_domain="raw-rgb48le")
    b = dict(a, frames_digest="b" * 64)
    assert any("frames_digest differs" in p
               for p in fc.compare_contracts(a, b))


def test_format_difference_is_reported():
    """rgb48le vs gbrp16le：位深一样但**通道布局不同** —— 必须报。"""
    a = fc.build_contract(frames=5, width=640, height=480,
                          pix_fmt="rgb48le", frames_digest="a" * 64)
    b = fc.build_contract(frames=5, width=640, height=480,
                          pix_fmt="gbrp16le", frames_digest="a" * 64)
    problems = fc.compare_contracts(a, b)
    assert any("pix_fmt differs" in p for p in problems), problems


def test_unproven_digest_blocks_the_comparison():
    """有一侧未证明就**不能**声称"两路吃了同一串帧"。"""
    a = fc.build_contract(frames=5, width=640, height=480,
                          pix_fmt="rgb48le", frames_digest="a" * 64)
    b = fc.build_contract(frames=5, width=640, height=480,
                          pix_fmt="rgb48le")
    assert any("unproven" in p for p in fc.compare_contracts(a, b))


def test_different_digest_domains_are_not_a_frame_difference():
    """摘要域不同（raw 字节 vs PNG 字节）= "没法这样比"，不是"帧不一样"。

    分开报的原因：混成一条会让真正的错位被误读成"表示不同可以忽略"，
    反过来也会让"只是换了表示"被当成致命差异。
    """
    a = fc.build_contract(frames=5, width=640, height=480,
                          pix_fmt="rgb48le", frames_digest="a" * 64,
                          frames_digest_domain="raw-rgb48le")
    b = dict(a, frames_digest="b" * 64, frames_digest_domain="png-bytes")
    problems = fc.compare_contracts(a, b)
    assert any("domains differ" in p for p in problems), problems
    assert not any("frames_digest differs" in p for p in problems), problems


def test_same_domain_different_digest_is_a_real_difference():
    a = fc.build_contract(frames=5, width=640, height=480,
                          pix_fmt="rgb48le", frames_digest="a" * 64,
                          frames_digest_domain="raw-rgb48le")
    b = dict(a, frames_digest="b" * 64)
    problems = fc.compare_contracts(a, b)
    assert any("frames_digest differs" in p for p in problems), problems


# ------------------------------------------------ 侧车读写

def test_contract_sidecar_round_trip(tmp_path):
    p = str(tmp_path / "out" / "clip.frame_contract.json")
    c = fc.build_contract(frames=3, width=4, height=4, pix_fmt="rgb48le",
                          frames_digest="c" * 64, source="fixed-frames")
    assert fc.write_contract(p, c) is True
    back = fc.read_contract(p)
    assert back == json.loads(json.dumps(c))


def test_reading_a_corrupt_sidecar_returns_none(tmp_path):
    p = _w(str(tmp_path / "bad.json"), b"{not json")
    assert fc.read_contract(p) is None
    assert fc.read_contract(str(tmp_path / "missing.json")) is None
