# -*- coding: utf-8 -*-
"""共享内存帧传输的**结果收集侧**（_drain_ring_to_pngs）。

为什么单独测这一侧
-----------------
2026-10-25 的教训：3c 先接了生产侧（建 ring + 设环境变量）却没接收集侧，
而 auto 模式会选共享内存 → AI 写进 ring 没人读 → 下游拿不到帧 → 断链。
所以本测试**先锁住收集侧**：给定一个已被"假 AI"写过的 ring，
必须能完整落回 PNG 且内容一致。

这里用一个"假 AI"：不加载任何模型，只往 ring 里写已知数据，
因此测试是毫秒级、不占 GPU、可反复跑。
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "shared_modules"))
sys.path.insert(0, os.path.join(_ROOT, "video_restoration_pipeline", "stages"))

import frame_transport as ft                      # noqa: E402
import stage3_ai_upscale as s3                    # noqa: E402


@pytest.fixture()
def ring_env(tmp_path):
    """造一个输入 ring（装 N 帧）和一个输出 ring（假 AI 会往里写）。"""
    W, H, C, DT, N = 8, 6, 3, "uint16", 5
    fb = W * H * C * np.dtype(DT).itemsize
    slots = 6                      # 必须 >= N，见 test_ring_requires_slots_ge_frames
    rin = ft.FrameRing(name="wb_t_in", frame_bytes=fb, slots=slots)
    rout = ft.FrameRing(name="wb_t_out", frame_bytes=fb, slots=slots)
    frames = []
    for i in range(N):
        a = (np.arange(W * H * C, dtype=np.uint16) + i * 7) % 60000
        a = a.reshape(H, W, C)
        frames.append(a.copy())
        rin.slot(i)[:] = np.frombuffer(np.ascontiguousarray(a).tobytes(),
                                   dtype=np.uint8)
    yield {"rin": rin, "rout": rout, "frames": frames, "W": W, "H": H,
           "C": C, "DT": DT, "N": N, "fb": fb, "slots": slots,
           "out_dir": str(tmp_path / "out")}
    rin.close()
    rout.close()


def _fake_ai(env, scale=1):
    """假 AI：从输入 ring 读、乘个系数、写进输出 ring（模拟 enhance）。"""
    N, slots, fb = env["N"], env["slots"], env["fb"]
    W, H, C, DT = env["W"], env["H"], env["C"], env["DT"]
    again_in = ft.FrameRing.attach(env["rin"].name, fb, slots)
    out = ft.FrameRing.attach(env["rout"].name, fb, slots)
    try:
        for i in range(N):
            off = (i % slots) * fb
            raw = bytes(again_in.buf_shm().buf[off:off + fb]) \
                if hasattr(again_in, "buf_shm") else \
                bytes(again_in._shm.buf[off:off + fb])
            a = np.frombuffer(raw, dtype=np.dtype(DT)).reshape(H, W, C)
            out.slot(i)[:] = np.frombuffer(
                np.ascontiguousarray((a * 2) % 60000).tobytes(),
                dtype=np.uint8)
    finally:
        again_in._safe_close()
        out._safe_close()


def test_drain_writes_all_frames(ring_env):
    _fake_ai(ring_env)
    n = s3._drain_ring_to_pngs(
        ring_env["rout"].name, ring_env["slots"], ring_env["fb"],
        ring_env["N"], ring_env["out_dir"],
        ring_env["W"], ring_env["H"], ring_env["C"], ring_env["DT"])
    assert n == ring_env["N"], "收集侧没有落全所有帧"
    pngs = sorted(f for f in os.listdir(ring_env["out_dir"])
                  if f.endswith(".png"))
    assert len(pngs) == ring_env["N"]


def test_drained_frames_match_what_ai_wrote(ring_env):
    """核心：落盘的 PNG 必须与"假 AI 写进 ring 的内容"一致。"""
    import cv2
    _fake_ai(ring_env)
    s3._drain_ring_to_pngs(
        ring_env["rout"].name, ring_env["slots"], ring_env["fb"],
        ring_env["N"], ring_env["out_dir"],
        ring_env["W"], ring_env["H"], ring_env["C"], ring_env["DT"])
    for i, orig in enumerate(ring_env["frames"]):
        got = cv2.imread(os.path.join(ring_env["out_dir"],
                                      "shm_%06d.png" % i),
                         cv2.IMREAD_UNCHANGED)
        assert got is not None, f"frame {i} 没落盘"
        assert np.array_equal(got, (orig * 2) % 60000), \
            f"frame {i} 内容与 AI 写入的不一致"


def test_drain_when_slots_exceed_frames(ring_env):
    """slots > N 时所有帧都占不同槽位，必须全部正确读出。"""
    assert ring_env["slots"] > ring_env["N"], "fixture 应满足 slots > N"
    _fake_ai(ring_env)
    n = s3._drain_ring_to_pngs(
        ring_env["rout"].name, ring_env["slots"], ring_env["fb"],
        ring_env["N"], ring_env["out_dir"],
        ring_env["W"], ring_env["H"], ring_env["C"], ring_env["DT"])
    assert n == ring_env["N"]


def test_drain_bad_shm_name_returns_zero(tmp_path):
    """挂不上 ring 必须返回 0（让调用方知道失败），不能静默过半。"""
    n = s3._drain_ring_to_pngs("wb_no_such_ring_xyz", 4, 8 * 6 * 3 * 2,
                               2, str(tmp_path / "o"), 8, 6, 3, "uint16")
    assert n == 0


def test_load_then_drain_roundtrip(tmp_path):
    """载入侧 → 假 AI → 收集侧，整条 ring 链路自洽。"""
    import cv2
    W, H, C, DT = 8, 6, 3, "uint16"
    fb = W * H * C * np.dtype(DT).itemsize
    src = tmp_path / "src"
    src.mkdir()
    want = []
    for i in range(3):
        a = ((np.arange(W * H * C, dtype=np.uint16) + i * 11) % 50000)
        a = a.reshape(H, W, C)
        cv2.imwrite(str(src / ("f%03d.png" % i)), a)
        want.append(a)
    ring = ft.FrameRing(name="wb_t_rt", frame_bytes=fb, slots=3)
    try:
        n, w, h, c, dt = s3._load_window_into_ring(str(src), ring, fb, 3)
        assert (n, w, h, c, dt) == (3, W, H, C, DT)
        # 假 AI：原样搬过去
        out = ft.FrameRing(name="wb_t_rt_o", frame_bytes=fb, slots=3)
        try:
            for i in range(n):
                out.slot(i)[:] = np.frombuffer(bytes(ring.slot(i)[:]), dtype=np.uint8)
            m = s3._drain_ring_to_pngs(out.name, 3, fb, n,
                                       str(tmp_path / "o"), W, H, C, DT)
            assert m == 3
        finally:
            out.close()
        for i, a in enumerate(want):
            got = cv2.imread(str(tmp_path / "o" / ("shm_%06d.png" % i)),
                             cv2.IMREAD_UNCHANGED)
            assert np.array_equal(got, a), f"roundtrip frame {i} 不一致"
    finally:
        ring.close()


def test_ring_requires_slots_ge_frames():
    """设计约束：ring 槽数必须 >= 批次帧数，否则生产者会覆盖未读走的帧。

    2026-10-25 实测：slots=4、N=5 时，第 5 帧写进 slot 0 覆盖第 1 帧，
    收集侧读到 [4,1,2,3,4] —— 丢一帧且重复一帧。
    所以真正的接线必须**把窗口按 buffer_frames 分块**，
    每块帧数 <= slots。本测试把这个约束钉住，防止后人误用。
    """
    W, H, C, DT = 8, 6, 3, "uint16"
    fb = W * H * C * np.dtype(DT).itemsize
    ring = ft.FrameRing(name="wb_t_over", frame_bytes=fb, slots=2)
    try:
        # 写 3 帧进 2 槽 → 第 3 帧覆盖第 1 帧
        for i in range(3):
            ring.slot(i)[:] = np.frombuffer(
                np.full(fb, i + 1, dtype=np.uint8).tobytes(), dtype=np.uint8)
        got = [int(bytes(ring.slot(i)[:1])[0]) for i in range(3)]
        assert got == [3, 2, 3], f"覆盖行为变了：{got}"
    finally:
        ring.close()


def test_shm_env_set_implies_drain_called():
    """结构性不变量：设了 WB_FRAME_OUT_SHM_NAME 就必须调用 drain。

    2026-10-25 的断链事故正是漏了这条：生产侧设了环境变量，AI 把结果
    写进输出 ring，却没有代码去读 → 下游拿不到帧。
    本测试用源码静态检查把这个配对钉住。
    """
    import io
    src = io.open(s3.__file__, encoding="utf-8", errors="replace").read()
    i_set = src.find('env["WB_FRAME_OUT_SHM_NAME"]')
    i_drain = src.find("_drain_ring_to_pngs(")
    assert i_set > 0, "找不到设置 WB_FRAME_OUT_SHM_NAME 的代码"
    assert i_drain > 0, "找不到调用 _drain_ring_to_pngs 的代码"
    # drain 的调用点必须在设置点之后（收集侧在生产侧之后）
    assert i_drain > i_set, "drain 调用点在设置环境变量之前？顺序反了"
    # drain 必须受"环境变量存在"保护
    guard = src[i_drain - 400:i_drain]
    assert "WB_FRAME_OUT_SHM_NAME" in guard, \
        "drain 调用没有检查 WB_FRAME_OUT_SHM_NAME 是否存在"


def test_shm_failure_clears_env_vars():
    """结构性不变量：shm 建立失败时必须清掉所有 WB_FRAME_* 再回退 PNG。

    否则残留的环境变量会让 AI 子进程尝试挂一个不存在的 ring。
    """
    import io
    src = io.open(s3.__file__, encoding="utf-8", errors="replace").read()
    assert 'if _k.startswith("WB_FRAME_")' in src, \
        "缺少清理 WB_FRAME_* 的代码（失败回退会残留环境变量）"
    assert "STAGE3_FRAME_RING_FALLBACK" in src, "缺少回退日志"
