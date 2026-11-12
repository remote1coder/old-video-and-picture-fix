# -*- coding: utf-8 -*-
"""R22 §5.1：帧边界的**两组对照实验**。

## 为什么要这两组

审核把"融合与经典是否等价"拆成三组，并明确它们**不能互相替代**：

| 组 | 固定什么 | 回答什么 |
|---|---|---|
| ① 不含 AI 的边界夹具 | 同一已知帧序列、时间、颜色、后处理参数 | 接线、次序、格式转换、异常处理 |
| ② 同一次 AI 输出 | AI 只跑一次，冻结其输出，两条后半链消费相同字节 | 经典物化 vs 融合串流的差异 |
| ③ 两条独立 AI 全链 | 各自从同一源跑 | 全链重复性（**不能**替代 ①②）|

③ 已有（`tests/probes/g34_equiv_det.py`，R22 重跑仍 IDENTICAL）。
本文件做 ① 和 ②。

## 边界怎么"固定"

`stage3._stream_raw_upscale(..., frame_source_dir=...)` 是内部注入点：
给了它就用一个**接口完全一致**的替身脚本，按顺序吐出预置的 rgb48le 帧，
绕过模型。经典路径（不带 `post_filter_chain`）与融合路径（带）消费同一串字节。

## 断言什么

- 两条路径的帧数、尺寸、像素格式一致
- **逐帧** PSNR y/u/v 全为 inf（不是"平均 inf"）
- 帧序一致（本文件用**逐帧不同**的帧，错位会立刻体现为有限值）
"""
import json
import os
import subprocess
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT,
           os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import frame_contract as fc  # noqa: E402
import stage3_ai_upscale as s3  # noqa: E402
import stage4_postprocess as s4  # noqa: E402

_MODELS = os.path.join(_ROOT, "models")

#: 输入尺寸 / 放大倍数 —— 小到几秒跑完，又能覆盖 split=5 的掩模分支
IN_W, IN_H, SCALE = 320, 240, 2


def _ffmpeg(*args, timeout=120):
    return subprocess.run(["ffmpeg", *args], capture_output=True, text=True,
                          errors="replace", timeout=timeout, shell=False)


def _make_source(tmp_path, frames=5, rate=10):
    """造一个 5 帧的小视频（给解码器用；内容不重要，帧数要对）。"""
    src = str(tmp_path / "src.mkv")
    r = _ffmpeg("-v", "error", "-y", "-f", "lavfi",
                "-i", f"testsrc2=size={IN_W}x{IN_H}:rate={rate}"
                      f":duration={frames / rate}",
                "-c:v", "ffv1", "-pix_fmt", "yuv420p", src)
    assert r.returncode == 0, (r.stderr or "")[-400:]
    return src


def _write_raw_frames(target_dir, frames, *, pattern="distinct"):
    """写 rgb48le 原始帧。

    ``pattern="distinct"``：每帧颜色不同 + 各帧内部有纹理（用行列渐变），
    这样"帧序错了"和"掩模判错了"都会立刻反映成有限 PSNR。
    """
    import numpy as np
    os.makedirs(target_dir, exist_ok=True)
    for f in os.listdir(target_dir):
        os.remove(os.path.join(target_dir, f))
    h, w = IN_H * SCALE, IN_W * SCALE
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(frames):
        if pattern == "flat":
            arr = np.zeros((h, w, 3), dtype="<u2")
            arr[:, :, 0] = (i * 13000) % 65536
            arr[:, :, 1] = (i * 7000) % 65536
            arr[:, :, 2] = (i * 3000) % 65536
        else:
            # 每帧一个主色 + 行列渐变（制造边缘/纹理，让掩模真的有东西可判）
            base = [(60000, 8000, 8000), (8000, 60000, 8000),
                    (8000, 8000, 60000), (45000, 45000, 8000),
                    (8000, 45000, 45000)][i % 5]
            arr = np.zeros((h, w, 3), dtype="<u2")
            ramp = ((xx * 65535) // max(1, w - 1)).astype("<u2")
            for c in range(3):
                arr[:, :, c] = (base[c] // 2 + ramp // 2).astype("<u2")
        arr.tofile(os.path.join(target_dir, f"frame_{i:03d}.raw"))
    return sorted(os.path.join(target_dir, n)
                  for n in os.listdir(target_dir))


def _run_classic(tmp_path, src, frame_dir):
    """经典：流式出 step3（**不带**后处理链），再单独跑 stage4 的滤镜。"""
    st3 = str(tmp_path / "step3.mkv")
    ok, why, _ = s3._stream_raw_upscale(
        input_video=src, output_mkv=st3, width=IN_W, height=IN_H,
        scale=SCALE, fps_str="10", model="unused", model_dir=str(tmp_path),
        ai_py=sys.executable, frame_source_dir=frame_dir, inactivity_s=180)
    assert ok, f"classic step3 failed: {why}"

    chain = s4.build_filter_chain({})
    st4 = str(tmp_path / "classic_step4.mkv")
    r = _ffmpeg("-v", "error", "-y", "-i", st3,
                "-filter_complex", chain, "-map", "[vout]",
                "-c:v", "ffv1", "-pix_fmt", "yuv420p16le", st4)
    assert r.returncode == 0, (r.stderr or "")[-600:]
    return st4


def _run_fused(tmp_path, src, frame_dir):
    """融合：一次流式跑完，滤镜链直接接在编码器上。"""
    fu = str(tmp_path / "fused_step4.mkv")
    ok, why, _ = s3._stream_raw_upscale(
        input_video=src, output_mkv=fu, width=IN_W, height=IN_H,
        scale=SCALE, fps_str="10", model="unused", model_dir=str(tmp_path),
        ai_py=sys.executable, frame_source_dir=frame_dir, inactivity_s=180,
        post_filter_chain=s4.build_filter_chain({}),
        out_pix_fmt="yuv420p16le")
    assert ok, f"fused step4 failed: {why}"
    return fu


def _psnr_rows(a, b):
    """逐帧 PSNR 行（stats_file=- 的 `n:` 行）。"""
    r = _ffmpeg("-v", "info", "-i", a, "-i", b,
                "-lavfi", "[0:v][1:v]psnr=stats_file=-", "-f", "null", "-")
    return [ln for ln in (r.stdout or "").splitlines() if ln.startswith("n:")]


def _assert_identical(a, b, label=""):
    rows = _psnr_rows(a, b)
    assert rows, f"{label}: 没有比较结果（命令失败？）"
    bad = [ln for ln in rows
           if "psnr_y:inf" not in ln or "psnr_u:inf" not in ln
           or "psnr_v:inf" not in ln]
    assert not bad, (
        f"{label}: {len(bad)}/{len(rows)} 帧不是逐像素相同\n"
        f"第一处：{bad[0][:200]}")
    return len(rows)


# ------------------------------------------------ 实验 ①：不含 AI 的边界夹具

@pytest.mark.parametrize("pattern", ["distinct", "flat"])
def test_classic_and_fused_consume_the_same_fixed_frames(tmp_path, pattern):
    """① 同一已知帧序列 → 两条后半链逐像素相同。

    `distinct` 用带纹理的渐变帧（让掩模真的起作用）；
    `flat` 用纯色帧（掩模退化的边界）。
    """
    src = _make_source(tmp_path)
    frame_dir = str(tmp_path / "frames")
    _write_raw_frames(frame_dir, 5, pattern=pattern)

    classic = _run_classic(tmp_path, src, frame_dir)
    fused = _run_fused(tmp_path, src, frame_dir)
    n = _assert_identical(classic, fused, f"exp1/{pattern}")
    assert n == 5, f"应比较 5 帧，实际 {n}"


def test_frame_order_matters_for_the_comparison(tmp_path):
    """反例：把冻结帧**顺序打乱**，两条链就不再相同。

    这条用来证明上面的 `inf` 不是"比较器没在比"：
    帧序一变，PSNR 立刻掉下来。
    """
    src = _make_source(tmp_path)
    fa = str(tmp_path / "fa")
    fb = str(tmp_path / "fb")
    order = _write_raw_frames(fa, 5, pattern="distinct")
    _write_raw_frames(fb, 5, pattern="distinct")
    # 把 fb 的第 2、3 帧对调
    import shutil
    a2, a3 = order[1], order[2]
    tmp_swap = str(tmp_path / "swap.raw")
    shutil.copyfile(a2, tmp_swap)
    shutil.copyfile(a3, a2)
    shutil.copyfile(tmp_swap, a3)

    ca = _run_classic(tmp_path / "A", src, fa) if False else None
    # 直接比较两串帧本身（不必跑完整管线）：顺序不同 → 逐帧 PSNR 有限
    rows_a = _psnr_rows(order[0], a2)
    rows_b = _psnr_rows(order[0], a2)
    assert rows_a or True      # 占位：真正的判据在下面
    # 用"两串帧目录"各跑一条链，逐帧比
    os.makedirs(str(tmp_path / "A"), exist_ok=True)
    os.makedirs(str(tmp_path / "B"), exist_ok=True)
    ca = _run_classic(tmp_path / "A", src, fa)
    cb = _run_classic(tmp_path / "B", src, fb)
    rows = _psnr_rows(ca, cb)
    assert rows, "没有比较结果"
    finite = [ln for ln in rows
              if "psnr_y:inf" not in ln or "psnr_u:inf" not in ln
              or "psnr_v:inf" not in ln]
    assert finite, "帧序被打乱后仍逐像素相同 —— 比较器根本没在比"


def test_fixed_frames_with_wrong_byte_size_fail_loudly(tmp_path):
    """异常处理：预置帧字节数不对时，必须**大声失败**而不是截断。"""
    src = _make_source(tmp_path)
    bad_dir = str(tmp_path / "bad")
    _write_raw_frames(bad_dir, 5, pattern="distinct")
    # 把第 3 帧截短一半
    p = os.path.join(bad_dir, "frame_002.raw")
    with open(p, "rb") as f:
        data = f.read()
    with open(p, "wb") as f:
        f.write(data[: len(data) // 2])

    out = str(tmp_path / "out.mkv")
    ok, why, _ = s3._stream_raw_upscale(
        input_video=src, output_mkv=out, width=IN_W, height=IN_H,
        scale=SCALE, fps_str="10", model="unused", model_dir=str(tmp_path),
        ai_py=sys.executable, frame_source_dir=bad_dir, inactivity_s=60)
    assert ok is False, "字节数不对却报成功"
    assert "expected" in str(why) or "bytes" in str(why), why


def test_frozen_set_shorter_than_the_input_fails_loudly(tmp_path):
    """预置帧比输入少 → 必须失败，不能静默截断。"""
    src = _make_source(tmp_path, frames=5)
    short = str(tmp_path / "short")
    _write_raw_frames(short, 3, pattern="distinct")
    out = str(tmp_path / "out.mkv")
    ok, why, _ = s3._stream_raw_upscale(
        input_video=src, output_mkv=out, width=IN_W, height=IN_H,
        scale=SCALE, fps_str="10", model="unused", model_dir=str(tmp_path),
        ai_py=sys.executable, frame_source_dir=short, inactivity_s=60)
    assert ok is False, "帧数不足却报成功"
    assert "frozen" in str(why) or "more frames" in str(why), why


# ------------------------------------------------ 合同落盘（生产路径）

def _run_classic_meta(tmp_path, src, frame_dir, **kw):
    st3 = str(tmp_path / "step3_meta.mkv")
    meta = {}
    ok, why, n = s3._stream_raw_upscale(
        input_video=src, output_mkv=st3, width=IN_W, height=IN_H,
        scale=SCALE, fps_str="10", model="unused", model_dir=str(tmp_path),
        ai_py=sys.executable, frame_source_dir=frame_dir, inactivity_s=180,
        meta=meta, **kw)
    return ok, why, n, meta, st3


def test_production_path_writes_a_provable_frame_contract(tmp_path):
    """合同必须是**生产路径自己写的**，而且帧摘要真的算得出来。

    这条针对"S2 剩余：合同还没接进 stage3 生产落盘"。以前帧摘要只能
    在测试里用 `digest_ordered_frames()` 自己造 —— 生产不落盘，等于
    两份合同无从比较。现在 AI 子进程对自己写出的字节算同一个算法。
    """
    src = _make_source(tmp_path)
    fz = str(tmp_path / "frames")
    _write_raw_frames(fz, 5, pattern="distinct")

    ok, why, n, meta, st3 = _run_classic_meta(tmp_path, src, fz)
    assert ok, why
    # 帧数必须以**编码器落盘**为准（曾经固定帧源下恒为 0）
    assert n == 5, f"返回帧数应为 5，实际 {n}"
    assert meta["ai_frames"] == 5 and meta["enc_frames"] == 5, meta["frames"]

    sidecar = st3 + ".frame_contract.json"
    assert os.path.exists(sidecar), "生产路径没有写帧边界合同"
    c = fc.read_contract(sidecar)
    assert c is not None
    # R23-B08：完整性的判据变严了 —— 时间/颜色/实现指纹也要有声明。
    # 这里的生产调用没传 `timeline_ref`（直接调 `_stream_raw_upscale`），
    # 所以时间引用缺内容身份是**如实记录**的缺口，不是缺陷。
    _probs = fc.contract_problems(c)
    assert not [x for x in _probs if "frames_digest" in x
                or "pixel format" in x], _probs
    assert any("timeline" in x for x in _probs), (
        "缺时间引用时必须报出来，而不是静默当作完整：" + str(_probs))
    assert c["frames"] == 5
    assert c["pix_fmt"] == "yuv420p16le", c["pix_fmt"]
    assert c["bit_depth"] == 16
    assert c["frames_digest_proven"] is True
    assert c["frames_digest_domain"] == "raw-rgb48le"
    assert c["source"] == "fixed-frames"
    assert c["gaps"] == [], c["gaps"]
    # 摘要必须与"按同样顺序读磁盘帧"算出来的一致 —— 否则两条链不可比
    expect = fc.digest_ordered_frames(
        sorted(os.path.join(fz, x) for x in os.listdir(fz)))
    assert c["frames_digest"] == expect, (c["frames_digest"], expect)


def test_both_halves_contracts_prove_the_same_frame_set(tmp_path):
    """经典物化与融合串流的合同必须**逐字段相同**（帧集+格式）。

    这比"PSNR 全 inf"更强：PSNR 只说明输出相同，合同说明**输入边界**
    也相同 —— 审核要的正是"两条链消费同一份帧源"。
    """
    src = _make_source(tmp_path)
    fz = str(tmp_path / "frames")
    _write_raw_frames(fz, 5, pattern="distinct")

    ok, why, _n, _m, st3 = _run_classic_meta(tmp_path, src, fz)
    assert ok, why
    fu = str(tmp_path / "fused.mkv")
    ok, why, _ = s3._stream_raw_upscale(
        input_video=src, output_mkv=fu, width=IN_W, height=IN_H,
        scale=SCALE, fps_str="10", model="unused", model_dir=str(tmp_path),
        ai_py=sys.executable, frame_source_dir=fz, inactivity_s=180,
        post_filter_chain=s4.build_filter_chain({}),
        out_pix_fmt="yuv420p16le")
    assert ok, why

    ca = fc.read_contract(st3 + ".frame_contract.json")
    cb = fc.read_contract(fu + ".frame_contract.json")
    assert ca and cb, (ca, cb)
    assert fc.compare_contracts(ca, cb) == [], fc.compare_contracts(ca, cb)


def test_requested_pix_fmt_is_not_silently_overridden(tmp_path):
    """`out_pix_fmt` 曾经被后面那句固定的 `-pix_fmt yuv420p16le` 覆盖。

    ffmpeg 的输出选项是"后者覆盖前者"，所以传 10-bit 进去、出来还是
    16-bit，而且**没有任何报错**。合同现在按**探测到的成品**写格式，
    这条用例同时守住"参数真生效"和"合同不说谎"。
    """
    src = _make_source(tmp_path)
    fz = str(tmp_path / "frames")
    _write_raw_frames(fz, 5, pattern="distinct")
    out = str(tmp_path / "ten_bit.mkv")
    meta = {}
    ok, why, _ = s3._stream_raw_upscale(
        input_video=src, output_mkv=out, width=IN_W, height=IN_H,
        scale=SCALE, fps_str="10", model="unused", model_dir=str(tmp_path),
        ai_py=sys.executable, frame_source_dir=fz, inactivity_s=180,
        post_filter_chain=s4.build_filter_chain({}),
        out_pix_fmt="yuv420p10le", meta=meta)
    assert ok, why
    probed = (meta.get("contract") or {}).get("pix_fmt")
    assert probed == "yuv420p10le", (
        f"请求 10-bit，实际落盘 {probed}（参数被后面的 -pix_fmt 覆盖了）")
    assert meta["contract"]["bit_depth"] == 10


def test_missing_digest_falls_back_to_unproven_not_to_a_fake(tmp_path,
                                                             monkeypatch):
    """拿不到摘要时：`frames_digest_proven=False` + gaps 写清原因。

    绝不允许"摘要没有就编一个顺序号"—— 那是本项目反复踩的坑。
    """
    import stream_ai_template as sat

    src = _make_source(tmp_path)
    fz = str(tmp_path / "frames")
    _write_raw_frames(fz, 5, pattern="distinct")

    real = sat.build_fixed_frame_script

    def _no_digest(frame_dir, in_w, in_h, out_w, out_h):
        # 模拟"老版本脚本只报 FRAMES、不报摘要"
        txt = real(frame_dir, in_w, in_h, out_w, out_h)
        return "\n".join(ln for ln in txt.splitlines()
                         if "FRAMES_DIGEST" not in ln
                         and "_h.update(b'|frames='" not in ln) + "\n"

    monkeypatch.setattr(sat, "build_fixed_frame_script", _no_digest)
    out = str(tmp_path / "nodigest.mkv")
    meta = {}
    ok, why, n = s3._stream_raw_upscale(
        input_video=src, output_mkv=out, width=IN_W, height=IN_H,
        scale=SCALE, fps_str="10", model="unused", model_dir=str(tmp_path),
        ai_py=sys.executable, frame_source_dir=fz, inactivity_s=180,
        meta=meta)
    assert ok, why
    assert n == 5
    c = meta["contract"]
    assert c["frames_digest"] is None
    assert c["frames_digest_proven"] is False
    assert any("FRAMES_DIGEST" in g for g in c["gaps"]), c["gaps"]
    assert fc.contract_problems(c), "缺摘要的合同不该被判为完整"


# ------------------------------------------------ 实验 ②：同一次 AI 输出

def _can_run_ai():
    if not os.path.isdir(_MODELS):
        return False
    if not os.path.exists(os.path.join(_MODELS, "RealESRGAN_x2plus.pth")):
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@pytest.mark.skipif(not _can_run_ai(),
                    reason="需要 RealESRGAN_x2plus 权重 + CUDA")
def test_same_ai_output_feeds_both_back_halves(tmp_path):
    """② AI **只跑一次**，冻结其输出，两条后半链消费同一串字节。

    与 ① 的区别：这里帧是**真实模型**产的（16-bit），
    所以也覆盖了"模型输出位深/通道序"这一环。
    """
    import cv2

    src = _make_source(tmp_path)
    ind = str(tmp_path / "ai_in")
    outd = str(tmp_path / "ai_out")
    os.makedirs(ind, exist_ok=True)
    os.makedirs(outd, exist_ok=True)

    # 16-bit 抽帧（PNG 用 rgb48be）
    r = _ffmpeg("-v", "error", "-y", "-i", src,
                "-pix_fmt", "rgb48be", os.path.join(ind, "f_%03d.png"))
    assert r.returncode == 0, (r.stderr or "")[-400:]

    # AI 跑**一次**（一次调用 = 一次模型加载 + 5 帧）
    res = s3.run_realesrgan(ind, outd, "RealESRGAN_x2plus", SCALE, 0)
    assert res is True, f"AI failed: {res}"
    names = sorted(n for n in os.listdir(outd) if n.endswith(".png"))
    assert len(names) == 5, names

    # 冻结成 rgb48le raw
    fz = str(tmp_path / "frozen")
    os.makedirs(fz, exist_ok=True)
    for i, n in enumerate(names):
        arr = cv2.imread(os.path.join(outd, n), cv2.IMREAD_UNCHANGED)
        assert arr is not None and arr.dtype.name == "uint16", (
            f"{n}: 期望 uint16，实际 {None if arr is None else arr.dtype}")
        arr[:, :, ::-1].copy().tofile(os.path.join(fz, f"frame_{i:03d}.raw"))

    # 合同：证明两路吃的是同一串帧
    paths = sorted(os.path.join(fz, n) for n in os.listdir(fz))
    digest = fc.digest_ordered_frames(paths)
    contract = fc.build_contract(
        frames=len(paths), width=IN_W * SCALE, height=IN_H * SCALE,
        pix_fmt="rgb48le", frames_digest=digest, source="model",
        # R23-B08：合同要能证明"这一串帧在什么时候显示、什么颜色、谁产的"
        timeline={"plan_path": str(tmp_path / "timeline.json"),
                  "fps": "10"},
        color={"probed": True, "pix_fmt": "rgb48le"},
        ai={"impl": "realesrgan", "model": "RealESRGAN_x2plus"})
    assert fc.contract_problems(contract) == [], fc.contract_problems(contract)

    classic = _run_classic(tmp_path, src, fz)
    fused = _run_fused(tmp_path, src, fz)
    n = _assert_identical(classic, fused, "exp2/ai")
    assert n == 5, f"应比较 5 帧，实际 {n}"


def test_ai_rejects_an_empty_input_directory(tmp_path):
    """AI 接口：空输入目录必须失败（曾经返回 True 但零产出）。"""
    empty = str(tmp_path / "empty")
    outd = str(tmp_path / "out")
    os.makedirs(empty, exist_ok=True)
    os.makedirs(outd, exist_ok=True)
    # 不需要模型：脚本在加载模型**之前**就检查输入了？—— 实际在之后，
    # 所以这条用例只在有权重时跑。
    if not _can_run_ai():
        pytest.skip("需要 RealESRGAN_x2plus 权重 + CUDA")
    res = s3.run_realesrgan(empty, outd, "RealESRGAN_x2plus", SCALE, 0)
    assert res is False, f"空输入目录返回了 {res!r}"
