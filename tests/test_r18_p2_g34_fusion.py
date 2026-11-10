# -*- coding: utf-8 -*-
"""第十八轮 P2：G34 融合执行组（stage3+stage4 一次跑完）。

## 设计要点

- **默认关闭**（`chain.mode: none`）。它只决定"怎么执行"，不决定"能用哪些功能"。
- 只有 **raw + CFR** 的流式路径参与融合；PNG / VFR 仍走经典两阶段
  —— 那是本轮未验证的范围，不是"技术上不可能"。
- stage3 直接把 stage4 的滤镜图接在编码器上，产物**直接叫 step4**，
  因此不产生 step3。下游（stage5/6）是按 glob `*_step4_postprocess*`
  找输入的，不依赖文件名里的 hash。
- 留下 `*.fused_stage34.json` 回执，main 据此**跳过** stage4；否则 stage4
  会去找不存在的 step3 → NO_INPUT，把一次成功的融合判成失败。
- 回执只在"产物存在且尺寸一致"时才被采信 —— 过期回执会导致跳过 stage4
  而缺少一道处理，那比多跑一次严重得多。

## 本轮未验证的部分（如实标注）

融合与经典的**逐帧像素等价**尚未实测。审核明确指出：
"FFV1 本身无损，不代表省掉文件边界后 FFmpeg 的自动格式协商与原流程完全一致"。
所以本轮只保证：开关存在、默认关闭、回退可用、回执可校验。
等价性要靠真实跑批对照（见交接文档的验收清单）。
"""
import json
import os
import sys

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import main as vmain  # noqa: E402


def _job(inp, jid="j1"):
    return {"job_id": jid, "input_path": str(inp)}


#: 测试统一使用的 attempt 标识（R20-A03 起回执必须带它）
_ATT = "att-test"


def _write_receipt(temp_dir, job, size=None, jid=None, attempt=_ATT):
    stem = os.path.splitext(os.path.basename(str(job["input_path"])))[0]
    name = f"{jid or job['job_id']}_{stem}_step4_postprocess_pabc.mkv"
    out = os.path.join(str(temp_dir), name)
    with open(out, "wb") as f:
        f.write(b"X" * (size if size is not None else 100))
    rcpt = os.path.join(str(temp_dir), name + ".fused_stage34.json")
    with open(rcpt, "w", encoding="utf-8") as f:
        json.dump({"attempt_id": attempt,
                   "chain_mode": "stage34", "output": out,
                   "output_size": os.path.getsize(out),
                   "logical_stages": ["stage3", "stage4"],
                   "frames": 10}, f)
    return out, rcpt


# ------------------------------------------------ 模式判定

def test_chain_mode_defaults_to_none():
    assert vmain._chain_mode_on({}) is False
    assert vmain._chain_mode_on({"chain_mode": "none"}) is False


def test_chain_mode_stage34_and_both_are_on():
    assert vmain._chain_mode_on({"chain_mode": "stage34"}) is True
    assert vmain._chain_mode_on({"chain_mode": "BOTH"}) is True


def test_chain_mode_stage67_alone_does_not_enable_stage34():
    """stage67 还没实现，不能顺带把 stage34 打开。"""
    assert vmain._chain_mode_on({"chain_mode": "stage67"}) is False


# ------------------------------------------------ 回执校验

def test_fused_step34_finds_a_valid_receipt(tmp_path):
    job = _job(tmp_path / "clip.mp4")
    out, _ = _write_receipt(tmp_path, job)
    got = vmain._fused_step34({"temp_dir": str(tmp_path), "attempt_id": _ATT}, job)
    assert got == os.path.abspath(out) or os.path.normcase(got) == \
        os.path.normcase(out), got


def test_fused_step34_returns_none_without_receipt(tmp_path):
    job = _job(tmp_path / "clip.mp4")
    assert vmain._fused_step34({"temp_dir": str(tmp_path), "attempt_id": _ATT}, job) is None


def test_fused_step34_rejects_a_receipt_whose_output_is_gone(tmp_path):
    """**关键**：回执在、产物没了 → 必须走经典 stage4，不能跳过。"""
    job = _job(tmp_path / "clip.mp4")
    out, _ = _write_receipt(tmp_path, job)
    os.unlink(out)

    class L:
        def __init__(self):
            self.ev = []

        def warning(self, e, *a, **k):
            self.ev.append(e)

        info = error = warning

    log = L()
    assert vmain._fused_step34({"temp_dir": str(tmp_path), "attempt_id": _ATT}, job, log) is None
    assert "STAGE4_FUSED_RECEIPT_STALE" in log.ev, log.ev


def test_fused_step34_rejects_a_receipt_whose_size_changed(tmp_path):
    """产物被改过（尺寸不符）→ 回执视为过期。"""
    job = _job(tmp_path / "clip.mp4")
    out, _ = _write_receipt(tmp_path, job, size=100)
    with open(out, "ab") as f:
        f.write(b"Y" * 50)

    class L:
        def __init__(self):
            self.ev = []

        def warning(self, e, *a, **k):
            self.ev.append(e)

        info = error = warning

    log = L()
    assert vmain._fused_step34({"temp_dir": str(tmp_path), "attempt_id": _ATT}, job, log) is None
    assert "STAGE4_FUSED_RECEIPT_SIZE_MISMATCH" in log.ev, log.ev


def test_fused_step34_ignores_a_receipt_for_another_job(tmp_path):
    """别的 job 的回执不能被当成本 job 的。"""
    job = _job(tmp_path / "clip.mp4", jid="j1")
    _write_receipt(tmp_path, job, jid="j2")
    assert vmain._fused_step34({"temp_dir": str(tmp_path), "attempt_id": _ATT}, job) is None


# ------------------------------------------------ 结构守卫

def test_g34_only_applies_to_the_stream_path():
    """融合只在流式路径生效；PNG 路径必须仍然产出 step3 走经典两阶段。"""
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage3_ai_upscale.py"),
               encoding="utf-8", errors="replace").read()
    i_stream = src.find('elif params.get("ai_stream_raw"):')
    i_fuse = src.find("STAGE3_FUSE34_ENABLED")
    assert i_stream > 0 and i_fuse > i_stream, \
        "G34 必须位于流式路径分支内（PNG/VFR 不得参与）"


def test_fusion_is_off_by_default_in_config():
    import io
    cfg = io.open(os.path.join(_ROOT, "video_restoration_pipeline",
                               "config.yaml"), encoding="utf-8").read()
    assert "chain:" in cfg
    assert "mode: none" in cfg, "默认必须是 none（行为与改动前一致）"


# ------------------------------------------------ 像素格式等价性（本轮定位的根因）

def test_fusion_chain_forces_unified_pix_fmt_before_filters():
    """结构守卫：融合进链前**必须**把输入统一成 `yuv420p16le`。

    这条走了两步（都被实测钉住）：

    **第一步**：经典 step3 原本是 `gbrp16le`（planar RGB 16），而融合的
    rawvideo 是 `rgb48le`（packed RGB 16）。滤镜图里
    `format=yuv420p` + `geq(cb/cr)` 的肤色掩模对格式敏感 ——
    swscale 的 packed→4:2:0 与 planar→4:2:0 不是同一条路径，
    色度下采样结果不同，掩模判错区域。实测：

    | 喂法 | Y | U | V |
    |---|---|---|---|
    | packed rgb48le 直接进链 | 43.19 | **12.97** | **12.37** |
    | 补 format=gbrp16le | inf | inf | inf |

    **第二步**：随后又把 step3 统一成 `yuv420p16le`（raw 路径原来不指定
    pix_fmt，PNG 路径显式写了它，两条路格式不同）。统一后省 52% 磁盘
    （909.6 MB → 434.7 MB），成品差异 66.00 dB。融合侧的进链格式
    随之改成 `yuv420p16le`，与经典路径进入 stage4 时保持一致。
    """
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage3_ai_upscale.py"),
               encoding="utf-8", errors="replace").read()
    i_block = src.find("if post_filter_chain:")
    assert i_block > 0, "找不到融合注入块"
    block = src[i_block:i_block + 3000]
    assert "format=yuv420p16le" in block, \
        "融合进链前必须补 format=yuv420p16le（否则色度会整片偏）"
    assert '_tag + "format=yuv420p16le,"' in block or \
        '"format=yuv420p16le," + _chain' in block, \
        "必须在 [0:v] 之后立刻统一格式，不能插到链中间"


def test_output_pix_fmt_written_by_both_paths():
    """结构守卫：raw 编码器必须显式写 `-pix_fmt yuv420p16le`。

    不写的话 ffv1 会把 rgb48le 存成 gbrp16le，与 PNG 路径分叉。
    """
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage3_ai_upscale.py"),
               encoding="utf-8", errors="replace").read()
    assert '"-pix_fmt", "yuv420p16le", "-c:v", "ffv1"' in src, \
        "raw 路径的编码器必须显式指定 yuv420p16le，否则会存成 gbrp16le"


def test_probe_script_exists_and_documents_the_numbers():
    """根因探针必须留在仓库里，且注释里带着实测数字（可复跑可核对）。"""
    probe = os.path.join(_ROOT, "tests", "probes",
                         "g34_pixfmt_probe.py")
    assert os.path.isfile(probe), "根因探针脚本必须留在仓库中"
    txt = open(probe, encoding="utf-8", errors="replace").read()
    for token in ("gbrp16le", "rgb48le", "ROOT_CAUSE_CONFIRMED"):
        assert token in txt, f"探针里缺少 {token}"
