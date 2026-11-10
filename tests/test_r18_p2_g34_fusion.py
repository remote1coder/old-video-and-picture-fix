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

#: 与各测试传给 `_fused_step34` 的 params 保持一致。
#: `compute_param_hash` 只用输入路径 + 质量参数，temp_dir 不参与摘要。
_P = {"attempt_id": _ATT}


def _write_receipt(temp_dir, job, size=None, jid=None, attempt=_ATT,
                   params=None):
    stem = os.path.splitext(os.path.basename(str(job["input_path"])))[0]
    name = f"{jid or job['job_id']}_{stem}_step4_postprocess_pabc.mkv"
    out = os.path.join(str(temp_dir), name)
    with open(out, "wb") as f:
        f.write(b"X" * (size if size is not None else 100))
    rcpt = os.path.join(str(temp_dir), name + ".fused_stage34.json")
    with open(rcpt, "w", encoding="utf-8") as f:
        # R20-A03：回执必须带 input_digest（= compute_param_hash(源, params)），
        # 消费方用它证明"这份回执属于本次源与参数"。
        _dig = ""
        try:
            from checkpoint import compute_param_hash as _cph
            _dig = _cph(str(job.get("input_path") or ""), params or {})
        except Exception:
            pass
        json.dump({"attempt_id": attempt,
                   "input_digest": _dig,
                   "chain_mode": "stage34", "output": out,
                   "output_size": os.path.getsize(out),
                   "logical_stages": ["stage3", "stage4"],
                   # R20-A03：回执必须带**证据块**。没有它们，
                   # 消费方无法区分"这次转换是对的"和"没人检查过"。
                   "color": {"pix_fmt": "yuv420p16le", "range": "limited",
                             "matrix": "bt709", "primaries": "bt709",
                             "transfer": "bt709", "probed": True,
                             "complete": True, "unknown_fields": []},
                   "bounds": {"ok": True, "performed": True, "reason": "",
                              "report": {"stage": "stage34"}},
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
    out, _ = _write_receipt(tmp_path, job, params=_P)
    got = vmain._fused_step34({"temp_dir": str(tmp_path), "attempt_id": _ATT}, job)
    assert got == os.path.abspath(out) or os.path.normcase(got) == \
        os.path.normcase(out), got


def test_fused_step34_returns_none_without_receipt(tmp_path):
    job = _job(tmp_path / "clip.mp4")
    assert vmain._fused_step34({"temp_dir": str(tmp_path), "attempt_id": _ATT}, job) is None


def test_fused_step34_rejects_a_receipt_whose_output_is_gone(tmp_path):
    """**关键**：回执在、产物没了 → 必须走经典 stage4，不能跳过。"""
    job = _job(tmp_path / "clip.mp4")
    out, _ = _write_receipt(tmp_path, job, params=_P)
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
    out, _ = _write_receipt(tmp_path, job, size=100, params=_P)
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
    _write_receipt(tmp_path, job, jid="j2", params=_P)
    assert vmain._fused_step34({"temp_dir": str(tmp_path), "attempt_id": _ATT}, job) is None


# ------------------------------------------------ 证据块（R20-A03）

class _L:
    def __init__(self):
        self.ev = []

    def warning(self, e, *a, **k):
        self.ev.append(e)

    info = error = warning


def _mutated_receipt(tmp_path, job, mutate):
    out, rcpt = _write_receipt(tmp_path, job, params=_P)
    with open(rcpt, encoding="utf-8") as f:
        doc = json.load(f)
    mutate(doc)
    with open(rcpt, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return out, rcpt


def test_fused_step34_rejects_a_receipt_without_color_evidence(tmp_path):
    """只有 pix_fmt 一个字段的时代结束了：颜色必须**成组**可查。"""
    job = _job(tmp_path / "clip.mp4")
    _mutated_receipt(tmp_path, job, lambda d: d.pop("color", None))
    log = _L()
    assert vmain._fused_step34(
        {"temp_dir": str(tmp_path), "attempt_id": _ATT}, job, log) is None
    # "过期"是"这不是本次的"，"未证明"是"是本次的，但没人证明它对"。
    assert "STAGE4_FUSED_RECEIPT_UNPROVEN" in log.ev, log.ev


def test_fused_step34_rejects_a_receipt_with_incomplete_color(tmp_path):
    """字段未知 ≠ 颜色错了，但同样不足以支撑"这次转换是对的"。"""
    job = _job(tmp_path / "clip.mp4")

    def _drop_matrix(d):
        d["color"]["matrix"] = None
        d["color"]["complete"] = False
        d["color"]["unknown_fields"] = ["matrix"]

    _mutated_receipt(tmp_path, job, _drop_matrix)
    assert vmain._fused_step34(
        {"temp_dir": str(tmp_path), "attempt_id": _ATT}, job, _L()) is None


def test_fused_step34_rejects_a_receipt_whose_bounds_were_never_checked(tmp_path):
    """**没查过 ≠ 查过了没问题**：performed=False 必须当"未证明"。"""
    job = _job(tmp_path / "clip.mp4")

    def _skip(d):
        d["bounds"] = {"ok": False, "performed": False,
                       "reason": "helpers unavailable"}

    _mutated_receipt(tmp_path, job, _skip)
    assert vmain._fused_step34(
        {"temp_dir": str(tmp_path), "attempt_id": _ATT}, job, _L()) is None


def test_fused_step34_rejects_a_receipt_whose_bounds_failed(tmp_path):
    job = _job(tmp_path / "clip.mp4")

    def _fail(d):
        d["bounds"] = {"ok": False, "performed": True,
                       "reason": "r_frame_rate changed 24 -> 25"}

    _mutated_receipt(tmp_path, job, _fail)
    assert vmain._fused_step34(
        {"temp_dir": str(tmp_path), "attempt_id": _ATT}, job, _L()) is None


def test_stage3_fused_path_runs_the_boundary_check():
    """结构守卫：融合路径必须在**写回执之前**跑经典 stage4 那道边界门。"""
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage3_ai_upscale.py"),
               encoding="utf-8", errors="replace").read()
    i_check = src.find("from generation_evidence import collect")
    i_rcpt = src.find('".fused_stage34.json"')
    assert 0 < i_check < i_rcpt, (i_check, i_rcpt)
    assert "STAGE3_FUSE34_BOUNDS_BROKEN" in src


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
