# -*- coding: utf-8 -*-
"""P3：S6→S7 执行组的数据面（交接单 + 组回执）。

## 这一层要证明什么

融合路线把 stage6 的 lossless 中间文件省掉了，于是"stage6 到底跑没跑"
**没有任何物证**。整套机制只有两个文件：

1. **交接单** `{job}_{stem}_step67_group.json` —— stage6 写、stage7 读，
   说明"用这个 vpy、它的输入是这一份、摘要是这个"；
2. **组回执** `<成品>.fused_stage67.json` —— stage7 写、编排器读，
   说明"这一代确实按组跑的、产物就是这些字节"。

本文件把这两份东西的**判定**钉住，尤其是所有"过期/被换过/没查过"
的路径：这些路径上如果返回 True，融合就变成一条可以伪造的近路。

## 与 `fused_receipt` 的关系

组回执的字段与 `fused_stage34.json` 对齐，身份/产物摘要那部分**直接复用**
`fused_receipt.verify_fused_receipt`（唯一实现，不会漂移）。只有两处是
组特有的：vpy 摘要一致、`intermediate_written is False`。
"""
import json
import os

import pytest

import conftest  # noqa: F401

import stage67_fusion as S


# --------------------------------------------------------------------------
# 夹具
# --------------------------------------------------------------------------
def _write(path, data=b"x"):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(str(path), "wb") as fh:
        fh.write(data)
    return str(path)


def _vpy(tmp_path, text="clip.set_output()\n"):
    return _write(tmp_path / "grain.vpy", text.encode("utf-8"))


def _spec(tmp_path, *, attempt="att1", vs=None, src=None):
    vs = vs or _vpy(tmp_path)
    src = src or _write(tmp_path / "step5_face.mkv", b"SRC" * 32)
    return S.build_spec(
        attempt_id=attempt, chain_mode="stage67", vs_script=vs,
        probe_source=src,
        grain={"g1str": 1.0, "g2str": 0.2, "g3str": 0.2, "temp_avg": 0.0},
        vs_format="vs.YUV420P16", pix_fmt="yuv420p16le",
        deterministic=True)


# --------------------------------------------------------------------------
# 1) 交接单
# --------------------------------------------------------------------------
def test_spec_path_is_deterministic_and_absolute(tmp_path):
    a = S.spec_path(str(tmp_path), "job1", "clip")
    b = S.spec_path(str(tmp_path), "job1", "clip")
    assert a == b and os.path.isabs(a)
    assert a.endswith("job1_clip_step67_group.json")
    # 换个 stem 必须换文件（同一 job 多素材不能互相覆盖）
    assert S.spec_path(str(tmp_path), "job1", "clip2") != a


def test_spec_roundtrip_and_validation(tmp_path):
    spec = _spec(tmp_path)
    p = str(tmp_path / "spec.json")
    S.write_spec(p, spec)
    back = S.load_spec(p)
    assert back == spec
    ok, why = S.validate_spec(back, expected_attempt="att1")
    assert ok, why
    assert back["vs_script_sha256"] == S.sha256_file(back["vs_script"])


def test_spec_rejects_wrong_attempt(tmp_path):
    """上一轮的交接单不能给本轮用 —— 否则会用旧颗粒参数跑新素材。"""
    spec = _spec(tmp_path, attempt="attOLD")
    ok, why = S.validate_spec(spec, expected_attempt="attNEW")
    assert not ok and "attOLD" in why, why


def test_spec_rejects_edited_script(tmp_path):
    """vpy 被改过 → 交接单过期。摘要必须真读文件算，不看 mtime。"""
    spec = _spec(tmp_path)
    with open(spec["vs_script"], "a", encoding="utf-8") as fh:
        fh.write("# someone changed the filter chain\n")
    ok, why = S.validate_spec(spec, expected_attempt="att1")
    assert not ok and "changed" in why, why


def test_spec_rejects_missing_script_or_source(tmp_path):
    spec = _spec(tmp_path)
    os.remove(spec["vs_script"])
    ok, why = S.validate_spec(spec, expected_attempt="att1")
    assert not ok and "gone" in why, why

    spec2 = _spec(tmp_path)
    os.remove(spec2["probe_source"])
    ok, why = S.validate_spec(spec2, expected_attempt="att1")
    assert not ok and "probe source" in why, why


@pytest.mark.parametrize("mutate,needle", [
    (lambda s: s.update(schema="other/1"), "schema"),
    (lambda s: s.update(group="stage34"), "group"),
    (lambda s: s.update(stages=[4, 5]), "stages"),
    (lambda s: s.update(vs_script_sha256=""), "vs_script_sha256"),
])
def test_spec_rejects_structural_tampering(tmp_path, mutate, needle):
    spec = _spec(tmp_path)
    mutate(spec)
    ok, why = S.validate_spec(spec, expected_attempt="att1")
    assert not ok and needle in why, why


# --------------------------------------------------------------------------
# 2) 组回执
# --------------------------------------------------------------------------
def _receipt(tmp_path, *, spec=None, params=None, bounds_ok=True,
             bounds_performed=True, color_probed=True, src_probed=True,
             lost=False, intermediate=False):
    spec = spec or _spec(tmp_path)
    out = _write(tmp_path / "final.mkv", b"OUT" * 64)
    params = params or {"attempt_id": "att1", "chain_mode": "stage67",
                        "job_id": "job1"}
    r = S.build_receipt(spec=spec, params=params, output_path=out, frames=42,
                        extra={
                            "color": {"probed": color_probed,
                                      "pix_fmt": "yuv420p10le"},
                            # 「丢标签」必须由**字段**体现：验证器按
                            # `derive_color_relations()` 从源/输出重新推导，
                            # 不看回执自报的列表（R22-A05 的教训：自报的
                            # 结论不是证据）。
                            "color_source": (
                                {"probed": src_probed,
                                 "pix_fmt": "yuv420p16le",
                                 "matrix": "bt709"} if lost
                                else {"probed": src_probed,
                                      "pix_fmt": "yuv420p16le"}),
                            "color_lost_tags": ["matrix"] if lost else [],
                            "bounds": {"ok": bounds_ok,
                                       "performed": bounds_performed},
                        })
    r["group_evidence"]["intermediate_written"] = intermediate
    p = S.receipt_path(out)
    S.write_receipt(p, r)
    return out, p, params, spec


def test_receipt_happy_path(tmp_path):
    out, p, params, spec = _receipt(tmp_path)
    ok, why, rcpt = S.verify_group_receipt(
        p, params, expected_attempt="att1", src_path=spec["probe_source"])
    assert ok, why
    assert rcpt["group"] == "stage67"
    assert rcpt["stages"] == [6, 7]
    assert rcpt["output"] == out
    assert rcpt["output_sha256"] == S.sha256_file(out)


def test_receipt_rejects_wrong_attempt(tmp_path):
    _out, p, params, spec = _receipt(tmp_path)
    ok, why, _ = S.verify_group_receipt(
        p, params, expected_attempt="attNEW", src_path=spec["probe_source"])
    assert not ok and "attempt" in why, why


def test_receipt_rejects_replaced_output(tmp_path):
    """同大小替换也必须被发现 —— 靠内容摘要，不是靠大小。"""
    out, p, params, spec = _receipt(tmp_path)
    n = os.path.getsize(out)
    with open(out, "wb") as fh:
        fh.write(b"Z" * n)
    ok, why, _ = S.verify_group_receipt(
        p, params, expected_attempt="att1", src_path=spec["probe_source"])
    assert not ok and ("bytes changed" in why or "digest" in why), why


def test_receipt_rejects_changed_vs_script(tmp_path):
    """vpy 被换掉 → 回执描述的已经不是这一代跑的东西。"""
    _out, p, params, spec = _receipt(tmp_path)
    with open(spec["vs_script"], "a", encoding="utf-8") as fh:
        fh.write("# tampered\n")
    ok, why, _ = S.verify_group_receipt(
        p, params, expected_attempt="att1", src_path=spec["probe_source"])
    assert not ok and "vs script" in why, why


def test_receipt_rejects_when_it_admits_an_intermediate(tmp_path):
    """回执自己说"写了中间文件" → 这一代根本不是融合。"""
    _out, p, params, spec = _receipt(tmp_path, intermediate=True)
    ok, why, _ = S.verify_group_receipt(
        p, params, expected_attempt="att1", src_path=spec["probe_source"])
    assert not ok and "intermediate" in why, why


def test_receipt_rejects_unprobed_colour(tmp_path):
    """「没查过」不是「查过了没问题」。"""
    _out, p, params, spec = _receipt(tmp_path, color_probed=False)
    ok, why, _ = S.verify_group_receipt(
        p, params, expected_attempt="att1", src_path=spec["probe_source"])
    assert not ok and "never probed" in why, why

    _out, p, params, spec = _receipt(tmp_path, src_probed=False)
    ok, why, _ = S.verify_group_receipt(
        p, params, expected_attempt="att1", src_path=spec["probe_source"])
    assert not ok and "never probed" in why, why


def test_receipt_rejects_lost_colour_tags(tmp_path):
    """组内把色彩标签丢了 = 组级缺陷（转换发生在组内不算"丢"）。"""
    _out, p, params, spec = _receipt(tmp_path, lost=True)
    ok, why, _ = S.verify_group_receipt(
        p, params, expected_attempt="att1", src_path=spec["probe_source"])
    assert not ok and "lost" in why, why


@pytest.mark.parametrize("performed,okv,needle", [
    (False, True, "never performed"),
    (True, False, "does not conserve"),
])
def test_receipt_rejects_unproven_or_broken_bounds(tmp_path, performed, okv,
                                                   needle):
    _out, p, params, spec = _receipt(tmp_path, bounds_performed=performed,
                                     bounds_ok=okv)
    ok, why, _ = S.verify_group_receipt(
        p, params, expected_attempt="att1", src_path=spec["probe_source"])
    assert not ok and needle in why, why


# --------------------------------------------------------------------------
# 3) 定位：回执跟着**产物**走，不跟着 temp 走
# --------------------------------------------------------------------------
def test_find_group_receipt_reads_next_to_the_output(tmp_path):
    out, p, params, spec = _receipt(tmp_path)
    rcpt, path, why = S.find_group_receipt(
        out, params, expected_attempt="att1", src_path=spec["probe_source"])
    assert rcpt is not None, why
    assert path == p


def test_find_group_receipt_reports_absence_explicitly(tmp_path):
    """缺失必须说"没有"，不能返回空 dict 让调用方以为有。"""
    out = _write(tmp_path / "lonely.mkv", b"OUT")
    rcpt, path, why = S.find_group_receipt(
        out, {"attempt_id": "att1"}, expected_attempt="att1")
    assert rcpt is None
    assert "no stage67 group receipt" in why, why


def test_build_receipt_records_source_only_digest(tmp_path):
    """源身份必须是 source-only（跨 stage 稳定），不能掺 params。"""
    spec = _spec(tmp_path)
    params = {"attempt_id": "att1", "crf": 99, "preset": 1}
    out = _write(tmp_path / "final.mkv", b"OUT" * 8)
    r = S.build_receipt(spec=spec, params=params, output_path=out, frames=1)
    assert r["input_digest_kind"] == "source-only"
    from fused_receipt import source_digest
    assert r["input_digest"] == source_digest(spec["probe_source"])
    assert json.dumps(r)  # 必须可序列化（会被写进 temp 供下一轮读）


# --------------------------------------------------------------------------
# 4) 组边界的守恒判据必须用**视频流**口径
# --------------------------------------------------------------------------
def _load_stage7():
    import sys
    vstages = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "video_restoration_pipeline", "stages")
    if vstages not in sys.path:
        sys.path.insert(0, vstages)
    import stage7_encode as s7                      # noqa: E402
    return s7


def test_group_bounds_uses_video_stream_semantics():
    """音轨比视频长时，"容器时长变了"**不能**判成视频被拉伸。

    这是真实跑批抓到的（2026-11-14）：40 帧 25fps 的中间产物 1.600s，
    成品带 1.642s 的 AAC → 容器时长 1.642s，`check_conservation` 报
    "duration changed 1.600s -> 1.642s (2.6%)" 并把一次完全正确的融合判成
    未完成。视频一帧都没动。
    """
    s7 = _load_stage7()
    pre = {
        "source": {"frames": 40, "duration": 1.600},
        "output": {"frames": 40, "duration": 1.642},
        "checks": {
            # verify_output 用 video_duration 比，所以这里是"通过"
            "frames": {"src": 40, "out": 40, "allowed": [40], "ok": True},
            "duration": {"src": 1.600, "out": 1.600, "ok": True},
        },
    }
    b = s7._group_bounds(pre)
    assert b["ok"] is True, b
    assert b["performed"] is True
    assert "video-stream" in b["basis"], b["basis"]


def test_group_bounds_fails_on_frame_loss():
    s7 = _load_stage7()
    pre = {"source": {"frames": 40}, "output": {"frames": 39},
           "checks": {"frames": {"src": 40, "out": 39, "ok": False,
                                 "allowed": [40]},
                      "duration": {"ok": True}}}
    b = s7._group_bounds(pre)
    assert b["ok"] is False
    assert "frames" in b["reason"], b


def test_group_bounds_unperformed_is_not_a_pass():
    """两侧都量不到 → 未验证，不是通过。"""
    s7 = _load_stage7()
    b = s7._group_bounds({"checks": {"frames": {"ok": None},
                                     "duration": {"ok": None}}})
    assert b["performed"] is False
    assert b["ok"] is False
    assert b["reason"]


# --------------------------------------------------------------------------
# 5) resume 时（没有 pin）必须靠身份侧车锚定，不能拿原始素材顶替
# --------------------------------------------------------------------------
def test_unanchored_source_requires_identity_sidecar(tmp_path, monkeypatch):
    """resume 到 6/7 时 pin 不存在；此时**只能**用身份侧车定基准。

    反例（必须被拒）：回执里的 input 是别人/上一代的中间产物 —— 侧车
    对不上就不给基准，最终判"未证明"，而不是拿用户原始素材去凑一个
    "通过"（原始素材本来就不是组读的东西）。
    """
    import main as vmain

    out = _write(tmp_path / "final.mkv", b"OUT" * 8)
    src = _write(tmp_path / "step2.mkv", b"MID" * 8)
    rc = {"group": "stage67", "stages": [6, 7], "input_path": src}
    S.write_receipt(S.receipt_path(out), rc)

    monkeypatch.setattr(vmain, "_s67_intermediate_provenance",
                        lambda *a, **k: (True, ""))
    assert vmain._s67_unanchored_source(out, {}, None) == src

    monkeypatch.setattr(vmain, "_s67_intermediate_provenance",
                        lambda *a, **k: (False, "identity sidecar missing"))
    assert vmain._s67_unanchored_source(out, {}, None) == ""


def test_unanchored_source_rejects_missing_input(tmp_path):
    import main as vmain
    out = _write(tmp_path / "final.mkv", b"OUT" * 8)
    S.write_receipt(S.receipt_path(out),
                    {"group": "stage67", "input_path": str(tmp_path / "gone.mkv")})
    assert vmain._s67_unanchored_source(out, {}, None) == ""


def test_unanchored_source_rejects_unreadable_receipt(tmp_path):
    import main as vmain
    out = _write(tmp_path / "final.mkv", b"OUT" * 8)
    assert vmain._s67_unanchored_source(out, {}, None) == ""


# --------------------------------------------------------------------------
# 6) 产物出处判据：只认 job / 字节 / 源素材，**不**把 chain_mode 算进去
# --------------------------------------------------------------------------
def _sidecar(tmp_path, *, job_id="jobX", tamper_digest=False,
             other_source=False):
    import main as vmain
    src = _write(tmp_path / "in" / "src.mov", b"SRC" * 32)
    other = _write(tmp_path / "in" / "other.mov", b"OTHER" * 32)
    mid = _write(tmp_path / "step2.mkv", b"MID" * 32)
    dig = vmain._artifact_digest(mid)
    side = {
        "job_id": job_id,
        "artifact_digest": ("0" * 64) if tamper_digest else dig,
        "source_identity": vmain.build_input_identity(
            other if other_source else src),
    }
    with open(vmain.intermediate_identity_path(mid), "w",
              encoding="utf-8") as fh:
        json.dump(side, fh)
    job = {"job_id": "jobX", "input_path": src}
    return mid, job


def test_provenance_accepts_own_intermediate(tmp_path):
    import main as vmain
    mid, job = _sidecar(tmp_path)
    ok, why = vmain._s67_intermediate_provenance(mid, job, None)
    assert ok, why


def test_provenance_rejects_tampered_bytes(tmp_path):
    import main as vmain
    mid, job = _sidecar(tmp_path, tamper_digest=True)
    ok, why = vmain._s67_intermediate_provenance(mid, job, None)
    assert not ok and "changed" in why, why


def test_provenance_rejects_other_job(tmp_path):
    import main as vmain
    mid, job = _sidecar(tmp_path, job_id="jobOTHER")
    ok, why = vmain._s67_intermediate_provenance(mid, job, None)
    assert not ok and "jobOTHER" in why, why


def test_provenance_rejects_other_source(tmp_path):
    import main as vmain
    mid, job = _sidecar(tmp_path, other_source=True)
    ok, why = vmain._s67_intermediate_provenance(mid, job, None)
    assert not ok and "different source" in why, why


def test_provenance_missing_sidecar_is_not_a_pass(tmp_path):
    import main as vmain
    mid = _write(tmp_path / "mid.mkv", b"MID")
    ok, why = vmain._s67_intermediate_provenance(mid, {"job_id": "j"}, None)
    assert not ok and "no identity sidecar" in why, why


def test_provenance_ignores_chain_mode(tmp_path):
    """`chain_mode` 只改**怎么执行**，不该让一份好的中间产物失效。

    反例（2026-11-14 真实续跑）：上一轮 `--chain stage67` 产出的 step2，
    这一轮 `--chain both` 续跑 —— 复用计划说"可复用"，而
    `intermediate_identity_matches()` 说"effective parameters changed"，
    两个判据互相矛盾。出处判据只看 job / 字节 / 源素材。
    """
    import main as vmain
    mid, job = _sidecar(tmp_path)
    assert vmain._s67_intermediate_provenance(mid, job, None)[0]


# --------------------------------------------------------------------------
# 7) 大 mkv 没有容器帧数时，守恒判定必须用**实测的两个数**兜底
# --------------------------------------------------------------------------
def test_group_bounds_falls_back_to_counted_frames():
    """`verify_output` 读不到帧数时不能直接判"未验证"。

    长片实测（2026-11-14，8160 帧 4.4 GB mkv）：容器里没有 `nb_frames`，
    `frames.ok=None` → 组回执判"未验证" → 一次完全正确的 48 分钟跑批
    被判失败并触发整轮重跑。而这两个数其实都测过：
    源在编码前 `-count_frames`、成品由 `measure_final_candidate()` 实测。
    """
    s7 = _load_stage7()
    pre = {"checks": {"frames": {"ok": None, "note": "无法推算允许帧数，未校验"},
                      "duration": {"ok": None}}}
    b = s7._group_bounds(pre, src_frames=8160, out_frames=8160)
    assert b["performed"] is True, b
    assert b["ok"] is True, b
    assert "counted" in b["basis"], b["basis"]


def test_group_bounds_counted_fallback_detects_frame_loss():
    s7 = _load_stage7()
    pre = {"checks": {"frames": {"ok": None}, "duration": {"ok": None}}}
    b = s7._group_bounds(pre, src_frames=8160, out_frames=8159)
    assert b["performed"] is True
    assert b["ok"] is False, b


def test_group_bounds_without_measured_output_stays_unproven():
    """只给了源帧数、没给成品帧数 → 仍然是"未验证"，不能默认相等。"""
    s7 = _load_stage7()
    pre = {"checks": {"frames": {"ok": None}, "duration": {"ok": None}}}
    b = s7._group_bounds(pre, src_frames=8160, out_frames=None)
    assert b["performed"] is False
    assert b["ok"] is False


def test_group_bounds_never_compares_a_number_with_itself():
    """回归：`out_frames` 若被写成带兜底的 `frames`，比较会恒真。

    这个坑很隐蔽 —— 传错一个参数，"源=8160、成品=8160"看着完全正常，
    但实际比的是同一个变量。所以这里显式钉住：只给 src、不给 out 时，
    结论必须是"未证明"，而不是"通过"。
    """
    s7 = _load_stage7()
    pre = {"checks": {"frames": {"ok": None}, "duration": {"ok": None}}}
    assert s7._group_bounds(pre, src_frames=1, out_frames=None)["ok"] is False
    assert s7._group_bounds(pre, src_frames=1, out_frames=2)["ok"] is False
