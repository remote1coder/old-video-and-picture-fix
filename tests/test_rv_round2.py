"""第二轮复核 RV-01 … RV-14 的反例回归测试。

设计原则（来自复核报告 §5）：
1. 每条用例对应一个**已复现的反例**；修复前必须失败，修复后转绿。
2. 每条用例自带最小输入，不依赖其它用例的产物（消除顺序依赖）。
3. 能用真工具就用真工具；用替身的地方在 docstring 里写明"这是替身"。
4. 缺失必需工具时 skip 并记为"未验收"，不静默算通过。
"""

import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tests", "_out", "tools")
SHARED = os.path.join(ROOT, "shared_modules")
STAGES = os.path.join(ROOT, "photo_restoration_pipeline", "stages")

import media_io  # noqa: E402
import manifest_loader as ml  # noqa: E402
import stage6_encode as s6  # noqa: E402
import main as vmain  # noqa: E402  (video_restoration_pipeline/main.py)


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


class _Log:
    """Minimal logger stub capturing events."""

    def __init__(self):
        self.events = []

    def _rec(self, name, msg, **kw):
        self.events.append((name, msg))

    info = warning = error = debug = _rec


def _srgb_icc_bytes():
    from PIL import ImageCms
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _write_png_with_icc(path, arr, icc=None):
    from PIL import Image
    img = Image.fromarray(arr, mode="RGB")
    kwargs = {"format": "PNG"}
    if icc:
        kwargs["icc_profile"] = icc
    img.save(path, **kwargs)
    return str(path)


def _grad16(h=8, w=8, c=3):
    """16-bit 测试素材：**高字节与低字节必须不同**。

    踩过的坑：最初用 ``n * 257`` 生成，结果每个样本的高字节都等于低字节，
    这恰好就是"8-bit 数据乘 257"的形态 —— 8-bit 截断后再乘回去仍然逐像素相等，
    于是"是否真 16-bit"的断言变成了恒真。这里用两条不同的数列分别填高/低字节。
    """
    n = np.arange(h * w, dtype=np.uint32).reshape(h, w)
    hi = n % 256
    lo = (n * 37 + 11) % 256
    v = (hi * 256 + lo).astype(np.uint16)
    if c == 1:
        return v[:, :, None]
    return np.stack([v] * c, axis=2).astype(np.uint16)


# ---------------------------------------------------------------------------
# RV-01 · 视频正式入口 / logger API
# ---------------------------------------------------------------------------

def test_rv01_logger_accepts_level_and_actually_filters(tmp_path):
    """get_logger(level=...) 曾被当成非法实参 → 正式入口 rc=1。"""
    import logger as _logger

    assert "level" in __import__("inspect").signature(
        _logger.StructuredLogger.__init__).parameters, \
        "StructuredLogger 必须真正接受 level，而不是用 **kwargs 吞掉"

    lg = _logger.get_logger(log_dir=str(tmp_path / "logs"), level="INFO")
    lg.debug("DBG_EVENT", "must be filtered out")
    lg.info("INFO_EVENT", "must be kept")

    with open(lg.log_file, encoding="utf-8") as f:
        events = [json.loads(line)["event"] for line in f if line.strip()]
    assert "INFO_EVENT" in events
    assert "DBG_EVENT" not in events, "级别过滤没有真正生效"

    lg.set_level("DEBUG")
    lg.debug("DBG2", "kept after lowering the threshold")
    with open(lg.log_file, encoding="utf-8") as f:
        events = [json.loads(line)["event"] for line in f if line.strip()]
    assert "DBG2" in events


def test_rv01_real_cli_reaches_dependency_check(tmp_path):
    """真实 CLI：不能死在 logger 初始化（rc=1 TypeError）。"""
    import yaml
    cfg_path = os.path.join(ROOT, "video_restoration_pipeline", "config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("global", {})["log_dir"] = str(tmp_path / "logs")
    tmp_cfg = tmp_path / "video_config.yaml"
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)

    empty_in = tmp_path / "in"
    empty_in.mkdir()
    env = dict(os.environ)
    if os.path.isdir(TOOLS):
        env["PATH"] = TOOLS + os.pathsep + env.get("PATH", "")
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "video_restoration_pipeline",
                                      "main.py"),
         "--mode", "scan", "--input", str(empty_in),
         "--config", str(tmp_cfg),
         "--output-manifest", str(tmp_path / "m.yaml")],
        capture_output=True, text=True, shell=False, env=env, cwd=ROOT)
    # 依赖缺失时是 80；依赖齐全时是 0。**唯一不允许的是 1**（初始化 TypeError）
    assert r.returncode != 1, (
        f"CLI 在初始化阶段崩了（RV-01 未修复）: rc={r.returncode}\n"
        f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    log_file = tmp_path / "logs" / "general.log"
    assert log_file.exists(), "日志初始化失败：连 general.log 都没写出来"
    body = log_file.read_text(encoding="utf-8")
    assert "PIPELINE_START" in body


# ---------------------------------------------------------------------------
# RV-02 · 色彩决议
# ---------------------------------------------------------------------------

def test_rv02_auto_must_not_override_the_sidecar(tmp_path):
    """默认 auto 曾盖过 stage1 sidecar：写了 AdobeRGB 也被忽略。"""
    work = _write_png_with_icc(tmp_path / "work.png",
                               np.full((4, 4, 3), 128, np.uint8))
    (tmp_path / "j1_clip_color_meta.json").write_text(
        json.dumps({"input_color_space": "adobergb"}), encoding="utf-8")

    log = _Log()
    color = s6.resolve_color_space(
        work, {"input_color_space": "auto"}, str(tmp_path), "j1", "clip", log)
    assert color["source_space"] == "adobergb", \
        "auto 仍然盖过了 sidecar（RV-02 未修复）"
    # 宽色域 + 没有真实 ICC → 必须阻断，而不是当 sRGB 编下去
    assert color["status"] == s6.COLOR_FAILED


def test_rv02_auto_unknown_with_strict_fails(tmp_path):
    work = _write_png_with_icc(tmp_path / "work.png",
                               np.full((4, 4, 3), 128, np.uint8))
    log = _Log()
    color = s6.resolve_color_space(
        work, {"input_color_space": "auto",
               "require_known_color_space": True},
        str(tmp_path), "j2", "clip", log)
    assert color["status"] == s6.COLOR_FAILED, \
        "无来源 + strict=true 仍然被当成 not-needed（RV-02 未修复）"


def test_rv02_unknown_source_is_not_labelled_srgb(tmp_path):
    work = _write_png_with_icc(tmp_path / "work.png",
                               np.full((4, 4, 3), 128, np.uint8))
    log = _Log()
    color = s6.resolve_color_space(
        work, {"input_color_space": "auto", "archival_mode": True},
        str(tmp_path), "j3", "clip", log)
    assert color["status"] == s6.COLOR_UNKNOWN
    assert color["cicp_override"] == "2/2/0", \
        "未知来源仍在沿用配置里的 sRGB 标签"


def test_rv02_detected_srgb_overrides_conflicting_config_label(tmp_path):
    work = _write_png_with_icc(tmp_path / "work.png",
                               np.full((4, 4, 3), 128, np.uint8),
                               icc=_srgb_icc_bytes())
    log = _Log()
    color = s6.resolve_color_space(
        work, {"input_color_space": "auto", "archival_mode": True},
        str(tmp_path), "j4", "clip", log)
    assert color["source_space"] == "srgb"
    assert color["cicp_override"] == "1/13/0", "已确认 sRGB 必须覆盖旧标签"


def test_rv02_degraded_cicp_reaches_the_actual_command(tmp_path):
    """resolver 给的值必须进入命令行，而不是只写进 sidecar。"""
    work = _write_png_with_icc(tmp_path / "work.png",
                               np.full((4, 4, 3), 128, np.uint8))
    params = {"avif_rgb444_params": {"depth": 16, "yuv": "444",
                                     "cicp": "1/13/0", "lossless": True}}
    cmd, contract = media_io.build_avifenc_cmd(
        work, str(tmp_path / "out.avif"), params, use_rgb444=True,
        capability=media_io.AvifencCapability(available=True,
                                              version=(1, 4, 2)),
        cicp_override="2/2/0")
    assert cmd[cmd.index("--cicp") + 1] == "2/2/0", \
        "cicp_override 没有进入命令（RV-02 未修复）"
    assert contract["cicp"] == "2/2/0"
    assert contract["cicp_override"] == "2/2/0"


def test_new_rv03_lossless_requires_identity_matrix(tmp_path):
    """真机实测的硬约束：lossless 模式不接受非 identity 的 matrix。"""
    work = _write_png_with_icc(tmp_path / "work.png",
                               np.full((4, 4, 3), 128, np.uint8))
    params = {"avif_rgb444_params": {"depth": 16, "yuv": "444",
                                     "cicp": "1/13/1", "lossless": True}}
    _cmd, contract = media_io.build_avifenc_cmd(
        work, str(tmp_path / "out.avif"), params, use_rgb444=True,
        capability=media_io.AvifencCapability(available=True,
                                              version=(1, 4, 2)))
    assert contract["cicp"].endswith("/0")
    assert contract["matrix_forced_identity_by_lossless"] is True


# ---------------------------------------------------------------------------
# RV-03 · 高位深读取
# ---------------------------------------------------------------------------

def test_rv03_rgb16_png_keeps_uint16_and_low_bytes(tmp_path):
    """独立解码器读到 uint16；项目 reader 也曾返回 uint8。"""
    src = _grad16()
    p = str(tmp_path / "rgb16.png")
    media_io.write_png(p, src)
    arr, meta = media_io.read_image_array(p)
    assert arr.dtype == np.uint16, f"reader 返回了 {arr.dtype}（RV-03 未修复）"
    assert np.array_equal(arr, src), "低位字节被丢了"
    assert "imagecodecs" in (meta.get("reader") or "")


def test_rv03_high_depth_read_failure_is_loud(tmp_path, monkeypatch):
    """没有能读 16-bit 的依赖时必须报错，不能静默塌成 8-bit。"""
    src = _grad16()
    p = str(tmp_path / "rgb16.png")
    media_io.write_png(p, src)
    monkeypatch.setitem(sys.modules, "imagecodecs", None)
    with pytest.raises(media_io.MediaPrecisionError):
        media_io.read_image_array(p)


def test_rv03_tiff_without_high_depth_reader_is_loud(tmp_path, monkeypatch):
    p = str(tmp_path / "src16.tif")
    import tifffile
    tifffile.imwrite(p, _grad16(), compression="zlib")
    monkeypatch.setitem(sys.modules, "tifffile", None)
    monkeypatch.setitem(sys.modules, "imagecodecs", None)
    with pytest.raises(media_io.MediaPrecisionError):
        media_io.read_image_array(p)


def test_rv03_stage5_keeps_16bit_from_png_input(tmp_path):
    """RGB16 PNG → stage5 → 母版，整条不能掉到 8-bit。"""
    src = _grad16(16, 16, 3)
    temp = tmp_path / "temp"
    logs = tmp_path / "logs"
    temp.mkdir()
    logs.mkdir()
    media_io.write_png(str(temp / "job001_clip_step1_preprocess.png"), src)
    (temp / "clip.png").write_bytes(b"")  # stage5 只用茎名，不需要真实输入文件

    params = {"temp_dir": str(temp), "log_dir": str(logs),
              "photo_bit_depth": 16, "local_mask_denoise": False,
              "grain_synthesis": False}
    r = subprocess.run(
        [sys.executable, os.path.join(STAGES, "stage5_postprocess.py"),
         "--input", str(temp / "clip.png"), "--job-id", "job001",
         "--params", json.dumps(params)],
        capture_output=True, text=True, shell=False, cwd=ROOT)
    assert r.returncode == 0, f"stage5 rc={r.returncode}\n{r.stdout[-1500:]}"
    out = temp / "job001_clip_step5_postprocess.tif"
    import tifffile
    back = tifffile.imread(str(out))
    assert back.dtype == np.uint16, "stage5 把 16-bit PNG 输入压成了 8-bit"
    assert np.array_equal(back, src)


# ---------------------------------------------------------------------------
# RV-04 · ICC 贯穿
# ---------------------------------------------------------------------------

def test_rv04_tiff_icc_is_read_and_carried_into_the_master(tmp_path):
    import tifffile
    icc = _srgb_icc_bytes()
    p = str(tmp_path / "with_icc.tif")
    tifffile.imwrite(p, _grad16(), compression="zlib",
                     extratags=[(34675, 1, len(icc), icc, True)])

    _arr, meta = media_io.read_image_array(p)
    assert meta.get("icc_profile") == icc, \
        "TIFF 高精度读取成功却把 ICC 丢了（RV-04 未修复）"

    master = str(tmp_path / "master.tif")
    info = s6.write_archival_master(p, master, {"photo_bit_depth": 16}, _Log())
    assert info["icc"] == "source", info
    with tifffile.TiffFile(master) as tf:
        tag = tf.pages[0].tags.get(34675)
    assert tag is not None and bytes(tag.value) == icc


def test_rv04_png_icc_is_carried_into_the_encoder_input(tmp_path):
    """母版 → 编码器输入这一跳也不能丢 ICC。"""
    icc = _srgb_icc_bytes()
    src = _grad16()
    p = str(tmp_path / "src.tif")
    import tifffile
    tifffile.imwrite(p, src, compression="zlib",
                     extratags=[(34675, 1, len(icc), icc, True)])
    res = media_io.prepare_encoder_input(p, str(tmp_path / "work"))
    assert res["icc_carried"] is True, res
    assert media_io._png_icc_profile(res["path"]) == icc


# ---------------------------------------------------------------------------
# RV-05 · 内部 TIFF writer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_rv05_internal_tiff_writer_roundtrip(tmp_path, monkeypatch,
                                             channels, dtype):
    """强制 fallback 时用第三方解码器核验 dtype/shape/像素。"""
    import tifffile
    if channels == 1:
        arr = _grad16(6, 5, 1)[:, :, 0]
    else:
        arr = _grad16(6, 5, channels)
    arr = arr.astype(dtype)
    p = str(tmp_path / f"internal_{channels}_{np.dtype(dtype).name}.tif")

    monkeypatch.setitem(sys.modules, "tifffile", None)
    writer = media_io.write_tiff(p, arr)
    monkeypatch.undo()

    assert writer == "internal-tiff"
    back = tifffile.imread(p)
    assert back.dtype == dtype, f"读回 {back.dtype}，应为 {dtype}"
    expected = arr if channels > 1 else arr
    assert np.array_equal(np.asarray(back).reshape(expected.shape), expected), \
        f"像素不一致 (channels={channels}, dtype={dtype})"


# ---------------------------------------------------------------------------
# RV-06 · 请求的操作必须真的执行
# ---------------------------------------------------------------------------

def test_rv06_requested_resize_is_performed_on_the_16bit_path(tmp_path):
    """请求 8x8 → 4x4：输出必须是 4x4，而不是原尺寸 + SUCCESS。"""
    src = _grad16(8, 8, 3)
    temp = tmp_path / "temp"
    logs = tmp_path / "logs"
    temp.mkdir()
    logs.mkdir()
    media_io.write_tiff(str(temp / "job001_clip_step1_preprocess.tif"), src)

    params = {"temp_dir": str(temp), "log_dir": str(logs),
              "photo_bit_depth": 16, "local_mask_denoise": False,
              "grain_synthesis": False, "lanczos_target": "4x4"}
    r = subprocess.run(
        [sys.executable, os.path.join(STAGES, "stage5_postprocess.py"),
         "--input", str(temp / "clip.tif"), "--job-id", "job001",
         "--params", json.dumps(params)],
        capture_output=True, text=True, shell=False, cwd=ROOT)
    assert r.returncode == 0, f"stage5 rc={r.returncode}\n{r.stdout[-1500:]}"
    out = temp / "job001_clip_step5_postprocess.tif"
    import tifffile
    back = tifffile.imread(str(out))
    assert back.shape[:2] == (4, 4), \
        f"请求了 4x4 但输出是 {back.shape}（RV-06 未修复）"
    with open(str(out) + ".status.json", encoding="utf-8") as f:
        status = json.load(f)
    assert status["status"] == "SUCCESS"
    assert status["skipped_operations"] == []


def test_rv06_unresizable_request_fails_instead_of_pretending(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "scipy", None)
    out, reason = s6.__dict__ and None, None  # 占位，避免 linter 误判
    import stage5_postprocess as s5
    out, reason = s5._resize_array_high_depth(_grad16(8, 8, 3), 4, 4)
    assert out is None and reason, "缺少重采样能力时必须显式报告"


def test_rv06_portrait_fallback_does_not_claim_success(tmp_path):
    """注入依赖失败后，兜底复制原图不能再返回"已应用"。"""
    import stage5_postprocess as s5
    fake_png = _write_png_with_icc(tmp_path / "in.png",
                                   np.full((4, 4, 3), 100, np.uint8))
    out_path = str(tmp_path / "out.png")
    res = s5._match_portrait_color(fake_png, [[0, 0, 4, 4]], out_path, _Log())
    assert isinstance(res, dict), "必须返回结构化结果"
    assert res["applied"] is False
    assert res["skipped"] is True
    assert res["reason"]


# ---------------------------------------------------------------------------
# RV-07 / RV-08 · manifest 状态一致性
# ---------------------------------------------------------------------------

def _manifest_with_jobs(path, job_ids):
    doc = {"meta": {}, "jobs": [{"job_id": j, "input_path": f"/in/{j}.mp4",
                                 "detected_class": "dv_digital",
                                 "status": "PENDING"} for j in job_ids]}
    ml.save_manifest(doc, str(path))
    return str(path)


def test_rv07_external_update_is_not_overwritten_by_the_batch(tmp_path):
    """B 已 DONE；外部 writer 更新了 C；本批结束不得把 B 盖回 PENDING。"""
    path = _manifest_with_jobs(tmp_path / "m.yaml", ["A", "B", "C"])
    gen = ml.acquire_run_ownership(path, ["A", "B", "C"], owner="run-1")
    assert gen is not None
    assert ml.save_manifest_job_result(path, "B", "DONE", owner="run-1",
                                       generation=gen)
    # 外部 writer（编辑器/恢复流程）不带凭据更新 C —— 已被领取的 job 不接受
    ml.save_manifest_job_result(path, "C", "DONE")
    doc = ml.load_manifest(path)
    statuses = {j["job_id"]: j["status"] for j in doc["jobs"]}
    assert statuses["B"] == "DONE", "已完成的进度被覆盖了（RV-07 未修复）"


def test_rv07_each_result_is_persisted_immediately(tmp_path):
    """每收到一个结果就落盘：A 完成后立刻可读到 DONE。"""
    path = _manifest_with_jobs(tmp_path / "m.yaml", ["A", "B"])
    gen = ml.acquire_run_ownership(path, ["A", "B"], owner="run-1")
    assert ml.save_manifest_job_result(path, "A", "DONE", owner="run-1",
                                       generation=gen)
    doc = ml.load_manifest(path)
    assert next(j for j in doc["jobs"] if j["job_id"] == "A")["status"] == "DONE"


def test_rv08_late_old_run_cannot_overwrite_a_newer_result(tmp_path):
    """先有旧 run 的 A，再有新 run 的 B：旧 A 的迟到写必须被拒绝。"""
    path = _manifest_with_jobs(tmp_path / "m.yaml", ["A", "B"])
    old_gen = ml.acquire_run_ownership(path, ["A", "B"], owner="run-old")
    new_gen = ml.acquire_run_ownership(path, ["A", "B"], owner="run-new")
    assert new_gen > old_gen
    assert ml.save_manifest_job_result(path, "B", "DONE", owner="run-new",
                                       generation=new_gen)
    # 旧 run 迟到的失败结果
    applied = ml.save_manifest_job_result(path, "A", "FAILED",
                                          owner="run-old",
                                          generation=old_gen)
    assert applied is False, "旧 run 的迟到结果覆盖了新 run（RV-08 未修复）"
    doc = ml.load_manifest(path)
    assert next(j for j in doc["jobs"] if j["job_id"] == "A")["status"] != "FAILED"


def test_rv08_attempt_group_alone_is_not_ordering(tmp_path):
    """换个 UUID 不能证明"更新"，必须带 generation。"""
    path = _manifest_with_jobs(tmp_path / "m.yaml", ["A"])
    g1 = ml.acquire_run_ownership(path, ["A"], owner="run-1")
    g2 = ml.acquire_run_ownership(path, ["A"], owner="run-2")
    assert ml.save_manifest_job_result(path, "A", "DONE", owner="run-2",
                                       generation=g2,
                                       attempt_group="uuid-newer")
    assert not ml.save_manifest_job_result(
        path, "A", "FAILED", owner="run-1", generation=g1,
        attempt_group="uuid-older")


# ---------------------------------------------------------------------------
# RV-09 · 视频恢复身份
# ---------------------------------------------------------------------------

def test_rv09_output_without_completion_marker_is_not_reused(tmp_path):
    job = {"job_id": "j1", "input_path": str(tmp_path / "clip.mp4")}
    (tmp_path / "clip.mp4").write_bytes(b"\x00" * 32)
    final = tmp_path / "clip_restored.mkv"
    final.write_bytes(b"\x00" * 32)  # 存在且"能解码"，但没有身份
    ok, reason = vmain.resume_artifact_matches(str(final), job, {}, None)
    assert ok is False and reason


def test_rv09_matching_marker_is_accepted(tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x01" * 64)
    final = tmp_path / "clip_restored.mkv"
    final.write_bytes(b"\x02" * 64)
    job = {"job_id": "j1", "input_path": str(src)}
    assert vmain.write_completion_marker(str(final), job, {}, None)
    ok, reason = vmain.resume_artifact_matches(str(final), job, {}, None)
    assert ok is True, reason


def test_rv09_changed_input_invalidates_the_artifact(tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x01" * 64)
    final = tmp_path / "clip_restored.mkv"
    final.write_bytes(b"\x02" * 64)
    job = {"job_id": "j1", "input_path": str(src)}
    vmain.write_completion_marker(str(final), job, {}, None)
    src.write_bytes(b"\x09" * 64)  # 素材换了
    ok, _reason = vmain.resume_artifact_matches(str(final), job, {}, None)
    assert ok is False


def test_rv09_same_basename_different_job_is_not_reused(tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"\x01" * 64)
    final = tmp_path / "clip_restored.mkv"
    final.write_bytes(b"\x02" * 64)
    vmain.write_completion_marker(str(final),
                                  {"job_id": "j1", "input_path": str(src)},
                                  {}, None)
    ok, _reason = vmain.resume_artifact_matches(
        str(final), {"job_id": "j2", "input_path": str(src)}, {}, None)
    assert ok is False


def test_rv09_colliding_final_outputs_are_disambiguated(tmp_path):
    """两个不同 job、同名素材不能共用同一个最终输出路径。"""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    jobs = [{"job_id": "jobA", "input_path": "/in/dir1/clip.mp4"},
            {"job_id": "jobB", "input_path": "/in/dir2/clip.mp4"}]
    vmain._ensure_unique_final_outputs(jobs, {"global": {}},
                                       {"output": str(out_dir)}, None)
    second = jobs[1].get("output_path") or ""
    assert "jobB" in second, f"第二个 job 仍会覆盖第一个的输出: {second}"


# ---------------------------------------------------------------------------
# RV-10 · 结构化结果与退出码聚合
# ---------------------------------------------------------------------------

def test_rv10_worker_result_is_structured(tmp_path, monkeypatch):
    path = _manifest_with_jobs(tmp_path / "m.yaml", ["j1"])
    manifest = ml.load_manifest(path)
    monkeypatch.setattr(vmain, "process_job",
                        lambda job, config, args, logger: (86, "injected"))
    res = vmain._run_single_job({"job_id": "j1"}, {}, None, _Log(),
                                manifest, path)
    assert res["status"] == "FAILED"
    assert res["exit_code"] == 86


def test_rv10_persist_failure_is_not_success(tmp_path, monkeypatch):
    path = _manifest_with_jobs(tmp_path / "m.yaml", ["j1"])
    manifest = ml.load_manifest(path)
    monkeypatch.setattr(vmain, "process_job",
                        lambda job, config, args, logger: (0, ""))
    monkeypatch.setattr(vmain, "save_manifest_job_result",
                        lambda *a, **k: False)
    res = vmain._run_single_job({"job_id": "j1"}, {}, None, _Log(),
                                manifest, path)
    assert res["status"] == "persist-failed"
    assert res["exit_code"] == 82


def test_rv10_aggregate_exit_codes():
    assert vmain.aggregate_results(
        [vmain._result("a", vmain.JOB_DONE, 0)]) == 0
    assert vmain.aggregate_results(
        [vmain._result("a", vmain.JOB_DONE, 0),
         vmain._result("b", vmain.JOB_FAILED, 86)]) == 86
    assert vmain.aggregate_results(
        [vmain._result("a", vmain.JOB_CANCELLED, 88)]) == 88
    assert vmain.aggregate_results(
        [vmain._result("a", vmain.JOB_PERSIST_FAILED, 82)]) == 82


def _cli_env(tmp_path, monkeypatch, process_rc):
    """把 CLI 的依赖预检和 stage 执行换成受控替身，驱动真正的调度器。"""
    import yaml
    cfg = {"global": {"log_dir": str(tmp_path / "logs"), "retry_count": 0,
                      "temp_dir": str(tmp_path / "temp"),
                      "output_dir": str(tmp_path / "out")}}
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    mpath = _manifest_with_jobs(tmp_path / "m.yaml", ["j1"])

    import dependency_check
    monkeypatch.setattr(dependency_check, "run_all_checks",
                        lambda *a, **k: (True, [], []))
    monkeypatch.setattr(vmain, "process_job",
                        lambda *a, **k: (process_rc, "stub"))
    monkeypatch.setattr(sys, "argv",
                        ["main.py", "--manifest", mpath,
                         "--config", str(cfg_path)])
    return mpath


def test_rv10_cli_returns_nonzero_when_a_job_fails(tmp_path, monkeypatch):
    """调度器层：失败任务必须让 CLI 以非零码退出（不能无条件打印 Done）。"""
    _cli_env(tmp_path, monkeypatch, 86)
    with pytest.raises(SystemExit) as exc:
        vmain.main()
    assert exc.value.code == 86, "CLI 把失败任务当成了成功"


def test_rv10_cli_success_persists_done(tmp_path, monkeypatch):
    """成功路径：manifest 必须落 DONE，且不会抛非 0 的 SystemExit。"""
    mpath = _cli_env(tmp_path, monkeypatch, 0)
    exit_code = 0
    try:
        vmain.main()
    except SystemExit as exc:          # 成功时不应退出；若退出也必须是 0
        exit_code = exc.code or 0
    assert exit_code == 0
    doc = ml.load_manifest(mpath)
    assert next(j for j in doc["jobs"] if j["job_id"] == "j1")["status"] == "DONE"


# ---------------------------------------------------------------------------
# RV-11 · 缓存身份必须包含人脸元数据
# ---------------------------------------------------------------------------

def test_rv11_face_metadata_changes_the_cache_identity(tmp_path):
    import stage5_postprocess as s5
    p = str(tmp_path / "in.tif")
    media_io.write_tiff(p, _grad16())
    base = s5.compute_stage5_cache_identity(p, {}, 16)
    with_meta = s5.compute_stage5_cache_identity(
        p, {}, 16, face_boxes=[[1, 2, 3, 4]], face_landmarks=[[[1, 1]]])
    assert base["cache_id"] != with_meta["cache_id"], \
        "人脸元数据变了 cache_id 却不变（RV-11 未修复）"

    moved = s5.compute_stage5_cache_identity(
        p, {}, 16, face_boxes=[[1, 2, 30, 40]], face_landmarks=[[[1, 1]]])
    assert moved["cache_id"] != with_meta["cache_id"]

    # 无关字段（不在 STAGE5_CACHE_PARAM_KEYS 里）不应导致失效
    same = s5.compute_stage5_cache_identity(
        p, {"some_logging_flag": "x"}, 16,
        face_boxes=[[1, 2, 3, 4]], face_landmarks=[[[1, 1]]])
    assert same["cache_id"] == with_meta["cache_id"]


# ---------------------------------------------------------------------------
# RV-12 · 产物事务发布（编码器替身：写半截 + 失败）
# ---------------------------------------------------------------------------

class _SubprocessShim:
    """把 stage6 看到的 `subprocess` 换成"只有 Popen 是假的"。

    原来直接 `monkeypatch.setattr(s6.subprocess, "Popen", ...)` ——
    但 `s6.subprocess` 就是**全局 subprocess 模块**，于是该测试期间
    任何 `subprocess.run()`（内部是 `with Popen(...) as p:`）都会拿到
    假 Popen 并抛 "does not support the context manager protocol"。
    这会让测试结果依赖**前面跑过哪些测试**（全量跑时失败、单跑通过）。

    这里只替换 stage6 这一侧的引用，`run()` 也走同一个替身，
    不再污染进程全局。
    """

    def __init__(self, real, fake_popen):
        self._real = real
        self.Popen = fake_popen

    def run(self, cmd, **kw):
        p = self.Popen(cmd, **kw)
        out, err = p.communicate()

        class _Completed:
            pass

        r = _Completed()
        r.returncode = p.returncode
        r.stdout = out
        r.stderr = err
        r.args = cmd
        return r

    def __getattr__(self, name):
        return getattr(self._real, name)


class _FakeProc:
    """编码器替身：把"写了一半再返回失败"做成确定性行为。"""

    def __init__(self, cmd, returncode=1, write_bytes=0, **kw):
        self.cmd = cmd
        self.returncode = returncode
        self._write = write_bytes
        self.stdout = None
        self.stderr = None

    # `subprocess.run()` 内部是 `with Popen(...) as process:` ——
    # 桩缺上下文管理器协议时，只要被测代码走了 run() 而不是直接 Popen，
    # 就会抛 "does not support the context manager protocol"。
    # 这是桩不完整，不是产品缺陷；补齐后两条路径都能被替身覆盖。
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.wait()
        return False

    def communicate(self, timeout=None):
        if self._write:
            with open(self.cmd[-1], "wb") as f:
                f.write(b"\x00" * self._write)
        return b"", (b"stub failure" if self.returncode else b"")

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def _seed_stage5_input(tmp_path, arr=None):
    temp = tmp_path / "temp"
    logs = tmp_path / "logs"
    out = tmp_path / "output"
    for d in (temp, logs, out):
        d.mkdir(parents=True, exist_ok=True)
    media_io.write_tiff(str(temp / "job001_clip_step5_postprocess.tif"),
                        arr if arr is not None else _grad16())
    return temp, logs, out


def _run_stage6(tmp_path, monkeypatch, fake_cls, extra_params=None):
    temp, logs, out = _seed_stage5_input(tmp_path)
    params = {"temp_dir": str(temp), "log_dir": str(logs),
              "output_dir": str(out), "photo_bit_depth": 16,
              "archival_mode": True}
    params.update(extra_params or {})
    monkeypatch.setattr(s6.shutil, "which", lambda name: "avifenc")
    monkeypatch.setattr(s6.media_io, "detect_avifenc_capability",
                        lambda *a, **k: media_io.AvifencCapability(
                            available=True, version=(1, 4, 2)))
    # 只替换 stage6 看到的 subprocess，不动进程全局模块（见 _SubprocessShim）
    monkeypatch.setattr(s6, "subprocess", _SubprocessShim(subprocess, fake_cls))
    argv = [sys.argv[0], "--input", str(temp / "clip.tif"),
            "--job-id", "job001", "--params", json.dumps(params)]
    monkeypatch.setattr(sys, "argv", argv)
    rc = s6.main()
    return rc, out


def test_rv12_failed_encode_leaves_no_partial_and_records_failed(tmp_path,
                                                                 monkeypatch):
    """编码器写了一半再失败：最终路径不能有半成品，且必须有 FAILED 记录。"""

    def _fake(cmd, **kw):
        return _FakeProc(cmd, returncode=1, write_bytes=4096)

    rc, out = _run_stage6(tmp_path, monkeypatch, _fake)
    final = out / "clip_restored.avif"
    assert rc == 84
    assert not final.exists(), "失败后最终路径上仍留有半成品（RV-12 未修复）"
    assert not (out / "clip_restored.avif.partial").exists()
    # R13-04：**提交前失败不再覆写正式 status**。
    # 正式 generation 状态只由 publisher 在事务内更新；一次失败尝试的诊断
    # 写到 attempt 专属侧车，调度器读那一份即可。
    status_file = out / "clip_restored.avif.status.json"
    assert not status_file.exists(), \
        "失败尝试不应创建/覆写正式 status（会污染旧代的有效状态）"

    attempts = sorted(out.glob("clip_restored.avif.status.json.attempt-*.json"))
    assert attempts, "失败没有留下 attempt 级 FAILED 记录"
    status = json.loads(attempts[-1].read_text(encoding="utf-8"))
    assert status["status"] == "FAILED"
    assert status["failed_operation"] == "avif_encode"
    # 同一提交集合：母版也不能留下"半个提交"
    assert not (out / "clip_restored.tif").exists()


def test_rv12_success_publishes_atomically_and_records_success(tmp_path,
                                                               monkeypatch):
    def _fake(cmd, **kw):
        return _FakeProc(cmd, returncode=0, write_bytes=4096)

    rc, out = _run_stage6(tmp_path, monkeypatch, _fake)
    assert rc == 0
    final = out / "clip_restored.avif"
    assert final.exists(), "成功路径没有发布产物"
    assert not (out / "clip_restored.avif.partial").exists()
    status = json.loads((out / "clip_restored.avif.status.json")
                        .read_text(encoding="utf-8"))
    assert status["status"] == "SUCCESS"


# ---------------------------------------------------------------------------
# RV-13 · WSL 回拷必须使用工作路径（替身验证参数）
# ---------------------------------------------------------------------------

def test_rv13_copyback_uses_the_work_path(tmp_path, monkeypatch):
    """AVIF 与 TIFF 的回拷源都必须是工作路径，不能从发布路径派生。"""
    temp, logs, out = _seed_stage5_input(tmp_path)
    calls = []

    def _fake_redirect(path, no_wsl_redirect=False, target_platform="auto"):
        # 只重定向最终产物（.avif），temp_dir 保持原样，否则 stage6 找不到输入
        if not str(path).endswith(".avif"):
            return path, None, None, None
        work = str(tmp_path / "wsl_work" / os.path.basename(path))
        os.makedirs(os.path.dirname(work), exist_ok=True)
        return work, path, None, None

    def _fake_copy_back(src, dst):
        calls.append((src, dst))
        shutil.copyfile(src, dst)
        return True

    monkeypatch.setattr(s6, "wsl_redirect_path", _fake_redirect)
    monkeypatch.setattr(s6, "wsl_copy_back", _fake_copy_back)
    monkeypatch.setattr(s6.shutil, "which", lambda name: "avifenc")
    monkeypatch.setattr(s6.media_io, "detect_avifenc_capability",
                        lambda *a, **k: media_io.AvifencCapability(
                            available=True, version=(1, 4, 2)))
    monkeypatch.setattr(s6.subprocess, "Popen",
                        lambda cmd, **kw: _FakeProc(cmd, returncode=0,
                                                    write_bytes=2048))
    params = {"temp_dir": str(temp), "log_dir": str(logs),
              "output_dir": str(out), "photo_bit_depth": 16,
              "archival_mode": True}
    monkeypatch.setattr(sys, "argv",
                        [sys.argv[0], "--input", str(temp / "clip.tif"),
                         "--job-id", "job001", "--params", json.dumps(params)])
    rc = s6.main()
    assert rc == 0, f"stage6 rc={rc}"
    assert calls, "没有发生任何回拷"
    for src, _dst in calls:
        assert str(tmp_path / "wsl_work") in src, \
            f"回拷源用了发布路径而不是工作路径: {src}"
    assert any(dst.endswith(".tif") for _src, dst in calls), \
        "TIFF 母版没有被回拷"


# ---------------------------------------------------------------------------
# RV-14 · GCD 只是量化网格诊断
# ---------------------------------------------------------------------------

def test_rv14_heuristic_is_documented_not_authoritative():
    flat16 = np.full((8, 8, 3), 40000, np.uint16)   # 真 16-bit 常量图
    scaled8 = (np.arange(64).reshape(8, 8).astype(np.uint16) * 257)[:, :, None]
    scaled8 = np.repeat(scaled8, 3, axis=2)
    perturbed = scaled8.copy()
    perturbed[0, 0, 0] = (int(perturbed[0, 0, 0]) + 1) & 0xFFFF

    assert s6.quantization_grid_bits(flat16) <= 8, \
        "常量 16-bit 图的网格诊断值很小 —— 这正是它不能当证据的原因"
    assert s6.quantization_grid_bits(scaled8) == 8
    assert s6.quantization_grid_bits(perturbed) > 8, \
        "1-LSB 扰动就能把 GCD 打散 —— 指标可以被绕过"
    # 容器位深与网格诊断必须分开记录
    assert s6.container_bit_depth(flat16) == 16
    assert s6.effective_bit_depth(scaled8) == s6.quantization_grid_bits(scaled8)


# ---------------------------------------------------------------------------
# 真实编码过程中的取消（RV-12 验收要求：不能只测 sleep 脚本）
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_avifenc() is None,
                    reason="avifenc 不可用 —— 真实编码取消未验收")
def test_rv12_real_cancel_during_encoding(tmp_path):
    """真实 avifenc 编码过程中取消：不留半成品、留 FAILED、退出码 88。"""
    rng = np.random.default_rng(20261005)
    big = rng.integers(0, 65536, (900, 900, 3), dtype=np.uint64).astype(
        np.uint16)
    temp, logs, out = _seed_stage5_input(tmp_path, big)
    params = {"temp_dir": str(temp), "log_dir": str(logs),
              "output_dir": str(out), "photo_bit_depth": 16,
              "archival_mode": True,
              "avif_rgb444_params": {"speed": 0, "lossless": True,
                                     "depth": 16, "yuv": "444",
                                     "cicp": "1/13/0"}}
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(_avifenc()) + os.pathsep + env.get("PATH", "")
    popen_kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    proc = subprocess.Popen(
        [sys.executable, os.path.join(STAGES, "stage6_encode.py"),
         "--input", str(temp / "clip.tif"), "--job-id", "job001",
         "--params", json.dumps(params)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, cwd=ROOT, **popen_kwargs)

    partial = out / "clip_restored.avif.partial"
    final = out / "clip_restored.avif"
    # avifenc 只在收尾时创建输出文件，所以"等 partial 出现"是不可靠的。
    # 改为等编码器输入（enc_input.png）就绪 —— 它的下一步就是启动 avifenc。
    enc_input = temp / "job001_clip_step5_postprocess_enc_input.png"
    deadline = time.time() + 90
    while time.time() < deadline and not enc_input.exists():
        if proc.poll() is not None:
            break
        time.sleep(0.2)
    assert enc_input.exists(), (
        "编码器输入没有生成，无法验证取消（stage 提前结束）")
    # 600x600 @ speed 0 实测约 16s，这里等 3s 确保 avifenc 已经在跑
    time.sleep(3.0)
    assert proc.poll() is None, "编码已经结束，取消时机不成立"
    proc.send_signal(getattr(__import__("signal"), "CTRL_BREAK_EVENT"))
    try:
        _out, _err = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise AssertionError("取消后进程没有退出")

    assert proc.returncode == 88, \
        f"取消后的退出码应为 88（interrupted），实际 {proc.returncode}"
    # 稍等，确认没有孤儿编码器继续写回
    time.sleep(1.0)
    assert not (out / "clip_restored.avif").exists(), \
        "取消之后最终路径上出现了产物（可能有孤儿写回）"
    assert not partial.exists(), "临时产物没有被清理"
    status = json.loads((out / "clip_restored.avif.status.json")
                        .read_text(encoding="utf-8"))
    assert status["status"] == "FAILED"
    assert status["failed_operation"] == "cancelled"


# ---------------------------------------------------------------------------
# NEW-RV-07：A/V 同步门在 MKV 上 100% 静默跳过
# ---------------------------------------------------------------------------

def _load_stage7():
    """加载 video stage7 模块（加 video stages 目录到 sys.path）。"""
    vstages = os.path.join(ROOT, "video_restoration_pipeline", "stages")
    if vstages not in sys.path:
        sys.path.insert(0, vstages)
    import stage7_encode as s7  # noqa: E402
    return s7


def test_new_rv07_mkv_stream_duration_is_na_so_old_probe_returns_zero():
    """反例复现：MKV 的 stream=duration 是 N/A，旧实现只能返回 0.0。

    真机证据（ffmpeg 2023-11-13 gyan 版）：
        ffprobe -show_entries stream=duration clip_restored.mkv → N/A
        ffprobe -show_entries format=duration clip_restored.mkv → 2.071000
    """
    s7 = _load_stage7()

    class _R:
        def __init__(self, out, rc=0):
            self.stdout = out
            self.returncode = rc
            self.stderr = ""

    calls = []

    def fake_run(cmd, *a, **k):
        joined = " ".join(str(c) for c in cmd)
        calls.append(joined)
        if "stream=duration" in joined:
            return _R("N/A\n")          # MKV 的真实行为
        if "packet=pts_time" in joined:
            return _R("")               # 退路也取不到时必须诚实地返回 0
        raise AssertionError("unexpected probe: " + joined)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(s7.subprocess, "run", fake_run)
        assert s7._get_stream_duration("x.mkv", "v") == 0.0
    finally:
        monkeypatch.undo()
    # 必须真的试过两级取值，而不是第一次就拿不到就放弃
    assert any("packet=pts_time" in c for c in calls), \
        "没有走 packet 退路，MKV 上同步门仍然等于关闭"


def test_new_rv07_packet_span_fallback_returns_real_duration():
    """packet 时间戳跨度退路：MKV 上应得到真实时长（2.0s）。"""
    s7 = _load_stage7()

    class _R:
        def __init__(self, out, rc=0):
            self.stdout = out
            self.returncode = rc
            self.stderr = ""

    packets = "".join(
        f"{i * 0.04:.6f},0.040000\n" for i in range(50))  # 50 帧 × 40ms = 2.0s

    def fake_run(cmd, *a, **k):
        joined = " ".join(str(c) for c in cmd)
        if "stream=duration" in joined:
            return _R("N/A\n")
        if "packet=pts_time" in joined:
            return _R(packets)
        raise AssertionError("unexpected probe: " + joined)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(s7.subprocess, "run", fake_run)
        dur = s7._get_stream_duration("x.mkv", "v")
    finally:
        monkeypatch.undo()
    assert dur == pytest.approx(2.0, abs=1e-3), f"packet 退路算出的时长不对：{dur}"


def test_new_rv07_sync_gate_measures_real_mkv(tmp_path):
    """真机：对一个真实 MKV，同步门必须量到时长，而不是 SKIP。"""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("无 ffmpeg/ffprobe —— NEW-RV-07 真机部分记为未验收")
    s7 = _load_stage7()

    mkv = tmp_path / "av.mkv"
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=25",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(mkv)],
        capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"ffmpeg 合成测试片失败（未验收）：{r.stderr[:200]}")
    assert mkv.exists()

    vd = s7._get_stream_duration(str(mkv), "v")
    ad = s7._get_stream_duration(str(mkv), "a")
    assert vd > 0, "MKV 视频时长仍取不到 —— 同步门在默认容器上依旧是关闭的"
    assert ad > 0, "MKV 音频时长仍取不到 —— 同步门在默认容器上依旧是关闭的"
    assert abs(vd - ad) * 1000 < 500, \
        f"合成的等长素材不应报 500ms 以上失步：v={vd} a={ad}"


# ---------------------------------------------------------------------------
# NEW-RV-05 / NEW-RV-06 / NEW-RV-04（真机视频，缺 ffmpeg 则记为未验收）
# ---------------------------------------------------------------------------

def test_new_rv06_filter_option_path_survives_ffmpeg_escaping():
    """NEW-RV-06 → R2-09：滤镜选项里的路径必须按 FFmpeg **两层**转义。

    契约变更说明（必须有理由，不能只改断言）：
    第二轮只做了第 1 层（``\\`` → ``/``，``:`` → ``\\:``），这条断言当时是
    ``E\\:/m/std.rnnn``。第三轮 R2-09 用真实 ffmpeg 实测证明该输出在
    Windows 绝对路径上**会失败** —— ``\\:`` 的反斜杠在第 2 层被吃掉，
    盘符冒号失去转义，被 arnndn 当成 ``mix`` 选项分隔符。
    新契约是两层转义后的 ``E\\\\:/m/std.rnnn``，8 种路径实测全部 rc=0
    （证据 `_repair_r3_20261010/evidence/a19_probe.json`）。
    等价反例由下面的 `test_r2_09_real_ffmpeg_model_path_matrix` 承担
    （真实 ffmpeg 调用，而不是字符串比对）。
    """
    sys.path.insert(0, SHARED)
    import audio_utils  # noqa: E402

    # 真机报错（第二轮）：Failed to open model file: modelsstd.rnnn
    assert audio_utils.filter_option_path("models\\std.rnnn") == "models/std.rnnn"
    # 盘符的 ':' 是滤镜选项分隔符；第 1 层转义成 `\:` 后，第 2 层要把 `\` 再转义一次
    win_path = "E:" + chr(92) + "m" + chr(92) + "std.rnnn"   # E:\m\std.rnnn
    assert audio_utils.filter_option_path(win_path) == "E" + chr(92) * 2 + ":/m/std.rnnn"
    # 第 2 层特殊字符：, ; [ ] '
    assert audio_utils.filter_option_path("models/a,b.rnnn") == "models/a" + chr(92) + ",b.rnnn"
    assert audio_utils.filter_option_path("models/a;b.rnnn") == "models/a" + chr(92) + ";b.rnnn"
    assert audio_utils.filter_option_path("models/a[b].rnnn") == "models/a" + chr(92) + "[" + "b" + chr(92) + "].rnnn"
    # 第 1 层+第 2 层都处理单引号：' → \' → \\'
    assert audio_utils.filter_option_path("models/it's.rnnn") == \
        "models/it" + chr(92) * 2 + chr(92) + "'s.rnnn"
    # 正斜杠输入不应被二次破坏
    assert audio_utils.filter_option_path("models/std.rnnn") == "models/std.rnnn"


def test_new_rv05_zscale_without_format_option_degrades_honestly(monkeypatch):
    """NEW-RV-05 / R2-12：zscale 没有 format= 时不得硬拼，必须走降级链。

    契约变更说明：第二轮该用例断言事件名 `STAGE7_DITHER_DEGRADED`，
    且日志声称"dither 不作用于降位深"。第三轮 R2-12 实测证明**这个说法是错的**
    （dither=ordered/error_diffusion 与 none 相比确实改变像素），
    因此事件改名为 `STAGE7_DITHER_FALLBACK_CHAIN`，日志改为陈述事实。
    断言同步更新，并新增下面两条真实像素/限帧用例作为等价（更强）反例。
    """
    s7 = _load_stage7()

    class _Log:
        def __init__(self):
            self.warnings = []
            self.infos = []

        def warning(self, event, message, **kw):
            self.warnings.append((event, message))

        def info(self, *a, **k):
            self.infos.append((a[0] if a else "",))

    monkeypatch.setattr(s7, "_zscale_probe", lambda expr: "unsupported")
    monkeypatch.setattr(s7, "_zscale_has_option", lambda expr: False)
    log = _Log()
    filters = s7._zscale_dither_filters("ordered", log)
    assert filters, "降级链必须给出可用的滤镜串，不能返回空"
    # 关键：绝不能把 zscale 不支持的 format= 拼进 zscale 里（NEW-RV-05）
    assert not any(f.startswith("zscale=") and "format=" in f for f in filters), \
        "又把 format= 拼进 zscale 了 —— 该构建会 rc=84"
    assert any(ev == "STAGE7_DITHER_FALLBACK_CHAIN" for ev, _ in log.warnings), \
        "降级了却没记录 —— 等于静默降级，正是 NEW-RV-05 要禁止的"
    # R2-12：不能再声称"抖动无效"
    joined = " ".join(m for _, m in log.warnings)
    assert "does NOT apply" not in joined, (
        "仍在声称 dither 不作用于降位深 —— 实测该说法错误（R2-12）")


def test_r212_zscale_probe_is_bounded(monkeypatch):
    """R2-12：能力探测必须限帧，不能靠超时兜底。

    无限 lavfi color 源会让"支持的选项"跑到超时（实测 15s 才返回 False）。
    这里断言探测命令里带 -frames:v，且限时内返回结论。
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("无 ffmpeg —— R2-12 限帧探测记为未验收")
    s7 = _load_stage7()
    s7._ZSCALE_CAP_CACHE.clear()
    s7._ZSCALE_CAP_REASON.clear()

    seen = {}
    orig_run = s7.subprocess.run      # 先取原函数，否则 spy 会递归调自己

    def spy_run(cmd, *a, **kw):
        seen["cmd"] = list(cmd)
        return orig_run(cmd, *a, **kw)

    monkeypatch.setattr(s7.subprocess, "run", spy_run)
    start = time.time()
    reason = s7._zscale_probe("dither=ordered")
    elapsed = time.time() - start

    assert reason == "supported", f"受支持的选项被判成 {reason}（限帧探测仍有误判）"
    assert "-frames:v" in seen.get("cmd", []), \
        "探测没有限帧 —— 无限 lavfi 源会靠超时兜底（实测 15s 判 False）"
    assert elapsed < 10, f"限帧探测仍耗时 {elapsed:.1f}s，说明没有真正生效"


def test_r212_dither_changes_pixels_in_fallback_chain(tmp_path):
    """R2-12 真实负向对照：降级链里 dither 必须真的改变像素。

    没有这条，"dither 有效/无效"都只是日志里的形容词。
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("无 ffmpeg —— R2-12 像素对照记为未验收")

    def run(dither: str) -> bytes:
        vf = (f"format=yuv444p16,zscale=rangein=limited:range=limited:"
              f"dither={dither},format=yuv444p10")
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi",
             "-i", "testsrc2=s=64x64:d=1:r=1", "-vf", vf,
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv444p10le", "-"],
            capture_output=True, timeout=60)
        assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[:300]
        return r.stdout

    base = run("none")
    assert base, "参考输出为空，对照不成立"
    for dither in ("ordered", "error_diffusion"):
        other = run(dither)
        diff = sum(1 for a, b in zip(base, other) if a != b)
        assert diff > 0, (
            f"dither={dither} 与 none 的输出完全相同 —— "
            "说明该链里抖动确实无效，日志/文档必须据此修正")


def test_new_rv04_real_stage7_with_audio_track(tmp_path):
    """NEW-RV-04/05/06 的真机闸门：源片带音轨时 stage7 必须 rc=0。

    修复前这条必然失败：
      Option vf ... cannot be applied to input url ...mka → rc=84
    """
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("无 ffmpeg/ffprobe —— NEW-RV-04 真机部分记为未验收")
    s7 = _load_stage7()

    src = tmp_path / "clip.mp4"
    temp = tmp_path / "temp"
    out = tmp_path / "output"
    logs = tmp_path / "logs"
    for d in (temp, out, logs):
        d.mkdir()

    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=25",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(src)],
        capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"ffmpeg 合成测试片失败（未验收）：{r.stderr[:200]}")

    job, stem = "jobRV4", "clip"
    # 音轨（stage0 的产物形态）
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                    "-vn", "-c:a", "copy", str(temp / f"{job}_{stem}.mka")],
                   check=True, capture_output=True)
    # step6 上游产物（本轮 stage1–6 无法真机跑，这里用源片复制件作替身）
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                    "-c", "copy", str(temp / f"{job}_{stem}_step6_grain.mkv")],
                   check=True, capture_output=True)

    params = json.dumps({"temp_dir": str(temp), "log_dir": str(logs),
                         "output_dir": str(out)})
    rc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "video_restoration_pipeline",
                                      "stages", "stage7_encode.py"),
         "--input", str(src), "--job-id", job, "--params", params],
        capture_output=True, text=True, cwd=str(tmp_path))
    if rc.returncode == 80:
        pytest.skip("依赖预检未通过（环境缺件）——记为未验收")

    produced = list(out.glob(f"{stem}_restored.*"))
    assert rc.returncode == 0, (
        f"带音轨的源片 stage7 必须成功（修复前稳定 rc=84），实际 {rc.returncode}。"
        f"\nstderr 尾部：{rc.stderr[-600:]}")
    assert produced and produced[0].stat().st_size > 0, "stage7 没有产出非空成品"


# ---------------------------------------------------------------------------
# RV-08 的真实多进程压测（补 §10.17 里登记的"并发压测未做"缺口）
# ---------------------------------------------------------------------------

def _stress_worker(path, job_ids, owner, generation, status, barrier_sec):
    """子进程体：用**指定**的 owner/generation 写结果（spawn 安全，必须模块级）。"""
    import time
    import conftest  # noqa: F401  (sys.path bootstrap)
    import manifest_loader as m

    time.sleep(barrier_sec)  # 让新旧 run 真正重叠
    for jid in job_ids:
        m.save_manifest_job_result(path, jid, status, owner=owner,
                                   generation=generation)
    return 0


def test_rv08_real_multiprocess_stress_with_stale_late_writer(tmp_path):
    """真实多进程压测：8 进程 × 160 job，其中 2 个是"迟到的旧 run"。

    补的是 `修复清单.md` §10.17 里"并发压测未做"这条缺口。
    判据不只是"最后都写进去了"，而是**旧 run 的迟到结果不得覆盖新 run**。
    """
    path = str(tmp_path / "stress.jsonl")
    jobs = [{"job_id": f"j{i:03d}", "status": "PENDING"} for i in range(160)]
    with open(path, "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    ml.write_meta(path, {"total": len(jobs)}) if hasattr(ml, "write_meta") else None

    # 旧 run 先认领（拿到较小的 generation），随后新 run 再认领（generation 更大）
    gen_old = ml.acquire_run_ownership(path, owner="run-old")
    gen_new = ml.acquire_run_ownership(path, owner="run-new")
    assert gen_old is not None and gen_new is not None, "未能取得 run 所有权"
    assert gen_new > gen_old, f"新 run 的 generation 必须更大：{gen_new} vs {gen_old}"

    ids = [j["job_id"] for j in jobs]
    # 160 个 job 分给 8 个进程：6 个属于新 run，2 个属于"迟到的旧 run"
    chunks = [ids[i::8] for i in range(8)]
    ctx = __import__("multiprocessing").get_context("spawn")
    procs = []
    for i, chunk in enumerate(chunks):
        stale = (i >= 6)          # 最后两个是迟到旧 run
        owner = "run-old" if stale else "run-new"
        gen = gen_old if stale else gen_new
        status = "FAILED" if stale else "DONE"
        # 旧 run 睡得更久，确保它在新 run 之后才落笔
        p = ctx.Process(target=_stress_worker,
                        args=(path, chunk, owner, gen, status, 1.5 if stale else 0.0))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=180)
        assert p.exitcode == 0, f"worker 失败：exitcode={p.exitcode}"

    final = ml.load_manifest(path)
    by_status = {}
    for j in final["jobs"]:
        by_status[j["status"]] = by_status.get(j["status"], 0) + 1
    assert len(final["jobs"]) == 160, f"job 数被写坏了：{len(final['jobs'])}"
    # 新 run 覆盖 6/8 的 job；旧 run 只分到 2/8，且必须全部被拒
    assert by_status.get("FAILED", 0) == 0, (
        f"迟到的旧 run 覆盖了新 run 的结果：{by_status}")
    assert by_status.get("DONE", 0) >= 100, (
        f"并发写入丢更新，只落了 {by_status.get('DONE', 0)} 条 DONE：{by_status}")
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert not leftovers, f"残留临时文件：{leftovers}"


# ---------------------------------------------------------------------------
# R2-09：FFmpeg 模型路径两层转义 —— 真实 ffmpeg 矩阵（不是字符串比对）
# ---------------------------------------------------------------------------

_R209_CASES = ["relative", "absolute", "space", "chinese", "comma",
               "semicolon", "bracket", "quote"]


def test_r209_real_ffmpeg_model_path_matrix(tmp_path):
    """R2-09：同一有效模型在 8 种路径下都必须经**真实 ffmpeg** 处理成功。

    第二轮只做了一层转义，实测 5 种失败（absolute/comma/semicolon/bracket/quote）。
    这里不做字符串比对，而是真的跑 ffmpeg 并断言 rc=0 —— 只断言"长得像转义过"
    是自欺（交接明确禁止）。
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("无 ffmpeg —— R2-09 真机部分记为未验收")
    sys.path.insert(0, SHARED)
    import audio_utils  # noqa: E402

    model_src = os.path.join(ROOT, "models", "std.rnnn")
    if not os.path.exists(model_src):
        pytest.skip("仓库内没有 models/std.rnnn —— R2-09 真机部分记为未验收")

    modeldir = tmp_path / "models"
    modeldir.mkdir()
    names = {
        "relative": "std.rnnn",
        "absolute": "std.rnnn",
        "space": "my model.rnnn",
        "chinese": "模型 中文.rnnn",
        "comma": "a,b.rnnn",
        "semicolon": "a;b.rnnn",
        "bracket": "a[b].rnnn",
        "quote": "it's.rnnn",
    }
    failures = {}
    for kind in _R209_CASES:
        model = modeldir / names[kind]
        shutil.copyfile(model_src, str(model))
        argument = str(model) if kind == "absolute" else "models/" + names[kind]
        chain = "arnndn=m=" + audio_utils.filter_option_path(argument)
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi",
             "-i", "sine=duration=0.2:sample_rate=48000",
             "-af", chain, "-f", "null", "-"],
            cwd=str(tmp_path), capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            failures[kind] = {"argument": argument, "chain": chain,
                              "stderr": (r.stderr or "")[:300]}
    assert not failures, (
        "R2-09：以下路径经真实 ffmpeg 失败（滤镜图两层转义不完整）："
        + json.dumps(failures, ensure_ascii=False, indent=2))


def test_r209_second_level_escape_is_actually_needed(tmp_path):
    """负向对照：只做第 1 层转义时，真实 ffmpeg 必须失败 —— 证明两层是必要的。

    没有这条，"两层转义"就可能只是把字符串改复杂了，而没解决任何真实缺陷。
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("无 ffmpeg —— R2-09 负向对照记为未验收")
    sys.path.insert(0, SHARED)
    import audio_utils  # noqa: E402

    model_src = os.path.join(ROOT, "models", "std.rnnn")
    if not os.path.exists(model_src):
        pytest.skip("仓库内没有 models/std.rnnn —— R2-09 负向对照记为未验收")

    modeldir = tmp_path / "models"
    modeldir.mkdir()
    model = modeldir / "a,b.rnnn"      # 逗号只在第 2 层是特殊字符
    shutil.copyfile(model_src, str(model))

    # 只做第 1 层（逗号不是第 1 层特殊字符，等于原样）
    only_l1 = "arnndn=m=models/a,b.rnnn"
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "sine=duration=0.2:sample_rate=48000",
         "-af", only_l1, "-f", "null", "-"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=30)
    assert r.returncode != 0, (
        "只做第 1 层转义居然通过了 —— 说明第 2 层转义可能并不需要，需要重新核对")


# ---------------------------------------------------------------------------
# R2-10：立体声内容保留（真实波形，不是只看声道数恢复成 2）
# ---------------------------------------------------------------------------

def _make_stereo_wav(path, seconds=1.0, rate=48000):
    """左右**不同**频率的双声道素材：L=440Hz，R=880Hz。"""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}:sample_rate={rate}",
         "-f", "lavfi", "-i", f"sine=frequency=880:duration={seconds}:sample_rate={rate}",
         "-filter_complex", "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]",
         "-map", "[a]", "-c:a", "pcm_s16le", str(path)],
        capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        pytest.skip(f"无法合成双声道素材（未验收）：{r.stderr[:200]}")
    return path


def _extract_channel(path, index):
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-af",
         f"pan=mono|c0=c{index}", "-f", "s16le", "-"],
        capture_output=True, timeout=60)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[:200]
    return r.stdout


def test_r210_stereo_information_survives_arnndn(tmp_path):
    """R2-10：arnndn 之后左右声道**不得**变成同一份内容。

    旧实现 `aformat=channel_layouts=mono → arnndn → aformat=...stereo`
    会把左右下混成一个单声道再复制回去，声道数看起来还是 2，但内容已经合并。
    实测（本机 ffmpeg）：旧链处理后 L 与 R **完全相同**。
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("无 ffmpeg —— R2-10 声道保留记为未验收")
    s7 = _load_stage7()
    model = os.path.join(ROOT, "models", "std.rnnn")
    if not os.path.exists(model):
        pytest.skip("仓库内没有 models/std.rnnn —— R2-10 声道保留记为未验收")

    sys.path.insert(0, SHARED)
    from audio_utils import filter_option_path

    src = _make_stereo_wav(tmp_path / "stereo.wav")
    assert _extract_channel(src, 0) != _extract_channel(src, 1), \
        "素材本身左右就相同，对照不成立"

    chain = ",".join(s7._arnndn_filters(
        model, {"channels": 2, "layout": "stereo"}, filter_option_path, None))
    out = tmp_path / "denoised.wav"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                        "-af", chain, "-f", "wav", str(out)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"逐声道 arnndn 链跑不起来：{r.stderr[:300]}"
    left = _extract_channel(out, 0)
    right = _extract_channel(out, 1)
    assert left and right, "输出声道为空"
    assert left != right, \
        "arnndn 之后左右声道完全相同 —— 说明左右信息被下混合并了（R2-10）"


def test_r210_old_mono_roundtrip_really_merges_channels(tmp_path):
    """负向对照：旧的 mono 回绕链**必须**被判为合并。

    没有这条，上一条可能只是"本来就没差异"，证明不了门有效。
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("无 ffmpeg —— R2-10 负向对照记为未验收")
    model = os.path.join(ROOT, "models", "std.rnnn")
    if not os.path.exists(model):
        pytest.skip("仓库内没有 models/std.rnnn —— R2-10 负向对照记为未验收")

    sys.path.insert(0, SHARED)
    from audio_utils import filter_option_path

    src = _make_stereo_wav(tmp_path / "stereo.wav")
    old_chain = (f"aformat=channel_layouts=mono,"
                 f"arnndn=m={filter_option_path(model)},"
                 f"aformat=channel_layouts=stereo")
    out = tmp_path / "old.wav"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                        "-af", old_chain, "-f", "wav", str(out)],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        pytest.skip(f"旧链在本机跑不起来（未验收）：{r.stderr[:200]}")
    assert _extract_channel(out, 0) == _extract_channel(out, 1), (
        "旧 mono 回绕链居然没有合并左右 —— 那么上一条用例就没有证明力，"
        "需要重新设计对照")


# ---------------------------------------------------------------------------
# R2-04 / R2-05：调度入口（视频不覆盖并发结果、照片完成顺序落盘）
# ---------------------------------------------------------------------------

def test_r204_video_startup_does_not_clobber_concurrent_done(tmp_path,
                                                             monkeypatch):
    """R2-04 等价反例：视频启动期不得用旧快照把并发写好的 DONE 盖回 PENDING。

    与独立用例 A09 同一注入设置；这里不依赖"启动期一定会整表写盘"这一前提，
    而是直接断言磁盘最终状态（因此更强）。
    """
    path = _cli_env(tmp_path, monkeypatch, 0)
    original_load = vmain.load_manifest
    injected = {"done": False}

    def stale_load(p):
        old = original_load(p)
        if str(p) == path and not injected["done"]:
            injected["done"] = True
            gen = ml.acquire_run_ownership(path, ["j1"], owner="concurrent-A")
            assert ml.save_manifest_job_result(
                path, "j1", "DONE", owner="concurrent-A", generation=gen)
        return old          # 本次进程拿到的是**注入前**的旧快照

    # 必须在"作业真正被处理之前"停下来看磁盘：否则后面 job 处理完又会写一次
    # DONE，把启动期的覆盖掩盖掉（第一版用例就是这样在基线上也通过的）。
    class _Stop(Exception):
        pass

    def _boom(*_a, **_k):
        raise _Stop()

    monkeypatch.setattr(vmain, "load_manifest", stale_load)
    monkeypatch.setattr(vmain, "process_job", _boom)
    try:
        vmain.main()
    except _Stop:
        pass
    except SystemExit:
        pass
    final = ml.load_manifest(path)
    assert final["jobs"][0]["status"] == "DONE", (
        "视频启动期用旧快照整表覆盖，把并发运行的 DONE 打回 "
        f"{final['jobs'][0]['status']}（R2-04）")


def test_r205_photo_scheduler_persists_completed_job_without_waiting(
        tmp_path, monkeypatch):
    """R2-05：已完成的 job 不得被前面那个慢 job 挡住落盘。

    驱动**真正的 photo main() 调度入口**：
      * job A 在 worker 里阻塞（等事件）；job B 立即返回；
      * A 只有在"读到磁盘上 B 已是 DONE"之后才放行 —— 这消除了竞态，
        如果父进程是按提交顺序 future.get() 阻塞，A 永远等不到 B 落盘，
        于是断言失败。
    进程池换成线程池，是为了让父进程里替换的 _run_single_job 在 worker 中
    可见（spawn 子进程看不到 monkeypatch）；被测对象是**父进程的收集逻辑**，
    这正是 R2-05 的缺陷所在。
    """
    import importlib.util as _ilu
    import multiprocessing as _mp
    import multiprocessing.pool as _mppool
    import threading

    pmain_path = os.path.join(ROOT, "photo_restoration_pipeline", "main.py")
    spec = _ilu.spec_from_file_location("photo_main_r205", pmain_path)
    pmain = _ilu.module_from_spec(spec)
    spec.loader.exec_module(pmain)

    manifest_path = str(tmp_path / "m.jsonl")
    jobs = [{"job_id": "A", "status": "PENDING", "input_path": "a.png",
             "detected_class": "scan_print",
             "output_path": str(tmp_path / "a.out")},
            {"job_id": "B", "status": "PENDING", "input_path": "b.png",
             "detected_class": "scan_print",
             "output_path": str(tmp_path / "b.out")}]
    with open(manifest_path, "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    ml.write_meta(manifest_path, {"total": len(jobs)}) if hasattr(
        ml, "write_meta") else None

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("global:/n  temp_dir: .\n", encoding="utf-8")

    observed = {"b_done_before_a": False}
    release = threading.Event()

    def fake_run_single_job(job, config_path, config, cli_params):
        if job["job_id"] == "A":
            # 轮询等待：B 的结果必须**已经在磁盘上**才放行
            deadline = time.time() + 20
            while time.time() < deadline:
                doc = ml.load_manifest(manifest_path)
                hit = [j for j in doc["jobs"] if j["job_id"] == "B"]
                if hit and hit[0].get("status") == "DONE":
                    observed["b_done_before_a"] = True
                    break
                time.sleep(0.05)
            return 0
        rc = 0
        return rc

    class _ThreadContext:
        def Pool(self, processes=None):
            return _mppool.ThreadPool(processes=processes or 2)

    class _FakeMp:
        @staticmethod
        def get_context(_name=None):
            return _ThreadContext()

    # photo main 在函数内部 `import multiprocessing as mp` 后调用
    # `mp.get_context('spawn')`，因此要 patch multiprocessing 模块本身。
    monkeypatch.setattr(pmain, "_run_single_job", fake_run_single_job)
    monkeypatch.setattr(_mp, "get_context", _FakeMp.get_context)
    monkeypatch.setattr(sys, "argv",
                        ["main.py", "--manifest", manifest_path,
                         "--config", str(cfg_path), "--workers", "2"])

    try:
        pmain.main()
    except SystemExit as e:
        assert e.code in (0, None, 3), f"photo main 退出码异常：{e.code}"

    assert observed["b_done_before_a"], (
        "R2-05：job B 已经完成，但在 job A 结束前它的结果没有被落盘 —— "
        "说明父进程仍按提交顺序阻塞收集（此时调度器退出就会丢掉 B 的状态）")
    final = ml.load_manifest(manifest_path)
    statuses = {j["job_id"]: j["status"] for j in final["jobs"]}
    assert statuses.get("B") == "DONE", f"B 最终状态不对：{statuses}"
    assert statuses.get("A") == "DONE", f"A 最终状态不对：{statuses}"


# ---------------------------------------------------------------------------
# R2-10 扩展：>2 声道逐声道处理（原列为未完成，现补实测）
# ---------------------------------------------------------------------------

def _make_multichannel_wav(path, freqs=(220, 330, 440, 550, 660, 770),
                           layout="5.1", seconds=1.0, rate=48000):
    """合成 N 声道素材：每个声道放**不同**频率，便于检测是否被合并。"""
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for f in freqs:
        cmd += ["-f", "lavfi", "-i",
                f"sine=frequency={f}:duration={seconds}:sample_rate={rate}"]
    cmd += ["-filter_complex",
            "".join(f"[{i}:a]" for i in range(len(freqs)))
            + f"join=inputs={len(freqs)}:channel_layout={layout}[a]",
            "-map", "[a]", "-c:a", "pcm_s16le", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        pytest.skip(f"无法合成多声道素材（未验收）：{r.stderr[:200]}")
    return path


def test_r210_multichannel_arnndn_keeps_channels_distinct(tmp_path):
    """>2 声道：逐声道 arnndn 后各声道**不得**被合并。

    之前 >2ch 直接退回 afftdn；现在实现了 channelsplit → 逐声道 → join。
    实测（5.1，六声道不同频率）：rc=0 且六个声道仍两两不同。
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("无 ffmpeg —— 多声道逐声道记为未验收")
    s7 = _load_stage7()
    model = os.path.join(ROOT, "models", "std.rnnn")
    if not os.path.exists(model):
        pytest.skip("仓库内没有 models/std.rnnn —— 多声道逐声道记为未验收")

    sys.path.insert(0, SHARED)
    from audio_utils import filter_option_path

    src = _make_multichannel_wav(tmp_path / "src51.wav")
    src_ch = [_extract_channel(src, i) for i in range(6)]
    assert len(set(src_ch)) == 6, "素材本身声道就相同，对照不成立"

    chain = ",".join(s7._arnndn_filters(
        model, {"channels": 6, "layout": "5.1"}, filter_option_path, None))
    assert "channelsplit" in chain and "join=" in chain, (
        ">2 声道没有走逐声道方案：" + chain)
    out = tmp_path / "denoised51.wav"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                        "-af", chain, "-f", "wav", str(out)],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, f"逐声道 arnndn 链跑不起来：{r.stderr[:300]}"
    got = [_extract_channel(out, i) for i in range(6)]
    assert len(set(got)) == 6, (
        "处理之后声道被合并了（应保留 6 个互不相同的声道）")


def test_r207_intermediate_without_identity_is_not_reused(tmp_path):
    """R2-07 补完：没有身份侧车的中间产物**不得**被复用。

    "能解码"只证明文件没坏，证明不了它属于本次输入/参数/实现。
    """
    art = tmp_path / "j1_clip_step6_grain.mkv"
    art.write_bytes(b"OLD-INTERMEDIATE")
    job = {"job_id": "j1", "input_path": str(tmp_path / "src.mkv")}
    (tmp_path / "src.mkv").write_bytes(b"SOURCE")
    ok, why = vmain.intermediate_identity_matches(str(art), job, {}, None)
    assert not ok, "没有身份侧车却被判为可复用"
    assert "identity" in why.lower() or "sidecar" in why.lower(), why


def test_r207_intermediate_identity_detects_source_change(tmp_path):
    """R2-07 补完：源内容变了，已登记身份的中间产物必须失效。"""
    src = tmp_path / "src.mkv"
    src.write_bytes(b"SOURCE-V1")
    job = {"job_id": "j1", "input_path": str(src)}
    art = tmp_path / "j1_clip_step6_grain.mkv"
    art.write_bytes(b"INTERMEDIATE")
    assert vmain.record_stage_artifact_identity(
        str(art), job, {}, 6, str(src), None)
    ok1, _ = vmain.intermediate_identity_matches(str(art), job, {}, None)
    assert ok1, "同一输入下已登记的产物应当可复用"

    src.write_bytes(b"SOURCE-V2-CHANGED")
    ok2, why = vmain.intermediate_identity_matches(str(art), job, {}, None)
    assert not ok2, f"源变了却仍判定可复用：{why}"


def test_r207_intermediate_identity_detects_size_change(tmp_path):
    """R2-07 补完：产物自身大小与登记不符时拒绝（防半截文件）。"""
    src = tmp_path / "src.mkv"
    src.write_bytes(b"SOURCE")
    job = {"job_id": "j1", "input_path": str(src)}
    art = tmp_path / "j1_clip_step6_grain.mkv"
    art.write_bytes(b"INTERMEDIATE")
    assert vmain.record_stage_artifact_identity(
        str(art), job, {}, 6, str(src), None)
    art.write_bytes(b"INTERMEDIATE-TRUNCATED-OR-REWRITTEN")
    ok, why = vmain.intermediate_identity_matches(str(art), job, {}, None)
    assert not ok, f"产物大小变了却仍判定可复用：{why}"


# ---------------------------------------------------------------------------
# R2-04 补完：多个 worker **争抢同一批 job** 的完整并发压测
# ---------------------------------------------------------------------------

def _contend_worker(path, owner, barrier):
    """子进程：和别人**争抢同一批** job（不是各写各的集合）。"""
    import conftest  # noqa: F401  (sys.path bootstrap)
    import manifest_loader as m

    barrier.wait()                      # 尽量同时开抢，制造真实竞争
    gen = m.acquire_run_ownership(path, owner=owner)
    if gen is None:
        return []
    written = []
    doc = m.load_manifest(path)
    for job in doc.get("jobs", []):
        ok = m.save_manifest_job_result(path, job["job_id"], "DONE",
                                        owner=owner, generation=gen)
        written.append((job["job_id"], ok, gen, owner))
    return written


def test_r204_workers_contending_for_the_same_jobs_stay_consistent(tmp_path):
    """R2-04 补完：8 个 worker 争抢**同一批** job，结果必须自洽。

    交接指出旧压测让新旧 run 写**不相交**的集合，证明不了竞争场景。
    这里让所有 worker 争抢同一批 job，并用 Barrier 尽量同时开抢，然后逐 job
    断言：状态不丢、归属唯一、generation 一致、meta 计数正确。
    """
    import multiprocessing

    job_ids = [f"j{i:02d}" for i in range(12)]
    path = _manifest_with_jobs(tmp_path / "contend.jsonl", job_ids)

    workers = 8
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(workers)
    procs = []
    for i in range(workers):
        p = ctx.Process(target=_contend_worker,
                        args=(path, f"worker-{i}", barrier))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=180)
        assert p.exitcode == 0, f"worker 失败：exitcode={p.exitcode}"

    final = ml.load_manifest(path)
    jobs = final["jobs"]
    assert len(jobs) == len(job_ids)

    statuses = {j["status"] for j in jobs}
    assert statuses == {"DONE"}, f"并发争抢后有 job 没落盘：{statuses}"

    owners = {str(j.get("run_owner")) for j in jobs}
    assert len(owners) == 1, (
        f"同一批 job 出现了多个归属 —— 竞争下归属不唯一：{owners}")

    gens = {j.get("run_generation") for j in jobs}
    assert len(gens) == 1, (
        f"同一批 job 的 generation 不一致（可能混代）：{gens}")

    # 代际必须单调：最终 generation 不小于任务数（每个 worker 领取都会递增）
    assert final["meta"].get("done") == len(job_ids), (
        f"meta.done 与实际不符：{final['meta'].get('done')}")
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert not leftovers, f"残留临时文件：{leftovers}"


# ---------------------------------------------------------------------------
# R2-11 补完：内容级同步（首/中/尾已知 flash / click 事件）
# ---------------------------------------------------------------------------

_EVENT_TIMES = [0.5, 1.5, 2.5]      # 秒：首 / 中 / 尾


def _build_flash_click(tmp_path, audio_offset: float):
    """合成"已知事件"素材：视频在这些时刻白闪，音频在同一时刻脉冲。

    audio_offset > 0 表示音轨整体延后（制造不同步）。
    """
    import struct
    duration = 3.0
    fps = 25
    sr = 48000

    # --- 视频：黑底 + 指定时刻画白框 ---
    video = tmp_path / "v.mkv"
    vf = ",".join(
        f"drawbox=x=0:y=0:w=64:h=64:color=white@1:t=fill:"
        f"enable='between(t,{t},{t + 0.08})'" for t in _EVENT_TIMES)
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"color=c=black:s=64x64:d={duration}:r={fps}",
         "-vf", vf, "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(video)],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        pytest.skip(f"无法合成 flash 素材（未验收）：{r.stderr[:200]}")

    # --- 音频：静音 + 指定时刻脉冲（用 numpy 直接写 PCM，最可靠）---
    import numpy as _np
    n = int(duration * sr)
    pcm = _np.zeros(n, dtype=_np.int16)
    for t in _EVENT_TIMES:
        start = int(t * sr)
        pcm[start:start + int(0.02 * sr)] = 30000
    raw = tmp_path / "a.pcm"
    raw.write_bytes(pcm.tobytes())

    out = tmp_path / "av.mkv"
    # 注意：`-itsoffset` 必须放在它要作用的那个 `-i` **之前**，
    # 放到 -i 之后不会生效（第一版就写错了位置，导致这条用例被 skip）。
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video)]
    if audio_offset:
        cmd += ["-itsoffset", str(audio_offset)]
    cmd += ["-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", str(raw),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "ffv1", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le",
            str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        pytest.skip(f"无法合成 A/V 素材（未验收）：{r.stderr[:200]}")
    return out, fps, sr


def _first_packet_pts(path, stream):
    """容器里该流第一个 packet 的 pts_time。

    **方法学要点**：把流解码成 raw PCM / rawvideo 后，输出只是字节流，
    **容器里的时间偏移会丢掉**（第一版检测就栽在这里：音轨延后 0.5s，
    raw 解码测出的脉冲位置却几乎没变）。因此必须把首 packet 的 pts 加回去，
    raw 里测到的只是"流内相对时间"。
    """
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream,
         "-show_entries", "packet=pts_time", "-of", "csv=p=0", path],
        capture_output=True, text=True, timeout=60)
    vals = []
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vals.append(float(line))
        except ValueError:
            continue
    return min(vals) if vals else 0.0


def _video_event_times(path, fps):
    """逐帧求平均亮度，找白闪帧的时间戳（已加回流起始 pts）。"""
    import numpy as _np
    base = _first_packet_pts(path, "v:0")
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", "format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, timeout=120)
    assert r.returncode == 0, r.stderr[:200]
    frames = _np.frombuffer(r.stdout, dtype=_np.uint8).reshape(-1, 64 * 64)
    mean = frames.mean(axis=1)
    bright = _np.where(mean > 128)[0]
    if bright.size == 0:
        return []
    # 把连续帧聚成事件
    events, run = [], [bright[0]]
    for idx in bright[1:]:
        if idx - run[-1] <= 2:
            run.append(idx)
        else:
            events.append(run[0]); run = [idx]
    events.append(run[0])
    return [round(base + e / fps, 4) for e in events]


def _audio_event_times(path, sr):
    """逐样本求绝对值，找脉冲的时间戳（已加回流起始 pts）。"""
    import numpy as _np
    base = _first_packet_pts(path, "a:0")
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn",
         "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, timeout=120)
    assert r.returncode == 0, r.stderr[:200]
    pcm = _np.frombuffer(r.stdout, dtype=_np.int16).astype(_np.int32)
    loud = _np.where(_np.abs(pcm) > 20000)[0]
    if loud.size == 0:
        return []
    events, run = [], [loud[0]]
    for idx in loud[1:]:
        if idx - run[-1] <= sr // 10:      # 同一脉冲内
            run.append(idx)
        else:
            events.append(run[0]); run = [idx]
    events.append(run[0])
    return [round(base + e / sr, 4) for e in events]


def test_r211_content_events_align_when_in_sync(tmp_path):
    """R2-11 补完：内容级验证 —— 同步时，flash 与 click 应在同一时刻。

    这是"真正同步"而不是"时长接近"：直接比对已知事件的内容时间戳。
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("无 ffmpeg —— 内容级同步记为未验收")
    av, fps, sr = _build_flash_click(tmp_path, audio_offset=0.0)
    v_t = _video_event_times(av, fps)
    a_t = _audio_event_times(av, sr)
    assert len(v_t) >= 2 and len(a_t) >= 2, f"事件没被检出：v={v_t} a={a_t}"

    deltas = []
    for vt in v_t:
        if not a_t:
            break
        nearest = min(a_t, key=lambda x: abs(x - vt))
        deltas.append(abs(nearest - vt))
    worst = max(deltas)
    assert worst < 0.1, (
        f"同步素材里 flash 与 click 的最大时间差 {worst:.3f}s 超过 100ms "
        f"(video={v_t} audio={a_t})")


def test_r211_content_events_reveal_real_offset(tmp_path):
    """负向对照：音轨整体延后 0.5s 时，内容事件必须**暴露**出来。

    没有这条，"内容级验证"就没有证明力 —— 它可能对真正的错位也报 OK。
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("无 ffmpeg —— 内容级同步负向对照记为未验收")
    s7 = _load_stage7()
    # 用 0.8s 而不是 0.5s：规范的失败阈值是 "> 500ms"，0.5s 正好卡在边界上
    # （第一版用 0.5s，门按规范只 warn 不 fail，看起来像漏判）。
    # 边界口径见 修复清单.md §12.14：正好 500ms 按规范不判失败。
    offset = 0.8
    av, fps, sr = _build_flash_click(tmp_path, audio_offset=offset)
    v_t = _video_event_times(av, fps)
    a_t = _audio_event_times(av, sr)
    assert len(v_t) >= 2 and len(a_t) >= 2, f"事件没被检出：v={v_t} a={a_t}"

    deltas = [min(abs(a - vt) for a in a_t) for vt in v_t]
    worst = max(deltas)
    assert worst > 0.3, (
        f"音轨延后 {offset}s，内容事件却只差 {worst:.3f}s —— 检测没有暴露错位")

    # 现有包头级同步门对这种固定偏移也应当判失败
    class _Log:
        def __init__(self):
            self.events = []

        def _rec(self, ev, msg, **kw):
            self.events.append(ev)

        info = warning = error = debug = critical = _rec

    log = _Log()
    rc = s7._validate_av_sync(str(av), log)
    assert rc != 0, f"音轨延后 {offset}s，包头级同步门仍判定通过"


# ---------------------------------------------------------------------------
# R3-03：发布备份/回滚的三态语义
# ---------------------------------------------------------------------------

def _load_stage6():
    import importlib.util as _ilu
    path = os.path.join(ROOT, "photo_restoration_pipeline", "stages",
                        "stage6_encode.py")
    spec = _ilu.spec_from_file_location("stage6_encode_r303", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_r303_backup_failure_does_not_destroy_previous_generation(tmp_path,
                                                                  monkeypatch):
    """R3-03：备份被拒时，回滚**不得删除**原本存在的旧产物。

    这是独立用例 B03 想表达的核心危害（"回滚把该路径直接删除"）的
    **自洽版本**：不手工篡改文件内容，只验证"不销毁"。

    关于 B03 本身：它在备份被注入失败**之后**又手工把文件写成
    `NEW-PARTIAL`，然后断言回滚后等于 `OLD-VALID`。备份失败意味着磁盘上
    不存在 OLD-VALID 的任何副本，回滚不可能凭空还原 —— 断言与故障注入前提
    不自洽。我在不绕过故障注入的前提下无法满足它；这里用等价断言覆盖
    评估指出的真实危害。
    """
    s6 = _load_stage6()
    final = tmp_path / "old.avif"
    final.write_bytes(b"OLD-VALID")

    def denied(*_a, **_k):
        raise PermissionError("controlled backup denied")

    monkeypatch.setattr(s6.shutil, "copyfile", denied)

    class _Log:
        def __init__(self):
            self.events = []

        def _rec(self, ev, msg, **kw):
            self.events.append(ev)

        info = warning = error = debug = critical = _rec

    log = _Log()
    snapshot = s6._snapshot_for_rollback([str(final)], str(tmp_path / "backup"),
                                         log)
    # 1) 备份失败必须与"原先不存在"区分开
    assert s6._snapshot_has_failure(snapshot), \
        "备份失败没有被识别（与 absent 混为一谈）"
    assert snapshot[str(final)]["state"] == s6.BACKUP_FAILED
    # 2) 回滚不得销毁旧产物
    s6._rollback_publish(snapshot, log)
    assert final.exists(), "备份失败后回滚把旧产物删除了（R3-03 的核心危害）"
    assert final.read_bytes() == b"OLD-VALID"
    # R12-02 之后，这类"备份不可用"会在**预检阶段**就整体阻断（更好：
    # 不是恢复一半才失败）。事件名随之变为 PRECHECK_FAILED；
    # 旧的 RESTORE_IMPOSSIBLE 语义仍保留给"预检通过但执行失败"。
    assert ("STAGE6_ROLLBACK_PRECHECK_FAILED" in log.events
            or "STAGE6_RESTORE_IMPOSSIBLE" in log.events), \
        "回滚失败却仍然声称已恢复"


def test_r303_rollback_restores_backed_up_generation(tmp_path):
    """R3-03：备份成功时，回滚必须把每个目标恢复成提交前的内容。"""
    s6 = _load_stage6()
    final = tmp_path / "old.avif"
    final.write_bytes(b"OLD-VALID")
    tiff = tmp_path / "old.tif"
    tiff.write_bytes(b"OLD-TIFF")

    snapshot = s6._snapshot_for_rollback([str(final), str(tiff)],
                                         str(tmp_path / "backup"))
    assert not s6._snapshot_has_failure(snapshot)
    # 模拟"提交进行到一半"
    final.write_bytes(b"NEW-AVIF")
    tiff.write_bytes(b"NEW-TIFF")
    s6._rollback_publish(snapshot)
    assert final.read_bytes() == b"OLD-VALID"
    assert tiff.read_bytes() == b"OLD-TIFF"


def test_r303_rollback_removes_files_created_by_this_run(tmp_path):
    """R3-03：提交前不存在的路径（本次新建），回滚时应撤下，不留半代。"""
    s6 = _load_stage6()
    fresh = tmp_path / "not_there_before.avif"
    snapshot = s6._snapshot_for_rollback([str(fresh)], str(tmp_path / "backup"))
    assert snapshot[str(fresh)]["state"] == s6.BACKUP_ABSENT
    fresh.write_bytes(b"NEW-PARTIAL")
    s6._rollback_publish(snapshot)
    assert not fresh.exists(), "本次新建的产物在回滚后仍留在磁盘上"


def test_r211_content_events_survive_real_stage7(tmp_path):
    """R3-10：已知事件经过**真实 stage7（含 AV1/AAC 编码）**后仍然对齐。

    之前两条内容同步用例只检查刚合成的 FFV1/PCM 素材，没有经过产品处理链，
    因此不能称为"生产链内容同步通过"。这条把带 flash/click 的素材
    真正送进 stage7_encode，再从产物里检测首/中/尾事件。
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("无 ffmpeg —— 生产链内容同步记为未验收")

    # --- 1. 造带事件的有声源片 ---
    # stage7 按 `--input` 的 stem 匹配中间产物（j1_<stem>_step6_grain*），
    # 因此源片统一叫 clip.mkv。
    av, fps, sr = _build_flash_click(tmp_path, audio_offset=0.0)
    src = tmp_path / "clip.mkv"
    shutil.copyfile(av, src)
    # --- 2. 准备 stage7 需要的 temp 输入 ---
    temp = tmp_path / "temp"
    out = tmp_path / "out"
    logs = tmp_path / "logs"
    for d in (temp, out, logs):
        d.mkdir(exist_ok=True)
    step6 = temp / "j1_clip_step6_grain.mkv"
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:v", "ffv1",
         "-pix_fmt", "yuv444p16le", str(step6)],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[:200]
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-c:a", "copy",
         str(temp / "j1_clip.mka")],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[:200]

    params = {"temp_dir": str(temp), "output_dir": str(out),
              "log_dir": str(logs), "duration": 3,
              "audio_denoise": False, "hum_freq": 0, "input_bit_depth": 16}
    cmd = [sys.executable,
           os.path.join(ROOT, "video_restoration_pipeline", "stages",
                        "stage7_encode.py"),
           "--input", str(src), "--job-id", "j1",
           "--params", json.dumps(params)]
    r = subprocess.run(cmd, cwd=str(tmp_path), capture_output=True, text=True,
                       timeout=300)
    final = out / "clip_restored.mkv"
    assert r.returncode == 0, f"stage7 失败：{r.stderr[:400]}"
    assert final.exists(), "没有最终产物"

    # --- 3. 从最终产物回读事件 ---
    v_events = _video_event_times(final, fps)
    a_events = _audio_event_times(final, 48000)
    assert len(v_events) >= 2, f"产物里没检出 flash：{v_events}"
    assert len(a_events) >= 2, f"产物里没检出 click：{a_events}"

    deltas = [min(abs(a - vt) for a in a_events) for vt in v_events]
    worst = max(deltas)
    # 容差 150ms：AAC priming(约 23ms) + 编码/容器对齐都会带来几十毫秒量级偏移
    assert worst < 0.15, (
        f"经过真实 stage7 后，flash 与 click 的最大时间差 {worst:.3f}s "
        f"超过 150ms（video={v_events} audio={a_events}）")


# ---------------------------------------------------------------------------
# 第四轮交付要求：阶段衔接 / 正式 CLI 联动 / 真实产物并发归属
# ---------------------------------------------------------------------------

def test_r301_stage_artifact_roundtrip_not_confused_by_sidecars(tmp_path):
    """阶段衔接：**stage 产物 → 登记侧车 → 下游消费 → 清理** 全链路。

    B10 只证明了"stage7 不被侧车骗到"；这条把调度器侧也纳入：
    登记会产生侧车，下游枚举必须仍然只看媒体，清理时两者一起消失。
    """
    import checkpoint
    temp = tmp_path / "temp"
    temp.mkdir()
    src = tmp_path / "src.mkv"
    src.write_bytes(b"SOURCE")
    art = temp / "j1_src_step6_grain.mkv"
    art.write_bytes(b"MEDIA-CONTENT")
    job = {"job_id": "j1", "input_path": str(src)}

    # 1) 登记侧车
    assert vmain.record_stage_artifact_identity(str(art), job, {}, 6, str(src))
    sidecar = vmain.intermediate_identity_path(str(art))
    assert os.path.exists(sidecar)

    # 2) 下游枚举不得选中侧车，且仍能选中媒体
    found = vmain.find_stage_artifacts(str(temp), "j1", str(src), 6)
    assert found == [str(art)], f"下游枚举选中了非媒体文件：{found}"

    # 3) 侧车必须与媒体匹配（同一输入/参数下）
    ok, _why = vmain.intermediate_identity_matches(str(art), job, {})
    assert ok, "刚登记的身份却在校验时被拒"

    # 4) 清理：媒体与侧车一起消失，不留孤儿 JSON
    checkpoint.cleanup_all_intermediates(str(temp), "j1")
    assert not art.exists()
    assert not os.path.exists(sidecar), "媒体已删但侧车残留（孤儿 JSON）"
    assert not any(p.suffix == ".json" for p in temp.iterdir()), \
        "清理后仍有 JSON 残留"


def test_r304_startup_decision_prevents_reexecution(tmp_path, monkeypatch):
    """正式 CLI 联动：启动期把任务判成 DONE 后，不得再执行它。

    B02 用受控替换隔离了根因；这条跑**真正的 vmain.main()**，
    验证"重新读取 manifest"这条修复在实际调度路径上生效。
    """
    path = _cli_env(tmp_path, monkeypatch, 0)

    # 启动期决议（与 resume/FAILED 降级同一入口）把它置为 DONE
    assert vmain._apply_startup_job_decision(
        path, "j1", {"status": "DONE"}, ["PENDING"]), \
        "启动期决议没有写成功"

    calls = []

    def process(job, *_a, **_k):
        calls.append(dict(job))
        return 0, ""

    monkeypatch.setattr(vmain, "process_job", process)
    try:
        vmain.main()
    except SystemExit:
        pass
    assert calls == [], f"磁盘已 DONE 的任务仍被送进 process_job：{calls}"


def _contend_artifact_worker(path, owner, artifact, barrier):
    """子进程：争抢同一 job，并把**自己的 owner 写进真实产物**。"""
    import conftest  # noqa: F401
    import manifest_loader as m

    barrier.wait()
    gen = m.acquire_run_ownership(path, owner=owner)
    if gen is None:
        return None
    # 真实产物：内容标出是谁写的
    tmp = artifact + f".{owner}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(f"PRODUCED-BY-{owner}")
    import os as _os
    _os.replace(tmp, artifact)
    m.save_manifest_job_result(path, "j1", "DONE", owner=owner,
                               generation=gen)
    return owner


def test_r204_contending_workers_artifact_belongs_to_winner(tmp_path):
    """真实产物归属：并发争抢后，磁盘上的产物必须属于胜出的 owner。

    已有的 `..._stay_consistent` 只验证 manifest 字段；评估指出那不足以
    证明**真实媒体产物**与合法 owner 一致。这条让每个 worker 把可识别的
    内容写进产物，最后核对"文件里的 owner == manifest 里的 owner"。
    """
    import multiprocessing

    path = _manifest_with_jobs(tmp_path / "m.yaml", ["j1"])
    artifact = str(tmp_path / "j1_product.bin")

    workers = 4
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(workers)
    procs = []
    for i in range(workers):
        p = ctx.Process(target=_contend_artifact_worker,
                        args=(path, f"w{i}", artifact, barrier))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=180)
        assert p.exitcode == 0, f"worker 失败：exitcode={p.exitcode}"

    final = ml.load_manifest(path)
    job = final["jobs"][0]
    assert job["status"] == "DONE"
    owner = str(job.get("run_owner"))

    with open(artifact, encoding="utf-8") as f:
        content = f.read()
    assert content == f"PRODUCED-BY-{owner}", (
        f"产物内容（{content}）与 manifest 记录的 owner（{owner}）不一致 —— "
        "最终文件不是合法 owner 产出的")
