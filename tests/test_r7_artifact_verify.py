"""产物核对：让"整批偏移一位"第一次跑就被拦住。

**它要解决的问题**

2026-10-19 的 60 帧 → 59 帧 bug，根因只有一行 `- 1`，我猜了三轮才找到。
如果跑完就有一道机械核对，第一次就能拦住。

**设计被实测修正过两次，必须写清**

1. **第一版给了 ±1 帧容差**（`allowed = {n-1, n, n+1}`），本意是容忍
   编码器尾部处理。结果恰好放过了这个 bug —— `{59,60,61}` 里包含 59，
   核对直接通过。**容差不能这么给。**
2. **第一版把感知哈希当偏移检测器**。实测（testsrc2，64bit pHash）：

   | 对比 | 距离 |
   |---|---|
   | 同源相邻帧 | 17 / 14 |
   | 源 vs 成品**同帧** | 14 / 12 |
   | 源 vs 成品**偏移一位** | 17 / 27 |

   正常差异和偏移差异**区间重叠** —— 相邻帧本身像的素材上，pHash
   分不出来。所以它只能报 warning，不能当错误判据。

**现在靠什么判**

* 帧数（精确集合，差 1 帧单独报 `FRAME_COUNT_OFF_BY_ONE`）；
* 时长（±3%）；
* 首末帧画面（warning 级，用于提示"值得看一下"）；
* 非黑图（error 级）。
"""

import os
import subprocess
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "shared_modules") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "shared_modules"))

import artifact_verify as av  # noqa: E402

HAS_FFMPEG = bool(__import__("shutil").which("ffmpeg"))
SRC = r"E:/codex/_repair_r4_20261013/real_run/input/real2s.mov"
GOOD = r"E:/codex/_round7_20261019/perf/out/real2s_restored.mkv"
BAD59 = r"E:/codex/_round7_20261019/v59/only59.mkv"


# ---------------------------------------------------------------------------
# 帧数允许集合：不能有 ±1 容差
# ---------------------------------------------------------------------------

def test_expected_counts_has_no_plus_minus_one_slack():
    """60 帧输入的允许集合必须是 {60}，不能是 {59,60,61}。"""
    allowed = av.expected_frame_counts(60, {"deinterlace": "QTGMC",
                                           "qtgmc_fps_divisor": 1})
    assert 60 in allowed
    assert 59 not in allowed, "±1 容差会放过整批丢一帧（2026-10-19 的 bug）"
    assert 61 not in allowed


def test_expected_counts_allows_bob_doubling():
    """bob 会让帧数翻倍，这属于合法变换。"""
    allowed = av.expected_frame_counts(60, {"deinterlace": "BOB"})
    assert 120 in allowed


def test_expected_counts_allows_ivtc_4_of_5():
    """IVTC 后帧数 = 4/5。"""
    allowed = av.expected_frame_counts(60, {"ivtc": True,
                                           "vdecimate_cycle": 5})
    assert 48 in allowed


def test_expected_counts_unknown_when_source_frames_missing():
    """源帧数未知 → 返回 None，调用方必须标"未校验"。"""
    assert av.expected_frame_counts(None, {}) is None
    assert av.expected_frame_counts(0, {}) is None


# ---------------------------------------------------------------------------
# 感知哈希：能跨分辨率，但不能当偏移检测器
# ---------------------------------------------------------------------------

def test_hash_distance_basics():
    assert av.hash_distance("0000000000000000", "0000000000000000") == 0
    assert av.hash_distance("0000000000000000", "000000000000000f") == 4
    assert av.hash_distance(None, "0") is None


def test_phash_survives_rescaling(tmp_path):
    """同一画面缩放到不同尺寸，pHash 必须接近（这是它能跨 1080p/4K 的原因）。"""
    if not HAS_FFMPEG:
        pytest.skip("需要 ffmpeg")
    src = tmp_path / "a.png"
    small = tmp_path / "b.png"
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi",
                        "-i", "testsrc2=s=640x360", "-frames:v", "1",
                        "-y", str(src)], capture_output=True)
    if r.returncode != 0:
        pytest.skip("生成失败")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(src),
                    "-vf", "scale=1280:720", "-y", str(small)],
                   capture_output=True)
    d = av.hash_distance(av.perceptual_hash(str(src)),
                         av.perceptual_hash(str(small)))
    assert d is not None and d <= 4, f"缩放后 pHash 距离 {d} 偏大"


# ---------------------------------------------------------------------------
# 端到端：好产物通过，59 帧产物必须不通过
# ---------------------------------------------------------------------------

pytestmark_real = pytest.mark.skipif(
    not (HAS_FFMPEG and os.path.exists(SRC) and os.path.exists(GOOD)),
    reason="需要真实样本")


@pytestmark_real
def test_good_output_passes():
    r = av.verify_output(SRC, GOOD, {"deinterlace": "QTGMC",
                                     "qtgmc_fps_divisor": 1})
    assert r["ok"] is True, av.render_verdict(r)
    assert r["checks"]["frames"]["ok"] is True
    assert r["checks"]["frames"]["allowed"] == [60]


@pytest.mark.skipif(not (HAS_FFMPEG and os.path.exists(SRC)
                         and os.path.exists(BAD59)),
                    reason="需要真实样本（源 + 59 帧成品）")
def test_off_by_one_output_fails_with_specific_code():
    """59 帧（模拟丢帧）必须报 FRAME_COUNT_OFF_BY_ONE，而不是默默通过。"""
    r = av.verify_output(SRC, BAD59, {"deinterlace": "QTGMC",
                                      "qtgmc_fps_divisor": 1})
    assert r["ok"] is False
    codes = [i["code"] for i in r["issues"] if i["level"] == "error"]
    assert "FRAME_COUNT_OFF_BY_ONE" in codes, codes


# ---------------------------------------------------------------------------
# 自包含版本：不依赖任何仓库外素材，用 lavfi 现场生成
# ---------------------------------------------------------------------------
#
# 为什么要有这一组：上面的用例依赖三个外部文件（SRC / GOOD / BAD59），
# 2026-10-22 磁盘清理把它们删掉后，这组用例整体失效（SRC 缺失时甚至
# 不是跳过、而是假装通过）。帧数判据是**纯算术**，与画面内容无关，
# 完全可以自包含验证 —— 这样它永远不会再被外部文件的状态绑架。
#
# 注意两组并存的理由：真实样本那组额外覆盖「真实编码产物能否通过」
# （含 pHash / 时长 / 黑图判据在真实数据上的表现），自包含组只保证
# 帧数判据本身不回退。


def _mk_video(path, n_frames, extra_vf=""):
    """用 lavfi 生成精确 n_frames 帧的 h264 视频。

    注意 `d` 必须足够长：`-frames:v` 只能**截断**、不能补足，
    原料帧数不足时会静默给出更少的帧（2026-10-22 实测踩到：
    `d=1:r=30` 只产 30 帧，请求 60 帧实际只得 30）。
    这里按请求帧数 +2s 余量给原料。
    """
    need_d = n_frames / 30.0 + 2.0
    vf = f"scale=1920:1080:flags=lanczos{extra_vf}"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi",
           "-i", f"testsrc2=s=640x360:d={need_d:.1f}:r=30",
           "-frames:v", str(n_frames),
           "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", "-y", path]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       errors="replace", timeout=180)
    if r.returncode != 0 or not os.path.exists(path):
        pytest.skip(f"无法生成测试视频: {(r.stderr or '')[:150]}")
    # 防御：帧数不对就明说，不要让后续断言报看不懂的错
    got = av.probe_media_facts(path).get("frames")
    if got != n_frames:
        pytest.skip(f"生成的帧数不符（要 {n_frames}，实得 {got}）")
    return path


@pytest.mark.skipif(not HAS_FFMPEG, reason="需要 ffmpeg")
def test_selfcontained_frame_count_off_by_one(tmp_path):
    """自包含：源 60 帧 / 产物 59 帧 → FRAME_COUNT_OFF_BY_ONE。

    与上面的真实样本版断言同一个判据，但不依赖任何外部文件。
    """
    src = _mk_video(str(tmp_path / "src.mkv"), 60)
    bad = _mk_video(str(tmp_path / "bad59.mkv"), 59)

    r = av.verify_output(src, bad, {"deinterlace": "QTGMC",
                                    "qtgmc_fps_divisor": 1})
    assert r["ok"] is False
    codes = [i["code"] for i in r["issues"] if i["level"] == "error"]
    assert "FRAME_COUNT_OFF_BY_ONE" in codes, codes


@pytest.mark.skipif(not HAS_FFMPEG, reason="需要 ffmpeg")
def test_selfcontained_frame_count_mismatch_is_error(tmp_path):
    """自包含：差 2 帧以上 → FRAME_COUNT_MISMATCH（不是 OFF_BY_ONE）。"""
    src = _mk_video(str(tmp_path / "src.mkv"), 60)
    bad = _mk_video(str(tmp_path / "bad55.mkv"), 55)

    r = av.verify_output(src, bad, {"deinterlace": "QTGMC",
                                    "qtgmc_fps_divisor": 1})
    assert r["ok"] is False
    codes = [i["code"] for i in r["issues"] if i["level"] == "error"]
    assert "FRAME_COUNT_MISMATCH" in codes, codes
    assert "FRAME_COUNT_OFF_BY_ONE" not in codes, \
        "差 5 帧不该报 OFF_BY_ONE（那是专属于差 1 帧的）"


@pytest.mark.skipif(not HAS_FFMPEG, reason="需要 ffmpeg")
def test_selfcontained_exact_frame_count_passes(tmp_path):
    """自包含：帧数精确相等时帧数判据必须通过（正例，防误报）。"""
    src = _mk_video(str(tmp_path / "src.mkv"), 60)
    ok = _mk_video(str(tmp_path / "ok60.mkv"), 60)

    r = av.verify_output(src, ok, {"deinterlace": "QTGMC",
                                   "qtgmc_fps_divisor": 1})
    assert r["checks"]["frames"]["ok"] is True, av.render_verdict(r)
    assert r["checks"]["frames"]["allowed"] == [60]
    # 帧数判据通过即整体通过（pHash/时长在自包含同源素材上应无告警）
    assert r["ok"] is True, av.render_verdict(r)


@pytest.mark.skipif(not (HAS_FFMPEG and os.path.exists(BAD59)),
                    reason="需要 59 帧成品样本")
def test_unreadable_source_is_reported_instead_of_silent_pass():
    """源不可读时必须报 SOURCE_UNREADABLE，绝不能"一项没核却判通过"。

    2026-10-22 实测：verify_output 原本只判 `out["exists"]`，源被删掉后
    5 项判据全部退化成 `ok: None`（"未校验"），issues 里零 error
    → 返回 ok=True。核对是最后一道防线，它自己必须会报错。
    """
    r = av.verify_output("/nonexistent/src.mov", BAD59,
                         {"deinterlace": "QTGMC", "qtgmc_fps_divisor": 1})
    assert r["ok"] is False
    codes = [i["code"] for i in r["issues"] if i["level"] == "error"]
    assert "SOURCE_UNREADABLE" in codes, codes


def test_missing_output_is_reported():
    r = av.verify_output(SRC, "/nonexistent/out.mkv", {})
    assert r["ok"] is False
    assert r["issues"][0]["code"] == "OUTPUT_MISSING"


def test_render_verdict_is_readable():
    r = {"ok": False,
         "source": {"frames": 60, "duration": 2.0, "width": 1920,
                    "height": 1080},
         "output": {"frames": 59, "duration": 1.99, "width": 3840,
                    "height": 2160, "pix_fmt": "yuv420p10le"},
         "checks": {"frames": {"src": 60, "out": 59, "allowed": [60],
                               "ok": False}},
         "issues": [{"level": "error", "code": "FRAME_COUNT_OFF_BY_ONE",
                     "message": "帧数差 1"}]}
    text = av.render_verdict(r)
    assert "不通过" in text
    assert "FRAME_COUNT_OFF_BY_ONE" in text


# ---------------------------------------------------------------------------
# 接入守卫
# ---------------------------------------------------------------------------

def test_stage7_runs_the_check_and_fails_with_86():
    """stage7 必须在编码成功后核对，失败时返回 86（验证失败）。"""
    src = open(os.path.join(ROOT, "video_restoration_pipeline", "stages",
                            "stage7_encode.py"), encoding="utf-8").read()
    assert "from artifact_verify import verify_output" in src
    assert "STAGE7_VERIFY_OK" in src
    assert "STAGE7_VERIFY_FAIL" in src
    assert "return 86" in src, "核对失败必须让运行不算成功"
    # 核对器自身异常不能把成功编码判死
    assert "STAGE7_VERIFY_ERROR" in src


# ---------------------------------------------------------------------------
# 逐帧对齐校验：只在"帧序不变"的两端可靠
# ---------------------------------------------------------------------------

def test_alignment_uses_relative_comparison_not_absolute_similarity():
    """判据必须是"自己那帧比邻居更像"，不是"绝对相似度够高"。

    实测依据：合成素材上 `src[k] vs out[k]` 的绝对距离和
    `src[k] vs out[k+1]` 只差 0.001，绝对判据分不开；相对比较不受影响。
    """
    src = open(os.path.join(ROOT, "shared_modules", "artifact_verify.py"),
               encoding="utf-8").read()
    assert "best_nbr" in src
    assert "self_sim >= best_nbr" in src
    # 比例判据（不是"每个点都必须过"）
    assert "ALIGNMENT_FLIP_RATIO" in src
    # 判别力过滤（相邻帧太像就跳过）
    assert "MIN_SRC_FRAME_DIFF" in src
    assert "discriminative" in src
    # 可判定点太少时不下结论
    assert "MIN_ALIGNMENT_SAMPLES" in src


def test_alignment_documents_its_scope():
    """必须写明"不能拿它比源 vs 全链成品"—— 去隔行会改变时间对应。"""
    src = open(os.path.join(ROOT, "shared_modules", "artifact_verify.py"),
               encoding="utf-8").read()
    assert "QTGMC" in src
    assert "不要拿它去比" in src or "不该用在" in src
    assert "stage7" in src


def test_alignment_rejects_too_short_clips():
    r = av.verify_frame_alignment("E:/does/not/exist.mkv",
                                  "E:/does/not/exist2.mkv")
    assert r["ok"] is None
    assert r["error"]


@pytest.mark.skipif(not (os.path.exists(GOOD)
                         and os.path.exists(r"E:/codex/_round7_20261019/align/reenc.mkv")),
                    reason="需要重编码样本")
def test_alignment_passes_on_reencode():
    """纯重编码（帧序不变）必须判为对齐 —— 这是它唯一可靠的用法。"""
    r = av.verify_frame_alignment(
        GOOD, r"E:/codex/_round7_20261019/align/reenc.mkv", samples=4)
    assert r["ok"] is True, av.render_alignment(r)
    assert r["flipped"] == 0


@pytest.mark.skipif(not (os.path.exists(GOOD)
                         and os.path.exists(r"E:/codex/_round7_20261019/align/shifted.mkv")),
                    reason="需要错位样本")
def test_alignment_catches_shifted_output():
    """整体前移一位必须被抓出来。"""
    r = av.verify_frame_alignment(
        GOOD, r"E:/codex/_round7_20261019/align/shifted.mkv", samples=4)
    assert r["ok"] is not True, av.render_alignment(r)
