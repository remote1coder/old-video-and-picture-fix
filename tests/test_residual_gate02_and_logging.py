# -*- coding: utf-8 -*-
"""残留问题回归：探测失败必须算 error、硬错误与模式无关都要拦。

核查报告 GATE-02 里还有两个子问题，本轮补修：

1. `probe_media_facts()` 只在"文件不存在"时设 `error`。ffprobe 超时 /
   非零退出 / 输出解析失败时，facts **全是 None 但 error 仍为 None**，
   于是 verify_output 里所有判据都变成 ok=None，最终 `ok = not errors`
   仍然 True —— "什么都没量到"反而通过。
2. `BLANK_FRAMES`（源有内容、成品纯色）只是 error 级 issue，而
   `_fullchain_blockers` 只看 `checks` 与 `error`，于是 frames_only 模式下
   黑图也不会阻断。黑图/不可读属于**跨模式硬错误**，不该被模式放宽。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import artifact_verify as av  # noqa: E402


def test_probe_sets_error_when_nothing_usable(tmp_path):
    """完全读不出事实时必须留下 error，而不是静默返回全 None。"""
    p = tmp_path / "garbage.mkv"
    p.write_bytes(b"not a media file" * 200)
    facts = av.probe_media_facts(str(p))
    assert facts.get("error"), f"应设置 error，实际 facts={facts}"


def test_missing_file_sets_error(tmp_path):
    facts = av.probe_media_facts(str(tmp_path / "nope.mkv"))
    assert facts.get("error") == "file not found"


def test_verify_output_unreadable_is_not_ok(tmp_path):
    """源不可读 -> ok=False 且 issues 里有 PROBE_UNUSABLE。"""
    bad = tmp_path / "bad.mkv"
    bad.write_bytes(b"garbage" * 300)
    good = tmp_path / "good.mkv"
    good.write_bytes(b"another garbage" * 300)
    rep = av.verify_output(str(bad), str(good), {})
    assert rep["ok"] is False, rep
    codes = [i.get("code") for i in rep.get("issues") or []]
    # 源不可读时已有更具体的 SOURCE_UNREADABLE；探测整体不可用是
    # PROBE_UNUSABLE。两者都算"未验收"，这里接受任一。
    assert ("SOURCE_UNREADABLE" in codes
            or "PROBE_UNUSABLE" in codes), codes
    assert rep.get("error"), "顶层 error 不能为空"


def test_verify_output_blank_output_is_hard_error(tmp_path, monkeypatch):
    """源有内容而成品纯色 -> BLANK_FRAMES，且标记为硬错误。"""
    def fake_facts(path):
        if "src" in path:
            return {"exists": True,
                    "width": 640, "height": 480, "frames": 10,
                    "duration": 1.0, "pix_fmt": "yuv420p16le",
                    "first_frame_hash": "0000000000000000",
                    "last_frame_hash": "0000000000000000",
                    "mean": 120.0, "std": 40.0, "error": None}
        return {"exists": True,
                "width": 1280, "height": 960, "frames": 10,
                "duration": 1.0, "pix_fmt": "yuv420p16le",
                "first_frame_hash": "0000000000000000",
                    "last_frame_hash": "0000000000000000",
                "mean": 0.0, "std": 0.0, "error": None}

    monkeypatch.setattr(av, "probe_media_facts", fake_facts)
    rep = av.verify_output(str(tmp_path / "src.mp4"),
                           str(tmp_path / "out.mp4"), {})
    codes = [i.get("code") for i in rep.get("issues") or []]
    assert "BLANK_FRAMES" in codes, codes
    assert rep["ok"] is False
    assert rep.get("hard_error") == "BLANK_FRAMES", rep.get("hard_error")


def test_fullchain_blocks_hard_error_in_frames_only():
    """frames_only 模式也必须拦硬错误（不能因为放宽时长就放过黑图）。"""
    import stage7_encode as s7
    got = s7._fullchain_blockers({"frames": {"ok": True}}, "frames_only",
                                 error="BLANK_FRAMES")
    assert got == ["ERROR"], got


def test_sync_log_grading_has_no_shadowed_branch():
    """回归护栏：>100ms 分支不能再被 `diff_ms > 0` 遮蔽。

    原实现顺序是 `>0` 在前，于是任何正差值都先命中，写出硬编码的
    "exceeds 300ms threshold"，而真正的 >100ms 告警永远不触发。
    """
    src = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                       "stage7_encode.py")
    text = open(src, encoding="utf-8", errors="replace").read()
    # 允许出现在注释里（说明历史），但不能有一行**代码**写着它
    for line in text.splitlines():
        if "exceeds 300ms" in line:
            assert line.lstrip().startswith("#"), \
                f"仍有代码行硬编码 300ms: {line.strip()[:80]}"
    i100 = text.find("STAGE7_SYNC_WARN_100")
    iminor = text.find("STAGE7_SYNC_MINOR")
    assert i100 != -1 and iminor != -1, "两个分支都应存在"
    assert i100 < iminor, ">100ms 分支必须在剩余区间之前，否则又会被遮蔽"


def test_verifier_exception_blocks_instead_of_passing():
    """校验器自己抛异常时必须阻断发布，不能只记 warning（fail-open）。

    另外 `_fc_blocked` 必须在 try 之前初始化：否则异常跳到 except 后，
    后面的 `if _fc_blocked:` 会 NameError，把"校验异常"升级成整个 stage 崩溃。
    """
    src = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                       "stage7_encode.py")
    text = open(src, encoding="utf-8", errors="replace").read()

    i_init = text.find("_fc_blocked = []")
    i_try = text.find("from artifact_verify import verify_output as _fc_verify")
    assert i_init != -1 and i_try != -1
    assert i_init < i_try, "_fc_blocked 必须在校验 try 之前初始化"

    # 异常分支必须赋值阻断项，且用 error 级
    i_exc = text.find("STAGE7_FULLCHAIN_VERIFY_ERROR")
    assert i_exc != -1, "异常分支应使用 ERROR 级事件"
    window = text[i_exc:i_exc + 500]
    assert '_fc_blocked = ["VERIFY_ERROR"]' in window, \
        "校验器异常必须标记为阻断项"
