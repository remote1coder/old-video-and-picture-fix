# -*- coding: utf-8 -*-
"""R16-03：一代成果必须能冻结，标记与回拷都不能混代。

审核确认的两条**无锁交错**（本次未做真实并发复现，但代码路径确定）：

1. `stage7` 的 WSL 回拷：第一个 publisher 返回时已释放锁，随后从**共享正式
   路径**复制媒体、再复制 verify/fullchain。A 刚拷完媒体、B 发布了下一代，
   A 就会拷到 B 的报告 → 第二个事务把 A 媒体与 B 报告一起提交。
   而且原实现"报告不存在就 `continue`"，缺报告会被静默跳过。

2. `main` 的完成标记：stage7 结束后**锁外**重读共享正式媒体，配上本次
   job 的输入/参数写 `.done.json`。另一进程可能已替换目标，
   使 A 的输入身份绑定到 B 的结果。`.done.json.tmp` 还是共享名。

本文件用**真实函数**覆盖完成标记的拒绝路径，并用源码守卫锁住回拷的冻结语义。
"""
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import main as vmain  # noqa: E402

_STAGE7 = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                       "stage7_encode.py")


# ------------------------------------------------ 完成标记：代际核对

def _job(inp):
    return {"job_id": "j1", "input_path": inp, "detected_class": "HDV_AVCHD",
            "recommended_params": {}, "user_overrides": {}}


def test_marker_refuses_when_media_changed_after_publish(tmp_path):
    """媒体被下一代替换后，**不许**把本 job 的身份写进标记。

    这是审核第 2 条的直接反例：锁外重读会读到别人的结果。
    """
    media = tmp_path / "clip_restored.mkv"
    media.write_bytes(b"GENERATION-B")
    inp = tmp_path / "in.mp4"
    inp.write_bytes(b"src")

    # verify.json 记录的是**本代**发布时的摘要（A 代）
    other = tmp_path / "other.bin"
    other.write_bytes(b"GENERATION-A")
    (tmp_path / "clip_restored.mkv.verify.json").write_text(
        json.dumps({"media_sha256": vmain._chunked_sha256(str(other))}),
        encoding="utf-8")

    class L:
        def __init__(self):
            self.events = []

        def info(self, e, m="", **k):
            self.events.append(e)

        def warning(self, e, m="", **k):
            self.events.append(e)

        def error(self, e, m="", **k):
            self.events.append(e)

    log = L()
    ok = vmain.write_completion_marker(str(media), _job(str(inp)), {}, log)

    assert ok is False, "媒体已换代，标记必须拒绝写入"
    assert "COMPLETION_MARKER_GENERATION_CHANGED" in log.events, log.events
    # 关键：不能留下一个把 A 的身份绑到 B 的标记
    assert not os.path.exists(vmain.completion_marker_path(str(media)))


def test_marker_written_when_digest_matches(tmp_path):
    """摘要一致时标记正常写入，并记录**已核对**的摘要。"""
    media = tmp_path / "clip_restored.mkv"
    media.write_bytes(b"GENERATION-A")
    dig = vmain._chunked_sha256(str(media))
    (tmp_path / "clip_restored.mkv.verify.json").write_text(
        json.dumps({"media_sha256": dig}), encoding="utf-8")
    inp = tmp_path / "in.mp4"
    inp.write_bytes(b"src")

    class L:
        def info(self, *a, **k):
            pass
        warning = error = info

    assert vmain.write_completion_marker(str(media), _job(str(inp)),
                                         {}, L()) is True
    marker = json.load(open(vmain.completion_marker_path(str(media)),
                            encoding="utf-8"))
    assert marker["output_sha256"] == dig
    assert marker["job_id"] == "j1"


def test_marker_tmp_name_is_private():
    """`.done.json.tmp` 是共享名，并发会互相踩 → 必须带 pid/随机后缀。"""
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "main.py"),
               encoding="utf-8", errors="replace").read()
    assert ("completion_marker_path(final_path)}.tmp-{os.getpid()}"
            in src), "标记的临时文件名必须私有化"
    assert 'completion_marker_path(final_path) + ".tmp"' not in src, \
        "不得再用共享的 `<final>.done.json.tmp`"


# ------------------------------------------------ WSL 回拷：冻结语义

def test_wsl_copyback_uses_frozen_digests_and_full_set():
    """WSL 回拷必须在源锁内、按发布前冻结的摘要复制，且集合与发布一致。

    审核的两条具体要求：
      * 不能从"共享正式路径上碰巧是什么"复制（会 A 媒体 + B 报告）；
      * 缺报告不能 `continue` 静默跳过（目标侧会成为不完整的一代）。
    """
    src = open(_STAGE7, encoding="utf-8", errors="replace").read()

    assert "_pub_digests" in src, "必须在发布前冻结本代身份"
    assert "acquire_target_locks([output_path]" in src or \
           "_acq_src([output_path]" in src, "回拷必须在源端锁内进行"
    assert "STAGE7_WSL_SOURCE_CHANGED" in src, \
        "源被换代时必须中止并报出这个事件"

    # 反例：老的"存在就拷、不存在就 continue"必须消失
    import re
    m = re.search(r"if not os\.path\.exists\(_src\):\s*\n\s*continue", src)
    assert m is None, "缺报告仍被静默跳过（WSL 目标侧会是不完整的一代）"

    # 复制后必须再核对一次摘要
    assert src.count("_fdig(_stage_path) != _want") >= 1, \
        "复制后必须核对摘要，否则复制期间的替换检测不到"
