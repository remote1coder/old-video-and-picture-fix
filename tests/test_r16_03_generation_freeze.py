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
           os.path.join(_ROOT, "video_restoration_pipeline"),
           # R22-A01：夹具改用真实生产者 `build_marker_candidate`（在 stages/ 下）
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import main as vmain  # noqa: E402

_STAGE7 = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                       "stage7_encode.py")


# ------------------------------------------------ 完成标记：代际核对

def _job(inp):
    return {"job_id": "j1", "input_path": inp, "detected_class": "HDV_AVCHD",
            "recommended_params": {}, "user_overrides": {}}


def _publish_full(tmp_path, *, media=b"GENUINE", attempt="att1",
                  input_bytes=b"src"):
    """用生产函数发布一整代，返回 ``(media_path, params)``（R22-A01）。"""
    import publisher as _P
    import stage7_encode as _s7

    out = str(tmp_path / "clip_restored.mkv")
    stage = str(tmp_path / "staging")
    inp = str(tmp_path / "in.mp4")
    if not os.path.exists(inp):
        with open(inp, "wb") as f:
            f.write(input_bytes)
    job = _job(inp)

    os.makedirs(stage, exist_ok=True)
    with open(os.path.join(stage, "media.bin"), "wb") as f:
        f.write(media)
    with open(os.path.join(stage, "verify.json"), "w", encoding="utf-8") as f:
        # R24-02：报告要带检查结论与发布决定（真实 stage7 就是这么写的）
        json.dump({"attempt_id": "att1",
            "run_identity": {"schema": "run-identity/1", "digest": "0" * 64},
            "media_sha256": vmain._plain_sha256(
            os.path.join(stage, "media.bin")),
            "checks": {
                "mechanical": {"name": "mechanical", "status": "PASS",
                               "required": True},
                "frame_order": {"name": "frame_order", "status": "PASS",
                                "required": True},
                "final_timeline": {"name": "final_timeline", "status": "PASS",
                                   "required": True},
                "fullchain": {"name": "fullchain", "status": "PASS",
                              "required": True},
            },
            "publish_decision": {"publish": True, "verified": True},
        }, f)
    with open(os.path.join(stage, "fullchain.json"), "w",
              encoding="utf-8") as f:
        json.dump({"ok": True, "attempt_id": attempt, "checks": {"frames": {"ok": True}, "duration": {"ok": True}, "not_black": {"ok": True}}}, f)

    arts = [
        _P.Artifact("media", os.path.join(stage, "media.bin"), out),
        _P.Artifact("verify", os.path.join(stage, "verify.json"),
                    out + ".verify.json"),
        _P.Artifact("fullchain", os.path.join(stage, "fullchain.json"),
                    out + ".fullchain.json"),
    ]
    digests = [(a.name, a.final, vmain._plain_sha256(a.staging)) for a in arts]
    eff = vmain.merge_params(job, {}, {}, None)
    eff["attempt_id"] = attempt
    seed = {
        "impl": vmain.VIDEO_IMPL_VERSION,
        "job_id": job["job_id"],
        "input_path": inp,
        "input_identity": vmain.build_input_identity(inp),
        "params_identity": vmain.build_params_identity({}, job, None,
                                                       params=eff),
    }
    _mk, mk_stage, mk_dig, _pm = _s7.build_marker_candidate(
        seed=seed, pub_digests=digests, attempt_id=attempt,
        job_id=job["job_id"], output_path=out, media_digest=digests[0][2],
        media_size=os.path.getsize(arts[0].staging))
    arts.append(_P.Artifact("marker", mk_stage, out + ".done.json"))
    # R23-B01：真实 stage7 把 RunIdentity 摘要交给 publisher；回执里有它
    # 才能被"当前计划身份"核对。夹具此前没传，回执的 run_identity 恒为空。
    _ri0 = vmain.build_run_identity({}, job, None, params=eff)
    res = _P.publish(arts, out + ".journal.json", str(tmp_path / "backup"),
                     attempt_id=attempt, decision={"publish": True},
                     allowed_output_roots=[str(tmp_path)],
                     required_objects=[a.final for a in arts],
                     marker_digest=mk_dig,
                     run_identity=str(_ri0["digest"]))
    assert res.ok, res.errors

    rp = str(tmp_path / "stage7_result.json")
    eff["_stage7_result_path"] = rp
    with open(rp, "w", encoding="utf-8") as f:
        json.dump({
            "schema": 1, "attempt_id": attempt, "ok": True,
            "terminal_pending": False, "journal_phase": "COMMITTED",
            "journal_path": out + ".journal.json",
            "generation_receipt": (res.generation_receipt.to_dict()
                                   if res.generation_receipt else None),
            "final_output_path": out,
            "publish_result": res.as_dict(),
            "run_identity": vmain.build_run_identity({}, job, None,
                                                     params=eff),
        }, f, ensure_ascii=False, default=str)
    return out, eff


class _L:
    def __init__(self):
        self.events = []

    def _rec(self, e, m="", **k):
        self.events.append(e)

    info = warning = error = critical = _rec


def test_marker_refuses_when_media_changed_after_publish(tmp_path):
    """媒体被换 → 验证必须拒绝（R22-A01：拒绝点在 payload 摘要）。

    旧版断言的是"不写标记 + COMPLETION_MARKER_GENERATION_CHANGED"；
    现在 main 不写标记，判据由 `verify_generation_completion` 给出，
    事件统一为 `COMPLETION_UNPROVEN`（附具体原因）。
    """
    media, params = _publish_full(tmp_path)
    with open(media, "wb") as f:
        f.write(b"OTHER-BYTES")      # 与 GENUINE 同为 8 字节

    log = _L()
    ok = vmain.write_completion_marker(str(media), _job(str(tmp_path / "in.mp4")),
                                       {}, params=params, logger=log)
    assert ok is False
    assert "COMPLETION_UNPROVEN" in log.events, log.events


def test_marker_is_verified_when_everything_matches(tmp_path):
    """摘要一致、journal 已提交、私有 result 齐全 → 验证通过。"""
    media, params = _publish_full(tmp_path)
    log = _L()
    ok = vmain.write_completion_marker(str(media), _job(str(tmp_path / "in.mp4")),
                                       {}, params=params, logger=log)
    assert ok is True, log.events
    assert "COMPLETION_VERIFIED" in log.events, log.events
    # marker 是生产者在事务里写的，验证不该重写它
    assert os.path.exists(vmain.completion_marker_path(str(media)))


def test_marker_staging_name_is_attempt_private():
    """暂存名必须带 attempt —— 两个 process 不会同名（R22 迁移）。

    写入已从 `main` 移到 `stage7.build_marker_candidate()`，
    所以守卫对象也跟着移过去。
    """
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage7_encode.py"),
               encoding="utf-8", errors="replace").read()
    i = src.index("staging = f\"{marker_final}.staging-{attempt_id}\"")
    assert i > 0, "找不到 marker 的私有暂存名"
    # 旧写法（共享 .tmp）必须彻底消失
    assert ".done.json.tmp" not in src


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

    # 复制后必须再核对一次摘要。
    # R21-C07 抽成 `stage_generation_copy()` 后，局部别名 `_fstate/_fdig`
    # 换成函数内 import 的 `_fs/_fd` —— 断言改为在**该函数块内**找比较，
    # 而不是在整文件里找某一行原文。
    j = src.index("def stage_generation_copy(")
    block = src[j:j + 5000]
    assert "_fd(stage_path) != want" in block or \
        "_fdig(_stage_path) != _want" in block, \
        "复制后必须核对摘要，否则复制期间的替换检测不到"
    assert "_fs(src)" in block, "复制前也必须核对源摘要（冻结代）"


# ------------------------------------------------ B09 两进程写标记

import subprocess  # noqa: E402
import sys as _sys  # noqa: E402

_WRITER = r"""
import json, os, sys
sys.path.insert(0, {root!r})
sys.path.insert(0, {shared!r})
sys.path.insert(0, {pipe!r})
import main as vmain

final = {final!r}
job = {{"job_id": {jid!r}, "input_path": {inp!r},
        "detected_class": "HDV_AVCHD", "recommended_params": {{}},
        "user_overrides": {{}}}}
class L:
    def info(self, *a, **k): pass
    warning = error = info
ok = vmain.write_completion_marker(final, job, {{}}, L())
print("OK" if ok else "REFUSED")
"""


def test_two_attempts_do_not_corrupt_the_marker(tmp_path):
    """两次**独立发布**同一个目标：后一次成功，且 marker 仍可验证。

    旧版是"两个 Python 进程同时写 marker"；R22 起 marker 由事务发布，
    所以改成"两次 attempt 的事务先后落位"，核对的仍是"不能写出坏 marker"。
    """
    import publisher as _P
    import stage7_encode as _s7

    out = str(tmp_path / "clip_restored.mkv")
    stage = str(tmp_path / "staging")
    inp = str(tmp_path / "in.mp4")
    with open(inp, "wb") as f:
        f.write(b"src")
    job = _job(inp)
    os.makedirs(stage, exist_ok=True)

    markers = []
    for attempt in ("attA", "attB"):
        with open(os.path.join(stage, "media.bin"), "wb") as f:
            f.write(b"GEN-" + attempt.encode())
        with open(os.path.join(stage, "verify.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"attempt_id": "att1",
                "run_identity": {"schema": "run-identity/1", "digest": "0" * 64},
                "media_sha256": vmain._plain_sha256(
                os.path.join(stage, "media.bin")),
                "checks": {
                    "mechanical": {"name": "mechanical", "status": "PASS",
                                   "required": True},
                    "frame_order": {"name": "frame_order", "status": "PASS",
                                    "required": True},
                    "final_timeline": {"name": "final_timeline",
                                       "status": "PASS", "required": True},
                    "fullchain": {"name": "fullchain", "status": "PASS",
                                  "required": True},
                },
                "publish_decision": {"publish": True, "verified": True},
            }, f)
        with open(os.path.join(stage, "fullchain.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"ok": True, "attempt_id": attempt, "checks": {"frames": {"ok": True}, "duration": {"ok": True}, "not_black": {"ok": True}}}, f)
        arts = [
            _P.Artifact("media", os.path.join(stage, "media.bin"), out),
            _P.Artifact("verify", os.path.join(stage, "verify.json"),
                        out + ".verify.json"),
            _P.Artifact("fullchain", os.path.join(stage, "fullchain.json"),
                        out + ".fullchain.json"),
        ]
        dig = [(a.name, a.final, vmain._plain_sha256(a.staging)) for a in arts]
        seed = {
            "impl": vmain.VIDEO_IMPL_VERSION,
            "job_id": job["job_id"],
            "input_path": inp,
            "input_identity": vmain.build_input_identity(inp),
            "params_identity": vmain.build_params_identity({}, job, None),
        }
        _mk, mk_stage, mk_dig, _pm = _s7.build_marker_candidate(
            seed=seed, pub_digests=dig, attempt_id=attempt,
            job_id=job["job_id"], output_path=out,
            media_digest=dig[0][2],
            media_size=os.path.getsize(arts[0].staging))
        markers.append(mk_stage)
        arts.append(_P.Artifact("marker", mk_stage, out + ".done.json"))
        res = _P.publish(arts, out + ".journal.json",
                         str(tmp_path / "backup"), attempt_id=attempt,
                         decision={"publish": True},
                         allowed_output_roots=[str(tmp_path)],
                         required_objects=[a.final for a in arts],
                         marker_digest=mk_dig)
        assert res.ok, res.errors

    # 两次的暂存名不同（不会互相覆盖）
    assert len(set(markers)) == 2
    # 最终 markert 是完整、可解析的 JSON
    with open(vmain.completion_marker_path(out), "r",
              encoding="utf-8") as f:
        doc = json.load(f)
    assert doc.get("payload_manifest"), "最终 marker 必须带 payload"
    assert doc.get("attempt_id") in ("attA", "attB")


def test_b09_marker_staging_is_unique_per_attempt():
    """源码守卫：暂存名带 attempt（等价于"带 pid"的私有性）。"""
    src = open(os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                            "stage7_encode.py"),
               encoding="utf-8", errors="replace").read()
    assert ".staging-{attempt_id}" in src, "marker 暂存名必须带 attempt"


def test_plain_and_chunked_digests_differ_by_construction(tmp_path):
    """`_plain_sha256` 与 `_chunked_sha256` **必然不同**（后者拼了 size）。

    这条锁住"为什么守卫必须用纯内容口径"。
    """
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    a = vmain._plain_sha256(str(f))
    b = vmain._chunked_sha256(str(f))
    assert a and b and a != b, (a, b)
    # 纯内容口径要和项目其它模块一致
    from publish_journal import file_digest
    assert a == file_digest(str(f))


def test_guard_compares_like_with_like(tmp_path):
    """报告写纯内容摘要 → 验证必须通过（原来会被误拒）。

    实测反例：同一份成品 `3f1fbf3c…`（纯内容）vs `bb52754f…`（内容+size），
    旧守卫拿后者比前面记录的值，**每个视频 job 都被拒**。
    """
    media, params = _publish_full(tmp_path)
    log = _L()
    assert vmain.write_completion_marker(
        str(media), _job(str(tmp_path / "in.mp4")), {},
        params=params, logger=log) is True, log.events


