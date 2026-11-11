# -*- coding: utf-8 -*-
"""R21-C02：整代的生产集成（生产者 → publisher → main 消费）。

## 为什么要有这一组

审核 R20 §4.1：

> 这些测试不需要 AI、模型、GPU 或授权真实媒体。用自产字节作为发布
> artifacts，通过生产 publisher/stage7 发布适配层/main 完成消费路径执行。
> 只替换耗时媒体生成边界，**不能把关键回执直接手造为理想状态绕过生产者**。

所以这里不手写 marker JSON，而是调用 stage7 的
`build_marker_candidate()`（生产函数）生成 marker，再用**真实**
`publisher.publish()` 把 4 个对象作为**一个事务**发布，最后由
`main.write_completion_marker()` / `verify_generation_objects()` /
`resume_artifact_matches()` 消费。

## 反例

- 媒体被同大小替换 → payload manifest 对不上 → 拒绝
- manifest 被改一个字符 → 摘要对不上 → 拒绝
- marker 缺失 → 走兼容路径（不影响本组断言）
"""
import hashlib
import json
import os
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

import main as vmain  # noqa: E402
import publisher as P  # noqa: E402
import publish_journal as J  # noqa: E402
import stage7_encode as s7  # noqa: E402


class _Log:
    def __init__(self):
        self.events = []

    def _rec(self, ev, msg="", **kw):
        self.events.append((ev, msg, kw))

    info = warning = error = critical = _rec

    def names(self):
        return [e[0] for e in self.events]


def _w(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(path, mode, **({} if mode == "wb" else {"encoding": "utf-8"})) as f:
        f.write(data)
    return path


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _publish_generation(tmp_path, job, *, media=b"NEW-MEDIA-BYTES",
                        seed=None):
    """用真实生产者 + 真实 publisher 发布一整代，返回上下文。"""
    out = str(tmp_path / "out" / "clip_restored.mkv")
    stage = str(tmp_path / "staging")
    _w(os.path.join(stage, "media.bin"), media)
    _w(os.path.join(stage, "verify.json"),
       json.dumps({"media_sha256": _digest(os.path.join(stage, "media.bin"))}))
    _w(os.path.join(stage, "fullchain.json"), json.dumps({"ok": True}))

    arts = [
        P.Artifact("media", os.path.join(stage, "media.bin"), out),
        P.Artifact("verify", os.path.join(stage, "verify.json"),
                   out + ".verify.json"),
        P.Artifact("fullchain", os.path.join(stage, "fullchain.json"),
                   out + ".fullchain.json"),
    ]
    pub_digests = [(a.name, a.final, _digest(a.staging)) for a in arts]

    if seed is None:
        seed = {
            "impl": vmain.VIDEO_IMPL_VERSION,
            "job_id": job["job_id"],
            "input_path": job["input_path"],
            "input_identity": vmain.build_input_identity(job["input_path"]),
            "params_identity": vmain.build_params_identity({}, job, None),
        }
    marker, staging, mdig, pmdig = s7.build_marker_candidate(
        seed=seed, pub_digests=pub_digests, attempt_id="att1",
        job_id=job["job_id"], output_path=out,
        media_digest=_digest(arts[0].staging),
        media_size=os.path.getsize(arts[0].staging))
    arts.append(P.Artifact("marker", staging, out + ".done.json"))

    res = P.publish(arts, out + ".journal.json", str(tmp_path / "backup"),
                    attempt_id="att1", decision={"publish": True},
                    allowed_output_roots=[str(tmp_path / "out")],
                    required_objects=[a.final for a in arts],
                    marker_digest=mdig)
    return {"out": out, "res": res, "marker": marker, "mdigest": mdig,
            "pm_digest": pmdig, "arts": arts}


# ------------------------------------------------ 正常路径

def test_marker_is_published_in_the_same_transaction(tmp_path):
    job = {"job_id": "j1", "input_path": str(tmp_path / "src.mp4")}
    _w(job["input_path"], b"SOURCE")
    ctx = _publish_generation(tmp_path, job)

    assert ctx["res"].ok is True, ctx["res"].errors
    for suffix in ("", ".verify.json", ".fullchain.json", ".done.json"):
        assert os.path.exists(ctx["out"] + suffix), suffix

    # marker 里带着冻结的 payload manifest，且**不含自己**
    doc = json.loads(open(ctx["out"] + ".done.json", encoding="utf-8").read())
    names = [o["name"] for o in doc["payload_manifest"]["objects"]]
    assert names == ["media", "verify", "fullchain"]
    assert "marker" not in names
    assert doc["payload_manifest_digest"] == ctx["pm_digest"]


def test_main_verifies_the_published_marker_without_rewriting_it(tmp_path):
    job = {"job_id": "j1", "input_path": str(tmp_path / "src.mp4")}
    _w(job["input_path"], b"SOURCE")
    ctx = _publish_generation(tmp_path, job)
    marker_path = ctx["out"] + ".done.json"
    before = (_digest(marker_path), os.path.getmtime(marker_path))

    log = _Log()
    ok = vmain.write_completion_marker(ctx["out"], job, {}, log)
    assert ok is True
    assert "COMPLETION_MARKER_VERIFIED_PUBLISHED" in log.names()
    after = (_digest(marker_path), os.path.getmtime(marker_path))
    assert after == before, "验证通过的 marker 不该被重写"


def test_resume_accepts_the_published_generation(tmp_path):
    job = {"job_id": "j1", "input_path": str(tmp_path / "src.mp4")}
    _w(job["input_path"], b"SOURCE")
    ctx = _publish_generation(tmp_path, job)
    ok, why = vmain.resume_artifact_matches(ctx["out"], job, {}, None)
    assert ok is True, why


# ------------------------------------------------ 反例

def test_same_size_media_replacement_is_rejected(tmp_path):
    """同大小替换：文件在、字节换 → manifest 必须发现。"""
    job = {"job_id": "j1", "input_path": str(tmp_path / "src.mp4")}
    _w(job["input_path"], b"SOURCE")
    ctx = _publish_generation(tmp_path, job)
    assert len(b"NEW-MEDIA-BYTES") == len(b"OLD-MEDIA-BYTES")
    _w(ctx["out"], b"OLD-MEDIA-BYTES")

    ok, why = vmain.verify_published_marker(ctx["out"], job, {}, None, None)
    assert ok is False
    assert "changed since it was published" in why, why


def test_tampered_manifest_is_rejected(tmp_path):
    job = {"job_id": "j1", "input_path": str(tmp_path / "src.mp4")}
    _w(job["input_path"], b"SOURCE")
    ctx = _publish_generation(tmp_path, job)
    p = ctx["out"] + ".done.json"
    doc = json.loads(open(p, encoding="utf-8").read())
    doc["payload_manifest"]["objects"][1]["sha256"] = "0" * 64
    _w(p, json.dumps(doc, ensure_ascii=False))

    ok, why = vmain.verify_published_marker(ctx["out"], job, {}, None, None)
    assert ok is False
    assert "does not match the digest" in why, why


def test_missing_report_breaks_the_whole_generation(tmp_path):
    job = {"job_id": "j1", "input_path": str(tmp_path / "src.mp4")}
    _w(job["input_path"], b"SOURCE")
    ctx = _publish_generation(tmp_path, job)
    os.remove(ctx["out"] + ".fullchain.json")

    ok, why = vmain.verify_published_marker(ctx["out"], job, {}, None, None)
    assert ok is False
    assert "missing" in why, why


def test_marker_from_another_job_is_rejected(tmp_path):
    job = {"job_id": "j1", "input_path": str(tmp_path / "src.mp4")}
    _w(job["input_path"], b"SOURCE")
    other = {"job_id": "j2", "input_path": str(tmp_path / "src2.mp4")}
    _w(other["input_path"], b"OTHER")
    ctx = _publish_generation(tmp_path, job,
                              seed={
                                  "impl": vmain.VIDEO_IMPL_VERSION,
                                  "job_id": "j2",
                                  "input_path": other["input_path"],
                                  "input_identity": vmain.build_input_identity(
                                      other["input_path"]),
                                  "params_identity": vmain.build_params_identity(
                                      {}, other, None),
                              })
    ok, why = vmain.verify_published_marker(ctx["out"], job, {}, None, None)
    assert ok is False
    assert "belongs to job" in why, why
