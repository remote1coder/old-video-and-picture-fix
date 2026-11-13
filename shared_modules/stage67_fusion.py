# -*- coding: utf-8 -*-
"""S6→S7 执行组（`chain.mode: stage67`）的共享数据面。

## 这个模块解决什么

stage6（加颗粒）与 stage7（编码）在**经典路线**上是两个进程、一个
lossless FFV1 中间文件：

    stage5_out --vspipe--> ffmpeg(ffv1) --写盘--> step6_grain.mkv
               --读盘--> ffmpeg(libsvtav1) --> 成品

中间那一份 FFV1 是 **16bit 4:2:0 全量帧**，1080p 60 分钟约 300 GB 量级，
写一次读一次纯粹是"为了让下一个进程能拿到帧"。融合后：

    stage5_out --vspipe--> ffmpeg(libsvtav1) --> 成品

## 为什么这样切才叫"复用同一核心"

融合**不重写**任何滤镜或编码参数：

* 滤镜部分：stage6 仍然用它自己的 ``_build_vs_script()`` 生成 vpy，
  只是不再把它喂给 FFV1 编码器，而是把 **脚本路径** 交给 stage7；
* 编码部分：stage7 仍然用它自己那一整段 ``video_cmd`` / ``output_opts``
  （色彩转换、黑边裁还原、SVTAV1 参数、元数据），只把输入段从
  ``-i <文件>`` 换成 ``-f yuv4mpegpipe -i pipe:0``；
* 两个进程的编排：直接调用 stage6 的 ``_run_grain_pipeline()``
  —— 同时创建、同时回收、检查**每一个**进程的返回码、失败清半成品。

所以"融合结果和经典结果一样"这件事，本质上是**同一个 vpy + 同一段
ffmpeg 参数**。任何一边改了参数，两条路线会一起变。

## 身份与回执

融合组必须留下**组级回执**（``<成品>.fused_stage67.json``），否则
编排器/下一轮无法回答"stage7 到底有没有把 stage6 的颗粒做进去"。
回执的字段刻意与 ``fused_stage34.json`` 保持一致，好让
``fused_receipt.verify_fused_receipt()`` 直接复用它的
attempt / 源身份 / 产物摘要 / 证据块校验，本模块只补组特有的三项：

1. ``schema`` + ``group`` 必须是 stage67；
2. ``vs_script_sha256`` 与**当前**磁盘上的 vpy 一致（脚本被换过 → 过期）；
3. ``stages`` 必须恰好是 ``[6, 7]``。

## 边界

* 融合是优化，不是必须。组回执不成立时**走经典路线**，不报错。
* 但"计划里选了 stage67、stage7 却按经典跑完"必须被发现 —— 那意味着
  中间文件被写出来了，用户以为省了盘、其实没省。见
  :func:`verify_group_receipt` 的 ``require_fused`` 用法。
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

GROUP = "stage67"
SCHEMA = "stage67_group/1"


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def sha256_file(path: str, chunk: int = 1 << 20) -> Optional[str]:
    """文件内容的 sha256；读不到返回 None（**不**返回空串冒充成功）。"""
    if not path:
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def spec_path(temp_dir: str, job_id: str, stem: str) -> str:
    """组规格文件（stage6 写、stage7 读）的确定路径。

    **确定路径**而不是 glob：融合组的两端必须看同一个文件。用 glob 选
    "最新的那个"会在上一轮残留 + 本轮重跑时选错（R20-A03 的病根）。
    """
    return os.path.abspath(os.path.join(
        str(temp_dir), f"{job_id}_{stem}_step67_group.json"))


def receipt_path(output_path: str) -> str:
    """组回执与成品**同名**，加 ``.fused_stage67.json`` 后缀。

    这样"产物被换过"这件事天然成立：换掉成品不一定换回执，但回执里的
    ``output_sha256`` 会对不上。
    """
    return str(output_path) + ".fused_stage67.json"


# --------------------------------------------------------------------------
# 组规格（stage6 → stage7 的交接单）
# --------------------------------------------------------------------------
def build_spec(*, attempt_id: str, chain_mode: str, vs_script: str,
               probe_source: str, grain: Dict[str, Any],
               vs_format: str, pix_fmt: str,
               deterministic: bool) -> Dict[str, Any]:
    """构造交接单。**所有路径必须是绝对路径**（vspipe 会 chdir）。"""
    return {
        "schema": SCHEMA,
        "group": GROUP,
        "stages": [6, 7],
        "attempt_id": str(attempt_id or ""),
        "chain_mode": str(chain_mode or ""),
        "vs_script": os.path.abspath(str(vs_script)),
        "vs_script_sha256": sha256_file(str(vs_script)) or "",
        "probe_source": os.path.abspath(str(probe_source)),
        "grain": dict(grain or {}),
        "vs_format": str(vs_format or ""),
        "pix_fmt": str(pix_fmt or ""),
        "deterministic": bool(deterministic),
    }


def write_spec(path: str, spec: Dict[str, Any]) -> None:
    Path_dir = os.path.dirname(os.path.abspath(path))
    if Path_dir:
        os.makedirs(Path_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=2)


def load_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh) or {}
    if not isinstance(doc, dict):
        raise ValueError("stage67 spec is not an object")
    return doc


def validate_spec(spec: Dict[str, Any], *, expected_attempt: str = ""
                  ) -> Tuple[bool, str]:
    """交接单自洽性。**任何一项不成立都不能开跑**。

    这里查的是"这份交接单能不能用"，不是"上一轮干得对不对"：文件不存在、
    脚本被改过、attempt 不是本次，都属于"不能用"，应回经典路线或失败，
    绝不允许 stage7 一边读着别人的脚本一边宣称自己做了 stage6。
    """
    if not isinstance(spec, dict):
        return False, "spec is not an object"
    if spec.get("schema") != SCHEMA:
        return False, f"spec schema is {spec.get('schema')!r}, not {SCHEMA!r}"
    if spec.get("group") != GROUP:
        return False, f"spec group is {spec.get('group')!r}, not {GROUP!r}"
    if list(spec.get("stages") or []) != [6, 7]:
        return False, f"spec stages are {spec.get('stages')!r}, not [6, 7]"
    if expected_attempt:
        got = str(spec.get("attempt_id") or "")
        if got != str(expected_attempt):
            return False, (f"spec belongs to attempt {got!r}, "
                           f"expected {expected_attempt!r}")
    vs = str(spec.get("vs_script") or "")
    if not vs or not os.path.exists(vs):
        return False, f"vs script is gone: {vs!r}"
    want = str(spec.get("vs_script_sha256") or "")
    if not want:
        return False, "spec carries no vs_script_sha256"
    cur = sha256_file(vs)
    if cur is None:
        return False, "vs script is unreadable"
    if cur != want:
        return False, (f"vs script changed since the spec was written "
                       f"(recorded {want[:16]}, now {cur[:16]})")
    ps = str(spec.get("probe_source") or "")
    if not ps or not os.path.exists(ps):
        return False, f"probe source is gone: {ps!r}"
    return True, ""


# --------------------------------------------------------------------------
# 组回执（stage7 写，编排器/下一轮读）
# --------------------------------------------------------------------------
def build_receipt(*, spec: Dict[str, Any], params: Dict[str, Any],
                  output_path: str, frames: Optional[int],
                  extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """构造组回执。

    字段与 ``fused_stage34.json`` 对齐（好让共享验证器直接复用），
    另外带上只有 6→7 组才有的东西：vpy 摘要、颗粒参数、探测源。
    """
    try:
        from fused_receipt import source_digest as _src_digest
        _rdig = _src_digest(str(spec.get("probe_source") or ""))
    except Exception:                              # pragma: no cover
        _rdig = None
    try:
        _size = os.path.getsize(str(output_path))
    except OSError:
        _size = 0
    rcpt: Dict[str, Any] = {
        "schema": SCHEMA,
        "group": GROUP,
        "stages": [6, 7],
        # 消费方（fused_receipt.verify_fused_receipt）按这些键判定
        "attempt_id": str((params or {}).get("attempt_id") or ""),
        "chain_mode": str((params or {}).get("chain_mode") or ""),
        "logical_stages": ["stage6", "stage7"],
        "impl": {
            "group": "stage67-fusion",
            "producer_script": "stage6_grain_synthesis.py",
            "consumer_script": "stage7_encode.py",
        },
        "job_id": str((params or {}).get("job_id") or ""),
        # 源身份：**只含文件身份**（空 params），跨 stage 稳定
        "input_path": str(spec.get("probe_source") or ""),
        "input_digest": (_rdig or ""),
        "input_digest_kind": "source-only",
        # 参数身份：main 在计划阶段冻结的那一份，消费方**不重算**
        "param_digest": str((params or {}).get("_frozen_param_digest") or ""),
        "output": str(output_path),
        "output_size": _size,
        "output_sha256": sha256_file(str(output_path)) or "",
        "frames": frames,
        # ---- 组特有 ----
        "group_evidence": {
            "vs_script": str(spec.get("vs_script") or ""),
            "vs_script_sha256": str(spec.get("vs_script_sha256") or ""),
            "probe_source": str(spec.get("probe_source") or ""),
            "grain": dict(spec.get("grain") or {}),
            "vs_format": str(spec.get("vs_format") or ""),
            "pix_fmt": str(spec.get("pix_fmt") or ""),
            "deterministic": bool(spec.get("deterministic")),
            "intermediate_written": False,
        },
        "written_at": __import__("datetime").datetime.now().astimezone().isoformat(),
    }
    if extra:
        rcpt.update(extra)
    return rcpt


def write_receipt(path: str, rcpt: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rcpt, fh, ensure_ascii=False, indent=2)


def verify_group_receipt(rcpt_path: str, params: Dict[str, Any],
                         expected_attempt: str = "",
                         src_path: str = "") -> Tuple[bool, str,
                                                      Optional[Dict[str, Any]]]:
    """校验组回执能否用于**本次**执行。

    先跑共享验证器 ``fused_receipt.verify_fused_receipt`` 的**身份与产物**
    部分（attempt / 源身份 / 参数身份 / 产物存在 + 大小 + 内容摘要），
    再补组特有的三项 + 组边界上的证据：

    * 第 4 步「颜色转换合同」在这一层**不适用**（转换发生在组内），
      但**标签丢失**与**帧数/时长守恒**是组级属性，必须查。
    """
    try:
        from fused_receipt import verify_fused_receipt as _vfr
    except Exception as exc:                        # pragma: no cover
        return False, f"shared verifier unavailable: {exc}", None
    ok, why, rcpt = _vfr(rcpt_path, params,
                         expected_attempt=expected_attempt, src_path=src_path,
                         require_evidence=False)
    if not ok:
        return False, why, rcpt

    if rcpt.get("group") != GROUP:                  # pragma: no cover - 防御
        return False, f"receipt group is {rcpt.get('group')!r}", rcpt
    if list(rcpt.get("stages") or []) != [6, 7]:    # pragma: no cover - 防御
        return False, f"receipt stages are {rcpt.get('stages')!r}", rcpt

    ge = rcpt.get("group_evidence") or {}
    vs = str(ge.get("vs_script") or "")
    want = str(ge.get("vs_script_sha256") or "")
    if not vs or not want:
        return False, "receipt carries no group evidence (vpy identity)", rcpt
    if not os.path.exists(vs):
        return False, "the fused vs script is gone", rcpt
    cur = sha256_file(vs)
    if cur is None:
        return False, "the fused vs script is unreadable", rcpt
    if cur != want:
        return False, ("the fused vs script changed since this receipt "
                       f"(recorded {want[:16]}, now {cur[:16]})"), rcpt
    if ge.get("intermediate_written") is not False:
        return False, ("receipt says an intermediate was written; the "
                       "execution group did not actually fuse"), rcpt

    # ---- 组边界上的证据：颜色「没丢」+ 帧数/时长「守恒」----
    color = rcpt.get("color")
    src_color = rcpt.get("color_source")
    if not isinstance(color, dict) or not color.get("probed"):
        return False, "the output colour state was never probed", rcpt
    if not isinstance(src_color, dict) or not src_color.get("probed"):
        return False, ("the group input's colour state was never probed "
                       "(unverified, not 'no tags')"), rcpt
    try:
        from generation_evidence import derive_color_relations as _dcr
        _lost, _ = _dcr(src_color, color)
    except Exception:                               # pragma: no cover
        _lost = []
    if _lost:
        return False, ("colour tags were lost inside the group: "
                       + ",".join(_lost)), rcpt
    bounds = rcpt.get("bounds") or {}
    if not bounds.get("performed"):
        return False, ("the frame count / duration conservation check was "
                       "never performed, so the group is unproven"), rcpt
    if not bounds.get("ok"):
        return False, ("the group does not conserve frame count / duration: "
                       f"{bounds.get('reason') or bounds.get('detail') or ''}"
                       ).strip(), rcpt
    return True, "", rcpt


def find_group_receipt(output_path: str, params: Dict[str, Any],
                       expected_attempt: str = "",
                       src_path: str = ""):
    """取**这一份成品**旁边的组回执并校验。

    位置固定为 ``<output>.fused_stage67.json`` —— 与 ``.verify.json`` /
    ``.fullchain.json`` 同一套约定：回执跟着产物走，不跟着 temp 目录走
    （temp 会被跨轮次清理，而且同名 job 重跑时会留下上一轮的残骸）。

    Returns:
        ``(rcpt, path, reason)``。找不到/不通过时 ``rcpt`` 为 None。
    """
    path = receipt_path(output_path)
    if not os.path.exists(path):
        return None, path, "no stage67 group receipt next to the output"
    ok, why, rcpt = verify_group_receipt(
        path, params, expected_attempt=expected_attempt, src_path=src_path)
    if ok:
        return rcpt, path, ""
    return None, path, why
