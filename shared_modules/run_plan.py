# -*- coding: utf-8 -*-
"""运行计划（RunPlan）：在**首次落盘之前**把有效路径一次性定清楚。

**为什么需要它**

`main.py` 原来分三处各自取值：

```python
params = merge_params(job, config, cli_overrides, logger)   # CLI 的 --output → output_dir
params["output_path"] = job.get("output_path", "")          # job 又覆盖最终媒体
log_dir = params.get("log_dir", "./logs")                   # 日志再取一次
```

于是 `--output` 只换了 `output_dir`，**没有**换 `output_path`。
2026-10-28 那次事故就是这么来的：复制了别人的 manifest、用 `--output`
指向隔离目录，结果仍然写回了原目录，覆盖了既有验收产物。

**本模块做什么**

1. 把每个 job 的**全部**读写路径物化一次，并记下每个值的来源与优先级；
2. 解析符号链接/junction、Windows 大小写、相对路径（不存在的尾部按最深
   已存在祖先解析）；
3. 预检：写路径是否在授权根内、输入是否与输出重合、跨 job 目标是否冲突、
   侧车是否互相覆盖；
4. 返回**唯一**的计划；调用方把它写回 params，后续 stage 只消费这份计划。

**不做什么**

- 不创建目录（除显式 `materialise=True`）；
- 不改变逐 job 输出配置的优先级（那是用户意图）；
- 不做业务判断（时间轴、检查结果等不在这里）。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# 路径来源的优先级（数字越小越强）。记录用途：让报告能解释"这个值听谁的"。
SOURCE_CLI_RELOCATE = "cli:relocate"
SOURCE_JOB = "job"
SOURCE_CLI = "cli"
SOURCE_CONFIG = "config"
SOURCE_DEFAULT = "default"

_SOURCE_RANK = {
    SOURCE_CLI_RELOCATE: 0,
    SOURCE_JOB: 1,
    SOURCE_CLI: 2,
    SOURCE_CONFIG: 3,
    SOURCE_DEFAULT: 4,
}

# 同一目标的侧车后缀。集中在这里，避免各处自己拼。
SIDECAR_SUFFIXES: Tuple[str, ...] = (
    ".verify.json",
    ".fullchain.json",
    ".status.json",
    ".contract.json",
    ".done.json",
    ".journal.json",
    ".lock",
    ".pubbak",
)


class PathValue:
    """一个路径 + 它凭什么取这个值。"""

    __slots__ = ("key", "raw", "resolved", "source", "is_write")

    def __init__(self, key: str, raw: Optional[str], resolved: Optional[str],
                 source: str, is_write: bool):
        self.key = key
        self.raw = raw
        self.resolved = resolved
        self.source = source
        self.is_write = is_write

    def as_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "raw": self.raw,
                "resolved": self.resolved, "source": self.source,
                "is_write": self.is_write}

    def __repr__(self) -> str:      # pragma: no cover - 调试用
        return f"PathValue({self.key!r}={self.resolved!r}, {self.source})"


def resolve_path(path: Optional[str]) -> Optional[str]:
    """绝对化 + 解析 symlink/junction。**保留原始大小写。**

    > 不要在这里 ``normcase``：返回值会被写回 params 供各 stage 消费，
    > 而 Windows 上 ``normcase`` 会小写化 —— 那会改变产物文件名，
    > 也会改变 ``param_hash`` 的输入。需要比较时用 ``path_key()``。


    **不存在的尾部**按"最深的已存在祖先"解析后再拼回去。
    这样"父目录是个 junction 指向别处"的情形也能被发现——
    裸字符串拼接会把 junction 当普通目录，判断就错了。

    Returns:
        归一后的绝对路径；输入为空返回 ``None``。
    """
    if not path:
        return None
    p = os.path.abspath(os.path.expanduser(str(path)))
    tail: List[str] = []
    cur = p
    while not os.path.exists(cur):
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        tail.append(os.path.basename(cur))
        cur = parent
    base = os.path.realpath(cur) if os.path.exists(cur) else cur
    for name in reversed(tail):
        base = os.path.join(base, name)
    return os.path.normpath(base)


def path_key(path: Optional[str]) -> Optional[str]:
    """用于**比较**的键：归一大小写与分隔符，但只给比较用，不写回参数。"""
    r = resolve_path(path)
    return os.path.normcase(r) if r else None


def is_within(path: Optional[str], root: Optional[str]) -> bool:
    """``path`` 是否在 ``root`` 之内（含 root 本身）。

    用 ``os.path.commonpath``，**不用** `startswith`：
    `C:\\out2` 以 `C:\\out` 为前缀但不是它的子路径；
    大小写与分隔符差异也由 ``normcase`` 处理。
    """
    if not path or not root:
        return False
    try:
        a = path_key(path)
        b = path_key(root)
        if a is None or b is None:
            return False
        return os.path.commonpath([a, b]) == b
    except (ValueError, OSError):
        # 不同盘符时 commonpath 抛异常 → 必然不在同一根下
        return False


def _pick(candidates: Sequence[Tuple[str, Optional[str]]]
          ) -> Tuple[Optional[str], str]:
    """按 SOURCE_RANK 顺序取第一个非空值。返回 ``(raw, source)``。"""
    ordered = sorted(candidates, key=lambda kv: _SOURCE_RANK.get(kv[0], 99))
    for source, raw in ordered:
        if raw not in (None, ""):
            return str(raw), source
    return None, SOURCE_DEFAULT


class RunPlan:
    """一个 job 的有效读写路径集合 + 预检结论。

    Attributes:
        job_id: 作业标识。
        paths: ``{key: PathValue}``，键见 ``_WRITE_KEYS`` / ``_READ_KEYS``。
        write_roots: 授权写根（空 = 不做越界判定，只记事实）。
        errors: 阻断性问题（有任一条就不得开始落盘）。
        warnings: 需要说明但不阻断的问题。
    """

    def __init__(self, job_id: str, paths: Dict[str, PathValue],
                 write_roots: Sequence[str], errors: List[str],
                 warnings: List[str]):
        self.job_id = job_id
        self.paths = paths
        self.write_roots = list(write_roots)
        self.errors = errors
        self.warnings = warnings

    # -- 取值便利 ----------------------------------------------------------
    def get(self, key: str) -> Optional[str]:
        pv = self.paths.get(key)
        return pv.resolved if pv else None

    def source_of(self, key: str) -> Optional[str]:
        pv = self.paths.get(key)
        return pv.source if pv else None

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "ok": self.ok,
            "write_roots": self.write_roots,
            "errors": self.errors,
            "warnings": self.warnings,
            "paths": {k: v.as_dict() for k, v in self.paths.items()},
        }

    def __repr__(self) -> str:      # pragma: no cover - 调试用
        return f"RunPlan({self.job_id!r}, ok={self.ok}, errors={len(self.errors)})"


# 最终用户目标（缺任一项都不是完整一代）
_WRITE_KEYS = (
    "output_path",       # 最终媒体
    "verify_path",
    "fullchain_path",
    "status_path",
    "contract_path",
    "journal_path",
    "temp_dir",
    "log_dir",
    "wsl_copyback_path",
)
_READ_KEYS = ("input_path",)


def _derive_sidecars(output_path: str) -> Dict[str, str]:
    """由最终媒体路径推导侧车路径（与各 stage 现在的拼法一致）。"""
    return {
        "verify_path": output_path + ".verify.json",
        "fullchain_path": output_path + ".fullchain.json",
        "status_path": output_path + ".status.json",
        "contract_path": output_path + ".contract.json",
        "journal_path": output_path + ".journal.json",
    }


def resolve_run_plan(job: Dict[str, Any], params: Dict[str, Any],
                     write_roots: Optional[Sequence[str]] = None,
                     relocate_output: Optional[str] = None) -> RunPlan:
    """物化一个 job 的有效路径并预检。

    Args:
        job: manifest 里的 job 字典（可能含 ``output_path`` / ``temp_dir`` /
            ``log_path``）。
        params: 合并后的参数（含 CLI 覆盖产生的 ``output_dir`` /
            ``temp_dir`` / ``log_dir``）。
        write_roots: 授权写根。**为空时不做越界判定**（只记录事实），
            以便不破坏既有调用方；显式给出时越界即阻断。
        relocate_output: 显式"重新定位输出"目录。给定时**所有**由输出
            派生的路径（媒体、四个侧车、journal）都改到这个目录下，
            只保留原 basename；temp/log 不受影响（它们有独立语义）。

    Returns:
        ``RunPlan``。``plan.ok`` 为 False 时调用方**不得**开始任何落盘。
    """
    errors: List[str] = []
    warnings: List[str] = []
    roots = [r for r in (write_roots or []) if r]

    # ---- 最终媒体路径 ----
    job_out = job.get("output_path") or ""
    out_dir = params.get("output_dir") or params.get("output_dir_default") or ""

    if relocate_output:
        # 显式重定位：用原 basename 落到新目录，来源记为 cli:relocate
        basename = os.path.basename(str(job_out).replace("\\", "/")) if job_out \
            else f"{job.get('job_id', 'job')}_restored.mkv"
        out_raw = os.path.join(relocate_output, basename)
        out_source = SOURCE_CLI_RELOCATE
    elif job_out:
        out_raw, out_source = str(job_out), SOURCE_JOB
    elif out_dir:
        # 没有 job 级输出时才用目录 + 约定名
        name = f"{job.get('job_id', 'job')}_restored.mkv"
        out_raw = os.path.join(str(out_dir), name)
        out_source = SOURCE_CLI if params.get("_output_from_cli") else SOURCE_CONFIG
    else:
        out_raw, out_source = "", SOURCE_DEFAULT
        errors.append(
            "输出路径无法确定：既没有 job.output_path，也没有可用的 "
            "output_dir。请在 manifest 或 CLI 中明确指定。")

    out_res = resolve_path(out_raw) if out_raw else None
    paths: Dict[str, PathValue] = {
        "output_path": PathValue("output_path", out_raw, out_res,
                                 out_source, is_write=True)
    }

    if out_res:
        for k, v in _derive_sidecars(out_raw).items():
            paths[k] = PathValue(k, v, resolve_path(v), out_source, True)

    # ---- 输入 ----
    in_raw = job.get("input_path") or ""
    in_res = resolve_path(in_raw) if in_raw else None
    paths["input_path"] = PathValue("input_path", in_raw or None, in_res,
                                    SOURCE_JOB if in_raw else SOURCE_DEFAULT,
                                    is_write=False)
    if not in_raw:
        errors.append("输入路径缺失：job.input_path 为空。")
    elif in_res and not os.path.exists(in_res):
        errors.append(f"输入不存在或不可访问：{in_raw}")

    # ---- 临时目录 ----
    tmp_raw, tmp_source = _pick([
        (SOURCE_JOB, job.get("temp_dir")),
        (SOURCE_CLI, params.get("temp_dir")),
        (SOURCE_CONFIG, params.get("temp_dir_default")),
    ])
    paths["temp_dir"] = PathValue("temp_dir", tmp_raw,
                                  resolve_path(tmp_raw) if tmp_raw else None,
                                  tmp_source, True)

    # ---- 日志目录 ----
    log_raw, log_source = _pick([
        (SOURCE_JOB, job.get("log_path")),
        (SOURCE_CLI, params.get("log_dir")),
        (SOURCE_CONFIG, params.get("log_dir_default")),
    ])
    if not log_raw:
        log_raw, log_source = "./logs", SOURCE_DEFAULT
        warnings.append("日志目录未指定，将使用默认 ./logs")
    paths["log_dir"] = PathValue("log_dir", log_raw, resolve_path(log_raw),
                                 log_source, True)

    # ---- WSL 最终回拷目标（仅重定向时存在）----
    wsl_target = job.get("original_output_path") or params.get("wsl_original_output")
    if wsl_target:
        paths["wsl_copyback_path"] = PathValue(
            "wsl_copyback_path", str(wsl_target), resolve_path(wsl_target),
            SOURCE_JOB, True)

    # ---- 预检 1：输入不得与输出重合 ----
    if in_res and out_res:
        if path_key(in_res) == path_key(out_res):
            errors.append(
                f"输入与最终输出是同一个路径：{in_raw}。这会覆盖原始素材。")
        elif os.path.dirname(in_res) == os.path.dirname(out_res) and \
                os.path.splitext(in_res)[1].lower() == \
                os.path.splitext(out_res)[1].lower() and \
                os.path.basename(in_res) == os.path.basename(out_res):
            errors.append(f"输入与输出同名同目录：{in_raw}")

    # ---- 预检 2：写路径必须在授权根内 ----
    if roots:
        for key, pv in paths.items():
            if not pv.is_write or not pv.resolved:
                continue
            if not any(is_within(pv.resolved, r) for r in roots):
                errors.append(
                    f"{key} 落在授权写根之外：{pv.raw}"
                    f"（来源 {pv.source}）。授权根：{', '.join(roots)}")
    else:
        warnings.append(
            "未提供授权写根：不做越界判定，仅记录事实路径。"
            "（无人值守运行建议显式给出 --allow-write-root）")

    # ---- 预检 3：侧车不得与其它 job 的媒体路径相同（单 job 内自查）----
    seen: Dict[str, str] = {}
    for key, pv in paths.items():
        if not pv.is_write or not pv.resolved:
            continue
        _k = path_key(pv.resolved) or pv.resolved
        if _k in seen:
            errors.append(
                f"{key} 与 {seen[_k]} 指向同一路径：{pv.raw}")
        else:
            seen[_k] = key

    return RunPlan(str(job.get("job_id") or "unknown"), paths, roots,
                   errors, warnings)


def detect_cross_job_conflicts(plans: Iterable[RunPlan]) -> List[str]:
    """多个 job 的目标集合是否互相踩到（侧车碰撞、媒体同名等）。

    Returns:
        冲突描述列表；空表示无冲突。
    """
    owners: Dict[str, List[str]] = {}
    for plan in plans:
        for key, pv in plan.paths.items():
            if not pv.is_write or not pv.resolved:
                continue
            owners.setdefault(path_key(pv.resolved) or pv.resolved,
                              []).append(f"{plan.job_id}:{key}")

    conflicts: List[str] = []
    for path, who in owners.items():
        if len(who) > 1:
            conflicts.append(f"路径被多个目标占用：{path} ← {', '.join(who)}")
    return conflicts
