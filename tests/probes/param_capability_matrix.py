# -*- coding: utf-8 -*-
"""R23-13：逐参数能力表（功能 × 配置键 × 读取点 × 阶段 × 身份 × 证据）。

## 它回答什么问题

审核 R22-B14：

> 功能表存在总括项而非逐参数证据。

> 参数进入 hash 但没有真实读取、代码读取但没有依赖声明，都应被发现；
> 仍要有行为测试，不能只靠 grep。

所以这张表由**两段**组成：

1. **静态段（本脚本）**：从 `config.yaml`（210 个叶子键）、
   `checkpoint.CORE_QUALITY_PARAMS`、`stage_deps` 三处取集合，
   再去源码里找**读取点**（`.get("k")` / `["k"]` / `_cfg(params,"k")`）。
   它给出"这个键到底在哪被读"和"三处声明是否一致"。
2. **行为段（`tests/test_r23_param_capability.py`）**：对静态段判定为
   "危险"的键，用**真实函数调用**证明它要么改变行为、要么确实是死的。
   静态扫描只能发现"值得怀疑"，不能单独定案。

## 用法

    python tests/probes/param_capability_matrix.py [--json out.json]
"""
import argparse
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
CONFIG = os.path.join(ROOT, "video_restoration_pipeline", "config.yaml")
SRC_DIRS = ("shared_modules", "video_restoration_pipeline",
            "photo_restoration_pipeline")

#: 读取模式（按优先级）。命中任一即算"有读取点"。
READ_PATTERNS = (
    r"\.get\(\s*['\"]{k}['\"]",
    r"\[\s*['\"]{k}['\"]\s*\]",
    r"_cfg\([^)]*['\"]{k}['\"]",
    r"params\.pop\(\s*['\"]{k}['\"]",
)

#: 这些词太通用（`cfg.get("enabled")` 遍地都是），静态扫描无法归因到具体参数。
#: 它们在表里单独标 "generic"，**不参与**一致性判定，避免噪音掩盖真问题。
GENERIC_KEYS = frozenset({
    "enabled", "mode", "type", "name", "path", "state", "value", "size",
    "deterministic", "sample_rate", "channels", "codec", "bitrate",
    "denoise", "preset", "crf", "platform", "format", "version",
})

#: 这些键由**位置参数**消费（不按名字读），静态扫描必然漏；单独列出。
POSITIONAL_CONSUMERS = {
    "output_dir": "main：路径层（不是媒体语义）",
    "temp_dir": "main：路径层",
    "log_dir": "main：路径层",
    "output_path": "main：路径层",
    "input_path": "main：路径层",
}


def _py_files():
    out = []
    for d in SRC_DIRS:
        for base, _dirs, files in os.walk(os.path.join(ROOT, d)):
            if "__pycache__" in base:
                continue
            for n in files:
                if n.endswith(".py"):
                    out.append(os.path.join(base, n))
    return sorted(out)


def _leaf_keys(doc, prefix=""):
    for k, v in doc.items():
        p = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _leaf_keys(v, p + ".")
        else:
            yield p, v


def _bare(path):
    return path.rsplit(".", 1)[-1]


def scan():
    import yaml
    cfg = yaml.safe_load(io.open(CONFIG, encoding="utf-8"))
    leaves = list(_leaf_keys(cfg))

    sys.path.insert(0, os.path.join(ROOT, "shared_modules"))
    import checkpoint as cp
    import stage_deps as sd
    from typing import get_args  # noqa: F401

    hash_keys = set(getattr(cp, "CORE_QUALITY_PARAMS", frozenset()))
    stage_keys = set()
    for s in getattr(sd, "_STAGE_KEYS", {}).values():
        stage_keys |= set(s)
    global_params = set(getattr(sd, "GLOBAL_PARAMS", set()))
    non_sem = set(getattr(sd, "NON_SEMANTIC_PARAMS", set()))
    known_unimplemented = set(getattr(sd, "DECLARED_NOT_IMPLEMENTED", set()))

    files = _py_files()
    texts = {}
    for f in files:
        try:
            texts[f] = io.open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue

    rows = []
    _seen = {}
    for path, default in leaves:
        key = _bare(path)
        hits = []
        for rx in READ_PATTERNS:
            pat = re.compile(rx.format(k=re.escape(key)))
            for f, txt in texts.items():
                for m in pat.finditer(txt):
                    line = txt.count("\n", 0, m.start()) + 1
                    rel = os.path.relpath(f, ROOT).replace("\\", "/")
                    hit = f"{rel}:{line}"
                    if hit not in hits:
                        hits.append(hit)
        row = {
            "config_path": path,
            "key": key,
            "default": default,
            "read_points": hits[:8],
            "read_count": len(hits),
            "in_param_hash": key in hash_keys,
            "in_stage_deps": key in stage_keys or key in global_params,
            "non_semantic": key in non_sem,
            "positional": key in POSITIONAL_CONSUMERS,
            "generic": key in GENERIC_KEYS,
        }
        rows.append(row)
        # 同一个 bare key 可能在多个段里出现（global / paths / audio_override）；
        # 一致性判定按**并集**看，不能因为某一段没读到就判死。
        agg = _seen.setdefault(key, {
            "key": key, "config_paths": [], "read_points": [],
            "read_count": 0, "in_param_hash": False,
            "in_stage_deps": False, "non_semantic": False,
            "positional": False, "generic": row["generic"]})
        agg["config_paths"].append(path)
        agg["read_points"] = list(dict.fromkeys(
            agg["read_points"] + hits))[:8]
        agg["read_count"] = max(agg["read_count"], len(hits))
        for flag in ("in_param_hash", "in_stage_deps", "non_semantic",
                     "positional"):
            agg[flag] = agg[flag] or row[flag]

    rows = list(_seen.values())

    # ---- 三个方向的一致性检查 ----
    problems = []
    for r in rows:
        if r["generic"]:
            continue
        # 已知未实现的键单独登记在 `stage_deps.DECLARED_NOT_IMPLEMENTED`，
        # 不算"新问题"；但它们仍然要有读取点检查（实现后要移出登记表）。
        r["known_unimplemented"] = r["key"] in known_unimplemented
        if r["known_unimplemented"]:
            continue
        if r["in_param_hash"] and r["read_count"] == 0 and not r["positional"]:
            problems.append({
                "kind": "hash_without_reader",
                "key": r["key"],
                "detail": ("in CORE_QUALITY_PARAMS but no code reads it; "
                           "changing it invalidates intermediates for nothing"),
            })
        if r["read_count"] > 0 and not r["in_stage_deps"] \
                and not r["in_param_hash"] and not r["non_semantic"]:
            problems.append({
                "kind": "reader_without_declaration",
                "key": r["key"],
                "detail": ("read by code but neither in stage_deps nor in "
                           "CORE_QUALITY_PARAMS: changing it will not "
                           "invalidate anything"),
            })
        if r["in_stage_deps"] and r["read_count"] == 0 and not r["positional"]:
            problems.append({
                "kind": "stage_dep_without_reader",
                "key": r["key"],
                "detail": ("declared as a stage dependency but never read; "
                           "it will recompute stages without changing output"),
            })
    return rows, problems


def render(rows, problems):
    lines = ["# 逐参数能力表（R23-13，自动生成）", ""]
    lines.append("> 由 `tests/probes/param_capability_matrix.py` 生成。")
    lines.append("> **静态扫描只用来发现可疑项**；每一项的定论在 "
                 "`tests/test_r23_param_capability.py` 的行为断言里。")
    lines.append("")
    lines.append("## 1. 一致性结论")
    lines.append("")
    if not problems:
        lines.append("没有发现「进 hash 却没人读」或「读了却没声明」的键。")
    else:
        lines.append("| 类型 | 键 | 说明 |")
        lines.append("|---|---|---|")
        for p in problems:
            lines.append(f"| `{p['kind']}` | `{p['key']}` | {p['detail']} |")
    lines.append("")
    lines.append("## 2. 逐键表")
    lines.append("")
    lines.append("| 配置键 | 默认 | 读取点 | 进 hash | 进 stage_deps | 非语义 |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        pts = ", ".join(f"`{x}`" for x in r["read_points"][:3]) or "**无**"
        if r["read_count"] > 3:
            pts += f" (+{r['read_count'] - 3})"
        _paths = r.get("config_paths") or [r.get("config_path")]
        _plabel = _paths[0] + (f" (+{len(_paths) - 1})" if len(_paths) > 1 else "")
        lines.append("| `{}` | `{}` | {} | {} | {} | {}{} |".format(
            _plabel, str(r.get("default"))[:24], pts,
            "是" if r["in_param_hash"] else "",
            "是" if r["in_stage_deps"] else "",
            "是" if r["non_semantic"] else "",
            "（通用词）" if r.get("generic") else ""))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    rows, problems = scan()
    md = render(rows, problems)
    print(md[:3000])
    print(f"\n[matrix] keys={len(rows)} problems={len(problems)}")
    for p in problems:
        print(f"  - {p['kind']}: {p['key']}")
    if args.json:
        io.open(args.json, "w", encoding="utf-8").write(
            json.dumps({"rows": rows, "problems": problems},
                       ensure_ascii=False, indent=2, default=str))
    if args.md:
        io.open(args.md, "w", encoding="utf-8").write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
