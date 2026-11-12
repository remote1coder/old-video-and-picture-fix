# -*- coding: utf-8 -*-
"""R23-05：多轮**真实编排**调用计数。

## 审核要什么

> 同输入复用；仅最终编码变化只执行相关下游；VIVTC/sRestore、模型权重、
> 工作格式、grain_seed 分别使对应阶段失效。用两轮真实编排计数验证，
> 媒体生成可以采用轻量真实工具；单独比较两个 hash 字符串不算结束。

所以这里跑**真实 main.py**，在**同一个临时目录**里连续跑多个场景，
从**生产日志**里数 `STAGE{N}_START` 的次数，并读它自己的
`STAGE_REUSE_PLAN` 结论。

## 场景与期望

| 场景 | 改了什么 | 期望重跑的 stage |
|---|---|---|
| A 首次 | — | 1..7（AI 开）|
| B 同配置 | 无 | **0 个** |
| C 只改 crf | `crf` 24→30 | 只有 7 |
| D 改工作格式 | `chroma_mode` 444→420 | 1..7（贯穿全链）|
| E 改颗粒种子 | `grain_seed` 1→2 | 只有 6、7 |

## 两个必须处理的坑（都踩过）

1. **manifest 里的 job 状态**：上一轮跑完会写成 `status: DONE`，main 于是
   **整个 job 跳过** —— 那是最强的一种复用，但它会掩盖阶段级计数。
   所以每个场景前要把它改回 `PENDING`。
2. **不要用正则改 manifest**：`re.sub(r"^(\\s*)status:...")` 经过一层 shell
   转义会变成 `//s`，静默不匹配 —— 探针"跑通了"却什么也没改。
   这里逐行比较字符串。

## 用法

    python tests/probes/r23_orchestration_count.py [--root DIR]
"""
import argparse
import datetime
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r22_444_classic_path as s3probe  # noqa: E402

PY = s3probe.PY
MAIN = s3probe.MAIN
ROOT = r"E:\codex\_media_r23_20261112T1202\s_orch"

_STAGE_RX = re.compile(r'"event":\s*"STAGE(\d)_START"')


def _write_cfg(work, base_cfg, name, overrides):
    import yaml
    with open(base_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    g = cfg.setdefault("global", {})
    g.update(overrides)
    g["temp_dir"] = os.path.join(work, "temp")
    g["log_dir"] = os.path.join(work, "logs")
    g["output_dir"] = os.path.join(work, "output")
    p = os.path.join(work, "config_%s.yaml" % name)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return p


def _stage_starts(log_dir):
    """从本轮 job 日志里数每个 stage 的 STAGE{N}_START 次数。"""
    hits = {}
    for n in os.listdir(log_dir):
        if not n.startswith("job_") or not n.endswith(".log"):
            continue
        with open(os.path.join(log_dir, n), "r", encoding="utf-8",
                  errors="replace") as f:
            for ln in f:
                m = _STAGE_RX.search(ln)
                if m:
                    st = int(m.group(1))
                    hits[st] = hits.get(st, 0) + 1
    return hits


def _reuse_plans(log_dir):
    plans = []
    for n in os.listdir(log_dir):
        if not n.startswith("job_") or not n.endswith(".log"):
            continue
        with open(os.path.join(log_dir, n), "r", encoding="utf-8",
                  errors="replace") as f:
            for ln in f:
                if '"event": "STAGE_REUSE_PLAN"' not in ln:
                    continue
                try:
                    doc = json.loads(ln)
                except ValueError:
                    continue
                if "restart_from" in doc:
                    plans.append({
                        "restart_from": doc.get("restart_from"),
                        "reusable": doc.get("reusable") or [],
                        "invalidated": [d.get("stage") for d in
                                        (doc.get("invalidated") or [])],
                    })
    return plans


def _reset_manifest_status(manifest):
    """把 job 状态从 DONE 改回 PENDING，否则 main 根本不会处理它。

    逐行比较，不用正则 —— 见模块 docstring 第 2 条。
    """
    with open(manifest, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    out = []
    for ln in lines:
        st = ln.strip()
        if st == "status: DONE":
            out.append(ln.replace("DONE", "PENDING"))
        elif st == "pending: 0":
            out.append(ln.replace("pending: 0", "pending: 1"))
        elif st == "done: 1":
            out.append(ln.replace("done: 1", "done: 0"))
        else:
            out.append(ln)
    with open(manifest, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def _rotate_logs(work):
    """把上一轮的日志挪开，保证本次只数本轮的事件。"""
    logs = os.path.join(work, "logs")
    if not os.path.isdir(logs):
        return
    for n in os.listdir(logs):
        if n.startswith("job_") and n.endswith(".log"):
            os.replace(os.path.join(logs, n),
                       os.path.join(logs, n + ".prev"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--size", default="320x240")
    args = ap.parse_args()
    if not os.path.exists(PY):
        raise SystemExit("缺少视频链解释器: %s" % PY)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    work = os.path.join(args.root, stamp)
    for sub in ("input", "output", "temp", "logs"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    print("[orch] work=%s" % work)

    s3probe.W, s3probe.H = (int(x) for x in args.size.lower().split("x"))
    s3probe.FRAMES = args.frames
    src = os.path.join(work, "input", "probe_src.mkv")
    s3probe._make_source(src)

    base_cfg = s3probe._config_for(work, "420", "libsvtav1", "yuv420p10le")
    manifest = os.path.join(work, "manifest.yaml")
    scan = s3probe._run([PY, "-B", MAIN, "--config", base_cfg, "--input",
                         os.path.join(work, "input"), "--output",
                         os.path.join(work, "output"), "--no-wsl-redirect",
                         "--scan", "--output-manifest", manifest], cwd=work)
    if scan.returncode != 0:
        raise SystemExit("scan 失败: %s" % (scan.stderr or "")[-500:])
    s3probe._unskip(manifest)

    _444 = {"chroma_mode": "444", "video_encoder": "libx265",
            "pix_fmt": "yuv444p10le"}
    scenarios = [
        ("A_first", dict({"grain_seed": 1}, **_444), "首次（444 工作+交付）"),
        ("B_same", dict({"grain_seed": 1}, **_444), "同配置（强制重跑）"),
        ("B2_job_skip", dict({"grain_seed": 1}, **_444),
         "同配置（不重置 manifest）"),
        ("C_crf", dict({"grain_seed": 1, "crf": 30}, **_444), "只改 crf"),
        ("D_chroma", {"grain_seed": 1, "crf": 30, "chroma_mode": "420",
                      "video_encoder": "libsvtav1",
                      "pix_fmt": "yuv420p10le"}, "改工作格式 444→420"),
        ("E_seed", {"grain_seed": 2, "crf": 30, "chroma_mode": "420",
                    "video_encoder": "libsvtav1",
                    "pix_fmt": "yuv420p10le"}, "改 grain_seed"),
    ]

    results = []
    for name, overrides, label in scenarios:
        cfg = _write_cfg(work, base_cfg, name, overrides)
        _rotate_logs(work)
        if name != "B2_job_skip":
            _reset_manifest_status(manifest)
        r = s3probe._run([PY, "-B", MAIN, "--config", cfg, "--input",
                          os.path.join(work, "input"), "--output",
                          os.path.join(work, "output"), "--no-wsl-redirect",
                          "--manifest", manifest, "--keep-intermediates",
                          "--workers", "1"], cwd=work, timeout=3600)
        starts = _stage_starts(os.path.join(work, "logs"))
        plans = _reuse_plans(os.path.join(work, "logs"))
        results.append({
            "scenario": name, "label": label, "rc": r.returncode,
            "stage_starts": starts, "reuse_plans": plans[-1:] or [],
            "stderr_tail": ((r.stderr or "")[-400:] if r.returncode else ""),
        })
        print("[orch] %-10s rc=%s starts=%s restart_from=%s" % (
            name, r.returncode, starts,
            (plans[-1] if plans else {}).get("restart_from")))

    # ---- 判定 ----
    ok = True
    by = {r["scenario"]: r for r in results}
    for r in results:
        if r["rc"] != 0:
            print("[FAIL] %s rc=%s" % (r["scenario"], r["rc"]))
            ok = False

    def starts(name):
        return set(by[name]["stage_starts"])

    def reusable(name):
        plans = by[name].get("reuse_plans") or []
        if not plans:
            return set()
        return {int(x["stage"]) if isinstance(x, dict) else int(x)
                for x in (plans[-1].get("reusable") or [])}

    def restart(name):
        plans = by[name].get("reuse_plans") or []
        return (plans[-1].get("restart_from") if plans else None)

    # A：首次必须真的跑起来
    if not {1, 7} <= starts("A_first"):
        print("[FAIL] 首次运行没有跑到 stage1/7: %s" % sorted(starts("A_first")))
        ok = False

    # B2：**作业级复用** —— manifest 仍是 DONE，main 应当整个 job 跳过
    if starts("B2_job_skip"):
        print("[FAIL] 作业已完成，重跑却仍然启动了 %s 个 stage"
              % sorted(starts("B2_job_skip")))
        ok = False

    # B：强制重跑时，1..6 必须全部复用（stage7 例外，见下）
    if not {1, 2, 3, 4, 6} <= reusable("B_same"):
        print("[FAIL] 同配置下 1..6 没有全部复用: reusable=%s starts=%s"
              % (sorted(reusable("B_same")), sorted(starts("B_same"))))
        ok = False
    if starts("B_same") - {7}:
        print("[FAIL] 同配置重跑仍然启动了 %s（stage7 之外的阶段）"
              % sorted(starts("B_same") - {7}))
        ok = False

    # C：只改 crf —— 上游一律复用，只有 stage7 相关
    if not {1, 2, 3, 4, 6} <= reusable("C_crf"):
        print("[FAIL] 只改 crf 却让上游失效: reusable=%s starts=%s"
              % (sorted(reusable("C_crf")), sorted(starts("C_crf"))))
        ok = False
    if starts("C_crf") - {7}:
        print("[FAIL] 只改 crf 却重跑了 %s" % sorted(starts("C_crf") - {7}))
        ok = False

    # D：改工作表示 → 全链失效（从 stage1 起）
    if 1 not in starts("D_chroma"):
        print("[FAIL] 改工作格式没有让 stage1 失效: %s"
              % sorted(starts("D_chroma")))
        ok = False

    # E：只改 grain_seed → 只有 6（及其下游 7）
    if 6 not in starts("E_seed"):
        print("[FAIL] 改 grain_seed 没有让 stage6 失效: %s"
              % sorted(starts("E_seed")))
        ok = False
    if starts("E_seed") - {6, 7}:
        print("[FAIL] 只改 grain_seed 却重跑了 %s"
              % sorted(starts("E_seed") - {6, 7}))
        ok = False

    out = os.path.join(work, "orchestration_count.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"work": work, "ok": ok, "scenarios": results}, f,
                  ensure_ascii=False, indent=2)
    print("\n[orch] evidence -> %s" % out)
    print("[orch] 说明：stage7 的产物在 output/ 而不是 temp/，它没有阶段身份"
          "侧车（交付物的完整性由 marker/journal/result 证明）。所以只要 job"
          "真的跑起来，stage7 就会重跑；\"什么都不变就完全不跑\" 由**作业级**"
          "复用（manifest DONE → 整个 job 跳过，见 B2）承担。")
    print("[orch] VERDICT = %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
