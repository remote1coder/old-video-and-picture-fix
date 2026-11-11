# -*- coding: utf-8 -*-
"""R21-C01 复现探针：真实 `stage6.main()` + 真实小视频，只替换管道执行边界。

## 为什么不用"抽出来的尾段"复现

审核 R20-C01 用"提取原 main 的结果处理尾段"复现出两个 `None`。那能证明尾段逻辑
有问题，但**不能证明 `main()` 真的会返回 None** —— 中间还隔着输入解析、缓存
命中、配置读取等分支。本探针直接调**生产 `main()`**，只把
`_run_grain_pipeline`（真正跑 vspipe/ffmpeg 的那一步）替换成按用例返回的桩，
其余全部走真实代码。

## 用例

| case | 桩返回 | 修复前 | 修复后 |
|---|---|---|---|
| `ok` | ok=True | 0 | 0 |
| `encoder_fail` | ffmpeg rc=1（普通失败） | **None（=成功）** | 84 |
| `plugin_no_fallback` | GF3 缺失 + 配置禁用回退 | 84 | 84 |
| `plugin_fallback` | GF3 缺失 + 允许回退（回退也由桩返回成功） | **None（=成功）** | 0 + 降级侧车 |
| `cancel` | cancelled=True | 88 | 88 |

## 用法

    python tests/probes/r21_c01_stage6_tail.py --root E:\\...\\_media_r21_xxx
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _sh(cmd, timeout=120):
    return subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace", timeout=timeout, shell=False)


def make_clip(path):
    """造一个 2 帧的真视频（ffprobe 能读出宽高即可）。

    注意必须是 **.mkv**：FFV1 不能写进 mp4（实测
    `Could not find tag for codec ffv1 in stream #0`）。
    """
    r = _sh(["ffmpeg", "-hide_banner", "-v", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=0.2",
             "-c:v", "ffv1", "-pix_fmt", "yuv420p", path])
    if r.returncode != 0:
        raise RuntimeError(f"造片失败: {(r.stderr or '')[:300]}")
    return path


def write_config(path, addgrainc_enabled):
    text = f"""grain_factory3:
  enabled: true
  addgrainc_enabled: {str(addgrainc_enabled).lower()}
  addgrainc_ratio: 0.2
  g1str: 8.0
  g2str: 6.0
  g3str: 6.0
  temp_avg: 0.0
  analog_tape:
    g1str: 8.0
    g2str: 6.0
    g3str: 6.0
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def run_case(mod, case, work):
    """在独立目录里跑一次真实 main()，返回 (rc, stdout_tail)。"""
    import io
    import contextlib

    tdir = os.path.join(work, case)
    os.makedirs(tdir, exist_ok=True)
    clip = make_clip(os.path.join(tdir, "src.mkv"))
    # stage6 按 glob 找 step4 产物；把造好的片子复制成那个名字
    stem = "src"
    step4 = os.path.join(tdir, f"job1_{stem}_step4_postprocess_pdeadbeef.mkv")
    with open(clip, "rb") as a, open(step4, "wb") as b:
        b.write(a.read())

    # 是否允许 AddGrain 回退：按用例名决定
    allow_fb = case.startswith("plugin_enabled")
    cfg = write_config(os.path.join(tdir, "config.yaml"), allow_fb)
    params = {
        "temp_dir": tdir,
        "log_dir": os.path.join(tdir, "logs"),
        "no_wsl_redirect": True,
        "ai_enabled": False,
        "face_restore": "",
        "stage_inactivity_timeout": 60,
    }
    pfile = os.path.join(tdir, "params.json")
    with open(pfile, "w", encoding="utf-8") as f:
        json.dump(params, f)

    # 本机实测（2026-11-11，tools_r4/vs_portable/r72）：
    #   hasattr(core,'grainfactory') → False
    #   core.grainfactory.GrainFactory3 →
    #   AttributeError: No attribute with the name grainfactory exists.
    #                   Did you mistype a plugin namespace or forget to install a plugin?
    # 注意这句里**没有** "grainfactory3" —— 旧关键词表永远匹配不到真实报错。
    GF3 = ("AttributeError: No attribute with the name grainfactory exists. "
           "Did you mistype a plugin namespace or forget to install a plugin?")
    # 另一种真实形态：havsfunc 存在但版本太老，没有 GrainFactory3
    GF3_OLD = ("AttributeError: module 'havsfunc' has no attribute "
               "'GrainFactory3'")

    def _mk(**kw):
        """构造结果对象，**只传该版本真实存在的字段**。

        这样同一个探针既能跑修复前（没有 vspipe_tail/encoder_tail）
        也能跑修复后，不必为两个版本各维护一份。
        """
        allowed = set(getattr(mod.GrainPipelineResult,
                              "__dataclass_fields__", {}) or {})
        return mod.GrainPipelineResult(
            **{k: v for k, v in kw.items() if k in allowed})

    def _stub(vs_script_str, vs_path, vspipe_cmd, ffmpeg_cmd, output_path,
              logger, fallback_label="", inactivity_s=1800.0):
        is_fallback = bool(fallback_label)
        if case == "ok":
            with open(output_path, "wb") as f:
                f.write(b"FAKEVIDEO")
            return _mk(ok=True, vspipe_rc=0, ffmpeg_rc=0)
        if case == "cancel":
            return _mk(ok=False, cancelled=True, failed_stage="wait")
        if case == "encoder_fail":
            return _mk(ok=False, vspipe_rc=0, ffmpeg_rc=1,
                       failed_stage="ffmpeg",
                       encoder_tail="disk quota exceeded",
                       stderr_tail="disk quota exceeded",
                       detail="encoder (ffmpeg) failed rc=1")
        # plugin_* 用例：ffmpeg 没失败（rc=0），只是 vspipe 起不来
        if case.startswith("plugin_disabled"):
            msg = GF3_OLD if case.endswith("oldmsg") else GF3
            return _mk(ok=False, vspipe_rc=1, ffmpeg_rc=0,
                       failed_stage="vspipe",
                       vspipe_tail=msg, stderr_tail=msg,
                       detail="producer (vspipe) failed rc=1")
        # plugin_enabled_*：**连带失败** —— ffmpeg 也因为没输入挂了
        if is_fallback:                      # 回退这一趟按成功返回
            with open(output_path, "wb") as f:
                f.write(b"FAKEVIDEO-ADDGRAIN")
            return _mk(ok=True, vspipe_rc=0, ffmpeg_rc=0)
        msg = GF3_OLD if case.endswith("oldmsg") else GF3
        return _mk(ok=False, vspipe_rc=1, ffmpeg_rc=1, failed_stage="vspipe",
                   vspipe_tail=msg, stderr_tail=msg,
                   detail="producer (vspipe) failed rc=1")

    mod._run_grain_pipeline = _stub
    old_argv = sys.argv
    sys.argv = ["stage6", "--input", clip, "--job-id", "job1",
                "--config", cfg, "--params-file", pfile]
    buf = io.StringIO()
    rc = None
    try:
        with contextlib.redirect_stdout(buf):
            rc = mod.main()
    finally:
        sys.argv = old_argv
    return rc, buf.getvalue()[-300:]


CASES = ["ok", "encoder_fail",
         "plugin_disabled_realmsg", "plugin_disabled_oldmsg",
         "plugin_enabled_realmsg", "plugin_enabled_oldmsg",
         "cancel"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="本轮隔离根（自建目录，绝不复用旧产物）")
    ap.add_argument("--out", default=None, help="结果 JSON 路径")
    args = ap.parse_args()

    work = os.path.join(os.path.abspath(args.root), "c01_probe")
    os.makedirs(work, exist_ok=True)

    for p in (_ROOT,
              os.path.join(_ROOT, "shared_modules"),
              os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import stage6_grain_synthesis as mod

    results = {}
    for case in CASES:
        try:
            rc, tail = run_case(mod, case, work)
        except Exception as exc:
            results[case] = {"rc": "EXCEPTION", "error": str(exc)}
            print(f"  {case:22s} EXCEPTION {exc}")
            continue
        note = os.path.join(work, case)
        degraded = any(n.endswith(".degraded.json")
                       for n in os.listdir(note)) if os.path.isdir(note) else False
        results[case] = {"rc": rc, "type": type(rc).__name__,
                         "degraded_sidecar": degraded, "stdout_tail": tail}
        print(f"  {case:22s} rc={rc!r:8s} type={type(rc).__name__:8s} "
              f"degraded={degraded}")

    out = args.out or os.path.join(work, "c01_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  json: {out}")

    # 判据：任何 None 都是失败（sys.exit(None) = 0 = 成功）
    bad = [k for k, v in results.items() if v.get("rc") is None]
    err = [k for k, v in results.items() if v.get("rc") == "EXCEPTION"]
    if err:
        print(f"  [FAIL] 这些用例抛异常、根本没跑起来: {err}")
        return 2
    if bad:
        print(f"  [FAIL] 这些用例返回 None（= 成功退出）: {bad}")
        return 1
    print("  [OK] 没有用例返回 None")
    return 0


if __name__ == "__main__":
    sys.exit(main())
