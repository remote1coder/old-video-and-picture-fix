# -*- coding: utf-8 -*-
"""G34 融合 vs 经典：**严格**等价性对照（第十九轮重写）。

## 审核 R18-A08 指出的问题（逐条对应）

1. **项目根定位错误**：
   > 归档到 tests/probes 后，PROJ 仍用两次 dirname，得到的是 tests 目录。
   > 从项目根启动不能改变 `__file__` 的这个结果。

   旧代码 `PROJ = dirname(dirname(abspath(__file__)))`：
   `tests/probes/x.py` → `tests/probes` → `tests`。**少了最后一级**。
   修：`Path(__file__).resolve().parents[2]`（probes → tests → repo），
   并允许 `--repo` 显式覆盖。

2. **硬编码历史测试根 / 会删既有目录**：
   > 脚本还硬编码历史测试根，并包含删除既有 work 目录的操作。

   旧代码 `ROOT = E:\\codex\\_media_r17_20261031T0004` + `shutil.rmtree(work)`。
   修：`--work` 可选，默认在 repo 的**父目录**下 `mkdtemp`（一定是新目录）；
   调用方给了已存在的非空 `--work` 就**拒绝**，绝不 rmtree。

3. **判据不充分**：
   > 探针修了零帧假通过，但没有要求全部帧比较、比较进程成功、
   > 两次流水线成功或确实执行融合。

   受控输入下旧版会误报 IDENTICAL 的情形：
   - 两个文件声明 60 帧，只比较到 1 帧 inf → IDENTICAL
   - 两条流水线均 rc=86，仍留有可比较的 step4 → IDENTICAL
   - 两例都没有融合回执、fused 仍有 step3 → 仍不阻断

   修：见 `evaluate()`，**任一条不满足就不得报 IDENTICAL**。

4. **失败也退出 0**：现在失败返回非零（1=未通过，2=参数/环境错误）。

## 用法

    python tests/probes/g34_equiv_det.py --frames 20
    python tests/probes/g34_equiv_det.py --repo E:\\..\\repo --work E:\\..\\out2 --input x.mp4

## 判据（全部满足才算 IDENTICAL）

- 两条流水线 rc 均为 0；
- classic 有 step3、fused **没有** step3（否则说明融合没真正跑）；
- fused 有 `*.fused_stage34.json` 回执；
- 两边尺寸 / pix_fmt / 帧数一致，且等于 `--frames`；
- 逐帧 PSNR 覆盖**全部**帧（部分比较不是证据）；
- y/u/v 三个平面**都有**数据；
- 所有帧都有限且为 inf（完全相同）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _repo_root() -> str:
    """tests/probes/x.py → parents[0]=probes, [1]=tests, [2]=repo。"""
    return str(Path(__file__).resolve().parents[2])


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace", shell=False, **kw)


def probe_timing(path):
    """用**视频流**的 packet 时间戳观测时间轴（R22-A07 重写）。

    R21 只留了"帧数 / 首 PTS / 末 PTS / 末端覆盖 / 中位间隔"五个摘要，
    审核给出了一组摘要相同、逐帧显然不同的反例：

        0, 0.04, 0.08, 0.12, 0.16
        0, 0.04, 0.10, 0.12, 0.16      ← 中段错位，摘要却一样

    所以现在**保留逐帧 PTS**（`pts`），由 `evaluate()` 逐点比较。
    另外：不再 `sort()` —— 排序会把"原始顺序乱掉"这个事实抹掉，
    而乱序本身就是缺陷。

    Returns:
        ``{frames, pts, durations, first_pts, last_pts, last_end,
        last_frame_duration, median_interval}``；
        失败时 ``{"error": str}``。
    """
    r = sh(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts_time,duration_time",
            "-of", "csv=p=0", path])
    if r.returncode != 0:
        return {"error": f"ffprobe rc={r.returncode}"}

    pts_list = []
    dur_list = []
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        try:
            pts = float(parts[0])
        except (TypeError, ValueError):
            continue
        dur = 0.0
        if len(parts) > 1:
            try:
                dur = float(parts[1])
            except (TypeError, ValueError):
                dur = 0.0
        pts_list.append(pts)
        dur_list.append(dur)

    if not pts_list:
        return {"error": "no packet timestamps"}

    # ---- 文件顺序 vs 显示顺序 ----
    # ffprobe 的 `packet=pts_time` 按**文件/解码顺序**输出；有 B 帧时
    # PTS 本来就不单调（实测 1080p 合成片：0.133467 -> 0.066733）。
    # 所以这里**排序得到显示顺序**，同时把原始顺序留作证据。
    # 真正的缺陷是"排序后仍有重复 PTS"，那说明两帧抢同一个时刻。
    raw_order = [round(x, 6) for x in pts_list]
    pairs = sorted(zip(pts_list, dur_list), key=lambda t: t[0])
    disp_pts = [p for p, _d in pairs]
    disp_dur = [d for _p, d in pairs]
    for i in range(1, len(disp_pts)):
        if abs(disp_pts[i] - disp_pts[i - 1]) < 1e-9:
            return {"error": (f"duplicate presentation timestamp at index {i} "
                              f"({disp_pts[i]!r})")}

    last_pts, last_dur = disp_pts[-1], disp_dur[-1]
    # 是否与文件顺序一致（不一致 = 有 B 帧，属于正常，但要留证据）
    reordered = (raw_order != [round(x, 6) for x in disp_pts])
    d = [b - a for a, b in zip(disp_pts, disp_pts[1:]) if b > a]
    med = sorted(d)[len(d) // 2] if d else 0.0
    return {
        "frames": len(disp_pts),
        # 逐帧证据（evaluate 逐点比；result.json 里也留着）
        "pts": disp_pts,
        "durations": disp_dur,
        # 原始（文件/解码）顺序 —— 与上面不同说明有 B 帧
        "pts_file_order": raw_order,
        "reordered_by_pts": bool(reordered),
        "first_pts": round(disp_pts[0], 6),
        "last_pts": round(last_pts, 6),
        # 末帧覆盖 = 末帧 pts + **它自己的** duration；缺则等于 last_pts
        "last_end": round(last_pts + (last_dur if last_dur > 0 else 0.0), 6),
        "last_frame_duration": round(last_dur, 6),
        "median_interval": round(med, 6),
    }


def probe(path):
    """探测媒体参数。

    审核 R19-B07 指出三个误判来源，这里逐一堵住：

    1. **ffprobe 返回非零但 stdout 留有可解析 JSON** —— 旧实现只看 JSON，
       返回码被丢弃。现在 rc != 0 直接返回 ``{"error": ...}``。
    2. **两份探测结果都缺 width/height/pix_fmt，缺值彼此相等** —— 旧实现
       把这些字段原样放进 dict，``None == None`` 就算"格式一致"。
       现在缺失/非法直接 error。
    3. 帧数必须**可证明**（正整数），不能拿 None 当 0 用。

    Returns:
        dict：成功时含已校验的字段；失败时只有 ``{"error": str}``。
    """
    r = sh(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries",
            "stream=nb_read_frames,width,height,pix_fmt,r_frame_rate,time_base",
            "-of", "json", path])
    if r.returncode != 0:
        return {"error": f"ffprobe rc={r.returncode}: "
                         f"{(r.stderr or '')[-300:]}"}
    try:
        streams = json.loads(r.stdout or "{}").get("streams") or []
    except Exception as exc:
        return {"error": f"ffprobe stdout is not JSON: {exc}"}
    if not streams:
        return {"error": "ffprobe returned no video stream"}

    s = streams[0]
    info = {k: s.get(k) for k in
            ("width", "height", "pix_fmt", "nb_read_frames", "r_frame_rate",
             "time_base")}

    for k in ("width", "height", "nb_read_frames"):
        try:
            v = int(info[k])
        except (TypeError, ValueError):
            return {"error": f"{k} missing or not an integer: {info[k]!r}"}
        if v <= 0:
            return {"error": f"{k} must be positive, got {v}"}
        info[k] = v

    if not info.get("pix_fmt") or not isinstance(info["pix_fmt"], str):
        return {"error": f"pix_fmt missing/invalid: {info.get('pix_fmt')!r}"}

    # ---- R21-C08：时间轴也必须**独立**核对 ----
    # 审核："evaluator 比较尺寸、格式、帧数，却未核对实际 PTS、尾帧覆盖和
    #       时间映射。受控输入中，声明时间不同仍判 IDENTICAL。"
    # 只比帧数/尺寸无法发现"像素一样但时间轴不同"或"末帧被重复填充凑数"。
    tl = probe_timing(path)
    if "error" in tl:
        return {"error": f"timing probe failed: {tl['error']}"}
    if tl["frames"] != info["nb_read_frames"]:
        return {"error": (f"packet count {tl['frames']} != nb_read_frames "
                          f"{info['nb_read_frames']}")}
    info.update({k: tl[k] for k in
                 ("frames", "first_pts", "last_pts", "last_end",
                  "median_interval", "last_frame_duration",
                  # R22-A07：逐帧证据也要带出来（evaluate 逐点比）
                  "pts", "durations")})
    return info


def make_clip(dst_dir, frames):
    """造测试片。只在**自己创建的**目录里替换自产素材。"""
    os.makedirs(dst_dir, exist_ok=True)
    clip = os.path.join(dst_dir, "src.mp4")
    if os.path.exists(clip) and os.path.getsize(clip) > 0:
        info = probe(clip)
        if int(info.get("nb_read_frames") or 0) == frames:
            return clip
        os.remove(clip)
    r = sh(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "lavfi",
            "-i", "testsrc2=size=1920x1080:rate=30000/1001:duration=10",
            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
            "-frames:v", str(frames), clip])
    if r.returncode != 0:
        raise SystemExit("造片失败: " + (r.stderr or "")[-300:])
    info = probe(clip)
    if "error" in info:
        raise SystemExit(f"合成片探测失败：{info['error']}")
    if info["nb_read_frames"] != frames:
        raise SystemExit(
            f"合成片帧数不符：期望 {frames}，实际 {info['nb_read_frames']}")
    return clip, info


def _fps_of(info):
    """从 probe 结果算 fps（失败给 None）。"""
    raw = str(info.get("r_frame_rate") or "")
    if "/" in raw:
        try:
            n, d = raw.split("/", 1)
            n, d = float(n), float(d)
            return n / d if d > 0 else None
        except Exception:
            return None
    try:
        return float(raw) or None
    except Exception:
        return None


def write_manifest(work, clip, frames, info=None):
    """写 manifest。

    审核 R19-B08：
    > `--input` 没有实际裁到 `--frames`，manifest 却写死 1920×1080、
    > 30000/1001、progressive 和 `frames/30` 时长。

    现在所有媒体字段都来自 ``probe(clip)`` 的**实测**结果；
    ``info`` 缺失时直接拒绝（不允许用假元数据凑合）。
    """
    import yaml
    if not info or "error" in info:
        raise SystemExit("write_manifest 需要真实的 probe 结果，"
                         f"拒绝写死元数据：{info!r}")

    for sub in ("temp", "out", "logs"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)

    _fps = _fps_of(info)
    _dur = (frames / _fps) if (_fps and _fps > 0) else 0.0

    job = {
        "job_id": "g34", "input_path": clip,
        "detected_class": "HDV_AVCHD", "confidence": 0.98,
        "field_order": "progressive", "needs_ivtc": False, "is_psf": False,
        "field_segments": [], "status": "PENDING", "priority": 0,
        "ai_upscale_banned": False, "recommended_params": {},
        "user_overrides": {
            "ai_stream_raw": True,
            "force_allow_banned_upscale": True,
            "original_fps": str(info.get("r_frame_rate") or ""),
            "cleanup_intermediates": False,
            # 确定性模式：不开的话 stage1 的抖动会盖住真实差异
            "vs_deterministic": True,
        },
        "duration": _dur,
        "original_width": int(info["width"]),
        "original_height": int(info["height"]),
        "file_size": os.path.getsize(clip), "file_mtime": 0.0,
        "file_hash_prefix": "", "run_generation": 1,
        "temp_dir": os.path.join(work, "temp"),
        "log_path": os.path.join(work, "logs", "job.log"),
        "output_path": os.path.join(work, "out", "g34_restored.mkv"),
    }
    path = os.path.join(work, "m.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"meta": {"generated_at": "2026-11-09T22:00:00+08:00",
                                 "source_dirs": [os.path.dirname(clip)],
                                 "config_path": "", "total_files": 1,
                                 "pending": 1, "done": 0, "failed": 0,
                                 "skipped": 0, "run_generation": 1},
                        "jobs": [job]}, f, allow_unicode=True, sort_keys=False)
    return path


def trim_to_frames(src, dst_dir, frames):
    """把外部素材**真裁**到 ``frames`` 帧，返回 (路径, probe)。

    审核 R19-B08：
    > `--frames` 必须实际限制所处理样本或拒绝不匹配输入。

    做法：先 probe 源（拿真实尺寸/fps/pix_fmt），再用 ``-frames:v N``
    精确裁帧并重编码到 mp4（避免长 GOP 首帧依赖）。
    裁出来的片子仍按**实测**结果写 manifest。
    """
    info_src = probe(src)
    if "error" in info_src:
        raise SystemExit(f"--input 无法探测：{info_src['error']}")

    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"trim_{frames}f.mp4")
    r = sh(["ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-i", src, "-frames:v", str(frames),
            "-an", "-c:v", "libx264", "-crf", "14",
            "-pix_fmt", "yuv420p", dst])
    if r.returncode != 0 or not os.path.exists(dst):
        raise SystemExit("裁帧失败：" + (r.stderr or "")[-300:])

    info = probe(dst)
    if "error" in info:
        raise SystemExit(f"裁出的样本无法探测：{info['error']}")
    if info["nb_read_frames"] != frames:
        raise SystemExit(
            f"裁帧结果帧数不符：期望 {frames}，实际 {info['nb_read_frames']}")
    return dst, info


def run_mode(repo, work, clip, frames, mode, info=None):
    """跑一次；`work` 必须是**调用方已创建好的**新目录。"""
    os.makedirs(work, exist_ok=True)
    man = write_manifest(work, clip, frames, info=info)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([
        os.path.join(repo, "tools_r4"),
        os.path.join(repo, "tools_r4", "gmic", "gmic-4.0.5-cli-win64"),
        os.path.join(repo, "tools_r4", "exiftool", "exiftool-13.59_64"),
        env.get("PATH", "")])
    py = os.path.join(repo, "tools_r4", "vs_portable", "r72", "python.exe")
    r = sh([py, os.path.join(repo, "video_restoration_pipeline", "main.py"),
            "--manifest", man,
            "--config", os.path.join(repo, "video_restoration_pipeline",
                                     "config.yaml"),
            "--mode", "batch", "--no-wsl-redirect", "--ai-enabled",
            "--ai-upscale-factor", "2", "--force-allow-banned-upscale",
            "--workers", "1", "--allow-write-root", work,
            "--chain", mode], env=env)

    td = os.path.join(work, "temp")
    names = sorted(os.listdir(td)) if os.path.isdir(td) else []
    step3 = [n for n in names if "_step3_" in n and n.endswith(".mkv")]
    step4 = [n for n in names if "_step4_" in n and n.endswith(".mkv")]
    receipts = [n for n in names if n.endswith(".fused_stage34.json")]
    return {
        "mode": mode, "rc": r.returncode, "work": work,
        "step3": os.path.join(td, step3[-1]) if step3 else None,
        "step4": os.path.join(td, step4[-1]) if step4 else None,
        "receipts": receipts,
        "stdout_tail": (r.stdout or "").splitlines()[-4:],
        "stderr_tail": (r.stderr or "").splitlines()[-6:],
    }


def compare(a, b):
    """逐帧 PSNR。`frames` 是**实际比较到的帧数**。"""
    r = sh(["ffmpeg", "-hide_banner", "-v", "info", "-i", a, "-i", b,
            "-lavfi", "psnr=stats_file=-", "-f", "null", "-"])
    rows = []
    for ln in (r.stdout or "").splitlines():
        if ln.startswith("n:"):
            rows.append(dict(x.split(":", 1) for x in ln.split() if ":" in x))
    summary = [l.strip()[:120] for l in (r.stderr or "").splitlines()
               if "PSNR" in l]

    # 审核 R19-B07：比较日志重复同一帧序号、行数凑够目标数也会被判通过。
    # 这里显式核对 n: 序列必须是 1..N 的严格连续整数。
    seq = []
    for d in rows:
        try:
            seq.append(int(d.get("n")))
        except (TypeError, ValueError):
            seq.append(None)
    dup = sorted({x for x in seq if seq.count(x) > 1})
    noncontig = (seq != list(range(1, len(seq) + 1)))

    return {"returncode": r.returncode, "frames": len(rows),
            "rows": rows, "summary": summary,
            "seq": seq, "dup": dup[:5], "noncontiguous": bool(noncontig),
            "stderr_tail": (r.stderr or "").splitlines()[-3:]}


#: 只接受这些字面量作为 "完全相同"。审核 R19-B07：
#: 「拒绝随便以 inf 开头的坏值」—— ``startswith("inf")`` 会放过 ``infinity_bad``。
#: 只接受**正**无穷 —— 它才是"逐像素完全相同"（MSE=0）的唯一合法表示。
#:
#: R21-C08：审核指出旧集合接受 `-inf/-infinity`。
#: 负无穷在 PSNR 里没有"零误差"的含义（它来自坏的 1/x 或错误统计），
#: 把它当 inf 计数等于**用错误数值证明等价**。
_INF_TOKENS = frozenset({"inf", "+inf", "infinity", "+infinity"})

#: 负无穷：显式判为**非法值**，既不算 inf 也不算有限值。
_NEG_INF_TOKENS = frozenset({"-inf", "-infinity"})


def _plane(rows, key, total=None):
    """某个平面在**每一行**都必须有合法值。

    审核 R19-B07 的第一个误判：
    > 20 行比较数据，仅第 1 行有 V 平面，另外 19 行缺 V —— 旧实现
    > ``if v is not None`` 跳过缺失行，``present`` 仍为 True。

    现在要求：缺失行数 == 0、非法值行数 == 0、有效值个数 == ``total``。

    Args:
        rows: psnr 逐帧行（dict）。
        key: ``psnr_y`` / ``psnr_u`` / ``psnr_v``。
        total: 期望行数；``None`` 时用 ``len(rows)``。

    Returns:
        dict：``present`` / ``n`` / ``n_missing`` / ``missing_rows`` /
        ``bad_rows`` / ``n_inf`` / ``min`` / ``total``。
    """
    total = len(rows) if total is None else int(total)
    vals = []
    missing_idx = []
    bad_idx = []
    for i, d in enumerate(rows):
        v = d.get(key)
        if v is None:
            missing_idx.append(i + 1)
            continue
        tok = v.strip().lower()
        if tok in _NEG_INF_TOKENS:
            # 负无穷不能证明零误差 → 记成非法行
            bad_idx.append(i + 1)
            continue
        if tok in _INF_TOKENS:
            vals.append(None)
            continue
        try:
            vals.append(float(tok))
        except ValueError:
            bad_idx.append(i + 1)

    ok = (not missing_idx) and (not bad_idx) and len(vals) == total and total > 0
    return {
        "present": ok,
        "n": len(vals),
        "total": total,
        "n_missing": len(missing_idx),
        "missing_rows": missing_idx[:5],
        "bad_rows": bad_idx[:5],
        "n_inf": sum(1 for x in vals if x is None),
        "min": min((x for x in vals if x is not None), default=None),
    }


def evaluate(classic, fused, info_c, info_f, cmp_result, frames):
    """严格判定。**任一条不满足就不能报 IDENTICAL。**

    Returns:
        ``(verdict, reasons)``
    """
    reasons = []

    # ---- 1) 两条流水线都必须成功 ----
    if classic["rc"] != 0:
        reasons.append(f"classic pipeline rc={classic['rc']} (must be 0)")
    if fused["rc"] != 0:
        reasons.append(f"fused pipeline rc={fused['rc']} (must be 0)")

    # ---- 2) 必须**确实执行了**融合 ----
    if classic["step3"] is None:
        reasons.append("classic produced no step3 (unexpected)")
    if fused["step3"] is not None:
        reasons.append("fused produced a step3 — fusion did not actually run")
    if not fused["receipts"]:
        reasons.append("fused produced no *.fused_stage34.json receipt")
    if classic["step4"] is None or fused["step4"] is None:
        reasons.append("missing step4 on one side")

    # ---- 3) 帧数：两边相等且等于请求值 ----
    try:
        n_c = int(info_c.get("nb_read_frames") or 0)
        n_f = int(info_f.get("nb_read_frames") or 0)
    except (TypeError, ValueError):
        n_c = n_f = 0
    if n_c != n_f:
        reasons.append(f"frame count differs: classic={n_c} fused={n_f}")
    if frames and n_c != frames:
        reasons.append(f"classic frame count {n_c} != requested {frames}")

    # ---- 4) 尺寸与像素格式（先拒绝 error，再比真值）----
    for tag, info in (("classic", info_c), ("fused", info_f)):
        if "error" in info:
            reasons.append(f"{tag} probe failed: {info['error']}")
    if "error" not in info_c and "error" not in info_f:
        for k in ("width", "height", "pix_fmt"):
            # 审核 R19-B07：缺值彼此相等不能算"格式一致"
            if info_c.get(k) is None or info_f.get(k) is None:
                reasons.append(
                    f"{k} unproven on one side: "
                    f"classic={info_c.get(k)!r} fused={info_f.get(k)!r}")
            elif info_c.get(k) != info_f.get(k):
                reasons.append(
                    f"{k} differs: {info_c.get(k)} vs {info_f.get(k)}")

    # ---- 4b) 时间轴：数值有限、长度完整、末帧覆盖（R23-B10）----
    # 审核用原 evaluator 的受控输入复现四类"仍然 IDENTICAL"：
    #   * 一侧中段 PTS = NaN（`abs(nan - x) > tol` 恒为 False，从不报差异）
    #   * 两侧 PTS 数组**都为空**却声称 5 帧
    #   * 两侧 durations 数组缺失
    #   * 尾帧 duration 均为 0
    # 现在逐条变成判据：先证明**观测本身完整**，再谈相等。
    _tol = 0.002

    def _all_finite(seq):
        try:
            return all(isinstance(x, (int, float)) and not isinstance(x, bool)
                       and math.isfinite(float(x)) for x in seq)
        except (TypeError, ValueError):
            return False

    if "error" not in info_c and "error" not in info_f:
        for key in ("frames", "first_pts", "last_pts", "last_end"):
            a, b = info_c.get(key), info_f.get(key)
            if a is None or b is None:
                reasons.append(f"timing field {key} unproven on one side")
                continue
            if key == "frames":
                if int(a) != int(b):
                    reasons.append(f"timing frame count differs: {a} vs {b}")
                continue
            if not (_all_finite([a]) and _all_finite([b])):
                reasons.append(f"timing field {key} is not finite: "
                               f"classic={a!r} fused={b!r}")
                continue
            if abs(float(a) - float(b)) > _tol:
                reasons.append(
                    f"{key} differs: classic={a} fused={b} "
                    f"(tolerance {_tol}s)")

        # ---- 逐帧 PTS：长度必须等于解码帧数，数值必须有限 ----
        _pc, _pf = info_c.get("pts"), info_f.get("pts")
        if not isinstance(_pc, list) or not isinstance(_pf, list):
            reasons.append(
                "per-frame PTS arrays are missing; timing equality is "
                "unproven (a summary is not evidence)")
        elif not _pc or not _pf:
            reasons.append("per-frame PTS arrays are empty; timing equality "
                           "is unproven (an empty table is not evidence)")
        elif not _all_finite(_pc) or not _all_finite(_pf):
            reasons.append("per-frame PTS contains non-finite values")
        elif n_c and (len(_pc) != n_c or len(_pf) != n_c):
            reasons.append(
                f"per-frame PTS length {len(_pc)}/{len(_pf)} != decoded "
                f"frames {n_c}")
        elif len(_pc) != len(_pf):
            reasons.append(
                f"per-frame PTS length differs: {len(_pc)} vs {len(_pf)}")
        else:
            for _i, (_x, _y) in enumerate(zip(_pc, _pf)):
                if abs(float(_x) - float(_y)) > _tol:
                    reasons.append(
                        f"per-frame PTS differs at frame {_i}: "
                        f"classic={_x} fused={_y} (tolerance {_tol}s)")
                    break

        # ---- 逐帧时长：必须存在、长度一致、数值有限、尾帧 > 0 ----
        _dc, _df = info_c.get("durations"), info_f.get("durations")
        if not isinstance(_dc, list) or not isinstance(_df, list):
            reasons.append(
                "per-frame durations are missing; the last-frame coverage "
                "cannot be proven")
        elif not _dc or not _df:
            reasons.append("per-frame durations are empty")
        elif not _all_finite(_dc) or not _all_finite(_df):
            reasons.append("per-frame durations contain non-finite values")
        elif n_c and (len(_dc) != n_c or len(_df) != n_c):
            reasons.append(
                f"per-frame durations length {len(_dc)}/{len(_df)} != decoded "
                f"frames {n_c}")
        else:
            if not (float(_dc[-1]) > 0) or not (float(_df[-1]) > 0):
                reasons.append(
                    "the last frame duration is zero on at least one side; "
                    "the tail is not covered")
            elif abs(float(_dc[-1]) - float(_df[-1])) > _tol:
                reasons.append(
                    f"the last frame duration differs: "
                    f"classic={_dc[-1]} fused={_df[-1]}")

    # ---- 5) 比较本身必须成功且覆盖全部帧 ----
    if cmp_result.get("returncode") != 0:
        reasons.append(f"psnr comparison rc={cmp_result.get('returncode')}")
    got = cmp_result.get("frames", 0)
    if cmp_result.get("dup"):
        reasons.append(
            f"psnr log repeats frame numbers: {cmp_result['dup']}")
    if cmp_result.get("noncontiguous"):
        reasons.append(
            "psnr log frame numbers are not 1..N contiguous "
            "(padding/duplicate rows are not evidence)")
    if got <= 0:
        reasons.append("psnr comparison produced no frames")
    elif n_c and got != n_c:
        reasons.append(
            f"psnr covered {got} of {n_c} frames — partial comparison "
            f"is not evidence")

    # ---- 6) 三个平面都要有数据，且全部 inf ----
    _rows = cmp_result.get("rows") or []
    planes = {k: _plane(_rows, f"psnr_{k}", total=got)
              for k in ("y", "u", "v")}
    missing = [k for k, v in planes.items() if not v["present"]]
    if missing:
        reasons.append(f"planes missing from comparison: {missing}")
    else:
        finite = {k: v for k, v in planes.items() if v["n_inf"] != v["n"]}
        if finite:
            detail = ", ".join(
                f"{k}:{v['n'] - v['n_inf']} finite (min {v['min']})"
                for k, v in finite.items())
            reasons.append(f"frames are not identical — {detail}")

    if reasons:
        return "NOT_IDENTICAL", reasons
    return "IDENTICAL", []


def main():
    ap = argparse.ArgumentParser(description="G34 vs classic strict equivalence")
    ap.add_argument("--repo", default=None, help="项目根（默认按 __file__ 推断）")
    ap.add_argument("--work", default=None,
                    help="输出根（默认在 repo 父目录下 mkdtemp，绝不覆盖已有目录）")
    ap.add_argument("--input", default=None, help="用指定素材（跳过造片）")
    ap.add_argument("--frames", type=int, default=20)
    args = ap.parse_args()

    repo = os.path.abspath(args.repo or _repo_root())
    if not os.path.isfile(os.path.join(repo, "video_restoration_pipeline",
                                       "main.py")):
        print("  [错误] --repo 下找不到 video_restoration_pipeline/main.py：%s"
              % repo)
        return 2

    created_here = False
    if args.work:
        work_root = os.path.abspath(args.work)
        if os.path.isdir(work_root) and os.listdir(work_root):
            print("  [错误] --work 目录已存在且非空，拒绝覆盖：%s" % work_root)
            return 2
        os.makedirs(work_root, exist_ok=True)
    else:
        parent = os.path.dirname(repo) or tempfile.gettempdir()
        work_root = tempfile.mkdtemp(prefix="r19_g34_", dir=parent)
        created_here = True

    print("  repo    : %s" % repo)
    print("  work    : %s%s" % (work_root,
                                "  (本轮新建)" if created_here else ""))
    print("  frames  : %d" % args.frames)

    if args.input:
        clip = os.path.abspath(args.input)
        if not os.path.isfile(clip):
            print("  [错误] --input 不存在：%s" % clip)
            return 2
        print("  input   : %s（外部素材，按 --frames 实测裁剪）" % clip)
        clip, info = trim_to_frames(clip, os.path.join(work_root, "in"),
                                    args.frames)
        print("  trimmed : %s" % clip)
    else:
        clip, info = make_clip(os.path.join(work_root, "in"), args.frames)
        print("  input   : %s（本轮生成）" % clip)

    classic = run_mode(repo, os.path.join(work_root, "classic"), clip,
                       args.frames, "none", info=info)
    fused = run_mode(repo, os.path.join(work_root, "fused"), clip,
                     args.frames, "stage34", info=info)
    print("  classic rc=%d step3=%s" % (classic["rc"],
                                        bool(classic["step3"])))
    print("  fused   rc=%d step3=%s receipts=%s"
          % (fused["rc"], bool(fused["step3"]), fused["receipts"]))

    info_c = info_f = {}
    cmp_result = {"frames": 0, "rows": [], "returncode": None}
    if classic["step4"] and fused["step4"]:
        info_c = probe(classic["step4"])
        info_f = probe(fused["step4"])
        cmp_result = compare(classic["step4"], fused["step4"])
        verdict, reasons = evaluate(classic, fused, info_c, info_f,
                                    cmp_result, args.frames)
    else:
        verdict = "NOT_IDENTICAL"
        reasons = ["missing step4 on one side — nothing to compare"]

    print("\n  === 判定 ===")
    print("    classic step4: %s" % (classic["step4"] or "(无)"))
    print("    fused   step4: %s" % (fused["step4"] or "(无)"))
    print("    classic: %sx%s %s frames=%s"
          % (info_c.get("width"), info_c.get("height"),
             info_c.get("pix_fmt"), info_c.get("nb_read_frames")))
    print("    fused  : %sx%s %s frames=%s"
          % (info_f.get("width"), info_f.get("height"),
             info_f.get("pix_fmt"), info_f.get("nb_read_frames")))
    print("    psnr   : %d 帧" % cmp_result.get("frames", 0))
    print("    verdict = %s" % verdict)
    if reasons:
        print("    未通过原因：")
        for r in reasons:
            print("      - %s" % r)
    elif cmp_result.get("summary"):
        print("    %s" % cmp_result["summary"][-1])

    report = {
        "repo": repo, "work": work_root, "input": clip,
        "frames_requested": args.frames,
        "classic": {k: v for k, v in classic.items() if k != "stdout_tail"},
        "fused": {k: v for k, v in fused.items() if k != "stdout_tail"},
        "info_classic": info_c, "info_fused": info_f,
        "comparison": {"frames": cmp_result.get("frames"),
                       "returncode": cmp_result.get("returncode"),
                       "summary": cmp_result.get("summary")},
        "verdict": verdict, "reasons": reasons,
        "argv": sys.argv,
    }
    out = os.path.join(work_root, "result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("    报告: %s" % out)

    return 0 if verdict == "IDENTICAL" else 1


if __name__ == "__main__":
    sys.exit(main())
