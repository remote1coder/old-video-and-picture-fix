#!/usr/bin/env python3
"""
Manifest Generator
Scans input directories, classifies files, generates batch manifest.
Supports YAML (default) and JSON Lines formats.

v6.3 Compliance:
- Includes dpi field in manifest jobs
- Maps photo categories (SCAN_PRINT, DIGITAL_PHOTO) to path-specific params
- Computes native_width/native_height/native_dar
"""

import sys
import json
import yaml  # type: ignore[import-untyped]
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load config.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def now_iso() -> str:
    """ISO-8601 timestamp with timezone."""
    return datetime.now().astimezone().isoformat()


# ISSUE-V-MGEN-001: module-level cache to avoid repeated importlib loading
_CLASSIFY_CACHE: Dict[str, Any] = {}


def classify_item(input_path: str, pipeline_type: str = "video",
                  config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Classify a single file using importlib to avoid sys.path conflicts.

    Caches the loaded module so large directories don't pay importlib overhead
    per file.

    FIX-B7：必须把 `config` 透传给分类器。`classify_video()` 内部有一处
    `(config or {}).get("classify", {}).get("hdv_ivtc")`，但它的签名原本
    **没有 config 参数** → 走到"NTSC + 隔行"分支就 `NameError`；而这里的
    `except Exception` 会把异常**吞成 UNKNOWN/0.0**，于是素材被判
    "低置信度 → SKIP → 不修"，且日志里只有一句 Low confidence，
    看不出是崩溃。实测真实素材必现。
    config 为 None 时仍按旧行为调用（向后兼容）。
    """
    cache_key = pipeline_type
    if cache_key in _CLASSIFY_CACHE:
        _fn = _CLASSIFY_CACHE[cache_key]
        try:
            return _fn(input_path, config) if config is not None else _fn(input_path)
        except TypeError:
            return _fn(input_path)

    if pipeline_type == "video":
        module_path = Path(__file__).parent.parent / "video_restoration_pipeline" / "classify.py"
        func_name = "classify_video"
    else:
        module_path = Path(__file__).parent.parent / "photo_restoration_pipeline" / "classify.py"
        func_name = "classify_photo"

    try:
        spec = importlib.util.spec_from_file_location(f"classify_module_{pipeline_type}", module_path)
        if spec is None or spec.loader is None:
            return {"detected_class": "UNKNOWN", "confidence": 0.0, "error": "Cannot load classify module"}
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        classify_func = getattr(module, func_name)
        _CLASSIFY_CACHE[cache_key] = classify_func
        try:
            return (classify_func(input_path, config)
                    if config is not None else classify_func(input_path))
        except TypeError:
            # 该分类器不接受 config（如 classify_photo）→ 回退单参调用
            return classify_func(input_path)
    except Exception as e:
        return {"detected_class": "UNKNOWN", "confidence": 0.0, "error": str(e)}


def map_class_to_path_params(detected_class: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Map detected class to path-specific parameters.
    
    For photo categories, merges config["photo"] settings into recommended_params.
    Also merges audio_override if present.
    """
    paths = config.get("paths", {})
    result: Dict[str, Any] = {}
    
    if detected_class == "ANALOG_TAPE":
        result = dict(paths.get("analog_tape", {}))
    elif detected_class == "DV_DIGITAL":
        result = dict(paths.get("dv_digital", {}))
    elif detected_class == "HDV_AVCHD":
        result = dict(paths.get("hdv_avchd", {}))
    elif detected_class == "SCAN_PRINT":
        result = dict(paths.get("scan_print", {}))
        # Merge photo-specific defaults if scan_print is empty
        if not result:
            result = dict(config.get("photo", {}))
    elif detected_class == "DIGITAL_PHOTO":
        result = dict(paths.get("digital_photo", {}))
        if not result:
            result = dict(config.get("photo", {}))
    
    # ISSUE-069 fix: filter out encoding-specific keys from photo params
    # to keep recommended_params focused on processing parameters
    encoding_keys = {"avif_params", "avif_rgb444_params"}
    result = {k: v for k, v in result.items() if k not in encoding_keys}
    
    # FIX-P4-1: Merge audio_override per detected_class, not entire dict
    audio_override = config.get("audio_override", {})
    if audio_override:
        type_key = None
        if detected_class == "ANALOG_TAPE":
            type_key = "analog_tape"
        elif detected_class == "DV_DIGITAL":
            type_key = "dv_digital"
        elif detected_class == "HDV_AVCHD":
            type_key = "hdv_avchd"
        if type_key and type_key in audio_override:
            result["audio_override"] = dict(audio_override[type_key])
        elif "default" in audio_override:
            result["audio_override"] = dict(audio_override["default"])

    # FIX-P4-14: Add path reference key per v7.32 §3.4.1
    class_to_path_key = {
        "ANALOG_TAPE": "analog_tape",
        "DV_DIGITAL": "dv_digital",
        "HDV_AVCHD": "hdv_avchd",
        "SCAN_PRINT": "scan_print",
        "DIGITAL_PHOTO": "digital_photo",
    }
    if detected_class in class_to_path_key:
        result["path"] = class_to_path_key[detected_class]

    return result


def _compute_photo_ai_banned(orig_w: int, orig_h: int) -> bool:
    """Compute ai_upscale_banned for photo categories per v7.32 §3.9.

    Uses long_edge/short_edge logic:
    long_edge = max(w,h), short_edge = min(w,h)
    Ban if long_edge > 3840 OR short_edge > 2160.
    """
    long_edge = max(orig_w, orig_h)
    short_edge = min(orig_w, orig_h)
    return long_edge > 3840 or short_edge > 2160


def _get_duration(input_path: str, classification: Dict[str, Any]) -> float:
    """Return duration in seconds from classify result or ffprobe fallback."""
    if "duration" in classification and classification["duration"]:
        try:
            return float(classification["duration"])
        except Exception:
            pass
    try:
        import subprocess
        r = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", input_path],
            capture_output=True, text=True, timeout=30, shell=False
        )
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return 0.0


def _truncate_jsonl_line(job: Dict[str, Any]) -> str:
    """Truncate oversized fields so a JSONL line stays <= 1MB.

    FIX-G16: per v7.32 §3.10.19, when a JSONL line exceeds 1MB, replace the
    largest offending top-level field (by serialized footprint — covers
    dict/list/string values like recommended_params or field_segments) with a
    ``<truncated_field name=... len=... sha=...>`` placeholder preserving the
    field name, original byte length, and a SHA-256 prefix-8 fingerprint, and
    append a ``_truncated_meta`` record. Repeats until under the limit or no
    large field remains.
    """
    import hashlib
    MAX = 1024 * 1024
    job = dict(job)  # shallow copy; only top-level values are swapped
    truncated_meta: List[Dict[str, Any]] = []

    def _serialize() -> str:
        return json.dumps(job, ensure_ascii=False)

    while len(_serialize().encode("utf-8")) > MAX:
        candidate = None  # (key, value, size, serialized)
        for k, v in job.items():
            if k == "_truncated_meta":
                continue
            try:
                serialized = json.dumps(v, ensure_ascii=False)
            except (TypeError, ValueError):
                continue
            size = len(serialized.encode("utf-8"))
            if candidate is None or size > candidate[2]:
                candidate = (k, v, size, serialized)
        # Stop if nothing left to truncate (or remaining fields are tiny)
        if candidate is None or candidate[2] < 256:
            break
        k, _v, size, serialized = candidate
        h = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]
        job[k] = f"<truncated_field name={k} len={size} sha={h}>"
        truncated_meta.append({"field": k, "orig_bytes": size, "sha256_prefix8": h})

    if truncated_meta:
        job["_truncated_meta"] = truncated_meta
    return _serialize()


def generate_manifest(input_dirs: List[str], output_path: str,
                      config: Dict[str, Any],
                      pipeline_type: str = "video",
                      incremental: bool = False,
                      config_path: str = "",
                      cli_output_dir: Optional[str] = None) -> str:
    """Generate manifest from input directories.

    Args:
        input_dirs: List of input directories to scan
        output_path: Output manifest path
        config: Loaded config
        pipeline_type: "video" or "photo"
        cli_output_dir: FIX-R5-07 — 扫描时 --output 指定的输出目录，
            优先于 config 的 global.output_dir。原来扫描完全忽略
            --output，manifest 里记的是 config 的相对 "./output"，
            执行时按进程 cwd 解析，换 cwd 就落错地方甚至直接崩。

    Returns:
        Path to generated manifest
    """
    # Deduplicate input_dirs
    seen_dirs = set()
    unique_dirs = []
    for d in input_dirs:
        resolved = str(Path(d).resolve())
        if resolved not in seen_dirs:
            seen_dirs.add(resolved)
            unique_dirs.append(d)
    input_dirs = unique_dirs

    # ISSUE-V-MGEN-003: incremental scan preserves existing user edits
    existing_jobs: Dict[str, Dict[str, Any]] = {}
    if incremental and Path(output_path).exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                old_manifest = yaml.safe_load(f)
            for old_job in old_manifest.get("jobs", []):
                key = old_job.get("input_path", "")
                if key:
                    existing_jobs[key] = old_job
        except Exception:
            pass

    manifest: Dict[str, Any] = {
        "meta": {
            "generated_at": now_iso(),
            "source_dirs": input_dirs,
            "config_path": config_path,  # FIX-P4-6: record config path for editor
            "total_files": 0,
            "pending": 0,
            "done": 0,
            "failed": 0,
            "skipped": 0,
        },
        "jobs": []
    }

    # Extensions to include
    if pipeline_type == "video":
        exts = {".avi", ".mkv", ".mp4", ".mov", ".mxf", ".dv", ".mpeg", ".mpg",
                ".ts", ".vob", ".wmv", ".flv"}
    else:
        exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".webp"}

    # FIX-R5-07: --output 优先于 config 的 global.output_dir；两者都缺省
    # 才用 "./output"。原来扫描忽略 --output，用户传了也不生效。
    output_dir = (cli_output_dir
                  or config.get("global", {}).get("output_dir", "./output"))
    log_dir = config.get("global", {}).get("log_dir", "./logs")
    # FIX-L9: record temp_dir on each job so manifest_loader can locate
    # intermediates when the user used --temp-dir or a non-default config
    # (otherwise loader falls back to ./temp and silently misses them).
    temp_dir = config.get("global", {}).get("temp_dir", "./temp")

    # FIX-R5-07: manifest 是**持久产物**，扫描后可能换 cwd/换机器再执行，
    # 所有落盘路径必须写绝对路径（input_path 上面已经 resolve 过，这三个
    # 原来却是相对的，正是本轮 stage7 exit 84 的根因：
    # cwd=仓库根时 "output\\xxx_restored.mkv" 落到仓库根的 output\ 下，
    # 目录不存在 -> ffmpeg "No such file or directory"）。
    # 基准 = 扫描进程的 cwd，与 input_path 的 resolve 基准一致。
    output_dir_abs = str(Path(output_dir).resolve())
    log_dir_abs = str(Path(log_dir).resolve())
    temp_dir_abs = str(Path(temp_dir).resolve())

    job_id = 0
    for inp_dir in input_dirs:
        inp_path = Path(inp_dir).resolve()
        if not inp_path.exists():
            continue

        for f in sorted(inp_path.rglob("*")):
            if f.is_file() and f.suffix.lower() in exts:
                job_id += 1
                input_str = str(f)

                # Classify
                # FIX-B7：透传 config —— classify_video 需要它读
                # classify.hdv_ivtc 等开关；不透传会在 NTSC+隔行分支
                # NameError，被吞成 UNKNOWN/0.0 → 素材被判不修。
                classification = classify_item(input_str, pipeline_type, config)

                detected_class = classification.get("detected_class", "UNKNOWN")
                confidence = classification.get("confidence", 0.0)

                # Determine status
                # FIX-P4-13: Record skip_reason per v7.32 §3.4.3
                skip_reason = ""
                if confidence < 0.80:
                    status = "SKIP"
                    skip_reason = (
                        f"Low confidence ({confidence:.2f} < 0.80), "
                        f"detected as {detected_class}"
                    )
                    manifest["meta"]["skipped"] += 1
                else:
                    status = "PENDING"
                    manifest["meta"]["pending"] += 1

                # Map to path params
                path_params = map_class_to_path_params(detected_class, config)

                # FIX-P4-3: generate.py is the sole authority for ai_upscale_banned
                # Do NOT preserve classify.py's initial value; recompute from pixel dims
                orig_w = classification.get("original_width", 0) or 0
                orig_h = classification.get("original_height", 0) or 0
                # ORIGINAL: ai_banned = classification.get("ai_upscale_banned", False)
                ai_banned = False

                # FIX-P4-10: Hard recalculation based on original pixel dimensions
                # Uses long_edge/short_edge per v7.32 §3.9
                if detected_class in ("SCAN_PRINT", "DIGITAL_PHOTO"):
                    if _compute_photo_ai_banned(orig_w, orig_h):
                        ai_banned = True
                elif detected_class == "HDV_AVCHD":
                    if orig_h >= 1080 or orig_w >= 1920:
                        ai_banned = True
                elif detected_class == "ANALOG_TAPE":
                    if orig_h > 1080 or orig_w > 1920:
                        ai_banned = True
                # DV_DIGITAL: never banned (SD material, well below thresholds)

                # Determine output path
                out_name = f.stem + "_restored"
                if pipeline_type == "video":
                    out_name += ".mkv"
                else:
                    out_name += ".avif"
                # FIX-R5-07: 绝对路径，见上面 output_dir_abs 注释
                job_out_path = str(Path(output_dir_abs) / out_name)

                # ISSUE-V-MGEN-002 / BUG-045: ensure duration is populated
                duration = _get_duration(input_str, classification)

                job = {
                    "job_id": f"job_{job_id:03d}",
                    "input_path": input_str,
                    "detected_class": detected_class,
                    "confidence": confidence,
                    "field_order": classification.get("field_order", "unknown"),
                    "needs_ivtc": classification.get("needs_ivtc", False),
                    "is_psf": classification.get("is_psf", False),
                    "field_segments": classification.get("field_segments", []),
                    "status": status,
                    "priority": 0,
                    "ai_upscale_banned": ai_banned,
                    "recommended_params": path_params,
                    "user_overrides": {},
                    "output_path": job_out_path,
                    "temp_dir": temp_dir_abs,  # FIX-L9+FIX-R5-07: per-job temp_dir (absolute) for loader
                    "log_path": str(Path(log_dir_abs) / f"job_{job_id:03d}.log"),
                    "error_message": "",
                    "skip_reason": skip_reason,  # FIX-P4-13: v7.32 §3.4.3
                    "created_at": now_iso(),
                    "updated_at": now_iso(),
                    "original_width": classification.get("original_width", 0),
                    "original_height": classification.get("original_height", 0),
                    "native_width": classification.get("native_width", 0),
                    "native_height": classification.get("native_height", 0),
                    "native_dar": classification.get("native_dar", ""),
                    "duration": duration,
                    "dpi": classification.get("dpi", None),
                    "file_size": classification.get("file_size", 0),
                    "file_mtime": classification.get("file_mtime", 0),
                    "file_hash_prefix": classification.get("file_hash_prefix", ""),
                }

                # ISSUE-V-MGEN-003: preserve user edits from existing manifest
                old_job = existing_jobs.get(input_str)
                if old_job:
                    for preserve_key in ("user_overrides", "status", "priority", "error_message"):
                        if preserve_key in old_job:
                            job[preserve_key] = old_job[preserve_key]
                    # If old status was DONE but file changed, reset to PENDING for safety
                    old_hash = old_job.get("file_hash_prefix", "")
                    new_hash = job.get("file_hash_prefix", "")
                    if old_hash and new_hash and old_hash != new_hash:
                        job["status"] = "PENDING"
                        job["error_message"] = "File changed since last scan, reset to PENDING"

                manifest["jobs"].append(job)

    manifest["meta"]["total_files"] = job_id

    # Write manifest
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # BUG-011 fix: atomic write to avoid corruption during concurrent generation
    out_path = Path(output_path)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        if output_path.endswith(".jsonl"):
            # FIX-P4-7: JSONL stores only job lines; meta goes to separate .meta.json
            # FIX-G13/L12: write .meta.json atomically (.tmp + replace) so a crash
            # mid-write cannot leave meta inconsistent with the jobs file.
            meta_path = out_path.with_suffix(".meta.json")
            meta_tmp = meta_path.parent / (meta_path.name + ".tmp")
            with open(meta_tmp, "w", encoding="utf-8") as fh:
                json.dump(manifest["meta"], fh, ensure_ascii=False, indent=2)
            meta_tmp.replace(meta_path)
            with open(tmp_path, "w", encoding="utf-8") as fh:
                for job in manifest["jobs"]:
                    line = json.dumps(job, ensure_ascii=False)
                    # FIX-G16: enforce 1MB JSONL line limit per v7.32 §3.10.19
                    if len(line.encode("utf-8")) > 1024 * 1024:
                        line = _truncate_jsonl_line(job)
                    fh.write(line + "\n")
        else:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                yaml.dump(manifest, fh, allow_unicode=True, sort_keys=False)
        # Atomic rename on same filesystem
        tmp_path.replace(out_path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to write manifest atomically: {e}") from e

    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate batch manifest")
    parser.add_argument("--input", "-i", nargs="+", required=True,
                        help="Input directories")
    parser.add_argument("--output", "-o", default="batch_manifest.yaml",
                        help="Output manifest path")
    parser.add_argument("--config", "-c", default="config.yaml",
                        help="Config file path")
    parser.add_argument("--type", choices=["video", "photo"],
                        default="video", help="Pipeline type")
    parser.add_argument("--incremental", action="store_true",
                        help="Preserve existing manifest user_overrides/status/priority")
    args = parser.parse_args()

    config = load_config(args.config)
    path = generate_manifest(args.input, args.output, config, args.type,
                             incremental=args.incremental,
                             config_path=args.config)
    print(f"Manifest generated: {path}")
    # Count jobs from the generated manifest
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
        total_jobs = len(manifest.get("jobs", []))
        print(f"Total files: {total_jobs}")
    except Exception:
        print("Total files: unknown")


if __name__ == "__main__":
    main()
