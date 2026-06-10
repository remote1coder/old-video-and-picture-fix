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
from typing import Dict, Any, List


def load_config(config_path: str) -> Dict[str, Any]:
    """Load config.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def now_iso() -> str:
    """ISO-8601 timestamp with timezone."""
    return datetime.now().astimezone().isoformat()


# ISSUE-V-MGEN-001: module-level cache to avoid repeated importlib loading
_CLASSIFY_CACHE: Dict[str, Any] = {}


def classify_item(input_path: str, pipeline_type: str = "video") -> Dict[str, Any]:
    """Classify a single file using importlib to avoid sys.path conflicts.

    Caches the loaded module so large directories don't pay importlib overhead
    per file.
    """
    cache_key = pipeline_type
    if cache_key in _CLASSIFY_CACHE:
        return _CLASSIFY_CACHE[cache_key](input_path)

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
    
    return result


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


def generate_manifest(input_dirs: List[str], output_path: str,
                      config: Dict[str, Any],
                      pipeline_type: str = "video",
                      incremental: bool = False,
                      config_path: str = "") -> str:
    """Generate manifest from input directories.

    Args:
        input_dirs: List of input directories to scan
        output_path: Output manifest path
        config: Loaded config
        pipeline_type: "video" or "photo"

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

    output_dir = config.get("global", {}).get("output_dir", "./output")
    log_dir = config.get("global", {}).get("log_dir", "./logs")

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
                classification = classify_item(input_str, pipeline_type)

                detected_class = classification.get("detected_class", "UNKNOWN")
                confidence = classification.get("confidence", 0.0)

                # Determine status
                if confidence < 0.80:
                    status = "SKIP"
                    manifest["meta"]["skipped"] += 1
                else:
                    status = "PENDING"
                    manifest["meta"]["pending"] += 1

                # Map to path params
                path_params = map_class_to_path_params(detected_class, config)

                # FIX-P4-4: AI upscale banned: compute directly in generate.py per spec
                orig_w = classification.get("original_width", 0) or 0
                orig_h = classification.get("original_height", 0) or 0
                ai_banned = classification.get("ai_upscale_banned", False)
                
                # Hard recalculation based on original pixel dimensions
                if detected_class in ("SCAN_PRINT", "DIGITAL_PHOTO"):
                    if orig_w > 3840 or orig_h > 2160:
                        ai_banned = True
                elif detected_class == "HDV_AVCHD":
                    if orig_h >= 1080 or orig_w >= 1920:
                        ai_banned = True
                elif detected_class == "ANALOG_TAPE":
                    if orig_h > 1080 or orig_w > 1920:
                        ai_banned = True

                # Determine output path
                out_name = f.stem + "_restored"
                if pipeline_type == "video":
                    out_name += ".mkv"
                else:
                    out_name += ".avif"
                job_out_path = str(Path(output_dir) / out_name)

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
                    "log_path": str(Path(log_dir) / f"job_{job_id:03d}.log"),
                    "error_message": "",
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
            meta_path = str(out_path.with_suffix(".meta.json"))
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(manifest["meta"], fh, ensure_ascii=False, indent=2)
            with open(tmp_path, "w", encoding="utf-8") as fh:
                for job in manifest["jobs"]:
                    fh.write(json.dumps(job, ensure_ascii=False) + "\n")
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
