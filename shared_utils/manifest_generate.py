#!/usr/bin/env python3
"""
Manifest Generator
Scans input directories, classifies files, generates batch manifest.
Supports YAML (default) and JSON Lines formats.
"""

import sys
import json
import yaml  # type: ignore[import-untyped]
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


def classify_item(input_path: str, pipeline_type: str = "video") -> Dict[str, Any]:
    """Classify a single file."""
    if pipeline_type == "video":
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "video_restoration_pipeline"))
            from classify import classify_video
            return classify_video(input_path)
        except Exception as e:
            return {"detected_class": "UNKNOWN", "confidence": 0.0, "error": str(e)}
    else:
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "photo_restoration_pipeline"))
            from classify import classify_photo
            return classify_photo(input_path)
        except Exception as e:
            return {"detected_class": "UNKNOWN", "confidence": 0.0, "error": str(e)}


def map_class_to_path_params(detected_class: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Map detected class to path-specific parameters."""
    paths = config.get("paths", {})
    if detected_class == "ANALOG_TAPE":
        return dict(paths.get("analog_tape", {}))
    elif detected_class == "DV_DIGITAL":
        return dict(paths.get("dv_digital", {}))
    elif detected_class == "HDV_AVCHD":
        return dict(paths.get("hdv_avchd", {}))
    return {}


def generate_manifest(input_dirs: List[str], output_path: str,
                      config: Dict[str, Any],
                      pipeline_type: str = "video") -> str:
    """Generate manifest from input directories.

    Args:
        input_dirs: List of input directories to scan
        output_path: Output manifest path
        config: Loaded config
        pipeline_type: "video" or "photo"

    Returns:
        Path to generated manifest
    """
    manifest: Dict[str, Any] = {
        "meta": {
            "generated_at": now_iso(),
            "source_dirs": input_dirs,
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

                # AI upscale banned
                ai_banned = classification.get("ai_upscale_banned", False)

                # Determine output path
                out_name = f.stem + "_restored"
                if pipeline_type == "video":
                    out_name += ".mkv"
                else:
                    out_name += ".avif"
                job_out_path = str(Path(output_dir) / out_name)

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
                    "file_size": classification.get("file_size", 0),
                    "file_mtime": classification.get("file_mtime", 0),
                    "file_hash_prefix": classification.get("file_hash_prefix", ""),
                }

                manifest["jobs"].append(job)

    manifest["meta"]["total_files"] = job_id

    # Write manifest
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.endswith(".jsonl"):
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(manifest["meta"], ensure_ascii=False) + "\n")
            for job in manifest["jobs"]:
                fh.write(json.dumps(job, ensure_ascii=False) + "\n")
    else:
        with open(output_path, "w", encoding="utf-8") as fh:
            yaml.dump(manifest, fh, allow_unicode=True, sort_keys=False)

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
    args = parser.parse_args()

    config = load_config(args.config)
    path = generate_manifest(args.input, args.output, config, args.type)
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
