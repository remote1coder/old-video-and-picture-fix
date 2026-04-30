#!/usr/bin/env python3
"""
Manifest Loader
Loads manifests (YAML/JSONL), validates, filters by status, sorts by priority.
"""

import yaml  # type: ignore[import-untyped]
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_manifest(path: str) -> Dict[str, Any]:
    """Load manifest from YAML or JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    if path.endswith(".jsonl"):
        return _load_jsonl(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def _load_jsonl(path: str) -> Dict[str, Any]:
    """Load JSON Lines format manifest."""
    manifest: Dict[str, Any] = {"meta": {}, "jobs": []}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if lines:
            manifest["meta"] = json.loads(lines[0])
            for line in lines[1:]:
                line = line.strip()
                if line:
                    manifest["jobs"].append(json.loads(line))
    return manifest


def validate_job(job: Dict[str, Any]) -> bool:
    """Validate required job fields."""
    required = ["job_id", "input_path", "detected_class", "status"]
    for field in required:
        if field not in job or job[field] is None:
            return False
    return True


def filter_jobs(manifest: Dict[str, Any],
                statuses: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Filter jobs by status. Default: PENDING and FAILED.

    For FAILED jobs with existing intermediates, verify integrity.
    If valid, downgrade to PENDING; if invalid, delete intermediates and set PENDING.
    """
    if statuses is None:
        statuses = ["PENDING", "FAILED"]

    filtered = []
    for job in manifest.get("jobs", []):
        if job.get("status") in statuses:
            if job.get("status") == "FAILED":
                job = _handle_failed_job(job)
            filtered.append(job)

    # Sort by priority descending
    filtered.sort(key=lambda j: j.get("priority", 0), reverse=True)
    return filtered


def _handle_failed_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Handle FAILED job: check intermediates, verify integrity.

    - verify_frame_integrity on latest intermediate
    - keep recent 2 stages, delete older ones
    - downgrade to PENDING if latest is valid or no intermediates
    """
    try:
        import sys
        import re
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from shared_utils.checkpoint import verify_frame_integrity, verify_ffprobe_decodeable

        temp_dir = job.get("temp_dir", "./temp")
        job_id = job["job_id"]
        input_name = Path(job["input_path"]).stem

        import glob
        # Normalize path separator for cross-platform glob compatibility
        temp_dir_norm = temp_dir.replace("\\", "/")
        pattern = f"{temp_dir_norm}/{job_id}_{input_name}_step*"
        intermediates = glob.glob(pattern)

        if not intermediates:
            job["status"] = "PENDING"
            return job

        # Extract step number and sort
        def _step_key(path: str) -> int:
            m = re.search(r'_step(\d+)_', Path(path).name)
            return int(m.group(1)) if m else 0

        intermediates.sort(key=_step_key)

        # Verify latest intermediate with frame integrity check
        latest = intermediates[-1]
        if verify_ffprobe_decodeable(latest) and verify_frame_integrity(latest):
            job["status"] = "PENDING"
        else:
            Path(latest).unlink(missing_ok=True)
            job["status"] = "PENDING"

        # Keep recent 2 stages, delete older ones
        if len(intermediates) > 2:
            for old_path in intermediates[:-2]:
                Path(old_path).unlink(missing_ok=True)

    except Exception:
        # If we can't verify, just allow retry
        job["status"] = "PENDING"

    return job


def save_manifest(manifest: Dict[str, Any], path: str):
    """Save manifest back to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest["meta"], ensure_ascii=False) + "\n")
            for job in manifest["jobs"]:
                f.write(json.dumps(job, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(manifest, f, allow_unicode=True, sort_keys=False)


def update_job_status(manifest: Dict[str, Any], job_id: str,
                      status: str, error_message: str = ""):
    """Update job status in manifest."""
    for job in manifest.get("jobs", []):
        if job.get("job_id") == job_id:
            job["status"] = status
            if error_message:
                job["error_message"] = error_message
            from datetime import datetime
            job["updated_at"] = datetime.now().astimezone().isoformat()
            break

    # Update meta counts
    _update_meta_counts(manifest)


def _update_meta_counts(manifest: Dict[str, Any]):
    """Recalculate meta status counts."""
    meta = manifest.get("meta", {})
    jobs = manifest.get("jobs", [])
    meta["total_files"] = len(jobs)
    meta["pending"] = sum(1 for j in jobs if j.get("status") == "PENDING")
    meta["done"] = sum(1 for j in jobs if j.get("status") == "DONE")
    meta["failed"] = sum(1 for j in jobs if j.get("status") == "FAILED")
    meta["skipped"] = sum(1 for j in jobs if j.get("status") == "SKIP")
