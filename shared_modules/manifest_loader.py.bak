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


def _verify_photo_integrity(path: str) -> bool:
    """Verify photo file integrity using PIL.
    
    Per v6.32: Only mark as corrupted if PIL verify/load fails.
    Solid colors and all-black frames are legitimate content.
    
    ISSUE-SHARED-ML-001: Added multi-position spot checks (top-left, center,
    bottom-right) to detect partial writes or corruption that PIL load might miss.
    """
    try:
        from PIL import Image
        import numpy as np
        with Image.open(path) as img:
            # Ensure image can be loaded
            img.load()
            # Check non-trivial size
            if img.width <= 0 or img.height <= 0:
                return False
            # For 16-bit images, preserve precision
            if img.mode == "I;16":
                arr = np.array(img, dtype=np.uint16)
            else:
                arr = np.array(img)
            # Only reject if array is empty (not based on content)
            if arr.size == 0:
                return False
            
            # ISSUE-SHARED-ML-001 fix: Multi-position spot checks
            # Extract top-left, center, bottom-right regions and verify
            # they are not all-zero (indicates partial write / corruption)
            h, w = arr.shape[:2]
            region_h = max(1, h // 10)
            region_w = max(1, w // 10)
            
            # Top-left region
            top_left = arr[0:region_h, 0:region_w]
            # Center region
            cy, cx = h // 2, w // 2
            center = arr[cy:cy+region_h, cx:cx+region_w]
            # Bottom-right region
            br = arr[h-region_h:h, w-region_w:w]
            
            for region_name, region in [("top-left", top_left),
                                         ("center", center),
                                         ("bottom-right", br)]:
                if region.size == 0:
                    continue
                # Check if region is all zeros (indicates corruption/partial write)
                if np.all(region == 0):
                    return False
                # Check if region is all max value (also suspicious for corruption)
                max_val = 65535 if arr.dtype == np.uint16 else 255
                if np.all(region == max_val):
                    return False
        return True
    except Exception:
        return False


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

    - For video intermediates: verify_frame_integrity
    - For photo intermediates: PIL-based integrity check
    - keep recent 2 stages, delete older ones
    - downgrade to PENDING if latest is valid or no intermediates
    """
    try:
        import sys
        import re
        import glob
        # BUG-060 fix: Use parent directory (shared_modules/) instead of project root
        # to ensure checkpoint.py is found regardless of project structure.
        sys.path.insert(0, str(Path(__file__).parent))
        from checkpoint import verify_ffprobe_decodeable, verify_frame_integrity

        temp_dir = job.get("temp_dir", "./temp")
        job_id = job["job_id"]
        input_name = Path(job["input_path"]).stem

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

        # Verify latest intermediate
        latest = intermediates[-1]
        # Photo intermediates are TIFF/PNG, video are MKV
        is_photo = Path(latest).suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".avif")

        valid = False
        if is_photo:
            valid = _verify_photo_integrity(latest)
        else:
            valid = verify_ffprobe_decodeable(latest) and verify_frame_integrity(latest)

        if valid:
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
