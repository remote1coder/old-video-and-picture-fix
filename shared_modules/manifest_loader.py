#!/usr/bin/env python3
"""
Manifest Loader
Loads manifests (YAML/JSONL), validates, filters by status, sorts by priority.
"""

import yaml  # type: ignore[import-untyped]
import json
import logging
from datetime import datetime
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
    """Load JSON Lines format manifest.
    
    FIX-P4-7: meta is stored in a separate .meta.json file.
    """
    manifest: Dict[str, Any] = {"meta": {}, "jobs": []}
    p = Path(path)
    meta_path = p.with_suffix(".meta.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            manifest["meta"] = json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
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


def _verify_photo_integrity(path: str, bit_depth: int = 8) -> bool:
    """Verify photo intermediate file integrity.
    
    FIX-P4-5: Use statistical features + three-region check instead of
    all-zero/all-max spot checks to avoid false positives on legitimate
    black-bordered scans.
    
    Strategy:
    1. Load image (PIL)
    2. Check three vertical regions (top/middle/bottom) for corruption
    3. Use statistical features (mean/std) instead of all-zero/all-max
    4. Allow legitimate black borders (detected via edge analysis)
    """
    try:
        from PIL import Image
        import numpy as np
        
        # FIX-P4-15: Use context manager for PIL Image
        # ORIGINAL: img = Image.open(path); arr = np.array(img)
        with Image.open(path) as img:
            arr = np.array(img)
        
        # Handle various modes
        if len(arr.shape) == 2:
            arr = arr[:, :, np.newaxis]  # Grayscale -> H,W,1
        h, w, c = arr.shape
        
        # Determine max value based on bit_depth
        if arr.dtype == np.uint16:
            max_val = 65535
            effective_depth = 16
        elif arr.dtype == np.uint8:
            max_val = 255
            effective_depth = 8
        else:
            max_val = 255
            effective_depth = 8
        
        # --- Step 1: Detect globally corrupt image ---
        global_mean = float(np.mean(arr))
        global_std = float(np.std(arr))
        
        # All-zero detection
        if global_mean < 1.0 and global_std < 1.0:
            return False  # Entire image almost black
        
        # All-max detection
        if global_mean > max_val - 1.0 and global_std < 1.0:
            return False  # Entire image almost white/max
        
        # --- Step 2: Three-region validation ---
        region_h = h // 3
        regions = {
            "top": arr[0:region_h, :, :],
            "middle": arr[region_h:2*region_h, :, :],
            "bottom": arr[2*region_h:h, :, :]
        }
        
        for region_name, region in regions.items():
            r_mean = float(np.mean(region))
            r_std = float(np.std(region))
            
            if r_std < 1.0:
                # Region is nearly solid color
                if r_mean < 1.0 or r_mean > max_val - 1.0:
                    # Allow legitimate black/white borders on edges
                    is_edge_region = (region_name in ("top", "bottom"))
                    if is_edge_region and global_std > 10.0:
                        continue
                    return False
        
        # --- Step 3: Luminance mean range check (v6.32 §7.9) ---
        if effective_depth == 8:
            low, high = 10, 245
        elif effective_depth == 10:
            low, high = 40, 980
        else:  # 16bit — full range for photos per v6.32 §7.9
            # ORIGINAL: low, high = 4096, 60160
            low, high = 0, max_val
        
        if not (low <= global_mean <= high):
            if global_mean < 5 or global_mean > max_val - 5:
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
        # FIX-P4-1: Validate required fields per v7.32 §3.4.5
        if not validate_job(job):
            logging.warning(
                "Invalid job %s: missing required fields "
                "(input_path/detected_class/status), skipping",
                job.get("job_id", "UNKNOWN")
            )
            continue
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
        # BUG-060 fix: Use parent directory (shared_modules/) instead of project root
        # to ensure checkpoint.py is found regardless of project structure.
        sys.path.insert(0, str(Path(__file__).parent))
        from checkpoint import verify_ffprobe_decodeable, verify_frame_integrity

        temp_dir = job.get("temp_dir", "./temp")
        job_id = job["job_id"]
        input_name = Path(job["input_path"]).stem

        # FIX-P4-6: Use pathlib.Path.glob() instead of glob.glob()
        # ORIGINAL: import glob; temp_dir_norm = temp_dir.replace("\\", "/");
        #           pattern = f"{temp_dir_norm}/{job_id}_{input_name}_step*";
        #           intermediates = glob.glob(pattern)
        temp_dir_path = Path(temp_dir)
        pattern = f"{job_id}_{input_name}_step*"
        intermediates = [str(p) for p in sorted(temp_dir_path.glob(pattern))]

        if not intermediates:
            job["status"] = "PENDING"
            return job

        # Extract step number and sort
        def _step_key(path: str) -> int:
            m = re.search(r'_step(\d+)_', Path(path).name)
            return int(m.group(1)) if m else 0

        intermediates.sort(key=_step_key)

        # FIX-P4-5: Verify from latest backwards, keep up to 2 most recent valid stages
        valid_stage_path = None
        for cand in reversed(intermediates):
            is_photo = Path(cand).suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".avif")
            if is_photo:
                valid = _verify_photo_integrity(cand)
            else:
                valid = verify_ffprobe_decodeable(cand) and verify_frame_integrity(cand)
            if valid:
                valid_stage_path = cand
                break
            else:
                Path(cand).unlink(missing_ok=True)
        
        job["status"] = "PENDING"
        
        # Keep only up to 2 most recent valid stages
        if valid_stage_path:
            keep_paths = {valid_stage_path}
            valid_idx = intermediates.index(valid_stage_path)
            if valid_idx > 0:
                keep_paths.add(intermediates[valid_idx - 1])
            for p in intermediates:
                if p not in keep_paths:
                    Path(p).unlink(missing_ok=True)
        else:
            # None valid, clean up all
            for p in intermediates:
                Path(p).unlink(missing_ok=True)

    except Exception:
        # If we can't verify, just allow retry
        job["status"] = "PENDING"

    return job


def save_manifest(manifest: Dict[str, Any], path: str):
    """Save manifest back to disk atomically. FIX-P4-16: atomic write."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(p.suffix + ".tmp")
    try:
        if path.endswith(".jsonl"):
            # FIX-P4-7: write meta to separate .meta.json file
            meta_path = p.with_suffix(".meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(manifest["meta"], f, ensure_ascii=False, indent=2)
            with open(tmp_path, "w", encoding="utf-8") as f:
                for job in manifest["jobs"]:
                    f.write(json.dumps(job, ensure_ascii=False) + "\n")
        else:
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.dump(manifest, f, allow_unicode=True, sort_keys=False)
        # Atomic rename on same filesystem
        tmp_path.replace(p)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to save manifest atomically: {e}") from e


def update_job_status(manifest: Dict[str, Any], job_id: str,
                      status: str, error_message: str = ""):
    """Update job status in manifest."""
    for job in manifest.get("jobs", []):
        if job.get("job_id") == job_id:
            job["status"] = status
            if error_message:
                job["error_message"] = error_message
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
