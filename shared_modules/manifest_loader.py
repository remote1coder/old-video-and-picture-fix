#!/usr/bin/env python3
"""
Manifest Loader
Loads manifests (YAML/JSONL), validates, filters by status, sorts by priority.
"""

import yaml  # type: ignore[import-untyped]
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:  # POSIX
    import fcntl  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:  # Windows
    import msvcrt  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]


# FIX-V08: manifest 跨进程并发保护 ------------------------------------------
# 背景：每个 worker 各自 load_manifest 后整份 save_manifest（固定 .tmp + replace）。
# 原子 rename 只能避免半个文件，不能避免丢更新：B 用自己的旧快照覆盖掉 A 的 DONE。
# 方案：跨进程文件锁 + 锁内重新读取当前版本 + 合并单任务更新 + 唯一临时文件名。

class ManifestLockError(RuntimeError):
    """Raised when the cross-process manifest lock cannot be acquired."""


@contextmanager
def _file_lock(lock_path: Path, timeout: float = 30.0, poll: float = 0.05):
    """Cross-process exclusive advisory lock.

    POSIX: fcntl.flock(LOCK_EX|LOCK_NB). Windows: msvcrt.locking(LK_NBLCK).
    Raises ManifestLockError on timeout instead of silently proceeding.
    """
    if fcntl is None and msvcrt is None:
        # 无可用锁实现：退化为"不加锁"但必须显式记录，不能假装已保护
        logging.warning(
            "MANIFEST_LOCK_UNAVAILABLE: no fcntl/msvcrt; manifest updates "
            "are NOT protected against concurrent writers")
        yield None
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # NEW-V-07: 打开锁文件本身也可能失败。
    # 实测（Windows，4 进程并发写同一 manifest）：另一个进程持有 msvcrt 锁区间时，
    # `open(lock_path, "a+b")` 会抛 `PermissionError: [Errno 13]` ——
    # 而它不在原来的重试循环里，worker 因此直接崩溃（exitcode=1），
    # 既没等到锁，也没报"锁超时"。这里把 open 纳入同一套重试/超时语义：
    # 等不到就抛 ManifestLockError（显式失败），绝不静默继续。
    open_deadline = time.monotonic() + max(0.0, timeout)
    fh = None
    while True:
        try:
            fh = open(lock_path, "a+b")
            break
        except OSError as exc:
            if time.monotonic() >= open_deadline:
                raise ManifestLockError(
                    f"Could not open manifest lock file within {timeout}s "
                    f"({type(exc).__name__}: {exc}): {lock_path}") from exc
            time.sleep(poll)

    acquired = False
    deadline = time.monotonic() + max(0.0, timeout)
    try:
        while True:
            try:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
                break
            except Exception:
                if time.monotonic() >= deadline:
                    break
                time.sleep(poll)
        if not acquired:
            raise ManifestLockError(
                f"Could not acquire manifest lock within {timeout}s: {lock_path}")
        yield fh
    finally:
        if acquired:
            try:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                else:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
        try:
            fh.close()
        except Exception:
            pass


def _unique_tmp_path(p: Path, suffix: str = ".tmp") -> Path:
    """FIX-V08: unique temp name so concurrent writers never share one file."""
    ident = f"{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    return p.parent / f"{p.name}.{ident}{suffix}"


def _write_manifest_locked(manifest: Dict[str, Any], path: str) -> None:
    """Write a full manifest using a unique temp file (must hold the lock)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _unique_tmp_path(p)
    is_jsonl = path.endswith(".jsonl")
    meta_path = None
    meta_tmp = None
    try:
        if is_jsonl:
            meta_path = p.with_suffix(".meta.json")
            meta_tmp = _unique_tmp_path(meta_path)
            with open(meta_tmp, "w", encoding="utf-8") as f:
                json.dump(manifest["meta"], f, ensure_ascii=False, indent=2)
            with open(tmp_path, "w", encoding="utf-8") as f:
                for job in manifest["jobs"]:
                    f.write(json.dumps(job, ensure_ascii=False) + "\n")
        else:
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.dump(manifest, f, allow_unicode=True, sort_keys=False)
        tmp_path.replace(p)
        if is_jsonl and meta_tmp is not None and meta_path is not None:
            meta_tmp.replace(meta_path)
    except Exception as e:
        for cand in (tmp_path, meta_tmp):
            if cand is not None:
                try:
                    Path(cand).unlink(missing_ok=True)
                except Exception:
                    pass
        raise RuntimeError(f"Failed to save manifest atomically: {e}") from e


def update_manifest(path: str,
                    mutator: Callable[[Dict[str, Any]], Any],
                    timeout: float = 30.0) -> bool:
    """FIX-V08: locked read-merge-write of a manifest.

    The mutator receives the *current on-disk* manifest (re-read inside the
    lock) and mutates it in place. The result is written only if the mutator
    did not raise. Returns True when written.

    This is the primitive that fixes the lost-update bug: worker A's DONE is
    visible to worker B because B re-reads after acquiring the lock.
    """
    p = Path(path)
    lock_path = p.parent / (p.name + ".lock")
    with _file_lock(lock_path, timeout=timeout):
        try:
            manifest = load_manifest(path)
        except Exception as e:
            # 读取失败不能清空所有任务：宁可不写
            logging.error("MANIFEST_LOCKED_UPDATE_SKIP: cannot re-read %s: %s",
                          path, e)
            return False
        if not isinstance(manifest, dict):
            logging.error("MANIFEST_LOCKED_UPDATE_SKIP: %s is not a mapping",
                          path)
            return False
        manifest.setdefault("jobs", [])
        manifest.setdefault("meta", {})
        result = mutator(manifest)
        if result is False:
            return False
        _update_meta_counts(manifest)
        _write_manifest_locked(manifest, path)
        return True


def _should_apply_job_result(job: Dict[str, Any], status: str,
                             attempt: Optional[int],
                             attempt_group: Optional[str]) -> Tuple[bool, str]:
    """State-transition guard for a single-job result (late/stale attempts).

    Rules:
    - No attempt info (or no recorded attempt) → apply.
    - A different attempt group (new pipeline run) → apply (newer run wins).
    - Same group, older attempt → reject (stale late writer).
    - Same group, same-or-newer attempt → apply.
    """
    recorded_attempt = job.get("result_attempt")
    recorded_group = job.get("result_attempt_group")
    if attempt is None or recorded_attempt is None:
        return True, "no-attempt-info"
    if recorded_group != attempt_group:
        return True, "new-attempt-group"
    try:
        if int(attempt) < int(recorded_attempt):
            return False, "stale-attempt"
    except (TypeError, ValueError):
        return True, "unparsable-attempt"
    return True, "same-or-newer-attempt"


def save_manifest_job_result(path: str, job_id: str, status: str,
                             error_message: str = "",
                             attempt: Optional[int] = None,
                             attempt_group: Optional[str] = None,
                             timeout: float = 30.0) -> bool:
    """FIX-V08: report a single job's result under the cross-process lock.

    Replaces "worker saves its whole stale snapshot". Only the targeted job is
    modified, everything else is re-read from disk inside the lock.
    """
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status: {status!r} (must be one of {sorted(VALID_STATUSES)})"
        )
    applied = {"ok": False, "reason": "not-found"}

    def _mutate(manifest: Dict[str, Any]) -> bool:
        for job in manifest.get("jobs", []):
            if job.get("job_id") != job_id:
                continue
            ok, reason = _should_apply_job_result(job, status, attempt,
                                                  attempt_group)
            if not ok:
                applied["reason"] = reason
                logging.warning(
                    "MANIFEST_RESULT_REJECTED: job %s status %s rejected (%s)",
                    job_id, status, reason)
                return False
            job["status"] = status
            if error_message:
                job["error_message"] = error_message
            job["updated_at"] = datetime.now().astimezone().isoformat()
            if attempt is not None:
                job["result_attempt"] = int(attempt)
            if attempt_group is not None:
                job["result_attempt_group"] = attempt_group
            applied.update({"ok": True, "reason": reason})
            return True
        logging.warning("MANIFEST_RESULT_MISSING_JOB: %s not in %s", job_id, path)
        return False

    written = update_manifest(path, _mutate, timeout=timeout)
    return bool(written and applied["ok"])


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
        # FIX-L2: reject empty strings too, not just None/missing. ORIGINAL:
        # `job[field] is None` let input_path="" / detected_class="" pass.
        if not job.get(field):
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

        # FIX-L9: temp_dir now comes from the per-job field written by
        # manifest_generate.py (config global.temp_dir). Falls back to ./temp
        # only for legacy manifests, so non-default --temp-dir is honored.
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
                # FIX-L10: video intermediates are verified via checkpoint.py
                # (verify_ffprobe_decodeable + verify_frame_integrity). Photo
                # intermediates use this local stricter implementation
                # (three-region + statistical features). Whether photo MUST also
                # route through checkpoint.py is 待规范澄清 (§3.4.5); the local
                # impl is equivalent-or-stricter. Inline note stands in for a
                # shixian.md deviation entry (out of this file's edit scope).
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

    except Exception as e:
        # FIX-L11: if verification itself crashed, do NOT silently leave
        # potentially-corrupt intermediates that a later stage could reuse via
        # param_hash. Log a WARNING with context and flag the job for manual
        # inspection (status still PENDING so the user can retry/inspect).
        logging.warning(
            "Intermediate verification crashed for job %s: %s — status set to "
            "PENDING; attempting best-effort cleanup of collected intermediates",
            job.get("job_id", "UNKNOWN"), e
        )
        job["status"] = "PENDING"
        job["error_message"] = (
            f"Intermediate file verification raised an exception ({e}); "
            "best-effort cleanup attempted — please inspect temp_dir manually"
        )
        # FIX-L11: best-effort cleanup of any intermediates already collected
        # before the crash. A corrupt intermediate that survives here can be
        # reused via param_hash by a later stage, producing silent corruption.
        # If `intermediates` was bound before the crash, unlink each (missing_ok
        # so files already gone don't re-raise). On any cleanup failure, leave
        # the file in place and let the manual-inspection error_message stand.
        try:
            for p in intermediates:  # type: ignore[name-defined]
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
        except (NameError, UnboundLocalError):
            # `intermediates` never got bound (crash happened before collection)
            # - nothing to clean, manual inspection message already set.
            pass

    return job


def save_manifest(manifest: Dict[str, Any], path: str,
                  timeout: float = 30.0):
    """Save manifest back to disk atomically. FIX-P4-16: atomic write.

    FIX-V08: the write now happens under the cross-process lock and uses a
    unique temp file (previously a fixed `.tmp` shared by all writers).

    NOTE: this still writes the *caller's* snapshot. Multi-worker call sites
    must use :func:`save_manifest_job_result` / :func:`update_manifest`, which
    re-read the on-disk manifest inside the lock and only merge their own job.
    """
    p = Path(path)
    lock_path = p.parent / (p.name + ".lock")
    with _file_lock(lock_path, timeout=timeout):
        _write_manifest_locked(manifest, path)


# FIX-L13: legal job status values per v7.32 §3.4.3
VALID_STATUSES = {"PENDING", "DONE", "FAILED", "SKIP"}


def update_job_status(manifest: Dict[str, Any], job_id: str,
                      status: str, error_message: str = ""):
    """Update job status in manifest."""
    # FIX-L13: validate status so arbitrary strings from other callers cannot
    # corrupt the manifest. The argparse CLI already restricts to 4 values,
    # but this function is a public API.
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status: {status!r} (must be one of {sorted(VALID_STATUSES)})"
        )
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
