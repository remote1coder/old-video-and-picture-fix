#!/usr/bin/env python3
"""
Structured JSON Lines Logger
Creates one log file per job with structured JSON entries.

v7.3 §3.3, §4.3 Compliance:
- JSON Lines format, UTF-8
- ISO-8601 timestamps with timezone
- Per-job log files
- Multi-process safe (file locking + buffering)
- Log rotation by line count
"""

import time
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Cross-platform file locking
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False
    try:
        import msvcrt
        _HAS_MSVCRT = True
    except ImportError:
        _HAS_MSVCRT = False


class StructuredLogger:
    """Structured JSON Lines logger with per-job log files."""

    def __init__(self,
                 log_dir: str = "./logs",
                 job_id: Optional[str] = None,
                 log_max_lines: int = 10000,
                 log_backup_count: int = 5,
                 log_max_line_bytes: int = 1048576):
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.job_id = job_id
        self.log_max_lines = log_max_lines
        self.log_backup_count = log_backup_count
        self.log_max_line_bytes = log_max_line_bytes
        self.lock = threading.Lock()

        # Determine log file
        if job_id:
            self.log_file = self.log_dir / f"job_{job_id}.log"
        else:
            self.log_file = self.log_dir / f"general_{int(time.time())}.log"

    def _get_current_line_count(self) -> int:
        """Count current lines in log file (called before each write)."""
        if not self.log_file.exists():
            return 0
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def _rotate_if_needed(self):
        """Rotate log file if line count exceeds limit.
        
        Uses timestamp-based backup names to avoid rename race conditions
        in multi-process environments.
        """
        line_count = self._get_current_line_count()
        if line_count >= self.log_max_lines:
            # Use timestamp-based unique name to avoid rename race
            ts = int(time.time() * 1000)
            backup = self.log_file.parent / f"{self.log_file.name}.{ts}"
            try:
                self.log_file.rename(backup)
                # Clean up old backups if exceeding log_backup_count
                backups = sorted(
                    self.log_file.parent.glob(f"{self.log_file.name}.*"),
                    key=lambda p: p.stat().st_mtime
                )
                while len(backups) > self.log_backup_count:
                    try:
                        backups.pop(0).unlink()
                    except OSError:
                        pass
            except OSError:
                pass

    def _rotate_locked(self, f):
        """Rotate log file when holding a cross-process lock.
        
        FIX-P1-4: Uses file size estimate instead of line count to avoid
        re-reading the entire file under lock. Per v7.32 §3.10.12 / v6.32 §7.4.
        """
        f.seek(0, 2)  # EOF
        size = f.tell()
        # Rough estimate: assume average 200 bytes per line
        estimated_lines = size // 200
        if estimated_lines >= self.log_max_lines:
            ts = int(time.time() * 1000)
            backup = self.log_file.parent / f"{self.log_file.name}.{ts}"
            try:
                f.flush()
                import os
                os.rename(str(self.log_file), str(backup))
                open(self.log_file, 'a').close()
                # Clean up old backups if exceeding log_backup_count
                backups = sorted(
                    self.log_file.parent.glob(f"{self.log_file.name}.*"),
                    key=lambda p: p.stat().st_mtime
                )
                while len(backups) > self.log_backup_count:
                    try:
                        backups.pop(0).unlink()
                    except OSError:
                        pass
            except OSError:
                pass  # Another process already rotated

    def log(self, level: str, event: str, message: str, **kwargs):
        """Write a structured log entry."""
        # FIX-P1-3: Truncate large fields by byte length per v7.32 §3.10.12
        for key, value in kwargs.items():
            vbytes = str(value).encode("utf-8")
            if len(vbytes) > self.log_max_line_bytes:
                import hashlib
                h = hashlib.sha256(vbytes).hexdigest()[:8]
                kwargs[key] = f"<truncated_field name={key} len={len(vbytes)} sha={h}>"

        # Use local timezone with offset per v7.3 §3.10.12
        entry = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "level": level,
            "event": event,
            "message": message,
            "job_id": self.job_id,
            "thread": threading.current_thread().name,
        }
        entry.update(kwargs)

        line = json.dumps(entry, ensure_ascii=False) + "\n"
        line_bytes = line.encode("utf-8")

        # FIX-P1-3: Final safety net — if entire line exceeds 1MB, replace with safe stub
        if len(line_bytes) > self.log_max_line_bytes:
            safe = {
                "timestamp": entry["timestamp"],
                "level": level,
                "event": "LOG_TRUNCATED",
                "message": f"Original line {len(line_bytes)} bytes exceeds 1MB",
                "original_event": event,
                "truncated_fields": list(kwargs.keys()),
            }
            line_bytes = (json.dumps(safe, ensure_ascii=False) + "\n").encode("utf-8")

        with self.lock:
            self._rotate_if_needed()
            with open(self.log_file, "a+b") as f:
                if _HAS_FCNTL:
                    # FIX-P1-5: flock → lockf per v6.32 §7.4
                    # FIX-P1-4: Rotate under cross-process lock per v7.32 §3.10.12
                    fcntl.lockf(f.fileno(), fcntl.LOCK_EX)
                    self._rotate_locked(f)
                    f.write(line_bytes)
                    f.flush()
                    fcntl.lockf(f.fileno(), fcntl.LOCK_UN)
                elif _HAS_MSVCRT:
                    # Windows: use LK_LOCK (blocking) instead of LK_NBLCK
                    f.seek(0, 2)  # EOF
                    lock_start = f.tell()
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, len(line_bytes))
                    f.write(line_bytes)
                    f.flush()
                    # msvcrt.locking requires file pointer at lock start to unlock
                    f.seek(lock_start)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, len(line_bytes))
                else:
                    # Fallback: no locking (single-process only)
                    f.write(line_bytes)
                    f.flush()

    def info(self, event: str, message: str, **kwargs):
        self.log("INFO", event, message, **kwargs)

    def warn(self, event: str, message: str, **kwargs):
        self.log("WARN", event, message, **kwargs)

    def warning(self, event: str, message: str, **kwargs):
        self.log("WARN", event, message, **kwargs)

    def error(self, event: str, message: str, **kwargs):
        self.log("ERROR", event, message, **kwargs)

    def critical(self, event: str, message: str, **kwargs):
        """FIX-P1-2: Added critical() per v7.32 §3.10.2/§3.10.5/§3.10.9."""
        self.log("CRITICAL", event, message, **kwargs)

    def debug(self, event: str, message: str, **kwargs):
        self.log("DEBUG", event, message, **kwargs)

    def param_override(self, param_name: str, old_value: Any, new_value: Any,
                       source: str = "unknown"):
        """Log a parameter override event per v7.3 §3.10.12.
        
        Source must be one of: user_overrides, paths, global, cli, unknown.
        """
        # BUG-016 fix: restrict source to allowed enumeration values
        allowed_sources = ("user_overrides", "paths", "global", "cli", "unknown")
        if source not in allowed_sources:
            self.log("WARN", "PARAM_OVERRIDE_INVALID_SOURCE",
                     f"Invalid source '{source}' for param_override, defaulting to 'unknown'",
                     invalid_source=source, allowed=list(allowed_sources))
            source = "unknown"
        self.log("INFO", "PARAM_OVERRIDE",
                 f"Parameter '{param_name}' overridden from '{old_value}' to '{new_value}'",
                 param_name=param_name, old_value=old_value,
                 new_value=new_value, source=source)


# ---------------------------------------------------------------------------
# get_logger singleton helper for stages
# ---------------------------------------------------------------------------

# Fix: job_id-based map to avoid creating multiple loggers for same job
_loggers: Dict[str, StructuredLogger] = {}
_loggers_lock = threading.Lock()


def get_logger(log_dir: str = "./logs",
               job_id: Optional[str] = None,
               **kwargs) -> StructuredLogger:
    """Singleton factory: returns existing logger for job_id, or creates new one."""
    if job_id:
        key = f"{log_dir}/{job_id}"
        with _loggers_lock:
            if key not in _loggers:
                _loggers[key] = StructuredLogger(log_dir=log_dir, job_id=job_id, **kwargs)
            return _loggers[key]
    return StructuredLogger(log_dir=log_dir, job_id=job_id, **kwargs)


# NOTE: CORE_QUALITY_PARAMS is defined in checkpoint.py (single source of truth).
# Do not duplicate here to avoid divergence.
