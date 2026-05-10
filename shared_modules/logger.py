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
        self.line_count = 0
        self.lock = threading.Lock()

        # Determine log file
        if job_id:
            self.log_file = self.log_dir / f"job_{job_id}.log"
        else:
            self.log_file = self.log_dir / f"general_{int(time.time())}.log"

        # Initialize line count
        self._init_line_count()

    def _init_line_count(self):
        """Count existing lines in log file."""
        if self.log_file.exists():
            with open(self.log_file, "r", encoding="utf-8") as f:
                self.line_count = sum(1 for _ in f)

    def _rotate_if_needed(self):
        """Rotate log file if line count exceeds limit."""
        if self.line_count >= self.log_max_lines:
            # Rotate: job.log -> job.log.1 -> job.log.2 -> ...
            for i in range(self.log_backup_count, 0, -1):
                old = self.log_file.parent / f"{self.log_file.name}.{i-1}" if i > 1 else self.log_file
                new = self.log_file.parent / f"{self.log_file.name}.{i}"
                if old.exists():
                    old.rename(new)
            self.line_count = 0

    def log(self, level: str, event: str, message: str, **kwargs):
        """Write a structured log entry."""
        # Truncate large fields per v7.3 §3.10.12
        for key, value in kwargs.items():
            vstr = str(value)
            if len(vstr) > self.log_max_line_bytes:
                import hashlib
                h = hashlib.sha256(vstr.encode()).hexdigest()[:8]
                kwargs[key] = f"<truncated_field name={key} len={len(vstr)} sha={h}>"

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

        with self.lock:
            self._rotate_if_needed()
            with open(self.log_file, "a+b") as f:
                if _HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    f.write(line_bytes)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                elif _HAS_MSVCRT:
                    # Windows: lock the entire file
                    f.seek(0, 2)  # EOF
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, len(line_bytes))
                    f.write(line_bytes)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, len(line_bytes))
                else:
                    # Fallback: no locking (single-process only)
                    f.write(line_bytes)
            self.line_count += 1

    def info(self, event: str, message: str, **kwargs):
        self.log("INFO", event, message, **kwargs)

    def warn(self, event: str, message: str, **kwargs):
        self.log("WARN", event, message, **kwargs)

    def warning(self, event: str, message: str, **kwargs):
        self.log("WARN", event, message, **kwargs)

    def error(self, event: str, message: str, **kwargs):
        self.log("ERROR", event, message, **kwargs)

    def debug(self, event: str, message: str, **kwargs):
        self.log("DEBUG", event, message, **kwargs)

    def param_override(self, param_name: str, old_value: Any, new_value: Any,
                       source: str = "unknown"):
        """Log a parameter override event per v7.3 §3.10.12."""
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
