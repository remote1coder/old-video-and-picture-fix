#!/usr/bin/env python3
"""
Structured JSON Lines Logger
Implements per-job log separation, multi-process safety, and log rotation.
"""

import json
import datetime
import threading
from pathlib import Path
from typing import Dict, Any, Optional


# Cross-platform file locking helpers
def _acquire_file_lock(f):
    """Acquire exclusive file lock. Returns True if lock acquired."""
    # Linux / macOS
    try:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        return True
    except (ImportError, OSError):
        pass

    # portalocker cross-platform (if installed)
    try:
        import portalocker
        portalocker.lock(f, portalocker.LOCK_EX)
        return True
    except (ImportError, OSError):
        pass

    # Windows built-in msvcrt
    try:
        import msvcrt
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        return True
    except (ImportError, OSError):
        pass

    return False


def _release_file_lock(f):
    """Release file lock."""
    try:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except (ImportError, OSError):
        pass

    try:
        import portalocker
        portalocker.unlock(f)
    except (ImportError, OSError):
        pass

    try:
        import msvcrt
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    except (ImportError, OSError):
        pass


class StructuredLogger:
    """Structured JSON Lines logger with per-job separation."""

    # Excluded keys from param_hash (non-quality parameters)
    EXCLUDED_FROM_HASH = {
        "log_level", "output_path", "temp_path", "parallel_workers",
        "log_dir", "no_wsl_redirect", "preview", "log_max_lines",
        "log_backup_count", "log_max_line_bytes"
    }

    def __init__(self, log_dir: str = "./logs", level: str = "INFO",
                 max_line_bytes: int = 1048576, job_id: Optional[str] = None):
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.level = level.upper()
        self.max_line_bytes = max_line_bytes
        self.job_id = job_id

        self.levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        self._local = threading.local()

        # Global master log (job-level events only)
        self.master_log_path = self.log_dir / "master.log"

    def _get_log_path(self) -> Path:
        """Get log path: per-job if job_id set, else master log."""
        if self.job_id:
            return self.log_dir / f"{self.job_id}.log"
        return self.master_log_path

    def _now_iso(self) -> str:
        """ISO-8601 timestamp with timezone."""
        tz = datetime.datetime.now().astimezone().strftime("%z")
        tz = tz[:3] + ":" + tz[3:]
        return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S") + tz

    def _should_log(self, level: str) -> bool:
        return self.levels.get(level.upper(), 0) >= self.levels.get(self.level, 20)

    def _write_jsonl(self, log_path: Path, record: Dict[str, Any]):
        """Write a JSON Lines record with multi-process safety."""
        # Truncate oversized fields
        record = self._truncate_record(record)

        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"

        try:
            with open(log_path, "a", encoding="utf-8") as f:
                _acquire_file_lock(f)
                try:
                    f.write(line)
                    f.flush()
                finally:
                    _release_file_lock(f)
        except Exception as e:
            # Fallback: print to stderr
            print(f"[LOGGER ERROR] Failed to write to {log_path}: {e}", flush=True)
            print(line, flush=True)

    def _truncate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Truncate fields exceeding max_line_bytes while keeping valid JSON."""
        import hashlib
        result = {}
        for k, v in record.items():
            val_str = json.dumps(v, ensure_ascii=False, default=str)
            if len(val_str.encode("utf-8")) > self.max_line_bytes:
                val_hash = hashlib.sha256(val_str.encode("utf-8")).hexdigest()[:8]
                result[k] = f"<truncated_field:{k}:len={len(val_str)}:hash={val_hash}>"
            else:
                result[k] = v
        return result

    def log(self, level: str, event: str, message: str = "", **kwargs):
        """Emit a structured log record."""
        if not self._should_log(level):
            return

        record = {
            "timestamp": self._now_iso(),
            "level": level.upper(),
            "event": event,
            "message": message,
            "job_id": self.job_id,
        }
        record.update(kwargs)

        self._write_jsonl(self._get_log_path(), record)

    def debug(self, event: str, message: str = "", **kwargs):
        self.log("DEBUG", event, message, **kwargs)

    def info(self, event: str, message: str = "", **kwargs):
        self.log("INFO", event, message, **kwargs)

    def warning(self, event: str, message: str = "", **kwargs):
        self.log("WARNING", event, message, **kwargs)

    def error(self, event: str, message: str = "", **kwargs):
        self.log("ERROR", event, message, **kwargs)

    def critical(self, event: str, message: str = "", **kwargs):
        self.log("CRITICAL", event, message, **kwargs)

    def param_override(self, param_name: str, old_value: Any, new_value: Any,
                       source: str):
        """Log parameter override event."""
        self.info("PARAM_OVERRIDE",
                  f"Parameter '{param_name}' overridden",
                  param_name=param_name,
                  old_value=str(old_value),
                  new_value=str(new_value),
                  override_source=source)


def get_logger(log_dir: str = "./logs", level: str = "INFO",
               job_id: Optional[str] = None) -> StructuredLogger:
    """Factory function to create a logger instance."""
    return StructuredLogger(log_dir=log_dir, level=level, job_id=job_id)
