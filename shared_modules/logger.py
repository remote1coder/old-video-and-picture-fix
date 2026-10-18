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


# --- RV-01: 真正的级别过滤 -------------------------------------------------
# 背景（真实 CLI 复现）：video_restoration_pipeline/main.py:799 / :743 调用
# ``get_logger(..., level=cfg["log_level"])``，而 ``StructuredLogger.__init__``
# 不接受 level，于是正式入口在**依赖预检之前**就以
# ``TypeError: unexpected keyword argument 'level'`` 退出（rc=1）。
# 修法不是用 **kwargs 把参数吞掉（那只是让测试不报错），而是真正实现级别过滤，
# 并让 get_logger 的 ``level`` 实参落到实现上。
LOG_LEVEL_ORDER: Dict[str, int] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARN": 30,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}
#: 默认不过滤任何级别（保持修复前的行为，避免悄悄丢日志）
DEFAULT_LOG_LEVEL = "DEBUG"


def _normalize_level(level: Any) -> str:
    """Map a configured level onto a known name. Unknown → DEBUG (no filter)."""
    if level is None:
        return DEFAULT_LOG_LEVEL
    name = str(level).strip().upper()
    if name in LOG_LEVEL_ORDER:
        return "WARNING" if name == "WARN" else name
    # 未知级别不猜：退回"不过滤"，由调用方决定是否记录告警
    return DEFAULT_LOG_LEVEL


class StructuredLogger:
    """Structured JSON Lines logger with per-job log files."""

    def __init__(self,
                 log_dir: str = "./logs",
                 job_id: Optional[str] = None,
                 log_max_lines: int = 10000,
                 log_backup_count: int = 5,
                 log_max_line_bytes: int = 1048576,
                 level: Any = DEFAULT_LOG_LEVEL):
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.job_id = job_id
        self.log_max_lines = log_max_lines
        self.log_backup_count = log_backup_count
        self.log_max_line_bytes = log_max_line_bytes
        self.lock = threading.Lock()
        # RV-01: 真实实现级别过滤（此前该实参根本不被接受）。
        self.level = _normalize_level(level)
        self.level_value = LOG_LEVEL_ORDER[self.level]
        # FIX-P3-1: Track line count for rotation instead of size estimation
        # per v7.32 §3.10.12
        self._line_count = 0

        # Determine log file
        if job_id:
            self.log_file = self.log_dir / f"job_{job_id}.log"
        else:
            # FIX-P4-1: Use a stable "general.log" instead of timestamped
            # "general_{ts}.log". ORIGINAL created a new file per
            # instantiation (general_{int(time.time())}.log), which produced
            # an unbounded number of files across runs and defeated rotation.
            # Per v7.32 §3.10.12 / v6.32 §7.4: one stable general log file,
            # rotated by line count. A single shared file is safe here
            # because all writes go through the cross-process lock below.
            self.log_file = self.log_dir / "general.log"

    def _is_filtered(self, level: str) -> bool:
        """RV-01: True when `level` is below the configured threshold."""
        try:
            return LOG_LEVEL_ORDER[str(level).strip().upper()] < self.level_value
        except KeyError:
            # 未知级别一律放行：宁可多写一条，也不能悄悄丢掉诊断信息
            return False

    def set_level(self, level: Any) -> None:
        """RV-01: change the threshold at runtime (used by worker/main)."""
        self.level = _normalize_level(level)
        self.level_value = LOG_LEVEL_ORDER[self.level]

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
        """Rotate log file if line count exceeds limit (no-lock fallback path).

        Uses timestamp-based backup names to avoid rename race conditions
        in multi-process environments.
        FIX-P2-6: ORIGINAL used self.log_file.rename(backup), which is
        inconsistent with the locked path (_rotate_locked uses
        copyfile+truncate). rename also invalidates any open fd held by
        another process mid-write. Per v7.32 §3.10.12, the fallback path
        now mirrors the locked path: shutil.copyfile -> truncate(0), so
        the live log file keeps its inode and continues to receive writes.
        """
        line_count = self._get_current_line_count()
        if line_count >= self.log_max_lines:
            # Use timestamp-based unique name to avoid rename race
            ts = int(time.time() * 1000)
            backup = self.log_file.parent / f"{self.log_file.name}.{ts}"
            try:
                import shutil
                shutil.copyfile(str(self.log_file), str(backup))
                # Truncate the live file in place (keeps inode / open fds valid)
                with open(self.log_file, "r+b") as f:
                    f.truncate(0)
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

        FIX-P3-1: Uses tracked line count instead of size estimation per
        v7.32 §3.10.12. ORIGINAL used size//200 estimate which was inaccurate.
        FIX-P2-3-REV: Always use copyfile+truncate for rotation.
        ORIGINAL (FIX-P2-3): Used os.rename on same filesystem, which invalidated
        the open file descriptor — after rename, f still pointed to the old
        inode (now the backup), so subsequent f.write() went to the backup file
        instead of the new empty log. copyfile+truncate keeps fd valid and
        avoids the issue entirely. Per v6.32 §7.3, os.rename is forbidden
        across filesystems; for log rotation, copyfile+truncate is always
        safe under lock regardless of filesystem.
        FIX-P2-2: ORIGINAL rotated based on self._line_count, an in-process
        variable. With N processes each writing to the same job log, each
        process counts only its own lines, so rotation never triggered
        (per-process count never reaches log_max_lines). Per v7.32 §3.10.12,
        rotation must reflect the ACTUAL file line count. We now read the
        true line count from disk while holding the cross-process lock, and
        fall back to the in-memory estimate only if the read fails.
        """
        # FIX-P2-2: Read the true on-disk line count under the lock so that
        # multi-process writers rotate based on the shared file size, not a
        # per-process counter that never reaches the threshold.
        true_count = self._line_count
        try:
            pos = f.tell()
            f.seek(0, 0)
            true_count = sum(1 for _ in f)
            f.seek(pos, 0)
        except Exception:
            true_count = self._line_count  # fallback to in-memory estimate

        if true_count >= self.log_max_lines:
            ts = int(time.time() * 1000)
            backup = self.log_file.parent / f"{self.log_file.name}.{ts}"
            try:
                f.flush()
                import shutil
                shutil.copyfile(str(self.log_file), str(backup))
                f.truncate(0)
                f.seek(0, 2)
                self._line_count = 0

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
        self._line_count += 1

    def log(self, level: str, event: str, message: str, **kwargs):
        """Write a structured log entry."""
        # RV-01: 级别过滤真正生效（entry 的 level 字段保持调用方给的原文）
        if self._is_filtered(level):
            return
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
                    # FIX-P1-2: Windows rotation under cross-process lock
                    # ORIGINAL: rotation was done outside the lock via _rotate_if_needed()
                    # FIX-P2-2: Removed redundant _rotate_if_needed() outside lock.
                    # FIX-P1-4: ORIGINAL locked [old_EOF, old_EOF+MAX_LOCK_BYTES)
                    # (lock_start = f.tell() after seek(0,2)). After
                    # _rotate_locked() calls truncate(0), the write moves to
                    # offset 0 — OUTSIDE that locked region — leaving the
                    # post-rotation write unprotected. A naive fix locking only
                    # [0, MAX_LOCK_BYTES) is also wrong: in the non-rotating
                    # case the write appends at old_EOF, which for a grown log
                    # file is far beyond MAX_LOCK_BYTES and would be unlocked.
                    # Correct approach per v7.32 §3.10.12 / v6.32 §7.4: lock a
                    # single contiguous range [0, file_size + MAX_LOCK_BYTES)
                    # from offset 0. This covers BOTH write positions — the
                    # post-truncate write at 0 AND the normal append at
                    # old_EOF (file_size). Windows permits locking byte ranges
                    # beyond EOF, so the over-length tail is harmless. The
                    # same length is reused for unlock.
                    MAX_LOCK_BYTES = self.log_max_line_bytes + 1024  # safety margin
                    f.seek(0, 2)  # EOF
                    file_size = f.tell()
                    lock_len = file_size + MAX_LOCK_BYTES
                    f.seek(0, 0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, lock_len)
                    try:
                        # Rotate under cross-process lock per v7.32 §3.10.12
                        self._rotate_locked(f)
                        f.seek(0, 2)  # EOF (offset 0 after truncate, else appended)
                        f.write(line_bytes)
                        f.flush()
                    finally:
                        # Unlock the exact [0, lock_len) range we locked.
                        f.seek(0, 0)
                        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, lock_len)
                else:
                    # Fallback: no locking (single-process only)
                    self._rotate_if_needed()
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

        Source must be one of: user_overrides, audio_override,
        recommended_params, paths, global, cli, unknown.
        """
        # FIX-P2-1: Added audio_override per v7.32 §3.2 + §3.10.12
        # FIX-P1-1: Added recommended_params per v7.32 §3.2 + §3.10.12
        # ORIGINAL: allowed_sources = ("user_overrides", "paths", "global", "cli", "unknown")
        allowed_sources = ("user_overrides", "audio_override", "recommended_params", "paths", "global", "cli", "unknown")
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
_loggers_lock = threading.Lock()  # spawn-safe: 无副作用初始化器，spawn 子进程重导入安全


def get_logger(log_dir: str = "./logs",
               job_id: Optional[str] = None,
               level: Any = None,
               **kwargs) -> StructuredLogger:
    """Singleton factory: returns existing logger for job_id, or creates new one.

    RV-01: ``level`` is an explicitly supported argument (video main.py passes
    ``global.log_level``). ORIGINAL forwarded every kwarg straight into
    ``StructuredLogger.__init__``, which did not accept ``level`` — the video
    CLI died with ``TypeError: unexpected keyword argument 'level'`` before the
    dependency pre-check. The parameter is now implemented as real threshold
    filtering instead of being swallowed by ``**kwargs``.
    """
    if level is not None:
        kwargs["level"] = level
    if job_id:
        key = f"{log_dir}/{job_id}"
        with _loggers_lock:
            existing = _loggers.get(key)
            if existing is None:
                _loggers[key] = StructuredLogger(log_dir=log_dir,
                                                 job_id=job_id, **kwargs)
                return _loggers[key]
            if level is not None:
                # 单例已存在时也要让新传入的级别生效，否则"配置改了没效果"
                existing.set_level(level)
            return existing
    return StructuredLogger(log_dir=log_dir, job_id=job_id, **kwargs)


# NOTE: CORE_QUALITY_PARAMS is defined in checkpoint.py (single source of truth).
# Do not duplicate here to avoid divergence.
