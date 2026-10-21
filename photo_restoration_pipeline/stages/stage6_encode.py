#!/usr/bin/env python3
"""
Stage 6: Photo Final Encoding
- AVIF encoding with avifenc (10bit YUV420 or RGB444)
- CICP metadata
- Archival mode: RGB444 lossless when archival_mode=true

v6.3/v7.3 Compliance:
- Supports both avif_params (YUV420 10-bit) and avif_rgb444_params (RGB444)
- Grain synthesis moved to stage5_postprocess.py per v7.32 §3.1
- Proper CICP tagging

FIX-V01/V02/V04/V05 (2026-10-02): the encode boundary now follows the
documented avifenc contract instead of assumptions:
  * input must be jpg/jpeg/png/y4m — TIFF intermediates are converted
    losslessly (16-bit preserved) in an explicit adapter (V05);
  * `--depth` only accepts 8/10/12; a 16-bit result requires the documented
    depth extension (e.g. `12,8`) on libavif >= 1.4.0 (V01);
  * the TIFF master is written at the *actual* pixel precision and only ever
    carries an ICC profile that matches the pixels it labels (V02);
  * wide-gamut conversion needs a real source profile and fails loudly instead
    of warning and continuing (V04).
Every run writes an `*.encode_contract.json` sidecar describing what was
actually encoded — not what was requested.
"""

import os
import sys
import json
import argparse
import glob
import subprocess
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from checkpoint import wsl_redirect_path, wsl_copy_back, safe_input_stem  # noqa: E402
# PUB-04（第七轮）：发布目标互斥锁（逐路径加锁，非集合摘要）
from publish_lock import acquire_target_locks, TargetLockTimeout  # noqa: E402
# 持久 journal（第七轮）：让硬终止后"上次做到哪一步"可判断
from publish_journal import (  # noqa: E402
    JOURNAL_SCHEMA, PHASE_PREPARED, PHASE_COMMITTED,
    PHASE_RECOVERY_REQUIRED, file_digest, journal_path_for, write_journal,
    read_journal, validate_journal_paths, classify_recovery)

try:  # FIX-V01/V05: encode-boundary contracts
    sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
    import media_io  # noqa: E402
except Exception:  # pragma: no cover - defensive; stage must fail loudly later
    media_io = None  # type: ignore[assignment]

try:  # NEW-V-06: 取消/中断守卫（except Exception 抓不到 KeyboardInterrupt）
    import cancel_guard  # noqa: E402
except Exception:  # pragma: no cover - 缺失时不阻断 stage，仅失去该保护
    cancel_guard = None  # type: ignore[assignment]


# FIX-V04: colour-space handling outcomes
COLOR_NOT_NEEDED = "not-needed"      # source already sRGB / no conversion required
COLOR_CONVERTED = "converted"        # real profile → real pixel transform → sRGB
COLOR_UNKNOWN = "unknown-source"     # source space not established
COLOR_DEGRADED = "degraded"          # conversion impossible, explicitly accepted
COLOR_FAILED = "failed"              # required conversion impossible → stage fails

WIDE_GAMUT_SPACES = ("adobergb", "prophoto", "displayp3", "p3")


def _get_avifenc_version() -> tuple:
    """Parse avifenc version. Returns (major, minor, patch) or (0, 0, 0)."""
    if media_io is not None:
        cap = media_io.detect_avifenc_capability()
        return cap.version
    return (0, 0, 0)


def _resolve_jobs_value(jobs: str) -> Optional[str]:
    """Resolve --jobs value based on avifenc version (FIX-P12-10)."""
    if media_io is not None:
        return media_io.resolve_jobs_value(jobs, media_io.detect_avifenc_capability())
    if jobs != "all":
        return str(jobs)
    version = _get_avifenc_version()
    if version >= (1, 1, 0):
        return "all"
    elif version >= (0, 9, 2):
        return str(min(os.cpu_count() or 1, 32))
    return None


def _build_avifenc_cmd(input_path: str, output_path: str, params: dict,
                       use_rgb444: bool = False) -> list:
    """Build a validated avifenc command from config parameters.

    FIX-V01: raises media_io.UnsupportedAvifConfig instead of emitting a bare
    ``--depth 16`` that no avifenc version accepts.
    """
    if media_io is None:
        raise RuntimeError("media_io module unavailable")
    cmd, _contract = media_io.build_avifenc_cmd(
        input_path, output_path, params, use_rgb444=use_rgb444)
    return cmd


def _run_encoder(cmd: List[str], timeout: int, logger=None) -> Tuple[bool, str, int]:
    """RV-12: run the encoder as a *managed* child process.

    ``subprocess.run`` cannot be interrupted: a cancellation or a timeout
    would kill only the Python stage while avifenc keeps writing. Here the
    child is registered with the cancel guard and is killed+awaited on
    timeout, cancellation or any ``BaseException``.

    Returns ``(ok, stderr, returncode)``.
    """
    popen_kwargs: Dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "shell": False,
    }
    try:  # Windows: 独立进程组，便于 CTRL_BREAK_EVENT
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    except Exception:
        pass
    proc = subprocess.Popen(cmd, **popen_kwargs)
    if cancel_guard is not None:
        cancel_guard.watch_process(proc)
    try:
        _out, _err = proc.communicate(timeout=timeout)
        stderr = (_err or b"").decode("utf-8", "replace")
        return proc.returncode == 0, stderr, int(proc.returncode)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=10)
        except BaseException:  # noqa: BLE001
            pass
        if logger:
            logger.error("STAGE6_ENCODER_TIMEOUT",
                         f"Encoder exceeded {timeout}s and was terminated")
        return False, f"timeout after {timeout}s", 124
    except BaseException as exc:  # noqa: BLE001 - KeyboardInterrupt 也走这里
        # 取消/中断：必须先把子进程收掉，再让异常继续向上（由守卫写 FAILED）
        try:
            proc.kill()
            proc.wait(timeout=10)
        except BaseException:  # noqa: BLE001
            pass
        if logger:
            logger.error("STAGE6_ENCODER_ABORTED",
                         f"Encoder terminated: {type(exc).__name__}")
        raise
    finally:
        if cancel_guard is not None:
            cancel_guard.unwatch_process(proc)


def _write_json_sidecar(path: str, payload: Dict[str, Any]) -> bool:
    """Atomically write a JSON sidecar (temp + replace)."""
    tmp = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False


def _color_skipped(color: Dict[str, Any]) -> List[str]:
    """RV-06: operations that were requested but not actually performed."""
    skipped: List[str] = []
    if color.get("status") == COLOR_DEGRADED:
        skipped.append("color_conversion")
    return skipped


class PublishAbort(Exception):
    """R2-01: 成套提交过程中任一步失败 —— 必须整批回滚，不能留下半代。"""

    def __init__(self, operation: str, reason: str):
        super().__init__(f"{operation}: {reason}")
        self.operation = operation
        self.reason = reason


# R3-03 / B03：提交前每个目标路径的状态必须**三态可区分**。
# 旧实现用 None 同时表示"原先不存在"和"备份失败"，于是回滚把"备份失败"的
# 路径直接**删除** —— 旧产物被销毁，且日志还声称"完整旧代已恢复"。
BACKUP_ABSENT = "absent"        # 提交前不存在 → 回滚时撤下本次新建的文件
BACKUP_OK = "backed_up"         # 已备份 → 回滚时拷回
BACKUP_FAILED = "backup_failed"  # 备份失败 → 回滚时**绝不能动**


def _snapshot_for_rollback(paths: List[str], backup_dir: str,
                           logger=None) -> Dict[str, Dict[str, Optional[str]]]:
    """R3-03: 提交前把**现存**产物复制到备份目录，并明确记录每个目标的状态。

    Returns ``{目标路径: {"state": ..., "backup": 备份路径 或 None}}``。
    """
    snapshot: Dict[str, Dict[str, Optional[str]]] = {}
    for idx, path in enumerate([p for p in paths if p]):
        if not os.path.isfile(path):
            snapshot[path] = {"state": BACKUP_ABSENT, "backup": None}
            continue
        try:
            os.makedirs(backup_dir, exist_ok=True)
            backup = os.path.join(backup_dir, f"{idx}_{os.path.basename(path)}")
            # R3-03：必须**真实复制**，不能图快用硬链接。
            # 实测（本机）：os.link(path, backup) 之后若调用方**原地写**
            # path（open(path,'w') / write_bytes），备份文件的內容也会跟着变，
            # 于是回滚恢复出来的就是被改坏的数据 —— 等于没有备份。
            # 提交阶段虽然主要用 os.replace 换 inode，但无法强制所有调用方
            # 都这样做，备份必须自己站得住。
            shutil.copyfile(path, backup)
            snapshot[path] = {"state": BACKUP_OK, "backup": backup}
        except OSError as e:
            snapshot[path] = {"state": BACKUP_FAILED, "backup": None}
            if logger:
                logger.error("STAGE6_ROLLBACK_BACKUP_FAIL",
                             f"Cannot back up {path} before publishing: {e}. "
                             "This path will NOT be touched on rollback.",
                             path=path)
    return snapshot


def _snapshot_has_failure(snapshot: Dict[str, Dict[str, Optional[str]]]) -> bool:
    """R3-03: 有任何一个目标备份失败 → 提交必须先停，不能动旧产物。"""
    return any(entry.get("state") == BACKUP_FAILED
               for entry in snapshot.values())


def _recover_prior_publish(work_output_path: str,
                           extra_targets: Optional[List[str]] = None,
                           logger=None) -> Dict[str, Any]:
    """A4 方案 1：进入发布临界区前，按 journal 收拾上一次未完成的事务。

    **判断层早已就绪，缺的只是这个调用点。**
    `publish_journal` 里的 `read_journal` / `validate_journal_paths` /
    `classify_recovery` 都已实现，`tests/test_r7_kill9_recovery.py` 还用
    **真 kill -9**（`TerminateProcess`，子进程没有任何机会跑清理代码）
    演练过"旧成品完好 + journal 停在 PREPARED + classify_recovery 给出
    明确结论"。但**生产路径从来没读过 journal** —— 于是硬杀后残留的
    `.pubbak` 与可能存在的混合代一直没人处理。

    沿用 publish_journal 的原则：
    * 用**内容摘要**识别新旧代，不相信阶段字段；
    * 有任何目标匹配不上已记录的代 → `KEEP_AND_ASK`，**绝不自动恢复**；
    * journal 是磁盘数据不是可信指令，先过路径归属校验。

    返回 `{"status": ..., "action": ..., "report": ...}`：
      `none`           没有上次事务的记录（journal 不存在/读不出）
      `already_new`    上次已完整提交，只需清残留
      `rolled_back`    上次是混合代/未提交，已用备份恢复旧代
      `needs_human`    状态无法判定，保留现场等人工（调用方应失败）
    """
    jp = journal_path_for(work_output_path)
    doc = read_journal(jp)
    if doc is None:
        return {"status": "none", "action": None, "report": None}

    # 路径归属校验：journal 可能被外部改写，恢复/清理前必须确认它指的
    # 每个路径都落在本次事务授权的根目录内。
    roots: List[str] = []
    for p in ([work_output_path] + list(extra_targets or [])):
        if p:
            roots.append(str(Path(p).resolve().parent))
    problems = validate_journal_paths(doc, roots)
    if problems:
        if logger:
            logger.error("STAGE6_JOURNAL_UNTRUSTED",
                         "Prior publish journal failed path validation; "
                         "leaving everything untouched",
                         journal=jp, problems=problems[:5])
        return {"status": "needs_human", "action": None,
                "report": {"problems": problems}}

    rep = classify_recovery(doc)
    rec = str(rep.get("recommendation") or "")
    if logger:
        logger.info("STAGE6_PRIOR_PUBLISH_CHECK",
                    "Recovered the state of a previous publish transaction",
                    journal=jp, phase=rep.get("phase"),
                    recommendation=rec,
                    targets={k: v.get("state")
                             for k, v in (rep.get("targets") or {}).items()})

    if rec.startswith("KEEP_AND_ASK"):
        # 不猜：保留现场，让调用方失败并等人工介入。
        return {"status": "needs_human", "action": None, "report": rep}

    if rec.startswith("ALREADY_COMMITTED"):
        # 上次已完整提交。清掉这次事务的 journal 与备份残留即可。
        _purge_publish_residue(jp, rep, logger)
        return {"status": "already_new", "action": "purged", "report": rep}

    # ROLLBACK_SAFE（全是旧代，恢复是 no-op）与 MIXED（混合代，必须恢复）
    # 都走同一条路：按 journal 记录的备份把每个目标恢复成旧代。
    snapshot: Dict[str, Dict[str, Optional[str]]] = {}
    for t in (doc.get("targets") or []):
        r = (doc.get("records") or {}).get(t) or {}
        bk = r.get("backup")
        if bk and os.path.exists(bk):
            snapshot[t] = {"state": BACKUP_OK, "backup": bk}
        elif r.get("existed") is False:
            snapshot[t] = {"state": BACKUP_ABSENT, "backup": None}
        else:
            snapshot[t] = {"state": BACKUP_FAILED, "backup": None}
    rb = _rollback_publish(snapshot, logger)
    if not rb.get("fully_restored"):
        if logger:
            logger.error("STAGE6_PRIOR_PUBLISH_ROLLBACK_FAILED",
                         "Could not fully restore the previous generation "
                         "from journal backups; leaving everything untouched",
                         failed=rb.get("failed"),
                         impossible=rb.get("impossible"))
        return {"status": "needs_human", "action": None,
                "report": {"rollback": rb}}
    _purge_publish_residue(jp, rep, logger)
    return {"status": "rolled_back", "action": "rolled_back",
            "report": {"recovery": rep, "rollback": rb}}


def _purge_publish_residue(journal_path: str, report: Dict[str, Any],
                           logger=None) -> None:
    """删掉一次已了结事务留下的 journal 与 `.pubbak` 备份。

    只删 journal 里**记录过**的备份路径，不做目录遍历猜测
    （`publish_journal` 的 docstring 专门批评过"发现备份就回滚"是猜测）。
    """
    removed: List[str] = []
    for t, info in (report.get("targets") or {}).items():
        bk = info.get("backup")
        if bk and os.path.isfile(bk):
            try:
                os.remove(bk)
                removed.append(bk)
            except OSError:
                pass
    try:
        if os.path.exists(journal_path):
            os.remove(journal_path)
            removed.append(journal_path)
    except OSError:
        pass
    # 备份文件都删掉后，`.pubbak` 目录通常是空的 —— 顺手移除。
    # 用 rmdir 而不是 rmtree：**只有目录为空才会成功**，万一里面还有
    # journal 没记录到的文件（不该有，但不猜），就留着不动。
    for d in {str(Path(b).parent)
              for b in (info.get("backup")
                        for info in (report.get("targets") or {}).values())
              if b}:
        try:
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
                removed.append(d)
        except OSError:
            pass
    if removed and logger:
        logger.info("STAGE6_PRIOR_PUBLISH_RESIDUE_PURGED",
                    "Removed journal and backups of a concluded transaction",
                    removed=removed)


def _rollback_publish(snapshot: Dict[str, Dict[str, Optional[str]]],
                      logger=None) -> Dict[str, Any]:
    """R3-03 / PUB-02（第七轮）：把每个目标路径恢复成提交前的内容。

    返回**可直接核验的逐文件结果**（旧实现返回 None，调用方无法知道
    部分恢复失败，却仍打印"Previous generation restored"）：

    ```
    {
      "restored":  [path, ...],       # 从备份拷回成功
      "removed":   [path, ...],       # 本次新建、已正确删除
      "failed":    [{"path","error"}],# 恢复动作本身失败
      "impossible":[path, ...],       # 备份失败/缺失，未改动（没得恢复）
      "fully_restored": bool,         # = not failed and not impossible
    }
    ```

    * ``backed_up``  → 从备份拷回；
    * ``absent``     → 删除本次新建的文件；
    * ``backup_failed`` → **不动**（没有备份可恢复，删掉就等于销毁旧产物），
      记 `STAGE6_RESTORE_IMPOSSIBLE`，绝不声称"完整旧代已恢复"。
    """
    result: Dict[str, Any] = {"restored": [], "removed": [], "failed": [],
                              "impossible": [], "fully_restored": False}
    for path, entry in snapshot.items():
        state = entry.get("state")
        backup = entry.get("backup")
        try:
            if state == BACKUP_OK and backup and os.path.isfile(backup):
                # R3-03 / B14：优先 `os.replace` —— 单次系统调用、原子，
                # 比 copyfile 的失败面小得多。取消处理器里做文件 I/O 有额外
                # 风险（Windows 实测会再抛 KeyboardInterrupt），
                # 越少的系统调用越好；跨卷失败时再退回复制。
                try:
                    os.replace(backup, path)
                    result["restored"].append(path)
                    continue
                except OSError:
                    pass
                shutil.copyfile(backup, path)
                result["restored"].append(path)
            elif state == BACKUP_ABSENT and os.path.exists(path):
                os.remove(path)
                result["removed"].append(path)
            elif state == BACKUP_FAILED:
                result["impossible"].append(path)
            elif state == BACKUP_OK:
                # PUB-02（测试抓到的真实漏洞）：snapshot 声明"备份已成功"，
                # 但备份文件此刻不可用（被外部删除 / 损坏 / 被清理）。
                # 旧实现的三个分支都不匹配 → **静默跳过该目标** →
                # fully_restored 仍为 True，等于假报"完整恢复"。
                # 这里必须显式记为失败，阻止假报成功。
                result["failed"].append({
                    "path": path,
                    "error": "declared backup is missing or unreadable",
                })
                if logger:
                    logger.error("STAGE6_ROLLBACK_BACKUP_MISSING",
                                 f"Snapshot claims a backup for {path} but it "
                                 "is not readable; cannot restore this target",
                                 path=path, backup=backup)
        except OSError as e:
            # PUB-02：不再静默吞掉 —— 记录到结构化结果，由调用方决定
            # 是否进入 RECOVERY_REQUIRED，绝不假报"已完整恢复"。
            result["failed"].append({"path": path, "error": str(e)})
            if logger:
                logger.error("STAGE6_ROLLBACK_FAIL",
                             f"Could not restore {path}: {e}", path=path)

    result["fully_restored"] = (not result["failed"]
                               and not result["impossible"])
    if result["impossible"] and logger:
        logger.error("STAGE6_RESTORE_IMPOSSIBLE",
                     f"{len(result['impossible'])} path(s) had no usable backup "
                     "and were left untouched; the previous generation could NOT "
                     "be fully restored — do not report a clean rollback",
                     impossible_paths=result["impossible"])
    if logger:
        logger.info("STAGE6_ROLLBACK_RESULT",
                    f"restored={len(result['restored'])} "
                    f"removed={len(result['removed'])} "
                    f"failed={len(result['failed'])} "
                    f"impossible={len(result['impossible'])}",
                    fully_restored=result["fully_restored"])
    return result
    if logger:
        logger.info("STAGE6_PUBLISH_ROLLBACK",
                    f"Rollback finished: {restored} restored, "
                    f"{impossible} impossible")


def _fail(logger, status_path: str, operation: str, reason: str,
          paths: Optional[List[str]] = None,
          preserve_valid_status: bool = False,
          recovery_required: bool = False,
          rollback_detail: Optional[Dict[str, Any]] = None) -> None:
    """RV-12: clean this run's temp artifacts and record an honest FAILED.

    The status sidecar is what lets the scheduler and the next run tell
    "failed" apart from "an old, still valid product".

    PUB-03（第七轮）：**当前有效成品状态**与**本次失败 attempt 状态**必须分开。

    * ``preserve_valid_status=True`` —— 回滚已完整恢复旧代，旧代仍是有效
      成品。此时**不覆写**固定 status（否则会把一份好产物标成 FAILED，
      下次运行误判为需要重跑，甚至触发清理删除它）。本次失败记到
      `<status>.attempt-<pid>-<ms>.json`。
    * ``recovery_required=True`` —— 回滚**没有**完整恢复，磁盘处于未知
      混合状态。此时覆写固定 status 为 ``RECOVERY_REQUIRED`` 是正确的：
      必须阻止下一次运行把这些路径当成有效产物。
    """
    for p in paths or []:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    payload: Dict[str, Any] = {
        "status": "RECOVERY_REQUIRED" if recovery_required else "FAILED",
        "stage": 6,
        "failed_operation": operation,
        "reason": str(reason)[:2000],
    }
    if rollback_detail is not None:
        payload["rollback"] = {
            "restored": rollback_detail.get("restored", []),
            "removed": rollback_detail.get("removed", []),
            "failed": rollback_detail.get("failed", []),
            "impossible": rollback_detail.get("impossible", []),
            "fully_restored": rollback_detail.get("fully_restored", False),
        }

    if preserve_valid_status:
        # 旧代完好 → 固定 status 不动，只落一个 attempt 级记录
        attempt_sidecar = (f"{status_path}.attempt-"
                           f"{os.getpid()}-{int(time.time() * 1000)}.json")
        _write_json_sidecar(attempt_sidecar, payload)
        if logger:
            logger.info("STAGE6_ATTEMPT_FAILURE_RECORDED",
                        "This attempt failed but the previously published "
                        "generation was fully restored and remains valid; "
                        "the fixed status sidecar was left untouched",
                        attempt_sidecar=attempt_sidecar)
    else:
        _write_json_sidecar(status_path, payload)

    if cancel_guard is not None:
        cancel_guard.release_outputs()


def _calculate_avifenc_timeout(input_path: str, base_timeout: int = 600) -> int:
    """BUG-057 / ISSUE-P-S6-003: Dynamic timeout based on image pixel count."""
    try:
        from PIL import Image
        with Image.open(input_path) as img:
            w, h = img.size
        mp = (w * h) / 1_000_000
        dynamic = int(mp * 60)
        return max(base_timeout, dynamic)
    except Exception:
        return base_timeout


def run_avifenc(input_path: str, output_path: str, params: dict,
                use_rgb444: bool = False) -> Tuple[bool, str]:
    """Run avifenc. Returns (ok, stderr)."""
    cmd = _build_avifenc_cmd(input_path, output_path, params, use_rgb444)
    timeout = _calculate_avifenc_timeout(input_path)
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, shell=False)
        return r.returncode == 0, (r.stderr or b"").decode("utf-8", "replace")
    except Exception as e:
        return False, str(e)


# --- FIX-V04: colour contract ----------------------------------------------

def _read_color_meta(temp_dir: str, job_id: str, stem: str) -> Dict[str, Any]:
    """Read the stage1 colour sidecar if present."""
    path = Path(temp_dir) / f"{job_id}_{stem}_color_meta.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


#: RV-02: 这些值都不是"已知色彩空间"，不能当成声明值参与判断。
#: 默认配置就是 "auto" —— 它是非空字符串，会盖过 stage1 sidecar，也不读嵌入
#: ICC，于是"检测"这条路径永远走不到（复核实测：sidecar 写 AdobeRGB 仍被忽略，
#: 无来源 + strict=true 也不失败）。
AUTO_COLOR_VALUES = frozenset({"", "auto", "none", "null", "detect", "unknown",
                               "unspecified", "default"})
#: RV-02: 未指定标签（libavif 手册：CICP 中 2 = unspecified）。
#: NEW-RV-03: 矩阵位不能写成 2 —— lossless 模式只接受 identity(0)/YCgCo，
#: 真机实测 avifenc 1.4.2 会直接 rc=1（"Matrix coefficients have to be
#: identity, YCgCo-Re, or YCgCo-Ro in lossless mode."）。因此这里用
#: "2/2/0"：primaries/transfer 未指定 + identity 矩阵。
CICP_UNSPECIFIED_RGB444 = "2/2/0"
CICP_UNSPECIFIED_YUV = "2/2/1"
#: 真实 sRGB 像素对应的标签：primaries=1(BT.709) / transfer=13(sRGB)
CICP_SRGB_RGB444 = "1/13/0"   # RGB444 → matrix 0 (identity)
CICP_SRGB_YUV = "1/13/1"      # YUV → matrix 1 (BT.709)


def _normalize_declared_color(value: Any) -> Optional[str]:
    """RV-02: normalise a declared colour space; ``auto`` means 'undecided'."""
    text = str(value or "").strip().lower()
    if text in AUTO_COLOR_VALUES:
        return None
    return text


def _extract_source_icc(working_path: str) -> Optional[bytes]:
    """Return the ICC profile actually embedded in the working image.

    RV-04: Pillow cannot represent 16-bit multi-channel data, so the profile
    is read through media_io (TIFF tag 34675 / PNG iCCP) first and Pillow is
    only a fallback.
    """
    if media_io is not None:
        try:
            _arr, meta = media_io.read_image_array(working_path)
            icc = meta.get("icc_profile")
            if icc:
                return icc
        except Exception:
            pass
    try:
        from PIL import Image
        with Image.open(working_path) as im:
            return im.info.get("icc_profile")
    except Exception:
        return None


def _describe_profile(icc_bytes: bytes) -> str:
    try:
        import io
        from PIL import ImageCms
        prof = ImageCms.getOpenProfile(io.BytesIO(icc_bytes))
        return str(ImageCms.getProfileDescription(prof))
    except Exception:
        return ""


def resolve_color_space(working_path: str, params: Dict[str, Any],
                        temp_dir: str, job_id: str, stem: str,
                        logger) -> Dict[str, Any]:
    """Establish (and enforce) the colour contract before encoding (FIX-V04).

    ORIGINAL called ``ImageCms.createProfile("ADBE"/"ROMM")`` — Pillow only
    supports on-the-fly creation for LAB/XYZ/sRGB, so both calls raised and the
    code only warned before encoding and reporting success. Verified on this
    machine with Pillow 10.4.0 and 12.3.0: ``PyCMSError``.

    Returns a dict with keys: status, working_path, source_space, icc_path,
    cicp_override, precision, reason.
    """
    result: Dict[str, Any] = {
        "status": COLOR_NOT_NEEDED,
        "working_path": working_path,
        "source_space": "unknown",
        "icc_path": None,
        "cicp_override": None,
        "precision": None,
        "reason": None,
        "profile_description": None,
        # RV-02: 这个 job 最终采用的色彩标签依据（用户声明 / 实测 ICC / 未指定）
        "color_basis": None,
        "source_icc_bytes": None,
    }
    color_meta = _read_color_meta(temp_dir, job_id, stem)
    # RV-02: 配置的 auto/空值不再被当成"已知声明"，因此会继续往下走
    # sidecar → 嵌入 ICC 的检测路径。
    declared = _normalize_declared_color(params.get("input_color_space"))
    declared_source = "params" if declared else None
    if declared is None:
        declared = _normalize_declared_color(color_meta.get("input_color_space"))
        declared_source = "stage1-sidecar" if declared else None
    if declared is None:
        # No sidecar/override: fall back to the profile embedded in the file.
        icc = _extract_source_icc(working_path)
        if icc:
            result["source_icc_bytes"] = icc
            desc = _describe_profile(icc).lower()
            result["profile_description"] = _describe_profile(icc) or None
            if "adobe" in desc:
                declared = "adobergb"
            elif "prophoto" in desc or "romm" in desc:
                declared = "prophoto"
            elif "display p3" in desc or "displayp3" in desc:
                declared = "displayp3"
            elif "srgb" in desc or "iec61966" in desc:
                declared = "srgb"
            else:
                declared = "unknown"
            declared_source = "embedded-icc"
        else:
            declared = "unknown"
    result["source_space"] = declared or "unknown"
    result["color_basis"] = declared_source

    if result["source_space"] not in WIDE_GAMUT_SPACES:
        if result["source_space"] == "unknown":
            # FIX-V04 / RV-02: 未知源色彩不猜，也不假装是 sRGB。
            # 之前这里直接 return，配置里的 sRGB 标签就原样落到产物上 ——
            # 等于给来源不明的像素贴了一个没依据的标签。现在改成"未指定"。
            result["status"] = COLOR_UNKNOWN
            result["reason"] = ("source colour space not established; no ICC "
                                "in file and no input_color_space metadata")
            result["cicp_override"] = (CICP_UNSPECIFIED_RGB444
                                       if params.get("archival_mode", False)
                                       else CICP_UNSPECIFIED_YUV)
            result["color_basis"] = "unspecified"
            logger.warning("STAGE6_COLOR_UNKNOWN",
                           "Source colour space unknown; encoding with "
                           "UNSPECIFIED CICP (2/2/2) instead of claiming "
                           "sRGB. Set require_known_color_space=true to fail "
                           "instead.")
            if params.get("require_known_color_space", False):
                result["status"] = COLOR_FAILED
                result["reason"] += " (require_known_color_space=true)"
                logger.error("STAGE6_COLOR_FAILED",
                             "Unknown source colour space is not accepted "
                             "(require_known_color_space=true)")
        elif result["source_space"] == "srgb":
            # 已确认是 sRGB：覆盖与之冲突的旧配置标签（RV-02 验收项）
            result["cicp_override"] = (CICP_SRGB_RGB444
                                       if params.get("archival_mode", False)
                                       else CICP_SRGB_YUV)
        return result

    # --- wide-gamut source: a real transform is mandatory -------------------
    allow_degrade = bool(params.get("allow_unconverted_colorspace", False))
    icc_bytes = _extract_source_icc(working_path)
    if not icc_bytes:
        # FIX-2026-10-20（G6 实测）：中间产物一律不带 ICC（G'MIC 重写像素时
        # 丢掉），而宽色域转换必须有它。stage1 检测到 ICC 时会按**同一命名**
        # 落一份 sidecar（见 stage1 的 STAGE1_ICC_SIDECAR），这里回退读它。
        _icc_sidecar = Path(temp_dir) / f"{job_id}_{stem}_source.icc"
        try:
            if _icc_sidecar.is_file():
                icc_bytes = _icc_sidecar.read_bytes()
                logger.info("STAGE6_ICC_SIDECAR",
                            "Working file carries no ICC; using the sidecar "
                            "saved by stage1",
                            path=str(_icc_sidecar),
                            profile_size=len(icc_bytes))
        except OSError as _ice:
            logger.warning("STAGE6_ICC_SIDECAR_FAIL", str(_ice))
    source_icc_file: Optional[str] = None
    src_profile = None
    try:
        from PIL import ImageCms
        import io as _io
        if icc_bytes:
            source_icc_file = str(Path(temp_dir) / f"{job_id}_{stem}_source.icc")
            with open(source_icc_file, "wb") as f:
                f.write(icc_bytes)
            src_profile = ImageCms.getOpenProfile(source_icc_file)
            result["profile_description"] = _describe_profile(icc_bytes) or None
        elif params.get("source_icc_path"):
            source_icc_file = str(params["source_icc_path"])
            src_profile = ImageCms.getOpenProfile(source_icc_file)
            result["profile_description"] = _describe_profile(
                Path(source_icc_file).read_bytes()) or None
    except Exception as e:
        src_profile = None
        result["reason"] = f"source ICC could not be opened: {e}"

    if src_profile is None:
        result["icc_path"] = source_icc_file
        if not allow_degrade:
            result["status"] = COLOR_FAILED
            result["reason"] = (result["reason"] or "no usable source ICC profile")
            logger.error("STAGE6_COLOR_FAILED",
                         f"{result['source_space']} source needs a real ICC "
                         "profile for conversion; none available. Refusing to "
                         "encode pixels that would be mislabelled as sRGB.")
            return result
        result["status"] = COLOR_DEGRADED
        # R2-13：按**最终布局**选合法矩阵，不能硬写 identity 0。
        # 常规 YUV420 下传 0 时 avifenc **不报错**，而是警告后自动改成 6，
        # 于是契约（0）与真实文件头（6）不一致（A20 实测）。
        result["cicp_override"] = (CICP_UNSPECIFIED_RGB444
                                   if params.get("archival_mode", False)
                                   else CICP_UNSPECIFIED_YUV)
        result["reason"] = (result["reason"] or "no usable source ICC profile")
        logger.warning("STAGE6_COLOR_DEGRADED",
                       "Encoding without colour conversion; CICP set to "
                       "unspecified primaries/transfer so the sRGB label is "
                       "not claimed",
                       source_space=result["source_space"])
        return result

    try:
        from PIL import Image as _PILImg, ImageCms as _ImageCms
        img = _PILImg.open(working_path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        if img.mode == "L":
            img = img.convert("RGB")
        # Pillow's C-API conversion works on 8-bit data. Honest recording:
        result["precision"] = "8-bit"
        if params.get("require_16bit_color_conversion", False):
            result["status"] = COLOR_FAILED
            result["reason"] = "16-bit colour conversion not available via Pillow"
            logger.error("STAGE6_COLOR_FAILED",
                         "require_16bit_color_conversion=true but only 8-bit "
                         "CMS conversion is available")
            return result
        srgb_profile = _ImageCms.createProfile("sRGB")  # supported on-the-fly
        converted = _ImageCms.profileToProfile(img, src_profile, srgb_profile,
                                               outputMode="RGB")
        conv_path = str(Path(temp_dir) / f"{job_id}_{stem}_srgb.png")
        converted.save(conv_path, format="PNG")
        result["status"] = COLOR_CONVERTED
        result["working_path"] = conv_path
        # RV-02: 转换后的像素确实已经是 sRGB，因此必须覆盖与之冲突的旧配置
        # 标签；同时**不能**把原始宽色域 profile 贴回转换后的像素（RV-04）。
        result["cicp_override"] = (CICP_SRGB_RGB444
                                   if params.get("archival_mode", False)
                                   else CICP_SRGB_YUV)
        result["icc_path"] = None
        result["color_basis"] = "converted-to-srgb"
        logger.info("STAGE6_COLOR_CONVERTED",
                    f"Converted {result['source_space']} -> sRGB with a real "
                    "ICC transform (8-bit CMS precision)",
                    source_profile=result["profile_description"])
        return result
    except Exception as e:
        if not allow_degrade:
            result["status"] = COLOR_FAILED
            result["reason"] = f"colour conversion failed: {e}"
            logger.error("STAGE6_COLOR_FAILED",
                         f"Mandatory colour conversion failed: {e}")
            return result
        result["status"] = COLOR_DEGRADED
        # R2-13：CICP 必须按**最终布局**选合法矩阵。
        # 旧代码这里硬写 "2/2/0"（矩阵 0 = Identity）：在 RGB444 下合法，
        # 但在常规 YUV420 下 avifenc 不会拒绝，而是**警告后自动把 0 改成 6**，
        # 于是保存的契约（0）与真实文件头（6）不一致 —— 契约失真。
        # 实测（A20）：avifdec 读回 Matrix Coeffs = 6，contract 仍写 0。
        result["cicp_override"] = (CICP_UNSPECIFIED_RGB444
                                   if params.get("archival_mode", False)
                                   else CICP_UNSPECIFIED_YUV)
        result["icc_path"] = source_icc_file
        result["reason"] = f"colour conversion failed: {e}"
        logger.warning("STAGE6_COLOR_DEGRADED",
                       "Colour conversion failed; degraded encode accepted "
                       "with unspecified CICP + source ICC attached",
                       reason=str(e))
        return result


# --- FIX-V02: archival TIFF master -----------------------------------------

def quantization_grid_bits(arr) -> int:
    """RV-14: **diagnostic** estimate of the sample value grid, NOT proof of
    information depth.

    What it actually measures: the greatest common divisor of the sample
    values. If every sample is a multiple of ``g``, the data can only live on
    a grid of ``2**16 / g`` levels — which detects patterns such as
    "8-bit data multiplied by 257".

    What it cannot prove:
    * a genuinely 16-bit constant image uses very few levels, so this reports
      a small number even though the source really was 16-bit;
    * adding a 1-LSB perturbation to an 8-bit×257 image destroys the GCD
      without restoring any real detail.

    Therefore this value is recorded as a diagnostic only. Lossless acceptance
    is decided by comparing pixels against a known reference with an
    independent decoder (``np.array_equal``), never by this number.
    """
    import numpy as np
    a = np.asarray(arr)
    if a.dtype != np.uint16:
        return 8
    flat = a.reshape(-1)
    if flat.size == 0:
        return 0
    if flat.size > 200000:  # 大图抽样，够用且快
        idx = np.linspace(0, flat.size - 1, 200000).astype(np.int64)
        flat = flat[idx]
    g = int(np.gcd.reduce(flat.astype(np.uint32)))
    if g <= 0:
        return 0
    levels = max(1, 65536 // g)
    bits = int(np.ceil(np.log2(levels)))
    return max(1, min(16, bits))


def effective_bit_depth(arr) -> int:
    """Deprecated alias of :func:`quantization_grid_bits`.

    Kept for backwards compatibility with existing callers. The name suggested
    "how many bits of information the image really carries", which the GCD
    heuristic cannot establish (RV-14) — use
    :func:`quantization_grid_bits` and treat it as a diagnostic.
    """
    return quantization_grid_bits(arr)


def container_bit_depth(arr) -> int:
    """RV-14: the bit depth of the **container** (what the dtype can hold)."""
    import numpy as np
    return 16 if getattr(arr, "dtype", None) == np.uint16 else 8


def write_archival_master(working_path: str, tiff_path: str,
                          params: Dict[str, Any], logger) -> Dict[str, Any]:
    """Write the TIFF master at its *real* precision (FIX-V02).

    ORIGINAL: ``img.convert("RGB")`` then ``save(tiff)`` while logging "16bit
    TIFF"; the resulting file has BitsPerSample (8,8,8) — verified on this
    machine. It also attached an Adobe RGB / ProPhoto profile found by scanning
    the system colour directory, i.e. a label that does not match the pixels.
    """
    info: Dict[str, Any] = {
        "written": False,
        "requested_bit_depth": params.get("photo_bit_depth", 16),
        "actual_bit_depth": None,
        "precision_loss": False,
        "icc": "none",
        "writer": None,
        "dtype": None,
        "error": None,
    }
    if media_io is None:
        info["error"] = "media_io unavailable"
        return info
    try:
        import numpy as np
        arr, meta = media_io.read_image_array(working_path)
        arr = media_io._ensure_uint(arr)
        info["dtype"] = str(arr.dtype)
        # RV-14: 三件事分开记录 —— 容器位深 / 量化网格诊断 / 像素保真。
        # 旧代码把 GCD 结果叫 "effective_bit_depth" 并当成"真实信息位深"的
        # 证据，这个推论不成立（常量 16-bit 图、加 1LSB 扰动都能反例）。
        info["container_bit_depth"] = container_bit_depth(arr)
        info["actual_bit_depth"] = info["container_bit_depth"]
        info["precision_loss"] = (info["requested_bit_depth"] == 16 and
                                  info["container_bit_depth"] == 8)
        info["quantization_grid_bits"] = quantization_grid_bits(arr)
        # 兼容旧键名（下游/测试仍在读），但语义已降级为"诊断值"
        info["effective_bit_depth"] = info["quantization_grid_bits"]
        info["container_mismatch"] = (info["container_bit_depth"] == 16 and
                                      info["quantization_grid_bits"] <= 8)
        # RV-04: 只有在读取阶段确认了"这份像素的 ICC"时才带过去；转换后的像素
        # 不贴原 profile。
        icc_bytes = meta.get("icc_profile")
        writer = media_io.write_tiff(tiff_path, arr, icc=icc_bytes)
        info["writer"] = writer
        info["written"] = True
        info["icc_source"] = meta.get("icc_source")
        if icc_bytes:
            info["icc"] = ("source" if media_io.embed_icc_in_tiff(tiff_path, icc_bytes)
                           else "embed-failed")
        if info["precision_loss"]:
            logger.warning("STAGE6_TIFF_PRECISION_LOSS",
                           f"Declared {info['requested_bit_depth']}-bit master but "
                           f"the working image is {info['actual_bit_depth']}-bit; "
                           "the master is written at the real precision and the "
                           "loss is recorded (upstream stage dropped to 8-bit)",
                           dtype=info["dtype"])
        if info["container_mismatch"]:
            logger.warning(
                "STAGE6_TIFF_CONTAINER_MISMATCH",
                "Master is stored in a 16-bit container but its sample values "
                f"sit on a {info['quantization_grid_bits']}-bit quantisation "
                "grid (heuristic: e.g. 8-bit data scaled by 257). The master "
                "is kept; this is a DIAGNOSTIC and must not be read as proof "
                "of the source's information depth (RV-14).")
        if info["icc"] == "none":
            logger.info("STAGE6_TIFF_NO_SOURCE_ICC",
                        "No source ICC in the working image; TIFF master "
                        "carries no profile rather than a guessed one")
        logger.info("STAGE6_TIFF_MASTER",
                    f"TIFF master written ({info['actual_bit_depth']}-bit, "
                    f"writer={writer}, icc={info['icc']})",
                    output=tiff_path)
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        logger.warning("STAGE6_TIFF_FAIL", f"TIFF master failed: {e}")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--params")
    parser.add_argument("--params-file")
    parser.add_argument("--config")
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()

    # BUG-057 support: --params-file fallback for long JSON
    if getattr(args, "params_file", None):
        with open(args.params_file, "r", encoding="utf-8") as f:
            params = json.load(f)
    else:
        params = json.loads(args.params) if args.params else {}
    log_dir = params.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, job_id=args.job_id)

    # NEW-V-06: 注册取消守卫（KeyboardInterrupt 继承 BaseException）。
    if cancel_guard is not None:
        cancel_guard.install_cancel_guard(stage=6, logger=logger)

    temp_dir = params.get("temp_dir", "./temp")
    job_id = args.job_id
    input_name = Path(args.input).name

    # Bug 11.21 fix: pre-check avifenc existence
    if not shutil.which("avifenc"):
        logger.error("STAGE6_AVIFENC_MISSING", "avifenc not found in PATH")
        return 80
    if media_io is None:
        logger.error("STAGE6_MEDIA_IO_MISSING",
                     "shared_modules/media_io.py unavailable; refusing to "
                     "encode with unverified parameters")
        return 80

    capability = media_io.detect_avifenc_capability()
    logger.info("STAGE6_AVIFENC_CAPABILITY",
                f"avifenc version={'.'.join(map(str, capability.version))} "
                f"depth_extension={capability.supports_depth_extension}")

    # FIX-L12-16: WSL2 redirect for temp_dir per v7.32 §3.10.10
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    _platform = params.get("platform", "auto")
    try:
        _redir_temp, _orig_temp, _, _ = wsl_redirect_path(
            temp_dir, no_wsl_redirect=no_wsl_redirect,
            target_platform=_platform
        )
        if _redir_temp != temp_dir:
            logger.info("STAGE6_WSL_REDIRECT_TEMP",
                        f"temp_dir: {temp_dir} -> {_redir_temp}")
            temp_dir = _redir_temp
            params["temp_dir"] = temp_dir
    except Exception:
        pass  # wsl_redirect_path may be unavailable on native Windows

    # FIX-P10-1: input search order step5 > step4 > step3 > step2 > step1
    input_source = None
    for pat in ["step5_postprocess*", "step4_face*", "step3_upscale*", "step2_ai_pre*", "step1_preprocess*"]:
        pattern = f"{job_id}_{safe_input_stem(input_name)}_{pat}"
        candidates = glob.glob(os.path.join(temp_dir, pattern))
        # NEW-V-04: 侧车文件（.status.json / .cache.json）也是 stepN_* 前缀，
        # 不过滤就会被当成"最新产物"选中，随后 Pillow 抛
        # UnidentifiedImageError。这里只保留真实图像扩展名。
        candidates = [
            p for p in candidates
            if Path(p).suffix.lower() in set(
                getattr(media_io, "IMAGE_EXTS", ()) or
                {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
            )
        ]
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            input_source = candidates[0]
            break

    # FIX-NEW-V-13 / R15: 与 stage4/stage5 同型——stage3 的真实产物在
    # <temp>/<job_id>_upscaled/<stem>_x<N>.tif。若前序 stage 的 stepN_*
    # 产物都不在（实测），没有这条回落就会 STAGE6_NO_INPUT 退 82。
    if not input_source:
        _up = os.path.join(temp_dir, f"{job_id}_upscaled")
        if os.path.isdir(_up):
            _c = glob.glob(os.path.join(
                _up, f"{safe_input_stem(input_name)}_x*"))
            _c = [p for p in _c if Path(p).suffix.lower()
                  in {".tif", ".tiff", ".png"}]
            if _c:
                _c.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                input_source = _c[0]

    if not input_source:
        # BUG-023 fix: require a processed intermediate
        logger.error("STAGE6_NO_INPUT", "No processed intermediate file found, refusing to encode raw input")
        return 82

    output_path = params.get("output_path", "")
    if not output_path:
        output_dir = params.get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir,
            f"{safe_input_stem(input_name)}_restored.avif")

    # FIX-R5-08: 与视频链 stage7（FIX-R5-07）同型——显式 output_path
    # 既不转绝对路径也不建父目录：相对路径按进程 cwd 解析，manifest 是
    # 持久产物，换 cwd 执行就落错地方甚至直接崩（父目录不存在）。
    # 统一 resolve + mkdir 父目录，cwd 从此无关。
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # WSL2 redirect
    # RV-13: **工作路径与发布路径必须分别保存**。ORIGINAL 在回拷成功后把
    # output_path 改成发布路径，随后 TIFF 的源路径又从这个已换过的变量派生
    # （取的是发布路径下的 tif，而母版其实写在工作路径），且回拷返回值没检查。
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    platform = params.get("platform", "auto")
    redirected_path, original_output_path, _, _ = wsl_redirect_path(
        output_path, no_wsl_redirect=no_wsl_redirect, target_platform=platform
    )
    work_output_path = redirected_path if original_output_path else output_path
    publish_output_path = original_output_path or output_path
    if original_output_path:
        logger.info("STAGE6_WSL_REDIRECT",
                   f"WSL redirect: {publish_output_path} -> {work_output_path}")

    logger.info("STAGE6_START", "Final photo encoding",
                input=input_source, output=publish_output_path)

    # RV-12: 事务式发布 —— 先写临时文件，校验通过后再原子替换到发布路径。
    # 这样"编码失败/中断"不会在最终路径留下半截 AVIF，也不会删掉上一轮的
    # 有效成品（守卫只删本轮临时产物）。
    # FIX-A3：临时名必须带 job 维度。
    # 原实现是 `work_output_path + ".partial"`，纯由**输出路径**派生。
    # 当多个 job 并发发布到同一输出路径（或同一 job 重跑而上一轮的
    # .partial 尚未清理）时，两个进程会**共用同一个临时名**：
    # 一个进程写了一半被另一个覆盖，回滚时也可能把对方的备份删掉。
    # 加 job_id 后各 job 的暂存/备份互不干扰；路径本身不参与发布，
    # 因此不影响正常单 job 行为的产物布局。
    _iso = f".{job_id}" if job_id else ""
    tmp_avif = f"{work_output_path}{_iso}.partial"
    status_path = publish_output_path + ".status.json"
    contract_path = publish_output_path + ".encode_contract.json"
    if cancel_guard is not None:
        cancel_guard.watch_outputs(
            publish_output_path,
            extra_paths=[tmp_avif],
            logger=logger, stage=6,
            delete_output=False,          # 不删上一次已发布的有效成品
            status_path=status_path)      # 取消记录写在发布路径旁

    # --- FIX-V04: 色彩契约（必需转换失败即阻断）---
    _stem = safe_input_stem(input_name)
    color = resolve_color_space(input_source, params, temp_dir, job_id, _stem,
                                logger)
    if color["status"] == COLOR_FAILED:
        logger.error("STAGE6_COLOR_BLOCKED",
                     "Colour contract could not be satisfied; not writing a "
                     "success artifact",
                     source_space=color["source_space"], reason=color["reason"])
        return 86
    working_path = color["working_path"]

    # --- FIX-V05: 编码输入适配（avifenc 不接受 TIFF）---
    try:
        adapter = media_io.prepare_encoder_input(working_path, temp_dir, logger)
    except media_io.MediaWriteError as e:
        logger.error("STAGE6_INPUT_ADAPTER_FAIL",
                     f"Could not produce an avifenc-acceptable input: {e}")
        return 82
    encode_input = adapter["path"]
    if Path(encode_input).suffix.lower() not in media_io.AVIFENC_SUPPORTED_INPUT_EXTS:
        logger.error("STAGE6_INPUT_FORMAT_UNSUPPORTED",
                     f"avifenc accepts only "
                     f"{sorted(media_io.AVIFENC_SUPPORTED_INPUT_EXTS)}; got "
                     f"{Path(encode_input).suffix}")
        return 86

    # --- FIX-V02: 高位深母版（真实精度，同样走临时文件）---
    tiff_info: Dict[str, Any] = {"written": False}
    work_tiff_path = ""
    publish_tiff_path = ""
    tmp_tiff = ""
    # RV-12: 本轮已经"发布"的产物集合；失败时要整批撤下，不能留下
    # "有母版、没 AVIF"这种半个提交。
    round_artifacts: List[str] = []
    if params.get("photo_bit_depth", 16) == 16:
        publish_tiff_path = publish_output_path.rsplit(".", 1)[0] + ".tif"
        work_tiff_path = work_output_path.rsplit(".", 1)[0] + ".tif"
        tmp_tiff = f"{work_tiff_path}{_iso}.partial"
        if cancel_guard is not None:
            cancel_guard.watch_outputs(
                publish_output_path,
                extra_paths=[tmp_avif, tmp_tiff],
                logger=logger, stage=6,
                delete_output=False, status_path=status_path)
        tiff_info = write_archival_master(working_path, tmp_tiff, params, logger)
        if tiff_info.get("written"):
            # R2-01：母版**先留在暂存路径**，不在这里覆盖旧母版。
            # 旧实现在此 `os.replace(tmp_tiff, work_tiff_path)` 直接盖掉上一代
            # 母版，随后编码失败时 `_fail` 又把这个路径删掉 —— 实测结果是
            # 旧 AVIF 还在、旧 TIFF 已丢失（A01）。移动动作推迟到提交阶段，
            # 且提交阶段有备份可回滚。
            tiff_info["staging_path"] = tmp_tiff
        if not tiff_info.get("written") and params.get("archival_tiff_required", False):
            logger.error("STAGE6_TIFF_REQUIRED_FAIL",
                         "Archival TIFF master is required but could not be "
                         "written", error=tiff_info.get("error"))
            _fail(logger, status_path, "tiff_master",
                  tiff_info.get("error") or "archival TIFF master missing",
                  paths=[tmp_avif] + round_artifacts)
            return 86

    # --- FIX-V01: 命令构造（不支持的组合在启动前失败）---
    archival_mode = params.get("archival_mode", False)
    use_rgb444 = archival_mode
    try:
        cmd, contract = media_io.build_avifenc_cmd(
            encode_input, tmp_avif, params, use_rgb444=use_rgb444,
            capability=capability, icc_path=color.get("icc_path"),
            cicp_override=color.get("cicp_override"))
    except media_io.UnsupportedAvifConfig as e:
        logger.error("STAGE6_AVIF_CONFIG_UNSUPPORTED", str(e))
        _fail(logger, status_path, "avif_config", str(e),
              paths=[tmp_avif] + round_artifacts)
        return 86

    # 实际编码契约：记录"实际会编出什么"，而不是"用户想要什么"
    contract.update({
        # RV-04: source_icc_bytes 是字节，不能进 JSON；长度单独记录
        "color": {k: (len(v) if k == "source_icc_bytes" and isinstance(v, (bytes, bytearray))
                      else v)
                  for k, v in color.items()
                  if k not in ("working_path", "source_icc_bytes")},
        "color_source_icc_bytes": len(color.get("source_icc_bytes") or b""),
        "encoder_input": adapter,
        "tiff_master": tiff_info,
        "output_path": publish_output_path,
        "work_output_path": work_output_path,
        "published": False,
    })
    logger.info("STAGE6_ENCODE_CONTRACT",
                f"depth_arg={contract['depth_arg']} "
                f"effective_depth={contract['effective_decode_depth']} "
                f"yuv={contract['yuv']} cicp={contract['cicp']} "
                f"lossless={contract['lossless']} "
                f"icc_attached={contract['icc_attached']}",
                input=encode_input)

    logger.info("STAGE6_ENCODE", f"Encoding AVIF (archival={use_rgb444})",
                cmd=" ".join(cmd))
    timeout = _calculate_avifenc_timeout(encode_input)
    # RV-12: 用 Popen 而不是 subprocess.run —— 取消/超时必须能真正终止编码器
    # 子进程并等它退出，否则会出现"stage 已退出、avifenc 还在写回"的孤儿写。
    ok, stderr, rc = _run_encoder(cmd, timeout, logger)
    contract["exit_code"] = None if ok else 84
    if not ok:
        logger.error("STAGE6_AVIF_FAIL", "avifenc encoding failed",
                     stderr=stderr[:2000], exit_code=rc)
        # FIX-H12-05: encoding failure must not be masked by another format
        # RV-12: 失败必须清掉本轮临时产物，并在发布路径旁留下 FAILED 记录，
        # 不能让"半截 AVIF"或"上一轮成品"被当成这次的成功结果。
        _fail(logger, status_path, "avif_encode",
              f"avifenc exited {rc}: {stderr[:500]}",
              paths=[tmp_avif] + round_artifacts)
        return 84
    if not os.path.exists(tmp_avif) or os.path.getsize(tmp_avif) == 0:
        # 退出码 0 也不代表成功：必须验证产物本身
        logger.error("STAGE6_AVIF_OUTPUT_MISSING",
                     "avifenc returned 0 but the output is missing/empty",
                     output=tmp_avif)
        _fail(logger, status_path, "avif_encode",
              "encoder produced an empty/missing file",
              paths=[tmp_avif] + round_artifacts)
        return 84

    # --- R2-13: 发布前读回真实文件头，核对色彩标签 -----------------------
    # 命令参数只是"意图"，编码器可能接受并改写（实测 identity 矩阵 0 → 6）。
    # 这里用 avifdec 独立回读实际头部，与契约逐项比对并记录；不一致时按
    # require_header_match 决定是否判失败，绝不默默把契约当事实。
    try:
        actual = media_io.read_avif_header(tmp_avif, capability=capability)
    except Exception as e:  # 回读失败本身要留痕，但不能因此丢掉已经编好的产物
        actual = {}
        logger.warning("STAGE6_HEADER_READBACK_FAIL",
                       f"Could not read back the encoded AVIF header: {e}",
                       output=tmp_avif)
    if not actual:
        # R3-09 / B07：回读失败/结果未知 —— 严格模式下**不能当作验收通过**。
        # 旧实现 `if actual:` 一跳而过，于是 require_header_match=True 时
        # 未知结果照样发布、rc=0（实测 B07）。
        logger.warning("STAGE6_HEADER_UNKNOWN",
                       "Could not determine the encoded file's real header; "
                       "the contract is NOT verified",
                       output=tmp_avif)
        contract["actual_header"] = None
        contract["header_matches_contract"] = None      # 未知 ≠ 一致
        contract["header_verification"] = "unknown"
        if params.get("require_header_match", False):
            _fail(logger, status_path, "header_verification_unknown",
                  "the encoded header could not be read back; strict mode "
                  "refuses to publish an unverified product",
                  paths=[tmp_avif])
            return 86
    if actual:
        contract["actual_header"] = actual
        contract["header_verification"] = "checked"
        declared_matrix = media_io.cicp_matrix(contract.get("cicp"))
        actual_matrix = actual.get("matrix")
        contract["header_matches_contract"] = (
            declared_matrix is not None
            and actual_matrix is not None
            and int(declared_matrix) == int(actual_matrix))
        if not contract["header_matches_contract"]:
            logger.warning(
                "STAGE6_HEADER_MISMATCH",
                "Encoder wrote different colour tags than the contract "
                "claims (command arguments are intent, the file header is "
                "fact)",
                declared_cicp=contract.get("cicp"),
                declared_matrix=declared_matrix,
                actual_matrix=actual_matrix)
            if params.get("require_header_match", False):
                _fail(logger, status_path, "header_verification",
                      f"encoded header matrix {actual_matrix} != contract "
                      f"{declared_matrix}",
                      paths=[tmp_avif] + round_artifacts)
                return 86

    # --- R2-01: 成套提交（备份旧代 → 逐个提交 → 任一步失败整批回滚）---
    # 连续几次 os.replace **不等于**整组原子：AVIF 换成了新代、TIFF 回拷失败
    # 时，磁盘上就是"新 AVIF + 旧 TIFF"的混合代（A02 实测）。
    # 这里用"提交前整体备份 + 失败整体恢复"的协议，保证最终只会看到
    # 完整旧代或完整新代。
    commit_targets: List[str] = [work_output_path]
    if tiff_info.get("written"):
        commit_targets.append(work_tiff_path)
    if original_output_path:
        commit_targets.append(publish_output_path)
        if tiff_info.get("written"):
            commit_targets.append(publish_tiff_path)
    # R3-03 / B04：**契约和状态也属于这一代**。
    # 旧实现只快照媒体文件，于是状态写失败回滚后 —— 媒体恢复了旧代，
    # 但 encode_contract 仍是新代且 published=true（实测 B04）。
    # 契约是下一次 resume/验收的依据，与媒体不同代会让下游读到自相矛盾的记录。
    commit_targets.append(contract_path)
    commit_targets.append(status_path)

    # --- A4 方案 1：先收拾上一次未完成的事务，再开始这一次 ---
    # 提交协议是"备份 + 逐个 os.replace + 失败整批回滚"，整组**不是**原子的。
    # PublishAbort 与取消信号都能回滚，但 kill -9 / 断电时 finally 和取消钩子
    # 都跑不到 —— 磁盘上可能留下混合代（新 AVIF + 旧 TIFF）加一堆没有归属
    # 信息的 `.pubbak`。判断层（read_journal + validate_journal_paths +
    # classify_recovery）与 kill -9 演练早在第七轮就绪，缺的只是这个调用点。
    #
    # 放在这里（而不是进程启动时）的理由：此刻才确定这次事务要动哪些文件，
    # 可以用作 journal 路径归属校验的授权根目录。
    _prior = _recover_prior_publish(work_output_path, commit_targets, logger)
    if _prior.get("status") == "needs_human":
        # 状态无法判定 —— 不猜、不动，让人工介入。此时动任何文件都可能
        # 把一次可恢复的残留变成不可恢复的损失。
        logger.error("STAGE6_PRIOR_PUBLISH_UNRESOLVED",
                     "A previous publish transaction could not be resolved "
                     "automatically; refusing to start a new one",
                     report=_prior.get("report"))
        _fail(logger, status_path, "prior_publish_recovery",
              "unresolved prior publish transaction; inspect the journal "
              "and .pubbak manually",
              paths=[tmp_avif] + round_artifacts)
        return 82
    if _prior.get("status") == "rolled_back" and logger:
        logger.warning("STAGE6_PRIOR_PUBLISH_RECOVERED",
                       "Rolled a previous uncommitted publish back to the "
                       "old generation before starting this one",
                       report=_prior.get("report"))

    # --- FIX-2026-10-20（G6 实跑的教训）：发布前的色彩保真守卫 -----------
    #
    # 为什么加：8-bit 输入的照片链曾长期产出**纯黑成品**而 rc=0 —— 契约、
    # CICP、avifdec 头部核验全部正常，只有像素是黑的。那次是靠人工拿
    # avifdec 解出来算 mean 才发现的。这里把这一步变成例行守卫，让
    # "整体崩坏"（黑/白/内容丢失）不可能再带着 rc=0 流出去。
    #
    # 判据与视频链 stage7 **同一份实现**（shared_modules/color_probe.py），
    # 这里只决定"超了阈值要不要失败"。阈值默认 60.0，实测标定：
    #   纯黑成品 delta≈116 → 拦；ProPhoto→sRGB delta≈26.7 → 放行；
    #   普通 sRGB 正常处理 delta≈1 → 放行。
    # 即它专门抓"整体崩坏"，不抓细微色偏（后者需针对素材标定更严的值）。
    try:
        from color_probe import (COLOR_DRIFT_DEFAULT_THRESHOLD,
                                 check_color_drift)
        _thr = float(params.get("color_drift_threshold",
                                COLOR_DRIFT_DEFAULT_THRESHOLD))
        _enforce = bool(params.get("color_drift_enforce", False))
        _drift = check_color_drift(args.input, tmp_avif, threshold=_thr,
                                   enforce=_enforce, logger=logger)
        _fields = {k: v for k, v in _drift.items() if k != "status"}
        if _drift["status"] == "drift":
            _msg = ("Colour drift between original input and encoded output "
                    "exceeds threshold")
            if _enforce:
                logger.error("STAGE6_COLOR_DRIFT", _msg, **_fields)
                # tmp_avif / tmp_tiff 是本次 attempt 的暂存，round_artifacts 是
                # 本轮其余临时产物，都要丢掉，否则 .partial 会留在 output 目录。
                # 既有成品一个字节都不动（PUB-01 语义），与视频链 stage7 的
                # _discard_attempt_tmp 对称。tmp_tiff 未写时为 ""，_fail 会跳过。
                _fail(logger, status_path, "color_drift",
                      f"delta={_drift['delta']:.2f} > {_thr}",
                      paths=[tmp_avif, tmp_tiff] + round_artifacts)
                return 86
            logger.warning("STAGE6_COLOR_DRIFT", _msg, **_fields)
        elif _drift["status"] == "ok":
            logger.info("STAGE6_COLOR_OK",
                        "Colour distribution within threshold", **_fields)
        else:
            # 探测失败 → 如实记"未校验"，不假装通过，也不阻断。
            logger.warning("STAGE6_COLOR_SKIPPED",
                           "Could not probe RGB means on one side; colour "
                           "fidelity left unchecked")
    except Exception as _ce:
        logger.warning("STAGE6_COLOR_DRIFT_ERROR", str(_ce))

    # FIX-A3：同 tmp_avif，备份目录也带 job 维度（见上方说明）。
    backup_dir = f"{work_output_path}{_iso}.pubbak"
    snapshot = _snapshot_for_rollback(commit_targets, backup_dir, logger)
    # R3-03 / B14：进入发布临界区前把"成套回滚"挂到取消路径上。
    # 取消处理器最后是 os._exit，普通 try/finally 根本跑不到，
    # 所以必须注册为取消钩子；提交成功后由 release_outputs() 清除。
    if cancel_guard is not None:
        cancel_guard.add_cancel_hook(
            lambda: _rollback_publish(snapshot, logger))
    # R3-03 / B03：任一目标备份失败 → **在改动任何旧产物之前停止**。
    if _snapshot_has_failure(snapshot):
        logger.error("STAGE6_PUBLISH_BACKUP_INCOMPLETE",
                     "Cannot back up every artifact of the current generation; "
                     "refusing to start the commit (a later failure could "
                     "destroy the previous generation)",
                     failed=[p for p, e in snapshot.items()
                             if e.get("state") == BACKUP_FAILED])
        _fail(logger, status_path, "publish_backup",
              "incomplete backup; commit not started", paths=[])
        return 82

    # 持久 journal（第七轮）：备份已完整、即将改动目标 —— 此刻把"这次事务
    # 要动哪些文件、旧内容是什么、备份在哪"落盘。
    #
    # 为什么必须落盘：进程被硬终止后，内存 snapshot 与取消钩子都没了，
    # 只剩一堆 `.pubbak`。没有 journal 时，谁也说不清这些备份属于哪次
    # 事务、事务走到哪一步、还完不完整 —— 只能靠"发现备份 + 摘要不符"
    # 去**猜**，而那可能是一次早已成功的旧尝试。
    _journal_targets = list(commit_targets)
    _journal_records: Dict[str, Any] = {}
    for _t in _journal_targets:
        _entry = snapshot.get(_t) or {}
        _existed = os.path.exists(_t)
        _bk = _entry.get("backup")
        _journal_records[_t] = {
            "existed": _existed,
            "old_digest": file_digest(_t) if _existed else None,
            "backup": _bk,
            "backup_digest": file_digest(_bk) if _bk else None,
            "state": _entry.get("state"),
        }
    _journal_path = journal_path_for(work_output_path)
    _journal_base = {
        "schema": JOURNAL_SCHEMA,
        "job_id": params.get("job_id") or "",
        "attempt_id": f"{os.getpid()}-{int(time.time() * 1000)}",
        "stage": 6,
        "targets": _journal_targets,
        "records": _journal_records,
        "roots": [os.path.dirname(os.path.abspath(_t)) for _t in _journal_targets],
    }
    if not write_journal(_journal_path,
                         dict(_journal_base, phase=PHASE_PREPARED)):
        # journal 写不下去 → **不改任何目标**（否则硬终止后无法判断状态）
        logger.error("STAGE6_JOURNAL_WRITE_FAIL",
                     "Cannot persist the publish journal; refusing to modify "
                     "any target (a crash would leave an unidentifiable state)",
                     journal=_journal_path)
        _fail(logger, status_path, "publish_journal",
              "journal not persisted; commit not started", paths=[],
              preserve_valid_status=True)
        return 82
    logger.info("STAGE6_JOURNAL_PREPARED",
                "Publish journal persisted before touching any target",
                journal=_journal_path,
                attempt_id=_journal_base["attempt_id"])

    # PUB-04（第七轮）：提交段必须持有**全部目标路径的互斥锁**。
    # 逐路径加锁（不是集合摘要），任何重叠目标集合都会互斥；
    # 锁从"改任何目标之前"开始，覆盖备份、替换、侧车写入与回滚。
    try:
        _lock_cm = acquire_target_locks(
            commit_targets,
            timeout=float(params.get("publish_lock_timeout", 60)),
            logger=logger)
        _lock_cm.__enter__()
    except TargetLockTimeout as _lt:
        logger.error("STAGE6_PUBLISH_LOCK_TIMEOUT",
                     "Could not acquire publish locks; nothing was modified",
                     targets=_lt.targets, timeout=_lt.timeout)
        _fail(logger, status_path, "publish_lock", str(_lt), paths=[],
              preserve_valid_status=True)
        return 82

    try:
        try:
            os.replace(tmp_avif, work_output_path)
            contract["published"] = True
            if tiff_info.get("written") and tmp_tiff:
                os.replace(tmp_tiff, work_tiff_path)
                tiff_info["master_path"] = work_tiff_path
            if original_output_path:
                # RV-13: 源路径一律用工作路径（work_*）
                if not wsl_copy_back(work_output_path, publish_output_path):
                    raise PublishAbort("wsl_copyback",
                                       f"copy back failed: {work_output_path}")
                logger.info("STAGE6_WSL_COPYBACK",
                            f"Copied result back to {publish_output_path}")
                if tiff_info.get("written") and work_tiff_path:
                    if not wsl_copy_back(work_tiff_path, publish_tiff_path):
                        raise PublishAbort(
                            "wsl_copyback_tiff",
                            f"copy back failed: {work_tiff_path}")
                    logger.info("STAGE6_WSL_COPYBACK_TIFF",
                                f"Copied master back to {publish_tiff_path}")
            # 契约与状态必须**写成功**才算提交完成（A12：侧车写失败仍 rc=0 是错的）
            if not _write_json_sidecar(contract_path, contract):
                raise PublishAbort("contract",
                                   "could not write the encode contract sidecar")
            if not _write_json_sidecar(status_path, {
                "status": "SUCCESS",
                "stage": 6,
                "skipped_operations": _color_skipped(color),
                "output_path": publish_output_path,
                "published": True,
            }):
                raise PublishAbort("status",
                                   "could not write the SUCCESS status sidecar")
        except PublishAbort as abort:
            logger.error("STAGE6_PUBLISH_ABORT",
                         f"Publish aborted at {abort.operation}: {abort.reason}; "
                         "rolling the whole generation back",
                         operation=abort.operation)
            # PUB-02/03（第七轮）：按**可核验的回滚结果**分流，不再无条件
            # 宣布"Previous generation restored"；也不再把当前有效代的
            # status 覆写成 FAILED（旧代若完好，它仍然是有效成品）。
            _rb = _rollback_publish(snapshot, logger)
            if _rb["fully_restored"]:
                logger.warning("STAGE6_PUBLISH_ROLLBACK",
                               "Previous generation restored and verified; no "
                               "mixed-generation output is left behind",
                               restored=_rb["restored"], removed=_rb["removed"])
                # 旧代仍有效 → 固定 status 保持原样（可能是 SUCCESS），
                # 本次失败只记到 attempt 级 sidecar。
                _fail(logger, status_path, abort.operation, abort.reason,
                      paths=[], preserve_valid_status=True)
            else:
                logger.error("STAGE6_RECOVERY_REQUIRED",
                             "Rollback did NOT fully restore the previous "
                             "generation; manual recovery is required before the "
                             "next run may treat these paths as valid",
                             failed=_rb["failed"], impossible=_rb["impossible"],
                             restored=_rb["restored"])
                _fail(logger, status_path, abort.operation, abort.reason,
                      paths=[], recovery_required=True,
                      rollback_detail=_rb)
            return 82
        except OSError as e:
            logger.error("STAGE6_PUBLISH_FAIL", f"Could not publish the output: {e}")
            _rb = _rollback_publish(snapshot, logger)
            if _rb["fully_restored"]:
                _fail(logger, status_path, "publish", str(e), paths=[],
                      preserve_valid_status=True)
            else:
                logger.error("STAGE6_RECOVERY_REQUIRED",
                             "Rollback after publish failure did NOT fully "
                             "restore the previous generation",
                             failed=_rb["failed"], impossible=_rb["impossible"])
                _fail(logger, status_path, "publish", str(e), paths=[],
                      recovery_required=True, rollback_detail=_rb)
            # FIX-2026-10-20：恢复信息写不下去 = 下次启动不知道需要人工恢复。
            # 这一行原来是静默的（返回值没人看）。
            if not write_journal(_journal_path,
                                 dict(_journal_base,
                                      phase=PHASE_RECOVERY_REQUIRED,
                                      rollback=_rb)):
                logger.error("STAGE6_JOURNAL_RECOVERY_FAIL",
                             "Could not record RECOVERY_REQUIRED; the next "
                             "run will not know manual recovery is needed",
                             journal=_journal_path)
            return 82

        # 已发布：守卫不再删除任何产物
        if cancel_guard is not None:
            cancel_guard.release_outputs()

        # journal 推进到 COMMITTED（含新代摘要，供下次判断"是否已提交"）
        _done_records = {}
        for _t in _journal_targets:
            _r = dict(_journal_records.get(_t) or {})
            _r["new_digest"] = file_digest(_t)
            _done_records[_t] = _r
        # FIX-2026-10-20：journal 写失败**不能静默**。
        # 此刻媒体已经发布（上面的 os.replace 已完成），若 journal 仍停在
        # PREPARED / COMMITTING，下次启动会认为"未提交"——可能重复发布，
        # 或把已发布的成品当成发布残留处理。"发布了却记不下状态"
        # 比"没发布"更危险，所以这里必须让本次运行失败。
        if not write_journal(_journal_path,
                             dict(_journal_base, phase=PHASE_COMMITTED,
                                  records=_done_records)):
            logger.error("STAGE6_JOURNAL_COMMIT_FAIL",
                         "Output was promoted but the COMMITTED phase could "
                         "not be recorded; the next run may treat this "
                         "publish as uncommitted and re-publish",
                         journal=_journal_path)
            return 82

        logger.info("STAGE6_DONE", "Photo encoding complete", output=publish_output_path)
        return 0
    finally:
        try:
            _lock_cm.__exit__(None, None, None)
        except Exception:
            pass


if __name__ == "__main__":
    # FIX-P11-5: main() must return explicit int exit code per v7.32 §3.5
    rc = main()
    if rc is None:
        rc = 86  # validation failure — main() must always return explicit exit code
    sys.exit(rc)
