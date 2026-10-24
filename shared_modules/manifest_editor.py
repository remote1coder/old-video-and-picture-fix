#!/usr/bin/env python3
"""
Manifest Editor (Optional)
Interactive or command-line tool to edit manifest jobs.

v7.32 Compliance:
- CLI subcommands for batch operations
- Interactive TUI using rich for visual job review
- Nested dict override support (key.subkey syntax)
- PRIORITY setting support
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from manifest_loader import load_manifest, save_manifest, update_job_status, _update_meta_counts  # noqa: E402


def _set_nested(d: dict, key_path: str, value):
    """Set nested dict value by dot-separated key path."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _parse_value(value: str):
    """Parse user input string to Python value."""
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    elif value.lower() == "none":
        return None
    else:
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value


def _find_config_upwards(manifest_path: str, max_levels: int = 5) -> str:
    """Search for config.yaml upwards from manifest directory."""
    p = Path(manifest_path).resolve().parent
    for _ in range(max_levels):
        cand = p / "config.yaml"
        if cand.exists():
            return str(cand)
        if p.parent == p:
            break
        p = p.parent
    return ""


def list_jobs(manifest_path: str, status_filter: Optional[str] = None,
              show_params: bool = False, low_confidence: bool = False):
    """List jobs in manifest.

    low_confidence: 只列会被 main.py 的 0.80 门禁拦下的 job
        （needs_confirmation=True 或 confidence<0.80），方便确认前先看清。
    """
    manifest = load_manifest(manifest_path)
    jobs = manifest.get("jobs", [])

    print(f"{'Job ID':<12} {'Status':<10} {'Class':<14} {'Priority':<8} {'Confidence':<10} {'Path'}")
    print("-" * 100)

    shown = 0
    for job in jobs:
        if status_filter and job.get("status") != status_filter:
            continue
        if low_confidence:
            c = job.get("confidence", 0)
            if not (job.get("needs_confirmation") or
                    (isinstance(c, (int, float)) and c < 0.80)):
                continue
        shown += 1
        print(f"{job.get('job_id', ''):<12} "
              f"{job.get('status', ''):<10} "
              f"{job.get('detected_class', ''):<14} "
              f"{job.get('priority', 0):<8} "
              f"{job.get('confidence', 0):.2f}       "
              f"{Path(job.get('input_path', '')).name}")
        if show_params:
            rec = job.get("recommended_params", {})
            over = job.get("user_overrides", {})
            if rec:
                print(f"    recommended: {rec}")
            if over:
                print(f"    overrides: {over}")

    if low_confidence:
        print("-" * 100)
        print(f"{shown} job(s) below the 0.80 gate. "
              f"Use 'confirm <job_id>' or 'confirm --all-low-confidence' "
              f"after reviewing them.")


def set_job_status(manifest_path: str, job_id: str, status: str):
    """Set job status."""
    manifest = load_manifest(manifest_path)
    update_job_status(manifest, job_id, status)
    save_manifest(manifest, manifest_path)
    print(f"Job {job_id} status set to {status}")


def set_user_override(manifest_path: str, job_id: str,
                      key: str, value: str):
    """Set user override parameter for a job.

    Supports nested keys via dot notation (e.g., 'avif_params.speed').
    Values are parsed as int, float, bool, or string.
    """
    manifest = load_manifest(manifest_path)
    for job in manifest.get("jobs", []):
        if job.get("job_id") == job_id:
            overrides = job.setdefault("user_overrides", {})
            # FIX-P4-4: Reuse module-level _parse_value(), eliminate code duplication
            parsed_value = _parse_value(value)
            _set_nested(overrides, key, parsed_value)
            # FIX-P4-8: update timestamp on modification
            job["updated_at"] = datetime.now().astimezone().isoformat()
            print(f"Set {key}={parsed_value} for job {job_id}")
            break
    save_manifest(manifest, manifest_path)


def set_priority(manifest_path: str, job_id: str, priority: int):
    """Set job priority."""
    manifest = load_manifest(manifest_path)
    for job in manifest.get("jobs", []):
        if job.get("job_id") == job_id:
            job["priority"] = priority
            # FIX-P4-8: update timestamp on modification
            job["updated_at"] = datetime.now().astimezone().isoformat()
            print(f"Set priority={priority} for job {job_id}")
            break
    save_manifest(manifest, manifest_path)


def confirm_job(manifest_path: str, job_id: str = None, *,
                confidence: float = None,
                all_low_confidence: bool = False) -> int:
    """人工确认低置信度分类 —— 清掉 needs_confirmation 让 job 放行。

    背景（2026-10-24 修）：`needs_confirmation` **只有分类器会写**，
    全库没有任何地方清除它。于是编辑器的 `c`（手动指定类别）即使把
    `confidence` 设成 1.0，只要 `needs_confirmation` 还是 True，
    main.py 的门禁 `if needs_confirmation or conf < 0.80` 仍然命中，
    非交互批处理照样 SKIP —— **人工确认这条路径实际上是断的**。

    本函数补上"清除"这一环：清 needs_confirmation、可选抬 confidence、
    记 manual_confirmed / confirmed_at 便于追溯。

    Args:
        manifest_path: manifest.yaml 路径
        job_id: 确认单个 job；None 时需配合 all_low_confidence
        confidence: 可选，抬到的置信度（0-1）。None 表示保持原值
        all_low_confidence: 一次确认所有低置信度 job（几百上千素材用这个）

    Returns:
        确认的 job 数。
    """
    from datetime import datetime as _dt
    manifest = load_manifest(manifest_path)
    if not manifest:
        return 0
    jobs = manifest.get("jobs", [])
    now = _dt.now().astimezone().isoformat()

    targets = []
    if all_low_confidence:
        for j in jobs:
            c = j.get("confidence", 0.0)
            if j.get("needs_confirmation") or (
                    isinstance(c, (int, float)) and c < 0.80):
                targets.append(j)
    elif job_id:
        for j in jobs:
            if j.get("job_id") == job_id:
                targets.append(j)
                break
        if not targets:
            print(f"[red]Job not found: {job_id}[/red]")
            return 0
    else:
        print("[red]Specify job_id or --all-low-confidence[/red]")
        return 0

    for j in targets:
        j["needs_confirmation"] = False
        # FIX-2026-10-24：光清 needs_confirmation **不够**。
        # main.py 的门禁是两个条件的**或**：
        #     if needs_confirmation or confidence < 0.80:
        # 所以 0.55 的 job 清掉 flag 后仍会被 `confidence < 0.80` 拦住
        # （实测确认过 6 个 job，list --low-confidence 仍显示 6 个）。
        # 人工确认 = 人的把握，因此把置信度至少抬到过线值。
        # 用户可用 --confidence 覆盖成任意值。
        _floor = 0.80
        if confidence is not None:
            try:
                _floor = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                pass
        try:
            j["confidence"] = max(float(j.get("confidence") or 0.0), _floor)
        except (TypeError, ValueError):
            j["confidence"] = _floor
        j["manual_confirmed"] = True
        j["confirmed_at"] = now
        j["updated_at"] = now

    _update_meta_counts(manifest)
    save_manifest(manifest, manifest_path)
    print(f"[green]Confirmed {len(targets)} job(s)[/green]")
    for j in targets[:20]:
        print("  %s  %s (conf=%s)" % (
            j.get("job_id"), j.get("detected_class"), j.get("confidence")))
    if len(targets) > 20:
        print("  ... and %d more" % (len(targets) - 20))
    return len(targets)


def interactive_tui(manifest_path: str):
    """Interactive TUI for manifest review using rich.
    
    Workflow: y (accept) / e (edit) / s (skip) / p (priority) / q (quit)
    """
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
    except ImportError:
        print("rich library not installed. Falling back to basic interactive mode.")
        _interactive_basic(manifest_path)
        return

    manifest = load_manifest(manifest_path)
    jobs = manifest.get("jobs", [])
    console = Console()
    
    console.print(Panel(f"Manifest Editor: {manifest_path}\n"
                        f"Total jobs: {len(jobs)}\n"
                        f"Commands: [y]=accept [e]=edit [s]=skip [p]=priority [c]=class [q]=quit",
                        title="Interactive Review"))
    
    for i, job in enumerate(jobs):
        # FIX-E8: include SKIP jobs so users can review and re-queue them per
        # v7.32 §3.4.3.10 (the TUI is the spec-defined core review entry point;
        # pressing 'y' on a SKIP job resets it to PENDING via existing logic).
        if job.get("status") not in ("PENDING", "FAILED", "SKIP"):
            continue
        
        table = Table(title=f"Job {i+1}/{len(jobs)}: {job.get('job_id')}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Input", str(job.get("input_path")))
        table.add_row("Class", str(job.get("detected_class")))
        table.add_row("Confidence", f"{job.get('confidence', 0):.2f}")
        table.add_row("Status", str(job.get("status")))
        table.add_row("Priority", str(job.get("priority", 0)))
        table.add_row("AI Banned", str(job.get("ai_upscale_banned", False)))
        # FIX-P4-13: Show skip_reason per v7.32 §3.4.3
        if job.get("skip_reason"):
            table.add_row("Skip Reason", str(job.get("skip_reason")))
        
        rec = job.get("recommended_params", {})
        if rec:
            table.add_row("Params", str(rec)[:80])
        
        console.print(table)
        
        while True:
            choice = console.input("[y/e/s/p/c/q] ? ").strip().lower()
            if choice == "y":
                job["status"] = "PENDING"
                _update_meta_counts(manifest)
                break
            elif choice == "e":
                key = console.input("Parameter key (dot notation supported): ").strip()
                val = console.input("Value: ").strip()
                overrides = job.setdefault("user_overrides", {})
                parsed_value = _parse_value(val)
                _set_nested(overrides, key, parsed_value)
                # FIX-P4-2: edit in-memory only; save at loop end
                job["updated_at"] = datetime.now().astimezone().isoformat()
                console.print(f"Set {key}={parsed_value} for job {job['job_id']}")
                break
            elif choice == "s":
                job["status"] = "SKIP"
                _update_meta_counts(manifest)
                break
            elif choice == "p":
                # FIX-P4-17: Default priority=1 per v7.32 §3.4.4
                # ORIGINAL: p = int(console.input("Priority (int): ").strip())
                try:
                    raw = console.input("Priority (default=1): ").strip()
                    job["priority"] = int(raw) if raw else 1
                except ValueError:
                    job["priority"] = 1  # fallback to spec default
                job["updated_at"] = datetime.now().astimezone().isoformat()
                console.print(f"Priority set to {job['priority']} for job {job['job_id']}")
                break
            elif choice == "c":
                new_class = console.input(
                    "New class (ANALOG_TAPE/DV_DIGITAL/HDV_AVCHD/SCAN_PRINT/DIGITAL_PHOTO): "
                ).strip().upper()
                valid_classes = ("ANALOG_TAPE", "DV_DIGITAL", "HDV_AVCHD", "SCAN_PRINT", "DIGITAL_PHOTO")
                if new_class in valid_classes:
                    old_class = job.get("detected_class", "UNKNOWN")
                    job["detected_class"] = new_class
                    job["confidence"] = 1.0  # FIX-P4-5: manual class = certainty
                    # FIX-2026-10-24：手动指定类别 = 已人工确认，必须清
                    # needs_confirmation。否则 main.py 的门禁
                    # `needs_confirmation or conf<0.80` 仍命中，
                    # 非交互批处理照样 SKIP（实测该路径此前是断的）。
                    job["needs_confirmation"] = False
                    job["manual_confirmed"] = True

                    config_path = manifest.get("meta", {}).get("config_path")
                    if not config_path:
                        config_path = _find_config_upwards(manifest_path)
                    if not Path(config_path).exists():
                        console.print(f"[red]Config not found: {config_path}[/red]")
                        break
                    
                    from manifest_generate import load_config, map_class_to_path_params, _compute_photo_ai_banned
                    config = load_config(config_path)
                    
                    job["recommended_params"] = map_class_to_path_params(new_class, config)
                    
                    audio_override = config.get("audio_override", {})
                    type_key_map = {
                        "ANALOG_TAPE": "analog_tape",
                        "DV_DIGITAL": "dv_digital",
                        "HDV_AVCHD": "hdv_avchd",
                    }
                    type_key = type_key_map.get(new_class)
                    if type_key and type_key in audio_override:
                        job["recommended_params"]["audio_override"] = dict(audio_override[type_key])
                    
                    orig_w = job.get("original_width", 0)
                    orig_h = job.get("original_height", 0)
                    # FIX-P4-10/P4-12: Use _compute_photo_ai_banned for photo classes
                    # ORIGINAL: job["ai_upscale_banned"] = (orig_w > 3840 or orig_h > 2160)
                    if new_class in ("SCAN_PRINT", "DIGITAL_PHOTO"):
                        job["ai_upscale_banned"] = _compute_photo_ai_banned(orig_w, orig_h)
                    elif new_class == "HDV_AVCHD":
                        job["ai_upscale_banned"] = (orig_h >= 1080 or orig_w >= 1920)
                    elif new_class == "ANALOG_TAPE":
                        job["ai_upscale_banned"] = (orig_h > 1080 or orig_w > 1920)
                    else:
                        # FIX-P4-2: DV_DIGITAL etc. must reset ai_upscale_banned to False
                        job["ai_upscale_banned"] = False

                    job["updated_at"] = datetime.now().astimezone().isoformat()
                    console.print(f"[green]Class changed from {old_class} to {new_class}[/green]")
                    console.print(f"[yellow]ai_upscale_banned = {job['ai_upscale_banned']}[/yellow]")
                    # FIX-E11: re-derive field_order/needs_ivtc/is_psf/field_segments
                    # from the input file so they match the new class (e.g.
                    # DV_DIGITAL => needs_ivtc recomputed). Reuses
                    # manifest_generate.classify_item but keeps the user's chosen
                    # detected_class/confidence. Falls back to a prompt on failure.
                    try:
                        from manifest_generate import classify_item
                        ptype = "photo" if new_class in ("SCAN_PRINT", "DIGITAL_PHOTO") else "video"
                        reclass = classify_item(job.get("input_path", ""), ptype)
                        job["field_order"] = reclass.get("field_order", job.get("field_order", "unknown"))
                        job["needs_ivtc"] = reclass.get("needs_ivtc", job.get("needs_ivtc", False))
                        job["is_psf"] = reclass.get("is_psf", job.get("is_psf", False))
                        job["field_segments"] = reclass.get("field_segments", job.get("field_segments", []))
                        console.print(f"[cyan]field_order={job['field_order']} needs_ivtc={job['needs_ivtc']} is_psf={job['is_psf']}[/cyan]")
                    except Exception as ce:
                        console.print(f"[yellow]Could not auto-refresh field_order/needs_ivtc/is_psf ({ce}); please verify manually.[/yellow]")
                else:
                    console.print("[red]Invalid class[/red]")
                break
            elif choice == "q":
                _update_meta_counts(manifest)
                save_manifest(manifest, manifest_path)
                console.print("Manifest saved. Quitting.")
                return
    
    _update_meta_counts(manifest)
    save_manifest(manifest, manifest_path)
    console.print("All jobs reviewed. Manifest saved.")


def _interactive_basic(manifest_path: str):
    """Basic interactive mode without rich."""
    manifest = load_manifest(manifest_path)
    jobs = manifest.get("jobs", [])
    
    print(f"Manifest Editor: {manifest_path}")
    print(f"Total jobs: {len(jobs)}")
    print("Commands: y=accept e=edit s=skip p=priority c=class q=quit")
    
    for i, job in enumerate(jobs):
        # FIX-E8: include SKIP jobs so users can review and re-queue them per
        # v7.32 §3.4.3.10 (the TUI is the spec-defined core review entry point;
        # pressing 'y' on a SKIP job resets it to PENDING via existing logic).
        if job.get("status") not in ("PENDING", "FAILED", "SKIP"):
            continue
        
        print(f"\n--- Job {i+1}/{len(jobs)}: {job.get('job_id')} ---")
        print(f"  Input: {job.get('input_path')}")
        print(f"  Class: {job.get('detected_class')} (confidence: {job.get('confidence', 0):.2f})")
        print(f"  Status: {job.get('status')} | Priority: {job.get('priority', 0)}")
        # FIX-P4-13: Show skip_reason per v7.32 §3.4.3
        if job.get("skip_reason"):
            print(f"  Skip Reason: {job.get('skip_reason')}")
        
        while True:
            choice = input("[y/e/s/p/c/q] ? ").strip().lower()
            if choice == "y":
                job["status"] = "PENDING"
                _update_meta_counts(manifest)
                break
            elif choice == "e":
                key = input("Parameter key: ").strip()
                val = input("Value: ").strip()
                overrides = job.setdefault("user_overrides", {})
                parsed_value = _parse_value(val)
                _set_nested(overrides, key, parsed_value)
                # FIX-P4-2: edit in-memory only; save at loop end
                job["updated_at"] = datetime.now().astimezone().isoformat()
                print(f"Set {key}={parsed_value} for job {job['job_id']}")
                break
            elif choice == "s":
                job["status"] = "SKIP"
                _update_meta_counts(manifest)
                break
            elif choice == "p":
                # FIX-P4-17: Default priority=1 per v7.32 §3.4.4
                # ORIGINAL: job["priority"] = int(input("Priority: ").strip())
                try:
                    raw = input("Priority (default=1): ").strip()
                    job["priority"] = int(raw) if raw else 1
                except ValueError:
                    job["priority"] = 1  # fallback to spec default
                job["updated_at"] = datetime.now().astimezone().isoformat()
                print(f"Priority set to {job['priority']}")
                break
            elif choice == "c":
                new_class = input(
                    "New class (ANALOG_TAPE/DV_DIGITAL/HDV_AVCHD/SCAN_PRINT/DIGITAL_PHOTO): "
                ).strip().upper()
                valid_classes = ("ANALOG_TAPE", "DV_DIGITAL", "HDV_AVCHD", "SCAN_PRINT", "DIGITAL_PHOTO")
                if new_class in valid_classes:
                    old_class = job.get("detected_class", "UNKNOWN")
                    job["detected_class"] = new_class
                    job["confidence"] = 1.0  # FIX-P4-5: manual class = certainty
                    # FIX-2026-10-24：手动指定类别 = 已人工确认，必须清
                    # needs_confirmation。否则 main.py 的门禁
                    # `needs_confirmation or conf<0.80` 仍命中，
                    # 非交互批处理照样 SKIP（实测该路径此前是断的）。
                    job["needs_confirmation"] = False
                    job["manual_confirmed"] = True

                    config_path = manifest.get("meta", {}).get("config_path")
                    if not config_path:
                        config_path = _find_config_upwards(manifest_path)
                    if not Path(config_path).exists():
                        print(f"Config not found: {config_path}")
                        break
                    
                    from manifest_generate import load_config, map_class_to_path_params, _compute_photo_ai_banned
                    config = load_config(config_path)
                    
                    job["recommended_params"] = map_class_to_path_params(new_class, config)
                    
                    audio_override = config.get("audio_override", {})
                    type_key_map = {
                        "ANALOG_TAPE": "analog_tape",
                        "DV_DIGITAL": "dv_digital",
                        "HDV_AVCHD": "hdv_avchd",
                    }
                    type_key = type_key_map.get(new_class)
                    if type_key and type_key in audio_override:
                        job["recommended_params"]["audio_override"] = dict(audio_override[type_key])
                    
                    orig_w = job.get("original_width", 0)
                    orig_h = job.get("original_height", 0)
                    # FIX-P4-10/P4-12: Use _compute_photo_ai_banned for photo classes
                    # ORIGINAL: job["ai_upscale_banned"] = (orig_w > 3840 or orig_h > 2160)
                    if new_class in ("SCAN_PRINT", "DIGITAL_PHOTO"):
                        job["ai_upscale_banned"] = _compute_photo_ai_banned(orig_w, orig_h)
                    elif new_class == "HDV_AVCHD":
                        job["ai_upscale_banned"] = (orig_h >= 1080 or orig_w >= 1920)
                    elif new_class == "ANALOG_TAPE":
                        job["ai_upscale_banned"] = (orig_h > 1080 or orig_w > 1920)
                    else:
                        # FIX-P4-2: DV_DIGITAL etc. must reset ai_upscale_banned to False
                        job["ai_upscale_banned"] = False

                    job["updated_at"] = datetime.now().astimezone().isoformat()
                    print(f"Class changed from {old_class} to {new_class}")
                    print(f"ai_upscale_banned = {job['ai_upscale_banned']}")
                    # FIX-E11: re-derive field_order/needs_ivtc/is_psf/field_segments
                    # from the input file so they match the new class. Reuses
                    # manifest_generate.classify_item but keeps the user's chosen
                    # detected_class/confidence. Falls back to a prompt on failure.
                    try:
                        from manifest_generate import classify_item
                        ptype = "photo" if new_class in ("SCAN_PRINT", "DIGITAL_PHOTO") else "video"
                        reclass = classify_item(job.get("input_path", ""), ptype)
                        job["field_order"] = reclass.get("field_order", job.get("field_order", "unknown"))
                        job["needs_ivtc"] = reclass.get("needs_ivtc", job.get("needs_ivtc", False))
                        job["is_psf"] = reclass.get("is_psf", job.get("is_psf", False))
                        job["field_segments"] = reclass.get("field_segments", job.get("field_segments", []))
                        print(f"field_order={job['field_order']} needs_ivtc={job['needs_ivtc']} is_psf={job['is_psf']}")
                    except Exception as ce:
                        print(f"Could not auto-refresh field_order/needs_ivtc/is_psf ({ce}); please verify manually.")
                else:
                    print("Invalid class")
                break
            elif choice == "q":
                _update_meta_counts(manifest)
                save_manifest(manifest, manifest_path)
                print("Saved. Quitting.")
                return
    
    _update_meta_counts(manifest)
    save_manifest(manifest, manifest_path)
    print("All jobs reviewed. Manifest saved.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Manifest Editor")
    parser.add_argument("manifest", help="Manifest file path")
    subparsers = parser.add_subparsers(dest="command")

    # List command
    list_parser = subparsers.add_parser("list", help="List jobs")
    list_parser.add_argument("--status", help="Filter by status")
    list_parser.add_argument("--params", action="store_true", help="Show parameters")
    list_parser.add_argument("--low-confidence", action="store_true",
                             help="Only jobs below the 0.80 gate (need confirmation)")

    # Set-status command
    status_parser = subparsers.add_parser("set-status", help="Set job status")
    status_parser.add_argument("job_id", help="Job ID")
    status_parser.add_argument("status", choices=["PENDING", "DONE", "FAILED", "SKIP"])

    # Set-override command
    override_parser = subparsers.add_parser("set-override", help="Set parameter override")
    override_parser.add_argument("job_id", help="Job ID")
    override_parser.add_argument("key", help="Parameter name (dot notation supported)")
    override_parser.add_argument("value", help="Parameter value")

    # Set-priority command
    priority_parser = subparsers.add_parser("set-priority", help="Set job priority")
    priority_parser.add_argument("job_id", help="Job ID")
    priority_parser.add_argument("priority", type=int, help="Priority value")

    # Confirm command（人工确认低置信度分类）
    confirm_parser = subparsers.add_parser(
        "confirm", help="Confirm low-confidence classification (clears "
                        "needs_confirmation so batch runs stop skipping it)")
    confirm_parser.add_argument("job_id", nargs="?", help="Job ID")
    confirm_parser.add_argument("--all-low-confidence", action="store_true",
                                help="Confirm ALL jobs below the 0.80 gate "
                                     "(几百上千素材用这个)")
    confirm_parser.add_argument("--confidence", type=float, default=None,
                                help="Optional confidence to set (0-1); "
                                     "default keeps the classifier's value")

    # Interactive TUI command
    tui_parser = subparsers.add_parser("interactive", help="Interactive TUI review")

    args = parser.parse_args()

    if args.command == "list":
        list_jobs(args.manifest, args.status,
                  show_params=getattr(args, "params", False),
                  low_confidence=getattr(args, "low_confidence", False))
    elif args.command == "set-status":
        set_job_status(args.manifest, args.job_id, args.status)
    elif args.command == "set-override":
        set_user_override(args.manifest, args.job_id, args.key, args.value)
    elif args.command == "set-priority":
        set_priority(args.manifest, args.job_id, args.priority)
    elif args.command == "confirm":
        confirm_job(args.manifest, args.job_id,
                    confidence=args.confidence,
                    all_low_confidence=args.all_low_confidence)
    elif args.command == "interactive":
        interactive_tui(args.manifest)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
