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
              show_params: bool = False):
    """List jobs in manifest."""
    manifest = load_manifest(manifest_path)
    jobs = manifest.get("jobs", [])

    print(f"{'Job ID':<12} {'Status':<10} {'Class':<14} {'Priority':<8} {'Confidence':<10} {'Path'}")
    print("-" * 100)

    for job in jobs:
        if status_filter and job.get("status") != status_filter:
            continue
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
        if job.get("status") not in ("PENDING", "FAILED"):
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
        if job.get("status") not in ("PENDING", "FAILED"):
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

    # Interactive TUI command
    tui_parser = subparsers.add_parser("interactive", help="Interactive TUI review")

    args = parser.parse_args()

    if args.command == "list":
        list_jobs(args.manifest, args.status, show_params=getattr(args, "params", False))
    elif args.command == "set-status":
        set_job_status(args.manifest, args.job_id, args.status)
    elif args.command == "set-override":
        set_user_override(args.manifest, args.job_id, args.key, args.value)
    elif args.command == "set-priority":
        set_priority(args.manifest, args.job_id, args.priority)
    elif args.command == "interactive":
        interactive_tui(args.manifest)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
