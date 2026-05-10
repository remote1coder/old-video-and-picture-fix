#!/usr/bin/env python3
"""
Manifest Editor (Optional)
Interactive or command-line tool to edit manifest jobs.
"""

from pathlib import Path
from typing import Optional

from manifest_loader import load_manifest, save_manifest, update_job_status


def list_jobs(manifest_path: str, status_filter: Optional[str] = None):
    """List jobs in manifest."""
    manifest = load_manifest(manifest_path)
    jobs = manifest.get("jobs", [])

    print(f"{'Job ID':<12} {'Status':<10} {'Class':<14} {'Confidence':<10} {'Path'}")
    print("-" * 100)

    for job in jobs:
        if status_filter and job.get("status") != status_filter:
            continue
        print(f"{job.get('job_id', ''):<12} "
              f"{job.get('status', ''):<10} "
              f"{job.get('detected_class', ''):<14} "
              f"{job.get('confidence', 0):.2f}       "
              f"{Path(job.get('input_path', '')).name}")


def set_job_status(manifest_path: str, job_id: str, status: str):
    """Set job status."""
    manifest = load_manifest(manifest_path)
    update_job_status(manifest, job_id, status)
    save_manifest(manifest, manifest_path)
    print(f"Job {job_id} status set to {status}")


def set_user_override(manifest_path: str, job_id: str,
                      key: str, value: str):
    """Set user override parameter for a job."""
    manifest = load_manifest(manifest_path)
    for job in manifest.get("jobs", []):
        if job.get("job_id") == job_id:
            job.setdefault("user_overrides", {})[key] = value
            print(f"Set {key}={value} for job {job_id}")
            break
    save_manifest(manifest, manifest_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Manifest Editor")
    parser.add_argument("manifest", help="Manifest file path")
    subparsers = parser.add_subparsers(dest="command")

    # List command
    list_parser = subparsers.add_parser("list", help="List jobs")
    list_parser.add_argument("--status", help="Filter by status")

    # Set-status command
    status_parser = subparsers.add_parser("set-status", help="Set job status")
    status_parser.add_argument("job_id", help="Job ID")
    status_parser.add_argument("status", choices=["PENDING", "DONE", "FAILED", "SKIP"])

    # Set-override command
    override_parser = subparsers.add_parser("set-override", help="Set parameter override")
    override_parser.add_argument("job_id", help="Job ID")
    override_parser.add_argument("key", help="Parameter name")
    override_parser.add_argument("value", help="Parameter value")

    args = parser.parse_args()

    if args.command == "list":
        list_jobs(args.manifest, args.status)
    elif args.command == "set-status":
        set_job_status(args.manifest, args.job_id, args.status)
    elif args.command == "set-override":
        set_user_override(args.manifest, args.job_id, args.key, args.value)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
