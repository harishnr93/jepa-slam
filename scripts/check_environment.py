#!/usr/bin/env python
"""Inspect the hardware and software environment and save a JSON report.

Usage:
    python scripts/check_environment.py [--output results/metrics/environment.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latent_loop.utils.config import REPO_ROOT
from latent_loop.utils.device import collect_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "results/metrics/environment.json"
    )
    args = parser.parse_args()

    env = collect_environment(REPO_ROOT)
    print(f"Platform     : {env['platform']} (WSL: {env['wsl']})")
    print(f"Python       : {env['python']} ({env['python_executable']})")
    print(f"CPU          : {env['cpu_model']} ({env['logical_cpus']} logical CPUs)")
    print(
        f"RAM          : {env['ram_total_gib']} GiB total, {env['ram_available_gib']} GiB available"
    )
    print(f"Disk free    : {env['disk_free_gib']} GiB at {env['disk_path']}")
    if env["gpus"]:
        for gpu in env["gpus"]:
            print(
                f"GPU          : {gpu['name']}, {gpu['memory_total_mib']} MiB, "
                f"driver {gpu['driver_version']}, compute capability {gpu['compute_capability']}"
            )
    else:
        print("GPU          : none detected (nvidia-smi unavailable) -> CPU execution")
    torch = env["torch"]
    if torch["installed"]:
        print(
            f"PyTorch      : {torch['version']} (CUDA build {torch['cuda_build']}, "
            f"available: {torch['cuda_available']})"
        )
    else:
        print("PyTorch      : not installed (required from Phase 2)")
    missing = [name for name, version in env["packages"].items() if version is None]
    print("Packages     : " + ", ".join(f"{k}={v}" for k, v in env["packages"].items() if v))
    if missing:
        print("Not installed: " + ", ".join(missing))
    if env["pythonpath"]:
        print(
            "WARNING      : PYTHONPATH is set (e.g. by ROS). Run with `env -u PYTHONPATH` "
            "to keep the virtual environment isolated."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(env, indent=2))
    print(f"Report saved : {args.output}")


if __name__ == "__main__":
    main()
