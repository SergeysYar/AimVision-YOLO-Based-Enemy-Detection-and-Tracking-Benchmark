"""Convenience benchmark runner for tracker comparison."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AimVision benchmark.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--trackers", nargs="+", default=["sort", "bytetrack", "opencv", "custom"])
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/benchmark.md"))
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "src/evaluation/compare_trackers.py",
        "--weights",
        str(args.weights),
        "--videos",
        str(args.videos),
        "--output",
        str(args.output),
        "--trackers",
        *args.trackers,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
