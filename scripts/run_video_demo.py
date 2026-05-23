"""Convenience demo runner for single video inference."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-video detector demo.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/videos/demo_detected.mp4"))
    args = parser.parse_args()
    subprocess.run(
        [
            sys.executable,
            "src/detection/infer_yolo.py",
            "--weights",
            str(args.weights),
            "--video",
            str(args.video),
            "--output",
            str(args.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
