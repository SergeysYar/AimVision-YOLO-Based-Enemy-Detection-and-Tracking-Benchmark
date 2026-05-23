"""Convenience script for training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO training with standard paths.")
    parser.add_argument("--data", type=Path, default=Path("datasets/dataset.yaml"))
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "src/detection/train_yolo.py",
        "--data",
        str(args.data),
        "--model",
        args.model,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
