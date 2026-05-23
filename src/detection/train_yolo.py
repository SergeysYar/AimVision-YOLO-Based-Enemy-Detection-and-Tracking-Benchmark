"""Train YOLO model with configurable arguments."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def train_model(
    data: Path,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    lr0: float,
    project: Path,
    name: str,
) -> None:
    detector = YOLO(model)
    detector.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        project=str(project),
        name=name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO detector.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", type=str, default="train")
    args = parser.parse_args()
    train_model(args.data, args.model, args.epochs, args.imgsz, args.batch, args.lr0, args.project, args.name)


if __name__ == "__main__":
    main()
