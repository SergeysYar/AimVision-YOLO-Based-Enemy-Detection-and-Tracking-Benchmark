"""Run YOLO inference on a single video and export annotated output."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.utils.io_utils import ensure_dir


def run_inference(weights: Path, video: Path, output: Path, conf: float = 0.25) -> None:
    model = YOLO(str(weights))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    ensure_dir(output.parent)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_count = 0
    t0 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=conf, verbose=False)[0]
        annotated = results.plot()
        frame_count += 1
        elapsed = max(1e-6, time.perf_counter() - t0)
        inst_fps = frame_count / elapsed
        cv2.putText(annotated, f"FPS: {inst_fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
        writer.write(annotated)

    cap.release()
    writer.release()
    print(f"Saved annotated video: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO video inference.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    run_inference(args.weights, args.video, args.output, args.conf)


if __name__ == "__main__":
    main()
