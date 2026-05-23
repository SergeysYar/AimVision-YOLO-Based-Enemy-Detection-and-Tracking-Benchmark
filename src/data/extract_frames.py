"""Extract frames from video for dataset creation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from src.utils.io_utils import ensure_dir


def extract_frames(video: Path, output: Path, every_n: int = 1) -> int:
    ensure_dir(output)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_id = 0
    saved = 0
    pbar = tqdm(total=total if total > 0 else None, desc=f"Extracting {video.name}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % every_n == 0:
            out_file = output / f"{video.stem}_{frame_id:06d}.jpg"
            cv2.imwrite(str(out_file), frame)
            saved += 1
        frame_id += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--every-n", type=int, default=1, help="Save every N-th frame.")
    args = parser.parse_args()
    saved = extract_frames(args.video, args.output, args.every_n)
    print(f"Saved {saved} frames to {args.output}")


if __name__ == "__main__":
    main()
