"""Video metadata utilities."""

from __future__ import annotations

from pathlib import Path

import cv2


def get_video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    cap.release()

    codec = "".join([chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)]).strip() or "unknown"
    duration = frame_count / fps if fps > 0 else 0.0

    return {
        "video_name": video_path.name,
        "codec": codec,
        "width": width,
        "height": height,
        "fps": float(fps),
        "duration_s": float(duration),
        "frame_count": frame_count,
    }
