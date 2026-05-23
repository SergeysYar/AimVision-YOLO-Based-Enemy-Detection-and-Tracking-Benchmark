"""Visualization utilities for detections and tracks."""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np


def color_for_id(track_id: int) -> tuple[int, int, int]:
    np.random.seed(track_id + 7)
    return tuple(int(v) for v in np.random.randint(60, 255, size=3))


def draw_tracks(
    frame: np.ndarray,
    tracks: list[dict],
    class_names: dict[int, str],
    trajectories: dict[int, list[tuple[int, int]]],
    show_conf: bool = True,
) -> np.ndarray:
    for tr in tracks:
        x1, y1, x2, y2 = map(int, tr["bbox"])
        tid = int(tr["track_id"])
        cid = int(tr["class_id"])
        conf = float(tr.get("confidence", 0.0))
        c = color_for_id(tid)
        label = f"ID {tid} | {class_names.get(cid, str(cid))}"
        if show_conf:
            label += f" {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        trajectories[tid].append(center)
        for i in range(1, len(trajectories[tid])):
            cv2.line(frame, trajectories[tid][i - 1], trajectories[tid][i], c, 2)
    return frame


def new_trajectory_store() -> dict[int, list[tuple[int, int]]]:
    return defaultdict(list)
