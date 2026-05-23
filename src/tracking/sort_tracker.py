"""SORT-style tracker using IoU + Hungarian assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class TrackState:
    track_id: int
    bbox: np.ndarray
    class_id: int
    confidence: float
    hits: int = 1
    miss: int = 0
    age: int = 1


class SortTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 20, min_hits: int = 1) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[TrackState] = []
        self.next_id = 1

    def update(self, detections: list[dict]) -> list[dict]:
        for t in self.tracks:
            t.age += 1
            t.miss += 1

        if not self.tracks:
            self._spawn_all(detections)
            return self._active_tracks()

        if not detections:
            self._prune()
            return self._active_tracks()

        cost = np.ones((len(self.tracks), len(detections)), dtype=np.float32)
        for i, tr in enumerate(self.tracks):
            for j, det in enumerate(detections):
                cost[i, j] = 1.0 - iou_xyxy(tr.bbox, np.array(det["bbox"], dtype=np.float32))

        rows, cols = linear_sum_assignment(cost)
        matched_t, matched_d = set(), set()
        for r, c in zip(rows, cols):
            iou = 1.0 - float(cost[r, c])
            if iou < self.iou_threshold:
                continue
            tr = self.tracks[r]
            det = detections[c]
            tr.bbox = np.array(det["bbox"], dtype=np.float32)
            tr.class_id = int(det["class_id"])
            tr.confidence = float(det["confidence"])
            tr.hits += 1
            tr.miss = 0
            matched_t.add(r)
            matched_d.add(c)

        for i, det in enumerate(detections):
            if i not in matched_d:
                self._spawn(det)

        self._prune()
        return self._active_tracks()

    def _spawn_all(self, detections: list[dict]) -> None:
        for det in detections:
            self._spawn(det)

    def _spawn(self, det: dict) -> None:
        self.tracks.append(
            TrackState(
                track_id=self.next_id,
                bbox=np.array(det["bbox"], dtype=np.float32),
                class_id=int(det["class_id"]),
                confidence=float(det["confidence"]),
            )
        )
        self.next_id += 1

    def _prune(self) -> None:
        self.tracks = [t for t in self.tracks if t.miss <= self.max_age]

    def _active_tracks(self) -> list[dict]:
        return [
            {
                "track_id": t.track_id,
                "bbox": t.bbox.tolist(),
                "class_id": t.class_id,
                "confidence": t.confidence,
            }
            for t in self.tracks
            if t.hits >= self.min_hits
        ]
