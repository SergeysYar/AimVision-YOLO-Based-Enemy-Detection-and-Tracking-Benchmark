"""Custom center-distance tracker from scratch."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass
class CustomTrack:
    track_id: int
    bbox: list[float]
    class_id: int
    confidence: float
    missed: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


class CustomTracker:
    def __init__(self, max_distance: float = 80.0, max_missed: int = 15) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.tracks: list[CustomTrack] = []
        self.next_id = 1

    def update(self, detections: list[dict]) -> list[dict]:
        for tr in self.tracks:
            tr.missed += 1

        assigned_det: set[int] = set()
        for tr in self.tracks:
            tx, ty = tr.center
            best_idx = -1
            best_dist = float("inf")
            for i, det in enumerate(detections):
                if i in assigned_det:
                    continue
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                d = hypot(cx - tx, cy - ty)
                if d < best_dist:
                    best_idx, best_dist = i, d
            if best_idx >= 0 and best_dist <= self.max_distance:
                det = detections[best_idx]
                tr.bbox = [float(v) for v in det["bbox"]]
                tr.class_id = int(det["class_id"])
                tr.confidence = float(det["confidence"])
                tr.missed = 0
                assigned_det.add(best_idx)

        for i, det in enumerate(detections):
            if i in assigned_det:
                continue
            self.tracks.append(
                CustomTrack(
                    track_id=self.next_id,
                    bbox=[float(v) for v in det["bbox"]],
                    class_id=int(det["class_id"]),
                    confidence=float(det["confidence"]),
                )
            )
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return [
            {"track_id": t.track_id, "bbox": t.bbox, "class_id": t.class_id, "confidence": t.confidence}
            for t in self.tracks
        ]
