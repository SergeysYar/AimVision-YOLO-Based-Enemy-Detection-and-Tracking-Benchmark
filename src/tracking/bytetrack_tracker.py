"""Simplified ByteTrack-inspired tracker (high/low confidence association)."""

from __future__ import annotations

from src.tracking.sort_tracker import SortTracker


class ByteTrackLikeTracker:
    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        iou_threshold: float = 0.3,
        max_age: int = 20,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.primary = SortTracker(iou_threshold=iou_threshold, max_age=max_age, min_hits=1)

    def update(self, detections: list[dict]) -> list[dict]:
        high = [d for d in detections if float(d["confidence"]) >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= float(d["confidence"]) < self.high_thresh]

        tracks = self.primary.update(high)
        if low:
            tracks = self.primary.update(low)
        return tracks
