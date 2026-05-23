"""OpenCV tracker wrapper (CSRT/KCF) with periodic detector refresh."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.tracking.sort_tracker import iou_xyxy


def _create_tracker(name: str):
    name = name.lower()
    if name == "kcf":
        return cv2.TrackerKCF_create()
    return cv2.TrackerCSRT_create()


@dataclass
class CvTrack:
    track_id: int
    tracker: object
    bbox: list[float]
    class_id: int
    confidence: float
    miss: int = 0


class OpenCVMultiObjectTracker:
    def __init__(self, tracker_type: str = "csrt", iou_threshold: float = 0.3, max_age: int = 20) -> None:
        self.tracker_type = tracker_type
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: list[CvTrack] = []
        self.next_id = 1

    def update(self, frame: np.ndarray, detections: list[dict]) -> list[dict]:
        for t in self.tracks:
            ok, box = t.tracker.update(frame)
            if ok:
                x, y, w, h = box
                t.bbox = [x, y, x + w, y + h]
                t.miss = 0
            else:
                t.miss += 1

        unmatched = []
        for det in detections:
            db = np.array(det["bbox"], dtype=np.float32)
            best_iou = 0.0
            best_idx = -1
            for idx, tr in enumerate(self.tracks):
                iou = iou_xyxy(np.array(tr.bbox, dtype=np.float32), db)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_iou < self.iou_threshold or best_idx < 0:
                unmatched.append(det)
            else:
                tr = self.tracks[best_idx]
                x1, y1, x2, y2 = det["bbox"]
                tracker = _create_tracker(self.tracker_type)
                tracker.init(frame, (float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
                tr.tracker = tracker
                tr.bbox = [float(x1), float(y1), float(x2), float(y2)]
                tr.class_id = int(det["class_id"])
                tr.confidence = float(det["confidence"])
                tr.miss = 0

        for det in unmatched:
            x1, y1, x2, y2 = det["bbox"]
            tracker = _create_tracker(self.tracker_type)
            tracker.init(frame, (float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
            self.tracks.append(
                CvTrack(
                    track_id=self.next_id,
                    tracker=tracker,
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    class_id=int(det["class_id"]),
                    confidence=float(det["confidence"]),
                )
            )
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.miss <= self.max_age]
        return [
            {"track_id": t.track_id, "bbox": t.bbox, "class_id": t.class_id, "confidence": t.confidence}
            for t in self.tracks
        ]
