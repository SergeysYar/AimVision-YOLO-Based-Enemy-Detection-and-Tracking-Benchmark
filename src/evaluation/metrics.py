"""Benchmark metric computation for tracking logs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MetricBundle:
    average_fps: float
    lost_tracks: int
    id_switches: int
    average_detection_confidence: float
    average_track_duration_frames: float
    reid_after_occlusion_rate: float
    processing_time_per_frame_ms: float
    detection_success_rate: float


def compute_metrics(
    log_df: pd.DataFrame,
    frame_times_ms: list[float],
    total_frames: int,
    detection_frames: int,
) -> MetricBundle:
    if log_df.empty:
        return MetricBundle(0.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    avg_conf = float(log_df["confidence"].mean())
    avg_ms = float(sum(frame_times_ms) / max(1, len(frame_times_ms)))
    avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    detection_success = float(100.0 * detection_frames / max(1, total_frames))

    durations = log_df.groupby("track_id")["frame_id"].count()
    avg_dur = float(durations.mean()) if not durations.empty else 0.0

    lost_tracks = int((durations < 5).sum())

    id_switches = 0
    for _, g in log_df.sort_values(["frame_id", "track_id"]).groupby("class_id"):
        prev = None
        for _, row in g.iterrows():
            if prev is not None and row["frame_id"] == prev["frame_id"] + 1 and row["track_id"] != prev["track_id"]:
                id_switches += 1
            prev = row

    reid_hits = 0
    reid_events = 0
    for _, g in log_df.groupby("track_id"):
        frames = g["frame_id"].sort_values().to_list()
        for i in range(1, len(frames)):
            gap = frames[i] - frames[i - 1]
            if gap > 1:
                reid_events += 1
                if gap <= 15:
                    reid_hits += 1
    reid_rate = float(100.0 * reid_hits / reid_events) if reid_events > 0 else 0.0

    return MetricBundle(
        average_fps=avg_fps,
        lost_tracks=lost_tracks,
        id_switches=id_switches,
        average_detection_confidence=avg_conf,
        average_track_duration_frames=avg_dur,
        reid_after_occlusion_rate=reid_rate,
        processing_time_per_frame_ms=avg_ms,
        detection_success_rate=detection_success,
    )
