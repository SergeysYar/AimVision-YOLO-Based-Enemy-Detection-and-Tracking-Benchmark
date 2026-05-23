"""Run multi-tracker benchmark on video files."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from src.evaluation.metrics import compute_metrics
from src.tracking.bytetrack_tracker import ByteTrackLikeTracker
from src.tracking.custom_tracker import CustomTracker
from src.tracking.opencv_tracker import OpenCVMultiObjectTracker
from src.tracking.sort_tracker import SortTracker
from src.utils.io_utils import ensure_dir, list_video_files
from src.utils.video_info import get_video_info
from src.utils.visualization import draw_tracks, new_trajectory_store


def make_tracker(name: str):
    lname = name.lower()
    if lname == "sort":
        return SortTracker()
    if lname == "bytetrack":
        return ByteTrackLikeTracker()
    if lname == "opencv":
        return OpenCVMultiObjectTracker(tracker_type="csrt")
    if lname == "custom":
        return CustomTracker()
    raise ValueError(f"Unknown tracker: {name}")


def detections_from_result(result) -> list[dict]:
    out = []
    if result.boxes is None:
        return out
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy()
    for box, conf, cls_id in zip(boxes, confs, clss):
        out.append(
            {
                "bbox": [float(v) for v in box.tolist()],
                "confidence": float(conf),
                "class_id": int(cls_id),
            }
        )
    return out


def run_single(weights: Path, video: Path, tracker_name: str, out_video: Path, out_log: Path, conf: float) -> dict:
    detector = YOLO(str(weights))
    tracker = make_tracker(tracker_name)
    info = get_video_info(video)

    cap = cv2.VideoCapture(str(video))
    ensure_dir(out_video.parent)
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info["fps"] if info["fps"] > 0 else 30.0,
        (info["width"], info["height"]),
    )

    trajectories = new_trajectory_store()
    logs: list[dict] = []
    frame_times: list[float] = []
    frame_id = 0
    detection_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        result = detector.predict(frame, conf=conf, verbose=False)[0]
        detections = detections_from_result(result)
        if detections:
            detection_frames += 1

        if tracker_name.lower() == "opencv":
            tracks = tracker.update(frame, detections)
        else:
            tracks = tracker.update(detections)

        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            logs.append(
                {
                    "frame_id": frame_id,
                    "track_id": int(t["track_id"]),
                    "class_id": int(t["class_id"]),
                    "confidence": float(t["confidence"]),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "center_x": float(cx),
                    "center_y": float(cy),
                }
            )

        frame = draw_tracks(frame, tracks, detector.names, trajectories)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        frame_times.append(dt_ms)
        fps_now = 1000.0 / dt_ms if dt_ms > 0 else 0.0
        cv2.putText(frame, f"{tracker_name} FPS {fps_now:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()
    log_df = pd.DataFrame(logs)
    ensure_dir(out_log.parent)
    log_df.to_csv(out_log, index=False)

    m = compute_metrics(log_df, frame_times, frame_id, detection_frames)
    summary = {
        "video_name": info["video_name"],
        "tracker": tracker_name,
        "codec": info["codec"],
        "resolution": f'{info["width"]}x{info["height"]}',
        "video_fps": info["fps"],
        "duration_s": info["duration_s"],
        "frames": info["frame_count"],
        "detected_objects": int(len(log_df)),
        "average_fps": m.average_fps,
        "lost_tracks": m.lost_tracks,
        "id_switches": m.id_switches,
        "average_detection_confidence": m.average_detection_confidence,
        "average_track_duration_frames": m.average_track_duration_frames,
        "reid_after_occlusion_rate": m.reid_after_occlusion_rate,
        "processing_time_per_frame_ms": m.processing_time_per_frame_ms,
        "detection_success_rate": m.detection_success_rate,
        "log_csv": str(out_log.as_posix()),
        "annotated_video": str(out_video.as_posix()),
    }
    return summary


def benchmark(weights: Path, videos_dir: Path, trackers: list[str], output_md: Path, conf: float) -> None:
    videos = list_video_files(videos_dir)[:5]
    if not videos:
        raise FileNotFoundError(f"No videos found in {videos_dir}")

    reports_dir = output_md.parent
    logs_dir = reports_dir.parent / "logs"
    vids_dir = reports_dir.parent / "videos"
    ensure_dir(reports_dir)
    ensure_dir(logs_dir)
    ensure_dir(vids_dir)

    rows = []
    for video in videos:
        for tracker_name in trackers:
            out_video = vids_dir / f"{video.stem}_{tracker_name}.mp4"
            out_log = logs_dir / f"{video.stem}_{tracker_name}.csv"
            rows.append(run_single(weights, video, tracker_name, out_video, out_log, conf))
            print(f"Processed {video.name} with {tracker_name}")

    df = pd.DataFrame(rows)
    csv_path = output_md.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    with output_md.open("w", encoding="utf-8") as f:
        f.write("# AimVision Tracker Benchmark Report\n\n")
        f.write("## Comparison Table\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"Saved report: {output_md}")
    print(f"Saved CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple trackers on same videos.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True, help="Directory with input videos.")
    parser.add_argument("--trackers", nargs="+", default=["sort", "bytetrack", "opencv"])
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/benchmark.md"))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    benchmark(args.weights, args.videos, args.trackers, args.output, args.conf)


if __name__ == "__main__":
    main()
