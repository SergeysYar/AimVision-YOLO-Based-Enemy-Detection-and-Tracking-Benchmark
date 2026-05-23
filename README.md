# AimVision: YOLO-Based Enemy Detection and Tracking Benchmark

Offline computer vision benchmark for detecting and tracking player-like entities in Counter-Strike-style gameplay videos.

Русская документация: [docs/README.ru.md](docs/README.ru.md)

## Safety Warning
This repository is **not** a cheat, aimbot, automation, or game-hacking tool.
- No mouse control
- No process/memory interaction
- No DLL injection or anti-cheat bypass
- No runtime interaction with games

It only processes offline videos, screenshots, or synthetic sandbox clips for research and education.

## Features
- Dataset creation pipeline from recorded videos
- YOLO training workflow (Ultralytics YOLOv8/YOLOv11-compatible checkpoints)
- Video inference with annotated output
- Tracking backends:
  - SORT-style IoU + Hungarian
  - ByteTrack-like high/low confidence association
  - OpenCV CSRT/KCF tracker
  - Custom center-distance tracker (from scratch)
- Tracker comparison on shared videos
- Exported artifacts:
  - Annotated videos
  - Per-frame tracking CSV logs
  - Markdown + CSV benchmark reports

## Demo
- GIF placeholder: `docs/assets/demo.gif`
- Screenshot placeholders:
  - `docs/assets/frame_detection.png`
  - `docs/assets/frame_tracking.png`

## Dataset Structure
```text
datasets/
  raw_videos/
  frames/
  labels/
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
  dataset.yaml
```

## Installation
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Training
```bash
python src/detection/train_yolo.py --data datasets/dataset.yaml --model yolov8n.pt --epochs 50 --imgsz 640
```

## Inference
```bash
python src/detection/infer_yolo.py --weights runs/detect/train/weights/best.pt --video datasets/raw_videos/demo.mp4 --output outputs/videos/demo_detected.mp4
```

## Frame Extraction
```bash
python src/data/extract_frames.py --video datasets/raw_videos/demo.mp4 --output datasets/frames/demo
```

## Benchmark
```bash
python scripts/run_benchmark.py --weights runs/detect/train/weights/best.pt --videos datasets/raw_videos --trackers sort bytetrack opencv custom --output outputs/reports/benchmark.md
```

## GUI Control Center
```bash
python scripts/run_gui.py
```

Tabs:
- `Dataset`: frame extraction runner.
- `Training`: YOLO training controls.
- `Inference`: video detection export.
- `Benchmark`: tracker comparison run.
- `Annotation Studio`: full dataset labeling workspace.

Annotation Studio includes:
- Manual box drawing with class picker.
- Save labels in YOLO format.
- Prev/Next navigation with autosave.
- `Auto-annotate with YOLO` using your trained weights.
- `Smart Copy Box to Next Frame` for fast sequential labeling.

## Metrics
- Average FPS
- Lost tracks
- ID switches
- Average detection confidence
- Average track duration
- Re-identification after occlusion
- Processing time per frame
- Detection success rate (% frames with detections)

## Architecture
```text
AimVision/
  README.md
  requirements.txt
  pyproject.toml
  .gitignore
  configs/
    config.yaml
  datasets/
    README.md
    dataset.yaml
  src/
    data/
      extract_frames.py
      split_dataset.py
    detection/
      train_yolo.py
      infer_yolo.py
    tracking/
      sort_tracker.py
      bytetrack_tracker.py
      opencv_tracker.py
      custom_tracker.py
    evaluation/
      metrics.py
      compare_trackers.py
    gui/
      main_app.py
      annotation_tool.py
    utils/
      video_info.py
      visualization.py
      io_utils.py
  scripts/
    run_training.py
    run_video_demo.py
    run_benchmark.py
    run_gui.py
  outputs/
    videos/
    reports/
    logs/
  docs/
    project_description.md
    math_model.md
    dataset_preparation.md
    benchmark_results_template.md
```

## ML Pipeline Diagram
```text
Raw Video -> Frame Extraction -> Annotation -> YOLO Training
      -> Detection on Video -> Tracker Update (SORT/ByteTrack/OpenCV/Custom)
      -> Metrics + Logs -> Annotated Video + CSV + Markdown Report
```

## Ethics and Responsible Use
- Offline analysis only
- No interaction with game memory/processes
- Not for multiplayer cheating or competitive advantage
- Intended for CV learning, research, and portfolio demonstration

## Future Improvements
1. Plug in official ByteTrack/BoT-SORT implementations with appearance embeddings.
2. Add MOTChallenge-compatible evaluation with labeled track ground truth.
3. Add hyperparameter sweep for tracker thresholds.
4. Add unit tests and CI pipeline.
