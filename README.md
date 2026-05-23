# AimVision: YOLO-Based Enemy Detection and Tracking Benchmark

Offline computer vision benchmark for detection and tracking of player-like entities in FPS-style gameplay videos.

Russian documentation: [docs/README.ru.md](docs/README.ru.md)

## Safety Warning
This repository is **not** a cheat, aimbot, automation, or game-hacking tool.

- No mouse control
- No process/memory interaction
- No DLL injection or anti-cheat bypass
- No runtime interaction with games

The project processes only offline videos, screenshots, or synthetic sandbox clips.

## What You Get
- Dataset preparation pipeline from recorded videos
- YOLO training workflow (Ultralytics, YOLOv8/YOLOv11 checkpoints)
- Video inference with annotated output
- Tracking backends:
  - SORT-style IoU + Hungarian
  - ByteTrack-like high/low-confidence association
  - OpenCV CSRT/KCF
  - Custom center-distance tracker
- Benchmark runner with per-video, per-tracker comparison
- Exported artifacts:
  - Annotated videos
  - CSV tracking logs
  - Markdown and CSV reports
- Desktop GUI control center + annotation studio

## Quick Start (uv)
1. Install `uv`: https://docs.astral.sh/uv/
2. Sync dependencies:
```bash
uv sync
```
3. Run GUI:
```bash
uv run python scripts/run_gui.py
```

## End-to-End CLI Workflow
1. Extract frames:
```bash
uv run python src/data/extract_frames.py --video datasets/raw_videos/demo.mp4 --output datasets/frames/demo --every-n 2
```
2. Annotate frames in YOLO format (`class_id x_center y_center width height`).
3. Split train/val/test:
```bash
uv run python src/data/split_dataset.py --frames datasets/frames/demo --labels datasets/labels --dataset-root datasets
```
4. Train detector:
```bash
uv run python src/detection/train_yolo.py --data datasets/dataset.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --lr0 0.01
```
5. Run inference:
```bash
uv run python src/detection/infer_yolo.py --weights runs/detect/train/weights/best.pt --video datasets/raw_videos/demo.mp4 --output outputs/videos/demo_detected.mp4 --conf 0.25
```
6. Run tracker benchmark:
```bash
uv run python scripts/run_benchmark.py --weights runs/detect/train/weights/best.pt --videos datasets/raw_videos --trackers sort bytetrack opencv custom --output outputs/reports/benchmark.md
```

## GUI Control Center
Launch:
```bash
uv run python scripts/run_gui.py
```

Tabs:
- `Dataset`: frame extraction
- `Training`: YOLO training controls
- `Inference`: single-video detection export
- `Benchmark`: tracker comparison runner
- `Annotation Studio`: labeling workspace

Annotation Studio features:
- Manual box drawing + class picker
- YOLO label save/load
- Prev/Next navigation with autosave
- `Auto-annotate with YOLO` using trained weights
- `Smart Copy Box to Next Frame` for fast sequential labeling

## Metrics Reported
- Average FPS
- Lost tracks
- ID switches
- Average detection confidence
- Average track duration
- Re-identification after occlusion
- Processing time per frame
- Detection success rate (% frames with detections)

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

## Output Artifacts
- `outputs/videos/*.mp4`: annotated detector/tracker videos
- `outputs/logs/*.csv`: per-frame track logs
- `outputs/reports/*.md`: human-readable benchmark report
- `outputs/reports/*.csv`: benchmark table for analysis

## Project Architecture
```text
AimVision/
  configs/
  datasets/
  docs/
  outputs/
  scripts/
  src/
    data/
    detection/
    tracking/
    evaluation/
    gui/
    utils/
```

## Documentation Index
- Dataset guide: [datasets/README.md](datasets/README.md)
- Dataset preparation: [docs/dataset_preparation.md](docs/dataset_preparation.md)
- Mathematical model: [docs/math_model.md](docs/math_model.md)
- Project description: [docs/project_description.md](docs/project_description.md)
- Benchmark template: [docs/benchmark_results_template.md](docs/benchmark_results_template.md)
- GUI and annotation guide: [docs/gui_guide.md](docs/gui_guide.md)
- Troubleshooting: [docs/troubleshooting.md](docs/troubleshooting.md)

## Reproducibility Notes
- Keep source videos and labels versioned per experiment.
- Store trained weights path used in each benchmark report.
- Use fixed tracker list/order for comparable runs.
- Use the same 5 input videos when comparing trackers.

## Ethics and Responsible Use
- Offline analysis only
- No interaction with game processes or memory
- Not for multiplayer cheating or competitive advantage
- Intended for CV learning, research, and portfolio demonstration
