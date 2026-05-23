# GUI Guide

## Launch
```bash
uv run python scripts/run_gui.py
```

## Tabs Overview
- `Dataset`: run frame extraction from source video.
- `Training`: run YOLO training with configurable hyperparameters.
- `Inference`: run detector on a single video and export annotated output.
- `Benchmark`: run multi-tracker comparison and produce reports.
- `Annotation Studio`: label dataset frames in YOLO format.

## Annotation Studio Workflow
1. Select `images` directory (usually `datasets/frames/...`).
2. Select `labels` directory (usually `datasets/labels`).
3. Click `Load Images`.
4. Choose class from class dropdown.
5. Draw box with left mouse drag.
6. Click `Save Labels`.
7. Use `Prev/Next` for navigation.

## Intelligent Features
- `Auto-annotate with YOLO`:
  - Loads model weights from the weights field.
  - Predicts boxes for the current frame.
  - Converts predictions into YOLO label entries.
- `Smart Copy Box to Next Frame`:
  - Copies last selected/created box to next frame.
  - Useful when object motion is small between adjacent frames.

## Tips
- Start with auto-annotation, then quickly clean up manually.
- Label difficult occlusion frames carefully to improve tracker robustness.
- Keep class usage consistent (`enemy`, `teammate`, `weapon`, `head`).
