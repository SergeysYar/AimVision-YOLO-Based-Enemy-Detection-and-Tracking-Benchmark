# Troubleshooting

## `uv` command not found
Install uv first: https://docs.astral.sh/uv/

## `No module named src` on script launch
Run commands from repository root (`AimVision/`) and use:
```bash
uv run python ...
```

## YOLO training/inference fails due to missing weights
- For training: ensure model checkpoint exists (e.g. `yolov8n.pt`).
- For inference/benchmark/auto-annotation: verify path to `best.pt`.

## OpenCV tracker creation error (`TrackerCSRT_create` / `TrackerKCF_create`)
Make sure `opencv-python` is installed and up to date:
```bash
uv sync
```

## Empty benchmark report
- Verify input directory contains supported video formats (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`).
- Ensure detector weights are valid.
- Confirm videos are readable by OpenCV.

## Low FPS
- Use smaller detector model (`yolov8n.pt`).
- Reduce input resolution (`--imgsz 512` or lower).
- Run on GPU-enabled environment if available.

## Poor tracking quality
- Improve detector quality first (better labels + more epochs).
- Tune confidence threshold and tracker params.
- Include more occlusion-heavy samples in training/validation set.
