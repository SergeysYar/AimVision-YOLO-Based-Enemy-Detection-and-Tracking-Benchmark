# Dataset Guide

Directory layout:

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
```

1. Put source gameplay recordings in `datasets/raw_videos/`.
2. Extract frames:
   `uv run python src/data/extract_frames.py --video datasets/raw_videos/demo.mp4 --output datasets/frames/demo`
3. Annotate frames with any YOLO-compatible labeling tool (LabelImg, CVAT, Roboflow).
4. Save YOLO label `.txt` files into `datasets/labels/`.
5. Split:
   `uv run python src/data/split_dataset.py --frames datasets/frames/demo --labels datasets/labels --dataset-root datasets`

Ethical collection:
- Use your own recordings or assets you are allowed to use.
- Do not scrape private or copyrighted material without permission.
- Do not collect or publish identifiable personal information.
