# Dataset Preparation

## 1. Record or Collect Source Clips
- Use your own gameplay recordings, public benchmark clips, or synthetic aim-trainer footage.
- Store clips in `datasets/raw_videos/`.

## 2. Extract Frames
```bash
python src/data/extract_frames.py --video datasets/raw_videos/demo.mp4 --output datasets/frames/demo --every-n 2
```

## 3. Annotate in YOLO Format
For each frame, create a label file:
`class_id x_center y_center width height` (normalized to `[0,1]`).

Classes:
- `0 enemy`
- `1 teammate`
- `2 weapon`
- `3 head`

## 4. Train/Val/Test Split
```bash
python src/data/split_dataset.py --frames datasets/frames/demo --labels datasets/labels --dataset-root datasets
```

## 5. Verify Dataset YAML
Check `datasets/dataset.yaml` before training.
