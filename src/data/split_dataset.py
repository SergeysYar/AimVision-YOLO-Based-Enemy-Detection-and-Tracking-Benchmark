"""Split prepared image/label pairs into train/val/test."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from src.utils.io_utils import ensure_dir


def split_dataset(
    frames_dir: Path,
    labels_dir: Path,
    dataset_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    random.seed(seed)
    images = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    pairs = [(img, labels_dir / f"{img.stem}.txt") for img in images if (labels_dir / f"{img.stem}.txt").exists()]
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }

    for split, items in splits.items():
        img_out = ensure_dir(dataset_root / split / "images")
        lbl_out = ensure_dir(dataset_root / split / "labels")
        for img, lbl in items:
            shutil.copy2(img, img_out / img.name)
            shutil.copy2(lbl, lbl_out / lbl.name)
        print(f"{split}: {len(items)} samples")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split YOLO dataset into train/val/test.")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_dataset(args.frames, args.labels, args.dataset_root, args.train_ratio, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
