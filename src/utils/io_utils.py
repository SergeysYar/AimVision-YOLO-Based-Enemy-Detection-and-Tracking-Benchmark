"""I/O helper utilities for AimVision."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_video_files(directory: Path) -> list[Path]:
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in VIDEO_EXTS and p.is_file()])


def write_csv(rows: Iterable[dict], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    pd.DataFrame(list(rows)).to_csv(output_path, index=False)


def write_markdown_table(df: pd.DataFrame, title: str, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")


def parse_classes(raw: Sequence[str]) -> list[str]:
    if not raw:
        return ["enemy"]
    return [item.strip() for item in raw if item.strip()]
