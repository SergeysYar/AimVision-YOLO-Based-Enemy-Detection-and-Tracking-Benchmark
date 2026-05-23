"""AimVision desktop GUI."""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.gui.annotation_tool import AnnotationTab


class CommandPanel(ttk.Frame):
    """Reusable command runner block with log output."""

    def __init__(self, master: tk.Misc, title: str, command_builder):
        super().__init__(master, padding=10)
        self.command_builder = command_builder
        ttk.Label(self, text=title, font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))
        self.form = ttk.Frame(self)
        self.form.pack(fill="x")
        self.log = tk.Text(self, height=14, wrap="word")
        self.log.pack(fill="both", expand=True, pady=(10, 0))
        self.run_btn = ttk.Button(self, text="Run", command=self.run_command)
        self.run_btn.pack(anchor="e", pady=(8, 0))

    def add_row(self, label: str, default: str = "", browse_file: bool = False, browse_dir: bool = False) -> ttk.Entry:
        row = ttk.Frame(self.form)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=20).pack(side="left")
        entry = ttk.Entry(row)
        entry.insert(0, default)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        if browse_file:
            ttk.Button(row, text="...", width=4, command=lambda: self._pick_file(entry)).pack(side="left")
        if browse_dir:
            ttk.Button(row, text="...", width=4, command=lambda: self._pick_dir(entry)).pack(side="left")
        return entry

    def _pick_file(self, entry: ttk.Entry) -> None:
        path = filedialog.askopenfilename()
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def _pick_dir(self, entry: ttk.Entry) -> None:
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def write_log(self, text: str) -> None:
        self.log.insert(tk.END, text)
        self.log.see(tk.END)

    def run_command(self) -> None:
        cmd = self.command_builder()
        if not cmd:
            return
        self.run_btn.configure(state="disabled")
        self.write_log(f"\n$ {' '.join(cmd)}\n")

        def worker() -> None:
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    self.log.after(0, self.write_log, line)
                code = process.wait()
                self.log.after(0, self.write_log, f"\n[Exit code: {code}]\n")
            except Exception as exc:  # pragma: no cover
                self.log.after(0, self.write_log, f"\n[Error] {exc}\n")
            finally:
                self.log.after(0, lambda: self.run_btn.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()


class AimVisionApp(tk.Tk):
    """Main multi-tab application window."""

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        self.title("AimVision Control Center")
        self.geometry("1320x860")
        self.minsize(1100, 760)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        notebook.add(self._build_dataset_tab(notebook), text="Dataset")
        notebook.add(self._build_training_tab(notebook), text="Training")
        notebook.add(self._build_inference_tab(notebook), text="Inference")
        notebook.add(self._build_benchmark_tab(notebook), text="Benchmark")
        notebook.add(AnnotationTab(notebook, project_root=self.project_root), text="Annotation Studio")

    def _build_dataset_tab(self, master: ttk.Notebook) -> ttk.Frame:
        panel = CommandPanel(master, "Extract Frames", self._dataset_command)
        self.ds_video = panel.add_row("Input video", str(self.project_root / "datasets/raw_videos/demo.mp4"), browse_file=True)
        self.ds_out = panel.add_row("Output folder", str(self.project_root / "datasets/frames/demo"), browse_dir=True)
        self.ds_every = panel.add_row("Every N frame", "1")
        return panel

    def _dataset_command(self) -> list[str]:
        return [
            sys.executable,
            "src/data/extract_frames.py",
            "--video",
            self.ds_video.get().strip(),
            "--output",
            self.ds_out.get().strip(),
            "--every-n",
            self.ds_every.get().strip() or "1",
        ]

    def _build_training_tab(self, master: ttk.Notebook) -> ttk.Frame:
        panel = CommandPanel(master, "YOLO Training", self._train_command)
        self.tr_data = panel.add_row("dataset.yaml", str(self.project_root / "datasets/dataset.yaml"), browse_file=True)
        self.tr_model = panel.add_row("Model checkpoint", "yolov8n.pt")
        self.tr_epochs = panel.add_row("Epochs", "50")
        self.tr_imgsz = panel.add_row("Image size", "640")
        self.tr_batch = panel.add_row("Batch", "16")
        self.tr_lr0 = panel.add_row("Learning rate", "0.01")
        return panel

    def _train_command(self) -> list[str]:
        return [
            sys.executable,
            "src/detection/train_yolo.py",
            "--data",
            self.tr_data.get().strip(),
            "--model",
            self.tr_model.get().strip(),
            "--epochs",
            self.tr_epochs.get().strip(),
            "--imgsz",
            self.tr_imgsz.get().strip(),
            "--batch",
            self.tr_batch.get().strip(),
            "--lr0",
            self.tr_lr0.get().strip(),
        ]

    def _build_inference_tab(self, master: ttk.Notebook) -> ttk.Frame:
        panel = CommandPanel(master, "Video Inference", self._infer_command)
        self.inf_weights = panel.add_row("Weights", str(self.project_root / "runs/detect/train/weights/best.pt"), browse_file=True)
        self.inf_video = panel.add_row("Input video", str(self.project_root / "datasets/raw_videos/demo.mp4"), browse_file=True)
        self.inf_out = panel.add_row("Output video", str(self.project_root / "outputs/videos/demo_detected.mp4"))
        self.inf_conf = panel.add_row("Confidence", "0.25")
        return panel

    def _infer_command(self) -> list[str]:
        return [
            sys.executable,
            "src/detection/infer_yolo.py",
            "--weights",
            self.inf_weights.get().strip(),
            "--video",
            self.inf_video.get().strip(),
            "--output",
            self.inf_out.get().strip(),
            "--conf",
            self.inf_conf.get().strip(),
        ]

    def _build_benchmark_tab(self, master: ttk.Notebook) -> ttk.Frame:
        panel = CommandPanel(master, "Tracking Benchmark", self._benchmark_command)
        self.bm_weights = panel.add_row("Weights", str(self.project_root / "runs/detect/train/weights/best.pt"), browse_file=True)
        self.bm_videos = panel.add_row("Videos directory", str(self.project_root / "datasets/raw_videos"), browse_dir=True)
        self.bm_trackers = panel.add_row("Trackers", "sort bytetrack opencv custom")
        self.bm_output = panel.add_row("Report .md", str(self.project_root / "outputs/reports/benchmark.md"))
        return panel

    def _benchmark_command(self) -> list[str]:
        trackers = [t.strip() for t in self.bm_trackers.get().split() if t.strip()]
        if len(trackers) < 1:
            messagebox.showerror("Trackers missing", "Specify at least one tracker.")
            return []
        return [
            sys.executable,
            "scripts/run_benchmark.py",
            "--weights",
            self.bm_weights.get().strip(),
            "--videos",
            self.bm_videos.get().strip(),
            "--trackers",
            *trackers,
            "--output",
            self.bm_output.get().strip(),
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch AimVision GUI.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    app = AimVisionApp(project_root=args.project_root.resolve())
    app.mainloop()


if __name__ == "__main__":
    main()
