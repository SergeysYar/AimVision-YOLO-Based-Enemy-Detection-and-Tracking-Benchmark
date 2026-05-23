"""Dataset annotation tab with manual and assisted labeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ["enemy", "teammate", "weapon", "head"]


@dataclass
class Box:
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int


class AnnotationTab(ttk.Frame):
    """Annotation workspace with model-assisted utilities."""

    def __init__(self, master: tk.Misc, project_root: Path) -> None:
        super().__init__(master, padding=10)
        self.project_root = project_root
        self.images_dir = project_root / "datasets/frames"
        self.labels_dir = project_root / "datasets/labels"
        self.images: list[Path] = []
        self.index = 0
        self.current_img = None
        self.current_shape = (1, 1)
        self.boxes: list[Box] = []
        self.start_xy: tuple[int, int] | None = None
        self.temp_rect = None
        self.model = None
        self.last_selected: Box | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        root = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root, width=350, padding=8)
        right = ttk.Frame(root, padding=8)
        root.add(left, weight=0)
        root.add(right, weight=1)

        ttk.Label(left, text="Annotation Studio", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))
        path_row = ttk.Frame(left)
        path_row.pack(fill="x", pady=2)
        self.images_entry = ttk.Entry(path_row)
        self.images_entry.insert(0, str(self.images_dir))
        self.images_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(path_row, text="...", width=4, command=self._pick_images_dir).pack(side="left")
        ttk.Button(left, text="Load Images", command=self.load_images).pack(fill="x", pady=4)

        label_row = ttk.Frame(left)
        label_row.pack(fill="x", pady=2)
        self.labels_entry = ttk.Entry(label_row)
        self.labels_entry.insert(0, str(self.labels_dir))
        self.labels_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(label_row, text="...", width=4, command=self._pick_labels_dir).pack(side="left")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left, text="Class").pack(anchor="w")
        self.class_var = tk.StringVar(value=CLASSES[0])
        self.class_box = ttk.Combobox(left, textvariable=self.class_var, values=CLASSES, state="readonly")
        self.class_box.pack(fill="x", pady=4)

        nav = ttk.Frame(left)
        nav.pack(fill="x", pady=4)
        ttk.Button(nav, text="Prev", command=self.prev_image).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(nav, text="Next", command=self.next_image).pack(side="left", fill="x", expand=True)

        ttk.Button(left, text="Save Labels", command=self.save_labels).pack(fill="x", pady=4)
        ttk.Button(left, text="Delete Selected Box", command=self.delete_last).pack(fill="x", pady=4)
        ttk.Button(left, text="Clear All Boxes", command=self.clear_boxes).pack(fill="x", pady=4)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)
        ttk.Label(left, text="Intelligent Features").pack(anchor="w")
        self.weights_entry = ttk.Entry(left)
        self.weights_entry.insert(0, str(self.project_root / "runs/detect/train/weights/best.pt"))
        self.weights_entry.pack(fill="x", pady=4)
        ttk.Button(left, text="Auto-annotate with YOLO", command=self.auto_annotate).pack(fill="x", pady=4)
        ttk.Button(left, text="Smart Copy Box to Next Frame", command=self.smart_copy_to_next).pack(fill="x", pady=4)

        self.info_lbl = ttk.Label(left, text="No images loaded.", justify="left")
        self.info_lbl.pack(anchor="w", pady=(12, 0))

        self.canvas = tk.Canvas(right, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def _pick_images_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.images_entry.delete(0, tk.END)
            self.images_entry.insert(0, path)

    def _pick_labels_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.labels_entry.delete(0, tk.END)
            self.labels_entry.insert(0, path)

    def load_images(self) -> None:
        self.images_dir = Path(self.images_entry.get().strip())
        self.labels_dir = Path(self.labels_entry.get().strip())
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.images = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        self.index = 0
        if not self.images:
            messagebox.showwarning("No images", f"Images not found in {self.images_dir}")
            return
        self._load_current()

    def _load_current(self) -> None:
        if not self.images:
            return
        img_path = self.images[self.index]
        img = cv2.imread(str(img_path))
        if img is None:
            messagebox.showerror("Image error", f"Cannot read image:\n{img_path}")
            return
        self.current_img = img
        self.current_shape = (img.shape[1], img.shape[0])
        self.boxes = self._read_labels(img_path)
        self._render_canvas()
        self._update_info()

    def _update_info(self) -> None:
        if not self.images:
            self.info_lbl.configure(text="No images loaded.")
            return
        text = (
            f"Image: {self.images[self.index].name}\n"
            f"Index: {self.index + 1}/{len(self.images)}\n"
            f"Boxes: {len(self.boxes)}"
        )
        self.info_lbl.configure(text=text)

    def _render_canvas(self) -> None:
        if self.current_img is None:
            return
        rgb = cv2.cvtColor(self.current_img.copy(), cv2.COLOR_BGR2RGB)
        for b in self.boxes:
            cv2.rectangle(rgb, (b.x1, b.y1), (b.x2, b.y2), (80, 240, 80), 2)
            cls = CLASSES[b.class_id] if 0 <= b.class_id < len(CLASSES) else str(b.class_id)
            cv2.putText(rgb, cls, (b.x1, max(18, b.y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 240, 80), 2)

        h, w = rgb.shape[:2]
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        scale = min(canvas_w / w, canvas_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        png = cv2.imencode(".png", resized)[1].tobytes()
        self.tk_img = tk.PhotoImage(data=png)
        self.canvas.delete("all")
        self.canvas.create_image((canvas_w - nw) // 2, (canvas_h - nh) // 2, image=self.tk_img, anchor="nw")
        self.display_scale = scale
        self.display_offset = ((canvas_w - nw) // 2, (canvas_h - nh) // 2)

    def _canvas_to_image_xy(self, x: int, y: int) -> tuple[int, int]:
        ox, oy = self.display_offset
        ix = int((x - ox) / max(self.display_scale, 1e-6))
        iy = int((y - oy) / max(self.display_scale, 1e-6))
        iw, ih = self.current_shape
        ix = max(0, min(iw - 1, ix))
        iy = max(0, min(ih - 1, iy))
        return ix, iy

    def on_mouse_down(self, event) -> None:
        if self.current_img is None:
            return
        self.start_xy = self._canvas_to_image_xy(event.x, event.y)

    def on_mouse_drag(self, event) -> None:
        if self.current_img is None or self.start_xy is None:
            return
        self._render_canvas()
        x0, y0 = self.start_xy
        x1, y1 = self._canvas_to_image_xy(event.x, event.y)
        sx = int(x0 * self.display_scale + self.display_offset[0])
        sy = int(y0 * self.display_scale + self.display_offset[1])
        ex = int(x1 * self.display_scale + self.display_offset[0])
        ey = int(y1 * self.display_scale + self.display_offset[1])
        self.temp_rect = self.canvas.create_rectangle(sx, sy, ex, ey, outline="#ffcc00", width=2)

    def on_mouse_up(self, event) -> None:
        if self.current_img is None or self.start_xy is None:
            return
        x0, y0 = self.start_xy
        x1, y1 = self._canvas_to_image_xy(event.x, event.y)
        x1n, x2n = sorted([x0, x1])
        y1n, y2n = sorted([y0, y1])
        if (x2n - x1n) < 6 or (y2n - y1n) < 6:
            self.start_xy = None
            self._render_canvas()
            return
        class_id = CLASSES.index(self.class_var.get()) if self.class_var.get() in CLASSES else 0
        box = Box(class_id=class_id, x1=x1n, y1=y1n, x2=x2n, y2=y2n)
        self.boxes.append(box)
        self.last_selected = box
        self.start_xy = None
        self._render_canvas()
        self._update_info()

    def prev_image(self) -> None:
        if not self.images:
            return
        self.save_labels()
        self.index = max(0, self.index - 1)
        self._load_current()

    def next_image(self) -> None:
        if not self.images:
            return
        self.save_labels()
        self.index = min(len(self.images) - 1, self.index + 1)
        self._load_current()

    def delete_last(self) -> None:
        if self.boxes:
            self.boxes.pop()
            self._render_canvas()
            self._update_info()

    def clear_boxes(self) -> None:
        self.boxes = []
        self._render_canvas()
        self._update_info()

    def _label_path(self, image_path: Path) -> Path:
        return self.labels_dir / f"{image_path.stem}.txt"

    def _read_labels(self, image_path: Path) -> list[Box]:
        label_path = self._label_path(image_path)
        if not label_path.exists():
            return []
        iw, ih = self.current_shape
        out: list[Box] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = parts
            cls_id = int(float(cls))
            xc_f, yc_f, w_f, h_f = map(float, (xc, yc, w, h))
            x1 = int((xc_f - w_f / 2.0) * iw)
            y1 = int((yc_f - h_f / 2.0) * ih)
            x2 = int((xc_f + w_f / 2.0) * iw)
            y2 = int((yc_f + h_f / 2.0) * ih)
            out.append(Box(cls_id, x1, y1, x2, y2))
        return out

    def save_labels(self) -> None:
        if not self.images:
            return
        img_path = self.images[self.index]
        iw, ih = self.current_shape
        lines: list[str] = []
        for b in self.boxes:
            xc = ((b.x1 + b.x2) / 2.0) / iw
            yc = ((b.y1 + b.y2) / 2.0) / ih
            w = (b.x2 - b.x1) / iw
            h = (b.y2 - b.y1) / ih
            lines.append(f"{b.class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        self._label_path(img_path).write_text("\n".join(lines), encoding="utf-8")
        self._update_info()

    def _load_model(self):
        if self.model is not None:
            return self.model
        if YOLO is None:
            raise RuntimeError("Ultralytics not installed.")
        weights = Path(self.weights_entry.get().strip())
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        self.model = YOLO(str(weights))
        return self.model

    def auto_annotate(self) -> None:
        if not self.images or self.current_img is None:
            return
        try:
            model = self._load_model()
            result = model.predict(self.current_img, conf=0.2, verbose=False)[0]
            self.boxes = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy()
                for box, cls_id in zip(boxes, clss):
                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                    mapped_class = int(cls_id) if int(cls_id) < len(CLASSES) else 0
                    self.boxes.append(Box(mapped_class, x1, y1, x2, y2))
            self._render_canvas()
            self._update_info()
            self.save_labels()
        except Exception as exc:
            messagebox.showerror("Auto-annotation error", str(exc))

    def smart_copy_to_next(self) -> None:
        if not self.images or self.index >= len(self.images) - 1:
            return
        source = self.last_selected if self.last_selected is not None else (self.boxes[-1] if self.boxes else None)
        if source is None:
            messagebox.showinfo("No source box", "Create/select at least one box first.")
            return
        self.save_labels()
        self.index += 1
        self._load_current()
        cloned = Box(source.class_id, source.x1, source.y1, source.x2, source.y2)
        self.boxes.append(cloned)
        self.last_selected = cloned
        self._render_canvas()
        self._update_info()
        self.save_labels()
