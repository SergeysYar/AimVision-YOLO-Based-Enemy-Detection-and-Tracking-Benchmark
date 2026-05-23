# AimVision: YOLO-бенчмарк детекции и трекинга

Офлайн-проект по компьютерному зрению для детекции и трекинга игровых сущностей (в стиле FPS) на записанных видео.

## Важно: безопасность
Проект **не является** читом, аимботом или автоматизацией игры.

- Нет управления мышью/клавиатурой
- Нет чтения памяти процесса
- Нет DLL/process injection
- Нет обхода античита
- Нет взаимодействия с запущенной игрой

Использование только для офлайн-анализа, обучения и портфолио.

## Быстрый старт (uv)
1. Установите `uv`: https://docs.astral.sh/uv/
2. Установите зависимости:
```bash
uv sync
```
3. Запустите GUI:
```bash
uv run python scripts/run_gui.py
```

## Основной workflow (CLI)
Извлечение кадров:
```bash
uv run python src/data/extract_frames.py --video datasets/raw_videos/demo.mp4 --output datasets/frames/demo --every-n 2
```

Сплит датасета:
```bash
uv run python src/data/split_dataset.py --frames datasets/frames/demo --labels datasets/labels --dataset-root datasets
```

Обучение:
```bash
uv run python src/detection/train_yolo.py --data datasets/dataset.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --lr0 0.01
```

Инференс:
```bash
uv run python src/detection/infer_yolo.py --weights runs/detect/train/weights/best.pt --video datasets/raw_videos/demo.mp4 --output outputs/videos/demo_detected.mp4 --conf 0.25
```

Бенчмарк трекеров:
```bash
uv run python scripts/run_benchmark.py --weights runs/detect/train/weights/best.pt --videos datasets/raw_videos --trackers sort bytetrack opencv custom --output outputs/reports/benchmark.md
```

## GUI (центр управления)
Запуск:
```bash
uv run python scripts/run_gui.py
```

Вкладки:
- `Dataset`: извлечение кадров
- `Training`: обучение YOLO
- `Inference`: детекция на одном видео
- `Benchmark`: сравнение трекеров
- `Annotation Studio`: разметка датасета

Интеллектуальные функции в `Annotation Studio`:
- `Auto-annotate with YOLO`
- `Smart Copy Box to Next Frame`

## Метрики
- Average FPS
- Lost tracks
- ID switches
- Average detection confidence
- Average track duration
- Re-identification after occlusion
- Processing time per frame
- Detection success rate

## Индекс документации
- Основной README: `../README.md`
- Подготовка датасета: `dataset_preparation.md`
- Математическая модель: `math_model.md`
- Описание проекта: `project_description.md`
- Шаблон отчета: `benchmark_results_template.md`
- Гайд по GUI: `gui_guide.md`
- Troubleshooting: `troubleshooting.md`
