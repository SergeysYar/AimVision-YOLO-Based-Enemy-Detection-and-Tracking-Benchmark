# AimVision: YOLO-бенчмарк детекции и трекинга

Офлайн-проект по компьютерному зрению для детекции и трекинга игровых сущностей (в стиле Counter-Strike) на видео.

## Важно: безопасность и ограничения
Этот проект **не является** читом, аимботом или системой автоматизации игры.

- Нет управления мышью/клавиатурой
- Нет чтения памяти процесса
- Нет DLL/process injection
- Нет обхода античита
- Нет взаимодействия с запущенной игрой

Проект работает только с офлайн-видео, скриншотами и синтетическими роликами.

## Возможности
- Подготовка датасета из видеозаписей
- Обучение детектора на базе YOLO (Ultralytics)
- Инференс по видео с аннотированным выводом
- Трекинг несколькими методами:
  - SORT
  - ByteTrack-подобный подход
  - OpenCV CSRT/KCF
  - Кастомный трекер (с нуля)
- Сравнение трекеров на одном наборе видео
- Экспорт:
  - аннотированные видео
  - CSV-логи треков
  - Markdown/CSV отчёты с метриками

## Структура датасета
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

## Установка
```bash
uv sync
```

## Основные команды
Извлечение кадров:
```bash
uv run python src/data/extract_frames.py --video datasets/raw_videos/demo.mp4 --output datasets/frames/demo
```

Обучение:
```bash
uv run python src/detection/train_yolo.py --data datasets/dataset.yaml --model yolov8n.pt --epochs 50 --imgsz 640
```

Инференс:
```bash
uv run python src/detection/infer_yolo.py --weights runs/detect/train/weights/best.pt --video datasets/raw_videos/demo.mp4 --output outputs/videos/demo_detected.mp4
```

Бенчмарк трекеров:
```bash
uv run python scripts/run_benchmark.py --weights runs/detect/train/weights/best.pt --videos datasets/raw_videos --trackers sort bytetrack opencv custom --output outputs/reports/benchmark.md
```

## GUI (центр управления)
```bash
uv run python scripts/run_gui.py
```

Вкладки:
- `Dataset`: извлечение кадров из видео.
- `Training`: запуск обучения YOLO.
- `Inference`: инференс и экспорт аннотированного видео.
- `Benchmark`: сравнение трекеров.
- `Annotation Studio`: отдельная рабочая зона для разметки датасета.

Интеллектуальные функции в `Annotation Studio`:
- `Auto-annotate with YOLO` для авторазметки текущего кадра.
- `Smart Copy Box to Next Frame` для быстрого переноса бокса на следующий кадр.

## Метрики
- Average FPS
- Lost tracks
- ID switches
- Average detection confidence
- Average track duration
- Re-identification after occlusion
- Processing time per frame
- Detection success rate

## Дополнительная документация
- Подготовка датасета: `docs/dataset_preparation.md`
- Математическая модель: `docs/math_model.md`
- Шаблон отчёта: `docs/benchmark_results_template.md`

## Назначение
Проект предназначен для обучения, исследований и портфолио в области CV/ML.
