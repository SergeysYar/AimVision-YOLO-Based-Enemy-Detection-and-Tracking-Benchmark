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
python -m venv .venv
# Windows
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Основные команды
Извлечение кадров:
```bash
python src/data/extract_frames.py --video datasets/raw_videos/demo.mp4 --output datasets/frames/demo
```

Обучение:
```bash
python src/detection/train_yolo.py --data datasets/dataset.yaml --model yolov8n.pt --epochs 50 --imgsz 640
```

Инференс:
```bash
python src/detection/infer_yolo.py --weights runs/detect/train/weights/best.pt --video datasets/raw_videos/demo.mp4 --output outputs/videos/demo_detected.mp4
```

Бенчмарк трекеров:
```bash
python scripts/run_benchmark.py --weights runs/detect/train/weights/best.pt --videos datasets/raw_videos --trackers sort bytetrack opencv custom --output outputs/reports/benchmark.md
```

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
