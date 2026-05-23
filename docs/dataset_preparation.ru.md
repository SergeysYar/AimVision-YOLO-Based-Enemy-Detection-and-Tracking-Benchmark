# Подготовка датасета

## 1. Сбор исходных материалов
- Используйте собственные записи геймплея, открытые benchmark-видео или синтетические ролики.
- Сохраняйте видео в `datasets/raw_videos/`.

## 2. Извлечение кадров
```bash
uv run python src/data/extract_frames.py --video datasets/raw_videos/demo.mp4 --output datasets/frames/demo --every-n 2
```

## 3. Разметка в формате YOLO
Каждому кадру соответствует `.txt`:
`class_id x_center y_center width height` (координаты нормализованы в диапазоне `[0,1]`).

Классы:
- `0 enemy`
- `1 teammate`
- `2 weapon`
- `3 head`

## 4. Разделение на train/val/test
```bash
uv run python src/data/split_dataset.py --frames datasets/frames/demo --labels datasets/labels --dataset-root datasets
```

## 5. Проверка dataset.yaml
Перед обучением проверьте `datasets/dataset.yaml`.
