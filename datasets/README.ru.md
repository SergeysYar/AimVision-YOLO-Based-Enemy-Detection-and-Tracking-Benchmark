# Руководство по датасету

Структура:

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

1. Поместите исходные видео в `datasets/raw_videos/`.
2. Извлеките кадры:
   `uv run python src/data/extract_frames.py --video datasets/raw_videos/demo.mp4 --output datasets/frames/demo`
3. Разметьте кадры в YOLO-формате.
4. Сохраните `.txt`-метки в `datasets/labels/`.
5. Выполните split:
   `uv run python src/data/split_dataset.py --frames datasets/frames/demo --labels datasets/labels --dataset-root datasets`

Этичный сбор данных:
- Используйте только материалы, на которые у вас есть права.
- Не публикуйте персональные данные.
