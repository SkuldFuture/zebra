# Структура проекта

```
project/
├── train.py                # Скрипт обучения, валидации и экспорта
├── annotation.py           # Пример использования модели для инференса
├── yolo11n.pt              # Базовые веса YOLO
├── augmented_dataset_old/  # Датасет с аугментацией
├── runs_dishes/            # Папки с результатами обучения
└── requirements.txt        # Список зависимостей
```


# Как запускать

1. Нужно создать виртуальное окружение и установить все необходимые библиотеки:

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Запустить обучение:

   ```bash
   python train.py
   ```
