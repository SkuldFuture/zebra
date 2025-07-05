# Структура проекта

```
zebra/
└───REPORT_MD             # Папка с Markdown файлом отчета о проделанной работе 
        output_13_1.png
        output_19_1.png
        output_22_1.png
        output_24_1.png
        output_26_1.png
        output_29_1.png
        output_5_1.png
        output_6_1.png
        output_7_1.png
        output_8_1.png
        REPORT.md
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

2. Опционально: для воспроизведения ячеек Jupyter необходимо поместить оригинальный 
датасет с видео в папку `dataset`

2. Запустить обучение:

   ```bash
   python train.py
   ```

<i style="color: blue;">Честно времени ушло около 11 часов в субботу и по паре часов в будни,
потому что было желание играться с методами разбиения видео (датасет 
порой важнее гиперпараметров) и вспоминать теорию</i>
