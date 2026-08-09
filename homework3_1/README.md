# Домашнее задание №3

## Задумка:
1) вычитать датасет через API и сохранить данные БД
2) выполнить ряд sql-запросов
3) визуализировать исходный датасет
4) применить к датасету алгоритмы классификации и сравнить результаты

```
HOMEWORK2/
├── data/
│   ├── personality_synthetic_dataset.csv   - датасет типов личности
│   ├── db_screen.png                       - скриншот БД
│   └── data_analysis_report.html           - отчет профилирования ds_4_regression_for_boston_houses
├── private/               - папка для вспоиогательных ресурсов
│   └── utils.py           - методы для вычитки датасетов, менеджеры контекстов и т.д. 
├── public/                - универсальные классы
│   ├── data_loader_utils.py          - загрузчик
│   ├── classification_utils.py       - утилиты для models.py
│   ├── models.py                     - класс "ClassificationProcessor"
│   └── log_utils.py                  - логеры
├── ds_3_1_load_and_create_tables.ipynb  - получение по API данных, создание и заполнение таблиц в БД
├── ds_3_2_query_examples.ipynb          - примеры sql-query на базе созданных таблиц
├── ds_3_3_profolong.ipynb               - предварительный анализ датасета путем профилирования
├── ds_3_4_visualization.ipynb           - применение к датасету алгоритмов классияикации с визуализацией
└── README.md
```
