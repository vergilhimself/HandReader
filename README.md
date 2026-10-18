# 🖐️ HandReader (HandSight v2)

Высокопроизводительная модульная система распознавания статических жестов рук в реальном времени. Построена на базе современного **MediaPipe Tasks Vision API** и легковесных моделей **TensorFlow Lite (MLP)**.

Система устойчива к перепадам освещения, инвариантна к масштабу и положению руки в кадре, а также включает полный цикл: от интерактивного сбора датасета до обучения и вывода инференса с Teachable Machine-style HUD.

---

## 🚀 Особенности

- **Современный пайплайн MediaPipe**: использует актуальный `mediapipe.tasks.python.vision.HandLandmarker` в режиме потокового видео (`RunningMode.VIDEO`).
- **Инвариантный препроцессинг**: координаты 21 ключевой точки нормализуются относительно основания ладони (запястья) и масштабируются по максимальному отклонению, что исключает зависимость от расстояния до камеры.
- **Устойчивость к плохому освещению**: встроенная предобработка через **CLAHE** (Contrast Limited Adaptive Histogram Equalization) в цветовом пространстве LAB.
- **Полный цикл MLOps**:
  - `collect_data.py` — интерактивный сборщик датасета прямо с веб-камеры по горячим клавишам.
  - `train_model.py` — обучение компактной MLP-сети на Keras за ~5 секунд с автоматическим квантованием/экспортом в `.tflite`.
  - `main_demo.py` — боевой запуск с отрисовкой скелета, вероятностей каждого класса и FPS.
- **Минимальный оверхед**: инференс на CPU занимает доли миллисекунды, обеспечивая стабильные 30–60+ FPS.

---

## 📂 Структура проекта
HandReader/
├── models/
│   ├── hand_landmarker.task             # Базовая модель лендмарков MediaPipe
│   ├── keypoint_classifier.tflite       # Обученная модель классификации жестов
│   ├── keypoint_classifier_label.csv    # Список названий классов жестов
│   └── keypoint_dataset.csv             # Сохраненный датасет (генерируется)
├── collect_data.py                      # Скрипт записи обучающих примеров
├── train_model.py                       # Обучение и экспорт модели в TFLite
├── gesture_engine.py                    # Изолированное ядро инференса и отрисовки
├── main_demo.py                         # Главный демонстрационный скрипт
└── requirements.txt                     # Зависимости проекта
```
---

## 🛠️ Установка и настройка

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-username/HandReader.git
cd HandReader
```

### 2. Создание и активация виртуального окружения

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Скачивание базовой модели MediaPipe

Для работы трекера рук необходим файл модели `hand_landmarker.task`. Скачайте его и поместите в директорию `models/`:

- **Прямая ссылка:** [Google MediaPipe Hand Landmarker Model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

**Linux / macOS (через curl):**
```bash
mkdir -p models
curl -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Force -Path models
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "models/hand_landmarker.task"
```

---

## 🎮 Использование

### 1. Демонстрация (готов из коробки)

Запустите распознавание с уже имеющейся моделью:

```bash
python main_demo.py
```

*Нажмите `q` для выхода из приложения.*

---

### 2. Сбор собственного датасета

Вы можете переобучить модель на свои жесты или добавить новые.

1. Откройте `collect_data.py` и при необходимости скорректируйте словарь `GESTURE_NAMES`:
   ```python
   GESTURE_NAMES = {
       0: "Open Palm",
       1: "Fist",
       2: "Peace",
       3: "Thumbs Up",
       4: "Pointing"
   }
   ```
2. Запустите сборщик:
   ```bash
   python collect_data.py
   ```
3. Покажите жест перед камерой и **зажмите соответствующую цифру (`0`–`9`)** на клавиатуре. Набирайте ~150–300 сэмплов на каждый жест под разными углами и расстояниями.
4. Нажмите `q` для сохранения в `models/keypoint_dataset.csv`.

---

### 3. Обучение классификатора

После сбора данных обновите список `LABELS` в файле `train_model.py` (порядок индексов должен совпадать с `collect_data.py`) и запустите обучение:

```bash
python train_model.py
```

Скрипт:
- Разделит данные на обучающую (80%) и валидационную (20%) выборки;
- Обучит компактную нейросеть (Dense 64 → Dropout 0.2 → Dense 32 → Softmax);
- Сгенерирует `models/keypoint_classifier_label.csv`;
- Экспортирует оптимизированную модель в `models/keypoint_classifier.tflite`.

После завершения просто перезапустите `python main_demo.py`!

---

## 🧠 Архитектура пайплайна

```text
[ Камера (RGB) ] 
       │
       ▼
[ Предобработка: CLAHE (выравнивание контраста) ]
       │
       ▼
[ MediaPipe Tasks: HandLandmarker ] ──► Извлечение 21 точки (x, y)
       │
       ▼
[ Нормализация ] ──► Смещение к точке 0 (Wrist) + деление на max(|v|)
       │
       ▼
[ TFLite Classifier (MLP) ] ──► Вектор вероятностей классов (Softmax)
       │
       ▼
[ HUD Отрисовка ] ──► Скелет руки + Индикатор вероятностей Teachable Machine
```

---

## 📋 Требования к окружению

- **Python**: `3.10` – `3.12`
- **Камера**: любая стандартная веб-камера (720p/1080p).
- Библиотеки:
  - `mediapipe >= 0.10.0`
  - `opencv-python >= 4.8.0`
  - `tensorflow >= 2.15.0`
  - `scikit-learn`, `numpy`

---

## 🗺️ Дорожная карта (Roadmap)

- [ ] Экспоненциальное сглаживание / фильтр Калмана (OneEuroFilter) для устранения дрожания ключевых точек.
- [ ] Активация `point_history_classifier` для динамических жестов (свайпы, круговые движения).
- [ ] Поддержка одновременного распознавания двух рук с независимыми триггерами.
- [ ] Модуль управления виртуальной мышью и горячими клавишами системы (`pyautogui`).

---

## 📄 Лицензия

Проект распространяется под лицензией [MIT](LICENSE). Вы можете свободно модифицировать и использовать его в учебных и коммерческих целях.
