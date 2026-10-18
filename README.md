# 🖐️ HandReader (HandSight v2)

Высокопроизводительная модульная система распознавания жестов рук и бесконтактного управления компьютером в реальном времени. Построена на базе современного **MediaPipe Tasks Vision API** и легковесных моделей **TensorFlow Lite (MLP)**.

Система позволяет управлять курсором мыши указательным пальцем, выполнять клики смыканием пальцев без смещения прицела, распознавать статические жесты, а также обучать собственные модели.

---

## 🚀 Особенности

- **Управление ПК без инерции**:
  - Указательный палец — наведение курсора.
  - Большой + Средний пальцы — **ЛКМ** (клик, двойной клик, Drag-and-Drop).
  - Большой + Безымянный пальцы — **ПКМ** (контекстное меню).
  - Алгоритм **One Euro Filter** + мягкая мертвая зона: нулевая задержка при движении и полное отсутствие дрожания в покое.
- **Современный пайплайн MediaPipe**: использует актуальный `HandLandmarker` в режиме потокового видео (`RunningMode.VIDEO`).
- **Инвариантный препроцессинг**: координаты 21 ключевой точки нормализуются относительно запястья и масштабируются по максимальному размаху.
- **Устойчивость к освещению**: встроенная предобработка через **CLAHE** в цветовом пространстве LAB.
- **Полный цикл MLOps**:
  - `collect_data.py` — интерактивный сборщик датасета по горячим клавишам `0`–`9`.
  - `train_model.py` — обучение компактной MLP-сети на Keras за ~5 секунд с автоматическим квантованием/экспортом в `.tflite`.
  - `main_demo.py` — запуск с визуализацией прицела, линий смыкания пальцев, рамки экрана и FPS в разрешении 1280×720.

---

## 📂 Структура проекта

```text
HandReader/
├── models/
│   ├── hand_landmarker.task             # Базовая модель лендмарков MediaPipe
│   ├── keypoint_classifier.tflite       # Обученная модель классификации жестов
│   ├── keypoint_classifier_label.csv    # Список названий классов жестов
│   └── keypoint_dataset.csv             # Сохраненный датасет (генерируется)
├── action_controller.py                 # Контроллер мыши, фильтрация координат и жесты кликов
├── collect_data.py                      # Скрипт записи обучающих примеров
├── train_model.py                       # Обучение и экспорт модели в TFLite
├── gesture_engine.py                    # Изолированное ядро инференса и отрисовки
├── main_demo.py                         # Главный скрипт демонстрации и управления
└── requirements.txt                     # Зависимости проекта
```

---

## 🛠️ Установка и настройка

### 1. Клонирование репозитория

```bash
git clone https://github.com/vergilhimself/HandReader.git
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

Для работы трекера рук необходим файл модели `hand_landmarker.task`. Скачайте его и поместите в папку `models/`:

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

### 1. Управление мышью и демонстрация

Запустите главный скрипт:

```bash
python main_demo.py
```

- **Управление курсором**: перемещайте указательный палец внутри зеленой рамки (Active Screen Area).
- **ЛКМ**: сомкните подушечки большого и среднего пальцев.
- **ПКМ**: сомкните подушечки большого и безымянного пальцев.
- **Клавиша `m`**: включить / выключить управление мышью.
- **Клавиша `q`**: выход из приложения.

---

### 2. Сбор собственного датасета жестов

1. Откройте `collect_data.py` и задайте нужные названия жестов:
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
3. Покажите жест перед камерой и **зажмите соответствующую цифру (`0`–`9`)** на клавиатуре. Набирайте ~150–300 сэмплов на каждый жест под разными углами.
4. Нажмите `q` для сохранения данных в `models/keypoint_dataset.csv`.

---

### 3. Обучение классификатора

После записи обновите список `LABELS` в файле `train_model.py` и запустите обучение:

```bash
python train_model.py
```

Скрипт автоматически обучит нейросеть, создаст файл меток `keypoint_classifier_label.csv` и экспортирует готовую модель в `models/keypoint_classifier.tflite`.

---

## 🧠 Архитектура пайплайна

```text
[ Веб-камера (1280x720) ] 
            │
            ▼
[ Предобработка: CLAHE (выравнивание освещения) ]
            │
            ▼
[ MediaPipe Tasks: HandLandmarker ] ──► 21 ключевая точка руки
            │
            ├──► [ Указательный палец ] ──► One Euro Filter ──► Курсор мыши
            ├──► [ Большой + Средний ]  ──► Порог смыкания  ──► ЛКМ (Click/Drag)
            ├──► [ Большой + Безымянный]──► Порог смыкания  ──► ПКМ
            │
            ▼
[ Нормализация лендмарков ] ──► TFLite MLP-классификатор ──► Название жеста
```

---

## 📋 Требования к окружению

- **Python**: `3.10` – `3.12`
- **Камера**: любая веб-камера с поддержкой 720p/1080p.
- Зависимости: `mediapipe`, `opencv-python`, `tensorflow`, `pynput`, `pyautogui`, `scikit-learn`, `numpy`.

---

## 📄 Лицензия

Проект распространяется под лицензией [MIT](LICENSE).
