import csv
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Список названий в строгом порядке индексов 0, 1, 2...
LABELS = [
    "Open Palm",   # Класс 0
    "Fist",        # Класс 1
    "Peace",       # Класс 2
    "Thumbs Up",   # Класс 3
    "Pointing"     # Класс 4
]

dataset_path = "models/keypoint_dataset.csv"
model_save_path = "models/keypoint_classifier.tflite"
labels_save_path = "models/keypoint_classifier_label.csv"

def main():
    if not Path(dataset_path).exists():
        print(f"Ошибка: файл {dataset_path} не найден. Сначала запустите collect_data.py!")
        return

    # 1. Загрузка данных
    raw_data = np.loadtxt(dataset_path, delimiter=",", dtype=np.float32)
    if raw_data.ndim == 1:
        raw_data = np.expand_dims(raw_data, axis=0)

    y = raw_data[:, 0].astype(int)
    X = raw_data[:, 1:]

    num_classes = len(LABELS)
    print(f"Загружено примеров: {len(X)}")
    print(f"Количество классов: {num_classes}")

    # Разделение выборки на обучение и валидацию (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # 2. Архитектура нейросети (MLP для 42 координат)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(42,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 3. Обучение (обычно длится ~5–10 секунд)
    print("\nСтарт обучения...")
    model.fit(
        X_train, y_train,
        epochs=40,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nТочность на проверочной выборке: {test_acc * 100:.2f}%")

    # 4. Сохранение файла меток (CSV)
    with open(labels_save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for label in LABELS:
            writer.writerow([label])
    print(f"Файл меток сохранен: {labels_save_path}")

    # 5. Экспорт в TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(model_save_path, "wb") as f:
        f.write(tflite_model)
    print(f"Новая модель сохранена: {model_save_path}")

if __name__ == "__main__":
    main()