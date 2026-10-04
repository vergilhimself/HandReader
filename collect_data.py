import os
import csv
from collections import Counter
from pathlib import Path
import cv2 as cv
from gesture_engine import GestureEngine

# Задайте соответствие клавиш и названий ваших жестов
GESTURE_NAMES = {
    0: "Open Palm",
    1: "Fist",
    2: "Peace",
    3: "Thumbs Up",
    4: "Pointing"
}


def main():
    Path("models").mkdir(parents=True, exist_ok=True)
    csv_path = "models/keypoint_dataset.csv"

    # Подсчет уже имеющихся записей в файле, если он существует
    counts = Counter()
    if Path(csv_path).exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row:
                    counts[int(row[0])] += 1

    cap = cv.VideoCapture(0)
    engine = GestureEngine()

    print("=== РЕЖИМ ЗАПИСИ ДАТАСЕТА ===")
    print("Зажмите цифру 0..9, чтобы непрерывно записывать жест.")
    print("Нажмите 'q' для выхода.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        result = engine.process(frame, draw_overlay=True)
        img = result.annotated_image

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Если нажата цифра 0..9 и в кадре обнаружена рука
        if ord('0') <= key <= ord('9') and result.hands:
            class_id = int(chr(key))
            landmarks = result.hands[0].landmarks

            # Получаем 42 нормализованных числа
            pre_processed = GestureEngine._pre_process_landmarks(landmarks)

            # Сохраняем в CSV: [class_id, x0, y0, ..., x20, y20]
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([class_id, *pre_processed])

            counts[class_id] += 1

            # Индикатор записи (зеленый кружок в углу)
            cv.circle(img, (img.shape[1] - 30, 30), 12, (0, 255, 0), -1)

        # Вывод статистики записанных кадров на экран
        y_offset = 30
        cv.putText(img, "Collected samples:", (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for cid, name in GESTURE_NAMES.items():
            y_offset += 25
            cnt = counts[cid]
            color = (0, 255, 0) if cnt >= 150 else (0, 165, 255)
            cv.putText(img, f"[{cid}] {name}: {cnt}", (10, y_offset), cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        cv.imshow("Dataset Collector", img)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()