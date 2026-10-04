import cv2 as cv
import time
from gesture_engine import GestureEngine

def main():
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    # Инициализируем изолированное ядро
    engine = GestureEngine(
        keypoint_model_path="models/keypoint_classifier.tflite",
        keypoint_labels_path="models/keypoint_classifier_label.csv",
        # history_model_path="models/point_history_classifier.tflite", # Раскомментируйте, если используете
    )

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Зеркалим кадр для естественного восприятия человеком
        frame = cv.flip(frame, 1)

        # Обрабатываем кадр
        result = engine.process(frame, draw_overlay=True)

        # Расчет FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        # Вывод FPS на экран
        cv.putText(
            result.annotated_image,
            f"FPS: {int(fps)}",
            (20, 40),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        # В консоль можно получать данные без GUI
        for hand in result.hands:
            if hand.keypoint_label != "Unknown":
                print(f"Жест: {hand.keypoint_label} ({hand.handedness})")

        cv.imshow("HandSight v2 Core Demo", result.annotated_image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()