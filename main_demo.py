import cv2 as cv
import time
from gesture_engine import GestureEngine
from action_controller import ActionController

WINDOW_TITLE = "HandSight Dual-Click Mouse (1280x720)"

def main():
    cap = cv.VideoCapture(0)
    # Задаем камере режим 1280x720
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    # Принудительно создаем масштабируемое окно нужного размера
    cv.namedWindow(WINDOW_TITLE, cv.WINDOW_NORMAL)
    cv.resizeWindow(WINDOW_TITLE, 1280, 720)

    engine = GestureEngine(
        keypoint_model_path="models/keypoint_classifier.tflite",
        keypoint_labels_path="models/keypoint_classifier_label.csv",
    )

    action_ctrl = ActionController(
        box_margin_x=0.18,
        box_margin_y=0.18,
        click_press_ratio=0.32,
        click_release_ratio=0.42,
        beta=0.65,        # Высокая отзывчивость
        deadzone=0.0022   # Точная фиксация при остановке
    )

    prev_time = time.time()
    mouse_control_enabled = True

    print("=== УПРАВЛЕНИЕ МЫШЬЮ ОБНОВЛЕНО ===")
    print("Указательный: Точное наведение (без инерции)")
    print("Большой + Средний:   ЛКМ")
    print("Большой + Безымянный: ПКМ")
    print("'m' — вкл/выкл мышь. 'q' — выход.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)

        # Если веб-камера аппаратно не умеет 1280x720, программно доводим до этого разрешения
        if frame.shape[1] != 1280 or frame.shape[0] != 720:
            frame = cv.resize(frame, (1280, 720), interpolation=cv.INTER_LINEAR)

        h, w = frame.shape[:2]

        result = engine.process(frame, draw_overlay=True)
        img = result.annotated_image

        # Активная рамка экрана
        bx1, by1, bx2, by2 = action_ctrl.get_interaction_box(w, h)
        box_color = (0, 255, 0) if mouse_control_enabled else (100, 100, 100)
        cv.rectangle(img, (bx1, by1), (bx2, by2), box_color, 2)
        cv.putText(img, "Active Screen Area", (bx1 + 10, by1 + 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)

        if result.hands and mouse_control_enabled:
            hand = result.hands[0]
            action_ctrl.update(hand, w, h)

            thumb = hand.landmarks[4]
            index = hand.landmarks[8]
            middle = hand.landmarks[12]
            ring = hand.landmarks[16]

            # 1. Прицел указательного пальца (КУРСОР)
            cv.circle(img, index, 8, (255, 0, 255), -1)
            cv.circle(img, index, 14, (255, 255, 255), 2)

            # 2. Линия ЛКМ (Большой + Средний)
            lmb_color = (0, 0, 255) if action_ctrl.is_left_pressed else (255, 180, 0)
            cv.line(img, thumb, middle, lmb_color, 2)
            cv.circle(img, middle, 6, lmb_color, -1)

            # 3. Линия ПКМ (Большой + Безымянный)
            rmb_color = (255, 0, 128) if action_ctrl.is_right_pressed else (0, 200, 255)
            cv.line(img, thumb, ring, rmb_color, 2)
            cv.circle(img, ring, 6, rmb_color, -1)

            # Текстовые индикаторы клика
            if action_ctrl.is_left_pressed:
                cv.putText(img, "[ LMB CLICK ]", (middle[0] + 15, middle[1]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            elif action_ctrl.is_right_pressed:
                cv.putText(img, "[ RMB CLICK ]", (ring[0] + 15, ring[1]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 128), 2)

        # Инфопанель
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time

        status_text = f"Mouse: {'ON' if mouse_control_enabled else 'OFF'} (Press 'M')"
        cv.putText(img, status_text, (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        cv.putText(img, f"FPS: {int(fps)}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv.imshow(WINDOW_TITLE, img)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mouse_control_enabled = not mouse_control_enabled

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()