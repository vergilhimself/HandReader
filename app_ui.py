import math
import threading
import time
from typing import Optional
import cv2 as cv
import numpy as np
from PIL import Image
import customtkinter as ctk

from gesture_engine import GestureEngine
from action_controller import ActionController

# Настройки оформления CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class HandReaderApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("HandReader Studio — Touchless PC Control")
        self.geometry("1280x820")
        self.minsize(1100, 720)

        # 1. Инициализация ядерных модулей
        self.engine = GestureEngine(
            keypoint_model_path="models/keypoint_classifier.tflite",
            keypoint_labels_path="models/keypoint_classifier_label.csv",
        )
        self.action_ctrl = ActionController(
            box_margin_x=0.18,
            box_margin_y=0.18,
            click_press_ratio=0.32,
            click_release_ratio=0.42,
            beta=0.65,
            deadzone=0.0022
        )

        # 2. Флаги и переменные состояния
        self.is_running = True
        self.camera_paused = False
        self.mouse_enabled = False
        self.show_screen_box = True
        self.show_skeleton = True
        self.show_probabilities = True

        # Телеметрия
        self.fps = 0
        self.detected_gesture = "None"
        self.confidence = 0.0
        self.is_lmb = False
        self.is_rmb = False

        # Буфер для передачи кадров между потоками
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

        # 3. Построение графического интерфейса
        self._build_ui()

        # 4. Запуск камеры в отдельном рабочем потоке
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

        self.worker_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.worker_thread.start()

        # 5. Цикл обновления UI (таймер Tkinter ~45 FPS)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_gui()

    def _build_ui(self):
        # Главная сетка: 2 колонки (слева видео, справа панель)
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================= ЛЕВАЯ ПАНЕЛЬ (ВИДЕО) =================
        self.left_frame = ctk.CTkFrame(self, corner_radius=15)
        self.left_frame.grid(row=0, column=0, padx=(15, 10), pady=15, sticky="nsew")
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        # Верхняя панель видео
        self.video_header = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.video_header.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="ew")

        self.lbl_stream_title = ctk.CTkLabel(
            self.video_header,
            text="🎥 Видеопоток в реальном времени (1280x720)",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.lbl_stream_title.pack(side="left")

        # Контейнер для вывода изображения
        self.video_label = ctk.CTkLabel(self.left_frame, text="Запуск камеры...", corner_radius=12)
        self.video_label.grid(row=1, column=0, padx=15, pady=10, sticky="nsew")

        # Нижняя панель управления видео
        self.video_footer = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.video_footer.grid(row=2, column=0, padx=15, pady=(5, 12), sticky="ew")

        self.btn_pause = ctk.CTkButton(
            self.video_footer,
            text="⏸ Пауза камеры",
            width=140,
            command=self._toggle_camera_pause
        )
        self.btn_pause.pack(side="left", padx=(0, 15))

        self.chk_box = ctk.CTkCheckBox(
            self.video_footer,
            text="Рамка экрана",
            command=self._toggle_screen_box
        )
        self.chk_box.select()
        self.chk_box.pack(side="left", padx=10)

        self.chk_skeleton = ctk.CTkCheckBox(
            self.video_footer,
            text="Скелет руки",
            command=self._toggle_skeleton
        )
        self.chk_skeleton.select()
        self.chk_skeleton.pack(side="left", padx=10)

        # ================= ПРАВАЯ ПАНЕЛЬ (НАСТРОЙКИ) =================
        self.right_frame = ctk.CTkScrollableFrame(self, corner_radius=15, width=340)
        self.right_frame.grid(row=0, column=1, padx=(10, 15), pady=15, sticky="nsew")

        # 1. Заголовок и главный переключатель
        lbl_app = ctk.CTkLabel(
            self.right_frame,
            text="🖐️ HandReader Studio",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        lbl_app.pack(anchor="w", pady=(5, 12))

        self.switch_mouse = ctk.CTkSwitch(
            self.right_frame,
            text="Управление мышью (ПК)",
            font=ctk.CTkFont(size=15, weight="bold"),
            progress_color="#10b981",
            command=self._toggle_mouse_control
        )
        self.switch_mouse.pack(fill="x", pady=(0, 15))

        # 2. Карточка статуса и индикаторов
        lbl_sec_status = ctk.CTkLabel(
            self.right_frame,
            text="СТАТУС И ТЕЛЕМЕТРИЯ",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray"
        )
        lbl_sec_status.pack(anchor="w", pady=(5, 4))

        self.status_card = ctk.CTkFrame(self.right_frame, corner_radius=10, fg_color="#2b2d30")
        self.status_card.pack(fill="x", pady=(0, 15), padx=2)

        self.lbl_fps = ctk.CTkLabel(self.status_card, text="FPS: --", font=ctk.CTkFont(size=13))
        self.lbl_fps.pack(anchor="w", padx=12, pady=(8, 2))

        self.lbl_gesture = ctk.CTkLabel(
            self.status_card,
            text="Жест: Ожидание руки...",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#60a5fa"
        )
        self.lbl_gesture.pack(anchor="w", padx=12, pady=2)

        # Индикаторы кликов
        self.btn_status_box = ctk.CTkFrame(self.status_card, fg_color="transparent")
        self.btn_status_box.pack(fill="x", padx=12, pady=(6, 10))

        self.badge_lmb = ctk.CTkLabel(
            self.btn_status_box,
            text="ЛКМ: ВЫКЛ",
            width=100,
            corner_radius=6,
            fg_color="#374151",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.badge_lmb.pack(side="left", padx=(0, 8))

        self.badge_rmb = ctk.CTkLabel(
            self.btn_status_box,
            text="ПКМ: ВЫКЛ",
            width=100,
            corner_radius=6,
            fg_color="#374151",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.badge_rmb.pack(side="left")

        # 3. Тонкая настройка манипулятора
        lbl_sec_tune = ctk.CTkLabel(
            self.right_frame,
            text="ТОНКАЯ НАСТРОЙКА МЫШИ",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray"
        )
        lbl_sec_tune.pack(anchor="w", pady=(5, 4))

        self.tune_card = ctk.CTkFrame(self.right_frame, corner_radius=10, fg_color="#2b2d30")
        self.tune_card.pack(fill="x", pady=(0, 15), padx=2)

        # Слайдер: Размер рабочей зоны (Margin)
        self.lbl_margin_title = ctk.CTkLabel(
            self.tune_card,
            text="Размер рабочей зоны экрана:",
            font=ctk.CTkFont(size=13)
        )
        self.lbl_margin_title.pack(anchor="w", padx=12, pady=(10, 0))

        self.slider_margin = ctk.CTkSlider(
            self.tune_card,
            from_=0.10,
            to=0.35,
            number_of_steps=25,
            command=self._on_margin_change
        )
        self.slider_margin.set(self.action_ctrl.margin_x)
        self.slider_margin.pack(fill="x", padx=12, pady=(2, 8))

        # Слайдер: Порог смыкания для клика
        self.lbl_click_title = ctk.CTkLabel(
            self.tune_card,
            text="Чувствительность клика (смыкание):",
            font=ctk.CTkFont(size=13)
        )
        self.lbl_click_title.pack(anchor="w", padx=12, pady=(4, 0))

        self.slider_click = ctk.CTkSlider(
            self.tune_card,
            from_=0.20,
            to=0.45,
            number_of_steps=25,
            command=self._on_click_ratio_change
        )
        self.slider_click.set(self.action_ctrl.click_press_ratio)
        self.slider_click.pack(fill="x", padx=12, pady=(2, 8))

        # Слайдер: Мертвая зона (Deadzone)
        self.lbl_deadzone_title = ctk.CTkLabel(
            self.tune_card,
            text="Мертвая зона покоя (гашение тремора):",
            font=ctk.CTkFont(size=13)
        )
        self.lbl_deadzone_title.pack(anchor="w", padx=12, pady=(4, 0))

        self.slider_deadzone = ctk.CTkSlider(
            self.tune_card,
            from_=0.0010,
            to=0.0060,
            number_of_steps=25,
            command=self._on_deadzone_change
        )
        self.slider_deadzone.set(self.action_ctrl.deadzone)
        self.slider_deadzone.pack(fill="x", padx=12, pady=(2, 8))

        # Слайдер: Отзывчивость/Скорость разгона (Beta)
        self.lbl_beta_title = ctk.CTkLabel(
            self.tune_card,
            text="Отзывчивость / Разгон (Beta):",
            font=ctk.CTkFont(size=13)
        )
        self.lbl_beta_title.pack(anchor="w", padx=12, pady=(4, 0))

        self.slider_beta = ctk.CTkSlider(
            self.tune_card,
            from_=0.2,
            to=1.2,
            number_of_steps=20,
            command=self._on_beta_change
        )
        self.slider_beta.set(self.action_ctrl.filter_x.beta)
        self.slider_beta.pack(fill="x", padx=12, pady=(2, 12))

        # Подсказка
        lbl_hint = ctk.CTkLabel(
            self.right_frame,
            text="💡 Подсказка:\n• Указательный — прицел\n• Средний + Большой — ЛКМ\n• Безымянный + Большой — ПКМ",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            justify="left"
        )
        lbl_hint.pack(anchor="w", padx=4, pady=10)

    # ================= СЛОТЫ / ОБРАБОТЧИКИ СОБЫТИЙ =================
    def _toggle_mouse_control(self):
        self.mouse_enabled = self.switch_mouse.get() == 1
        if not self.mouse_enabled:
            # Сброс зажатых кнопок мыши при выключении
            if self.action_ctrl.is_left_pressed:
                self.action_ctrl.mouse.release(self.action_ctrl.mouse.Button.left)
                self.action_ctrl.is_left_pressed = False
            if self.action_ctrl.is_right_pressed:
                self.action_ctrl.mouse.release(self.action_ctrl.mouse.Button.right)
                self.action_ctrl.is_right_pressed = False

    def _toggle_camera_pause(self):
        self.camera_paused = not self.camera_paused
        self.btn_pause.configure(text="▶ Возобновить" if self.camera_paused else "⏸ Пауза камеры")

    def _toggle_screen_box(self):
        self.show_screen_box = self.chk_box.get() == 1

    def _toggle_skeleton(self):
        self.show_skeleton = self.chk_skeleton.get() == 1

    def _on_margin_change(self, val):
        self.action_ctrl.margin_x = float(val)
        self.action_ctrl.margin_y = float(val)

    def _on_click_ratio_change(self, val):
        self.action_ctrl.click_press_ratio = float(val)
        self.action_ctrl.click_release_ratio = float(val) + 0.10

    def _on_deadzone_change(self, val):
        self.action_ctrl.deadzone = float(val)

    def _on_beta_change(self, val):
        self.action_ctrl.filter_x.beta = float(val)
        self.action_ctrl.filter_y.beta = float(val)

    # ================= ФОНОВЫЙ ПОТОК КАМЕРЫ =================
    def _video_loop(self):
        prev_time = time.time()

        while self.is_running:
            if self.camera_paused:
                time.sleep(0.04)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv.flip(frame, 1)

            # Приведение к эталонному разрешению 1280x720
            if frame.shape[1] != 1280 or frame.shape[0] != 720:
                frame = cv.resize(frame, (1280, 720), interpolation=cv.INTER_LINEAR)

            h, w = frame.shape[:2]

            # 1. Прогон через GestureEngine
            result = self.engine.process(frame, draw_overlay=self.show_skeleton)
            annotated = result.annotated_image

            # 2. Отрисовка рабочей рамки
            if self.show_screen_box:
                bx1, by1, bx2, by2 = self.action_ctrl.get_interaction_box(w, h)
                box_color = (16, 185, 129) if self.mouse_enabled else (80, 80, 80)
                cv.rectangle(annotated, (bx1, by1), (bx2, by2), box_color, 2)

            # 3. Обновление контроллера мыши
            if result.hands:
                hand = result.hands[0]
                if self.mouse_enabled:
                    self.action_ctrl.update(hand, w, h)

                # Телеметрия для UI
                self.detected_gesture = hand.keypoint_label
                self.confidence = hand.confidence
                self.is_lmb = self.action_ctrl.is_left_pressed
                self.is_rmb = self.action_ctrl.is_right_pressed

                # Отрисовка прицела указательного пальца
                idx_pt = hand.landmarks[8]
                cv.circle(annotated, idx_pt, 7, (255, 0, 255), -1)
                cv.circle(annotated, idx_pt, 12, (255, 255, 255), 2)
            else:
                self.detected_gesture = "Рука не найдена"
                self.confidence = 0.0
                self.is_lmb = False
                self.is_rmb = False

            # Расчет FPS
            curr_time = time.time()
            self.fps = int(1 / (curr_time - prev_time)) if curr_time != prev_time else 0
            prev_time = curr_time

            # Потокобезопасное сохранение кадра для UI
            with self.frame_lock:
                self.latest_frame = annotated

            time.sleep(0.001)

    # ================= ОБНОВЛЕНИЕ UI =================
    def _update_gui(self):
        if not self.is_running:
            return

        # 1. Обновление видео-картинки
        frame_to_display = None
        with self.frame_lock:
            if self.latest_frame is not None:
                frame_to_display = self.latest_frame.copy()

        if frame_to_display is not None:
            # Получаем текущие размеры контейнера для адаптивного ресайза
            lbl_w = max(self.video_label.winfo_width(), 640)
            lbl_h = max(self.video_label.winfo_height(), 360)

            # Конвертируем BGR -> RGB
            rgb_img = cv.cvtColor(frame_to_display, cv.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

            # Отрисовываем через CTkImage
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(lbl_w, lbl_h))
            self.video_label.configure(image=ctk_img, text="")
            self.video_label.image = ctk_img

        # 2. Обновление текста телеметрии
        self.lbl_fps.configure(text=f"FPS: {self.fps}")

        if self.confidence > 0:
            pct = int(self.confidence * 100)
            self.lbl_gesture.configure(text=f"Жест: {self.detected_gesture} ({pct}%)")
        else:
            self.lbl_gesture.configure(text=f"Жест: {self.detected_gesture}")

        # 3. Индикаторы ЛКМ и ПКМ
        if self.is_lmb:
            self.badge_lmb.configure(fg_color="#ef4444", text="ЛКМ: АКТИВЕН")
        else:
            self.badge_lmb.configure(fg_color="#374151", text="ЛКМ: ВЫКЛ")

        if self.is_rmb:
            self.badge_rmb.configure(fg_color="#8b5cf6", text="ПКМ: АКТИВЕН")
        else:
            self.badge_rmb.configure(fg_color="#374151", text="ПКМ: ВЫКЛ")

        # Перезапуск таймера (~45 кадров в секунду для UI)
        self.after(22, self._update_gui)

    def _on_close(self):
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = HandReaderApp()
    app.mainloop()