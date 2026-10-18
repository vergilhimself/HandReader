import math
import time
from typing import Tuple
import numpy as np
from pynput.mouse import Button, Controller as MouseController
import pyautogui

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False


class OneEuroFilter:
    """
    One Euro Filter, настроенный на минимальную задержку (без инерции).
    """
    def __init__(self, min_cutoff: float = 0.015, beta: float = 0.6, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _smoothing_factor(self, t_e: float, cutoff: float) -> float:
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def filter(self, x: float, timestamp: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = timestamp
            return x

        t_e = max(timestamp - self.t_prev, 1e-4)
        self.t_prev = timestamp

        # Оценка текущей скорости
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # При движении cutoff резко возрастает -> задержка падает до нуля
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat


class ActionController:
    def __init__(
            self,
            box_margin_x: float = 0.18,          # Отступ рабочей зоны по бокам
            box_margin_y: float = 0.18,          # Отступ рабочей зоны сверху/снизу
            click_press_ratio: float = 0.32,     # Порог клика (смыкание)
            click_release_ratio: float = 0.42,   # Порог отпускания
            click_freeze_duration: float = 0.10, # Заморозка при нажатии (сек)
            deadzone: float = 0.0022,            # Мертвая зона покоя (~3-4 пикселя)
            min_cutoff: float = 0.012,           # Стабильность в статике
            beta: float = 0.65                   # Мгновенная реакция на разгон (без скольжения)
    ):
        self.mouse = MouseController()
        self.screen_w, self.screen_h = pyautogui.size()

        self.margin_x = box_margin_x
        self.margin_y = box_margin_y
        self.click_press_ratio = click_press_ratio
        self.click_release_ratio = click_release_ratio
        self.freeze_duration = click_freeze_duration
        self.deadzone = deadzone

        # Фильтры координат
        self.filter_x = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.filter_y = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

        self.current_norm_x: float = 0.5
        self.current_norm_y: float = 0.5
        self.is_first_frame = True

        # Состояния кнопок мыши
        self.is_left_pressed = False
        self.is_right_pressed = False
        self.last_click_time = 0.0

    def get_interaction_box(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        x1 = int(frame_w * self.margin_x)
        y1 = int(frame_h * self.margin_y)
        x2 = int(frame_w * (1.0 - self.margin_x))
        y2 = int(frame_h * (1.0 - self.margin_y))
        return x1, y1, x2, y2

    @staticmethod
    def _calc_distance(pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    def update(self, hand_result, frame_w: int, frame_h: int):
        if not hand_result or not hand_result.landmarks:
            if self.is_left_pressed:
                self.mouse.release(Button.left)
                self.is_left_pressed = False
            if self.is_right_pressed:
                self.mouse.release(Button.right)
                self.is_right_pressed = False
            return

        landmarks = hand_result.landmarks
        now = time.time()

        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_base = landmarks[9]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]

        # 1. Нормализация координат указательного пальца внутри рабочей зоны
        bx1, by1, bx2, by2 = self.get_interaction_box(frame_w, frame_h)
        target_norm_x = float(np.clip((index_tip[0] - bx1) / max(bx2 - bx1, 1), 0.0, 1.0))
        target_norm_y = float(np.clip((index_tip[1] - by1) / max(by2 - by1, 1), 0.0, 1.0))

        # 2. Фильтрация координат (быстрый отклик без запаздывания)
        filtered_norm_x = self.filter_x.filter(target_norm_x, now)
        filtered_norm_y = self.filter_y.filter(target_norm_y, now)

        if self.is_first_frame:
            self.current_norm_x = filtered_norm_x
            self.current_norm_y = filtered_norm_y
            self.is_first_frame = False

        # 3. Мертвая зона: если палец сместился меньше порога — курсор стоит мертво
        dx = filtered_norm_x - self.current_norm_x
        dy = filtered_norm_y - self.current_norm_y
        dist = math.hypot(dx, dy)

        if dist > self.deadzone:
            # Мгновенная передача координат без вязкой интерполяции
            self.current_norm_x = filtered_norm_x
            self.current_norm_y = filtered_norm_y

        # 4. Дистанции смыкания пальцев
        palm_scale = self._calc_distance(wrist, middle_base) or 1.0
        middle_pinch = self._calc_distance(thumb_tip, middle_tip) / palm_scale
        ring_pinch = self._calc_distance(thumb_tip, ring_tip) / palm_scale

        # --- ЛКМ (Средний + Большой) ---
        if not self.is_right_pressed:
            if not self.is_left_pressed:
                if middle_pinch < self.click_press_ratio and middle_pinch < ring_pinch:
                    self.mouse.press(Button.left)
                    self.is_left_pressed = True
                    self.last_click_time = now
            else:
                if middle_pinch > self.click_release_ratio:
                    self.mouse.release(Button.left)
                    self.is_left_pressed = False

        # --- ПКМ (Безымянный + Большой) ---
        if not self.is_left_pressed:
            if not self.is_right_pressed:
                if ring_pinch < self.click_press_ratio and ring_pinch < middle_pinch:
                    self.mouse.press(Button.right)
                    self.is_right_pressed = True
                    self.last_click_time = now
            else:
                if ring_pinch > self.click_release_ratio:
                    self.mouse.release(Button.right)
                    self.is_right_pressed = False

        # 5. Перемещение курсора мыши
        any_pressed = self.is_left_pressed or self.is_right_pressed
        if any_pressed and (now - self.last_click_time < self.freeze_duration):
            return

        screen_x = int(self.current_norm_x * self.screen_w)
        screen_y = int(self.current_norm_y * self.screen_h)
        self.mouse.position = (screen_x, screen_y)