import csv
import copy
import itertools
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow import lite as tflite
    except ImportError:
        tflite = None


@dataclass
class HandDetectionResult:
    handedness: str
    keypoint_label: str
    keypoint_id: int
    point_history_label: str
    landmarks: List[Tuple[int, int]]
    bounding_box: Tuple[int, int, int, int]


@dataclass
class FrameResult:
    annotated_image: np.ndarray
    hands: List[HandDetectionResult] = field(default_factory=list)


class GestureEngine:
    def __init__(
        self,
        landmarker_model_path: str = "models/hand_landmarker.task",
        keypoint_model_path: str = "models/keypoint_classifier.tflite",
        keypoint_labels_path: str = "models/keypoint_classifier_label.csv",
        history_model_path: Optional[str] = "models/point_history_classifier.tflite",
        history_labels_path: Optional[str] = "models/point_history_classifier_label.csv",
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence: float = 0.5,

    ):
        # 1. Инициализация современного MediaPipe HandLandmarker
        if not Path(landmarker_model_path).exists():
            raise FileNotFoundError(
                f"Файл {landmarker_model_path} не найден! "
                f"Скачайте его по ссылке: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        base_options = python.BaseOptions(model_asset_path=landmarker_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self._last_timestamp_ms = 0
        # 2. Буфер истории кончика указательного пальца (точка №8)
        self.history_length = 16
        self.point_history = deque([(0, 0)] * self.history_length, maxlen=self.history_length)

        # 3. Загрузка классификатора жестов
        self.keypoint_interpreter = self._load_tflite_model(keypoint_model_path)
        self.keypoint_labels = self._load_labels(keypoint_labels_path)

        self.history_interpreter = self._load_tflite_model(history_model_path) if history_model_path else None
        self.history_labels = self._load_labels(history_labels_path) if history_labels_path else []

    def _load_tflite_model(self, model_path: str):
        if not tflite or not Path(model_path).exists():
            return None
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=2)
        interpreter.allocate_tensors()
        return interpreter

    def _load_labels(self, labels_path: str) -> List[str]:
        if not Path(labels_path).exists():
            return []
        with open(labels_path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            return [row[0] for row in reader if row]

    def _predict_tflite(self, interpreter, input_vector: list) -> int:
        if interpreter is None:
            return -1
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_data = np.array([input_vector], dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        return int(np.argmax(np.squeeze(output)))

    # --- Предобработка координат (полностью сохранена оригинальная математика) ---

    @staticmethod
    def _calc_landmarks(w: int, h: int, landmarks_list) -> List[Tuple[int, int]]:
        return [
            (min(int(landmark.x * w), w - 1), min(int(landmark.y * h), h - 1))
            for landmark in landmarks_list
        ]

    @staticmethod
    def _calc_bounding_box(landmarks: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        xs = [pt[0] for pt in landmarks]
        ys = [pt[1] for pt in landmarks]
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def _pre_process_landmarks(landmarks: List[Tuple[int, int]]) -> List[float]:
        temp = copy.deepcopy(landmarks)
        base_x, base_y = temp[0]

        for i in range(len(temp)):
            temp[i] = (temp[i][0] - base_x, temp[i][1] - base_y)

        flat = list(itertools.chain.from_iterable(temp))
        max_val = max(map(abs, flat)) or 1.0
        return [v / max_val for v in flat]

    def _pre_process_point_history(self, w: int, h: int) -> List[float]:
        temp = list(self.point_history)
        base_x, base_y = temp[0]

        flat = []
        for pt in temp:
            flat.append((pt[0] - base_x) / w)
            flat.append((pt[1] - base_y) / h)
        return flat

    # --- Обработка кадра ---

    def process(self, frame: np.ndarray, draw_overlay: bool = True) -> FrameResult:
        import time

        debug_image = copy.deepcopy(frame)
        h, w = frame.shape[:2]

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Для режима VIDEO обязательно передавать монотонно растущее время кадра в мс
        current_timestamp_ms = int(time.time() * 1000)
        if current_timestamp_ms <= self._last_timestamp_ms:
            current_timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = current_timestamp_ms

        # Используем detect_for_video вместо detect
        detection_result = self.detector.detect_for_video(mp_image, current_timestamp_ms)
        detected_hands: List[HandDetectionResult] = []

        if detection_result.hand_landmarks:
            for hand_landmarks, handedness_info in zip(
                detection_result.hand_landmarks, detection_result.handedness
            ):
                landmarks = self._calc_landmarks(w, h, hand_landmarks)
                bbox = self._calc_bounding_box(landmarks)
                raw_handedness = handedness_info[0].category_name
                # Инвертируем определение руки из-за зеркалирования кадра (cv.flip)
                handedness = "Right" if raw_handedness == "Left" else "Left"

                # Классификация статического жеста
                pre_processed_landmarks = self._pre_process_landmarks(landmarks)
                keypoint_id = self._predict_tflite(self.keypoint_interpreter, pre_processed_landmarks)
                keypoint_label = (
                    self.keypoint_labels[keypoint_id]
                    if 0 <= keypoint_id < len(self.keypoint_labels)
                    else "Unknown"
                )

                # Классификация динамического жеста
                self.point_history.append(landmarks[8])
                pre_processed_history = self._pre_process_point_history(w, h)
                history_id = self._predict_tflite(self.history_interpreter, pre_processed_history)
                history_label = (
                    self.history_labels[history_id]
                    if 0 <= history_id < len(self.history_labels)
                    else ""
                )

                hand_result = HandDetectionResult(
                    handedness=handedness,
                    keypoint_label=keypoint_label,
                    keypoint_id=keypoint_id,
                    point_history_label=history_label,
                    landmarks=landmarks,
                    bounding_box=bbox,
                )
                detected_hands.append(hand_result)

                if draw_overlay:
                    self._draw_overlay(debug_image, hand_result)
        else:
            self.point_history.append((0, 0))

        return FrameResult(annotated_image=debug_image, hands=detected_hands)

    def _draw_overlay(self, image: np.ndarray, hand: HandDetectionResult):
        x1, y1, x2, y2 = hand.bounding_box
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        for p1, p2 in connections:
            cv.line(image, hand.landmarks[p1], hand.landmarks[p2], (255, 255, 255), 2)

        for pt in hand.landmarks:
            cv.circle(image, pt, 4, (0, 128, 255), -1)

        text = f"{hand.handedness}: {hand.keypoint_label}"
        if hand.point_history_label:
            text += f" | {hand.point_history_label}"
        cv.putText(image, text, (x1, max(y1 - 10, 20)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)