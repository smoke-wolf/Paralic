"""Thin wrapper around MediaPipe's HandLandmarker neural network.

HandLandmarker runs a palm detector and a hand-landmark network to predict 21
3-D landmarks per hand. This is the Hand-mode counterpart of tracker.py's
FaceTracker; it keeps the same shape of interface (``process`` → observation,
``close``) so the session code reads the same way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class HandObservation:
    points_norm: np.ndarray               # (21, 3) landmarks normalised to the image (0..1)
    image_size: tuple[int, int]           # (width, height)
    handedness: str                       # "Left" | "Right" | ""


class HandTracker:
    """Runs HandLandmarker in VIDEO mode (it tracks the hand between frames)."""

    def __init__(self, model_bytes: bytes, *, num_hands: int = 1, min_detection: float = 0.5,
                 min_presence: float = 0.5, min_tracking: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> Optional[HandObservation]:
        ts = max(int(timestamp_ms), self._last_ts + 1)      # VIDEO mode needs increasing timestamps
        self._last_ts = ts
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            return None
        lms = result.hand_landmarks[0]
        pts = np.empty((len(lms), 3), dtype=np.float64)
        for i, p in enumerate(lms):
            pts[i, 0] = p.x
            pts[i, 1] = p.y
            pts[i, 2] = p.z
        hand = ""
        if result.handedness:
            try:
                hand = result.handedness[0][0].category_name
            except (IndexError, AttributeError):  # pragma: no cover
                hand = ""
        return HandObservation(points_norm=pts, image_size=(w, h), handedness=hand)

    def close(self) -> None:
        lm, self._landmarker = getattr(self, "_landmarker", None), None
        if lm is not None:
            try:
                lm.close()
            except Exception:  # pragma: no cover - best effort cleanup
                log.debug("Error closing HandLandmarker", exc_info=True)
