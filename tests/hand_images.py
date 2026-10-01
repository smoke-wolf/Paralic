"""Public MediaPipe test photos of hands, for tests with the real hand network.

Downloaded once into tests/.cache: a hand pointing up, a "victory" sign, a
thumbs-up and two open hands.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / ".cache"
URLS = {
    "pointing_up": "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/pointing_up.jpg",
    "victory": "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/victory.jpg",
    "thumbs_up": "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/thumbs_up.jpg",
    "open_hands": "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg",
}


def hand_image(name: str) -> np.ndarray | None:
    """RGB image, or None if it cannot be downloaded."""
    import cv2

    path = CACHE / f"hand_{name}.jpg"
    if not path.exists():
        try:
            CACHE.mkdir(exist_ok=True)
            urllib.request.urlretrieve(URLS[name], path)
        except Exception:
            return None
    img = cv2.imread(str(path))
    return None if img is None else np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
