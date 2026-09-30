"""Thin wrapper around MediaPipe's FaceLandmarker neural network.

FaceLandmarker runs a pipeline of neural networks on every frame: a face
detector (BlazeFace), a face-mesh network that predicts 478 3-D landmarks
including the irises, and a blendshape network that scores 52 facial
expressions (e.g. "eyeBlinkLeft", "eyeLookUpRight"). It also estimates the
head's 3-D pose as a transformation matrix.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FaceObservation:
    points_px: np.ndarray                 # (478, 3) landmarks in pixel units (z scaled like x)
    image_size: tuple[int, int]           # (width, height)
    blendshapes: dict[str, float]
    matrix: Optional[np.ndarray]          # 4x4 facial transformation matrix


def decode_image(data: bytes) -> np.ndarray:
    """Decode JPEG/PNG/WebP bytes into a contiguous RGB uint8 array."""
    try:
        import cv2  # installed together with mediapipe

        buf = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode camera frame")
        return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    except ImportError:  # pragma: no cover - fallback when OpenCV is unavailable
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return np.ascontiguousarray(np.asarray(im.convert("RGB")))


class FaceTracker:
    """Runs FaceLandmarker in VIDEO mode (it tracks the face between frames)."""

    def __init__(self, model_bytes: bytes, *, min_detection: float = 0.5, min_presence: float = 0.5,
                 min_tracking: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection,
            min_face_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> Optional[FaceObservation]:
        # VIDEO mode requires strictly increasing timestamps.
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, ts)
        if not result.face_landmarks:
            return None
        lms = result.face_landmarks[0]
        pts = np.empty((len(lms), 3), dtype=np.float64)
        for i, p in enumerate(lms):
            pts[i, 0] = p.x * w
            pts[i, 1] = p.y * h
            pts[i, 2] = p.z * w
        blend = {}
        if result.face_blendshapes:
            blend = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        matrix = None
        if result.facial_transformation_matrixes:
            matrix = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64).reshape(4, 4)
        return FaceObservation(points_px=pts, image_size=(w, h), blendshapes=blend, matrix=matrix)

    def close(self) -> None:
        lm, self._landmarker = getattr(self, "_landmarker", None), None
        if lm is not None:
            try:
                lm.close()
            except Exception:  # pragma: no cover - best effort cleanup
                log.debug("Error closing FaceLandmarker", exc_info=True)
