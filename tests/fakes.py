"""Test doubles that let the session run without MediaPipe or a camera."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from paralic.features import EyeMeasure, FrameFeatures


def tiny_jpeg() -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
    assert ok
    return buf.tobytes()


def fake_features(vector: np.ndarray, closure: float = 0.12) -> FrameFeatures:
    eye = EyeMeasure(center=np.zeros(2), iris=np.zeros(2), iris_radius=1.0, width=30.0,
                     dx=float(vector[0]), dy=float(vector[1]), aperture=float(vector[4]))
    return FrameFeatures(vector=np.asarray(vector, float), closure=closure, closure_left=closure,
                         closure_right=closure, aperture=0.3, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
                         distance_cm=60.0, right_eye=eye, left_eye=eye, face_box=(0.3, 0.3, 0.7, 0.7))


@dataclass
class FakeObservation:
    points_px: FrameFeatures          # carries the prepared features (see patch_session)
    image_size: tuple = (640, 480)
    blendshapes: dict = field(default_factory=dict)
    matrix: object = None


class FakeTracker:
    """Returns queued observations; None means "no face"."""

    def __init__(self):
        self.queue: list = []
        self.closed = False

    def push(self, features: FrameFeatures | None) -> None:
        self.queue.append(None if features is None else FakeObservation(points_px=features))

    def process(self, rgb, timestamp_ms):
        return self.queue.pop(0) if self.queue else None

    def close(self):
        self.closed = True


def patch_session(monkeypatch):
    """Make the session use the prepared FrameFeatures carried by FakeObservation."""
    import paralic.session as session_mod

    monkeypatch.setattr(session_mod, "extract_features", lambda pts, size, bs=None, m=None: pts)
    monkeypatch.setattr(session_mod, "overlay_points", lambda pts, size, feats: {"r": [], "l": [], "ri": [0, 0, 0],
                                                                                 "li": [0, 0, 0], "box": [0, 0, 1, 1]})


class ManualClock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float = 1 / 30) -> None:
        self.t += dt
