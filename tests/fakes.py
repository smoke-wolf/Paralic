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


def fake_features(vector: np.ndarray, closure: float = 0.12, closure_left: float | None = None,
                  closure_right: float | None = None) -> FrameFeatures:
    eye = EyeMeasure(center=np.zeros(2), iris=np.zeros(2), iris_radius=1.0, width=30.0,
                     dx=float(vector[0]), dy=float(vector[1]), aperture=float(vector[4]))
    cl = closure if closure_left is None else closure_left
    cr = closure if closure_right is None else closure_right
    return FrameFeatures(vector=np.asarray(vector, float), closure=0.5 * (cl + cr), closure_left=cl,
                         closure_right=cr, aperture=0.3, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
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
    monkeypatch.setattr(session_mod, "mesh_overlay", lambda pts, size: {"pts": [[0.5, 0.5]], "lines": [], "iris": []})


class ManualClock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float = 1 / 30) -> None:
        self.t += dt


# -- hand mode ---------------------------------------------------------------------

def make_hand(centre=(0.5, 0.6), tip=None, pinch=0.9, fingers="spread", thumb_out=True, width=0.16):
    """A plausible 21-point normalised hand (image coordinates, fingers up).

    ``centre``: the middle knuckle; ``tip``: where the index fingertip is
    (default: straight above its knuckle); ``pinch``: thumb-index distance in
    palm widths; ``fingers``: "spread" (an open "stop" hand), "together" (a flat
    pointing hand) or "curled" (only the index finger out).
    """
    cx, cy = centre
    w = width
    p = np.zeros((21, 3))
    p[0] = (cx, cy + 1.6 * w, 0)                              # wrist
    p[5] = (cx - 0.5 * w, cy, 0)                              # index knuckle
    p[9] = (cx, cy, 0)                                        # middle knuckle
    p[13] = (cx + 0.3 * w, cy, 0)
    p[17] = (cx + 0.5 * w, cy, 0)                             # pinky knuckle (palm width = w)
    fan = {"spread": 0.45, "together": -0.06, "curled": 0.08}[fingers]
    for k, (mcp, base) in enumerate(((5, -0.5), (9, 0.0), (13, 0.3), (17, 0.5))):
        x0 = cx + base * w
        up = 1.0
        if fingers == "curled" and k > 0:
            p[mcp + 1] = (x0, cy - 0.5 * w, 0)                # PIP
            p[mcp + 2] = (x0, cy - 0.35 * w, 0)
            p[mcp + 3] = (x0, cy - 0.15 * w, 0)               # tip curled back towards the wrist
            continue
        dx = (k - 1.5) * fan * w
        p[mcp + 1] = (x0 + dx * 0.4, cy - 0.7 * w * up, 0)    # PIP
        p[mcp + 2] = (x0 + dx * 0.7, cy - 1.0 * w * up, 0)
        p[mcp + 3] = (x0 + dx, cy - 1.4 * w * up, 0)          # tip
    if tip is not None:
        p[8] = (tip[0], tip[1], 0)
        p[7] = ((p[6][0] + tip[0]) / 2, (p[6][1] + tip[1]) / 2, 0)
    thumb_base = (cx - 0.9 * w, cy + 0.6 * w) if thumb_out else (cx - 0.35 * w, cy + 0.3 * w)
    p[1] = (cx - 0.4 * w, cy + 1.2 * w, 0)
    p[2] = (cx - 0.7 * w, cy + 0.9 * w, 0)
    p[3] = (thumb_base[0], thumb_base[1], 0)
    if pinch < 0.85:
        # Thumb tip next to the index fingertip.
        p[4] = (p[8][0] - pinch * w, p[8][1], 0)
    else:
        p[4] = (thumb_base[0] - (0.35 * w if thumb_out else 0.0), thumb_base[1] - 0.2 * w, 0)
    return p


@dataclass
class FakeHandObservation:
    points_norm: np.ndarray
    image_size: tuple = (640, 480)
    handedness: str = "Right"


class FakeHandTracker:
    """Returns queued hands (21x3 landmark arrays); None means "no hand"."""

    def __init__(self):
        self.queue: list = []
        self.closed = False

    def push(self, pts) -> None:
        self.queue.append(None if pts is None else FakeHandObservation(points_norm=np.asarray(pts, float)))

    def process(self, rgb, timestamp_ms):
        return self.queue.pop(0) if self.queue else None

    def close(self):
        self.closed = True
