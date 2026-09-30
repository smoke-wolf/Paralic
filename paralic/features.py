"""Feature extraction: MediaPipe face landmarks -> numbers the gaze network can learn from.

For every camera frame we compute a fixed-length feature vector made of

* iris position inside each eye (relative to the eye corners, so it is invariant
  to where the face is in the picture and how far away it is),
* eyelid opening of each eye (the upper lid follows the eye when looking down),
* the eye-movement blendshapes predicted by MediaPipe's blendshape network,
* head rotation (yaw / pitch / roll) and head position from the facial
  transformation matrix, so the gaze network can compensate for head movement.

We also compute an eye *closure* score used for blink detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from . import landmarks as L

FEATURE_NAMES: tuple[str, ...] = (
    "r_dx", "r_dy", "l_dx", "l_dy",
    "r_open", "l_open",
    "bs_look_in_l", "bs_look_out_l", "bs_look_up_l", "bs_look_down_l",
    "bs_look_in_r", "bs_look_out_r", "bs_look_up_r", "bs_look_down_r",
    "yaw", "pitch", "roll",
    "tx", "ty", "tz",
)
NUM_FEATURES = len(FEATURE_NAMES)

# Indices into the feature vector that describe the eyes themselves (used for
# outlier rejection during calibration).
EYE_FEATURE_IDX: tuple[int, ...] = (0, 1, 2, 3)

# Typical noise / movement scale of each feature. Used as a lower bound for the
# standard deviation when standardising, so that a feature that barely changed
# during calibration (e.g. head position when the user kept perfectly still)
# does not get blown up into a huge input later on.
FEATURE_MIN_STD = np.array(
    [0.01, 0.01, 0.01, 0.01,
     0.01, 0.01,
     0.03, 0.03, 0.03, 0.03,
     0.03, 0.03, 0.03, 0.03,
     0.035, 0.035, 0.035,   # ~2 degrees
     1.0, 1.0, 1.5],        # centimetres
    dtype=np.float64,
)

_BLENDSHAPE_KEYS = (
    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight",
)


@dataclass
class EyeMeasure:
    """Geometry of one eye in image pixels."""

    center: np.ndarray      # midpoint between the two corners
    iris: np.ndarray        # iris centre
    iris_radius: float
    width: float            # corner-to-corner distance
    dx: float               # iris offset along the eye axis, in eye widths
    dy: float               # iris offset perpendicular to the eye axis (down = +)
    aperture: float         # lid opening, in eye widths


@dataclass
class FrameFeatures:
    vector: np.ndarray                  # shape (NUM_FEATURES,)
    closure: float                      # 0 = wide open, 1 = closed (blink signal)
    closure_left: float
    closure_right: float
    aperture: float                     # mean lid opening of both eyes (eye widths)
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    distance_cm: float
    right_eye: EyeMeasure
    left_eye: EyeMeasure
    face_box: tuple[float, float, float, float]  # normalised x0, y0, x1, y1
    blendshapes: dict[str, float] = field(default_factory=dict)


def _eye_measure(pts: np.ndarray, corner_a: int, corner_b: int,
                 upper: Sequence[int], lower: Sequence[int], iris: Sequence[int]) -> EyeMeasure:
    """Measure one eye. ``corner_a`` must be the corner with the smaller image x."""
    a = pts[corner_a]
    b = pts[corner_b]
    axis = b - a
    width = float(np.hypot(axis[0], axis[1]))
    width = max(width, 1e-6)
    ex = axis / width
    ey = np.array([-ex[1], ex[0]])  # rotated +90 degrees: points "down" in image space
    center = (a + b) * 0.5
    iris_pts = pts[list(iris)]
    iris_c = iris_pts.mean(axis=0)
    rel = iris_c - center
    dx = float(rel @ ex) / width
    dy = float(rel @ ey) / width
    up = pts[list(upper)].mean(axis=0)
    lo = pts[list(lower)].mean(axis=0)
    aperture = float((lo - up) @ ey) / width
    radius = float(np.mean(np.hypot(*(iris_pts[1:] - iris_c).T)))
    return EyeMeasure(center=center, iris=iris_c, iris_radius=radius, width=width,
                      dx=dx, dy=dy, aperture=aperture)


def head_pose_from_matrix(matrix: Optional[np.ndarray]) -> tuple[float, float, float, np.ndarray]:
    """Return (yaw, pitch, roll) in radians and translation (cm) from a 4x4 face transform."""
    if matrix is None:
        return 0.0, 0.0, 0.0, np.zeros(3)
    m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    rot = m[:3, :3]
    forward = rot[:, 2]      # canonical face normal expressed in camera space
    right = rot[:, 0]
    yaw = math.atan2(forward[0], forward[2])
    pitch = math.asin(max(-1.0, min(1.0, forward[1])))
    roll = math.atan2(right[1], right[0])
    return yaw, pitch, roll, m[:3, 3].copy()


def extract_features(
    points_px: np.ndarray,
    image_size: tuple[int, int],
    blendshapes: Optional[Mapping[str, float]] = None,
    matrix: Optional[np.ndarray] = None,
) -> FrameFeatures:
    """Compute features for one face.

    Args:
        points_px: (478, 2+) landmark array in *pixel* coordinates.
        image_size: (width, height) of the analysed image.
        blendshapes: blendshape name -> score, if available.
        matrix: 4x4 facial transformation matrix, if available.
    """
    if points_px.shape[0] < L.NUM_LANDMARKS_WITH_IRIS:
        raise ValueError("Iris landmarks missing: a 478-point face model is required")
    pts = np.asarray(points_px[:, :2], dtype=np.float64)

    right = _eye_measure(pts, L.RIGHT_EYE_OUTER, L.RIGHT_EYE_INNER,
                         L.RIGHT_UPPER_LID, L.RIGHT_LOWER_LID, L.RIGHT_IRIS)
    left = _eye_measure(pts, L.LEFT_EYE_INNER, L.LEFT_EYE_OUTER,
                        L.LEFT_UPPER_LID, L.LEFT_LOWER_LID, L.LEFT_IRIS)

    bs = dict(blendshapes or {})
    looks = [float(bs.get(k, 0.0)) for k in _BLENDSHAPE_KEYS]

    yaw, pitch, roll, t = head_pose_from_matrix(matrix)

    vector = np.array(
        [right.dx, right.dy, left.dx, left.dy,
         right.aperture, left.aperture,
         *looks,
         yaw, pitch, roll,
         t[0], t[1], t[2]],
        dtype=np.float64,
    )

    # Eye closure: blend the blink blendshapes with a geometric eye-aspect
    # measure. An open eye has an aperture of roughly 0.25-0.35 eye widths and
    # a closed one close to 0.05.
    def geometric_closure(aperture: float) -> float:
        return float(np.clip((0.30 - aperture) / 0.24, 0.0, 1.0))

    if "eyeBlinkLeft" in bs and "eyeBlinkRight" in bs:
        # MediaPipe's "Left" blendshapes are for the subject's left eye.
        c_left = 0.5 * float(bs["eyeBlinkLeft"]) + 0.5 * geometric_closure(left.aperture)
        c_right = 0.5 * float(bs["eyeBlinkRight"]) + 0.5 * geometric_closure(right.aperture)
    else:
        c_left = geometric_closure(left.aperture)
        c_right = geometric_closure(right.aperture)

    w, h = image_size
    xs, ys = pts[:, 0], pts[:, 1]
    face_box = (float(xs.min() / w), float(ys.min() / h), float(xs.max() / w), float(ys.max() / h))

    return FrameFeatures(
        vector=vector,
        closure=0.5 * (c_left + c_right),
        closure_left=c_left,
        closure_right=c_right,
        aperture=0.5 * (left.aperture + right.aperture),
        yaw_deg=math.degrees(yaw),
        pitch_deg=math.degrees(pitch),
        roll_deg=math.degrees(roll),
        distance_cm=float(-t[2]) if matrix is not None else 0.0,
        right_eye=right,
        left_eye=left,
        face_box=face_box,
        blendshapes={k: float(v) for k, v in bs.items() if k.startswith("eye")},
    )


def overlay_points(points_px: np.ndarray, image_size: tuple[int, int], feats: FrameFeatures) -> dict:
    """Compact, normalised eye geometry for drawing the camera-preview overlay."""
    w, h = image_size

    def norm(idx: Sequence[int]) -> list[list[float]]:
        return [[round(float(points_px[i, 0] / w), 4), round(float(points_px[i, 1] / h), 4)] for i in idx]

    def iris(e: EyeMeasure) -> list[float]:
        return [round(float(e.iris[0] / w), 4), round(float(e.iris[1] / h), 4), round(e.iris_radius / w, 4)]

    return {
        "r": norm(L.RIGHT_EYE_CONTOUR),
        "l": norm(L.LEFT_EYE_CONTOUR),
        "ri": iris(feats.right_eye),
        "li": iris(feats.left_eye),
        "box": [round(v, 4) for v in feats.face_box],
    }
