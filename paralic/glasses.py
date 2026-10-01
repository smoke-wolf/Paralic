"""Glasses: whether the person wears them, and reflections on their lenses.

Glasses change how the eyes look to the camera, so each person gets a
calibration made with them and one without (see ``session.py``), and a
reflection on a lens - a lamp or a window caught in it - hides that eye from
the face mesh. Both are measured on the camera frame, in a crop of the face
cut out level and at a fixed scale (64 pixels between the eye centres),
cheaply enough for every frame:

* **Glasses.** A frame crosses the nose between the eyes: its bridge makes a
  horizontal edge right across the nose, where bare skin has hardly any (the
  nose's own edges run up and down). The *bridge* score is the strongest row
  of the vertical gradient averaged across the nose, the *rims* score the
  strongest horizontal edge below each eye (the lower rim: median over the
  columns, the weaker eye counts). Both are measured in units of the skin's
  own edges nearby - the median of the forehead's and both cheeks' mean
  gradient, but at least 4% of the skin's brightness - so the lighting
  cancels out and textured skin needs stronger edges. The bridge decides
  (rimless and half-rim frames have one too); rims below both eyes add up to
  half again. A smoothed score above 2.1 means glasses, below 1.4 none.
* **Glare**, per eye. Around the eye (the area a lens covers), pixels much
  brighter than the skin - over 60% of the way from the skin's brightness to
  white - and nearly colourless; inside the eye opening only almost white
  ones (not the white of the eye). The largest such blob, in units of the
  iris's area: the small corneal reflection everyone has is a few percent of
  it, a reflection on a lens about one or more. Smoothed, above 0.5 the eye
  has glare, below 0.25 not any more. A face lit almost white is not judged.

The thresholds come from glasses and reflections painted onto public test
faces and run through the real face mesh, at several brightnesses, sizes and
head tilts (``tests/test_glasses.py``, ``docs/glasses.md``): without glasses
the score stayed below 0.6 and frown lines across the nose below 1.8, while
dark full-rim, half-rim and rimless frames scored 2.4 or more. Thin frames
the colour of the skin may go unnoticed.

"Left" and "right" are the person's eyes, like everywhere in Paralic.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import landmarks as L

log = logging.getLogger(__name__)

SCALE = 64                                 # crop pixels per eye distance (between the eye centres)
# The crop and the regions in it, in eye distances from the point between the
# eye centres: x along the eyes (towards the person's left eye), y downwards.
CROP = (-1.15, 1.15, -0.85, 0.85)
BRIDGE_BOX = (-0.12, 0.12, -0.26, 0.06)
SKIN_BOXES = ((-0.35, 0.35, -0.80, -0.52), (-0.85, -0.40, 0.50, 0.78), (0.40, 0.85, 0.50, 0.78))
RIM_BOX = (-0.26, 0.26, 0.16, 0.44)         # below an eye (x relative to its centre)
LENS_BOX = (-0.42, 0.42, -0.30, 0.36)       # the area a lens covers (x relative to the eye's centre)
EYE_X = {"left": 0.5, "right": -0.5}
EYE_CONTOURS = {"left": L.LEFT_EYE_CONTOUR, "right": L.RIGHT_EYE_CONTOUR}

SKIN_FLOOR = 0.04            # the skin's edges count as at least this fraction of its brightness
RIM_PLAIN, RIM_SPAN, RIM_BOOST = 1.8, 1.5, 0.5
GLASSES_ON, GLASSES_OFF = 2.1, 1.4
GLARE_LEVEL = 0.6            # glare: this fraction of the way from the skin's brightness to white...
GLARE_MIN_RISE = 25.0        # ...and at least this much brighter than the skin (0-255)
GLARE_SCLERA = 235.0         # inside the eye opening only this bright counts
GLARE_MAX_SAT = 0.28         # (max - min) / max of the colour channels: nearly colourless
GLARE_MAX_SKIN = 220.0       # skin brighter than this: glare cannot be told apart
IRIS_AREA = 0.03             # an iris, in square eye distances
GLARE_ON, GLARE_OFF = 0.5, 0.25
MAX_ANGLE = 25.0             # degrees of head turn or tilt beyond which frames are not judged
MIN_EYE_PX = 24.0            # eye distance in the frame below which the face is too small


@dataclass
class GlassesMeasure:
    """The evidence in one frame."""

    score: float             # glasses (see the module docstring): above GLASSES_ON = glasses
    bridge: float
    rims: float
    glare: dict              # eye -> largest bright blob around it, in iris areas
    skin: float              # brightness of the skin (0-255)


def _which(eyes: dict) -> Optional[str]:
    """None, "left", "right" or "both": the eyes flagged in ``eyes`` (eye -> bool)."""
    on = [eye for eye in ("left", "right") if eyes[eye]]
    return None if not on else on[0] if len(on) == 1 else "both"


def _rows_cols(box: tuple, dx: float = 0.0) -> tuple[slice, slice]:
    x0, x1, y0, y1 = box
    cx0, _, cy0, _ = CROP
    return (slice(int(round((y0 - cy0) * SCALE)), int(round((y1 - cy0) * SCALE))),
            slice(int(round((x0 + dx - cx0) * SCALE)), int(round((x1 + dx - cx0) * SCALE))))


def measure(rgb: np.ndarray, points: np.ndarray) -> Optional[GlassesMeasure]:
    """Glasses and glare evidence in one RGB frame with its face mesh (pixel
    coordinates). None when the face is too small or partly out of the picture."""
    import cv2

    if not isinstance(points, np.ndarray) or not isinstance(rgb, np.ndarray) or rgb.ndim != 3:
        return None
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[0] < 468 or P.shape[1] < 2:
        return None
    P = P[:, :2]
    right = (P[L.RIGHT_EYE_OUTER] + P[L.RIGHT_EYE_INNER]) / 2
    left = (P[L.LEFT_EYE_INNER] + P[L.LEFT_EYE_OUTER]) / 2
    dist = float(np.hypot(*(left - right)))
    if dist < MIN_EYE_PX:
        return None
    ex = (left - right) / dist
    ey = np.array([-ex[1], ex[0]])
    mid = (left + right) / 2
    x0, x1, y0, y1 = CROP
    h, w = rgb.shape[:2]
    corners = np.array([mid + dist * (x * ex + y * ey) for x in (x0, x1) for y in (y0, y1)])
    if (corners < -0.02 * np.array([w, h])).any() or (corners > 1.02 * np.array([w, h])).any():
        return None
    # Image -> crop: the eye line level, SCALE pixels per eye distance.
    A = (SCALE / dist) * np.array([ex, ey])
    M = np.hstack([A, (np.array([-x0, -y0]) * SCALE - A @ mid)[:, None]])
    size = (int(round((x1 - x0) * SCALE)), int(round((y1 - y0) * SCALE)))
    crop = cv2.warpAffine(np.ascontiguousarray(rgb), M, size, flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    grey = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Edges, in units of the skin's own edges.
    g = cv2.GaussianBlur(grey, (3, 3), 0.8)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.sqrt(gx * gx + gy * gy)
    skin_boxes = [_rows_cols(b) for b in SKIN_BOXES]
    skin = float(np.median([np.median(grey[b]) for b in skin_boxes]))
    texture = float(np.median([mag[b].mean() for b in skin_boxes]))
    ref = max(texture, SKIN_FLOOR * skin, 0.3)
    bridge = float(np.abs(gy[_rows_cols(BRIDGE_BOX)].mean(axis=1)).max()) / ref
    rims = min(float(np.median(np.abs(gy[_rows_cols(RIM_BOX, cx)]).max(axis=0))) for cx in EYE_X.values()) / ref
    score = bridge * (1.0 + RIM_BOOST * float(np.clip((rims - RIM_PLAIN) / RIM_SPAN, 0.0, 1.0)))

    glare = {eye: 0.0 for eye in EYE_X}
    if skin <= GLARE_MAX_SKIN:
        level = skin + max(GLARE_LEVEL * (255.0 - skin), GLARE_MIN_RISE)
        Q = P @ M[:, :2].T + M[:, 2]              # the landmarks in the crop
        for eye, cx in EYE_X.items():
            rows, cols = _rows_cols(LENS_BOX, cx)
            lens = crop[rows, cols].astype(np.float32)
            hi = np.maximum(np.maximum(lens[..., 0], lens[..., 1]), lens[..., 2])
            lo = np.minimum(np.minimum(lens[..., 0], lens[..., 1]), lens[..., 2])
            opening = np.zeros(hi.shape, np.uint8)
            contour = Q[list(EYE_CONTOURS[eye])] - [cols.start, rows.start]
            cv2.fillPoly(opening, [np.round(contour).astype(np.int32)], 1)
            opening = cv2.dilate(opening, np.ones((5, 5), np.uint8))
            bright = (((hi - lo) <= GLARE_MAX_SAT * np.maximum(hi, 1.0))
                      & (grey[rows, cols] >= np.where(opening > 0, max(level, GLARE_SCLERA), level)))
            n, _, stats, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8), connectivity=8)
            if n > 1:
                glare[eye] = float(stats[1:, cv2.CC_STAT_AREA].max()) / (IRIS_AREA * SCALE * SCALE)
    return GlassesMeasure(score=score, bridge=bridge, rims=rims, glare=glare, skin=skin)


class GlassesDetector:
    """Glasses and glare from frame to frame, smoothed, with hysteresis.

    Call :meth:`update` with every frame that shows a face and :meth:`missing`
    for frames without one. ``glasses`` is None until a few frames were judged,
    then True or False; ``settled`` follows it once a change has held for
    ``settle_s`` (glasses put on or taken off, not a flicker); ``glare`` is
    None, "left", "right" or "both", and ``frame_glare`` the same for the
    latest frame alone (before any smoothing: for leaving frames out).
    """

    def __init__(self, tau_s: float = 0.6, glare_tau_s: float = 0.2, settle_s: float = 3.0,
                 min_frames: int = 5, persist_s: float = 3.0, persist_share: float = 0.7):
        self.tau_s = tau_s
        self.glare_tau_s = glare_tau_s
        self.settle_s = settle_s
        self.min_frames = min_frames
        self.persist_s = persist_s
        self.persist_share = persist_share
        self.reset()

    def reset(self) -> None:
        self.glasses: Optional[bool] = None
        self.settled: Optional[bool] = None
        self.score = 0.0                     # smoothed glasses score
        self.glare_score = {eye: 0.0 for eye in EYE_X}
        self.glare_since: Optional[float] = None   # when the current glare began
        self.frame_glare: Optional[str] = None
        self._log_score: Optional[float] = None
        self._frames = 0
        self._changed_at = 0.0
        self._last: Optional[float] = None
        self._last_glare: Optional[float] = None
        self._glare_eyes = {eye: False for eye in EYE_X}
        self._glare_log: deque = deque()     # (time, any glare) of recent frames with a face

    @property
    def glare(self) -> Optional[str]:
        return _which(self._glare_eyes)

    def update(self, t: float, rgb: np.ndarray, points: np.ndarray, yaw_deg: float = 0.0,
               pitch_deg: float = 0.0) -> Optional[GlassesMeasure]:
        """Judge one frame with a face (head turned too far: not judged). Returns its measurements."""
        m = None
        if abs(yaw_deg) <= MAX_ANGLE and abs(pitch_deg) <= MAX_ANGLE:
            try:
                m = measure(rgb, points)
            except Exception:  # noticing glasses must never break tracking
                log.debug("Glasses measurement failed", exc_info=True)
        self.frame_glare = None
        if m is not None:
            self._update_glasses(t, m.score)
            self._update_glare(t, m.glare)
            self.frame_glare = _which({eye: m.glare[eye] >= GLARE_ON for eye in EYE_X})
        self._glare_log.append((t, self.glare is not None))
        while self._glare_log and self._glare_log[0][0] < t - self.persist_s:
            self._glare_log.popleft()
        return m

    def missing(self, t: float) -> None:
        """A frame without a face: no glare to see (the glasses state stays)."""
        self.frame_glare = None
        self._update_glare(t, {eye: 0.0 for eye in EYE_X})

    def glare_persists(self, t: float) -> bool:
        """Glare in most frames of the last ``persist_s`` (not just a moment)."""
        log = self._glare_log
        if not log or t - log[0][0] < 0.6 * self.persist_s:
            return False
        return sum(g for _, g in log) >= self.persist_share * len(log)

    def view(self) -> dict:
        """The frame message fields."""
        return {"glasses": bool(self.glasses), "glare": self.glare,
                "glare_score": [round(min(1.0, self.glare_score[eye]), 2) for eye in ("left", "right")]}

    # -- internals ----------------------------------------------------------------
    @staticmethod
    def _alpha(dt: Optional[float], tau: float) -> float:
        return 1.0 if dt is None else 1.0 - math.exp(-min(max(dt, 0.0), 1.0) / tau)

    def _update_glasses(self, t: float, score: float) -> None:
        # Smoothed on a log scale; over the first frames a plain mean, so the
        # first decision does not hang on the first frame.
        x = math.log(max(score, 1e-3))
        self._frames += 1
        a = max(self._alpha(None if self._last is None else t - self._last, self.tau_s), 1.0 / self._frames)
        self._log_score = x if self._log_score is None else self._log_score + a * (x - self._log_score)
        self._last = t
        self.score = math.exp(self._log_score)
        if self._frames < self.min_frames:
            return
        if self.glasses is None:
            state = self.score >= math.sqrt(GLASSES_ON * GLASSES_OFF)
        elif self.glasses:
            state = self.score > GLASSES_OFF
        else:
            state = self.score >= GLASSES_ON
        if state != self.glasses:
            self.glasses = state
            self._changed_at = t
        if self.settled is None or (self.settled != self.glasses and t - self._changed_at >= self.settle_s):
            self.settled = self.glasses

    def _update_glare(self, t: float, areas: dict) -> None:
        a = self._alpha(None if self._last_glare is None else t - self._last_glare, self.glare_tau_s)
        self._last_glare = t
        for eye in EYE_X:
            s = self.glare_score[eye] + a * (float(areas.get(eye, 0.0)) - self.glare_score[eye])
            self.glare_score[eye] = s
            self._glare_eyes[eye] = s > GLARE_OFF if self._glare_eyes[eye] else s >= GLARE_ON
        if self.glare is None:
            self.glare_since = None
        elif self.glare_since is None:
            self.glare_since = t
