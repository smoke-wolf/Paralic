"""Finger-gesture recognition for Hand mode.

MediaPipe Hands gives 21 landmarks per hand. This module turns a stream of those
landmarks into the cursor position and discrete gestures that drive the whole
site - the hand equivalent of blink.py for eyes. It is pure NumPy and holds no
MediaPipe or I/O, so the state machine can be unit-tested directly.

Gesture set (usable hand-only):

* **Point**     - the index fingertip is the cursor.
* **Pinch**     - thumb tip touches index tip. A quick pinch-and-release is a
  **click** on what the cursor pointed at just before the fingers started to
  close; holding the pinch and moving the hand up or down **scrolls**
  (pinch-drag), so you never need a mouse wheel.
* **Open palm** - a "stop" sign: all five fingers out and spread, held still
  for a moment, toggles **pause**. It fires once per gesture: the hand has to
  leave the pose before it can toggle again.

Distances are measured in palm widths (the knuckle span), so they do not
depend on how close the hand is to the camera. A hand lost for a moment (a
frame or two of failed tracking) keeps its pinch.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Landmark indices (MediaPipe Hands).
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20


# ---------------------------------------------------------------------------
# Calibration: pointing map, pinch thresholds, hand span
# ---------------------------------------------------------------------------

def _design(tips: np.ndarray, quadratic: bool) -> np.ndarray:
    """Feature matrix for the pointing fit: [1, x, y] (+ [x^2, y^2, x*y])."""
    x, y = tips[:, 0], tips[:, 1]
    cols = [np.ones_like(x), x, y]
    if quadratic:
        cols += [x * x, y * y, x * y]
    return np.stack(cols, axis=1)


class PointingMap:
    """Maps the (mirrored, normalised) index-fingertip position to a screen
    fraction. Fit by least squares; affine with 3+ points, 2nd-order with 6+ so
    the whole screen is reachable even if the finger's comfortable range is small
    or slightly curved. Everything is clamped to the screen.
    """

    def __init__(self, coef: np.ndarray, quadratic: bool):
        self.coef = np.asarray(coef, float)      # (n_features, 2)
        self.quadratic = quadratic

    @classmethod
    def fit(cls, tips: np.ndarray, targets: np.ndarray) -> "PointingMap":
        tips = np.asarray(tips, float)
        targets = np.asarray(targets, float)
        if len(tips) < 3:
            raise ValueError("need at least 3 pointing samples")
        quadratic = len(tips) >= 6
        A = _design(tips, quadratic)
        coef, *_ = np.linalg.lstsq(A, targets, rcond=None)
        return cls(coef, quadratic)

    def apply(self, tip: np.ndarray) -> np.ndarray:
        tip = np.atleast_2d(np.asarray(tip, float))
        out = _design(tip, self.quadratic) @ self.coef
        out = np.clip(out, 0.0, 1.0)
        return out[0]

    def to_dict(self) -> dict:
        return {"coef": self.coef.tolist(), "quadratic": self.quadratic}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Optional[PointingMap]":
        if not d:
            return None
        coef = np.asarray(d["coef"], float)
        quadratic = bool(d["quadratic"])
        if coef.shape != ((6 if quadratic else 3), 2) or not np.isfinite(coef).all():
            raise ValueError("bad pointing map")
        return cls(coef, quadratic)


def estimate_pinch_thresholds(open_dists: np.ndarray, pinch_dists: np.ndarray,
                              default_on: float = 0.45, default_off: float = 0.60) -> tuple[float, float]:
    """Personal make/break pinch thresholds (thumb-index distance / palm width).

    ``make`` (pinch_on) sits just above the closed-pinch distances; ``break``
    (pinch_off) sits below the open distances, with a guaranteed gap for
    hysteresis so a single pinch cannot chatter into several clicks.
    """
    open_dists = np.asarray(open_dists, float)
    pinch_dists = np.asarray(pinch_dists, float)
    if len(open_dists) < 3 or len(pinch_dists) < 3:
        return default_on, default_off
    closed = float(np.percentile(pinch_dists, 80))      # generous "closed"
    openp = float(np.percentile(open_dists, 20))        # cautious "open"
    if openp - closed < 0.08:                           # not separable → fall back
        return default_on, default_off
    make = closed + 0.35 * (openp - closed)
    brk = closed + 0.70 * (openp - closed)
    make = float(np.clip(make, 0.12, 0.9))
    brk = float(np.clip(max(brk, make + 0.08), make + 0.08, 1.1))
    return make, brk


def split_pinch_cycle(dists: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the thumb-index distances of "pinch a few times" into (open, closed).

    Otsu's threshold: the cut that best separates the two groups (largest
    between-group variance), whatever this person's closed and open distances
    are - a hand that cannot close fully still has two groups.
    """
    d = np.sort(np.asarray(dists, float))
    if len(d) < 6:
        return d[:0], d[:0]
    best, cut = -1.0, 1
    for k in range(2, len(d) - 1):
        a, b = d[:k], d[k:]
        between = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if between > best:
            best, cut = between, k
    return d[cut:], d[:cut]


@dataclass
class HandCalibration:
    span: float = 0.0                       # measured palm width (normalised)
    pinch_on: Optional[float] = None
    pinch_off: Optional[float] = None
    pointing: Optional["PointingMap"] = None

    def to_dict(self) -> dict:
        return {"span": self.span, "pinch_on": self.pinch_on, "pinch_off": self.pinch_off,
                "pointing": self.pointing.to_dict() if self.pointing else None}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Optional[HandCalibration]":
        if not isinstance(d, dict):
            return None
        try:
            on, off = d.get("pinch_on"), d.get("pinch_off")
            return cls(span=float(d.get("span") or 0.0),
                       pinch_on=None if on is None else float(on), pinch_off=None if off is None else float(off),
                       pointing=PointingMap.from_dict(d.get("pointing")))
        except (TypeError, ValueError, KeyError):
            return None




@dataclass
class HandGestureConfig:
    pinch_on: float = 0.45        # thumb-index distance (palm widths) that starts a pinch
    pinch_off: float = 0.60       # hysteresis: the pinch ends above this
    click_max_ms: float = 700.0   # a pinch held longer than this is not a click
    click_move: float = 0.45      # most the hand may travel (palm widths) during a click
    scroll_start: float = 0.5     # vertical pinch-drag (palm widths) that starts scrolling
    scroll_gain: float = 0.5      # screen heights scrolled per palm width of hand travel
    palm_hold_ms: float = 700.0   # a held "stop" palm toggles pause after this long
    palm_still: float = 0.35      # ...if the hand moved less than this (palm widths) meanwhile
    palm_rearm_ms: float = 300.0  # the hand must leave the pose this long before the next toggle
    dropout_ms: float = 250.0     # a hand lost for less than this keeps its pinch
    aim_ms: float = 350.0         # how far back the click looks for where the fingers started to close
    extend_ratio: float = 1.15    # a fingertip is "extended" when this much farther from the wrist than its PIP


@dataclass
class HandEvent:
    type: str                     # "pinch_start" | "click" | "pinch_end" | "scroll" | "palm"
    t: float
    dy: float = 0.0               # scroll amount (screen heights, + = down)
    aim_t: Optional[float] = None  # click: when the cursor last pointed where the person aimed


@dataclass
class HandState:
    present: bool
    cursor: Optional[tuple[float, float]]   # index fingertip mapped to a screen fraction
    pinching: bool
    scrolling: bool
    open_palm: bool
    pinch_dist: float = 0.0
    span: float = 0.0                        # palm width (normalised) this frame
    tip_raw: Optional[tuple[float, float]] = None   # mirrored fingertip before the pointing map
    aim: Optional[tuple[float, float]] = None       # while pinching: the cursor where the pinch aimed


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def palm_width(pts: np.ndarray) -> float:
    """A scale invariant to distance from the camera: the knuckle span."""
    return max(_dist(pts[INDEX_MCP], pts[PINKY_MCP]), 1e-6)


def palm_centre(pts: np.ndarray) -> np.ndarray:
    """Barely moves when the fingers do (a pinch moves the fingertips, not this)."""
    return (pts[WRIST] + pts[INDEX_MCP] + pts[MIDDLE_MCP] + pts[PINKY_MCP]) / 4.0


def finger_extended(pts: np.ndarray, tip: int, pip: int, ratio: float = 1.15) -> bool:
    w = pts[WRIST]
    return _dist(pts[tip], w) > _dist(pts[pip], w) * ratio


def extended_fingers(pts: np.ndarray, ratio: float = 1.15) -> dict[str, bool]:
    return {
        "index": finger_extended(pts, INDEX_TIP, INDEX_PIP, ratio),
        "middle": finger_extended(pts, MIDDLE_TIP, MIDDLE_PIP, ratio),
        "ring": finger_extended(pts, RING_TIP, RING_PIP, ratio),
        "pinky": finger_extended(pts, PINKY_TIP, PINKY_PIP, ratio),
    }


def stop_palm(pts: np.ndarray, ratio: float = 1.15) -> bool:
    """An open "stop" palm: four fingers out, spread apart, and the thumb out.

    Pointing with a flat hand (fingers together) is not this, so it cannot
    pause by accident.
    """
    if not all(extended_fingers(pts, ratio).values()):
        return False
    width = palm_width(pts)
    spread = _dist(pts[INDEX_TIP], pts[PINKY_TIP]) / width
    thumb_out = _dist(pts[THUMB_TIP], pts[INDEX_MCP]) / width
    return spread >= 1.25 and thumb_out >= 0.8


class HandGestureRecognizer:
    """Turns per-frame hand landmarks into a cursor and click / scroll / pause events.

    Pause itself belongs to the page (it also has a button): the recogniser
    only reports the "stop" palm, once per gesture. ``setup`` (during the hand
    calibration) reports no clicks and no palms - the setup asks for open hands
    and pinches.
    """

    def __init__(self, config: Optional[HandGestureConfig] = None,
                 calibration: Optional[HandCalibration] = None):
        self.config = config or HandGestureConfig()
        self.calibration = calibration
        self.setup = False
        self._recent: deque = deque(maxlen=40)      # (t, cursor, pinch_dist)
        self.reset_motion()

    def reset_motion(self) -> None:
        self._pinching = False
        self._pinch_start_t = 0.0
        self._pinch_centre: Optional[np.ndarray] = None
        self._aim: Optional[tuple[float, float]] = None
        self._aim_t: Optional[float] = None
        self._scrolling = False
        self._scroll_last_y = 0.0
        self._palm_since: Optional[float] = None
        self._palm_from: Optional[np.ndarray] = None
        self._palm_fired = False
        self._palm_left_since: Optional[float] = None
        self._lost_since: Optional[float] = None
        self._last_cursor: Optional[tuple[float, float]] = None
        self._recent.clear()

    def apply_calibration(self, cal: Optional[HandCalibration]) -> None:
        self.calibration = cal

    def thresholds(self) -> tuple[float, float]:
        on, off = self.config.pinch_on, self.config.pinch_off
        if self.calibration is not None:
            if self.calibration.pinch_on is not None:
                on = self.calibration.pinch_on
            if self.calibration.pinch_off is not None:
                off = self.calibration.pinch_off
        return on, max(off, on + 0.05)

    def _aim_point(self, t: float, pinch_off: float) -> tuple[Optional[float], Optional[tuple[float, float]]]:
        """Where the cursor pointed when the fingers started to close: the last
        frame (not too long ago) on which the pinch was still fully open."""
        aim_t, aim = None, None
        for rt, cur, d in reversed(self._recent):
            if t - rt > self.config.aim_ms / 1000.0:
                break
            aim_t, aim = rt, cur
            if d >= pinch_off:
                break
        return aim_t, aim

    def update(self, t: float, pts: Optional[np.ndarray], *, mirror: bool = True) -> tuple[HandState, list[HandEvent]]:
        """``pts`` is a (21, 2+) array of normalised (0..1) landmarks, or None.

        Returns the current state and any events fired this frame. The cursor is
        the index fingertip in *screen fraction* coordinates (x mirrored for a
        front camera when ``mirror`` is True).
        """
        cfg = self.config
        events: list[HandEvent] = []
        if pts is None or len(pts) < 21:
            # Hand lost. A moment's dropout keeps a pinch going; after that it
            # ends without a click.
            if self._lost_since is None:
                self._lost_since = t
            if (t - self._lost_since) * 1000.0 >= cfg.dropout_ms and (self._pinching or self._palm_since):
                if self._pinching:
                    events.append(HandEvent("pinch_end", t))
                self._pinching = self._scrolling = False
                self._pinch_centre = None
                self._aim = self._aim_t = None
                self._palm_since = None
            return HandState(False, self._last_cursor, self._pinching, self._scrolling, False,
                             aim=self._aim), events
        self._lost_since = None

        pts = np.asarray(pts, dtype=np.float64)[:, :2]
        tip = pts[INDEX_TIP]
        cx = (1.0 - tip[0]) if mirror else tip[0]
        tip_raw = (float(np.clip(cx, 0.0, 1.0)), float(np.clip(tip[1], 0.0, 1.0)))
        # The pointing map (fit during the hand setup) stretches the comfortable
        # finger range to the whole screen; without one the raw fingertip is the cursor.
        if self.calibration is not None and self.calibration.pointing is not None:
            mapped = self.calibration.pointing.apply(np.array(tip_raw))
            cursor = (float(mapped[0]), float(mapped[1]))
        else:
            cursor = tip_raw
        self._last_cursor = cursor

        width = palm_width(pts)
        centre = palm_centre(pts)
        pinch_dist = _dist(pts[THUMB_TIP], pts[INDEX_TIP]) / width
        palm = stop_palm(pts, cfg.extend_ratio)
        pinch_on, pinch_off = self.thresholds()

        # -- pinch: click or scroll --------------------------------------------
        if not self._pinching and pinch_dist < pinch_on:
            self._pinching = True
            self._pinch_start_t = t
            self._pinch_centre = centre
            self._aim_t, self._aim = self._aim_point(t, pinch_off)
            if self._aim is None:
                self._aim_t, self._aim = t, cursor
            self._scrolling = False
            self._scroll_last_y = float(centre[1])
            events.append(HandEvent("pinch_start", t, aim_t=self._aim_t))
        elif self._pinching:
            travel = centre - self._pinch_centre
            moved = float(np.hypot(*travel)) / width
            if not self._scrolling and abs(travel[1]) / width > cfg.scroll_start:
                self._scrolling = True
                # Scrolling starts from where the pinch began, so the first
                # frame already scrolls the distance moved so far.
                self._scroll_last_y = float(self._pinch_centre[1])
            if self._scrolling:
                dy = (float(centre[1]) - self._scroll_last_y) / width * cfg.scroll_gain
                self._scroll_last_y = float(centre[1])
                if abs(dy) > 1e-4:
                    events.append(HandEvent("scroll", t, dy=dy))
            if pinch_dist > pinch_off:                     # released (hysteresis)
                held_ms = (t - self._pinch_start_t) * 1000.0
                click = not self._scrolling and held_ms <= cfg.click_max_ms and moved <= cfg.click_move
                if click and not self.setup:
                    events.append(HandEvent("click", t, aim_t=self._aim_t))
                else:
                    events.append(HandEvent("pinch_end", t))
                self._pinching = self._scrolling = False
                self._pinch_centre = None
                self._aim = self._aim_t = None

        # -- "stop" palm: pause toggle, once per gesture -------------------------
        if palm and not self._pinching:
            self._palm_left_since = None
            if self._palm_since is None:
                self._palm_since, self._palm_from = t, centre
            elif float(np.hypot(*(centre - self._palm_from))) / width > cfg.palm_still:
                self._palm_since, self._palm_from = t, centre        # moving: not a held sign
            elif (not self._palm_fired and not self.setup
                  and (t - self._palm_since) * 1000.0 >= cfg.palm_hold_ms):
                self._palm_fired = True
                events.append(HandEvent("palm", t))
        else:
            self._palm_since = None
            if self._palm_fired:
                if self._palm_left_since is None:
                    self._palm_left_since = t
                if (t - self._palm_left_since) * 1000.0 >= cfg.palm_rearm_ms:
                    self._palm_fired = False

        self._recent.append((t, cursor, pinch_dist))
        state = HandState(present=True, cursor=cursor, pinching=self._pinching, scrolling=self._scrolling,
                          open_palm=palm, pinch_dist=round(pinch_dist, 3), span=round(float(width), 4),
                          tip_raw=tip_raw, aim=self._aim)
        return state, events
