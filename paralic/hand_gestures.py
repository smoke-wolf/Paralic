"""Finger-gesture recognition for Hand mode.

MediaPipe Hands gives 21 landmarks per hand. This module turns a stream of those
landmarks into the cursor position and discrete gestures that drive the whole
site — the hand equivalent of blink.py for eyes. It is pure NumPy and holds no
MediaPipe or I/O, so the state machine can be unit-tested directly.

Gesture set (usable hand-only):

* **Point**     - the index fingertip is the cursor.
* **Pinch**     - thumb tip touches index tip. A quick pinch-and-release is a
  **click**; holding the pinch and moving the hand up/down is a **scroll**
  (pinch-drag), so you never need a mouse wheel.
* **Open palm** - all four fingers extended, held briefly, toggles **pause**
  (nothing is clicked while paused; open-palm again to resume).

Thresholds are expressed relative to the palm width, so they do not depend on
how close the hand is to the camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Landmark indices (MediaPipe Hands).
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20


@dataclass
class HandGestureConfig:
    pinch_on: float = 0.45        # thumb-index distance / palm width to start a pinch
    pinch_off: float = 0.60       # hysteresis: release above this
    click_max_ms: float = 450.0   # a pinch shorter than this (and still) is a click
    click_move: float = 0.04      # max cursor travel (screen fraction) to still count as a click
    scroll_start: float = 0.05    # pinch-drag beyond this (screen fraction) starts scrolling
    palm_hold_ms: float = 450.0   # open palm held this long toggles pause
    palm_cooldown_ms: float = 1200.0
    extend_ratio: float = 1.15    # tip is "extended" when this much farther from the wrist than its PIP


@dataclass
class HandEvent:
    type: str                     # "click" | "scroll" | "pause_toggle"
    t: float
    dy: float = 0.0               # scroll amount (screen fractions, + = down)


@dataclass
class HandState:
    present: bool
    cursor: Optional[tuple[float, float]]   # index fingertip, screen fraction (already mirrored)
    pinching: bool
    scrolling: bool
    open_palm: bool
    paused: bool
    pinch_dist: float = 0.0


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def palm_width(pts: np.ndarray) -> float:
    """A scale invariant to distance from the camera: the knuckle span."""
    return max(_dist(pts[INDEX_MCP], pts[PINKY_MCP]), 1e-6)


def finger_extended(pts: np.ndarray, tip: int, pip: int) -> bool:
    w = pts[WRIST]
    return _dist(pts[tip], w) > _dist(pts[pip], w) * 1.15


def extended_fingers(pts: np.ndarray) -> dict[str, bool]:
    return {
        "index": finger_extended(pts, INDEX_TIP, INDEX_PIP),
        "middle": finger_extended(pts, MIDDLE_TIP, MIDDLE_PIP),
        "ring": finger_extended(pts, RING_TIP, RING_PIP),
        "pinky": finger_extended(pts, PINKY_TIP, PINKY_PIP),
    }


class HandGestureRecognizer:
    """Turns per-frame hand landmarks into a cursor and click/scroll/pause events."""

    def __init__(self, config: Optional[HandGestureConfig] = None):
        self.config = config or HandGestureConfig()
        self._pinching = False
        self._pinch_start_t = 0.0
        self._pinch_start_xy: Optional[np.ndarray] = None
        self._scrolling = False
        self._scroll_last_y = 0.0
        self._palm_since: Optional[float] = None
        self._palm_cooldown_until = 0.0
        self._paused = False
        self._last_cursor: Optional[tuple[float, float]] = None

    @property
    def paused(self) -> bool:
        return self._paused

    def reset(self) -> None:
        self.__init__(self.config)

    def update(self, t: float, pts: Optional[np.ndarray], *, mirror: bool = True) -> tuple[HandState, list[HandEvent]]:
        """``pts`` is a (21, 2+) array of normalised (0..1) landmarks, or None.

        Returns the current state and any events fired this frame. The cursor is
        the index fingertip in *screen fraction* coordinates (x mirrored for a
        front camera when ``mirror`` is True).
        """
        cfg = self.config
        events: list[HandEvent] = []
        if pts is None or len(pts) < 21:
            # Hand lost: end any pinch/scroll without firing a click.
            self._pinching = self._scrolling = False
            self._pinch_start_xy = None
            self._palm_since = None
            return HandState(False, self._last_cursor, False, False, False, self._paused), events

        pts = np.asarray(pts, dtype=np.float64)[:, :2]
        tip = pts[INDEX_TIP]
        cx = (1.0 - tip[0]) if mirror else tip[0]
        cursor = (float(np.clip(cx, 0.0, 1.0)), float(np.clip(tip[1], 0.0, 1.0)))
        self._last_cursor = cursor
        cur = np.array(cursor)

        width = palm_width(pts)
        pinch_dist = _dist(pts[THUMB_TIP], pts[INDEX_TIP]) / width
        fingers = extended_fingers(pts)
        open_palm = all(fingers.values())

        # -- pinch (click / scroll) --------------------------------------------
        if not self._pinching and pinch_dist < cfg.pinch_on:
            self._pinching = True
            self._pinch_start_t = t
            self._pinch_start_xy = cur
            self._scrolling = False
            self._scroll_last_y = cursor[1]
        elif self._pinching:
            moved = _dist(cur, self._pinch_start_xy) if self._pinch_start_xy is not None else 0.0
            if not self._scrolling and moved > cfg.scroll_start:
                self._scrolling = True
                # Keep _scroll_last_y at the pinch-start y so the first frame of
                # scrolling already reports the distance travelled since the pinch.
            if self._scrolling:
                dy = cursor[1] - self._scroll_last_y
                self._scroll_last_y = cursor[1]
                if abs(dy) > 1e-4 and not self._paused:
                    events.append(HandEvent("scroll", t, dy=dy))
            if pinch_dist > cfg.pinch_off:                 # released
                held_ms = (t - self._pinch_start_t) * 1000.0
                was_click = (not self._scrolling and held_ms <= cfg.click_max_ms and moved <= cfg.click_move)
                self._pinching = False
                self._scrolling = False
                self._pinch_start_xy = None
                if was_click and not self._paused:
                    events.append(HandEvent("click", t))

        # -- open palm (pause toggle) ------------------------------------------
        if open_palm and not self._pinching:
            if self._palm_since is None:
                self._palm_since = t
            elif (t - self._palm_since) * 1000.0 >= cfg.palm_hold_ms and t >= self._palm_cooldown_until:
                self._paused = not self._paused
                self._palm_cooldown_until = t + cfg.palm_cooldown_ms / 1000.0
                self._palm_since = None
                events.append(HandEvent("pause_toggle", t))
        else:
            self._palm_since = None

        state = HandState(present=True, cursor=cursor, pinching=self._pinching, scrolling=self._scrolling,
                          open_palm=open_palm, paused=self._paused, pinch_dist=round(pinch_dist, 3))
        return state, events
