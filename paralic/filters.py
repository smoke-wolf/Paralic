"""Signal smoothing for the gaze cursor.

* :class:`OneEuroFilter` - the classic adaptive low-pass filter (Casiez et al.,
  CHI 2012). It smooths heavily while the eyes rest on something (removing
  jitter) and lightly during fast eye movements (keeping lag low).
* :class:`GazeStabilizer` - wraps the filter and freezes the cursor while the
  eyes are closing / closed, rewinding to where you were looking just before
  the blink. Without this the cursor would jump every time you blink, which
  would make "blink twice to click" land in the wrong place.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _alpha(dt: float, cutoff_hz: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


@dataclass
class OneEuroParams:
    min_cutoff: float = 0.5   # Hz: lower = smoother but laggier while fixating
    beta: float = 0.0008      # speed coefficient (per px/s): higher = snappier saccades
    d_cutoff: float = 1.0     # Hz: smoothing of the speed estimate


# Tuned on simulated gaze with ~50 px per-frame noise: "medium" cuts the jitter
# to about a third and settles on a new target ~0.1 s after a large eye jump.
SMOOTHING_PRESETS: dict[str, OneEuroParams] = {
    "low": OneEuroParams(min_cutoff=1.0, beta=0.002),
    "medium": OneEuroParams(min_cutoff=0.5, beta=0.0008),
    "high": OneEuroParams(min_cutoff=0.3, beta=0.0004),
}


class OneEuroFilter:
    """One Euro filter for a vector signal (the cutoff adapts to the vector speed)."""

    def __init__(self, params: Optional[OneEuroParams] = None):
        self.params = params or OneEuroParams()
        self._t: Optional[float] = None
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None

    def reset(self, value: Optional[np.ndarray] = None, t: Optional[float] = None) -> None:
        if value is None:
            self._t = self._x = self._dx = None
        else:
            self._x = np.asarray(value, dtype=np.float64).copy()
            self._dx = np.zeros_like(self._x)
            self._t = t

    @property
    def value(self) -> Optional[np.ndarray]:
        return None if self._x is None else self._x.copy()

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None:
            self._t, self._x, self._dx = t, x.copy(), np.zeros_like(x)
            return x.copy()
        dt = t - self._t
        if dt <= 0:
            dt = 1e-3
        p = self.params
        a_d = _alpha(dt, p.d_cutoff)
        dx = (x - self._x) / dt
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = p.min_cutoff + p.beta * float(np.linalg.norm(self._dx))
        a = _alpha(dt, cutoff)
        self._x = a * x + (1.0 - a) * self._x
        self._t = t
        return self._x.copy()


class GazeStabilizer:
    """Smooths gaze predictions and holds the cursor still through blinks.

    ``update`` is called once per frame with the raw gaze prediction (or None
    when there is no prediction) and whether the eyes are currently closing or
    closed. It returns the position to display and whether the cursor is frozen.
    """

    def __init__(self, params: Optional[OneEuroParams] = None, rewind_s: float = 0.10,
                 settle_s: float = 0.15, max_freeze_s: float = 0.9):
        self.filter = OneEuroFilter(params)
        self.rewind_s = rewind_s
        self.settle_s = settle_s
        self.max_freeze_s = max_freeze_s
        self._history: deque[tuple[float, np.ndarray]] = deque()
        self._frozen = False
        self._freeze_pos: Optional[np.ndarray] = None
        self._freeze_start = 0.0
        self._release_at: Optional[float] = None
        self._freeze_expired = False
        self._last: Optional[np.ndarray] = None

    def set_params(self, params: OneEuroParams) -> None:
        self.filter.params = params

    def reset(self) -> None:
        self.filter.reset()
        self._history.clear()
        self._frozen = False
        self._freeze_pos = None
        self._release_at = None
        self._freeze_expired = False
        self._last = None

    @property
    def frozen(self) -> bool:
        return self._frozen

    def position_before(self, t: float) -> Optional[np.ndarray]:
        """Smoothed position at (or just before) time ``t``."""
        best = None
        for ts, pos in self._history:
            if ts <= t:
                best = pos
            else:
                break
        if best is None and self._history:
            best = self._history[0][1]
        if best is None:
            best = self._last
        return None if best is None else best.copy()

    def update(self, t: float, raw: Optional[np.ndarray], eyes_closing: bool) -> tuple[Optional[np.ndarray], bool]:
        if not eyes_closing:
            self._freeze_expired = False
        if eyes_closing and not self._freeze_expired:
            if not self._frozen:
                self._frozen = True
                self._freeze_start = t
                self._freeze_pos = self.position_before(t - self.rewind_s)
            self._release_at = None
            if t - self._freeze_start <= self.max_freeze_s:
                return self._copy(self._freeze_pos), True
            # Eyes have been "closing" for too long: it is not a blink (probably
            # looking down). Let the cursor move again.
            self._freeze_expired = True
            self._release(t)
        elif self._frozen:
            if self._release_at is None:
                self._release_at = t + self.settle_s
            if t < self._release_at:
                return self._copy(self._freeze_pos), True
            self._release(t)

        if raw is None:
            return self._copy(self._last), False
        out = self.filter(t, raw)
        self._last = out
        self._history.append((t, out.copy()))
        while self._history and self._history[0][0] < t - 1.0:
            self._history.popleft()
        return out.copy(), False

    def _release(self, t: float) -> None:
        self._frozen = False
        self._release_at = None
        if self._freeze_pos is not None:
            # Restart smoothing from the frozen point, as if it had been
            # observed one frame ago, so the cursor glides on from there.
            self.filter.reset(self._freeze_pos, t - 1.0 / 30.0)
            self._last = self._freeze_pos.copy()

    @staticmethod
    def _copy(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return None if v is None else v.copy()
