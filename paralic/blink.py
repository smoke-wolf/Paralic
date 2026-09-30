"""Blink and double-blink detection.

The input is a per-frame eye *closure* score between 0 (wide open) and 1
(closed), produced from MediaPipe's blink blendshapes and eyelid geometry.

Everyone's eyes look different and the lids drop when you look down, so the
thresholds are *adaptive*: we track the median closure of recent open-eye frames
(the baseline) and call the eyes closed when the score rises well above it.

A *blink* is a closure that lasts between ``min_closed_ms`` and
``max_closed_ms``. A *double blink* is two blinks where the eyes were open for at
most ``double_gap_ms`` in between. Longer closures (resting your eyes, looking
far down) never count as blinks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

BLINK_SENSITIVITY_PRESETS = {"low": 0.40, "normal": 0.30, "high": 0.22}
DOUBLE_BLINK_GAP_PRESETS = {"fast": 380.0, "normal": 550.0, "relaxed": 800.0}


@dataclass
class BlinkConfig:
    sensitivity: float = 0.30        # closure rise above baseline needed to count as closed
    min_closed_ms: float = 25.0
    max_closed_ms: float = 700.0
    double_gap_ms: float = 550.0     # longest allowed "open" pause between the two blinks
    min_gap_ms: float = 30.0
    baseline_window_s: float = 1.6
    initial_baseline: float = 0.2
    missing_reset_ms: float = 600.0


@dataclass
class BlinkEvent:
    type: str            # "blink" | "double_blink" | "blink_expired"
    t: float             # event time (seconds)
    count: int = 0       # blinks in the current sequence (for "blink")
    first_start: Optional[float] = None   # when the first blink of the sequence started


@dataclass
class BlinkState:
    closing: bool          # closure is above the midway threshold (blink may be starting)
    closed: bool           # eyes are considered closed
    baseline: float
    close_threshold: float
    open_threshold: float
    pending: int           # blinks waiting for a partner (0 or 1)


@dataclass
class _Blink:
    start: float
    end: float


class BlinkDetector:
    def __init__(self, config: Optional[BlinkConfig] = None):
        self.config = config or BlinkConfig()
        self._closed = False
        self._closed_since = 0.0
        self._pending: Optional[_Blink] = None
        self._samples: deque[tuple[float, float]] = deque()
        self._baseline = self.config.initial_baseline
        self._last_seen: Optional[float] = None
        self._closure = 0.0

    # -- thresholds ---------------------------------------------------------
    @property
    def baseline(self) -> float:
        return self._baseline

    def thresholds(self) -> tuple[float, float]:
        """(close threshold, open threshold)."""
        b = self._baseline
        t_close = float(np.clip(b + self.config.sensitivity, 0.30, 0.92))
        t_open = b + 0.5 * (t_close - b)
        return t_close, t_open

    def state(self) -> BlinkState:
        t_close, t_open = self.thresholds()
        return BlinkState(
            closing=self._closed or self._closure >= t_open,
            closed=self._closed,
            baseline=self._baseline,
            close_threshold=t_close,
            open_threshold=t_open,
            pending=1 if self._pending else 0,
        )

    def reset(self) -> None:
        self._closed = False
        self._pending = None
        self._samples.clear()
        self._baseline = self.config.initial_baseline
        self._last_seen = None
        self._closure = 0.0

    # -- updates ------------------------------------------------------------
    def update_missing(self, t: float) -> list[BlinkEvent]:
        """Call when no face was found in a frame."""
        events: list[BlinkEvent] = []
        if self._last_seen is not None and (t - self._last_seen) * 1000.0 > self.config.missing_reset_ms:
            self._closed = False
            if self._pending is not None:
                events.append(BlinkEvent("blink_expired", t))
            self._pending = None
            self._closure = 0.0
        return events

    def update(self, t: float, closure: float) -> list[BlinkEvent]:
        cfg = self.config
        events: list[BlinkEvent] = []
        self._last_seen = t
        self._closure = closure
        t_close, t_open = self.thresholds()

        if not self._closed:
            self._add_sample(t, closure)
            if closure >= t_close:
                self._closed = True
                self._closed_since = t
            elif self._pending is not None and (t - self._pending.end) * 1000.0 > cfg.double_gap_ms:
                self._pending = None
                events.append(BlinkEvent("blink_expired", t))
        else:
            duration_ms = (t - self._closed_since) * 1000.0
            if duration_ms > cfg.max_closed_ms:
                # Not a blink. Let long closures (e.g. looking far down) teach the
                # baseline so the detector recovers instead of staying "closed".
                self._add_sample(t, closure)
            if closure < t_open:
                self._closed = False
                events.extend(self._on_reopen(t, duration_ms))
        return events

    # -- internals ----------------------------------------------------------
    def _on_reopen(self, t: float, duration_ms: float) -> list[BlinkEvent]:
        cfg = self.config
        if not (cfg.min_closed_ms <= duration_ms <= cfg.max_closed_ms):
            expired = self._pending is not None
            self._pending = None
            return [BlinkEvent("blink_expired", t)] if expired else []

        blink = _Blink(start=self._closed_since, end=t)
        if self._pending is not None:
            gap_ms = (blink.start - self._pending.end) * 1000.0
            if cfg.min_gap_ms <= gap_ms <= cfg.double_gap_ms:
                first = self._pending
                self._pending = None
                return [BlinkEvent("blink", t, count=2, first_start=first.start),
                        BlinkEvent("double_blink", t, count=2, first_start=first.start)]
        self._pending = blink
        return [BlinkEvent("blink", t, count=1, first_start=blink.start)]

    def _add_sample(self, t: float, closure: float) -> None:
        self._samples.append((t, closure))
        horizon = t - self.config.baseline_window_s
        while self._samples and self._samples[0][0] < horizon:
            self._samples.popleft()
        if len(self._samples) >= 5:
            self._baseline = float(np.median([c for _, c in self._samples]))
