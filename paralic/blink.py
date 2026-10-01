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

A *long close* is a deliberate closure of both eyes for ``long_close_ms`` to
``long_close_max_ms`` (an alternative to winking for people who cannot close
one eye on its own). It has to be *deep* - close to a full blink - for most of
that time, so that looking at the bottom of the screen (which lowers the lids
too) does not count.
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
    min_threshold: float = 0.30      # lowest "closed" threshold (personalised for light blinkers)
    long_close_ms: float = 1000.0
    long_close_max_ms: float = 6000.0
    deep_rise: Optional[float] = None  # rise above the baseline that counts as fully closed (personal)


@dataclass
class BlinkEvent:
    type: str            # "blink" | "double_blink" | "blink_expired" | "long_close_ready" | "long_close"
    t: float             # event time (seconds)
    count: int = 0       # blinks in the current sequence (for "blink")
    first_start: Optional[float] = None   # when the first blink of the sequence (or the closure) started
    duration_ms: float = 0.0


@dataclass
class BlinkState:
    closing: bool          # closure is above the midway threshold (blink may be starting)
    closed: bool           # eyes are considered closed
    baseline: float
    close_threshold: float
    open_threshold: float
    pending: int           # blinks waiting for a partner (0 or 1)
    deep: bool = False     # closed as in a full blink (keeps the cursor frozen during a long close)


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
        self._closed_base = self._baseline   # baseline when the current closure started
        self._closed_frames = 0
        self._deep_frames = 0
        self._ready_sent = False

    # -- thresholds ---------------------------------------------------------
    @property
    def baseline(self) -> float:
        return self._baseline

    def thresholds(self) -> tuple[float, float]:
        """(close threshold, open threshold).

        The rise needed to count as "closed" is a fraction of the headroom
        above the baseline, so blinks are still caught when the baseline is
        high (e.g. while looking at the bottom of the screen).
        """
        b = self._baseline
        t_close = float(np.clip(b + self.config.sensitivity * (1.0 - b), self.config.min_threshold, 0.92))
        t_open = b + 0.5 * (t_close - b)
        return t_close, t_open

    def deep_level(self, base: Optional[float] = None) -> float:
        """Closure that counts as eyes fully shut (for long closes)."""
        b = self._closed_base if base is None else base
        rise = self.config.deep_rise if self.config.deep_rise else 0.6 * (1.0 - b)
        return float(min(0.97, max(b + max(rise, 0.2), 0.5)))

    def state(self) -> BlinkState:
        t_close, t_open = self.thresholds()
        return BlinkState(
            closing=self._closed or self._closure >= t_open,
            closed=self._closed,
            baseline=self._baseline,
            close_threshold=t_close,
            open_threshold=t_open,
            pending=1 if self._pending else 0,
            deep=self._closed and self._closure >= self.deep_level(),
        )

    def reset(self) -> None:
        self._closed = False
        self._pending = None
        self._samples.clear()
        self._baseline = self.config.initial_baseline
        self._last_seen = None
        self._closure = 0.0

    def cancel(self) -> bool:
        """Forget the closure in progress and any blink waiting for a partner.

        Returns True if a pending blink was dropped (the caller may tell the
        page that the first blink of a double blink expired).
        """
        dropped = self._pending is not None
        self._closed = False
        self._pending = None
        return dropped

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
                self._closed_base = self._baseline
                self._closed_frames = 0
                self._deep_frames = 0
                self._ready_sent = False
            elif self._pending is not None and (t - self._pending.end) * 1000.0 > cfg.double_gap_ms:
                self._pending = None
                events.append(BlinkEvent("blink_expired", t))
        if self._closed:
            duration_ms = (t - self._closed_since) * 1000.0
            deep = closure >= self.deep_level()
            self._closed_frames += 1
            self._deep_frames += int(deep)
            if duration_ms > cfg.max_closed_ms and not deep:
                # Not a blink. Let long closures (e.g. looking far down) teach the
                # baseline so the detector recovers instead of staying "closed".
                # (Eyes that are really shut keep the old baseline.)
                self._add_sample(t, closure)
            if closure < t_open:
                self._closed = False
                events.extend(self._on_reopen(t, duration_ms))
            elif (not self._ready_sent and cfg.long_close_ms <= duration_ms <= cfg.long_close_max_ms
                  and self._mostly_deep()):
                self._ready_sent = True
                events.append(BlinkEvent("long_close_ready", t, first_start=self._closed_since,
                                         duration_ms=duration_ms))
        return events

    def _mostly_deep(self) -> bool:
        return self._closed_frames > 0 and self._deep_frames >= 0.8 * self._closed_frames

    # -- internals ----------------------------------------------------------
    def _on_reopen(self, t: float, duration_ms: float) -> list[BlinkEvent]:
        cfg = self.config
        # Once "long close" was announced, this closure is never a blink.
        if self._ready_sent or not (cfg.min_closed_ms <= duration_ms <= cfg.max_closed_ms):
            expired = self._pending is not None
            self._pending = None
            events = [BlinkEvent("blink_expired", t)] if expired else []
            long_enough = self._ready_sent or duration_ms >= cfg.long_close_ms
            if long_enough and duration_ms <= cfg.long_close_max_ms and self._mostly_deep():
                events.append(BlinkEvent("long_close", t, first_start=self._closed_since, duration_ms=duration_ms))
            return events

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
