"""Eye gestures beyond blinking: winks, and their personal tuning.

A *wink* is one eye closing while the other stays open. Held for a moment it
works like holding a mouse button down: the page drags whatever was under the
cursor for as long as the eye stays closed (or opens a menu when the gaze
stays still - a "long press"), and drops it when the eye opens again. Short
winks are reported as well, for people who prefer a wink to a double blink.

Everyone's eyes are different - one lid may droop, many people squint the
other eye while winking, a blink may be a little lopsided - so each eye has
its own adaptive baseline, and the thresholds can be personalised from a
short recording (:func:`analyze_winks`).

Per-eye closure scores follow MediaPipe's convention: "left" is the
*person's* left eye (it appears on the right of an un-mirrored camera image).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

EYES = ("left", "right")
BLINK_SIGNALS = ("both", "mean", "left", "right")


def other_eye(eye: str) -> str:
    return "right" if eye == "left" else "left"


def blink_signal(closure_left: float, closure_right: float, mode: str = "both") -> float:
    """The closure score the blink detector watches.

    * ``both``  - the more open eye: only a closure of *both* eyes counts, so a
      wink never looks like a blink (the default);
    * ``mean``  - the average of the eyes (for lopsided blinks);
    * ``left`` / ``right`` - one eye only (e.g. when the other eye cannot close).
    """
    if mode == "left":
        return float(closure_left)
    if mode == "right":
        return float(closure_right)
    if mode == "mean":
        return 0.5 * (float(closure_left) + float(closure_right))
    return float(min(closure_left, closure_right))


@dataclass
class EyeWinkConfig:
    enabled: bool = True
    rise: Optional[float] = None    # closure rise above this eye's baseline that counts as closed
    asym: Optional[float] = None    # how much more it must rise than the other eye


@dataclass
class WinkConfig:
    hold_ms: float = 350.0          # a wink held this long becomes a "press"
    quick_min_ms: float = 120.0     # shorter one-eye closures are ignored
    start_frames: int = 2           # consecutive wink frames before it counts
    release_frames: int = 2         # consecutive open frames that end a press
    sensitivity: float = 0.40       # default rise: this fraction of the headroom above the baseline
    asym_min: float = 0.20          # default asymmetry between the eyes
    max_hold_s: float = 60.0
    missing_reset_ms: float = 600.0
    adapt_after_s: float = 2.0      # a lid that stays this "closed" this long (not winking) is its new normal
    baseline_window_s: float = 2.0
    initial_baseline: float = 0.25
    left: EyeWinkConfig = field(default_factory=EyeWinkConfig)
    right: EyeWinkConfig = field(default_factory=EyeWinkConfig)

    def eye(self, eye: str) -> EyeWinkConfig:
        return self.left if eye == "left" else self.right


@dataclass
class WinkEvent:
    type: str              # "wink_start" (held: press) | "wink_end" (released) | "wink" (short wink)
    eye: str
    t: float
    start: float           # when the wink started
    duration_ms: float = 0.0
    cancelled: bool = False


IDLE, CANDIDATE, ACTIVE = "idle", "candidate", "active"


class WinkDetector:
    """Detects one-eye closures from the two per-eye closure scores.

    States: ``idle`` -> ``candidate`` (one eye closed, other open, for a couple
    of frames: the gaze estimate should switch to the open eye) -> ``active``
    (held for ``hold_ms``: a press) -> back to ``idle`` when the eye reopens.
    A candidate that ends early is a short ``wink``; one where the other eye
    closes too is a blink, and is dropped.
    """

    def __init__(self, config: Optional[WinkConfig] = None):
        self.config = config or WinkConfig()
        self._base = {e: self.config.initial_baseline for e in EYES}
        self._samples: dict[str, deque[tuple[float, float]]] = {e: deque() for e in EYES}
        # An eye must be seen open before it can wink (an eye that looks closed
        # from the start - a patch, a drooping lid - never triggers a press).
        self._armed = {e: False for e in EYES}
        self._high_since: dict[str, Optional[float]] = {e: None for e in EYES}
        self.state = IDLE
        self.eye: Optional[str] = None
        self.since = 0.0
        self._pending: Optional[str] = None
        self._run = 0
        self._run_start = 0.0
        self._open_run = 0
        self._last_seen: Optional[float] = None

    # -- thresholds ---------------------------------------------------------
    def baseline(self, eye: str) -> float:
        return self._base[eye]

    def thresholds(self, eye: str) -> tuple[float, float, float]:
        """(closed level, open level, asymmetry) for ``eye``."""
        cfg = self.config
        ec = cfg.eye(eye)
        b = self._base[eye]
        rise = ec.rise if ec.rise is not None else cfg.sensitivity * (1.0 - b)
        close = min(b + max(rise, 0.08), 0.95)
        return close, b + 0.5 * (close - b), ec.asym if ec.asym is not None else cfg.asym_min

    def _winking(self, eye: str, c: dict[str, float]) -> bool:
        close, _, asym = self.thresholds(eye)
        o = other_eye(eye)
        return c[eye] >= close and (c[eye] - self._base[eye]) - (c[o] - self._base[o]) >= asym

    @property
    def winking(self) -> Optional[str]:
        """The eye that is (probably) winking right now, including candidates."""
        return self.eye if self.state in (CANDIDATE, ACTIVE) else None

    @property
    def pressed(self) -> Optional[str]:
        return self.eye if self.state == ACTIVE else None

    def reset(self) -> None:
        self.__init__(self.config)

    # -- updates ------------------------------------------------------------
    def update_missing(self, t: float) -> list[WinkEvent]:
        """Call when no face was found in a frame."""
        self._pending, self._run = None, 0   # a wink that was only starting is forgotten
        if self.state == IDLE or self._last_seen is None:
            return []
        if (t - self._last_seen) * 1000.0 <= self.config.missing_reset_ms:
            return []
        return self._finish(t, cancelled=True)

    def update(self, t: float, closure_left: float, closure_right: float) -> list[WinkEvent]:
        cfg = self.config
        c = {"left": float(closure_left), "right": float(closure_right)}
        if self._last_seen is not None and t - self._last_seen > 0.25:
            self._pending, self._run = None, 0   # frames were missing: start counting again
        self._last_seen = t
        events: list[WinkEvent] = []
        for e in EYES:
            if not self._armed[e] and c[e] < self.thresholds(e)[1]:
                self._armed[e] = True

        if self.state == IDLE:
            ready = [e for e in EYES if cfg.eye(e).enabled and self._armed[e] and self._winking(e, c)]
            if ready:
                eye = max(ready, key=lambda e: c[e] - self._base[e])
                if eye != self._pending:
                    self._pending, self._run, self._run_start = eye, 0, t
                self._run += 1
                if self._run >= cfg.start_frames:
                    self.state, self.eye, self.since = CANDIDATE, eye, self._run_start
                    self._pending, self._run = None, 0
            else:
                self._pending, self._run = None, 0
        elif self.state == CANDIDATE:
            eye = self.eye
            o = other_eye(eye)
            close, open_level, asym = self.thresholds(eye)
            duration_ms = (t - self.since) * 1000.0
            if c[eye] < open_level:
                if duration_ms >= cfg.quick_min_ms:
                    events.append(WinkEvent("wink", eye, t, self.since, duration_ms))
                self._to_idle()
            elif (c[eye] - self._base[eye]) - (c[o] - self._base[o]) < 0.5 * asym:
                # The other eye closed as well: a blink or a squeeze, not a wink.
                self._armed[eye] = False
                self._to_idle()
            elif duration_ms >= cfg.hold_ms:
                self.state = ACTIVE
                self._open_run = 0
                events.append(WinkEvent("wink_start", eye, t, self.since, duration_ms))
        elif self.state == ACTIVE:
            eye = self.eye
            if c[eye] < self.thresholds(eye)[1]:
                self._open_run += 1
                if self._open_run >= cfg.release_frames:
                    events.extend(self._finish(t))
            else:
                self._open_run = 0
                if t - self.since > cfg.max_hold_s:
                    events.extend(self._finish(t, cancelled=True))

        # Baselines: learn from open eyes, never from a winking eye. A lid that
        # stays low for a while without a wink - a drooping lid, an eye patch,
        # looking at the bottom of the screen - becomes the new normal. Both
        # eyes learn together: if one lid is up and the other down (both drop
        # when looking down, but not equally far past their thresholds), moving
        # only one baseline would fake a wink.
        winking = self.winking
        high = {e: c[e] >= self.thresholds(e)[0] for e in EYES}
        for e in EYES:
            if e == winking or not high[e]:
                self._high_since[e] = None
            elif self._high_since[e] is None:
                self._high_since[e] = t
        settled = {e: self._high_since[e] is not None and t - self._high_since[e] >= cfg.adapt_after_s
                   for e in EYES}
        if winking is None and all(not high[e] or settled[e] for e in EYES):
            for e in EYES:
                self._add_sample(e, t, c[e])
        return events

    def _finish(self, t: float, cancelled: bool = False) -> list[WinkEvent]:
        eye, since, state = self.eye, self.since, self.state
        self._to_idle()
        if cancelled:
            self._armed[eye] = False
        if state != ACTIVE:
            return []
        return [WinkEvent("wink_end", eye, t, since, (t - since) * 1000.0, cancelled)]

    def _to_idle(self) -> None:
        self.state, self.eye = IDLE, None
        self._pending, self._run, self._open_run = None, 0, 0

    def _add_sample(self, eye: str, t: float, value: float) -> None:
        samples = self._samples[eye]
        samples.append((t, value))
        horizon = t - self.config.baseline_window_s
        while samples and samples[0][0] < horizon:
            samples.popleft()
        if len(samples) >= 5:
            self._base[eye] = float(np.median([v for _, v in samples]))


def wink_config(personal: dict, gestures: Optional[dict] = None) -> WinkConfig:
    """This person's wink settings: calibrated thresholds, enabled eyes and hold time."""
    cfg = WinkConfig()
    wink = personal.get("wink") or {}
    g = gestures if gestures is not None else (personal.get("gestures") or {})
    blink = personal.get("blink") or {}
    if isinstance(g.get("hold_ms"), (int, float)):
        cfg.hold_ms = float(np.clip(g["hold_ms"], 200.0, 1500.0))
    # Winks must be clearly more lopsided than this person's ordinary blinks,
    # and a held wink clearly longer than a blink.
    blink_floor = 0.0
    lopsided = False
    if blink.get("blink_asym") is not None:
        wanted = 1.5 * float(blink["blink_asym"])
        # Ordinary blinks this one-sided (e.g. one eye hardly closes) cannot be
        # told apart from winks: winks stay off unless the person turns them on.
        lopsided = wanted > 0.6
        blink_floor = float(np.clip(wanted, 0.0, 0.6))
        cfg.asym_min = max(cfg.asym_min, blink_floor)
    if blink.get("blink_ms") is not None:
        cfg.hold_ms = max(cfg.hold_ms, float(blink["blink_ms"]) * 1.5)
    for eye in EYES:
        ec = cfg.eye(eye)
        w = wink.get(eye) or {}
        if w.get("ok"):
            ec.rise = float(w["threshold_rise"])
            ec.asym = float(max(w["threshold_asym"], blink_floor))
        mapping = g.get(f"{eye}_hold", "drag")
        ec.enabled = mapping != "off" or g.get(f"{eye}_quick", "off") != "off"
        if w and not w.get("ok") and not g.get(f"{eye}_forced"):
            ec.enabled = False  # tested: this eye cannot wink reliably on its own
        if lopsided and not g.get(f"{eye}_forced"):
            ec.enabled = False
    return cfg


# ---------------------------------------------------------------------------
# Personalisation: a short "close your left eye / right eye" recording
# ---------------------------------------------------------------------------

def analyze_winks(samples: Sequence[tuple], settle_ms: float = 400.0) -> dict:
    """Learn how someone winks.

    ``samples`` are (time s, closure left, closure right, phase) per frame,
    where phase is "rest" (both eyes open), "left" or "right" (asked to keep
    that eye closed). The first ``settle_ms`` of every prompt are skipped
    (the eye is still closing). For each eye we measure how far it closes,
    how much the other eye closes with it, and whether a detector with
    thresholds at half of those values would hold the wink steadily.
    """
    from .personalize import PersonalizationError

    rows = [s for s in samples if len(s) >= 4 and s[3] in ("rest", "left", "right")]
    rest = np.array([(s[1], s[2]) for s in rows if s[3] == "rest"], float)
    if len(rest) < 10:
        raise PersonalizationError("Not enough camera frames with both eyes open - keep your face in view.")
    base = {"left": float(np.median(rest[:, 0])), "right": float(np.median(rest[:, 1]))}
    noise = {e: max(1.4826 * float(np.median(np.abs(rest[:, i] - base[e]))), 0.01)
             for i, e in enumerate(EYES)}

    result: dict = {"updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for eye in EYES:
        o = other_eye(eye)
        held = []
        seg_start = None
        prev_phase = None
        for s in rows:
            t, phase = float(s[0]), s[3]
            if phase != prev_phase:
                seg_start = t
                prev_phase = phase
            if phase == eye and (t - seg_start) * 1000.0 >= settle_ms:
                held.append((s[1], s[2]))
        if len(held) < 8:
            result[eye] = {"ok": False, "reason": "not recorded"}
            continue
        H = np.array(held, float)
        ce = H[:, 0] if eye == "left" else H[:, 1]
        co = H[:, 1] if eye == "left" else H[:, 0]
        rise = float(np.median(ce)) - base[eye]
        other_rise = float(np.median(co)) - base[o]
        asym = rise - other_rise
        thr_rise = max(0.5 * rise, 4.0 * noise[eye], 0.08)
        thr_asym = max(0.5 * asym, 4.0 * max(noise[eye], noise[o]), 0.08)
        hits = ((ce - base[eye]) >= thr_rise) & (((ce - base[eye]) - (co - base[o])) >= thr_asym)
        hold_fraction = float(np.mean(hits))
        ok = rise >= max(0.15, 6.0 * noise[eye]) and asym >= max(0.12, 6.0 * noise[eye]) and hold_fraction >= 0.8
        reason = None
        if not ok:
            if rise < max(0.15, 6.0 * noise[eye]):
                reason = "the eye did not close far enough"
            elif asym < max(0.12, 6.0 * noise[eye]):
                reason = "the other eye closed too"
            else:
                reason = "the eye did not stay closed steadily"
        result[eye] = {
            "ok": bool(ok),
            "reason": reason,
            "baseline": round(base[eye], 3),
            "noise": round(noise[eye], 4),
            "rise": round(rise, 3),
            "other_rise": round(other_rise, 3),
            "asym": round(asym, 3),
            "hold_fraction": round(hold_fraction, 3),
            "threshold_rise": round(thr_rise, 3),
            "threshold_asym": round(thr_asym, 3),
            "n_frames": int(len(H)),
        }
    if all(not result[e].get("n_frames") for e in EYES):
        raise PersonalizationError("I couldn't see either eye closed - try again, keeping one eye closed "
                                   "until the circle fills.")
    return result
