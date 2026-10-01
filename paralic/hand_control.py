"""Hand mode: the cursor follows the index fingertip, a pinch clicks.

Used by a :class:`~paralic.session.TrackerSession` opened in hand mode, so the
people, their settings and desktop control are the same as with the eyes; only
the camera pipeline differs. For each frame it runs the hand recogniser
(``hand_gestures.py``), smooths the cursor, collects the hand setup's samples
and turns gestures into the messages the page already understands for eyes:

* a ``frame`` with the cursor as ``gaze`` (held still while the fingers pinch),
* ``blink`` (n=1) when a pinch starts and ``double_blink`` when it ends as a
  click - with ``pre_frame``/``at`` pointing at the moment just before the
  fingers started to close, so the click lands on what was aimed at,
* ``blink_expired`` when a pinch ends without a click,
* ``hand_scroll`` (pinch-drag) and ``hand_palm`` (the "stop" palm: the page
  toggles its pause).

Each person's hand setup (pointing map, pinch thresholds, hand size) is saved
in their own folder (``users/<id>/hand.json``).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np

from .filters import OneEuroFilter, OneEuroParams
from .hand_gestures import (HandCalibration, HandGestureConfig, HandGestureRecognizer, PointingMap,
                            estimate_pinch_thresholds, palm_centre, palm_width, split_pinch_cycle)

HAND_PROFILE_VERSION = 1
MIN_POINT_SAMPLES = 4
DEFAULT_SCREEN = (1920.0, 1080.0)

GESTURE_HELP = {
    "point": "Move the cursor with your index fingertip.",
    "click": "Pinch your thumb and index finger together, then let go.",
    "scroll": "Pinch, then move your hand up or down.",
    "pause": "Hold up an open hand, fingers spread, to pause or resume.",
}


class HandSetupError(ValueError):
    """The hand setup could not be fitted (the page shows the message)."""


class HandSelector:
    """Another hand in view - someone else's, or the person's other hand: keep
    following the hand in control while it stays in view; when it is gone, the
    largest hand (the one nearest the camera) takes over."""

    FOLLOW_GAP_S = 0.6
    FOLLOW_JUMP = 1.5          # palm widths the followed hand may move between frames

    def __init__(self):
        self.centre: Optional[np.ndarray] = None
        self.width = 1.0
        self.seen_at = -np.inf

    def select(self, t: float, hands: list[np.ndarray]) -> Optional[int]:
        if not hands:
            return None
        pts = [np.asarray(h, float)[:, :2] for h in hands]
        centres = [palm_centre(p) for p in pts]
        widths = [palm_width(p) for p in pts]
        pick = None
        if self.centre is not None and t - self.seen_at <= self.FOLLOW_GAP_S:
            k = min(range(len(pts)), key=lambda j: float(np.hypot(*(centres[j] - self.centre))))
            if float(np.hypot(*(centres[k] - self.centre))) / self.width <= self.FOLLOW_JUMP:
                pick = k
        if pick is None:
            pick = max(range(len(pts)), key=lambda j: widths[j])
        self.centre, self.width, self.seen_at = centres[pick], widths[pick], t
        return pick


def _round(v, nd=3):
    return None if v is None else round(float(v), nd)


class HandControl:
    def __init__(self, config: Optional[HandGestureConfig] = None):
        self.recognizer = HandGestureRecognizer(config or HandGestureConfig())
        self.selector = HandSelector()
        # Light smoothing in screen pixels: the fingertip is far steadier than gaze.
        self.filter = OneEuroFilter(OneEuroParams(min_cutoff=1.2, beta=0.004))
        self.calibration: Optional[HandCalibration] = None
        self.summary: Optional[dict] = None
        self.calibrating = False
        self._mode = "full"
        self._span: list[float] = []
        self._open: list[float] = []
        self._cycle: list[float] = []
        self._points: dict[int, dict] = {}
        self._frames: deque = deque(maxlen=90)    # (t, frame id, cursor in screen px)
        self._last: Optional[np.ndarray] = None

    # -- the person's saved setup ------------------------------------------------
    def load(self, doc: Optional[dict]) -> bool:
        """Use a saved hand setup (``users/<id>/hand.json``); False if none or unreadable."""
        cal = HandCalibration.from_dict((doc or {}).get("calibration")) if isinstance(doc, dict) else None
        self.calibration = cal
        self.summary = dict(doc.get("summary") or {}) if cal is not None else None
        self.recognizer.apply_calibration(cal)
        self.recognizer.reset_motion()
        self.filter.reset()
        return cal is not None

    def document(self, updated: str) -> dict:
        return {"version": HAND_PROFILE_VERSION, "updated": updated,
                "calibration": self.calibration.to_dict() if self.calibration else None,
                "summary": self.summary}

    # -- frames --------------------------------------------------------------------
    def frame(self, t: float, frame_id: Any, pts: Optional[np.ndarray], label: Optional[dict],
              screen: Optional[dict]) -> tuple[dict, list[dict], list]:
        """One camera frame: (frame message fields, event messages, recogniser events)."""
        w, h = DEFAULT_SCREEN
        if isinstance(screen, dict):
            w = float(screen.get("w") or w)
            h = float(screen.get("h") or h)
        state, events = self.recognizer.update(t, pts)
        msg: dict[str, Any] = {"face": state.present, "pinching": state.pinching, "scrolling": state.scrolling,
                               "frozen": False, "labeled": False}
        if state.present:
            msg["hand"] = {"span": state.span, "pinch": state.pinch_dist, "open": state.open_palm,
                           "tip": [round(state.tip_raw[0], 4), round(state.tip_raw[1], 4)]}
            if isinstance(label, dict) and self.calibrating:
                msg["labeled"] = self._collect(label, state)
            raw = np.array([state.cursor[0] * w, state.cursor[1] * h])
            gaze = self.filter(t, raw)
            if state.pinching and state.aim is not None:
                # The fingers closing pull the fingertip: hold the cursor where it aimed.
                gaze = np.array([state.aim[0] * w, state.aim[1] * h])
                msg["frozen"] = True
            self._last = gaze
            msg["raw"] = [round(float(raw[0]), 1), round(float(raw[1]), 1)]
        else:
            msg["raw"] = None
        gaze = self._last
        msg["gaze"] = None if gaze is None else [round(float(gaze[0]), 1), round(float(gaze[1]), 1)]
        self._frames.append((t, frame_id, None if gaze is None else gaze.copy()))
        if not state.present and not state.pinching:
            self.filter.reset()

        out: list[dict] = []
        for ev in events:
            if ev.type == "pinch_start":
                out.append({"type": "blink", "n": 1, "frame": frame_id, "hand": True})
            elif ev.type == "click":
                pre = self._frame_at(ev.aim_t)
                out.append({"type": "double_blink", "frame": frame_id, "hand": True,
                            "pre_frame": pre[1] if pre else None,
                            "at": None if pre is None or pre[2] is None
                            else [round(float(pre[2][0]), 1), round(float(pre[2][1]), 1)]})
            elif ev.type == "pinch_end":
                out.append({"type": "blink_expired", "frame": frame_id, "hand": True})
            elif ev.type == "scroll":
                out.append({"type": "hand_scroll", "frame": frame_id, "dy": round(ev.dy * h, 1)})
            elif ev.type == "palm":
                out.append({"type": "hand_palm", "frame": frame_id})
        return msg, out, events

    def _frame_at(self, t: Optional[float]):
        """The last frame at or before time ``t``."""
        if t is None:
            return None
        best = None
        for rec in self._frames:
            if rec[0] > t + 1e-9:
                break
            best = rec
        return best

    # -- the hand setup --------------------------------------------------------------
    def start_calibration(self, mode: str) -> dict:
        self._mode = "point" if mode == "point" else "full"
        if self._mode == "point" and self.calibration is None:
            raise HandSetupError("There is no hand setup to re-point yet - run the full setup")
        self._span.clear()
        self._open.clear()
        self._cycle.clear()
        self._points.clear()
        self.calibrating = True
        self.recognizer.setup = True
        return {"mode": self._mode}

    def cancel_calibration(self) -> None:
        self.calibrating = False
        self.recognizer.setup = False

    def _collect(self, label: dict, state) -> bool:
        kind = label.get("kind")
        if kind == "hspan":
            if state.open_palm:
                self._span.append(state.span)
                return True
            return False
        if kind == "hpinch":
            (self._cycle if label.get("state") == "cycle" else self._open).append(state.pinch_dist)
            return True
        if kind == "hpoint":
            try:
                i = int(label["i"])
                target = (float(np.clip(float(label["fx"]), 0, 1)), float(np.clip(float(label["fy"]), 0, 1)))
            except (KeyError, TypeError, ValueError):
                return False
            entry = self._points.setdefault(i, {"target": target, "tips": []})
            entry["tips"].append(state.tip_raw)
            return True
        return False

    def fit(self, screen: Optional[dict] = None) -> dict:
        """Fit the setup from the collected samples and use it. Raises HandSetupError."""
        w, h = DEFAULT_SCREEN
        if isinstance(screen, dict):
            w = float(screen.get("w") or w)
            h = float(screen.get("h") or h)
        prev = self.calibration if self._mode == "point" else None
        tips, targets = [], []
        for _, p in sorted(self._points.items()):
            if len(p["tips"]) >= MIN_POINT_SAMPLES:
                # The finger is still on its way at first: the later samples count.
                tail = np.array(p["tips"][len(p["tips"]) // 3:])
                tips.append(np.median(tail, axis=0))
                targets.append(p["target"])
        if len(tips) < 6:
            raise HandSetupError("Your finger was not seen at enough of the dots. Keep your hand in view and try "
                                 "again.")
        tips, targets = np.array(tips), np.array(targets)
        if np.ptp(tips[:, 0]) < 0.03 or np.ptp(tips[:, 1]) < 0.03:
            raise HandSetupError("Your fingertip hardly moved between the dots. Point at each dot, moving your "
                                 "hand, and try again.")
        pointing = PointingMap.fit(tips, targets)
        # Leave-one-out: how far off a dot it was not fitted on would be.
        errs = []
        for k in range(len(tips)):
            keep = np.arange(len(tips)) != k
            m = PointingMap.fit(tips[keep], targets[keep]) if keep.sum() >= 3 else pointing
            d = m.apply(tips[k]) - targets[k]
            errs.append(float(np.hypot(d[0] * w, d[1] * h)))
        span = float(np.median(self._span)) if self._span else (prev.span if prev else 0.0)
        pinch_on = pinch_off = None
        tuned = False
        if prev is not None:
            pinch_on, pinch_off = prev.pinch_on, prev.pinch_off
            tuned = bool((self.summary or {}).get("pinch_tuned"))
        if self._cycle:
            opened, closed = split_pinch_cycle(np.array(self._cycle))
            opened = np.concatenate([opened, np.array(self._open)])
            if len(opened) >= 3 and len(closed) >= 3:
                on, off = estimate_pinch_thresholds(opened, closed, default_on=-1.0, default_off=-1.0)
                if on > 0:
                    pinch_on, pinch_off, tuned = on, off, True
        self.calibration = HandCalibration(span=span, pinch_on=pinch_on, pinch_off=pinch_off, pointing=pointing)
        self.summary = {
            "points": int(len(tips)),
            "pointing_error_px": round(float(np.mean(errs)), 1),
            "span": _round(span, 4),
            "pinch_on": _round(pinch_on),
            "pinch_off": _round(pinch_off),
            "pinch_tuned": tuned,
            "mode": self._mode,
        }
        self.recognizer.apply_calibration(self.calibration)
        self.recognizer.reset_motion()
        self.filter.reset()
        self.cancel_calibration()
        return dict(self.summary)

    def view(self) -> Optional[dict]:
        """What the page shows about the saved setup (None: no setup yet)."""
        return dict(self.summary or {}, calibrated=True) if self.calibration is not None else None

    def set_options(self, cmd: dict) -> dict:
        cfg = self.recognizer.config
        ranges = {"click_max_ms": (250.0, 1500.0), "palm_hold_ms": (300.0, 2500.0),
                  "scroll_gain": (0.1, 2.0), "scroll_start": (0.2, 1.5)}
        for key, (lo, hi) in ranges.items():
            if isinstance(cmd.get(key), (int, float)):
                setattr(cfg, key, float(np.clip(cmd[key], lo, hi)))
        return {k: getattr(cfg, k) for k in ranges}
