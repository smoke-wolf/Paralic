"""Per-connection pipeline for Hand mode.

The eye pipeline lives in session.py; this is its deliberately separate hand
counterpart. It decodes each camera frame, runs HandLandmarker, turns the
landmarks into a cursor (index fingertip) and gestures (pinch = click,
pinch-drag = scroll, open palm = pause), and emits the *same* WebSocket messages
the browser already understands for eyes — a ``frame`` with a ``gaze`` point and
a ``double_blink`` for a click — so the page's cursor / snapping / click / scroll
code is reused unchanged. Scroll and pause add two small hand-only messages.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .filters import OneEuroFilter, OneEuroParams
from .hand_gestures import (HandCalibration, HandGestureConfig, HandGestureRecognizer, PointingMap,
                            estimate_pinch_thresholds)
from .hands import HandTracker
from .session import FrameFormatError, parse_frame
from .tracker import decode_image

log = logging.getLogger(__name__)


class HandSession:
    """Mirrors TrackerSession's frame/command interface for hand control."""

    def __init__(self, tracker_factory: Callable[[], HandTracker],
                 clock: Callable[[], float] = time.monotonic,
                 push: Optional[Callable[[dict], None]] = None,
                 profile_path: Optional[Path] = None):
        self._tracker_factory = tracker_factory
        self._tracker: Optional[HandTracker] = None
        self.clock = clock
        self._push = push or (lambda msg: None)
        self._lock = threading.RLock()
        self.filter = OneEuroFilter(OneEuroParams(min_cutoff=1.5, beta=0.01))
        self.recognizer = HandGestureRecognizer(HandGestureConfig())
        self.screen = {"w": 1920.0, "h": 1080.0}
        self._tracker_errors = 0
        self._profile_path = Path(profile_path) if profile_path else None
        # Calibration collection buffers (filled from labelled frames).
        self._span: list[float] = []
        self._points: dict[int, dict] = {}      # i -> {"target": (fx, fy), "tips": [...]}
        self._pinch_open: list[float] = []
        self._pinch_closed: list[float] = []
        self.calibration: Optional[HandCalibration] = None
        self._saved_summary: Optional[dict] = None
        self._load_profile()

    @property
    def tracker(self) -> HandTracker:
        if self._tracker is None:
            self._tracker = self._tracker_factory()
        return self._tracker

    def close(self) -> None:
        with self._lock:
            if self._tracker is not None:
                self._tracker.close()
                self._tracker = None

    # ------------------------------------------------------------------ frames
    def handle_frame(self, payload: bytes) -> list[dict]:
        with self._lock:
            return self._handle_frame(payload)

    def _handle_frame(self, payload: bytes) -> list[dict]:
        try:
            header, image = parse_frame(payload)
        except (FrameFormatError, ValueError) as exc:
            return [{"type": "frame", "id": None, "face": False, "error": str(exc)}]
        frame_id = header.get("id")
        t = self.clock()
        try:
            rgb = decode_image(image)
        except Exception as exc:
            return [{"type": "frame", "id": frame_id, "face": False, "error": f"decode: {exc}"}]
        try:
            obs = self.tracker.process(rgb, int(t * 1000))
        except Exception as exc:
            self._tracker_errors += 1
            if self._tracker_errors in (1, 10) or self._tracker_errors % 500 == 0:
                log.exception("Hand tracking failed (%d times)", self._tracker_errors)
            return [{"type": "frame", "id": frame_id, "face": False, "error": f"tracker: {exc}"}]

        pts = obs.points_norm if obs is not None else None
        state, events = self.recognizer.update(t, pts)

        # Collect calibration samples if this frame was labelled.
        label = header.get("label") if isinstance(header.get("label"), dict) else None
        labeled = bool(label) and state.present
        if labeled:
            self._collect(label, state)

        msg: dict[str, Any] = {"type": "frame", "id": frame_id, "face": state.present,
                               "frozen": False, "labeled": labeled,
                               "pinching": state.pinching, "paused": state.paused}
        if state.present:
            msg["hand"] = {"span": state.span, "pinch": state.pinch_dist, "open": state.open_palm,
                           "tip": [round(state.tip_raw[0], 4), round(state.tip_raw[1], 4)] if state.tip_raw else None}
        if state.present and state.cursor is not None:
            sx = state.cursor[0] * float(self.screen.get("w", 1920.0))
            sy = state.cursor[1] * float(self.screen.get("h", 1080.0))
            msg["gaze"] = [round(sx, 1), round(sy, 1)]
            msg["raw"] = msg["gaze"]
        else:
            msg["gaze"] = None
            msg["raw"] = None

        out: list[dict] = [msg]
        for ev in events:
            if ev.type == "click":
                # Reuse the browser's double-blink click path.
                out.append({"type": "blink", "n": 2, "frame": frame_id})
                out.append({"type": "double_blink", "frame": frame_id})
            elif ev.type == "scroll":
                out.append({"type": "hand_scroll", "frame": frame_id,
                            "dy": round(ev.dy * float(self.screen.get("h", 1080.0)), 1)})
            elif ev.type == "pause_toggle":
                out.append({"type": "hand_pause", "paused": state.paused})
        return out

    def _collect(self, label: dict, state) -> None:
        kind = label.get("kind")
        if kind == "hspan":
            self._span.append(state.span)
        elif kind == "hpinch":
            (self._pinch_closed if label.get("state") == "closed" else self._pinch_open).append(state.pinch_dist)
        elif kind == "hpoint" and state.tip_raw is not None:
            i = int(label.get("i", 0))
            entry = self._points.setdefault(i, {"target": (float(label.get("fx", 0.5)),
                                                           float(label.get("fy", 0.5))), "tips": []})
            entry["tips"].append(state.tip_raw)

    # ---------------------------------------------------------------- commands
    def handle_command(self, cmd: dict) -> list[dict]:
        kind = cmd.get("type")
        if kind == "hello":
            screen = cmd.get("screen") or {}
            if screen.get("w") and screen.get("h"):
                self.screen = {"w": float(screen["w"]), "h": float(screen["h"])}
            return [{"type": "hello", "mode": "hand", "model": False, "profile": None, "hand": True,
                     "gestures": _gesture_info(), "calibration": self._saved_summary,
                     "calibrated": self.calibration is not None}]
        if kind == "ping":
            return [{"type": "pong", "t": cmd.get("t")}]
        if kind == "set_gestures":
            cfg = self.recognizer.config
            for key in ("pinch_on", "pinch_off", "click_max_ms", "palm_hold_ms", "scroll_start"):
                if key in cmd:
                    setattr(cfg, key, float(cmd[key]))
            return [{"type": "gestures_set", "ok": True}]
        if kind == "hand_calibration_start":
            self._span.clear(); self._points.clear()
            self._pinch_open.clear(); self._pinch_closed.clear()
            return [{"type": "hand_calibration_started", "ok": True, "mode": cmd.get("mode", "full")}]
        if kind == "hand_calibration_fit":
            return [self._fit()]
        if kind == "hand_profile_use":
            if self.calibration is None:
                return [{"type": "hand_calibration_result", "ok": False, "error": "No saved hand calibration"}]
            self.recognizer.apply_calibration(self.calibration)
            return [{"type": "hand_calibration_result", "ok": True, "used": True, **(self._saved_summary or {})}]
        return []

    def _fit(self) -> dict:
        try:
            span = float(np.median(self._span)) if self._span else 0.0
            on = off = None
            if len(self._pinch_open) >= 3 and len(self._pinch_closed) >= 3:
                on, off = estimate_pinch_thresholds(np.array(self._pinch_open), np.array(self._pinch_closed))
            pointing = None
            point_err = None
            usable = [p for p in self._points.values() if len(p["tips"]) >= 3]
            if len(usable) >= 3:
                tips = np.array([np.median(np.array(p["tips"]), axis=0) for p in usable])
                targets = np.array([p["target"] for p in usable])
                pointing = PointingMap.fit(tips, targets)
                pred = np.array([pointing.apply(t) for t in tips])
                point_err = float(np.mean(np.linalg.norm(pred - targets, axis=1)))
            if pointing is None and on is None and span <= 0:
                return {"type": "hand_calibration_result", "ok": False,
                        "error": "Not enough calibration data was collected. Please try again."}
            self.calibration = HandCalibration(span=span, pinch_on=on, pinch_off=off, pointing=pointing)
            self.recognizer.apply_calibration(self.calibration)
            saved = self._save_profile()
            self._saved_summary = self._summary(point_err)
            return {"type": "hand_calibration_result", "ok": True, "saved": saved, **self._saved_summary}
        except Exception as exc:  # keep the connection alive
            log.exception("Hand calibration fit failed")
            return {"type": "hand_calibration_result", "ok": False, "error": str(exc)}

    def _summary(self, point_err: Optional[float] = None) -> dict:
        c = self.calibration
        return {
            "span": round(c.span, 4) if c else None,
            "pinch_on": round(c.pinch_on, 3) if c and c.pinch_on is not None else None,
            "pinch_off": round(c.pinch_off, 3) if c and c.pinch_off is not None else None,
            "has_pointing": bool(c and c.pointing is not None),
            "pointing_error": round(point_err, 4) if point_err is not None else None,
            "n_points": len([p for p in self._points.values() if len(p["tips"]) >= 3]),
        }

    def _save_profile(self) -> bool:
        if self._profile_path is None or self.calibration is None:
            return False
        try:
            self._profile_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"calibration": self.calibration.to_dict(), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self._profile_path.write_text(json.dumps(data))
            return True
        except OSError:
            log.warning("Could not save hand profile", exc_info=True)
            return False

    def _load_profile(self) -> None:
        if self._profile_path is None or not self._profile_path.exists():
            return
        try:
            data = json.loads(self._profile_path.read_text())
            cal = HandCalibration.from_dict(data.get("calibration"))
            if cal is not None:
                self.calibration = cal
                self.recognizer.apply_calibration(cal)
                self._saved_summary = self._summary()
        except (OSError, ValueError, KeyError):
            log.warning("Could not load hand profile", exc_info=True)


def _gesture_info() -> dict:
    return {
        "point": "Move the cursor with your index fingertip.",
        "click": "Pinch your thumb and index finger together.",
        "scroll": "Pinch and move your hand up or down.",
        "pause": "Hold an open palm to the camera to pause or resume.",
    }
