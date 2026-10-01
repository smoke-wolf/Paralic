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

import logging
import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from .filters import OneEuroFilter, OneEuroParams
from .hand_gestures import HandGestureConfig, HandGestureRecognizer
from .hands import HandTracker
from .session import FrameFormatError, parse_frame
from .tracker import decode_image

log = logging.getLogger(__name__)


class HandSession:
    """Mirrors TrackerSession's frame/command interface for hand control."""

    def __init__(self, tracker_factory: Callable[[], HandTracker],
                 clock: Callable[[], float] = time.monotonic,
                 push: Optional[Callable[[dict], None]] = None):
        self._tracker_factory = tracker_factory
        self._tracker: Optional[HandTracker] = None
        self.clock = clock
        self._push = push or (lambda msg: None)
        self._lock = threading.RLock()
        self.filter = OneEuroFilter(OneEuroParams(min_cutoff=1.5, beta=0.01))
        self.recognizer = HandGestureRecognizer(HandGestureConfig())
        self.screen = {"w": 1920.0, "h": 1080.0}
        self._tracker_errors = 0

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

        msg: dict[str, Any] = {"type": "frame", "id": frame_id, "face": state.present,
                               "frozen": False, "labeled": False,
                               "pinching": state.pinching, "paused": state.paused}
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

    # ---------------------------------------------------------------- commands
    def handle_command(self, cmd: dict) -> list[dict]:
        kind = cmd.get("type")
        if kind == "hello":
            screen = cmd.get("screen") or {}
            if screen.get("w") and screen.get("h"):
                self.screen = {"w": float(screen["w"]), "h": float(screen["h"])}
            return [{"type": "hello", "mode": "hand", "model": False, "profile": None,
                     "hand": True, "gestures": _gesture_info()}]
        if kind == "ping":
            return [{"type": "pong", "t": cmd.get("t")}]
        if kind == "set_gestures":
            cfg = self.recognizer.config
            for key in ("pinch_on", "pinch_off", "click_max_ms", "palm_hold_ms", "scroll_start"):
                if key in cmd:
                    setattr(cfg, key, float(cmd[key]))
            return [{"type": "gestures_set", "ok": True}]
        # Hand mode ignores calibration / lab commands rather than erroring.
        return []


def _gesture_info() -> dict:
    return {
        "point": "Move the cursor with your index fingertip.",
        "click": "Pinch your thumb and index finger together.",
        "scroll": "Pinch and move your hand up or down.",
        "pause": "Hold an open palm to the camera to pause or resume.",
    }
