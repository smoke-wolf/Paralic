"""Per-connection eye-tracking pipeline.

Every browser tab that connects to the WebSocket gets its own
:class:`TrackerSession`. For each camera frame it:

1. decodes the JPEG sent by the browser,
2. runs MediaPipe FaceLandmarker (face mesh + iris + blendshapes + head pose),
3. extracts eye / head features,
4. updates the blink detector (single and double blinks),
5. predicts the on-screen gaze point with GazeNet,
6. smooths it and freezes it through blinks,
7. stores the frame as a calibration sample if the browser labelled it.

All methods are synchronous and are called from one worker thread per
connection (see ``server.py``), so no locking is needed.

Wire format of a frame message (binary WebSocket message)::

    uint32 little-endian N | N bytes of UTF-8 JSON header | JPEG bytes

The header carries the frame ``id`` and optionally a calibration ``label``
({"x", "y", "kind", "pt"}) with the dot position in *screen* CSS pixels.
"""

from __future__ import annotations

import json
import logging
import struct
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .blink import (BLINK_SENSITIVITY_PRESETS, DOUBLE_BLINK_GAP_PRESETS, BlinkDetector, BlinkEvent)
from .calibration import (KINDS, CalibrationData, CalibrationError, LabeledFrame, ProfileStore,
                          evaluate_validation, fit_adjustment, fit_full_calibration)
from .features import extract_features, overlay_points
from .filters import SMOOTHING_PRESETS, GazeStabilizer
from .gazenet import GazeNet
from .tracker import FaceTracker, decode_image

log = logging.getLogger(__name__)

MAX_HEADER_BYTES = 8192

# Reply message type of each command, so failures reach the same handler in the
# browser as successes (the page waits for these types).
_REPLY_TYPES = {
    "calibration_start": "calibration_started",
    "calibration_fit": "calibration_result",
    "validation_finish": "validation_result",
    "profile_load": "profile",
    "profile_delete": "profile",
    "hello": "hello",
    "settings": "settings",
    "ping": "pong",
}


class FrameFormatError(ValueError):
    pass


def parse_frame(payload: bytes) -> tuple[dict, bytes]:
    if len(payload) < 4:
        raise FrameFormatError("frame too short")
    (n,) = struct.unpack_from("<I", payload, 0)
    if n > MAX_HEADER_BYTES or len(payload) < 4 + n:
        raise FrameFormatError("bad frame header length")
    header = json.loads(payload[4:4 + n].decode("utf-8")) if n else {}
    if not isinstance(header, dict):
        raise FrameFormatError("frame header must be a JSON object")
    return header, payload[4 + n:]


def pack_frame(header: dict, image: bytes) -> bytes:
    """Inverse of :func:`parse_frame` (used by tests and tools)."""
    h = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(h)) + h + image


def _xy(v: Optional[np.ndarray]) -> Optional[list[float]]:
    return None if v is None else [round(float(v[0]), 1), round(float(v[1]), 1)]


@dataclass
class _FrameRecord:
    t: float
    frame_id: int
    closing: bool
    gaze: Optional[np.ndarray]


class TrackerSession:
    def __init__(self, tracker_factory: Callable[[], FaceTracker], profiles: ProfileStore,
                 clock: Callable[[], float] = time.monotonic):
        self._tracker_factory = tracker_factory
        self._tracker: Optional[FaceTracker] = None
        self.profiles = profiles
        self.clock = clock
        self.blink = BlinkDetector()
        self.stabilizer = GazeStabilizer(SMOOTHING_PRESETS["medium"])
        self.model: Optional[GazeNet] = None
        self.data = CalibrationData()
        self.screen: Optional[dict] = None
        self._history: deque[_FrameRecord] = deque(maxlen=150)
        self._frame_times: deque[float] = deque(maxlen=30)
        self._tracker_errors = 0
        self.settings = {"smoothing": "medium", "blink_sensitivity": "normal", "double_blink": "normal"}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def tracker(self) -> FaceTracker:
        if self._tracker is None:
            self._tracker = self._tracker_factory()
        return self._tracker

    def close(self) -> None:
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------
    def handle_frame(self, payload: bytes) -> list[dict]:
        started = time.perf_counter()
        try:
            header, image = parse_frame(payload)
        except (FrameFormatError, ValueError) as exc:
            return [{"type": "frame", "id": None, "face": False, "error": str(exc)}]
        frame_id = header.get("id")
        t = self.clock()
        try:
            rgb = decode_image(image)
        except Exception as exc:  # corrupted JPEG etc.
            return [{"type": "frame", "id": frame_id, "face": False, "error": f"decode: {exc}"}]

        try:
            obs = self.tracker.process(rgb, int(t * 1000))
        except Exception as exc:  # a MediaPipe failure should not end the session
            self._tracker_errors += 1
            if self._tracker_errors in (1, 10) or self._tracker_errors % 500 == 0:
                log.exception("Face tracking failed (%d times)", self._tracker_errors)
            return [{"type": "frame", "id": frame_id, "face": False, "error": f"tracker: {exc}"}]
        self._frame_times.append(t)
        msg: dict[str, Any] = {"type": "frame", "id": frame_id}

        if obs is None:
            events = self.blink.update_missing(t)
            gaze, frozen = self.stabilizer.update(t, None, False)
            msg.update(face=False, gaze=_xy(gaze), raw=None, frozen=frozen, labeled=False)
            closing = False
        else:
            feats = extract_features(obs.points_px, obs.image_size, obs.blendshapes, obs.matrix)
            events = self.blink.update(t, feats.closure)
            state = self.blink.state()
            closing = state.closing
            raw = self.model.predict(feats.vector)[0] if self.model is not None else None
            gaze, frozen = self.stabilizer.update(t, raw, closing)

            labeled = False
            label = header.get("label")
            if isinstance(label, dict) and not closing and label.get("kind") in KINDS:
                try:
                    target = (float(label["x"]), float(label["y"]))
                    point = int(label.get("pt", 0))
                except (KeyError, TypeError, ValueError):
                    pass
                else:
                    self.data.add(LabeledFrame(t=t, features=feats.vector.copy(), target=target,
                                               kind=label["kind"], point=point))
                    labeled = True

            msg.update(
                face=True,
                gaze=_xy(gaze),
                raw=_xy(raw),
                frozen=frozen,
                labeled=labeled,
                closure=round(feats.closure, 3),
                closed=state.closed,
                closing=closing,
                thr=[round(state.close_threshold, 3), round(state.open_threshold, 3)],
                head=[round(feats.yaw_deg, 1), round(feats.pitch_deg, 1), round(feats.roll_deg, 1)],
                dist=round(feats.distance_cm, 1),
            )
            if header.get("overlay"):
                msg["eyes"] = overlay_points(obs.points_px, obs.image_size, feats)

        self._history.append(_FrameRecord(t=t, frame_id=frame_id, closing=closing, gaze=gaze))
        msg["ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        msg["fps"] = self._fps()
        out = [msg]
        out.extend(self._event_message(ev, frame_id) for ev in events)
        return out

    def _fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return round((len(self._frame_times) - 1) / span, 1) if span > 0 else 0.0

    def _event_message(self, ev: BlinkEvent, frame_id: Any) -> dict:
        if ev.type == "double_blink":
            pre = self._frame_before(ev.first_start)
            return {
                "type": "double_blink",
                "frame": frame_id,
                "pre_frame": pre.frame_id if pre else None,
                "at": _xy(pre.gaze) if pre else None,
            }
        if ev.type == "blink":
            return {"type": "blink", "n": ev.count, "frame": frame_id}
        return {"type": ev.type, "frame": frame_id}

    def _frame_before(self, t: Optional[float]) -> Optional[_FrameRecord]:
        """Last frame with open eyes before time ``t`` (just before the first blink)."""
        if t is None:
            return None
        best = None
        for rec in self._history:
            if rec.t >= t:
                break
            if not rec.closing:
                best = rec
        return best

    # ------------------------------------------------------------------
    # Commands (JSON text messages)
    # ------------------------------------------------------------------
    def handle_command(self, cmd: dict) -> list[dict]:
        kind = cmd.get("type")
        handler = getattr(self, f"_cmd_{kind}", None) if isinstance(kind, str) else None
        if handler is None:
            return [{"type": "error", "error": f"unknown command {kind!r}"}]
        try:
            return handler(cmd)
        except CalibrationError as exc:
            error = str(exc)
        except Exception as exc:  # keep the session alive; report to the page
            log.exception("Command %r failed", kind)
            error = f"{type(exc).__name__}: {exc}"
        reply = {"type": _REPLY_TYPES.get(kind, f"{kind}_result"), "ok": False, "mode": cmd.get("mode"),
                 "error": error}
        if reply["type"] == "profile":
            reply["loaded"] = False
        return [reply]

    def _cmd_ping(self, cmd: dict) -> list[dict]:
        return [{"type": "pong", "t": cmd.get("t")}]

    def _cmd_hello(self, cmd: dict) -> list[dict]:
        screen = cmd.get("screen")
        if isinstance(screen, dict):
            self.screen = {k: screen.get(k) for k in ("w", "h", "dpr")}
        if isinstance(cmd.get("settings"), dict):
            self._apply_settings(cmd["settings"])
        return [{"type": "hello", "profile": self.profiles.summary(), "model": self.model is not None}]

    def _cmd_settings(self, cmd: dict) -> list[dict]:
        self._apply_settings(cmd)
        return [{"type": "settings", **self.settings}]

    def _apply_settings(self, s: dict) -> None:
        smoothing = s.get("smoothing")
        if smoothing in SMOOTHING_PRESETS:
            self.settings["smoothing"] = smoothing
            self.stabilizer.set_params(SMOOTHING_PRESETS[smoothing])
        sens = s.get("blink_sensitivity")
        if sens in BLINK_SENSITIVITY_PRESETS:
            self.settings["blink_sensitivity"] = sens
            self.blink.config.sensitivity = BLINK_SENSITIVITY_PRESETS[sens]
        speed = s.get("double_blink")
        if speed in DOUBLE_BLINK_GAP_PRESETS:
            self.settings["double_blink"] = speed
            self.blink.config.double_gap_ms = DOUBLE_BLINK_GAP_PRESETS[speed]

    def _cmd_calibration_start(self, cmd: dict) -> list[dict]:
        mode = cmd.get("mode", "full")
        if mode == "adjust":
            if self.model is None:
                return [{"type": "calibration_started", "ok": False, "error": "No calibration to adjust"}]
            self.data.clear(["adjust"])
        else:
            self.data.clear(["cal", "head", "val", "adjust"])
        return [{"type": "calibration_started", "ok": True, "mode": mode}]

    def _cmd_calibration_fit(self, cmd: dict) -> list[dict]:
        mode = cmd.get("mode", "full")
        if mode == "adjust":
            if self.model is None:
                raise CalibrationError("No calibration to adjust")
            info = fit_adjustment(self.model, self.data)
            self.stabilizer.reset()
            if self.profiles.exists():
                try:
                    self.profiles.save_correction(self.model)
                except (OSError, ValueError):
                    log.warning("Could not update saved profile", exc_info=True)
            return [{"type": "calibration_result", "ok": True, "mode": "adjust", **info}]
        model, info = fit_full_calibration(self.data)
        self.model = model
        self.stabilizer.reset()
        return [{"type": "calibration_result", "ok": True, "mode": "full", **info}]

    def _cmd_validation_finish(self, cmd: dict) -> list[dict]:
        if self.model is None:
            raise CalibrationError("Calibrate first")
        result = evaluate_validation(self.model, self.data)
        # Use the validation dots as extra training data for the final network.
        try:
            model, info = fit_full_calibration(self.data, include_validation=True)
            self.model = model
            result["refit"] = info
        except CalibrationError:
            log.info("Refit with validation data failed; keeping the first model")
        self.stabilizer.reset()
        saved = False
        if cmd.get("save", True):
            try:
                self.profiles.save(self.model, self.data, self.screen, round(result["mean_error_px"], 1))
                saved = True
            except OSError:
                log.warning("Could not save calibration profile", exc_info=True)
        return [{"type": "validation_result", "ok": True, "saved": saved, **result}]

    def _cmd_profile_load(self, cmd: dict) -> list[dict]:
        if not self.profiles.exists():
            return [{"type": "profile", "loaded": False, "error": "No saved calibration"}]
        try:
            model, data, meta = self.profiles.load()
        except (OSError, ValueError, KeyError) as exc:
            return [{"type": "profile", "loaded": False, "error": f"Could not load saved calibration: {exc}"}]
        self.model = model
        self.data = data
        self.stabilizer.reset()
        return [{"type": "profile", "loaded": True, **meta}]

    def _cmd_profile_delete(self, cmd: dict) -> list[dict]:
        try:
            self.profiles.delete()
        except OSError as exc:  # e.g. the file is locked on Windows
            return [{"type": "profile", "ok": False, "loaded": self.model is not None,
                     "error": f"Could not delete the saved calibration: {exc}"}]
        return [{"type": "profile", "ok": True, "loaded": self.model is not None, "deleted": True}]
