"""Per-connection eye-tracking pipeline.

Every browser tab that connects to the WebSocket gets its own
:class:`TrackerSession`. For each camera frame it:

1. decodes the JPEG sent by the browser,
2. runs MediaPipe FaceLandmarker (face mesh + iris + blendshapes + head pose),
3. extracts eye / head features,
4. updates the blink detector (single and double blinks, long closes) and the
   wink detector (one eye closed: short winks, and held winks that work like
   holding a mouse button down),
5. predicts the on-screen gaze point with GazeNet - with a one-eye network
   while the other eye is closed,
6. smooths it and freezes it through blinks,
7. stores the frame as a calibration sample if the browser labelled it.

A session belongs to one *person* (see ``users.py``) and applies their
personal settings: blink thresholds, cursor smoothing and button magnet
learned from their calibration and A/B experiments. While they use the site,
the frames just before each practice-target hit or click become labelled
fine-tuning samples; a background job periodically trains challenger networks
on them and keeps whichever model predicts this person's recent gaze best.

Frames and commands are handled on one worker thread per connection (see
``server.py``); background jobs publish their results under ``_lock`` and
notify the page through ``push``.

Wire format of a frame message (binary WebSocket message)::

    uint32 little-endian N | N bytes of UTF-8 JSON header | JPEG bytes

The header carries the frame ``id`` and optionally a calibration ``label``
({"x", "y", "kind", "pt"}) with the dot position in *screen* CSS pixels.
"""

from __future__ import annotations

import json
import logging
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .blink import BLINK_SENSITIVITY_PRESETS, DOUBLE_BLINK_GAP_PRESETS, BlinkDetector, BlinkEvent
from .calibration import (CalibrationData, CalibrationError, LabeledFrame, ProfileStore, evaluate_validation,
                          fit_adjustment, fit_eye_models, fit_full_calibration)
from .features import extract_features, overlay_points
from .filters import GazeStabilizer
from .gazenet import GazeNet, ModelConfig
from .gestures import BLINK_SIGNALS, WinkDetector, WinkEvent, analyze_winks, blink_signal, other_eye, wink_config
from .personalize import (EXPERIMENTS, SMOOTHING_LEVELS, TRIAL_TIMEOUT_MS, PersonalizationError, analyze_blinks,
                          analyze_experiment, experiment_arms, recommend_magnet, run_finetune, smoothing_params,
                          tune_smoothing)
from .tracker import FaceTracker, decode_image
from .users import UnknownUser, UserStore

log = logging.getLogger(__name__)

MAX_HEADER_BYTES = 8192
AUTO_FINETUNE_EVENTS = 15        # new labelled events before an automatic fine-tune
AUTO_FINETUNE_MIN_GAP_S = 90.0
SAVE_EVERY_EVENTS = 10
LABEL_WINDOW_S = 0.45            # frames before the first blink that describe the fixation
FIXATION_RADIUS_PX = 160.0       # ...and whose gaze estimate stayed near the final one
MAX_BLINK_RECORDING = 30 * 40    # frames
MAX_WINK_RECORDING = 30 * 60
BLINK_MUTE_AFTER_WINK_S = 0.3    # ignore blinks while a reopening eye settles
WINK_OFFSET_FRAMES = 6           # frames before a wink that align the one-eye network

# Per-person gesture choices (stored in personal["gestures"]). The detection
# settings are used here; the page reads the rest to decide what each gesture does.
GESTURE_DEFAULTS = {
    "left_hold": "drag",     # holding the left eye closed: drag | menu | off
    "right_hold": "drag",    # holding the right eye closed
    "left_quick": "off",     # a short wink of the left eye: off | click | menu
    "right_quick": "off",
    "long_close": "off",     # closing both eyes for about a second: off | menu | grab | click
    "long_close_ms": 1000,
    "hold_ms": 350,          # how long a wink must last to count as holding
    "long_press_ms": 1000,   # holding still this long opens the menu instead of dragging
    "dwell": False,          # click by resting the eyes on a button
    "dwell_ms": 1000,
    "tracking_eye": "auto",  # which network leads: auto | both | left | right
    "left_forced": False,    # use an eye for winks even though the wink test found it unreliable
    "right_forced": False,
}
_GESTURE_CHOICES = {
    "left_hold": ("drag", "menu", "off"), "right_hold": ("drag", "menu", "off"),
    "left_quick": ("off", "click", "menu"), "right_quick": ("off", "click", "menu"),
    "long_close": ("off", "menu", "grab", "click"), "tracking_eye": ("auto", "both", "left", "right"),
}
_GESTURE_RANGES = {"long_close_ms": (600, 3000), "hold_ms": (200, 1500), "long_press_ms": (500, 3000),
                   "dwell_ms": (400, 3000)}

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
    "users": "users",
    "user_select": "users",
    "user_create": "users",
    "user_rename": "users",
    "user_delete": "users",
    "label_event": "label_stored",
    "blink_calibration_start": "blink_calibration_started",
    "blink_calibration_finish": "blink_calibration_result",
    "finetune": "finetune_started",
    "experiment_plan": "experiment_plan",
    "experiment_log": "experiment_result",
    "experiment_reset": "experiment_plan",
    "personal_get": "personal",
    "personal_reset": "personal",
    "gestures_set": "personal",
    "wink_calibration_start": "wink_calibration_started",
    "wink_calibration_finish": "wink_calibration_result",
    "blink_calibration_cancel": "blink_calibration_cancelled",
    "wink_calibration_cancel": "wink_calibration_cancelled",
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


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class _FrameRecord:
    t: float
    frame_id: int
    closing: bool
    gaze: Optional[np.ndarray]
    features: Optional[np.ndarray] = None
    raw: Optional[np.ndarray] = None
    wall: float = 0.0
    winking: Optional[str] = None     # eye that was closed in a (possible) wink

    @property
    def steady(self) -> bool:
        """Both eyes open: the features describe normal gaze."""
        return not self.closing and self.winking is None


class TrackerSession:
    def __init__(self, tracker_factory: Callable[[], FaceTracker], users: UserStore,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 push: Optional[Callable[[dict], None]] = None):
        self._tracker_factory = tracker_factory
        self._tracker: Optional[FaceTracker] = None
        self.users = users
        self.clock = clock
        self.wall = wall
        self._push = push or (lambda msg: None)
        self._lock = threading.RLock()
        self.blink = BlinkDetector()
        self.wink = WinkDetector()
        self.blink_mode = "both"
        self._wink_eye: Optional[str] = None       # open eye whose network leads during a wink
        self._wink_offset = np.zeros(2)
        self._blink_mute_until = 0.0
        self._wink_recording: Optional[list[tuple]] = None
        self.stabilizer = GazeStabilizer(smoothing_params(SMOOTHING_LEVELS["medium"]))
        self.model: Optional[GazeNet] = None
        self.data = CalibrationData()
        self.screen: Optional[dict] = None
        self.profile_meta: dict = {}
        self._history: deque[_FrameRecord] = deque(maxlen=150)
        self._frame_times: deque[float] = deque(maxlen=30)
        self._tracker_errors = 0
        self._blink_recording: Optional[list[tuple[float, float]]] = None
        self._job: Optional[threading.Thread] = None
        self._last_job_at = 0.0
        self._unsaved_events = 0
        self._overrides: dict = {}
        self.settings = {"smoothing": "auto", "blink_sensitivity": "personal", "double_blink": "personal",
                         "learning": True}
        self.user = users.ensure_active()
        self.personal = users.load_personal(self.user["id"])
        self._apply_effective()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @property
    def tracker(self) -> FaceTracker:
        if self._tracker is None:
            self._tracker = self._tracker_factory()
        return self._tracker

    @property
    def profiles(self) -> ProfileStore:
        return self.users.profile_store(self.user["id"])

    def close(self) -> None:
        with self._lock:
            self._save_if_dirty()
            if self._tracker is not None:
                self._tracker.close()
                self._tracker = None

    # ------------------------------------------------------------------
    # Personal settings
    # ------------------------------------------------------------------
    def effective(self) -> dict:
        """The settings in force now: personal values, presets and experiment overrides."""
        p = self.personal
        sm = p.get("smoothing") or {}
        auto_level = float(np.clip(float(sm.get("tuned_level", SMOOTHING_LEVELS["medium"]))
                                   + float(sm.get("bias", 0.0)), 0.0, 10.0))
        choice = self.settings["smoothing"]
        level = auto_level if choice == "auto" else SMOOTHING_LEVELS.get(choice, SMOOTHING_LEVELS["medium"])
        level = float(self._overrides.get("smoothing_level", level))

        blink = p.get("blink") or {}
        personal_blink = self.settings["blink_sensitivity"] == "personal" and "sensitivity" in blink
        sensitivity = (float(blink["sensitivity"]) if personal_blink
                       else BLINK_SENSITIVITY_PRESETS.get(self.settings["blink_sensitivity"], 0.30))
        if self.settings["double_blink"] == "personal":
            gap = float(blink.get("double_gap_ms", DOUBLE_BLINK_GAP_PRESETS["normal"])) + float(blink.get("gap_bias_ms", 0))
        else:
            gap = DOUBLE_BLINK_GAP_PRESETS.get(self.settings["double_blink"], DOUBLE_BLINK_GAP_PRESETS["normal"])
        gap = float(np.clip(self._overrides.get("double_gap_ms", gap), 250.0, 1200.0))

        mg = p.get("magnet")
        magnet = None
        if mg:
            magnet = ({"radius_px": 0.0, "pull": 0.0} if mg.get("off") else
                      {"radius_px": round(float(mg["radius_px"]) * float(mg.get("scale", 1.0)), 1),
                       "pull": float(mg.get("pull", 0.3))})
        # Thresholds learned before per-eye signals existed were measured on the average.
        signal = (blink["signal"] if blink.get("signal") in BLINK_SIGNALS
                  else "mean" if "sensitivity" in blink else "both")
        return {
            "smoothing_level": round(level, 2),
            "auto_smoothing_level": round(auto_level, 2),
            "sensitivity": sensitivity,
            "min_threshold": float(blink.get("min_threshold", 0.30)) if personal_blink else 0.30,
            "max_closed_ms": float(blink.get("max_closed_ms", 700.0)) if personal_blink else 700.0,
            "double_gap_ms": gap,
            "personal_blink": personal_blink,
            "blink_signal": signal,
            "deep_rise": round(0.6 * float(blink["peak_rise"]), 3) if blink.get("peak_rise") else None,
            "magnet": magnet,
            "gestures": self.gestures(),
        }

    def gestures(self) -> dict:
        """This person's gesture settings (defaults for anything not chosen)."""
        g = self.personal.get("gestures") or {}
        return {**GESTURE_DEFAULTS, **{k: v for k, v in g.items() if k in GESTURE_DEFAULTS}}

    def _apply_effective(self) -> None:
        e = self.effective()
        self.stabilizer.set_params(smoothing_params(e["smoothing_level"]))
        cfg = self.blink.config
        cfg.sensitivity = e["sensitivity"]
        cfg.min_threshold = e["min_threshold"]
        cfg.max_closed_ms = e["max_closed_ms"]
        cfg.double_gap_ms = e["double_gap_ms"]
        cfg.deep_rise = e["deep_rise"]
        # A long close must be clearly longer than this person's longest blink.
        cfg.long_close_ms = max(float(e["gestures"]["long_close_ms"]), cfg.max_closed_ms + 200.0)
        self.blink_mode = e["blink_signal"]
        self.wink.config = wink_config(self.personal, e["gestures"])

    def _save_personal(self) -> None:
        try:
            self.users.save_personal(self.user["id"], self.personal)
        except OSError:
            log.warning("Could not save personal settings", exc_info=True)

    def personal_view(self) -> dict:
        """What the page shows in the Personalization Lab (and uses for the magnet)."""
        e = self.effective()
        p = self.personal
        events = self.data.ft_events()
        trained_ts = float(self.model.meta.get("trained_ts", 0.0)) if self.model else 0.0
        meta = self.model.meta if self.model else {}
        return {
            "user": {"id": self.user["id"], "name": self.user["name"]},
            "smoothing_level": e["smoothing_level"],
            "auto_smoothing_level": e["auto_smoothing_level"],
            "smoothing_profile": p.get("smoothing"),
            "magnet": e["magnet"],
            "magnet_profile": p.get("magnet"),
            "blink": {"sensitivity": round(e["sensitivity"], 3), "double_gap_ms": round(e["double_gap_ms"]),
                      "personal": e["personal_blink"]},
            "blink_profile": p.get("blink"),
            "model": None if not self.model else {
                "version": meta.get("version"),
                "config": ModelConfig.from_dict(meta.get("config")).name,
                "source": meta.get("source", "calibration"),
                "trained_at": meta.get("trained_at"),
                "cv_error_px": meta.get("cv_error_px"),
            },
            "accuracy_px": self.profile_meta.get("accuracy_px"),
            "ft_events": len(events),
            "ft_new_events": sum(1 for frames in events.values() if frames[0].t > trained_ts),
            "finetune_jobs": (p.get("finetune") or {}).get("jobs", [])[-8:],
            "model_history": p.get("model_history", [])[-12:],
            "experiments": p.get("experiments", {}),
            "learning": bool(self.settings.get("learning", True)),
            "job_running": self._job_running(),
            "gestures": e["gestures"],
            "blink_signal": e["blink_signal"],
            "wink_profile": p.get("wink"),
            "winks": {eye: self.wink.config.eye(eye).enabled for eye in ("left", "right")},
            "eye_models": None if not self.model else {
                "available": sorted(self.model.eyes),
                "preferred": self.model.eye,
                "cv_px": meta.get("eye_cv_px"),
            },
        }

    def _record_model(self, source: str, **extra) -> None:
        """Give the current model a new version number and log it."""
        version = int(self.personal.get("model_version", 0)) + 1
        self.personal["model_version"] = version
        self.model.meta["version"] = version
        self.model.meta.setdefault("source", source)
        entry = {"version": version, "source": source, "time": _now(),
                 "config": ModelConfig.from_dict(self.model.meta.get("config")).name, **extra}
        self.personal["model_history"] = (self.personal.get("model_history", []) + [entry])[-50:]

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------
    def handle_frame(self, payload: bytes) -> list[dict]:
        with self._lock:
            return self._handle_frame(payload)

    def _handle_frame(self, payload: bytes) -> list[dict]:
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
        features = raw = None
        winking = None

        if obs is None:
            events: list = self.blink.update_missing(t)
            events += self.wink.update_missing(t)
            if self.wink.winking is None:
                self._wink_eye = None
            gaze, frozen = self.stabilizer.update(t, None, False)
            msg.update(face=False, gaze=_xy(gaze), raw=None, frozen=frozen, labeled=False)
            closing = False
        else:
            feats = extract_features(obs.points_px, obs.image_size, obs.blendshapes, obs.matrix)
            features = feats.vector
            cl, cr = feats.closure_left, feats.closure_right
            was_winking = self.wink.winking
            wink_events = self.wink.update(t, cl, cr)
            winking = self.wink.winking
            if winking and winking != was_winking:
                self._start_wink_tracking(t, winking)
            elif winking is None:
                self._wink_eye = None
            # The blink detector always watches this person's blink signal (it is
            # never switched in the middle of a closure). Blinks during a held
            # wink (the open eye blinking during a long drag) or just after a
            # wink are dropped *and forgotten*, so they cannot pair up into a
            # double blink. A wink that is only starting does not mute blinks: a
            # lid that leads by a frame is part of an ordinary blink.
            events = self.blink.update(t, blink_signal(cl, cr, self.blink_mode))
            winked = False
            for ev in wink_events:
                if ev.type in ("wink_start", "wink"):
                    winked = True        # that closure was a wink, not a blink
                if ev.type in ("wink", "wink_end"):
                    self._blink_mute_until = t + BLINK_MUTE_AFTER_WINK_S
            if winked or self.wink.pressed is not None or t < self._blink_mute_until:
                dropped = any(e.type != "blink_expired" for e in events)
                events = [e for e in events if e.type == "blink_expired"]
                if (dropped or winked) and self.blink.cancel() and not events:
                    events.append(BlinkEvent("blink_expired", t))
            events += wink_events
            state = self.blink.state()
            closing = state.closing
            if winking is not None:
                # During a wink the cursor follows the open eye; it holds still
                # only if that eye closes as well.
                closing = min(cl, cr) >= state.open_threshold
            raw = self._predict(feats.vector)
            gaze, frozen = self.stabilizer.update(t, raw, closing, hold=state.deep and winking is None)
            if self._blink_recording is not None and len(self._blink_recording) < MAX_BLINK_RECORDING:
                self._blink_recording.append((t, cl, cr))
            phase = header.get("gesture")
            if (self._wink_recording is not None and phase in ("rest", "left", "right")
                    and len(self._wink_recording) < MAX_WINK_RECORDING):
                self._wink_recording.append((t, cl, cr, phase))

            labeled = False
            label = header.get("label")
            if (isinstance(label, dict) and not closing and winking is None
                    and label.get("kind") in ("cal", "head", "val", "adjust")):
                try:
                    target = (float(label["x"]), float(label["y"]))
                    point = int(label.get("pt", 0))
                except (KeyError, TypeError, ValueError):
                    pass
                else:
                    self.data.add(LabeledFrame(t=self.wall(), features=feats.vector.copy(), target=target,
                                               kind=label["kind"], point=point))
                    labeled = True

            msg.update(
                face=True,
                gaze=_xy(gaze),
                raw=_xy(raw),
                frozen=frozen,
                labeled=labeled,
                closure=round(feats.closure, 3),
                cl=round(cl, 3),
                cr=round(cr, 3),
                closed=state.closed,
                closing=closing,
                wink=self.wink.pressed,
                winking=winking,
                net=(self._wink_eye or self._preferred_eye()) if raw is not None else None,
                thr=[round(state.close_threshold, 3), round(state.open_threshold, 3)],
                head=[round(feats.yaw_deg, 1), round(feats.pitch_deg, 1), round(feats.roll_deg, 1)],
                dist=round(feats.distance_cm, 1),
            )
            if header.get("overlay"):
                msg["eyes"] = overlay_points(obs.points_px, obs.image_size, feats)

        self._history.append(_FrameRecord(t=t, frame_id=frame_id, closing=closing, gaze=gaze,
                                          features=None if features is None else features.copy(),
                                          raw=raw, wall=self.wall(), winking=winking))
        msg["ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        msg["fps"] = self._fps()
        out = [msg]
        out.extend(self._event_message(ev, frame_id, gaze) for ev in events)
        return out

    # -- gaze prediction ---------------------------------------------------------
    def _preferred_eye(self) -> str:
        """The network that leads while both eyes are open."""
        choice = self.gestures()["tracking_eye"]
        if self.model is None:
            return "both"
        if choice in ("left", "right") and choice in self.model.eyes:
            return choice
        return "both" if choice == "both" else self.model.eye

    def _predict(self, vector: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        if self._wink_eye is not None:
            if self._wink_eye not in self.model.eyes:
                return None  # no one-eye network yet: hold the cursor still
            return self.model.predict(vector, self._wink_eye)[0] + self._wink_offset
        return self.model.predict(vector, self._preferred_eye())[0]

    def _start_wink_tracking(self, t: float, winking: str) -> None:
        """One eye started closing: let the open eye's network take over seamlessly.

        The one-eye network is aligned with the usual one on the frames just
        before the wink, so the cursor does not jump, and smoothing restarts
        from where the cursor was before the closing eye disturbed it.
        """
        self._wink_eye = other_eye(winking)
        self._wink_offset = np.zeros(2)
        if self.model is not None and self._wink_eye in self.model.eyes:
            before = [r.features for r in self._history
                      if r.t < self.wink.since and r.steady and r.features is not None][-WINK_OFFSET_FRAMES:]
            if before:
                F = np.array(before)
                usual = self.model.predict(F, self._preferred_eye())
                one_eye = self.model.predict(F, self._wink_eye)
                self._wink_offset = (usual - one_eye).mean(axis=0)
        self.stabilizer.rewind(t, self.wink.since - self.stabilizer.rewind_s)

    def _fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return round((len(self._frame_times) - 1) / span, 1) if span > 0 else 0.0

    def _event_message(self, ev, frame_id: Any, gaze: Optional[np.ndarray]) -> dict:
        if isinstance(ev, WinkEvent):
            msg = {"type": ev.type, "eye": ev.eye, "frame": frame_id, "duration_ms": round(ev.duration_ms)}
            if ev.type in ("wink_start", "wink"):
                pre = self._frame_before(ev.start - 0.05)
                msg.update(pre_frame=pre.frame_id if pre else None, at=_xy(pre.gaze) if pre else None)
            else:
                msg.update(at=_xy(gaze), cancelled=ev.cancelled)
            return msg
        if ev.type in ("double_blink", "long_close"):
            pre = self._frame_before(ev.first_start)
            msg = {
                "type": ev.type,
                "frame": frame_id,
                "pre_frame": pre.frame_id if pre else None,
                "at": _xy(pre.gaze) if pre else None,
            }
            if ev.type == "long_close":
                msg["duration_ms"] = round(ev.duration_ms)
            return msg
        if ev.type == "blink":
            return {"type": "blink", "n": ev.count, "frame": frame_id}
        return {"type": ev.type, "frame": frame_id}

    def _frame_before(self, t: Optional[float]) -> Optional[_FrameRecord]:
        """Last frame with both eyes open before time ``t`` (just before a blink or wink)."""
        if t is None:
            return None
        best = None
        for rec in self._history:
            if rec.t >= t:
                break
            if rec.steady:
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
            with self._lock:
                return handler(cmd)
        except (CalibrationError, PersonalizationError) as exc:
            error = str(exc)
        except UnknownUser:
            error = "Unknown person"
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
        return [{"type": "hello", "profile": self.profiles.summary(), "model": self.model is not None,
                 "user": self.user, "users": self.users.list(), "personal": self.personal_view()}]

    def _cmd_settings(self, cmd: dict) -> list[dict]:
        self._apply_settings(cmd)
        return [{"type": "settings", **self.settings, "effective": self.effective()}]

    def _apply_settings(self, s: dict) -> None:
        if s.get("smoothing") in (*SMOOTHING_LEVELS, "auto"):
            self.settings["smoothing"] = s["smoothing"]
        if s.get("blink_sensitivity") in (*BLINK_SENSITIVITY_PRESETS, "personal"):
            self.settings["blink_sensitivity"] = s["blink_sensitivity"]
        if s.get("double_blink") in (*DOUBLE_BLINK_GAP_PRESETS, "personal"):
            self.settings["double_blink"] = s["double_blink"]
        if isinstance(s.get("learning"), bool):
            self.settings["learning"] = s["learning"]
        # Temporary overrides used while an A/B experiment round runs.
        if s.get("clear_overrides"):
            self._overrides = {}
        for key, lo, hi in (("smoothing_level", 0.0, 10.0), ("double_gap_ms", 250.0, 1200.0)):
            if isinstance(s.get(key), (int, float)):
                self._overrides[key] = float(np.clip(s[key], lo, hi))
        self._apply_effective()

    # -- calibration ---------------------------------------------------------------
    def _cmd_calibration_start(self, cmd: dict) -> list[dict]:
        mode = cmd.get("mode", "full")
        if mode == "adjust":
            if self.model is None:
                return [{"type": "calibration_started", "ok": False, "error": "No calibration to adjust"}]
            self.data.clear(["adjust"])
        else:
            # Keep the fine-tuning samples: they still describe this person's eyes.
            self.data.clear(["cal", "head", "val", "adjust"])
        return [{"type": "calibration_started", "ok": True, "mode": mode}]

    def _cmd_calibration_fit(self, cmd: dict) -> list[dict]:
        mode = cmd.get("mode", "full")
        if mode == "adjust":
            if self.model is None:
                raise CalibrationError("No calibration to adjust")
            info = fit_adjustment(self.model, self.data)
            self.model.meta["trained_ts"] = self.wall()
            self._record_model("adjust", error_px=round(info["error_after_px"], 1))
            self.stabilizer.reset()
            self._save_profile()
            self._save_personal()
            return [{"type": "calibration_result", "ok": True, "mode": "adjust", **info,
                     "personal": self.personal_view()}]
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
        # Personalise smoothing and the button magnet from the measured precision / accuracy.
        old_sm = self.personal.get("smoothing") or {}
        old_mg = self.personal.get("magnet") or {}
        self.personal["smoothing"] = {**tune_smoothing(result["precision_px"]), "bias": old_sm.get("bias", 0.0)}
        self.personal["magnet"] = {**recommend_magnet(result["mean_error_px"]), "scale": old_mg.get("scale", 1.0),
                                   "off": old_mg.get("off", False)}
        self.profile_meta["accuracy_px"] = round(result["mean_error_px"], 1)
        self._record_model("calibration", accuracy_px=round(result["mean_error_px"], 1),
                           cv_error_px=self.model.meta.get("cv_error_px"))
        self._apply_effective()
        saved = False
        if cmd.get("save", True):
            saved = self._save_profile()
        self._save_personal()
        return [{"type": "validation_result", "ok": True, "saved": saved, **result,
                 "personal": self.personal_view()}]

    # -- profile / people -----------------------------------------------------------------
    def _save_profile(self) -> bool:
        if self.model is None:
            return False
        try:
            with self.users.lock_for(self.user["id"]):
                self.profiles.save(self.model, self.data, self.screen, self.profile_meta.get("accuracy_px"))
            self._unsaved_events = 0
            return True
        except OSError:
            log.warning("Could not save calibration profile", exc_info=True)
            return False

    def _save_if_dirty(self) -> None:
        if self._unsaved_events and self.model is not None:
            self._save_profile()

    def _cmd_profile_load(self, cmd: dict) -> list[dict]:
        if not self.profiles.exists():
            return [{"type": "profile", "loaded": False, "error": "No saved calibration",
                     "personal": self.personal_view()}]
        try:
            model, data, meta = self.profiles.load()
        except (OSError, ValueError, KeyError) as exc:
            return [{"type": "profile", "loaded": False, "error": f"Could not load saved calibration: {exc}"}]
        self.model = model
        self.data = data
        self.profile_meta = meta
        self.stabilizer.reset()
        if not model.eyes:
            # Saved before one-eye networks existed: add them now (a second or
            # two, once). Done here rather than in the background so that nothing
            # - a quick adjust, say - can change the model while they train.
            try:
                fit_eye_models(model, data)
                self._save_profile()
            except CalibrationError:
                log.info("Not enough saved data for one-eye networks")
        return [{"type": "profile", "loaded": True, **meta, "personal": self.personal_view()}]

    def _cmd_profile_delete(self, cmd: dict) -> list[dict]:
        try:
            self.profiles.delete()
        except OSError as exc:  # e.g. the file is locked on Windows
            return [{"type": "profile", "ok": False, "loaded": self.model is not None,
                     "error": f"Could not delete the saved calibration: {exc}"}]
        return [{"type": "profile", "ok": True, "loaded": self.model is not None, "deleted": True}]

    def _users_reply(self) -> list[dict]:
        return [{"type": "users", "ok": True, "users": self.users.list(), "user": self.user,
                 "profile": self.profiles.summary(), "personal": self.personal_view()}]

    def _switch_user(self, user: dict) -> None:
        self._save_if_dirty()
        self.user = user
        self.personal = self.users.load_personal(user["id"])
        self.model = None
        self.data = CalibrationData()
        self.profile_meta = {}
        self._overrides = {}
        self._blink_recording = None
        self._wink_recording = None
        self._wink_eye = None
        self.stabilizer.reset()
        self.blink.reset()
        self.wink.reset()
        self._apply_effective()

    def _cmd_users(self, cmd: dict) -> list[dict]:
        return self._users_reply()

    def _cmd_user_select(self, cmd: dict) -> list[dict]:
        self._switch_user(self.users.select(str(cmd.get("id"))))
        return self._users_reply()

    def _cmd_user_create(self, cmd: dict) -> list[dict]:
        self._switch_user(self.users.create(cmd.get("name")))
        return self._users_reply()

    def _cmd_user_rename(self, cmd: dict) -> list[dict]:
        user_id = str(cmd.get("id") or self.user["id"])
        user = self.users.rename(user_id, str(cmd.get("name", "")))
        if user_id == self.user["id"]:
            self.user = user
        return self._users_reply()

    def _cmd_user_delete(self, cmd: dict) -> list[dict]:
        user_id = str(cmd.get("id"))
        self.users.delete(user_id)
        if user_id == self.user["id"]:
            self._unsaved_events = 0  # nothing left to save for them
            self._switch_user(self.users.ensure_active())
        return self._users_reply()

    # -- learning from use ---------------------------------------------------------------------
    def _find_record(self, frame_id) -> Optional[int]:
        for i in range(len(self._history) - 1, -1, -1):
            if self._history[i].frame_id == frame_id:
                return i
        return None

    def _cmd_label_event(self, cmd: dict) -> list[dict]:
        """A practice target was popped or a button clicked: learn where the eyes were."""
        def skip(reason: str) -> list[dict]:
            return [{"type": "label_stored", "ok": True, "stored": False, "reason": reason}]

        if not self.settings.get("learning", True):
            return skip("learning is off")
        if self.model is None:
            return skip("not calibrated")
        idx = self._find_record(cmd.get("pre_frame"))
        if idx is None:
            return skip("frame not found")
        anchor = self._history[idx]

        def same_fixation(r: _FrameRecord) -> bool:
            # Skip frames from while the eyes were still travelling to the target
            # (and frames without a gaze estimate, which we cannot check).
            if anchor.raw is None:
                return True
            if r.raw is None:
                return False
            return float(np.hypot(*(r.raw - anchor.raw))) <= FIXATION_RADIUS_PX

        window = [r for r in list(self._history)[:idx + 1]
                  if r.features is not None and r.steady and anchor.t - r.t <= LABEL_WINDOW_S
                  and same_fixation(r)]
        if len(window) < 4:
            return skip("eyes were not steady")
        kind = cmd.get("kind")
        if kind == "practice":
            try:
                target = (float(cmd["target"][0]), float(cmd["target"][1]))
            except (KeyError, TypeError, ValueError, IndexError):
                raise PersonalizationError("practice events need a target")
            weight = 1.0
        elif kind == "click":
            try:
                x0, y0, x1, y1 = (float(v) for v in cmd["rect"])
            except (KeyError, TypeError, ValueError):
                raise PersonalizationError("click events need a rect")
            sw = float((self.screen or {}).get("w") or 1920)
            sh = float((self.screen or {}).get("h") or 1080)
            if x1 - x0 > 0.45 * sw or y1 - y0 > 0.45 * sh:
                return skip("target too large to be informative")
            pred = np.mean([r.raw for r in window if r.raw is not None], axis=0)
            ix, iy = 0.25 * (x1 - x0) / 2, 0.25 * (y1 - y0) / 2
            target = (float(np.clip(pred[0], x0 + ix, x1 - ix)), float(np.clip(pred[1], y0 + iy, y1 - iy)))
            weight = 0.5
        else:
            raise PersonalizationError("unknown event kind")
        event = self.data.next_event_id()
        for r in window:
            self.data.add(LabeledFrame(t=r.wall, features=r.features, target=target, kind="ft",
                                       point=event, weight=weight))
        self.data.prune_ft()
        self._unsaved_events += 1
        if self._unsaved_events >= SAVE_EVERY_EVENTS:
            self._save_profile()
        started = self._maybe_autostart_finetune()
        view = self.personal_view()
        return [{"type": "label_stored", "ok": True, "stored": True, "event": event,
                 "ft_new_events": view["ft_new_events"], "finetune_started": started}]

    def _job_running(self) -> bool:
        return self._job is not None and self._job.is_alive()

    def _maybe_autostart_finetune(self) -> bool:
        if self._job_running() or self.model is None or not self.settings.get("learning", True):
            return False
        if self.wall() - self._last_job_at < AUTO_FINETUNE_MIN_GAP_S:
            return False
        if self.personal_view()["ft_new_events"] < AUTO_FINETUNE_EVENTS:
            return False
        return self._start_finetune(auto=True)

    def _start_finetune(self, auto: bool) -> bool:
        if self._job_running() or self.model is None:
            return False
        champion = self.model
        snapshot = self.data.copy()
        user_id = self.user["id"]
        self._last_job_at = self.wall()

        def run() -> None:
            try:
                new_model, report = run_finetune(champion, snapshot)
            except (PersonalizationError, CalibrationError) as exc:
                self._push({"type": "finetune_result", "ok": False, "auto": auto, "error": str(exc)})
                return
            except Exception as exc:
                log.exception("Fine-tuning failed")
                self._push({"type": "finetune_result", "ok": False, "auto": auto, "error": f"{exc}"})
                return
            with self._lock:
                if self.user["id"] != user_id or self.model is not champion:
                    self._push({"type": "finetune_result", "ok": False, "auto": auto,
                                "error": "Skipped: the model changed while training"})
                    return
                if new_model is not None:
                    self.model = new_model
                    self._record_model("fine-tune", error_px=report["candidate_errors_px"][report["winner"]],
                                       previous_error_px=report["champion_error_px"])
                    report["version"] = new_model.meta["version"]
                    self._save_profile()
                ft = self.personal.setdefault("finetune", {})
                ft["jobs"] = (ft.get("jobs", []) + [{**report, "auto": auto}])[-30:]
                self._save_personal()
                view = self.personal_view()
            self._push({"type": "finetune_result", "ok": True, "auto": auto, **report, "personal": view})

        self._job = threading.Thread(target=run, name="paralic-finetune", daemon=True)
        self._job.start()
        return True

    def _cmd_finetune(self, cmd: dict) -> list[dict]:
        if self.model is None:
            raise PersonalizationError("Calibrate first")
        if self._job_running():
            return [{"type": "finetune_started", "ok": False, "error": "Already fine-tuning"}]
        self._start_finetune(auto=bool(cmd.get("auto", False)))
        return [{"type": "finetune_started", "ok": True}]

    def wait_for_job(self, timeout: float = 60.0) -> None:
        """Block until a running background job finishes (used by tests and tools)."""
        job = self._job
        if job is not None:
            job.join(timeout)

    # -- blink personalisation ------------------------------------------------------------------
    def _cmd_blink_calibration_start(self, cmd: dict) -> list[dict]:
        self._blink_recording = []
        return [{"type": "blink_calibration_started", "ok": True}]

    def _cmd_blink_calibration_cancel(self, cmd: dict) -> list[dict]:
        self._blink_recording = None
        return [{"type": "blink_calibration_cancelled", "ok": True}]

    def _cmd_blink_calibration_finish(self, cmd: dict) -> list[dict]:
        samples, self._blink_recording = self._blink_recording or [], None
        result = analyze_blinks(samples)
        result["gap_bias_ms"] = (self.personal.get("blink") or {}).get("gap_bias_ms", 0)
        self.personal["blink"] = result
        self._save_personal()
        self._apply_effective()
        return [{"type": "blink_calibration_result", "ok": True, **result, "personal": self.personal_view()}]

    # -- gestures -------------------------------------------------------------------------------------
    def _cmd_gestures_set(self, cmd: dict) -> list[dict]:
        patch = cmd.get("gestures")
        if not isinstance(patch, dict):
            raise PersonalizationError("No gesture settings")
        g = dict(self.personal.get("gestures") or {})
        for key, value in patch.items():
            if key in _GESTURE_CHOICES and value in _GESTURE_CHOICES[key]:
                g[key] = value
            elif key in _GESTURE_RANGES and isinstance(value, (int, float)) and not isinstance(value, bool):
                lo, hi = _GESTURE_RANGES[key]
                g[key] = int(np.clip(value, lo, hi))
            elif key == "dwell" and isinstance(value, bool):
                g[key] = value
            elif key in ("left_forced", "right_forced") and isinstance(value, bool):
                g[key] = value
        self.personal["gestures"] = g
        self._save_personal()
        self._apply_effective()
        return [{"type": "personal", "ok": True, "personal": self.personal_view()}]

    def _cmd_wink_calibration_start(self, cmd: dict) -> list[dict]:
        self._wink_recording = []
        return [{"type": "wink_calibration_started", "ok": True}]

    def _cmd_wink_calibration_cancel(self, cmd: dict) -> list[dict]:
        self._wink_recording = None
        return [{"type": "wink_calibration_cancelled", "ok": True}]

    def _cmd_wink_calibration_finish(self, cmd: dict) -> list[dict]:
        samples, self._wink_recording = self._wink_recording or [], None
        result = analyze_winks(samples)
        self.personal["wink"] = result
        # A fresh test decides again which eyes may wink.
        g = self.personal.setdefault("gestures", {})
        for eye in ("left", "right"):
            g.pop(f"{eye}_forced", None)
        self._save_personal()
        self._apply_effective()
        return [{"type": "wink_calibration_result", "ok": True, **result, "personal": self.personal_view()}]

    # -- A/B experiments -----------------------------------------------------------------------------
    def _experiment(self, cmd: dict) -> str:
        name = cmd.get("experiment")
        if name not in EXPERIMENTS:
            raise PersonalizationError("Unknown experiment")
        return name

    def _current_trials(self, name: str) -> tuple[dict, list[dict]]:
        exps = self.users.load_experiments(self.user["id"])
        entry = exps.setdefault(name, {"epoch": 0, "trials": []})
        return exps, [t for t in entry["trials"] if t.get("epoch") == entry["epoch"]]

    def _cmd_experiment_plan(self, cmd: dict) -> list[dict]:
        name = self._experiment(cmd)
        arms = experiment_arms(name, self.effective())
        _, trials = self._current_trials(name)
        analysis = analyze_experiment(trials, [a["id"] for a in arms])
        return [{"type": "experiment_plan", "ok": True, "experiment": name, "title": EXPERIMENTS[name]["title"],
                 "arms": arms, "analysis": analysis}]

    def _cmd_experiment_reset(self, cmd: dict) -> list[dict]:
        name = self._experiment(cmd)
        exps = self.users.load_experiments(self.user["id"])
        entry = exps.setdefault(name, {"epoch": 0, "trials": []})
        entry["epoch"] += 1
        self.users.save_experiments(self.user["id"], exps)
        return self._cmd_experiment_plan(cmd)

    def _cmd_experiment_log(self, cmd: dict) -> list[dict]:
        name = self._experiment(cmd)
        arms = experiment_arms(name, self.effective() if not self._overrides else self._baseline_effective())
        arm_ids = [a["id"] for a in arms]
        raw_trials = cmd.get("trials")
        if not isinstance(raw_trials, list) or not raw_trials:
            raise PersonalizationError("No trials")
        clean = []
        for t in raw_trials[:200]:
            if not isinstance(t, dict) or t.get("arm") not in arm_ids:
                continue
            clean.append({
                "arm": t["arm"],
                "time_ms": float(np.clip(float(t.get("time_ms", TRIAL_TIMEOUT_MS)), 0.0, TRIAL_TIMEOUT_MS)),
                "misses": int(np.clip(int(t.get("misses", 0)), 0, 50)),
                "timeout": bool(t.get("timeout", False)),
            })
        exps = self.users.load_experiments(self.user["id"])
        entry = exps.setdefault(name, {"epoch": 0, "trials": []})
        for t in clean:
            entry["trials"].append({**t, "epoch": entry["epoch"], "time": _now()})
        entry["trials"] = entry["trials"][-2000:]
        current = [t for t in entry["trials"] if t.get("epoch") == entry["epoch"]]
        analysis = analyze_experiment(current, arm_ids)
        applied = None
        if analysis["decision"] == "adopt":
            applied = self._adopt(name, analysis["best"], {a["id"]: a for a in arms})
            entry["epoch"] += 1  # a new baseline: start counting again
        self.users.save_experiments(self.user["id"], exps)
        self.personal.setdefault("experiments", {})[name] = {
            "decision": analysis["decision"], "best": analysis["best"], "p_value": analysis["p_value"],
            "effect": analysis["effect"], "arms": analysis["arms"], "applied": applied, "updated": _now(),
        }
        self._save_personal()
        self._overrides = {}
        self._apply_effective()
        return [{"type": "experiment_result", "ok": True, "experiment": name, **analysis, "applied": applied,
                 "logged": len(clean), "personal": self.personal_view()}]

    def _baseline_effective(self) -> dict:
        saved, self._overrides = self._overrides, {}
        try:
            return self.effective()
        finally:
            self._overrides = saved

    def _adopt(self, name: str, arm_id: str, arms: dict) -> dict:
        """Make the winning arm this person's new normal."""
        arm = arms[arm_id]
        if name == "smoothing":
            sm = self.personal.setdefault("smoothing", {"tuned_level": SMOOTHING_LEVELS["medium"]})
            sm["bias"] = float(arm["smoothing_level"]) - float(sm.get("tuned_level", SMOOTHING_LEVELS["medium"]))
            self.settings["smoothing"] = "auto"
            return {"arm": arm_id, "smoothing_level": arm["smoothing_level"], "settings": {"smoothing": "auto"}}
        if name == "magnet":
            mg = self.personal.setdefault("magnet", {"radius_px": 110.0, "pull": 0.3, "scale": 1.0})
            if arm_id == "off":
                mg["off"] = True
            else:
                mg["off"] = False
                mg["scale"] = float(arm["magnet"]["radius_px"]) / max(float(mg["radius_px"]), 1.0)
                mg["pull"] = float(arm["magnet"]["pull"])
            return {"arm": arm_id, "magnet": arm["magnet"], "settings": {"snap": "auto"}}
        if name == "double_blink":
            bl = self.personal.setdefault("blink", {})
            base = float(bl.get("double_gap_ms", DOUBLE_BLINK_GAP_PRESETS["normal"]))
            bl["gap_bias_ms"] = float(arm["double_gap_ms"]) - base
            self.settings["double_blink"] = "personal"
            return {"arm": arm_id, "double_gap_ms": arm["double_gap_ms"], "settings": {"doubleBlink": "personal"}}
        if name == "dwell":
            self.personal.setdefault("gestures", {})["dwell_ms"] = int(arm["dwell_ms"])
            return {"arm": arm_id, "dwell_ms": arm["dwell_ms"], "settings": {}}
        raise PersonalizationError("Unknown experiment")

    # -- personal overview ---------------------------------------------------------------------------
    def _cmd_personal_get(self, cmd: dict) -> list[dict]:
        return [{"type": "personal", "ok": True, "personal": self.personal_view()}]

    def _cmd_personal_reset(self, cmd: dict) -> list[dict]:
        part = cmd.get("part")
        if part == "experiments":
            self.personal.pop("experiments", None)
            self.users.save_experiments(self.user["id"], {})
            for key in ("smoothing", "magnet"):
                if key in self.personal:
                    self.personal[key].update({"bias": 0.0} if key == "smoothing" else {"scale": 1.0, "off": False})
            if "blink" in self.personal:
                self.personal["blink"]["gap_bias_ms"] = 0
        elif part == "blink":
            self.personal.pop("blink", None)
        elif part == "gestures":
            self.personal.pop("gestures", None)
            self.personal.pop("wink", None)
        elif part == "learning":
            self.data.clear(["ft"])
            self._unsaved_events += 1
            self._save_if_dirty()
            self.personal.pop("finetune", None)
        else:
            raise PersonalizationError("Unknown part")
        self._save_personal()
        self._apply_effective()
        return [{"type": "personal", "ok": True, "personal": self.personal_view()}]
