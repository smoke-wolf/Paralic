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

A session opened in *hand mode* runs the hand pipeline instead (see
``hand_control.py``): the cursor follows the index fingertip and a pinch
clicks. Everything else - the people, their settings, desktop control - is
shared, so both ways of controlling Paralic are one application.

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
from .calibration import (FIXATION_KINDS, CalibrationData, CalibrationError, LabeledFrame, ProfileStore,
                          SettleTracker, calibrated_pose, evaluate_validation, fit_adjustment, fit_eye_models,
                          fit_full_calibration)
from .faceprint import MIN_SAMPLES, FaceRecognizer, FaceSample, choose_samples, make_sample
from .features import extract_features, face_lighting, mesh_overlay, overlay_points
from .filters import GazeStabilizer
from .gazenet import GazeNet, ModelConfig
from .gestures import BLINK_SIGNALS, WinkDetector, WinkEvent, analyze_winks, blink_signal, other_eye, wink_config
from .hand_control import GESTURE_HELP as HAND_GESTURES, HandControl, HandSetupError
from .personalize import (EXPERIMENTS, SMOOTHING_LEVELS, TRIAL_TIMEOUT_MS, PersonalizationError, analyze_blinks,
                          analyze_experiment, experiment_arms, recommend_magnet, run_finetune, smoothing_params,
                          tune_smoothing)
from .oscontrol import OSController
from .system_control import SystemController, hand_config
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
    # How the cursor moves (web/js/motion.js).
    "motion": "balanced",    # glide | balanced | snappy: how quickly it glides to a new place
    "hold_still": True,      # rest on the average of each fixation
    "head_nudge": False,     # small head tilts move the cursor
    "nudge_speed": 60,       # px/s per degree of tilt beyond the dead zone
    "nudge_deadzone": 5,     # degrees of tilt that do nothing
    "nudge_right": -1,       # the sign of the head yaw that means "right" for this person
    "nudge_up": 1,           # the sign of the head pitch that means "up"
}
_GESTURE_CHOICES = {
    "left_hold": ("drag", "menu", "off"), "right_hold": ("drag", "menu", "off"),
    "left_quick": ("off", "click", "menu"), "right_quick": ("off", "click", "menu"),
    "long_close": ("off", "menu", "grab", "click"), "tracking_eye": ("auto", "both", "left", "right"),
    "motion": ("glide", "balanced", "snappy"), "nudge_right": (-1, 1), "nudge_up": (-1, 1),
}
_GESTURE_RANGES = {"long_close_ms": (600, 3000), "hold_ms": (200, 1500), "long_press_ms": (500, 3000),
                   "dwell_ms": (400, 3000), "nudge_speed": (15, 250), "nudge_deadzone": (2, 15)}
_GESTURE_FLAGS = ("dwell", "left_forced", "right_forced", "hold_still", "head_nudge")

# Point numbers of the extra dots of accuracy-improving round n start at n * this.
REFINE_POINT_BASE = 1000

# Face print (see faceprint.py): how often a frame is described, how many
# frames a recognition looks at, how often the print may grow, when it is saved.
FACE_SAMPLE_GAP_S = 0.25
FACE_WINDOW = 6
FACE_LEARN_GAP_S = 2.0
FACE_SAVE_GAP_S = 10.0
FACE_CHECK_GAP_S = 6.0
FACE_LOST_S = 1.0
# A frame joins a ready print when it is at least this unlike the stored ones
# (in units of the person's typical difference).
FACE_NOVEL = 1.0

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
    "hand_calibration_start": "hand_calibration_started",
    "hand_calibration_fit": "hand_calibration_result",
    "hand_calibration_cancel": "hand_calibration_cancelled",
    "hand_profile_load": "hand_profile",
    "hand_profile_delete": "hand_profile",
    "hand_settings": "hand_settings",
}

# Commands that need the face camera pipeline; a hand mode session answers them
# with an error rather than doing something meaningless.
_EYE_ONLY = frozenset({"calibration_start", "calibration_fit", "validation_finish", "label_event", "finetune",
                       "blink_calibration_start", "blink_calibration_finish", "wink_calibration_start",
                       "wink_calibration_finish", "face_recognize", "experiment_log"})
_HAND_ONLY = frozenset({"hand_calibration_start", "hand_calibration_fit", "hand_calibration_cancel",
                        "hand_settings"})


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
    def __init__(self, tracker_factory: Callable[[], Any], users: UserStore,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 push: Optional[Callable[[dict], None]] = None, mode: str = "eyes"):
        # ``tracker_factory`` makes a FaceTracker - or a HandTracker in hand mode.
        self.mode = "hand" if mode == "hand" else "eyes"
        self.hand: Optional[HandControl] = HandControl() if self.mode == "hand" else None
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
        # How long the eyes have rested on the calibration dot being recorded.
        self._settle = SettleTracker()
        # An accuracy-improving round in progress: the model before it, the
        # round number and where its extra dots' point numbers start.
        self._refine: Optional[dict] = None
        # Face prints: everyone's (loaded when first needed), the recogniser
        # built from them, a recognition in progress. `_face_trust`: whose face
        # the camera is following - set when the person chose themselves,
        # calibrated or was recognised, and kept only while the face stays in
        # view (losing it may mean someone else sits down). Only then are new
        # frames learned, so a print never takes in someone else's face.
        self._prints: Optional[dict[str, list[FaceSample]]] = None
        self._recognizer: Optional[FaceRecognizer] = None
        self._face_next = 0.0
        self._face_window: Optional[dict] = None
        self._face_trust: Optional[str] = None
        self._face_seen = 0.0
        self._face_last_add = 0.0
        self._face_dirty_since = 0.0
        self._face_recent: deque = deque(maxlen=4)
        self._face_check = {"next": 0.0, "hits": 0, "quiet_until": 0.0}
        self.stabilizer = GazeStabilizer(smoothing_params(SMOOTHING_LEVELS["medium"]))
        # System-wide ("control my whole computer") desktop cursor control. Off
        # by default; constructing the OSController is a safe no-op off macOS.
        self.system = SystemController(OSController(), hand_config() if self.mode == "hand" else None)
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
        self._load_hand()

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
            self._save_face()
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
            "pose": calibrated_pose(self.data),
            "legacy": bool(self.model is not None and self.model.meta.get("legacy")),
            "faceprint": self.faceprint_view(),
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
                log.exception("%s tracking failed (%d times)", "Hand" if self.hand else "Face",
                              self._tracker_errors)
            return [{"type": "frame", "id": frame_id, "face": False, "error": f"tracker: {exc}"}]
        self._frame_times.append(t)
        if self.hand is not None:
            return self._hand_frame(t, frame_id, header, obs, started)
        msg: dict[str, Any] = {"type": "frame", "id": frame_id}
        face_events: list[dict] = []
        features = raw = None
        winking = None

        if obs is None:
            if t - self._face_seen > FACE_LOST_S:
                # The face left the camera: whoever comes back must be recognised again.
                self._face_trust = None
                self._face_recent.clear()
            events: list = self.blink.update_missing(t)
            events += self.wink.update_missing(t)
            if self.wink.winking is None:
                self._wink_eye = None
            gaze, frozen = self.stabilizer.update(t, None, False)
            msg.update(face=False, gaze=_xy(gaze), raw=None, frozen=frozen, labeled=False)
            closing = False
        else:
            self._face_seen = t
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
                    if label["kind"] in FIXATION_KINDS:
                        msg["settled"] = self._settle.update((label["kind"], point, target), feats.vector)

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
                pos=[round(float(features[17]), 1), round(float(features[18]), 1)],
            )
            if header.get("setup"):
                msg["light"] = face_lighting(rgb, feats.face_box)
            if header.get("overlay"):
                msg["eyes"] = overlay_points(obs.points_px, obs.image_size, feats)
            if header.get("mesh"):
                msg["mesh"] = mesh_overlay(obs.points_px, obs.image_size)
            if t >= self._face_next and self._face_wanted(t):
                self._face_next = t + FACE_SAMPLE_GAP_S
                if not closing and max(cl, cr) < 0.45 and abs(feats.yaw_deg) <= 30 and abs(feats.pitch_deg) <= 25:
                    try:
                        sample = make_sample(rgb, obs.points_px, feats.yaw_deg, feats.pitch_deg, self.wall())
                    except Exception:  # a face print must never break tracking
                        log.debug("Face description failed", exc_info=True)
                        sample = None
                    if sample is not None:
                        face_events.extend(self._face_frame(sample, t))

        self._history.append(_FrameRecord(t=t, frame_id=frame_id, closing=closing, gaze=gaze,
                                          features=None if features is None else features.copy(),
                                          raw=raw, wall=self.wall(), winking=winking))
        w = self._face_window
        if w is not None and t >= w["until"]:
            face_events.append(self._finish_face_window())
        if self._face_dirty_since and t - self._face_dirty_since >= FACE_SAVE_GAP_S:
            self._save_face()
        msg["ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        msg["fps"] = self._fps()
        out = [msg]
        out.extend(self._event_message(ev, frame_id, gaze) for ev in events)
        out.extend(face_events)

        # System-wide desktop control: move the real OS cursor to the gaze point,
        # click on a double blink, scroll at the screen edges. Off unless the user
        # turned it on (and Accessibility is granted); may auto-disable itself via
        # its kill switches, in which case we tell the page.
        if self.system.enabled:
            g = None if gaze is None else (float(gaze[0]), float(gaze[1]))
            res = self.system.update(t, g, obs is not None)
            if self.system.enabled and any(getattr(ev, "type", None) == "double_blink" for ev in events):
                self.system.click(g)
            if res.disabled_reason:
                out.append(self.system.state())
        return out

    def _hand_frame(self, t: float, frame_id: Any, header: dict, obs, started: float) -> list[dict]:
        """Hand mode: the fingertip is the cursor, a pinch clicks (see hand_control.py)."""
        label = header.get("label") if isinstance(header.get("label"), dict) else None
        fields, messages, events = self.hand.frame(t, frame_id, None if obs is None else obs.points_norm,
                                                   label, self.screen)
        msg: dict[str, Any] = {"type": "frame", "id": frame_id, **fields}
        msg["ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        msg["fps"] = self._fps()
        out = [msg, *messages]
        if self.system.enabled:
            g = None if fields["gaze"] is None else (float(fields["gaze"][0]), float(fields["gaze"][1]))
            res = self.system.update(t, g, fields["face"])
            if self.system.enabled and any(ev.type == "click" for ev in events):
                at = next((m.get("at") for m in messages if m["type"] == "double_blink"), None)
                self.system.click(tuple(at) if at else g)
            for m in messages:
                if m["type"] == "hand_scroll":
                    self.system.scroll_by(m["dy"])
            if res.disabled_reason:
                out.append(self.system.state())
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
            if self.hand is not None and kind in _EYE_ONLY:
                raise CalibrationError("This works with eye tracking - Paralic is in hand mode now")
            if self.hand is None and kind in _HAND_ONLY:
                raise CalibrationError("This works in hand mode - Paralic is using eye tracking now")
            with self._lock:
                return handler(cmd)
        except (CalibrationError, PersonalizationError, HandSetupError) as exc:
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
                 "user": self.user, "users": self.users.list(), "personal": self.personal_view(),
                 "mode": self.mode, **self._hand_view()}]

    def _cmd_settings(self, cmd: dict) -> list[dict]:
        self._apply_settings(cmd)
        return [{"type": "settings", **self.settings, "effective": self.effective()}]

    def _cmd_system_control(self, cmd: dict) -> list[dict]:
        """Turn whole-computer desktop control on/off. ``_cmd_`` prefix is the
        command dispatcher's convention (see handle_command)."""
        if cmd.get("status"):
            return [self.system.state()]
        if cmd.get("enabled"):
            return [self.system.enable()]
        return [self.system.disable("Desktop control turned off.")]

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
        self._settle.reset()
        self._face_trust = self.user["id"]           # calibrating as themselves
        if mode in ("adjust", "refine") and self.model is None:
            return [{"type": "calibration_started", "ok": False, "error": "No calibration to improve"}]
        if mode == "adjust":
            self.data.clear(["adjust"])
            self._refine = None
        elif mode == "refine":
            # Another round to improve the accuracy: the dots that measured it
            # (and a quick adjust's) are good fixations, so they become ordinary
            # calibration dots, and new ones will measure the accuracy afresh.
            # The current network stays the champion until a new one beats it.
            rnd = (self._refine or {}).get("round", 0) + 1
            first = max((f.point for f in self.data.of_kind("cal")), default=0) + 1
            moved = {}
            for f in self.data.frames:
                if f.kind in ("val", "adjust"):
                    key = (f.kind, f.point)
                    moved.setdefault(key, first + len(moved))
                    f.kind, f.point = "cal", moved[key]
            base = REFINE_POINT_BASE * rnd
            self._refine = {"champion": self.model, "round": rnd, "base": base}
            return [{"type": "calibration_started", "ok": True, "mode": mode, "round": rnd, "point_base": base}]
        else:
            # Keep the fine-tuning samples: they still describe this person's eyes.
            self.data.clear(["cal", "head", "val", "adjust"])
            self._refine = None
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
        refine = self._refine
        if refine is not None and refine["champion"] is not self.model:
            # An improving round: both networks are measured on the same new
            # dots, which neither was trained on, and the better one stays.
            before = evaluate_validation(refine["champion"], self.data)
            kept = "new" if result["mean_error_px"] <= before["mean_error_px"] else "previous"
            summary = {"round": refine["round"], "before_px": round(before["mean_error_px"], 1),
                       "after_px": round(result["mean_error_px"], 1), "kept": kept}
            if kept == "previous":
                # The extra dots did not help (the eyes may have been elsewhere): forget them.
                base = refine["base"]
                self.data.frames = [f for f in self.data.frames
                                    if not (f.kind == "cal" and base <= f.point < base + REFINE_POINT_BASE)]
                self.model = refine["champion"]
                result = before
            result["refine"] = summary
            refine["champion"] = self.model
        # Use the validation dots as extra training data for the final network
        # (unless only measuring, e.g. after a quick adjust).
        refit = bool(cmd.get("refit", True))
        if refit:
            try:
                model, info = fit_full_calibration(self.data, include_validation=True)
                self.model = model
                result["refit"] = info
            except CalibrationError:
                log.info("Refit with validation data failed; keeping the first model")
            if refine is not None:
                refine["champion"] = self.model
        self.stabilizer.reset()
        # Personalise smoothing and the button magnet from the measured precision / accuracy.
        old_sm = self.personal.get("smoothing") or {}
        old_mg = self.personal.get("magnet") or {}
        self.personal["smoothing"] = {**tune_smoothing(result["precision_px"]), "bias": old_sm.get("bias", 0.0)}
        self.personal["magnet"] = {**recommend_magnet(result["mean_error_px"]), "scale": old_mg.get("scale", 1.0),
                                   "off": old_mg.get("off", False)}
        self.profile_meta["accuracy_px"] = round(result["mean_error_px"], 1)
        if refit:
            self._record_model("refine" if refine is not None else "calibration",
                               accuracy_px=round(result["mean_error_px"], 1),
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
        self._refine = None
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
                 "profile": self.profiles.summary(), "personal": self.personal_view(), **self._hand_view()}]

    def _switch_user(self, user: dict) -> None:
        self._save_if_dirty()
        self._save_face()
        self._face_trust = None
        self._face_recent.clear()
        self.user = user
        self.personal = self.users.load_personal(user["id"])
        self.model = None
        self.data = CalibrationData()
        self.profile_meta = {}
        self._refine = None
        self._overrides = {}
        self._blink_recording = None
        self._wink_recording = None
        self._wink_eye = None
        self.stabilizer.reset()
        self.blink.reset()
        self.wink.reset()
        self._apply_effective()
        self._load_hand()

    def _cmd_users(self, cmd: dict) -> list[dict]:
        return self._users_reply()

    def _cmd_user_select(self, cmd: dict) -> list[dict]:
        self._switch_user(self.users.select(str(cmd.get("id"))))
        self._face_trust = self.user["id"]            # they picked themselves
        return self._users_reply()

    def _cmd_user_create(self, cmd: dict) -> list[dict]:
        self._switch_user(self.users.create(cmd.get("name")))
        self._face_trust = self.user["id"]
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
        if self._prints is not None:
            self._prints.pop(user_id, None)
        self._recognizer = None
        if user_id == self.user["id"]:
            self._unsaved_events = 0  # nothing left to save for them
            self._switch_user(self.users.ensure_active())
        return self._users_reply()

    # -- hand mode -------------------------------------------------------------------------------
    def _load_hand(self) -> None:
        if self.hand is not None:
            self.hand.load(self.users.load_hand(self.user["id"]))

    def _hand_view(self) -> dict:
        if self.hand is None:
            return {}
        return {"hand": self.hand.view(), "hand_gestures": HAND_GESTURES}

    def _cmd_hand_calibration_start(self, cmd: dict) -> list[dict]:
        info = self.hand.start_calibration(str(cmd.get("mode", "full")))
        return [{"type": "hand_calibration_started", "ok": True, **info}]

    def _cmd_hand_calibration_cancel(self, cmd: dict) -> list[dict]:
        self.hand.cancel_calibration()
        return [{"type": "hand_calibration_cancelled", "ok": True}]

    def _cmd_hand_calibration_fit(self, cmd: dict) -> list[dict]:
        try:
            summary = self.hand.fit(self.screen)
        except HandSetupError:
            self.hand.cancel_calibration()
            raise
        saved = True
        try:
            self.users.save_hand(self.user["id"], self.hand.document(_now()))
        except OSError:
            log.warning("Could not save the hand setup", exc_info=True)
            saved = False
        return [{"type": "hand_calibration_result", "ok": True, "saved": saved, **summary}]

    def _cmd_hand_profile_load(self, cmd: dict) -> list[dict]:
        """Reload the person's saved hand setup (also answers in eye mode: whether one exists)."""
        if self.hand is not None:
            self._load_hand()
            return [{"type": "hand_profile", "ok": True, **self._hand_view()}]
        doc = self.users.load_hand(self.user["id"])
        return [{"type": "hand_profile", "ok": True, "hand": (doc or {}).get("summary") if doc else None}]

    def _cmd_hand_profile_delete(self, cmd: dict) -> list[dict]:
        self.users.delete_hand(self.user["id"])
        self._load_hand()
        return [{"type": "hand_profile", "ok": True, "deleted": True, **self._hand_view()}]

    def _cmd_hand_settings(self, cmd: dict) -> list[dict]:
        return [{"type": "hand_settings", "ok": True, **self.hand.set_options(cmd)}]

    # -- face print ----------------------------------------------------------------------------
    def _face_enabled(self) -> bool:
        return bool((self.personal.get("faceprint") or {}).get("enabled", True))

    def _all_prints(self) -> dict[str, list[FaceSample]]:
        if self._prints is None:
            self._prints = {}
            for u in self.users.list():
                samples = self.users.face_store(u["id"]).load()
                if samples:
                    self._prints[u["id"]] = samples
        return self._prints

    def _face_recognizer(self) -> Optional[FaceRecognizer]:
        if self._recognizer is None:
            ready = {u: s for u, s in self._all_prints().items() if len(s) >= MIN_SAMPLES}
            self._recognizer = FaceRecognizer(ready) if ready else None
        return self._recognizer

    def _face_wanted(self, t: float) -> bool:
        """Describe this frame's face? (Recognising, learning, or checking who is there.)"""
        if self._face_window is not None:
            return True
        if not self._face_enabled():
            return False
        if self._face_trust == self.user["id"]:
            return t - self._face_last_add >= FACE_LEARN_GAP_S
        # Not sure whose face this is: look now and then (see _check_face).
        return t >= self._face_check["next"] and self._face_recognizer() is not None

    def _face_frame(self, sample: FaceSample, t: float) -> list[dict]:
        out = []
        w = self._face_window
        if w is not None:
            w["samples"].append(sample)
            if len(w["samples"]) >= FACE_WINDOW:
                out.append(self._finish_face_window())
        self._face_recent.append(sample)
        if self._face_enabled():
            self._learn_face(sample, t)
            event = self._check_face(t)
            if event:
                out.append(event)
        return out

    def _names(self) -> dict[str, str]:
        return {u["id"]: u["name"] for u in self.users.list()}

    def _finish_face_window(self) -> dict:
        w, self._face_window = self._face_window, None
        rec = self._face_recognizer()
        if rec is None or not w["samples"]:
            result = {"user": None, "confident": False, "scores": {},
                      "reason": "no face prints yet" if rec is None else "no clear view of the face"}
        else:
            result = rec.identify(w["samples"])
        if result.get("user"):
            # Recognised: while this face stays in view it is theirs.
            self._face_trust = result["user"]
        names = self._names()
        return {"type": "face_recognition", "ok": True, **result, "name": names.get(result.get("user")),
                "frames": len(w["samples"]), "scores": {names.get(u, u): v for u, v in result.get("scores", {}).items()}}

    def _learn_face(self, sample: FaceSample, t: float) -> None:
        """Keep frames that are new for this person - only while the camera
        follows their face (see ``_face_trust``)."""
        uid = self.user["id"]
        if self._face_trust != uid or t - self._face_last_add < FACE_LEARN_GAP_S:
            return
        prints = self._all_prints()
        mine = prints.get(uid, [])
        rec = None
        if len(mine) < MIN_SAMPLES:
            # Starting a print: frames that differ (pose, light) from the ones kept so far.
            if mine and min(abs(sample.yaw - s.yaw) + abs(sample.pitch - s.pitch) + abs(sample.light - s.light) / 4
                            for s in mine) < 3.0:
                return
        else:
            rec = self._face_recognizer()
            if rec is not None and uid in rec.prints:
                if rec.novelty(sample, uid) < FACE_NOVEL:
                    return          # nothing new
                if rec.relative_scores([sample]).get(uid, 0.0) > 12.0:
                    return          # far too unlike them: a tracking glitch, not a new look
        prints[uid] = choose_samples(mine, sample, rec.metric if rec else None)
        self._face_last_add = t
        self._face_dirty_since = self._face_dirty_since or t
        self._recognizer = None

    def _check_face(self, t: float) -> Optional[dict]:
        """While unsure whose face this is, look every few seconds: the current
        person (then trust it again), or someone else with a face print?"""
        c = self._face_check
        if self._face_trust is not None or t < c["next"]:
            return None
        c["next"] = t + FACE_CHECK_GAP_S
        rec = self._face_recognizer()
        if rec is None or len(self._face_recent) < 3:
            return None
        result = rec.identify(list(self._face_recent))
        other = result.get("user")
        if other == self.user["id"]:
            # Trusting the face again lets it be learned: ask for a close match.
            if rec.identify(list(self._face_recent), strict=True).get("user") == other:
                self._face_trust = other
            c["hits"] = 0
            return None
        if other:
            c["hits"] += 1
        else:
            c["hits"] = 0
        if c["hits"] >= 2 and t >= c["quiet_until"]:
            c["hits"] = 0
            c["quiet_until"] = t + 60.0
            return {"type": "face_changed", "user": other, "name": self._names().get(other)}
        return None

    def _save_face(self) -> None:
        if not self._face_dirty_since or self._prints is None:
            return
        self._face_dirty_since = 0.0
        uid = self.user["id"]
        try:
            with self.users.lock_for(uid):
                self.users.face_store(uid).save(self._prints.get(uid, []))
        except OSError:
            log.warning("Could not save the face print", exc_info=True)

    def _forget_face(self) -> None:
        uid = self.user["id"]
        self.users.face_store(uid).forget()
        if self._prints is not None:
            self._prints.pop(uid, None)
        self._recognizer = None
        self._face_dirty_since = 0.0

    def faceprint_view(self) -> dict:
        n = len(self._all_prints().get(self.user["id"], []))
        return {"enabled": self._face_enabled(), "samples": n, "ready": n >= MIN_SAMPLES}

    def _cmd_face_recognize(self, cmd: dict) -> list[dict]:
        if self._face_recognizer() is None:
            return [{"type": "face_recognition", "ok": True, "user": None, "confident": False, "scores": {},
                     "reason": "no face prints yet"}]
        seconds = float(np.clip(float(cmd.get("seconds", 4.0)), 1.0, 10.0))
        self._face_window = {"samples": [], "until": self.clock() + seconds}
        self._face_next = 0.0
        return []      # the answer (face_recognition) comes with the frames

    def _cmd_face_confirm(self, cmd: dict) -> list[dict]:
        """The person at the camera said they are the current person (e.g. chose to go on as them)."""
        self._face_trust = self.user["id"]
        return [{"type": "face_confirmed", "ok": True}]

    def _cmd_faceprint_set(self, cmd: dict) -> list[dict]:
        enabled = cmd.get("enabled")
        if not isinstance(enabled, bool):
            raise PersonalizationError("enabled must be true or false")
        self.personal["faceprint"] = {"enabled": enabled}
        if not enabled:
            self._forget_face()
        self._save_personal()
        return [{"type": "personal", "ok": True, "personal": self.personal_view()}]

    def _cmd_faceprint_forget(self, cmd: dict) -> list[dict]:
        self._forget_face()
        return [{"type": "personal", "ok": True, "personal": self.personal_view()}]

    def _cmd_faceprint_faces(self, cmd: dict) -> list[dict]:
        """The kept faces of the current person (small grey JPEGs) for the Lab page."""
        import base64

        self._save_face()
        store = self.users.face_store(self.user["id"])
        faces = []
        for s in self._all_prints().get(self.user["id"], []):
            pic = store.picture(s.id)
            if pic:
                faces.append({"id": s.id, "t": round(s.t), "yaw": round(s.yaw), "pitch": round(s.pitch),
                              "src": "data:image/jpeg;base64," + base64.b64encode(pic).decode("ascii")})
        return [{"type": "faceprint_faces", "ok": True, "faces": faces}]

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
        if self._job_running() or self.model is None or self.model.meta.get("legacy"):
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
        if self.model.meta.get("legacy"):
            raise PersonalizationError("This calibration comes from an older version of Paralic: "
                                       "do a full calibration first, then fine-tuning can improve it.")
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
            if isinstance(value, bool):
                if key in _GESTURE_FLAGS:
                    g[key] = value
            elif key in _GESTURE_CHOICES and value in _GESTURE_CHOICES[key]:
                g[key] = value
            elif key in _GESTURE_RANGES and isinstance(value, (int, float)):
                lo, hi = _GESTURE_RANGES[key]
                g[key] = int(np.clip(value, lo, hi))
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
