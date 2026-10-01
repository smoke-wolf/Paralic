"""What the Inspector's panels show, computed from a recording.

:class:`RecordingView` wraps a :class:`~paralic.recording.Recording` and
caches what is expensive to compute again:

* **Signals.** Every frame's numbers - eye closures with the blink and wink
  detectors' thresholds and states, raw and smoothed gaze, head pose, timing,
  glasses, which network led, and (eye mode) the gaze network's ensemble spread
  and how far its output, computed again here, lies from the recorded one - as
  columns. They are parsed in blocks of frames when first needed and kept; a
  request for a stretch of time returns the frames themselves, or, when there
  are more than the screen can show, the minimum and maximum of each column per
  bucket (so a blink of three frames still shows in an hour-long view).
* **Timeline.** Markers for the gesture events, clicks, model and person
  changes, and stretches: calibration phases, the wink test, frames without a
  face, other faces in view, glasses and glare, winks and pinches.
* **Calibrations.** Each calibration, quick adjust or hand setup in the
  events: its dots (from the labelled frames: how many frames each dot was
  shown for and how many were kept as samples), the fit, the per-dot errors of
  the validation and the accuracy over time.
* **Config.** ``meta.json`` and every settings, gesture and person change.
* **Frame detail.** One frame with its landmarks, video frame and - in eye
  mode - the gaze networks' internals (``internals.explain``) and the
  detectors' state; in hand mode the recogniser's measurements.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Optional

import numpy as np

from .. import landmarks as L
from ..features import EYE_INPUTS, FEATURE_NAMES
from ..recording import Recording
from . import internals

log = logging.getLogger(__name__)

BLOCK = 1024
ON_DEMAND_BLOCKS = 8        # a request computes at most this many blocks itself; the rest come from the warm-up

# What the camera panel draws: the face mesh's outlines and eye landmarks (MediaPipe
# indices, see landmarks.py) and the hand skeleton's bones.
MESH = {
    "lines": [list(g) for g in L.MESH_OUTLINES],
    "right_eye": list(L.RIGHT_EYE_CONTOUR), "left_eye": list(L.LEFT_EYE_CONTOUR),
    "right_iris": list(L.RIGHT_IRIS), "left_iris": list(L.LEFT_IRIS),
    "right_corners": [L.RIGHT_EYE_OUTER, L.RIGHT_EYE_INNER], "left_corners": [L.LEFT_EYE_INNER, L.LEFT_EYE_OUTER],
    "nose_tip": 1,
}
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

LABEL_KINDS = ["", "cal", "head", "val", "adjust", "ft", "hspan", "hpinch", "hpoint"]
NETS = ["both", "left", "right"]
EYE_SIDES = {None: 0.0, "left": 1.0, "right": 2.0, "both": 3.0}
WINK_STATES = {"idle": 0.0, "candidate": 1.0, "active": 2.0}
TEST_ASKS = {"rest": 0.0, "left": 1.0, "right": 2.0}      # the wink test's request (a frame's "gesture")

# Columns per mode; booleans and categories are stored as numbers (NaN = missing).
EYE_COLUMNS = ("face", "faces", "waiting", "cl", "cr", "closure", "signal", "close_thr", "open_thr", "baseline",
               "pending", "closing", "closed", "deep", "winking", "pressed", "wstate", "wclose_l", "wclose_r",
               "wopen_l", "wopen_r", "gesture", "gx", "gy", "rx", "ry", "tx", "ty", "frozen", "stored", "label",
               "lpt", "yaw", "pitch", "roll", "dist", "fps", "ms", "glasses", "glare_l", "glare_r", "glare",
               "net", "spread", "check")
HAND_COLUMNS = ("face", "hands", "pinch", "span", "pinching", "scrolling", "open", "tipx", "tipy", "gx", "gy", "rx",
                "ry", "tx", "ty", "frozen", "stored", "label", "lpt", "fps", "ms")
# Columns whose buckets keep the first value rather than the extremes.
CATEGORICAL = {"net", "label", "lpt", "winking", "pressed", "wstate", "gesture", "glare"}

# Event types shown as markers on the timeline (kind "event" unless given).
MARKERS = {
    "blink": "blink", "double_blink": "click", "blink_expired": "expired", "long_close": "long_close",
    "long_close_ready": "long_close", "wink": "wink", "wink_start": "wink", "wink_end": "wink",
    "hand_scroll": "scroll", "hand_palm": "palm", "face_recognition": "face", "face_changed": "face",
    "glasses_changed": "glasses", "person_changed": "person", "video_stopped": "video",
}
COMMAND_MARKERS = {"label_event": "click", "settings": "setting", "gestures_set": "setting",
                   "hand_settings": "setting", "user_select": "person", "user_create": "person",
                   "profile_load": "model", "calibration_start": "calibration", "hand_calibration_start": "calibration",
                   "face_recognize": "face"}

DEFAULT_PINCH = (0.45, 0.60)            # HandGestureConfig's pinch_on / pinch_off


def _num(v) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else math.nan


def _flag(v) -> float:
    return math.nan if v is None else (1.0 if v else 0.0)


def _pair(v, k: int) -> float:
    return _num(v[k]) if isinstance(v, (list, tuple)) and len(v) > k else math.nan


def jsonable(v: Any, digits: int = 5, sig: Optional[int] = None):
    """NumPy arrays and scalars as JSON values, rounded to ``digits`` decimals (or ``sig``
    significant digits, which keeps tiny values); NaN and infinity as null."""
    if isinstance(v, dict):
        return {str(k): jsonable(x, digits, sig) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x, digits, sig) for x in v]
    if isinstance(v, np.ndarray):
        if v.dtype == bool:
            return v.tolist()
        if v.dtype.kind == "f" and sig is None and v.ndim == 1:
            # The signal columns: round in NumPy, then only the gaps need a look.
            r = np.round(v.astype(np.float64), digits)
            out = r.tolist()
            if not np.isfinite(r).all():
                out = [x if math.isfinite(x) else None for x in out]
            return out
        return jsonable(v.tolist(), digits, sig)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            return None
        return float(f"{f:.{sig}g}") if sig else round(f, digits)
    return v


class RecordingView:
    """The panels' data for one recording, with caches (thread-safe)."""

    def __init__(self, rec: Recording):
        self.rec = rec
        self._lock = threading.RLock()
        self._blocks: dict[int, dict[str, np.ndarray]] = {}
        self._seen = (rec.version, len(rec))
        self._models_key = self._model_key()
        self._timeline: Optional[dict] = None
        self._warming: Optional[threading.Thread] = None
        self.closed = False

    # -- keeping up with a growing recording -----------------------------------------
    def _model_key(self) -> tuple:
        return tuple((t, m.name if m is not None else None) for t, m in self.rec.model_changes())

    def refresh(self) -> None:
        self.rec.refresh()
        with self._lock:
            if (self.rec.version, len(self.rec)) == self._seen:
                return
            old_len = self._seen[1]
            self._seen = (self.rec.version, len(self.rec))
            self._timeline = None
            key = self._model_key()
            if key != self._models_key:
                self._models_key = key
                self._blocks.clear()            # model-derived columns depend on when each model applies
            else:
                self._blocks.pop(old_len // BLOCK, None)    # the last block may have grown
        self.warm()

    # -- the whole recording, in the background ----------------------------------------
    def warm(self) -> None:
        """Compute every block's columns on a background thread, in order, so the
        whole-recording views (overview, timeline) fill in while the panels that
        follow the current frame are already usable."""
        with self._lock:
            if self.closed or (self._warming is not None and self._warming.is_alive()):
                return
            self._warming = threading.Thread(target=self._warm, name="paralic-inspector-warm", daemon=True)
            self._warming.start()

    def _warm(self) -> None:
        try:
            b = 0
            while not self.closed and b * BLOCK < len(self.rec):
                self._block(b)
                b += 1
        except Exception:  # a damaged recording must not take the server down
            log.exception("Reading %s in the background failed", self.rec.id)

    def progress(self) -> float:
        """Share of the recording's frames whose columns are computed."""
        n = len(self.rec)
        if not n:
            return 1.0
        with self._lock:
            done = sum(min(BLOCK, n - b * BLOCK) for b in self._blocks if b * BLOCK < n)
        return min(1.0, done / n)

    def close(self) -> None:
        self.closed = True

    @property
    def mode(self) -> str:
        return self.rec.mode

    # -- info ---------------------------------------------------------------------------
    def info(self) -> dict:
        rec = self.rec
        t0, t1 = rec.span()
        return jsonable({
            **rec.summary(),
            "t0": t0, "t1": t1,
            "screen": rec.screen,
            "meta": rec.meta,
            "models": [m.info() for m in rec.models()],
            "feature_names": list(FEATURE_NAMES),
            "eye_inputs": {eye: list(cols) for eye, cols in EYE_INPUTS.items()},
            "mesh": MESH if self.mode == "eyes" else None,
            "hand_bones": HAND_BONES if self.mode == "hand" else None,
            "label_kinds": LABEL_KINDS,
            "nets": NETS,
            "columns": list(EYE_COLUMNS if self.mode == "eyes" else HAND_COLUMNS),
            "landmark_chunks": rec.landmark_chunks,
            "image": self.image_size(),
            "pinch": self.pinch_thresholds() if self.mode == "hand" else None,
        }, 4)

    def index(self) -> dict:
        """Every frame's time and id, and whether it has a camera image (for the player)."""
        ids = self.rec.frame_ids
        video = np.isin(ids, self.rec.video_ids()).astype(np.int8)
        return {"t": np.round(self.rec.times, 4).tolist(), "i": ids.tolist(), "v": video.tolist()}

    # -- signals ------------------------------------------------------------------------
    def _block(self, b: int) -> dict[str, np.ndarray]:
        with self._lock:
            cached = self._blocks.get(b)
        if cached is not None:
            return cached
        lo, hi = b * BLOCK, min((b + 1) * BLOCK, len(self.rec))
        frames = self.rec.frames(lo, hi, light=True)
        cols = self._eye_columns(frames) if self.mode == "eyes" else self._hand_columns(frames)
        cols["t"] = self.rec.times[lo:hi].astype(np.float64)
        with self._lock:
            if hi - lo == BLOCK or hi == len(self.rec):
                self._blocks[b] = cols
        return cols

    def _eye_columns(self, frames: list[dict]) -> dict[str, np.ndarray]:
        n = len(frames)
        out = {name: np.full(n, np.nan, np.float32) for name in EYE_COLUMNS}
        feats = np.full((n, len(FEATURE_NAMES)), np.nan)
        rows = []
        for k, f in enumerate(frames):
            m = f.get("msg") or {}
            b = f.get("blink") or {}
            w = f.get("wink") or {}
            face = bool(m.get("face"))
            out["face"][k] = 1.0 if face else 0.0
            out["faces"][k] = _num(m.get("faces")) if m.get("faces") is not None else (1.0 if face else 0.0)
            out["waiting"][k] = 1.0 if m.get("waiting") else 0.0     # faces in view, none of them this person's
            out["cl"][k], out["cr"][k], out["closure"][k] = _num(m.get("cl")), _num(m.get("cr")), _num(m.get("closure"))
            out["signal"][k] = _num(b.get("signal"))
            thr = m.get("thr")
            out["close_thr"][k] = _num(b.get("close_thr")) if "close_thr" in b else _pair(thr, 0)
            out["open_thr"][k] = _num(b.get("open_thr")) if "open_thr" in b else _pair(thr, 1)
            out["baseline"][k], out["pending"][k] = _num(b.get("baseline")), _num(b.get("pending"))
            out["closing"][k] = _flag(b.get("closing", m.get("closing")))
            out["closed"][k] = _flag(b.get("closed", m.get("closed")))
            out["deep"][k] = _flag(b.get("deep"))
            out["winking"][k] = EYE_SIDES.get(w.get("winking", m.get("winking")), np.nan)
            out["pressed"][k] = EYE_SIDES.get(w.get("pressed", m.get("wink")), np.nan)
            out["wstate"][k] = WINK_STATES.get(w.get("state"), np.nan)
            out["wclose_l"][k], out["wclose_r"][k] = _pair(w.get("close_thr"), 0), _pair(w.get("close_thr"), 1)
            out["wopen_l"][k], out["wopen_r"][k] = _pair(w.get("open_thr"), 0), _pair(w.get("open_thr"), 1)
            out["gesture"][k] = TEST_ASKS.get(f.get("gesture"), np.nan)
            out["gx"][k], out["gy"][k] = _pair(m.get("gaze"), 0), _pair(m.get("gaze"), 1)
            out["rx"][k], out["ry"][k] = _pair(m.get("raw"), 0), _pair(m.get("raw"), 1)
            self._label(out, k, f)
            out["frozen"][k] = _flag(m.get("frozen"))
            out["yaw"][k], out["pitch"][k], out["roll"][k] = (_pair(m.get("head"), j) for j in range(3))
            out["dist"][k] = _num(m.get("dist"))
            out["fps"][k], out["ms"][k] = _num(m.get("fps")), _num(m.get("ms"))
            out["glasses"][k] = _flag(m.get("glasses"))
            gs = m.get("glare_score")
            glare = m.get("glare")
            out["glare_l"][k] = _pair(gs, 0) if gs is not None else (1.0 if glare in ("left", "both") else
                                                                    0.0 if "glare" in m else np.nan)
            out["glare_r"][k] = _pair(gs, 1) if gs is not None else (1.0 if glare in ("right", "both") else
                                                                    0.0 if "glare" in m else np.nan)
            out["glare"][k] = EYE_SIDES.get(glare, np.nan) if "glare" in m else np.nan
            net = f.get("net", m.get("net"))
            out["net"][k] = float(NETS.index(net)) if net in NETS else np.nan
            vec = f.get("features")
            if isinstance(vec, list) and len(vec) == len(FEATURE_NAMES) and net in NETS:
                feats[k] = [_num(v) for v in vec]
                rows.append(k)
        self._model_columns(frames, out, feats, rows)
        return out

    def _model_columns(self, frames: list[dict], out: dict, feats: np.ndarray, rows: list[int]) -> None:
        """The gaze network's member spread and the recorded raw gaze computed again (eye mode)."""
        groups: dict[tuple, list[int]] = {}
        for k in rows:
            f = frames[k]
            snap = self.rec.model_at(float(f.get("t", math.nan)))
            if snap is None:
                continue
            groups.setdefault((snap.name, f.get("net", (f.get("msg") or {}).get("net"))), []).append(k)
        snaps = {m.name: m for m in self.rec.models()}
        for (name, net), idx in groups.items():
            model = snaps[name].net
            if net != "both" and net not in model.eyes:
                continue
            res = internals.batch_outputs(model, feats[idx], net)
            out["spread"][idx] = res["spread"]
            raw = np.stack([out["rx"][idx], out["ry"][idx]], axis=1)
            check = np.hypot(*(res["final"] - raw).T)
            shifted = np.array([self._shifted(frames[k], model) for k in idx], bool)
            check[shifted] = np.nan
            out["check"][idx] = check

    def _shifted(self, f: dict, model) -> bool:
        """True if the session moved this frame's gaze by an offset: while a wink (or, in newer
        sessions, glare on glasses) hands the cursor to one eye's network, it is lined up with
        the usual network - so the recorded gaze is not that network's output alone."""
        if (f.get("wink") or {}).get("winking", (f.get("msg") or {}).get("winking")) is not None:
            return True
        return f.get("net", (f.get("msg") or {}).get("net")) != self._usual_eye(model, _num(f.get("t")))

    def _usual_eye(self, model, t: float) -> str:
        """The network that leads while both eyes are open (TrackerSession._preferred_eye)."""
        choice = None
        for when, value in self._tracking_choices():
            if when > t:
                break
            choice = value
        if choice in ("left", "right") and choice in model.eyes:
            return choice
        return "both" if choice == "both" else model.eye

    def _tracking_choices(self) -> list[tuple[float, Optional[str]]]:
        """The person's choice of the leading network (the tracking_eye gesture) over time."""
        with self._lock:
            cached = getattr(self, "_tracking", None)
            if cached is not None and cached[0] == self.rec.version:
                return cached[1]

        def choice(conf: dict) -> Optional[str]:
            eff = conf.get("effective") if isinstance(conf.get("effective"), dict) else {}
            gestures = eff.get("gestures") or (conf.get("personal") or {}).get("gestures") or {}
            return gestures.get("tracking_eye") if isinstance(gestures, dict) else None

        out = [(-math.inf, choice(self.rec.meta))]
        for e in self.rec.events():
            d = e.get("data") if isinstance(e.get("data"), dict) else {}
            t = _num(e.get("t"))
            if e.get("type") == "person_changed":
                out.append((t, choice(d)))
            elif e.get("kind") == "command" and e.get("type") == "gestures_set" and \
                    "tracking_eye" in (d.get("gestures") or {}):
                out.append((t, d["gestures"]["tracking_eye"]))
        with self._lock:
            self._tracking = (self.rec.version, out)
        return out

    def _hand_columns(self, frames: list[dict]) -> dict[str, np.ndarray]:
        n = len(frames)
        out = {name: np.full(n, np.nan, np.float32) for name in HAND_COLUMNS}
        for k, f in enumerate(frames):
            m = f.get("msg") or {}
            hand = m.get("hand") or {}
            out["face"][k] = 1.0 if m.get("face") else 0.0
            out["hands"][k] = _num(m.get("hands")) if m.get("hands") is not None else out["face"][k]
            out["pinch"][k], out["span"][k] = _num(hand.get("pinch")), _num(hand.get("span"))
            out["pinching"][k], out["scrolling"][k] = _flag(m.get("pinching")), _flag(m.get("scrolling"))
            out["open"][k] = _flag(hand.get("open")) if m.get("face") else np.nan
            out["tipx"][k], out["tipy"][k] = _pair(hand.get("tip"), 0), _pair(hand.get("tip"), 1)
            out["gx"][k], out["gy"][k] = _pair(m.get("gaze"), 0), _pair(m.get("gaze"), 1)
            out["rx"][k], out["ry"][k] = _pair(m.get("raw"), 0), _pair(m.get("raw"), 1)
            out["frozen"][k] = _flag(m.get("frozen"))
            out["fps"][k], out["ms"][k] = _num(m.get("fps")), _num(m.get("ms"))
            self._label(out, k, f)
        return out

    def _label(self, out: dict, k: int, f: dict) -> None:
        label = f.get("label")
        out["stored"][k] = _flag(f.get("stored", (f.get("msg") or {}).get("labeled")))
        if not isinstance(label, dict):
            out["label"][k] = 0.0
            return
        kind = label.get("kind")
        out["label"][k] = float(LABEL_KINDS.index(kind)) if kind in LABEL_KINDS else float(len(LABEL_KINDS))
        if "fx" in label:                     # hand setup dots: screen fractions
            s = self.rec.screen
            out["tx"][k], out["ty"][k] = _num(label.get("fx")) * s["w"], _num(label.get("fy")) * s["h"]
        else:
            out["tx"][k], out["ty"][k] = _num(label.get("x")), _num(label.get("y"))
        out["lpt"][k] = _num(label.get("pt", label.get("i")))

    def _missing(self, lo: int, hi: int) -> int:
        """How many blocks of frames ``lo .. hi - 1`` are not computed yet."""
        if hi <= lo:
            return 0
        with self._lock:
            return sum(1 for b in range(lo // BLOCK, (hi - 1) // BLOCK + 1) if b not in self._blocks)

    def columns(self, lo: int, hi: int, wait: bool = True) -> dict[str, np.ndarray]:
        """Columns for frames ``lo .. hi - 1``. Without ``wait``, blocks not computed
        yet (by the warm-up) are left missing (NaN) instead of being computed now."""
        lo, hi = max(0, lo), min(hi, len(self.rec))
        names = (EYE_COLUMNS if self.mode == "eyes" else HAND_COLUMNS)
        if hi <= lo:
            return {name: np.zeros(0, np.float32) for name in names + ("t",)}
        parts = []
        for b in range(lo // BLOCK, (hi - 1) // BLOCK + 1):
            a, z = max(lo - b * BLOCK, 0), min(hi - b * BLOCK, BLOCK)
            with self._lock:
                cols = self._blocks.get(b)
            if cols is None and not wait:
                k = min(z, len(self.rec) - b * BLOCK) - a
                cols = {name: np.full(k, np.nan, np.float32) for name in names}
                cols["t"] = self.rec.times[b * BLOCK + a:b * BLOCK + a + k].astype(np.float64)
                parts.append(cols)
                continue
            if cols is None:
                cols = self._block(b)
            parts.append({k: v[a:z] for k, v in cols.items()})
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

    def signals(self, t0: Optional[float] = None, t1: Optional[float] = None, points: int = 2000) -> dict:
        """Columns between ``t0`` and ``t1`` (session time), at most ``points`` samples (buckets).

        A long stretch whose frames are not all read yet comes back partial (the
        missing frames as gaps) with ``progress``; ask again for the rest.
        """
        rec = self.rec
        first, last = rec.span()
        t0 = first if t0 is None else t0
        t1 = last if t1 is None else t1
        if first is None or not len(rec):
            return {"t0": t0, "t1": t1, "lo": 0, "hi": 0, "decimated": False, "t": [], "cols": {}}
        lo, hi = rec.range_of(t0, t1)
        lo, hi = max(lo - 1, 0), min(hi + 1, len(rec))
        partial = self._missing(lo, hi) > ON_DEMAND_BLOCKS
        if partial:
            self.warm()
        cols = self.columns(lo, hi, wait=not partial)
        t = cols.pop("t")
        n = hi - lo
        points = max(16, int(points))
        head = {"t0": t0, "t1": t1, "lo": lo, "hi": hi, "partial": partial, "progress": round(self.progress(), 3)}
        if n <= points:
            return {**head, "decimated": False, "t": jsonable(t, 4), "cols": {k: jsonable(v, 4) for k, v in cols.items()}}
        edges = np.linspace(0, n, points + 1).astype(int)
        starts = np.unique(edges[:-1])
        out = {}
        with np.errstate(all="ignore"):
            for name, v in cols.items():
                if name in CATEGORICAL:
                    first_v = v[starts]
                    out[name] = {"min": jsonable(first_v, 4), "max": jsonable(first_v, 4)}
                else:
                    out[name] = {"min": jsonable(np.fmin.reduceat(v, starts), 4),
                                 "max": jsonable(np.fmax.reduceat(v, starts), 4)}
        ends = np.append(starts[1:], n) - 1
        centre = (t[starts] + t[ends]) / 2
        return {**head, "decimated": True, "t": jsonable(centre, 4),
                "t_lo": jsonable(t[starts], 4), "t_hi": jsonable(t[ends], 4), "cols": out}

    # -- timeline -----------------------------------------------------------------------
    def timeline(self) -> dict:
        with self._lock:
            if self._timeline is not None:
                return self._timeline
        rec = self.rec
        markers = []
        for k, e in enumerate(rec.events()):
            t = _num(e.get("t"))
            if not math.isfinite(t):
                continue
            kind, typ = e.get("kind"), e.get("type")
            d = e.get("data") if isinstance(e.get("data"), dict) else {}
            cat = None
            if kind == "event" and typ in MARKERS:
                cat = MARKERS[typ]
                if typ == "blink" and d.get("n") == 2:
                    cat = "blink2"
            elif kind == "command" and typ in COMMAND_MARKERS:
                cat = COMMAND_MARKERS[typ]
            elif kind == "reply" and d.get("ok") is False:
                cat = "error"
            elif kind == "push":
                cat = "push"
            if cat:
                markers.append({"t": t, "cat": cat, "type": typ, "kind": kind, "event": k,
                                "eye": d.get("eye"), "hand": bool(d.get("hand"))})
        for t, m in rec.model_changes():
            if math.isfinite(t):
                markers.append({"t": t, "cat": "model", "type": m.why if m else "no model", "kind": "model",
                                "name": m.name if m else None})
        # Stretches need every frame: those not read yet are left out until the warm-up has them.
        complete = self._missing(0, len(rec)) <= ON_DEMAND_BLOCKS
        if not complete:
            self.warm()
        stretches = self._stretches(wait=complete)
        out = jsonable({"markers": markers, "stretches": stretches, "calibrations": self._calibration_spans(),
                        "complete": complete, "progress": self.progress()}, 4)
        if complete:
            with self._lock:
                self._timeline = out
        return out

    def _stretches(self, wait: bool = True) -> list[dict]:
        cols = self.columns(0, len(self.rec), wait=wait)
        t = cols.get("t", np.zeros(0))
        if not len(t):
            return []
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30

        def runs(mask: np.ndarray, kind: str, value=None, gap: float = 0.0) -> list[dict]:
            mask = np.asarray(mask, bool)
            if not mask.any():
                return []
            d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
            starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
            out = []
            for a, b in zip(starts, ends):
                t0, t1 = float(t[a]), float(t[b]) + dt
                if out and t0 - out[-1]["t1"] <= gap:
                    out[-1]["t1"] = t1
                else:
                    out.append({"kind": kind, "t0": t0, "t1": t1, **({"value": value} if value is not None else {})})
            return out

        out = []
        out += runs(cols["face"] == 0, "noface")
        label = cols["label"]
        for code, kind in enumerate(LABEL_KINDS):
            if code:
                out += runs(label == code, "label", kind, gap=1.2)
        if self.mode == "eyes":
            out += runs(cols["faces"] >= 2, "faces")
            out += runs(cols["waiting"] == 1, "waiting")
            out += runs(cols["glasses"] == 1, "glasses")
            out += runs((cols["glare"] >= 1) & (cols["glare"] <= 3), "glare")
            for code, eye in ((1.0, "left"), (2.0, "right")):
                out += runs(cols["winking"] == code, "wink", eye)
            out += runs(cols["closed"] == 1, "closed")
            for ask, code in TEST_ASKS.items():
                out += runs(cols["gesture"] == code, "wink_test", ask)
        else:
            out += runs(cols["hands"] >= 2, "hands")
            out += runs(cols["pinching"] == 1, "pinch")
            out += runs(cols["scrolling"] == 1, "scroll")
            out += runs(cols["open"] == 1, "palm")
        return out

    # -- calibrations -------------------------------------------------------------------
    def _calibration_spans(self) -> list[dict]:
        return [{"t0": c["t0"], "t1": c["t1"], "mode": c["mode"], "kind": c["kind"]} for c in self._sessions()]

    def _sessions(self) -> list[dict]:
        """Calibration sessions from the events: start, fit, validation (eye mode) or the hand setup."""
        sessions: list[dict] = []
        cur = None
        last_t = self.rec.span()[1] or 0.0
        for k, e in enumerate(self.rec.events()):
            typ, kind = e.get("type"), e.get("kind")
            d = e.get("data") if isinstance(e.get("data"), dict) else {}
            t = _num(e.get("t"))
            if kind == "command" and typ in ("calibration_start", "hand_calibration_start"):
                cur = {"kind": "hand" if typ.startswith("hand") else "eyes", "mode": d.get("mode", "full"),
                       "t0": t, "t1": None, "events": [k], "started": None, "fit": None, "validation": None}
                sessions.append(cur)
                continue
            if cur is None:
                continue
            if typ in ("calibration_started", "hand_calibration_started") and kind == "reply":
                cur["started"] = d
                cur["events"].append(k)
                if d.get("ok") is False:
                    cur["t1"] = t
                    cur = None
            elif typ in ("calibration_result", "hand_calibration_result") and kind == "reply":
                cur["fit"] = {**d, "t": t}
                cur["events"].append(k)
                cur["t1"] = t
                if cur["kind"] == "hand" or d.get("ok") is False:
                    cur = None
            elif typ == "validation_result" and kind == "reply":
                cur["validation"] = {**d, "t": t}
                cur["events"].append(k)
                cur["t1"] = t
                cur = None
            elif typ == "hand_calibration_cancelled":
                cur["t1"] = t
                cur["cancelled"] = True
                cur = None
        for s in sessions:
            if s["t1"] is None:
                s["t1"] = last_t
        return sessions

    def calibrations(self) -> dict:
        """Each calibration with its dots, fit and validation, and the accuracy over time."""
        rec = self.rec
        out = []
        for s in self._sessions():
            lo, hi = rec.range_of(s["t0"], s["t1"])
            cols = self.columns(lo, hi)
            dots: dict[tuple, dict] = {}
            label, pt, stored = cols["label"], cols["lpt"], cols["stored"]
            for k in np.flatnonzero(label > 0):
                kind = LABEL_KINDS[int(label[k])] if int(label[k]) < len(LABEL_KINDS) else "other"
                key = (kind, None if not np.isfinite(pt[k]) else int(pt[k]))
                d = dots.get(key)
                if d is None:
                    d = dots[key] = {"kind": kind, "pt": key[1], "x": None, "y": None, "frames": 0, "stored": 0,
                                     "t0": float(cols["t"][k]), "t1": float(cols["t"][k]), "raw": []}
                if d["x"] is None and np.isfinite(cols["tx"][k]):
                    d["x"], d["y"] = float(cols["tx"][k]), float(cols["ty"][k])
                d["frames"] += 1
                d["stored"] += int(stored[k] == 1)
                d["t1"] = float(cols["t"][k])
                if np.isfinite(cols["rx"][k]) and stored[k] == 1:
                    d["raw"].append((float(cols["rx"][k]), float(cols["ry"][k])))
            dot_list = []
            for d in dots.values():
                raw = np.array(d.pop("raw")) if d["raw"] else None
                d["raw_mean"] = None if raw is None else raw.mean(axis=0).tolist()
                dot_list.append(d)
            fit = s.get("fit") or {}
            val = s.get("validation") or {}
            errors = {}
            for p in val.get("points") or []:
                if isinstance(p, dict) and p.get("point") is not None:
                    errors[int(p["point"])] = p
            for d in dot_list:
                if d["kind"] == "val" and d["pt"] in errors:
                    p = errors[d["pt"]]
                    d.update(error=p.get("error"), spread=p.get("spread"), mean=p.get("mean"), used=p.get("n"))
            refit = val.get("refit") if isinstance(val.get("refit"), dict) else None
            dropped = set((refit or {}).get("dropped_points") or []) | set(fit.get("dropped_points") or [])
            for d in dot_list:
                d["dropped"] = f"{d['kind']}:{d['pt']}" in dropped
            out.append({**{k: v for k, v in s.items() if k not in ("fit", "validation")},
                        "fit": {k: v for k, v in fit.items() if k != "personal"} or None,
                        "validation": {k: v for k, v in val.items() if k != "personal"} or None,
                        "dots": dot_list})
        return jsonable({"sessions": out, "accuracy": self.accuracy_over_time()}, 3)

    def accuracy_over_time(self) -> list[dict]:
        out = []
        for e in self.rec.events():
            d = e.get("data") if isinstance(e.get("data"), dict) else {}
            if e.get("kind") != "reply" or d.get("ok") is False:
                continue
            if e.get("type") == "validation_result" and d.get("mean_error_px") is not None:
                out.append({"t": e.get("t"), "kind": "validation", "error_px": d.get("mean_error_px"),
                            "precision_px": d.get("precision_px"), "max_px": d.get("max_error_px")})
                refine = d.get("refine")
                if isinstance(refine, dict):
                    out[-1]["refine"] = refine
            elif e.get("type") == "calibration_result":
                if d.get("mode") == "adjust" and d.get("error_after_px") is not None:
                    out.append({"t": e.get("t"), "kind": "adjust", "error_px": d.get("error_after_px"),
                                "before_px": d.get("error_before_px")})
                elif d.get("cv_error_px") is not None:
                    out.append({"t": e.get("t"), "kind": "cv", "error_px": d.get("cv_error_px")})
            elif e.get("type") == "hand_calibration_result" and d.get("pointing_error_px") is not None:
                out.append({"t": e.get("t"), "kind": "hand", "error_px": d.get("pointing_error_px")})
        return out

    # -- config -------------------------------------------------------------------------
    def config(self) -> dict:
        """meta.json and every settings / gesture change, with what it changed."""
        meta = self.rec.meta
        settings = dict(meta.get("settings") or {})
        effective = meta.get("effective") if isinstance(meta.get("effective"), dict) else {}
        gestures = dict((effective or {}).get("gestures") or ((meta.get("personal") or {}).get("gestures") or {}))
        hand: dict = {}
        changes = []
        for k, e in enumerate(self.rec.events()):
            d = e.get("data") if isinstance(e.get("data"), dict) else {}
            typ, kind = e.get("type"), e.get("kind")
            diff = []
            if kind == "command" and typ in ("settings", "hello"):
                patch = d.get("settings") if typ == "hello" else d
                for key, value in (patch or {}).items():
                    if key == "type" or key == "screen":
                        continue
                    if settings.get(key) != value:
                        diff.append({"key": key, "from": settings.get(key), "to": value})
                        settings[key] = value
            elif kind == "command" and typ == "gestures_set":
                for key, value in (d.get("gestures") or {}).items():
                    if gestures.get(key) != value:
                        diff.append({"key": key, "from": gestures.get(key), "to": value})
                        gestures[key] = value
            elif kind == "command" and typ == "hand_settings":
                for key, value in d.items():
                    if key != "type" and hand.get(key) != value:
                        diff.append({"key": key, "from": hand.get(key), "to": value})
                        hand[key] = value
            elif kind == "event" and typ == "person_changed":
                # Someone else was chosen: their settings and gestures from here on.
                diff.append({"key": "person", "from": (d.get("previous") or {}).get("name"),
                             "to": (d.get("person") or {}).get("name")})
                for key, value in (d.get("settings") or {}).items():
                    if settings.get(key) != value:
                        diff.append({"key": key, "from": settings.get(key), "to": value})
                        settings[key] = value
                eff = d.get("effective") if isinstance(d.get("effective"), dict) else {}
                gestures = dict(eff.get("gestures") or ((d.get("personal") or {}).get("gestures") or {}))
                hand = {}
            elif kind == "reply" and typ == "settings" and isinstance(d.get("effective"), dict):
                changes.append({"t": e.get("t"), "event": k, "type": "effective", "kind": kind,
                                "effective": d["effective"]})
                continue
            if diff or (kind == "command" and typ in ("settings", "gestures_set", "hand_settings")):
                changes.append({"t": e.get("t"), "event": k, "type": typ, "kind": kind, "changes": diff})
        return jsonable({"meta": meta, "changes": changes, "settings": settings, "gestures": gestures}, 4)

    # -- one frame ----------------------------------------------------------------------
    def image_size(self) -> Optional[dict]:
        """The analysed camera frame's size (eye-mode landmarks are in its pixels), if it can be told.

        The video may be smaller than the frames MediaPipe saw, so it is
        measured where the frames carry normalised geometry (the camera
        overlay's face box, the calibration mesh, the followed face's box)
        next to the landmarks in pixels.
        """
        if self.mode != "eyes":
            return None
        with self._lock:
            cached = getattr(self, "_image", None)
            if cached is not None and cached[0] == self.rec.version:
                return cached[1]
        rec = self.rec
        size = None
        meta = rec.meta
        for key in ("camera", "frame", "image"):
            c = meta.get(key)
            if isinstance(c, dict) and _num(c.get("w")) > 0 and _num(c.get("h")) > 0:
                size = {"w": float(c["w"]), "h": float(c["h"]), "source": f"meta.{key}"}
                break
        if size is None and len(rec.landmark_ids()):
            # Each frame gives an estimate (the landmarks are stored as float16, so one
            # estimate can be a pixel off): the median of several.
            n = len(rec)
            ws, hs = [], []
            for k in range(0, n, max(1, n // 80)):
                f = rec.frame(k)
                m = f.get("msg") or {}
                box = None
                if isinstance(m.get("eyes"), dict) and isinstance(m["eyes"].get("box"), list):
                    box = m["eyes"]["box"]
                elif isinstance(m.get("you"), list):
                    box = m["you"]
                if not box or len(box) != 4:
                    continue
                lm = rec.landmarks(int(f.get("i", -1)))
                if lm is None:
                    continue
                P = lm["points"][:, :2].astype(float)
                bw, bh = float(box[2]) - float(box[0]), float(box[3]) - float(box[1])
                if bw <= 0.01 or bh <= 0.01:
                    continue
                ws.append((P[:, 0].max() - P[:, 0].min()) / bw)
                hs.append((P[:, 1].max() - P[:, 1].min()) / bh)
                if len(ws) >= 15:
                    break
            if ws:
                size = {"w": round(float(np.median(ws))), "h": round(float(np.median(hs))), "source": "face box"}
        with self._lock:
            self._image = (rec.version, size)
        return size

    def _configs(self) -> list[tuple[Optional[float], dict]]:
        """The session's configuration over time: meta.json's, then each person_changed event's
        ((time, {"effective", "detectors", "hand"}), the first with time None)."""
        meta = self.rec.meta
        out = [(None, meta)]
        for e in self.rec.events():
            d = e.get("data")
            if e.get("type") == "person_changed" and isinstance(d, dict) and math.isfinite(_num(e.get("t"))):
                out.append((float(e["t"]), d))
        return out

    def _config_at(self, t: float, key: str) -> dict:
        """``key`` ("effective", "detectors", ...) of the configuration in force at ``t`` ({} if unknown)."""
        found: dict = {}
        for when, conf in self._configs():
            if when is not None and math.isfinite(t) and when > t:
                break
            value = conf.get(key)
            found = value if isinstance(value, dict) else {}
        return found

    def pinch_thresholds(self, t: Optional[float] = None) -> dict:
        """Hand mode: the pinch thresholds in force at ``t``, and their history.

        HandGestureConfig's, overridden by the person's hand setup - as meta.json
        (and a person_changed event) records them - and then by each hand setup
        made during the recording.
        """
        def thresholds(conf: dict, on: float, off: float) -> tuple[float, float]:
            det = (conf.get("detectors") or {}).get("hand") if isinstance(conf.get("detectors"), dict) else None
            det = det if isinstance(det, dict) else {}
            for src in (det.get("config"), det.get("calibration"), conf.get("hand")):
                if isinstance(src, dict):
                    on = _num(src["pinch_on"]) if math.isfinite(_num(src.get("pinch_on"))) else on
                    off = _num(src["pinch_off"]) if math.isfinite(_num(src.get("pinch_off"))) else off
            return on, off

        on, off = thresholds(self.rec.meta, *DEFAULT_PINCH)
        history = [{"t": None, "on": on, "off": max(off, on + 0.05)}]
        for e in self.rec.events():
            d = e.get("data") if isinstance(e.get("data"), dict) else {}
            typ = e.get("type")
            if typ == "person_changed":
                on, off = thresholds(d, *DEFAULT_PINCH)
            elif typ == "hand_calibration_result" and d.get("ok") is not False:
                if math.isfinite(_num(d.get("pinch_on"))):
                    on = _num(d["pinch_on"])
                if math.isfinite(_num(d.get("pinch_off"))):
                    off = _num(d["pinch_off"])
            else:
                continue
            history.append({"t": e.get("t"), "on": on, "off": max(off, on + 0.05)})
        now = history[0]
        for h_ in history[1:]:
            if t is not None and h_["t"] is not None and h_["t"] <= t:
                now = h_
        return {"on": now["on"], "off": now["off"], "history": history}

    def detail(self, n: int) -> dict:
        """Everything about frame ``n`` for the panels that follow the current frame."""
        rec = self.rec
        f = rec.frame(n)
        t = _num(f.get("t"))
        fid = f.get("i")
        out: dict[str, Any] = {"n": n, "frame": f, "video": None, "landmarks": None}
        internals_json = None
        if isinstance(fid, int):
            out["video"] = rec.video_frame(fid)
            lm = rec.landmarks(fid)
            if lm is not None:
                out["landmarks"] = {"points": np.round(lm["points"], 2 if self.mode == "eyes" else 5),
                                    "others": lm.get("others"), "chunk": lm["chunk"]}
        if self.mode == "eyes":
            out["model"] = None
            snap = rec.model_at(t) if math.isfinite(t) else None
            vec = f.get("features")
            if snap is not None:
                out["model"] = snap.info()
            if snap is not None and isinstance(vec, list) and len(vec) == len(FEATURE_NAMES):
                ex = internals.explain(snap.net, vec)
                net = f.get("net", (f.get("msg") or {}).get("net"))
                raw = (f.get("msg") or {}).get("raw")
                ex["led"] = net if net in ex["networks"] else None
                if ex["led"] and isinstance(raw, list) and len(raw) == 2:
                    final = ex["networks"][ex["led"]]["final"]
                    delta = np.asarray(raw, float) - final
                    winking = (f.get("wink") or {}).get("winking", (f.get("msg") or {}).get("winking"))
                    why = None
                    if self._shifted(f, snap.net):
                        why = "wink" if winking else "glare" if (f.get("msg") or {}).get("glare") else "switch"
                    ex["check"] = {"recorded": raw, "computed": final, "delta": delta,
                                   "distance": float(np.hypot(*delta)), "winking": winking, "shifted": why}
                # Significant digits, not decimals: a path that adds 1e-7 px must not read as 0.
                internals_json = jsonable(self._compact(ex), sig=7)
            out["blink"] = self._blink_view(f, t)
            out["wink"] = self._wink_view(f, t)
        else:
            pinch = self.pinch_thresholds(t)
            out["pinch"] = {"on": pinch["on"], "off": pinch["off"]}
            if out["landmarks"] is not None:
                cfg = (self._config_at(t, "detectors").get("hand") or {}).get("config") or {}
                ratio = _num(cfg.get("extend_ratio"))         # HandGestureConfig's, when recorded
                out["hand"] = internals.hand_trace(out["landmarks"]["points"], pinch["on"], pinch["off"],
                                                   ratio if math.isfinite(ratio) else 1.15)
        out["recent"] = self._recent_events(t)
        out = jsonable(out, 5)
        if internals_json is not None:
            out["internals"] = internals_json
        return out

    @staticmethod
    def _compact(ex: dict) -> dict:
        """The internals the page draws (the per-member input shares are averaged over the members)."""
        for net in ex["networks"].values():
            members = net["members"]
            net["skip_by_input_px"] = np.mean([m.pop("skip_by_input_px") for m in members], axis=0)
        return ex

    def _blink_view(self, f: dict, t: float) -> dict:
        """The blink detector's state, with its deep level (BlinkDetector.deep_level) worked out."""
        b = f.get("blink") or {}
        m = f.get("msg") or {}
        close, open_ = _num(b.get("close_thr")), _num(b.get("open_thr"))
        if not math.isfinite(close):
            close, open_ = _pair(m.get("thr"), 0), _pair(m.get("thr"), 1)
        out = {"closing": b.get("closing", m.get("closing")), "closed": b.get("closed", m.get("closed")),
               "deep": b.get("deep"), "close_thr": close, "open_thr": open_, "signal": _num(b.get("signal")),
               "pending": b.get("pending"), "cl": _num(m.get("cl")), "cr": _num(m.get("cr")),
               "watches": self._config_at(t, "detectors").get("blink_signal")
               or self._config_at(t, "effective").get("blink_signal")}
        base = _num(b.get("baseline"))
        out["baseline_recorded"] = math.isfinite(base)
        if not math.isfinite(base) and math.isfinite(close) and math.isfinite(open_):
            base = 2.0 * open_ - close             # open_thr is halfway between the baseline and close_thr
        if math.isfinite(base):
            cfg = self._config_at(t, "detectors").get("blink") or {}
            rise = _num(cfg.get("deep_rise", self._config_at(t, "effective").get("deep_rise")))
            rise = rise if math.isfinite(rise) and rise > 0 else 0.6 * (1.0 - base)
            out["baseline"] = base
            out["deep_thr"] = min(0.97, max(base + max(rise, 0.2), 0.5))
        return out

    def _wink_view(self, f: dict, t: float) -> dict:
        """The wink detector's state; per eye ([left, right]) its baseline and levels when recorded."""
        w = f.get("wink") or {}
        m = f.get("msg") or {}
        winking, pressed = w.get("winking", m.get("winking")), w.get("pressed", m.get("wink"))
        state = w.get("state")
        if state not in WINK_STATES:
            state = "active" if pressed else "candidate" if winking else "idle"
        cfg = self._config_at(t, "detectors").get("wink") or {}
        asym = []
        for eye in ("left", "right"):
            own = _num((cfg.get(eye) or {}).get("asym")) if isinstance(cfg.get(eye), dict) else math.nan
            asym.append(own if math.isfinite(own) else _num(cfg.get("asym_min")))
        return {"state": state, "eye": pressed or winking, "cl": _num(m.get("cl")), "cr": _num(m.get("cr")),
                "base": w.get("base"), "close_thr": w.get("close_thr"), "open_thr": w.get("open_thr"),
                "asym": asym, "hold_ms": _num(cfg.get("hold_ms"))}

    def _recent_events(self, t: float, before: float = 1.5, after: float = 0.1) -> list[dict]:
        """Events close to time ``t`` (for the state machines and the screen map)."""
        if not math.isfinite(t):
            return []
        out = []
        for k, e in enumerate(self.rec.events()):
            et = _num(e.get("t"))
            if t - before <= et <= t + after and e.get("kind") in ("event", "command"):
                d = e.get("data") if isinstance(e.get("data"), dict) else {}
                out.append({"t": et, "event": k, "kind": e.get("kind"), "type": e.get("type"),
                            "data": {key: d.get(key) for key in ("n", "at", "eye", "pre_frame", "frame", "dy",
                                                                   "duration_ms", "target", "kind", "hand")
                                     if key in d}})
        return out

    # -- events -------------------------------------------------------------------------
    def events(self) -> list[dict]:
        out = []
        for k, e in enumerate(self.rec.events()):
            d = e.get("data")
            out.append({"n": k, "t": e.get("t"), "wall": e.get("wall"), "kind": e.get("kind"), "type": e.get("type"),
                        "data": d})
        return jsonable(out, 4)
