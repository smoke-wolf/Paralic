"""Make demo session recordings (format 1) with the real pipeline, without a camera.

    python tools/make_demo_recording.py                 # an eye-mode and a hand-mode demo in data/recordings/
    python tools/make_demo_recording.py --quick --mode eyes --out /tmp/recordings

Recordings are what the Paralic Inspector (``python -m paralic.inspector``)
replays; this makes them without a camera, for trying it and for its tests.
The directories are written the way ``paralic/recorder.py`` writes them
(``docs/recording-format.md``), from a session driven the way the page drives it.

**Eye mode.** A real :class:`~paralic.session.TrackerSession` runs the real
MediaPipe face model on the test portrait (``tests/face_images.py``), framed
like a webcam picture and moved a little from frame to frame. For blinks and
winks its eyes are painted closed, so the closures, the blink and wink
detectors and their events are real. The gaze features come from the
simulated person of ``tests/synthetic.py`` (``VirtualUser``) looking at a
scripted sequence of targets - calibration dots (a real
``fit_full_calibration`` trains the network the recording then uses), a
validation, browsing with blinks, double blinks, a held wink, settings
changes, a quick adjust, a wink test - because a photo cannot move its eyes.
The face geometry and the head pose in the frame messages are MediaPipe's;
the 28 features the network reads are the simulated person's.

Some stretches have glasses painted on (with a reflection on one lens), a
second face pasted into the picture, or the face leaving the picture. Where
the session does not report these itself, this script fills in their frame
fields (``glasses``, ``glare``, ``glare_score``, ``faces``, ``you``,
``others``) and the ``glasses_changed`` event as a session that notices them
would.

**Hand mode.** A hand-mode session reads the 21-point hands of
``tests/fakes.py`` (``make_hand``): the hand setup, pointing, pinch clicks, a
pinch-drag scroll and the open-hand pause. The camera image is the hand drawn
from its landmarks.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import paralic.session as session_mod  # noqa: E402
from paralic import __version__  # noqa: E402
from paralic import landmarks as L  # noqa: E402
from paralic.features import EYE_INPUTS  # noqa: E402
from paralic.recording import CHUNK_FRAMES, FORMAT  # noqa: E402
from paralic.session import TrackerSession, pack_frame  # noqa: E402
from paralic.users import UserStore  # noqa: E402
from tests import face_images  # noqa: E402
from tests.fakes import FakeHandTracker, make_hand  # noqa: E402
from tests.synthetic import SCREEN_H, SCREEN_W, Head, VirtualUser  # noqa: E402

FPS = 30.0
SCREEN = {"w": SCREEN_W, "h": SCREEN_H, "dpr": 1}
MODEL = ROOT / "models" / "face_landmarker.task"

# The page's dots (web/js/calibration.js): the 5x4 grid in snake order after
# the centre, the validation dots and the quick adjust's dots.
FULL_POINTS = [(0.5, 0.5)] + [(x, y) for row, y in enumerate((0.08, 0.37, 0.63, 0.92))
                              for x in ((0.06, 0.28, 0.5, 0.72, 0.94) if row % 2 == 0
                                        else (0.94, 0.72, 0.5, 0.28, 0.06))]
QUICK_POINTS = [(0.5, 0.5)] + [(x, y) for row, y in enumerate((0.08, 0.5, 0.92))
                               for x in ((0.07, 0.36, 0.64, 0.93) if row % 2 == 0 else (0.93, 0.64, 0.36, 0.07))]
VALIDATION_POINTS = [(0.5, 0.55), (0.27, 0.28), (0.73, 0.28), (0.73, 0.8), (0.27, 0.8)]
ADJUST_POINTS = [(0.5, 0.5), (0.1, 0.12), (0.5, 0.12), (0.9, 0.12), (0.9, 0.5),
                 (0.9, 0.88), (0.5, 0.88), (0.1, 0.88), (0.1, 0.5)]
# Head poses during the calibration's head step: (yaw, pitch, roll) in radians.
HEAD_POSES = [(0.12, 0.0, 0.0), (-0.12, 0.0, 0.0), (0.0, -0.09, 0.0), (0.0, 0.09, 0.0),
              (0.0, 0.0, 0.06), (0.0, 0.0, -0.06)]
# Places on the page the eyes visit while browsing (screen fractions).
BROWSE_TARGETS = [(0.08, 0.06), (0.25, 0.06), (0.42, 0.06), (0.6, 0.06), (0.3, 0.35), (0.62, 0.32),
                  (0.45, 0.55), (0.2, 0.7), (0.7, 0.72), (0.96, 0.3), (0.96, 0.75), (0.5, 0.88)]
# A person who did "Test my winks": weak but steady winks, thresholds personalised.
WINK_PROFILE = {eye: {"ok": True, "reason": None, "baseline": 0.3, "noise": 0.01, "rise": 0.12, "other_rise": 0.0,
                      "asym": 0.12, "hold_fraction": 0.95, "threshold_rise": 0.08, "threshold_asym": 0.07,
                      "n_frames": 40} for eye in ("left", "right")}


# ---------------------------------------------------------------------------
# Writing a recording directory, as paralic/recorder.py does
# ---------------------------------------------------------------------------

# A model snapshot is named after the model history's source or the command
# that changed the model (as recorder.py names them).
SNAPSHOT_NAMES = {"calibration_fit": "calibration", "calibration": "refit", "fine-tune": "finetune"}
OTHER_FACES = 3                         # boxes of other faces kept per frame (landmarks' "others")
MAX_BYTES = 3_000_000_000               # the recorder's limit for camera images (meta.json "video")
DATA_URL_CHARS = 256                    # longer data: URLs (pictures) in messages are not recorded


def _plain(v):
    """JSON for NumPy values, tuples and dataclasses too."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (set, frozenset, tuple)):
        return list(v)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return dataclasses.asdict(v)
    raise TypeError(f"{type(v).__name__} is not JSON serialisable")


def _finite(v):
    """``v`` with NaN and infinities as None (strict JSON has no such numbers)."""
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, np.ndarray):
        return _finite(v.tolist())
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    return v


def _dumps(doc, indent: Optional[int] = None) -> str:
    """Strict JSON, compact unless ``indent``."""
    separators = (",", ":") if indent is None else (",", ": ")
    try:
        return json.dumps(doc, separators=separators, indent=indent, default=_plain, allow_nan=False)
    except ValueError:
        return json.dumps(_finite(doc), separators=separators, indent=indent, default=_plain, allow_nan=False)


def _scrub(v):
    """``v`` without the pictures embedded in it as data: URLs."""
    if isinstance(v, str):
        if v.startswith("data:") and len(v) > DATA_URL_CHARS:
            return f"{v[:v.find(',') + 1]}... ({len(v)} characters, not recorded)"
        return v
    if isinstance(v, dict):
        return {k: _scrub(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_scrub(x) for x in v]
    return v


def _slug(name) -> str:
    """A person's name as a part of a directory name."""
    out, gap = [], False
    for ch in str(name or ""):
        if ch.isalnum():
            out.append(ch)
            gap = False
        elif not gap and out:
            out.append("_")
            gap = True
    return "".join(out).strip("_")[:32] or "person"


def _write_atomic(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=path.suffix)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _vector(v) -> Optional[list]:
    if v is None:
        return None
    return [round(x, 6) if math.isfinite(x) else None for x in np.asarray(v, float).ravel().tolist()]


def _other_faces(others) -> Optional[np.ndarray]:
    """The frame message's other faces as an (OTHER_FACES, 4) array, NaN where there are fewer."""
    if not isinstance(others, (list, tuple)):
        return None
    out = np.full((OTHER_FACES, 4), np.nan, np.float16)
    for k, item in enumerate(others[:OTHER_FACES]):
        out[k] = np.asarray(item.get("box") if isinstance(item, dict) else item, float).reshape(4)
    return out


def _blink_state(detector, face: bool) -> dict:
    s = detector.state()
    return {"closing": s.closing, "closed": s.closed, "deep": s.deep,
            "close_thr": round(s.close_threshold, 4), "open_thr": round(s.open_threshold, 4),
            "signal": round(detector._closure, 4) if face else None,
            "baseline": round(s.baseline, 4), "pending": s.pending}


def _wink_state(detector) -> dict:
    eyes = ("left", "right")
    thr = [detector.thresholds(e) for e in eyes]
    return {"winking": detector.winking, "pressed": detector.pressed, "state": detector.state,
            "base": [round(detector.baseline(e), 4) for e in eyes],
            "close_thr": [round(t[0], 4) for t in thr], "open_thr": [round(t[1], 4) for t in thr]}


def _session_config(s) -> dict:
    """The session's configuration as meta.json holds it."""
    if s.hand is not None:
        hand = s.hand
        detectors = {"hand": {"config": dataclasses.asdict(hand.recognizer.config),
                              "calibration": hand.calibration.to_dict() if hand.calibration else None,
                              "smoothing": dataclasses.asdict(hand.filter.params)}}
    else:
        st = s.stabilizer
        detectors = {"blink": dataclasses.asdict(s.blink.config), "blink_signal": s.blink_mode,
                     "wink": dataclasses.asdict(s.wink.config),
                     "smoothing": {**dataclasses.asdict(st.filter.params), "rewind_s": st.rewind_s,
                                   "settle_s": st.settle_s, "max_freeze_s": st.max_freeze_s}}
    user = s.user or {}
    return {"mode": s.mode, "person": {"id": user.get("id"), "name": user.get("name")}, "screen": s.screen,
            "settings": dict(s.settings), "effective": s.effective(), "personal": s.personal,
            "profile": s.profiles.summary(), "hand": s.hand.view() if s.hand is not None else None,
            "detectors": detectors}


class RecordingWriter:
    """Writes one recording directory the way paralic/recorder.py does (docs/recording-format.md)."""

    def __init__(self, root: Path, config: dict, started_t: float, started_wall: float, video: dict):
        base = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(started_wall))}-" \
               f"{_slug(config['person']['name'])}-{config['mode']}"
        root.mkdir(parents=True, exist_ok=True)
        for k in range(1, 100):
            folder = root / (base if k == 1 else f"{base}-{k}")
            try:
                folder.mkdir()
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError(f"Too many recordings named {base} in {root}")
        self.folder, self.id = folder, folder.name
        for sub in ("landmarks", "models", "video"):
            (folder / sub).mkdir()
        self.video = dict(video)
        self.head = {"format": FORMAT, "paralic": __version__, "id": self.id,
                     "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_wall))}
        self.config = {"clock": {"t": round(started_t, 4), "wall": round(started_wall, 3)},
                       **json.loads(_dumps(config))}
        self.frames = self.video_frames = self.events = self.models = 0
        self._n = 0                             # frames seen (for video "every")
        self._chunk: list[tuple] = []
        self._chunks = 0
        self._lock = threading.Lock()           # background jobs record pushes and models too
        self.write_meta(ended=None, final=False)
        self.frames_fh = open(folder / "frames.jsonl", "w", encoding="utf-8")
        self.events_fh = open(folder / "events.jsonl", "w", encoding="utf-8")

    def write_meta(self, ended: Optional[str], final: bool) -> None:
        size = sum(f.stat().st_size for f in self.folder.rglob("*") if f.is_file() and f.name != "meta.json")
        meta = {**self.head, "ended": ended, "complete": final, **self.config,
                "video": {**self.video, "max_bytes": MAX_BYTES, "stopped": None},
                "summary": {"frames": self.frames, "video_frames": self.video_frames, "events": self.events,
                            "models": self.models, "dropped": 0, "bytes": size},
                "error": None}
        meta["summary"]["bytes"] = size + len(_dumps(meta, indent=1))
        _write_atomic(self.folder / "meta.json", _dumps(meta, indent=1).encode("utf-8"))

    def frame(self, line: dict, image: Optional[tuple], points, others: Optional[np.ndarray]) -> None:
        """One frame line (without "video"), its camera image (the JPEG as sent, and the
        picture) and its tracked points."""
        n, self._n = self._n, self._n + 1
        fid = line["i"]
        every = self.video["every"]
        video = image is not None and every > 0 and n % every == 0 and isinstance(fid, int) and fid >= 0
        if video:
            self._write_video(fid, *image)
        if points is not None:
            pts = np.asarray(points, np.float16)
            if self._chunk and self._chunk[0][1].shape != pts.shape:
                self._write_chunk()
            self._chunk.append((fid, pts, others))
            if len(self._chunk) >= CHUNK_FRAMES:
                self._write_chunk()
        text = _dumps(line)
        with self._lock:
            self.frames_fh.write(f'{text[:-1]},"video":{"true" if video else "false"}}}\n')
            self.frames += 1

    def _write_video(self, fid: int, jpeg: bytes, bgr: np.ndarray) -> None:
        """Images up to max_width as the page sent them; wider ones shrunk (with "quality")."""
        import cv2

        h, w = bgr.shape[:2]
        data = jpeg
        if w > self.video["max_width"]:
            mw = self.video["max_width"]
            small = cv2.resize(bgr, (mw, max(1, round(h * mw / w))), interpolation=cv2.INTER_AREA)
            data = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(self.video["quality"])])[1].tobytes()
        (self.folder / "video" / f"{fid}.jpg").write_bytes(data)
        self.video_frames += 1

    def _write_chunk(self) -> None:
        chunk, self._chunk = self._chunk, []
        if not chunk:
            return
        arrays = {"ids": np.array([fid if isinstance(fid, int) else -1 for fid, _, _ in chunk], np.int64),
                  "points": np.stack([p for _, p, _ in chunk])}
        if any(o is not None for _, _, o in chunk):
            none = np.full((OTHER_FACES, 4), np.nan, np.float16)
            arrays["others"] = np.stack([none if o is None else o for _, _, o in chunk])
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        _write_atomic(self.folder / "landmarks" / f"{self._chunks:06d}.npz", buf.getvalue())
        self._chunks += 1

    def event(self, t: float, wall: float, kind: str, type_, data) -> None:
        line = _dumps({"t": round(t, 4), "wall": round(wall, 3), "kind": kind, "type": type_, "data": data})
        with self._lock:
            self.events_fh.write(line + "\n")
            self.events += 1

    def model(self, why: str, model, t: float, wall: float) -> None:
        """A snapshot of the gaze model, and the event that says from when it is used."""
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in why)[:40] or "change"
        with self._lock:
            k, self.models = self.models, self.models + 1
        name = f"models/{k:03d}-{safe}.json"
        path = self.folder / name
        path.write_text(_dumps(model.to_dict()), encoding="utf-8")
        os.utime(path, (wall, wall))           # written when the model changed
        self.event(t, wall, "event", "model", {"why": why, "file": name, "version": model.meta.get("version")})

    def close(self, ended: str) -> None:
        self._write_chunk()
        self.frames_fh.close()
        self.events_fh.close()
        self.write_meta(ended=ended, final=True)


# ---------------------------------------------------------------------------
# The session driver (commands, frames, events, models)
# ---------------------------------------------------------------------------

class Driver:
    """Runs a TrackerSession like the page would, and records it like the recorder would:
    the page says hello, then recording is switched on (the ● Rec button)."""

    def __init__(self, root: Path, name: str, mode: str, tracker, start_wall: float, *, video: dict,
                 personal: Optional[dict] = None, settings: Optional[dict] = None):
        self.tmp = tempfile.TemporaryDirectory(prefix="paralic-demo-")
        self.users = UserStore(Path(self.tmp.name))
        user = self.users.create(name)
        if personal:
            self.users.save_personal(user["id"], personal)
        self.t = 3617.25                       # the session clock (monotonic seconds)
        self.t0 = self.t
        self.wall0 = start_wall
        self.writer: Optional[RecordingWriter] = None
        self.session = TrackerSession(lambda: tracker, self.users, clock=lambda: self.t,
                                      wall=lambda: self.wall0 + (self.t - self.t0), push=self._pushed, mode=mode)
        self.frame_id = 0
        self.last_events: list[dict] = []
        self.event_log: list[dict] = []
        s = self.session
        s.handle_command({"type": "hello", "screen": dict(SCREEN), "settings": settings or {}})
        self.writer = RecordingWriter(root, _session_config(s), self.t, self.wall, video)
        self.folder = self.writer.folder
        # A snapshot whenever the session gives its model a new version, and after each command.
        self._model_key: tuple = (None, None)
        record_model = s._record_model

        def recorded(source: str, **extra) -> None:
            record_model(source, **extra)
            self._snapshot(source)

        s._record_model = recorded
        self._snapshot("start")
        cmd = {"type": "recording", "on": True, "video": dict(video)}
        self.writer.event(self.t, self.wall, "command", "recording", cmd)
        self.writer.event(self.t, self.wall, "reply", "recording", {
            "type": "recording", "ok": True, "on": True, "id": self.writer.id, "dir": str(self.folder.absolute()),
            "frames": 0, "video_frames": 0, "events": 1, "bytes": 0, "seconds": 0.0, "video": dict(video),
            "video_stopped": None, "error": None})

    @property
    def wall(self) -> float:
        return self.wall0 + (self.t - self.t0)

    def _pushed(self, msg: dict) -> None:
        if self.writer is not None:
            self.writer.event(self.t, self.wall, "push", msg.get("type"), _scrub(msg))

    def _snapshot(self, why: str) -> None:
        """Save the session's model if it changed since the last snapshot (Recorder.model)."""
        model = self.session.model
        version = None if model is None else model.meta.get("version")
        last, last_version = self._model_key
        if model is last and version == last_version:
            return
        self._model_key = (model, version)
        why = SNAPSHOT_NAMES.get(why, why)
        if model is not None:
            self.writer.model(why, model, self.t, self.wall)
        elif last is not None:                 # no model any more
            self.writer.event(self.t, self.wall, "event", "model", {"why": why, "file": None, "version": None})

    def command(self, cmd: dict) -> list[dict]:
        self.writer.event(self.t, self.wall, "command", cmd.get("type"), _scrub(cmd))
        replies = self.session.handle_command(cmd)
        self._snapshot(str(cmd.get("type")))
        for r in replies:
            self.writer.event(self.t, self.wall, "reply", r.get("type"), _scrub(r))
        self.t += 0.004                        # a command takes a moment between two frames
        return replies

    def wait_for_job(self) -> None:
        self.session.wait_for_job(120.0)

    def frame(self, jpeg: bytes, *, image_bgr: Optional[np.ndarray], label: Optional[dict] = None,
              points_of=None, features: Optional[np.ndarray] = None, extra=None, header: Optional[dict] = None,
              extra_events: Optional[list] = None) -> dict:
        """One camera frame through the session; returns the frame message.

        ``extra(msg)`` returns fields the session does not produce (yet); they
        are added to the frame message unless the session produced them
        itself, and so are ``extra_events`` to its events.
        """
        self.frame_id += 1
        h = {"id": self.frame_id, **(header or {})}
        if label:
            h["label"] = label
        out = self.session.handle_frame(pack_frame(h, jpeg))
        msg, events = out[0], list(out[1:])
        for k, v in ((extra(msg) if extra is not None else None) or {}).items():
            msg.setdefault(k, v)
        for e in extra_events or []:
            if not any(x.get("type") == e.get("type") for x in events):
                events.append(e)
        s = self.session
        eyes = s.hand is None
        face = bool(msg.get("face"))
        line = {"i": msg.get("id", self.frame_id), "t": round(self.t, 4), "wall": round(self.wall, 3),
                # The calibration's copy of the face mesh is in the landmarks.
                "msg": {**msg, "mesh": True} if "mesh" in msg else msg,
                "features": _vector(features) if eyes and face else None,
                "blink": _blink_state(s.blink, face) if eyes else None,
                "wink": _wink_state(s.wink) if eyes else None,
                "net": msg.get("net"), "label": h.get("label"), "stored": bool(msg.get("labeled")),
                "events": [m.get("type") for m in events]}
        if h.get("gesture"):
            line["gesture"] = h["gesture"]
        points = points_of() if points_of is not None and face else None
        others = _other_faces(msg.get("others")) if points is not None and eyes else None
        self.writer.frame(line, None if image_bgr is None else (jpeg, image_bgr), points, others)
        for m in events:
            self.writer.event(self.t, self.wall, "event", m.get("type"), m)
        self.last_events = events
        self.event_log.extend(events)
        self.t += 1.0 / FPS
        return msg

    def finish(self) -> Path:
        self.session.close()
        self.writer.close(time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.wall)))
        self.tmp.cleanup()
        return self.folder


# ---------------------------------------------------------------------------
# Eye mode: the test portrait as a webcam picture
# ---------------------------------------------------------------------------

def _paint_glasses(img: np.ndarray, pts: np.ndarray, glare: bool) -> np.ndarray:
    """Dark frames over the eyes (and a reflection on the left lens)."""
    import cv2

    out = img.copy()
    lenses = []
    for a, b in ((L.RIGHT_EYE_OUTER, L.RIGHT_EYE_INNER), (L.LEFT_EYE_INNER, L.LEFT_EYE_OUTER)):
        p, q = pts[a], pts[b]
        c = (p + q) / 2 + [0, 0.05 * np.hypot(*(q - p))]
        ew = float(np.hypot(*(q - p)))
        angle = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
        lenses.append((c, ew, angle))
    thick = max(3, int(round(lenses[0][1] * 0.09)))
    for c, ew, angle in lenses:
        cv2.ellipse(out, (int(c[0]), int(c[1])), (int(0.92 * ew), int(0.62 * ew)), angle, 0, 360, (28, 26, 30),
                    thick, cv2.LINE_AA)
    (c1, e1, _), (c2, e2, _) = lenses
    y = int(min(c1[1], c2[1]) - 0.25 * e1)
    cv2.line(out, (int(c1[0] + 0.9 * e1), y), (int(c2[0] - 0.9 * e2), y), (28, 26, 30), thick, cv2.LINE_AA)
    cv2.line(out, (int(c1[0] - 0.92 * e1), int(c1[1])), (int(c1[0] - 1.6 * e1), int(c1[1] - 0.1 * e1)),
             (28, 26, 30), thick, cv2.LINE_AA)
    cv2.line(out, (int(c2[0] + 0.92 * e2), int(c2[1])), (int(c2[0] + 1.6 * e2), int(c2[1] - 0.1 * e2)),
             (28, 26, 30), thick, cv2.LINE_AA)
    if glare:                                   # the person's left eye: on the right of the picture
        c, ew, _ = lenses[1]
        blob = np.zeros(out.shape[:2], np.float32)
        cv2.ellipse(blob, (int(c[0] + 0.25 * ew), int(c[1] - 0.15 * ew)), (int(0.45 * ew), int(0.3 * ew)), -20,
                    0, 360, 1.0, -1, cv2.LINE_AA)
        blob = cv2.GaussianBlur(blob, (0, 0), 0.12 * ew)[..., None]
        out = (out * (1 - 0.9 * blob) + 250 * 0.9 * blob).astype(np.uint8)
    return out


class PortraitScene:
    """The eye-mode demo's camera picture: the test portrait framed like a webcam, moving a little."""

    def __init__(self, tracker_factory, width: int = 960):
        import cv2

        self.w, self.h = width, int(round(width * 9 / 16))
        src = face_images.portrait(960)
        if src is None:
            raise SystemExit("The test portrait is not available (offline?) - see tests/face_images.py")
        # The head reaches the top of the photo: continue the background above it.
        pad = 260
        top = src[:30].astype(np.float32).mean(axis=0)
        a, b = int(0.3 * src.shape[1]), int(0.7 * src.shape[1])
        for col in range(a, b):
            top[col] = top[a] + (top[b] - top[a]) * (col - a) / (b - a)
        band = cv2.GaussianBlur(np.repeat(top[None], pad, axis=0), (0, 0), 8)
        src = np.vstack([band.astype(np.uint8), src])
        tracker = tracker_factory()
        try:
            rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            obs = None
            for i in range(5):
                obs = tracker.process(rgb, i * 33)
        finally:
            tracker.close()
        if obs is None:
            raise SystemExit("MediaPipe found no face in the test portrait")
        pts = obs.points_px[:, :2]
        self.pts = pts
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        self.centre = ((x0 + x1) / 2, (y0 + y1) / 2)
        self.scale = 0.36 * self.h / (y1 - y0)
        both = (L.RIGHT_EYE_CONTOUR, L.LEFT_EYE_CONTOUR)
        self.src = {"open": src, "closed": face_images.close_eyes(src, pts, both),
                    "left": face_images.close_eyes(src, pts, (L.LEFT_EYE_CONTOUR,)),
                    "right": face_images.close_eyes(src, pts, (L.RIGHT_EYE_CONTOUR,))}
        self._glasses: dict = {}
        # A second, smaller face for the stretch with someone else in view.
        hx0, hy0 = int(x0 - 0.25 * (x1 - x0)), int(y0 - 0.35 * (y1 - y0))
        hx1, hy1 = int(x1 + 0.25 * (x1 - x0)), int(y1 + 0.15 * (y1 - y0))
        head = src[max(0, hy0):hy1, max(0, hx0):hx1]
        size = (int(head.shape[1] * 0.42 * self.scale), int(head.shape[0] * 0.42 * self.scale))
        self.second = cv2.resize(head, size, interpolation=cv2.INTER_AREA)
        mask = np.zeros(self.second.shape[:2], np.float32)
        cv2.ellipse(mask, (size[0] // 2, size[1] // 2), (int(size[0] * 0.42), int(size[1] * 0.47)), 0, 0, 360,
                    1.0, -1)
        self.second_mask = cv2.GaussianBlur(mask, (0, 0), 4)[..., None]
        fx0, fy0 = (x0 - max(0, hx0)) * size[0] / head.shape[1], (y0 - max(0, hy0)) * size[1] / head.shape[0]
        fx1, fy1 = (x1 - max(0, hx0)) * size[0] / head.shape[1], (y1 - max(0, hy0)) * size[1] / head.shape[0]
        self.second_face = (fx0, fy0, fx1, fy1)       # the face's box inside the pasted picture

    def _variant(self, eyes: str, glasses: bool, glare: bool) -> np.ndarray:
        if not glasses:
            return self.src[eyes]
        key = (eyes, glare)
        if key not in self._glasses:
            self._glasses[key] = _paint_glasses(self.src[eyes], self.pts, glare)
        return self._glasses[key]

    def render(self, t: float, *, eyes: str = "open", blend: float = 1.0, glasses: bool = False,
               glare: bool = False, second: bool = False, away: bool = False,
               pose: tuple = (0.0, 0.0, 0.0)) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """The picture at time ``t`` and the boxes (image fractions) of other faces in it."""
        import cv2

        img = self._variant(eyes, glasses, glare)
        if eyes != "open" and blend < 1.0:
            img = cv2.addWeighted(self._variant("open", glasses, glare), 1.0 - blend, img, blend, 0)
        yaw, pitch, roll = pose
        dx = 9.0 * math.sin(0.7 * t) + 160.0 * yaw
        dy = 6.0 * math.sin(0.5 * t + 1.0) - 120.0 * pitch
        rot = 1.5 * math.sin(0.4 * t) + math.degrees(roll)
        zoom = 1.0 + 0.025 * math.sin(0.3 * t)
        M = cv2.getRotationMatrix2D(self.centre, rot, self.scale * zoom)
        M[0, 2] += self.w / 2 + dx - self.centre[0]
        M[1, 2] += 0.42 * self.h + dy - self.centre[1]
        frame = cv2.warpAffine(img, M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        if away:
            # Turned away from the camera: the back of the head instead of the face.
            P = self.pts @ M[:, :2].T + M[:, 2]
            (x0, y0), (x1, y1) = P.min(axis=0), P.max(axis=0)
            hair = np.zeros(frame.shape[:2], np.float32)
            cv2.ellipse(hair, (int((x0 + x1) / 2), int((y0 + y1) / 2 - 0.08 * (y1 - y0))),
                        (int(0.62 * (x1 - x0)), int(0.66 * (y1 - y0))), rot, 0, 360, 1.0, -1, cv2.LINE_AA)
            hair = cv2.GaussianBlur(hair, (0, 0), 3)[..., None]
            texture = 26 + 8 * np.sin(np.arange(frame.shape[1])[None, :, None] * 0.9)
            frame = (frame * (1 - hair) + texture * hair).astype(np.uint8)
        others = None
        if second and not away:
            sh, sw = self.second.shape[:2]
            x, y = int(0.86 * self.w - sw / 2), int(0.30 * self.h - sh / 2)
            roi = frame[y:y + sh, x:x + sw].astype(np.float32)
            frame[y:y + sh, x:x + sw] = (roi * (1 - self.second_mask) + self.second * self.second_mask).astype(np.uint8)
            fx0, fy0, fx1, fy1 = self.second_face
            others = np.array([[(x + fx0) / self.w, (y + fy0) / self.h, (x + fx1) / self.w, (y + fy1) / self.h]])
        return frame, others


@dataclasses.dataclass
class Look:
    """What the person does in one frame of the script."""

    target: tuple                           # where the eyes go (screen px)
    eyes: str = "open"                      # open | closed | left | right
    blend: float = 1.0                      # how far the eyes are shut (painted)
    label: Optional[dict] = None
    pose: tuple = (0.0, 0.0, 0.0)           # head yaw, pitch, roll (radians)
    glasses: bool = False
    glare: bool = False
    second: bool = False
    away: bool = False
    header: Optional[dict] = None


class EyeDemo:
    def __init__(self, root: Path, *, quick: bool, seed: int, width: int, start_wall: float, name: str = "Sam",
                 video_width: int = 640):
        import cv2

        from paralic.tracker import FaceTracker

        if not MODEL.exists():
            raise SystemExit(f"The face model is missing: {MODEL} (run python -m paralic once to download it)")
        model_bytes = MODEL.read_bytes()
        self.cv2 = cv2
        self.quick = quick
        self.scene = PortraitScene(lambda: FaceTracker(model_bytes), width)
        self.user = VirtualUser(seed=seed)
        self.rng = np.random.default_rng(seed + 1)
        self.tap = _TapTracker(FaceTracker(model_bytes))
        self.driver = Driver(root, name, "eyes", self.tap, start_wall,
                             video={"every": 1, "max_width": video_width, "quality": 70},
                             personal={"wink": WINK_PROFILE},
                             settings={"smoothing": "auto", "blink_sensitivity": "high", "double_blink": "personal",
                                       "learning": True})
        self.eye_pos = np.array([SCREEN_W / 2, SCREEN_H / 2])
        self.goal = self.eye_pos.copy()
        self.move_at = 0
        self.frame_no = 0
        self.vector: Optional[np.ndarray] = None
        self.glasses_seen = 0.0                 # how long the glasses have been on (for the simulated detector)
        self.glasses_state = False
        self.glare_score = np.zeros(2)
        self.pending_glasses_event: Optional[float] = None

    # -- the simulated eyes ------------------------------------------------------------
    def _features(self, look: Look) -> np.ndarray:
        goal = np.asarray(look.target, float)
        if not np.allclose(goal, self.goal):
            self.goal = goal
            self.move_at = self.frame_no + 5            # ~170 ms before the eyes jump
        if self.frame_no >= self.move_at:
            self.eye_pos = self.eye_pos + 0.62 * (self.goal - self.eye_pos)
            if np.hypot(*(self.goal - self.eye_pos)) < 2.0:
                self.eye_pos = self.goal.copy()
        t = self.driver.t - self.driver.t0
        yaw, pitch, roll = look.pose
        head = Head(x=0.6 * math.sin(0.7 * t), y=9.0 + 0.4 * math.sin(0.5 * t + 1.0), dist=60.0 + 0.8 * math.sin(0.3 * t),
                    yaw=yaw + 0.01 * math.sin(0.37 * t), pitch=pitch + 0.01 * math.sin(0.29 * t),
                    roll=roll + 0.026 * math.sin(0.4 * t))
        sx, sy = self.eye_pos
        shut = look.eyes != "open" and look.blend >= 0.95
        if shut and look.eyes == "closed":
            v = self.user.features(sx, sy, head, closed="left")
            v_right = self.user.features(sx, sy, head, closed="right")
            right = [c for c in EYE_INPUTS["right"] if c not in EYE_INPUTS["left"]]
            v[right] = v_right[right]
            return v
        return self.user.features(sx, sy, head, closed=look.eyes if shut else None)

    def _glasses_fields(self, look: Look) -> dict:
        """What a glasses detector would report (used where the session reports nothing itself)."""
        dt = 1.0 / FPS
        self.glasses_seen = self.glasses_seen + dt if look.glasses else 0.0
        before = self.glasses_state
        if look.glasses and self.glasses_seen >= 0.5:
            self.glasses_state = True
        elif not look.glasses:
            self.glasses_state = False
        a = 1.0 - math.exp(-dt / 0.2)
        target = np.array([1.4 if look.glare else 0.03, 0.04])
        self.glare_score += a * (target - self.glare_score)
        glare = "left" if self.glare_score[0] >= 0.5 else None
        if self.glasses_state != before:
            self.pending_glasses_event = self.driver.t + 3.0     # announced once it has held a moment
        return {"glasses": self.glasses_state, "glare": glare,
                "glare_score": [round(float(min(1.0, s)), 2) for s in self.glare_score]}

    def look(self, look: Look) -> dict:
        frame, others = self.scene.render(self.driver.t - self.driver.t0, eyes=look.eyes, blend=look.blend,
                                          glasses=look.glasses, glare=look.glare, second=look.second,
                                          away=look.away, pose=look.pose)
        ok, buf = self.cv2.imencode(".jpg", frame, [self.cv2.IMWRITE_JPEG_QUALITY, 82])
        self.vector = self._features(look)
        glasses = self._glasses_fields(look)

        def extra(msg: dict) -> Optional[dict]:
            if not msg.get("face"):
                return None
            fields = dict(glasses)
            if others is not None:
                fields.update(faces=2, you=self.tap.box(), others=[[round(float(v), 4) for v in o] for o in others])
            return fields

        announce = None
        pending = self.pending_glasses_event
        if pending is not None and self.driver.t >= pending:
            self.pending_glasses_event = None
            announce = [{"type": "glasses_changed", "glasses": self.glasses_state, "slot_available": False,
                         "in_use": False}]
        msg = self.driver.frame(buf.tobytes(), image_bgr=frame, label=look.label,
                                points_of=lambda: self.tap.last.points_px if self.tap.last else None,
                                features=self.vector, extra=extra, header=look.header, extra_events=announce)
        self.frame_no += 1
        return msg

    # -- script pieces -------------------------------------------------------------------
    def hold(self, target, seconds: float, **kw) -> Iterator[Look]:
        for _ in range(max(1, int(round(seconds * FPS)))):
            yield Look(tuple(target), **kw)

    def blink(self, target, closed_frames: int = 4, **kw) -> Iterator[Look]:
        yield Look(tuple(target), eyes="closed", blend=0.75, **kw)
        for _ in range(closed_frames):
            yield Look(tuple(target), eyes="closed", blend=1.0, **kw)
        yield Look(tuple(target), eyes="closed", blend=0.75, **kw)

    def double_blink(self, target, **kw) -> Iterator[Look]:
        yield from self.blink(target, **kw)
        yield from self.hold(target, 0.2, **kw)
        yield from self.blink(target, **kw)

    def dots(self, points, kind: str, seconds: float, blink_at: Optional[int] = None) -> Iterator[Look]:
        n = int(round(seconds * FPS))
        for pt, (fx, fy) in enumerate(points):
            target = (fx * SCREEN_W, fy * SCREEN_H)
            label = {"x": round(target[0], 1), "y": round(target[1], 1), "kind": kind, "pt": pt}
            k = 0
            while k < n:
                if blink_at is not None and pt == blink_at and k == n // 2:
                    for look in self.blink(target, label=label, header={"mesh": True, "overlay": True}):
                        yield look
                        k += 1
                    continue
                yield Look(target, label=label, header={"mesh": True, "overlay": True})
                k += 1

    def script(self) -> Iterator:
        q = self.quick
        centre = (SCREEN_W / 2, SCREEN_H / 2)
        cam = {"overlay": True}
        yield from self.hold(centre, 1.0, header=cam)
        # Full calibration: dots, head movements, training, validation, refit.
        yield {"type": "calibration_start", "mode": "full"}
        yield from self.dots(QUICK_POINTS if q else FULL_POINTS, "cal", 0.75 if q else 1.0, blink_at=4)
        for i, pose in enumerate(HEAD_POSES):
            yield from self.hold(centre, 0.15, pose=pose, header=cam)
            yield from self.hold(centre, 0.55 if q else 0.8, pose=pose, header={"mesh": True, "overlay": True},
                                 label={"x": centre[0], "y": centre[1], "kind": "head", "pt": i})
        yield {"type": "calibration_fit", "mode": "full"}
        yield from self.dots(VALIDATION_POINTS, "val", 0.7 if q else 0.9)
        yield {"type": "validation_finish"}
        # Browsing.
        scale = 0.6 if q else 1.0
        targets = [(fx * SCREEN_W, fy * SCREEN_H) for fx, fy in BROWSE_TARGETS]

        def wander(seconds: float, **kw) -> Iterator[Look]:
            left = seconds
            while left > 0:
                d = min(left, float(self.rng.uniform(0.45, 0.9)) * scale)
                target = targets[int(self.rng.integers(len(targets)))]
                yield from self.hold(target, d, header=cam, **kw)
                left -= d

        yield from wander(2.0)
        yield from self.blink(targets[4], header=cam)
        yield from wander(1.5)
        practice = targets[5]
        yield from self.hold(practice, 0.6, header=cam)
        yield from self.double_blink(practice, header=cam)
        yield from self.hold(practice, 0.2, header=cam)
        yield ("practice", practice)
        yield {"type": "settings", "smoothing": "high"}
        yield from wander(1.5)
        # A held wink (left eye) that drags something from one place to another.
        start, end = targets[7], targets[8]
        yield from self.hold(start, 0.5, header=cam)
        yield Look(start, eyes="left", blend=0.75, header=cam)
        for k in range(int(1.1 * FPS)):
            yield Look(start if k < 10 else end, eyes="left", blend=1.0, header=cam)
        yield Look(end, eyes="left", blend=0.75, header=cam)
        yield from self.hold(end, 0.6, header=cam)
        yield from wander(1.0)
        practice = targets[6]
        yield from self.hold(practice, 0.6, header=cam)
        yield from self.double_blink(practice, header=cam)
        yield from self.hold(practice, 0.2, header=cam)
        yield ("practice", practice)
        # Glasses go on, with a reflection on the left lens for a while.
        yield from wander(2.0 * scale, glasses=True)
        yield from wander(2.0 * scale, glasses=True, glare=True)
        yield from wander(2.0 * scale, glasses=True)
        yield from self.blink(targets[2], header=cam, glasses=True)
        yield from wander(1.0 * scale, glasses=True)
        # Someone else comes into view, then the face leaves the picture.
        yield from wander(3.0 * scale, second=True)
        yield from self.hold(centre, 1.2, away=True, header=cam)
        yield from wander(1.0)
        yield {"type": "gestures_set", "gestures": {"dwell": True, "dwell_ms": 800}}
        yield {"type": "face_recognize", "seconds": 2}
        yield from wander(2.5)
        practice = targets[3]
        yield from self.hold(practice, 0.6, header=cam)
        yield from self.double_blink(practice, header=cam)
        yield from self.hold(practice, 0.2, header=cam)
        yield ("practice", practice)
        yield {"type": "settings", "smoothing": "auto"}
        yield from wander(1.5)
        # A quick adjust: nine dots, a correction, five dots to measure it.
        yield {"type": "calibration_start", "mode": "adjust"}
        yield from self.dots(ADJUST_POINTS, "adjust", 0.6 if q else 0.8)
        yield {"type": "calibration_fit", "mode": "adjust"}
        yield from self.dots(VALIDATION_POINTS, "val", 0.6 if q else 0.8)
        yield {"type": "validation_finish", "refit": False}
        yield from wander(1.5)
        yield {"type": "finetune"}
        yield "wait"
        yield from wander(1.0)
        # "Test my winks": one eye closed, then the other, as the page asks (frames carry "gesture").
        yield {"type": "wink_calibration_start"}
        for ask, seconds in (("rest", 1.8), ("left", 2.8), ("rest", 1.9), ("right", 2.8), ("rest", 1.6)):
            yield from self.hold(centre, seconds * (0.5 if q else 1.0), eyes="open" if ask == "rest" else ask,
                                 header={"gesture": ask, "overlay": True})
        yield {"type": "wink_calibration_finish"}
        yield from wander(1.0)

    def run(self) -> Path:
        with _simulated_features(self):
            for step in self.script():
                if isinstance(step, Look):
                    self.look(step)
                elif isinstance(step, dict):
                    self.driver.command(step)
                elif step == "wait":
                    self.driver.wait_for_job()
                elif isinstance(step, tuple) and step[0] == "practice":
                    # The page pops a practice target after a double blink and tells the session.
                    double = [e for e in self.driver.event_log if e.get("type") == "double_blink"]
                    if double:
                        self.driver.command({"type": "label_event", "kind": "practice",
                                             "pre_frame": double[-1].get("pre_frame"),
                                             "target": [round(step[1][0], 1), round(step[1][1], 1)]})
        return self.driver.finish()


class _TapTracker:
    """A face tracker that keeps its last observation (for the landmark chunks)."""

    def __init__(self, inner):
        self.inner = inner
        self.last = None

    def process(self, rgb, timestamp_ms):
        self.last = self.inner.process(rgb, timestamp_ms)
        return self.last

    def box(self) -> Optional[list[float]]:
        if self.last is None:
            return None
        w, h = self.last.image_size
        P = self.last.points_px[:, :2]
        return [round(float(P[:, 0].min() / w), 4), round(float(P[:, 1].min() / h), 4),
                round(float(P[:, 0].max() / w), 4), round(float(P[:, 1].max() / h), 4)]

    def close(self):
        self.inner.close()


class _simulated_features:
    """While active, the session's feature vector is the simulated person's (MediaPipe's
    geometry, closures and head pose stay real)."""

    def __init__(self, demo: EyeDemo):
        self.demo = demo
        self.saved = None

    def __enter__(self):
        real = session_mod.extract_features
        demo = self.demo

        def extract(points_px, image_size, blendshapes=None, matrix=None):
            feats = real(points_px, image_size, blendshapes, matrix)
            return dataclasses.replace(feats, vector=demo.vector.copy())

        self.saved = real
        session_mod.extract_features = extract
        return self

    def __exit__(self, *exc):
        session_mod.extract_features = self.saved
        return False


# ---------------------------------------------------------------------------
# Hand mode: drawn hands
# ---------------------------------------------------------------------------

HAND_GRID = [(fx, fy) for fy in (0.12, 0.5, 0.88) for fx in (0.1, 0.4, 0.6, 0.9)] + [(0.5, 0.5)]
_FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))


def draw_hand(pts: Optional[np.ndarray], size=(640, 480), t: float = 0.0) -> np.ndarray:
    """A camera-like picture with the hand drawn from its landmarks (image fractions)."""
    import cv2

    w, h = size
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    base = np.zeros((h, w, 3), np.float32)
    base[:] = (58, 46, 40)
    base += (yy * np.array([22, 26, 30], np.float32))[:, None, :].reshape(h, 1, 3)
    img = base.astype(np.uint8)
    cv2.rectangle(img, (0, int(0.82 * h)), (w, h), (52, 62, 74), -1)          # the desk
    if pts is None:
        return img
    P = np.asarray(pts, float)[:, :2] * [w, h]
    width = float(np.hypot(*(P[5] - P[17])))
    thick = max(6, int(0.32 * width))
    skin, edge = (128, 160, 205), (70, 92, 128)
    palm = np.round(P[[0, 1, 2, 5, 9, 13, 17]]).astype(np.int32)
    cv2.fillConvexPoly(img, cv2.convexHull(palm), skin, cv2.LINE_AA)
    for chain in _FINGERS:
        poly = np.round(P[list(chain)]).astype(np.int32)
        cv2.polylines(img, [poly], False, edge, thick + 4, cv2.LINE_AA)
    cv2.polylines(img, [cv2.convexHull(palm)], True, edge, 3, cv2.LINE_AA)
    cv2.fillConvexPoly(img, cv2.convexHull(palm), skin, cv2.LINE_AA)
    for chain in _FINGERS:
        poly = np.round(P[list(chain)]).astype(np.int32)
        cv2.polylines(img, [poly], False, skin, thick, cv2.LINE_AA)
        tip = tuple(int(v) for v in np.round(P[chain[-1]]))
        cv2.circle(img, tip, max(2, thick // 4), (150, 180, 222), -1, cv2.LINE_AA)
    return img


class HandDemo:
    def __init__(self, root: Path, *, quick: bool, seed: int, start_wall: float, name: str = "Alex"):
        import cv2

        self.cv2 = cv2
        self.quick = quick
        self.tracker = FakeHandTracker()
        self.driver = Driver(root, name, "hand", self.tracker, start_wall,
                             video={"every": 2, "max_width": 640, "quality": 70})
        self.rng = np.random.default_rng(seed)
        self.tip = np.array([0.5, 0.45])
        self.last_pts: Optional[np.ndarray] = None

    def hand(self, tip, pinch=1.0, fingers="curled", centre_shift=(0.0, 0.0)) -> np.ndarray:
        w = 0.16
        tx, ty = tip
        centre = (tx + 0.5 * w + centre_shift[0], ty + 1.25 * w + centre_shift[1])
        return make_hand(centre=centre, tip=(tx, ty), pinch=pinch, fingers=fingers, width=w)

    def frame(self, pts: Optional[np.ndarray], label: Optional[dict] = None) -> dict:
        self.tracker.push(pts)
        img = draw_hand(pts, t=self.driver.t)
        ok, buf = self.cv2.imencode(".jpg", img, [self.cv2.IMWRITE_JPEG_QUALITY, 82])
        self.last_pts = pts
        return self.driver.frame(buf.tobytes(), image_bgr=img, label=label,
                                 points_of=lambda: pts)

    @staticmethod
    def screen_tip(fx: float, fy: float) -> tuple[float, float]:
        """Where the fingertip is (image fractions, un-mirrored) to point at screen fraction (fx, fy)."""
        return 1 - (0.35 + 0.3 * fx), 0.3 + 0.3 * fy

    def move(self, goal, seconds: float, **kw) -> None:
        start = self.tip.copy()
        goal = np.asarray(goal, float)
        n = max(1, int(round(seconds * FPS)))
        for k in range(1, n + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * min(1.0, k / (0.45 * n)))
            self.tip = start + (goal - start) * s
            jitter = self.rng.normal(0, 0.0012, 2)
            self.frame(self.hand(self.tip + jitter, **kw))

    def click(self) -> None:
        for d in (0.95, 0.7, 0.45, 0.28, 0.18, 0.15, 0.15, 0.3, 0.6, 0.95, 1.05):
            self.frame(self.hand(self.tip, pinch=d))

    def run(self) -> Path:
        d = self.driver
        q = self.quick
        for _ in range(int(FPS)):
            self.frame(self.hand(self.tip))
        d.command({"type": "hand_calibration_start", "mode": "full"})
        for _ in range(25):
            self.frame(make_hand(centre=(0.5, 0.55), fingers="spread"), {"kind": "hspan"})
        for i, (fx, fy) in enumerate(HAND_GRID):
            goal = np.array(self.screen_tip(fx, fy))
            for k in range(12 if q else 16):
                self.tip = self.tip + 0.5 * (goal - self.tip)
                self.frame(self.hand(self.tip + self.rng.normal(0, 0.001, 2)),
                           {"kind": "hpoint", "i": i, "fx": fx, "fy": fy})
        for k in range(90 if q else 120):
            self.frame(self.hand(self.tip, pinch=0.2 if (k // 12) % 2 else 1.0), {"kind": "hpinch", "state": "cycle"})
        d.command({"type": "hand_calibration_fit"})
        targets = [(0.1, 0.06), (0.3, 0.06), (0.5, 0.06), (0.4, 0.4), (0.7, 0.35), (0.25, 0.7), (0.8, 0.8)]
        for k, (fx, fy) in enumerate(targets):
            self.move(self.screen_tip(fx, fy), 0.7 if q else 1.0)
            if k in (2, 4, 6):
                self.click()
        # Pinch and pull the hand down: scroll.
        self.move(self.screen_tip(0.5, 0.4), 0.6)
        for k in range(24):
            self.frame(self.hand(self.tip, pinch=0.18, centre_shift=(0.0, 0.0)) + [0, 0.006 * k, 0])
        self.frame(self.hand(self.tip + [0, 0.006 * 23], pinch=1.05))
        self.tip = self.tip + [0, 0.006 * 23]
        self.move(self.screen_tip(0.6, 0.5), 0.6)
        # The open "stop" hand pauses; again resumes.
        for _ in range(int(1.1 * FPS)):
            self.frame(make_hand(centre=(0.5, 0.55), fingers="spread"))
        self.move(self.screen_tip(0.5, 0.5), 0.6)
        for _ in range(int(0.6 * FPS)):        # the hand leaves the picture
            self.frame(None)
        self.move(self.screen_tip(0.45, 0.45), 0.4)
        for _ in range(int(1.1 * FPS)):
            self.frame(make_hand(centre=(0.5, 0.55), fingers="spread"))
        for fx, fy in ((0.3, 0.3), (0.62, 0.62)):
            self.move(self.screen_tip(fx, fy), 0.8)
            self.click()
        self.move(self.screen_tip(0.5, 0.5), 0.8)
        return d.finish()


# ---------------------------------------------------------------------------

def make_demo(out: Path, *, mode: str = "eyes", quick: bool = False, seed: int = 0, width: int = 960,
              video_width: int = 640, start_wall: Optional[float] = None, name: Optional[str] = None) -> Path:
    """Make one demo recording in ``out``; returns its folder.

    ``width``: the analysed camera frames (eye mode; the page sends up to 960
    pixels wide); ``video_width``: the stored images (narrower ones are scaled
    down, as the recorder does with ``video.max_width``).
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start_wall = time.time() if start_wall is None else start_wall
    if mode == "hand":
        return HandDemo(out, quick=quick, seed=seed, start_wall=start_wall, name=name or "Alex").run()
    return EyeDemo(out, quick=quick, seed=seed, width=width, start_wall=start_wall, name=name or "Sam",
                   video_width=video_width).run()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Make demo session recordings for the Paralic Inspector.")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "recordings", help="folder for the recordings")
    parser.add_argument("--mode", choices=("eyes", "hand", "both"), default="both")
    parser.add_argument("--quick", action="store_true", help="shorter recordings (fewer dots, shorter browsing)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=960, help="width of the analysed camera frames (eye mode)")
    parser.add_argument("--video-width", type=int, default=640, help="largest width of the stored camera images")
    args = parser.parse_args(argv)
    now = time.time()
    modes = ("eyes", "hand") if args.mode == "both" else (args.mode,)
    for k, mode in enumerate(modes):
        started = time.perf_counter()
        # Recordings made one after the other, a few minutes apart, so their names differ.
        folder = make_demo(args.out, mode=mode, quick=args.quick, seed=args.seed, width=args.width,
                           video_width=args.video_width, start_wall=now - 600.0 * (len(modes) - k))
        print(f"{mode:5s} demo: {folder} ({time.perf_counter() - started:.1f} s)")


if __name__ == "__main__":
    main()
