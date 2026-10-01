"""Session recordings for the Paralic Inspector (see docs/recording-format.md).

A :class:`Recorder` keeps everything about one session in one directory: the
configuration at the start (``meta.json``), every camera frame's measurements
and what the pipeline made of them (``frames.jsonl``), the tracked points
(``landmarks/``), every command, reply, pushed message and gesture event
(``events.jsonl``), the gaze model whenever it changes (``models/``) and camera
images (``video/``).

The session calls it from its worker thread (frames, commands) and from
background jobs (pushes, a fine-tuned model). Those calls only turn what they
are given into JSON lines and float16 points and queue them; one writer thread
does all the file work - and shrinks camera images wider than ``max_width`` -
so a slow disk never slows down tracking. When the writer falls behind,
camera images are left out first. A problem with the recording is logged and
switches the recording off; tracking goes on.
"""

from __future__ import annotations

import atexit
import dataclasses
import io
import json
import logging
import math
import os
import queue
import shutil
import tempfile
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import __version__

log = logging.getLogger(__name__)

FORMAT = 1
CHUNK_FRAMES = 300                  # frames per landmarks/NNNNNN.npz
FLUSH_EVERY_S = 60.0                # flush the files and refresh meta.json this often
MAX_BYTES = 3_000_000_000           # beyond this, a recording keeps everything but camera images
MIN_FREE_BYTES = 1_000_000_000      # ...and so it does with less than this free on the disk
SPACE_CHECK_EVERY = 100             # camera images between free-space checks
VIDEO_BACKLOG = 90                  # queued items beyond which new frames come without their image
MAX_BACKLOG = 3000                  # queued items beyond which frames are dropped (counted in meta.json)
OTHER_FACES = 3                     # boxes of other faces kept per frame (landmarks' ``others``)
DATA_URL_CHARS = 256                # longer data: URLs (pictures) in commands and replies are not recorded

# What a model snapshot (models/NNN-<why>.json) is called after the model history's
# source (TrackerSession._record_model: "calibration" is the refit with the accuracy-check
# dots) or after the command that changed the model; anything else keeps its own name.
SNAPSHOT_NAMES = {"calibration_fit": "calibration", "calibration": "refit", "fine-tune": "finetune"}

_STOP = object()
_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})


@dataclass
class VideoOptions:
    every: int = 2           # the camera image of every n-th frame (0: no images)
    max_width: int = 640     # wider images are shrunk to this width...
    quality: int = 70        # ...and saved with this JPEG quality

    @classmethod
    def from_dict(cls, d: Any) -> "VideoOptions":
        """Options from a ``recording`` command's ``video`` (anything invalid keeps its default)."""
        opts = cls()
        if d is False:
            opts.every = 0
        elif isinstance(d, dict):
            for key, lo, hi in (("every", 0, 60), ("max_width", 160, 1920), ("quality", 30, 95)):
                value = d.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                    setattr(opts, key, int(min(max(value, lo), hi)))
        return opts


# -- camera images -------------------------------------------------------------------------

def jpeg_size(data: bytes) -> Optional[tuple[int, int]]:
    """(width, height) from a JPEG's frame header without decoding it; None if it is not a JPEG."""
    if data[:2] != b"\xff\xd8":
        return None
    i, n = 2, len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:                         # fill byte
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:   # markers without a length
            i += 2
            continue
        if marker in _SOF:
            if i + 9 > n:
                return None
            return int.from_bytes(data[i + 7:i + 9], "big"), int.from_bytes(data[i + 5:i + 7], "big")
        if marker in (0xD9, 0xDA):                 # the end, or image data, before any frame header
            return None
        i += 2 + int.from_bytes(data[i + 2:i + 4], "big")
    return None


def fit_width(image: bytes, max_width: int, quality: int) -> bytes:
    """``image`` as a JPEG at most ``max_width`` pixels wide: the bytes as they
    are when they fit, otherwise shrunk and re-encoded with ``quality``."""
    size = jpeg_size(image)
    if size is not None and size[0] <= max_width:
        return image
    try:
        import cv2
    except ImportError:  # pragma: no cover - OpenCV comes with MediaPipe
        from PIL import Image

        with Image.open(io.BytesIO(image)) as im:
            im = im.convert("RGB")
            if im.width > max_width:
                im = im.resize((max_width, max(1, round(im.height * max_width / im.width))), Image.LANCZOS)
            out = io.BytesIO()
            im.save(out, "JPEG", quality=int(quality))
            return out.getvalue()
    # Let the JPEG decoder do most of the shrinking (it can decode at 1/2, 1/4, 1/8 size).
    reduce = 1
    while size is not None and reduce < 8 and size[0] // (2 * reduce) >= max_width:
        reduce *= 2
    flags = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4,
             8: cv2.IMREAD_REDUCED_COLOR_8}[reduce]
    img = cv2.imdecode(np.frombuffer(image, np.uint8), flags)
    if img is None:
        raise ValueError("could not decode a camera image")
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, max(1, round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("could not encode a camera image")
    return buf.tobytes()


# -- what gets recorded ----------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    """json.dumps fallback: NumPy values become plain numbers and lists; anything else its text
    (a stray value must not end the recording)."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return str(obj)


def _finite(value: Any) -> Any:
    """``value`` with NaN and infinities as None (strict JSON has no such numbers)."""
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _dumps(obj: Any, indent: Optional[int] = None) -> str:
    """Strict JSON (a browser's JSON.parse reads it), compact unless ``indent``."""
    separators = (",", ":") if indent is None else (",", ": ")
    try:
        return json.dumps(obj, separators=separators, indent=indent, default=_jsonable, allow_nan=False)
    except ValueError:
        return json.dumps(_finite(obj), separators=separators, indent=indent, default=_jsonable, allow_nan=False)


def _scrub(value: Any) -> Any:
    """``value`` without the pictures embedded in it as data: URLs (e.g. the face print's faces)."""
    if isinstance(value, str):
        if value.startswith("data:") and len(value) > DATA_URL_CHARS:
            return f"{value[:value.find(',') + 1]}... ({len(value)} characters, not recorded)"
        return value
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def _quiet(cmd: dict) -> bool:
    """Commands not worth recording: keep-alives and the page asking how the recording is going."""
    kind = cmd.get("type")
    return kind == "ping" or (kind == "recording" and "on" not in cmd)


def _vector(v: Any) -> Optional[list]:
    if v is None:
        return None
    return [round(x, 6) if math.isfinite(x) else None for x in np.asarray(v, float).ravel().tolist()]


def _points(points: Any) -> Optional[np.ndarray]:
    """Tracked points as a (P, 3) float16 array, or None (no face or hand, or not points at all)."""
    if points is None:
        return None
    try:
        p = np.asarray(points, dtype=np.float16)
    except (TypeError, ValueError):
        return None
    return p if p.ndim == 2 and p.shape[1] == 3 and len(p) else None


def _other_faces(others: Any) -> Optional[np.ndarray]:
    """Boxes of other faces in view from the frame message's ``others`` ([x0, y0, x1, y1] each,
    or {"box": [...]}): an (OTHER_FACES, 4) float16 array, NaN where there are fewer."""
    if not isinstance(others, (list, tuple)):
        return None
    out = np.full((OTHER_FACES, 4), np.nan, np.float16)
    k = 0
    for item in others:
        box = item.get("box") if isinstance(item, dict) else item
        try:
            out[k] = np.asarray(box, float).reshape(4)
        except (TypeError, ValueError):
            continue
        k += 1
        if k == OTHER_FACES:
            break
    return out


def _blink_state(detector, face: bool) -> dict:
    s = detector.state()
    return {"closing": s.closing, "closed": s.closed, "deep": s.deep,
            "close_thr": round(s.close_threshold, 4), "open_thr": round(s.open_threshold, 4),
            "signal": round(detector.closure, 4) if face else None,
            "baseline": round(s.baseline, 4), "pending": s.pending}


def _wink_state(detector) -> dict:
    eyes = ("left", "right")
    thr = [detector.thresholds(e) for e in eyes]
    return {"winking": detector.winking, "pressed": detector.pressed, "state": detector.state,
            "base": [round(detector.baseline(e), 4) for e in eyes],
            "close_thr": [round(t[0], 4) for t in thr], "open_thr": [round(t[1], 4) for t in thr]}


def detector_config(session) -> dict:
    """The settings the session's detectors and filters work with right now."""
    hand = session.hand
    if hand is not None:
        return {"hand": {"config": dataclasses.asdict(hand.recognizer.config),
                         "calibration": hand.calibration.to_dict() if hand.calibration else None,
                         "smoothing": dataclasses.asdict(hand.filter.params)}}
    st = session.stabilizer
    return {"blink": dataclasses.asdict(session.blink.config), "blink_signal": session.blink_mode,
            "wink": dataclasses.asdict(session.wink.config),
            "smoothing": {**dataclasses.asdict(st.filter.params), "rewind_s": st.rewind_s,
                          "settle_s": st.settle_s, "max_freeze_s": st.max_freeze_s}}


def session_config(session) -> dict:
    """The session's configuration as meta.json records it (and a person_changed event)."""
    user = session.user or {}
    return {
        "mode": session.mode,
        "person": {"id": user.get("id"), "name": user.get("name")},
        "screen": session.screen,
        "settings": dict(session.settings),
        "effective": session.effective(),
        "personal": session.personal,
        "profile": session.profiles.summary(),
        "hand": session.hand.view() if session.hand is not None else None,
        "detectors": detector_config(session),
    }


def _slug(name: Any) -> str:
    """A person's name as a safe part of a directory name."""
    out, gap = [], False
    for ch in str(name or ""):
        if ch.isalnum():
            out.append(ch)
            gap = False
        elif not gap and out:
            out.append("_")
            gap = True
    return "".join(out).strip("_")[:32] or "person"


def _new_directory(root: Path, name: str, mode: str, when: float) -> Path:
    """<YYYYmmdd-HHMMSS>-<name>-<mode> (-2, -3... when that exists), readable only by this user."""
    base = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(when))}-{_slug(name)}-{mode}"
    root.mkdir(parents=True, exist_ok=True)
    for k in range(1, 100):
        path = root / (base if k == 1 else f"{base}-{k}")
        try:
            path.mkdir(mode=0o700)
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"Too many recordings named {base} in {root}")


def _write_atomic(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# Recordings still running when Python exits (e.g. Ctrl+C) are stopped cleanly.
_running: "weakref.WeakSet[Recorder]" = weakref.WeakSet()


@atexit.register
def _stop_all() -> None:  # pragma: no cover - runs at interpreter exit
    for rec in list(_running):
        rec.stop(timeout=10.0)


class Recorder:
    """Records one session into a new directory under ``root`` (see the module docstring).

    ``session`` is the :class:`~paralic.session.TrackerSession`: its
    configuration goes into meta.json, its clocks time everything, and the
    recorder reads its detectors, model and person when asked to record a
    frame, a model or the replies to a command. ``video`` holds the camera
    image options of a ``recording`` command (see :class:`VideoOptions`).
    Camera images stop beyond ``max_bytes`` in the recording or with less
    than ``min_free`` bytes free on the disk (default: MAX_BYTES, MIN_FREE_BYTES).
    """

    def __init__(self, session, root: Path, video: Any = None, *, max_bytes: Optional[int] = None,
                 min_free: Optional[int] = None):
        self.session = session
        self.video = VideoOptions.from_dict(video)
        self.max_bytes = int(MAX_BYTES if max_bytes is None else max_bytes)
        self.min_free = int(MIN_FREE_BYTES if min_free is None else min_free)
        self.error: Optional[str] = None
        # Counted by the writer thread, read by anyone.
        self.frames = self.video_frames = self.events = self.models = self.bytes = self.dropped = 0
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._active = False
        self._stopping = False
        self._ended_t: Optional[float] = None
        self._n = 0                                # frames seen (for video.every)
        self._video_on = self.video.every > 0
        self._video_stopped: Optional[dict] = None
        self._model_key: tuple = (None, None)      # (model, version) last recorded
        self._snapshots = 0
        self._chunk: list[tuple] = []
        self._chunks = 0
        self._meta_bytes = 0
        self._since_space_check = 0

        config = json.loads(_dumps(session_config(session)))
        self.started_t = float(session.clock())
        self.started_wall = float(session.wall())
        self._person = config["person"]
        self.dir = _new_directory(Path(root), config["person"]["name"], config["mode"], self.started_wall)
        self.id = self.dir.name
        try:
            for sub in ("landmarks", "models", "video"):
                (self.dir / sub).mkdir()
            if self._video_on:
                reason = self._low_space()
                if reason:
                    self._stop_video(reason)
            self._head = {"format": FORMAT, "paralic": __version__, "id": self.id,
                          "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started_wall))}
            self._config = {"clock": {"t": round(self.started_t, 4), "wall": round(self.started_wall, 3)}, **config}
            self._write_meta(final=False, ended=False)
            self._frames_fh = open(self.dir / "frames.jsonl", "wb", buffering=1 << 16)
            self._events_fh = open(self.dir / "events.jsonl", "wb", buffering=1 << 16)
        except BaseException:
            shutil.rmtree(self.dir, ignore_errors=True)
            raise
        self._active = True
        self._thread = threading.Thread(target=self._run, name="paralic-recorder", daemon=True)
        self._thread.start()
        _running.add(self)
        log.info("Recording %s in %s", self.id, self.dir)
        self.model("start")

    # ------------------------------------------------------------------
    # Called by the session (any thread)
    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._active

    def frame(self, t: float, header: dict, image: Optional[bytes], out: list[dict], features: Any = None,
              points: Any = None) -> None:
        """One processed camera frame: ``out`` is what the page gets (the frame message, then the
        events it produced); ``points`` the face mesh (pixels) or hand (image fractions)."""
        if not self._active:
            return
        try:
            msg = out[0] if out else {}
            events = out[1:]
            fid = msg.get("id", header.get("id"))
            wall = self.session.wall()
            n, self._n = self._n, self._n + 1
            busy = self._queue.qsize()
            if busy >= MAX_BACKLOG:
                self.dropped += 1
                return
            video = (bool(image) and self._video_on and n % self.video.every == 0 and busy < VIDEO_BACKLOG
                     and isinstance(fid, int) and not isinstance(fid, bool) and fid >= 0)
            eyes = self.session.hand is None
            face = bool(msg.get("face"))
            if "mesh" in msg:
                # The calibration preview's copy of the face mesh: the landmarks hold all of it.
                msg = {**msg, "mesh": True}
            # "video" (whether video/<i>.jpg exists) is added by the writer, which knows.
            line = {"i": fid, "t": round(t, 4), "wall": round(wall, 3), "msg": msg,
                    "features": _vector(features) if eyes else None,
                    "blink": _blink_state(self.session.blink, face) if eyes else None,
                    "wink": _wink_state(self.session.wink) if eyes else None,
                    "net": msg.get("net"), "label": header.get("label"), "stored": bool(msg.get("labeled")),
                    "events": [m.get("type") for m in events]}
            if header.get("gesture"):
                line["gesture"] = header["gesture"]
            event_lines = [_dumps({"t": round(t, 4), "wall": round(wall, 3), "kind": "event", "type": m.get("type"),
                                   "data": m}) for m in events]
            pts = _points(points)
            others = _other_faces(msg.get("others")) if pts is not None and eyes else None
            self._put(("frame", fid, t, wall, _dumps(line), event_lines, image if video else None, pts, others))
        except Exception as exc:
            self._fail(exc)

    def command(self, cmd: dict) -> None:
        """A command from the page (before it is handled)."""
        if self._active and not _quiet(cmd):
            self._event("command", cmd.get("type"), _scrub(cmd))

    def replies(self, cmd: dict, replies: list[dict]) -> None:
        """The replies to ``cmd`` - after recording what the command changed: who is using Paralic
        (a ``person_changed`` event with their configuration) and the gaze model."""
        if not self._active:
            return
        try:
            user = self.session.user or {}
            if user.get("id") != self._person.get("id"):
                previous, self._person = self._person, {"id": user.get("id"), "name": user.get("name")}
                self._event("event", "person_changed", {**session_config(self.session), "previous": previous})
            self.model(str(cmd.get("type")))
            if not _quiet(cmd):
                for reply in replies:
                    self._event("reply", reply.get("type"), _scrub(reply))
        except Exception as exc:
            self._fail(exc)

    def push(self, msg: dict) -> None:
        """A message a background job sent to the page."""
        if self._active:
            self._event("push", msg.get("type"), _scrub(msg))

    def model(self, why: str) -> None:
        """Snapshot the session's gaze model if it changed since the last one (call it under the
        session lock). ``why``: the model history's source, or the command that changed it."""
        if not self._active:
            return
        try:
            model = self.session.model
            version = None if model is None else model.meta.get("version")
            last, last_version = self._model_key
            if model is last and version == last_version:
                return
            self._model_key = (model, version)
            why = SNAPSHOT_NAMES.get(why, why)
            name = None
            if model is not None:
                with self._lock:
                    k, self._snapshots = self._snapshots, self._snapshots + 1
                safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in why)[:40] or "change"
                name = f"models/{k:03d}-{safe}.json"
                self._put(("model", name, _dumps(model.to_dict())))
            elif last is None:
                return                                    # still no model
            self._event("event", "model", {"why": why, "file": name, "version": version})
        except Exception as exc:
            self._fail(exc)

    def status(self) -> dict:
        """How the recording is going (the ``recording`` reply's fields)."""
        end = self._ended_t if self._ended_t is not None else self.session.clock()
        return {"on": self._active, "id": self.id, "dir": str(self.dir.absolute()), "frames": self.frames,
                "video_frames": self.video_frames, "events": self.events, "bytes": self.bytes,
                "seconds": round(max(0.0, end - self.started_t), 1), "video": dataclasses.asdict(self.video),
                "video_stopped": (self._video_stopped or {}).get("reason"), "error": self.error}

    def stop(self, timeout: float = 30.0) -> dict:
        """Write everything still queued, close the files and refresh meta.json one last time."""
        with self._lock:
            if not self._stopping:
                self._stopping = True
                self._active = False
                self._ended_t = self.session.clock()
                self._queue.put(_STOP)
        self._thread.join(timeout)
        _running.discard(self)
        return self.status()

    # ------------------------------------------------------------------
    # Internals: the calling side
    # ------------------------------------------------------------------
    def _event(self, kind: str, type_: Any, data: Any) -> None:
        try:
            line = {"t": round(self.session.clock(), 4), "wall": round(self.session.wall(), 3), "kind": kind,
                    "type": type_, "data": data}
            self._put(("event", _dumps(line)))
        except Exception as exc:
            self._fail(exc)

    def _put(self, item: tuple) -> None:
        with self._lock:
            if self._active:
                self._queue.put(item)

    def _fail(self, exc: BaseException) -> None:
        with self._lock:
            first = self.error is None
            if first:
                self.error = f"{type(exc).__name__}: {exc}"
            self._active = False
        if first:
            log.error("Recording %s stopped: %s", getattr(self, "id", "?"), self.error, exc_info=exc)

    # ------------------------------------------------------------------
    # Internals: the writer thread
    # ------------------------------------------------------------------
    def _run(self) -> None:
        last_flush = time.monotonic()
        broken = False
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                item = None
            if item is _STOP:
                break
            try:
                if item is not None:
                    self._write(item)
                if time.monotonic() - last_flush >= FLUSH_EVERY_S:
                    last_flush = time.monotonic()
                    self._flush()
            except Exception as exc:
                self._fail(exc)
                broken = True
                break
        try:
            if not broken:
                self._write_chunk()
        except Exception as exc:
            self._fail(exc)
        for fh in (self._frames_fh, self._events_fh):
            try:
                fh.close()
            except Exception as exc:
                self._fail(exc)
        try:
            self._write_meta(final=True)
        except Exception:
            log.warning("Could not finish meta.json of recording %s", self.id, exc_info=True)
        if self.error is None:
            log.info("Recording %s finished: %d frames, %.1f MB", self.id, self.frames, self.bytes / 1e6)

    def _write(self, item: tuple) -> None:
        kind = item[0]
        if kind == "frame":
            _, fid, t, wall, line, event_lines, image, points, others = item
            video = image is not None and self._video_on
            if video:
                self._write_video(fid, t, wall, image)
            if points is not None:
                if self._chunk and self._chunk[0][1].shape != points.shape:
                    self._write_chunk()
                self._chunk.append((fid, points, others))
                if len(self._chunk) >= CHUNK_FRAMES:
                    self._write_chunk()
            self._write_line(self._frames_fh, f'{line[:-1]},"video":{"true" if video else "false"}}}')
            self.frames += 1
            for ev in event_lines:
                self._write_line(self._events_fh, ev)
                self.events += 1
        elif kind == "event":
            self._write_line(self._events_fh, item[1])
            self.events += 1
        elif kind == "model":
            data = item[2].encode("utf-8")
            (self.dir / item[1]).write_bytes(data)
            self.bytes += len(data)
            self.models += 1

    def _write_line(self, fh, line: str) -> None:
        data = line.encode("utf-8") + b"\n"
        fh.write(data)
        self.bytes += len(data)

    def _write_video(self, fid: int, t: float, wall: float, image: bytes) -> None:
        data = fit_width(image, self.video.max_width, self.video.quality)
        (self.dir / "video" / f"{fid}.jpg").write_bytes(data)
        self.bytes += len(data)
        self.video_frames += 1
        self._since_space_check += 1
        reason = None
        if self.bytes >= self.max_bytes:
            reason = f"the recording reached {self.max_bytes / 1e9:g} GB"
        elif self._since_space_check >= SPACE_CHECK_EVERY:
            reason = self._low_space()
        if reason:
            self._stop_video(reason, fid, t, wall)

    def _low_space(self) -> Optional[str]:
        """Why the camera images (the bulk of a recording) should stop: the disk is getting full."""
        self._since_space_check = 0
        if shutil.disk_usage(self.dir).free >= self.min_free:
            return None
        return f"less than {self.min_free / 1e9:g} GB free on the disk"

    def _stop_video(self, reason: str, fid: Any = None, t: Optional[float] = None,
                    wall: Optional[float] = None) -> None:
        """No more camera images; everything else goes on (meta.json and a video_stopped event say why)."""
        self._video_on = False
        self._video_stopped = {"i": fid, "t": None if t is None else round(t, 4), "reason": reason,
                               "bytes": self.bytes}
        log.warning("Recording %s: no more camera images (%s)", self.id, reason)
        if t is None:
            return                                        # starting: meta.json is written next
        self._write_line(self._events_fh, _dumps({"t": round(t, 4), "wall": round(wall, 3), "kind": "event",
                                                  "type": "video_stopped", "data": self._video_stopped}))
        self.events += 1
        self._write_meta(final=False)

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
        data = buf.getvalue()
        _write_atomic(self.dir / "landmarks" / f"{self._chunks:06d}.npz", data)
        self._chunks += 1
        self.bytes += len(data)

    def _flush(self) -> None:
        self._frames_fh.flush()
        self._events_fh.flush()
        self._write_meta(final=False)

    def _write_meta(self, final: bool, ended: bool = True) -> None:
        """meta.json: the configuration at the start, and (refreshed every minute, so that a
        crash loses little) when it ended and what it holds; ``complete`` once it stopped cleanly."""
        meta = {**self._head,
                "ended": time.strftime("%Y-%m-%dT%H:%M:%S") if ended else None,
                "complete": final and self.error is None,
                **self._config,
                "video": {**dataclasses.asdict(self.video), "max_bytes": self.max_bytes,
                          "stopped": self._video_stopped},
                "summary": {"frames": self.frames, "video_frames": self.video_frames, "events": self.events,
                            "models": self.models, "dropped": self.dropped, "bytes": self.bytes},
                "error": self.error}
        data = _dumps(meta, indent=1).encode("utf-8")
        _write_atomic(self.dir / "meta.json", data)
        self.bytes += len(data) - self._meta_bytes
        self._meta_bytes = len(data)
