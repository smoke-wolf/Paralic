"""Reading session recordings (format 1, see ``docs/recording-format.md``).

A recording is one folder, usually under ``data/recordings/``::

    meta.json              the configuration at the start, a summary at the end
    frames.jsonl           one line per camera frame
    events.jsonl           commands, replies, pushed messages and gesture events
    landmarks/NNNNNN.npz   the tracked points, in chunks of up to 300 frames
    models/NNN-<why>.json  the gaze model whenever it changed
    video/<frame id>.jpg   camera images

:class:`Recording` reads one without loading all of it. An index of where each
frame's line starts in ``frames.jsonl`` is built once (a pass over the bytes
that parses only the frame id and time at the start of each line) and extended
when the file grows, so any frame or stretch of time can be read directly and
an hour-long recording opens in about a second.

A recording may still be growing, or may have ended in a crash: a half-written
last line is skipped (and read once it is complete), unreadable lines and
landmark chunks are left out, and every part except ``frames.jsonl`` may be
missing.

Frames are numbered by their position in ``frames.jsonl`` (the *frame index*
``n``); each also carries the browser's frame id ``i``, which names its video
image and its landmarks. ``t`` is the session clock in seconds as recorded (it
does not start at 0).
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from .gazenet import GazeNet, ModelConfig

FORMAT = 1                  # the format version this module reads (meta.json "format")
CHUNK_FRAMES = 300          # frames per landmark chunk
LIVE_S = 120.0              # an unfinished recording written to this recently is probably still going

# <YYYYmmdd-HHMMSS>-<person name>-<eyes|hand>, then -2, -3... for more started in the same second
_NAME_RE = re.compile(r"^(\d{8})-(\d{6})-(.+)-(eyes|hand)(?:-\d+)?$")
# The start of a frame line as the recorder writes it: {"i": 1234, "t": 81.532, ...
_HEAD_RE = re.compile(rb'\s*\{\s*"i"\s*:\s*(-?\d+|null)\s*,\s*"t"\s*:\s*(-?\d[0-9.eE+-]*)')
_MODEL_RE = re.compile(r"^(\d+)-(.+)\.json$")
_READ_BLOCK = 1 << 22


class RecordingError(Exception):
    """The folder is not a readable recording."""


def recordings_root(data_dir: Path) -> Path:
    """The folder that holds the recordings of a data directory.

    That is ``<data>/recordings``; a folder that itself holds recordings (or is
    one) is used as it is.
    """
    data_dir = Path(data_dir)
    sub = data_dir / "recordings"
    if sub.is_dir() or not data_dir.is_dir():
        return sub
    if is_recording(data_dir) or any(is_recording(p) for p in _subdirs(data_dir)):
        return data_dir
    return sub


def is_recording(path: Path) -> bool:
    path = Path(path)
    return (path / "frames.jsonl").is_file() or (path / "meta.json").is_file()


def _subdirs(path: Path) -> list[Path]:
    try:
        return [Path(e.path) for e in os.scandir(path) if e.is_dir()]
    except OSError:
        return []


def parse_name(name: str) -> dict:
    """What a recording's folder name says: start time, person and mode (empty if it says nothing)."""
    m = _NAME_RE.match(name)
    if not m:
        return {}
    d, t, person, mode = m.groups()
    return {"started": f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}", "person": person, "mode": mode}


def list_recordings(root: Path) -> list[dict]:
    """Summaries of the recordings in ``root`` (or of ``root`` itself if it is one), newest first."""
    root = Path(root)
    folders = [root] if is_recording(root) else [p for p in _subdirs(root) if is_recording(p)]
    out = []
    for path in folders:
        try:
            out.append(Recording(path).summary())
        except (RecordingError, OSError) as exc:
            out.append({"id": path.name, "path": str(path), "error": str(exc)})
    return sorted(out, key=lambda s: (s.get("started") or "", s["id"]), reverse=True)


def _read_json(path: Path) -> Optional[dict]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _number(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


# Drawing aids the page asks for in some frames (session.py: the camera preview's
# eye outlines; the calibration's face mesh, which the recorder keeps only as
# "mesh": true): the bulk of a frame line, and not a measurement. Each is found
# by its key, its last key and how it ends.
OVERLAY_FIELDS = ("mesh", "eyes")
_OVERLAYS = ((b'"mesh"', b'"iris"', b"]]}"), (b'"eyes"', b'"box"', b"]}"))


def _strip_overlays(raw: bytes) -> bytes:
    """A frame line without its overlay objects (parsing what is left is much faster).

    Only cuts what it recognises - an object, its brackets balanced, written
    compactly or with spaces - and leaves anything else (the caller parses the
    whole line if the result is not valid JSON).
    """
    for key, last, end in _OVERLAYS:
        a = raw.find(key)
        if a < 0:
            continue
        v = a + len(key)
        while raw[v:v + 1] in (b":", b" "):
            v += 1
        if raw[v:v + 1] != b"{":
            continue                        # not an object (e.g. "mesh": true)
        b = raw.find(last, v)
        c = raw.find(end, b) if b >= 0 else -1
        if c < 0:
            continue
        c += len(end)
        part = raw[v:c]
        if part.count(b"{") != part.count(b"}") or part.count(b"[") != part.count(b"]"):
            continue
        before = raw[:a].rstrip(b" ")
        if before.endswith(b","):           # "...": 1, "eyes": {...}
            raw = before[:-1] + raw[c:]
        else:                               # the first key: {"eyes": {...}, "...": 1}
            after = raw[c:].lstrip(b" ")
            raw = before + (after[1:].lstrip(b" ") if after.startswith(b",") else after)
    return raw


# ---------------------------------------------------------------------------
# frames.jsonl: the line index
# ---------------------------------------------------------------------------

class _FrameIndex:
    """Where each complete line of ``frames.jsonl`` starts, its length, time and frame id.

    ``update`` reads only what was appended since the last call. A last line
    without its newline is a write in progress or a crash: it is indexed only
    if it is a complete JSON object, and read again on the next update. The
    four arrays are replaced together (``arrays``), so a reader on another
    thread always sees a consistent set.
    """

    def __init__(self, path: Path):
        self.path = path
        self.reset()

    @property
    def starts(self) -> np.ndarray:
        return self.arrays[0]

    @property
    def lengths(self) -> np.ndarray:
        return self.arrays[1]

    @property
    def t(self) -> np.ndarray:
        return self.arrays[2]

    @property
    def ids(self) -> np.ndarray:
        return self.arrays[3]

    def reset(self) -> None:
        self._parts: list[tuple[np.ndarray, ...]] = []
        self.arrays = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float64), np.zeros(0, np.int64))
        self.done = 0           # bytes covered by complete lines
        self.bad = 0            # unreadable lines left out
        self.tail = False       # the last entry is a line without its newline
        self.sorted_ids = True
        self._stamp: Optional[tuple] = None

    def __len__(self) -> int:
        return len(self.arrays[2])

    def update(self) -> bool:
        """Index what was appended. Returns True if anything changed."""
        try:
            st = os.stat(self.path)
        except OSError:
            if self._stamp is not None:
                self.reset()
                return True
            return False
        stamp = (st.st_size, st.st_mtime_ns)
        if stamp == self._stamp:
            return False
        if st.st_size < self.done:         # replaced or truncated: start again
            self.reset()
        self._stamp = stamp
        starts, lengths, ts, ids = [], [], [], []
        tail = None
        with open(self.path, "rb") as fh:
            fh.seek(self.done)
            offset = self.done
            pending = b""
            while True:
                data = fh.read(_READ_BLOCK)
                if not data:
                    break
                buf = pending + data if pending else data
                base = offset - len(pending)
                offset += len(data)
                nl = np.flatnonzero(np.frombuffer(buf, np.uint8) == 10)
                begin = 0
                for end in nl.tolist():
                    head = self._parse(buf, begin, end)
                    if head is None:
                        if end > begin and buf[begin:end].strip():
                            self.bad += 1
                    else:
                        starts.append(base + begin)
                        lengths.append(end - begin)
                        ids.append(head[0])
                        ts.append(head[1])
                    begin = end + 1
                pending = buf[begin:]
            self.done = offset - len(pending)
            if pending.strip():
                head = self._parse(pending, 0, len(pending), strict=True)
                if head is not None:
                    tail = (self.done, len(pending), head)
        if self.tail:                       # the old unterminated line is read again above
            self._parts.pop()
            self.tail = False
        if starts:
            self._parts.append((np.array(starts, np.int64), np.array(lengths, np.int64),
                                np.array(ts, np.float64), np.array(ids, np.int64)))
        if tail is not None:
            start, length, (i, t) = tail
            self._parts.append((np.array([start], np.int64), np.array([length], np.int64),
                                np.array([t], np.float64), np.array([i], np.int64)))
            self.tail = True
        if self._parts:
            arrays = tuple(np.concatenate(c) for c in zip(*self._parts))
            self._parts = [arrays]
            if self.tail:                   # keep the unterminated line separate, so it can be dropped
                k = len(arrays[0]) - 1
                self._parts = [tuple(a[:k] for a in arrays), tuple(a[k:] for a in arrays)]
        else:
            arrays = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float64), np.zeros(0, np.int64))
        ids = arrays[3]
        self.sorted_ids = bool(len(ids) < 2 or np.all(np.diff(ids) > 0))
        self.arrays = arrays
        return True

    @staticmethod
    def _parse(buf: bytes, begin: int, end: int, strict: bool = False) -> Optional[tuple[int, float]]:
        """(frame id, t) of the line ``buf[begin:end]``, or None if it is not a complete frame line."""
        stop = end
        while stop > begin and buf[stop - 1] in b" \t\r":
            stop -= 1
        if stop <= begin:
            return None
        m = None if strict else _HEAD_RE.match(buf, begin, min(stop, begin + 200))
        if m is not None and buf[stop - 1:stop] == b"}":
            try:
                return (-1 if m.group(1) == b"null" else int(m.group(1))), float(m.group(2))
            except ValueError:
                pass
        try:                                # another key order, or a line to check fully
            doc = json.loads(buf[begin:stop])
        except ValueError:
            return None
        if not isinstance(doc, dict) or _number(doc.get("t")) is None:
            return None
        i = doc.get("i")
        return (int(i) if isinstance(i, int) and not isinstance(i, bool) else -1), float(doc["t"])


class _Lines:
    """A small JSON-lines file read whole (events.jsonl), re-read when it changes."""

    def __init__(self, path: Path):
        self.path = path
        self.items: list[dict] = []
        self.bad = 0
        self._stamp: Optional[tuple] = None

    def update(self) -> bool:
        try:
            st = os.stat(self.path)
        except OSError:
            changed = self._stamp is not None
            self.items, self.bad, self._stamp = [], 0, None
            return changed
        stamp = (st.st_size, st.st_mtime_ns)
        if stamp == self._stamp:
            return False
        items, bad = [], 0
        with open(self.path, "rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    doc = json.loads(line)
                except ValueError:
                    bad += 1                # the half-written last line of a crash
                    continue
                if isinstance(doc, dict):
                    items.append(doc)
        self.items, self.bad, self._stamp = items, bad, stamp
        return True


# ---------------------------------------------------------------------------
# landmarks/*.npz
# ---------------------------------------------------------------------------

class _Landmarks:
    """Which chunk holds each frame's landmarks, and an LRU cache of loaded chunks.

    ``table`` is (frame ids sorted, (chunk, row) of each, chunk files), replaced
    as a whole when chunks are added, so readers on other threads see one state.
    """

    def __init__(self, folder: Path, cache: int = 6):
        self.folder = folder
        self.table: tuple = (np.zeros(0, np.int64), np.zeros((0, 2), np.int64), [])
        self.bad: list[str] = []
        self._known: dict[str, tuple] = {}       # file name -> (size, mtime, ids)
        self._cache: OrderedDict = OrderedDict()  # chunk file -> its arrays
        self._size = cache
        self._lock = threading.Lock()

    @property
    def ids(self) -> np.ndarray:
        return self.table[0]

    @property
    def files(self) -> list[Path]:
        return self.table[2]

    def update(self) -> bool:
        try:
            entries = sorted((e for e in os.scandir(self.folder) if e.name.endswith(".npz")), key=lambda e: e.name)
        except OSError:
            entries = []
        changed = False
        known = {}
        for e in entries:
            try:
                st = e.stat()
            except OSError:
                continue
            old = self._known.get(e.name)
            if old is not None and old[:2] == (st.st_size, st.st_mtime_ns):
                known[e.name] = old
                continue
            changed = True
            try:
                with np.load(e.path) as z:
                    ids = np.asarray(z["ids"], np.int64).ravel()
            except Exception:               # half-written (a crash) or not a chunk
                known[e.name] = (st.st_size, st.st_mtime_ns, None)
                continue
            known[e.name] = (st.st_size, st.st_mtime_ns, ids)
        if set(known) != set(self._known):
            changed = True
        if not changed:
            return False
        self._known = known
        files = [self.folder / name for name in known]
        self.bad = [name for name, v in known.items() if v[2] is None]
        ids, where = [], []
        for k, name in enumerate(known):
            chunk_ids = known[name][2]
            if chunk_ids is None:
                continue
            ids.append(chunk_ids)
            where.append(np.stack([np.full(len(chunk_ids), k), np.arange(len(chunk_ids))], axis=1))
        if ids:
            all_ids = np.concatenate(ids)
            order = np.argsort(all_ids, kind="stable")
            self.table = (all_ids[order], np.concatenate(where)[order], files)
        else:
            self.table = (np.zeros(0, np.int64), np.zeros((0, 2), np.int64), files)
        with self._lock:
            self._cache.clear()
        return True

    def _chunk(self, path: Path) -> Optional[dict]:
        with self._lock:
            if path in self._cache:
                self._cache.move_to_end(path)
                return self._cache[path]
        try:
            with np.load(path) as z:
                chunk = {name: np.asarray(z[name]) for name in z.files}
        except Exception:
            return None
        with self._lock:
            self._cache[path] = chunk
            while len(self._cache) > self._size:
                self._cache.popitem(last=False)
        return chunk

    def get(self, frame_id: int) -> Optional[dict]:
        ids, where, files = self.table
        k = int(np.searchsorted(ids, frame_id))
        if k >= len(ids) or int(ids[k]) != int(frame_id):
            return None
        chunk_no, row = (int(v) for v in where[k])
        chunk = self._chunk(files[chunk_no])
        if chunk is None or "points" not in chunk or row >= len(chunk["points"]):
            return None
        out = {"points": np.asarray(chunk["points"][row], np.float32), "chunk": files[chunk_no].name}
        others = chunk.get("others")
        if others is not None and row < len(others):
            boxes = np.asarray(others[row], np.float32).reshape(-1, 4)
            boxes = boxes[np.isfinite(boxes).all(axis=1)]
            out["others"] = boxes
        return out


# ---------------------------------------------------------------------------
# models/NNN-<why>.json
# ---------------------------------------------------------------------------

@dataclass
class ModelSnapshot:
    """One saved gaze model and when it came into force."""

    number: int
    why: str
    name: str                       # file name
    doc: dict                       # GazeNet.to_dict()
    t: float = -math.inf            # session time it came into force (-inf: from the start)
    t_source: str = "start"         # how ``t`` was found (see Recording._place_models)
    extra: dict = field(default_factory=dict)   # whatever else the file held besides the model
    _net: Any = None

    @property
    def net(self):
        """The model as a :class:`paralic.gazenet.GazeNet` (built on first use)."""
        if self._net is None:
            self._net = GazeNet.from_dict(self.doc)
        return self._net

    def info(self) -> dict:
        meta = self.doc.get("meta") or {}
        return {
            "number": self.number, "why": self.why, "name": self.name,
            "t": None if not math.isfinite(self.t) else self.t, "t_source": self.t_source,
            "members": len(self.doc.get("nets") or []),
            "config": ModelConfig.from_dict(meta["config"]).name if isinstance(meta.get("config"), dict) else None,
            "eyes": sorted((self.doc.get("eyes") or {}).keys()),
            "eye": meta.get("eye"),
            "cv_error_px": meta.get("cv_error_px"),
            "eye_cv_px": meta.get("eye_cv_px"),
            "version": meta.get("version"),
            "source": meta.get("source"),
            "trained_at": meta.get("trained_at"),
            "corrected": _corrected(self.doc),
        }


def _corrected(doc: dict) -> bool:
    c = doc.get("correction") or {}
    try:
        return not (np.allclose(np.asarray(c.get("A", np.eye(2)), float), np.eye(2))
                    and np.allclose(np.asarray(c.get("b", [0, 0]), float), 0.0))
    except (TypeError, ValueError):
        return False


def _is_model(doc) -> bool:
    return isinstance(doc, dict) and isinstance(doc.get("nets"), list) and "x_scaler" in doc


# A model file says why it was saved; the reply or push that announced that
# change tells when (used when the file does not say).
_MODEL_EVENTS = {
    "calibration": ("calibration_result", lambda d: d.get("mode", "full") != "adjust"),
    "refit": ("validation_result", lambda d: bool(d.get("refit"))),
    "refine": ("validation_result", lambda d: bool(d.get("refine") or d.get("refit"))),
    "adjust": ("calibration_result", lambda d: d.get("mode") == "adjust"),
    "finetune": ("finetune_result", lambda d: d.get("version") is not None),
    "fine-tune": ("finetune_result", lambda d: d.get("version") is not None),
    "profile_load": ("profile", lambda d: bool(d.get("loaded"))),
    "profile": ("profile", lambda d: bool(d.get("loaded"))),
}


# ---------------------------------------------------------------------------
# The recording
# ---------------------------------------------------------------------------

class Recording:
    """One recording folder (see the module docstring)."""

    def __init__(self, path: Path, *, refresh_s: float = 1.0):
        self.path = Path(path)
        if not self.path.is_dir():
            raise RecordingError(f"{self.path} is not a folder")
        if not is_recording(self.path):
            raise RecordingError(f"{self.path} has no meta.json or frames.jsonl")
        self.id = self.path.name
        self.refresh_s = refresh_s
        self._lock = threading.RLock()
        self._index = _FrameIndex(self.path / "frames.jsonl")
        self._events = _Lines(self.path / "events.jsonl")
        self._landmarks = _Landmarks(self.path / "landmarks")
        self._video: np.ndarray = np.zeros(0, np.int64)
        self._video_stamp: Optional[int] = None
        self._meta: dict = {}
        self._meta_stamp: Optional[tuple] = None
        self._models: list[ModelSnapshot] = []
        self._changes: list[tuple[float, Optional[ModelSnapshot]]] = []
        self._models_stamp: Optional[tuple] = None
        self._checked = 0.0
        self.version = 0                    # bumped whenever something changed (for caches built on top)
        self.refresh(force=True)

    # -- staying up to date --------------------------------------------------------
    def refresh(self, force: bool = False) -> bool:
        """Pick up what was written since the last look (at most every ``refresh_s``). True if anything changed."""
        now = time.monotonic()
        if not force and now - self._checked < self.refresh_s:
            return False
        with self._lock:
            self._checked = now
            changed = self._update_meta()
            changed |= self._index.update()
            changed |= self._events.update()
            changed |= self._landmarks.update()
            changed |= self._update_video()
            # New events or frames can tell when a model came into force: place them again.
            changed |= self._update_models(force=changed)
            if changed:
                self.version += 1
            return changed

    def _update_meta(self) -> bool:
        path = self.path / "meta.json"
        try:
            st = os.stat(path)
            stamp = (st.st_size, st.st_mtime_ns)
        except OSError:
            stamp = None
        if stamp == self._meta_stamp:
            return False
        self._meta_stamp = stamp
        doc = _read_json(path) if stamp else None
        if doc is None and stamp is not None and self._meta:
            return False                    # being rewritten right now: keep the last good one
        self._meta = doc or {}
        return True

    def _update_video(self) -> bool:
        folder = self.path / "video"
        try:
            stamp = os.stat(folder).st_mtime_ns
        except OSError:
            stamp = None
        if stamp == self._video_stamp:
            return False
        self._video_stamp = stamp
        ids = []
        if stamp is not None:
            for e in os.scandir(folder):
                stem, _, ext = e.name.rpartition(".")
                if ext.lower() in ("jpg", "jpeg") and stem.lstrip("-").isdigit():
                    ids.append(int(stem))
        self._video = np.array(sorted(ids), np.int64)
        return True

    def _update_models(self, force: bool = False) -> bool:
        folder = self.path / "models"
        try:
            entries = sorted(e.name for e in os.scandir(folder) if _MODEL_RE.match(e.name))
            stamp = tuple((name, os.stat(folder / name).st_mtime_ns) for name in entries)
        except OSError:
            entries, stamp = [], ()
        if stamp == self._models_stamp and not force:
            return False
        changed = stamp != self._models_stamp
        self._models_stamp = stamp
        old = {m.name: m for m in self._models}
        models = []
        for name in entries:
            number, why = _MODEL_RE.match(name).groups()
            prev = old.get(name)
            if prev is not None and not changed:
                models.append(prev)
                continue
            doc = _read_json(folder / name)
            if doc is None:
                continue                    # half-written: read it next time
            extra = {}
            if not _is_model(doc) and _is_model(doc.get("model")):
                extra = {k: v for k, v in doc.items() if k != "model"}
                doc = doc["model"]
            elif _is_model(doc):
                extra = {k: v for k, v in doc.items()
                         if k not in ("x_scaler", "y_scaler", "nets", "correction", "meta", "inputs", "eyes")}
            else:
                continue
            snap = ModelSnapshot(number=int(number), why=why, name=name, doc=doc, extra=extra)
            try:
                snap.extra.setdefault("_mtime", os.stat(folder / name).st_mtime)
            except OSError:
                pass
            models.append(snap)
        self._models = models
        self._place_models()
        return True

    # -- meta ----------------------------------------------------------------------
    @property
    def meta(self) -> dict:
        """``meta.json`` (empty if it is missing or unreadable)."""
        return self._meta

    @property
    def mode(self) -> str:
        mode = self._meta.get("mode") or parse_name(self.id).get("mode")
        return "hand" if mode == "hand" else "eyes"

    @property
    def person(self) -> Optional[str]:
        person = self._meta.get("person")
        if isinstance(person, dict) and person.get("name"):
            return str(person["name"])
        return parse_name(self.id).get("person")

    @property
    def screen(self) -> dict:
        """The screen size in CSS pixels (from meta.json, or the hello command, or 1920x1080)."""
        s = self._meta.get("screen")
        if not (isinstance(s, dict) and _number(s.get("w")) and _number(s.get("h"))):
            s = None
            for ev in self.events():
                d = ev.get("data") or {}
                if ev.get("type") == "hello" and isinstance(d.get("screen"), dict):
                    s = d["screen"]
                    break
        s = s or {}
        return {"w": _number(s.get("w")) or 1920.0, "h": _number(s.get("h")) or 1080.0,
                "dpr": _number(s.get("dpr")) or 1.0}

    # -- frames --------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index.t)

    @property
    def times(self) -> np.ndarray:
        """Session time of every frame (by frame index)."""
        return self._index.t

    @property
    def frame_ids(self) -> np.ndarray:
        """Browser frame id of every frame (by frame index; -1 where it had none)."""
        return self._index.ids

    @property
    def bad_lines(self) -> dict:
        return {"frames": self._index.bad, "events": self._events.bad}

    @property
    def truncated(self) -> bool:
        """True if ``frames.jsonl`` ends in an unterminated line (a write in progress, or a crash)."""
        try:
            return os.path.getsize(self._index.path) > self._index.done and not self._index.tail
        except OSError:
            return False

    def span(self) -> tuple[Optional[float], Optional[float]]:
        """First and last time in the recording (frames, else events)."""
        t = self._index.t
        if len(t):
            return float(t[0]), float(t[-1])
        ts = [e["t"] for e in self._events.items if _number(e.get("t")) is not None]
        return (min(ts), max(ts)) if ts else (None, None)

    def _read(self, lo: int, hi: int) -> list[bytes]:
        if lo >= hi:
            return []
        starts, lengths = self._index.arrays[:2]
        a = int(starts[lo])
        b = int(starts[hi - 1] + lengths[hi - 1])
        with open(self._index.path, "rb") as fh:
            fh.seek(a)
            buf = fh.read(b - a)
        return [buf[int(s) - a:int(s) - a + int(n)] for s, n in zip(starts[lo:hi], lengths[lo:hi])]

    def _decode(self, n: int, raw: bytes, light: bool = False) -> dict:
        doc = None
        if light:
            try:
                doc = json.loads(_strip_overlays(raw))
            except ValueError:
                doc = None
        if doc is None:
            try:
                doc = json.loads(raw)
            except ValueError:
                doc = None
        if not isinstance(doc, dict):      # damaged in the middle of the file: keep its place
            _, _, t, ids = self._index.arrays
            return {"i": int(ids[n]), "t": float(t[n]), "msg": {}, "damaged": True}
        if light and isinstance(doc.get("msg"), dict):
            for key in OVERLAY_FIELDS:
                doc["msg"].pop(key, None)
        return doc

    def frame(self, n: int) -> dict:
        """The frame at index ``n`` (negative counts from the end)."""
        count = len(self)
        if n < 0:
            n += count
        if not 0 <= n < count:
            raise IndexError(f"frame index {n} out of range (0..{count - 1})")
        return self._decode(n, self._read(n, n + 1)[0])

    def frames(self, start: int = 0, stop: Optional[int] = None, light: bool = False) -> list[dict]:
        """Frames ``start`` .. ``stop - 1`` (clamped to the recording).

        ``light`` leaves out the frame message's bulky drawing aids (the face
        mesh and the eye overlay the page asked for, see OVERLAY_FIELDS),
        which also makes reading several times faster.
        """
        count = len(self)
        stop = count if stop is None else min(int(stop), count)
        start = max(0, int(start))
        return [self._decode(start + k, raw, light) for k, raw in enumerate(self._read(start, stop))]

    def iter_frames(self, start: int = 0, stop: Optional[int] = None, block: int = 2000,
                    light: bool = False) -> Iterator[dict]:
        """All frames from ``start``, read ``block`` at a time."""
        stop = len(self) if stop is None else min(int(stop), len(self))
        for lo in range(max(0, int(start)), stop, block):
            yield from self.frames(lo, min(lo + block, stop), light)

    def index_at(self, t: float) -> int:
        """Index of the frame on screen at time ``t``: the last one at or before it (0 before the first)."""
        n = int(np.searchsorted(self._index.t, t, side="right")) - 1
        return min(max(n, 0), max(len(self) - 1, 0))

    def range_of(self, t0: float, t1: float) -> tuple[int, int]:
        """Frame indices ``(start, stop)`` of the frames with ``t0 <= t <= t1``."""
        t = self._index.t
        return int(np.searchsorted(t, t0, side="left")), int(np.searchsorted(t, t1, side="right"))

    def frames_between(self, t0: float, t1: float) -> list[dict]:
        return self.frames(*self.range_of(t0, t1))

    def find(self, frame_id: int) -> Optional[int]:
        """Frame index of the frame with browser frame id ``frame_id``."""
        ids = self._index.ids
        if self._index.sorted_ids:
            k = int(np.searchsorted(ids, frame_id))
            return k if k < len(ids) and int(ids[k]) == int(frame_id) else None
        hits = np.flatnonzero(ids == int(frame_id))
        return int(hits[0]) if len(hits) else None

    # -- events --------------------------------------------------------------------
    def events(self) -> list[dict]:
        """Every event (see the format), in file order."""
        return self._events.items

    # -- landmarks -----------------------------------------------------------------
    def landmarks(self, frame_id: int) -> Optional[dict]:
        """The frame's tracked points: ``points`` (P, 3) - face-mesh pixels in eye mode, hand
        landmarks in image fractions in hand mode - and ``others`` (k, 4): other faces' boxes
        (if recorded). None if the frame had no face or hand, or its chunk is unreadable."""
        return self._landmarks.get(int(frame_id))

    def landmark_ids(self) -> np.ndarray:
        return self._landmarks.ids

    @property
    def landmark_chunks(self) -> dict:
        return {"files": len(self._landmarks.files), "unreadable": list(self._landmarks.bad)}

    # -- video ---------------------------------------------------------------------
    def video_ids(self) -> np.ndarray:
        """Frame ids that have a camera image, sorted."""
        return self._video

    def video_frame(self, frame_id: int, earlier: bool = True) -> Optional[int]:
        """The frame id whose image shows frame ``frame_id``: its own, or (``earlier``) the nearest earlier one."""
        ids = self._video
        k = int(np.searchsorted(ids, frame_id, side="right")) - 1
        if k < 0:
            return None
        if int(ids[k]) == int(frame_id) or earlier:
            return int(ids[k])
        return None

    def video(self, frame_id: int, earlier: bool = True) -> Optional[tuple[int, bytes]]:
        """(frame id, JPEG bytes) of the image for frame ``frame_id`` (see :meth:`video_frame`)."""
        vid = self.video_frame(frame_id, earlier)
        if vid is None:
            return None
        try:
            return vid, (self.path / "video" / f"{vid}.jpg").read_bytes()
        except OSError:
            try:
                return vid, (self.path / "video" / f"{vid}.jpeg").read_bytes()
            except OSError:
                return None

    # -- models --------------------------------------------------------------------
    def models(self) -> list[ModelSnapshot]:
        """The saved gaze models in the order they came into force."""
        return self._models

    def model_changes(self) -> list[tuple[float, Optional[ModelSnapshot]]]:
        """Every change of the gaze model in force, in order: (time, model), where the model
        is None from a ``model`` event without a file on (another person, without one, was chosen)."""
        return self._changes

    def model_at(self, t: float) -> Optional[ModelSnapshot]:
        """The model in force at time ``t`` (None before the first one, or while there is none)."""
        best = None
        for when, m in self._changes:
            if when > t:
                break
            best = m
        return best

    def wall_to_t(self, wall: float, pairs: Optional[list] = None) -> Optional[float]:
        """Session time of a Unix time, from the nearest (wall, t) pair of the frames and events."""
        pairs = self._clock_pairs() if pairs is None else pairs
        if not pairs:
            return None
        walls = [p[0] for p in pairs]
        k = int(np.searchsorted(walls, wall))
        if k >= len(walls) or (k > 0 and wall - walls[k - 1] < walls[k] - wall):
            k -= 1
        return float(pairs[k][1] + (wall - pairs[k][0]))

    def _clock_pairs(self) -> list[tuple[float, float]]:
        pairs = []
        clock = self._meta.get("clock")
        if isinstance(clock, dict) and _number(clock.get("wall")) is not None and _number(clock.get("t")) is not None:
            pairs.append((float(clock["wall"]), float(clock["t"])))
        if len(self):
            for n in sorted({0, len(self) - 1}):
                f = self.frame(n)
                if _number(f.get("wall")) is not None and _number(f.get("t")) is not None:
                    pairs.append((float(f["wall"]), float(f["t"])))
        for e in self._events.items:
            if _number(e.get("wall")) is not None and _number(e.get("t")) is not None:
                pairs.append((float(e["wall"]), float(e["t"])))
        return sorted(pairs)

    def _place_models(self) -> None:
        """Work out when each saved model came into force.

        Normally the ``model`` event that names the file says so. Otherwise, in
        this order: a time in the file (``t``, ``wall``, or the frame id ``i`` /
        ``frame``); another event naming it (a type with "model" in it); the
        reply or push that reported the change (``calibration_result`` for
        "calibration", ``validation_result`` for "refit", ...); the training
        time in its meta; the file's modification time. The model saved first
        (or "start") is in force from the start; a model nothing places stays
        in force from the one before it. A ``model`` event without a file
        means no model from then on.
        """
        pairs = self._clock_pairs()
        wall_span = (pairs[0][0] - 2.0, pairs[-1][0] + 2.0) if pairs else None
        lo, hi = self.span()
        events = sorted((e for e in self._events.items if _number(e.get("t")) is not None),
                        key=lambda e: e["t"])
        used: set[int] = set()
        order: dict[str, int] = {}          # the event that placed a model (orders changes at one time)
        previous = -math.inf
        def naming(m: ModelSnapshot, strict: bool) -> Optional[int]:
            """The first unused event that names the file (``strict``: a ``model`` event)."""
            for j, e in enumerate(events):
                typ = str(e.get("type", ""))
                if j in used or (typ != "model" if strict else "model" not in typ):
                    continue
                d = e.get("data") if isinstance(e.get("data"), dict) else {}
                names = {str(d.get(key)) for key in ("file", "name", "path") if d.get(key)}
                if m.name in names or any(n_.endswith("/" + m.name) for n_ in names) or \
                        (not strict and d.get("number") == m.number and d.get("why", m.why) == m.why):
                    return j
            return None

        for k, m in enumerate(sorted(self._models, key=lambda m: (m.number, m.name))):
            t, source = None, None
            x = m.extra
            j = naming(m, strict=True)
            if j is not None:
                t, source = float(events[j]["t"]), "event"
            elif _number(x.get("t")) is not None:
                t, source = float(x["t"]), "file"
            elif _number(x.get("wall")) is not None and pairs:
                t, source = self.wall_to_t(float(x["wall"]), pairs), "file"
            else:
                fid = x.get("frame", x.get("i"))
                n = self.find(int(fid)) if isinstance(fid, int) and not isinstance(fid, bool) else None
                if n is not None:
                    t, source = float(self._index.t[n]), "file"
                else:
                    j = naming(m, strict=False)
                    if j is not None:
                        t, source = float(events[j]["t"]), "event"
            if j is not None:
                used.add(j)
                order[m.name] = j
            if t is None and m.why in _MODEL_EVENTS:
                kind, accept = _MODEL_EVENTS[m.why]
                for j, e in enumerate(events):
                    d = e.get("data") if isinstance(e.get("data"), dict) else {}
                    if (j not in used and e.get("type") == kind and e["t"] >= previous
                            and d.get("ok", True) is not False and accept(d)):
                        t, source = float(e["t"]), "reply"
                        used.add(j)
                        break
            if t is None and pairs:
                trained = _number((m.doc.get("meta") or {}).get("trained_ts"))
                mtime = _number(x.get("_mtime"))
                for wall, label in ((trained, "trained"), (mtime, "mtime")):
                    if wall is not None and wall_span[0] <= wall <= wall_span[1]:
                        t, source = self.wall_to_t(wall, pairs), label
                        break
            if m.why == "start" or (t is None and k == 0):
                t, source = -math.inf, "start"
            elif t is None:
                t, source = previous, "order"
            elif k == 0 and lo is not None and t <= lo:
                t = -math.inf               # in force before the first frame
            m.t, m.t_source = t, source
            previous = max(previous, t)
        self._models.sort(key=lambda m: (m.t, m.number))
        changes = [(m.t, order.get(m.name, -1), m) for m in self._models]
        for j, e in enumerate(events):
            d = e.get("data")
            if e.get("type") == "model" and isinstance(d, dict) and "file" in d and not d["file"]:
                changes.append((float(e["t"]), j, None))
        changes.sort(key=lambda c: (c[0], c[1]))
        self._changes = [(t, m) for t, _, m in changes]

    # -- summary -------------------------------------------------------------------
    def glasses_seen(self) -> Optional[bool]:
        """True if any frame says the person wore glasses, False if frames say so but never did,
        None if the frames do not carry ``glasses`` (recorded before glasses were noticed)."""
        for e in self._events.items:
            d = e.get("data") or {}
            if e.get("type") == "glasses_changed" and isinstance(d, dict) and d.get("glasses"):
                return True
        found = None
        try:
            with open(self._index.path, "rb") as fh:
                carry = b""
                while True:
                    data = fh.read(_READ_BLOCK)
                    if not data:
                        break
                    buf = carry + data
                    if b'"glasses": true' in buf or b'"glasses":true' in buf:
                        return True
                    if found is None and b'"glasses":' in buf:
                        found = False
                    carry = buf[-24:]
        except OSError:
            return None
        return found

    def accuracy(self) -> Optional[dict]:
        """The last accuracy measured in the recording (a validation), else None."""
        best = None
        for e in self._events.items:
            d = e.get("data")
            if e.get("type") == "validation_result" and isinstance(d, dict) and d.get("ok", True) is not False:
                err = _number(d.get("mean_error_px"))
                if err is not None:
                    best = {"mean_error_px": round(err, 1), "t": e.get("t"),
                            "precision_px": _number(d.get("precision_px"))}
        return best

    def _written_since(self, seconds: float) -> bool:
        """True if the recording's files changed in the last ``seconds`` (the recorder
        writes its buffers out and refreshes meta.json at least every minute)."""
        newest = 0.0
        for name in ("frames.jsonl", "events.jsonl", "meta.json"):
            try:
                newest = max(newest, os.stat(self.path / name).st_mtime)
            except OSError:
                pass
        return time.time() - newest < seconds

    def size_bytes(self) -> int:
        total = 0
        for root, _dirs, files in os.walk(self.path):
            for name in files:
                try:
                    total += os.stat(os.path.join(root, name)).st_size
                except OSError:
                    pass
        return total

    def summary(self) -> dict:
        """What the list of recordings shows."""
        meta = self._meta
        named = parse_name(self.id)
        t0, t1 = self.span()
        ended = meta.get("ended")
        # "ended" is refreshed every minute while recording; "complete" says it stopped cleanly.
        complete = meta.get("complete") if isinstance(meta.get("complete"), bool) else bool(ended)
        summary = meta.get("summary") if isinstance(meta.get("summary"), dict) else {}
        size = summary.get("bytes") if complete and _number(summary.get("bytes")) else self.size_bytes()
        person = meta.get("person") if isinstance(meta.get("person"), dict) else {}
        video = meta.get("video") if isinstance(meta.get("video"), dict) else {}
        fmt = meta.get("format")
        return {
            "id": self.id,
            "path": str(self.path),
            "format": fmt,
            "newer_format": isinstance(fmt, int) and fmt > FORMAT,   # made by a newer Paralic: may read partly
            "paralic": meta.get("paralic"),
            "person": self.person,
            "person_id": person.get("id"),
            "mode": self.mode,
            "started": meta.get("started") or named.get("started"),
            "ended": ended,
            "complete": complete,
            "live": not complete and self._written_since(LIVE_S),
            "recorder_error": meta.get("error"),    # why the recorder stopped by itself
            "duration_s": None if t0 is None else round(t1 - t0, 3),
            "frames": len(self),
            "dropped": summary.get("dropped"),      # frames left out: the disk could not keep up
            "video_frames": int(len(self._video)),
            "video_every": video.get("every"),
            "video_stopped": video.get("stopped"),
            "landmark_frames": int(len(self._landmarks.ids)),
            "events": len(self._events.items),
            "models": len(self._models),
            "bytes": int(size),
            "glasses": self.glasses_seen(),
            "accuracy": self.accuracy(),
            "screen": self.screen,
            "truncated": self.truncated,
            "bad_lines": self.bad_lines,
        }
