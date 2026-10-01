"""The Inspector's local web server: a JSON API over the recordings and the page in ``static/``.

Everything is read from the recordings folder on demand (see
``paralic/recording.py``): the page asks for the frames, signals and video
images it shows, never for a whole recording, so hour-long recordings stay
quick. The server only reads; it listens on 127.0.0.1 (see ``__main__.py``).

API (``{rid}`` is a recording's folder name; times are the session clock):

* ``GET /api/recordings`` - the recordings, with their summaries
* ``GET /api/recordings/{rid}`` - summary, meta.json, models, screen, image size
* ``GET /api/recordings/{rid}/index`` - every frame's time and frame id
* ``GET /api/recordings/{rid}/frames?start=&count=&light=`` - frames by index
* ``GET /api/recordings/{rid}/frame/{n}`` - one frame with landmarks and model internals
* ``GET /api/recordings/{rid}/video/{frame_id}`` - its JPEG (or the nearest earlier one;
  header ``X-Frame-Id``)
* ``GET /api/recordings/{rid}/signals?t0=&t1=&points=`` - signal columns over time
* ``GET /api/recordings/{rid}/timeline`` - markers and stretches for the scrubber
* ``GET /api/recordings/{rid}/events`` - every event
* ``GET /api/recordings/{rid}/calibrations`` - calibrations, their dots, fits and accuracy
* ``GET /api/recordings/{rid}/config`` - meta.json and every settings change
* ``GET /api/recordings/{rid}/models/{name}`` - one saved model as recorded
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..recording import LIVE_S, Recording, RecordingError, is_recording, recordings_root
from ..server import host_allowed
from .views import RecordingView, jsonable

STATIC_DIR = Path(__file__).resolve().parent / "static"
LIGHT_DROP = ("mesh", "eyes", "light")       # bulky frame fields the timeline windows do not need


def _strict_json(obj) -> Response:
    """JSON a browser can parse: recorded NaN or infinity (not strict JSON) become null."""
    try:
        body = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    except ValueError:
        body = json.dumps(jsonable(obj, 6), separators=(",", ":"), allow_nan=False)
    return Response(body, media_type="application/json")


class RecordingStore:
    """Open recordings (one view each, kept while the server runs) and the list's summaries."""

    def __init__(self, root: Path, keep: int = 4):
        self.root = Path(root)
        self.keep = keep
        self._views: dict[str, RecordingView] = {}
        self._order: list[str] = []
        self._summaries: dict[str, tuple] = {}
        self._lock = threading.Lock()

    def folder(self, rid: str) -> Path:
        if not rid or rid in (".", "..") or "/" in rid or "\\" in rid or os.sep in rid:
            raise HTTPException(404, "No such recording")
        path = self.root / rid if self.root.name != rid or not is_recording(self.root) else self.root
        if not path.is_dir() or not is_recording(path):
            raise HTTPException(404, f"No recording {rid!r} in {self.root}")
        return path

    def view(self, rid: str) -> RecordingView:
        path = self.folder(rid)
        with self._lock:
            view = self._views.get(rid)
            if view is None:
                try:
                    view = RecordingView(Recording(path))
                except RecordingError as exc:
                    raise HTTPException(404, str(exc))
                self._views[rid] = view
                view.warm()                 # read the whole recording in the background
            if rid in self._order:
                self._order.remove(rid)
            self._order.append(rid)
            while len(self._order) > self.keep:
                old = self._views.pop(self._order.pop(0), None)
                if old is not None:
                    old.close()
        view.refresh()
        return view

    def _stamp(self, path: Path) -> tuple:
        out = []
        for name in ("meta.json", "frames.jsonl", "events.jsonl", "models", "video"):
            try:
                st = os.stat(path / name)
                out.append((st.st_size, st.st_mtime_ns))
            except OSError:
                out.append(None)
        return tuple(out)

    def list(self) -> list[dict]:
        """Summaries of every recording (computed again only for recordings that changed)."""
        if not self.root.is_dir():
            return []
        if is_recording(self.root):
            folders = [self.root]
        else:
            folders = [Path(e.path) for e in os.scandir(self.root) if e.is_dir() and is_recording(Path(e.path))]
        out = []
        for path in folders:
            stamp = self._stamp(path)
            cached = self._summaries.get(path.name)
            if cached is None or cached[0] != stamp:
                try:
                    summary = Recording(path).summary()
                except (RecordingError, OSError) as exc:
                    summary = {"id": path.name, "path": str(path), "error": str(exc)}
                cached = (stamp, jsonable(summary, 3))
                self._summaries[path.name] = cached
            summary = cached[1]
            if summary.get("live"):
                # Unchanged since: still recording only if that was a moment ago (see Recording.summary).
                newest = max((s[1] for s in stamp[:3] if s is not None), default=0) / 1e9
                summary = {**summary, "live": time.time() - newest < LIVE_S}
            out.append(summary)
        return sorted(out, key=lambda s: (s.get("started") or "", s["id"]), reverse=True)


def create_app(data_dir: Path, *, root: Optional[Path] = None) -> FastAPI:
    """The Inspector app for the recordings of ``data_dir`` (``<data_dir>/recordings``), or of ``root``."""
    root = Path(root) if root is not None else recordings_root(Path(data_dir))
    app = FastAPI(title="Paralic Inspector", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    store = RecordingStore(root)
    app.state.store = store

    @app.middleware("http")
    async def no_cache(request, call_next):
        # Recordings hold camera images of faces: answer only requests made to
        # this computer's own address (no other site can rebind its domain here).
        if not host_allowed(request.headers.get("host")):
            return PlainTextResponse("The Inspector answers only on this computer's own address.", status_code=403)
        response = await call_next(request)
        if request.url.path.startswith("/api/") and "/video/" not in request.url.path:
            response.headers["Cache-Control"] = "no-cache"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/api/recordings")
    def recordings() -> JSONResponse:
        return JSONResponse({"root": str(root), "version": __version__, "recordings": store.list()})

    @app.get("/api/recordings/{rid}")
    def info(rid: str) -> JSONResponse:
        return JSONResponse(store.view(rid).info())

    @app.get("/api/recordings/{rid}/index")
    def index(rid: str) -> JSONResponse:
        return JSONResponse(store.view(rid).index())

    @app.get("/api/recordings/{rid}/frames")
    def frames(rid: str, start: int = 0, count: int = Query(300, ge=1, le=5000), light: bool = False) -> Response:
        view = store.view(rid)
        out = view.rec.frames(start, start + count, light=light)
        if light:
            for f in out:
                msg = f.get("msg")
                if isinstance(msg, dict):
                    for key in LIGHT_DROP:
                        msg.pop(key, None)
        return _strict_json({"start": max(0, start), "frames": out})

    @app.get("/api/recordings/{rid}/frame/{n}")
    def frame(rid: str, n: int) -> JSONResponse:
        view = store.view(rid)
        if not 0 <= n < len(view.rec):
            raise HTTPException(404, "No such frame")
        return JSONResponse(view.detail(n))

    @app.get("/api/recordings/{rid}/video/{frame_id}")
    def video(rid: str, frame_id: int, exact: bool = False) -> Response:
        view = store.view(rid)
        found = view.rec.video(frame_id, earlier=not exact)
        if found is None:
            raise HTTPException(404, "No video for this frame")
        vid, data = found
        return Response(data, media_type="image/jpeg",
                        headers={"X-Frame-Id": str(vid), "Cache-Control": "max-age=3600"})

    @app.get("/api/recordings/{rid}/signals")
    def signals(rid: str, t0: Optional[float] = None, t1: Optional[float] = None,
                points: int = Query(2000, ge=16, le=20000)) -> JSONResponse:
        return JSONResponse(store.view(rid).signals(t0, t1, points))

    @app.get("/api/recordings/{rid}/timeline")
    def timeline(rid: str) -> JSONResponse:
        return JSONResponse(store.view(rid).timeline())

    @app.get("/api/recordings/{rid}/events")
    def events(rid: str) -> JSONResponse:
        return JSONResponse({"events": store.view(rid).events()})

    @app.get("/api/recordings/{rid}/calibrations")
    def calibrations(rid: str) -> JSONResponse:
        return JSONResponse(store.view(rid).calibrations())

    @app.get("/api/recordings/{rid}/config")
    def config(rid: str) -> JSONResponse:
        return JSONResponse(store.view(rid).config())

    @app.get("/api/recordings/{rid}/models/{name}")
    def model(rid: str, name: str) -> JSONResponse:
        for m in store.view(rid).rec.models():
            if m.name == name or str(m.number) == name:
                return JSONResponse(jsonable({**m.info(), "model": m.doc, "extra": m.extra}, 8))
        raise HTTPException(404, "No such model")

    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app
