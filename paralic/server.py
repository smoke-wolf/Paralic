"""FastAPI web server: serves the website and the eye-tracking WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .hands import HandTracker
from .session import TrackerSession
from .tracker import FaceTracker
from .users import UserStore

log = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_WEB_DIR = PROJECT_ROOT / "web"

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_LOOPBACK_CLIENTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _origin_allowed(origin: Optional[str], host_header: Optional[str], client_host: Optional[str],
                    allowed_hosts=LOCAL_HOSTS) -> bool:
    """Decide whether a WebSocket connection may drive the tracker.

    Browsers always send an Origin: it must be exactly the origin of this
    server (same host and port as the Host header), so pages from other
    sites or other local servers cannot connect, and its host name must be
    one we serve under (defeats DNS-rebinding tricks). Connections without an
    Origin (scripts, tests) are only accepted from this machine.
    """
    if not origin:
        return client_host in _LOOPBACK_CLIENTS
    if not host_header:
        return False
    try:
        parsed = urlparse(origin)
        hostname = parsed.hostname
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not hostname:
        return False
    if parsed.netloc.lower() != host_header.lower():
        return False
    return allowed_hosts == "*" or hostname.lower() in allowed_hosts


def host_allowed(host_header: Optional[str], allowed_hosts=LOCAL_HOSTS) -> bool:
    """May a request naming ``host_header`` be answered?

    Pages are only served under the host names this server is meant to be
    reached by, so a web page elsewhere cannot point its own domain at this
    computer (DNS rebinding) and read what is served here.
    """
    if allowed_hosts == "*":
        return True
    if not host_header:
        return False
    try:
        name = urlparse(f"//{host_header}").hostname
    except ValueError:
        return False
    return bool(name) and name.lower() in allowed_hosts


def _unavailable(error: Optional[str]) -> Callable[[], FaceTracker]:
    def factory() -> FaceTracker:
        raise RuntimeError(error or "Eye tracker unavailable")
    return factory



class HandModel:
    """Hand mode's tracker factory, possibly still loading in the background
    (the hand model is downloaded on first use; eye mode never waits for it)."""

    def __init__(self, factory: Optional[Callable[[], HandTracker]] = None,
                 loader: Optional[Callable[[], Callable[[], HandTracker]]] = None):
        self.factory = factory
        self.error: Optional[str] = None
        self.loading = False
        if factory is None and loader is not None:
            self.loading = True
            threading.Thread(target=self._load, args=(loader,), name="paralic-hand-model", daemon=True).start()
        elif factory is None:
            self.error = "Hand mode is not available"

    def _load(self, loader) -> None:
        try:
            self.factory = loader()
        except Exception as exc:  # ModelUnavailable or MediaPipe problems
            log.warning("Hand mode unavailable: %s", exc)
            self.error = f"Hand mode is not available: {exc}"
        finally:
            self.loading = False

    def status(self) -> str:
        return "ready" if self.factory is not None else "loading" if self.loading else "unavailable"


def create_app(*, data_dir: Path, web_dir: Path = DEFAULT_WEB_DIR,
               tracker_factory: Optional[Callable[[], FaceTracker]] = None,
               hand_tracker_factory: Optional[Callable[[], HandTracker]] = None,
               hand_loader: Optional[Callable[[], Callable[[], HandTracker]]] = None,
               model_error: Optional[str] = None, allowed_hosts=LOCAL_HOSTS, record_all: bool = False,
               recordings_dir: Optional[Path] = None) -> FastAPI:
    """Build the application.

    ``data_dir`` holds everyone's profiles (see ``users.py``).
    ``record_all`` records every session, from the page's hello on, into
    ``recordings_dir`` (default: ``data_dir/recordings``; see ``recorder.py``).
    Without it the page's ● Rec button records single sessions there.
    ``tracker_factory`` creates one MediaPipe FaceTracker per connection. When
    it is None (e.g. the model could not be downloaded) the website still
    loads, shows ``model_error`` and works in mouse demo mode.
    ``hand_tracker_factory`` creates a HandTracker for hand mode; or
    ``hand_loader`` returns one, slowly (it is run on a background thread).
    ``allowed_hosts`` lists the host names the site may be opened under ("*"
    for any).
    """
    app = FastAPI(title="Paralic", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    users = UserStore(Path(data_dir))
    hands = HandModel(hand_tracker_factory, hand_loader)
    app.state.hands = hands
    recordings = Path(recordings_dir) if recordings_dir is not None else Path(data_dir) / "recordings"

    @app.middleware("http")
    async def no_cache(request, call_next):
        if not host_allowed(request.headers.get("host"), allowed_hosts):
            return PlainTextResponse("Paralic answers only under this computer's own address.", status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/status")
    async def status() -> JSONResponse:
        active = users.active_id()
        return JSONResponse({
            "version": __version__,
            "tracker": tracker_factory is not None,
            "error": model_error,
            "hands": hands.status(),
            "hands_error": hands.error,
            "profile": users.profile_summary(active) if active else None,
            "users": len(users.list()),
            "recording": record_all,
        })

    @app.get("/api/trail/{kind}")
    async def trail(kind: str, n: int = 300, seed: int = 0) -> JSONResponse:
        """Procedural pursuit trail (normalised 0..1 waypoints) for smooth-pursuit
        calibration and the eye-tracking games. See paralic/trails.py."""
        from .trails import KINDS, trail as make_trail
        if kind not in KINDS:
            return JSONResponse({"error": f"unknown trail {kind!r}", "kinds": list(KINDS)}, status_code=404)
        n = max(2, min(int(n), 2000))
        pts = make_trail(kind, n=n, seed=int(seed))
        return JSONResponse({"kind": kind, "seed": int(seed),
                             "points": [{"x": round(x, 5), "y": round(y, 5)} for x, y in pts]})

    @app.websocket("/ws")
    async def eye_tracking_socket(websocket: WebSocket) -> None:
        client_host = websocket.client.host if websocket.client else None
        if not _origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host"), client_host,
                               allowed_hosts):
            log.warning("Rejected WebSocket connection from origin %r", websocket.headers.get("origin"))
            await websocket.close(code=1008)
            return
        await websocket.accept()

        loop = asyncio.get_running_loop()
        outbox: asyncio.Queue = asyncio.Queue()

        async def sender() -> None:
            # The only task that writes to the socket (replies and pushed results).
            while True:
                msg = await outbox.get()
                if msg is None:
                    return
                await websocket.send_text(json.dumps(msg))

        def push(msg: dict) -> None:  # called from background threads
            try:
                loop.call_soon_threadsafe(outbox.put_nowait, msg)
            except RuntimeError:  # the server is shutting down
                pass

        mode = "hand" if websocket.query_params.get("mode") == "hand" else "eyes"
        if mode == "hand":
            factory = hands.factory
            if factory is None:
                error = "Hand mode is still getting ready - try again in a moment" if hands.loading \
                    else hands.error or "Hand mode is not available"
                outbox.put_nowait({"type": "fatal", "error": error})
                factory = _unavailable(error)
        else:
            if tracker_factory is None:
                # No face tracking, but keep the connection for people / lab commands (demo mode).
                outbox.put_nowait({"type": "fatal", "error": model_error or "Eye tracker unavailable"})
            factory = tracker_factory or _unavailable(model_error)
        # One worker thread per connection keeps MediaPipe calls ordered.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paralic-session")
        session = await loop.run_in_executor(executor, lambda: TrackerSession(
            factory, users, push=push, mode=mode, recordings=recordings, record=record_all))
        send_task = asyncio.create_task(sender())
        try:
            while not send_task.done():
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data is not None:
                    replies = await loop.run_in_executor(executor, session.handle_frame, data)
                else:
                    text = message.get("text")
                    if not text:
                        continue
                    try:
                        cmd = json.loads(text)
                    except ValueError:
                        replies = [{"type": "error", "error": "invalid JSON"}]
                    else:
                        if not isinstance(cmd, dict):
                            replies = [{"type": "error", "error": "commands must be JSON objects"}]
                        else:
                            replies = await loop.run_in_executor(executor, session.handle_command, cmd)
                for reply in replies:
                    outbox.put_nowait(reply)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # keep the server alive, report to the page
            log.exception("Eye-tracking session failed")
            outbox.put_nowait({"type": "fatal", "error": f"Tracker error: {exc}"})
        finally:
            outbox.put_nowait(None)
            # Close the session (saving it, finishing a recording) on its own thread after
            # anything still running there - queued before any await and shielded, so that
            # it happens even when this task is being cancelled.
            closing = loop.run_in_executor(executor, session.close)
            executor.shutdown(wait=False)
            try:
                await asyncio.wait_for(send_task, timeout=2.0)
            except Exception:
                send_task.cancel()
            await asyncio.shield(closing)

    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    return app
