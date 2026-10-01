"""FastAPI web server: serves the website and the eye-tracking WebSocket."""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .hand_session import HandSession
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


def _unavailable(error: Optional[str]) -> Callable[[], FaceTracker]:
    def factory() -> FaceTracker:
        raise RuntimeError(error or "Eye tracker unavailable")
    return factory


def create_app(*, data_dir: Path, web_dir: Path = DEFAULT_WEB_DIR,
               tracker_factory: Optional[Callable[[], FaceTracker]] = None,
               hand_tracker_factory: Optional[Callable[[], HandTracker]] = None,
               model_error: Optional[str] = None, allowed_hosts=LOCAL_HOSTS) -> FastAPI:
    """Build the application.

    ``data_dir`` holds everyone's profiles (see ``users.py``).
    ``tracker_factory`` creates one MediaPipe FaceTracker per connection. When
    it is None (e.g. the model could not be downloaded) the website still
    loads, shows ``model_error`` and works in mouse demo mode.
    ``allowed_hosts`` lists the host names the site may be opened under ("*"
    for any).
    """
    app = FastAPI(title="Paralic", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    users = UserStore(Path(data_dir))

    @app.middleware("http")
    async def no_cache(request, call_next):
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
            "profile": users.profile_store(active).summary() if active else None,
            "users": len(users.list()),
        })

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

        hand_mode = websocket.query_params.get("mode") == "hand"
        # One worker thread per connection keeps MediaPipe calls ordered.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paralic-session")
        if hand_mode:
            if hand_tracker_factory is None:
                outbox.put_nowait({"type": "fatal", "error": "Hand tracker unavailable"})
                hfac = _unavailable("Hand tracker unavailable")
            else:
                hfac = hand_tracker_factory
            hand_profile = Path(data_dir) / "hand_profile.json"
            session = await loop.run_in_executor(
                executor, lambda: HandSession(hfac, push=push, profile_path=hand_profile))
        else:
            if tracker_factory is None:
                # No face tracking, but keep the connection for people / lab commands (demo mode).
                outbox.put_nowait({"type": "fatal", "error": model_error or "Eye tracker unavailable"})
            factory = tracker_factory or _unavailable(model_error)
            session = await loop.run_in_executor(executor, lambda: TrackerSession(factory, users, push=push))
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
            try:
                await asyncio.wait_for(send_task, timeout=2.0)
            except Exception:
                send_task.cancel()
            await loop.run_in_executor(executor, session.close)
            executor.shutdown(wait=False)

    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    return app
