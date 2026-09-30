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
from .calibration import ProfileStore
from .session import TrackerSession
from .tracker import FaceTracker

log = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_WEB_DIR = PROJECT_ROOT / "web"

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _origin_allowed(origin: Optional[str]) -> bool:
    """Only pages served from this machine may drive the tracker."""
    if not origin:
        return True  # non-browser clients (tests, tools) send no Origin
    try:
        host = urlparse(origin).hostname
    except ValueError:
        return False
    return host in _LOCAL_HOSTS


def create_app(*, profile_path: Path, web_dir: Path = DEFAULT_WEB_DIR,
               tracker_factory: Optional[Callable[[], FaceTracker]] = None,
               model_error: Optional[str] = None) -> FastAPI:
    """Build the application.

    ``tracker_factory`` creates one MediaPipe FaceTracker per connection. When
    it is None (e.g. the model could not be downloaded) the website still
    loads and shows ``model_error``.
    """
    app = FastAPI(title="Paralic", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    profiles = ProfileStore(profile_path)

    @app.middleware("http")
    async def no_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/status")
    async def status() -> JSONResponse:
        return JSONResponse({
            "version": __version__,
            "tracker": tracker_factory is not None,
            "error": model_error,
            "profile": profiles.summary(),
        })

    @app.websocket("/ws")
    async def eye_tracking_socket(websocket: WebSocket) -> None:
        if not _origin_allowed(websocket.headers.get("origin")):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        if tracker_factory is None:
            await websocket.send_text(json.dumps({"type": "fatal", "error": model_error or "Eye tracker unavailable"}))
            await websocket.close()
            return

        loop = asyncio.get_running_loop()
        # One worker thread per connection keeps MediaPipe calls ordered.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paralic-session")
        session = TrackerSession(tracker_factory, profiles)
        try:
            while True:
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
                    await websocket.send_text(json.dumps(reply))
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # keep the server alive, report to the page
            log.exception("Eye-tracking session failed")
            try:
                await websocket.send_text(json.dumps({"type": "fatal", "error": f"Tracker error: {exc}"}))
                await websocket.close()
            except Exception:
                pass
        finally:
            await loop.run_in_executor(executor, session.close)
            executor.shutdown(wait=False)

    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    return app
