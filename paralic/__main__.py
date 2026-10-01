"""Command-line entry point: ``python -m paralic``."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

from . import __version__
from .model_assets import ModelUnavailable, ensure_face_model, ensure_hand_model
from .server import DEFAULT_WEB_DIR, PROJECT_ROOT, create_app

log = logging.getLogger("paralic")


def _port_free(host: str, port: int) -> bool:
    """True if every address ``host`` resolves to (IPv4 and/or IPv6) can bind ``port``."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SystemExit(f"Cannot use host {host!r}: {exc}") from None
    for family, socktype, proto, _, addr in infos:
        with socket.socket(family, socktype, proto) as s:
            if os.name != "nt":  # match uvicorn, which reuses ports left in TIME_WAIT
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(addr)
            except OSError:
                return False
    return True


def _pick_port(host: str, preferred: int, attempts: int = 20) -> int:
    for port in range(preferred, preferred + attempts):
        if _port_free(host, port):
            return port
    raise SystemExit(f"No free port found between {preferred} and {preferred + attempts - 1}")


def _allowed_hosts(host: str):
    """Host names pages may be served from (see server._origin_allowed)."""
    if host in ("0.0.0.0", "::", ""):
        return "*"  # listening on every interface: any address of this machine
    names = {"localhost", "127.0.0.1", "::1"}
    names.add(host.strip("[]").lower())
    return names


def _check_tracker(model_bytes: bytes) -> None:
    """Load MediaPipe once at startup so problems are reported immediately."""
    from .tracker import FaceTracker

    FaceTracker(model_bytes).close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="paralic", description="Eye-controlled website (webcam eye tracking).")
    parser.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: 8000, next free one if busy)")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "face_landmarker.task",
                        help="path of the MediaPipe face landmarker model (downloaded if missing)")
    parser.add_argument("--hand-model", type=Path, default=PROJECT_ROOT / "models" / "hand_landmarker.task",
                        help="path of the MediaPipe hand landmarker model for Hand mode (downloaded if missing)")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data",
                        help="where each person's calibration and personal settings are saved")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    tracker_factory = None
    model_error = None
    try:
        model_path = ensure_face_model(args.model)
        model_bytes = model_path.read_bytes()
        _check_tracker(model_bytes)
    except ModelUnavailable as exc:
        model_error = str(exc)
    except Exception as exc:  # e.g. MediaPipe native library problems
        model_error = f"MediaPipe could not start: {exc}"
        if "libEGL" in str(exc) or "libGLES" in str(exc):
            model_error += " (on Linux install the OpenGL ES libraries: sudo apt install libegl1 libgles2)"
    else:
        from .tracker import FaceTracker

        def tracker_factory():
            return FaceTracker(model_bytes)

    if model_error:
        log.error(model_error)

    # Hand mode is optional: load its model best-effort and never block eye mode.
    hand_tracker_factory = None
    try:
        hand_bytes = ensure_hand_model(args.hand_model).read_bytes()
        from .hands import HandTracker

        HandTracker(hand_bytes).close()      # fail fast if MediaPipe can't start it

        def hand_tracker_factory():
            return HandTracker(hand_bytes)
    except Exception as exc:  # ModelUnavailable or native MediaPipe problems
        log.warning("Hand mode unavailable: %s", exc)

    import uvicorn

    port = _pick_port(args.host, args.port)
    app = create_app(data_dir=args.data_dir, web_dir=DEFAULT_WEB_DIR,
                     tracker_factory=tracker_factory, hand_tracker_factory=hand_tracker_factory,
                     model_error=model_error, allowed_hosts=_allowed_hosts(args.host))

    shown_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0", "::", "::1") else args.host
    url = f"http://{shown_host}:{port}/"
    print(f"\n  Paralic {__version__} - eye-controlled browsing\n  Open {url} in Chrome, Edge or Firefox.\n"
          f"  Press Ctrl+C to stop.\n", flush=True)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("Browsers only allow camera access on localhost or HTTPS pages.")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host=args.host, port=port, log_level="warning", ws_max_size=8 * 1024 * 1024)
    except KeyboardInterrupt:  # pragma: no cover
        pass


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
