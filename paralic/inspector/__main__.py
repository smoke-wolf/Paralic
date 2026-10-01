"""Command line: ``python -m paralic.inspector [--data-dir data] [--port 8100] [--no-browser] [recording-id]``."""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import quote

from .. import __version__
from ..recording import is_recording, recordings_root

HOST = "127.0.0.1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, port))
        except OSError:
            return False
    return True


def _pick_port(preferred: int, attempts: int = 20) -> int:
    for port in range(preferred, preferred + attempts):
        if _free(port):
            return port
    raise SystemExit(f"No free port found between {preferred} and {preferred + attempts - 1}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m paralic.inspector",
                                     description="Replay Paralic session recordings and look inside the models.")
    parser.add_argument("recording", nargs="?", help="open this recording (its folder name)")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data",
                        help="Paralic's data folder (recordings are in its recordings/ folder), "
                             "or a folder of recordings (python -m paralic --recordings-dir)")
    parser.add_argument("--port", type=int, default=8100, help="port (default: 8100, next free one if busy)")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    import uvicorn

    from .server import create_app

    root = recordings_root(args.data_dir)
    app = create_app(args.data_dir)
    port = _pick_port(args.port)
    url = f"http://localhost:{port}/"
    if args.recording:
        if is_recording(root / args.recording) or (root.name == args.recording and is_recording(root)):
            url += f"#/rec/{quote(args.recording)}"
        else:
            logging.getLogger("paralic").warning("No recording %r in %s", args.recording, root)
    print(f"\n  Paralic Inspector {__version__}\n  Recordings: {root}\n  Open {url}\n  Press Ctrl+C to stop.\n",
          flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host=HOST, port=port, log_level="warning")
    except KeyboardInterrupt:  # pragma: no cover
        pass


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
