"""Download / locate the MediaPipe FaceLandmarker model file."""

from __future__ import annotations

import logging
import os
import shutil
import ssl
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


class ModelUnavailable(RuntimeError):
    pass


def _is_valid_model(path: Path) -> bool:
    # .task files are zip archives bundling the TFLite networks.
    try:
        return path.is_file() and path.stat().st_size > 1_000_000 and zipfile.is_zipfile(path)
    except OSError:
        return False


def _ssl_contexts():
    yield ssl.create_default_context()
    try:  # python.org builds on macOS often lack system certificates
        import certifi

        yield ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover
        pass


def ensure_face_model(path: Path, url: str = FACE_LANDMARKER_URL) -> Path:
    """Return ``path``, downloading the model there first if needed."""
    path = Path(path)
    if _is_valid_model(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading MediaPipe face landmarker model (~3.7 MB) to %s", path)
    last_error: Exception | None = None
    for ctx in _ssl_contexts():
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60, context=ctx) as resp:
                shutil.copyfileobj(resp, out)
            tmp_path = Path(tmp)
            if not _is_valid_model(tmp_path):
                raise ModelUnavailable("Downloaded file is not a valid model")
            os.replace(tmp_path, path)
            return path
        except (urllib.error.URLError, OSError, ssl.SSLError, ModelUnavailable) as exc:
            last_error = exc
            log.debug("Model download attempt failed: %s", exc)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    raise ModelUnavailable(
        f"Could not download the face landmarker model ({last_error}).\n"
        f"Download it manually from\n  {url}\nand save it as\n  {path}"
    )
