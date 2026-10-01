"""People who use Paralic on this computer, each with their own eye profile.

Layout of the data directory::

    data/
      users.json                  who exists, and who used Paralic last
      users/<id>/profile.json     GazeNet + calibration and fine-tuning samples
      users/<id>/profile-glasses.json  the same, calibrated wearing glasses
      users/<id>/personal.json    personal blink / smoothing / magnet settings,
                                  model history, experiment decisions
      users/<id>/experiments.json raw A/B trial results
      users/<id>/hand.json        Hand mode setup (pointing map, pinch thresholds)
      users/<id>/faceprint/       face print (see faceprint.py)

A single-user ``data/profile.json`` from older versions is migrated into the
first person's folder automatically, and so is the shared
``data/hand_profile.json`` of the first Hand mode preview.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from .calibration import ProfileStore
from .faceprint import FacePrintStore

_ID_RE = re.compile(r"^u[0-9a-f]{8}$")
MAX_NAME = 32


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_json_atomic(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_json(path: Path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def clean_name(name: Optional[str]) -> str:
    name = re.sub(r"\s+", " ", str(name or ""))          # tabs / newlines -> spaces
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()  # drop other control characters
    return name[:MAX_NAME]


class UnknownUser(KeyError):
    pass


class UserStore:
    """Thread-safe store of user profiles (shared by all browser tabs)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.index_path = self.root / "users.json"
        self._lock = threading.RLock()
        self._user_locks: dict[str, threading.RLock] = {}
        self._migrate_legacy()

    # -- index ----------------------------------------------------------------
    def _read_index(self) -> dict:
        doc = read_json(self.index_path, {})
        users = [u for u in doc.get("users", []) if isinstance(u, dict) and _ID_RE.match(str(u.get("id", "")))]
        active = doc.get("active") if any(u["id"] == doc.get("active") for u in users) else None
        return {"active": active, "users": users}

    def _write_index(self, doc: dict) -> None:
        write_json_atomic(self.index_path, doc)

    def _migrate_legacy(self) -> None:
        legacy = self.root / "profile.json"
        with self._lock:
            if legacy.is_file() and not self._read_index()["users"]:
                user = self.create("Person 1")
                target = self.user_dir(user["id"]) / "profile.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(legacy), str(target))
            hand = self.root / "hand_profile.json"
            if hand.is_file():
                # One shared hand setup: it becomes the active person's, unless they have their own.
                user = self.ensure_active()
                target = self.user_dir(user["id"]) / "hand.json"
                doc = read_json(hand, None)
                if not target.exists() and isinstance(doc, dict):
                    write_json_atomic(target, {"version": 1, "updated": doc.get("updated", _now()),
                                               "calibration": doc.get("calibration"), "summary": {}})
                hand.unlink(missing_ok=True)

    def list(self) -> list[dict]:
        with self._lock:
            doc = self._read_index()
        out = []
        for u in doc["users"]:
            calibrated = self.profile_store(u["id"]).exists() or self.profile_store(u["id"], glasses=True).exists()
            out.append({**u, "calibrated": calibrated, "face": self.face_store(u["id"]).exists(),
                        "hands": (self.user_dir(u["id"]) / "hand.json").is_file()})
        return out

    def active_id(self) -> Optional[str]:
        with self._lock:
            return self._read_index()["active"]

    def get(self, user_id: str) -> dict:
        with self._lock:
            for u in self._read_index()["users"]:
                if u["id"] == user_id:
                    return u
        raise UnknownUser(user_id)

    def create(self, name: Optional[str] = None) -> dict:
        with self._lock:
            doc = self._read_index()
            existing = {u["name"] for u in doc["users"]}
            name = clean_name(name)
            if not name:
                k = len(doc["users"]) + 1
                while f"Person {k}" in existing:
                    k += 1
                name = f"Person {k}"
            user = {"id": "u" + uuid.uuid4().hex[:8], "name": name, "created": _now(), "last_used": _now()}
            doc["users"].append(user)
            doc["active"] = user["id"]
            self._write_index(doc)
            self.user_dir(user["id"]).mkdir(parents=True, exist_ok=True)
            return user

    def ensure_active(self) -> dict:
        """The active person, creating "Person 1" on first run."""
        with self._lock:
            doc = self._read_index()
            if doc["active"]:
                return self.get(doc["active"])
            if doc["users"]:
                latest = max(doc["users"], key=lambda u: u.get("last_used", ""))
                self.select(latest["id"])
                return latest
            return self.create()

    def select(self, user_id: str) -> dict:
        with self._lock:
            doc = self._read_index()
            for u in doc["users"]:
                if u["id"] == user_id:
                    u["last_used"] = _now()
                    doc["active"] = user_id
                    self._write_index(doc)
                    return u
        raise UnknownUser(user_id)

    def rename(self, user_id: str, name: str) -> dict:
        name = clean_name(name)
        if not name:
            raise ValueError("Name must not be empty")
        with self._lock:
            doc = self._read_index()
            for u in doc["users"]:
                if u["id"] == user_id:
                    u["name"] = name
                    self._write_index(doc)
                    return u
        raise UnknownUser(user_id)

    def delete(self, user_id: str) -> None:
        with self._lock:
            doc = self._read_index()
            if not any(u["id"] == user_id for u in doc["users"]):
                raise UnknownUser(user_id)
            doc["users"] = [u for u in doc["users"] if u["id"] != user_id]
            if doc["active"] == user_id:
                doc["active"] = None
            self._write_index(doc)
            shutil.rmtree(self.user_dir(user_id), ignore_errors=True)

    # -- per-user files ----------------------------------------------------------
    def user_dir(self, user_id: str) -> Path:
        if not _ID_RE.match(str(user_id)):
            raise UnknownUser(user_id)
        return self.root / "users" / user_id

    def lock_for(self, user_id: str) -> threading.RLock:
        with self._lock:
            return self._user_locks.setdefault(user_id, threading.RLock())

    def face_store(self, user_id: str) -> FacePrintStore:
        return FacePrintStore(self.user_dir(user_id) / "faceprint")

    def profile_store(self, user_id: str, glasses: bool = False) -> ProfileStore:
        """The person's calibration made wearing glasses, or the other one (made
        without them - or before Paralic noticed glasses)."""
        return ProfileStore(self.user_dir(user_id) / ("profile-glasses.json" if glasses else "profile.json"))

    def profile_summary(self, user_id: str) -> Optional[dict]:
        """The saved calibration's summary: the one made without glasses, else the one with."""
        for glasses in (False, True):
            summary = self.profile_store(user_id, glasses=glasses).summary()
            if summary is not None:
                return summary
        return None

    def load_personal(self, user_id: str) -> dict:
        return read_json(self.user_dir(user_id) / "personal.json", {})

    def save_personal(self, user_id: str, personal: dict) -> None:
        with self.lock_for(user_id):
            write_json_atomic(self.user_dir(user_id) / "personal.json", personal)

    def load_experiments(self, user_id: str) -> dict:
        return read_json(self.user_dir(user_id) / "experiments.json", {})

    def save_experiments(self, user_id: str, experiments: dict) -> None:
        with self.lock_for(user_id):
            write_json_atomic(self.user_dir(user_id) / "experiments.json", experiments)

    def load_hand(self, user_id: str) -> Optional[dict]:
        doc = read_json(self.user_dir(user_id) / "hand.json", None)
        return doc if isinstance(doc, dict) else None

    def save_hand(self, user_id: str, doc: dict) -> None:
        with self.lock_for(user_id):
            write_json_atomic(self.user_dir(user_id) / "hand.json", doc)

    def delete_hand(self, user_id: str) -> None:
        with self.lock_for(user_id):
            (self.user_dir(user_id) / "hand.json").unlink(missing_ok=True)
