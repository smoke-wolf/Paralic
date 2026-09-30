"""Calibration data handling, GazeNet training and saved profiles.

During calibration the browser shows a dot and labels every camera frame with
the dot's on-screen position. Frames are grouped per dot ("point"); we reject
frames where you glanced away or blinked, average short runs of frames to
reduce landmark noise, then train GazeNet.

Kinds of labelled frames:

* ``cal``    - the calibration grid (used for training and cross-validation)
* ``head``   - looking at the centre dot while gently moving the head
               (teaches the network to compensate for head movement)
* ``val``    - validation dots shown after training (measure accuracy, then
               also used for the final fit)
* ``adjust`` - the quick 5-point re-adjustment of a saved calibration
* ``ft``     - fine-tuning samples collected while you use the site: the
               frames just before you popped a practice target or clicked a
               button, labelled with where that target was. ``point`` is the
               event number and ``weight`` how much we trust the label.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .features import EYE_FEATURE_IDX, FEATURE_MIN_STD, FEATURE_NAMES, NUM_FEATURES
from .gazenet import AffineCorrection, GazeNet

PROFILE_VERSION = 1
KINDS = ("cal", "head", "val", "adjust", "ft")
TRAIN_KINDS = ("cal", "head", "val", "adjust", "ft")
# Kinds whose samples may be held out in cross-validation (not the head-motion ones).
HOLDOUT_KINDS = ("cal", "val", "adjust", "ft")
MAX_FT_EVENTS = 1500


class CalibrationError(Exception):
    """Raised when there is not enough usable calibration data."""


@dataclass
class LabeledFrame:
    t: float                      # wall-clock time (seconds since the epoch)
    features: np.ndarray
    target: tuple[float, float]
    kind: str
    point: int
    weight: float = 1.0


@dataclass
class CalibrationData:
    frames: list[LabeledFrame] = field(default_factory=list)

    def add(self, frame: LabeledFrame) -> None:
        self.frames.append(frame)

    def copy(self) -> "CalibrationData":
        return CalibrationData(list(self.frames))

    def ft_events(self) -> dict[int, list[LabeledFrame]]:
        """Fine-tuning events, oldest first: event number -> frames."""
        events: dict[int, list[LabeledFrame]] = {}
        for f in self.frames:
            if f.kind == "ft":
                events.setdefault(f.point, []).append(f)
        return dict(sorted(events.items()))

    def next_event_id(self) -> int:
        return max((f.point for f in self.frames if f.kind == "ft"), default=0) + 1

    def prune_ft(self, max_events: int = MAX_FT_EVENTS) -> None:
        """Forget the oldest fine-tuning events beyond ``max_events``."""
        ids = sorted({f.point for f in self.frames if f.kind == "ft"})
        if len(ids) > max_events:
            cutoff = ids[len(ids) - max_events]
            self.frames = [f for f in self.frames if f.kind != "ft" or f.point >= cutoff]

    def clear(self, kinds: Optional[Iterable[str]] = None) -> None:
        if kinds is None:
            self.frames.clear()
        else:
            ks = set(kinds)
            self.frames = [f for f in self.frames if f.kind not in ks]

    def of_kind(self, *kinds: str) -> list[LabeledFrame]:
        return [f for f in self.frames if f.kind in kinds]

    def count(self, kind: str) -> int:
        return sum(1 for f in self.frames if f.kind == kind)


# ---------------------------------------------------------------------------
# Preparing training data
# ---------------------------------------------------------------------------

def _group_frames(frames: list[LabeledFrame]) -> dict[str, list[LabeledFrame]]:
    groups: dict[str, list[LabeledFrame]] = {}
    for f in sorted(frames, key=lambda f: f.t):
        groups.setdefault(f"{f.kind}:{f.point}", []).append(f)
    return groups


def reject_outliers(F: np.ndarray, z_max: float = 3.5) -> np.ndarray:
    """Boolean mask of frames whose eye features are consistent with the rest."""
    if len(F) < 5:
        return np.ones(len(F), bool)
    eye = F[:, list(EYE_FEATURE_IDX)]
    med = np.median(eye, axis=0)
    mad = np.median(np.abs(eye - med), axis=0) * 1.4826
    mad = np.maximum(mad, 0.004)
    z = np.abs(eye - med) / mad
    return (z <= z_max).all(axis=1)


def chunk_average(F: np.ndarray, size: int, stride: int) -> np.ndarray:
    """Average consecutive frames in overlapping windows (reduces input noise)."""
    if len(F) < size:
        return F.mean(axis=0, keepdims=True)
    starts = range(0, len(F) - size + 1, stride)
    return np.array([F[s:s + size].mean(axis=0) for s in starts])


def prepare_training_set(frames: list[LabeledFrame], chunk: int = 4, stride: int = 2,
                         head_weight: float = 3.0, min_frames: int = 4):
    """Turn labelled frames into (X, Y, groups, weights, holdout) for GazeNet."""
    Xs, Ys, Gs, Ws, Hs = [], [], [], [], []
    for name, group in _group_frames(frames).items():
        F = np.array([f.features for f in group])
        keep = reject_outliers(F)
        F = F[keep]
        if len(F) < min_frames:
            continue
        # Moving the head is continuous motion, so average less there.
        kind = group[0].kind
        C = chunk_average(F, 2 if kind == "head" else chunk, 1 if kind == "head" else stride)
        target = np.array(group[0].target, float)
        gw = (head_weight if kind == "head" else 1.0) * float(np.mean([f.weight for f in group]))
        Xs.append(C)
        Ys.append(np.repeat(target[None], len(C), axis=0))
        Gs.extend([name] * len(C))
        Ws.append(np.full(len(C), gw / len(C)))
        Hs.extend([kind in HOLDOUT_KINDS] * len(C))
    if not Xs:
        raise CalibrationError("No usable calibration data")
    W = np.concatenate(Ws)
    return (np.vstack(Xs), np.vstack(Ys), np.array(Gs), W * (len(W) / W.sum()), np.array(Hs))


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def fit_full_calibration(data: CalibrationData, include_validation: bool = False, *,
                         kinds: Optional[Iterable[str]] = None, configs=None) -> tuple[GazeNet, dict]:
    """Train GazeNet on the labelled frames of the given kinds.

    ``configs`` is the list of candidate architectures (see
    :func:`paralic.gazenet.search_configs`); by default only the weight decay
    is tuned, which keeps the first calibration fast.
    """
    if kinds is None:
        kinds = ("cal", "head", "val") if include_validation else ("cal", "head")
    frames = data.of_kind(*kinds)
    n_points = len({f.point for f in frames if f.kind == "cal"})
    if n_points < 6:
        raise CalibrationError(
            "Not enough calibration data - make sure your face is well lit and visible to the camera.")
    X, Y, G, W, H = prepare_training_set(frames)
    usable_points = len({g for g, h in zip(G, H) if h})
    if usable_points < 6:
        raise CalibrationError(
            "Too few calibration points had steady eye data. Try again, keeping your eyes on each dot.")
    started = time.perf_counter()
    model, report = GazeNet.train(X, Y, G, W, holdout=H, min_std=FEATURE_MIN_STD, configs=configs)
    elapsed = time.perf_counter() - started
    model.meta.update({
        "trained_ts": time.time(),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_frames": len(frames),
        "l2": report.l2,
        "cv_error_px": _round(report.cv_error_px),
        "last_event": max((f.point for f in frames if f.kind == "ft"), default=0),
    })
    info = {
        "n_frames": len(frames),
        "n_samples": report.n_samples,
        "n_points": usable_points,
        "l2": report.l2,
        "config": report.config,
        "candidates": {k: _round(v) for k, v in report.candidates.items()},
        "cv_error_px": _round(report.cv_error_px),
        "linear_cv_error_px": _round(report.linear_cv_error_px),
        "train_error_px": _round(report.train_error_px),
        "train_seconds": round(elapsed, 2),
    }
    return model, info


def point_summaries(model: GazeNet, frames: list[LabeledFrame], corrected: bool = True) -> list[dict]:
    """Per-dot accuracy (distance of the mean prediction) and precision (spread)."""
    out = []
    for name, group in _group_frames(frames).items():
        F = np.array([f.features for f in group])
        F = F[reject_outliers(F)]
        if len(F) == 0:
            continue
        P = model.predict(F) if corrected else model.predict_uncorrected(F)
        target = np.array(group[0].target, float)
        mean = P.mean(axis=0)
        out.append({
            "point": group[0].point,
            "target": target.tolist(),
            "mean": mean.tolist(),
            "error": float(np.linalg.norm(mean - target)),
            "spread": float(np.sqrt(((P - mean) ** 2).sum(axis=1).mean())),
            "n": int(len(F)),
        })
    return out


def evaluate_validation(model: GazeNet, data: CalibrationData) -> dict:
    points = point_summaries(model, data.of_kind("val"))
    if not points:
        raise CalibrationError("No validation data was captured")
    errors = [p["error"] for p in points]
    return {
        "points": points,
        "mean_error_px": float(np.mean(errors)),
        "max_error_px": float(np.max(errors)),
        "precision_px": float(np.mean([p["spread"] for p in points])),
    }


def fit_adjustment(model: GazeNet, data: CalibrationData) -> dict:
    """Quick re-adjustment: fit an affine correction from a few dots."""
    points = point_summaries(model, data.of_kind("adjust"), corrected=False)
    if len(points) < 3:
        raise CalibrationError("Not enough adjustment data - keep your eyes on each dot.")
    pred = np.array([p["mean"] for p in points])
    target = np.array([p["target"] for p in points])
    before = model.correction.apply(pred)
    correction = AffineCorrection.fit(pred, target)
    after = correction.apply(pred)
    model.correction = correction
    return {
        "n_points": len(points),
        "error_before_px": float(np.linalg.norm(before - target, axis=1).mean()),
        "error_after_px": float(np.linalg.norm(after - target, axis=1).mean()),
    }


def _round(v: float) -> Optional[float]:
    return None if v != v else round(float(v), 1)  # NaN -> None (JSON friendly)


# ---------------------------------------------------------------------------
# Saved profile
# ---------------------------------------------------------------------------

class ProfileStore:
    """Stores the trained network (and its training data) as a JSON file."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.is_file()

    def summary(self) -> Optional[dict]:
        try:
            doc = self._read()
        except (OSError, ValueError, KeyError):
            return None
        return {
            "created": doc.get("created"),
            "screen": doc.get("screen"),
            "accuracy_px": doc.get("accuracy_px"),
            "model_version": doc.get("model", {}).get("meta", {}).get("version"),
        }

    def save(self, model: GazeNet, data: CalibrationData, screen: Optional[dict], accuracy_px: Optional[float]) -> None:
        frames = data.of_kind(*TRAIN_KINDS)
        doc = {
            "version": PROFILE_VERSION,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "screen": screen,
            "accuracy_px": accuracy_px,
            "feature_names": list(FEATURE_NAMES),
            "model": model.to_dict(),
            "data": {
                "features": [np.round(f.features, 5).tolist() for f in frames],
                "targets": [list(f.target) for f in frames],
                "kinds": [f.kind for f in frames],
                "points": [f.point for f in frames],
                "times": [round(f.t, 3) for f in frames],
                "weights": [round(f.weight, 3) for f in frames],
            },
        }
        self._write(doc)

    def save_correction(self, model: GazeNet) -> None:
        """Update just the model (e.g. after a quick adjustment)."""
        doc = self._read()
        doc["model"] = model.to_dict()
        self._write(doc)

    def _write(self, doc: dict) -> None:
        """Write atomically so a crash never leaves a half-written profile."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".profile-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def load(self) -> tuple[GazeNet, CalibrationData, dict]:
        doc = self._read()
        if doc.get("version") != PROFILE_VERSION or doc.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Saved calibration was made by an incompatible version")
        model = GazeNet.from_dict(doc["model"])
        data = CalibrationData()
        d = doc.get("data", {})
        n = len(d.get("features", []))
        weights = d.get("weights") or [1.0] * n
        for feats, target, kind, point, t, w in zip(d.get("features", []), d.get("targets", []), d.get("kinds", []),
                                                    d.get("points", []), d.get("times", []), weights):
            data.add(LabeledFrame(t=t, features=np.asarray(feats, float), target=tuple(target), kind=kind,
                                  point=point, weight=float(w)))
        return model, data, {"created": doc.get("created"), "screen": doc.get("screen"),
                             "accuracy_px": doc.get("accuracy_px")}

    def delete(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def _read(self) -> dict:
        doc = json.loads(self.path.read_text(encoding="utf-8"))
        if len(doc.get("feature_names", [])) != NUM_FEATURES:
            raise ValueError("feature mismatch")
        return doc
