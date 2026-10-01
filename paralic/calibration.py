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
import shutil
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .features import (EYE_FEATURE_IDX, EYE_INPUTS, FEATURE_MIN_STD, FEATURE_NAMES, FEATURE_VERSION,
                       NUM_FEATURES)
from .gazenet import AffineCorrection, GazeNet, ModelConfig

# Bumped to 2 with the full-mesh feature set (see features.FEATURE_VERSION).
PROFILE_VERSION = 2
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


# Kinds where every frame of a group should show the eyes resting on one dot.
FIXATION_KINDS = ("cal", "val", "adjust")
# Features that show where the eyes rest: the iris offsets and, for vertical
# moves (weak in the offsets), the iris height between the lids.
FIXATION_IDX: tuple[int, ...] = tuple(EYE_FEATURE_IDX) + (20, 21)


def noise_sigma(F: np.ndarray, floor: float = 0.003) -> np.ndarray:
    """Frame-to-frame noise of the eye features (robust: from the median step).

    Steps between neighbouring frames are mostly noise (the occasional jump to
    a new dot barely moves the median); for Gaussian noise of standard
    deviation s the median absolute step is 0.6745 * sqrt(2) * s.
    """
    eye = np.asarray(F, float)[:, list(FIXATION_IDX)]
    steps = np.abs(np.diff(eye, axis=0))
    return np.maximum(1.4826 * np.median(steps, axis=0) / np.sqrt(2.0), floor)


def segments(Z: np.ndarray, min_seg: int = 4, chi_min: float = 30.0) -> list[tuple[int, int]]:
    """Split a recording into steady stretches where the eyes rested in one place.

    ``Z``: eye features in units of the frame-to-frame noise (frames x
    features). Binary segmentation on shifts of the mean: a stretch is split
    where the means before and after differ most, if that difference is
    significant (the sum over features of the squared t statistics is at
    least ``chi_min``). Comparing means rather than single frames tells apart
    fixations closer together than the noise of one frame (neighbouring dots
    for someone with jittery tracking). Returns (start, end) index pairs in
    time order; stretches are >= ``min_seg`` frames.
    """
    out = []
    stack = [(0, len(Z))]
    while stack:
        lo, hi = stack.pop()
        n = hi - lo
        cut = None
        if n >= 2 * min_seg:
            c = np.cumsum(Z[lo:hi], axis=0)
            k = np.arange(min_seg, n - min_seg + 1)
            left = c[k - 1] / k[:, None]
            right = (c[-1] - c[k - 1]) / (n - k)[:, None]
            chi = (((right - left) / np.sqrt(1.0 / k + 1.0 / (n - k))[:, None]) ** 2).sum(axis=1)
            best = int(np.argmax(chi))
            if chi[best] >= chi_min:
                cut = lo + int(k[best])
        if cut is None:
            out.append((lo, hi))
        else:
            stack += [(cut, hi), (lo, cut)]
    return sorted(out)


def _matches(Z: np.ndarray, rest: Optional[tuple[np.ndarray, int]], chi_max: float) -> bool:
    """Whether frames ``Z`` (noise units) still show the eyes where they rested
    before: ``rest`` = (mean, number of frames it averages), in noise units."""
    if rest is None or len(Z) == 0:
        return False
    mean, n = rest
    d = (Z.mean(axis=0) - mean) / np.sqrt(1.0 / len(Z) + 1.0 / max(1, n))
    return bool((d ** 2).sum() <= chi_max)


def fixation_frames(F: np.ndarray, sigma: Optional[np.ndarray] = None,
                    previous: Optional[tuple[np.ndarray, int]] = None, min_keep: int = 4,
                    move_chi: float = 25.0, z_max: float = 4.5, settle_frames: int = 3) -> np.ndarray:
    """Boolean mask of the frames that show the eyes resting on the dot.

    The frames of one calibration dot are not all looking at it: slower eyes
    may still be on the previous dot (or in flight) when labelling starts, and
    people glance at the instructions. The recording is split into steady
    stretches (:func:`segments`); stretches where the eyes were still where
    they rested on the previous dot (``previous``: that rest's eye features
    and how many frames it averages) are dropped. Recording stops once the
    eyes have rested on the dot for a while, so the stretch that ends it is
    the one on the dot - unless it is short (under 8 frames, or under 30% of
    the longest; a glance away at the very end of an old fixed-length
    recording). Odd glitch frames
    go too, and so do the first ``settle_frames`` of a long stretch (after a
    jump the eyes often land a little short and correct). If every stretch
    looks like the previous dot, the tracking cannot tell the two apart (a weak
    signal, or eyes that barely move): the dot is kept as if there were no
    previous one rather than lost. Distances are in units of the
    frame-to-frame noise (``sigma``, best pooled over all the dots - see
    :func:`noise_sigma`), so the rule adapts to each person and camera.
    """
    n = len(F)
    if n < 6:
        return reject_outliers(F)
    sigma = noise_sigma(F) if sigma is None else np.maximum(np.asarray(sigma, float), 0.003)
    Z = F[:, list(FIXATION_IDX)] / sigma
    rest = None if previous is None else (np.asarray(previous[0], float) / sigma, previous[1])
    found = segments(Z)
    stretches = [(lo, hi) for lo, hi in found if not _matches(Z[lo:hi], rest, move_chi)] or found
    keep = np.zeros(n, bool)
    longest = max(hi - lo for lo, hi in stretches)
    lo, hi = [st for st in stretches if st[1] - st[0] >= min(longest, max(8, 0.3 * longest))][-1]
    if hi - lo >= 10:
        lo += settle_frames
    med = np.median(Z[lo:hi], axis=0)
    keep[lo:hi] = (np.abs(Z[lo:hi] - med) <= z_max).all(axis=1)
    if keep.sum() < min_keep:
        keep[lo:hi] = True
    return keep


def settled_run(F: np.ndarray, previous: Optional[tuple[np.ndarray, int]] = None,
                sigma: Optional[np.ndarray] = None, move_chi: float = 25.0) -> int:
    """How many of the most recent frames show the eyes resting on the new dot.

    Used while collecting a calibration dot: keep collecting until the eyes
    have rested on it long enough, so slower eyes get more time. The count is
    the length of the last steady stretch (:func:`segments`), or 0 while the
    eyes are still where they rested on the previous dot (``previous``: that
    rest's eye features and how many frames it averages). ``sigma``: the
    frame-to-frame noise, best pooled over the dots so far (a few frames give
    a poor estimate; see :func:`noise_sigma`).
    """
    n = len(F)
    if n < 4:
        return 0
    F = np.asarray(F, float)
    sigma = noise_sigma(F) if sigma is None else np.maximum(np.asarray(sigma, float), 0.003)
    Z = F[:, list(FIXATION_IDX)] / sigma
    lo, hi = segments(Z)[-1]
    rest = None if previous is None else (np.asarray(previous[0], float) / sigma, previous[1])
    if _matches(Z[lo:hi], rest, move_chi):
        return 0
    return hi - lo


class SettleTracker:
    """Live "settled" count for the calibration dot being recorded.

    Fed every labelled frame of a dot; reports how many of its latest frames
    show the eyes resting on it (:func:`settled_run`). Remembers where the
    eyes rested on the previous dot (nothing counts while they are still
    there) and pools the frame-to-frame noise over the dots so far (one dot's
    few frames estimate it poorly). The page keeps a dot up until the count
    reaches its target, so slower eyes get more time.
    """

    def __init__(self, max_frames: int = 240, max_steps: int = 600):
        self.max_frames = max_frames
        self.key: Optional[tuple] = None
        self.frames: list[np.ndarray] = []
        self.previous: Optional[tuple[np.ndarray, int]] = None
        self.steps: deque = deque(maxlen=max_steps)

    def reset(self) -> None:
        self.key, self.frames, self.previous = None, [], None
        self.steps.clear()

    def sigma(self) -> Optional[np.ndarray]:
        if len(self.steps) < 20:
            return None
        return np.maximum(1.4826 * np.median(np.array(self.steps), axis=0) / np.sqrt(2.0), 0.003)

    def _rest(self) -> Optional[tuple[np.ndarray, int]]:
        """Where the eyes rested at the end of the current dot (eye features, frames)."""
        if len(self.frames) < 4:
            return None
        E = np.array(self.frames)[:, list(FIXATION_IDX)]
        sigma = self.sigma()
        sigma = noise_sigma(np.array(self.frames)) if sigma is None else sigma
        lo, hi = segments(E / sigma)[-1]
        return E[lo:hi].mean(axis=0), hi - lo

    def update(self, key: tuple, vector: np.ndarray) -> int:
        """Add a frame of the dot ``key`` (e.g. (kind, point, target)); returns the count."""
        if key != self.key:
            same_kind = self.key is not None and key[0] == self.key[0]
            self.previous = self._rest() if same_kind else None
            self.key = key
            self.frames = []
        vector = np.asarray(vector, float)
        eye = list(FIXATION_IDX)
        if self.frames:
            self.steps.append(np.abs(vector[eye] - self.frames[-1][eye]))
        self.frames.append(vector)
        if len(self.frames) > self.max_frames:
            del self.frames[:-self.max_frames]
        return settled_run(np.array(self.frames), self.previous, sigma=self.sigma())


def chunk_average(F: np.ndarray, size: int, stride: int) -> np.ndarray:
    """Average consecutive frames in overlapping windows (reduces input noise)."""
    if len(F) < size:
        return F.mean(axis=0, keepdims=True)
    starts = range(0, len(F) - size + 1, stride)
    return np.array([F[s:s + size].mean(axis=0) for s in starts])


def _still_target(group: list[LabeledFrame], tol_px: float = 2.0) -> bool:
    """All frames of the group share one target (a dot that does not move)."""
    T = np.array([f.target for f in group], float)
    return bool(np.ptp(T, axis=0).max() <= tol_px) if len(T) else True


def _is_fixation(group: list[LabeledFrame]) -> bool:
    return group[0].kind in FIXATION_KINDS and _still_target(group)


def pooled_noise(groups: Iterable[list[LabeledFrame]]) -> Optional[np.ndarray]:
    """The eye features' frame-to-frame noise over all the dots (None if too few frames)."""
    steps = [np.diff(np.array([f.features for f in g])[:, list(FIXATION_IDX)], axis=0)
             for g in groups if len(g) > 1 and _is_fixation(g)]
    if sum(len(d) for d in steps) < 20:
        return None
    D = np.vstack(steps)
    return np.maximum(1.4826 * np.median(np.abs(D), axis=0) / np.sqrt(2.0), 0.003)


def _group_selection(group: list[LabeledFrame], F: np.ndarray, sigma: Optional[np.ndarray] = None,
                     previous: Optional[tuple[np.ndarray, int]] = None) -> np.ndarray:
    """Frames of a group to train on (see :func:`fixation_frames`)."""
    if _is_fixation(group):
        return fixation_frames(F, sigma=sigma, previous=previous)
    return reject_outliers(F)


def _selections(groups: dict[str, list[LabeledFrame]]) -> dict[str, np.ndarray]:
    """The frames to use of every group. Dots are shown in order, so each dot's
    selection knows where the eyes rested on the dot before it."""
    sigma = pooled_noise(groups.values())
    rest: dict[str, Optional[tuple[np.ndarray, int]]] = {}
    out = {}
    order = sorted(groups.items(), key=lambda item: (item[1][0].kind, item[1][0].point, item[1][0].t))
    for name, group in order:
        F = np.array([f.features for f in group])
        kind, point = group[0].kind, group[0].point
        keep = _group_selection(group, F, sigma, rest.get(f"{kind}:{point - 1}"))
        out[name] = keep
        if _is_fixation(group) and keep.sum() >= 4:
            rest[name] = (F[keep][:, list(FIXATION_IDX)].mean(axis=0), int(keep.sum()))
    return out


def prepare_training_set(frames: list[LabeledFrame], chunk: int = 4, stride: int = 2,
                         head_weight: float = 9.0, min_frames: int = 4):
    """Turn labelled frames into (X, Y, groups, weights, holdout) for GazeNet.

    Every dot (group) weighs 1; ``head_weight`` is the weight of all the
    head-movement steps together, shared equally between them - about 30% of
    a 21-dot calibration. On simulated users more head weight trades a little
    still-head accuracy for much better accuracy after the head moves (head
    weight 3 / 9 / 18: 20.3 / 21.2 / 22.0 px still, 68.7 / 57.8 / 55.1 px
    with the head moved); 9 keeps most of that gain.
    """
    Xs, Ys, Gs, Ws, Hs = [], [], [], [], []
    groups = _group_frames(frames)
    n_head = max(1, sum(1 for group in groups.values() if group[0].kind == "head"))
    selected = _selections(groups)
    for name, group in groups.items():
        F = np.array([f.features for f in group])
        T = np.array([f.target for f in group], float)
        keep = selected[name]
        F, T = F[keep], T[keep]
        kind = group[0].kind
        if len(F) < min_frames:
            continue
        # Moving the head is continuous motion, so average less there.
        size, step = (2, 1) if kind == "head" else (chunk, stride)
        C = chunk_average(F, size, step)
        # Each chunk's own targets (they differ when the target moves, e.g. a
        # pursuit sweep labelled with the pointer position).
        CT = chunk_average(T, size, step)
        gw = (head_weight / n_head if kind == "head" else 1.0) * float(np.mean([f.weight for f in group]))
        Xs.append(C)
        Ys.append(CT)
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

# A one-eye network becomes the main one only if it is clearly better than
# using both eyes (e.g. one eye squints, drifts or is covered).
EYE_PREFERENCE_GAIN = 0.10
EYE_PREFERENCE_MIN_PX = 10.0


def _training_frames(data: CalibrationData, include_validation: bool,
                     kinds: Optional[Iterable[str]]) -> list[LabeledFrame]:
    if kinds is None:
        kinds = ("cal", "head", "val") if include_validation else ("cal", "head")
    return data.of_kind(*kinds)


def train_eye_models(model: GazeNet, X: np.ndarray, Y: np.ndarray, G: np.ndarray, W: np.ndarray,
                     H: np.ndarray, cv_both: Optional[float] = None) -> dict:
    """Train the left- and right-eye networks and choose which network leads.

    Each one-eye network uses the architecture chosen for the two-eye one.
    Their cross-validated errors are compared with the two-eye network's:
    for most people both eyes together are best, but when one eye does not
    track reliably, the other eye alone gives a steadier, more accurate cursor.
    """
    config = ModelConfig.from_dict(model.meta.get("config"))
    errors = {"both": cv_both}
    model.eyes = {}
    for eye, idx in EYE_INPUTS.items():
        net, report = GazeNet.train(X, Y, G, W, holdout=H, min_std=FEATURE_MIN_STD, configs=[config],
                                    inputs=idx, always_cv=True, ensemble=2)
        net.meta["cv_error_px"] = _round(report.cv_error_px)
        model.eyes[eye] = net
        errors[eye] = report.cv_error_px
    preferred = "both"
    both = errors["both"]
    if both is not None and both == both:  # not NaN
        for eye in ("left", "right"):
            e = errors[eye]
            if e == e and e < both * (1.0 - EYE_PREFERENCE_GAIN) and both - e >= EYE_PREFERENCE_MIN_PX:
                if preferred == "both" or e < errors[preferred]:
                    preferred = eye
    model.meta["eye"] = preferred
    model.meta["eye_cv_px"] = {k: _round(v) if v is not None else None for k, v in errors.items()}
    return {"eye": preferred, "eye_cv_px": model.meta["eye_cv_px"]}


def fit_eye_models(model: GazeNet, data: CalibrationData, *, kinds: Iterable[str] = TRAIN_KINDS) -> dict:
    """Add one-eye networks to an existing model (e.g. one saved before they existed)."""
    frames = data.of_kind(*kinds)
    X, Y, G, W, H = prepare_training_set(frames)
    if len({g for g, h in zip(G, H) if h}) < 6:
        raise CalibrationError("Not enough calibration data for the one-eye networks")
    return train_eye_models(model, X, Y, G, W, H, cv_both=model.meta.get("cv_error_px"))


def fit_full_calibration(data: CalibrationData, include_validation: bool = False, *,
                         kinds: Optional[Iterable[str]] = None, configs=None,
                         eyes: bool = True) -> tuple[GazeNet, dict]:
    """Train GazeNet on the labelled frames of the given kinds.

    ``configs`` is the list of candidate architectures (see
    :func:`paralic.gazenet.search_configs`); by default only the weight decay
    is tuned, which keeps the first calibration fast. With ``eyes`` the
    one-eye networks are trained too (see :func:`train_eye_models`).
    """
    frames = _training_frames(data, include_validation, kinds)
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
    eye_info = train_eye_models(model, X, Y, G, W, H, cv_both=report.cv_error_px) if eyes else {}
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
        **eye_info,
    }
    return model, info


def point_summaries(model: GazeNet, frames: list[LabeledFrame], corrected: bool = True,
                    eye: Optional[str] = None) -> list[dict]:
    """Per-dot accuracy (distance of the mean prediction) and precision (spread)."""
    out = []
    groups = _group_frames(frames)
    selected = _selections(groups)
    for name, group in groups.items():
        F = np.array([f.features for f in group])
        T = np.array([f.target for f in group], float)
        keep = selected[name]
        F, T = F[keep], T[keep]
        if len(F) == 0:
            continue
        P = model.predict(F, eye) if corrected else model.predict_uncorrected(F, eye)
        target = T.mean(axis=0)
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
    """Quick re-adjustment: fit an affine correction from a few dots.

    Every network (both eyes and each single eye) gets its own correction;
    the reported numbers are for the one that normally leads.
    """
    frames = data.of_kind("adjust")
    fitted: dict[str, tuple[AffineCorrection, dict]] = {}
    for eye, member in model.members():
        points = point_summaries(member, frames, corrected=False, eye="both")
        if len(points) < 3:
            raise CalibrationError("Not enough adjustment data - keep your eyes on each dot.")
        pred = np.array([p["mean"] for p in points])
        target = np.array([p["target"] for p in points])
        before = member.correction.apply(pred)
        correction = AffineCorrection.fit(pred, target)
        after = correction.apply(pred)
        fitted[eye] = (correction, {
            "n_points": len(points),
            "error_before_px": float(np.linalg.norm(before - target, axis=1).mean()),
            "error_after_px": float(np.linalg.norm(after - target, axis=1).mean()),
        })
    for eye, member in model.members():
        member.correction = fitted[eye][0]
    return fitted[model.eye][1]


def calibrated_pose(data: CalibrationData) -> Optional[dict]:
    """Where the head was during the calibration dots (camera space, cm / degrees).

    The gaze network is most accurate near this position, so the page guides
    people back to it before a quick adjustment. ``x`` grows to the right of
    the camera image and ``y`` upwards (the face transform's convention);
    ``dist`` is the distance from the camera.
    """
    frames = data.of_kind("cal")
    if len(frames) < 10:
        return None
    F = np.array([f.features for f in frames])
    yaw, pitch, tx, ty, tz = np.median(F[:, [14, 15, 17, 18, 19]], axis=0)
    if tz == 0.0:   # no face transform available
        return None
    return {"x": round(float(tx), 1), "y": round(float(ty), 1), "dist": round(float(-tz), 1),
            "yaw": round(float(np.degrees(yaw)), 1), "pitch": round(float(np.degrees(pitch)), 1)}


def _round(v: float) -> Optional[float]:
    return None if v != v else round(float(v), 1)  # NaN -> None (JSON friendly)


# ---------------------------------------------------------------------------
# Saved profile
# ---------------------------------------------------------------------------

# Profiles saved before the full-mesh features (version 1) used these 20
# columns, which still come first and mean the same today.
LEGACY_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES[:20]


def _is_legacy(doc: dict) -> bool:
    return doc.get("version") == 1 and doc.get("feature_names") == list(LEGACY_FEATURE_NAMES)


class ProfileStore:
    """Stores the trained network (and its training data) as a JSON file."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @property
    def backup_path(self) -> Path:
        return self.path.with_name(self.path.stem + ".v1" + self.path.suffix)

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
            "legacy": _is_legacy(doc) or bool(doc.get("model", {}).get("meta", {}).get("legacy")),
        }

    def save(self, model: GazeNet, data: CalibrationData, screen: Optional[dict], accuracy_px: Optional[float]) -> None:
        frames = data.of_kind(*TRAIN_KINDS)
        model.meta["feature_version"] = FEATURE_VERSION
        doc = {
            "version": PROFILE_VERSION,
            "feature_version": FEATURE_VERSION,
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
        self._keep_legacy_backup()
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".profile-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _keep_legacy_backup(self) -> None:
        """Before a version-1 profile is first replaced, keep a copy of it (its
        recordings cannot be used with today's features, but are not thrown away)."""
        try:
            if self.backup_path.exists() or not self.path.is_file():
                return
            doc = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if _is_legacy(doc):
            try:
                shutil.copyfile(self.path, self.backup_path)
            except OSError:
                pass

    def load(self) -> tuple[GazeNet, CalibrationData, dict]:
        doc = self._read()
        if _is_legacy(doc):
            return self._load_legacy(doc)
        # A profile is only usable if it was made with this exact feature set:
        # the gaze network's inputs (count, order and meaning) must match, or it
        # would silently mispredict. A mismatch is not an error to the user —
        # they are simply asked to recalibrate (see session._cmd_profile_load,
        # which turns this into loaded=False and the UI offers calibration).
        if (doc.get("version") != PROFILE_VERSION
                or doc.get("feature_version") != FEATURE_VERSION
                or doc.get("feature_names") != list(FEATURE_NAMES)):
            raise ValueError("This calibration was made with an older face model — please recalibrate.")
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
                             "accuracy_px": doc.get("accuracy_px"), "legacy": bool(model.meta.get("legacy"))}

    def _load_legacy(self, doc: dict) -> tuple[GazeNet, CalibrationData, dict]:
        """A version-1 profile: its networks read the first 20 columns, which
        mean the same today, so they keep working (the one-eye networks already
        list their columns). Its recordings have the old layout and cannot be
        trained on; the model is marked ``legacy`` so fine-tuning waits for a
        new full calibration."""
        model = GazeNet.from_dict(doc["model"])
        for _, member in model.members():
            if member.inputs is None:
                member.inputs = list(range(len(LEGACY_FEATURE_NAMES)))
        model.meta["legacy"] = True
        return model, CalibrationData(), {"created": doc.get("created"), "screen": doc.get("screen"),
                                          "accuracy_px": doc.get("accuracy_px"), "legacy": True}

    def delete(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def _read(self) -> dict:
        doc = json.loads(self.path.read_text(encoding="utf-8"))
        if len(doc.get("feature_names", [])) != NUM_FEATURES and not _is_legacy(doc):
            raise ValueError("feature mismatch")
        return doc
