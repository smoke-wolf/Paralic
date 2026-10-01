"""Face print: recognise who is in front of the camera.

Paralic keeps, for each person who allows it, a *face print*: a set of frames
of their face that differ from each other (another head pose, other light,
another distance), each reduced to numbers:

* **face deltas** - the 3D positions of 47 face-mesh points that expressions
  barely move (eye corners, nose, forehead, cheekbones, temples, jaw angles),
  in a frame fixed to the face: origin between the inner eye corners, x along
  the line between the outer eye corners, y towards the base of the nose,
  lengths in eye distances. Head pose, distance and position drop out; what
  is left is the shape of the face (plus a little landmark noise);
* **texture** - histograms of local binary patterns of the grey face, aligned
  by the eyes and contrast-equalised (so overall brightness hardly matters),
  in a 4 x 4 grid.

and the aligned 112 x 112 grey face itself, so the print can be rebuilt when
the method improves.

**Matching** uses a matrix weighting learned from the prints themselves
(:class:`FaceMetric`):

1. *Within-person whitening.* The differences between frames of the same
   person (expressions, pose left over, light, landmark noise) are pooled into
   a covariance, regularised towards its average variance, and inverted:
   directions in which one person's own frames vary a lot count little, the
   steady ones count a lot (a Mahalanobis distance, computed through the
   Woodbury identity from the stored frames, never a 1085 x 1085 matrix).
2. *Discriminant directions.* With two or more people, the directions that
   separate their mean faces (in the whitened space) get extra weight.
3. *Channel weights.* Shape and texture distances are each scaled to "1 = a
   typical difference between two frames of the same person" and weighted by
   how well each separates the people it knows.

A person is recognised when the face is within a few times that person's own
typical frame-to-frame difference *and* clearly closer to them than to anyone
else; otherwise the answer is "not sure", and the page asks. This is a
convenience (who is using the computer), not security: a photo would fool it.

**Improving.** While someone uses Paralic, frames that match them and are
unlike their stored ones (new pose, light, distance - measured in the learned
metric) are added, up to :data:`MAX_SAMPLES`; when full, the most redundant
frame makes way. Every new frame also refines the learned weighting.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

FACEPRINT_VERSION = 1
MAX_SAMPLES = 40
CROP = 112

# Face-mesh points that expressions barely move.
RIGID_POINTS: tuple[int, ...] = (
    33, 133, 362, 263,                                          # eye corners
    168, 6, 197, 195, 5, 4, 1, 2, 98, 327, 48, 278, 129, 358,   # nose
    10, 151, 9, 108, 337, 67, 297, 109, 338,                    # forehead
    116, 345, 50, 280, 123, 352, 117, 346,                      # cheekbones
    127, 356, 234, 454, 93, 323, 162, 389,                      # temples, sides of the face
    132, 361, 58, 288,                                          # jaw angles
)
SHAPE_DIM = 3 * len(RIGID_POINTS)
GRID = 4
TEXTURE_DIM = GRID * GRID * 59
CHANNELS = {"shape": slice(0, SHAPE_DIM), "texture": slice(SHAPE_DIM, SHAPE_DIM + TEXTURE_DIM)}
# Smallest per-dimension within-person variance assumed (landmark noise in eye
# distances; noise of a square-rooted histogram bin).
PRIOR_VAR = {"shape": 0.004 ** 2, "texture": 0.008 ** 2}


# ---------------------------------------------------------------------------
# Describing one frame
# ---------------------------------------------------------------------------

def face_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(origin, rotation rows x/y/z, scale) of the frame fixed to the face."""
    P = np.asarray(points, float)
    if P.shape[1] == 2:
        P = np.column_stack([P, np.zeros(len(P))])
    origin = (P[133] + P[362]) / 2
    x = P[263] - P[33]
    scale = float(np.linalg.norm(x))
    if scale < 1e-6:
        raise ValueError("degenerate face")
    x = x / scale
    v = P[2] - origin
    y = v - (v @ x) * x
    ny = float(np.linalg.norm(y))
    if ny < 1e-6:
        raise ValueError("degenerate face")
    y = y / ny
    z = np.cross(x, y)
    return origin, np.stack([x, y, z]), scale


def shape_descriptor(points: np.ndarray) -> np.ndarray:
    """Face deltas: the rigid points in the face's own frame, in eye distances."""
    P = np.asarray(points, float)
    if P.shape[1] == 2:
        P = np.column_stack([P, np.zeros(len(P))])
    origin, R, scale = face_frame(P)
    return ((P[list(RIGID_POINTS)] - origin) @ R.T / scale).ravel()


def align_face(rgb: np.ndarray, points: np.ndarray, size: int = CROP) -> np.ndarray:
    """The grey face, rotated and scaled so the eyes sit at fixed places, contrast-equalised."""
    import cv2

    P = np.asarray(points, float)[:, :2]
    a = (P[33] + P[133]) / 2           # the eye on the image's left
    b = (P[362] + P[263]) / 2
    da, db = np.array([0.33, 0.40]) * size, np.array([0.67, 0.40]) * size
    src, dst = b - a, db - da
    s = np.linalg.norm(dst) / max(np.linalg.norm(src), 1e-6)
    ang = np.arctan2(dst[1], dst[0]) - np.arctan2(src[1], src[0])
    c, sn = s * np.cos(ang), s * np.sin(ang)
    M = np.array([[c, -sn, 0.0], [sn, c, 0.0]])
    M[:, 2] = da - M[:, :2] @ a
    grey = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    crop = cv2.warpAffine(grey, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(crop)


def _uniform_table() -> np.ndarray:
    """LBP code -> bin: the 58 "uniform" codes (at most two 0/1 changes) get a bin each, the rest share one."""
    table = np.full(256, 58, np.int64)
    k = 0
    for code in range(256):
        bits = [(code >> i) & 1 for i in range(8)]
        if sum(bits[i] != bits[(i + 1) % 8] for i in range(8)) <= 2:
            table[code] = k
            k += 1
    return table


_UNIFORM = _uniform_table()


def texture_descriptor(face: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Square-rooted histograms of uniform local binary patterns in a grid of cells."""
    g = face.astype(np.int16)
    c = g[1:-1, 1:-1]
    nbrs = (g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:], g[1:-1, 2:], g[2:, 2:], g[2:, 1:-1], g[2:, :-2], g[1:-1, :-2])
    code = np.zeros(c.shape, np.int64)
    for k, n in enumerate(nbrs):
        code |= (n >= c).astype(np.int64) << k
    bins = _UNIFORM[code]
    h, w = bins.shape
    out = []
    for i in range(grid):
        for j in range(grid):
            cell = bins[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid]
            hist = np.bincount(cell.ravel(), minlength=59).astype(float)
            out.append(np.sqrt(hist / max(1.0, hist.sum())))
    return np.concatenate(out)


@dataclass
class FaceSample:
    """One frame of a face print."""

    shape: np.ndarray
    texture: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0
    light: float = 0.0          # mean brightness of the face (before equalising)
    t: float = 0.0
    id: str = ""
    crop: Optional[np.ndarray] = None

    @property
    def vector(self) -> np.ndarray:
        return np.concatenate([self.shape, self.texture])


def make_sample(rgb: np.ndarray, points: np.ndarray, yaw: float = 0.0, pitch: float = 0.0,
                t: Optional[float] = None) -> Optional[FaceSample]:
    """Describe the face in one frame (None if it is too small or degenerate)."""
    if not isinstance(points, np.ndarray) or not isinstance(rgb, np.ndarray):
        return None
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[0] < 468:
        return None
    eye_px = float(np.linalg.norm(P[263, :2] - P[33, :2]))
    if eye_px < 30:
        return None
    try:
        shape = shape_descriptor(P)
    except ValueError:
        return None
    face = align_face(rgb, P)
    import cv2

    grey = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    x0, y0 = np.maximum(P[:, :2].min(axis=0).astype(int), 0)
    x1, y1 = P[:, :2].max(axis=0).astype(int)
    light = float(grey[y0:y1, x0:x1].mean()) if y1 > y0 and x1 > x0 else 0.0
    return FaceSample(shape=shape, texture=texture_descriptor(face), yaw=float(yaw), pitch=float(pitch),
                      light=light, t=time.time() if t is None else t, id=uuid.uuid4().hex[:12], crop=face)


# ---------------------------------------------------------------------------
# The learned matrix weighting
# ---------------------------------------------------------------------------

class FaceMetric:
    """Distance between face prints, weighted by what was learned from them.

    ``galleries``: person -> (frames x 1085) descriptor rows. See the module
    docstring for the layers (within-person whitening, discriminant
    directions, channel weights).

    The weighting is *cross-validated*: each person's frames are split into
    ``folds`` parts; for each part a weighting is learned from the rest and
    measured on that part (judged on the frames it learned from, any learned
    distance looks too good). A new frame is then scored by every fold's
    weighting against that fold's frames, each in its own measured units, and
    the scores averaged - so scores mean the same as in the measurement.
    ``refs[person]``: that person's typical score for a new frame of their face.
    """

    def __init__(self, galleries: dict[str, np.ndarray], boost: float = 1.0, folds: int = 4, k: int = 3):
        self.galleries = {u: np.asarray(X, float) for u, X in galleries.items() if len(X)}
        self.k = k
        self.models: list[dict] = []
        held: list[tuple[str, int, dict]] = []      # (person, model, {person: {channel: d2 array}})
        for f in range(folds):
            train, test = {}, {}
            for u, X in self.galleries.items():
                out = (np.arange(len(X)) % folds == f) if len(X) >= 3 else np.zeros(len(X), bool)
                train[u], test[u] = X[~out], X[out]
            if not any(len(T) for T in test.values()):
                continue
            model = {"train": train, "channels": self._build(train, boost)}
            m = len(self.models)
            self.models.append(model)
            for u, T in test.items():
                for x in T:
                    held.append((u, m, {v: {n: self._d2_channel(ch, (Xv - x)[:, ch["sl"]])
                                            for n, ch in model["channels"].items()}
                                        for v, Xv in train.items() if len(Xv)}))
        if not self.models:
            self.models = [{"train": self.galleries, "channels": self._build(self.galleries, boost)}]
        # Each fold's units: 1 = a held-out frame's typical distance to its own
        # person's nearest frames (pooled over the folds when a fold has few).
        pooled = {n: [self._knn(rec[u][n]) for u, _, rec in held if u in rec] for n in CHANNELS}
        for m, model in enumerate(self.models):
            for n, ch in model["channels"].items():
                mine = [self._knn(rec[u][n]) for u, mm, rec in held if mm == m and u in rec]
                vals = mine if len(mine) >= 3 else pooled[n]
                if vals:
                    ch["norm"] = max(float(np.median(vals)), 1e-9)
        # Weight each channel by how well it tells the people apart.
        within = {n: [self._knn(rec[u][n]) / self.models[m]["channels"][n]["norm"]
                      for u, m, rec in held if u in rec] for n in CHANNELS}
        between = {n: [self._knn(rec[v][n]) / self.models[m]["channels"][n]["norm"]
                       for u, m, rec in held for v in rec if v != u] for n in CHANNELS}
        if all(within[n] and between[n] for n in CHANNELS):
            raw = {n: float(np.clip(np.log(max(np.median(between[n]) / np.median(within[n]), 1.0001)), 0.2, 5.0))
                   for n in CHANNELS}
            mean = sum(raw.values()) / len(raw)
            for model in self.models:
                for n in CHANNELS:
                    model["channels"][n]["weight"] = raw[n] / mean
        # Each person's typical score for a new frame of their own face.
        own: dict[str, list[float]] = {}
        for u, m, rec in held:
            if u in rec:
                chans = self.models[m]["channels"]
                total = sum(chans[n]["weight"] * rec[u][n] / chans[n]["norm"] for n in CHANNELS) / len(CHANNELS)
                own.setdefault(u, []).append(self._knn(total))
        everyone = [v for vals in own.values() for v in vals]
        overall = float(np.median(everyone)) if everyone else 1.0
        self.refs = {u: max(float(np.median(own[u])) if len(own.get(u, [])) >= 3 else overall, 1e-6)
                     for u in self.galleries}

    @classmethod
    def _build(cls, galleries: dict[str, np.ndarray], boost: float) -> dict[str, dict]:
        groups = [X for X in galleries.values() if len(X) >= 2]
        n_w = sum(len(X) - 1 for X in groups)
        devs = [X - X.mean(axis=0) for X in groups]
        means = [X.mean(axis=0) for X in galleries.values() if len(X)]
        channels = {}
        for name, sl in CHANNELS.items():
            dim = sl.stop - sl.start
            E = np.vstack([d[:, sl] for d in devs]) if devs else np.zeros((0, dim))
            # The floor for directions the frames do not cover: the variance left
            # after the main ways one person's frames differ (those are handled
            # by the whitening itself) - not the overall average, which they
            # would inflate and so blunt every other direction.
            rho = PRIOR_VAR[name]
            if len(E) >= 2:
                lam = np.linalg.svd(E, compute_uv=False) ** 2 / max(1, n_w)
                top = max(1, min(len(lam) - 1, n_w // 3))
                rho = max(float(lam[top:].sum()) / max(1, dim - top), rho)
            K = np.linalg.inv(rho * max(1, n_w) * np.eye(len(E)) + E @ E.T) if len(E) else None
            ch = {"sl": sl, "E": E, "K": K, "rho": rho, "dirs": np.zeros((0, dim)), "norm": float(dim), "weight": 1.0}
            # Discriminant directions: whitened differences of the people's mean faces,
            # scaled so that one person's own frames vary by about 1 along each.
            if len(means) >= 2 and boost > 0:
                M = np.array([m[sl] for m in means])
                D = M - M.mean(axis=0)
                W = cls._apply_inverse(ch, D)
                norms = np.sqrt(np.maximum((W * D).sum(axis=1), 1e-12))
                ch["dirs"] = np.sqrt(boost) * W / norms[:, None]
            channels[name] = ch
        return channels

    def _knn(self, d: np.ndarray) -> float:
        return float(np.sort(d)[:min(self.k, len(d))].mean())

    def _total(self, model: dict, x: np.ndarray, Y: np.ndarray) -> np.ndarray:
        V = np.atleast_2d(Y) - np.asarray(x, float)
        chans = model["channels"]
        return sum(ch["weight"] * self._d2_channel(ch, V[:, ch["sl"]]) / ch["norm"]
                   for ch in chans.values()) / len(chans)

    @staticmethod
    def _apply_inverse(ch: dict, V: np.ndarray) -> np.ndarray:
        """Rows of V times (within covariance + rho I)^-1 (Woodbury)."""
        E, K, rho = ch["E"], ch["K"], ch["rho"]
        if K is None:
            return V / rho
        return (V - (V @ E.T) @ K @ E) / rho

    @staticmethod
    def _d2_channel(ch: dict, V: np.ndarray) -> np.ndarray:
        V = np.atleast_2d(V)
        E, K, rho = ch["E"], ch["K"], ch["rho"]
        d2 = (V * V).sum(axis=1)
        if K is not None:
            EV = V @ E.T
            d2 = d2 - ((EV @ K) * EV).sum(axis=1)
        d2 = d2 / rho
        if len(ch["dirs"]):
            d2 = d2 + ((V @ ch["dirs"].T) ** 2).sum(axis=1)
        return d2

    def d2(self, x: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Learned squared distances from descriptor ``x`` to each row of ``Y`` (all folds averaged)."""
        return np.mean([self._total(model, x, Y) for model in self.models], axis=0)

    def score(self, x: np.ndarray, user: str) -> float:
        """Distance from ``x`` to the person's ``k`` most similar frames, averaged over the folds."""
        vals = [self._knn(self._total(model, x, model["train"][user])) for model in self.models
                if len(model["train"].get(user, []))]
        return float(np.mean(vals)) if vals else float("inf")


# ---------------------------------------------------------------------------
# Recognising and learning
# ---------------------------------------------------------------------------

# A face is someone's when its score is within ACCEPT x their typical score and
# MARGIN x closer to them than to anyone else.
ACCEPT = 3.0
MARGIN = 1.6
STRICT = 1.8


# Frames a print needs before it is used to recognise someone.
MIN_SAMPLES = 6


class FaceRecognizer:
    """Recognise people from several frames, and decide which frames to keep."""

    def __init__(self, prints: dict[str, list[FaceSample]]):
        self.prints = {u: list(s) for u, s in prints.items() if len(s) >= MIN_SAMPLES}
        self.metric = FaceMetric({u: np.array([s.vector for s in ss]) for u, ss in self.prints.items()})
        self.refs = self.metric.refs

    def relative_scores(self, samples: Iterable[FaceSample]) -> dict[str, float]:
        """Each person's mean score over the samples, in units of their typical score."""
        vecs = [s.vector for s in samples]
        if not vecs:
            return {}
        return {u: float(np.mean([self.metric.score(v, u) for v in vecs])) / self.refs[u] for u in self.prints}

    def identify(self, samples: Iterable[FaceSample], strict: bool = False) -> dict:
        """Who these frames show: {"user": id or None, "confident": bool, "scores": {...}}.
        ``strict`` asks for a closer match (before frames may be learned)."""
        rel = self.relative_scores(list(samples))
        if not rel:
            return {"user": None, "confident": False, "scores": {}}
        ranked = sorted(rel.items(), key=lambda kv: kv[1])
        best, s1 = ranked[0]
        s2 = ranked[1][1] if len(ranked) > 1 else float("inf")
        confident = s1 <= (STRICT if strict else ACCEPT) and s2 >= MARGIN * s1
        return {"user": best if confident else None, "best": best, "confident": confident,
                "scores": {u: round(v, 2) for u, v in ranked}}

    def novelty(self, sample: FaceSample, user: str) -> float:
        """How unlike the person's stored frames this one is (their nearest frame, in typical scores)."""
        X = self.metric.galleries.get(user)
        if X is None or not len(X):
            return float("inf")
        return float(self.metric.d2(sample.vector, X).min()) / self.refs.get(user, 1.0)


def choose_samples(samples: list[FaceSample], new: FaceSample, metric: Optional[FaceMetric] = None,
                   cap: int = MAX_SAMPLES) -> list[FaceSample]:
    """Add ``new``; when over ``cap``, drop the frame most like another one."""
    out = samples + [new]
    if len(out) <= cap:
        return out
    X = np.array([s.vector for s in out])
    if metric is None:
        metric = FaceMetric({"_": X})
    nearest = []
    for i in range(len(out)):
        d = metric.d2(X[i], X)
        d[i] = np.inf
        nearest.append(d.min())
    drop = int(np.argmin(nearest))
    return out[:drop] + out[drop + 1:]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class FacePrintStore:
    """A person's face print on disk: ``print.npz`` plus the aligned faces as JPEGs."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)

    @property
    def faces(self) -> Path:
        return self.folder / "faces"

    def exists(self) -> bool:
        return (self.folder / "print.npz").is_file()

    def load(self) -> list[FaceSample]:
        path = self.folder / "print.npz"
        if not path.is_file():
            return []
        try:
            with np.load(path, allow_pickle=False) as z:
                if int(z["version"]) != FACEPRINT_VERSION or z["shape"].shape[1:] != (SHAPE_DIM,) \
                        or z["texture"].shape[1:] != (TEXTURE_DIM,):
                    return self._rebuild()
                meta = json.loads(str(z["meta"]))
                return [FaceSample(shape=z["shape"][i], texture=z["texture"][i], **meta[i])
                        for i in range(len(meta))]
        except (OSError, ValueError, KeyError):
            return []

    def _rebuild(self) -> list[FaceSample]:
        """An older print: describe the kept faces again with today's method (texture only
        can be recomputed from a picture; the shape needs the mesh, so such frames are dropped)."""
        return []

    def save(self, samples: list[FaceSample]) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.faces.mkdir(exist_ok=True)
        import cv2

        for s in samples:
            if s.crop is not None:
                pic = self.faces / f"{s.id}.jpg"
                if not pic.exists():
                    cv2.imwrite(str(pic), s.crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        keep = {f"{s.id}.jpg" for s in samples}
        for pic in self.faces.glob("*.jpg"):
            if pic.name not in keep:
                pic.unlink(missing_ok=True)
        meta = [{"yaw": s.yaw, "pitch": s.pitch, "light": s.light, "t": s.t, "id": s.id} for s in samples]
        fd, tmp = tempfile.mkstemp(dir=self.folder, prefix=".print-", suffix=".npz")
        os.close(fd)
        try:
            np.savez_compressed(tmp, version=FACEPRINT_VERSION,
                                shape=np.array([s.shape for s in samples]).reshape(len(samples), SHAPE_DIM),
                                texture=np.array([s.texture for s in samples]).reshape(len(samples), TEXTURE_DIM),
                                meta=json.dumps(meta))
            os.replace(tmp, self.folder / "print.npz")
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def picture(self, sample_id: str) -> Optional[bytes]:
        pic = self.faces / f"{sample_id}.jpg"
        try:
            return pic.read_bytes()
        except OSError:
            return None

    def forget(self) -> None:
        shutil.rmtree(self.folder, ignore_errors=True)
