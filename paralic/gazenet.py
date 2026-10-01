"""GazeNet: the personal gaze-estimation neural network.

MediaPipe's networks tell us *where the eyes and head are*; GazeNet learns, from
your calibration, *where on the screen that means you are looking*.

Architecture (per ensemble member)::

    features ──► Dense(32, tanh) ──► Dense(16, tanh) ──► Dense(2) ──┐
        │                                                           (+) ──► screen x, y
        └──────────────────── linear skip connection ───────────────┘

The linear skip path is initialised with a ridge-regression solution, so the
network starts out as a well-behaved linear model and the hidden layers learn
the non-linear corrections (eyeball curvature, head pose interactions, ...).
Because tanh units saturate, the non-linear part stays bounded when you move
outside the calibrated range, and the linear part extrapolates sensibly.

Training uses full-batch Adam with a Huber loss (robust to frames where you
glanced away), L2 weight decay chosen by grouped cross-validation, and an
ensemble of a few networks with different random initialisations.

A GazeNet can also carry one-eye ("monocular") networks in ``eyes``: they
read only one eye (plus the head pose), keep the cursor moving while the
other eye is closed, and become the main model for people whose other eye
does not track reliably.

Everything is implemented with NumPy only: it trains in about a second on a
laptop CPU and needs no deep-learning framework.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, min_std: Optional[np.ndarray] = None) -> "Scaler":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        floor = np.full_like(std, 1e-6) if min_std is None else np.maximum(np.asarray(min_std, float), 1e-6)
        return cls(mean=mean, std=np.maximum(std, floor))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        return cls(mean=np.asarray(d["mean"], float), std=np.asarray(d["std"], float))


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float, weights: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Weighted ridge regression with an unpenalised intercept. Returns (W, b)."""
    n = X.shape[0]
    w = np.ones(n) if weights is None else weights
    w = w / w.sum()
    xm = w @ X
    ym = w @ Y
    Xc = X - xm
    Yc = Y - ym
    A = (Xc * w[:, None]).T @ Xc + lam * np.eye(X.shape[1])
    W = np.linalg.solve(A, (Xc * w[:, None]).T @ Yc)
    b = ym - xm @ W
    return W, b


def grouped_folds(groups: Sequence, k: int, seed: int = 0) -> list[np.ndarray]:
    """Split sample indices into k folds so that each group lands in one fold."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    k = max(2, min(k, len(uniq)))
    folds = []
    for i in range(k):
        members = uniq[i::k]
        folds.append(np.flatnonzero(np.isin(groups, members)))
    return folds


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

class MLPRegressor:
    """Two hidden tanh layers plus a linear skip connection, trained with Adam.

    All parameters live in one flat vector (``theta``) with named views into
    it, which keeps the optimiser loop cheap in NumPy.
    """

    _PARAMS = ("W1", "b1", "W2", "b2", "W3", "Ws", "b")

    def __init__(self, n_in: int, n_out: int = 2, hidden: tuple[int, int] = (32, 16), seed: int = 0):
        h1, h2 = hidden
        self.shapes = {
            "W1": (n_in, h1), "b1": (h1,), "W2": (h1, h2), "b2": (h2,),
            "W3": (h2, n_out), "Ws": (n_in, n_out), "b": (n_out,),
        }
        sizes = [int(np.prod(self.shapes[k])) for k in self._PARAMS]
        self.theta = np.zeros(sum(sizes))
        self._bind_views()
        rng = np.random.default_rng(seed)
        self.W1[...] = rng.normal(0.0, 1.0 / math.sqrt(n_in), (n_in, h1))
        self.W2[...] = rng.normal(0.0, 1.0 / math.sqrt(h1), (h1, h2))
        # W3 starts at zero: the network begins as the pure linear model.

    def _bind_views(self) -> None:
        offset = 0
        for k in self._PARAMS:
            size = int(np.prod(self.shapes[k]))
            setattr(self, k, self.theta[offset:offset + size].reshape(self.shapes[k]))
            offset += size

    # -- forward / backward -------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        A1 = np.tanh(X @ self.W1 + self.b1)
        A2 = np.tanh(A1 @ self.W2 + self.b2)
        return X @ self.Ws + A2 @ self.W3 + self.b

    def loss_and_grads(self, X: np.ndarray, Y: np.ndarray, weights: np.ndarray, l2: float,
                       l2_skip: float, delta: float) -> tuple[float, np.ndarray]:
        """Loss and the flat gradient vector (same layout as ``theta``)."""
        n = X.shape[0]
        A1 = np.tanh(X @ self.W1 + self.b1)
        A2 = np.tanh(A1 @ self.W2 + self.b2)
        R = X @ self.Ws + A2 @ self.W3 + self.b - Y
        absR = np.abs(R)
        huber = np.where(absR <= delta, 0.5 * R * R, delta * (absR - 0.5 * delta))
        wn = weights[:, None] / n
        loss = float((huber * wn).sum())
        loss += 0.5 * l2 * float((self.W1 ** 2).sum() + (self.W2 ** 2).sum() + (self.W3 ** 2).sum())
        loss += 0.5 * l2_skip * float((self.Ws ** 2).sum())

        G = np.clip(R, -delta, delta) * wn
        dZ2 = (G @ self.W3.T) * (1.0 - A2 * A2)
        dZ1 = (dZ2 @ self.W2.T) * (1.0 - A1 * A1)
        grad = np.concatenate([
            (X.T @ dZ1 + l2 * self.W1).ravel(), dZ1.sum(axis=0),
            (A1.T @ dZ2 + l2 * self.W2).ravel(), dZ2.sum(axis=0),
            (A2.T @ G + l2 * self.W3).ravel(),
            (X.T @ G + l2_skip * self.Ws).ravel(), G.sum(axis=0),
        ])
        return loss, grad

    # -- training -----------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray, weights: Optional[np.ndarray] = None, *, l2: float = 1e-2,
            l2_skip: float = 1e-4, delta: float = 0.35, iters: int = 600, lr: float = 0.01,
            ridge_lam: float = 1e-2) -> "MLPRegressor":
        """Train from scratch: ridge-initialised skip path, then Adam on everything.

        With ``iters=0`` this is exactly a ridge-regression (linear) model.
        """
        n = X.shape[0]
        w = np.ones(n) if weights is None else weights * (n / weights.sum())
        Ws, b = ridge_fit(X, Y, ridge_lam, w)
        self.Ws[...] = Ws
        self.b[...] = b
        self._adam(X, Y, w, l2, l2_skip, delta, iters, lr)
        return self

    def continue_training(self, X: np.ndarray, Y: np.ndarray, weights: Optional[np.ndarray] = None, *,
                          l2: float = 1e-2, l2_skip: float = 1e-4, delta: float = 0.35, iters: int = 300,
                          lr: float = 0.003) -> "MLPRegressor":
        """Fine-tune: keep the current weights and train a little more (warm start)."""
        n = X.shape[0]
        w = np.ones(n) if weights is None else weights * (n / weights.sum())
        self._adam(X, Y, w, l2, l2_skip, delta, iters, lr)
        return self

    def _adam(self, X, Y, w, l2, l2_skip, delta, iters, lr) -> None:
        m = np.zeros_like(self.theta)
        v = np.zeros_like(self.theta)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for step in range(1, iters + 1):
            _, g = self.loss_and_grads(X, Y, w, l2, l2_skip, delta)
            # Cosine learning-rate decay down to 10% of the initial value.
            lr_t = lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * step / iters)))
            m *= beta1
            m += (1.0 - beta1) * g
            v *= beta2
            v += (1.0 - beta2) * g * g
            self.theta -= (lr_t / (1.0 - beta1 ** step)) * m / (np.sqrt(v / (1.0 - beta2 ** step)) + eps)

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in self._PARAMS}

    @classmethod
    def from_dict(cls, d: dict) -> "MLPRegressor":
        W1 = np.asarray(d["W1"], float)
        W2 = np.asarray(d["W2"], float)
        net = cls(W1.shape[0], np.asarray(d["b"]).shape[0], (W1.shape[1], W2.shape[1]))
        for k in cls._PARAMS:
            getattr(net, k)[...] = np.asarray(d[k], float)
        return net


# ---------------------------------------------------------------------------
# Affine correction (quick re-adjustment of a saved calibration)
# ---------------------------------------------------------------------------

@dataclass
class AffineCorrection:
    A: np.ndarray = field(default_factory=lambda: np.eye(2))
    b: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def apply(self, P: np.ndarray) -> np.ndarray:
        return P @ self.A.T + self.b

    @classmethod
    def fit(cls, pred: np.ndarray, target: np.ndarray, strength: float = 0.15) -> "AffineCorrection":
        """Least-squares affine map pred -> target, shrunk towards the identity.

        ``strength`` is the relative weight of the identity prior; with only a
        handful of points this keeps the correction from doing anything wild.
        """
        pred = np.asarray(pred, float)
        target = np.asarray(target, float)
        pm, tm = pred.mean(axis=0), target.mean(axis=0)
        Pc, Tc = pred - pm, target - tm
        spread = float(np.trace(Pc.T @ Pc)) / 2.0
        mu = max(strength * spread, 1e-6)
        At = np.linalg.solve(Pc.T @ Pc + mu * np.eye(2), Pc.T @ Tc + mu * np.eye(2))
        A = At.T
        # Keep the correction physically plausible.
        u, s, vt = np.linalg.svd(A)
        A = u @ np.diag(np.clip(s, 0.6, 1.6)) @ vt
        b = tm - A @ pm
        return cls(A=A, b=b)

    def to_dict(self) -> dict:
        return {"A": self.A.tolist(), "b": self.b.tolist()}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AffineCorrection":
        if not d:
            return cls()
        return cls(A=np.asarray(d["A"], float), b=np.asarray(d["b"], float))


# ---------------------------------------------------------------------------
# GazeNet = scalers + ensemble + correction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """One network architecture to try. ``hidden=None`` means a linear model."""

    hidden: Optional[tuple[int, int]] = (32, 16)
    l2: float = 1e-2

    @property
    def name(self) -> str:
        return "linear" if self.hidden is None else f"{self.hidden[0]}x{self.hidden[1]} l2={self.l2:g}"

    def build(self, n_in: int, seed: int) -> MLPRegressor:
        return MLPRegressor(n_in, 2, self.hidden or (4, 4), seed=seed)

    def fit(self, net: MLPRegressor, X, Y, w, iters: int) -> MLPRegressor:
        if self.hidden is None:  # ridge regression only (the non-linear branch stays at zero)
            return net.fit(X, Y, w, iters=0)
        return net.fit(X, Y, w, l2=self.l2, iters=iters)

    def to_dict(self) -> dict:
        return {"hidden": list(self.hidden) if self.hidden else None, "l2": self.l2}

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ModelConfig":
        if not d:
            return cls()
        return cls(hidden=tuple(d["hidden"]) if d.get("hidden") else None, l2=float(d.get("l2", 1e-2)))


def default_configs() -> list[ModelConfig]:
    """Fast search used right after calibration: weight decay only."""
    return [ModelConfig((32, 16), l2) for l2 in (1e-3, 1e-2, 1e-1)]


def search_configs() -> list[ModelConfig]:
    """Wider per-person model search (run in the background while fine-tuning)."""
    configs = [ModelConfig(h, l2) for h in ((16, 8), (32, 16), (64, 32)) for l2 in (1e-3, 1e-2, 1e-1)]
    return configs + [ModelConfig(None)]


@dataclass
class TrainReport:
    l2: float
    cv_error_px: float
    linear_cv_error_px: float
    train_error_px: float
    n_samples: int
    n_groups: int
    candidates: dict = field(default_factory=dict)
    config: str = ""


EYES = ("both", "left", "right")


class GazeNet:
    HIDDEN = (32, 16)
    L2_GRID = (1e-3, 1e-2, 1e-1)

    def __init__(self, x_scaler: Scaler, y_scaler: Scaler, nets: list[MLPRegressor],
                 correction: Optional[AffineCorrection] = None, meta: Optional[dict] = None,
                 inputs: Optional[Sequence[int]] = None, eyes: Optional[dict] = None):
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.nets = nets
        self.correction = correction or AffineCorrection()
        self.meta = meta or {}
        # Feature columns this network reads (None = all of them).
        self.inputs = None if inputs is None else [int(i) for i in inputs]
        # One-eye networks: "left" / "right" -> GazeNet.
        self.eyes: dict[str, GazeNet] = dict(eyes or {})

    # -- inference ----------------------------------------------------------
    @property
    def eye(self) -> str:
        """Which network ``predict`` uses by default: "both", "left" or "right"."""
        eye = self.meta.get("eye", "both")
        return eye if eye in self.eyes else "both"

    def member(self, eye: Optional[str] = None) -> "GazeNet":
        """The network for ``eye`` (default: the preferred one); falls back to both eyes."""
        eye = self.eye if eye is None else eye
        return self.eyes.get(eye, self) if eye != "both" else self

    def members(self) -> list[tuple[str, "GazeNet"]]:
        return [("both", self)] + [(k, self.eyes[k]) for k in EYES if k in self.eyes]

    def _forward(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        if self.inputs is not None:
            X = X[:, self.inputs]
        # Clamp extreme inputs (e.g. a glitchy frame) to keep outputs sane.
        Z = np.clip(self.x_scaler.transform(X), -6.0, 6.0)
        out = np.mean([net.forward(Z) for net in self.nets], axis=0)
        return self.y_scaler.inverse(out)

    def predict_uncorrected(self, X: np.ndarray, eye: Optional[str] = None) -> np.ndarray:
        return self.member(eye)._forward(X)

    def predict(self, X: np.ndarray, eye: Optional[str] = None) -> np.ndarray:
        """Gaze on screen for full feature vectors ``X`` (one-eye networks pick their columns)."""
        m = self.member(eye)
        return m.correction.apply(m._forward(X))

    def clone(self) -> "GazeNet":
        return GazeNet.from_dict(self.to_dict())

    # -- training -----------------------------------------------------------
    @classmethod
    def train(cls, X: np.ndarray, Y: np.ndarray, groups: Sequence, weights: Optional[np.ndarray] = None, *,
              holdout: Optional[np.ndarray] = None, min_std: Optional[np.ndarray] = None, ensemble: int = 3,
              iters: int = 600, l2_grid: Optional[Sequence[float]] = None,
              configs: Optional[Sequence[ModelConfig]] = None, folds: int = 3,
              seed: int = 0, inputs: Optional[Sequence[int]] = None,
              always_cv: bool = False) -> tuple["GazeNet", TrainReport]:
        """Train an ensemble, picking the architecture / weight decay by grouped cross-validation.

        ``groups`` identifies the calibration point each sample belongs to;
        cross-validation holds out whole points, so it measures how well the
        network *interpolates* to screen positions it has never seen.
        ``holdout`` marks the samples that may be held out (the head-movement
        samples, for example, should always stay in training).
        ``configs`` are the candidate models (an A/B/n test on this person's
        data); by default only the weight decay of the 32x16 network is tuned.
        ``inputs`` restricts the network to some feature columns (the one-eye
        networks); ``always_cv`` measures the cross-validated error even when
        there is only one candidate (without the linear reference).
        """
        X = np.asarray(X, float)
        if inputs is not None:
            inputs = [int(i) for i in inputs]
            X = X[:, inputs]
            if min_std is not None:
                min_std = np.asarray(min_std, float)[inputs]
        Y = np.asarray(Y, float)
        groups = np.asarray(groups)
        n = X.shape[0]
        w = np.ones(n) if weights is None else np.asarray(weights, float)
        can_hold = np.ones(n, bool) if holdout is None else np.asarray(holdout, bool)
        if configs is None:
            grid = l2_grid if l2_grid is not None else cls.L2_GRID
            configs = [ModelConfig(cls.HIDDEN, float(l2)) for l2 in grid]
        configs = list(configs)
        linear = ModelConfig(None)

        def fit_eval(train_idx: np.ndarray, test_idx: np.ndarray, config: ModelConfig) -> np.ndarray:
            xs = Scaler.fit(X[train_idx], min_std)
            ys = Scaler.fit(Y[train_idx])
            Xt, Yt = xs.transform(X[train_idx]), ys.transform(Y[train_idx])
            Xv = np.clip(xs.transform(X[test_idx]), -6.0, 6.0)
            net = config.fit(config.build(X.shape[1], seed), Xt, Yt, w[train_idx], iters)
            return np.linalg.norm(ys.inverse(net.forward(Xv)) - Y[test_idx], axis=1)

        hold_groups = np.unique(groups[can_hold])
        n_groups = len(np.unique(groups))
        candidates: dict[str, float] = {}
        cv_err = lin_err = float("nan")
        best = configs[len(configs) // 2] if len(configs) > 1 else configs[0]
        if len(hold_groups) >= 6 and (len(configs) > 1 or always_cv):
            fold_sets = grouped_folds(hold_groups, folds, seed)
            test_sets = [np.flatnonzero(np.isin(groups, hold_groups[f]) & can_hold) for f in fold_sets]
            all_idx = np.arange(n)

            def cv(config: ModelConfig) -> float:
                errs, ws = [], []
                for test in test_sets:
                    train = np.setdiff1d(all_idx, test)
                    errs.append(fit_eval(train, test, config))
                    ws.append(w[test])
                return float(np.average(np.concatenate(errs), weights=np.concatenate(ws)))

            for config in configs:
                candidates[config.name] = cv(config)
            if linear.name not in candidates and len(configs) > 1:
                candidates[linear.name] = cv(linear)
            best = min(configs, key=lambda c: candidates[c.name])
            cv_err = candidates[best.name]
            lin_err = candidates.get(linear.name, float("nan"))

        xs = Scaler.fit(X, min_std)
        ys = Scaler.fit(Y)
        Xs, Ys = xs.transform(X), ys.transform(Y)
        members = 1 if best.hidden is None else ensemble
        nets = [best.fit(best.build(X.shape[1], seed + 101 * i), Xs, Ys, w, iters) for i in range(members)]
        model = cls(xs, ys, nets, meta={"config": best.to_dict(), "l2": best.l2}, inputs=inputs)
        train_err = float(np.average(np.linalg.norm(
            model.y_scaler.inverse(np.mean([net.forward(Xs) for net in nets], axis=0)) - Y, axis=1), weights=w))
        report = TrainReport(l2=best.l2, cv_error_px=cv_err, linear_cv_error_px=lin_err,
                             train_error_px=train_err, n_samples=n, n_groups=n_groups, candidates=candidates,
                             config=best.name)
        return model, report

    def fine_tune(self, X: np.ndarray, Y: np.ndarray, weights: Optional[np.ndarray] = None, *,
                  iters: int = 300, lr: float = 0.003) -> "GazeNet":
        """Return a copy of this network trained a little further on (X, Y).

        The copy keeps this network's input/output scaling and weights as its
        starting point (a warm start), so a modest amount of new data nudges
        it rather than replacing what it learned during calibration. Any quick
        adjustment is folded in: the copy learns the full mapping itself.
        """
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        tuned = self.clone()
        for _, m in tuned.members():
            m.correction = AffineCorrection()
            Xm = X if m.inputs is None else X[:, m.inputs]
            Xs = np.clip(m.x_scaler.transform(Xm), -6.0, 6.0)
            Ys = m.y_scaler.transform(Y)
            config = ModelConfig.from_dict(m.meta.get("config"))
            for net in m.nets:
                if config.hidden is None:
                    net.fit(Xs, Ys, weights, iters=0)
                else:
                    net.continue_training(Xs, Ys, weights, l2=config.l2, iters=iters, lr=lr)
        tuned.meta = {**self.meta}
        return tuned

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        d = {
            "x_scaler": self.x_scaler.to_dict(),
            "y_scaler": self.y_scaler.to_dict(),
            "nets": [n.to_dict() for n in self.nets],
            "correction": self.correction.to_dict(),
            "meta": self.meta,
        }
        if self.inputs is not None:
            d["inputs"] = list(self.inputs)
        if self.eyes:
            d["eyes"] = {k: m.to_dict() for k, m in self.eyes.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GazeNet":
        return cls(
            x_scaler=Scaler.from_dict(d["x_scaler"]),
            y_scaler=Scaler.from_dict(d["y_scaler"]),
            nets=[MLPRegressor.from_dict(n) for n in d["nets"]],
            correction=AffineCorrection.from_dict(d.get("correction")),
            meta=copy.deepcopy(d.get("meta", {})),   # a clone must not share the original's meta
            inputs=d.get("inputs"),
            eyes={k: cls.from_dict(v) for k, v in (d.get("eyes") or {}).items() if k in ("left", "right")},
        )
