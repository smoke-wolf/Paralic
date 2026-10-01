"""Contrastive fine-tuning for the "manual" calibration mode.

In manual mode the person uses the app for a few minutes and:

* holds **Shift** while the cursor is on target  -> a *positive* sample: the
  current snapped cursor is the ground-truth gaze for those frames, so we pull
  the network's prediction toward it;
* presses **.** while the cursor is wrong -> the screen pauses and they press an
  **arrow key** for the direction the cursor *should* have gone (up/down/left/
  right, relative to where it is). That gives a *directional* negative: we push
  the prediction along that arrow, away from the frozen wrong point.

Negatives give a direction but not an exact target, so used naively they
destabilise the model (as the mode's trade-off warns). This module keeps them
safe:

* the pull on positives is the real objective; the push on negatives is a
  bounded **directional hinge** (move at least ``margin`` along the arrow, then
  stop) with a small weight ``lam_neg``;
* we measure progress from the *frozen* champion prediction, computed once,
  never from the model's moving output (no self-reference feedback loop);
* training is a short warm-start from the champion (a fine-tune, not a retrain);
* the result is accepted **only if** held-out positive accuracy does not get
  worse — otherwise the champion is kept unchanged.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .gazenet import GazeNet, MLPRegressor


class ContrastiveError(Exception):
    pass


def _backward_from_output_grad(net: MLPRegressor, X: np.ndarray, g_out: np.ndarray) -> np.ndarray:
    """Flat parameter gradient for an arbitrary upstream dL/d(output) = ``g_out``.

    Mirrors ``MLPRegressor.loss_and_grads`` exactly, but with the output-space
    gradient supplied directly (no Huber, no weight decay). Layout matches
    ``net.theta`` so the two gradients can simply be added.
    """
    A1 = np.tanh(X @ net.W1 + net.b1)
    A2 = np.tanh(A1 @ net.W2 + net.b2)
    dZ2 = (g_out @ net.W3.T) * (1.0 - A2 * A2)
    dZ1 = (dZ2 @ net.W2.T) * (1.0 - A1 * A1)
    return np.concatenate([
        (X.T @ dZ1).ravel(), dZ1.sum(axis=0),
        (A1.T @ dZ2).ravel(), dZ2.sum(axis=0),
        (A2.T @ g_out).ravel(),
        (X.T @ g_out).ravel(), g_out.sum(axis=0),
    ])


def _directional_grad(out: np.ndarray, frozen: np.ndarray, dirs: np.ndarray,
                      margin: float, weight: float) -> np.ndarray:
    """dL/d(output) for the directional hinge 0.5 * relu(margin - (out-frozen)·dir)^2.

    Pushes ``out`` along the unit direction ``dir`` (the arrow the user pressed),
    starting from the frozen wrong point, until it has moved ``margin`` along it.
    ``dirs`` are unit vectors in the standardised output space.
    """
    s = ((out - frozen) * dirs).sum(axis=1, keepdims=True)   # progress along the arrow
    hinge = np.clip(margin - s, 0.0, None)
    return -weight * hinge * dirs


def contrastive_finetune(
    champion: GazeNet,
    pos_X: np.ndarray, pos_Y: np.ndarray, pos_W: Optional[np.ndarray] = None,
    neg_X: Optional[np.ndarray] = None, neg_dir: Optional[np.ndarray] = None, *,
    margin: float = 0.6, lam_neg: float = 0.4, iters: int = 160, lr: float = 0.003,
    l2: float = 1e-2, delta: float = 0.35, test_fraction: float = 0.3,
    min_positives: int = 20, tol: float = 0.02, min_gain: float = 0.03, seed: int = 0,
    eye: Optional[str] = None,
) -> tuple[Optional[GazeNet], dict]:
    """Fine-tune ``champion`` with manual-mode positives and directional negatives.

    ``neg_dir`` are screen-space unit vectors (the arrow the user pressed for each
    negative frame; e.g. up = (0, -1)). Returns (new model or None, report).
    ``None`` means the result was rejected (not enough data, or it did not keep
    held-out positive accuracy) and the caller should keep the champion.

    ``margin`` is in units of the output standardisation (≈ std of calibration
    targets), so it is resolution independent.

    The network tuned (and judged) is the one that drives the cursor: ``eye``
    ("both", "left", "right"), by default the model's leading network - for
    some people a one-eye network leads (see ``GazeNet.eyes``). The other
    networks are left as they are.
    """
    started = time.perf_counter()
    pos_X = np.asarray(pos_X, float)
    pos_Y = np.asarray(pos_Y, float)
    n_pos = pos_X.shape[0]
    if n_pos < min_positives:
        raise ContrastiveError(
            f"Need at least {min_positives} 'working' (Shift) frames; have {n_pos}. "
            "Hold Shift for longer while the cursor is on target.")
    pos_W = np.ones(n_pos) if pos_W is None else np.asarray(pos_W, float)
    neg_X = np.zeros((0, pos_X.shape[1])) if neg_X is None else np.asarray(neg_X, float)
    neg_dir = np.zeros((0, 2)) if neg_dir is None else np.asarray(neg_dir, float)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_pos)
    n_test = max(5, int(round(test_fraction * n_pos)))
    n_test = min(n_test, n_pos - min_positives // 2) if n_pos - n_test >= 5 else max(1, n_pos // 5)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    def pos_error(model: GazeNet, idx: np.ndarray) -> float:
        if len(idx) == 0:
            return float("nan")
        return float(np.mean(np.linalg.norm(model.predict(pos_X[idx], eye) - pos_Y[idx], axis=1)))

    champ_err = pos_error(champion, test_idx)
    # Baseline: how well does simply guessing the training centroid do on the
    # held-out positives? Real on-target data is far more predictable than this;
    # random/garbage "positives" are not, so a model that cannot beat the
    # centroid has learned nothing generalisable and must be rejected.
    centroid = pos_Y[train_idx].mean(axis=0)
    baseline_err = float(np.mean(np.linalg.norm(pos_Y[test_idx] - centroid, axis=1)))

    tuned = champion.clone()
    lead = tuned.member(eye)                         # the network predict() uses
    lead.correction = type(lead.correction)()        # fold any affine adjust into the net
    cols = lead.inputs                               # a one-eye network reads only its columns
    sel = (lambda A: A) if cols is None else (lambda A: A[:, cols])
    xs, ys = lead.x_scaler, lead.y_scaler
    Xtr = np.clip(xs.transform(sel(pos_X[train_idx])), -6.0, 6.0)
    Ytr = ys.transform(pos_Y[train_idx])
    wtr = pos_W[train_idx]
    wtr = wtr * (len(wtr) / max(wtr.sum(), 1e-9))
    have_neg = neg_X.shape[0] > 0 and neg_dir.shape[0] == neg_X.shape[0]
    if have_neg:
        Xneg = np.clip(xs.transform(sel(neg_X)), -6.0, 6.0)
        # Arrow directions are in screen pixels; convert to the standardised
        # output space (divide by the per-axis target std) and renormalise.
        Dneg = neg_dir / np.maximum(ys.std, 1e-9)
        Dneg = Dneg / np.maximum(np.linalg.norm(Dneg, axis=1, keepdims=True), 1e-9)

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for net in lead.nets:
        # Frozen "wrong" predictions for the negatives (standardised output space).
        frozen = net.forward(Xneg) if have_neg else None
        m = np.zeros_like(net.theta)
        v = np.zeros_like(net.theta)
        for step in range(1, iters + 1):
            _, g_pos = net.loss_and_grads(Xtr, Ytr, wtr, l2, 1e-4, delta)
            g = g_pos
            if have_neg:
                out = net.forward(Xneg)
                g_out = _directional_grad(out, frozen, Dneg, margin, lam_neg / max(Xneg.shape[0], 1))
                g = g + _backward_from_output_grad(net, Xneg, g_out)
            lr_t = lr * (0.1 + 0.45 * (1.0 + np.cos(np.pi * step / iters)))
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            net.theta -= (lr_t / (1 - beta1 ** step)) * m / (np.sqrt(v / (1 - beta2 ** step)) + eps)

    tuned.meta = {**champion.meta}
    tuned_err = pos_error(tuned, test_idx)

    # Safety gate: keep the fine-tune only if, on held-out positives, it (a) does
    # not regress versus the champion and (b) clearly beats the centroid baseline
    # (so unlearnable / mislabelled data is rejected).
    accepted = (np.isfinite(tuned_err)
                and tuned_err <= champ_err * (1.0 + tol)
                and tuned_err <= (1.0 - min_gain) * baseline_err)
    report = {
        "accepted": bool(accepted),
        "n_positives": int(n_pos),
        "n_negatives": int(neg_X.shape[0]),
        "n_test": int(n_test),
        "champion_error_px": round(float(champ_err), 1) if np.isfinite(champ_err) else None,
        "tuned_error_px": round(float(tuned_err), 1) if np.isfinite(tuned_err) else None,
        "baseline_error_px": round(baseline_err, 1),
        "improvement": round(float((champ_err - tuned_err) / champ_err), 4) if champ_err else None,
        "margin": margin, "lam_neg": lam_neg, "iters": iters,
        "seconds": round(time.perf_counter() - started, 2),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if not accepted:
        return None, report

    # Refit on all positives (train + test) for the model we actually keep.
    Xall = np.clip(xs.transform(sel(pos_X)), -6.0, 6.0)
    Yall = ys.transform(pos_Y)
    wall = pos_W * (n_pos / max(pos_W.sum(), 1e-9))
    final = champion.clone()
    final_lead = final.member(eye)
    final_lead.correction = type(final_lead.correction)()
    for net in final_lead.nets:
        frozen = net.forward(Xneg) if have_neg else None
        m = np.zeros_like(net.theta)
        v = np.zeros_like(net.theta)
        for step in range(1, iters + 1):
            _, g_pos = net.loss_and_grads(Xall, Yall, wall, l2, 1e-4, delta)
            g = g_pos
            if have_neg:
                out = net.forward(Xneg)
                g_out = _directional_grad(out, frozen, Dneg, margin, lam_neg / max(Xneg.shape[0], 1))
                g = g + _backward_from_output_grad(net, Xneg, g_out)
            lr_t = lr * (0.1 + 0.45 * (1.0 + np.cos(np.pi * step / iters)))
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            net.theta -= (lr_t / (1 - beta1 ** step)) * m / (np.sqrt(v / (1 - beta2 ** step)) + eps)
    final.meta = {**champion.meta, "source": "manual-contrastive",
                  "trained_ts": time.time(), "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return final, report
