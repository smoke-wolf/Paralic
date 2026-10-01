"""What the gaze networks compute for one frame, step by step.

GazeNet (``paralic/gazenet.py``) turns a frame's feature vector into a screen
point. :func:`trace` runs the same steps on the same numbers and keeps
everything in between:

1. the inputs: the feature columns the network reads (all 28 for the network
   that reads both eyes, 17 for a one-eye network) and their names,
2. standardised with the mean and standard deviation of the calibration data,
   then clipped to +-6 (a glitchy frame cannot produce a wild output),
3. for each ensemble member: both hidden layers before and after tanh, and
   its output split into the linear skip path, the hidden path and the bias,
4. the mean of the members, scaled back to screen pixels,
5. the affine correction fitted by a quick adjust, and the final point.

The final point equals ``GazeNet.predict`` exactly - the same operations in
the same order (``tests/test_inspector.py`` checks it).

:func:`sensitivity` adds the exact derivatives of the final point with respect
to every input: how far the gaze moves per standard deviation of a feature,
and the feature's first-order share of the current point (derivative times
its distance from the calibration mean). :func:`ablation` replaces one input
at a time by its calibration mean and reports how far the point moves - the
whole effect, not only the first-order one.

:func:`hand_trace` does the same for hand mode's recogniser: palm width, the
pinch distance and the "stop" palm's tests, from the 21 hand landmarks.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..features import FEATURE_NAMES
from ..gazenet import EYES, GazeNet

CLIP = 6.0              # GazeNet clips standardised inputs to this


def _columns(net: GazeNet, n_features: int) -> list[int]:
    return list(range(n_features)) if net.inputs is None else list(net.inputs)


def trace(net: GazeNet, features: Sequence[float]) -> dict:
    """Every step of one network (no one-eye networks) for one feature vector.

    Arrays are float64; ``final`` is ``net.predict(features)`` for this network.
    Shapes: ``z`` (n_in,), member ``pre1``/``act1`` (h1,), ``pre2``/``act2`` (h2,).
    """
    X = np.atleast_2d(np.asarray(features, float))
    cols = _columns(net, X.shape[1])
    Xs = X if net.inputs is None else X[:, net.inputs]
    z_raw = net.x_scaler.transform(Xs)
    Z = np.clip(z_raw, -CLIP, CLIP)
    y_mean, y_std = net.y_scaler.mean, net.y_scaler.std
    members, outs = [], []
    for k, m in enumerate(net.nets):
        # The operations of MLPRegressor.forward, keeping the intermediate values.
        pre1 = Z @ m.W1 + m.b1
        act1 = np.tanh(pre1)
        pre2 = act1 @ m.W2 + m.b2
        act2 = np.tanh(pre2)
        out = Z @ m.Ws + act2 @ m.W3 + m.b
        outs.append(out)
        skip = (Z @ m.Ws)[0]
        hidden = (act2 @ m.W3)[0]
        members.append({
            "index": k,
            "linear": not bool(np.any(m.W3)),        # a ridge model: the hidden layers do not count
            "pre1": pre1[0], "act1": act1[0], "pre2": pre2[0], "act2": act2[0],
            "out": out[0],                              # standardised output
            "out_px": net.y_scaler.inverse(out)[0],
            # The output in pixels = skip + hidden + bias (each scaled by the output std).
            "skip_px": skip * y_std,
            "hidden_px": hidden * y_std,
            "bias_px": m.b * y_std + y_mean,
            # Per input: its share of the linear skip path; per layer-2 unit: its share of the hidden path.
            "skip_by_input_px": Z[0][:, None] * m.Ws * y_std,
            "hidden_by_unit_px": act2[0][:, None] * m.W3 * y_std,
        })
    ens = np.mean(outs, axis=0)                     # as GazeNet._forward
    ens_px = net.y_scaler.inverse(ens)
    final = net.correction.apply(ens_px)
    member_px = np.array([m["out_px"] for m in members])
    meta = net.meta or {}
    return {
        "inputs": cols,
        "names": [FEATURE_NAMES[c] if c < len(FEATURE_NAMES) else f"f{c}" for c in cols],
        "raw": Xs[0],
        "mean": net.x_scaler.mean,
        "std": net.x_scaler.std,
        "z_raw": z_raw[0],
        "z": Z[0],
        "clipped": np.abs(z_raw[0]) > CLIP,
        "hidden": [int(net.nets[0].W1.shape[1]), int(net.nets[0].W2.shape[1])] if net.nets else [],
        "members": members,
        "ensemble": ens[0],
        "ensemble_px": ens_px[0],
        "spread_px": float(np.sqrt(((member_px - ens_px[0]) ** 2).sum(axis=1).mean())) if len(members) > 1 else 0.0,
        "correction": {"A": net.correction.A, "b": net.correction.b,
                       "identity": bool(np.allclose(net.correction.A, np.eye(2))
                                        and np.allclose(net.correction.b, 0.0))},
        "final": final[0],
        "config": meta.get("config"),
        "cv_error_px": meta.get("cv_error_px"),
    }


def jacobian(net: GazeNet, features: Sequence[float]) -> dict:
    """Exact derivatives of the final point with respect to the network's inputs.

    ``dz``: pixels per standard deviation of each input (n_in, 2);
    ``dx``: pixels per unit of each raw feature (n_in, 2), zero for a clipped input.
    """
    X = np.atleast_2d(np.asarray(features, float))
    Xs = X if net.inputs is None else X[:, net.inputs]
    z_raw = net.x_scaler.transform(Xs)[0]
    Z = np.clip(z_raw, -CLIP, CLIP)[None, :]
    J = np.zeros((Z.shape[1], 2))
    for m in net.nets:
        act1 = np.tanh(Z @ m.W1 + m.b1)
        act2 = np.tanh(act1 @ m.W2 + m.b2)
        # d out / d z = Ws + W1 diag(1 - act1^2) W2 diag(1 - act2^2) W3
        J += m.Ws + (m.W1 * (1.0 - act1 ** 2)) @ ((m.W2 * (1.0 - act2 ** 2)) @ m.W3)
    J /= max(len(net.nets), 1)
    dz = (J * net.y_scaler.std) @ net.correction.A.T
    live = (np.abs(z_raw) <= CLIP).astype(float)
    dx = dz * (live / net.x_scaler.std)[:, None]
    return {"dz": dz * live[:, None], "dx": dx, "z": Z[0]}


def sensitivity(net: GazeNet, features: Sequence[float], top: Optional[int] = None) -> list[dict]:
    """Which inputs drive the gaze right now, strongest first.

    For each input: ``dz`` (px per standard deviation, x and y), ``dx`` (px
    per unit of the raw feature), ``share`` (``dz`` times the input's distance
    from the calibration mean, in px: its first-order part of the current
    point) and ``strength`` (the length of ``dz``).
    """
    jac = jacobian(net, features)
    cols = _columns(net, np.atleast_2d(features).shape[1])
    rows = []
    for k, c in enumerate(cols):
        dz = jac["dz"][k]
        rows.append({
            "input": c,
            "name": FEATURE_NAMES[c] if c < len(FEATURE_NAMES) else f"f{c}",
            "z": float(jac["z"][k]),
            "dz": dz,
            "dx": jac["dx"][k],
            "share": dz * jac["z"][k],
            "strength": float(np.hypot(*dz)),
        })
    rows.sort(key=lambda r: -r["strength"])
    return rows[:top] if top else rows


def ablation(net: GazeNet, features: Sequence[float]) -> np.ndarray:
    """How far the final point moves (n_in, 2 px) when each input is replaced by its calibration mean."""
    X = np.atleast_2d(np.asarray(features, float))[0]
    cols = _columns(net, len(X))
    # eye="both": this network itself, even if it is a model that prefers a one-eye member.
    base = net.predict(X[None, :], eye="both")[0]
    batch = np.repeat(X[None, :], len(cols), axis=0)
    for k, c in enumerate(cols):
        batch[k, c] = net.x_scaler.mean[k]
    return base - net.predict(batch, eye="both")


def explain(model: GazeNet, features: Sequence[float], top: int = 12) -> dict:
    """Everything for one frame: each network (both eyes, left, right), their
    sensitivities and ablations, and which one the model prefers."""
    features = np.asarray(features, float)
    networks = {}
    for eye, member in model.members():
        t = trace(member, features)
        t["sensitivity"] = sensitivity(member, features, top=top)
        t["ablation_px"] = ablation(member, features)
        t["eye"] = eye
        networks[eye] = t
    meta = model.meta or {}
    return {
        "preferred": model.eye,
        "available": [eye for eye in EYES if eye in networks],
        "eye_cv_px": meta.get("eye_cv_px"),
        "networks": networks,
    }


def batch_outputs(model: GazeNet, F: np.ndarray, eye: str = "both") -> dict:
    """For many frames at once: the network's final points and the spread of its members (px).

    Used for signals over time; matches ``predict`` to rounding (one frame at a
    time is exact, see :func:`trace`).
    """
    net = model.member(eye) if eye != "both" else model
    F = np.atleast_2d(np.asarray(F, float))
    Xs = F if net.inputs is None else F[:, net.inputs]
    Z = np.clip(net.x_scaler.transform(Xs), -CLIP, CLIP)
    outs = np.array([m.forward(Z) for m in net.nets])                  # (members, n, 2)
    px = outs * net.y_scaler.std + net.y_scaler.mean
    mean = px.mean(axis=0)
    spread = np.sqrt(((px - mean) ** 2).sum(axis=2).mean(axis=0)) if len(net.nets) > 1 else np.zeros(len(F))
    return {"final": net.correction.apply(mean), "spread": spread}


# ---------------------------------------------------------------------------
# Hand mode
# ---------------------------------------------------------------------------

def hand_trace(points: np.ndarray, pinch_on: float, pinch_off: float, extend_ratio: float = 1.15) -> dict:
    """The hand recogniser's measurements for one frame (21 landmarks in image fractions)."""
    from .. import hand_gestures as H

    pts = np.asarray(points, float)[:, :2]
    width = H.palm_width(pts)
    centre = H.palm_centre(pts)
    fingers = H.extended_fingers(pts, extend_ratio)
    wrist = pts[H.WRIST]
    reach = {}
    for name, (tip, pip) in {"index": (H.INDEX_TIP, H.INDEX_PIP), "middle": (H.MIDDLE_TIP, H.MIDDLE_PIP),
                             "ring": (H.RING_TIP, H.RING_PIP), "pinky": (H.PINKY_TIP, H.PINKY_PIP)}.items():
        reach[name] = float(np.hypot(*(pts[tip] - wrist)) / max(np.hypot(*(pts[pip] - wrist)), 1e-9))
    spread = float(np.hypot(*(pts[H.INDEX_TIP] - pts[H.PINKY_TIP])) / width)
    thumb_out = float(np.hypot(*(pts[H.THUMB_TIP] - pts[H.INDEX_MCP])) / width)
    thumb_free = float(np.hypot(*(pts[H.THUMB_TIP] - pts[H.INDEX_TIP])) / width)
    pinch = float(np.hypot(*(pts[H.THUMB_TIP] - pts[H.INDEX_TIP])) / width)
    return {
        "palm_width": float(width),
        "palm_centre": centre,
        "pinch": pinch,
        "pinch_on": float(pinch_on),
        "pinch_off": float(pinch_off),
        "fingers": fingers,
        "reach": reach,                     # tip-to-wrist over PIP-to-wrist; extended above extend_ratio
        "extend_ratio": extend_ratio,
        "palm_tests": {"fingers": all(fingers.values()), "spread": spread, "spread_min": 1.25,
                       "thumb_out": thumb_out, "thumb_out_min": 0.8, "thumb_free": thumb_free,
                       "thumb_free_min": 0.9},
        "stop_palm": bool(H.stop_palm(pts, extend_ratio)),
    }
