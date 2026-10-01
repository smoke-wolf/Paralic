"""Contrastive manual-mode fine-tuning (paralic/contrastive.py)."""

import numpy as np
import pytest

from paralic.gazenet import GazeNet
from paralic.contrastive import contrastive_finetune, ContrastiveError


def _mapping(seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (8, 2))
    def gen(n, bias=np.zeros(2), noise=8.0):
        X = rng.normal(0, 1, (n, 8))
        Y = X @ W
        Y = (Y - Y.min(0)) / (np.ptp(Y, axis=0) + 1e-9) * [1920.0, 1080.0] + bias
        return X, Y + rng.normal(0, noise, (n, 2))
    return gen


def _champion(gen):
    X, Y = gen(260)
    groups = np.repeat(np.arange(13), 20)
    model, _ = GazeNet.train(X, Y, groups, iters=300)
    return model


def test_positive_finetune_improves_and_persists_accuracy():
    gen = _mapping(1)
    champ = _champion(gen)
    Xp, Yp = gen(120, bias=np.array([60.0, -40.0]))   # a small seating shift
    before = np.mean(np.linalg.norm(champ.predict(Xp) - Yp, axis=1))
    model, rep = contrastive_finetune(champ, Xp, Yp, seed=1)
    assert rep["accepted"] and model is not None
    after = np.mean(np.linalg.norm(model.predict(Xp) - Yp, axis=1))
    assert after <= before            # fine-tune adapts to the shift


def test_directional_negative_moves_prediction_along_arrow():
    gen = _mapping(2)
    champ = _champion(gen)
    Xp, Yp = gen(100)
    Xn, Yn = gen(30, noise=0.0)
    pred = champ.predict(Xn)
    dirs = Yn - pred
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
    model, rep = contrastive_finetune(champ, Xp, Yp, neg_X=Xn, neg_dir=dirs, seed=3)
    assert model is not None
    progress = ((model.predict(Xn) - pred) * dirs).sum(1).mean()
    assert progress > 10.0            # moved meaningfully along the arrows


def test_guard_rejects_garbage_targets():
    gen = _mapping(4)
    champ = _champion(gen)
    rng = np.random.default_rng(9)
    Xg = rng.normal(0, 1, (120, 8))
    Yg = rng.uniform(0, [1920.0, 1080.0], (120, 2))   # random, unlearnable targets
    model, rep = contrastive_finetune(champ, Xg, Yg, seed=5)
    assert model is None and rep["accepted"] is False


def test_too_few_positives_raises():
    gen = _mapping(6)
    champ = _champion(gen)
    Xp, Yp = gen(5)
    with pytest.raises(ContrastiveError):
        contrastive_finetune(champ, Xp, Yp)


def test_tunes_the_network_that_drives_the_cursor():
    # For someone with a squint that comes and goes, a one-eye network leads;
    # manual-mode positives after a seating shift must improve *that* network.
    from paralic.calibration import fit_full_calibration
    from tests.synthetic import Head, VirtualUser
    from tests.test_eyes import calibration_data

    user = VirtualUser(seed=22, wander=(0.05, 0.0))
    champ, _ = fit_full_calibration(calibration_data(user))
    assert champ.eye == "left"
    rng = np.random.default_rng(1)
    pts = rng.uniform((150, 120), (1770, 960), (120, 2))
    X = np.array([user.features(x, y, Head(x=3.0, y=13.0)) for x, y in pts])
    model, rep = contrastive_finetune(champ, X, pts, seed=1)
    assert rep["accepted"] and model is not None
    assert rep["tuned_error_px"] < 0.7 * rep["champion_error_px"]
    assert model.eye == "left" and set(model.eyes) == {"left", "right"}
