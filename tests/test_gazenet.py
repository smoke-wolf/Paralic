import numpy as np
import pytest

from paralic.calibration import CalibrationData, LabeledFrame, fit_full_calibration, prepare_training_set
from paralic.features import FEATURE_MIN_STD
from paralic.gazenet import AffineCorrection, GazeNet, MLPRegressor, Scaler, grouped_folds, ridge_fit
from tests.synthetic import SCREEN_H, SCREEN_W, Head, VirtualUser, jitter_head, simulate_calibration


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 5))
    Y = rng.standard_normal((30, 2))
    w = rng.uniform(0.5, 1.5, 30)
    net = MLPRegressor(5, 2, (6, 4), seed=1)
    net.theta[:] += 0.3 * rng.standard_normal(net.theta.shape)
    _, grad = net.loss_and_grads(X, Y, w, 1e-2, 1e-3, 0.5)
    num = np.zeros_like(net.theta)
    for i in range(net.theta.size):
        old = net.theta[i]
        net.theta[i] = old + 1e-6
        lp, _ = net.loss_and_grads(X, Y, w, 1e-2, 1e-3, 0.5)
        net.theta[i] = old - 1e-6
        lm, _ = net.loss_and_grads(X, Y, w, 1e-2, 1e-3, 0.5)
        net.theta[i] = old
        num[i] = (lp - lm) / 2e-6
    assert np.max(np.abs(num - grad)) / np.max(np.abs(num)) < 1e-6


def test_mlp_learns_nonlinear_function():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, (300, 2))
    Y = np.stack([np.sin(X[:, 0]) + 0.3 * X[:, 1], X[:, 0] * X[:, 1] * 0.5], axis=1)
    net = MLPRegressor(2, 2, (32, 16), seed=0).fit(X, Y, l2=1e-4, iters=1500, lr=0.02)
    err = np.abs(net.forward(X) - Y).mean()
    W, b = ridge_fit(X, Y, 1e-3)
    lin_err = np.abs(X @ W + b - Y).mean()
    assert err < 0.5 * lin_err


def test_grouped_folds_keep_groups_together():
    groups = np.repeat(np.arange(10), 5)
    folds = grouped_folds(groups, 3)
    assert sorted(np.concatenate(folds).tolist()) == list(range(50))
    for f in folds:
        for g in np.unique(groups[f]):
            assert set(np.flatnonzero(groups == g)) <= set(f)


def _to_data(X, Y, G):
    data = CalibrationData()
    for i, (x, y, g) in enumerate(zip(X, Y, G)):
        kind, pt = g.split(":")
        data.add(LabeledFrame(t=i / 30, features=x, target=(y[0], y[1]), kind=kind, point=int(pt)))
    return data


def _eval(model, user, amount, n=200, seed=5):
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n):
        sx, sy = rng.uniform(0.05, 0.95) * SCREEN_W, rng.uniform(0.05, 0.95) * SCREEN_H
        f = user.features(sx, sy, jitter_head(rng, Head(), amount))
        errs.append(np.hypot(*(model.predict(f)[0] - (sx, sy))))
    return float(np.mean(errs))


def test_calibration_pipeline_is_accurate_on_simulated_user():
    user = VirtualUser(seed=3)
    X, Y, G = simulate_calibration(user, seed=13)
    model, info = fit_full_calibration(_to_data(X, Y, G))
    clean = VirtualUser(seed=3, noise=0.0)
    assert info["n_points"] == 13
    # Small head jitter: well under 1% of the screen diagonal off on average.
    assert _eval(model, clean, 0.3) < 30
    # Moderate head movement is compensated thanks to the head-motion samples.
    assert _eval(model, clean, 1.0) < 60


def test_gazenet_not_worse_than_linear_baseline():
    user = VirtualUser(seed=7)
    X, Y, G = simulate_calibration(user, seed=17)
    data = _to_data(X, Y, G)
    model, info = fit_full_calibration(data)
    Xp, Yp, _, Wp, _ = prepare_training_set(data.frames)
    xs, ys = Scaler.fit(Xp, FEATURE_MIN_STD), Scaler.fit(Yp)
    W, b = ridge_fit(xs.transform(Xp), ys.transform(Yp), 1e-2, Wp)

    class Linear:
        def predict(self, f):
            return ys.inverse(np.clip(xs.transform(np.atleast_2d(f)), -6, 6) @ W + b)

    clean = VirtualUser(seed=7, noise=0.0)
    assert _eval(model, clean, 1.0) <= _eval(Linear(), clean, 1.0) * 1.05


def test_serialisation_round_trip():
    user = VirtualUser(seed=2)
    X, Y, G = simulate_calibration(user, head_motion=False, seed=4)
    model, _ = fit_full_calibration(_to_data(X, Y, G))
    model.correction = AffineCorrection(A=np.array([[1.05, 0.0], [0.02, 0.97]]), b=np.array([12.0, -8.0]))
    clone = GazeNet.from_dict(model.to_dict())
    np.testing.assert_allclose(clone.predict(X[:20]), model.predict(X[:20]), rtol=1e-9, atol=1e-6)


def test_affine_correction_recovers_offset_and_scale():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 1000, (5, 2))
    target = pred * 1.1 + np.array([40.0, -25.0])
    corr = AffineCorrection.fit(pred, target, strength=0.01)
    assert np.abs(corr.apply(pred) - target).max() < 5


def test_affine_correction_is_bounded():
    pred = np.array([[0, 0], [100, 0], [0, 100], [100, 100], [50, 50]], float)
    target = pred * 5.0
    corr = AffineCorrection.fit(pred, target, strength=0.0)
    assert np.linalg.svd(corr.A, compute_uv=False).max() <= 1.6 + 1e-9


def test_not_enough_points_raises():
    from paralic.calibration import CalibrationError

    user = VirtualUser(seed=1)
    X, Y, G = simulate_calibration(user, head_motion=False)
    keep = np.isin(G, [f"cal:{i}" for i in range(3)])
    with pytest.raises(CalibrationError):
        fit_full_calibration(_to_data(X[keep], Y[keep], G[keep]))


def test_clone_does_not_share_meta():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6))
    Y = X[:, :2] * 100
    model, _ = GazeNet.train(X, Y, np.repeat(np.arange(12), 10), iters=50)
    model.meta["nested"] = {"a": 1}
    twin = model.clone()
    twin.meta["eye"] = "left"
    twin.meta["nested"]["a"] = 2
    assert "eye" not in model.meta and model.meta["nested"]["a"] == 1
