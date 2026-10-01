"""One-eye ("monocular") gaze networks: tracking through winks, and people
whose other eye does not track reliably."""

import time

import numpy as np
import pytest

from paralic.calibration import (CalibrationData, LabeledFrame, ProfileStore, fit_adjustment, fit_eye_models,
                                 fit_full_calibration)
from paralic.features import EYE_INPUTS
from tests.synthetic import SCREEN_H, SCREEN_W, Head, VirtualUser, simulate_calibration


def calibration_data(user, seed=1):
    X, Y, G = simulate_calibration(user, seed=seed)
    data = CalibrationData()
    t0 = time.time() - 3600
    for i, (x, y, g) in enumerate(zip(X, Y, G)):
        kind, pt = g.split(":")
        data.add(LabeledFrame(t=t0 + i / 30, features=x, target=(y[0], y[1]), kind=kind, point=int(pt)))
    return data


def mean_error(model, user, eye=None, closed=None, n=120, seed=7, shift=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n):
        sx, sy = rng.uniform(0.05, 0.95) * SCREEN_W, rng.uniform(0.05, 0.95) * SCREEN_H
        pred = model.predict(user.features(sx, sy, Head(), closed=closed), eye)[0]
        errs.append(np.hypot(pred[0] - sx - shift[0], pred[1] - sy - shift[1]))
    return float(np.mean(errs))


@pytest.fixture(scope="module")
def typical():
    user = VirtualUser(seed=21)
    data = calibration_data(user)
    model, info = fit_full_calibration(data)
    return user, data, model, info


def test_calibration_trains_one_eye_networks(typical):
    user, _, model, info = typical
    assert set(model.eyes) == {"left", "right"}
    assert info["eye"] == "both" and model.eye == "both"
    assert set(info["eye_cv_px"]) == {"both", "left", "right"}
    clean = VirtualUser(seed=21, noise=0.3)
    both = mean_error(model, clean)
    for eye in ("left", "right"):
        assert mean_error(model, clean, eye) < 2.0 * both


def test_one_eye_network_ignores_the_other_eye(typical):
    user, _, model, _ = typical
    v = user.features(700, 400, Head())
    w = v.copy()
    right_only = [i for i in EYE_INPUTS["right"] if i not in EYE_INPUTS["left"]]
    w[right_only] += 0.2
    np.testing.assert_allclose(model.predict(w, "left"), model.predict(v, "left"))
    assert np.hypot(*(model.predict(w, "both") - model.predict(v, "both"))[0]) > 20


def test_open_eye_keeps_tracking_during_a_wink(typical):
    _, _, model, _ = typical
    clean = VirtualUser(seed=21, noise=0.3)
    normal = mean_error(model, clean)
    # With the left eye shut, the two-eye network is thrown off; the right eye's is not.
    assert mean_error(model, clean, "both", closed="left") > 2.0 * normal
    assert mean_error(model, clean, "right", closed="left") < 1.8 * normal
    assert mean_error(model, clean, "left", closed="right") < 1.8 * normal


def fixation_error(model, user, eye=None, n=60, frames=8, seed=7):
    """Error of the average prediction over a short fixation (like the smoothed cursor)."""
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n):
        sx, sy = rng.uniform(0.05, 0.95) * SCREEN_W, rng.uniform(0.05, 0.95) * SCREEN_H
        F = np.array([user.features(sx, sy, Head()) for _ in range(frames)])
        errs.append(np.hypot(*(model.predict(F, eye).mean(axis=0) - (sx, sy))))
    return float(np.mean(errs))


def test_a_drifting_eye_is_left_out():
    # The right eye drifts a few degrees off target, differently on every
    # fixation (a squint that comes and goes): the left eye alone is better.
    user = VirtualUser(seed=22, wander=(0.05, 0.0))
    model, info = fit_full_calibration(calibration_data(user))
    assert info["eye"] == "left" and model.eye == "left"
    assert info["eye_cv_px"]["left"] < info["eye_cv_px"]["both"]
    test_user = VirtualUser(seed=22, wander=(0.05, 0.0))
    assert fixation_error(model, test_user) < 0.85 * fixation_error(model, test_user, "both")


def test_a_merely_noisy_eye_still_helps():
    # Noise alone is averaged out: the two-eye network learns how much to trust each eye.
    user = VirtualUser(seed=23, eye_noise=(4.0, 1.0))
    model, info = fit_full_calibration(calibration_data(user))
    assert info["eye"] == "both"


def test_eye_networks_survive_save_and_load(typical, tmp_path):
    user, data, model, _ = typical
    store = ProfileStore(tmp_path / "profile.json")
    store.save(model, data, {"w": 1920, "h": 1080}, 50.0)
    loaded, _, _ = store.load()
    assert set(loaded.eyes) == {"left", "right"} and loaded.eye == model.eye
    v = user.features(1200, 800, Head())
    for eye in ("both", "left", "right"):
        np.testing.assert_allclose(loaded.predict(v, eye), model.predict(v, eye), atol=1e-9)


def test_adjustment_corrects_every_network(typical):
    user, _, trained, _ = typical
    model = trained.clone()
    data = CalibrationData()
    clean = VirtualUser(seed=21, noise=0.3)
    # Everything now lands 70 px lower than predicted (the person moved).
    for i, (sx, sy) in enumerate([(960, 540), (230, 150), (1690, 150), (1690, 930), (230, 930)]):
        for _ in range(16):
            data.add(LabeledFrame(t=time.time(), features=clean.features(sx, sy, Head()), target=(sx, sy + 70),
                                  kind="adjust", point=i))
    before = {eye: mean_error(model, clean, eye, shift=(0, 70)) for eye in ("both", "left", "right")}
    info = fit_adjustment(model, data)
    assert info["error_after_px"] < info["error_before_px"]
    for eye in ("both", "left", "right"):
        assert mean_error(model, clean, eye, shift=(0, 70)) < 0.8 * before[eye]


def test_profiles_from_before_one_eye_networks_get_them(typical):
    _, data, _, _ = typical
    model, info = fit_full_calibration(data, eyes=False)
    assert model.eyes == {} and "eye" not in info
    fit_eye_models(model, data)
    assert set(model.eyes) == {"left", "right"} and model.meta["eye"] in ("both", "left", "right")


def test_fine_tuning_keeps_one_eye_networks(typical):
    user, _, model, _ = typical
    rng = np.random.default_rng(3)
    pts = rng.uniform((100, 100), (1800, 1000), (60, 2))
    X = np.array([user.features(x, y, Head(y=12.0)) for x, y in pts])
    tuned = model.fine_tune(X, pts)
    assert set(tuned.eyes) == {"left", "right"}
    v = user.features(900, 500, Head(y=12.0))
    assert not np.allclose(tuned.predict(v, "left"), model.predict(v, "left"))
