"""Continuous, mouse-labelled quick tune-up: the dense (eye-features -> mouse-
position) pairs fit the affine correction and reduce error, without touching the
network (so it stays quick and cannot destabilise)."""

import numpy as np

from paralic.calibration import CalibrationData, LabeledFrame, fit_adjustment, fit_full_calibration
from paralic.gazenet import AffineCorrection
from tests.synthetic import Head, SCREEN_H, SCREEN_W, VirtualUser, calibration_points


def _trained():
    user = VirtualUser(seed=1, noise=0.4)
    data = CalibrationData()
    for i, (sx, sy) in enumerate(calibration_points()):
        for _ in range(20):
            data.add(LabeledFrame(t=0.0, features=user.features(sx, sy, Head()), target=(sx, sy),
                                  kind="cal", point=i))
    model, _ = fit_full_calibration(data)
    return user, model


def _pursuit_path(n=40):
    """A smooth path across the screen (what the moving dot traces)."""
    pts = []
    for k in range(n):
        t = 2 * np.pi * k / n
        x = SCREEN_W * (0.5 + 0.42 * np.sin(3 * t))
        y = SCREEN_H * (0.5 + 0.42 * np.sin(2 * t))
        pts.append((float(x), float(y)))
    return pts


def test_mouse_labelled_adjust_reduces_error():
    user, model = _trained()
    # Simulate a seating drift: the saved correction is now off by a fixed bias.
    for _, member in model.members():
        member.correction = AffineCorrection(A=np.eye(2), b=np.array([90.0, -60.0]))

    # Dense continuous collection: every path sample contributes several frames,
    # each labelled with the (true) mouse position. Many pt buckets along the path.
    data = CalibrationData()
    for bucket, (sx, sy) in enumerate(_pursuit_path()):
        for _ in range(6):
            data.add(LabeledFrame(t=0.0, features=user.features(sx, sy, Head()), target=(sx, sy),
                                  kind="adjust", point=bucket))

    info = fit_adjustment(model, data)
    # Dense mouse-labelled data must clearly beat the drifted correction.
    assert info["error_after_px"] < info["error_before_px"]
    assert info["error_after_px"] < 0.5 * info["error_before_px"]
    assert info["n_points"] >= 20          # dense, continuous (not 5 dots)


def test_adjust_needs_enough_points():
    user, model = _trained()
    data = CalibrationData()
    for bucket, (sx, sy) in enumerate(_pursuit_path(2)):   # too few buckets
        data.add(LabeledFrame(t=0.0, features=user.features(sx, sy, Head()), target=(sx, sy),
                              kind="adjust", point=bucket))
    import pytest
    from paralic.calibration import CalibrationError
    with pytest.raises(CalibrationError):
        fit_adjustment(model, data)
