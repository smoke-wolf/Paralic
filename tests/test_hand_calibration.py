"""Hand-mode setup maths: pointing map, pinch thresholds, the fitted setup."""

import numpy as np
import pytest

from paralic.hand_control import HandControl, HandSetupError
from paralic.hand_gestures import (HandCalibration, HandGestureConfig, HandGestureRecognizer, PointingMap,
                                   estimate_pinch_thresholds, split_pinch_cycle)
from tests.fakes import make_hand

GRID = [(fx, fy) for fy in (0.12, 0.5, 0.88) for fx in (0.1, 0.4, 0.6, 0.9)] + [(0.5, 0.5)]


def test_pointing_map_recovers_affine_and_reaches_corners():
    rng = np.random.default_rng(0)
    # The finger only comfortably covers the middle of the frame...
    tips = rng.uniform(0.35, 0.65, (12, 2))
    # ...but should map to the whole screen (0..1).
    targets = np.clip((tips - 0.35) / 0.30, 0, 1)
    m = PointingMap.fit(tips, targets)
    pred = np.array([m.apply(t) for t in tips])
    assert np.mean(np.linalg.norm(pred - targets, axis=1)) < 0.02
    # A fingertip near the edge of the comfortable range reaches a screen corner.
    corner = m.apply(np.array([0.65, 0.65]))
    assert corner[0] > 0.9 and corner[1] > 0.9


def test_a_damaged_pointing_map_is_not_used():
    assert HandCalibration.from_dict({"pointing": {"coef": [[1, 2]], "quadratic": True}}) is None
    assert HandCalibration.from_dict("nonsense") is None
    ok = HandCalibration.from_dict({"span": 0.2, "pinch_on": 0.3, "pinch_off": 0.5,
                                    "pointing": {"coef": np.zeros((3, 2)).tolist(), "quadratic": False}})
    assert ok is not None and ok.pinch_on == 0.3


def test_pinch_thresholds_have_separated_hysteresis():
    rng = np.random.default_rng(1)
    openv = rng.uniform(0.6, 0.85, 40)
    closed = rng.uniform(0.08, 0.22, 40)
    make, brk = estimate_pinch_thresholds(openv, closed)
    assert closed.max() < make < brk < openv.min()
    assert brk - make >= 0.08          # usable hysteresis gap


def test_pinch_thresholds_fallback_when_not_separable():
    same = np.full(10, 0.4)
    assert estimate_pinch_thresholds(same, same) == (0.45, 0.60)


def test_pinch_cycle_splits_for_a_hand_that_cannot_close_fully():
    """Someone whose fingers only get to 0.5 palm widths apart still gets two groups."""
    rng = np.random.default_rng(2)
    cycle = np.concatenate([rng.normal(0.95, 0.04, 60), rng.normal(0.5, 0.03, 40), rng.uniform(0.55, 0.9, 10)])
    opened, closed = split_pinch_cycle(cycle)
    assert np.median(closed) < 0.6 < np.median(opened)
    make, brk = estimate_pinch_thresholds(opened, closed)
    assert np.percentile(closed, 80) < make < brk < np.percentile(opened, 20)


def test_calibrated_hysteresis_prevents_click_chatter():
    rec = HandGestureRecognizer(HandGestureConfig(), calibration=HandCalibration(pinch_on=0.30, pinch_off=0.50))
    events = []
    for k, d in enumerate([1.0, 0.25, 0.40, 0.25, 0.42, 0.60]):
        events += rec.update(k / 30, make_hand(pinch=d))[1]
    assert [e.type for e in events] == ["pinch_start", "click"]      # one pinch, one click


def test_calibrated_pointing_map_is_applied_to_cursor():
    tips = np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6], [0.5, 0.5], [0.45, 0.55]])
    cal = HandCalibration(pointing=PointingMap.fit(tips, np.clip((tips - 0.4) / 0.2, 0, 1)))
    rec = HandGestureRecognizer(HandGestureConfig(), calibration=cal)
    state, _ = rec.update(0.0, make_hand(tip=(0.40, 0.60)))    # mirrored: (0.60, 0.60)
    assert state.cursor[0] > 0.9 and state.cursor[1] > 0.9


def _setup(control, rng, *, mode="full", reach=0.30, closed=0.2, points=GRID):
    """Run the hand setup's collection: open hand, the dots, then pinching."""
    control.start_calibration(mode)
    t = 0.0

    def frame(pts, label):
        nonlocal t
        t += 1 / 30
        return control.frame(t, int(t * 30), pts, label, {"w": 1600, "h": 900})[0]

    if mode == "full":
        for _ in range(25):
            frame(make_hand(fingers="spread"), {"kind": "hspan"})
    for i, (fx, fy) in enumerate(points):
        # The comfortable finger range (reach) covers the whole screen.
        tip = (1 - (0.35 + reach * fx), 0.30 + reach * fy)                # image x is mirrored
        for k in range(18):
            jitter = rng.normal(0, 0.002, 2)
            frame(make_hand(tip=(tip[0] + jitter[0], tip[1] + jitter[1])), {"kind": "hpoint", "i": i, "fx": fx, "fy": fy})
    if mode == "full":
        for _ in range(20):
            frame(make_hand(pinch=1.0 + rng.normal(0, 0.03)), {"kind": "hpinch", "state": "open"})
        for k in range(150):
            d = closed if (k // 15) % 2 else 1.0
            frame(make_hand(pinch=d + rng.normal(0, 0.03)), {"kind": "hpinch", "state": "cycle"})
    return control.fit({"w": 1600, "h": 900})


def test_hand_setup_fits_pointing_and_pinch():
    control = HandControl()
    summary = _setup(control, np.random.default_rng(3))
    assert summary["points"] == len(GRID) and summary["pointing_error_px"] < 25
    assert summary["pinch_tuned"] and 0.2 < summary["pinch_on"] < summary["pinch_off"] < 1.0
    assert summary["span"] == pytest.approx(0.16, abs=0.01)
    assert control.recognizer.calibration is control.calibration and not control.calibrating
    # A saved document loads back into a fresh control.
    other = HandControl()
    assert other.load(control.document("now")) and other.view()["pinch_tuned"]
    assert other.calibration.to_dict() == control.calibration.to_dict()


def test_quick_re_point_keeps_the_pinch_and_hand_size():
    control = HandControl()
    rng = np.random.default_rng(4)
    first = _setup(control, rng, closed=0.45)
    again = _setup(control, rng, mode="point", reach=0.25)
    assert again["mode"] == "point" and again["pinch_on"] == first["pinch_on"] and again["span"] == first["span"]
    assert again["pinch_tuned"]


def test_hand_setup_fails_clearly_without_enough_dots():
    control = HandControl()
    with pytest.raises(HandSetupError):
        _setup(control, np.random.default_rng(5), points=GRID[:4])
    with pytest.raises(HandSetupError):
        HandControl().start_calibration("point")           # nothing to re-point yet


def test_setup_frames_are_labelled_only_while_setting_up():
    control = HandControl()
    fields = control.frame(0.0, 1, make_hand(), {"kind": "hpoint", "i": 0, "fx": 0.5, "fy": 0.5}, None)[0]
    assert fields["labeled"] is False
    control.start_calibration("full")
    fields = control.frame(0.1, 2, make_hand(), {"kind": "hpoint", "i": 0, "fx": 0.5, "fy": 0.5}, None)[0]
    assert fields["labeled"] is True
    # The span is only measured from an open hand.
    fields = control.frame(0.2, 3, make_hand(fingers="curled"), {"kind": "hspan"}, None)[0]
    assert fields["labeled"] is False
