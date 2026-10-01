"""Hand-mode calibration math: pointing map, pinch thresholds, hysteresis."""

import numpy as np

from paralic.hand_gestures import (HandCalibration, HandGestureConfig, HandGestureRecognizer,
                                    PointingMap, estimate_pinch_thresholds)

I_MCP, P_MCP = 5, 17
WRIST, THUMB_TIP, I_PIP, I_TIP = 0, 4, 6, 8
M_MCP, M_PIP, M_TIP = 9, 10, 12
R_PIP, R_TIP, P_PIP, P_TIP = 14, 16, 18, 20


def test_pointing_map_recovers_affine_and_reaches_corners():
    rng = np.random.default_rng(0)
    # The finger only comfortably covers the middle of the frame...
    tips = rng.uniform(0.35, 0.65, (12, 2))
    # ...but should map to the whole screen (0..1).
    targets = (tips - 0.35) / 0.30
    targets = np.clip(targets, 0, 1)
    m = PointingMap.fit(tips, targets)
    pred = np.array([m.apply(t) for t in tips])
    assert np.mean(np.linalg.norm(pred - targets, axis=1)) < 0.02
    # A fingertip near the edge of the comfortable range reaches a screen corner.
    corner = m.apply(np.array([0.65, 0.65]))
    assert corner[0] > 0.9 and corner[1] > 0.9


def test_pinch_thresholds_have_separated_hysteresis():
    rng = np.random.default_rng(1)
    openv = rng.uniform(0.6, 0.85, 40)
    closed = rng.uniform(0.08, 0.22, 40)
    make, brk = estimate_pinch_thresholds(openv, closed)
    assert closed.max() < make < brk < openv.min()
    assert brk - make >= 0.08          # usable hysteresis gap


def test_pinch_thresholds_fallback_when_not_separable():
    same = np.full(10, 0.4)
    make, brk = estimate_pinch_thresholds(same, same)
    assert (make, brk) == (0.45, 0.60)


def _hand(pinch_dist):
    """A hand whose thumb-index distance / palm width equals ``pinch_dist``."""
    p = np.zeros((21, 2))
    p[WRIST] = (0.5, 0.9)
    p[I_MCP] = (0.42, 0.60); p[M_MCP] = (0.50, 0.60); p[P_MCP] = (0.58, 0.60)  # width 0.16
    p[I_TIP] = (0.50, 0.30); p[I_PIP] = (0.50, 0.52)
    for tip, pip, x in ((M_TIP, M_PIP, 0.50), (R_TIP, R_PIP, 0.55), (P_TIP, P_PIP, 0.60)):
        p[tip] = (x, 0.30); p[pip] = (x, 0.52)
    width = 0.16
    p[THUMB_TIP] = (p[I_TIP][0] + pinch_dist * width, p[I_TIP][1])
    return p


def test_calibrated_hysteresis_prevents_click_chatter():
    cal = HandCalibration(pinch_on=0.30, pinch_off=0.50)
    rec = HandGestureRecognizer(HandGestureConfig(), calibration=cal)
    rec.update(0.00, _hand(0.70))                       # open
    rec.update(0.05, _hand(0.25))                       # below make → pinch starts
    _, mid = rec.update(0.10, _hand(0.40))              # between make/break → stays pinched
    assert all(e.type != "click" for e in mid)
    rec.update(0.15, _hand(0.25))                       # dips again, still one pinch
    _, rel = rec.update(0.20, _hand(0.60))              # above break → release → one click
    assert sum(e.type == "click" for e in rel) == 1


def test_hand_session_fit_builds_and_applies_calibration():
    from paralic.hand_session import HandSession

    s = HandSession(tracker_factory=lambda: None, profile_path=None)  # factory unused (no frames)
    s._span = [0.16] * 10
    s._pinch_open = [0.7] * 6
    s._pinch_closed = [0.15] * 6
    grid = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), (0.5, 0.5), (0.3, 0.7)]
    for i, (fx, fy) in enumerate(grid):
        tip = (0.35 + 0.30 * fx, 0.35 + 0.30 * fy)      # comfortable finger range → screen
        s._points[i] = {"target": (fx, fy), "tips": [tip] * 5}
    res = s._fit()
    assert res["ok"] and res["has_pointing"] and res["pinch_on"] is not None
    assert s.calibration is not None
    assert s.recognizer.calibration is s.calibration     # applied live


def test_calibrated_pointing_map_is_applied_to_cursor():
    tips = np.array([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6], [0.5, 0.5], [0.45, 0.55]])
    targets = (tips - 0.4) / 0.2
    cal = HandCalibration(pointing=PointingMap.fit(tips, np.clip(targets, 0, 1)))
    rec = HandGestureRecognizer(HandGestureConfig(), calibration=cal)
    # Index tip at raw-mirrored (0.6, 0.6) → should map near screen (1, 1).
    hand = _hand(0.70)
    hand[I_TIP] = (0.40, 0.60)                          # mirror(0.40) = 0.60
    state, _ = rec.update(0.0, hand)
    assert state.cursor[0] > 0.9 and state.cursor[1] > 0.9
