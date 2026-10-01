"""Calibration label quality: which frames of a dot really show the eyes on it."""

import time

import numpy as np

from paralic.calibration import (FIXATION_IDX, CalibrationData, LabeledFrame, SettleTracker, calibrated_pose,
                                 fixation_frames, noise_sigma, point_summaries, prepare_training_set, segments,
                                 settled_run)
from paralic.features import EYE_FEATURE_IDX
from tests.synthetic import Head, VirtualUser, jitter_head


def _frames(user, rng, looks):
    """Feature rows for a sequence of (screen x, screen y) gaze points."""
    return np.array([user.features(x, y, jitter_head(rng, Head(), 0.3)) for x, y in looks])


def _rest(user, rng, look, n=12):
    """Where the eyes rested on a dot: (mean of the fixation features, frames)."""
    F = _frames(user, rng, [look] * n)
    return F[:, list(FIXATION_IDX)].mean(axis=0), n


def test_fixation_frames_drop_the_way_there_and_an_early_glance():
    user, rng = VirtualUser(seed=1, noise=0.6), np.random.default_rng(0)
    old, new, text = (300, 300), (1500, 800), (960, 950)
    looks = [old] * 6 + [text] * 5 + [new] * 16
    keep = fixation_frames(_frames(user, rng, looks), previous=_rest(user, rng, old))
    assert keep[-13:].sum() >= 12       # (the first frames after a jump go too)
    assert not keep[:11].any()


def test_slow_eyes_still_on_the_previous_dot_are_not_used():
    """The eyes rest on the previous dot for longer than they rest on this one:
    the longer stretch is still dropped, because it is where they were before."""
    user, rng = VirtualUser(seed=6, noise=1.0), np.random.default_rng(5)
    old, new = (1700, 100), (1700, 420)       # a vertical hop between neighbouring dots
    F = _frames(user, rng, [old] * 30 + [new] * 16)
    keep = fixation_frames(F, previous=_rest(user, rng, old))
    assert keep[-13:].sum() >= 11 and not keep[:30].any()
    # Nothing can be told apart from the previous dot (a weak signal): the
    # dot is kept rather than lost.
    assert fixation_frames(_frames(user, rng, [old] * 20), previous=_rest(user, rng, old)).sum() >= 14


def test_fixation_frames_prefer_a_clearly_longer_stretch_over_a_late_glance():
    # An old-style fixed recording: on the dot, then a glance away at the very end.
    user, rng = VirtualUser(seed=2, noise=0.6), np.random.default_rng(1)
    dot, away = (600, 400), (1500, 950)
    F = _frames(user, rng, [dot] * 20 + [away] * 5)
    sigma = noise_sigma(_frames(user, rng, [(900, 500)] * 200))     # pooled over the dots, as in training
    keep = fixation_frames(F, sigma=sigma)
    assert keep[:20].sum() >= 15 and not keep[-5:].any()
    # Even with the noise estimated from this recording alone, only frames on the dot are used.
    keep = fixation_frames(F)
    assert keep[:20].sum() >= 8 and not keep[-5:].any()


def test_segments_find_where_the_eyes_moved():
    """Neighbouring dots (a hop to the side, then down) for jittery tracking."""
    user, rng = VirtualUser(seed=7, noise=1.0), np.random.default_rng(6)
    sigma = noise_sigma(_frames(user, rng, [(900, 500)] * 200))     # pooled over the dots
    F = _frames(user, rng, [(400, 300)] * 15 + [(820, 300)] * 15 + [(820, 610)] * 15)
    cuts = [lo for lo, _ in segments(F[:, list(FIXATION_IDX)] / sigma)]
    assert len(cuts) == 3 and abs(cuts[1] - 15) <= 1 and abs(cuts[2] - 30) <= 1
    for _ in range(20):
        still = _frames(user, rng, [(900, 500)] * 40)
        assert len(segments(still[:, list(FIXATION_IDX)] / sigma)) == 1


def test_settled_run_waits_for_the_eyes_to_leave_the_previous_dot():
    user, rng = VirtualUser(seed=3, noise=1.0), np.random.default_rng(2)
    old, new = (400, 300), (820, 300)          # neighbouring dots, jittery tracking
    rest = _rest(user, rng, old)
    sigma = noise_sigma(_frames(user, rng, [(900, 500)] * 40))
    still_there = _frames(user, rng, [old] * 25)
    assert settled_run(still_there, rest, sigma=sigma) == 0
    arrived = np.vstack([still_there, _frames(user, rng, [new] * 12)])
    assert 10 <= settled_run(arrived, rest, sigma=sigma) <= 13
    # A glance away restarts the count.
    glanced = np.vstack([arrived, _frames(user, rng, [(960, 1000)] * 5), _frames(user, rng, [new] * 6)])
    assert settled_run(glanced, rest, sigma=sigma) <= 7
    # Without a previous dot the steady stretch simply counts.
    assert settled_run(_frames(user, rng, [new] * 12), sigma=sigma) == 12
    assert settled_run(_frames(user, rng, [new] * 3), sigma=sigma) == 0


def test_settle_tracker_follows_a_run_of_dots():
    user, rng = VirtualUser(seed=8, noise=1.0), np.random.default_rng(7)
    tracker = SettleTracker()
    dots = [(960, 540), (100, 90), (540, 90)]
    eyes = (960, 540)
    for i, dot in enumerate(dots):
        counts = []
        for k in range(40):
            if k == 8:
                eyes = dot          # slow eyes: they arrive after 8 frames
            counts.append(tracker.update(("cal", i, dot), _frames(user, rng, [eyes])[0]))
            if counts[-1] >= 16:
                break
        if i > 0:
            assert max(counts[:8]) == 0          # nothing counts before they arrive
        assert counts[-1] >= 16 and len(counts) >= (16 if i == 0 else 24)
    tracker.reset()
    assert tracker.previous is None and len(tracker.steps) == 0


def test_moving_targets_keep_their_own_labels():
    """A pursuit sweep labels each frame with where the pointer was: the
    training targets follow it instead of repeating the first one."""
    user, rng = VirtualUser(seed=4, noise=0.3), np.random.default_rng(3)
    d = CalibrationData()
    t0 = time.time()
    xs = np.linspace(300, 1600, 24)
    for k, x in enumerate(xs):
        d.add(LabeledFrame(t=t0 + k / 30, features=user.features(x, 500, Head()), target=(float(x), 500.0),
                           kind="adjust", point=0))
    X, Y, G, W, H = prepare_training_set(d.frames)
    assert np.ptp(Y[:, 0]) > 1000
    # Summaries of a moving group use the mean target.
    class Echo:          # a "model" that predicts the true gaze
        def predict_uncorrected(self, F, eye=None):
            return np.array([[x, 500.0] for x in xs])[:len(F)]
    (p,) = point_summaries(Echo(), d.frames, corrected=False)
    assert abs(p["target"][0] - xs.mean()) < 1.0


def test_calibrated_pose_is_the_median_head_position_during_the_dots():
    user = VirtualUser(seed=5, noise=0.0)
    d = CalibrationData()
    assert calibrated_pose(d) is None
    for k in range(30):
        vec = user.features(800, 500, Head())
        vec[17], vec[18], vec[19] = 2.0, -6.0, -55.0       # tx, ty, tz (cm)
        vec[14], vec[15] = np.radians(4.0), np.radians(-8.0)
        d.add(LabeledFrame(t=float(k), features=vec, target=(800, 500), kind="cal", point=k // 10))
    pose = calibrated_pose(d)
    assert pose == {"x": 2.0, "y": -6.0, "dist": 55.0, "yaw": 4.0, "pitch": -8.0}
    assert set(EYE_FEATURE_IDX).isdisjoint({14, 15, 17, 18, 19})
