import numpy as np

from paralic.filters import SMOOTHING_PRESETS, GazeStabilizer, OneEuroFilter


def test_one_euro_reduces_jitter_while_fixating():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(SMOOTHING_PRESETS["medium"])
    raw = np.array([800.0, 500.0]) + 50 * rng.standard_normal((120, 2))
    out = np.array([f(i / 30, p) for i, p in enumerate(raw)])
    assert out[30:].std(axis=0).max() < 0.4 * raw[30:].std(axis=0).min()


def test_one_euro_follows_a_saccade_quickly():
    f = OneEuroFilter(SMOOTHING_PRESETS["medium"])
    t = 0.0
    for _ in range(30):
        f(t, np.array([200.0, 200.0]))
        t += 1 / 30
    for _ in range(9):  # 300 ms after a jump across the screen
        out = f(t, np.array([1400.0, 700.0]))
        t += 1 / 30
    assert np.linalg.norm(out - [1400, 700]) < 0.25 * np.linalg.norm([1200, 500])


def test_stabilizer_freezes_at_pre_blink_position():
    s = GazeStabilizer(SMOOTHING_PRESETS["low"])
    t = 0.0
    for _ in range(30):
        s.update(t, np.array([500.0, 400.0]), False)
        t += 1 / 30
    # The eyes start closing; the raw gaze drifts down as the lids fall.
    pos, frozen = s.update(t, np.array([500.0, 700.0]), True)
    assert frozen
    assert abs(pos[1] - 400) < 5
    for _ in range(4):
        t += 1 / 30
        pos, frozen = s.update(t, np.array([500.0, 900.0]), True)
        assert frozen and abs(pos[1] - 400) < 5
    # Eyes reopen: stays frozen for the settle time, then moves on smoothly.
    t += 1 / 30
    pos, frozen = s.update(t, np.array([520.0, 410.0]), False)
    assert frozen
    t += 0.2
    pos, frozen = s.update(t, np.array([520.0, 410.0]), False)
    assert not frozen and np.linalg.norm(pos - [500, 400]) < 30


def test_stabilizer_gives_up_on_very_long_closure():
    s = GazeStabilizer(SMOOTHING_PRESETS["low"], max_freeze_s=0.5)
    t = 0.0
    for _ in range(10):
        s.update(t, np.array([100.0, 100.0]), False)
        t += 1 / 30
    frozen_flags = []
    for _ in range(30):
        _, frozen = s.update(t, np.array([100.0, 900.0]), True)
        frozen_flags.append(frozen)
        t += 1 / 30
    assert frozen_flags[0] and not frozen_flags[-1]
