import numpy as np

from paralic.blink import BlinkConfig, BlinkDetector

FPS = 30.0


def run(detector, closures, t0=0.0):
    """Feed closure values at 30 fps; return list of (time, event type, count)."""
    events = []
    for i, c in enumerate(closures):
        t = t0 + i / FPS
        for ev in detector.update(t, c):
            events.append((round(t, 3), ev.type, ev.count))
    return events


def open_frames(n, level=0.15, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    return list(np.clip(level + noise * rng.standard_normal(n), 0, 1))


def blink(n_closed=5, peak=0.9):
    ramp = [0.45, peak] + [peak] * max(0, n_closed - 3) + [0.5]
    return ramp


def types(events):
    return [e[1] for e in events]


def test_single_blink_is_not_a_click():
    d = BlinkDetector()
    ev = run(d, open_frames(60) + blink() + open_frames(40, seed=1))
    assert types(ev) == ["blink", "blink_expired"]


def test_double_blink_detected():
    d = BlinkDetector()
    seq = open_frames(60) + blink() + open_frames(6, seed=2) + blink() + open_frames(30, seed=3)
    ev = run(d, seq)
    assert "double_blink" in types(ev)
    assert types(ev).count("double_blink") == 1


def test_double_blink_reports_first_blink_start():
    d = BlinkDetector()
    seq = open_frames(60) + blink() + open_frames(6, seed=2) + blink() + open_frames(10, seed=3)
    first_start = None
    for i, c in enumerate(seq):
        for e in d.update(i / FPS, c):
            if e.type == "blink" and e.count == 1:
                first_start = e.first_start
            if e.type == "double_blink":
                assert abs(e.first_start - first_start) < 1e-9
                assert 60 / FPS <= e.first_start <= 62 / FPS


def test_slow_blinks_are_two_singles():
    d = BlinkDetector(BlinkConfig(double_gap_ms=400))
    seq = open_frames(60) + blink() + open_frames(30, seed=2) + blink() + open_frames(30, seed=3)
    ev = run(d, seq)
    assert "double_blink" not in types(ev)
    assert types(ev).count("blink") == 2


def test_long_eye_closure_is_not_a_blink():
    d = BlinkDetector()
    seq = open_frames(60) + [0.95] * 45 + open_frames(30, seed=4) + blink() + open_frames(30, seed=5)
    ev = run(d, seq)
    # The long closure is ignored, the later blink starts a fresh sequence.
    assert "double_blink" not in types(ev)


def test_triple_blink_gives_one_click():
    d = BlinkDetector()
    seq = open_frames(60)
    for k in range(3):
        seq += blink() + open_frames(6, seed=10 + k)
    seq += open_frames(30, seed=20)
    ev = run(d, seq)
    assert types(ev).count("double_blink") == 1


def test_baseline_adapts_to_looking_down():
    d = BlinkDetector()
    run(d, open_frames(60, level=0.12))
    low_baseline = d.baseline
    # Looking down: the lids drop and the closure score stays higher.
    run(d, open_frames(120, level=0.42, seed=6), t0=2.0)
    assert d.baseline > low_baseline + 0.2
    assert not d.state().closed
    # A real blink on top of that is still detected.
    ev = run(d, blink(peak=0.97) + open_frames(6, level=0.42, seed=7) + blink(peak=0.97)
             + open_frames(20, level=0.42, seed=8), t0=6.0)
    assert "double_blink" in types(ev)


def test_state_reports_closing():
    d = BlinkDetector()
    run(d, open_frames(60))
    d.update(3.0, 0.9)
    s = d.state()
    assert s.closing and s.closed


def test_missing_face_resets_pending_sequence():
    d = BlinkDetector()
    run(d, open_frames(60) + blink() + open_frames(2, seed=9))
    assert d.state().pending == 1
    events = d.update_missing(10.0)
    assert [e.type for e in events] == ["blink_expired"]
    assert d.state().pending == 0
