"""Hand-mode gesture state machine (paralic/hand_gestures.py)."""

import numpy as np

from paralic.hand_gestures import HandGestureConfig, HandGestureRecognizer, stop_palm
from tests.fakes import make_hand

DT = 1 / 30


def run(rec, hands, t0=0.0):
    """Feed a list of hands (None = lost) at 30 fps: (states, events)."""
    states, events = [], []
    for k, pts in enumerate(hands):
        s, ev = rec.update(t0 + k * DT, None if pts is None else pts)
        states.append(s)
        events.extend(ev)
    return states, events


def types(events):
    return [e.type for e in events]


def test_cursor_is_index_tip_mirrored():
    rec = HandGestureRecognizer()
    state, _ = rec.update(0.0, make_hand(tip=(0.3, 0.4)))
    assert state.present
    assert abs(state.cursor[0] - 0.7) < 1e-6     # x mirrored for the front camera
    assert abs(state.cursor[1] - 0.4) < 1e-6


def test_quick_pinch_clicks_where_the_finger_pointed_before_closing():
    rec = HandGestureRecognizer()
    # The fingertip drifts towards the thumb as the fingers close.
    hands = [make_hand(tip=(0.40, 0.40), pinch=1.2)] * 5
    hands += [make_hand(tip=(0.40 + 0.004 * k, 0.40), pinch=0.8 - 0.12 * k) for k in range(1, 6)]
    hands += [make_hand(tip=(0.42, 0.40), pinch=0.2)] * 3 + [make_hand(tip=(0.42, 0.40), pinch=1.0)]
    states, events = run(rec, hands)
    assert types(events) == ["pinch_start", "click"]
    start, click = events
    assert click.aim_t == start.aim_t and click.aim_t <= 4 * DT + 1e-9     # before the fingers moved
    pinching = [s for s in states if s.pinching]
    assert pinching and all(abs(s.aim[0] - 0.60) < 1e-6 for s in pinching)   # mirror of 0.40


def test_a_long_pinch_is_not_a_click():
    rec = HandGestureRecognizer()
    hands = [make_hand(pinch=1.2)] * 3 + [make_hand(pinch=0.2)] * 30 + [make_hand(pinch=1.2)]
    _, events = run(rec, hands)
    assert types(events) == ["pinch_start", "pinch_end"]


def test_pinch_drag_scrolls_and_does_not_click():
    rec = HandGestureRecognizer()
    hands = [make_hand(centre=(0.5, 0.5), pinch=1.2)] * 3
    hands += [make_hand(centre=(0.5, 0.5 + 0.012 * k), pinch=0.2) for k in range(15)]   # hand moves down
    hands += [make_hand(centre=(0.5, 0.5 + 0.012 * 14), pinch=1.2)]
    _, events = run(rec, hands)
    scrolls = [e for e in events if e.type == "scroll"]
    assert scrolls and all(e.dy > 0 for e in scrolls)              # downward hand -> scroll down
    # The whole travel is scrolled: 0.168 / 0.16 palm widths at 0.5 screens each.
    assert abs(sum(e.dy for e in scrolls) - 0.168 / 0.16 * 0.5) < 0.02
    assert "click" not in types(events) and types(events)[-1] == "pinch_end"


def test_moving_the_hand_while_pinching_is_not_a_click():
    rec = HandGestureRecognizer()
    hands = [make_hand(centre=(0.5, 0.5), pinch=1.2)] * 3
    hands += [make_hand(centre=(0.5 + 0.02 * k, 0.5), pinch=0.2) for k in range(6)]   # sideways, 0.6 palm widths
    hands += [make_hand(centre=(0.6, 0.5), pinch=1.2)]
    _, events = run(rec, hands)
    assert "click" not in types(events)


def test_stop_palm_toggles_once_per_gesture():
    cfg = HandGestureConfig()
    rec = HandGestureRecognizer(cfg)
    open_hand = make_hand(fingers="spread")
    point = make_hand(fingers="curled")
    held = int(cfg.palm_hold_ms / 1000 / DT) + 3
    # Held for three times as long: still one toggle.
    _, events = run(rec, [open_hand] * (3 * held))
    assert types(events) == ["palm"]
    # A brief lowering does not re-arm it ...
    _, events = run(rec, [point] * 3 + [open_hand] * held, t0=10.0)
    assert types(events) == []
    # ... leaving the pose for a moment does.
    _, events = run(rec, [point] * 15 + [open_hand] * held, t0=20.0)
    assert types(events) == ["palm"]


def test_pointing_with_a_flat_hand_never_pauses():
    rec = HandGestureRecognizer()
    flat = make_hand(fingers="together", thumb_out=False)
    assert not stop_palm(flat[:, :2])
    _, events = run(rec, [flat] * 90)
    assert types(events) == []
    # Nor does a moving open hand.
    _, events = run(rec, [make_hand(centre=(0.3 + 0.01 * k, 0.6), fingers="spread") for k in range(40)], t0=5.0)
    assert types(events) == []


def test_a_short_dropout_keeps_the_pinch():
    rec = HandGestureRecognizer()
    hands = [make_hand(pinch=1.2)] * 3 + [make_hand(pinch=0.2)] * 3 + [None] * 4 + [make_hand(pinch=0.2)] * 2
    hands += [make_hand(pinch=1.2)]
    _, events = run(rec, hands)
    assert types(events) == ["pinch_start", "click"]


def test_hand_lost_mid_pinch_ends_without_a_click():
    rec = HandGestureRecognizer()
    hands = [make_hand(pinch=1.2)] * 3 + [make_hand(pinch=0.2)] * 3 + [None] * 12 + [make_hand(pinch=1.2)]
    states, events = run(rec, hands)
    assert types(events) == ["pinch_start", "pinch_end"]
    assert not states[-2].present and not states[-2].pinching


def test_setup_reports_no_clicks_or_palms():
    rec = HandGestureRecognizer()
    rec.setup = True
    hands = [make_hand(pinch=1.2)] * 3 + [make_hand(pinch=0.2)] * 3 + [make_hand(pinch=1.2)]
    hands += [make_hand(fingers="spread")] * 40
    _, events = run(rec, hands)
    assert "click" not in types(events) and "palm" not in types(events)


def test_thresholds_keep_a_hysteresis_gap():
    rec = HandGestureRecognizer()
    on, off = rec.thresholds()
    assert off - on >= 0.05
    assert np.isfinite([on, off]).all()
