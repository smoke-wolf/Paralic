import math

import numpy as np
import pytest

from paralic.blink import BlinkConfig, BlinkDetector
from paralic.gestures import (EyeWinkConfig, WinkConfig, WinkDetector, analyze_winks, blink_signal,
                              wink_config)
from paralic.personalize import PersonalizationError, analyze_blinks

FPS = 30.0


def run(detector, frames, t0=0.0):
    """Feed (left, right) closures at 30 fps; return [(time, type, eye, duration_ms, cancelled)]."""
    out = []
    for i, (cl, cr) in enumerate(frames):
        for ev in detector.update(t0 + i / FPS, cl, cr):
            out.append((round(t0 + i / FPS, 3), ev.type, ev.eye, round(ev.duration_ms), ev.cancelled))
    return out


def open_eyes(n, left=0.2, right=0.2, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return [(left + noise * rng.standard_normal(), right + noise * rng.standard_normal()) for _ in range(n)]


def wink(eye, frames, level=0.85, other=0.22, seed=1):
    rng = np.random.default_rng(seed)
    ramp = [0.5] + [level] * frames + [0.5]
    seq = []
    for v in ramp:
        o = other + 0.01 * rng.standard_normal()
        seq.append((v, o) if eye == "left" else (o, v))
    return seq


def types(events):
    return [e[1] for e in events]


# -- the detector ----------------------------------------------------------------------

def test_held_wink_is_a_press_and_release():
    d = WinkDetector()
    ev = run(d, open_eyes(60) + wink("left", 30) + open_eyes(30, seed=2))
    assert types(ev) == ["wink_start", "wink_end"]
    start, end = ev
    assert start[2] == end[2] == "left"
    # The press starts once the wink has lasted ~hold_ms, and lasts as long as the wink.
    assert 300 <= start[3] <= 450
    assert 950 <= end[3] <= 1200 and not end[4]


def test_right_eye_wink():
    d = WinkDetector()
    ev = run(d, open_eyes(60) + wink("right", 20) + open_eyes(30, seed=2))
    assert [(e[1], e[2]) for e in ev] == [("wink_start", "right"), ("wink_end", "right")]


def test_short_wink_is_reported_separately():
    d = WinkDetector()
    ev = run(d, open_eyes(60) + wink("left", 5) + open_eyes(30, seed=2))
    assert types(ev) == ["wink"] and ev[0][2] == "left" and 120 <= ev[0][3] <= 300


def test_blinks_are_not_winks():
    d = WinkDetector()
    blink = [(0.5, 0.5), (0.9, 0.9), (0.9, 0.9), (0.9, 0.9), (0.5, 0.5)]
    # A lopsided blink: the left eye leads by a frame and opens a frame later.
    lopsided = [(0.6, 0.2), (0.9, 0.6), (0.9, 0.9), (0.9, 0.9), (0.7, 0.4), (0.3, 0.2)]
    ev = run(d, open_eyes(60) + blink + open_eyes(20, seed=2) + lopsided + open_eyes(30, seed=3)
             + [(0.9, 0.9)] * 40 + open_eyes(20, seed=4))  # ...and closing both eyes for a while
    assert ev == []


def test_other_eye_may_blink_during_a_press():
    d = WinkDetector()
    held = wink("left", 40)
    for i in range(15, 20):  # the open right eye blinks in the middle of the drag
        held[i] = (held[i][0], 0.9)
    ev = run(d, open_eyes(60) + held + open_eyes(30, seed=2))
    assert types(ev) == ["wink_start", "wink_end"]
    assert ev[1][3] > 1200


def test_eye_must_be_seen_open_first():
    d = WinkDetector()
    # Starts with the left eye looking shut (a patch, a drooping lid): never a press.
    ev = run(d, [(0.85, 0.2)] * 90)
    assert ev == []
    # Once it has been seen open, winks work.
    ev = run(d, open_eyes(30) + wink("left", 20) + open_eyes(20, seed=2), t0=3.0)
    assert types(ev) == ["wink_start", "wink_end"]


def test_disabled_eye_does_not_wink():
    d = WinkDetector(WinkConfig())
    d.config.left.enabled = False
    ev = run(d, open_eyes(60) + wink("left", 30) + open_eyes(20, seed=2) + wink("right", 30)
             + open_eyes(20, seed=3))
    assert [(e[1], e[2]) for e in ev] == [("wink_start", "right"), ("wink_end", "right")]


def test_lost_face_cancels_a_press():
    d = WinkDetector()
    run(d, open_eyes(60) + wink("left", 15)[:-1])
    assert d.pressed == "left"
    assert d.update_missing(3.0) == []           # a short dropout is tolerated...
    ev = d.update_missing(4.5)                   # ...a long one ends the press
    assert [(e.type, e.cancelled) for e in ev] == [("wink_end", True)]


def test_droopy_eyelid_adapts_and_still_winks():
    # The left lid always hangs lower (closure 0.5 at rest).
    d = WinkDetector()
    ev = run(d, open_eyes(90, left=0.5) + wink("left", 25, level=0.92) + open_eyes(30, left=0.5, seed=2))
    assert types(ev) == ["wink_start", "wink_end"]
    assert d.baseline("left") == pytest.approx(0.5, abs=0.03)


def test_looking_down_with_a_drooping_lid_is_not_a_wink():
    # Both lids drop by the same amount while reading the bottom of the screen;
    # the drooping one ends up past its threshold, the other not. Learning only
    # one of the baselines would fake a wink.
    d = WinkDetector(WinkConfig(left=EyeWinkConfig(rise=0.17, asym=0.15)))
    ev = run(d, open_eyes(90, left=0.5) + open_eyes(150, left=0.8, right=0.5, seed=2)
             + open_eyes(60, left=0.5, seed=3))
    assert ev == []
    ev = run(d, wink("left", 25, level=0.92) + open_eyes(30, left=0.5, seed=4), t0=10.0)
    assert types(ev) == ["wink_start", "wink_end"]


def test_squinting_the_other_eye_is_fine():
    d = WinkDetector()
    ev = run(d, open_eyes(60) + wink("right", 25, other=0.42) + open_eyes(30, seed=2))
    assert types(ev) == ["wink_start", "wink_end"]


# -- personalisation ------------------------------------------------------------------------

def wink_recording(left_level=0.85, left_other=0.25, right_level=0.85, right_other=0.25, seed=0):
    """A prompted recording: rest, keep left closed 2 s, rest, keep right closed 2 s, rest."""
    rng = np.random.default_rng(seed)
    rows = []
    t = 0.0

    def add(n, phase, cl, cr):
        nonlocal t
        for i in range(n):
            ramp = 1.0 if phase == "rest" else min(1.0, i / 6)  # it takes ~0.2 s to close the eye
            l = 0.2 + (cl - 0.2) * ramp
            r = 0.2 + (cr - 0.2) * ramp
            rows.append((t, l + 0.01 * rng.standard_normal(), r + 0.01 * rng.standard_normal(), phase))
            t += 1 / FPS

    add(60, "rest", 0.2, 0.2)
    add(60, "left", left_level, left_other)
    add(45, "rest", 0.2, 0.2)
    add(60, "right", right_other, right_level)
    add(45, "rest", 0.2, 0.2)
    return rows


def test_wink_analysis_measures_both_eyes():
    res = analyze_winks(wink_recording(right_other=0.4))
    assert res["left"]["ok"] and res["right"]["ok"]
    assert res["left"]["rise"] == pytest.approx(0.65, abs=0.05)
    assert res["right"]["other_rise"] == pytest.approx(0.2, abs=0.05)  # squints the left eye a little
    assert res["left"]["threshold_rise"] < res["left"]["rise"]


def test_wink_analysis_spots_an_eye_that_cannot_wink_alone():
    # When asked to close the left eye, both eyes close.
    res = analyze_winks(wink_recording(left_other=0.8))
    assert not res["left"]["ok"] and res["left"]["reason"] == "the other eye closed too"
    assert res["right"]["ok"]
    cfg = wink_config({"wink": res})
    assert not cfg.left.enabled and cfg.right.enabled


def test_wink_analysis_needs_open_eyes():
    rows = [(i / FPS, 0.85, 0.2, "left") for i in range(90)]
    with pytest.raises(PersonalizationError):
        analyze_winks(rows)


def test_personal_thresholds_catch_gentle_winks():
    # This person's winks only raise the closure to 0.48, with the other eye at 0.3.
    res = analyze_winks(wink_recording(left_level=0.48, left_other=0.3, right_level=0.48, right_other=0.3))
    assert res["left"]["ok"] and res["right"]["ok"]
    usage = open_eyes(60) + wink("left", 25, level=0.48, other=0.3) + open_eyes(30, seed=2)
    assert run(WinkDetector(), usage) == []                       # too gentle for the defaults
    personal = WinkDetector(wink_config({"wink": res}))
    assert types(run(personal, usage)) == ["wink_start", "wink_end"]


def test_wink_config_follows_gesture_choices():
    cfg = wink_config({}, {"left_hold": "off", "left_quick": "off", "hold_ms": 600})
    assert not cfg.left.enabled and cfg.right.enabled and cfg.hold_ms == 600
    # Lopsided blinkers need clearly lopsided winks.
    cfg = wink_config({"blink": {"blink_asym": 0.3, "blink_ms": 300}})
    assert cfg.asym_min == pytest.approx(0.45) and cfg.hold_ms == pytest.approx(450)


def test_blink_signal_modes():
    assert blink_signal(0.9, 0.2) == 0.2
    assert blink_signal(0.9, 0.2, "mean") == pytest.approx(0.55)
    assert blink_signal(0.9, 0.2, "left") == 0.9 and blink_signal(0.9, 0.2, "right") == 0.2


# -- long closes ----------------------------------------------------------------------------------

def feed(detector, closures, t0=0.0):
    out = []
    for i, c in enumerate(closures):
        for ev in detector.update(t0 + i / FPS, c):
            out.append((ev.type, round(ev.duration_ms)))
    return out


def test_long_close_of_both_eyes():
    d = BlinkDetector()
    ev = feed(d, [0.2] * 60 + [0.5] + [0.92] * 45 + [0.5] + [0.2] * 30)
    assert [e[0] for e in ev] == ["long_close_ready", "long_close"]
    assert ev[0][1] >= 1000 and 1450 <= ev[1][1] <= 1650


def test_looking_down_is_not_a_long_close():
    d = BlinkDetector()
    # The lids drop to 0.55 while reading the bottom of the screen for 3 s.
    ev = feed(d, [0.2] * 60 + [0.55] * 90 + [0.2] * 30)
    assert all(e[0] not in ("long_close", "long_close_ready") for e in ev)


def test_very_long_closes_are_rest_not_gestures():
    d = BlinkDetector(BlinkConfig(long_close_max_ms=3000))
    ev = feed(d, [0.2] * 60 + [0.92] * 150 + [0.2] * 30)
    assert "long_close" not in [e[0] for e in ev]
    # ...and the eyes stayed "closed" the whole time (the baseline did not creep up).
    assert [e[0] for e in ev].count("long_close_ready") == 1


# -- blink analysis with both eyes -------------------------------------------------------------

def two_eye_blinks(left_rise=0.6, right_rise=0.6, pairs=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    t = 0.0

    def add(cl, cr):
        nonlocal t
        rows.append((t, cl + 0.01 * rng.standard_normal(), cr + 0.01 * rng.standard_normal()))
        t += 1 / FPS

    for _ in range(pairs):
        for _ in range(40):
            add(0.2, 0.2)
        for _ in range(2):
            for v in np.sin(np.linspace(0.3, math.pi - 0.3, 5)):
                add(0.2 + left_rise * v, 0.2 + right_rise * v)
            for _ in range(6):
                add(0.2, 0.2)
        for _ in range(30):
            add(0.2, 0.2)
    return rows


def test_blink_analysis_prefers_both_eyes():
    res = analyze_blinks(two_eye_blinks())
    assert res["signal"] == "both" and res["blink_asym"] < 0.1 and res["n_pairs"] == 3


def test_blink_analysis_handles_an_eye_that_hardly_closes():
    # Facial palsy: the right eye barely closes when blinking.
    res = analyze_blinks(two_eye_blinks(right_rise=0.04))
    assert res["signal"] in ("mean", "left")
    assert res["blink_asym"] > 0.4
    # The chosen signal and thresholds catch this person's double blinks...
    cfg = BlinkConfig(sensitivity=res["sensitivity"], min_threshold=res["min_threshold"],
                      double_gap_ms=res["double_gap_ms"])
    d = BlinkDetector(cfg)
    usage = two_eye_blinks(right_rise=0.04, pairs=4, seed=3)
    hits = sum(1 for t, l, r in usage for e in d.update(t, blink_signal(l, r, res["signal"]))
               if e.type == "double_blink")
    assert hits == 4
    # ...and their lopsided blinks are not mistaken for winks.
    w = WinkDetector(wink_config({"blink": res}))
    assert run(w, [(l, r) for _, l, r in usage]) == []


def test_a_wink_that_was_only_starting_is_forgotten_when_the_face_is_lost():
    d = WinkDetector()
    run(d, open_eyes(60))
    d.update(2.0, 0.85, 0.2)                     # one closing frame, then the face is gone for 3 s
    assert d.update_missing(2.5) == [] and d.update_missing(5.0) == []
    ev = run(d, [(0.85, 0.2)] * 3 + open_eyes(20), t0=5.03)
    assert all(e[1] != "wink_start" for e in ev)


def test_long_close_once_announced_is_not_a_blink():
    d = BlinkDetector(BlinkConfig(long_close_ms=600, max_closed_ms=700))
    ev = feed(d, [0.2] * 60 + [0.95] * 20 + [0.2] * 20)   # ~0.67 s, deep
    kinds = [e[0] for e in ev]
    assert kinds == ["long_close_ready", "long_close"]
