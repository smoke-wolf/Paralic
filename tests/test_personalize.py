import math
import time

import numpy as np
import pytest

from paralic.blink import BlinkConfig, BlinkDetector
from paralic.calibration import CalibrationData, LabeledFrame, fit_full_calibration
from paralic.personalize import (PersonalizationError, analyze_blinks, analyze_experiment, experiment_arms,
                                 paired_improvement_pvalue, permutation_pvalue, recommend_magnet, run_finetune,
                                 simulated_jitter, smoothing_params, tune_smoothing)
from tests.synthetic import SCREEN_H, SCREEN_W, Head, VirtualUser, simulate_calibration

FPS = 30.0


# -- blinks -------------------------------------------------------------------------

def blink_trace(baseline=0.1, rise=0.5, noise=0.01, pairs=3, gap_frames=6, blink_frames=5, seed=0):
    """Closure samples for a prompted recording: `pairs` double blinks, ~3 s apart."""
    rng = np.random.default_rng(seed)
    c = []
    for _ in range(pairs):
        c += [baseline] * 40
        for _ in range(2):
            c += [baseline + rise * v for v in np.sin(np.linspace(0.3, math.pi - 0.3, blink_frames))]
            c += [baseline] * gap_frames
        c += [baseline] * 30
    c = np.array(c) + noise * rng.standard_normal(len(c))
    return [(i / FPS, float(v)) for i, v in enumerate(c)]


def test_blink_analysis_learns_strength_and_timing():
    res = analyze_blinks(blink_trace())
    assert res["n_blinks"] == 6 and res["n_pairs"] == 3
    assert 0.2 < res["sensitivity"] < 0.35
    assert 120 <= res["gap_ms"] <= 280
    assert 300 <= res["double_gap_ms"] <= 700


def test_blink_analysis_needs_clear_blinks():
    flat = [(i / FPS, 0.1 + 0.005 * math.sin(i)) for i in range(300)]
    with pytest.raises(PersonalizationError):
        analyze_blinks(flat)


def _count_double_blinks(config, samples):
    d = BlinkDetector(config)
    return sum(1 for t, c in samples for e in d.update(t, c) if e.type == "double_blink")


def test_personal_thresholds_catch_light_blinkers():
    # Someone whose blinks only raise the closure score by 0.2.
    calib = blink_trace(rise=0.2, seed=1)
    res = analyze_blinks(calib)
    usage = blink_trace(rise=0.2, pairs=5, seed=2)
    default_hits = _count_double_blinks(BlinkConfig(), usage)
    personal = BlinkConfig(sensitivity=res["sensitivity"], min_threshold=res["min_threshold"],
                           double_gap_ms=res["double_gap_ms"])
    personal_hits = _count_double_blinks(personal, usage)
    assert default_hits == 0
    assert personal_hits == 5


# -- smoothing and magnet ---------------------------------------------------------------

def test_smoothing_family_is_monotonic():
    params = [smoothing_params(level) for level in range(11)]
    assert all(a.min_cutoff > b.min_cutoff and a.beta > b.beta for a, b in zip(params, params[1:]))
    assert simulated_jitter(8, 30) < simulated_jitter(2, 30)


def test_noisier_eyes_get_more_smoothing():
    calm = tune_smoothing(20.0)
    jittery = tune_smoothing(90.0)
    assert jittery["tuned_level"] > calm["tuned_level"]
    assert calm["expected_jitter_px"] <= calm["target_px"]


def test_magnet_scales_with_accuracy():
    assert recommend_magnet(40)["radius_px"] < recommend_magnet(120)["radius_px"]
    assert recommend_magnet(1000)["radius_px"] == 240


# -- A/B statistics ----------------------------------------------------------------------------

def _trials(arm, median_ms, n, seed, misses=0):
    rng = np.random.default_rng(seed)
    return [{"arm": arm, "time_ms": float(median_ms * math.exp(0.25 * rng.standard_normal())), "misses": misses}
            for _ in range(n)]


def test_clear_winner_is_adopted():
    trials = _trials("current", 2200, 12, 1) + _trials("smoother", 1300, 12, 2) + _trials("snappier", 2400, 12, 3)
    res = analyze_experiment(trials, ["current", "smoother", "snappier"])
    assert res["best"] == "smoother" and res["decision"] == "adopt"
    assert res["p_value"] < 0.05 and res["effect"] > 0.3


def test_no_difference_is_not_adopted():
    trials = _trials("current", 1500, 12, 4) + _trials("relaxed", 1500, 12, 5)
    res = analyze_experiment(trials, ["current", "relaxed"])
    assert res["decision"] in ("keep", "inconclusive")


def test_too_few_trials_need_more():
    trials = _trials("current", 2000, 3, 6) + _trials("off", 900, 3, 7)
    assert analyze_experiment(trials, ["current", "off"])["decision"] == "need_more"


def test_misses_count_against_an_arm():
    fast_but_sloppy = _trials("stronger", 1200, 10, 8, misses=3)
    careful = _trials("current", 1400, 10, 9)
    res = analyze_experiment(fast_but_sloppy + careful, ["current", "stronger"])
    assert res["best"] == "current"


def test_permutation_tests():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 30), rng.normal(1.5, 1, 30)
    assert permutation_pvalue(a, b) < 0.01
    assert permutation_pvalue(a, rng.normal(0, 1, 30)) > 0.05
    assert paired_improvement_pvalue(np.full(10, -5.0) + rng.normal(0, 1, 10)) < 0.01
    assert paired_improvement_pvalue(rng.normal(0, 1, 10)) > 0.05


def test_experiment_arms_are_relative_to_current_settings():
    eff = {"smoothing_level": 5.0, "double_gap_ms": 550.0, "magnet": {"radius_px": 100.0, "pull": 0.3}}
    sm = {a["id"]: a for a in experiment_arms("smoothing", eff)}
    assert sm["smoother"]["smoothing_level"] == 7.0 and sm["snappier"]["smoothing_level"] == 3.0
    mg = {a["id"]: a for a in experiment_arms("magnet", eff)}
    assert mg["stronger"]["magnet"]["radius_px"] == 150.0 and mg["off"]["magnet"]["radius_px"] == 0.0
    db = {a["id"]: a for a in experiment_arms("double_blink", eff)}
    assert db["relaxed"]["double_gap_ms"] == 750.0
    dw = {a["id"]: a for a in experiment_arms("dwell", {**eff, "gestures": {"dwell_ms": 1000}})}
    assert dw["faster"]["dwell_ms"] == 750 and dw["slower"]["dwell_ms"] == 1330


# -- fine-tuning ------------------------------------------------------------------------------------

def _calibrated(seed=3):
    user = VirtualUser(seed=seed)
    X, Y, G = simulate_calibration(user, seed=seed + 10)
    data = CalibrationData()
    t0 = time.time() - 3600
    for i, (x, y, g) in enumerate(zip(X, Y, G)):
        kind, pt = g.split(":")
        data.add(LabeledFrame(t=t0 + i / 30, features=x, target=(y[0], y[1]), kind=kind, point=int(pt)))
    model, _ = fit_full_calibration(data)
    return user, data, model


def _add_events(data, user, head, n, seed, corrupt=False):
    rng = np.random.default_rng(seed)
    t = time.time() + 1
    for k in range(n):
        sx, sy = rng.uniform(0.08, 0.92) * SCREEN_W, rng.uniform(0.08, 0.92) * SCREEN_H
        label = (rng.uniform(0, SCREEN_W), rng.uniform(0, SCREEN_H)) if corrupt else (sx, sy)
        event = data.next_event_id()
        for _ in range(8):
            data.add(LabeledFrame(t=t + k, features=user.features(sx, sy, head), target=label, kind="ft",
                                  point=event, weight=1.0))


def _drift_error(model, user, head, n=150, seed=11):
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n):
        sx, sy = rng.uniform(0.05, 0.95) * SCREEN_W, rng.uniform(0.05, 0.95) * SCREEN_H
        errs.append(np.hypot(*(model.predict(user.features(sx, sy, head))[0] - (sx, sy))))
    return float(np.mean(errs))


DRIFTED = Head(x=3.0, y=13.0, dist=68.0, pitch=0.06)


def test_finetune_adapts_to_a_new_sitting_position():
    user, data, champion = _calibrated()
    _add_events(data, user, DRIFTED, 24, seed=1)
    new_model, report = run_finetune(champion, data)
    assert report["accepted"], report
    assert report["candidate_errors_px"][report["winner"]] < report["champion_error_px"]
    assert set(report["model_search"]) >= {"linear", "32x16 l2=0.01"}
    clean = VirtualUser(seed=3, noise=0.0)
    assert _drift_error(new_model, clean, DRIFTED) < 0.8 * _drift_error(champion, clean, DRIFTED)
    assert new_model.meta["trained_ts"] > champion.meta["trained_ts"]


def test_finetune_refuses_garbage_labels():
    user, data, champion = _calibrated(seed=5)
    _add_events(data, user, Head(), 24, seed=2, corrupt=True)
    with pytest.raises(PersonalizationError, match="reliable"):
        run_finetune(champion, data)


def test_finetune_survives_some_wrong_labels():
    user, data, champion = _calibrated(seed=7)
    _add_events(data, user, DRIFTED, 22, seed=4)
    _add_events(data, user, DRIFTED, 5, seed=5, corrupt=True)   # ~20% mislabelled
    new_model, report = run_finetune(champion, data)
    clean = VirtualUser(seed=7, noise=0.0)
    if report["accepted"]:
        assert _drift_error(new_model, clean, DRIFTED) < _drift_error(champion, clean, DRIFTED)


def test_finetune_keeps_champion_when_nothing_changed():
    user, data, champion = _calibrated(seed=8)
    _add_events(data, user, Head(), 20, seed=6)  # same sitting position as the calibration
    new_model, report = run_finetune(champion, data)
    if report["accepted"]:  # allowed, but then it must really be at least as good
        clean = VirtualUser(seed=8, noise=0.0)
        assert _drift_error(new_model, clean, Head()) <= 1.05 * _drift_error(champion, clean, Head())


def test_finetune_needs_enough_new_events():
    user, data, champion = _calibrated(seed=6)
    _add_events(data, user, DRIFTED, 3, seed=3)
    with pytest.raises(PersonalizationError):
        run_finetune(champion, data)
