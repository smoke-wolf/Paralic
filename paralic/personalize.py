"""Per-person tuning: blink thresholds, smoothing, button magnet, fine-tuning
of the gaze network, and the statistics behind the A/B experiments.

Everything here works on one person's own data:

* :func:`analyze_blinks` - learns how strongly and how fast someone blinks
  from a short "blink twice when the dot turns purple" recording.
* :func:`tune_smoothing` - picks the least cursor smoothing that brings this
  person's measured gaze jitter under a target (less smoothing = less lag).
* :func:`recommend_magnet` - sizes the button magnet from measured accuracy.
* :func:`run_finetune` - champion/challenger fine-tuning: candidates trained on
  new practice and click data must beat the current network on the person's
  most recent, held-out events before they replace it.
* :func:`analyze_experiment` - statistics for the live A/B tests (blind,
  randomised target trials), with a permutation test.
"""

from __future__ import annotations

import math
import time
from typing import Optional, Sequence

import numpy as np

from .calibration import (CalibrationData, CalibrationError, TRAIN_KINDS, fit_full_calibration,
                          prepare_training_set, reject_outliers)
from .filters import OneEuroFilter, OneEuroParams
from .gazenet import GazeNet, ModelConfig, search_configs


class PersonalizationError(Exception):
    """Not enough (or not good enough) data to personalise something."""


# ---------------------------------------------------------------------------
# Cursor smoothing
# ---------------------------------------------------------------------------

SMOOTHING_LEVELS = {"low": 2.0, "medium": 5.0, "high": 8.0}


def smoothing_params(level: float) -> OneEuroParams:
    """One continuous family of One Euro settings: level 0 (snappy) .. 10 (very smooth)."""
    f = min(max(float(level), 0.0), 10.0) / 10.0
    return OneEuroParams(min_cutoff=1.2 * (0.2 / 1.2) ** f, beta=0.003 * 0.1 ** f)


def simulated_jitter(level: float, sigma_px: float, fps: float = 30.0, seconds: float = 6.0, seed: int = 0) -> float:
    """Per-axis output jitter of the filter on white noise with std ``sigma_px``."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    noise = rng.normal(0.0, sigma_px, (n, 2))
    filt = OneEuroFilter(smoothing_params(level))
    out = np.array([filt(i / fps, p) for i, p in enumerate(noise)])
    return float(out[int(fps):].std(axis=0).mean())


def tune_smoothing(raw_jitter_px: float, target_px: float = 14.0) -> dict:
    """Least smoothing whose expected cursor jitter is at most ``target_px``.

    ``raw_jitter_px`` is the RMS distance of unsmoothed gaze predictions from
    their mean while looking at a dot (the "precision" of the validation).
    """
    sigma = max(float(raw_jitter_px), 1.0) / math.sqrt(2.0)
    chosen, jitter = 10.0, None
    for level in np.arange(0.0, 10.01, 0.5):
        jitter = simulated_jitter(level, sigma)
        if jitter <= target_px:
            chosen = float(level)
            break
    return {
        "tuned_level": chosen,
        "raw_jitter_px": round(float(raw_jitter_px), 1),
        "expected_jitter_px": round(float(jitter), 1),
        "target_px": target_px,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Button magnet
# ---------------------------------------------------------------------------

def recommend_magnet(accuracy_px: float) -> dict:
    """Snap radius a bit larger than the typical error, so near misses still count."""
    acc = max(float(accuracy_px), 1.0)
    return {
        "radius_px": round(float(np.clip(1.1 * acc + 20.0, 50.0, 240.0)), 1),
        "pull": 0.3 if acc < 80 else 0.45,
        "accuracy_px": round(acc, 1),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Blinks
# ---------------------------------------------------------------------------

def _find_blinks(t: np.ndarray, c: np.ndarray) -> tuple[float, float, list[dict], list[float]]:
    """Blinks in one closure signal: (baseline, noise, blinks, double-blink gaps)."""
    baseline = float(np.median(c))
    open_part = c[c <= np.percentile(c, 70)]
    noise = max(1.4826 * float(np.median(np.abs(open_part - np.median(open_part)))), 0.005)
    theta = baseline + max(4.0 * noise, 0.08)

    blinks = []
    above = c > theta
    i = 0
    while i < len(c):
        if not above[i]:
            i += 1
            continue
        j = i
        while j < len(c) and above[j]:
            j += 1
        if j < len(c) and i > 0:  # skip blinks cut off by the window edges
            dur_ms = (t[j] - t[i]) * 1000.0
            if 20.0 <= dur_ms <= 900.0:
                blinks.append({"start": t[i], "end": t[j], "i": i, "j": j, "dur_ms": dur_ms,
                               "peak": float(c[i:j].max())})
        i = j

    gaps = []
    k = 0
    while k < len(blinks) - 1:
        gap_ms = (blinks[k + 1]["start"] - blinks[k]["end"]) * 1000.0
        if gap_ms <= 1000.0:
            gaps.append(gap_ms)
            k += 2
        else:
            k += 1
    return baseline, noise, blinks, gaps


def analyze_blinks(samples: Sequence[tuple], prompts: int = 3) -> dict:
    """Learn someone's blinks from a prompted recording.

    ``samples`` are (time in seconds, eye-closure score) - or (time, left eye,
    right eye) - for every frame of a window in which the person was asked to
    blink twice ``prompts`` times. A permissive threshold finds all blinks;
    their strength, length and the pause inside double blinks then set this
    person's detector.

    With per-eye scores it also chooses which signal to watch: normally the
    more open eye ("both" eyes must close, so a wink is never a blink), but
    the average or a single eye when one eye hardly closes (e.g. facial
    palsy), and it measures how lopsided ordinary blinks are, so that winks
    can be told apart from them.
    """
    if len(samples) < 30:
        raise PersonalizationError("Not enough camera frames - keep your face in view.")
    arr = np.array([tuple(float(v) for v in s[:3]) if len(s) >= 3 else (float(s[0]), float(s[1]), float(s[1]))
                    for s in samples], float)
    arr = arr[np.argsort(arr[:, 0])]
    t = arr[:, 0]
    per_eye = all(len(s) >= 3 for s in samples)
    if per_eye:
        from .gestures import blink_signal

        signals = {mode: np.array([blink_signal(l, r, mode) for l, r in arr[:, 1:3]])
                   for mode in ("both", "mean", "left", "right")}
    else:
        signals = {"given": arr[:, 1]}

    found = {}
    for mode, c in signals.items():
        baseline, noise, blinks, gaps = _find_blinks(t, c)
        if len(blinks) >= 3 and gaps:
            rise = float(np.median([b["peak"] for b in blinks])) - baseline
            found[mode] = (baseline, noise, blinks, gaps, rise)
    if not found:
        raise PersonalizationError("I couldn't see clear double blinks. Try again with two quick, full blinks.")
    if per_eye:
        snr = {m: v[4] / v[1] for m, v in found.items()}
        best = max(snr, key=snr.get)
        # Prefer "both eyes" unless it is much weaker than the alternatives.
        if "both" in found and found["both"][4] >= 0.1 and snr["both"] >= 0.5 * snr[best]:
            mode = "both"
        elif "mean" in found and found["mean"][4] >= 0.1 and snr["mean"] >= 0.6 * snr[best]:
            mode = "mean"
        else:
            mode = best
    else:
        mode = "given"
    baseline, noise, blinks, gaps, rise = found[mode]

    headroom = max(1.0 - baseline, 1e-3)
    # Trigger at about half of a typical blink, but safely above the noise.
    sensitivity = max(0.5 * rise / headroom, 3.5 * noise / headroom)
    sensitivity = float(np.clip(sensitivity, 0.12, 0.45))
    min_threshold = float(np.clip(baseline + 3.0 * noise + 0.03, 0.12, 0.30))
    durations = [b["dur_ms"] for b in blinks]
    result = {
        "sensitivity": round(sensitivity, 3),
        "min_threshold": round(min_threshold, 3),
        "double_gap_ms": round(float(np.clip(1.5 * max(gaps) + 120.0, 300.0, 1000.0))),
        "max_closed_ms": round(float(np.clip(max(700.0, 2.0 * max(durations)), 500.0, 1000.0))),
        "baseline": round(baseline, 3),
        "noise": round(noise, 4),
        "peak_rise": round(rise, 3),
        "blink_ms": round(float(np.median(durations))),
        "gap_ms": round(float(np.median(gaps))),
        "n_blinks": len(blinks),
        "n_pairs": len(gaps),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if per_eye:
        # How lopsided are this person's ordinary blinks? (Peak difference
        # between the eyes' rises above their own baselines.)
        bl, br = float(np.median(arr[:, 1])), float(np.median(arr[:, 2]))
        asym = [float(np.max(np.abs((arr[b["i"]:b["j"], 1] - bl) - (arr[b["i"]:b["j"], 2] - br))))
                for b in blinks]
        result["signal"] = mode
        result["blink_asym"] = round(float(np.percentile(asym, 90)), 3)
        result["signals"] = {m: {"rise": round(v[4], 3), "snr": round(v[4] / v[1], 1), "n_blinks": len(v[2])}
                             for m, v in found.items()}
    return result


# ---------------------------------------------------------------------------
# A/B experiment statistics
# ---------------------------------------------------------------------------

TRIAL_TIMEOUT_MS = 10000.0


def trial_score(trial: dict) -> float:
    """Lower is better: log time-to-hit plus penalties for misses and timeouts."""
    t = min(max(float(trial.get("time_ms", TRIAL_TIMEOUT_MS)), 150.0), TRIAL_TIMEOUT_MS) / 1000.0
    return math.log(t) + 0.3 * float(trial.get("misses", 0)) + (0.5 if trial.get("timeout") else 0.0)


def permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 0) -> float:
    """Two-sided permutation test for a difference in means."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    perms = rng.permuted(np.tile(pooled, (n_perm, 1)), axis=1)
    diffs = np.abs(perms[:, :len(a)].mean(axis=1) - perms[:, len(a):].mean(axis=1))
    return float((1 + np.count_nonzero(diffs >= observed - 1e-12)) / (n_perm + 1))


def paired_improvement_pvalue(diff: np.ndarray, n_perm: int = 5000, seed: int = 0) -> float:
    """One-sided sign-flip test that paired differences are below zero."""
    diff = np.asarray(diff, float)
    observed = diff.mean()
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    means = (signs * np.abs(diff)).mean(axis=1)
    return float((1 + np.count_nonzero(means <= observed + 1e-12)) / (n_perm + 1))


def analyze_experiment(trials: Sequence[dict], arms: Sequence[str], baseline: str = "current",
                       min_per_arm: int = 8, alpha: float = 0.1, min_effect: float = 0.05) -> dict:
    """Compare the arms of an A/B(/n) test and decide whether to switch.

    Decisions: ``need_more`` (too few trials), ``keep`` (the current setting
    is best), ``adopt`` (another arm is better, significantly and by at least
    ``min_effect``), ``inconclusive`` (better on average but not clearly).
    With several challengers the significance level is Bonferroni-corrected.
    """
    by_arm = {arm: [t for t in trials if t.get("arm") == arm] for arm in arms}
    stats = {}
    scores = {}
    for arm, ts in by_arm.items():
        sc = np.array([trial_score(t) for t in ts], float)
        scores[arm] = sc
        times = [min(float(t.get("time_ms", TRIAL_TIMEOUT_MS)), TRIAL_TIMEOUT_MS) for t in ts]
        stats[arm] = {
            "n": len(ts),
            "score": round(float(sc.mean()), 4) if len(sc) else None,
            "median_time_ms": round(float(np.median(times))) if times else None,
            "misses_per_trial": round(float(np.mean([t.get("misses", 0) for t in ts])), 3) if ts else None,
            "timeouts": int(sum(1 for t in ts if t.get("timeout"))),
        }
    result = {"arms": stats, "baseline": baseline, "best": None, "p_value": None, "effect": None,
              "decision": "need_more", "n_trials": len(trials)}
    ready = [a for a in arms if len(scores[a]) >= 3]
    if baseline not in ready or len(ready) < 2:
        return result
    best = min(ready, key=lambda a: scores[a].mean())
    result["best"] = best
    if len(scores[best]) < min_per_arm or len(scores[baseline]) < min_per_arm:
        return result
    if best == baseline:
        result["decision"] = "keep"
        return result
    p = permutation_pvalue(scores[best], scores[baseline])
    effect = 1.0 - math.exp(scores[best].mean() - scores[baseline].mean())
    result["p_value"] = round(p, 4)
    result["effect"] = round(effect, 4)
    level = alpha / max(1, len(arms) - 1)
    result["decision"] = "adopt" if (p < level and effect >= min_effect) else "inconclusive"
    return result


# ---------------------------------------------------------------------------
# Fine-tuning: champion vs challengers
# ---------------------------------------------------------------------------

def _event_errors(model: GazeNet, events: Sequence[list]) -> np.ndarray:
    errs = []
    for frames in events:
        F = np.array([f.features for f in frames])
        F = F[reject_outliers(F)]
        pred = model.predict(F).mean(axis=0)
        errs.append(float(np.linalg.norm(pred - np.array(frames[0].target, float))))
    return np.array(errs)


def implausible_error_px(data: CalibrationData) -> float:
    """22% of the screen diagonal, estimated from the calibration dots' spread.

    A popped target means the cursor was within about a target radius plus the
    magnet radius of it; together with the page's drift correction the raw
    prediction can be ~400 px off on a 1080p screen, but hardly more.
    """
    cal = np.array([f.target for f in data.frames if f.kind in ("cal", "val")], float)
    if len(cal) >= 2:
        span = float(np.hypot(*np.ptp(cal, axis=0)))
        diag = span / 0.88 if span > 100 else 2200.0  # the dots cover ~88% of the screen
    else:
        diag = 2200.0
    return 0.22 * diag


def _training_arrays(data: CalibrationData):
    frames = data.of_kind(*TRAIN_KINDS)
    X, Y, _, W, _ = prepare_training_set(frames)
    return X, Y, W


def run_finetune(champion: GazeNet, data: CalibrationData, *, min_new_events: int = 8, min_test_events: int = 4,
                 test_fraction: float = 0.4, min_gain: float = 0.03, alpha: float = 0.2,
                 configs: Optional[Sequence[ModelConfig]] = None, seed: int = 0) -> tuple[Optional[GazeNet], dict]:
    """Try to improve ``champion`` with the fine-tuning events in ``data``.

    Events recorded after the champion was trained are split in time: the
    earlier ones may be used for training, the most recent ones are held out.
    Two challengers are trained on everything except the held-out events:

    * ``fine-tuned`` - the champion's weights trained further (warm start),
    * ``retrained``  - a new network from a per-person model search
      (several architectures compete in cross-validation).

    The best challenger replaces the champion only if its error on the
    held-out events is lower by at least ``min_gain`` (relative) and a paired
    sign-flip test gives p < ``alpha``. The winner is then refit on all data.
    Returns (new model or None, report).
    """
    started = time.perf_counter()
    events = data.ft_events()
    trained_ts = float(champion.meta.get("trained_ts", 0.0))

    # Plausibility filter: a target cannot have been popped (or a button
    # clicked) while the eyes were a quarter of the screen away from it, so such
    # events have wrong labels (an accidental double blink, a mix-up). Drop
    # them; if many recent events look like that, don't learn from this batch.
    limit = implausible_error_px(data)
    champ_all = dict(zip(events, _event_errors(champion, list(events.values())))) if events else {}
    implausible = {i for i, err in champ_all.items() if err > limit}
    recent = [i for i, frames in events.items() if frames[0].t > trained_ts]
    if recent and len(implausible & set(recent)) > 0.3 * len(recent):
        raise PersonalizationError("The recent practice data don't look reliable (many targets were far from where "
                                   "the eyes pointed), so the gaze model was left unchanged.")
    if implausible:
        events = {i: f for i, f in events.items() if i not in implausible}
        data = CalibrationData([f for f in data.frames if not (f.kind == "ft" and f.point in implausible)])
    new_ids = [i for i, frames in events.items() if frames[0].t > trained_ts]
    if len(new_ids) < min_new_events:
        raise PersonalizationError(
            f"Need at least {min_new_events} new practice targets or clicks since the last update "
            f"(have {len(new_ids)}).")
    n_test = max(min_test_events, int(math.ceil(test_fraction * len(new_ids))))
    n_test = min(n_test, len(new_ids) - 1)
    test_ids = set(new_ids[-n_test:])
    test_events = [events[i] for i in sorted(test_ids)]

    train_data = CalibrationData([f for f in data.frames if not (f.kind == "ft" and f.point in test_ids)])
    # Events from before the champion describe older sessions: trust them less.
    for i, f in enumerate(train_data.frames):
        if f.kind == "ft" and f.t <= trained_ts:
            train_data.frames[i] = type(f)(t=f.t, features=f.features, target=f.target, kind=f.kind,
                                           point=f.point, weight=0.5 * f.weight)

    X, Y, W = _training_arrays(train_data)
    candidates: dict[str, GazeNet] = {"fine-tuned": champion.fine_tune(X, Y, W)}
    search_table: dict = {}
    retrained_config = None
    try:
        retrained, info = fit_full_calibration(train_data, kinds=TRAIN_KINDS,
                                               configs=list(configs) if configs else search_configs())
        candidates["retrained"] = retrained
        search_table = info.get("candidates", {})
        retrained_config = retrained.meta.get("config")
    except CalibrationError:
        pass

    # Robust comparison: cap each event's error so that one odd event cannot
    # decide the outcome, and require winning on at least half of the events.
    cap = 0.5 * limit
    champ_err = np.minimum(_event_errors(champion, test_events), cap)
    cand_errs = {name: np.minimum(_event_errors(m, test_events), cap) for name, m in candidates.items()}
    winner = min(cand_errs, key=lambda k: cand_errs[k].mean())
    diff = cand_errs[winner] - champ_err
    gain = float(-diff.mean() / max(champ_err.mean(), 1e-6))
    p = paired_improvement_pvalue(diff, seed=seed)
    win_rate = float(np.mean(diff < 0))
    accepted = gain >= min_gain and p < alpha and win_rate >= 0.5

    new_model = None
    if accepted:
        full = CalibrationData(list(data.frames))
        for i, f in enumerate(full.frames):
            if f.kind == "ft" and f.t <= trained_ts:
                full.frames[i] = type(f)(t=f.t, features=f.features, target=f.target, kind=f.kind,
                                         point=f.point, weight=0.5 * f.weight)
        if winner == "fine-tuned":
            Xa, Ya, Wa = _training_arrays(full)
            new_model = champion.fine_tune(Xa, Ya, Wa)
        else:
            new_model, _ = fit_full_calibration(full, kinds=TRAIN_KINDS,
                                                configs=[ModelConfig.from_dict(retrained_config)])
            # The refit has a single candidate, so it has no cross-validated
            # error to choose the leading eye with: keep the search's choice.
            for key in ("eye", "eye_cv_px", "cv_error_px"):
                if key in candidates["retrained"].meta:
                    new_model.meta[key] = candidates["retrained"].meta[key]
        new_model.meta.update({
            "trained_ts": time.time(),
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": winner,
            "last_event": max(events) if events else 0,
        })

    report = {
        "accepted": bool(accepted),
        "dropped_events": len(implausible),
        "winner": winner,
        "champion_error_px": round(float(champ_err.mean()), 1),
        "candidate_errors_px": {k: round(float(v.mean()), 1) for k, v in cand_errs.items()},
        "improvement": round(gain, 4),
        "p_value": round(p, 4),
        "win_rate": round(win_rate, 3),
        "n_test_events": len(test_events),
        "n_new_events": len(new_ids),
        "n_events": len(events),
        "model_search": search_table,
        "search_winner": ModelConfig.from_dict(retrained_config).name if retrained_config is not None else None,
        "seconds": round(time.perf_counter() - started, 2),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return new_model, report


# ---------------------------------------------------------------------------
# Live experiments: definitions of the arms
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "smoothing": {
        "title": "Cursor smoothing",
        "arms": ["current", "smoother", "snappier"],
    },
    "magnet": {
        "title": "Button magnet",
        "arms": ["current", "stronger", "off"],
    },
    "double_blink": {
        "title": "Double-blink timing",
        "arms": ["current", "relaxed"],
    },
    "dwell": {
        "title": "Dwell-click time",
        "arms": ["current", "faster", "slower"],
    },
}


def experiment_arms(name: str, effective: dict) -> list[dict]:
    """Concrete settings for each arm, relative to this person's current values."""
    level = float(effective["smoothing_level"])
    magnet = effective.get("magnet") or {"radius_px": 110.0, "pull": 0.3}
    gap = float(effective["double_gap_ms"])
    if name == "smoothing":
        return [
            {"id": "current", "label": "Current", "smoothing_level": level},
            {"id": "smoother", "label": "Smoother", "smoothing_level": min(10.0, level + 2.0)},
            {"id": "snappier", "label": "Snappier", "smoothing_level": max(0.0, level - 2.0)},
        ]
    if name == "magnet":
        return [
            {"id": "current", "label": "Current", "magnet": {"radius_px": magnet["radius_px"], "pull": magnet["pull"]}},
            {"id": "stronger", "label": "Stronger",
             "magnet": {"radius_px": round(min(300.0, 1.5 * magnet["radius_px"]), 1), "pull": 0.55}},
            {"id": "off", "label": "Off", "magnet": {"radius_px": 0.0, "pull": 0.0}},
        ]
    if name == "double_blink":
        return [
            {"id": "current", "label": "Current", "double_gap_ms": gap},
            {"id": "relaxed", "label": "Relaxed", "double_gap_ms": min(1100.0, gap + 200.0)},
        ]
    if name == "dwell":
        dwell = float((effective.get("gestures") or {}).get("dwell_ms", 1000))
        return [
            {"id": "current", "label": "Current", "dwell_ms": round(dwell)},
            {"id": "faster", "label": "Faster", "dwell_ms": round(max(400.0, 0.75 * dwell))},
            {"id": "slower", "label": "Slower", "dwell_ms": round(min(3000.0, 1.33 * dwell))},
        ]
    raise KeyError(name)
