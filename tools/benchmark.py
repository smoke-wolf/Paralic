"""Benchmark Paralic's personalisation on a population of *simulated* people.

    python tools/benchmark.py                 # ~5 minutes, writes docs/benchmark.md
    python tools/benchmark.py --people 4      # quicker

Real eyes cannot be put in a test suite, so this uses the geometric eye
simulation from ``tests/synthetic.py`` (feature noise, head movement, eyelids)
and synthetic eyelid traces. The numbers show whether each mechanism does
what it should and how large the effects are *in simulation*; they are not
measurements on real people. Every person who uses Paralic gets the same
procedures run on their own data on their own computer.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paralic.blink import BlinkConfig, BlinkDetector  # noqa: E402
from paralic.calibration import CalibrationData, LabeledFrame, fit_full_calibration  # noqa: E402
from paralic.filters import OneEuroFilter  # noqa: E402
from paralic.gazenet import search_configs  # noqa: E402
from paralic.gestures import WinkDetector, analyze_winks, blink_signal, wink_config  # noqa: E402
from paralic.personalize import (PersonalizationError, analyze_blinks, analyze_experiment,  # noqa: E402
                                 run_finetune, smoothing_params, tune_smoothing)
from tests.synthetic import SCREEN_H, SCREEN_W, Head, VirtualUser, simulate_calibration  # noqa: E402

FPS = 30.0
DRIFTED = Head(x=3.0, y=13.0, dist=68.0, pitch=0.06)


def med(values) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def fmt(v, unit="", digits=0) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.{digits}f}{unit}"


# ---------------------------------------------------------------------------
# Gaze networks
# ---------------------------------------------------------------------------

def person(seed: int, **kw) -> VirtualUser:
    rng = np.random.default_rng(1000 + seed)
    return VirtualUser(seed=seed, noise=float(rng.uniform(0.6, 1.6)), **kw)


def calibration_data(user: VirtualUser, seed: int) -> CalibrationData:
    X, Y, G = simulate_calibration(user, seed=seed)
    data = CalibrationData()
    t0 = time.time() - 3600
    for i, (x, y, g) in enumerate(zip(X, Y, G)):
        kind, pt = g.split(":")
        data.add(LabeledFrame(t=t0 + i / 30, features=x, target=(y[0], y[1]), kind=kind, point=int(pt)))
    return data


def fixation_error(model, user, head=Head(), eye=None, closed=None, n=60, frames=8, seed=7) -> float:
    """Mean error of the average prediction over short fixations at random targets."""
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n):
        sx, sy = rng.uniform(0.05, 0.95) * SCREEN_W, rng.uniform(0.05, 0.95) * SCREEN_H
        F = np.array([user.features(sx, sy, head, closed=closed) for _ in range(frames)])
        errs.append(float(np.hypot(*(model.predict(F, eye).mean(axis=0) - (sx, sy)))))
    return float(np.mean(errs))


def bench_model_search(n_people: int) -> dict:
    """Default (weight decay only) vs the wider per-person search that fine-tuning runs.

    As in fine-tuning, the training data are the calibration plus 24 practice hits.
    """
    rows = []
    for k in range(n_people):
        user = person(k)
        data = calibration_data(user, seed=k + 50)
        add_events(data, user, Head(), 24, seed=k + 200)
        kinds = ("cal", "head", "ft")
        default, _ = fit_full_calibration(data, eyes=False, kinds=kinds)
        searched, info = fit_full_calibration(data, eyes=False, kinds=kinds, configs=search_configs())
        test = person(k)
        rows.append({
            "default": fixation_error(default, test),
            "search": fixation_error(searched, test),
            "linear": info["candidates"].get("linear"),
            "chosen": info["config"],
        })
    return {
        "default_px": med([r["default"] for r in rows]),
        "search_px": med([r["search"] for r in rows]),
        "search_better": float(np.mean([r["search"] < r["default"] for r in rows])),
        "configs": sorted({r["chosen"] for r in rows}),
        "n": len(rows),
    }


def add_events(data, user, head, n, seed, corrupt=0.0):
    rng = np.random.default_rng(seed)
    t = time.time() + 1
    for k in range(n):
        sx, sy = rng.uniform(0.08, 0.92) * SCREEN_W, rng.uniform(0.08, 0.92) * SCREEN_H
        wrong = rng.random() < corrupt
        label = (rng.uniform(0, SCREEN_W), rng.uniform(0, SCREEN_H)) if wrong else (sx, sy)
        event = data.next_event_id()
        for _ in range(8):
            data.add(LabeledFrame(t=t + k, features=user.features(sx, sy, head), target=label, kind="ft",
                                  point=event, weight=1.0))


def bench_finetune(n_people: int) -> dict:
    out = {}
    scenarios = [
        ("Moved in the chair (24 practice hits)", DRIFTED, 24, 0.0),
        ("Moved, 20 % of labels wrong", DRIFTED, 24, 0.2),
        ("Same position as calibration", Head(), 24, 0.0),
        ("All labels wrong", Head(), 24, 1.0),
    ]
    for name, head, n_events, corrupt in scenarios:
        accepted = refused = 0
        before, after, harmed = [], [], 0
        for k in range(n_people):
            user = person(k)
            data = calibration_data(user, seed=k + 50)
            champion, _ = fit_full_calibration(data)
            add_events(data, user, head, n_events, seed=k + 200, corrupt=corrupt)
            try:
                new, report = run_finetune(champion, data)
            except PersonalizationError:
                refused += 1
                continue
            test = person(k)
            b = fixation_error(champion, test, head)
            if report["accepted"]:
                accepted += 1
                a = fixation_error(new, test, head)
                before.append(b)
                after.append(a)
                harmed += int(a > 1.05 * b)
        out[name] = {"accepted": accepted, "refused": refused, "n": n_people, "before_px": med(before),
                     "after_px": med(after), "harmed": harmed}
    return out


def bench_eyes(n_people: int) -> dict:
    wink_rows, squint_rows = [], []
    for k in range(n_people):
        user = person(k)
        data = calibration_data(user, seed=k + 50)
        model, info = fit_full_calibration(data)
        test = person(k)
        # During a wink the open eye's network takes over, aligned on the frames
        # just before the wink (as the session does).
        rng = np.random.default_rng(k)
        errs_both, errs_one = [], []
        for _ in range(30):
            ax, ay = rng.uniform(0.1, 0.9) * SCREEN_W, rng.uniform(0.1, 0.9) * SCREEN_H
            pre = np.array([test.features(ax, ay, Head()) for _ in range(6)])
            offset = (model.predict(pre) - model.predict(pre, "right")).mean(axis=0)
            bx, by = rng.uniform(0.05, 0.95) * SCREEN_W, rng.uniform(0.05, 0.95) * SCREEN_H
            F = np.array([test.features(bx, by, Head(), closed="left") for _ in range(8)])
            errs_both.append(float(np.hypot(*(model.predict(F).mean(axis=0) - (bx, by)))))
            errs_one.append(float(np.hypot(*((model.predict(F, "right") + offset).mean(axis=0) - (bx, by)))))
        wink_rows.append({"normal": fixation_error(model, test), "both": float(np.mean(errs_both)),
                          "one": float(np.mean(errs_one))})
        # A squint that comes and goes in the right eye.
        squinter = person(k, wander=(0.05, 0.0))
        sq_model, sq_info = fit_full_calibration(calibration_data(squinter, seed=k + 80))
        sq_test = person(k, wander=(0.05, 0.0))
        squint_rows.append({"picked": sq_info["eye"], "default": fixation_error(sq_model, sq_test),
                            "both": fixation_error(sq_model, sq_test, eye="both"),
                            "typical_picked": info["eye"]})
    return {
        "normal_px": med([r["normal"] for r in wink_rows]),
        "wink_both_px": med([r["both"] for r in wink_rows]),
        "wink_one_px": med([r["one"] for r in wink_rows]),
        "squint_left_picked": sum(r["picked"] == "left" for r in squint_rows),
        "typical_both_picked": sum(r["typical_picked"] == "both" for r in squint_rows),
        "squint_default_px": med([r["default"] for r in squint_rows]),
        "squint_both_px": med([r["both"] for r in squint_rows]),
        "n": n_people,
    }


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def settle_ms(level: float, jump_px: float = 600.0) -> float:
    """Time for the filtered cursor to cover 90% of a noise-free jump."""
    f = OneEuroFilter(smoothing_params(level))
    for i in range(30):
        f(i / FPS, np.zeros(2))
    for i in range(30, 120):
        y = f(i / FPS, np.array([jump_px, 0.0]))
        if y[0] >= 0.9 * jump_px:
            return (i - 29) * 1000.0 / FPS
    return float("inf")


def bench_smoothing() -> list:
    rows = []
    for raw in (20, 40, 70, 110):
        tuned = tune_smoothing(raw)
        rows.append({"raw": raw, "level": tuned["tuned_level"], "jitter": tuned["expected_jitter_px"],
                     "settle": settle_ms(tuned["tuned_level"]), "medium_settle": settle_ms(5.0)})
    return rows


# ---------------------------------------------------------------------------
# Blinks and winks (eyelid traces)
# ---------------------------------------------------------------------------

class Lids:
    """Per-eye closure traces for one simulated person."""

    def __init__(self, seed, left_rise=0.6, right_rise=0.6, base=(0.2, 0.2), noise=0.012, wink=(0.85, 0.25)):
        self.rng = np.random.default_rng(seed)
        self.left_rise, self.right_rise = left_rise, right_rise
        self.base = base
        self.noise = noise
        self.wink_level, self.wink_other = wink
        self.rows: list = []
        self.t = 0.0

    def add(self, cl, cr, phase=None, n=1):
        for _ in range(n):
            r = self.rng.standard_normal
            self.rows.append((self.t, cl + self.noise * r(), cr + self.noise * r(), phase))
            self.t += 1 / FPS

    def rest(self, n, phase=None, down=0.0):
        self.add(self.base[0] + down, self.base[1] + down, phase, n)

    def blink(self, frames=5):
        for v in np.sin(np.linspace(0.3, math.pi - 0.3, frames)):
            self.add(self.base[0] + self.left_rise * v, self.base[1] + self.right_rise * v)

    def double_blink(self):
        self.blink()
        self.rest(6)
        self.blink()

    def wink(self, eye, frames, phase=None):
        """One eye closes to ``wink_level``; the other rises by as much as it would from 0.2 to ``wink_other``."""
        ramp = [0.5, 1.0] + [1.0] * frames + [0.5]
        b_eye = self.base[0] if eye == "left" else self.base[1]
        b_other = self.base[1] if eye == "left" else self.base[0]
        for f in ramp:
            closed = b_eye + (max(self.wink_level, b_eye) - b_eye) * f
            other = b_other + (self.wink_other - 0.2) * f
            if eye == "left":
                self.add(closed, other, phase)
            else:
                self.add(other, closed, phase)


def blink_recording(lids: Lids):
    for _ in range(3):
        lids.rest(40)
        lids.double_blink()
        lids.rest(30)
    return [(t, l, r) for t, l, r, _ in lids.rows]


def usage(lids: Lids, double_blinks=10, single_blinks=15, winks=0):
    """Ordinary use: double blinks, natural single blinks, looking down, optional held winks."""
    start = len(lids.rows)
    marks = []
    for k in range(max(double_blinks, single_blinks, winks)):
        lids.rest(45)
        if k < double_blinks:
            lids.double_blink()
            lids.rest(30)
        if k < single_blinks:
            lids.blink()
            lids.rest(40)
        if k < winks:
            for eye in ("left", "right"):
                t0 = lids.t
                lids.wink(eye, 30)
                marks.append((eye, t0, lids.t))
                lids.rest(30)
    lids.rest(150, down=0.3)  # reading the bottom of the screen for 5 s
    lids.rest(30)
    return lids.rows[start:], marks


def count_double_blinks(cfg: BlinkConfig, rows, signal: str) -> int:
    d = BlinkDetector(cfg)
    return sum(1 for t, l, r, _ in rows for e in d.update(t, blink_signal(l, r, signal)) if e.type == "double_blink")


def bench_blinks(n_people: int) -> dict:
    groups = {
        "Typical blinks": dict(left_rise=0.6, right_rise=0.6),
        "Light blinks (closure rises only 0.2)": dict(left_rise=0.2, right_rise=0.2),
        "One eye hardly closes (facial palsy)": dict(left_rise=0.6, right_rise=0.04),
    }
    out = {}
    for name, kw in groups.items():
        default_hits = personal_hits = default_false = personal_false = total = 0
        signals = []
        for k in range(n_people):
            res = analyze_blinks(blink_recording(Lids(k, **kw)))
            rows, _ = usage(Lids(500 + k, **kw), double_blinks=10, single_blinks=15)
            n_true = 10
            total += n_true
            d = count_double_blinks(BlinkConfig(), rows, "both")
            p_cfg = BlinkConfig(sensitivity=res["sensitivity"], min_threshold=res["min_threshold"],
                                double_gap_ms=res["double_gap_ms"], max_closed_ms=res["max_closed_ms"])
            p = count_double_blinks(p_cfg, rows, res.get("signal", "both"))
            default_hits += min(d, n_true)
            personal_hits += min(p, n_true)
            default_false += max(0, d - n_true)
            personal_false += max(0, p - n_true)
            signals.append(res.get("signal"))
        out[name] = {"default": default_hits / total, "personal": personal_hits / total,
                     "default_false": default_false, "personal_false": personal_false,
                     "signals": {s: signals.count(s) for s in sorted(set(signals))}, "n": n_people}
    return out


def wink_test_recording(lids: Lids):
    lids.rest(60, "rest")
    lids.wink("left", 75, "left")
    lids.rest(45, "rest")
    lids.wink("right", 75, "right")
    lids.rest(45, "rest")
    return list(lids.rows)


def run_winks(cfg, rows):
    """Times and eyes of the presses (wink_start events)."""
    d = WinkDetector(cfg)
    found = []
    for t, l, r, _ in rows:
        for e in d.update(t, l, r):
            if e.type == "wink_start":
                found.append((e.eye, t))
    return found


def score_winks(found, marks):
    """(caught, wrong eye, false presses): a press counts for the wink it falls in."""
    caught = wrong = false = 0
    used = set()
    for eye, t in found:
        hit = next((i for i, (_, t0, t1) in enumerate(marks) if t0 <= t <= t1 + 0.2), None)
        if hit is None or hit in used:
            false += 1
        elif marks[hit][0] != eye:
            wrong += 1
            used.add(hit)
        else:
            caught += 1
            used.add(hit)
    return caught, wrong, false


def bench_winks(n_people: int) -> dict:
    groups = {
        "Typical winks": dict(),
        "Gentle winks (closed eye reaches 0.48)": dict(wink=(0.48, 0.30)),
        "Squints the other eye while winking": dict(wink=(0.85, 0.48)),
        "Drooping left lid (rests at 0.5)": dict(base=(0.5, 0.2)),
    }
    out = {}
    for name, kw in groups.items():
        stats = {"default": [0, 0, 0], "personal": [0, 0, 0]}  # hits, wrong eye, false presses
        total = 0
        for k in range(n_people):
            res = analyze_winks(wink_test_recording(Lids(k, **kw)))
            rows, marks = usage(Lids(700 + k, **kw), double_blinks=8, single_blinks=12, winks=5)
            total += len(marks)
            for label, cfg in (("default", None), ("personal", wink_config({"wink": res}))):
                caught, wrong, false = score_winks(run_winks(cfg, rows), marks)
                s = stats[label]
                s[0] += caught
                s[1] += wrong
                s[2] += false
        out[name] = {k: {"hit_rate": v[0] / total, "wrong_eye": v[1], "false": v[2]} for k, v in stats.items()}
        out[name]["n_winks"] = total
    return out


# ---------------------------------------------------------------------------
# A/B experiments
# ---------------------------------------------------------------------------

def bench_ab(n_sims: int = 200) -> dict:
    """How often the experiment adopts the truly better arm, and how often a non-existent difference."""
    rng = np.random.default_rng(0)
    arms = ["current", "smoother", "snappier"]

    def round_trials(medians):
        trials = []
        for arm in arms:
            for _ in range(4):
                trials.append({"arm": arm, "time_ms": medians[arm] * float(np.exp(0.35 * rng.standard_normal()))})
        return trials

    out = {}
    for name, medians in (("One arm 25 % faster", {"current": 2000, "smoother": 1500, "snappier": 2000}),
                          ("No real difference", {"current": 2000, "smoother": 2000, "snappier": 2000})):
        adopted_right = adopted_wrong = 0
        rounds_needed = []
        for _ in range(n_sims):
            trials = []
            for r in range(1, 5):  # up to 4 rounds of 12 targets
                trials += round_trials(medians)
                res = analyze_experiment(trials, arms)
                if res["decision"] == "adopt":
                    if medians[res["best"]] < medians["current"]:
                        adopted_right += 1
                        rounds_needed.append(r)
                    else:
                        adopted_wrong += 1
                    break
        out[name] = {"right": adopted_right / n_sims, "wrong": adopted_wrong / n_sims,
                     "rounds": med(rounds_needed) if rounds_needed else None}
    return out


# ---------------------------------------------------------------------------

def report(results: dict, seconds: float) -> str:
    L = []
    w = L.append
    w("# Personalisation benchmark (simulated people)")
    w("")
    w("Produced by `python tools/benchmark.py`. **These are simulations**: virtual eyes from "
      "`tests/synthetic.py` and synthetic eyelid traces, not measurements on real people. They check that each "
      "mechanism does what it should and show the size of the effect in simulation; on a real person the same "
      "procedures run on that person's own data.")
    w("")
    ms = results["model_search"]
    w("## Gaze network: per-person model search")
    w("")
    w(f"{ms['n']} simulated people; calibration plus 24 practice hits (the data fine-tuning works with). Error of "
      "the average prediction over a short fixation at new screen positions (1920×1080 screen).")
    w("")
    w("| Model | Median error |")
    w("| --- | --- |")
    w(f"| Default (32×16 network, weight decay chosen by cross-validation) | {fmt(ms['default_px'], ' px')} |")
    w(f"| Per-person search (3 sizes × 3 weight decays + linear) | {fmt(ms['search_px'], ' px')} |")
    w("")
    w(f"The search beat the default for {round(100 * ms['search_better'])} % of people; architectures chosen: "
      f"{', '.join(ms['configs'])}. On these simulated eyes the wider search brings no clear gain (their "
      "features are close to linear in the gaze position); it only replaces the current network when it wins on "
      "the person's own held-out recent clicks, so it costs training time but cannot make the cursor worse.")
    w("")
    w("## Fine-tuning while the site is used (champion / challenger)")
    w("")
    w("Calibrated in one position, then the practice hits / clicks of one session. A challenger replaces the "
      "current network only if it wins on the most recent, held-out hits.")
    w("")
    w("| Situation | Accepted | Refused (unreliable data) | Error before → after (accepted) | Made worse |")
    w("| --- | --- | --- | --- | --- |")
    for name, r in results["finetune"].items():
        ba = f"{fmt(r['before_px'], ' px')} → {fmt(r['after_px'], ' px')}" if r["accepted"] else "–"
        w(f"| {name} | {r['accepted']}/{r['n']} | {r['refused']}/{r['n']} | {ba} | {r['harmed']} |")
    w("")
    ey = results["eyes"]
    w("## One-eye networks")
    w("")
    w(f"{ey['n']} simulated people. While one eye is closed (a wink held to drag), its features say little:")
    w("")
    w("| Cursor during a left-eye wink | Median error |")
    w("| --- | --- |")
    w(f"| (both eyes open, for reference) | {fmt(ey['normal_px'], ' px')} |")
    w(f"| Two-eye network | {fmt(ey['wink_both_px'], ' px')} |")
    w(f"| Right-eye network, aligned before the wink (what Paralic does) | {fmt(ey['wink_one_px'], ' px')} |")
    w("")
    w(f"A squint that comes and goes (right eye off by ~3° on each fixation): cross-validation made the left eye "
      f"lead for {ey['squint_left_picked']}/{ey['n']} people (and kept both eyes for "
      f"{ey['typical_both_picked']}/{ey['n']} people without a squint). Error with the chosen network "
      f"{fmt(ey['squint_default_px'], ' px')} vs {fmt(ey['squint_both_px'], ' px')} with both eyes.")
    w("")
    w("## Cursor smoothing tuned to each person's jitter")
    w("")
    w("| Measured jitter | Tuned level (0–10) | Expected jitter | 90 % settle after a jump | (fixed *medium*) |")
    w("| --- | --- | --- | --- | --- |")
    for r in results["smoothing"]:
        w(f"| {r['raw']} px | {r['level']:.1f} | {r['jitter']:.0f} px | {r['settle']:.0f} ms | {r['medium_settle']:.0f} ms |")
    w("")
    w("Steadier eyes get less smoothing, so a quicker cursor; jittery ones get more.")
    w("")
    w("## Double blinks: standard vs personal thresholds")
    w("")
    w("Usage: 10 double blinks, 15 natural single blinks and 5 s of reading the bottom of the screen per person.")
    w("")
    w("| People | Caught (standard) | Caught (personal) | False double blinks (std / personal) | Signal chosen |")
    w("| --- | --- | --- | --- | --- |")
    for name, r in results["blinks"].items():
        sig = ", ".join(f"{k} ×{v}" for k, v in r["signals"].items())
        w(f"| {name} | {round(100 * r['default'])} % | {round(100 * r['personal'])} % | "
          f"{r['default_false']} / {r['personal_false']} | {sig} |")
    w("")
    w("## Held winks (press & drag): standard vs personal thresholds")
    w("")
    w("Usage: held winks of each eye mixed with double blinks, single blinks and looking down.")
    w("")
    w("| People | Presses caught (std / personal) | Wrong eye | False presses (std / personal) |")
    w("| --- | --- | --- | --- |")
    for name, r in results["winks"].items():
        d, p = r["default"], r["personal"]
        w(f"| {name} | {round(100 * d['hit_rate'])} % / {round(100 * p['hit_rate'])} % | "
          f"{d['wrong_eye']} / {p['wrong_eye']} | {d['false']} / {p['false']} |")
    w("")
    w("## A/B experiments")
    w("")
    w("Rounds of 12 targets (4 per variant), log-normal target times (σ = 0.35), up to 4 rounds.")
    w("")
    w("| Truth | Adopted the better variant | Adopted a variant wrongly | Rounds needed (median) |")
    w("| --- | --- | --- | --- |")
    for name, r in results["ab"].items():
        w(f"| {name} | {round(100 * r['right'])} % | {round(100 * r['wrong'])} % | {fmt(r['rounds'])} |")
    w("")
    w(f"_Run time {seconds:.0f} s._")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--people", type=int, default=8, help="simulated people per experiment (default 8)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "benchmark.md"))
    args = ap.parse_args(argv)
    started = time.perf_counter()
    results = {}
    steps = [
        ("model_search", lambda: bench_model_search(args.people)),
        ("finetune", lambda: bench_finetune(args.people)),
        ("eyes", lambda: bench_eyes(args.people)),
        ("smoothing", bench_smoothing),
        ("blinks", lambda: bench_blinks(max(args.people, 8))),
        ("winks", lambda: bench_winks(max(args.people, 8))),
        ("ab", bench_ab),
    ]
    for name, fn in steps:
        t0 = time.perf_counter()
        results[name] = fn()
        print(f"{name}: {time.perf_counter() - t0:.0f} s", flush=True)
    text = report(results, time.perf_counter() - started)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
