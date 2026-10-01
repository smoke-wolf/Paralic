# Personalisation benchmark (simulated people)

Produced by `python tools/benchmark.py`. **These are simulations**: virtual eyes from `tests/synthetic.py` and synthetic eyelid traces, not measurements on real people. They check that each mechanism does what it should and show the size of the effect in simulation; on a real person the same procedures run on that person's own data.

## Gaze network: per-person model search

8 simulated people; calibration plus 24 practice hits (the data fine-tuning works with). Error of the average prediction over a short fixation at new screen positions (1920×1080 screen).

| Model | Median error |
| --- | --- |
| Default (32×16 network, weight decay chosen by cross-validation) | 19 px |
| Per-person search (3 sizes × 3 weight decays + linear) | 18 px |

The search beat the default for 38 % of people; architectures chosen: 32x16 l2=0.01, 32x16 l2=0.1, linear. On these simulated eyes the wider search brings no clear gain (their features are close to linear in the gaze position); it only replaces the current network when it wins on the person's own held-out recent clicks, so it costs training time but cannot make the cursor worse.

## Fine-tuning while the site is used (champion / challenger)

Calibrated in one position, then the practice hits / clicks of one session. A challenger replaces the current network only if it wins on the most recent, held-out hits.

| Situation | Accepted | Refused (unreliable data) | Error before → after (accepted) | Made worse |
| --- | --- | --- | --- | --- |
| Moved in the chair (24 practice hits) | 8/8 | 0/8 | 134 px → 33 px | 0 |
| Moved, 20 % of labels wrong | 8/8 | 0/8 | 134 px → 52 px | 0 |
| Same position as calibration | 2/8 | 0/8 | 19 px → 18 px | 1 |
| All labels wrong | 0/8 | 8/8 | – | 0 |

## One-eye networks

8 simulated people. While one eye is closed (a wink held to drag), its features say little:

| Cursor during a left-eye wink | Median error |
| --- | --- |
| (both eyes open, for reference) | 19 px |
| Two-eye network | 227 px |
| Right-eye network, aligned before the wink (what Paralic does) | 30 px |

A squint that comes and goes (right eye off by ~3° on each fixation): cross-validation made the left eye lead for 7/8 people (and kept both eyes for 8/8 people without a squint). Error with the chosen network 29 px vs 34 px with both eyes.

## Cursor smoothing tuned to each person's jitter

| Measured jitter | Tuned level (0–10) | Expected jitter | 90 % settle after a jump | (fixed *medium*) |
| --- | --- | --- | --- | --- |
| 20 px | 0.0 | 6 px | 67 ms | 133 ms |
| 40 px | 0.0 | 13 px | 67 ms | 133 ms |
| 70 px | 7.5 | 14 px | 200 ms | 133 ms |
| 110 px | 10.0 | 19 px | 267 ms | 133 ms |

Steadier eyes get less smoothing, so a quicker cursor; jittery ones get more.

## Double blinks: standard vs personal thresholds

Usage: 10 double blinks, 15 natural single blinks and 5 s of reading the bottom of the screen per person.

| People | Caught (standard) | Caught (personal) | False double blinks (std / personal) | Signal chosen |
| --- | --- | --- | --- | --- |
| Typical blinks | 100 % | 100 % | 0 / 0 | both ×8 |
| Light blinks (closure rises only 0.2) | 0 % | 100 % | 0 / 0 | both ×8 |
| One eye hardly closes (facial palsy) | 0 % | 100 % | 0 / 0 | mean ×8 |

## Held winks (press & drag): standard vs personal thresholds

Usage: held winks of each eye mixed with double blinks, single blinks and looking down.

| People | Presses caught (std / personal) | Wrong eye | False presses (std / personal) |
| --- | --- | --- | --- |
| Typical winks | 100 % / 100 % | 0 / 0 | 0 / 0 |
| Gentle winks (closed eye reaches 0.48) | 0 % / 100 % | 0 / 0 | 0 / 0 |
| Squints the other eye while winking | 100 % / 100 % | 0 / 0 | 0 / 0 |
| Drooping left lid (rests at 0.5) | 100 % / 100 % | 0 / 0 | 0 / 0 |

## A/B experiments

Rounds of 12 targets (4 per variant), log-normal target times (σ = 0.35), up to 4 rounds.

| Truth | Adopted the better variant | Adopted a variant wrongly | Rounds needed (median) |
| --- | --- | --- | --- |
| One arm 25 % faster | 68 % | 0 % | 3 |
| No real difference | 0 % | 8 % | – |

_Run time 338 s._
