# Keeping the web app and the macOS app in step

Paralic has two implementations that are developed side by side on the same branch:

* **web** — Python server (`paralic/`) + browser front end (`web/`), MediaPipe FaceLandmarker;
* **macOS** — native Swift app (`mac/`), Apple Vision + Accelerate.

This page lists what must stay the same, what the web side does that the macOS side does not do yet,
and issues found when the two were compared (October 2026). Update it when either side changes one of
these behaviours.

## Contracts both sides follow

| Contract | Web | macOS | Status |
| --- | --- | --- | --- |
| Feature vector: columns 0–19 in the same order and meaning (`r_dx … tz`) | `features.py` `FEATURE_NAMES` | `Features.swift` `FeatureLayout` | ✅ same layout for 0–19 |
| Columns 20–27 (full mesh): `r_vlid`, `l_vlid` (iris height between the lids, 0 top … 1 bottom, 0.5 when the lid opening is under 4 % of the eye width, clipped to −0.5…1.5), `r_tilt`, `l_tilt`, squint and wide blendshapes | `features.py` (`FEATURE_VERSION = 2`) | — | ⏳ macOS has 20 columns; its `Features.swift` header still says it mirrors `features.py` |
| Head-pose columns 14–19 | `HEAD_FEATURE_IDX` | `headPoseColumns` | ✅ |
| One-eye inputs: right = 0,1,4,10–13,20,22,24,26 + head; left = 2,3,5,6–9,21,23,25,27 + head | `EYE_INPUTS` | (`dropColumns` can express them) | ⏳ macOS has no one-eye networks yet |
| "left" means the **person's own** left eye | MediaPipe convention, checked by `tests/test_mediapipe.py::test_closing_one_eye_raises_only_that_eyes_closure` | Vision `leftEye` / `rightEye` | ⚠️ verify on a Mac: close one eye and check which contour's aperture drops before using it for winks |
| GazeNet: 32×16 tanh MLP + ridge-initialised linear skip, Huber, Adam with cosine decay, grouped CV over L2 {1e-3, 1e-2, 1e-1}, ensemble of 3, affine correction clamped to singular values 0.6–1.6 | `gazenet.py` | `GazeNet.swift` | ✅ same algorithm |
| Blink detector: adaptive median baseline, `t_close = b + sensitivity·(1−b)`, double blink by gap | `blink.py` | `BlinkDetector.swift` | ✅ core identical (see gaps) |

Values differ even where the layout matches: Vision gives no blendshapes (the macOS "look" columns are
derived from iris geometry), head position is in face-box units instead of centimetres, and Vision's
normalised coordinates point **up** (so `dy` and the look-up/down columns have the opposite sign to the
web's). Each person's network is trained on its own platform's features, so this is fine for gaze, but
**profiles and models cannot be moved between the two apps**, and any rule that relies on the sign of
`dy` (e.g. "lids drop when looking down") has to be written per platform.

## Behaviours the web side has that macOS does not yet

1. **Blink signal**: the blink detector watches the *more open* eye (`min(left, right)`), so a wink is
   never a blink; the blink test may switch a person to the average or to one eye (facial palsy).
   macOS averages both eyes, so a held wink would count as a long blink and two quick winks as a click.
2. **Personal blink thresholds** from the prompted blink test (`personalize.analyze_blinks`):
   sensitivity, `min_threshold` (macOS hard-codes 0.30), double-blink gap, longest blink.
3. **Long close** (both eyes deeply closed 1–6 s; `blink.py`) and the stabiliser **holding** the cursor
   for the whole deep closure; the baseline does not learn from deep closures.
4. **Winks** (`gestures.py` `WinkDetector`): per-eye adaptive baselines; an eye must be seen open before
   it can press; both eyes' baselines learn together (one moving alone faked winks when looking down
   with a drooping lid); a lid that stays low for 2 s without a wink is that eye's new normal.
5. **One-eye networks** trained with every calibration, the open eye's network taking over during a
   wink with an alignment offset computed on the frames before it, and the leading network chosen by
   cross-validation (a single eye leads only when it is ≥ 10 % and ≥ 10 px better).
6. **Stabiliser**: on a blink the cursor rewinds to where it was ~100 ms before the eyes started closing
   (`rewind_s`); a non-deep closure releases the freeze after 0.9 s. macOS freezes at the last output.
7. **Per-person tuning**: smoothing tuned to the measured jitter (`tune_smoothing`), button magnet sized
   from the measured accuracy (`recommend_magnet`: 1.1 × error + 20 px, 50–240 px). The web magnet
   presets are 5 % (normal) and 8.5 % (strong) of the screen diagonal; macOS uses 10–20 %.
8. **Learning from use**: champion/challenger fine-tuning on practice hits and clicks, and live, blind
   A/B experiments (Lab). The macOS "A/B" is the calibration-time model bake-off (the web's equivalent
   is the model search run during fine-tuning).
9. **Interaction**: dwell click, held-wink press/drag/drop, the gaze menu (long press), drag lock,
   per-person gesture settings — not built on macOS yet (its `Control/` and `UI/` are still to come).
10. **Calibration labels** (`calibration.py`): a dot stays up until the server's `settled` count (frames
    of the last steady stretch, `SettleTracker`) reaches its target, at most 4 s, shortened after two
    timeouts in a row. Training splits each dot into steady stretches (`segments`: change points of the
    mean of columns 0–3, 20, 21 in units of the noise pooled over the dots; split when Σ t² ≥ 30), drops
    stretches still on the previous dot (Σ d² ≤ 25) and keeps the stretch that ended the recording, minus
    its first 3 frames. Runs of dots start in the centre (where the countdown is). Head-movement groups
    share a total weight of 9 (each dot weighs 1). In simulation this keeps typical eyes as accurate in about
    half the time on the dots and makes slow or glancing eyes much more accurate (`docs/benchmark.md`).
11. **Setup and quick adjust**: a seating-position check before calibrating (distance, centring, light;
    `web/js/position.js`; the frame message carries `pos` and, on request, `light`) that guides the person
    back to the calibrated pose (`calibrated_pose`) before a quick adjust; the quick adjust is 9 dots with
    the eyes alone.
12. **Improving rounds**: after the accuracy check, while it is below *good* (< 5.5 % of the screen diagonal),
    up to 3 rounds of 9 extra dots chosen where the measured error is largest, a refit, and 5 fresh measuring
    dots on which the previous and the new network are compared (`calibration_start` mode `refine`,
    `validation_finish`); the loser's extra dots are dropped. Dots the eyes were not on are left out of every
    fit (`suspect_dots`: leave-one-dot-out ridge residual > 3 × median and > 120 px, at most 25 %).
13. **Face print** (`faceprint.py`): per-person views of the face (47 rigid mesh points in a face-fixed frame
    + LBP texture of the eye-aligned face), compared with a cross-validated learned metric (within-person
    whitening, discriminant directions, channel weights); recognition at start-up, "Is that NAME?" when
    someone else sits down, new views learned only while the camera keeps following a confirmed face. macOS
    has no face print; Vision's 76-point landmarks would need their own rigid-point list.
14. **Cursor motion** (`web/js/motion.js`, per person): critically damped spring glide (ω = 9.5 / 17 / 30),
    "hold still while you look" (a running, then 0.6 s moving, average per fixation; a new fixation after
    two samples outside the radius), and the head nudge (joystick past a dead zone, signs learned by a
    head-direction check, the gaze point held still while nudging).
15. **Hand mode** (`hands.py`, `hand_gestures.py`, `hand_control.py`, `web/js/hand-calibration.js`): the
    index fingertip is the cursor (through a per-person quadratic pointing map fitted on 13 dots labelled in
    screen fractions), a pinch clicks at the cursor of the last frame before the fingers started to close
    (thresholds a quarter and half of the way from the person's closed pinch to their open hand, found by
    an Otsu split of a pinch cycle), pinch-drag scrolls by palm-centre travel in palm widths, and a held
    "stop" hand (four fingers spread, thumb out) asks the page to toggle pause once per gesture. It runs in
    the same `TrackerSession` (`mode="hand"`), so people, settings and desktop control are shared; each
    person's setup is `users/<id>/hand.json`. macOS has no hand mode yet (Vision's hand pose request would
    give the same 21 points).

16. **Someone else in view** (`face_select.py`): FaceLandmarker reports up to 3 faces; the person being
    followed keeps control (the face nearest its last box, within 1 face width per frame); when lost while others
    are or were in view in the last 30 s, only their face print (score ≤ 4.5) or their place (within 1.5 face widths,
    for 10 s) gives control back; otherwise nobody controls the cursor (the largest face after that, or 3 s later
    when a face print recognises nobody). Hand mode follows the hand in control (palm-width continuity).
17. **Glasses** (`glasses.py`): bridge and rim edges against the skin's own edges (glasses on above 2.1, off below
    1.4, settled after 3 s), lens glare as the largest bright colourless blob over an eye (on above 0.5); glare on
    one lens hands the cursor to the other eye's network (aligned like a wink); calibrations are kept per glasses
    state (`profile.json` / `profile-glasses.json`).
18. **Recordings and diagnosis** (`recorder.py`, `recording.py`, `inspector/`, `diagnose.py`): a session's
    frames, events, landmarks, models and images in the documented format (`docs/recording-format.md`); the
    diagnosis splits each accuracy check into the shared shift, the scatter between dots and the noise floor
    (jitter / √frames per dot).

## Issues found in the macOS code

* **`ABSelection.autoTrain` ignores each candidate's `hidden` size and `l2Grid`.** `GazeNet.train`
  always builds `HIDDEN = (32, 16)` and searches `L2_GRID`, so "small", "standard" and "large" are the same
  network trained for 500 / 600 / 800 iterations — which is why the self-test prints the same 0.58 cm for
  all three. Pass `hidden` and the L2 grid into `GazeNet.train` and use them in `fitEval` and in the final
  ensemble.
* **Selection by 3-fold grouped CV leans towards small and linear models** (holding out a third of the
  13 dots is mostly extrapolation). On the web side the wider search gave no measurable gain on simulated
  people (`docs/benchmark.md`) and often picked the linear model; the web keeps the standard network after
  calibration and only lets a searched model replace it when it wins on the person's own held-out recent
  clicks. Consider the same rule.
* **The velocity lead uses the raw frame-to-frame velocity**, which amplifies jitter (noise ÷ frame time ×
  lead). Use the One Euro filter's smoothed derivative instead.
* `AffineCorrection.clampSingularValues` matches NumPy's SVD (to 1e-15) whenever det(A) > 0; it takes
  `|q − r|` and so loses a reflection when det(A) < 0. Real corrections are near the identity, so this
  is harmless, but worth a comment or an assert.
* `GazeNet.train` computes `foldSets` and discards it (dead code).

## Notes for wiring the manual (contrastive) mode into the web session

`paralic/contrastive.py` tunes the network that drives the cursor — pass
`eye=session._preferred_eye()` so a forced *tracking eye* is respected. When connecting it:

* use only frames with both eyes open (`_FrameRecord.steady`): during blinks and winks the features say
  little about the gaze;
* run it like `_start_finetune` — one background job at a time, publish under `_lock`, then
  `_record_model("manual-contrastive")`, `_save_profile()`, `_save_personal()` and push a result message;
* the keys (Shift, ".", arrows) do not clash with existing ones: C / A / P / Esc and Shift+R (recording)
  for helpers and B / Q / E for the eyelids in mouse demo mode;
* the held-out positives are drawn per frame, and frames from one Shift-hold are near duplicates, so the
  held-out error is optimistic — split by hold (group) instead, as calibration and fine-tuning do.
