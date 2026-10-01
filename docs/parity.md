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
| Feature vector: 20 columns, same order and meaning (`r_dx … tz`) | `features.py` `FEATURE_NAMES` | `Features.swift` `FeatureLayout` | ✅ same layout |
| Head-pose columns 14–19 | `HEAD_FEATURE_IDX` | `headPoseColumns` | ✅ |
| One-eye inputs: right = 0,1,4,10–13 + head; left = 2,3,5,6–9 + head | `EYE_INPUTS` | (`dropColumns` can express them) | ⏳ macOS has no one-eye networks yet |
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
* the keys (Shift, ".", arrows) do not clash with existing ones: C / A / P / Esc for helpers and
  B / Q / E for the eyelids in mouse demo mode;
* the held-out positives are drawn per frame, and frames from one Shift-hold are near duplicates, so the
  held-out error is optimistic — split by hold (group) instead, as calibration and fine-tuning do.
