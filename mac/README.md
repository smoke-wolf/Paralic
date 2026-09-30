# Paralic for macOS — native app

A native Swift/SwiftUI rewrite of Paralic for accessibility: **no Python**, the
camera via **AVFoundation**, face/eye tracking via the **Vision** framework, and
a **native GazeNet** (Accelerate) for personalized gaze estimation.

> macOS has no ARKit face tracking (that is iOS + TrueDepth only), so the native
> tracker uses `VNDetectFaceLandmarksRequest` — pupils, eye contours and
> roll/yaw/pitch — and derives the same geometric features the Python used.

## Status

| Area | File(s) | State |
| --- | --- | --- |
| Linear algebra (BLAS) | `ML/Linalg.swift` | ✅ done |
| GazeNet (MLP + ridge skip, Adam, Huber, grouped CV, ensemble, affine correction) | `ML/GazeNet.swift` | ✅ done, verified |
| **A/B personalization** (per-user model bake-off by grouped CV) | `ML/ABSelection.swift` | ✅ done, verified |
| Fine-grained responsiveness (Speed↔Stability, saccade bypass, velocity lead, dead-zone) | `Model/Settings.swift`, `Tracking/OneEuroFilter.swift` | ✅ done |
| Blink / double-blink detector (adaptive thresholds) | `Tracking/BlinkDetector.swift` | ✅ done |
| Vision feature extraction (20-number vector) | `Tracking/Features.swift` | ✅ compiles against Vision |
| Camera capture | `Camera/` | ⏳ next |
| Face tracker (Vision request pipeline) | `Tracking/FaceTracker.swift` | ⏳ next |
| Gaze control (cursor, snapping, click, scroll, pause) | `Control/` | ⏳ next |
| Calibration flow | `Calibration/` | ⏳ next |
| SwiftUI pages (Home/Explore/Read/Talk/Practice/Settings/Help) | `UI/` | ⏳ next |

## Prove the personalization works (no camera needed)

```bash
cd mac
./selftest.sh
```

Synthesises a realistic per-user calibration (13 dots, head-pose-coupled iris
geometry + per-frame noise), runs the A/B bake-off, and reports accuracy in cm.
Representative run:

```
A/B model bake-off (grouped cross-validation — whole dots held out):
  config        CV error    train err
  small         0.58 cm     20 px       ◀ chosen
  standard      0.58 cm     20 px
  large         0.58 cm     20 px
  linear        0.59 cm     20 px
  no-head-pose  0.59 cm     21 px

fresh-gaze holdout (200 random points not in calibration):
  mean error: 0.58 cm   90th pct: 1.00 cm
✅ PASS — personalization is accurate, A/B selection works, profile persists.
```

## How personalization stays accurate

* **A/B selection** — five candidate models (linear, small/standard/large MLPs,
  and a head-pose-free variant) compete on *your* calibration data, scored by
  **grouped** cross-validation (whole calibration dots held out, so the score is
  true interpolation, not memorisation). The winner is kept; the comparison is
  surfaced in Settings.
* **Grouped CV also picks the L2** weight decay, and the winner is an ensemble of
  three networks averaged.
* **Fast *and* precise** — a low One-Euro cutoff holds the cursor steady while you
  fixate (plus a dead-zone that swallows sub-pixel jitter), a **saccade detector**
  blends straight to the raw estimate the instant your eyes jump, and a small
  **velocity lead** hides end-to-end latency. All exposed as a single Speed dial
  or individual knobs (`Model/Settings.swift`).

## Building the app

Full Xcode is required to build the `.app` (Command Line Tools alone cannot):

```bash
cd mac
xcodegen generate         # writes Paralic.xcodeproj from project.yml
open Paralic.xcodeproj     # build & run in Xcode (grants the camera permission)
```

The self-test compiles with Command Line Tools alone (`swiftc`), which is why it
is the verification path in CI / headless environments.

## License

Proprietary — All Rights Reserved. See the repository `LICENSE`.
