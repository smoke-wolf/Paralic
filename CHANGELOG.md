# Changelog

## 1.0.0 — 2026-10-01

The first release: everything below works together in one application, with
your eyes or with your hand.

### Two ways to control Paralic
- **Eyes.** Look to move the cursor (it snaps to the nearest button), blink
  twice to click, look at the arrows on the right edge to scroll. Hold one eye
  closed to press and drag, keep still for a right-click menu, or use dwell
  click or a long close instead of blinking.
- **Hand.** Point with your index finger to move, pinch to click, and pinch
  and move to scroll. Hold up an open hand to pause and resume. A short spoken
  hand setup learns your hand size, your comfortable pointing range and your
  own pinch, and *Quick re-point* redoes only the pointing dots.
- One start screen offers both. Paralic remembers the choice, and you can
  switch any time in **Settings → Control with**. People, personal settings
  and desktop control are shared between the two modes.

### Accuracy
- Calibration starts with a seating and lighting check. It then uses only the
  frames where your eyes actually rested on each dot, leaves out dots you were
  not looking at, includes head-movement steps and a smooth-pursuit sweep, and
  speaks every step.
- **Improve until good:** if the accuracy check comes out below good, Paralic
  offers extra rounds with fresh dots. It keeps whichever network measures
  better on dots neither has seen.
- **Quick adjust** (9 dots, about 20 seconds) and a **mouse tune-up** for
  sessions where a helper moves the mouse.
- **Smooth cursor motion:** the cursor glides instead of jumping and holds
  still while you look at something. **Head-tilt nudge** moves it a little
  with small head tilts for fine positioning.

### People and personalisation
- Several people can use one computer. Each has their own gaze network, blink
  and wink thresholds, smoothing, button magnet and hand setup.
- **Face print:** Paralic recognises who sits down at the camera and offers to
  switch to them. It learns new views of a face only while it keeps tracking
  that same person.
- Paralic keeps learning while you use it. The moments before your clicks
  become training data, and a background job keeps whichever gaze network
  predicts you best. The Lab runs A/B experiments to tune your settings.

### Desktop control (macOS, opt-in)
- Move the real mouse cursor across the whole computer with your eyes or
  hand, and click with a double blink or a pinch. To stop, rest the cursor in
  the top-left corner for a second, or turn it off in Settings.

### The website
- Explore the Solar System, read articles, and use *Talk*: spoken phrases with
  natural voices, plus an eye-typing keyboard with word prediction. Also a
  drag-and-drop game, a drawing canvas, target practice, Settings and Help.
- On macOS a small launcher window starts Paralic and has a Quit button.
  Calibration runs full screen.

### Upgrading from the preview on `main`
- Saved calibrations load automatically. Ones from before the full face-mesh
  features keep working; Paralic suggests a new full calibration for the best
  accuracy.
- A single `data/profile.json` moves into the first person's folder. The
  shared `data/hand_profile.json` of the hand-mode preview becomes the active
  person's hand setup.
