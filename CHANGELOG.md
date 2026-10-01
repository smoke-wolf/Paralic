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

### Glasses
- Paralic notices glasses, and reflections on their lenses, on every frame.
  Each person keeps one calibration made with glasses and one without. The
  one that fits is loaded; if only the other exists, Paralic suggests a quick
  adjust. Putting glasses on or taking them off mid-session brings up an offer
  to switch calibration.
- When a reflection hides one eye, the cursor follows the other eye without
  jumping. Calibration leaves out frames hit by a passing reflection, and the
  seating check and the dots say how to get rid of a lasting one.

### Someone else in view
- With several people in front of the camera, only the person being followed
  controls Paralic. The camera view outlines "You" and "Ignored". When you look
  away or step out, nobody else takes over: Paralic waits until your face print
  or your place says you are back. In hand mode, the hand in control keeps
  control when another hand appears.

### Desktop control (macOS, opt-in)
- Move the real mouse cursor across the whole computer with your eyes or
  hand, and click with a double blink or a pinch. To stop, rest the cursor in
  the top-left corner for a second, or turn it off in Settings.

### Recording sessions
- **● Rec** in the top bar (or Shift+R) records a session for the **Paralic
  Inspector**: the configuration, every frame's measurements and what the
  pipeline made of them, the face-mesh or hand points, commands and events,
  the gaze network whenever it changes, and camera images.
  `python -m paralic --record` records every session. Recordings stay on this
  computer in `data/recordings/`; the format is in
  `docs/recording-format.md`.
- The **Paralic Inspector** (`python -m paralic.inspector`) replays a
  recording frame by frame:
  - the camera with the face mesh, irises and head pose (or the hand
    skeleton), and the faces it ignored;
  - the screen map with gaze, cursor and calibration targets;
  - linked charts of eye closure against the blink thresholds, gaze, head
    pose, glare per lens and pinch distance;
  - "under the hood": every hidden layer of every gaze network for the
    current frame, the inputs that drive it most, and the blink and wink
    state machines;
  - each calibration dot by dot, every event, and the configuration over
    time.

### The website
- Explore the Solar System, read articles, and use *Talk*: 175 spoken phrases
  in 14 groups with natural voices, plus an eye-typing keyboard. Its word
  prediction covers about 1,600 words and suggests the next word after 131
  common words. It also learns the words and word pairs you use most.
- **Trail Shooter:** walk a preset trail at night and stop the creatures
  before they reach you. Look at one and blink to shoot; with a hand, point and
  pinch. There are four levels, from *Forest path* to *Night ridge*.
- Also a drag-and-drop game, a drawing canvas, target practice, Settings and
  Help.
- On macOS a small launcher window starts Paralic and has a Quit button.
  Calibration runs full screen.

### Upgrading from the preview on `main`
- Saved calibrations load automatically. Ones from before the full face-mesh
  features keep working; Paralic suggests a new full calibration for the best
  accuracy.
- A single `data/profile.json` moves into the first person's folder. The
  shared `data/hand_profile.json` of the hand-mode preview becomes the active
  person's hand setup.
