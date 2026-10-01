# Paralic — browse the web with your eyes

Paralic is a website that runs on your own computer (`http://localhost`) and can be used **with your eyes only**.
A Python server uses **neural networks** to track your eyes through an ordinary webcam:

* **look** somewhere and the cursor moves there (it gently snaps to the nearest button),
* **blink twice** to click,
* **hold one eye closed** to press and hold — look elsewhere to **drag**, keep still for a **right-click menu**,
* **look at the arrows on the right edge** to scroll,
* **blink twice on Pause** to rest your eyes, and blink twice again to resume.

For people who find blinking or winking hard there are alternatives: **dwell click** (rest your eyes on a
button), **closing both eyes for a second**, and a menu that can pick things up and drop them. Everything is
**personalised per person**: each person who uses the computer gets their own gaze network, blink and wink
thresholds and settings, which keep improving while they use the site.

The site itself is designed for eye control: big targets, a Solar System to explore, articles to read,
a *Talk* page with spoken phrases and an eye-typing keyboard with word prediction, a drag-and-drop game,
a drawing canvas, a target-practice game, and settings you can change with your eyes.

---

## Quick start

You need **Python 3.9 or newer** (3.10–3.12 recommended), a **webcam**, and **Chrome, Edge or Firefox**.

```bash
git clone https://github.com/smoke-wolf/Paralic.git
cd Paralic
./start.sh          # macOS / Linux
start.bat           # Windows (double-click or run in a terminal)
```

The start script creates a virtual environment, installs the dependencies (first run only), starts the
server and opens **http://localhost:8000** in your browser. On first start it also downloads MediaPipe's
face model (3.7 MB) into `models/`.

Prefer doing it by hand?

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m paralic                    # or: python run.py
```

Options: `--port 8080`, `--no-browser`, `--host`, `--model PATH`, `--data-dir DIR`, `--verbose`.
If port 8000 is busy the next free one is used.

> No webcam? Click **Try with a mouse** on the start screen (or open `http://localhost:8000/?demo`).
> The mouse then plays the part of your eyes: press **B** twice quickly for a double blink, hold **B** for a
> second to close both eyes, and hold **Q** / **E** to keep your left / right eye closed.

## Using it

1. **Start eye tracking** – the only click you need. Allow the camera when the browser asks.
   (Once allowed, Paralic starts the camera by itself next time.)
2. **Camera check** – centre your face; the eyes are outlined when they are found.
3. **Blink twice to calibrate** – follow the dot with your eyes for about 30 seconds. When asked, keep
   looking at the centre dot and gently move your head (skip that if moving is hard for you — just keep looking).
   Your personal neural networks are trained, five more dots measure the accuracy, and finally you blink twice
   three times when the dot turns purple, so double blinks are tuned to how *you* blink.
4. **Start browsing** – look at it and blink twice.

| Gesture | What it does |
| --- | --- |
| Look | Moves the cursor. The nearest button lights up. |
| Blink twice | Clicks what was highlighted *just before the first blink*. A small “1” shows after the first blink. |
| Hold one eye closed | Presses and holds where you were looking. Look elsewhere while it stays closed to **drag** (or draw); open it to drop. Keep looking at the same spot for a second for the **menu** (a right click: Click · Pick up to move · Read aloud). Open it again quickly to click. |
| Close both eyes ~1 s *(optional)* | Menu, pick up / drop, or click — for people who cannot close one eye on its own. |
| Rest your eyes on a button *(optional)* | Dwell click: a ring fills on the cursor, then it clicks. |
| Look at *Scroll up / Scroll down* (right edge) | Scrolls; look further towards the arrow to go faster. Blink twice there to jump a page. |
| Blink twice on *Pause* | Pauses (nothing gets clicked). Blink twice to resume. |

Each person's calibration is saved (numbers only, no images) in `data/users/<id>/`. Several people can share
the computer: Paralic asks who is using it. Next time you can use your calibration as is, do a
**Quick adjust** (5 dots, ~8 seconds) or a full calibration. Quick adjust and full calibration are also
on the Home and Settings pages.

**Helper shortcuts** (for someone assisting): `C` full calibration · `A` quick adjust · `P` pause/resume ·
`Esc` cancel calibration / drop a carried item / close the menu · `F11` full screen.

## Accessibility: other ways to click, hold and drag

Paralic is meant for people who cannot use their hands, and eyes differ a lot from person to person —
a drooping lid, a squint, facial palsy, involuntary blinks, fatigue. So there is more than one way to do
everything, chosen per person in **Settings → Eye gestures**:

* **Winks work like a mouse button.** Close one eye and keep it closed (≥ 0.35 s, adjustable): that presses
  where you were looking. Look somewhere else to drag; keep still for a second for the menu (a *long press*,
  i.e. a right click); open it quickly for a click. Each eye can be *press & drag*, *right-click menu* or off.
  On any page a held wink is an ordinary pressed pointer, so dragging and drawing work like with a mouse
  (try **Arrange** and **Draw**).
* **The cursor keeps following you while one eye is closed.** Besides the network that reads both eyes, every
  calibration trains a network for each eye alone; during a wink the open eye's network takes over, aligned to
  the usual one so the cursor does not jump.
* **Test my winks** checks each eye ("close your left eye… now your right eye") and personalises the wink
  thresholds — many people squint the other eye a little when winking, which is fine. An eye that cannot wink
  on its own is ignored, so it never presses by accident.
* **Closing both eyes for about a second** can open the menu, pick up / drop things, or click. It has to be a
  real closure: looking at the bottom of the screen (which also lowers the lids) does not count.
* **Dwell click** clicks by resting the eyes on a button (0.6–2 s) — no blinking at all.
* **Pick up to move** (in the menu) carries an item without holding anything; blink twice, dwell or close your
  eyes again to drop it.
* **Short winks** can click or open the menu (off by default).
* **Lopsided blinks and eyes that don't close fully**: the blink test chooses what to watch — both eyes, their
  average, or one eye (e.g. with facial palsy) — and how lopsided ordinary blinks are, so that winks have to
  be clearly more one-sided than blinks.
* **A squint or an unreliable eye**: cross-validation compares "both eyes" with each eye alone and lets a single
  eye lead when the other one misleads (e.g. a squint that comes and goes). *Tracking eye* in Settings can also
  force one eye (an eye patch, a prosthetic eye).

## Personalisation

* **People** — each person has their own profile (`data/users/<id>/`): gaze networks, calibration data, blink
  and wink thresholds, gestures, smoothing, magnet and experiment results.
* **Learning from use** — the moments before you pop a practice target or click a button become labelled
  training data. In the background a *challenger* network (the current one fine-tuned, and a fresh per-person
  model search over several architectures) must beat the current *champion* on your most recent, held-out
  clicks (paired sign-flip test, capped errors, ≥ 50 % win rate) before it replaces it; implausible labels are
  dropped first.
* **Auto settings** — cursor smoothing is tuned to your measured jitter, the button magnet to your accuracy.
* **Personalization Lab** (`#/lab`) — shows what was learned and runs **blind A/B experiments** (smoothing,
  magnet, double-blink timing, dwell time): each round is a shuffled, balanced set of targets where every
  target secretly uses one variant; a permutation test decides, and a clear winner becomes your default.

**Fully hands-free:** after the camera permission has been granted once, the site starts tracking without a
click. For a kiosk-style setup, launch the browser in full screen, e.g.
`chrome --kiosk --autoplay-policy=no-user-gesture-required http://localhost:8000` (the autoplay flag lets the
*Talk* page speak without a first click).

## Hand mode (finger gestures)

Prefer your hands? On the start screen choose **Use your hands** (or open
`http://localhost:8000/?hands`). A second neural network — MediaPipe
HandLandmarker — tracks your hand through the same webcam, and finger gestures do
everything:

| Gesture | What it does |
| --- | --- |
| Point your index finger | Moves the cursor (it still snaps to the nearest button). |
| Pinch (thumb + index) | Clicks the highlighted button. |
| Pinch and move up / down | Scrolls the page. |
| Hold an open palm to the camera | Pauses (and open palm again to resume). |

Thresholds scale with your hand size, so it works at any distance from the
camera. Hand mode runs as a completely separate pipeline from eye tracking
(`paralic/hands.py`, `paralic/hand_gestures.py`, `paralic/hand_session.py`); the
browser connects to the same `/ws` with `?mode=hand` and reuses the same cursor,
snapping and click code.

## How it works

```
 Browser (web/)                                   Python server (paralic/), on localhost
 ─────────────                                    ───────────────────────────────────────
 webcam ─► JPEG frames ──── WebSocket /ws ─────►  MediaPipe FaceLandmarker  (neural networks)
                                                    • face detector (BlazeFace)
                                                    • face mesh: 478 landmarks incl. irises
                                                    • blendshape network: eyeBlink, eyeLookUp…
                                                    • head pose (transformation matrix)
                                                            │
                                                  features: iris position in each eye, eyelid
                                                  opening, eye blendshapes, head rotation/position
                                                            │
                                                  GazeNet: your personal neural networks ─► screen x, y
                                                  (both eyes; each eye alone during a wink)
                                                            │
                                                  One Euro smoothing · blink freeze · blink, wink and
                                                  long-close detectors
 gaze cursor, snapping, clicks, drags, menu ◄─ JSON ─ gaze point + "double_blink", "wink_start/end",
 scrolling, dwell clicks                              "long_close" … events
```

* **MediaPipe FaceLandmarker** (Google) finds the face, places 478 landmarks (10 around the irises),
  scores 52 facial blendshapes and estimates the head pose, ~10 ms per frame on a laptop CPU.
* **Features** (`paralic/features.py`): for each eye, where the iris sits between the eye corners and how open
  the lids are (relative to the eye, so distance does not matter), the eye blendshapes, and head
  yaw/pitch/roll/position — 20 numbers per frame.
* **GazeNet** (`paralic/gazenet.py`) is a small multilayer perceptron written in NumPy: two tanh hidden
  layers (32 → 16) plus a linear skip connection that is initialised with ridge regression, trained with Adam
  and a Huber loss. The weight decay is chosen by *grouped cross-validation* (whole calibration dots are held
  out, so it measures how well the network interpolates to new screen positions) and three networks are
  averaged. The head-movement step of the calibration teaches it to compensate for head motion. Training
  takes about a second.
* **One-eye networks**: the same architecture trained on one eye's features plus the head pose. They keep the
  cursor moving during a wink and can lead for people whose other eye does not track reliably.
* **Blink detection** (`paralic/blink.py`) combines MediaPipe's blink blendshapes with the eyelid geometry,
  with thresholds that adapt to your eyes and to lids dropping when you look down. By default it watches the
  more open eye, so a wink is never a blink. A double blink is two blinks with a short pause between them;
  long eye closures never count (but a deliberate, deep one of 1–6 s is a *long close*).
* **Wink detection** (`paralic/gestures.py`) gives each eye its own adaptive baseline and looks for one eye
  closing while the other stays open; held for a moment it becomes a press.
* **Smoothing** (`paralic/filters.py`): a One Euro filter keeps the cursor steady while you look at something
  but quick when your eyes jump. When your eyes start to close the cursor freezes at where you were
  looking just before, so the double blink clicks the right thing.
* **Screen coordinates:** the network predicts positions on your *monitor*; the page converts them into page
  coordinates, so leaving full screen or moving the window keeps the calibration usable.
* In the page (`web/js/gaze.js`) the cursor is animated at 60 fps, snaps to the nearest button, and each click
  slightly corrects any drift (you can turn this off in Settings).

Note that the website draws its own gaze cursor; the operating system's mouse pointer is not moved (the
real mouse keeps working as usual, which is handy for a helper).

## Getting good accuracy

* Light your face evenly from the front; avoid a bright window behind you.
* Put the webcam at the top centre of the screen and sit about an arm's length (50–70 cm) away.
* Use full screen (`F11`) for the largest targets, and calibrate the way you will sit.
* If the cursor drifts, run **Quick adjust**; if it is far off, recalibrate.
* Webcam eye tracking is accurate to roughly 1–3 cm on the screen — that is why the site uses big buttons.

## Settings (all changeable with your eyes)

Per person: what holding each eye closed does · short winks · closing both eyes · dwell click and its time ·
wink hold time · tracking eye · wink and blink tests.
In this browser: cursor smoothing (Auto = learned) · snap to buttons (Auto) · double-blink speed (Personal) ·
blink sensitivity (Personal) · scroll speed · learn from clicks (drift correction) · keep learning my eyes
(fine-tuning) · cursor size · camera preview · sounds · speaking speed.
Missed double blinks → run the blink test, or *Relaxed* speed / *High* sensitivity. Unwanted clicks → *Low*
sensitivity or *Fast*, or switch to dwell click.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Camera blocked | Click the camera icon in the address bar, allow it, press Start again. On macOS also check *System Settings → Privacy & Security → Camera* for your browser. |
| “No camera found” / camera busy | Plug in a webcam; close other apps using it (Zoom, Teams…). |
| Camera doesn't work when opening the page by IP address | Browsers only allow cameras on `localhost` or HTTPS: use `http://localhost:8000`. |
| “MediaPipe could not start … libEGL” (Linux) | `sudo apt install libegl1 libgles2` |
| Model download failed | Download [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) and save it as `models/face_landmarker.task`. |
| `pip` can't find `mediapipe` | Use Python 3.9–3.12 (Intel Macs need ≤ 3.12). |
| Cursor jumpy | More smoothing, better light, recalibrate sitting still. |

## Privacy

Video frames go only from your browser to the Python program on the same computer. They are analysed in
memory and never stored or uploaded. The saved calibration contains numbers (eye/head measurements and the
network's weights), no images. The server listens on `127.0.0.1` and only accepts pages served from this
machine.

## Project layout

```
paralic/            Python package (server + eye tracking)
  __main__.py       command line entry point (python -m paralic)
  server.py         FastAPI app: website + /ws WebSocket
  session.py        per-connection pipeline (decode → MediaPipe → features → blink/wink → GazeNet → smoothing)
  tracker.py        MediaPipe FaceLandmarker wrapper
  features.py       landmark → feature extraction
  blink.py          blink / double-blink / long-close detector
  gestures.py       wink detector, blink signal choice, wink test analysis
  filters.py        One Euro filter + blink-aware cursor stabiliser
  gazenet.py        GazeNet neural networks (NumPy MLP, Adam, cross-validation, ensemble, one-eye networks)
  calibration.py    calibration data, training, saved profile
  personalize.py    blink test, auto smoothing/magnet, champion/challenger fine-tuning, A/B statistics
  users.py          people and their files
  model_assets.py   model download
web/                the website (vanilla HTML/CSS/JS modules, no build step)
  js/tracker.js     camera capture + WebSocket client (and the mouse demo mode)
  js/gaze.js        gaze cursor, snapping, double-blink and dwell clicks, scroll rail, pause
  js/gestures.js    press / drag / drop, the gaze menu ("right click"), drag lock
  js/calibration.js calibration / validation / quick adjust / blink and wink tests
  js/pages/…        Home, Explore, Read, Talk, Arrange, Draw, Practice, Lab, Settings, Help
tests/              unit, integration (real MediaPipe) and browser tests
```

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest                       # unit, server, MediaPipe integration and browser tests
python -m playwright install chromium  # once, for the browser tests (skipped when unavailable)

# Full end-to-end run with a fake webcam video (face + double blink) through the real pipeline:
python tests/make_fake_video.py /tmp/face.y4m
PARALIC_FAKE_VIDEO=/tmp/face.y4m python -m pytest --runslow tests/test_browser.py
```

The MediaPipe integration tests download a public-domain test portrait on first use and are skipped offline.

`python tools/benchmark.py` runs the personalisation benchmark on simulated people (model search,
fine-tuning under drift and with bad labels, one-eye networks, smoothing, blink and wink thresholds, A/B
decisions) and writes [docs/benchmark.md](docs/benchmark.md). It uses simulated eyes, not real people.

## License

**Proprietary — All Rights Reserved.** Copyright (c) 2026 Maliq Barnard.
No use, copying, modification, distribution, or reverse engineering is permitted without the Owner's prior written permission. See [LICENSE](LICENSE).

## Credits

* [MediaPipe](https://developers.google.com/mediapipe) Face Landmarker (Apache 2.0) for face, iris and
  blendshape detection.
* One Euro filter: Casiez, Roussel & Vogel, *1€ Filter*, CHI 2012.
