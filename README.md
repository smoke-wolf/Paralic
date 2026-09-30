# Paralic — browse the web with your eyes

Paralic is a website that runs on your own computer (`http://localhost`) and can be used **with your eyes only**.
A Python server uses **neural networks** to track your eyes through an ordinary webcam:

* **look** somewhere and the cursor moves there (it gently snaps to the nearest button),
* **blink twice** to click,
* **look at the arrows on the right edge** to scroll,
* **blink twice on Pause** to rest your eyes, and blink twice again to resume.

The site itself is designed for eye control: big targets, a Solar System to explore, articles to read,
a *Talk* page with spoken phrases and an eye-typing keyboard with word prediction, a target-practice game,
and settings you can change with your eyes.

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
> The mouse then plays the part of your eyes and pressing **B** twice quickly is a double blink.

## Using it

1. **Start eye tracking** – the only click you need. Allow the camera when the browser asks.
   (Once allowed, Paralic starts the camera by itself next time.)
2. **Camera check** – centre your face; the eyes are outlined when they are found.
3. **Blink twice to calibrate** – follow the dot with your eyes for about 30 seconds. When asked, keep
   looking at the centre dot and gently move your head (skip that if moving is hard for you — just keep looking).
   Your personal neural network is trained, five more dots measure the accuracy, and the result is shown.
4. **Start browsing** – look at it and blink twice.

| Gesture | What it does |
| --- | --- |
| Look | Moves the cursor. The nearest button lights up. |
| Blink twice | Clicks what was highlighted *just before the first blink*. A small “1” shows after the first blink. |
| Look at *Scroll up / Scroll down* (right edge) | Scrolls; look further towards the arrow to go faster. Blink twice there to jump a page. |
| Blink twice on *Pause* | Pauses (nothing gets clicked). Blink twice to resume. |

The calibration is saved (numbers only, no images) in `data/profile.json`. Next time you can use it as is,
do a **Quick adjust** (5 dots, ~8 seconds) or a full calibration. Quick adjust and full calibration are also
on the Home and Settings pages.

**Helper shortcuts** (for someone assisting): `C` full calibration · `A` quick adjust · `P` pause/resume ·
`Esc` cancel calibration · `F11` full screen.

**Fully hands-free:** after the camera permission has been granted once, the site starts tracking without a
click. For a kiosk-style setup, launch the browser in full screen, e.g.
`chrome --kiosk --autoplay-policy=no-user-gesture-required http://localhost:8000` (the autoplay flag lets the
*Talk* page speak without a first click).

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
                                                  GazeNet: your personal neural network ─► screen x, y
                                                            │
                                                  One Euro smoothing · blink freeze · blink detector
 gaze cursor, snapping, clicks, scrolling ◄─ JSON ─ gaze point + "blink" / "double_blink" events
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
* **Blink detection** (`paralic/blink.py`) combines MediaPipe's blink blendshapes with the eyelid geometry,
  with thresholds that adapt to your eyes and to lids dropping when you look down. A double blink is two
  blinks with a short pause between them; long eye closures never count.
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

Cursor smoothing · snap to buttons · double-blink speed · blink sensitivity · scroll speed · learn from
clicks (drift correction) · cursor size · camera preview · sounds · speaking speed.
Missed double blinks → *Relaxed* speed or *High* sensitivity. Unwanted clicks → *Low* sensitivity or *Fast*.

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
  session.py        per-connection pipeline (decode → MediaPipe → features → blink → GazeNet → smoothing)
  tracker.py        MediaPipe FaceLandmarker wrapper
  features.py       landmark → feature extraction
  blink.py          blink / double-blink detector
  filters.py        One Euro filter + blink-aware cursor stabiliser
  gazenet.py        GazeNet neural network (NumPy MLP, Adam, cross-validation, ensemble)
  calibration.py    calibration data, training, saved profile
  model_assets.py   model download
web/                the website (vanilla HTML/CSS/JS modules, no build step)
  js/tracker.js     camera capture + WebSocket client (and the mouse demo mode)
  js/gaze.js        gaze cursor, snapping, double-blink clicks, scroll rail, pause
  js/calibration.js calibration / validation / quick adjust
  js/pages/…        Home, Explore, Read, Talk, Practice, Settings, Help
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

## License

**Proprietary — All Rights Reserved.** Copyright (c) 2026 Maliq Barnard.
No use, copying, modification, distribution, or reverse engineering is permitted without the Owner's prior written permission. See [LICENSE](LICENSE).

## Credits

* [MediaPipe](https://developers.google.com/mediapipe) Face Landmarker (Apache 2.0) for face, iris and
  blendshape detection.
* One Euro filter: Casiez, Roussel & Vogel, *1€ Filter*, CHI 2012.
