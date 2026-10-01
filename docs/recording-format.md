# Session recordings (format 1)

A recording keeps everything about one session — the configuration, every
camera frame's measurements and what the pipeline made of them, every command
and event — so a real-world run can be replayed and inspected afterwards in
the **Paralic Inspector** (`python -m paralic.inspector`). Recording is off
unless it is switched on (the ● Rec button in the page or Shift+R, or `python
-m paralic --record` for every session). Recordings stay on this computer, under
`data/recordings/` (`--recordings-dir` puts them elsewhere); they contain camera
images of the face, so share them only knowingly. Written by
`paralic/recorder.py`.

## Layout

```
data/recordings/<YYYYmmdd-HHMMSS>-<person name>-<eyes|hand>/
  meta.json          the configuration at the start, and a summary at the end
  frames.jsonl       one line per camera frame
  events.jsonl       commands, replies, pushed messages and gesture events
  landmarks/NNNNNN.npz   the tracked points, in chunks of 300 frames
  models/NNN-<why>.json  the gaze model whenever it changes (also the one in use at the start)
  video/<frame id>.jpg   camera images (every frame, or every n-th: see meta.json)
```

The directory name uses the local time the recording started and the person's
name with anything but letters and digits turned into `_` (`Person 1` →
`Person_1`); a second recording started in the same second gets `-2`, `-3`...
On macOS and Linux the directory is readable only by the user who runs
Paralic. Each recording is
one browser connection: turning recording off and on again, reloading the page
or reconnecting starts a new directory.

All times: `t` is the session clock in seconds (monotonic, the same clock the
pipeline uses); `wall` is Unix time. Frame ids are the browser's frame ids. All
JSON is strict (no `NaN`), so a browser's `JSON.parse` reads it.

## meta.json

```json
{
  "format": 1,
  "paralic": "1.0.0",
  "id": "20261001-142233-Sam-eyes",
  "started": "2026-10-01T14:22:33", "ended": "2026-10-01T14:40:02",
  "complete": true,             // stopped cleanly (false while recording, or after a crash or error)
  "clock": {"t": 81.532, "wall": 1790862153.12},   // the session clock and Unix time at the start
  "mode": "eyes",
  "person": {"id": "u1a2b3c4d", "name": "Sam"},
  "screen": {"w": 1512, "h": 982, "dpr": 2},
  "settings": {...},            // the session's settings (smoothing, blink_sensitivity, ...)
  "effective": {...},           // TrackerSession.effective() at the start
  "personal": {...},            // the person's personal.json at the start
  "profile": {...},             // ProfileStore summary, or null
  "hand": {...},                // hand setup summary (hand mode), or null
  "detectors": {...},           // what the detectors and filters work with (below)
  "video": {"every": 2, "max_width": 640, "quality": 70, "max_bytes": 3000000000, "stopped": null},
  "summary": {"frames": 31000, "video_frames": 15500, "events": 812, "models": 4, "dropped": 0,
              "bytes": 512345678},
  "error": null                 // why the recording stopped by itself (e.g. the disk was full)
}
```

`ended` and `summary` are written when the recording stops, and refreshed every
minute, so a crash loses little (`ended` is null until the first refresh;
`complete` stays false). `summary.bytes` is the size of the whole recording;
`dropped` counts frames left out because the disk could not keep up (normally
0; camera images are left out before that).

`detectors`, eye mode: `blink` (the blink detector's `BlinkConfig`),
`blink_signal` (`both`, `mean`, `left` or `right`: what the blink detector
watches), `wink` (`WinkConfig`, with `left` and `right`) and `smoothing` (the
One Euro filter's `min_cutoff`, `beta`, `d_cutoff`, and the stabiliser's
`rewind_s`, `settle_s`, `max_freeze_s`). Hand mode: `hand` — `config`
(`HandGestureConfig`), `calibration` (the person's hand setup: palm `span`,
`pinch_on`, `pinch_off`, the `pointing` map) and `smoothing`.

`video`: the camera image of every `every`-th frame (0: none) is kept. Images
up to `max_width` pixels wide are kept exactly as the browser sent them (a
JPEG of quality 82); wider ones are shrunk to `max_width` and saved with JPEG
quality `quality`. Images stop — everything else goes on — once the recording
holds `max_bytes`, or when less than 1 GB is free on the disk; `stopped` then
says when and why: `{"i": 1234, "t": 81.5, "reason": "the recording reached 3
GB", "bytes": 3000012345}` (`i` and `t` are null when it was so from the start).

## frames.jsonl

One JSON object per processed camera frame:

```json
{"i": 1234, "t": 81.532, "wall": 1790862153.12,
 "msg": {...},          // the frame message exactly as sent to the page (gaze, raw, face, closures,
                        // thresholds, head pose, glasses/glare, faces, hand fields, ...)
 "features": [...],     // eye mode: the 28-number feature vector given to GazeNet (null without a face)
 "blink": {"closing": false, "closed": false, "deep": false,
           "close_thr": 0.42, "open_thr": 0.31, "signal": 0.12, "baseline": 0.2, "pending": 0},
 "wink": {"winking": null, "pressed": null, "state": "idle",
          "base": [0.21, 0.23], "close_thr": [0.53, 0.54], "open_thr": [0.37, 0.38]},
 "net": "both",         // the network that predicted (both / left / right), or null
 "label": {...},        // the calibration label the page attached, or null
 "stored": true,        // the frame was kept as a calibration sample
 "events": ["blink", "double_blink"],   // event types this frame produced
 "gesture": "left",     // only while the wink test asks to close an eye ("rest", "left", "right")
 "video": true}         // video/<i>.jpg exists
```

* `msg`: exactly what the page got, with one exception: during the calibration
  the page asks for the whole face mesh as well (`mesh`, up to 9 KB per frame);
  it is recorded as `"mesh": true`, since `landmarks/` holds all of it.
* `features`: rounded to 6 decimals.
* `blink` (eye mode, else null): the blink detector after this frame.
  `signal` is the closure it watched (see `detectors.blink_signal`; null
  without a face), `baseline` its adaptive open-eye level, `pending` 1 while a
  first blink waits for a second.
* `wink` (eye mode, else null): the wink detector after this frame. `state`:
  `idle`, `candidate` (one eye closing: the open eye's network leads) or
  `active` (held: a press); `base`, `close_thr`, `open_thr`: `[left, right]`,
  per eye, to compare with the message's `cl` and `cr`.
* `features`, `blink` and `wink` are null in hand mode (the hand's
  measurements are in `msg`: `hand`, `pinching`, `scrolling`...).

Frames the pipeline could not process at all (an image that does not decode, a
tracker failure) are not recorded.

## landmarks/NNNNNN.npz

Chunk number `NNNNNN` (000000, 000001, ...) holds up to 300 consecutive
frames (only the last chunk has fewer): `ids` (int64, frame ids) and `points`
(float16, shape (n, P, 3)) — the 478 face-mesh points in pixels in eye mode,
or the 21 hand points (image fractions) in hand mode. Frames without a face or
hand are not included. In eye mode there may also be `others` (float16, (n, 3,
4)): bounding boxes `[x0, y0, x1, y1]` of other faces in view, taken from the
frame message's `others`, NaN where there are fewer. Compressed `.npz`
(`numpy.load` reads it).

## events.jsonl

```json
{"t": 81.54, "wall": 1790862153.13, "kind": "command", "type": "calibration_start", "data": {...}}
{"t": 81.55, "wall": 1790862153.14, "kind": "reply",   "type": "calibration_started", "data": {...}}
{"t": 95.10, "wall": 1790862166.70, "kind": "push",    "type": "finetune_result", "data": {...}}
{"t": 96.20, "wall": 1790862167.80, "kind": "event",   "type": "double_blink", "data": {...}}
```

`kind`: `command` (from the page), `reply` (to a command), `push` (sent by a
background job), `event` (a gesture or other message produced by a frame:
blinks, winks, clicks, scrolls, face recognition, glasses changes...; and three
of the recording's own, below). `data` is the message itself, as sent or
received. Lines are in the order things happened: a command, then what it
changed, then its replies.

* Not recorded: `ping` / `pong`, and the page asking how the recording is
  going (a `recording` command without `on`). Pictures inside messages (the
  face print's faces, as `data:` URLs) are replaced by
  `"data:image/jpeg;base64,... (N characters, not recorded)"`.
* `model`: the gaze model changed — `{"why": "refit", "file":
  "models/002-refit.json", "version": 3}` (`version`: the person's model
  version, null before a calibration is finished); `file` is null when there is
  no model any more (another person was chosen).
* `person_changed`: someone else was chosen (`user_select`, `user_create`, or
  the current person was deleted). The recording goes on; `data` has the new
  `person` and their configuration (`mode`, `screen`, `settings`, `effective`,
  `personal`, `profile`, `hand`, `detectors`, as in meta.json) and the
  `previous` person (`{"id", "name"}`).
* `video_stopped`: no more camera images from here on (`data` as
  `video.stopped` in meta.json).

## models/NNN-<why>.json

The gaze model (`GazeNet.to_dict()`, with its `meta`) each time it changes:
`000-start` (if one was loaded), then `calibration` (the network a calibration
trains first), `refit` (refitted with the accuracy-check dots: the
calibration's final network), `adjust`, `refine` (an improving round),
`finetune`, `profile_load`... — or the name of whichever other command changed
the model. `NNN` counts from 000. Each one has a `model` event, whose `t` says
from when it was used. The Inspector runs these networks on the recorded
`features` to show what happens inside them (normalised inputs, every hidden
layer's activations, each ensemble member's output, the one-eye networks).

## Recording on and off

From the page: `{"type": "recording", "on": true, "video": {"every": 2,
"max_width": 640, "quality": 70}}` (`video` optional, any part of it; `false`:
no images) starts recording, `{"type": "recording", "on": false}` stops it, and
`{"type": "recording"}` asks how it is going. The reply:

```json
{"type": "recording", "ok": true, "on": true, "id": "20261001-142233-Sam-eyes",
 "dir": "/home/sam/Paralic/data/recordings/20261001-142233-Sam-eyes", "frames": 1234,
 "video_frames": 617, "events": 40, "bytes": 23456789, "seconds": 41.2,
 "video": {"every": 2, "max_width": 640, "quality": 70}, "video_stopped": null, "error": null}
```

After `on: false` it describes the finished recording. `{"type": "recording",
"ok": true, "on": false}` means none is running; `ok: false` and `error` that
one could not be started. With `python -m paralic --record` the session starts
recording when the page says hello, and sends this message after the `hello`
reply.

At the defaults, one minute at 30 frames per second takes about 24 MB (camera
images 17 MB, landmarks 4 MB, frames.jsonl 3 MB): about 480 MB for 20
minutes (320 MB at 20 frames per second).
