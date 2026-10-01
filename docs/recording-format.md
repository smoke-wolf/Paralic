# Session recordings (format 1)

A recording keeps everything about one session — the configuration, every
camera frame's measurements and what the pipeline made of them, every command
and event — so a real-world run can be replayed and inspected afterwards in
the **Paralic Inspector** (`python -m paralic.inspector`). Recording is off
unless it is switched on (the ● Rec button in the page, or `python -m paralic
--record` for every session). Recordings stay on this computer, under
`data/recordings/`; they contain camera images of the face, so share them only
knowingly.

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

All times: `t` is the session clock in seconds (monotonic, the same clock the
pipeline uses); `wall` is Unix time. Frame ids are the browser's frame ids.

## meta.json

```json
{
  "format": 1,
  "paralic": "1.0.0",
  "id": "20261001-142233-Sam-eyes",
  "started": "2026-10-01T14:22:33", "ended": "2026-10-01T14:40:02",
  "mode": "eyes",
  "person": {"id": "u1a2b3c4d", "name": "Sam"},
  "screen": {"w": 1512, "h": 982, "dpr": 2},
  "settings": {...},            // the session's settings (smoothing, blink_sensitivity, ...)
  "effective": {...},           // TrackerSession.effective() at the start
  "personal": {...},            // the person's personal.json at the start
  "profile": {...},             // ProfileStore summary, or null
  "hand": {...},                // hand setup summary (hand mode), or null
  "video": {"every": 1, "max_width": 640, "quality": 80},
  "summary": {"frames": 31000, "video_frames": 31000, "events": 812, "bytes": 912345678}
}
```

`ended` and `summary` are written when the recording stops (and refreshed every
minute, so a crash loses little).

## frames.jsonl

One JSON object per processed camera frame:

```json
{"i": 1234, "t": 81.532, "wall": 1790862153.12,
 "msg": {...},          // the frame message exactly as sent to the page (gaze, raw, face, closures,
                        // thresholds, head pose, glasses/glare, faces, hand fields, ...)
 "features": [...],     // eye mode: the 28-number feature vector given to GazeNet (null without a face)
 "blink": {"closing": false, "closed": false, "deep": false,
           "close_thr": 0.42, "open_thr": 0.31, "signal": 0.12},
 "wink": {"winking": null, "pressed": null},
 "net": "both",         // the network that predicted (both / left / right), or null
 "label": {...},        // the calibration label the page attached, if any
 "stored": true,        // the frame was kept as a calibration sample
 "events": ["blink", "double_blink"],   // event types this frame produced
 "video": true}         // video/<i>.jpg exists
```

## landmarks/NNNNNN.npz

Chunk number `NNNNNN` holds up to 300 consecutive frames: `ids` (int64, frame
ids) and `points` (float16, shape (n, P, 3)) — the 478 face-mesh points in
pixels in eye mode, or the 21 hand points (image fractions) in hand mode.
Frames without a face or hand are not included. In eye mode there may also be
`others` (float16, (n, 3, 4)): bounding boxes of other faces in view, NaN
where there are fewer.

## events.jsonl

```json
{"t": 81.54, "wall": 1790862153.13, "kind": "command", "type": "calibration_start", "data": {...}}
{"t": 81.55, "wall": 1790862153.14, "kind": "reply",   "type": "calibration_started", "data": {...}}
{"t": 95.10, "wall": 1790862166.70, "kind": "push",    "type": "finetune_result", "data": {...}}
{"t": 96.20, "wall": 1790862167.80, "kind": "event",   "type": "double_blink", "data": {...}}
```

`kind`: `command` (from the page), `reply` (to a command), `push` (sent by a
background job), `event` (a gesture or other message produced by a frame:
blinks, winks, clicks, scrolls, face recognition, glasses changes...).

## models/NNN-<why>.json

The gaze model (`GazeNet.to_dict()`, with its `meta`) each time it changes:
`000-start` (if one was loaded), then `calibration`, `refit`, `adjust`,
`refine`, `finetune`, `profile_load`... The Inspector runs these networks on
the recorded `features` to show what happens inside them (normalised inputs,
every hidden layer's activations, each ensemble member's output, the one-eye
networks).
