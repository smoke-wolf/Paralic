"""Session recordings (paralic/recorder.py), checked file by file against
docs/recording-format.md: eye mode (calibration, clicks, a background job, a
new person), hand mode, the size cap, failures, the server and the command
line - and the real MediaPipe face model on the test portrait."""

import json
import re
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import paralic.recorder as recorder_mod
import paralic.session as session_mod
from paralic import __version__
from paralic.gazenet import GazeNet
from paralic.recorder import VideoOptions, fit_width, jpeg_size
from paralic.server import create_app
from paralic.session import TrackerSession, pack_frame
from paralic.users import UserStore
from tests.fakes import FakeHandTracker, FakeTracker, ManualClock, fake_features, make_hand, patch_session, tiny_jpeg
from tests.synthetic import SCREEN_H, SCREEN_W, Head, VirtualUser, calibration_points

cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parent.parent
FRAME_KEYS = {"i", "t", "wall", "msg", "features", "blink", "wink", "net", "label", "stored", "events", "video"}
EVENT_KEYS = {"t", "wall", "kind", "type", "data"}
MESH = np.random.default_rng(7).uniform((380.0, 140.0, -40.0), (580.0, 420.0, 40.0), (478, 3))


def camera_jpeg(width: int = 960, height: int = 540, seed: int = 0) -> bytes:
    """A webcam-sized camera image (960 px: wider than a recording keeps)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    img = np.stack([xx * 255 // width, yy * 255 // height, np.full_like(xx, 120)], axis=-1)
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    assert ok
    return buf.tobytes()


JPEG = camera_jpeg()


class Mesh(np.ndarray):
    """478 face-mesh points (pixels) that also carry the frame's prepared features."""

    features = None


def mesh(features, rng) -> Mesh:
    m = (MESH + rng.normal(0, 0.4, MESH.shape)).view(Mesh)
    m.features = features
    return m


def _strict(text: str):
    """Parse JSON as strictly as a browser does (no NaN or Infinity)."""
    def refuse(name):
        raise ValueError(f"{name} is not JSON")
    return json.loads(text, parse_constant=refuse)


def read_recording(path: Path) -> dict:
    lines = lambda name: [_strict(s) for s in (path / name).read_text().splitlines()]  # noqa: E731
    return {
        "meta": _strict((path / "meta.json").read_text()),
        "frames": lines("frames.jsonl"),
        "events": lines("events.jsonl"),
        "chunks": [dict(np.load(p)) for p in sorted((path / "landmarks").glob("*.npz"))],
        "chunk_names": sorted(p.name for p in (path / "landmarks").glob("*.npz")),
        "models": sorted(p.name for p in (path / "models").glob("*.json")),
        "video": sorted((path / "video").glob("*.jpg"), key=lambda p: int(p.stem)),
        "size": sum(p.stat().st_size for p in path.rglob("*") if p.is_file()),
    }


def wait_until(check, timeout: float = 10.0):
    deadline = time.time() + timeout
    while not check():
        assert time.time() < deadline, "timed out"
        time.sleep(0.02)


@pytest.fixture
def eyes(monkeypatch, tmp_path):
    """An eye-mode session with fake face tracking whose frames carry a face mesh."""
    patch_session(monkeypatch)
    monkeypatch.setattr(session_mod, "extract_features", lambda pts, size, bs=None, m=None: pts.features)
    # Frames come much faster than from a camera here: never leave images out for that.
    monkeypatch.setattr(recorder_mod, "VIDEO_BACKLOG", 1 << 30)
    tracker = FakeTracker()
    clock = ManualClock()
    pushed = []
    session = TrackerSession(lambda: tracker, UserStore(tmp_path / "data"), clock=clock, push=pushed.append,
                             recordings=tmp_path / "recordings")
    session.handle_command({"type": "hello", "screen": {"w": SCREEN_W, "h": SCREEN_H, "dpr": 1}})
    session.handle_command({"type": "faceprint_set", "enabled": False})
    user = VirtualUser(seed=5, noise=0.5)
    rng = np.random.default_rng(1)
    state = {"id": 0}

    def frame(sx=SCREEN_W / 2, sy=SCREEN_H / 2, label=None, closure=0.12, face=True, image=JPEG, gesture=None):
        state["id"] += 1
        tracker.push(mesh(fake_features(user.features(sx, sy, Head()), closure), rng) if face else None)
        header = {"id": state["id"]}
        if label:
            header["label"] = label
        if gesture:
            header["gesture"] = gesture
        out = session.handle_frame(pack_frame(header, image))
        clock.tick()
        return out

    session.frame, session.pushed, session.clock_fake = frame, pushed, clock
    yield session
    session.close()


def calibrate(session):
    frame = session.frame
    session.handle_command({"type": "calibration_start", "mode": "full"})
    for i, (sx, sy) in enumerate(calibration_points()):
        for _ in range(20):
            frame(sx, sy, {"x": sx, "y": sy, "kind": "cal", "pt": i})
    for _ in range(60):
        frame(SCREEN_W / 2, SCREEN_H / 2, {"x": SCREEN_W / 2, "y": SCREEN_H / 2, "kind": "head", "pt": 0})
    return session.handle_command({"type": "calibration_fit", "mode": "full"})[0]


def validate(session):
    for i, (sx, sy) in enumerate([(500, 300), (1400, 300), (960, 600), (500, 900), (1400, 900)]):
        for _ in range(16):
            session.frame(sx, sy, {"x": sx, "y": sy, "kind": "val", "pt": i})
    return session.handle_command({"type": "validation_finish"})[0]


def click(session, sx, sy):
    """Look at (sx, sy), blink twice, and let the page report the click."""
    for _ in range(40):
        session.frame(sx, sy)
    msgs = []
    for closures in ([0.5, 0.9, 0.9, 0.9, 0.5], [0.12] * 5, [0.5, 0.9, 0.9, 0.9, 0.5], [0.12] * 5):
        for c in closures:
            msgs += session.frame(sx, sy, closure=c)
    db = next(m for m in msgs if m["type"] == "double_blink")
    return session.handle_command({"type": "label_event", "kind": "click", "pre_frame": db["pre_frame"],
                                   "rect": [sx - 80, sy - 40, sx + 80, sy + 40]})[0]


def quick_finetune(champion, data):
    """Fine-tuning that finishes at once (the real one searches models for a while)."""
    model = champion.clone()
    model.meta["source"] = "fine-tuned"
    return model, {"accepted": True, "winner": "fine-tuned", "champion_error_px": 52.0,
                   "candidate_errors_px": {"fine-tuned": 41.0}}


# -- eye mode -----------------------------------------------------------------------------------

def test_an_eye_session_is_recorded_as_the_format_says(eyes, monkeypatch, tmp_path):
    session = eyes
    monkeypatch.setattr(session_mod, "run_finetune", quick_finetune)
    first = dict(session.user)
    reply = session.handle_command({"type": "recording", "on": True})[0]
    assert reply["type"] == "recording" and reply["ok"] and reply["on"] and reply["frames"] == 0
    path = Path(reply["dir"])
    assert path.parent == tmp_path / "recordings" and path.name == reply["id"]
    assert re.fullmatch(r"\d{8}-\d{6}-Person_1-eyes", reply["id"])
    assert reply["video"] == {"every": 2, "max_width": 640, "quality": 70}

    sent = []                                       # what the page got for each frame
    frame = session.frame
    session.frame = lambda *a, **k: sent.append(frame(*a, **k)) or sent[-1]
    fit = calibrate(session)
    assert fit["ok"]
    probe = VirtualUser(seed=5, noise=0).features(700, 500, Head())
    fit_prediction = session.model.predict(probe)
    assert validate(session)["ok"]
    assert click(session, 1500, 250)["stored"] and click(session, 400, 800)["stored"]
    session.frame(face=False)
    session.frame(face=False)
    session.frame(gesture="left")
    session.handle_command({"type": "ping", "t": 1})
    assert session.handle_command({"type": "finetune"})[0]["ok"]
    session.wait_for_job()
    finetuned = session.model
    tuned_prediction = finetuned.predict(probe)
    session.handle_command({"type": "user_create", "name": "Alex"})
    session.frame()
    assert session.handle_command({"type": "recording"})[0]["on"]        # "how is it going?"
    stopped = session.handle_command({"type": "recording", "on": False})[0]
    assert stopped["on"] is False and stopped["frames"] == len(sent) and stopped["error"] is None
    assert session.handle_command({"type": "recording"})[0] == {"type": "recording", "ok": True, "on": False}

    rec = read_recording(path)
    meta, frames, events = rec["meta"], rec["frames"], rec["events"]

    # meta.json
    assert meta["format"] == 1 and meta["paralic"] == __version__ and meta["id"] == path.name
    assert meta["started"] <= meta["ended"] and meta["complete"] is True and meta["error"] is None
    assert meta["mode"] == "eyes" and meta["person"] == {"id": first["id"], "name": first["name"]}
    assert meta["screen"] == {"w": SCREEN_W, "h": SCREEN_H, "dpr": 1}
    assert meta["settings"] == {"smoothing": "auto", "blink_sensitivity": "personal", "double_blink": "personal",
                                "learning": True}
    assert meta["effective"]["double_gap_ms"] > 0 and meta["personal"]["faceprint"] == {"enabled": False}
    assert meta["profile"] is None and meta["hand"] is None
    assert meta["detectors"]["blink"]["sensitivity"] == meta["effective"]["sensitivity"]
    assert {"wink", "smoothing", "blink_signal"} <= set(meta["detectors"])
    assert meta["video"] == {"every": 2, "max_width": 640, "quality": 70, "max_bytes": 3_000_000_000, "stopped": None}
    assert meta["clock"]["t"] == pytest.approx(100.0, abs=0.1)
    summary = meta["summary"]
    assert summary["frames"] == len(frames) == len(sent) and summary["events"] == len(events)
    assert summary["video_frames"] == len(rec["video"]) and summary["models"] == len(rec["models"])
    assert summary["dropped"] == 0
    assert stopped["bytes"] == rec["size"] and summary["bytes"] == pytest.approx(rec["size"], abs=100)

    # frames.jsonl: one line per frame, the message exactly as the page got it.
    assert all(set(f) == FRAME_KEYS or set(f) == FRAME_KEYS | {"gesture"} for f in frames)
    assert [f["msg"] for f in frames] == [json.loads(json.dumps(out[0])) for out in sent]
    assert [f["i"] for f in frames] == [out[0]["id"] for out in sent]
    assert all(b["t"] > a["t"] for a, b in zip(frames, frames[1:]))
    seen = [f for f in frames if f["msg"]["face"]]
    assert all(len(f["features"]) == 28 for f in seen)
    assert all(set(f["blink"]) >= {"closing", "closed", "deep", "close_thr", "open_thr", "signal"} for f in seen)
    assert all(set(f["wink"]) >= {"winking", "pressed"} for f in seen)
    assert all(abs(f["blink"]["close_thr"] - f["msg"]["thr"][0]) <= 6e-4 for f in seen)
    blank = [f for f in frames if not f["msg"]["face"]]
    assert len(blank) == 2 and all(f["features"] is None and f["blink"]["signal"] is None for f in blank)
    cal = [f for f in frames if (f["label"] or {}).get("kind") == "cal"]
    assert len(cal) == 13 * 20 and cal[0]["label"] == {"x": calibration_points()[0][0], "y": calibration_points()[0][1],
                                                       "kind": "cal", "pt": 0}
    assert sum(f["stored"] for f in cal) > 200 and not any(f["stored"] for f in frames if f["label"] is None)
    assert [f["gesture"] for f in frames if "gesture" in f] == ["left"]
    assert sum("double_blink" in f["events"] for f in frames) == 2
    assert frames[-1]["net"] is None and frames[-2]["net"] == "both"             # Alex is not calibrated
    assert [f["video"] for f in frames] == [k % 2 == 0 for k in range(len(frames))]

    # video/: exactly the frames that say so, shrunk to 640 px.
    assert [int(p.stem) for p in rec["video"]] == [f["i"] for f in frames if f["video"]]
    img = cv2.imread(str(rec["video"][0]))
    assert img.shape == (360, 640, 3)

    # landmarks/: chunks of 300 frames with a face.
    assert rec["chunk_names"][:2] == ["000000.npz", "000001.npz"]
    assert [len(c["ids"]) for c in rec["chunks"][:-1]] == [300] * (len(rec["chunks"]) - 1)
    ids = np.concatenate([c["ids"] for c in rec["chunks"]])
    assert ids.dtype == np.int64 and ids.tolist() == [f["i"] for f in seen]
    for c in rec["chunks"]:
        assert set(c) == {"ids", "points"}
        assert c["points"].dtype == np.float16 and c["points"].shape == (len(c["ids"]), 478, 3)
        assert np.abs(c["points"].astype(float) - MESH).max() < 4

    # events.jsonl: commands, replies, pushes and the frames' events, in order.
    assert all(set(e) == EVENT_KEYS for e in events)
    assert all(b["t"] >= a["t"] for a, b in zip(events, events[1:]))
    kinds = {k: [e["type"] for e in events if e["kind"] == k] for k in ("command", "reply", "push", "event")}
    assert set(kinds["command"]) == {"recording", "calibration_start", "calibration_fit", "validation_finish",
                                     "label_event", "finetune", "user_create"}
    assert kinds["command"][0] == "recording" and kinds["command"][-1] == "recording"
    assert next(e for e in events if e["kind"] == "command")["data"] == {"type": "recording", "on": True}
    assert set(kinds["reply"]) == {"recording", "calibration_started", "calibration_result", "validation_result",
                                   "label_stored", "finetune_started", "users"}
    assert kinds["push"] == ["finetune_result"]
    assert {"blink", "double_blink", "model", "person_changed"} <= set(kinds["event"])
    assert "ping" not in kinds["command"] and "pong" not in kinds["reply"]
    db = next(e for e in events if e["type"] == "double_blink")
    assert db["data"]["type"] == "double_blink" and db["data"]["frame"] in ids
    switch = next(e for e in events if e["type"] == "person_changed")["data"]
    assert switch["person"]["name"] == "Alex" and switch["previous"] == {"id": first["id"], "name": first["name"]}
    assert {"settings", "effective", "personal", "profile", "detectors"} <= set(switch)
    order = [(e["kind"], e["type"]) for e in events]
    assert order.index(("command", "user_create")) < order.index(("event", "person_changed")) \
        < order.index(("reply", "users"))

    # models/: the network after the calibration fit, the refit, the fine-tune; then none (Alex).
    assert rec["models"] == ["000-calibration.json", "001-refit.json", "002-finetune.json"]
    model_events = [e["data"] for e in events if e["type"] == "model"]
    assert [m["file"] for m in model_events] == [f"models/{n}" for n in rec["models"]] + [None]
    assert [m["why"] for m in model_events] == ["calibration", "refit", "finetune", "user_create"]
    assert [m["version"] for m in model_events] == [None, 1, 2, None]
    after_fit = GazeNet.from_dict(json.loads((path / "models" / rec["models"][0]).read_text()))
    np.testing.assert_allclose(after_fit.predict(probe), fit_prediction, atol=1e-6)
    tuned = GazeNet.from_dict(json.loads((path / "models" / rec["models"][2]).read_text()))
    np.testing.assert_allclose(tuned.predict(probe), tuned_prediction, atol=1e-6)
    assert tuned.meta["version"] == 2 and tuned.meta["source"] == "fine-tuned"


def test_a_loaded_model_is_the_first_snapshot(eyes):
    session = eyes
    calibrate(session)
    validate(session)
    again = session.handle_command({"type": "recording", "on": True})[0]
    session.handle_command({"type": "profile_load"})
    session.handle_command({"type": "recording", "on": False})
    rec = read_recording(Path(again["dir"]))
    assert rec["models"] == ["000-start.json", "001-profile_load.json"]
    assert rec["meta"]["profile"]["accuracy_px"] is not None


def test_turning_recording_off_and_on_starts_a_new_directory(eyes):
    session = eyes
    a = session.handle_command({"type": "recording", "on": True, "video": {"every": 3}})[0]
    for _ in range(5):
        session.frame()
    assert session.handle_command({"type": "recording", "on": True})[0]["id"] == a["id"]   # still the same one
    session.handle_command({"type": "recording", "on": False})
    session.frame()                                                                       # not recorded
    b = session.handle_command({"type": "recording", "on": True})[0]
    for _ in range(4):
        session.frame()
    session.close()                                                                       # stops it too
    assert a["dir"] != b["dir"] and Path(a["dir"]).parent == Path(b["dir"]).parent
    ra, rb = read_recording(Path(a["dir"])), read_recording(Path(b["dir"]))
    assert ra["meta"]["complete"] and rb["meta"]["complete"]
    assert [f["i"] for f in ra["frames"]] == [1, 2, 3, 4, 5] and [f["i"] for f in rb["frames"]] == [7, 8, 9, 10]
    assert ra["meta"]["video"]["every"] == 3 and [p.stem for p in ra["video"]] == ["1", "4"]
    assert rb["meta"]["video"]["every"] == 2 and len(rb["video"]) == 2


def test_camera_images_stop_at_the_size_cap_but_everything_else_goes_on(eyes, monkeypatch):
    session = eyes
    monkeypatch.setattr(recorder_mod, "MAX_BYTES", 120_000)
    reply = session.handle_command({"type": "recording", "on": True, "video": {"every": 1}})[0]
    for _ in range(30):
        session.frame()
    session.handle_command({"type": "recording", "on": False})
    rec = read_recording(Path(reply["dir"]))
    stopped = rec["meta"]["video"]["stopped"]
    assert rec["meta"]["video"]["max_bytes"] == 120_000 and "reached" in stopped["reason"]
    assert 1 <= len(rec["video"]) < 30 and int(rec["video"][-1].stem) == stopped["i"]
    assert [f["video"] for f in rec["frames"]] == [f["i"] <= stopped["i"] for f in rec["frames"]]
    assert len(rec["frames"]) == 30 and sum(len(c["ids"]) for c in rec["chunks"]) == 30
    assert [e["data"]["i"] for e in rec["events"] if e["type"] == "video_stopped"] == [stopped["i"]]


def test_no_camera_images_on_a_nearly_full_disk(eyes, monkeypatch):
    session = eyes
    monkeypatch.setattr(recorder_mod, "MIN_FREE_BYTES", 1 << 62)
    reply = session.handle_command({"type": "recording", "on": True})[0]
    for _ in range(4):
        session.frame()
    session.handle_command({"type": "recording", "on": False})
    rec = read_recording(Path(reply["dir"]))
    assert "free on the disk" in rec["meta"]["video"]["stopped"]["reason"]
    assert not rec["video"] and len(rec["frames"]) == 4 and not any(f["video"] for f in rec["frames"])


def test_other_faces_in_view_go_into_the_landmarks(eyes):
    session = eyes
    reply = session.handle_command({"type": "recording", "on": True})[0]
    boxes = [[0.1, 0.2, 0.3, 0.4], {"box": [0.5, 0.5, 0.7, 0.8], "score": 0.9}, "not a box", [0, 0, 1, 1],
             [0.2, 0.2, 0.4, 0.4]]
    for i, others in enumerate((None, boxes, boxes[:1], 2)):
        msg = {"type": "frame", "id": i + 1, "face": True, **({} if others is None else {"others": others})}
        session._recorder.frame(100.0 + i, {"id": i + 1}, None, [msg], np.zeros(28), MESH)
    session.handle_command({"type": "recording", "on": False})
    rec = read_recording(Path(reply["dir"]))
    [chunk] = rec["chunks"]
    o = chunk["others"].astype(float)
    assert chunk["others"].dtype == np.float16 and o.shape == (4, 3, 4)
    assert np.isnan(o[0]).all() and np.isnan(o[3]).all()                 # none, or not boxes
    np.testing.assert_allclose(o[1], [[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.7, 0.8], [0, 0, 1, 1]], atol=1e-3)
    np.testing.assert_allclose(o[2, 0], [0.1, 0.2, 0.3, 0.4], atol=1e-3)
    assert np.isnan(o[2, 1:]).all()
    assert rec["frames"][1]["msg"]["others"][1] == {"box": [0.5, 0.5, 0.7, 0.8], "score": 0.9}


def test_a_slow_disk_costs_camera_images_first_then_frames(eyes, monkeypatch):
    session = eyes
    monkeypatch.setattr(recorder_mod, "VIDEO_BACKLOG", 0)
    reply = session.handle_command({"type": "recording", "on": True, "video": {"every": 1}})[0]
    for _ in range(3):
        session.frame()
    monkeypatch.setattr(recorder_mod, "MAX_BACKLOG", 0)
    for _ in range(2):
        session.frame()
    session.handle_command({"type": "recording", "on": False})
    rec = read_recording(Path(reply["dir"]))
    assert [f["i"] for f in rec["frames"]] == [1, 2, 3] and not rec["video"]
    assert rec["meta"]["summary"]["dropped"] == 2


def test_a_recording_problem_never_breaks_tracking(eyes, monkeypatch):
    session = eyes
    calibrate(session)
    working = recorder_mod.fit_width

    def disk_full(*args):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(recorder_mod, "fit_width", disk_full)
    first = session.handle_command({"type": "recording", "on": True, "video": {"every": 1}})[0]
    for _ in range(3):
        assert session.frame(800, 400)[0]["gaze"] is not None
    wait_until(lambda: not session.handle_command({"type": "recording"})[0]["on"])
    status = session.handle_command({"type": "recording"})[0]
    assert status["ok"] and "No space left" in status["error"]
    for _ in range(3):
        out = session.frame(800, 400)
        assert out[0]["face"] and out[0]["gaze"] is not None and "error" not in out[0]
    session.frame(800, 400, closure=0.9)
    # A problem while describing a frame (here: the blink state) does the same.
    monkeypatch.setattr(recorder_mod, "fit_width", working)
    monkeypatch.setattr(recorder_mod, "_blink_state", lambda detector, face: 1 / 0)
    second = session.handle_command({"type": "recording", "on": True})[0]
    assert second["on"] and second["id"] != first["id"]
    assert session.frame(800, 400)[0]["gaze"] is not None
    status = session.handle_command({"type": "recording"})[0]
    assert status["on"] is False and "ZeroDivisionError" in status["error"]
    session.handle_command({"type": "recording", "on": False})
    meta = read_recording(Path(first["dir"]))["meta"]
    assert meta["complete"] is False and "No space left" in meta["error"]


def test_pictures_in_messages_are_left_out_and_odd_numbers_stay_strict_json(eyes):
    session = eyes
    reply = session.handle_command({"type": "recording", "on": True})[0]
    big = "data:image/jpeg;base64," + "A" * 5000
    session._push({"type": "faceprint_faces", "faces": [{"id": 1, "src": big}], "note": "data:short"})
    session._push({"type": "odd", "nan": float("nan"), "values": np.array([1.5, np.inf]), "n": np.int64(3)})
    session.handle_command({"type": "recording", "on": False})
    rec = read_recording(Path(reply["dir"]))                                 # (parses strictly)
    pushed = next(e for e in rec["events"] if e["type"] == "faceprint_faces")["data"]
    assert pushed["faces"][0]["src"].startswith("data:image/jpeg;base64,...") and "not recorded" in \
        pushed["faces"][0]["src"] and pushed["note"] == "data:short"
    assert session.pushed[-2]["faces"][0]["src"] == big                     # the page still gets it
    odd = next(e for e in rec["events"] if e["type"] == "odd")["data"]
    assert odd == {"type": "odd", "nan": None, "values": [1.5, None], "n": 3}


# -- hand mode ----------------------------------------------------------------------------------

def test_a_hand_session_is_recorded(tmp_path):
    tracker = FakeHandTracker()
    clock = ManualClock()
    session = TrackerSession(lambda: tracker, UserStore(tmp_path / "data"), clock=clock, mode="hand",
                             recordings=tmp_path / "recordings")
    session.handle_command({"type": "hello", "screen": {"w": 1600, "h": 900, "dpr": 1}})
    reply = session.handle_command({"type": "recording", "on": True, "video": {"every": 1}})[0]
    assert re.fullmatch(r"\d{8}-\d{6}-Person_1-hand", reply["id"])
    image = camera_jpeg(640, 360)                     # fits: kept exactly as the browser sent it
    hands, state = [], {"id": 0}

    def frame(pts):
        state["id"] += 1
        tracker.push(pts)
        if pts is not None:
            hands.append(np.asarray(pts))
        out = session.handle_frame(pack_frame({"id": state["id"]}, image))
        clock.tick()
        return out

    for _ in range(5):
        frame(make_hand(tip=(0.40, 0.40), pinch=1.2))
    for k in range(1, 6):                             # a pinch: a click
        frame(make_hand(tip=(0.40 + 0.01 * k, 0.40), pinch=1.0 - 0.17 * k))
    frame(make_hand(tip=(0.45, 0.40), pinch=1.0))
    frame(None)
    session.close()

    rec = read_recording(Path(reply["dir"]))
    meta, frames = rec["meta"], rec["frames"]
    assert meta["mode"] == "hand" and meta["hand"] is None and meta["complete"]
    assert set(meta["detectors"]) == {"hand"} and meta["detectors"]["hand"]["calibration"] is None
    assert len(frames) == 12 and all(set(f) == FRAME_KEYS for f in frames)
    assert all(f["features"] is None and f["blink"] is None and f["wink"] is None for f in frames)
    assert all("pinching" in f["msg"] for f in frames) and not frames[-1]["msg"]["face"]
    assert sum("double_blink" in f["events"] for f in frames) == 1
    assert {"blink", "double_blink"} <= {e["type"] for e in rec["events"] if e["kind"] == "event"}
    [chunk] = rec["chunks"]
    assert chunk["ids"].tolist() == list(range(1, 12)) and chunk["points"].shape == (11, 21, 3)
    np.testing.assert_allclose(chunk["points"].astype(float), np.array(hands), atol=1e-3)
    assert len(rec["video"]) == 12 and all(p.read_bytes() == image for p in rec["video"])


# -- camera images --------------------------------------------------------------------------------

def test_camera_images_are_kept_as_they_are_when_they_fit():
    small = camera_jpeg(640, 360)
    assert jpeg_size(small) == (640, 360) and jpeg_size(JPEG) == (960, 540)
    assert jpeg_size(b"not a jpeg") is None and jpeg_size(tiny_jpeg()) == (8, 8)
    assert fit_width(small, 640, 70) is small
    shrunk = fit_width(JPEG, 640, 70)
    assert jpeg_size(shrunk) == (640, 360) and len(shrunk) < len(JPEG)
    assert jpeg_size(fit_width(camera_jpeg(1920, 1080), 640, 70)) == (640, 360)
    ok, png = cv2.imencode(".png", np.zeros((10, 20, 3), np.uint8))
    assert jpeg_size(fit_width(png.tobytes(), 640, 70)) == (20, 10)          # always a JPEG
    assert VideoOptions.from_dict({"every": 0, "max_width": 99999, "quality": "best"}) == \
        VideoOptions(every=0, max_width=1920, quality=70)
    assert VideoOptions.from_dict(False).every == 0 and VideoOptions.from_dict(None) == VideoOptions()


# -- the server and the command line ---------------------------------------------------------------

def test_python_m_paralic_record_records_every_session(monkeypatch, tmp_path):
    patch_session(monkeypatch)
    tracker = FakeTracker()
    app = create_app(data_dir=tmp_path, tracker_factory=lambda: tracker, record_all=True)
    with TestClient(app, base_url="http://localhost:8000", client=("127.0.0.1", 50000)) as client:
        assert client.get("/api/status").json()["recording"] is True
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "hello", "screen": {"w": 1920, "h": 1080, "dpr": 1}}))
            hello = json.loads(ws.receive_text())
            started = json.loads(ws.receive_text())
            assert hello["type"] == "hello" and started["type"] == "recording" and started["on"]
            tracker.push(fake_features(VirtualUser(seed=1).features(960, 540, Head())))
            ws.send_bytes(pack_frame({"id": 1}, JPEG))
            assert json.loads(ws.receive_text())["face"] is True
        path = Path(started["dir"])
        assert path.parent == tmp_path / "recordings"
        wait_until(lambda: _strict((path / "meta.json").read_text())["complete"])   # closed with the socket
    rec = read_recording(path)
    assert rec["meta"]["summary"]["frames"] == 1 and len(rec["video"]) == 1


def test_sessions_are_not_recorded_unless_asked(monkeypatch, tmp_path):
    patch_session(monkeypatch)
    app = create_app(data_dir=tmp_path, tracker_factory=FakeTracker, recordings_dir=tmp_path / "elsewhere")
    with TestClient(app, base_url="http://localhost:8000", client=("127.0.0.1", 50000)) as client:
        assert client.get("/api/status").json()["recording"] is False
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "hello", "screen": {"w": 1920, "h": 1080, "dpr": 1}}))
            assert json.loads(ws.receive_text())["type"] == "hello"
            ws.send_text(json.dumps({"type": "recording"}))
            assert json.loads(ws.receive_text()) == {"type": "recording", "ok": True, "on": False}
            ws.send_text(json.dumps({"type": "recording", "on": True}))
            started = json.loads(ws.receive_text())
            assert started["on"] and Path(started["dir"]).parent == tmp_path / "elsewhere"
        path = Path(started["dir"])
        wait_until(lambda: _strict((path / "meta.json").read_text())["complete"])
    assert not (tmp_path / "recordings").exists()


def test_command_line_options(monkeypatch, tmp_path):
    import uvicorn

    import paralic.__main__ as cli

    def offline(path):
        raise cli.ModelUnavailable("offline")

    seen = {}
    monkeypatch.setattr(cli, "ensure_face_model", offline)
    monkeypatch.setattr(cli, "create_app", lambda **kw: seen.update(kw))
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    base = ["--no-browser", "--no-window", "--data-dir", str(tmp_path), "--port", "18765"]
    cli.main(base)
    assert seen["record_all"] is False and seen["recordings_dir"] == tmp_path / "recordings"
    cli.main(base + ["--record", "--recordings-dir", str(tmp_path / "rec")])
    assert seen["record_all"] is True and seen["recordings_dir"] == tmp_path / "rec"


# -- the page ---------------------------------------------------------------------------------------

def test_the_rec_button_and_shift_r_record_the_session(tmp_path):
    """The page's controls in Chromium (fake webcam; the fake tracker finds no face)."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    from tests.test_browser import CHROMIUM_CANDIDATES, _serve

    app = create_app(data_dir=tmp_path, tracker_factory=FakeTracker)
    server, thread, url = _serve(app)
    try:
        with sync_playwright() as p:
            browser = None
            for path in [None, *[c for c in CHROMIUM_CANDIDATES if c]]:
                try:
                    browser = p.chromium.launch(executable_path=path, args=[
                        "--no-sandbox", "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
                    break
                except Exception:  # pragma: no cover - depends on the machine
                    continue
            if browser is None:
                pytest.skip("Chromium not available")
            ctx = browser.new_context(viewport={"width": 1600, "height": 900})
            ctx.grant_permissions(["camera"], origin=url)
            page = ctx.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"{url}/?demo")
            page.wait_for_function("window.paralic && window.paralic.state.started")
            assert page.locator(".rec-btn").count() == 0                   # demo mode: no camera to record
            page.goto(f"{url}/?nofs&manual")
            page.click("#start-btn")
            page.wait_for_function("window.paralic.tracker && window.paralic.tracker.connected")
            button = page.locator("#topbar .rec-btn")
            assert button.count() == 1 and "gaze-ignore" not in button.get_attribute("class")   # usable by gaze
            page.keyboard.press("Shift+R")
            page.wait_for_selector(".rec-btn.recording")
            assert "camera images included" in page.locator(".toast").first.text_content()
            page.wait_for_function("document.querySelector('.rec-info').textContent.includes('MB')")
            wait_until(lambda: len(list((tmp_path / "recordings").glob("*/video/*.jpg"))) >= 2)
            # The camera check is still on screen: close it to reach the button, as a helper would.
            page.evaluate("document.querySelector('#overlay-root').innerHTML = ''; window.paralic.appEl.inert = false")
            button.click()                    # it asks first (a stray double blink must not stop it)
            assert page.locator(".rec-btn.recording").count() == 1
            page.locator(".rec-confirm .btn", has_text="Stop recording").click()
            page.wait_for_selector(".rec-btn:not(.recording):not(.pending)")
            page.wait_for_function(
                "[...document.querySelectorAll('.toast')].some(t => t.textContent.includes('saved'))")
            assert errors == []
            ctx.close()
            browser.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    [path] = (tmp_path / "recordings").iterdir()
    rec = read_recording(path)
    assert rec["meta"]["complete"] and rec["meta"]["screen"]["w"] == 1600
    assert len(rec["frames"]) >= 3 and len(rec["video"]) >= 2
    assert jpeg_size(rec["video"][0].read_bytes())[0] <= 640
    commands = [(e["type"], e["data"].get("on")) for e in rec["events"] if e["kind"] == "command"]
    assert commands == [("recording", True), ("recording", False)]                # polls are not recorded


# -- the real face model -----------------------------------------------------------------------

def test_real_landmarks_and_camera_images_are_recorded(tmp_path):
    from tests import face_images

    model = ROOT / "models" / "face_landmarker.task"
    if not model.exists():
        pytest.skip("face landmarker model not downloaded")
    portrait = face_images.portrait(960)
    if portrait is None:
        pytest.skip("test portrait unavailable (offline?)")
    from paralic.tracker import FaceTracker

    model_bytes = model.read_bytes()
    session = TrackerSession(lambda: FaceTracker(model_bytes), UserStore(tmp_path / "data"),
                             recordings=tmp_path / "recordings")
    session.handle_command({"type": "hello", "screen": {"w": 1920, "h": 1080, "dpr": 1}})
    reply = session.handle_command({"type": "recording", "on": True, "video": {"every": 1}})[0]
    ok, buf = cv2.imencode(".jpg", portrait, [cv2.IMWRITE_JPEG_QUALITY, 82])
    for i in range(1, 7):
        assert session.handle_frame(pack_frame({"id": i}, buf.tobytes()))[0]["face"]
    session.close()

    rec = read_recording(Path(reply["dir"]))
    h, w = portrait.shape[:2]
    [chunk] = rec["chunks"]
    points = chunk["points"].astype(float)
    assert chunk["points"].dtype == np.float16 and points.shape == (6, 478, 3)
    assert chunk["ids"].tolist() == [1, 2, 3, 4, 5, 6]
    assert np.isfinite(points).all() and (points[..., 0] > 0).all() and (points[..., 0] < w).all()
    assert (points[..., 1] > 0).all() and (points[..., 1] < h).all()
    # The face is where the test portrait has it, and the same in every frame.
    tracker = FaceTracker(model_bytes)
    try:
        found = tracker.process(np.ascontiguousarray(cv2.cvtColor(portrait, cv2.COLOR_BGR2RGB)), 0).points_px
    finally:
        tracker.close()
    assert np.abs(points[0, :, :2] - found[:, :2]).max() < 6
    assert np.ptp(points[:, :, :2], axis=0).max() < 6
    assert all(len(f["features"]) == 28 and f["blink"]["signal"] is not None for f in rec["frames"])
    # The camera images: the frames shrunk to 640 px.
    assert len(rec["video"]) == 6
    img = cv2.imread(str(rec["video"][0]))
    assert img.shape[1] == 640 and img.shape[0] == round(h * 640 / w)
    expected = cv2.resize(portrait, (640, img.shape[0]), interpolation=cv2.INTER_AREA)
    assert np.abs(img.astype(float) - expected).mean() < 6
