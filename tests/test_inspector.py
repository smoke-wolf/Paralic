"""The Paralic Inspector: the recording reader, the gaze networks' internals, the API and the page.

The recordings are made by ``tools/make_demo_recording.py`` (once per test
session), written as ``paralic/recorder.py`` writes them: a hand-mode one from
drawn hands, and an eye-mode one from the real MediaPipe face model on the
test portrait with simulated gaze features through a real calibration. The
eye-mode tests are skipped when the face model or the test portrait is not
available.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from paralic.gazenet import GazeNet
from paralic.inspector import internals
from paralic.inspector.server import create_app
from paralic.inspector.views import RecordingView
from paralic.recording import Recording, _strip_overlays, list_recordings, parse_name, recordings_root

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "face_landmarker.task"
# The session rounds the gaze it sends to 0.1 px and the recording keeps the features to 6
# decimals: the gaze computed again from them lies this close to the recorded one.
CHECK_PX = 0.1
# A session that notices glasses itself (paralic/glasses.py) reports what it sees in the painted
# ones; otherwise the demo fills in what a session that notices them would report.
SESSION_SEES_GLASSES = importlib.util.find_spec("paralic.glasses") is not None


def _demo_tool():
    spec = importlib.util.spec_from_file_location("make_demo_recording", ROOT / "tools" / "make_demo_recording.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module       # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def demos(tmp_path_factory):
    """A data folder with the demo recordings: {"data", "root", "hand", "eyes" (None if unavailable)}."""
    data = tmp_path_factory.mktemp("inspector-data")
    root = data / "recordings"
    tool = _demo_tool()
    hand = tool.make_demo(root, mode="hand", quick=True, start_wall=1790834400.0)
    eyes = None
    if MODEL.exists():
        try:
            # Analysed frames 640 wide, images stored 480 wide: the overlay must be scaled.
            eyes = tool.make_demo(root, mode="eyes", quick=True, width=640, video_width=480, start_wall=1790833800.0)
        except SystemExit:                  # the test portrait could not be downloaded
            eyes = None
    return {"data": data, "root": root, "hand": hand, "eyes": eyes}


@pytest.fixture
def eyes(demos):
    if demos["eyes"] is None:
        pytest.skip("face model or test portrait unavailable")
    return demos["eyes"]


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# -- the reader -----------------------------------------------------------------------------------

def test_recordings_are_listed_with_their_summaries(demos, eyes):
    assert recordings_root(demos["data"]) == demos["root"]
    rows = list_recordings(demos["root"])
    assert [r["id"] for r in rows] == [demos["hand"].name, eyes.name]          # newest first
    hand, eye = rows
    assert hand["mode"] == "hand" and hand["person"] == "Alex" and hand["glasses"] is None
    assert hand["models"] == 0 and hand["accuracy"] is None and hand["complete"] and not hand["live"]
    assert hand["video_frames"] == (hand["frames"] + 1) // 2 and hand["video_every"] == 2   # every second image
    assert eye["mode"] == "eyes" and eye["person"] == "Sam"
    assert eye["glasses"] is True or (SESSION_SEES_GLASSES and eye["glasses"] is False)
    assert eye["models"] == 3 and 5 < eye["accuracy"]["mean_error_px"] < 200
    assert eye["frames"] == eye["video_frames"] > 1000 and 40 < eye["duration_s"] < 90
    assert eye["bytes"] > 1_000_000 and not eye["truncated"] and eye["dropped"] == 0
    assert eye["recorder_error"] is None and eye["video_stopped"] is None
    assert parse_name(eyes.name)["mode"] == "eyes"
    assert parse_name("20261001-142233-Sam_Lee-hand-2") == {"started": "2026-10-01T14:22:33", "person": "Sam_Lee",
                                                            "mode": "hand"}


def test_frames_are_read_by_index_time_and_id(eyes):
    rec = Recording(eyes)
    lines = _lines(eyes / "frames.jsonl")
    assert len(rec) == len(lines) == rec.meta["summary"]["frames"]
    keys = {"i", "t", "wall", "msg", "features", "blink", "wink", "net", "label", "stored", "events", "video"}
    assert all(set(f) - {"gesture"} == keys for f in lines)             # as docs/recording-format.md lists them
    assert np.array_equal(rec.times, [f["t"] for f in lines])
    assert np.array_equal(rec.frame_ids, [f["i"] for f in lines])
    assert rec.frame(123) == lines[123] and rec.frame(-1) == lines[-1]
    assert rec.frames(500, 510) == lines[500:510]
    assert list(rec.iter_frames(block=700)) == lines
    t = lines[800]["t"]
    assert rec.index_at(t) == 800 and rec.index_at(t + 0.001) == 800 and rec.index_at(t - 0.001) == 799
    lo, hi = rec.range_of(lines[100]["t"], lines[199]["t"])
    assert (lo, hi) == (100, 200) and rec.frames_between(lines[100]["t"], lines[199]["t"]) == lines[100:200]
    assert rec.find(lines[42]["i"]) == 42 and rec.find(10 ** 9) is None
    # Without the drawing aids, the rest of the frame is the same.
    light = rec.frames(0, 200, light=True)
    for full, lite in zip(lines[:200], light):
        assert "mesh" not in lite["msg"] and "eyes" not in lite["msg"]
        assert {k: v for k, v in full["msg"].items() if k not in ("mesh", "eyes")} == lite["msg"]
    kinds = {e["kind"] for e in rec.events()}
    assert kinds == {"command", "reply", "event", "push"}


def test_landmarks_video_and_models(eyes):
    rec = Recording(eyes)
    frames = rec.frames()
    with_face = [f for f in frames if f["msg"].get("face")]
    without = [f for f in frames if not f["msg"].get("face")]
    assert with_face and without
    lm = rec.landmarks(with_face[0]["i"])
    assert lm["points"].shape == (478, 3)
    assert rec.landmarks(without[0]["i"]) is None
    crowded = [f for f in frames if f["msg"].get("faces") == 2]
    assert crowded and rec.landmarks(crowded[0]["i"])["others"].shape == (1, 4)
    vid, jpeg = rec.video(frames[10]["i"])
    assert vid == frames[10]["i"] and jpeg[:2] == b"\xff\xd8"
    # The models, from when their "model" events say.
    models = rec.models()
    assert [m.why for m in models] == ["calibration", "refit", "adjust"]
    announced = [e for e in rec.events() if e["type"] == "model"]
    assert [m.t for m in models] == [e["t"] for e in announced] and all(m.t_source == "event" for m in models)
    assert [f"models/{m.name}" for m in models] == [e["data"]["file"] for e in announced]
    assert rec.model_at(rec.times[0]) is None
    assert rec.model_at(models[1].t + 0.01).name == models[1].name
    assert rec.model_at(rec.times[-1]).name == models[2].name
    assert isinstance(models[2].net, GazeNet) and models[2].info()["corrected"]


def test_hand_recording_landmarks_and_nearest_earlier_video(demos):
    rec = Recording(demos["hand"])
    assert rec.mode == "hand" and rec.models() == []
    frames = rec.frames()
    shown = next(f for f in frames if f["msg"].get("face"))
    assert rec.landmarks(shown["i"])["points"].shape == (21, 3)
    odd = next(f for f in frames if not f["video"])
    vid, _ = rec.video(odd["i"])
    assert vid < odd["i"] and rec.video(odd["i"], earlier=False) is None


def _small_copy(src: Path, dst: Path, frames: int = 400) -> Path:
    """The first frames of a recording with its events and models (no video, no landmarks)."""
    dst.mkdir(parents=True)
    shutil.copy2(src / "meta.json", dst / "meta.json")
    with open(src / "frames.jsonl", encoding="utf-8") as fh, open(dst / "frames.jsonl", "w", encoding="utf-8") as out:
        for k, line in enumerate(fh):
            if k >= frames:
                break
            out.write(line)
    shutil.copy2(src / "events.jsonl", dst / "events.jsonl")
    if (src / "models").is_dir():
        shutil.copytree(src / "models", dst / "models")
    return dst


def test_a_crash_leaves_a_readable_recording(demos, tmp_path):
    rec_dir = _small_copy(demos["hand"], tmp_path / "20261001-090000-Alex-hand")
    meta = json.loads((rec_dir / "meta.json").read_text())
    meta.update(ended=None, complete=False)     # as the recorder wrote it at the start
    meta.pop("summary")
    (rec_dir / "meta.json").write_text(json.dumps(meta))
    full = (rec_dir / "frames.jsonl").read_bytes()
    lines = full.splitlines(keepends=True)
    # The process died while writing the last frame and an event.
    (rec_dir / "frames.jsonl").write_bytes(b"".join(lines[:-1]) + lines[-1][:57])
    with open(rec_dir / "events.jsonl", "ab") as fh:
        fh.write(b'{"t": 99999.0, "wall": 1, "kind": "event", "type": "bl')
    rec = Recording(rec_dir, refresh_s=0.0)
    assert len(rec) == len(lines) - 1 and rec.truncated
    assert rec.bad_lines["events"] == 1 and all(e["type"] != "bl" for e in rec.events())
    assert rec.video(rec.frame_ids[5]) is None and rec.landmarks(rec.frame_ids[5]) is None
    summary = rec.summary()
    assert not summary["complete"] and summary["truncated"] and summary["frames"] == len(lines) - 1
    assert summary["live"]                      # written just now: probably still recording
    for name in ("meta.json", "frames.jsonl", "events.jsonl"):
        os.utime(rec_dir / name, (time.time() - 600, time.time() - 600))
    assert not rec.summary()["live"]            # nothing written for minutes: it stopped
    # The recording goes on: the line is completed and more frames arrive.
    with open(rec_dir / "frames.jsonl", "ab") as fh:
        fh.write(lines[-1][57:])
        nxt = json.loads(lines[-1])
        nxt.update(i=nxt["i"] + 1, t=nxt["t"] + 0.0333)
        fh.write(json.dumps(nxt).encode() + b"\n")
    assert rec.refresh(force=True)
    assert len(rec) == len(lines) + 1 and not rec.truncated
    assert rec.frame(-1)["i"] == nxt["i"] and rec.frame(-2) == json.loads(lines[-1])
    # A complete object without its newline (written, then the crash) is read.
    with open(rec_dir / "frames.jsonl", "ab") as fh:
        last = dict(nxt, i=nxt["i"] + 1, t=nxt["t"] + 0.0333)
        fh.write(json.dumps(last).encode())
    rec.refresh(force=True)
    assert len(rec) == len(lines) + 2 and rec.frame(-1)["i"] == last["i"]
    # Another key order, and a damaged line in the middle, are handled too.
    with open(rec_dir / "frames.jsonl", "ab") as fh:
        fh.write(b"\n{oops\n")
        fh.write(json.dumps({"t": last["t"] + 0.0333, "msg": {}, "i": last["i"] + 1}).encode() + b"\n")
    rec.refresh(force=True)
    assert len(rec) == len(lines) + 3 and rec.frame(-1)["i"] == last["i"] + 1 and rec.bad_lines["frames"] == 1


def test_missing_parts_are_tolerated(demos, tmp_path):
    rec_dir = tmp_path / "20261001-091500-Alex-hand"
    rec_dir.mkdir()
    shutil.copy2(demos["hand"] / "frames.jsonl", rec_dir / "frames.jsonl")
    rec = Recording(rec_dir)
    assert rec.meta == {} and rec.mode == "hand" and rec.person == "Alex"
    assert rec.events() == [] and rec.models() == [] and len(rec.video_ids()) == 0
    assert rec.summary()["started"] == "2026-10-01T09:15:00"
    # Only a meta.json (the recorder just started).
    empty = tmp_path / "20261001-092000-Sam-eyes"
    empty.mkdir()
    (empty / "meta.json").write_text(json.dumps({"format": 1, "mode": "eyes", "person": {"name": "Sam"}}))
    rec = Recording(empty)
    assert len(rec) == 0 and rec.span() == (None, None) and rec.summary()["frames"] == 0
    view = RecordingView(rec)
    assert view.signals()["cols"] == {} and view.timeline()["markers"] == []
    # Not strict JSON (NaN, as Python writes it by default): read, and sent to the page as null.
    odd = tmp_path / "odd" / "20261001-093000-Sam-eyes"
    odd.mkdir(parents=True)
    (odd / "frames.jsonl").write_text('{"i": 1, "t": 5.0, "msg": {"face": true, "cl": NaN}}\n')
    frames = TestClient(create_app(tmp_path / "odd"), base_url="http://localhost:8100").get(f"/api/recordings/{odd.name}/frames").json()["frames"]
    assert frames[0]["msg"] == {"face": True, "cl": None}


def test_model_times_without_model_events(eyes, tmp_path):
    rec_dir = _small_copy(eyes, tmp_path / "copy", frames=1700)
    first = sorted((rec_dir / "models").iterdir())
    events = _lines(rec_dir / "events.jsonl")

    def keep_events(keep):
        (rec_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events if keep(e)))

    # Without the model events, a time in the file places it (a recorder may wrap the model with it)...
    keep_events(lambda e: e["type"] != "model")
    doc = json.loads(first[1].read_text())
    first[1].write_text(json.dumps({"t": 3700.5, "why": "refit", "model": doc}))
    rec = Recording(rec_dir)
    refit = next(m for m in rec.models() if m.why == "refit")
    assert refit.t == 3700.5 and refit.t_source == "file" and refit.extra["why"] == "refit"
    # ...or the reply that reported the change...
    first[1].write_text(json.dumps(doc))
    rec = Recording(rec_dir)
    replies = [e for e in events if e["type"] in ("calibration_result", "validation_result")]
    assert [(m.t, m.t_source) for m in rec.models()[:2]] == [(replies[0]["t"], "reply"), (replies[1]["t"], "reply")]
    # ...or the file's modification time (written when the model changed)...
    keep_events(lambda e: e["type"] != "model" and e["kind"] != "reply")
    wall = json.loads((rec_dir / "frames.jsonl").read_text().splitlines()[1000])["wall"]
    os.utime(first[1], (wall, wall))
    rec = Recording(rec_dir)
    refit = next(m for m in rec.models() if m.why == "refit")
    assert refit.t_source == "mtime" and abs(refit.t - rec.times[1000]) < 0.01
    # ...and with nothing to go by, it follows the one before it.
    for p in first:
        os.utime(p, (1.0, 1.0))
    rec = Recording(rec_dir)
    for m in rec.models():
        m.doc.get("meta", {}).pop("trained_ts", None)
    rec._place_models()
    assert [m.t_source for m in rec.models()] == ["start", "order", "order"]


def test_a_model_event_without_a_file_means_no_model(eyes, tmp_path):
    rec_dir = _small_copy(eyes, tmp_path / "20261001-142233-Sam-eyes-2", frames=1700)
    rec = Recording(rec_dir)
    t_gone = float(rec.times[1600])
    assert rec.model_at(t_gone) is not None
    with open(rec_dir / "events.jsonl", "a", encoding="utf-8") as fh:     # another person, without a model
        fh.write(json.dumps({"t": t_gone, "wall": 0.0, "kind": "event", "type": "model",
                             "data": {"why": "user_select", "file": None, "version": None}}) + "\n")
    rec = Recording(rec_dir)
    assert rec.model_changes()[-1] == (t_gone, None) and len(rec.models()) == 3
    assert rec.model_at(t_gone - 0.01) is rec.models()[-1] and rec.model_at(t_gone) is None
    view = RecordingView(rec)
    assert view.detail(1650)["model"] is None and "internals" not in view.detail(1650)
    assert any(m["type"] == "no model" and abs(m["t"] - t_gone) < 1e-3 for m in view.timeline()["markers"])
    assert parse_name(rec_dir.name) == {"started": "2026-10-01T14:22:33", "person": "Sam", "mode": "eyes"}


def test_strip_overlays_only_cuts_what_it_recognises():
    eyes = {"r": [[0.1, 0.2]], "l": [[0.3, 0.2]], "ri": [0.1, 0.2, 0.01], "box": [0, 0, 1, 1]}
    mesh = {"pts": [[0.5, 0.5]], "lines": [[1, 2]], "iris": [[468, 469]]}
    for separators in ((",", ":"), (", ", ": ")):             # the recorder's, and json.dumps' default
        line = json.dumps({"i": 1, "t": 2.0, "msg": {"type": "frame", "eyes": eyes, "mesh": mesh, "face": True}},
                          separators=separators).encode()
        assert json.loads(_strip_overlays(line)) == {"i": 1, "t": 2.0, "msg": {"type": "frame", "face": True}}
        first = json.dumps({"msg": {"eyes": eyes, "face": True}}, separators=separators).encode()
        assert json.loads(_strip_overlays(first)) == {"msg": {"face": True}}
    # The recorder keeps the mesh as "mesh": true; what does not end as expected is left alone.
    for line in (b'{"msg":{"mesh":true,"face":true}}', b'{"msg":{"mesh":{"pts":[]}}}', b'{"msg":{"eyes":{"box":[1}}'):
        assert _strip_overlays(line) == line


# -- the networks' internals ---------------------------------------------------------------------

def test_internals_equal_gazenet_predict(eyes):
    """Every step is GazeNet's own: the final point is GazeNet.predict, bit for bit."""
    rec = Recording(eyes)
    checked = recorded = 0
    for f in rec.iter_frames():
        if not f.get("features"):
            continue
        snap = rec.model_at(f["t"])
        if snap is None:
            continue
        model = snap.net
        for eye, member in model.members():
            trace = internals.trace(member, f["features"])
            assert np.array_equal(trace["final"], model.predict(f["features"], eye)[0])
            assert np.array_equal(model.predict_uncorrected(f["features"], eye)[0], trace["ensemble_px"])
            checked += 1
        # The recorded raw gaze is the leading network's output, except while a wink hands the cursor
        # to one eye's network, which the session shifts to line up with the usual one.
        raw = f["msg"].get("raw")
        if raw and f["wink"]["winking"] is None and f["net"] == model.eye:
            member = model.member(f["net"])
            out = internals.trace(member, f["features"])["final"]
            # Rounded: the gaze to 0.1 px, each feature to 6 decimals (off by at most 5e-7).
            slack = 5e-7 * np.abs(internals.jacobian(member, f["features"])["dx"]).sum(axis=0)
            assert (np.abs(out - raw) <= 0.05 + slack + 1e-9).all()
            recorded += 1
    assert checked > 2000 and recorded > 500


def test_trace_details_and_sensitivity(eyes):
    rec = Recording(eyes)
    f = next(f for f in rec.frames(1000, 1100) if f.get("features") and f["net"] == "both")
    model = rec.model_at(f["t"]).net
    x = np.asarray(f["features"])
    t = internals.trace(model, x)
    assert len(t["names"]) == 28 and t["names"][0] == "r_dx" and len(t["members"]) == 3
    m = t["members"][0]
    assert m["act1"].shape == (32,) and m["act2"].shape == (16,)
    assert np.allclose(np.tanh(m["pre1"]), m["act1"]) and np.allclose(m["skip_px"] + m["hidden_px"] + m["bias_px"], m["out_px"])
    assert np.allclose(m["skip_by_input_px"].sum(axis=0), m["skip_px"])
    assert np.allclose(m["hidden_by_unit_px"].sum(axis=0), m["hidden_px"])
    # The derivatives are exact: compare with central differences of predict.
    for eye, member in model.members():
        jac = internals.jacobian(member, x)
        cols = member.inputs or list(range(len(x)))
        for k, c in enumerate(cols):
            h = 1e-5 * member.x_scaler.std[k]
            up, down = x.copy(), x.copy()
            up[c] += h
            down[c] -= h
            fd = (model.predict(up, eye)[0] - model.predict(down, eye)[0]) / (2 * h)
            assert np.allclose(fd, jac["dx"][k], rtol=1e-4, atol=1e-3)
    rows = internals.sensitivity(model, x, top=5)
    assert len(rows) == 5 and rows[0]["strength"] >= rows[-1]["strength"]
    ab = internals.ablation(model, x)
    assert ab.shape == (28, 2)
    y = x.copy()
    k = int(np.argmax(np.hypot(*ab.T)))
    y[k] = model.x_scaler.mean[k]
    assert np.allclose(model.predict(x, "both")[0] - model.predict(y, "both")[0], ab[k])
    ex = internals.explain(model, x)
    assert set(ex["networks"]) == {"both", "left", "right"} and len(ex["networks"]["left"]["inputs"]) == 17
    batch = internals.batch_outputs(model, np.array([x, x]))
    assert np.allclose(batch["final"][0], model.predict(x, "both")[0])


def test_hand_trace_reads_the_recogniser_measurements():
    from tests.fakes import make_hand

    open_hand = internals.hand_trace(make_hand(fingers="spread"), 0.45, 0.6)
    assert open_hand["stop_palm"] and open_hand["palm_tests"]["fingers"]
    pinched = internals.hand_trace(make_hand(tip=(0.4, 0.3), pinch=0.3, fingers="curled"), 0.45, 0.6)
    assert not pinched["stop_palm"] and abs(pinched["pinch"] - 0.3) < 1e-6 and pinched["fingers"]["index"]


# -- the API --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(demos):
    return TestClient(create_app(demos["data"]), base_url="http://localhost:8100")


def test_api_lists_and_opens_recordings(client, demos, eyes):
    rows = client.get("/api/recordings").json()["recordings"]
    assert {r["id"] for r in rows} == {demos["hand"].name, eyes.name}
    info = client.get(f"/api/recordings/{eyes.name}").json()
    assert info["mode"] == "eyes" and info["screen"]["w"] == 1920 and len(info["models"]) == 3
    assert info["image"]["w"] == 640 and info["image"]["h"] == 360        # measured: the video is 480 wide
    assert len(info["feature_names"]) == 28 and info["mesh"]["nose_tip"] == 1
    index = client.get(f"/api/recordings/{eyes.name}/index").json()
    assert len(index["t"]) == len(index["i"]) == len(index["v"]) == info["frames"]
    frames = client.get(f"/api/recordings/{eyes.name}/frames", params={"start": 600, "count": 50, "light": True}).json()
    assert len(frames["frames"]) == 50 and all("eyes" not in f["msg"] for f in frames["frames"])
    assert client.get("/api/recordings/nope").status_code == 404
    assert client.get("/api/recordings/..").status_code == 404
    assert client.get("/api/recordings/%2E%2E%2Fetc").status_code == 404


def test_api_frame_detail_and_video(client, eyes):
    rec = Recording(eyes)
    n = next(k for k, f in enumerate(rec.frames(1000, 1200), 1000) if f.get("features") and f["net"] == "both")
    d = client.get(f"/api/recordings/{eyes.name}/frame/{n}").json()
    assert d["n"] == n and len(d["landmarks"]["points"]) == 478 and d["video"] == d["frame"]["i"]
    ins = d["internals"]
    assert ins["led"] == "both" and ins["check"]["distance"] <= CHECK_PX and ins["check"]["shifted"] is None
    both = ins["networks"]["both"]
    assert len(both["members"]) == 3 and len(both["members"][0]["act1"]) == 32 and len(both["sensitivity"]) == 12
    blink, wink = d["blink"], d["wink"]
    assert blink["close_thr"] > blink["open_thr"] > blink["baseline"] and blink["baseline_recorded"]
    assert blink["deep_thr"] > blink["close_thr"] and blink["watches"] == "both" and blink["pending"] in (0, 1)
    assert wink["state"] in ("idle", "candidate", "active") and len(wink["close_thr"]) == len(wink["base"]) == 2
    assert wink["close_thr"][0] > wink["open_thr"][0] > wink["base"][0] and wink["hold_ms"] > 0
    assert client.get(f"/api/recordings/{eyes.name}/frame/999999").status_code == 404
    res = client.get(f"/api/recordings/{eyes.name}/video/{d['frame']['i']}")
    assert res.status_code == 200 and res.headers["content-type"] == "image/jpeg"
    assert res.headers["x-frame-id"] == str(d["frame"]["i"])
    model = client.get(f"/api/recordings/{eyes.name}/models/{d['model']['name']}").json()
    assert model["model"]["nets"] and model["why"] == d["model"]["why"]


def test_api_signals_timeline_calibrations_config(client, demos, eyes):
    view = client.app.state.store.view(eyes.name)
    view.warm()
    deadline = time.time() + 60
    while view.progress() < 1 and time.time() < deadline:
        time.sleep(0.05)
    sig = client.get(f"/api/recordings/{eyes.name}/signals", params={"points": 200}).json()
    assert sig["decimated"] and len(sig["t"]) == 200 and not sig["partial"]
    closure = sig["cols"]["signal"]
    assert max(v for v in closure["max"] if v is not None) > 0.4            # blinks survive the buckets
    rec = Recording(eyes)
    window = client.get(f"/api/recordings/{eyes.name}/signals",
                        params={"t0": rec.times[700], "t1": rec.times[760], "points": 2000}).json()
    assert not window["decimated"] and len(window["t"]) == 63
    check = [v for v in window["cols"]["check"] if v is not None]
    assert check and max(check) <= CHECK_PX
    tl = client.get(f"/api/recordings/{eyes.name}/timeline").json()
    cats = {m["cat"] for m in tl["markers"]}
    assert {"blink", "click", "wink", "model", "setting", "calibration", "push"} <= cats and tl["complete"]
    kinds = {(s["kind"], s.get("value")) for s in tl["stretches"]}
    assert {("noface", None), ("faces", None), ("wink", "left"), ("label", "cal"), ("label", "head"),
            ("label", "val"), ("label", "adjust"), ("wink_test", "left"), ("wink_test", "right")} <= kinds
    assert SESSION_SEES_GLASSES or {("glasses", None), ("glare", None)} <= kinds
    states = [v for v in sig["cols"]["wstate"]["max"] if v is not None]
    assert max(states) == 2 and len(sig["cols"]["wclose_l"]["min"]) == 200        # a held wink: a press
    cal = client.get(f"/api/recordings/{eyes.name}/calibrations").json()
    full, adjust = cal["sessions"]
    assert full["mode"] == "full" and adjust["mode"] == "adjust"
    assert full["fit"]["cv_error_px"] > 0 and full["validation"]["mean_error_px"] > 0
    dots = {(d["kind"], d["pt"]): d for d in full["dots"]}
    assert len([k for k in dots if k[0] == "cal"]) == 13 and len([k for k in dots if k[0] == "head"]) == 6
    assert any(d["stored"] < d["frames"] for d in full["dots"])                  # the blink during a dot
    assert all(d["error"] is not None for d in full["dots"] if d["kind"] == "val")
    assert [a["kind"] for a in cal["accuracy"]] == ["cv", "validation", "adjust", "validation"]
    conf = client.get(f"/api/recordings/{eyes.name}/config").json()
    assert conf["meta"]["format"] == 1
    changes = [c for c in conf["changes"] if c["type"] == "settings"]
    assert changes[0]["changes"] == [{"key": "smoothing", "from": "auto", "to": "high"}]
    assert any(c["type"] == "gestures_set" and c["changes"] for c in conf["changes"])
    events = client.get(f"/api/recordings/{eyes.name}/events").json()["events"]
    assert len(events) == len(rec.events())
    hand = client.get(f"/api/recordings/{demos['hand'].name}/calibrations").json()
    assert hand["sessions"][0]["kind"] == "hand" and hand["sessions"][0]["fit"]["points"] == 13


def test_api_hand_mode(client, demos):
    name = demos["hand"].name
    info = client.get(f"/api/recordings/{name}").json()
    assert info["mode"] == "hand" and info["image"] is None and info["pinch"]["history"][-1]["off"] > 0.6
    rec = Recording(demos["hand"])
    n = next(k for k, f in enumerate(rec.frames()) if f["msg"].get("face") and k > 300)
    d = client.get(f"/api/recordings/{name}/frame/{n}").json()
    assert len(d["landmarks"]["points"]) == 21 and "hand" in d and d["pinch"]["on"] < d["pinch"]["off"]
    sig = client.get(f"/api/recordings/{name}/signals", params={"t0": rec.times[0], "t1": rec.times[-1]}).json()
    assert "pinch" in sig["cols"] and "net" not in sig["cols"]


def test_the_page_needs_nothing_from_the_internet(client):
    page = client.get("/")
    assert page.status_code == 200 and "js/app.js" in page.text
    static = ROOT / "paralic" / "inspector" / "static"
    for path in static.rglob("*"):
        if path.suffix in (".js", ".css", ".html"):
            text = path.read_text(encoding="utf-8")
            assert "http://" not in text.replace("http://www.w3.org/2000/svg", "") and "https://" not in text, path


# -- the page (Playwright + Chromium) ------------------------------------------------------------

CHROMIUM_CANDIDATES = [os.environ.get("CHROMIUM_PATH"), "/opt/pw-browsers/chromium"]


@pytest.fixture(scope="module")
def chromium():
    playwright_api = pytest.importorskip("playwright.sync_api")
    with playwright_api.sync_playwright() as p:
        last_error = None
        for path in [None, *[c for c in CHROMIUM_CANDIDATES if c]]:
            try:
                browser = p.chromium.launch(executable_path=path, args=["--no-sandbox"])
                break
            except Exception as exc:  # pragma: no cover - depends on the machine
                last_error = exc
        else:
            pytest.skip(f"Chromium not available: {last_error}")
        yield browser
        browser.close()


@pytest.fixture(scope="module")
def inspector_url(demos):
    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(demos["data"]), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://localhost:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def test_the_inspector_plays_steps_and_draws_every_panel(chromium, inspector_url, demos, eyes):
    ctx = chromium.new_context(viewport={"width": 1600, "height": 1000})
    page = ctx.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    try:
        page.goto(inspector_url)
        page.wait_for_selector(".rec-row")
        assert page.locator(".rec-row").count() == 2
        page.locator(".rec-row", has_text="Sam").click()
        page.wait_for_function("window.inspector.view && window.inspector.view.player")
        page.evaluate("window.inspector.view.player.go(1000)")
        page.wait_for_function("window.inspector.view.detail && window.inspector.view.detail.n === 1000 "
                               "&& window.inspector.view.detail.internals")
        # Play a second, then pause: about 30 frames on.
        page.keyboard.press("Space")
        page.wait_for_timeout(1000)
        page.keyboard.press("Space")
        n = page.evaluate("window.inspector.view.player.n")
        assert 1015 <= n <= 1045, n
        page.keyboard.press("ArrowRight")
        page.keyboard.press("ArrowRight")
        page.keyboard.press("ArrowLeft")
        assert page.evaluate("window.inspector.view.player.n") == n + 1
        page.wait_for_function(f"window.inspector.view.detail && window.inspector.view.detail.n === {n + 1}")
        page.wait_for_timeout(400)

        def painted(selector):
            return page.evaluate("""(sel) => {
                const c = document.querySelector(sel);
                if (!c || !c.width) return 0;
                const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
                let k = 0; for (let i = 3; i < d.length; i += 4 * 97) if (d[i] > 0) k++;
                return k; }""", selector)

        for canvas in (".cam-canvas", ".map-canvas", ".scrub-canvas", ".sig-canvas", ".inputs-canvas",
                       ".layers-canvas", ".scatter-canvas", ".cal-canvas", ".acc-canvas"):
            assert painted(canvas) > 50, canvas
        assert "recorded raw" in page.locator(".pipeline").inner_text().lower()
        assert page.locator(".pipe-step.good").count() == 1
        assert page.locator(".machine").count() == 2
        assert page.locator(".sens tbody tr").count() == 12
        assert page.locator(".cal-sessions .session").count() == 2
        page.wait_for_selector(".diag-list li")              # the diagnosis, in sentences
        assert page.locator(".diag-table tbody tr").count() == 2
        assert page.locator(".ev-table tr.ev").count() > 40
        assert page.locator(".json-tree .jl").count() > 5
        # Events: filter, then jump to one.
        page.locator(".events .chip", has_text="command").click()
        page.locator(".events .chip", has_text="reply").click()
        page.locator(".events .chip", has_text="push").click()
        row = page.locator(".ev-table tr.ev", has_text="double_blink").first
        row.click()
        t_event = page.evaluate("window.inspector.view.player.time")
        assert abs(t_event - page.evaluate("window.inspector.view.events.shown.find(e => e.type === 'double_blink').t")) < 1e-6
        # Hand mode.
        page.goto(f"{inspector_url}/#/rec/{demos['hand'].name}?n=400")
        page.wait_for_function("window.inspector.view && window.inspector.view.detail && window.inspector.view.detail.hand")
        page.wait_for_timeout(300)
        assert painted(".cam-canvas") > 50 and painted(".sig-canvas") > 50
        assert "Pinch detector" in page.locator("#sec-hood").inner_text()
        assert errors == []
    finally:
        ctx.close()


def test_only_this_computers_address_is_answered(tmp_path):
    """Recordings hold faces: a site that points its own domain at 127.0.0.1
    (DNS rebinding) gets nothing."""
    app = create_app(tmp_path)
    for base, ok in (("http://localhost:8100", True), ("http://127.0.0.1:8100", True), ("http://[::1]:8100", True),
                     ("http://evil.example:8100", False), ("http://localhost.evil.example:8100", False)):
        r = TestClient(app, base_url=base).get("/api/recordings")
        assert (r.status_code == 200) is ok, (base, r.status_code)



def test_the_diagnosis_takes_each_accuracy_check_apart(client, demos, eyes):
    from paralic.diagnose import validation_parts

    # A pure shift: everything is the shared part, which a quick adjust removes.
    grid = [(x, y) for x in (200, 960, 1720) for y in (150, 540, 930)]
    shifted = [{"target": [x, y], "mean": [x + 60, y - 10], "spread": 30, "n": 16} for x, y in grid]
    v = validation_parts(shifted, {"w": 1920, "h": 1080})
    assert v["shift_px"] == [60.0, -10.0] and v["shift_share"] == 1.0 and v["scatter_px"] == 0.0
    assert v["noise_floor_px"] == 7.5 and v["horizontal_px"] == 60.0 and v["vertical_px"] == 10.0
    # Up-down errors, worse at the edges.
    rng = np.random.default_rng(0)
    vertical = [{"target": [x, y], "mean": [x + rng.normal(0, 5), y + (70 if y != 540 else 15) * rng.choice((-1, 1))],
                 "spread": 20, "n": 16} for x, y in grid]
    v = validation_parts(vertical, {"w": 1920, "h": 1080})
    assert v["vertical_px"] > 5 * v["horizontal_px"] and v["edge_px"] > 1.6 * v["middle_px"]
    from paralic.diagnose import _findings_for
    text = " ".join(_findings_for(v))
    assert "Up-down" in text and "edges" in text
    assert validation_parts([{"target": [1, 2], "mean": [3, 4]}]) is None      # too few dots
    # The whole recording, through the API.
    d = client.get(f"/api/recordings/{demos['eyes'].name}/diagnosis").json()
    assert len(d["checks"]) == 2 and d["findings"] and d["conditions"]["frames"] > 1000
    assert all(isinstance(f, str) and f for f in d["findings"])
    hand = client.get(f"/api/recordings/{demos['hand'].name}/diagnosis").json()
    assert hand["checks"] == [] and "No accuracy check" in hand["findings"][0]
