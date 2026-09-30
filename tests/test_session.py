import numpy as np
import pytest

from paralic.calibration import ProfileStore
from paralic.session import TrackerSession, pack_frame, parse_frame
from tests.fakes import FakeTracker, ManualClock, fake_features, patch_session, tiny_jpeg
from tests.synthetic import SCREEN_H, SCREEN_W, Head, VirtualUser, calibration_points

JPEG = tiny_jpeg()


@pytest.fixture
def env(monkeypatch, tmp_path):
    patch_session(monkeypatch)
    tracker = FakeTracker()
    clock = ManualClock()
    session = TrackerSession(lambda: tracker, ProfileStore(tmp_path / "profile.json"), clock=clock)
    user = VirtualUser(seed=5, noise=0.5)
    state = {"id": 0}

    def frame(sx=None, sy=None, label=None, closure=0.12, face=True, overlay=False):
        state["id"] += 1
        if face:
            vec = user.features(sx if sx is not None else SCREEN_W / 2, sy if sy is not None else SCREEN_H / 2, Head())
            tracker.push(fake_features(vec, closure))
        else:
            tracker.push(None)
        header = {"id": state["id"]}
        if label:
            header["label"] = label
        if overlay:
            header["overlay"] = True
        out = session.handle_frame(pack_frame(header, JPEG))
        clock.tick()
        return out

    def calibrate():
        session.handle_command({"type": "calibration_start", "mode": "full"})
        for i, (sx, sy) in enumerate(calibration_points()):
            for _ in range(20):
                frame(sx, sy, {"x": sx, "y": sy, "kind": "cal", "pt": i})
        for _ in range(60):
            frame(SCREEN_W / 2, SCREEN_H / 2, {"x": SCREEN_W / 2, "y": SCREEN_H / 2, "kind": "head", "pt": 0})
        return session.handle_command({"type": "calibration_fit", "mode": "full"})[0]

    return session, tracker, clock, frame, calibrate


def test_frame_header_round_trip():
    header, image = parse_frame(pack_frame({"id": 7, "label": {"x": 1}}, b"jpegdata"))
    assert header == {"id": 7, "label": {"x": 1}} and image == b"jpegdata"


def test_bad_frame_is_reported(env):
    session, *_ = env
    out = session.handle_frame(b"\x01")
    assert out[0]["type"] == "frame" and "error" in out[0]


def test_no_face_frame(env):
    session, tracker, clock, frame, _ = env
    msg = frame(face=False)[0]
    assert msg["face"] is False and msg["gaze"] is None


def test_uncalibrated_frames_have_no_gaze_but_report_blink_state(env):
    _, _, _, frame, _ = env
    msg = frame(overlay=True)[0]
    assert msg["face"] is True and msg["gaze"] is None and msg["closure"] == pytest.approx(0.12)
    assert "eyes" in msg and msg["thr"][0] > msg["thr"][1]


def test_calibration_trains_and_predicts(env):
    session, _, _, frame, calibrate = env
    result = calibrate()
    assert result["ok"] and result["n_points"] == 13
    for _ in range(40):
        msg = frame(400, 300)[0]
    assert msg["gaze"] is not None
    assert np.hypot(msg["gaze"][0] - 400, msg["gaze"][1] - 300) < 80


def test_labelled_frames_skip_blinks(env):
    session, _, _, frame, _ = env
    session.handle_command({"type": "calibration_start"})
    for _ in range(30):
        frame(100, 100)
    msg = frame(100, 100, {"x": 100, "y": 100, "kind": "cal", "pt": 0}, closure=0.95)[0]
    assert msg["labeled"] is False
    assert session.data.count("cal") == 0


def test_double_blink_reports_pre_blink_frame(env):
    session, _, _, frame, calibrate = env
    calibrate()
    for _ in range(40):
        frame(1500, 250)
    last_open = session._history[-1].frame_id
    msgs = []
    for closures in ([0.5, 0.9, 0.9, 0.9, 0.5], [0.12] * 5, [0.5, 0.9, 0.9, 0.9, 0.5], [0.12] * 5):
        for c in closures:
            msgs.extend(frame(1500, 900, closure=c))  # gaze "drifts" while blinking
    types = [m["type"] for m in msgs]
    assert "double_blink" in types
    ev = next(m for m in msgs if m["type"] == "double_blink")
    assert ev["pre_frame"] == last_open
    assert np.hypot(ev["at"][0] - 1500, ev["at"][1] - 250) < 120
    # The cursor was frozen during the blinks.
    assert any(m.get("frozen") for m in msgs if m["type"] == "frame")


def test_validation_saves_profile_and_profile_loads(env, tmp_path, monkeypatch):
    session, _, _, frame, calibrate = env
    calibrate()
    for i, (sx, sy) in enumerate([(500, 300), (1400, 300), (960, 600), (500, 900), (1400, 900)]):
        for _ in range(16):
            frame(sx, sy, {"x": sx, "y": sy, "kind": "val", "pt": i})
    res = session.handle_command({"type": "validation_finish"})[0]
    assert res["type"] == "validation_result" and res["ok"] and res["saved"]
    assert res["mean_error_px"] < 120 and len(res["points"]) == 5
    assert (tmp_path / "profile.json").exists()

    # A new session (e.g. after a page reload) can load it.
    tracker2 = FakeTracker()
    s2 = TrackerSession(lambda: tracker2, ProfileStore(tmp_path / "profile.json"))
    hello = s2.handle_command({"type": "hello", "screen": {"w": 1920, "h": 1080, "dpr": 1}})[0]
    assert hello["profile"]["accuracy_px"] is not None
    loaded = s2.handle_command({"type": "profile_load"})[0]
    assert loaded["loaded"] is True
    vec = VirtualUser(seed=5, noise=0).features(700, 500, Head())
    np.testing.assert_allclose(s2.model.predict(vec), session.model.predict(vec), atol=1e-6)


def test_quick_adjust_corrects_offset(env):
    session, _, _, frame, calibrate = env
    calibrate()
    session.handle_command({"type": "calibration_start", "mode": "adjust"})
    # The user now sits differently: every target appears 60 px lower than predicted.
    for i, (sx, sy) in enumerate([(960, 540), (230, 150), (1690, 150), (1690, 930), (230, 930)]):
        for _ in range(16):
            frame(sx, sy, {"x": sx, "y": sy + 60, "kind": "adjust", "pt": i})
    res = session.handle_command({"type": "calibration_fit", "mode": "adjust"})[0]
    assert res["ok"] and res["error_after_px"] < res["error_before_px"]


def test_adjust_without_model_fails_cleanly(env):
    session, *_ = env
    res = session.handle_command({"type": "calibration_fit", "mode": "adjust"})[0]
    assert res["type"] == "calibration_result" and res["ok"] is False


def test_calibration_with_too_little_data_reports_error(env):
    session, _, _, frame, _ = env
    session.handle_command({"type": "calibration_start"})
    for _ in range(10):
        frame(100, 100, {"x": 100, "y": 100, "kind": "cal", "pt": 0})
    res = session.handle_command({"type": "calibration_fit", "mode": "full"})[0]
    assert res["type"] == "calibration_result" and res["ok"] is False and res["error"]


def test_settings_command(env):
    session, *_ = env
    res = session.handle_command({"type": "settings", "smoothing": "high", "blink_sensitivity": "high",
                                  "double_blink": "relaxed"})[0]
    assert res["smoothing"] == "high"
    assert session.blink.config.sensitivity == pytest.approx(0.22)
    assert session.blink.config.double_gap_ms == pytest.approx(800)


def test_unknown_command(env):
    session, *_ = env
    assert session.handle_command({"type": "nope"})[0]["type"] == "error"


def test_close_closes_tracker(env):
    session, tracker, _, frame, _ = env
    frame()
    session.close()
    assert tracker.closed
