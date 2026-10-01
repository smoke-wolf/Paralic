import numpy as np
import pytest

from paralic.calibration import ProfileStore
from paralic.users import UserStore
from paralic.session import TrackerSession, pack_frame, parse_frame
from tests.fakes import FakeTracker, ManualClock, fake_features, patch_session, tiny_jpeg
from tests.synthetic import SCREEN_H, SCREEN_W, Head, VirtualUser, calibration_points

JPEG = tiny_jpeg()


@pytest.fixture
def env(monkeypatch, tmp_path):
    patch_session(monkeypatch)
    tracker = FakeTracker()
    clock = ManualClock()
    session = TrackerSession(lambda: tracker, UserStore(tmp_path), clock=clock)
    user = VirtualUser(seed=5, noise=0.5)
    state = {"id": 0}

    def frame(sx=None, sy=None, label=None, closure=0.12, face=True, overlay=False, closed=None, gesture=None,
              mesh=False, setup=False):
        """One camera frame; ``closed`` = "left" / "right" closes that eye (a wink)."""
        state["id"] += 1
        if face:
            vec = user.features(sx if sx is not None else SCREEN_W / 2, sy if sy is not None else SCREEN_H / 2,
                                Head(), closed=closed)
            if closed:
                shut, other = 0.85, closure
                cl, cr = (shut, other) if closed == "left" else (other, shut)
                tracker.push(fake_features(vec, closure, cl, cr))
            else:
                tracker.push(fake_features(vec, closure))
        else:
            tracker.push(None)
        header = {"id": state["id"]}
        if label:
            header["label"] = label
        if overlay:
            header["overlay"] = True
        if mesh:
            header["mesh"] = True
        if setup:
            header["setup"] = True
        if gesture:
            header["gesture"] = gesture
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


def test_mesh_is_streamed_only_when_requested(env):
    _, _, _, frame, _ = env
    # No mesh unless the calibration preview asks for it (bandwidth gate).
    assert "mesh" not in frame()[0]
    msg = frame(mesh=True)[0]
    assert "mesh" in msg and "pts" in msg["mesh"]


def test_dots_report_how_long_the_eyes_have_rested_on_them(env):
    session, _, _, frame, _ = env
    session.handle_command({"type": "calibration_start", "mode": "full"})
    a, b, c = (400, 300), (1400, 700), (700, 900)
    for k in range(25):
        out = frame(*a, {"x": a[0], "y": a[1], "kind": "cal", "pt": 0})
        if k == 9:
            assert out[0]["labeled"] and out[0]["settled"] >= 6
    assert out[0]["settled"] >= 20
    # The next dot: while the eyes are still on the previous one nothing counts.
    for _ in range(5):
        out = frame(*a, {"x": b[0], "y": b[1], "kind": "cal", "pt": 1})
    assert out[0]["settled"] == 0
    for _ in range(8):
        out = frame(*b, {"x": b[0], "y": b[1], "kind": "cal", "pt": 1})
    assert out[0]["settled"] >= 6
    # Arriving straight away: every frame counts.
    for _ in range(10):
        out = frame(*c, {"x": c[0], "y": c[1], "kind": "cal", "pt": 2})
    assert out[0]["settled"] >= 8
    # Head steps are not fixations on a dot.
    out = frame(*b, {"x": b[0], "y": b[1], "kind": "head", "pt": 0})
    assert out[0]["labeled"] and "settled" not in out[0]


def test_frames_report_the_head_position_and_lighting_when_asked(env):
    _, _, _, frame, _ = env
    out = frame()[0]
    assert len(out["pos"]) == 2 and out["dist"] > 0 and "light" not in out
    out = frame(setup=True)[0]
    assert set(out["light"]) == {"face", "frame", "balance"}


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
    # What would help most (paralic/diagnose.py), for the results screen.
    assert isinstance(res["advice"], list) and all(isinstance(a, str) for a in res["advice"])
    assert session.profiles.exists()
    # Smoothing and the button magnet were personalised from the validation.
    assert res["personal"]["smoothing_profile"]["tuned_level"] is not None
    assert res["personal"]["magnet"]["radius_px"] > 0
    assert res["personal"]["model"]["version"] == 1

    # A new session (e.g. after a page reload) can load it.
    tracker2 = FakeTracker()
    s2 = TrackerSession(lambda: tracker2, UserStore(tmp_path))
    hello = s2.handle_command({"type": "hello", "screen": {"w": 1920, "h": 1080, "dpr": 1}})[0]
    assert hello["profile"]["accuracy_px"] is not None
    loaded = s2.handle_command({"type": "profile_load"})[0]
    assert loaded["loaded"] is True
    vec = VirtualUser(seed=5, noise=0).features(700, 500, Head())
    np.testing.assert_allclose(s2.model.predict(vec), session.model.predict(vec), atol=1e-6)


VAL_A = [(960, 600), (500, 300), (1400, 300), (1400, 900), (500, 900)]
VAL_B = [(960, 480), (300, 540), (960, 160), (1620, 540), (960, 920)]


def _validate(frame, session, points, **cmd):
    for i, (sx, sy) in enumerate(points):
        for _ in range(16):
            frame(sx, sy, {"x": sx, "y": sy, "kind": "val", "pt": i})
    return session.handle_command({"type": "validation_finish", **cmd})[0]


def _refine_round(frame, session, looks=None, n=8):
    """One improving round: extra dots (looking at `looks` instead, if given), a fit, new dots."""
    started = session.handle_command({"type": "calibration_start", "mode": "refine"})[0]
    assert started["ok"] and started["point_base"] >= 1000
    rng = np.random.default_rng(started["round"])
    for i in range(n):
        sx, sy = float(rng.uniform(150, 1770)), float(rng.uniform(100, 980))
        lx, ly = looks if looks else (sx, sy)
        for _ in range(16):
            frame(lx, ly, {"x": sx, "y": sy, "kind": "cal", "pt": started["point_base"] + i})
    fit = session.handle_command({"type": "calibration_fit", "mode": "full"})[0]
    assert fit["ok"]
    return started


def test_improving_rounds_keep_the_better_network(env):
    session, _, _, frame, calibrate = env
    calibrate()
    first = _validate(frame, session, VAL_A)
    n_cal = len(session.data.of_kind("cal"))
    started = _refine_round(frame, session)
    assert started["round"] == 1
    # The dots that measured the accuracy became ordinary calibration dots.
    assert not session.data.of_kind("val") and len(session.data.of_kind("cal")) > n_cal
    res = _validate(frame, session, VAL_B)
    r = res["refine"]
    assert r["round"] == 1 and r["kept"] in ("new", "previous")
    assert res["mean_error_px"] == pytest.approx(min(r["before_px"], r["after_px"]), abs=0.1)
    assert res["personal"]["model_history"][-1]["source"] == "refine"
    assert first["ok"] and res["ok"] and res["saved"]


def test_an_unhelpful_round_is_rolled_back(env):
    session, _, _, frame, calibrate = env
    calibrate()
    _validate(frame, session, VAL_A)
    champion = session.model
    # The person looked at one spot the whole round: the extra dots teach nonsense.
    started = _refine_round(frame, session, looks=(960, 540))
    res = _validate(frame, session, VAL_B)
    assert res["refine"]["kept"] == "previous" and res["refine"]["after_px"] > res["refine"]["before_px"]
    base = started["point_base"]
    assert not [f for f in session.data.of_kind("cal") if base <= f.point < base + 1000]
    assert session.model is not champion        # refitted with the new measuring dots, from the kept data


def test_measuring_only_leaves_the_network_alone(env):
    session, _, _, frame, calibrate = env
    calibrate()
    _validate(frame, session, VAL_A)
    session.handle_command({"type": "calibration_start", "mode": "adjust"})
    for i, (sx, sy) in enumerate([(960, 540), (200, 150), (1700, 150), (1700, 930), (200, 930)]):
        for _ in range(16):
            frame(sx, sy, {"x": sx, "y": sy, "kind": "adjust", "pt": i})
    assert session.handle_command({"type": "calibration_fit", "mode": "adjust"})[0]["ok"]
    model = session.model
    version = model.meta["version"]
    res = _validate(frame, session, VAL_B, refit=False)
    assert res["ok"] and "refine" not in res and "refit" not in res
    assert session.model is model and model.meta["version"] == version
    assert res["personal"]["accuracy_px"] == pytest.approx(res["mean_error_px"], abs=0.1)
    # Improving from there: the quick adjust's dots join the calibration dots.
    started = session.handle_command({"type": "calibration_start", "mode": "refine"})[0]
    assert started["ok"] and not session.data.of_kind("adjust") and not session.data.of_kind("val")


def test_refine_needs_a_calibration(env):
    session, *_ = env
    reply = session.handle_command({"type": "calibration_start", "mode": "refine"})[0]
    assert reply["ok"] is False


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


def test_tracker_failure_is_reported_per_frame(env):
    session, tracker, _, frame, _ = env

    def broken(rgb, ts):
        raise RuntimeError("graph error")

    tracker.process = broken
    msg = session.handle_frame(pack_frame({"id": 99}, JPEG))[0]
    assert msg["type"] == "frame" and msg["id"] == 99 and msg["face"] is False and "graph error" in msg["error"]


def test_profile_delete_reports_os_errors(env, monkeypatch):
    session, *_ = env

    def locked():
        raise PermissionError("file in use")

    monkeypatch.setattr(ProfileStore, "delete", lambda self: locked())
    reply = session.handle_command({"type": "profile_delete"})[0]
    assert reply["type"] == "profile" and reply["ok"] is False and "file in use" in reply["error"]


# -- personalisation -------------------------------------------------------------------

def test_label_events_become_finetuning_samples(env):
    session, _, _, frame, calibrate = env
    calibrate()
    # The eyes just arrived at the target: the frames before belong to another
    # fixation (the centre of the screen), so there is nothing steady to learn from.
    msgs = frame(1600, 250)
    reply = session.handle_command({"type": "label_event", "kind": "practice", "pre_frame": msgs[0]["id"],
                                    "target": [1600, 250]})[0]
    assert reply["type"] == "label_stored" and reply["stored"] is False
    for _ in range(20):
        msgs = frame(1600, 250)
    reply = session.handle_command({"type": "label_event", "kind": "practice", "pre_frame": msgs[0]["id"],
                                    "target": [1600, 250]})[0]
    assert reply["stored"] is True and reply["event"] == 1
    events = session.data.ft_events()
    assert len(events) == 1 and 4 <= len(events[1]) <= 15
    assert events[1][0].target == (1600.0, 250.0) and events[1][0].weight == 1.0
    # Clicks: the label is the nearest point inside the clicked element; huge ones are ignored.
    session.handle_command({"type": "hello", "screen": {"w": 1920, "h": 1080}})
    reply = session.handle_command({"type": "label_event", "kind": "click", "pre_frame": msgs[0]["id"],
                                    "rect": [0, 0, 1900, 1000]})[0]
    assert reply["stored"] is False
    reply = session.handle_command({"type": "label_event", "kind": "click", "pre_frame": msgs[0]["id"],
                                    "rect": [1000, 600, 1200, 700]})[0]
    assert reply["stored"] is True and session.data.ft_events()[2][0].weight == 0.5
    x, y = session.data.ft_events()[2][0].target
    assert 1000 <= x <= 1200 and 600 <= y <= 700


def test_label_event_unknown_frame_is_skipped(env):
    session, _, _, frame, calibrate = env
    calibrate()
    reply = session.handle_command({"type": "label_event", "kind": "practice", "pre_frame": 99999,
                                    "target": [1, 2]})[0]
    assert reply["stored"] is False and reply["reason"] == "frame not found"


def test_finetune_job_runs_in_background_and_pushes_result(env, monkeypatch):
    import paralic.session as session_mod

    monkeypatch.setattr(session_mod, "AUTO_FINETUNE_EVENTS", 10_000)  # start it by hand below
    session, _, clock, frame, calibrate = env
    pushed = []
    session._push = pushed.append
    calibrate()
    user = VirtualUser(seed=5, noise=0.5)

    def drifted_frame(sx, sy):
        from paralic.session import pack_frame as pf
        session.tracker.push(fake_features(user.features(sx, sy, Head(x=3.0, y=13.0, dist=68.0, pitch=0.06))))
        out = session.handle_frame(pf({"id": 10_000 + len(pushed) + int(clock.t * 1000)}, JPEG))
        clock.tick()
        return out

    rng = np.random.default_rng(0)
    for _ in range(20):
        sx, sy = rng.uniform(200, 1700), rng.uniform(150, 950)
        for _ in range(12):
            msgs = drifted_frame(sx, sy)
        session.handle_command({"type": "label_event", "kind": "practice", "pre_frame": msgs[0]["id"],
                                "target": [sx, sy]})
    assert not session._job_running()
    reply = session.handle_command({"type": "finetune"})[0]
    # The job does a model search; on a busy machine (other test runs in
    # parallel) it can take well over a minute, so wait generously.
    session.wait_for_job(timeout=600)
    results = [m for m in pushed if m["type"] == "finetune_result"]
    assert reply["type"] == "finetune_started"
    assert results, f"no fine-tuning result yet (job still running: {session._job_running()})"
    last = results[-1]
    assert last["ok"] and last["accepted"], last
    assert last["personal"]["model"]["version"] >= 1 and last["personal"]["model"]["source"] in ("fine-tuned", "retrained")
    assert session.profiles.exists()


def test_blink_calibration_personalises_the_detector(env):
    session, _, clock, frame, calibrate = env
    session.handle_command({"type": "blink_calibration_start"})
    pattern = ([0.12] * 40 + [0.3, 0.55, 0.6, 0.5, 0.25] + [0.12] * 6 + [0.3, 0.55, 0.6, 0.5, 0.25] + [0.12] * 30) * 3
    for c in pattern:
        frame(closure=c)
    res = session.handle_command({"type": "blink_calibration_finish"})[0]
    assert res["ok"] and res["n_pairs"] == 3
    assert session.blink.config.sensitivity == pytest.approx(res["sensitivity"])
    assert res["personal"]["blink"]["personal"] is True
    # A failed recording keeps the previous settings.
    session.handle_command({"type": "blink_calibration_start"})
    for _ in range(40):
        frame()
    bad = session.handle_command({"type": "blink_calibration_finish"})[0]
    assert bad["type"] == "blink_calibration_result" and bad["ok"] is False
    assert session.blink.config.sensitivity == pytest.approx(res["sensitivity"])


def test_experiment_round_adopts_a_clear_winner(env):
    session, *_ = env
    plan = session.handle_command({"type": "experiment_plan", "experiment": "smoothing"})[0]
    arms = {a["id"]: a for a in plan["arms"]}
    assert set(arms) == {"current", "smoother", "snappier"} and plan["analysis"]["decision"] == "need_more"
    # During the round the page applies an arm through a temporary override.
    session.handle_command({"type": "settings", "smoothing_level": arms["smoother"]["smoothing_level"]})
    assert session.effective()["smoothing_level"] == arms["smoother"]["smoothing_level"]
    rng = np.random.default_rng(1)
    trials = []
    for _ in range(10):
        trials.append({"arm": "current", "time_ms": 2400 * float(np.exp(0.2 * rng.standard_normal()))})
        trials.append({"arm": "smoother", "time_ms": 1200 * float(np.exp(0.2 * rng.standard_normal()))})
        trials.append({"arm": "snappier", "time_ms": 2600 * float(np.exp(0.2 * rng.standard_normal()))})
    res = session.handle_command({"type": "experiment_log", "experiment": "smoothing", "trials": trials})[0]
    assert res["decision"] == "adopt" and res["best"] == "smoother"
    assert res["applied"]["settings"] == {"smoothing": "auto"}
    # The override is gone and the winner is the new personal level.
    assert session.effective()["smoothing_level"] == pytest.approx(arms["smoother"]["smoothing_level"])
    assert res["personal"]["experiments"]["smoothing"]["decision"] == "adopt"
    # A new epoch starts: the next plan is relative to the new setting.
    plan2 = session.handle_command({"type": "experiment_plan", "experiment": "smoothing"})[0]
    assert plan2["analysis"]["n_trials"] == 0


def test_experiment_rejects_bad_input(env):
    session, *_ = env
    assert session.handle_command({"type": "experiment_plan", "experiment": "nope"})[0]["ok"] is False
    res = session.handle_command({"type": "experiment_log", "experiment": "magnet",
                                  "trials": [{"arm": "bogus", "time_ms": 1}]})[0]
    assert res["logged"] == 0


def test_people_have_separate_profiles(env):
    session, _, _, frame, calibrate = env
    calibrate()
    session.handle_command({"type": "validation_finish"}) if session.data.count("val") else None
    first = session.user["id"]
    reply = session.handle_command({"type": "user_create", "name": "Sam"})[0]
    assert reply["user"]["name"] == "Sam" and session.model is None
    assert {u["name"] for u in reply["users"]} == {"Person 1", "Sam"}
    msg = frame(500, 500)[0]
    assert msg["gaze"] is None  # Sam isn't calibrated
    reply = session.handle_command({"type": "user_select", "id": first})[0]
    assert reply["user"]["id"] == first
    renamed = session.handle_command({"type": "user_rename", "name": "Alex"})[0]
    assert renamed["user"]["name"] == "Alex"
    bad = session.handle_command({"type": "user_select", "id": "u00000000"})[0]
    assert bad["type"] == "users" and bad["ok"] is False


def test_finetune_starts_automatically(env, monkeypatch):
    import paralic.session as session_mod

    monkeypatch.setattr(session_mod, "AUTO_FINETUNE_EVENTS", 3)
    session, _, _, frame, calibrate = env
    pushed = []
    session._push = pushed.append
    calibrate()
    started = []
    for sx in (300, 900, 1500, 600):
        for _ in range(12):
            msgs = frame(sx, 400)
        reply = session.handle_command({"type": "label_event", "kind": "practice", "pre_frame": msgs[0]["id"],
                                        "target": [sx, 400]})[0]
        started.append(reply["finetune_started"])
    session.wait_for_job(timeout=600)
    assert started.count(True) == 1  # once, then it waits (minimum gap between jobs)
    assert any(m["type"] == "finetune_result" and m["auto"] for m in pushed)


# -- eye gestures --------------------------------------------------------------------------

def test_held_wink_tracks_with_the_open_eye(env):
    session, _, _, frame, calibrate = env
    calibrate()
    for _ in range(45):
        frame(500, 400)
    msgs = []
    # Keep the left eye closed while looking from (500, 400) over to (1400, 700).
    for k in range(50):
        f = min(1.0, k / 15)
        msgs.extend(frame(500 + 900 * f, 400 + 300 * f, closed="left"))
    for _ in range(10):
        msgs.extend(frame(1400, 700))
    events = [m for m in msgs if m["type"] != "frame"]
    assert [m["type"] for m in events] == ["wink_start", "wink_end"], events
    start, end = events
    assert start["eye"] == "left" and np.hypot(start["at"][0] - 500, start["at"][1] - 400) < 90
    assert end["duration_ms"] > 1400 and np.hypot(end["at"][0] - 1400, end["at"][1] - 700) < 140
    frames = [m for m in msgs if m["type"] == "frame"]
    during = frames[5:48]
    assert all(m["winking"] == "left" and m["net"] == "right" and not m["frozen"] for m in during)
    assert any(m["wink"] == "left" for m in during)
    assert frames[-1]["net"] == "both" and frames[-1]["winking"] is None


def test_wink_does_not_click_or_freeze(env):
    session, _, _, frame, calibrate = env
    calibrate()
    for _ in range(45):
        frame(800, 500)
    msgs = []
    for _ in range(25):
        msgs.extend(frame(800, 500, closed="right"))
    for _ in range(10):
        msgs.extend(frame(800, 500))
    types = [m["type"] for m in msgs]
    assert "blink" not in types and "double_blink" not in types
    assert not any(m.get("frozen") for m in msgs if m["type"] == "frame")


def test_long_close_reports_where_the_eyes_were(env):
    session, _, _, frame, calibrate = env
    calibrate()
    for _ in range(45):
        frame(800, 300)
    msgs = []
    for _ in range(45):
        msgs.extend(frame(800, 900, closure=0.92))  # eyes shut: the features drift
    for _ in range(10):
        msgs.extend(frame(800, 300))
    events = [m for m in msgs if m["type"] != "frame"]
    assert [m["type"] for m in events] == ["long_close_ready", "long_close"]
    lc = events[-1]
    assert lc["duration_ms"] >= 1400 and np.hypot(lc["at"][0] - 800, lc["at"][1] - 300) < 90
    # The cursor stayed frozen for the whole closure (not just the first 0.9 s).
    closed = [m for m in msgs if m["type"] == "frame"][2:44]
    assert all(m["frozen"] for m in closed)


def test_gesture_settings_are_validated_and_applied(env):
    session, *_ = env
    reply = session.handle_command({"type": "gestures_set", "gestures": {
        "left_hold": "off", "dwell": True, "dwell_ms": 99999, "long_close": "menu", "hold_ms": "x",
        "bogus": 1, "right_quick": "explode"}})[0]
    assert reply["type"] == "personal" and reply["ok"]
    g = reply["personal"]["gestures"]
    assert g["left_hold"] == "off" and g["dwell"] is True and g["dwell_ms"] == 3000 and g["long_close"] == "menu"
    assert g["hold_ms"] == 350 and g["right_quick"] == "off" and "bogus" not in g
    assert reply["personal"]["winks"] == {"left": False, "right": True}
    assert session.users.load_personal(session.user["id"])["gestures"]["left_hold"] == "off"
    bad = session.handle_command({"type": "gestures_set"})[0]
    assert bad["type"] == "personal" and bad["ok"] is False


def test_cursor_motion_and_head_nudge_settings(env):
    session, *_ = env
    g = session.personal_view()["gestures"]
    assert g["motion"] == "balanced" and g["hold_still"] is True and g["head_nudge"] is False
    reply = session.handle_command({"type": "gestures_set", "gestures": {
        "motion": "glide", "hold_still": False, "head_nudge": True, "nudge_speed": 999, "nudge_deadzone": 1,
        "nudge_right": 1, "nudge_up": -1}})[0]
    g = reply["personal"]["gestures"]
    assert g["motion"] == "glide" and g["hold_still"] is False and g["head_nudge"] is True
    assert g["nudge_speed"] == 250 and g["nudge_deadzone"] == 2 and g["nudge_right"] == 1 and g["nudge_up"] == -1
    # Wrong types and values are ignored (a bool is not a number, nor a number a flag).
    g = session.handle_command({"type": "gestures_set", "gestures": {
        "motion": "warp", "nudge_right": 5, "nudge_up": True, "nudge_speed": True, "head_nudge": 1}})[0]["personal"]["gestures"]
    assert g["motion"] == "glide" and g["nudge_right"] == 1 and g["nudge_up"] == -1 and g["nudge_speed"] == 250
    assert g["head_nudge"] is True


def test_wink_calibration_personalises_winks(env):
    session, _, _, frame, _ = env
    session.handle_command({"type": "wink_calibration_start"})
    for phase, closed, n in (("rest", None, 40), ("left", "left", 60), ("rest", None, 30), ("right", "right", 60),
                             ("rest", None, 30)):
        for _ in range(n):
            frame(closed=closed, gesture=phase)
    res = session.handle_command({"type": "wink_calibration_finish"})[0]
    assert res["type"] == "wink_calibration_result" and res["ok"], res
    assert res["left"]["ok"] and res["right"]["ok"]
    assert session.wink.config.left.rise == pytest.approx(res["left"]["threshold_rise"])
    assert res["personal"]["wink_profile"]["left"]["ok"]
    # Nothing recorded -> a clear error, and the earlier result is kept.
    session.handle_command({"type": "wink_calibration_start"})
    bad = session.handle_command({"type": "wink_calibration_finish"})[0]
    assert bad["ok"] is False and session.personal["wink"]["left"]["ok"]


def test_old_profiles_get_one_eye_networks_on_load(env, tmp_path):
    session, _, _, frame, calibrate = env
    calibrate()
    session.model.eyes = {}
    session.model.meta.pop("eye", None)
    session._save_profile()
    s2 = TrackerSession(lambda: FakeTracker(), UserStore(tmp_path))
    reply = s2.handle_command({"type": "profile_load"})[0]
    assert reply["loaded"]
    # Trained during the load (so nothing can change the model meanwhile).
    assert set(s2.model.eyes) == {"left", "right"}
    assert reply["personal"]["eye_models"]["available"] == ["left", "right"]
    # ...and they were saved with the profile, together with all its training data.
    model, data, _ = s2.profiles.load()
    assert set(model.eyes) == {"left", "right"}
    assert len(data.frames) == len(session.data.frames) > 0


def test_dwell_experiment_adopts_a_new_dwell_time(env):
    session, *_ = env
    session.handle_command({"type": "gestures_set", "gestures": {"dwell": True, "dwell_ms": 1000}})
    plan = session.handle_command({"type": "experiment_plan", "experiment": "dwell"})[0]
    arms = {a["id"]: a for a in plan["arms"]}
    assert arms["faster"]["dwell_ms"] == 750
    rng = np.random.default_rng(2)
    trials = []
    for _ in range(10):
        trials.append({"arm": "current", "time_ms": 2600 * float(np.exp(0.15 * rng.standard_normal()))})
        trials.append({"arm": "faster", "time_ms": 1500 * float(np.exp(0.15 * rng.standard_normal()))})
        trials.append({"arm": "slower", "time_ms": 3000 * float(np.exp(0.15 * rng.standard_normal()))})
    res = session.handle_command({"type": "experiment_log", "experiment": "dwell", "trials": trials})[0]
    assert res["decision"] == "adopt" and res["best"] == "faster"
    assert res["personal"]["gestures"]["dwell_ms"] == 750


def test_an_eye_that_failed_the_wink_test_can_be_forced_back_on(env):
    session, _, _, frame, _ = env
    session.handle_command({"type": "wink_calibration_start"})
    # Asked to close the left eye, this person closes both; the right eye winks fine.
    for phase, n in (("rest", 40), ("left", 60), ("rest", 30), ("right", 60), ("rest", 30)):
        for _ in range(n):
            if phase == "left":
                frame(closure=0.85, gesture=phase)
            else:
                frame(closed="right" if phase == "right" else None, gesture=phase)
    res = session.handle_command({"type": "wink_calibration_finish"})[0]
    assert not res["left"]["ok"] and res["right"]["ok"]
    assert res["personal"]["winks"] == {"left": False, "right": True}
    forced = session.handle_command({"type": "gestures_set", "gestures": {"left_forced": True}})[0]
    assert forced["personal"]["winks"] == {"left": True, "right": True}
    # A new test decides again.
    session.handle_command({"type": "wink_calibration_start"})
    for phase, n in (("rest", 40), ("left", 60), ("rest", 30)):
        for _ in range(n):
            frame(closure=0.85 if phase == "left" else 0.12, gesture=phase)
    again = session.handle_command({"type": "wink_calibration_finish"})[0]
    assert again["personal"]["winks"]["left"] is False


def test_blink_profiles_from_before_per_eye_signals_keep_the_average(env):
    session, *_ = env
    session.personal["blink"] = {"sensitivity": 0.25, "min_threshold": 0.2, "double_gap_ms": 600}
    session._apply_effective()
    assert session.blink_mode == "mean"
    session.personal["blink"]["signal"] = "left"
    session._apply_effective()
    assert session.blink_mode == "left"


# -- regressions from the gesture code review ------------------------------------------------

@pytest.fixture
def lids(env, monkeypatch):
    """Feed per-eye closure levels directly (gaze fixed at the screen centre)."""
    session, tracker, clock, frame, _ = env
    user = VirtualUser(seed=5, noise=0.5)
    state = {"id": 50_000}

    def feed(seq):
        out = []
        for cl, cr in seq:
            state["id"] += 1
            tracker.push(fake_features(user.features(SCREEN_W / 2, SCREEN_H / 2, Head()), 0.5 * (cl + cr), cl, cr))
            out.extend(session.handle_frame(pack_frame({"id": state["id"]}, JPEG)))
            clock.tick()
        return [m for m in out if m["type"] != "frame"]

    return session, feed


OPEN2 = [(0.12, 0.12)]


def test_a_blink_during_a_held_wink_never_pairs_into_a_click(lids):
    session, feed = lids
    feed(OPEN2 * 60)
    seq = [(0.85, 0.12)] * 20 + [(0.85, 0.85)] * 4 + [(0.85, 0.12)] * 3   # the open eye blinks mid-drag
    seq += OPEN2 * 10 + [(0.85, 0.85)] * 4 + OPEN2 * 30                   # one ordinary blink afterwards
    types = [m["type"] for m in feed(seq)]
    assert "wink_start" in types and "wink_end" in types
    assert "double_blink" not in types


def test_lopsided_blinks_count_once_when_watching_the_average(lids):
    session, feed = lids
    session.personal["blink"] = {"sensitivity": 0.3, "min_threshold": 0.3, "double_gap_ms": 550, "signal": "mean"}
    session._apply_effective()
    feed(OPEN2 * 60)
    for lead in (1, 2, 3):
        blink = [(0.85, 0.12)] * lead + [(0.85, 0.85)] * 4 + [(0.12, 0.12)] * 40
        types = [m["type"] for m in feed(blink)]
        assert types.count("blink") == 1 and "double_blink" not in types, (lead, types)


def test_someone_whose_one_eye_hardly_closes_can_still_double_blink(lids):
    session, feed = lids
    lblink = [(0.5, 0.14)] + [(0.85, 0.16)] * 2 + [(0.5, 0.14)]
    session.handle_command({"type": "blink_calibration_start"})
    feed((OPEN2 * 40 + lblink + OPEN2 * 6 + lblink + OPEN2 * 30) * 3)
    res = session.handle_command({"type": "blink_calibration_finish"})[0]
    assert res["ok"] and res["signal"] in ("left", "mean")
    assert res["personal"]["winks"] == {"left": False, "right": False}   # winks would be their blinks
    types = [m["type"] for m in feed((OPEN2 * 40 + lblink + OPEN2 * 6 + lblink + OPEN2 * 30) * 3)]
    assert types.count("double_blink") == 3


def test_a_long_close_is_never_reported_as_a_blink(lids):
    session, feed = lids
    session.handle_command({"type": "gestures_set", "gestures": {"long_close": "menu", "long_close_ms": 600}})
    assert session.blink.config.long_close_ms >= session.blink.config.max_closed_ms + 200
    feed(OPEN2 * 60)
    types = [m["type"] for m in feed([(0.95, 0.95)] * 36 + OPEN2 * 20)]   # 1.2 s
    assert "long_close" in types and "blink" not in types
