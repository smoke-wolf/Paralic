"""Glasses: noticing them and reflections on their lenses (the real face mesh on
glasses painted onto the test portrait), the cursor following the other eye
through glare, and a calibration for each glasses state."""

import json
from pathlib import Path

import numpy as np
import pytest

from paralic import glasses as G
from paralic.glasses import GLARE_OFF, GLARE_ON, GLASSES_OFF, GLASSES_ON, GlassesDetector, GlassesMeasure, measure
from paralic.session import GLARE_RELEASE_S, TrackerSession, pack_frame
from paralic.users import UserStore
from tests import face_images
from tests.fakes import FakeTracker, ManualClock, fake_features, patch_session, tiny_jpeg
from tests.synthetic import SCREEN_H, SCREEN_W, Head, VirtualUser, calibration_points

MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
JPEG = tiny_jpeg()


# -- the real face mesh on painted glasses ------------------------------------------------------

@pytest.fixture(scope="module")
def mesh():
    """The face mesh (pixel coordinates) of a BGR picture, from the real network."""
    cv2 = pytest.importorskip("cv2")
    if not MODEL.exists():
        pytest.skip("face landmarker model not downloaded")
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=MODEL.read_bytes()), running_mode=vision.RunningMode.IMAGE))
    except Exception as exc:  # pragma: no cover - platform specific
        pytest.skip(f"MediaPipe unavailable: {exc}")

    def run(bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        assert res.face_landmarks, "no face found"
        h, w = rgb.shape[:2]
        return rgb, np.array([[p.x * w, p.y * h, p.z * w] for p in res.face_landmarks[0]])

    yield run
    lm.close()


@pytest.fixture(scope="module")
def face(mesh):
    """The test portrait and its face mesh (where glasses are painted)."""
    img = face_images.portrait(640)
    if img is None:
        pytest.skip("test portrait unavailable (offline?)")
    return img, mesh(img)[1]


def lit(img, gain):
    """The same picture in dimmer or brighter light."""
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("gain", [0.5, 0.75, 1.0, 1.3])
def test_glasses_are_noticed_in_dim_and_bright_light(mesh, face, gain):
    img, pts = face
    plain = measure(*mesh(lit(img, gain)))
    worn = measure(*mesh(lit(face_images.paint_glasses(img, pts), gain)))
    thin = measure(*mesh(lit(face_images.paint_glasses(img, pts, thickness=0.015, color=(60, 60, 70)), gain)))
    assert plain.score < GLASSES_OFF / 2, plain
    assert worn.score > 2 * GLASSES_ON and thin.score > 1.3 * GLASSES_ON, (worn, thin)


@pytest.mark.parametrize("style,kw", [("half", {"thickness": 0.03}),
                                      ("rimless", {"thickness": 0.015, "color": (70, 70, 80)})])
def test_frames_without_a_lower_rim_are_noticed_by_their_bridge(mesh, face, style, kw):
    img, pts = face
    m = measure(*mesh(face_images.paint_glasses(img, pts, style=style, **kw)))
    assert m.score > GLASSES_ON and m.rims < 2.0, m


@pytest.mark.parametrize("gain", [0.6, 1.0])
@pytest.mark.parametrize("eye", ["left", "right"])
def test_glare_is_reported_for_the_lens_it_is_on(mesh, face, eye, gain):
    img, pts = face
    worn = lit(face_images.paint_glasses(img, pts), gain)
    other = "right" if eye == "left" else "left"
    # A lamp's reflection stays white however the face is lit.
    for kw in ({}, {"size": 0.12, "at": (0.0, -0.02)}, {"size": 0.13, "at": (-0.2 if eye == "right" else 0.2, 0.12)}):
        m = measure(*mesh(face_images.paint_glare(worn, pts, eye, **kw)))
        assert m.glare[eye] > 1.5 * GLARE_ON and m.glare[other] < GLARE_OFF / 2, (kw, m.glare)


def test_the_corneal_reflection_everyone_has_is_not_glare(mesh, face):
    img, pts = face
    for picture in (img, face_images.paint_glasses(img, pts), lit(img, 0.6)):
        m = measure(*mesh(picture))
        assert max(m.glare.values()) < GLARE_OFF / 4, m.glare


def test_camera_frames_report_glasses_and_glare(face, tmp_path):
    """The whole pipeline: JPEG frames through MediaPipe to the frame messages."""
    import cv2

    from paralic.tracker import FaceTracker

    img, pts = face
    worn = face_images.paint_glasses(img, pts)
    clock = ManualClock()
    session = TrackerSession(lambda: FaceTracker(MODEL.read_bytes()), UserStore(tmp_path), clock=clock)

    def show(bgr, n):
        jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        out = []
        for i in range(n):
            out.append(session.handle_frame(pack_frame({"id": i}, jpeg))[0])
            clock.tick()
        return out

    try:
        assert show(img, 12)[-1]["glasses"] is False
        msgs = show(worn, 30)
        assert msgs[-1]["glasses"] is True and msgs[-1]["glare"] is None
        msgs = show(face_images.paint_glare(worn, pts, "left"), 8)
        assert msgs[-1]["glare"] == "left" and msgs[-1]["glare_score"][0] > 0.5 and msgs[-1]["glare_score"][1] < 0.1
    finally:
        session.close()


# -- smoothing and hysteresis ---------------------------------------------------------------------

def fake_measures(monkeypatch, frames):
    """measure() returns these (glasses score, left glare, right glare), one per frame."""
    it = iter(frames)

    def fake(rgb, points):
        score, left, right = next(it)
        return GlassesMeasure(score=score, bridge=score, rims=1.0, glare={"left": left, "right": right}, skin=140.0)

    monkeypatch.setattr(G, "measure", fake)


def run_detector(det, n, t0=0.0, **kw):
    states = []
    for i in range(n):
        det.update(t0 + i / 30, None, None, **kw)
        states.append((det.glasses, det.settled, det.glare))
    return states


def test_noisy_evidence_does_not_flicker(monkeypatch):
    rng = np.random.default_rng(0)
    # 4 s without glasses, then 4 s with: single frames scatter across the thresholds.
    scores = np.exp(np.r_[rng.normal(np.log(0.9), 0.35, 120), rng.normal(np.log(3.2), 0.35, 120)])
    fake_measures(monkeypatch, [(s, 0.0, 0.0) for s in scores])
    states = [g for g, _, _ in run_detector(GlassesDetector(), 240)]
    assert states[4] is False and states[-1] is True
    assert sum(a != b for a, b in zip(states[4:], states[5:])) == 1


def test_putting_glasses_on_settles_after_a_few_seconds(monkeypatch):
    fake_measures(monkeypatch, [(0.5, 0.0, 0.0)] * 30 + [(6.0, 0.0, 0.0)] * 150)
    det = GlassesDetector(settle_s=3.0)
    states = run_detector(det, 180)
    assert states[0] == (None, None, None) and states[29] == (False, False, None)
    on = next(i for i, s in enumerate(states) if s[0])
    settled = next(i for i, s in enumerate(states) if s[1])
    assert on - 30 < 25 and 3.0 <= (settled - on) / 30 < 3.2


def test_frames_with_the_head_turned_away_are_not_judged(monkeypatch):
    fake_measures(monkeypatch, [])           # measure() must not be called
    det = GlassesDetector()
    run_detector(det, 10, yaw_deg=40.0)
    assert det.glasses is None


def test_glare_needs_a_clear_reflection_and_lingers_a_moment(monkeypatch):
    # A faint reflection never counts; a clear one at once; once it is gone the flag
    # stays a few frames (no flicker), and glare that stays is seen to persist.
    frames = [(0.4, 0.0, 0.3)] * 30 + [(0.4, 0.0, 2.0)] * 100 + [(0.4, 0.0, 0.0)] * 30
    fake_measures(monkeypatch, frames)
    det = GlassesDetector()
    states, persists, raw = [], [], []
    for i in range(len(frames)):
        det.update(i / 30, None, None)
        states.append(det.glare)
        persists.append(det.glare_persists(i / 30))
        raw.append(det.frame_glare)
    assert states[:30] == [None] * 30 and raw[30] == "right" and states[31] == "right"
    assert set(states[31:130]) == {"right"} and not any(persists[:90]) and persists[129]
    off = states[130:].index(None)
    assert 3 <= off <= 15 and raw[130] is None


# -- in the session ---------------------------------------------------------------------------------

VAL_POINTS = [(960, 600), (500, 300), (1400, 300), (1400, 900), (500, 900)]
GLARE_COLUMNS = {"left": [2, 3], "right": [0, 1]}   # the eye's iris offsets (see features.py)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A session on fake frames; ``scene`` says whether glasses are worn and
    which lens has glare."""
    patch_session(monkeypatch)
    scene = {"glasses": False, "glare": None}

    def fake_measure(rgb, points):
        glare = {eye: 2.0 if scene["glare"] in (eye, "both") else 0.0 for eye in ("left", "right")}
        score = 6.0 if scene["glasses"] else 0.4
        return GlassesMeasure(score=score, bridge=score, rims=1.0, glare=glare, skin=140.0)

    monkeypatch.setattr(G, "measure", fake_measure)
    tracker, clock = FakeTracker(), ManualClock()
    users = UserStore(tmp_path)
    session = TrackerSession(lambda: tracker, users, clock=clock)
    person = VirtualUser(seed=5, noise=0.5)
    state = {"id": 0}

    def frame(sx=SCREEN_W / 2, sy=SCREEN_H / 2, label=None, closed=None, face=True):
        """One frame looking at (sx, sy). Behind glare the face mesh misplaces that eye's iris."""
        state["id"] += 1
        if not face:
            tracker.push(None)
        else:
            vec = person.features(sx, sy, Head(), closed=closed)
            for eye, cols in GLARE_COLUMNS.items():
                if scene["glare"] in (eye, "both"):
                    vec[cols] += (0.12, 0.09)
            cl, cr = {"left": (0.85, 0.12), "right": (0.12, 0.85)}.get(closed, (0.12, 0.12))
            tracker.push(fake_features(vec, 0.5 * (cl + cr), cl, cr))
        out = session.handle_frame(pack_frame({"id": state["id"], **({"label": label} if label else {})}, JPEG))
        clock.tick()
        return out

    def calibrate(validate=True):
        session.handle_command({"type": "calibration_start", "mode": "full"})
        for i, (sx, sy) in enumerate(calibration_points()):
            for _ in range(20):
                frame(sx, sy, {"x": sx, "y": sy, "kind": "cal", "pt": i})
        for _ in range(60):
            frame(label={"x": SCREEN_W / 2, "y": SCREEN_H / 2, "kind": "head", "pt": 0})
        fit = session.handle_command({"type": "calibration_fit", "mode": "full"})[0]
        assert fit["ok"], fit
        if not validate:
            return fit, None
        for i, (sx, sy) in enumerate(VAL_POINTS):
            for _ in range(16):
                frame(sx, sy, {"x": sx, "y": sy, "kind": "val", "pt": i})
        val = session.handle_command({"type": "validation_finish"})[0]
        assert val["ok"] and val["saved"], val
        return fit, val

    def wear(glasses, seconds=4.0):
        """Put glasses on or take them off, and wait until it has settled."""
        scene["glasses"] = glasses
        return [m for _ in range(int(seconds * 30)) for m in frame()]

    return session, scene, frame, calibrate, wear


def changes(msgs):
    return [m for m in msgs if m["type"] == "glasses_changed"]


def test_frames_carry_the_glasses_and_glare_fields(env):
    session, scene, frame, _, _ = env
    m = frame()[0]
    assert m["glasses"] is False and m["glare"] is None and m["glare_score"] == [0.0, 0.0]
    scene.update(glasses=True, glare="both")
    for _ in range(30):
        m = frame()[0]
    assert m["glasses"] is True and m["glare"] == "both" and m["glare_score"] == [1.0, 1.0]
    m = frame(face=False)[0]
    assert m["face"] is False and m["glasses"] is True and "glare" in m


def test_glare_on_one_lens_hands_the_cursor_to_the_other_eye(env):
    session, scene, frame, calibrate, _ = env
    calibrate(validate=False)
    for _ in range(45):
        m = frame(600, 400)[0]
    assert m["net"] == "both" and m["glare"] is None
    scene["glare"] = "left"
    during = [frame(600, 400)[0] for _ in range(40)]
    assert all(m["glare"] == "left" and m["net"] == "right" for m in during[1:])
    # Aligned with the usual network, the cursor stays where the eyes look, while
    # the network for both eyes, misled by the hidden one, would be far off.
    assert all(np.hypot(m["gaze"][0] - 600, m["gaze"][1] - 400) < 80 for m in during[1:])
    misled = session.model.predict(session._history[-1].features, "both")[0]
    assert np.hypot(misled[0] - 600, misled[1] - 400) > 150
    # The glare is gone: the usual network leads again after a moment, and only once.
    scene["glare"] = None
    after = [frame(600, 400)[0]["net"] for _ in range(60)]
    assert after[0] == "right" and after[-1] == "both" and sum(a != b for a, b in zip(after, after[1:])) == 1
    assert after.index("both") / 30 >= GLARE_RELEASE_S


def test_a_flickering_reflection_does_not_flap_between_networks(env):
    session, scene, frame, calibrate, _ = env
    calibrate(validate=False)
    for _ in range(30):
        frame(600, 400)
    nets = []
    for k in range(90):
        scene["glare"] = "left" if (k // 3) % 2 == 0 else None
        nets.append(frame(600, 400)[0]["net"])
    assert nets[-1] == "right" and sum(a != b for a, b in zip(nets, nets[1:])) == 1


def test_a_wink_takes_priority_over_glare(env):
    session, scene, frame, calibrate, _ = env
    calibrate(validate=False)
    for _ in range(45):
        frame(600, 400)
    scene["glare"] = "left"
    assert [frame(600, 400)[0]["net"] for _ in range(10)][-1] == "right"
    winking = [m for m in (frame(600, 400, closed="right")[0] for _ in range(20)) if m["winking"] == "right"]
    assert winking and all(m["net"] == "left" for m in winking)


def test_calibration_leaves_out_moments_of_glare_but_not_glare_that_stays(env):
    session, scene, frame, _, _ = env
    for _ in range(20):
        frame()
    session.handle_command({"type": "calibration_start", "mode": "full"})
    label = {"x": 400, "y": 300, "kind": "cal", "pt": 0}
    for _ in range(20):
        frame(400, 300, label)
    stored = session.data.count("cal")
    scene["glare"] = "right"                  # a moment of glare: left out, not counted as settled
    msgs = [frame(400, 300, label)[0] for _ in range(10)]
    assert not any(m["labeled"] or "settled" in m for m in msgs) and session.data.count("cal") == stored
    assert all(m["glare"] == "right" for m in msgs)              # the page can say why
    # Glare that stays: the frames are used again, so calibrating stays possible.
    msgs = [frame(400, 300, label)[0] for _ in range(100)]
    assert msgs[-1]["glare"] == "right" and msgs[-1]["labeled"] and msgs[-1]["settled"] > 0
    assert session.data.count("cal") > stored


def test_each_glasses_state_gets_its_own_calibration(env):
    session, scene, frame, calibrate, wear = env
    users, me = session.users, session.user["id"]
    wear(True, 1.0)
    fit, val = calibrate()
    assert fit["glasses"] is True and val["glasses"] is True
    store = users.profile_store(me, glasses=True)
    assert store.path.name == "profile-glasses.json" and json.loads(store.path.read_text())["glasses"] is True
    assert not users.profile_store(me).exists()
    assert users.list()[0]["calibrated"] is True               # calibrated with glasses only
    with_glasses = session.model
    wear(False, 1.0)
    fit, val = calibrate()
    assert fit["glasses"] is False and val["glasses"] is False
    assert json.loads(users.profile_store(me).path.read_text())["glasses"] is False
    without = session.model
    # Loading picks the one for the glasses worn now - or the one asked for.
    vec = VirtualUser(seed=5, noise=0).features(700, 500, Head())
    wear(True, 1.0)
    reply = session.handle_command({"type": "profile_load"})[0]
    assert reply["loaded"] and reply["glasses"] is True and reply["glasses_now"] is True
    assert reply["glasses_mismatch"] is False and reply["slots"] == {"glasses": True, "plain": True}
    np.testing.assert_allclose(session.model.predict(vec), with_glasses.predict(vec), atol=1e-6)
    reply = session.handle_command({"type": "profile_load", "glasses": False})[0]
    assert reply["glasses"] is False and reply["glasses_mismatch"] is True
    np.testing.assert_allclose(session.model.predict(vec), without.predict(vec), atol=1e-6)
    # Forgetting the calibration forgets both.
    assert session.handle_command({"type": "profile_delete"})[0]["ok"]
    assert not users.profile_store(me).exists() and not users.profile_store(me, glasses=True).exists()


def test_the_other_calibration_is_used_until_a_quick_adjust_makes_one(env):
    session, scene, frame, calibrate, wear = env
    users, me = session.users, session.user["id"]
    calibrate()                               # without glasses
    plain = users.profile_store(me).path.read_text()
    wear(True, 1.0)
    reply = session.handle_command({"type": "profile_load"})[0]
    assert reply["loaded"] and reply["glasses"] is False and reply["glasses_now"] is True
    assert reply["glasses_mismatch"] is True and reply["slots"] == {"glasses": False, "plain": True}
    # A quick adjust with the glasses on is saved as the calibration with glasses.
    session.handle_command({"type": "calibration_start", "mode": "adjust"})
    for i, (sx, sy) in enumerate([(960, 540), (230, 150), (1690, 150), (1690, 930), (230, 930)]):
        for _ in range(16):
            frame(sx, sy, {"x": sx, "y": sy, "kind": "adjust", "pt": i})
    res = session.handle_command({"type": "calibration_fit", "mode": "adjust"})[0]
    assert res["ok"] and res["glasses"] is True
    assert users.profile_store(me, glasses=True).exists() and users.profile_store(me).path.read_text() == plain
    reply = session.handle_command({"type": "profile_load"})[0]
    assert reply["glasses"] is True and reply["glasses_mismatch"] is False


def test_profiles_from_before_glasses_load_exactly_as_before(env):
    from tests.test_migration import _v1_profile

    session, scene, frame, calibrate, wear = env
    calibrate()
    path = session.profiles.path
    doc = json.loads(path.read_text())
    del doc["glasses"]
    path.write_text(json.dumps(doc))
    old = session.model
    vec = VirtualUser(seed=5, noise=0).features(700, 500, Head())
    for glasses in (True, False):
        wear(glasses, 1.0)
        reply = session.handle_command({"type": "profile_load"})[0]
        assert reply["loaded"] and reply["glasses"] is None and reply["glasses_mismatch"] is False
        np.testing.assert_allclose(session.model.predict(vec), old.predict(vec), atol=1e-6)
    session._save_profile()                   # saved again (learning from use): still says nothing
    assert "glasses" not in json.loads(path.read_text())
    # A version-1 profile too.
    _v1_profile(path)
    wear(True, 1.0)
    reply = session.handle_command({"type": "profile_load"})[0]
    assert reply["loaded"] and reply["legacy"] is True and reply["glasses"] is None and not reply["glasses_mismatch"]


def test_putting_glasses_on_is_announced_once(env):
    session, scene, frame, calibrate, wear = env
    calibrate()                               # without glasses, in use
    wear(False, 4.0)                          # (the dots are a few seconds ago)
    msgs = wear(True, 1.5) + wear(False, 4.0)  # on for a moment only: nothing
    assert not changes(msgs)
    msgs = wear(True, 5.0)
    assert changes(msgs) == [{"type": "glasses_changed", "glasses": True, "slot_available": False, "in_use": False}]
    assert all(m["glasses"] for m in msgs[-30:] if m["type"] == "frame")
    msgs = wear(False, 5.0)
    assert changes(msgs) == [{"type": "glasses_changed", "glasses": False, "slot_available": True, "in_use": True}]


def test_a_calibration_loaded_before_the_glasses_were_seen_is_questioned_once(env):
    """After a reconnect the calibration is loaded before any frame arrived: if
    the glasses seen then do not fit it, that is said (once)."""
    session, scene, _, calibrate, _ = env
    calibrate()                               # without glasses
    person = VirtualUser(seed=5, noise=0.5)
    for glasses, expected in ((False, []), (True, [{"type": "glasses_changed", "glasses": True,
                                                    "slot_available": False, "in_use": False}])):
        scene["glasses"] = glasses
        tracker, clock = FakeTracker(), ManualClock()
        again = TrackerSession(lambda: tracker, session.users, clock=clock)
        reply = again.handle_command({"type": "profile_load"})[0]
        assert reply["loaded"] and reply["glasses_now"] is None and reply["glasses_mismatch"] is False
        msgs = []
        for i in range(60):
            tracker.push(fake_features(person.features(960, 540, Head())))
            msgs += again.handle_frame(pack_frame({"id": i}, JPEG))
            clock.tick()
        assert changes(msgs) == expected


def test_no_announcement_before_a_calibration_is_in_use_or_while_calibrating(env):
    session, scene, frame, calibrate, wear = env
    wear(False, 1.0)
    assert not changes(wear(True, 5.0))       # nothing loaded yet: just noted
    assert session.glasses.settled is True
    calibrate()
    session.handle_command({"type": "calibration_start", "mode": "adjust"})
    scene["glasses"] = False
    label = {"x": 960, "y": 540, "kind": "adjust", "pt": 0}
    msgs = [m for _ in range(150) for m in frame(label=label)]
    assert not changes(msgs)
