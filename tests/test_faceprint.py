"""Face print: describing faces, the learned matrix weighting, storage, and the session."""

import numpy as np
import pytest

from paralic import faceprint as FP
from paralic.faceprint import (MIN_SAMPLES, SHAPE_DIM, TEXTURE_DIM, FaceMetric, FacePrintStore, FaceRecognizer,
                               FaceSample, choose_samples, shape_descriptor, texture_descriptor)


# -- describing a face ---------------------------------------------------------------------

def _random_face(rng):
    """478 plausible 3D points: a face-like cloud with the eye corners where the code expects them."""
    P = rng.normal(0, 30, (478, 3)) + [320, 240, 0]
    P[33], P[133], P[362], P[263] = [270, 230, 5], [300, 232, 0], [340, 232, 0], [370, 230, 5]
    P[2] = [320, 290, -10]
    return P


def test_face_deltas_ignore_pose_distance_and_position():
    rng = np.random.default_rng(0)
    P = _random_face(rng)
    d0 = shape_descriptor(P)
    assert d0.shape == (SHAPE_DIM,)
    # Turn the head (3D rotation), move it and bring it closer: same face deltas.
    a, b, c = 0.3, -0.2, 0.15
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    Q = (P - P.mean(axis=0)) @ (Rz @ Ry @ Rx).T * 1.7 + [100, -40, 12]
    np.testing.assert_allclose(shape_descriptor(Q), d0, atol=1e-9)
    # A different face shape gives different deltas.
    P2 = P.copy()
    P2[list(FP.RIGID_POINTS[4:18])] += [0, 12, 0]        # a longer nose
    assert np.abs(shape_descriptor(P2) - d0).max() > 0.05


def test_texture_ignores_overall_brightness():
    rng = np.random.default_rng(1)
    face = (rng.random((112, 112)) * 120 + 60).astype(np.uint8)
    t0 = texture_descriptor(face)
    assert t0.shape == (TEXTURE_DIM,)
    brighter = np.clip(face.astype(int) * 1.3 + 20, 0, 255).astype(np.uint8)
    other = (rng.random((112, 112)) * 120 + 60).astype(np.uint8)
    assert np.abs(texture_descriptor(brighter) - t0).sum() < 0.3 * np.abs(texture_descriptor(other) - t0).sum()


# -- the learned weighting -------------------------------------------------------------------

def _person(rng, n, mean, wobble):
    """n frames of one "person": their mean face plus variation along a few directions of their own."""
    D = SHAPE_DIM + TEXTURE_DIM
    X = mean + rng.normal(0, 0.004, (n, D))
    X += rng.normal(0, 1, (n, len(wobble))) @ wobble        # expressions, light: large but structured
    return [FaceSample(shape=x[:SHAPE_DIM], texture=x[SHAPE_DIM:], yaw=float(rng.normal(0, 8)),
                       pitch=float(rng.normal(0, 6)), light=float(rng.uniform(60, 180)), id=f"s{i}")
            for i, x in enumerate(X)]


@pytest.fixture
def population():
    rng = np.random.default_rng(3)
    D = SHAPE_DIM + TEXTURE_DIM
    wobble = rng.normal(0, 0.03, (4, D))                     # the same kinds of variation for everyone
    means = {name: rng.normal(0, 0.012, D) for name in "ABCDE"}
    return rng, wobble, means


def test_the_weighting_recognises_people_and_rejects_strangers(population):
    rng, wobble, means = population
    rec = FaceRecognizer({u: _person(rng, 16, means[u], wobble) for u in "ABC"})
    assert set(rec.refs) == {"A", "B", "C"}
    for u in "ABC":
        hits = [rec.identify(_person(rng, 3, means[u], wobble))["user"] for _ in range(5)]
        assert hits == [u] * 5
    for stranger in "DE":
        assert all(rec.identify(_person(rng, 3, means[stranger], wobble))["user"] is None for _ in range(5))
    # The whitening learned that the shared variation says nothing about who it is.
    a = _person(rng, 1, means["A"], wobble)[0]
    far_but_same = FaceSample(shape=a.shape + 3 * wobble[0, :SHAPE_DIM], texture=a.texture + 3 * wobble[0, SHAPE_DIM:])
    assert rec.identify([far_but_same])["user"] == "A"


def test_one_person_is_recognised_and_strangers_are_not(population):
    rng, wobble, means = population
    rec = FaceRecognizer({"A": _person(rng, 14, means["A"], wobble)})
    assert rec.identify(_person(rng, 4, means["A"], wobble))["user"] == "A"
    assert rec.identify(_person(rng, 4, means["B"], wobble))["user"] is None


def test_a_print_needs_enough_frames(population):
    rng, wobble, means = population
    rec = FaceRecognizer({"A": _person(rng, MIN_SAMPLES - 1, means["A"], wobble)})
    assert rec.prints == {}


def test_new_frames_join_and_the_most_redundant_one_makes_way(population):
    rng, wobble, means = population
    samples = _person(rng, 6, means["A"], wobble)
    dup = FaceSample(shape=samples[0].shape.copy(), texture=samples[0].texture.copy(), id="dup")
    out = choose_samples(samples, dup, cap=6)
    assert len(out) == 6 and sum(s.id in ("s0", "dup") for s in out) == 1
    rec = FaceRecognizer({"A": _person(rng, 12, means["A"], wobble)})
    assert rec.novelty(rec.prints["A"][0], "A") < 1e-6        # a stored frame is nothing new


# -- storage ---------------------------------------------------------------------------------

def test_store_keeps_numbers_and_faces(tmp_path, population):
    rng, wobble, means = population
    samples = _person(rng, 3, means["A"], wobble)
    for s in samples:
        s.crop = (rng.random((112, 112)) * 255).astype(np.uint8)
    store = FacePrintStore(tmp_path / "faceprint")
    assert not store.exists() and store.load() == []
    store.save(samples)
    loaded = store.load()
    assert [s.id for s in loaded] == [s.id for s in samples]
    np.testing.assert_allclose(loaded[1].shape, samples[1].shape)
    assert store.picture(samples[0].id)[:2] == b"\xff\xd8"           # a JPEG
    store.save(samples[1:])
    assert store.picture(samples[0].id) is None                     # dropped frames lose their picture
    store.forget()
    assert not store.exists() and not (tmp_path / "faceprint").exists()


# -- real faces through MediaPipe -------------------------------------------------------------

def test_real_faces_are_told_apart():
    """The test portrait, altered (turned, scaled, other light, blur), against two other test faces."""
    import cv2

    from tests import face_images
    from tests.conftest import ROOT

    model = ROOT / "models" / "face_landmarker.task"
    portrait = face_images.portrait(640)
    others = [face_images.face_image("business-person.png"), face_images.face_image("face_stylizer_raw_face_demo.png")]
    if not model.exists() or portrait is None or any(o is None for o in others):
        pytest.skip("face model or test images not available")
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model.read_bytes()), running_mode=vision.RunningMode.IMAGE))
    rng = np.random.default_rng(5)

    def frames(img, n):
        out = []
        h, w = img.shape[:2]
        while len(out) < n:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-9, 9), rng.uniform(0.85, 1.1))
            a = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT).astype(float)
            a = np.clip(a * rng.uniform(0.65, 1.25) + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
            if rng.random() < 0.4:
                a = cv2.GaussianBlur(a, (5, 5), 1.0)
            a = cv2.resize(a, (640, int(h * 640 / w)))
            rgb = np.ascontiguousarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if res.face_landmarks:
                hh, ww = rgb.shape[:2]
                pts = np.array([[p.x * ww, p.y * hh, p.z * ww] for p in res.face_landmarks[0]])
                s = FP.make_sample(rgb, pts)
                if s is not None:
                    out.append(s)
        return out

    big = [cv2.resize(o, (768, int(o.shape[0] * 768 / o.shape[1]))) for o in others]
    rec = FaceRecognizer({"portrait": frames(portrait, 10), "b": frames(big[0], 10)})
    assert rec.identify(frames(portrait, 3))["user"] == "portrait"
    assert rec.identify(frames(big[0], 3))["user"] == "b"
    assert rec.identify(frames(big[1], 3))["user"] is None          # never enrolled
    lm.close()


# -- in the session ---------------------------------------------------------------------------

@pytest.fixture
def face_env(monkeypatch, tmp_path, population):
    import uuid

    import paralic.session as S
    from paralic.session import TrackerSession, pack_frame
    from paralic.users import UserStore
    from tests.fakes import FakeTracker, ManualClock, fake_features, patch_session, tiny_jpeg
    from tests.synthetic import Head, VirtualUser

    patch_session(monkeypatch)
    monkeypatch.setattr(S, "FACE_LEARN_GAP_S", 0.0)
    monkeypatch.setattr(S, "FACE_CHECK_GAP_S", 0.5)
    monkeypatch.setattr(S, "FACE_SAVE_GAP_S", 0.0)
    rng, wobble, means = population
    who = {"face": "A"}

    def fake_sample(rgb, points, yaw, pitch, t):
        s = _person(rng, 1, means[who["face"]], wobble)[0]
        s.id = uuid.uuid4().hex[:12]
        s.crop = (rng.random((112, 112)) * 255).astype(np.uint8)
        return s

    monkeypatch.setattr(S, "make_sample", fake_sample)
    tracker, clock = FakeTracker(), ManualClock()
    users = UserStore(tmp_path)
    session = TrackerSession(lambda: tracker, users, clock=clock)
    eyes = VirtualUser(seed=2, noise=0.3)
    jpeg = tiny_jpeg()
    n = {"id": 0}

    def frames(count, face=None):
        out = []
        if face and face != who["face"]:
            # A different person sits down: the face leaves the camera for a moment.
            for _ in range(15):
                n["id"] += 1
                tracker.push(None)
                out += session.handle_frame(pack_frame({"id": n["id"]}, jpeg))
                clock.tick(0.1)
            who["face"] = face
        for _ in range(count):
            n["id"] += 1
            tracker.push(fake_features(eyes.features(960, 540, Head()), 0.12))
            out += session.handle_frame(pack_frame({"id": n["id"]}, jpeg))
            clock.tick(0.1)
        return out

    return session, frames


def test_a_person_who_chose_themselves_gets_a_face_print(face_env):
    session, frames = face_env
    me = session.user["id"]
    frames(30)                                  # not confirmed yet: nothing is learned
    assert session.faceprint_view()["samples"] == 0
    session.handle_command({"type": "face_confirm"})
    frames(40)
    view = session.faceprint_view()
    assert view["ready"] and view["samples"] >= MIN_SAMPLES
    users = session.handle_command({"type": "users"})[0]["users"]
    assert [u["face"] for u in users if u["id"] == me] == [True]
    faces = session.handle_command({"type": "faceprint_faces"})[0]["faces"]
    assert len(faces) == view["samples"] and faces[0]["src"].startswith("data:image/jpeg;base64,")


def test_recognising_who_is_at_the_camera(face_env):
    session, frames = face_env
    a = session.user["id"]
    session.handle_command({"type": "face_confirm"})
    frames(40, face="A")
    frames(1, face="B")                         # Bea sits down...
    b = session.handle_command({"type": "user_create", "name": "Bea"})[0]["user"]["id"]
    frames(40)                                  # ...and chose herself: her print starts
    seen = []
    assert session.handle_command({"type": "face_recognize"}) == []
    seen += frames(30, face="A")
    out = [m for m in seen if m["type"] == "face_recognition"]
    assert out and out[0]["user"] == a and out[0]["confident"] and out[0]["frames"] >= 6
    # Someone else (with a face print) sat down while Bea's profile is active.
    changed = [m for m in seen if m["type"] == "face_changed"]
    assert changed and changed[0]["user"] == a and changed[0]["name"]
    session.handle_command({"type": "face_recognize"})
    out = [m for m in frames(30, face="B") if m["type"] == "face_recognition"]
    assert out[0]["user"] == b and out[0]["name"] == "Bea"
    session.handle_command({"type": "face_recognize"})
    out = [m for m in frames(30, face="C") if m["type"] == "face_recognition"]
    assert out[0]["user"] is None               # a stranger



def test_strangers_and_unconfirmed_faces_are_not_learned(face_env):
    session, frames = face_env
    session.handle_command({"type": "face_confirm"})
    frames(40, face="A")
    n = session.faceprint_view()["samples"]
    frames(40, face="C")                        # not A: nothing is added to A's print
    assert session.faceprint_view()["samples"] == n


def test_face_print_can_be_turned_off_and_forgotten(face_env):
    session, frames = face_env
    session.handle_command({"type": "face_confirm"})
    frames(40)
    reply = session.handle_command({"type": "faceprint_set", "enabled": False})[0]
    assert reply["personal"]["faceprint"] == {"enabled": False, "samples": 0, "ready": False}
    frames(30)
    assert session.faceprint_view()["samples"] == 0
    assert not session.users.face_store(session.user["id"]).exists()
    assert session.handle_command({"type": "faceprint_set", "enabled": "yes"})[0]["ok"] is False
    session.handle_command({"type": "faceprint_set", "enabled": True})
    frames(40)
    assert session.faceprint_view()["ready"]
    session.handle_command({"type": "faceprint_forget"})
    assert session.faceprint_view()["samples"] == 0
