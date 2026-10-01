"""Glasses: noticing them and reflections on their lenses (the real face mesh on
glasses painted onto the test portrait), with smoothing and hysteresis."""

from pathlib import Path

import numpy as np
import pytest

from paralic import glasses as G
from paralic.glasses import GLARE_OFF, GLARE_ON, GLASSES_OFF, GLASSES_ON, GlassesDetector, GlassesMeasure, measure
from tests import face_images

MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


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
