"""Integration tests with the real MediaPipe FaceLandmarker networks.

Skipped automatically when the model file or the test portrait is unavailable
(run ``python -m paralic --no-browser`` once to download the model).
"""

from pathlib import Path

import numpy as np
import pytest

from paralic.calibration import ProfileStore
from paralic.features import extract_features
from paralic.session import TrackerSession, pack_frame
from tests import face_images

MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

cv2 = pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def model_bytes():
    if not MODEL.exists():
        pytest.skip("face landmarker model not downloaded")
    try:
        from paralic.tracker import FaceTracker

        FaceTracker(MODEL.read_bytes()).close()
    except Exception as exc:  # pragma: no cover - platform specific
        pytest.skip(f"MediaPipe unavailable: {exc}")
    return MODEL.read_bytes()


@pytest.fixture(scope="module")
def face():
    img = face_images.portrait()
    if img is None:
        pytest.skip("test portrait unavailable (offline?)")
    return img


def features_of(model_bytes, img_bgr, frames=6):
    from paralic.tracker import FaceTracker

    tracker = FaceTracker(model_bytes)
    try:
        rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        obs = None
        for i in range(frames):
            obs = tracker.process(rgb, i * 33)
        assert obs is not None, "no face found"
        return obs, extract_features(obs.points_px, obs.image_size, obs.blendshapes, obs.matrix)
    finally:
        tracker.close()


def test_face_is_found_with_iris_and_head_pose(model_bytes, face):
    obs, f = features_of(model_bytes, face)
    assert obs.points_px.shape == (478, 3)
    assert obs.matrix is not None and "eyeBlinkLeft" in obs.blendshapes
    assert abs(f.yaw_deg) < 20 and abs(f.pitch_deg) < 25
    assert 30 < f.distance_cm < 150
    assert 0.1 < f.aperture < 0.4


def test_closed_eyes_raise_closure(model_bytes, face):
    obs, open_f = features_of(model_bytes, face)
    closed = face_images.close_eyes(face, obs.points_px[:, :2])
    _, closed_f = features_of(model_bytes, closed)
    assert closed_f.closure > open_f.closure + 0.2


def test_iris_shift_moves_gaze_features(model_bytes, face):
    obs, centre = features_of(model_bytes, face)
    pts = obs.points_px[:, :2]
    _, left = features_of(model_bytes, face_images.shift_iris(face, pts, -0.12))
    _, right = features_of(model_bytes, face_images.shift_iris(face, pts, +0.12))
    for eye in ("right_eye", "left_eye"):
        assert getattr(left, eye).dx < getattr(centre, eye).dx - 0.02
        assert getattr(right, eye).dx > getattr(centre, eye).dx + 0.02


def test_end_to_end_calibration_with_synthetic_gaze(model_bytes, face, tmp_path):
    """Calibrate on images whose iris position encodes the screen x position."""
    from paralic.tracker import FaceTracker

    obs, _ = features_of(model_bytes, face)
    pts = obs.points_px[:, :2]
    rng = np.random.default_rng(0)
    cache = {}

    def jpeg(fx):
        if fx not in cache:
            cache[fx] = face_images.shift_iris(face, pts, (fx - 0.5) * 0.24)
        noisy = np.clip(cache[fx].astype(np.int16) + rng.normal(0, 2, cache[fx].shape), 0, 255).astype(np.uint8)
        return cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    session = TrackerSession(lambda: FaceTracker(model_bytes), ProfileStore(tmp_path / "p.json"))
    try:
        session.handle_command({"type": "calibration_start"})
        fid = 0
        xs = [0.06, 0.35, 0.65, 0.94]
        for i, (fx, fy) in enumerate([(x, y) for y in (0.1, 0.5, 0.9) for x in xs]):
            for k in range(14):
                fid += 1
                label = {"x": fx * 1920, "y": fy * 1080, "kind": "cal", "pt": i} if k >= 4 else None
                session.handle_frame(pack_frame({"id": fid, **({"label": label} if label else {})}, jpeg(fx)))
        res = session.handle_command({"type": "calibration_fit", "mode": "full"})[0]
        assert res["ok"], res

        preds = {}
        for fx in (0.2, 0.5, 0.8):
            raw = []
            for _ in range(6):
                fid += 1
                raw.append(session.handle_frame(pack_frame({"id": fid}, jpeg(fx)))[0]["raw"][0])
            preds[fx] = float(np.mean(raw[2:]))
        # Predictions are ordered like the targets and land in the right third of the screen.
        assert preds[0.2] < preds[0.5] < preds[0.8]
        for fx, px in preds.items():
            assert abs(px - fx * 1920) < 1920 / 5, preds
    finally:
        session.close()
