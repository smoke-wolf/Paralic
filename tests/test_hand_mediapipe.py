"""Hand mode with the real MediaPipe hand network on public test photos."""

from pathlib import Path

import numpy as np
import pytest

from paralic.hand_gestures import extended_fingers, stop_palm
from paralic.session import TrackerSession, pack_frame
from paralic.users import UserStore
from tests.fakes import ManualClock
from tests.hand_images import hand_image

MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


@pytest.fixture(scope="module")
def model_bytes():
    if not MODEL.is_file():
        try:
            from paralic.model_assets import ensure_hand_model

            ensure_hand_model(MODEL)
        except Exception as exc:
            pytest.skip(f"hand model not available: {exc}")
    return MODEL.read_bytes()


def _image(name):
    img = hand_image(name)
    if img is None:
        pytest.skip("could not download the hand test photos")
    return img


def _landmarks(model_bytes, rgb):
    from paralic.hands import HandTracker

    tracker = HandTracker(model_bytes)
    try:
        obs = None
        for k in range(3):                     # VIDEO mode settles on the hand
            obs = tracker.process(rgb, 33 * k)
        return obs
    finally:
        tracker.close()


def test_the_network_reads_the_hand_poses(model_bytes):
    pointing = _landmarks(model_bytes, _image("pointing_up"))
    assert pointing is not None
    pts = pointing.points_norm[:, :2]
    assert extended_fingers(pts) == {"index": True, "middle": False, "ring": False, "pinky": False}
    assert not stop_palm(pts)
    victory = _landmarks(model_bytes, _image("victory")).points_norm[:, :2]
    assert not stop_palm(victory) and extended_fingers(victory)["middle"]
    assert stop_palm(_landmarks(model_bytes, _image("open_hands")).points_norm[:, :2])


def test_hand_session_with_the_real_network(model_bytes, tmp_path):
    """Pointing moves the cursor to the (mirrored) fingertip; an open hand held
    up asks for one pause toggle."""
    import cv2

    from paralic.hands import HandTracker

    clock = ManualClock()
    session = TrackerSession(lambda: HandTracker(model_bytes), UserStore(tmp_path), clock=clock, mode="hand")
    session.handle_command({"type": "hello", "screen": {"w": 1000, "h": 1000, "dpr": 1}})

    def frames(rgb, n, first_id):
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        out = []
        for k in range(n):
            out += session.handle_frame(pack_frame({"id": first_id + k}, jpg.tobytes()))
            clock.tick()
        return out

    pointing = _image("pointing_up")
    out = frames(pointing, 6, 1)
    msg = [m for m in out if m["type"] == "frame"][-1]
    assert msg["face"] is True and msg["hand"]["open"] is False
    tip = _landmarks(model_bytes, pointing).points_norm[8]
    assert msg["gaze"] == pytest.approx([(1 - tip[0]) * 1000, tip[1] * 1000], abs=25)
    out = frames(_image("open_hands"), 40, 100)
    assert sum(m["type"] == "hand_palm" for m in out) == 1
    assert not any(m["type"] == "double_blink" for m in out)
    session.close()
