"""Two people in front of the camera, with the real face network: only the
person being followed controls Paralic."""

from pathlib import Path

import numpy as np
import pytest

from paralic.session import TrackerSession, pack_frame
from paralic.users import UserStore
from tests import face_images
from tests.fakes import ManualClock

MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


@pytest.fixture(scope="module")
def two_people():
    if not MODEL.is_file():
        pytest.skip("face model not available")
    img = face_images.face_image("man-woman-okay.jpg")
    if img is None:
        pytest.skip("could not download the two-person photo")
    return img


def _jpeg(bgr):
    import cv2

    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return buf.tobytes()


def test_only_the_followed_person_controls(two_people, tmp_path):
    from paralic.tracker import FaceTracker

    clock = ManualClock()
    session = TrackerSession(lambda: FaceTracker(MODEL.read_bytes()), UserStore(tmp_path), clock=clock)
    session.handle_command({"type": "hello", "screen": {"w": 1600, "h": 900, "dpr": 1}})
    state = {"id": 0}

    def frames(bgr, n):
        out = []
        jpg = _jpeg(bgr)
        for _ in range(n):
            state["id"] += 1
            out += session.handle_frame(pack_frame({"id": state["id"]}, jpg))
            clock.tick()
        return out

    both = [m for m in frames(two_people, 10) if m["type"] == "frame"]
    assert all(m["face"] and m["faces"] == 2 and len(m["others"]) == 1 for m in both[2:])
    me = both[-1]["you"]
    assert all(m["you"] == pytest.approx(me, abs=0.03) for m in both[2:])

    # The followed person moves out of view (their face is painted over); the other stays.
    h, w = two_people.shape[:2]
    alone = two_people.copy()
    x0, y0, x1, y1 = me
    alone[int(y0 * h) - 10:int(y1 * h) + 10, int(x0 * w) - 10:int(x1 * w) + 10] = 128
    out = frames(alone, 40)
    shown = [m for m in out if m["type"] == "frame"]
    assert all(not m["face"] and m["waiting"] and m["raw"] is None for m in shown[3:])
    assert not any(m["type"] in ("blink", "double_blink", "wink_start", "long_close") for m in out)

    # They come back: theirs again, not the other person's.
    back = [m for m in frames(two_people, 8) if m["type"] == "frame"]
    assert back[-1]["face"] and back[-1]["you"] == pytest.approx(me, abs=0.03)
    session.close()
