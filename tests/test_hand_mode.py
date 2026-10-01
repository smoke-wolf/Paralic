"""Hand mode inside the unified session: frames, gestures, per-person setups, the server."""

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from paralic.server import create_app
from paralic.session import TrackerSession, pack_frame
from paralic.users import UserStore
from tests.fakes import FakeHandTracker, ManualClock, make_hand, tiny_jpeg
from tests.test_hand_calibration import GRID

JPEG = tiny_jpeg()
SCREEN = {"w": 1600, "h": 900, "dpr": 1}


@pytest.fixture
def hand(tmp_path):
    tracker = FakeHandTracker()
    clock = ManualClock()
    users = UserStore(tmp_path)
    session = TrackerSession(lambda: tracker, users, clock=clock, mode="hand")
    session.handle_command({"type": "hello", "screen": SCREEN})
    state = {"id": 0}

    def frame(pts=None, label=None):
        state["id"] += 1
        tracker.push(pts)
        header = {"id": state["id"]}
        if label:
            header["label"] = label
        out = session.handle_frame(pack_frame(header, JPEG))
        clock.tick()
        return out

    def setup(mode="full"):
        assert session.handle_command({"type": "hand_calibration_start", "mode": mode})[0]["ok"]
        if mode == "full":
            for _ in range(25):
                frame(make_hand(fingers="spread"), {"kind": "hspan"})
        for i, (fx, fy) in enumerate(GRID):
            for _ in range(15):
                frame(make_hand(tip=(1 - (0.35 + 0.3 * fx), 0.3 + 0.3 * fy)),
                      {"kind": "hpoint", "i": i, "fx": fx, "fy": fy})
        if mode == "full":
            for k in range(120):
                frame(make_hand(pinch=0.2 if (k // 12) % 2 else 1.0), {"kind": "hpinch", "state": "cycle"})
        return session.handle_command({"type": "hand_calibration_fit"})[0]

    session.frame, session.setup, session.tracker_fake, session.clock_fake = frame, setup, tracker, clock
    return session


def _of(messages, kind):
    return [m for m in messages if m["type"] == kind]


def test_the_fingertip_moves_the_cursor_in_screen_pixels(hand):
    out = hand.frame(make_hand(tip=(0.25, 0.5)))
    msg = out[0]
    assert msg["type"] == "frame" and msg["face"] is True and "hand" in msg and msg["fps"] >= 0
    assert msg["gaze"] == pytest.approx([0.75 * 1600, 0.5 * 900], abs=1)
    lost = hand.frame(None)[0]
    assert lost["face"] is False and lost["gaze"] == msg["gaze"]       # stays where it was


def test_a_pinch_clicks_where_the_finger_aimed(hand):
    out = []
    for _ in range(5):
        out += hand.frame(make_hand(tip=(0.40, 0.40), pinch=1.2))
    aimed = out[-1]["id"]
    for k in range(1, 6):          # the fingertip drifts while the fingers close
        out += hand.frame(make_hand(tip=(0.40 + 0.01 * k, 0.40), pinch=1.0 - 0.17 * k))
    held = out[-1]
    out += hand.frame(make_hand(tip=(0.45, 0.40), pinch=1.0))
    first = _of(out, "blink")
    click = _of(out, "double_blink")
    assert len(first) == 1 and first[0]["n"] == 1 and first[0]["hand"]
    assert len(click) == 1 and click[0]["pre_frame"] == aimed
    assert click[0]["at"] == pytest.approx([0.60 * 1600, 0.40 * 900], abs=2)
    assert held["frozen"] and held["gaze"] == pytest.approx(click[0]["at"], abs=1)   # cursor held while pinching


def test_open_hand_asks_the_page_to_toggle_pause_once(hand):
    out = []
    for _ in range(60):
        out += hand.frame(make_hand(fingers="spread"))
    assert len(_of(out, "hand_palm")) == 1


def test_pinch_drag_scrolls_in_pixels(hand):
    out = []
    for _ in range(3):
        out += hand.frame(make_hand(centre=(0.5, 0.5), pinch=1.2))
    for k in range(12):
        out += hand.frame(make_hand(centre=(0.5, 0.5 + 0.012 * k), pinch=0.2))
    out += hand.frame(make_hand(centre=(0.5, 0.5 + 0.012 * 11), pinch=1.2))
    scroll = sum(m["dy"] for m in _of(out, "hand_scroll"))
    assert scroll == pytest.approx(0.132 / 0.16 * 0.5 * 900, rel=0.05)
    assert not _of(out, "double_blink") and _of(out, "blink_expired")


def test_hand_setup_is_saved_for_the_person(hand, tmp_path):
    hello = hand.handle_command({"type": "hello", "screen": SCREEN})[0]
    assert hello["mode"] == "hand" and hello["hand"] is None and "click" in hello["hand_gestures"]
    result = hand.setup()
    assert result["ok"] and result["saved"] and result["pinch_tuned"] and result["points"] == len(GRID)
    me = hand.user["id"]
    assert (tmp_path / "users" / me / "hand.json").is_file()
    assert any(u["hands"] for u in hand.users.list() if u["id"] == me)
    # A new tab (session) for the same person starts with it.
    again = TrackerSession(lambda: FakeHandTracker(), UserStore(tmp_path), mode="hand")
    view = again.handle_command({"type": "hello", "screen": SCREEN})[0]["hand"]
    assert view["calibrated"] and view["points"] == len(GRID)
    # Someone new has no hand setup; switching back brings it back.
    hand.handle_command({"type": "user_create", "name": "Sam"})
    assert hand.hand.calibration is None and hand.handle_command({"type": "users"})[0]["hand"] is None
    reply = hand.handle_command({"type": "user_select", "id": me})[0]
    assert reply["hand"]["calibrated"] and hand.hand.calibration is not None
    # Quick re-point keeps the pinch.
    repoint = hand.setup("point")
    assert repoint["ok"] and repoint["pinch_on"] == result["pinch_on"]


def test_failed_setup_says_why_and_ends_setup(hand):
    hand.handle_command({"type": "hand_calibration_start", "mode": "full"})
    reply = hand.handle_command({"type": "hand_calibration_fit"})[0]
    assert reply["type"] == "hand_calibration_result" and reply["ok"] is False and "dots" in reply["error"]
    assert not hand.hand.calibrating and not hand.hand.recognizer.setup


def test_eye_and_hand_commands_answer_in_the_wrong_mode(hand, tmp_path):
    reply = hand.handle_command({"type": "calibration_start", "mode": "full"})[0]
    assert reply["type"] == "calibration_started" and reply["ok"] is False and "hand mode" in reply["error"]
    eyes = TrackerSession(lambda: None, UserStore(tmp_path))
    reply = eyes.handle_command({"type": "hand_calibration_start"})[0]
    assert reply["type"] == "hand_calibration_started" and reply["ok"] is False
    # Shared commands work in both.
    assert hand.handle_command({"type": "personal_get"})[0]["type"] == "personal"
    assert eyes.handle_command({"type": "hand_profile_load"})[0]["hand"] is None


def test_the_shared_preview_hand_profile_moves_into_a_person(tmp_path):
    (tmp_path / "hand_profile.json").write_text(json.dumps(
        {"calibration": {"span": 0.2, "pinch_on": 0.3, "pinch_off": 0.5,
                         "pointing": {"coef": np.eye(3, 2).tolist(), "quadratic": False}}}))
    users = UserStore(tmp_path)
    me = users.ensure_active()["id"]
    assert not (tmp_path / "hand_profile.json").exists()
    assert users.load_hand(me)["calibration"]["pinch_on"] == 0.3
    session = TrackerSession(lambda: FakeHandTracker(), users, mode="hand")
    assert session.hand.calibration is not None and session.hand.recognizer.thresholds()[0] == 0.3


def test_desktop_control_follows_the_hand(hand):
    class FakeOS:
        available = True
        moves, clicks, scrolls = [], [], []

        def status(self):
            class S:
                available, trusted, reason = True, True, ""
            return S()

        def is_trusted(self):
            return True

        def request_trust(self):
            pass

        def screen_bounds(self):
            return (0.0, 0.0, 1600.0, 900.0)

        def move(self, x, y):
            self.moves.append((x, y))

        def click(self, x, y):
            self.clicks.append((x, y))

        def scroll(self, lines):
            self.scrolls.append(lines)

    fake = FakeOS()
    hand.system.os = fake
    assert hand.handle_command({"type": "system_control", "enabled": True})[0]["enabled"]
    for _ in range(4):
        hand.frame(make_hand(tip=(0.40, 0.40), pinch=1.2))
    for k in range(1, 6):
        hand.frame(make_hand(tip=(0.40 + 0.01 * k, 0.40), pinch=1.0 - 0.17 * k))
    hand.frame(make_hand(tip=(0.45, 0.40), pinch=1.0))
    assert fake.moves and fake.clicks == [pytest.approx((960.0, 360.0), abs=2)]
    # Lowering the hand for a few seconds does not turn desktop control off.
    for _ in range(150):
        hand.frame(None)
    assert hand.system.enabled
    # Pinch-drag scrolls the desktop too (down = negative lines).
    for k in range(12):
        hand.frame(make_hand(centre=(0.5, 0.5 + 0.02 * k), pinch=0.2))
    assert fake.scrolls and sum(fake.scrolls) < 0


def test_server_hand_mode_socket_and_status(tmp_path):
    tracker = FakeHandTracker()
    loaded = []

    def loader():
        time.sleep(0.2)
        loaded.append(True)
        return lambda: tracker

    app = create_app(data_dir=tmp_path, tracker_factory=None, hand_loader=loader)
    with TestClient(app, base_url="http://localhost:8000", client=("127.0.0.1", 50000)) as client:
        assert client.get("/api/status").json()["hands"] == "loading"
        with client.websocket_connect("/ws?mode=hand") as ws:
            assert json.loads(ws.receive_text())["type"] == "fatal"      # not ready yet
        for _ in range(50):
            if client.get("/api/status").json()["hands"] == "ready":
                break
            time.sleep(0.05)
        assert client.get("/api/status").json()["hands"] == "ready" and loaded
        with client.websocket_connect("/ws?mode=hand") as ws:
            ws.send_text(json.dumps({"type": "hello", "screen": SCREEN}))
            hello = json.loads(ws.receive_text())
            assert hello["type"] == "hello" and hello["mode"] == "hand"
            tracker.push(make_hand(tip=(0.5, 0.5)))
            ws.send_bytes(pack_frame({"id": 1}, JPEG))
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "frame" and msg["face"] is True and msg["gaze"] == pytest.approx([800, 450], abs=1)


def test_server_without_hand_mode(tmp_path):
    def broken():
        raise RuntimeError("no model")

    app = create_app(data_dir=tmp_path, tracker_factory=None, hand_loader=broken)
    with TestClient(app, base_url="http://localhost:8000", client=("127.0.0.1", 50000)) as client:
        for _ in range(50):
            status = client.get("/api/status").json()
            if status["hands"] != "loading":
                break
            time.sleep(0.02)
        assert status["hands"] == "unavailable" and "no model" in status["hands_error"]
