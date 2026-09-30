import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from paralic.server import create_app
from paralic.session import pack_frame
from tests.fakes import FakeTracker, fake_features, patch_session, tiny_jpeg
from tests.synthetic import Head, VirtualUser


@pytest.fixture
def client(monkeypatch, tmp_path):
    patch_session(monkeypatch)
    tracker = FakeTracker()
    app = create_app(data_dir=tmp_path, tracker_factory=lambda: tracker)
    # Like a client on this machine (the page is http://localhost:8000; relative
    # WebSocket URLs in the test client use the host name "testserver").
    with TestClient(app, base_url="http://localhost:8000", client=("127.0.0.1", 50000)) as c:
        c.tracker = tracker
        yield c


def test_status_endpoint(client):
    data = client.get("/api/status").json()
    assert data["tracker"] is True and data["error"] is None and data["profile"] is None


def test_serves_website(client):
    r = client.get("/")
    assert r.status_code == 200 and "Paralic" in r.text
    assert client.get("/js/main.js").status_code == 200


def test_websocket_hello_and_frames(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello", "screen": {"w": 1920, "h": 1080, "dpr": 1}}))
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "hello" and hello["model"] is False

        vec = VirtualUser(seed=1).features(960, 540, Head())
        client.tracker.push(fake_features(vec))
        ws.send_bytes(pack_frame({"id": 1}, tiny_jpeg()))
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "frame" and msg["id"] == 1 and msg["face"] is True

        ws.send_bytes(pack_frame({"id": 2}, tiny_jpeg()))  # no face queued
        msg = json.loads(ws.receive_text())
        assert msg["id"] == 2 and msg["face"] is False


def test_websocket_rejects_invalid_json(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_text("{not json")
        assert json.loads(ws.receive_text())["type"] == "error"


@pytest.mark.parametrize("origin", [
    "https://evil.example",          # another website
    "http://localhost:3000",         # another server on this machine
    "null",                          # file:// pages
])
def test_websocket_rejects_other_origins(client, origin):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("ws://localhost:8000/ws", headers={"origin": origin}) as ws:
            ws.receive_text()
    assert exc.value.code == 1008


def test_websocket_accepts_same_origin(client):
    with client.websocket_connect("ws://localhost:8000/ws", headers={"origin": "http://localhost:8000"}) as ws:
        ws.send_text(json.dumps({"type": "ping", "t": 5}))
        assert json.loads(ws.receive_text()) == {"type": "pong", "t": 5}


def test_origin_rules():
    from paralic.server import _origin_allowed

    assert _origin_allowed("http://localhost:8000", "localhost:8000", "127.0.0.1")
    assert _origin_allowed("http://127.0.0.1:8001", "127.0.0.1:8001", "127.0.0.1")
    assert _origin_allowed("http://[::1]:8000", "[::1]:8000", "::1")
    # DNS rebinding: same origin, but not a host name this server is opened under.
    assert not _origin_allowed("http://evil.example:8000", "evil.example:8000", "127.0.0.1")
    assert _origin_allowed("http://evil.example:8000", "evil.example:8000", "10.0.0.2", allowed_hosts="*")
    # No Origin (scripts): only from this machine.
    assert _origin_allowed(None, "localhost:8000", "127.0.0.1")
    assert not _origin_allowed(None, "localhost:8000", "192.168.1.20")
    assert not _origin_allowed("http://localhost:8000", None, "127.0.0.1")


def test_origin_less_remote_client_rejected(tmp_path):
    app = create_app(data_dir=tmp_path, tracker_factory=FakeTracker)
    with TestClient(app, client=("192.168.1.20", 5000)) as c:
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws") as ws:
                ws.receive_text()


def test_command_errors_do_not_end_the_session(client, monkeypatch):
    import paralic.session as session_mod

    def boom(self, cmd):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(session_mod.TrackerSession, "_cmd_profile_load", boom)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "profile_load"}))
        reply = json.loads(ws.receive_text())
        assert reply["type"] == "profile" and reply["ok"] is False and reply["loaded"] is False
        assert "disk on fire" in reply["error"]
        ws.send_text(json.dumps({"type": "ping", "t": 1}))  # still alive
        assert json.loads(ws.receive_text())["type"] == "pong"


def test_missing_model_reports_fatal(tmp_path):
    app = create_app(data_dir=tmp_path, tracker_factory=None, model_error="model missing")
    with TestClient(app, client=("127.0.0.1", 50000)) as c:
        assert c.get("/api/status").json()["error"] == "model missing"
        with c.websocket_connect("/ws") as ws:
            msg = json.loads(ws.receive_text())
            assert msg == {"type": "fatal", "error": "model missing"}
