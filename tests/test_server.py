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
    app = create_app(profile_path=tmp_path / "profile.json", tracker_factory=lambda: tracker)
    with TestClient(app) as c:
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


def test_websocket_rejects_foreign_origin(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_text()
    assert exc.value.code == 1008


def test_websocket_accepts_localhost_origin(client):
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8000"}) as ws:
        ws.send_text(json.dumps({"type": "ping", "t": 5}))
        assert json.loads(ws.receive_text()) == {"type": "pong", "t": 5}


def test_missing_model_reports_fatal(tmp_path):
    app = create_app(profile_path=tmp_path / "p.json", tracker_factory=None, model_error="model missing")
    with TestClient(app) as c:
        assert c.get("/api/status").json()["error"] == "model missing"
        with c.websocket_connect("/ws") as ws:
            msg = json.loads(ws.receive_text())
            assert msg == {"type": "fatal", "error": "model missing"}
