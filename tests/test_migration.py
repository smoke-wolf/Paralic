"""Safe migration: a profile from an older feature set is never used against
the wrong features. A version-1 profile (the 20 original columns, which mean
the same today) keeps working in a compatibility mode; anything else degrades
to "please recalibrate", never a crash or a silent misprediction."""

import json

import numpy as np
import pytest

from paralic.calibration import PROFILE_VERSION, ProfileStore
from paralic.features import FEATURE_NAMES, FEATURE_VERSION, NUM_FEATURES
from paralic.gazenet import GazeNet
from paralic.session import TrackerSession
from paralic.users import UserStore
from tests.fakes import FakeTracker, patch_session
from tests.synthetic import Head, VirtualUser, simulate_calibration


def _write(path, doc):
    path.write_text(json.dumps(doc), encoding="utf-8")


def _v1_profile(path, seed=3):
    """A real profile from before the full-mesh features: a network trained on
    the 20 original columns, saved with the version-1 layout."""
    user = VirtualUser(seed=seed, noise=0.5)
    X, Y, G = simulate_calibration(user, seed=seed)
    X20 = X[:, :20]
    model, _ = GazeNet.train(X20, Y, G, iters=150, ensemble=1, l2_grid=(1e-2,))
    _write(path, {
        "version": 1, "feature_version": 1,
        "created": "2026-01-01T00:00:00", "screen": {"w": 1920, "h": 1080},
        "accuracy_px": 42.0,
        "feature_names": list(FEATURE_NAMES[:20]),
        "model": model.to_dict(),
        "data": {"features": X20[:30].round(5).tolist(), "targets": Y[:30].tolist(),
                 "kinds": ["cal"] * 30, "points": [0] * 30, "times": [0.0] * 30},
    })
    return model, user


def test_old_profile_keeps_working_in_compatibility_mode(tmp_path):
    p = tmp_path / "profile.json"
    old_model, user = _v1_profile(p)
    store = ProfileStore(p)
    summary = store.summary()
    assert summary is not None and summary["legacy"] is True and summary["accuracy_px"] == 42.0
    model, data, meta = store.load()
    assert meta["legacy"] is True and model.meta["legacy"] is True
    # Its recordings have the old layout: not used for training.
    assert len(data.frames) == 0
    # Columns 0..19 mean the same today, so it predicts exactly as before.
    for sx, sy in [(300, 200), (960, 540), (1700, 900)]:
        vec = user.features(sx, sy, Head())
        assert vec.shape == (NUM_FEATURES,)
        np.testing.assert_allclose(model.predict(vec), old_model.predict(vec[:20]), atol=1e-6)


def test_replacing_an_old_profile_keeps_a_backup(tmp_path):
    p = tmp_path / "profile.json"
    _v1_profile(p)
    original = p.read_text(encoding="utf-8")
    store = ProfileStore(p)
    model, data, _ = store.load()
    store.save(model, data, None, 30.0)
    assert store.backup_path.read_text(encoding="utf-8") == original
    # The saved file is a current-version profile that still works.
    model2, _, meta = ProfileStore(p).load()
    assert meta["legacy"] is True
    vec = VirtualUser(seed=3, noise=0).features(800, 400, Head())
    np.testing.assert_allclose(model2.predict(vec), model.predict(vec), atol=1e-6)
    # A later save never overwrites the backup.
    store.save(model2, data, None, 25.0)
    assert store.backup_path.read_text(encoding="utf-8") == original


def test_unknown_feature_set_is_rejected(tmp_path):
    p = tmp_path / "profile.json"
    _write(p, {
        "version": 1, "feature_version": 1, "created": "2026-01-01T00:00:00", "screen": None,
        "accuracy_px": None, "feature_names": [f"f{i}" for i in range(20)], "model": {}, "data": {},
    })
    store = ProfileStore(p)
    assert store.summary() is None
    with pytest.raises(ValueError):
        store.load()


def test_session_uses_an_old_profile_but_waits_for_a_full_calibration_to_fine_tune(tmp_path, monkeypatch):
    patch_session(monkeypatch)
    users = UserStore(tmp_path)
    session = TrackerSession(lambda: FakeTracker(), users)
    _v1_profile(session.profiles.path)
    loaded = session.handle_command({"type": "profile_load"})[0]
    assert loaded["loaded"] is True and loaded["legacy"] is True
    assert loaded["personal"]["legacy"] is True
    reply = session.handle_command({"type": "finetune"})[0]
    assert reply["ok"] is False and "full calibration" in reply["error"]


def test_feature_version_bump_rejects_otherwise_matching_profile(tmp_path):
    p = tmp_path / "profile.json"
    # Same feature names/count as today but an older feature_version stamp.
    _write(p, {
        "version": PROFILE_VERSION, "feature_version": FEATURE_VERSION - 1,
        "created": "2026-01-01T00:00:00", "screen": None, "accuracy_px": None,
        "feature_names": list(FEATURE_NAMES), "model": {}, "data": {},
    })
    with pytest.raises(ValueError):
        ProfileStore(p).load()
