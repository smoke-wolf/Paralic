"""Safe migration: a profile from an older feature set must not be used against
the current features — it degrades to "please recalibrate", never a crash or a
silent misprediction."""

import json

import pytest

from paralic.calibration import PROFILE_VERSION, ProfileStore
from paralic.features import FEATURE_NAMES, FEATURE_VERSION


def _write(path, doc):
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_old_feature_set_profile_is_rejected(tmp_path):
    p = tmp_path / "profile.json"
    # An old 20-feature profile (pre full-mesh).
    _write(p, {
        "version": 1, "feature_version": 1,
        "created": "2026-01-01T00:00:00", "screen": {"w": 1920, "h": 1080},
        "accuracy_px": 42.0,
        "feature_names": ["r_dx", "r_dy", "l_dx", "l_dy", "r_open", "l_open",
                          "bs_look_in_l", "bs_look_out_l", "bs_look_up_l", "bs_look_down_l",
                          "bs_look_in_r", "bs_look_out_r", "bs_look_up_r", "bs_look_down_r",
                          "yaw", "pitch", "roll", "tx", "ty", "tz"],
        "model": {}, "data": {},
    })
    store = ProfileStore(p)
    assert store.exists()
    # summary() swallows the mismatch so /api/status shows no usable profile.
    assert store.summary() is None
    # load() refuses rather than mispredict; the message tells the user to redo it.
    with pytest.raises(ValueError):
        store.load()


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
