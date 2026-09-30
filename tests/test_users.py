import json

import pytest

from paralic.users import UnknownUser, UserStore, clean_name


def test_first_run_creates_person_1(tmp_path):
    store = UserStore(tmp_path)
    user = store.ensure_active()
    assert user["name"] == "Person 1"
    assert store.active_id() == user["id"]
    assert store.user_dir(user["id"]).is_dir()


def test_create_select_rename_delete(tmp_path):
    store = UserStore(tmp_path)
    a = store.create("Alex")
    b = store.create()
    assert b["name"] == "Person 2" and store.active_id() == b["id"]
    store.select(a["id"])
    assert store.active_id() == a["id"]
    store.rename(b["id"], "  Sam\n ")
    assert {u["name"] for u in store.list()} == {"Alex", "Sam"}
    store.delete(a["id"])
    assert store.active_id() is None
    assert store.ensure_active()["id"] == b["id"]
    with pytest.raises(UnknownUser):
        store.select(a["id"])


def test_rejects_unsafe_ids(tmp_path):
    store = UserStore(tmp_path)
    for bad in ("../etc", "u123", "uXXXXXXXX", ""):
        with pytest.raises(UnknownUser):
            store.user_dir(bad)


def test_names_are_cleaned():
    assert clean_name("  a\tb\x00c  ") == "a bc"
    assert len(clean_name("x" * 100)) == 32


def test_legacy_profile_is_migrated(tmp_path):
    (tmp_path / "profile.json").write_text(json.dumps({"hello": 1}))
    store = UserStore(tmp_path)
    users = store.list()
    assert len(users) == 1 and users[0]["name"] == "Person 1"
    moved = store.user_dir(users[0]["id"]) / "profile.json"
    assert moved.exists() and not (tmp_path / "profile.json").exists()


def test_personal_and_experiment_files(tmp_path):
    store = UserStore(tmp_path)
    user = store.ensure_active()
    assert store.load_personal(user["id"]) == {}
    store.save_personal(user["id"], {"blink": {"sensitivity": 0.2}})
    store.save_experiments(user["id"], {"smoothing": {"epoch": 1, "trials": []}})
    assert store.load_personal(user["id"])["blink"]["sensitivity"] == 0.2
    assert store.load_experiments(user["id"])["smoothing"]["epoch"] == 1
