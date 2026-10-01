"""System-wide gaze control: kill-switch logic and safe OS binding.

These tests NEVER move the real cursor: SystemController is driven with a fake
OS controller, and the real OSController is only exercised with read-only calls
(status / is_trusted / pure clamp) - never move/click/scroll/request_trust.
"""

from paralic.oscontrol import OSStatus
from paralic.system_control import SystemController, ControlConfig


class FakeOS:
    def __init__(self, available=True, trusted=True, bounds=(0.0, 0.0, 1440.0, 900.0)):
        self._available, self._trusted, self._bounds = available, trusted, bounds
        self.moves, self.clicks, self.scrolls, self.trust_requests = [], [], [], 0

    @property
    def available(self):
        return self._available

    def is_trusted(self):
        return self._trusted

    def request_trust(self):
        self.trust_requests += 1
        return self._trusted

    def status(self):
        if not self._available:
            return OSStatus(False, False, "macOS only")
        if not self._trusted:
            return OSStatus(True, False, "grant Accessibility")
        return OSStatus(True, True, "")

    def screen_bounds(self):
        return self._bounds

    def move(self, x, y):
        self.moves.append((x, y))

    def click(self, x, y):
        self.clicks.append((x, y))

    def scroll(self, dy):
        self.scrolls.append(dy)


def test_enable_requires_available():
    c = SystemController(FakeOS(available=False))
    assert c.enable()["enabled"] is False and c.enabled is False


def test_enable_prompts_when_not_trusted():
    os = FakeOS(trusted=False)
    c = SystemController(os)
    r = c.enable()
    assert os.trust_requests == 1 and r["enabled"] is False and r.get("needs_permission")
    assert c.enabled is False


def test_enable_trusted_then_moves_to_gaze():
    os = FakeOS()
    c = SystemController(os)
    assert c.enable()["enabled"] is True
    res = c.update(0.0, (200.0, 300.0), True)
    assert res.moved == (200.0, 300.0) and os.moves[-1] == (200.0, 300.0)


def test_disabled_is_a_noop():
    os = FakeOS()
    c = SystemController(os)              # never enabled
    res = c.update(0.0, (10.0, 10.0), True)
    assert res.moved is None and not os.moves


def test_no_face_timeout_disables():
    os = FakeOS()
    c = SystemController(os, ControlConfig(no_face_timeout_s=2.0))
    c.enable()
    c.update(0.0, (100.0, 100.0), True)          # last face at t=0
    assert c.update(1.0, None, False).disabled_reason is None and c.enabled
    r = c.update(3.1, None, False)
    assert r.disabled_reason == "no-face" and c.enabled is False


def test_corner_dwell_disables():
    os = FakeOS()
    c = SystemController(os, ControlConfig(corner_dwell_s=1.0, corner_px=60.0))
    c.enable()
    c.update(0.0, (10.0, 10.0), True)            # enter corner, start dwell
    assert c.enabled
    r = c.update(1.1, (10.0, 10.0), True)
    assert r.disabled_reason == "corner" and c.enabled is False


def test_corner_timer_resets_on_leaving():
    os = FakeOS()
    c = SystemController(os, ControlConfig(corner_dwell_s=1.0, corner_px=60.0))
    c.enable()
    c.update(0.0, (10.0, 10.0), True)
    c.update(0.5, (700.0, 500.0), True)          # left the corner
    assert c.update(1.1, (10.0, 10.0), True).disabled_reason is None and c.enabled


def test_edge_scroll_top_and_bottom():
    os = FakeOS(bounds=(0.0, 0.0, 1440.0, 900.0))
    c = SystemController(os, ControlConfig(edge_margin_frac=0.1, scroll_interval_s=0.0))
    c.enable()
    up = c.update(0.0, (700.0, 10.0), True)       # top band
    assert up.scrolled and up.scrolled > 0
    down = c.update(1.0, (700.0, 895.0), True)    # bottom band
    assert down.scrolled and down.scrolled < 0


def test_click_only_when_enabled():
    os = FakeOS()
    c = SystemController(os)
    assert c.click((5.0, 5.0)) is False and not os.clicks
    c.enable()
    assert c.click((5.0, 5.0)) is True and os.clicks[-1] == (5.0, 5.0)


def test_real_oscontrol_is_read_only_safe():
    # Construct the real controller and only touch read-only / pure paths.
    # Never call move/click/scroll/request_trust here (would move the cursor
    # or pop a permission prompt on a live Mac).
    from paralic.oscontrol import OSController
    c = OSController()
    assert isinstance(c.available, bool)
    st = c.status()
    assert isinstance(st.available, bool) and isinstance(st.reason, str)
    assert isinstance(c.is_trusted(), bool)
    c._bounds = (0.0, 0.0, 100.0, 100.0)          # pure clamp math, posts nothing
    assert c._clamp(-5.0, 200.0) == (0.0, 99.0)
    assert c._clamp(50.0, 50.0) == (50.0, 50.0)
