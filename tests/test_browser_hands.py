"""Browser test of hand mode (Playwright + Chromium): a scripted hand uses the site.

The server runs the real hand pipeline - the recogniser, the setup's fitting
and saving, the session - but its HandTracker is replaced by a scripted hand:
the test moves the fingertip, pinches and holds up an open hand the way a
person would, watching the page to see what it asks for, and checks what the
page does. The camera is Chromium's fake webcam (its picture is ignored).

The first visit runs the whole hand setup and then browses by hand; the second
(a new browser profile, same data) is welcomed back and re-points. Two short
tests check that hand mode waits while its model downloads, and that a dropped
connection cannot leave the hand setup stuck.

Skipped automatically when Playwright or a Chromium build is not available.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.fakes import FakeHandObservation, make_hand
from tests.test_browser import _serve, browser  # noqa: F401  (browser is a fixture)

OPEN, CLOSED = 1.0, 0.15         # thumb-index distance (palm widths): hand open, pinched
HAND_WIDTH = 0.12                # palm width in the camera image (normalised)
GRID_DOTS = 13                   # pointing dots in the hand setup (hand-calibration.js)


class Pose:
    """What the scripted hand is doing. The test changes it while the server's
    tracker reads it, one camera frame at a time: at once (``set``) or one
    step per frame (``play``, for movements such as a pinch)."""

    def __init__(self):
        self._cond = threading.Condition()
        self._state: dict = {}
        self._script: list[dict] = []    # changes still to come, one per camera frame
        self.frames = 0                  # camera frames the server has looked at
        self.reset()

    def reset(self) -> None:
        # Where this person's comfortable fingertip range starts: the mirrored
        # image x and the image y that point at the screen's top-left corner.
        # The range is 0.3 of the image each way.
        self.reach = (0.35, 0.30)
        with self._cond:
            self._script.clear()
        self.set(tip=(0.5, 0.45), drift=(0.0, 0.0), pinch=OPEN, fingers="curled", visible=False)

    def set(self, **changes) -> None:
        with self._cond:
            self._state = {**self._state, **changes}

    def get(self, key):
        with self._cond:
            return self._state[key]

    def point(self, fx: float, fy: float) -> None:
        """Point the fingertip at screen fraction (fx, fy)."""
        x0, y0 = self.reach
        self.set(tip=(1.0 - (x0 + 0.3 * fx), y0 + 0.3 * fy))

    def play(self, *steps: dict, timeout: float = 10.0) -> None:
        """Change the hand one step per camera frame; returns once every step was seen."""
        with self._cond:
            self._script.extend(steps)
            if not self._cond.wait_for(lambda: not self._script, timeout):
                raise AssertionError(f"the page stopped sending camera frames (after {self.frames})")

    def landmarks(self):
        """The hand in the next camera frame (None: no hand in view)."""
        with self._cond:
            if self._script:
                self._state = {**self._state, **self._script.pop(0)}
            state = dict(self._state)
            self.frames += 1
            self._cond.notify_all()
        if not state["visible"]:
            return None
        # The hand sits with its index fingertip at `tip`; `drift` moves only
        # the fingertip (closing fingers pull it), not the rest of the hand.
        rest = make_hand(fingers=state["fingers"], width=HAND_WIDTH)
        tip = np.asarray(state["tip"], float)
        centre = tip - (rest[8, :2] - rest[9, :2])               # point 9 is make_hand's centre
        drift = np.asarray(state["drift"], float)
        return make_hand(centre=tuple(centre), tip=tuple(tip + drift) if drift.any() else None,
                         pinch=state["pinch"], fingers=state["fingers"], width=HAND_WIDTH)

    def wait_frames(self, n: int = 3, timeout: float = 10.0) -> None:
        """Wait until the server has looked at ``n`` more camera frames."""
        with self._cond:
            target = self.frames + n
            if not self._cond.wait_for(lambda: self.frames >= target, timeout):
                raise AssertionError(f"the page stopped sending camera frames (after {self.frames})")


class ScriptedHandTracker:
    """Stands in for MediaPipe's HandTracker: ignores the image, returns the pose's hand."""

    def __init__(self, pose: Pose):
        self.pose = pose

    def process(self, rgb, timestamp_ms):
        pts = self.pose.landmarks()
        return None if pts is None else FakeHandObservation(points_norm=pts)

    def close(self):
        pass


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """Paralic with hand mode on a scripted hand; the data is kept between the visits."""
    from paralic.server import create_app

    pose = Pose()
    tracker = ScriptedHandTracker(pose)
    data_dir = tmp_path_factory.mktemp("data")
    app = create_app(data_dir=data_dir, tracker_factory=None, hand_tracker_factory=lambda: tracker)
    server, thread, url = _serve(app)
    yield SimpleNamespace(url=url, data_dir=data_dir, pose=pose)
    server.should_exit = True
    thread.join(timeout=5)


@contextmanager
def visit(browser, url):
    """Open Paralic in hand mode in a new browser profile (no full screen)."""
    ctx = browser.new_context(viewport={"width": 1600, "height": 900})
    try:
        ctx.grant_permissions(["camera"], origin=url)
        pg = ctx.new_page()
        pg.errors = []
        pg.on("pageerror", lambda e: pg.errors.append(str(e)))
        pg.on("console", lambda m: pg.errors.append(m.text) if m.type == "error" else None)
        pg.goto(f"{url}/?hands&nofs")
        yield pg
    finally:
        ctx.close()


# -- helpers -----------------------------------------------------------------------

# Page (CSS) pixels to a screen fraction, the way the page labels the setup's
# dots (screen-space.js): the server's pointing map works in screen fractions.
TO_SCREEN = """async ([x, y]) => {
    const { clientToScreen } = await import('/js/screen-space.js');
    const s = clientToScreen(x, y);
    return [s.x / window.screen.width, s.y / window.screen.height];
}"""

# The page position farthest from anything the cursor could be pulled to.
EMPTY_SPOT = """() => {
    const rects = [...document.querySelectorAll(window.paralic.gaze.selector)]
        .filter((el) => !el.closest('[inert], [hidden]'))
        .map((el) => el.getBoundingClientRect())
        .filter((r) => r.width >= 4 && r.height >= 4);
    const room = (x, y) => Math.min(...rects.map((r) =>
        Math.hypot(Math.max(r.left - x, 0, x - r.right), Math.max(r.top - y, 0, y - r.bottom))));
    let best = null;
    for (let y = 150; y <= innerHeight - 100; y += 25) {
        for (let x = 150; x <= innerWidth - 150; x += 25) {
            const d = room(x, y);
            if (!best || d > best[2]) best = [x, y, d];
        }
    }
    return best;
}"""

# The next pointing dot's position ("Xpx Ypx"), once it differs from `prev`.
NEXT_DOT = """(prev) => {
    const dot = document.querySelector('.calib-dot');
    const at = dot && !dot.classList.contains('done') ? dot.style.translate : '';
    return at && at !== prev ? at : null;
}"""

# Count what reaches the page: clicks (pinches), "stop" palms and popped practice targets.
COUNT_EVENTS = """() => {
    const n = window.handEvents = { double_blink: 0, hand_palm: 0, popped: 0 };
    window.paralic.tracker.on('double_blink', () => n.double_blink++);
    window.paralic.tracker.on('hand_palm', () => n.hand_palm++);
    document.addEventListener('click', (e) => { if (e.target.closest('.practice-target')) n.popped++; }, true);
}"""


def point_at(pg, pose: Pose, x: float, y: float) -> None:
    """Point the fingertip at page position (x, y)."""
    pose.point(*pg.evaluate(TO_SCREEN, [x, y]))


def centre(pg, selector: str) -> tuple[float, float]:
    box = pg.locator(selector).first.bounding_box()
    assert box, f"{selector} not visible"
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def aim(pg, pose: Pose, selector: str) -> None:
    """Point at an element until it lights up, and hold there a moment."""
    point_at(pg, pose, *centre(pg, selector))
    pg.wait_for_function("(s) => !!document.querySelector(s)?.classList.contains('gaze-hover')",
                         arg=selector, timeout=8000)
    pose.wait_frames(2)


def hold(pose: Pose, seconds: float) -> None:
    """Keep the hand as it is for ``seconds`` (and at least one camera frame)."""
    end = time.monotonic() + seconds
    pose.wait_frames(1)
    while time.monotonic() < end:
        pose.wait_frames(1)


def pinch(pose: Pose, touch: float = 0.15, pull: float = 0.012) -> None:
    """A quick pinch as a hand makes it: the closing fingers pull the fingertip
    towards the thumb and down (``pull`` image widths a frame, about 60 px on
    screen through the pointing map), touch for ``touch`` seconds and open.
    Timed in seconds, not frames, so it stays a quick click at any frame rate."""
    pose.play({"pinch": 0.6, "drift": (-pull, pull)}, {"pinch": CLOSED, "drift": (-2 * pull, 2 * pull)})
    hold(pose, touch)
    pose.play({"pinch": OPEN, "drift": (0.0, 0.0)}, {})


def pinch_drag(pose: Pose, dy: float, steps: int = 8) -> None:
    """Pinch, move the whole hand down by ``dy`` (image heights) and let go."""
    x, y = pose.get("tip")
    pose.play({"pinch": 0.45}, {"pinch": CLOSED},
              *[{"tip": (x, y + dy * k / steps)} for k in range(1, steps + 1)],
              {"pinch": OPEN}, {})


def setup_text(pg) -> str:
    return pg.evaluate("document.querySelector('.calib-text')?.textContent || ''")


def wait_text(pg, text: str, timeout: float = 30000) -> None:
    """Wait until the hand setup's instructions say ``text``."""
    pg.wait_for_function("(t) => (document.querySelector('.calib-text')?.textContent || '').includes(t)",
                         arg=text, timeout=timeout)


def cursor_at(pg):
    """Where the cursor is drawn (page pixels), or None while it is hidden."""
    t = pg.evaluate("(() => { const c = document.querySelector('#gaze-cursor');"
                    " return c.hidden ? null : c.style.transform; })()")
    m = re.match(r"translate3d\(([-\d.]+)px, ([-\d.]+)px", t or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


def poll(pg, read, ok, timeout_ms: float = 6000):
    """Read a value until ``ok(value)`` or the time is up; returns the last value."""
    for _ in range(int(timeout_ms / 100)):
        value = read()
        if ok(value):
            return value
        pg.wait_for_timeout(100)
    return read()


def scroll_top(pg) -> float:
    return pg.evaluate("document.querySelector('#page').scrollTop")


def hand_files(data_dir: Path) -> list[Path]:
    return sorted(Path(data_dir).glob("users/*/hand.json"))


# -- the person's part of the hand setup ---------------------------------------------------

def follow_dots(pg, pose: Pose) -> None:
    """The pointing step: point at each dot as it appears."""
    wait_text(pg, "Point at each dot")
    pose.set(fingers="curled")
    seen = ""
    for _ in range(GRID_DOTS):
        seen = pg.wait_for_function(NEXT_DOT, arg=seen, timeout=20000).json_value()
        x, y = ([float(v) for v in re.findall(r"-?[\d.]+", seen)] + [0.0])[:2]     # "Xpx" alone: y is 0
        point_at(pg, pose, x, y)


def pinch_four_times(pg, pose: Pose) -> None:
    """The pinch step: pinch until the page has counted four, then keep the
    hand open until it moves on."""
    pg.wait_for_selector(".calib-meter:not([hidden])", timeout=15000)       # counting from now
    for _ in range(200):
        text = setup_text(pg)
        if "Now pinch a few times" not in text:
            return
        counted = re.search(r"(\d+) of \d+", text)
        if counted is None or int(counted.group(1)) < 4:
            pinch(pose, touch=0.35)
            hold(pose, 0.4)
        else:
            pose.wait_frames(3)
    raise AssertionError(f"the pinch step did not end: {setup_text(pg)!r}")


def full_hand_setup(pg, pose: Pose) -> None:
    """The first hand setup, step by step, then the practice targets."""
    pg.evaluate(COUNT_EVENTS)
    wait_text(pg, "Hold up your open hand")
    pose.set(fingers="spread", pinch=OPEN)
    follow_dots(pg, pose)
    wait_text(pg, "Relax your open hand")
    pose.set(fingers="together", pinch=OPEN)
    wait_text(pg, "Now pinch a few times")
    pinch_four_times(pg, pose)
    wait_text(pg, "Hand setup complete")
    assert "pinch tuned to your hand" in setup_text(pg)
    pose.set(fingers="curled")
    for k in range(1, 4):
        wait_text(pg, f"Target {k} of 3")
        aim(pg, pose, ".practice-target")
        pinch(pose)
        pg.wait_for_function("(k) => window.handEvents.popped >= k", arg=k, timeout=5000)
    pg.wait_for_selector(".overlay.calib", state="detached", timeout=15000)


def raise_hand(pg, pose: Pose) -> None:
    """The camera check looks for a hand until one comes into view."""
    pg.wait_for_selector("text=Raise one hand into view, fingers spread", timeout=30000)
    pose.set(visible=True, fingers="spread", pinch=OPEN)
    pg.wait_for_selector("text=Hand found", timeout=10000)


# -- tests ------------------------------------------------------------------------------

def test_first_visit_sets_up_the_hand_then_browses_by_hand(browser, site):
    pose = site.pose
    pose.reset()
    with visit(browser, site.url) as pg:
        raise_hand(pg, pose)
        full_hand_setup(pg, pose)

        # The setup is saved in this person's folder and in use.
        files = hand_files(site.data_dir)
        person = pg.evaluate("window.paralic.state.person.id")
        assert [f.parent.name for f in files] == [person]
        doc = json.loads(files[0].read_text())
        summary, cal = doc["summary"], doc["calibration"]
        assert summary["mode"] == "full" and summary["points"] == GRID_DOTS and summary["pinch_tuned"]
        assert summary["pointing_error_px"] < 15
        assert cal["pointing"]["quadratic"] and CLOSED < cal["pinch_on"] < cal["pinch_off"] < 1.5
        hand = pg.evaluate("window.paralic.state.hand")
        assert hand and hand["calibrated"] and hand["points"] == GRID_DOTS
        assert pg.evaluate("window.paralic.state.calibrated && document.body.classList.contains('hand-mode')")

        # The cursor goes where the fingertip points.
        x, y, room = pg.evaluate(EMPTY_SPOT)
        assert room > 120                          # nothing near enough to pull the cursor
        point_at(pg, pose, x, y)
        at = poll(pg, lambda: cursor_at(pg), lambda p: p is not None and math.dist(p, (x, y)) < 60)
        assert at is not None and math.dist(at, (x, y)) < 60, (at, (x, y))

        # Point at a button and pinch: it is clicked.
        aim(pg, pose, 'a.nav-btn[data-route="explore"]')
        pinch(pose)
        pg.wait_for_function("location.hash === '#/explore'", timeout=5000)

        # Pinch and move the hand down: the page scrolls (and nothing is clicked).
        pg.evaluate("location.hash = '#/read/how-it-works'")
        pg.wait_for_selector(".article h1")
        point_at(pg, pose, 800, 500)
        pose.wait_frames(4)
        before = scroll_top(pg)
        pinch_drag(pose, dy=0.16)
        after = poll(pg, lambda: scroll_top(pg), lambda v: v > before + 300)
        assert after > before + 300, (before, after)
        assert pg.evaluate("location.hash") == "#/read/how-it-works"

        # The open "stop" hand pauses...
        palms = pg.evaluate("window.handEvents.hand_palm")
        pose.set(fingers="spread")
        pg.wait_for_selector("#paused-banner", state="visible", timeout=6000)
        assert pg.evaluate("document.body.classList.contains('is-paused')")
        # ...a pinch (here on a button) does not resume or click...
        pose.set(fingers="curled")
        point_at(pg, pose, *centre(pg, 'a.nav-btn[data-route="explore"]'))
        pose.wait_frames(4)
        pg.wait_for_timeout(600)                   # out of the stop pose long enough to toggle again
        clicks = pg.evaluate("window.handEvents.double_blink")
        pinch(pose)
        pg.wait_for_function("(n) => window.handEvents.double_blink > n", arg=clicks, timeout=5000)
        pg.wait_for_timeout(300)                   # time to act on it, if it would
        assert pg.locator("#paused-banner").is_visible()
        assert pg.evaluate("location.hash") == "#/read/how-it-works"
        # ...and the open hand again resumes.
        pose.set(fingers="spread")
        pg.wait_for_selector("#paused-banner", state="hidden", timeout=6000)
        assert pg.evaluate("window.handEvents.hand_palm") == palms + 2     # once per gesture
        pose.set(fingers="curled")
        aim(pg, pose, 'a.nav-btn[data-route="explore"]')
        pinch(pose)
        pg.wait_for_function("location.hash === '#/explore'", timeout=5000)
        assert pg.errors == []


def test_second_visit_welcomes_back_and_re_points(browser, site):
    pose = site.pose
    if not hand_files(site.data_dir):
        # Run on its own: make the first setup.
        pose.reset()
        with visit(browser, site.url) as pg:
            raise_hand(pg, pose)
            full_hand_setup(pg, pose)
    before = json.loads(hand_files(site.data_dir)[0].read_text())

    pose.reset()
    pose.set(visible=True, fingers="spread")
    with visit(browser, site.url) as pg:
        # The saved setup is loaded: point at a choice and pinch.
        pg.wait_for_selector(".choice-row", timeout=30000)
        labels = pg.locator(".choice-row .choice > span").all_text_contents()
        assert labels == ["Start browsing", "Quick re-point", "Full hand setup", "Switch person"]
        pose.set(fingers="curled")
        aim(pg, pose, '.choice[data-choice="point"]')
        pinch(pose)
        pg.wait_for_selector(".choice-row", state="detached", timeout=5000)
        # Sitting a little differently today: the comfortable range has moved.
        pose.reach = (0.40, 0.33)
        follow_dots(pg, pose)
        wait_text(pg, "Hand setup complete")
        pg.wait_for_selector(".overlay.calib", state="detached", timeout=15000)

        # Only the pointing was redone: the pinch is still this person's.
        after = json.loads(hand_files(site.data_dir)[0].read_text())
        assert after["summary"]["mode"] == "point" and after["summary"]["pinch_tuned"]
        for key in ("pinch_on", "pinch_off", "span"):
            assert after["calibration"][key] == before["calibration"][key], key
        assert after["calibration"]["pointing"] != before["calibration"]["pointing"]
        # The new pointing is in use.
        x, y, _ = pg.evaluate(EMPTY_SPOT)
        point_at(pg, pose, x, y)
        at = poll(pg, lambda: cursor_at(pg), lambda p: p is not None and math.dist(p, (x, y)) < 60)
        assert at is not None and math.dist(at, (x, y)) < 60, (at, (x, y))
        assert pg.errors == []


def test_hand_mode_waits_for_the_hand_model(browser, tmp_path):
    from paralic.server import create_app

    pose = Pose()
    pose.set(visible=True, fingers="spread")
    downloaded = threading.Event()

    def loader():
        downloaded.wait(30)                        # the hand model "downloading"
        return lambda: ScriptedHandTracker(pose)

    app = create_app(data_dir=tmp_path, tracker_factory=None, hand_loader=loader)
    server, thread, url = _serve(app)
    try:
        with visit(browser, url) as pg:
            pg.wait_for_selector(".error-box >> text=Getting hand control ready", timeout=15000)
            assert pg.evaluate("fetch('/api/status').then((r) => r.json()).then((s) => s.hands)") == "loading"
            assert pose.frames == 0                # the camera waits for the model
            downloaded.set()
            pg.wait_for_selector("text=Hand found", timeout=20000)
            wait_text(pg, "Hold up your open hand")
            assert pg.locator("text=Hand tracking stopped").count() == 0
            assert pg.errors == []
    finally:
        downloaded.set()
        server.should_exit = True
        thread.join(timeout=5)


def test_a_dropped_connection_cannot_leave_the_hand_setup_stuck(browser, tmp_path, monkeypatch):
    from paralic.server import create_app
    from paralic.session import TrackerSession

    asked, answer = threading.Event(), threading.Event()
    start = TrackerSession._cmd_hand_calibration_start

    def stalled_start(self, cmd):                  # the server stops answering (it is restarting, say)
        asked.set()
        answer.wait(10)
        return start(self, cmd)

    monkeypatch.setattr(TrackerSession, "_cmd_hand_calibration_start", stalled_start)
    pose = Pose()
    pose.set(visible=True, fingers="spread")
    app = create_app(data_dir=tmp_path, tracker_factory=None,
                     hand_tracker_factory=lambda: ScriptedHandTracker(pose))
    server, thread, url = _serve(app)
    try:
        with visit(browser, url) as pg:
            assert asked.wait(30)                  # the first hand setup is starting...
            pg.evaluate("window.paralic.tracker.ws.close()")         # ...when the connection drops
            answer.set()
            # The setup stops with a message, and the hand still moves the cursor:
            # without a setup, the (mirrored) fingertip itself.
            pg.wait_for_selector(".toast >> text=The hand setup stopped", timeout=10000)
            pg.wait_for_function("window.paralic.state.calibrated && window.paralic.tracker.connected",
                                 timeout=10000)
            pose.set(fingers="curled")
            x, y, _ = pg.evaluate(EMPTY_SPOT)
            fx, fy = pg.evaluate(TO_SCREEN, [x, y])
            pose.set(tip=(1.0 - fx, fy))
            at = poll(pg, lambda: cursor_at(pg), lambda p: p is not None and math.dist(p, (x, y)) < 60)
            assert at is not None and math.dist(at, (x, y)) < 60, (at, (x, y))
            assert pg.errors == []
    finally:
        answer.set()
        server.should_exit = True
        thread.join(timeout=5)
