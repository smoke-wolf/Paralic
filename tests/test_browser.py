"""Browser tests (Playwright + Chromium).

The fast tests use the website's mouse demo mode, where the mouse stands in
for your gaze and the B key for a blink, to exercise the eye-control UI:
hovering, double-blink clicks, gaze scrolling, pausing, eye typing...

``test_camera_flow`` (``--runslow``) runs the real pipeline: Chromium's fake
webcam plays a synthetic face video (with a double blink in it) that the
Python server analyses with MediaPipe, through calibration to browsing.

Skipped automatically when Playwright or a Chromium build is not available.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

import pytest

playwright_api = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parent.parent
CHROMIUM_CANDIDATES = [os.environ.get("CHROMIUM_PATH"), "/opt/pw-browsers/chromium"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app):
    import uvicorn

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    return server, thread, f"http://localhost:{port}"


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        last_error = None
        for path in [None, *[c for c in CHROMIUM_CANDIDATES if c]]:
            try:
                b = p.chromium.launch(executable_path=path, args=[
                    "--no-sandbox", "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                    "--autoplay-policy=no-user-gesture-required",
                    *([f"--use-file-for-fake-video-capture={os.environ['PARALIC_FAKE_VIDEO']}"]
                      if os.environ.get("PARALIC_FAKE_VIDEO") else []),
                ])
                break
            except Exception as exc:  # pragma: no cover - depends on the machine
                last_error = exc
        else:
            pytest.skip(f"Chromium not available: {last_error}")
        yield b
        b.close()


@pytest.fixture(scope="module")
def demo_server(tmp_path_factory):
    from paralic.server import create_app

    app = create_app(data_dir=tmp_path_factory.mktemp("data"), tracker_factory=None, model_error=None)
    server, thread, url = _serve(app)
    yield url
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def page(browser, demo_server):
    ctx = browser.new_context(viewport={"width": 1600, "height": 900})
    pg = ctx.new_page()
    pg.errors = []
    pg.on("pageerror", lambda e: pg.errors.append(str(e)))
    pg.on("console", lambda m: pg.errors.append(m.text) if m.type == "error" else None)
    pg.goto(f"{demo_server}/?demo")
    pg.wait_for_function("window.paralic && window.paralic.state.started")
    yield pg
    assert pg.errors == []
    ctx.close()


# -- helpers -----------------------------------------------------------------------

def look_at(pg, selector, dx=0.0, dy=0.0):
    """Move the "gaze" (mouse) to an element, offset by a fraction of its size."""
    pg.locator(selector).first.scroll_into_view_if_needed()
    box = pg.locator(selector).first.bounding_box()
    assert box, f"{selector} not visible"
    pg.mouse.move(box["x"] + box["width"] * (0.5 + dx), box["y"] + box["height"] * (0.5 + dy), steps=6)
    pg.wait_for_timeout(350)


def double_blink(pg):
    pg.keyboard.press("b")
    pg.wait_for_timeout(300)
    pg.keyboard.press("b")
    pg.wait_for_timeout(500)


def hovered_text(pg):
    return pg.evaluate("document.querySelector('.gaze-hover')?.textContent?.trim() || null")


# -- tests ----------------------------------------------------------------------------

def test_look_and_double_blink_navigates(page):
    look_at(page, 'a.nav-btn[data-route="explore"]')
    assert hovered_text(page) == "Explore"
    double_blink(page)
    assert page.evaluate("location.hash") == "#/explore"
    look_at(page, 'a[href="#/explore/mars"]')
    double_blink(page)
    assert page.evaluate("location.hash") == "#/explore/mars"
    assert "Mars" in page.locator("h1").first.text_content()
    look_at(page, '[data-action="back"]')
    double_blink(page)
    assert page.evaluate("location.hash") == "#/explore"


def test_single_blink_does_not_click(page):
    look_at(page, 'a.nav-btn[data-route="read"]')
    page.keyboard.press("b")
    page.wait_for_timeout(1200)
    assert page.evaluate("location.hash") in ("", "#/home")


def test_snapping_selects_nearby_button(page):
    # Look just outside the Talk button: the nearest button still lights up.
    box = page.locator('a.nav-btn[data-route="talk"]').bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] + 25, steps=6)
    page.wait_for_timeout(350)
    assert hovered_text(page) == "Talk"


def test_gaze_scrolling_with_the_rail(page):
    page.evaluate("location.hash = '#/read/how-it-works'")
    page.wait_for_timeout(400)
    before = page.evaluate("document.querySelector('#page').scrollTop")
    look_at(page, '[data-scroll="down"]', dy=0.3)
    page.wait_for_timeout(1500)
    after = page.evaluate("document.querySelector('#page').scrollTop")
    assert after > before + 200
    look_at(page, ".article p")  # looking away stops scrolling
    stopped = page.evaluate("document.querySelector('#page').scrollTop")
    page.wait_for_timeout(600)
    assert page.evaluate("document.querySelector('#page').scrollTop") == stopped


def test_pause_and_resume_with_double_blinks(page):
    look_at(page, "#pause-btn")
    double_blink(page)
    assert page.evaluate("document.body.classList.contains('is-paused')")
    assert page.locator("#paused-banner").is_visible()
    # While paused, looking at a link and double blinking resumes instead of clicking.
    look_at(page, 'a.nav-btn[data-route="read"]')
    double_blink(page)
    assert not page.evaluate("document.body.classList.contains('is-paused')")
    assert page.evaluate("location.hash") in ("", "#/home")


def test_eye_typing_with_predictions(page):
    page.evaluate("location.hash = '#/talk'")
    page.wait_for_timeout(300)
    look_at(page, ".tabs .btn:nth-child(2)")  # Keyboard tab
    double_blink(page)
    for letter in "HE":
        look_at(page, f'.key[aria-label="{letter}"]')
        double_blink(page)
    assert page.locator(".text-display").text_content().startswith("He")
    suggestions = page.locator(".suggestion").all_text_contents()
    assert "Hello" in suggestions or "Help" in suggestions
    look_at(page, '.suggestion:has-text("Help")')
    double_blink(page)
    assert page.locator(".text-display").text_content().startswith("Help ")


def test_practice_target_can_be_popped(page):
    page.evaluate("location.hash = '#/practice'")
    page.wait_for_timeout(300)
    look_at(page, ".arena .btn.primary")
    double_blink(page)
    page.wait_for_selector(".target", timeout=3000)
    look_at(page, ".target")
    double_blink(page)
    page.wait_for_timeout(300)
    assert "Hits 1" in page.locator(".hud .stats").text_content()


def test_settings_change_by_gaze(page):
    page.evaluate("location.hash = '#/settings'")
    page.wait_for_timeout(300)
    look_at(page, '.setting:has-text("Double-blink speed") .opt[data-value="relaxed"]')
    double_blink(page)
    stored = page.evaluate("JSON.parse(localStorage.getItem('paralic.settings.v1')).doubleBlink")
    assert stored == "relaxed"
    assert page.locator('.setting:has-text("Double-blink speed") .opt.selected').text_content() == "Relaxed"


# -- eye gestures (demo mode: hold Q / E = keep the left / right eye closed) ------------------

def center_of(pg, selector):
    box = pg.locator(selector).first.bounding_box()
    assert box, f"{selector} not visible"
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def test_hold_one_eye_closed_to_drag(page):
    page.evaluate("location.hash = '#/arrange'")
    page.wait_for_selector(".arrange-tray .arrange-card")
    look_at(page, '.arrange-tray .arrange-card[data-planet="mercury"]')
    page.keyboard.down("q")
    page.wait_for_timeout(600)                     # held long enough: the press starts
    assert page.evaluate("document.body.classList.contains('gaze-holding')")
    x, y = center_of(page, '.arrange-slot[data-slot="0"]')
    page.mouse.move(x, y, steps=12)                # look at the first place while holding
    page.wait_for_timeout(400)
    assert page.evaluate("document.body.classList.contains('gaze-dragging')")
    assert page.locator(".drag-ghost").count() == 1
    page.keyboard.up("q")                          # open the eye: drop
    page.wait_for_timeout(400)
    assert page.locator('.arrange-slot[data-slot="0"] .arrange-card[data-planet="mercury"]').count() == 1
    assert page.locator(".drag-ghost").count() == 0
    assert "1 of 8" in page.locator(".arrange-status").text_content()


def test_short_hold_is_a_click_and_quick_wink_does_nothing_by_default(page):
    look_at(page, 'a.nav-btn[data-route="read"]')
    page.keyboard.down("e")
    page.wait_for_timeout(200)                     # shorter than the hold time: a quick wink
    page.keyboard.up("e")
    page.wait_for_timeout(400)
    assert page.evaluate("location.hash") in ("", "#/home")
    page.keyboard.down("e")
    page.wait_for_timeout(550)                     # a held wink, released without moving: click
    page.keyboard.up("e")
    page.wait_for_timeout(400)
    assert page.evaluate("location.hash") == "#/read"


def test_holding_still_opens_the_menu(page):
    look_at(page, 'a.nav-btn[data-route="talk"]')
    page.keyboard.down("q")
    page.wait_for_timeout(1700)                    # hold time + long press
    page.keyboard.up("q")
    page.wait_for_selector(".gaze-menu")
    items = page.locator(".gaze-menu-item").all_text_contents()
    assert items[0] == "Click" and "Read aloud" in items and items[-1] == "Cancel"
    look_at(page, '.gaze-menu-item[data-menu="click"]')
    double_blink(page)
    assert page.evaluate("location.hash") == "#/talk"
    assert page.locator(".gaze-menu").count() == 0


def test_draw_with_a_held_wink(page):
    page.evaluate("location.hash = '#/draw'")
    page.wait_for_selector(".draw-canvas")
    x, y = center_of(page, ".draw-canvas")
    page.mouse.move(x - 200, y, steps=4)
    page.wait_for_timeout(200)
    page.keyboard.down("e")
    page.wait_for_timeout(600)
    page.mouse.move(x + 200, y + 60, steps=15)
    page.wait_for_timeout(200)
    page.keyboard.up("e")
    page.wait_for_timeout(200)
    assert page.evaluate("document.querySelector('.draw-canvas').dataset.strokes") == "1"
    # Pixels were painted along the path.
    painted = page.evaluate("""() => {
        const c = document.querySelector('.draw-canvas');
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++;
        return n; }""")
    assert painted > 2000


def test_dwell_click_when_enabled(page):
    page.evaluate("""window.paralic.tracker.request({type: 'gestures_set', gestures: {dwell: true, dwell_ms: 700}},
                                                  'personal')""")
    page.wait_for_timeout(300)
    look_at(page, 'a.nav-btn[data-route="explore"]')
    page.wait_for_timeout(1100)
    assert page.evaluate("location.hash") == "#/explore"
    page.evaluate("window.paralic.tracker.request({type: 'gestures_set', gestures: {dwell: false}}, 'personal')")
    page.wait_for_timeout(300)


def test_long_close_menu_and_drag_lock(page):
    page.evaluate("window.paralic.tracker.request({type: 'gestures_set', gestures: {long_close: 'menu'}}, 'personal')")
    page.evaluate("location.hash = '#/arrange'")
    page.wait_for_selector(".arrange-tray .arrange-card")
    look_at(page, '.arrange-tray .arrange-card[data-planet="venus"]')
    page.keyboard.down("b")                        # close both eyes for 1.2 s
    page.wait_for_timeout(1200)
    page.keyboard.up("b")
    page.wait_for_selector(".gaze-menu")
    look_at(page, '.gaze-menu-item[data-menu="move"]')
    double_blink(page)                             # "Pick up to move"
    page.wait_for_timeout(300)
    assert page.locator(".drag-ghost").count() == 1
    look_at(page, '.arrange-slot[data-slot="1"]')
    double_blink(page)                             # drop it here
    assert page.locator('.arrange-slot[data-slot="1"] .arrange-card[data-planet="venus"]').count() == 1
    page.evaluate("window.paralic.tracker.request({type: 'gestures_set', gestures: {long_close: 'off'}}, 'personal')")
    page.wait_for_timeout(300)


def test_lab_shows_personalisation_and_runs_an_experiment(page):
    page.evaluate("location.hash = '#/lab'")
    page.wait_for_selector(".lab-grid")
    assert "Tuned for" in page.locator("h1").first.text_content()
    look_at(page, '.lab-experiment[href="#/lab/run/magnet"]')
    double_blink(page)
    page.wait_for_selector(".lab-arena .btn.primary")
    look_at(page, ".lab-arena .btn.primary")
    double_blink(page)
    arms = []
    for _ in range(12):                            # 3 variants x 4 targets
        page.wait_for_selector(".lab-arena .target", timeout=4000)
        arms.append(page.evaluate("JSON.stringify(window.paralic.gaze.magnetOverride)"))
        look_at(page, ".lab-arena .target")
        double_blink(page)
    page.wait_for_selector(".lab-table", timeout=8000)
    assert len(set(arms)) == 3                     # every variant was used, blind
    assert page.locator(".lab-table tbody tr").count() == 3
    assert page.evaluate("window.paralic.gaze.magnetOverride") is None


@pytest.mark.slow
def test_camera_flow(browser, tmp_path):
    """Full pipeline: fake webcam -> MediaPipe -> blink to start -> calibration -> browsing."""
    if not os.environ.get("PARALIC_FAKE_VIDEO"):
        pytest.skip("set PARALIC_FAKE_VIDEO to a .y4m face video (see tests/make_fake_video.py)")
    model = ROOT / "models" / "face_landmarker.task"
    if not model.exists():
        pytest.skip("face model not downloaded")
    from paralic.server import create_app
    from paralic.tracker import FaceTracker

    model_bytes = model.read_bytes()
    app = create_app(data_dir=tmp_path, tracker_factory=lambda: FaceTracker(model_bytes))
    server, thread, url = _serve(app)
    ctx = browser.new_context(viewport={"width": 1600, "height": 900})
    try:
        ctx.grant_permissions(["camera"], origin=url)
        # The synthetic "closed eyes" are less closed than a real blink.
        ctx.add_init_script("localStorage.setItem('paralic.settings.v1', JSON.stringify({blinkSensitivity: 'high'}))")
        pg = ctx.new_page()
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(f"{url}/?nofs")
        pg.wait_for_timeout(1500)
        try:  # the camera starts by itself when permission was already granted
            pg.click("#start-btn", timeout=1000)
        except Exception:
            pass
        pg.wait_for_selector("text=Blink twice to calibrate", timeout=30000)
        pg.wait_for_selector(".calib-dot", timeout=20000)          # started by the video's double blink
        pg.wait_for_selector(".results", timeout=120000)
        # The results screen must be usable with the eyes: gaze cursor on and live.
        pg.wait_for_timeout(800)
        assert pg.evaluate("window.paralic.gaze.active && !window.paralic.gaze.suspended")
        assert pg.evaluate("!document.querySelector('#gaze-cursor').hidden")
        pg.click('[data-choice="go"]')
        pg.wait_for_timeout(1500)
        assert pg.evaluate("!document.querySelector('#gaze-cursor').hidden")
        assert pg.evaluate("window.paralic.state.calibrated")
        assert errors == []
    finally:
        ctx.close()
        server.should_exit = True
        thread.join(timeout=5)
