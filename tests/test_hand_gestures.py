"""Hand-mode gesture state machine (paralic/hand_gestures.py)."""

import numpy as np

from paralic.hand_gestures import HandGestureRecognizer, HandGestureConfig

# Landmark indices used by the recogniser.
WRIST, THUMB_TIP = 0, 4
I_MCP, I_PIP, I_TIP = 5, 6, 8
M_MCP, M_PIP, M_TIP = 9, 10, 12
R_PIP, R_TIP = 14, 16
P_MCP, P_PIP, P_TIP = 17, 18, 20


def make_hand(index_tip=(0.5, 0.3), pinch=False, fingers_up=True):
    """A plausible 21-point normalised hand. Fingers point up (smaller y)."""
    p = np.zeros((21, 2))
    p[WRIST] = (0.5, 0.9)
    p[I_MCP] = (0.42, 0.60)
    p[M_MCP] = (0.50, 0.60)
    p[P_MCP] = (0.58, 0.60)          # knuckle span ~0.16 → palm width
    tip_y, pip_y = (0.30, 0.52) if fingers_up else (0.78, 0.62)   # extended vs curled
    p[I_TIP] = index_tip
    p[I_PIP] = (index_tip[0], 0.52 if fingers_up else 0.72)
    p[M_TIP] = (0.50, tip_y); p[M_PIP] = (0.50, pip_y)
    p[R_TIP] = (0.55, tip_y); p[R_PIP] = (0.55, pip_y)
    p[P_TIP] = (0.60, tip_y); p[P_PIP] = (0.60, pip_y)
    # Thumb: close to the index tip for a pinch, far otherwise.
    p[THUMB_TIP] = (index_tip[0] + 0.02, index_tip[1] + 0.01) if pinch else (0.28, 0.60)
    return p


def test_cursor_is_index_tip_mirrored():
    rec = HandGestureRecognizer()
    state, _ = rec.update(0.0, make_hand(index_tip=(0.3, 0.4)))
    assert state.present
    assert abs(state.cursor[0] - 0.7) < 1e-6     # x mirrored for the front camera
    assert abs(state.cursor[1] - 0.4) < 1e-6


def test_quick_pinch_is_a_click():
    rec = HandGestureRecognizer()
    rec.update(0.0, make_hand(pinch=False))
    _, e1 = rec.update(0.05, make_hand(pinch=True))      # pinch down
    assert all(ev.type != "click" for ev in e1)
    _, e2 = rec.update(0.10, make_hand(pinch=False))     # release, still, < click_max_ms
    assert any(ev.type == "click" for ev in e2)


def test_pinch_drag_scrolls():
    rec = HandGestureRecognizer()
    rec.update(0.0, make_hand(index_tip=(0.5, 0.40), pinch=True))
    _, ev = rec.update(0.12, make_hand(index_tip=(0.5, 0.55), pinch=True))  # moved down > scroll_start
    scrolls = [e for e in ev if e.type == "scroll"]
    assert scrolls and scrolls[-1].dy > 0               # downward hand → scroll down
    # A scroll must not also be reported as a click on release.
    _, ev2 = rec.update(0.20, make_hand(index_tip=(0.5, 0.55), pinch=False))
    assert all(e.type != "click" for e in ev2)


def test_open_palm_held_toggles_pause():
    rec = HandGestureRecognizer()
    _, e0 = rec.update(0.0, make_hand(fingers_up=True))
    assert not rec.paused and all(ev.type != "pause_toggle" for ev in e0)
    _, e1 = rec.update(0.5, make_hand(fingers_up=True))  # held > palm_hold_ms
    assert any(ev.type == "pause_toggle" for ev in e1)
    assert rec.paused


def test_hand_lost_is_safe():
    rec = HandGestureRecognizer()
    rec.update(0.0, make_hand(pinch=True))
    state, ev = rec.update(0.1, None)                    # hand disappears mid-pinch
    assert not state.present
    assert all(e.type != "click" for e in ev)           # no accidental click
