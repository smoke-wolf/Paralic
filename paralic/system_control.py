"""System-wide gaze control: the decision logic + kill switches.

The gaze controls the real desktop cursor, so the dangerous part is *stopping*
it. This module keeps that logic pure and testable (it calls an injected
``OSController`` for the actual movement, which tests replace with a fake):

* off by default; only an explicit enable turns it on, and only if Accessibility
  is granted (otherwise it prompts and stays off);
* auto-disables if no face is seen for ``no_face_timeout_s`` (you walked away);
* auto-disables if the cursor dwells in the top-left corner for
  ``corner_dwell_s`` — a reachable emergency stop needing no keyboard;
* gaze in the top / bottom screen band scrolls (rate-limited).

Coordinates are in global display points == browser screen CSS px on a single
display (see OSController.screen_bounds / session.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class ControlConfig:
    no_face_timeout_s: float = 2.0
    corner_px: float = 60.0
    corner_dwell_s: float = 1.2
    edge_margin_frac: float = 0.06
    scroll_lines: float = 2.0
    scroll_interval_s: float = 0.12
    edge_scroll: bool = True          # hand mode scrolls with a pinch-drag instead
    lost_reason: str = "No face detected - desktop control off."
    scroll_px_per_line: float = 40.0


def hand_config() -> ControlConfig:
    """Hand mode: lowering the hand is normal (nothing moves or clicks without
    it), so only a long absence turns desktop control off; pinch-drag scrolls."""
    return ControlConfig(no_face_timeout_s=120.0, edge_scroll=False,
                         lost_reason="No hand seen for two minutes - desktop control off.")


@dataclass
class ControlResult:
    moved: Optional[tuple] = None
    clicked: bool = False
    scrolled: Optional[float] = None
    disabled_reason: Optional[str] = None


class SystemController:
    def __init__(self, os_controller, config: Optional[ControlConfig] = None) -> None:
        self.os = os_controller
        self.cfg = config or ControlConfig()
        self.enabled = False
        self._last_face_t: Optional[float] = None
        self._corner_since: Optional[float] = None
        self._last_scroll_t = 0.0

    def state(self) -> dict:
        avail = self.os.available
        trusted = self.os.is_trusted() if avail else False
        reason = "" if (avail and trusted) else self.os.status().reason
        return {"type": "system_control", "enabled": self.enabled,
                "available": avail, "trusted": trusted, "reason": reason}

    def enable(self) -> dict:
        st = self.os.status()
        if not st.available:
            self.enabled = False
            return {"type": "system_control", "enabled": False, "available": False,
                    "trusted": False, "reason": st.reason}
        if not st.trusted:
            self.os.request_trust()           # pop the Accessibility prompt if possible
            st = self.os.status()
            if not st.trusted:
                self.enabled = False
                return {"type": "system_control", "enabled": False, "available": True,
                        "trusted": False, "reason": st.reason, "needs_permission": True}
        self.enabled = True
        self._last_face_t = None
        self._corner_since = None
        return {"type": "system_control", "enabled": True, "available": True, "trusted": True, "reason": ""}

    def disable(self, reason: str = "") -> dict:
        self.enabled = False
        self._corner_since = None
        return {"type": "system_control", "enabled": False, "available": self.os.available,
                "trusted": self.os.is_trusted() if self.os.available else False, "reason": reason}

    def update(self, t: float, gaze: Optional[Sequence[float]], face: bool) -> ControlResult:
        """One frame. ``gaze`` is (x, y) in screen points, or None."""
        res = ControlResult()
        if not self.enabled:
            return res

        # Kill switch 1: no face for too long -> you left; stop.
        if face:
            self._last_face_t = t
        elif self._last_face_t is not None and (t - self._last_face_t) > self.cfg.no_face_timeout_s:
            self.disable(self.cfg.lost_reason)
            res.disabled_reason = "no-face"
            return res

        if not face or gaze is None:
            return res

        x, y = float(gaze[0]), float(gaze[1])
        bx, by, bw, bh = self.os.screen_bounds() or (0.0, 0.0, 0.0, 0.0)

        # Kill switch 2: dwell in the top-left corner.
        if (x - bx) <= self.cfg.corner_px and (y - by) <= self.cfg.corner_px:
            if self._corner_since is None:
                self._corner_since = t
            elif (t - self._corner_since) >= self.cfg.corner_dwell_s:
                self.disable("Desktop control off (held the top-left corner).")
                res.disabled_reason = "corner"
                return res
        else:
            self._corner_since = None

        self.os.move(x, y)
        res.moved = (x, y)

        # Edge scroll (needs a known screen height).
        if self.cfg.edge_scroll and bh > 0 and (t - self._last_scroll_t) >= self.cfg.scroll_interval_s:
            band = self.cfg.edge_margin_frac * bh
            if (y - by) <= band:
                self.os.scroll(self.cfg.scroll_lines)
                res.scrolled = self.cfg.scroll_lines
                self._last_scroll_t = t
            elif (by + bh - y) <= band:
                self.os.scroll(-self.cfg.scroll_lines)
                res.scrolled = -self.cfg.scroll_lines
                self._last_scroll_t = t
        return res

    def scroll_by(self, dy_px: float) -> float:
        """Pinch-drag scroll (``dy_px`` > 0 scrolls down); whole lines, the rest carried over."""
        if not self.enabled:
            return 0.0
        self._scroll_rest = getattr(self, "_scroll_rest", 0.0) + float(dy_px) / self.cfg.scroll_px_per_line
        lines = int(self._scroll_rest)
        if lines:
            self._scroll_rest -= lines
            self.os.scroll(-lines)
        return float(lines)

    def click(self, gaze: Optional[Sequence[float]]) -> bool:
        if self.enabled and gaze is not None:
            self.os.click(float(gaze[0]), float(gaze[1]))
            return True
        return False
