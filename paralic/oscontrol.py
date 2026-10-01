"""Drive the real macOS cursor from gaze (system-wide "control my whole computer").

A browser tab cannot move the OS pointer or click outside itself, so the local
Python server does it with CoreGraphics via ``ctypes`` (no extra dependencies).
Everything degrades to a safe no-op off macOS or if a framework / symbol is
missing, and nothing here moves the cursor unless :meth:`OSController.move` /
``click`` / ``scroll`` are actually called — which the session only does while
the user has explicitly turned system control on *and* Accessibility is granted.

macOS gates synthetic input behind the **Accessibility** permission (TCC): the
process must be in System Settings -> Privacy & Security -> Accessibility.
:meth:`is_trusted` reports it; :meth:`request_trust` pops the system prompt.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from dataclasses import dataclass


class CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class CGSize(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class CGRect(ctypes.Structure):
    _fields_ = [("origin", CGPoint), ("size", CGSize)]


# CoreGraphics event constants.
_kCGEventMouseMoved = 5
_kCGEventLeftMouseDown = 1
_kCGEventLeftMouseUp = 2
_kCGMouseButtonLeft = 0
_kCGHIDEventTap = 0
_kCGScrollEventUnitLine = 1


@dataclass
class OSStatus:
    available: bool          # the CoreGraphics bindings loaded (i.e. macOS)
    trusted: bool            # Accessibility permission granted
    reason: str = ""         # why unavailable / not trusted (for the UI)


class OSController:
    """Thin, defensive wrapper over CoreGraphics mouse control.

    Construction never raises: if anything is unavailable the controller simply
    reports ``available is False`` and every action is a no-op.
    """

    def __init__(self) -> None:
        self._cg = None
        self._cf = None
        self._reason = ""
        self._bounds: tuple[float, float, float, float] | None = None
        if sys.platform != "darwin":
            self._reason = "System control is macOS-only."
            return
        try:
            self._load()
        except Exception as exc:  # pragma: no cover - platform/linker specific
            self._cg = None
            self._reason = f"Could not load CoreGraphics: {exc}"

    # -- binding ------------------------------------------------------------
    def _load(self) -> None:
        cg_path = ctypes.util.find_library("CoreGraphics") or ctypes.util.find_library("ApplicationServices")
        app_path = ctypes.util.find_library("ApplicationServices")
        cf_path = ctypes.util.find_library("CoreFoundation")
        if not cg_path or not cf_path:
            raise OSError("CoreGraphics / CoreFoundation not found")
        cg = ctypes.cdll.LoadLibrary(cg_path)
        app = ctypes.cdll.LoadLibrary(app_path) if app_path else cg
        cf = ctypes.cdll.LoadLibrary(cf_path)

        cg.CGEventCreateMouseEvent.restype = ctypes.c_void_p
        cg.CGEventCreateMouseEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint32, CGPoint, ctypes.c_uint32]
        cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        cg.CGWarpMouseCursorPosition.restype = ctypes.c_int32
        cg.CGWarpMouseCursorPosition.argtypes = [CGPoint]
        cg.CGEventCreateScrollWheelEvent.restype = ctypes.c_void_p
        cg.CGEventCreateScrollWheelEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int32]
        cg.CGMainDisplayID.restype = ctypes.c_uint32
        cg.CGDisplayBounds.restype = CGRect
        cg.CGDisplayBounds.argtypes = [ctypes.c_uint32]
        cf.CFRelease.argtypes = [ctypes.c_void_p]

        # Accessibility trust check lives in the ApplicationServices umbrella.
        self._ax_trusted = getattr(app, "AXIsProcessTrusted", None) or getattr(cg, "AXIsProcessTrusted", None)
        if self._ax_trusted is not None:
            self._ax_trusted.restype = ctypes.c_bool
        self._ax_trusted_opts = getattr(app, "AXIsProcessTrustedWithOptions", None)
        if self._ax_trusted_opts is not None:
            self._ax_trusted_opts.restype = ctypes.c_bool
            self._ax_trusted_opts.argtypes = [ctypes.c_void_p]

        self._cg, self._app, self._cf = cg, app, cf
        try:
            b = cg.CGDisplayBounds(cg.CGMainDisplayID())
            self._bounds = (b.origin.x, b.origin.y, b.size.width, b.size.height)
        except Exception:
            self._bounds = None

    # -- status -------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._cg is not None

    def is_trusted(self) -> bool:
        """True if this process may post synthetic input (Accessibility granted)."""
        if not self.available or self._ax_trusted is None:
            return False
        try:
            return bool(self._ax_trusted())
        except Exception:  # pragma: no cover
            return False

    def request_trust(self) -> bool:
        """Prompt for Accessibility if possible; returns current trust state.

        Falls back to a plain check (no prompt) if the options dictionary cannot
        be built — the UI then tells the user to grant it manually.
        """
        if not self.available:
            return False
        if self._ax_trusted_opts is None:
            return self.is_trusted()
        try:
            opts = self._prompt_options()
            return bool(self._ax_trusted_opts(opts))
        except Exception:  # pragma: no cover - CF plumbing is best-effort
            return self.is_trusted()

    def _prompt_options(self) -> ctypes.c_void_p:  # pragma: no cover - needs live CF
        cf = self._cf
        cf.CFDictionaryCreate.restype = ctypes.c_void_p
        cf.CFDictionaryCreate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                                          ctypes.POINTER(ctypes.c_void_p), ctypes.c_long,
                                          ctypes.c_void_p, ctypes.c_void_p]
        key = ctypes.c_void_p.in_dll(self._app, "kAXTrustedCheckOptionPrompt")
        true_val = ctypes.c_void_p.in_dll(cf, "kCFBooleanTrue")
        kcb = ctypes.cast(ctypes.addressof(ctypes.c_void_p.in_dll(cf, "kCFTypeDictionaryKeyCallBacks")), ctypes.c_void_p)
        vcb = ctypes.cast(ctypes.addressof(ctypes.c_void_p.in_dll(cf, "kCFTypeDictionaryValueCallBacks")), ctypes.c_void_p)
        keys = (ctypes.c_void_p * 1)(key)
        vals = (ctypes.c_void_p * 1)(true_val)
        return ctypes.c_void_p(cf.CFDictionaryCreate(None, keys, vals, 1, kcb, vcb))

    def status(self) -> OSStatus:
        if not self.available:
            return OSStatus(available=False, trusted=False,
                            reason=self._reason or "System control is unavailable.")
        if not self.is_trusted():
            return OSStatus(available=True, trusted=False,
                            reason="Grant Accessibility in System Settings -> Privacy & Security -> "
                                   "Accessibility, then turn desktop control on again.")
        return OSStatus(available=True, trusted=True)

    def screen_bounds(self) -> tuple[float, float, float, float] | None:
        """(x, y, w, h) of the main display in global point coordinates, if known."""
        return self._bounds

    # -- actions (no-op unless available) -----------------------------------
    def _post_mouse(self, event_type: int, x: float, y: float) -> None:
        if not self.available:
            return
        x, y = self._clamp(x, y)
        ev = self._cg.CGEventCreateMouseEvent(None, event_type, CGPoint(x, y), _kCGMouseButtonLeft)
        if ev:
            try:
                self._cg.CGEventPost(_kCGHIDEventTap, ev)
            finally:
                self._cf.CFRelease(ev)

    def move(self, x: float, y: float) -> None:
        if not self.available:
            return
        x, y = self._clamp(x, y)
        try:
            self._cg.CGWarpMouseCursorPosition(CGPoint(x, y))   # keep the visible cursor synced
        except Exception:  # pragma: no cover
            pass
        self._post_mouse(_kCGEventMouseMoved, x, y)

    def click(self, x: float, y: float) -> None:
        self._post_mouse(_kCGEventLeftMouseDown, x, y)
        self._post_mouse(_kCGEventLeftMouseUp, x, y)

    def scroll(self, dy: float) -> None:
        if not self.available:
            return
        ev = self._cg.CGEventCreateScrollWheelEvent(None, _kCGScrollEventUnitLine, 1, int(dy))
        if ev:
            try:
                self._cg.CGEventPost(_kCGHIDEventTap, ev)
            finally:
                self._cf.CFRelease(ev)

    def _clamp(self, x: float, y: float) -> tuple[float, float]:
        if not self._bounds:
            return float(x), float(y)
        bx, by, bw, bh = self._bounds
        return (min(max(float(x), bx), bx + bw - 1), min(max(float(y), by), by + bh - 1))
