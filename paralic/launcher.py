"""A small native launcher window directing the user to finish setup in the
browser, then start controlling the computer with their eyes.

The gaze pipeline needs the browser (that is where the webcam is), so when
Paralic starts we pop a native window that points the user there. On macOS this
uses AppleScript (``osascript``) so there is no GUI dependency; on other
platforms it is a no-op and the normal "open browser" behaviour is used instead.
"""

from __future__ import annotations

import subprocess
import sys
import webbrowser


def available() -> bool:
    return sys.platform == "darwin"


def _dialog(message: str, buttons: list[str], default: str) -> str | None:
    """Show a native dialog; return the button clicked (or None on cancel/error)."""
    esc = (message.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n"))
    btns = ", ".join(f'"{b}"' for b in buttons)
    script = (f'display dialog "{esc}" with title "Paralic" '
              f'buttons {{{btns}}} default button "{default}" with icon note')
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
    except Exception:
        return None
    for b in buttons:
        if f"button returned:{b}" in out:
            return b
    return None  # user cancelled / error


def run_launcher(url: str, on_quit=None) -> None:
    """Show the launcher and keep it available until the user quits.

    Blocking — run it on its own (daemon) thread so the server keeps serving.
    Each ``osascript`` call is a separate process, so this does not need the
    main thread (unlike Tk).
    """
    if not available():
        return
    intro = ("Paralic eye tracking is running.\n\n"
             "1.  Click “Open in Browser”.\n"
             "2.  Allow the camera, then follow the dots to calibrate.\n"
             "3.  Browse — and control your computer — with your eyes.\n\n"
             + url)
    running = ("Paralic is running in your browser.\n\n"
               "Finish the setup there (follow the dots to calibrate).\n"
               "You can reopen the tab or quit Paralic here.\n\n" + url)
    first = True
    while True:
        if first:
            choice = _dialog(intro, ["Quit", "Open in Browser"], "Open in Browser")
            first = False
        else:
            choice = _dialog(running, ["Quit", "Reopen Paralic"], "Reopen Paralic")
        if choice in ("Open in Browser", "Reopen Paralic"):
            try:
                webbrowser.open(url)
            except Exception:
                pass
            continue
        break  # Quit / cancel
    if on_quit:
        on_quit()
