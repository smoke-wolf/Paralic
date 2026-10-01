"""Make the README's figures and screenshots.

    python tools/make_figures.py                         # everything, into docs/images/
    python tools/make_figures.py --only cursor-motion    # one figure (names: --list)

Two kinds of pictures:

* **Screenshots** of the website in its mouse demo mode (``?demo``: the mouse
  plays the part of the eyes), taken with Playwright and Chromium against a
  server with a temporary data folder.
* **Figures** computed with Paralic's own code: the calibration, GazeNet, the
  cursor filters, the face print and the hand recogniser. Eyes are the
  *simulated people* of ``tests/synthetic.py`` (as in ``tools/benchmark.py``),
  so the numbers show what each mechanism does in simulation, not how
  accurate it is on a real person. The hand pictures use MediaPipe's public
  test photos of hands; the face print uses its public test faces only to
  compute landmarks and scores - no photo of a face is drawn.

Everything uses fixed seeds, so running it again on the same machine gives
the same files (another machine's fonts or Chromium may change a few pixels).
Needs matplotlib, Pillow and Playwright with Chromium (``python -m playwright
install chromium``), and for the face and hand figures the models in
``models/`` and the test photos (downloaded once into ``tests/.cache``).
"""

from __future__ import annotations

import argparse
import functools
import io
import math
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

import benchmark as bench  # noqa: E402
from paralic import calibration as cal  # noqa: E402
from paralic.calibration import CalibrationData, LabeledFrame, SettleTracker  # noqa: E402
from tests.synthetic import SCREEN_H, SCREEN_W, Head, VirtualUser, jitter_head  # noqa: E402

OUT = ROOT / "docs" / "images"
FPS = 30.0
DPI = 150               # figures are ~11 in wide: ~1650 px, shown at about half that on GitHub

# The website's colours (web/css/styles.css).
BG = "#0a0f1f"          # --bg
PANEL = "#0e1530"       # --bg-2
CARD = "#151f3d"        # --card
GRID = "#232d4d"        # --line (rgba(148, 163, 214, 0.18)) over the panel
TEXT = "#eef2ff"        # --text
MUTED = "#a4b0d4"       # --muted
FAINT = "#6f7ba3"       # --faint
ACCENT = "#5eead4"      # --accent (teal)
VIOLET = "#a78bfa"      # --violet
DANGER = "#fb7185"      # --danger


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": PANEL,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 12, "text.color": TEXT, "axes.labelcolor": MUTED, "axes.labelsize": 12,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0, "axes.titlesize": 13.5, "axes.titleweight": "bold",
        "axes.titlecolor": TEXT, "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": FAINT, "ytick.color": FAINT, "xtick.labelcolor": MUTED, "ytick.labelcolor": MUTED,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.major.pad": 6, "ytick.major.pad": 6,
        "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
        "legend.frameon": False, "legend.fontsize": 11, "legend.labelcolor": MUTED,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def inch_y(fig, inches: float) -> float:
    """A distance from the figure's top, in inches, as a figure fraction (y)."""
    return 1.0 - inches / fig.get_figheight()


def header(fig, title: str, subtitle: str = "") -> None:
    """Title and subtitle at the figure's top left."""
    fig.text(0.03, inch_y(fig, 0.22), title, fontsize=16.5, fontweight="bold", color=TEXT, va="top")
    if subtitle:
        fig.text(0.03, inch_y(fig, 0.58), subtitle, fontsize=11.5, color=MUTED, va="top", linespacing=1.4)


def footer(fig, text: str) -> None:
    """Where the numbers come from, small at the bottom left."""
    fig.text(0.03, 0.12 / fig.get_figheight(), text, fontsize=10, color=FAINT, va="bottom", linespacing=1.4)


def save(fig, name: str, dither: bool = False) -> Path:
    """Write a PNG into docs/images, as small as it goes without visible loss."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    return save_image(Image.open(buf), name, dither=dither)


def save_image(img: Image.Image, name: str, dither: bool = False) -> Path:
    """Save as an optimised PNG, with a 256-colour palette when that is clearly
    smaller. Charts (flat colours) take the palette as it is; screenshots and
    photos are dithered onto it, so gradients do not band."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    img = img.convert("RGB")
    full = io.BytesIO()
    img.save(full, format="PNG", optimize=True)
    if dither:
        palette = img.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
        small_img = img.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
    else:
        small_img = img.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    small = io.BytesIO()
    small_img.save(small, format="PNG", optimize=True)
    data = small.getvalue() if len(small.getvalue()) < 0.8 * len(full.getvalue()) else full.getvalue()
    path.write_bytes(data)
    print(f"  {name}: {img.width}x{img.height}, {len(data) / 1024:.0f} KB")
    return path


SCREEN_PAD = 60          # px of margin around the drawn screen


def screen_box(fig, left: float, top: float, width: float) -> list[float]:
    """Axes rectangle (figure fractions) for a screen ``width`` wide whose top edge is at ``top``."""
    w_in = width * fig.get_figwidth()
    h = w_in * (SCREEN_H + 2 * SCREEN_PAD) / (SCREEN_W + 2 * SCREEN_PAD) / fig.get_figheight()
    return [left, top - h, width, h]


def screen_axes(ax) -> None:
    """Draw ``ax`` as a 1920 x 1080 screen (y down)."""
    ax.set_xlim(-SCREEN_PAD, SCREEN_W + SCREEN_PAD)
    ax.set_ylim(SCREEN_H + SCREEN_PAD, -SCREEN_PAD)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), SCREEN_W, SCREEN_H, boxstyle="round,pad=0,rounding_size=28",
                                facecolor=CARD, edgecolor=GRID, linewidth=1.2, zorder=0))


# ---------------------------------------------------------------------------
# Calibration: which frames count, and which dots
# ---------------------------------------------------------------------------

def record_dot(data, user, rng, tracker, point, target, look, t):
    """Record one calibration dot as the page does: labelling starts 0.6 s after
    the dot appears and lasts until the server reports 16 frames of a settled
    gaze on it (at most 4 s). ``look(ts)`` is where the eyes are ``ts`` seconds
    after the dot appeared. Returns the time after the last frame."""
    k = 0
    while True:
        ts = 0.6 + k / FPS
        f = user.features(*look(ts), jitter_head(rng, Head(), 0.3))
        data.add(LabeledFrame(t=t, features=f, target=target, kind="cal", point=point))
        t += 1 / FPS
        k += 1
        if tracker.update(("cal", point, target), f) >= 16 or k / FPS >= 4.0:
            return t


def eyes_to(prev, dot, latency, glance=None):
    """Eyes that reach ``dot`` ``latency`` s after it appears (with a short
    undershoot, as in tools/benchmark.py) and optionally glance at the
    instructions at the bottom of the screen during ``glance`` = (from, to)."""
    def look(ts):
        if glance and glance[0] <= ts < glance[1]:
            return TEXT_SPOT
        if ts < latency:
            return prev
        if ts < latency + 0.12:
            return prev[0] + 0.88 * (dot[0] - prev[0]), prev[1] + 0.88 * (dot[1] - prev[1])
        return dot
    return look


TEXT_SPOT = (SCREEN_W / 2, SCREEN_H * 0.85)     # where the instructions are


def grid_session(user, rng, wrong=()):
    """The 21 dots as the page shows them (18 frames each, like
    tests/test_labels.py); during the dots in ``wrong`` the eyes rest on the
    instructions instead of the dot."""
    data = CalibrationData()
    t = 0.0
    for i, (sx, sy) in enumerate(bench.grid21()):
        look = TEXT_SPOT if i in wrong else (sx, sy)
        for _ in range(18):
            data.add(LabeledFrame(t=t, features=user.features(*look, jitter_head(rng, Head(), 0.3)),
                                  target=(sx, sy), kind="cal", point=i))
            t += 1 / FPS
    return data


def phases_of(ts, look, prev, dot) -> list[list]:
    """[what the eyes did, first time, last time] for the frames at ``ts``."""
    out = []
    for x in ts:
        g = look(x)
        kind = ("previous dot" if g == prev else "at the text" if g == TEXT_SPOT
                else "on the dot" if g == dot else "")
        if not out or out[-1][0] != kind:
            out.append([kind, x, x])
        out[-1][2] = x
    return out


def fig_calibration_labels() -> None:
    # (a) Five dots of a calibration; at the last one the eyes are slow (still
    # on the previous dot when labelling starts) and glance at the instructions.
    user, rng = VirtualUser(seed=8), np.random.default_rng(4)
    dots = bench.grid21()[:5]
    slow = eyes_to(dots[-2], dots[-1], 1.0, glance=(1.38, 1.73))
    data, tracker, t = CalibrationData(), SettleTracker(), 0.0
    for i, dot in enumerate(dots):
        look = slow if i == len(dots) - 1 else eyes_to(dots[i - 1] if i else dot, dot, 0.3)
        t = record_dot(data, user, rng, tracker, i, dot, look, t)
    # The selection training uses, with the noise pooled over all the dots.
    groups = cal._group_frames(data.frames)
    name = f"cal:{len(dots) - 1}"
    kept = cal._selections(groups)[name]
    F = np.array([f.features for f in groups[name]])
    Z = F[:, list(cal.FIXATION_IDX)] / cal.pooled_noise(groups.values())
    cuts = [lo for lo, _ in cal.segments(Z)][1:]
    ts = 0.6 + np.arange(len(F)) / FPS
    ref = Z[kept].mean(axis=0)
    traces = [("iris left–right", (Z[:, [0, 2]] - ref[[0, 2]]).mean(axis=1)),
              ("iris up–down (between the lids)", (Z[:, [4, 5]] - ref[[4, 5]]).mean(axis=1))]

    # (b) A whole calibration where the eyes were on the instructions during three dots.
    wrong = (2, 7, 11)
    flagged = cal.suspect_dots(grid_session(VirtualUser(seed=9), np.random.default_rng(8), wrong=wrong).frames)

    fig = plt.figure(figsize=(11, 6.4))
    header(fig, "Calibration learns only from frames where the eyes rested on the dot",
           "Simulated eyes (tests/synthetic.py) run through the labelling code that training uses "
           "(paralic/calibration.py).")
    left, width = 0.085, 0.43
    strip = fig.add_axes([left, 0.73, width, 0.05])
    axes = [fig.add_axes([left, 0.47, width, 0.22]), fig.add_axes([left, 0.21, width, 0.22])]
    screen = fig.add_axes(screen_box(fig, 0.575, 0.78, 0.405))
    fig.text(left, 0.835, "One dot, slow eyes: which frames are used", fontsize=13, fontweight="bold")
    fig.text(0.575, 0.835, "21 dots: the ones the eyes missed are left out", fontsize=13, fontweight="bold")

    x0, x1 = ts[0] - 0.5 / FPS, ts[-1] + 0.5 / FPS
    for kind, a, b in phases_of(ts, slow, dots[-2], dots[-1]):
        on = kind == "on the dot"
        strip.add_patch(Rectangle((a - 0.5 / FPS, 0), b - a + 1 / FPS, 1, facecolor=ACCENT if on else GRID,
                                  edgecolor=BG, lw=2))
        if b - a > 0.2:
            strip.text((a + b) / 2, 0.5, kind, color="#042f2a" if on else MUTED, fontsize=9.5, ha="center",
                       va="center", fontweight="bold" if on else "normal")
    strip.set_xlim(x0, x1)
    strip.set_ylim(0, 1)
    strip.axis("off")
    strip.text(x0, 1.25, "what the simulated eyes did", color=FAINT, fontsize=10, va="bottom")

    a, b = np.flatnonzero(kept)[[0, -1]]
    for ax, (label, y) in zip(axes, traces):
        ax.axvspan(ts[a] - 0.5 / FPS, ts[b] + 0.5 / FPS, color=ACCENT, alpha=0.12, lw=0, zorder=0)
        for c in cuts:
            ax.axvline(ts[c] - 0.5 / FPS, color=MUTED, lw=1.2, alpha=0.7, zorder=1)
        ax.plot(ts, y, color=FAINT, lw=1.2, zorder=2)
        ax.scatter(ts[~kept], y[~kept], s=22, facecolor=PANEL, edgecolor=FAINT, lw=1.2, zorder=3)
        ax.scatter(ts[kept], y[kept], s=26, facecolor=ACCENT, edgecolor=PANEL, lw=1.0, zorder=4)
        ax.set_xlim(x0, x1)
        pad = 0.15 * np.ptp(y)
        ax.set_ylim(y.min() - pad, y.max() + pad)
        ax.text(0.01, 0.95, label, transform=ax.transAxes, color=MUTED, fontsize=10.5, va="top")
        ax.grid(axis="y")
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3, integer=True))
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("seconds after the dot appeared")
    fig.text(0.022, 0.45, "eye features (in units of the\nperson's frame-to-frame noise)", rotation=90,
             color=MUTED, fontsize=11, ha="center", va="center")
    axes[1].scatter([], [], s=26, facecolor=ACCENT, edgecolor=PANEL, label=f"used ({kept.sum()} frames)")
    axes[1].scatter([], [], s=22, facecolor=PANEL, edgecolor=FAINT, lw=1.2, label=f"left out ({(~kept).sum()})")
    axes[1].plot([], [], color=MUTED, lw=1.2, alpha=0.7, label="eyes moved (change point)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(-0.01, -0.3), ncol=3, handletextpad=0.4,
                   columnspacing=1.1, fontsize=10.5)

    screen_axes(screen)
    tx, ty = TEXT_SPOT
    screen.add_patch(FancyBboxPatch((tx - 170, ty - 26), 340, 52, boxstyle="round,pad=0,rounding_size=14",
                                    facecolor=GRID, edgecolor="none", zorder=1))
    screen.text(tx, ty, "instructions", color=MUTED, fontsize=9.5, ha="center", va="center", zorder=2)
    for i, (sx, sy) in enumerate(bench.grid21()):
        out = f"cal:{i}" in flagged
        if i in wrong:
            screen.annotate("", xy=(tx, ty - 26), xytext=(sx, sy), zorder=3,
                            arrowprops=dict(arrowstyle="-|>", color=DANGER, lw=1.6, alpha=0.9,
                                            shrinkA=9, shrinkB=1, mutation_scale=12))
        screen.scatter([sx], [sy], s=140, facecolor=DANGER if out else ACCENT, edgecolor=CARD, lw=2, zorder=4)
    screen.scatter([], [], s=110, facecolor=ACCENT, edgecolor=CARD, label="used for training")
    screen.scatter([], [], s=110, facecolor=DANGER, edgecolor=CARD, label="left out")
    screen.plot([], [], color=DANGER, lw=1.6, label="eyes were on the text")
    screen.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, handletextpad=0.4, columnspacing=1.1,
                  fontsize=10.5)
    fig.text(0.575, 0.2, "Each dot is predicted from all the others (leave one dot out);\n"
                         "a dot predicted far from where it was shown was\nprobably not looked at, and "
                         "training skips it.", color=MUTED, fontsize=10.5, va="bottom", linespacing=1.45)
    footer(fig, f"Left: the recording is split where the eye features change; the last steady stretch that is not "
                f"the previous dot is used (its first 3 frames go too).\nRight: the eyes were on the instructions "
                f"during 3 of 21 dots; suspect_dots() flagged "
                f"{', '.join(sorted(flagged, key=lambda s: int(s.split(':')[1])))}.")
    save(fig, "calibration-labels.png")


# ---------------------------------------------------------------------------
# GazeNet: what calibration learns
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def calibrated_person(seed: int = 61):
    """A typical simulated person (tests/synthetic.py) calibrated as the page
    does it (tools/benchmark.py's twin of the 21 dots + the head movement)."""
    data, _ = bench.label_session(VirtualUser(seed=seed), latency=0.25, glance_p=0.0, seed=seed, settled=True)
    model, info = cal.fit_full_calibration(data)
    return model, info


def test_fixations(model, user, n=60, frames=8, seed=7):
    """(targets, per-frame estimates) for short fixations at new screen
    positions - the same test as tools/benchmark.py's fixation_error()."""
    rng = np.random.default_rng(seed)
    targets, estimates = [], []
    for _ in range(n):
        sx, sy = rng.uniform(0.05, 0.95) * SCREEN_W, rng.uniform(0.05, 0.95) * SCREEN_H
        F = np.array([user.features(sx, sy, Head()) for _ in range(frames)])
        targets.append((sx, sy))
        estimates.append(model.predict(F))
    return np.array(targets), np.array(estimates)


def fig_calibration_accuracy() -> None:
    seed = 61
    model, info = calibrated_person(seed)
    targets, frames = test_fixations(model, VirtualUser(seed=seed))
    means = frames.mean(axis=1)
    errors = np.hypot(*(means - targets).T)
    assert abs(errors.mean() - bench.fixation_error(model, VirtualUser(seed=seed))) < 1e-6

    fig = plt.figure(figsize=(11, 6.3))
    header(fig, "What the personal network learns",
           "A simulated person looks at the 21 calibration dots (and moves the head at the centre one). GazeNet "
           "is trained on\nthose frames, then tested at 60 new places on the screen that it never saw.")
    screen = fig.add_axes(screen_box(fig, 0.03, inch_y(fig, 1.3), 0.62))
    screen_axes(screen)
    for sx, sy in bench.grid21():
        screen.scatter([sx], [sy], s=210, facecolor="none", edgecolor=MUTED, lw=1.6, zorder=2)
    for (tx, ty), (px, py) in zip(targets, means):
        screen.plot([tx, px], [ty, py], color=ACCENT, lw=1.6, zorder=3)
    screen.scatter(targets[:, 0], targets[:, 1], s=46, marker="+", color=TEXT, lw=1.4, zorder=4)
    screen.scatter(means[:, 0], means[:, 1], s=26, color=ACCENT, edgecolor=CARD, lw=0.8, zorder=5)
    # Up close: a fixation with a typical (near median) error and no other test fixation nearby.
    def window(i):
        return 10 * math.ceil(1.15 * max(np.abs(frames[i] - targets[i]).max(), 40) / 10)

    order = list(np.argsort(np.abs(errors - np.median(errors))))
    alone = [i for i in order if (np.abs(np.delete(np.vstack([targets, means]), [i, i + len(targets)], axis=0)
                                         - targets[i]).max(axis=1) > window(i) + 30).all()]
    pick = (alone or order)[0]
    tx, ty = targets[pick]
    P, m = frames[pick], means[pick]
    half = window(pick)
    screen.add_patch(Rectangle((tx - half, ty - half), 2 * half, 2 * half, facecolor="none", edgecolor=VIOLET,
                               lw=1.5, zorder=6))
    screen.scatter([], [], s=150, facecolor="none", edgecolor=MUTED, lw=1.6, label="calibration dot")
    screen.scatter([], [], s=60, marker="+", color=TEXT, lw=1.4, label="where the eyes looked")
    screen.scatter([], [], s=40, color=ACCENT, label="network's estimate")
    screen.legend(loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=3, handletextpad=0.3, columnspacing=1.2,
                  fontsize=10.5)

    # One fixation up close.
    side_in = 2.55
    x = 0.7
    fig.text(x, inch_y(fig, 1.25), "One fixation up close", fontsize=12.5, fontweight="bold", va="top")
    zoom = fig.add_axes([x, inch_y(fig, 1.6 + side_in), side_in / fig.get_figwidth(), side_in / fig.get_figheight()])
    zoom.set_facecolor(CARD)
    for edge in zoom.spines.values():
        edge.set_visible(True)
        edge.set_color(VIOLET)
        edge.set_linewidth(1.5)
    zoom.set_xticks([])
    zoom.set_yticks([])
    zoom.set_xlim(tx - half, tx + half)
    zoom.set_ylim(ty + half, ty - half)
    for sx, sy in bench.grid21():                      # what else is in the box on the screen
        if abs(sx - tx) < half and abs(sy - ty) < half:
            zoom.scatter([sx], [sy], s=210, facecolor="none", edgecolor=MUTED, lw=1.6, zorder=2)
    for i in range(len(targets)):
        if i != pick and abs(targets[i, 0] - tx) < half and abs(targets[i, 1] - ty) < half:
            zoom.scatter(*targets[i], s=46, marker="+", color=TEXT, lw=1.4, zorder=2)
            zoom.scatter(*means[i], s=26, color=ACCENT, edgecolor=CARD, lw=0.8, zorder=2)
    for p in P:
        zoom.plot([p[0], m[0]], [p[1], m[1]], color=ACCENT, lw=0.9, alpha=0.35)
    zoom.scatter(P[:, 0], P[:, 1], s=36, color=ACCENT, alpha=0.55, edgecolor="none")
    zoom.scatter([m[0]], [m[1]], s=110, color=ACCENT, edgecolor=TEXT, lw=1.5, zorder=4)
    zoom.scatter([tx], [ty], s=280, marker="+", color=TEXT, lw=2.2, zorder=5)
    bar = 50
    zoom.plot([tx - 0.85 * half, tx - 0.85 * half + bar], [ty + 0.85 * half] * 2, color=MUTED, lw=2,
              solid_capstyle="butt")
    zoom.text(tx - 0.85 * half + bar / 2, ty + 0.8 * half, f"{bar} px", color=MUTED, fontsize=9.5, ha="center",
              va="bottom")
    fig.text(x, inch_y(fig, 1.7 + side_in), f"8 camera frames (faint) average to\n{errors[pick]:.0f} px from "
             f"where the eyes looked.", color=MUTED, fontsize=10.5, va="top", linespacing=1.4)
    fig.text(x, inch_y(fig, 2.35 + side_in), f"{errors.mean():.0f} px", fontsize=24, fontweight="bold",
             color=TEXT, va="top")
    fig.text(x, inch_y(fig, 2.78 + side_in), "mean error at the 60 new places,\nin simulation (real webcams: "
             "~1–3 cm).", color=MUTED, fontsize=10.5, va="top", linespacing=1.4)
    footer(fig, f"VirtualUser(seed={seed}) from tests/synthetic.py on a 1920 × 1080 screen (55.6 px per cm). "
                "A simulation, not a real-world accuracy.")
    save(fig, "calibration-accuracy.png")


# ---------------------------------------------------------------------------
# Cursor motion: from noisy estimates to a calm cursor
# ---------------------------------------------------------------------------

class CursorMotion:
    """Python twin of CursorMotion in web/js/motion.js: the drawn cursor
    glides on a critically damped spring towards its goal, which (with "hold
    still while you look") is the running average of the current fixation."""

    OMEGA = {"snappy": 30.0, "balanced": 17.0, "glide": 9.5}
    HOLD_TAU = 0.6

    def __init__(self, style: str = "balanced", hold: bool = True, radius: float = 40.0):
        self.omega, self.hold, self.radius = self.OMEGA[style], hold, radius
        self.pos = self.goal = self.pending = None
        self.vel = np.zeros(2)
        self.fix = None                     # [centre, samples, time of the last one]

    def add_sample(self, p, t: float) -> None:
        p = np.asarray(p, float)
        if self.pos is None:
            self.pos, self.goal, self.fix = p.copy(), p.copy(), [p.copy(), 1, t]
            return
        centre, n, last = self.fix
        if np.hypot(*(p - centre)) <= self.radius:
            self.pending = None
            n += 1
            k = max(1 / n, 1 - math.exp(-max(0.0, t - last) / self.HOLD_TAU))
            self.fix = [centre + (p - centre) * k, n, t]
        elif self.pending is None:
            self.pending = p                # a single stray sample: wait for the next one
            return
        else:
            self.fix = [(p + self.pending) / 2, 2, t]
            self.pending = None
        self.goal = self.fix[0].copy() if self.hold else p.copy()

    def step(self, dt: float) -> np.ndarray:
        left = min(dt, 0.1)
        while left > 0:
            h = min(left, 1 / 240)
            self.vel = self.vel + (self.omega ** 2 * (self.goal - self.pos) - 2 * self.omega * self.vel) * h
            self.pos = self.pos + self.vel * h
            left -= h
        if np.hypot(*(self.goal - self.pos)) < 0.3 and np.hypot(*self.vel) < 2:
            self.pos, self.vel = self.goal.copy(), np.zeros(2)
        return self.pos.copy()


def fig_cursor_motion() -> None:
    from paralic.filters import GazeStabilizer
    from paralic.personalize import smoothing_params, tune_smoothing

    seed = 61
    model, _ = calibrated_person(seed)
    user, rng = VirtualUser(seed=seed), np.random.default_rng(3)
    # "Auto" smoothing, as after a calibration: tuned to the person's measured
    # jitter (the spread of the estimates while looking at the test dots).
    _, frames = test_fixations(model, VirtualUser(seed=seed), n=20)
    spread = float(np.sqrt(((frames - frames.mean(axis=1, keepdims=True)) ** 2).sum(axis=2).mean(axis=1)).mean())
    tuned = tune_smoothing(spread)
    stabilizer = GazeStabilizer(smoothing_params(tuned["tuned_level"]))
    motion = CursorMotion("balanced", hold=True, radius=float(np.clip(3 * tuned["expected_jitter_px"], 25, 90)))

    # One second on the first spot to settle the filters, then the part shown.
    a, b, jump, end = np.array([700.0, 560.0]), np.array([1240.0, 420.0]), 1.0, 2.6
    t_frames = np.arange(-1.0, end, 1 / FPS)
    truth = np.array([a if t < jump else b for t in t_frames])
    raw = np.array([model.predict(user.features(*g, jitter_head(rng, Head(), 0.3))[None])[0] for g in truth])
    smooth, t_draw, drawn = [], [], []
    for t, r in zip(t_frames, raw):
        s, _ = stabilizer.update(t, r, False)
        smooth.append(s)
        motion.add_sample(s, t)
        for k in range(2):                 # the page draws at 60 fps
            t_draw.append(t + k / 60)
            drawn.append(motion.step(1 / 60))
    smooth, t_draw, drawn = np.array(smooth), np.array(t_draw), np.array(drawn)

    def jitter(series, times):
        """RMS distance from the mean during the last 0.8 s (on the second spot)."""
        sel = series[times >= end - 0.8]
        return float(np.sqrt(((sel - sel.mean(axis=0)) ** 2).sum(axis=1).mean()))

    j_raw, j_smooth, j_drawn = jitter(raw, t_frames), jitter(smooth, t_frames), jitter(drawn, t_draw)
    reach = t_draw[(t_draw >= jump) & (np.hypot(*(drawn - b).T) < 0.1 * np.hypot(*(b - a)))][0] - jump

    fig = plt.figure(figsize=(11, 5.2))
    header(fig, "From noisy estimates to a calm cursor",
           "A simulated person looks at one spot, then jumps to another. Each camera frame gives a new estimate; "
           "the server\nsmooths them (One Euro filter) and the page glides the cursor to the average of each "
           "fixation.")
    ax = fig.add_axes([0.085, 0.17, 0.89, 0.5])
    shown = t_frames >= 0
    ax.plot(t_frames[shown], truth[shown, 0], color=TEXT, lw=1.3, alpha=0.85, zorder=2, drawstyle="steps-post",
            label="where the eyes looked")
    ax.scatter(t_frames[shown], raw[shown, 0], s=16, color=FAINT, alpha=0.9, edgecolor="none", zorder=3,
               label=f"network estimate, every frame (jitter {j_raw:.0f} px)")
    ax.plot(t_frames[shown], smooth[shown, 0], color=VIOLET, lw=1.7, zorder=4,
            label=f"after the One Euro filter on the server ({j_smooth:.0f} px)")
    seen = t_draw >= 0
    ax.plot(t_draw[seen], drawn[seen, 0], color=ACCENT, lw=2.8, zorder=5,
            label=f"the cursor on the page ({j_drawn:.0f} px)")
    ax.set_xlim(0, end)
    ax.set_xlabel("seconds")
    ax.set_ylabel("left–right position (px)")
    ax.grid(axis="y")
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.annotate("the eyes jump", xy=(jump, a[0] + 0.35 * (b[0] - a[0])),
                xytext=(jump - 0.42, a[0] + 0.62 * (b[0] - a[0])), color=MUTED, fontsize=10.5, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.annotate(f"the cursor arrives\n{reach * 1000:.0f} ms later", xy=(jump + reach, b[0] - 0.1 * (b[0] - a[0])),
                xytext=(jump + reach + 0.38, a[0] + 0.45 * (b[0] - a[0])), color=MUTED, fontsize=10.5,
                ha="left", arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.01), ncol=2, fontsize=10.5, handletextpad=0.5,
              columnspacing=1.6, markerscale=1.6)
    footer(fig, f"GazeNet → GazeStabilizer with 'Auto' smoothing tuned to this person (level "
                f"{tuned['tuned_level']:g}) → a twin of web/js/motion.js ('Balanced', hold still). "
                "Jitter: RMS over the last 0.8 s.")
    save(fig, "cursor-motion.png")


# ---------------------------------------------------------------------------
# Personalisation: the benchmark on simulated people (docs/benchmark.md)
# ---------------------------------------------------------------------------

def benchmark_table(text: str, heading: str) -> list[dict]:
    """The first table under ``## heading`` in docs/benchmark.md, as rows of {column: cell}."""
    lines = text.split("## " + heading, 1)[1].splitlines() if "## " + heading in text else []
    rows = []
    for line in lines:
        if line.startswith("|"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif rows:
            break
    if len(rows) < 3:
        raise SystemExit(f"docs/benchmark.md has no table under '{heading}' (run tools/benchmark.py)")
    return [dict(zip(rows[0], r)) for r in rows[2:]]


def numbers(cell: str) -> list[float]:
    """The numbers in a table cell ("134 px → 36 px" -> [134, 36])."""
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", cell)]


def dumbbell(ax, rows, unit: str, xmin: float, xmax: float) -> None:
    """Rows of (label, before, after, note) as a dumbbell chart: grey = before, teal = after."""
    for y, (label, before, after, note) in enumerate(rows):
        ax.text(-0.03, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", color=MUTED,
                fontsize=10.5)
        ax.axhline(y, color=GRID, lw=1.0, zorder=0)
        if before is None:
            ax.text(0.02, y, note, transform=ax.get_yaxis_transform(), ha="left", va="center", color=MUTED,
                    fontsize=10.5, style="italic")
            continue
        ax.plot([before, after], [y, y], color=FAINT, lw=2.2, zorder=1, clip_on=False)
        ax.scatter([before], [y], s=80, color=FAINT, edgecolor=PANEL, lw=2, zorder=2, clip_on=False)
        ax.scatter([after], [y], s=95, color=ACCENT, edgecolor=PANEL, lw=2, zorder=3, clip_on=False)
        text = f"{after:.0f}{unit}" if before == after else f"{before:.0f} → {after:.0f}{unit}"
        ax.text(max(before, after) + 0.06 * (xmax - xmin), y, text + (f"  {note}" if note else ""), va="center",
                color=TEXT, fontsize=10.5)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlim(xmin, xmax)
    ax.set_yticks([])
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)


DUMBBELL_ROW, DUMBBELL_GAP, DUMBBELL_HEADING = 0.4, 0.5, 0.36      # inches


def dumbbell_height(groups) -> float:
    """Inches the groups take, from the first heading to the bottom of the last row."""
    return sum(DUMBBELL_HEADING + DUMBBELL_ROW * len(rows) for _, rows in groups) + DUMBBELL_GAP * (len(groups) - 1)


def dumbbell_groups(fig, groups, x_label: float, x_axes: float, w_axes: float, top: float, unit: str,
                    xmin: float, xmax: float, xlabel: str, ticks=None) -> None:
    """Groups of dumbbell rows, one under the other: [(heading, rows), ...]; positions in inches."""
    row, gap = DUMBBELL_ROW, DUMBBELL_GAP
    y = top
    for k, (heading, rows) in enumerate(groups):
        fig.text(x_label / fig.get_figwidth(), inch_y(fig, y), heading, color=TEXT, fontsize=11.5,
                 fontweight="bold", va="top")
        y += DUMBBELL_HEADING
        ax = fig.add_axes([x_axes / fig.get_figwidth(), inch_y(fig, y + row * len(rows)),
                           w_axes / fig.get_figwidth(), row * len(rows) / fig.get_figheight()])
        dumbbell(ax, rows, unit, xmin, xmax)
        if ticks is not None:
            ax.set_xticks(ticks)
        if k < len(groups) - 1:
            ax.tick_params(labelbottom=False)
            ax.spines["bottom"].set_visible(False)
        else:
            ax.set_xlabel(xlabel)
        y += row * len(rows) + gap


def fig_personalisation() -> None:
    text = (ROOT / "docs" / "benchmark.md").read_text(encoding="utf-8")
    adjust = benchmark_table(text, "Quick adjust after sitting differently")
    finetune = benchmark_table(text, "Fine-tuning while the site is used")
    blinks = benchmark_table(text, "Double blinks: standard vs personal thresholds")
    winks = benchmark_table(text, "Held winks (press & drag): standard vs personal thresholds")
    n_people = int(numbers(finetune[0]["Accepted"])[1])

    def label_of(row_name: str) -> str:
        """A row's name from the report, shortened to fit beside the chart."""
        for cut in (" (24 practice hits)", " as calibration", " (closure rises only 0.2)", " (closed eye reaches 0.48)",
                    " (rests at 0.5)", " while winking", " (facial palsy)"):
            row_name = row_name.replace(cut, "")
        row_name = row_name.replace("One eye hardly closes", "One eye barely closes").replace("of labels", "labels")
        return row_name[:1].upper() + row_name[1:]

    adjusted = [(label_of(r["Sitting"]), numbers(r["No adjust"])[0], numbers(r["9 dots"])[0], "") for r in adjust]
    learned = []
    for r in finetune:
        accepted = numbers(r["Accepted"])
        if accepted[0] == 0:
            learned.append((label_of(r["Situation"]), None, None,
                            f"refused {r['Refused (unreliable data)'].replace('/', ' of ')}: the old network stays"))
            continue
        before, after = numbers(r["Median error before → after (accepted)"])
        note = "" if accepted[0] == accepted[1] else f"({accepted[0]:.0f} of {accepted[1]:.0f} swapped)"
        learned.append((label_of(r["Situation"]), before, after, note))
    blinked = [(label_of(r["People"]), numbers(r["Caught (standard)"])[0], numbers(r["Caught (personal)"])[0], "")
               for r in blinks]
    winked = [(label_of(r["People"]), *numbers(r["Presses caught (std / personal)"]), "") for r in winks]

    gaze = [("Quick adjust after sitting differently (9 dots)", adjusted),
            ("Learning from use (champion / challenger)", learned)]
    lids = [("Double blinks caught", blinked), ("Held winks caught (press & drag)", winked)]
    top = 1.72
    fig = plt.figure(figsize=(11, top + max(dumbbell_height(gaze), dumbbell_height(lids)) + 0.95))
    W = fig.get_figwidth()
    header(fig, "Personalisation, measured on simulated people",
           f"Medians over {n_people} simulated people per row, from docs/benchmark.md (eyes from tests/synthetic.py, "
           "synthetic eyelid traces):\nsimulations that show what each mechanism does, not measurements on real "
           "people.")
    dumbbell_groups(fig, gaze, 0.33, 2.25, 2.35, top, " px", 0, 230, "gaze error (px, lower is better)")
    dumbbell_groups(fig, lids, 6.05, 7.75, 1.75, top, " %", -6, 106, "caught (%)", ticks=[0, 50, 100])
    for x, before, after in ((0.33, "without it", "with it"), (6.05, "standard thresholds", "personal thresholds")):
        fig.legend(handles=[plt.scatter([], [], s=80, color=FAINT, label=before),
                            plt.scatter([], [], s=95, color=ACCENT, label=after)],
                   loc="lower left", bbox_to_anchor=(x / W - 0.008, inch_y(fig, top - 0.08)), ncol=2,
                   handletextpad=0.3, columnspacing=1.0, fontsize=10.5)
    footer(fig, "Gaze error: of the average estimate over a short fixation at new places on a 1920 × 1080 screen. "
                "No false clicks or presses in any blink or wink row.")
    save(fig, "personalisation-benchmark.png")


# ---------------------------------------------------------------------------
# Face print (public test faces: only landmarks and scores are drawn)
# ---------------------------------------------------------------------------

def strip(ax, y, values, colour, rng, size=34) -> None:
    """Values as a jittered row of dots at height ``y``."""
    values = np.asarray(values, float)
    ax.scatter(values, y + rng.uniform(-0.22, 0.22, len(values)), s=size, color=colour, alpha=0.85,
               edgecolor=PANEL, lw=0.8, zorder=3)


def fig_faceprint() -> None:
    import faceprint_eval
    from mediapipe.tasks.python import vision

    from paralic import faceprint as FP
    from tests import face_images

    if not (ROOT / "models" / "face_landmarker.task").is_file():
        raise SystemExit("models/face_landmarker.task is missing (python -m paralic downloads it)")
    # The same frames as tools/faceprint_eval.py: five people from public test
    # images, each altered many times and run through the real face mesh.
    imgs = faceprint_eval.people()
    rng = np.random.default_rng(1)
    with faceprint_eval.landmarker() as lm:
        enrol = {name: faceprint_eval.frames(lm, v, 12, rng) for name, v in imgs.items()}
        test = {name: faceprint_eval.frames(lm, v, 10, rng) for name, v in imgs.items()}
        _, faces = faceprint_eval.detect(lm, face_images.portrait(1024))
    if not faces:
        raise SystemExit("the face mesh found no face in the test portrait")
    names = list(imgs)
    everyone = FP.FaceRecognizer(enrol)
    own, others, results = [], [], []
    for who, samples in test.items():
        for s in samples:
            rel = everyone.relative_scores([s])
            own.append(rel[who])
            others += [v for u, v in rel.items() if u != who]
            results.append(everyone.identify([s])["user"] == who)
    three = FP.FaceRecognizer({u: enrol[u] for u in names[:3]})
    strangers, turned_away = [], 0
    for who in names[3:]:
        for s in test[who]:
            strangers.append(min(three.relative_scores([s]).values()))
            turned_away += three.identify([s])["user"] is None

    fig = plt.figure(figsize=(11, 6.35))
    header(fig, "Face print: who is at the camera",
           "Each view of a face is reduced to 47 face-mesh points that expressions barely move (plus skin texture), "
           "in a frame fixed\nto the face, and compared with a weighting learned from the person's own views.")
    mesh = fig.add_axes([0.03, inch_y(fig, 5.05), 0.27, 3.45 / fig.get_figheight()])
    origin, R, scale = FP.face_frame(faces[0])
    P = (faces[0] - origin) @ R.T / scale                 # the face's own frame, in eye distances
    lines = [[P[c.start, :2], P[c.end, :2]] for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION]
    mesh.add_collection(matplotlib.collections.LineCollection(lines, colors=GRID, linewidths=0.6, zorder=1))
    mesh.scatter(P[:, 0], P[:, 1], s=2.5, color=FAINT, zorder=2, edgecolor="none")
    rigid = P[list(FP.RIGID_POINTS)]
    mesh.scatter(rigid[:, 0], rigid[:, 1], s=30, color=ACCENT, edgecolor=BG, lw=0.8, zorder=3)
    mesh.annotate("", xy=(0.55, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=VIOLET, lw=1.6), zorder=4)
    mesh.annotate("", xy=(0, 0.55), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=VIOLET, lw=1.6), zorder=4)
    mesh.set_aspect("equal")
    lo, hi = P[:, :2].min(axis=0), P[:, :2].max(axis=0)
    mesh.set_xlim(lo[0] - 0.08, hi[0] + 0.08)
    mesh.set_ylim(hi[1] + 0.05, lo[1] - 0.05)
    mesh.axis("off")
    fig.text(0.03, inch_y(fig, 1.3), "Face mesh of the test portrait", fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.03, inch_y(fig, 5.15), "teal: the 47 points of the face print\nviolet: the face's own axes "
             "(length 1 = eye distance)", color=MUTED, fontsize=10, linespacing=1.4, va="top")

    x0 = 0.355
    ax = fig.add_axes([0.53, inch_y(fig, 4.75), 0.44, 2.45 / fig.get_figheight()])
    rows = [("own print", own, ACCENT), ("other enrolled people", others, FAINT),
            ("strangers (nearest print)", strangers, VIOLET)]
    jitter = np.random.default_rng(0)
    for y, (label, values, colour) in enumerate(rows):
        strip(ax, y, values, colour, jitter)
        ax.text(-0.02, y, label, ha="right", va="center", color=MUTED, fontsize=10.5,
                transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_xlim(0.3, 200)
    ax.set_ylim(len(rows) - 0.5, -0.65)
    ax.set_yticks([])
    ax.axvline(FP.ACCEPT, color=TEXT, lw=1.3, zorder=2)
    ax.text(FP.ACCEPT * 1.08, -0.56, f"recognised below {FP.ACCEPT:g}", color=TEXT, fontsize=10.5, va="center")
    ax.set_xticks([0.5, 1, 3, 10, 30, 100])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("score (1 = a typical new view of the person's own face)")
    fig.text(x0, inch_y(fig, 1.3), "Scores of new views against the prints", fontsize=12.5, fontweight="bold",
             va="top")
    fig.text(x0, inch_y(fig, 1.68), f"{sum(len(v) for v in test.values())} new views of 5 people from public "
             "test photos; strangers: 2 of them,\nscored against the prints of the other 3.", color=MUTED,
             fontsize=10.5, va="top", linespacing=1.4)
    footer(fig, "Frames as in tools/faceprint_eval.py: still photos, each altered (turned, scaled, lit, blurred) and "
                "run through the real face mesh,\nnot people over days. Single views: "
                f"{sum(results)} of {len(results)} recognised, the rest 'not sure' (Paralic then asks), none taken for "
                f"someone else;\n{turned_away} of {len(strangers)} stranger views turned away. Only landmarks and "
                "scores are drawn here, no photo.")
    save(fig, "faceprint.png")


# ---------------------------------------------------------------------------
# Hand mode: the hand network on public photos, and a personal pinch
# ---------------------------------------------------------------------------

def face_mask(rgb: np.ndarray, margin: int = 30) -> np.ndarray:
    """Pixels of every face MediaPipe finds in ``rgb``: the outline of its face
    mesh, grown by ``margin`` px. The hand photos also show people, so only
    hand crops clear of this mask are drawn."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=(ROOT / "models" / "face_landmarker.task").read_bytes()),
        running_mode=vision.RunningMode.IMAGE, num_faces=4, min_face_detection_confidence=0.2,
        min_face_presence_confidence=0.2)
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    with vision.FaceLandmarker.create_from_options(options) as lm:
        faces = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))).face_landmarks
    for face in faces:
        P = np.array([[p.x * w, p.y * h] for p in face], np.float32)
        cv2.fillConvexPoly(mask, np.round(cv2.convexHull(P)).astype(np.int32), 255)
    return cv2.dilate(mask, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) > 0


def hand_crop(rgb: np.ndarray, pts: np.ndarray, pad: float = 0.25):
    """The hand (``pts``: 21 landmarks in pixels) cut out of a photo that also
    shows a person: (x0, y0, x1, y1) of a crop around it and an alpha matte
    for the crop that is 0 on and around any face, with soft edges.

    The crop's padding is trimmed on the side of a face first; whatever face
    margin is left inside it (e.g. near the chin, beside a hand held low) is
    made fully transparent. No hand landmark may lie in the face margin."""
    import cv2

    h, w = rgb.shape[:2]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    p = max(20.0, pad * float((hi - lo).max()))
    x0, y0 = int(max(0, lo[0] - p)), int(max(0, lo[1] - p))
    x1, y1 = int(min(w, hi[0] + p)), int(min(h, hi[1] + p))
    faces = face_mask(rgb)
    inside = faces[np.clip(pts[:, 1].astype(int), 0, h - 1), np.clip(pts[:, 0].astype(int), 0, w - 1)]
    if inside.any():
        raise SystemExit("a hand in the test photos is too close to a face to be shown")
    # Trim padding from whichever side the face is on (never into the hand).
    for _ in range(4 * max(w, h)):
        hit = faces[y0:y1, x0:x1]
        if not hit.any():
            break
        ys, xs = np.nonzero(hit)
        fx, fy = xs.mean() + x0, ys.mean() + y0
        moves = [(fx > hi[0], "x1"), (fx < lo[0], "x0"), (fy > hi[1], "y1"), (fy < lo[1], "y0")]
        options = [m for ok, m in moves if ok and {"x0": x0 + 1 < lo[0], "x1": x1 - 1 > hi[0],
                                                  "y0": y0 + 1 < lo[1], "y1": y1 - 1 > hi[1]}[m]]
        if not options:
            break
        side = options[0]
        x0, x1 = x0 + (side == "x0"), x1 - (side == "x1")
        y0, y1 = y0 + (side == "y0"), y1 - (side == "y1")
    clear = (~faces[y0:y1, x0:x1]).astype(np.float32)
    edge = np.zeros_like(clear)
    edge[6:-6, 6:-6] = 1.0
    alpha = cv2.GaussianBlur(np.minimum(clear, edge), (0, 0), 4.0) * clear     # exactly 0 on any face margin
    return (x0, y0, x1, y1), alpha


def pinch_trace(rng, closed: float, opened: float, cycles: int, frames_per: int = 15) -> np.ndarray:
    """Thumb-index distance (palm widths) of a hand pinching ``cycles`` times:
    the fingers take ~5 frames to close or open, with tracking noise."""
    out = []
    for _ in range(cycles):
        for target, n in ((opened, frames_per), (closed, frames_per)):
            start = out[-1] if out else opened
            for k in range(n):
                f = min(1.0, (k + 1) / 5)
                out.append(start + (target - start) * (0.5 - 0.5 * math.cos(math.pi * f)))
    out += [opened] * frames_per
    return np.array(out) + rng.normal(0, 0.025, len(out))


def pinch_events(distances, calibration=None) -> list[tuple[int, str]]:
    """(frame, "pinch_start" / "click") from the real recogniser, for a hand
    whose thumb tip is ``distances[frame]`` palm widths from the index tip."""
    from paralic.hand_gestures import THUMB_TIP, HandGestureRecognizer
    from tests.fakes import make_hand

    rec = HandGestureRecognizer(calibration=calibration)
    base = make_hand(fingers="curled")
    width = float(np.hypot(*(base[5, :2] - base[17, :2])))
    out = []
    for i, d in enumerate(distances):
        pts = base.copy()
        pts[THUMB_TIP, :2] = pts[8, :2] + (d * width, 0.0)
        out += [(i, e.type) for e in rec.update(i / FPS, pts)[1] if e.type in ("pinch_start", "click")]
    return out


def fig_hands() -> None:
    from mediapipe.tasks.python import vision

    from paralic.hand_gestures import (HandCalibration, HandGestureConfig, HandGestureRecognizer,
                                       estimate_pinch_thresholds, extended_fingers, split_pinch_cycle, stop_palm)
    from paralic.hands import HandTracker
    from tests.hand_images import hand_image

    model = ROOT / "models" / "hand_landmarker.task"
    if not model.is_file():
        raise SystemExit("models/hand_landmarker.task is missing (hand mode downloads it on first use)")
    model_bytes = model.read_bytes()

    fig = plt.figure(figsize=(11, 8.3))
    header(fig, "Hand mode: point to move, pinch to click",
           "Top: MediaPipe's hand network on public test photos, cropped to the hand, and what Paralic's recogniser "
           "reads from\nthe 21 landmarks. Bottom: the hand setup learning a personal pinch, for a simulated hand.")
    shots = [("pointing_up", "Pointing", "index finger out:\nits tip is the cursor"),
             ("open_hands", "Open hand = pause", "four fingers out and spread,\nthumb out: pause or resume"),
             ("thumbs_up", "No pinch", "")]
    panels = []
    for name, title, detail in shots:
        rgb = hand_image(name)
        if rgb is None:
            raise SystemExit("could not download the hand test photos (tests/hand_images.py)")
        tracker = HandTracker(model_bytes)
        try:
            obs = None
            for i in range(3):                          # VIDEO mode settles on the hand
                obs = tracker.process(rgb, 33 * i)
        finally:
            tracker.close()
        h, w = rgb.shape[:2]
        pts = obs.points_norm[:, :2] * (w, h)
        (x0, y0, x1, y1), alpha = hand_crop(rgb, pts)
        state, _ = HandGestureRecognizer().update(0.0, obs.points_norm)
        norm = obs.points_norm[:, :2]
        if name == "pointing_up":
            assert extended_fingers(norm) == {"index": True, "middle": False, "ring": False, "pinky": False}
        if name == "open_hands":
            assert stop_palm(norm) and state.open_palm
        if name == "thumbs_up":
            assert not stop_palm(norm) and not state.pinching
            detail = f"thumb and index {state.pinch_dist:.1f} palm\nwidths apart: not pinching"
        panels.append((name, title, detail, rgb, pts, (x0, y0, x1, y1), alpha))

    # One row at a common height, each photo as wide as its crop, spread over the width.
    height, margin = 2.45, 0.33
    widths = [height * (b[2] - b[0]) / (b[3] - b[1]) for *_, b, _ in panels]
    slots = [max(w, 2.0) for w in widths]           # room for the captions under narrow photos
    gap = (fig.get_figwidth() - 2 * margin - sum(slots)) / (len(slots) - 1)
    x = margin
    for (name, title, detail, rgb, pts, (x0, y0, x1, y1), alpha), width, slot in zip(panels, widths, slots):
        left = x / fig.get_figwidth()
        x += slot + gap
        ax = fig.add_axes([left, inch_y(fig, 1.42 + height), width / fig.get_figwidth(), height / fig.get_figheight()])
        crop = rgb[y0:y1, x0:x1].astype(float) / 255.0
        bg = np.array(matplotlib.colors.to_rgb(BG))
        shown = alpha[..., None] * (0.8 * crop + 0.2 * bg) + (1 - alpha[..., None]) * bg
        ax.imshow(shown, extent=(x0, x1, y1, y0), interpolation="lanczos")
        segs = [[pts[c.start], pts[c.end]] for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
        ax.add_collection(matplotlib.collections.LineCollection(segs, colors=TEXT, linewidths=1.6, alpha=0.9))
        ax.scatter(pts[:, 0], pts[:, 1], s=16, color=ACCENT, edgecolor=BG, lw=0.6, zorder=3)
        if name == "pointing_up":
            ax.scatter([pts[8, 0]], [pts[8, 1]], s=260, facecolor="none", edgecolor=VIOLET, lw=2.4, zorder=4)
        if name == "thumbs_up":
            ax.plot(*pts[[4, 8]].T, color=VIOLET, lw=2.2, zorder=4)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.axis("off")
        fig.text(left, inch_y(fig, 1.55 + height), title, fontsize=12, fontweight="bold", va="top")
        fig.text(left, inch_y(fig, 1.85 + height), detail, fontsize=10.5, color=MUTED, va="top", linespacing=1.35)

    # A hand that cannot close fully: the setup's "pinch a few times" step.
    rng = np.random.default_rng(5)
    closed, opened = 0.48, 1.0
    setup = pinch_trace(rng, closed, opened, cycles=5)
    open_d, closed_d = split_pinch_cycle(setup)
    make, brk = estimate_pinch_thresholds(open_d, closed_d)
    use = pinch_trace(np.random.default_rng(6), closed, opened, cycles=6)
    personal = HandCalibration(pinch_on=make, pinch_off=brk)
    clicks = [i for i, kind in pinch_events(use, personal) if kind == "click"]
    standard = [i for i, kind in pinch_events(use) if kind == "click"]
    std_on, std_off = HandGestureConfig().pinch_on, HandGestureConfig().pinch_off

    ax = fig.add_axes([0.1, 0.13, 0.72, inch_y(fig, 5.85) - 0.13])
    t = np.arange(len(use)) / FPS
    ax.plot(t, use, color=FAINT, lw=1.6, zorder=2)
    ax.axhline(make, color=ACCENT, lw=2.0, zorder=1)
    ax.axhline(brk, color=VIOLET, lw=2.0, zorder=1)
    ax.axhline(std_on, color=MUTED, lw=1.0, alpha=0.7, zorder=1)
    right = t[-1] + 0.1
    ax.text(right, brk, f"personal break {brk:.2f}", color=TEXT, fontsize=10.5, va="bottom")
    ax.text(right, make, f"personal make {make:.2f}", color=TEXT, fontsize=10.5, va="top")
    ax.text(right, std_on, f"standard make {std_on:.2f}", color=MUTED, fontsize=10.5, va="top")
    ax.scatter(t[clicks], np.full(len(clicks), 1.13), marker="v", s=70, color=ACCENT, zorder=3, clip_on=False)
    ax.text(t[clicks[0]] - 0.08, 1.13, "click", color=TEXT, fontsize=10.5, va="center", ha="right")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0.3, 1.18)
    ax.set_xlabel("seconds")
    ax.set_ylabel("thumb–index distance\n(palm widths)")
    ax.grid(axis="y")
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
    fig.text(0.1, inch_y(fig, 5.18), "A simulated hand that cannot close fully pinches six times", fontsize=12,
             fontweight="bold", va="top")
    fig.text(0.1, inch_y(fig, 5.46), f"Clicks with its personal thresholds: {len(clicks)} of 6; with the standard "
             f"ones ({std_on:.2f} / {std_off:.2f}): {len(standard)} of 6.", color=MUTED, fontsize=10.5, va="top")
    footer(fig, "Photos: MediaPipe's public gesture test images (tests/hand_images.py), cropped to the hand by the "
                "script; the people in them are not shown. Thresholds:\nsplit_pinch_cycle() and "
                "estimate_pinch_thresholds() on 5 s of pinching; clicks from HandGestureRecognizer.")
    save(fig, "hand-mode.png")


# ---------------------------------------------------------------------------
# Screenshots of the website (mouse demo mode: the mouse plays the eyes)
# ---------------------------------------------------------------------------

VIEWPORT = {"width": 1600, "height": 900}
SHOT_WIDTH = 1280          # screenshots are scaled down to this width


def launch_chromium(p):
    """Chromium as tests/test_browser.py finds it."""
    from tests.test_browser import CHROMIUM_CANDIDATES

    last = None
    # One raster thread and full redraws: the same page gives the same pixels every run.
    args = ["--no-sandbox", "--force-color-profile=srgb", "--hide-scrollbars", "--num-raster-threads=1",
            "--disable-partial-raster", "--run-all-compositor-stages-before-draw"]
    for path in [None, *[c for c in CHROMIUM_CANDIDATES if c]]:
        try:
            return p.chromium.launch(executable_path=path, args=args)
        except Exception as exc:  # depends on the machine
            last = exc
    raise SystemExit(f"Chromium is not available ({last}); run: python -m playwright install chromium")


def save_shot(png: bytes, name: str, width: int = SHOT_WIDTH) -> None:
    img = Image.open(io.BytesIO(png)).convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    save_image(img, name, dither=True)


def look_at(page, selector: str, fx: float = 0.5, fy: float = 0.5) -> None:
    """Move the "gaze" (the mouse, in demo mode) onto an element and let the cursor settle.

    One jump, like a real eye movement: the cursor's new fixation then starts
    exactly there (web/js/motion.js), so the picture does not depend on timing.
    """
    box = page.locator(selector).first.bounding_box()
    if not box:
        raise SystemExit(f"{selector} is not on the page")
    page.mouse.move(box["x"] + box["width"] * fx, box["y"] + box["height"] * fy)
    page.wait_for_timeout(1200)
    if not page.evaluate("document.querySelector('.gaze-hover') !== null"):
        raise SystemExit(f"looking at {selector} highlighted nothing")


def shot(page, name: str) -> None:
    page.evaluate("document.querySelectorAll('.toast').forEach((t) => t.remove())")
    save_shot(page.screenshot(animations="disabled"), name)


def fig_screenshots() -> None:
    from playwright.sync_api import sync_playwright

    from paralic.server import create_app
    from tests.fakes import FakeHandTracker, FakeTracker
    from tests.test_browser import _serve

    with tempfile.TemporaryDirectory() as data:
        # Trackers that are never fed: the start screen shows both modes as
        # available, and the pages run in the mouse demo mode.
        app = create_app(data_dir=Path(data), tracker_factory=FakeTracker, hand_tracker_factory=FakeHandTracker,
                         model_error=None)
        server, thread, url = _serve(app)
        try:
            with sync_playwright() as p:
                browser = launch_chromium(p)
                errors = []

                # The start screen. At 1600 x 900 its card scrolls; a taller
                # window shows all of it (the layout depends on the width only).
                ctx = browser.new_context(viewport=VIEWPORT)
                page = ctx.new_page()
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(f"{url}/?manual&nofs")
                page.wait_for_selector(".overlay-card .version-tag")
                page.wait_for_function("!document.querySelector('#hand-btn').disabled")
                page.wait_for_timeout(300)
                for _ in range(6):
                    over = page.evaluate("(() => { const c = document.querySelector('.overlay-card');"
                                         " return c.scrollHeight - c.clientHeight; })()")
                    if over <= 0:
                        break
                    size = page.viewport_size
                    page.set_viewport_size({"width": size["width"], "height": size["height"] + int(over) + 40})
                    page.wait_for_timeout(300)
                box = page.locator(".overlay-card").first.bounding_box()
                clip = {"x": box["x"] - 60, "y": box["y"] - 50, "width": box["width"] + 120,
                        "height": box["height"] + 100}
                save_shot(page.screenshot(clip=clip, animations="disabled"), "screenshot-start.png", width=1100)
                ctx.close()

                ctx = browser.new_context(viewport=VIEWPORT)
                page = ctx.new_page()
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(f"{url}/?demo&nofs")
                page.wait_for_function("window.paralic && window.paralic.state.started")
                page.evaluate("window.paralic.tracker.request({type: 'user_rename', name: 'Sam'}, 'users')")
                page.wait_for_timeout(500)

                look_at(page, 'a.tile[href="#/talk"]', 0.3, 0.42)
                shot(page, "screenshot-home.png")

                page.evaluate("location.hash = '#/explore'")
                page.wait_for_selector(".planet-card")
                look_at(page, 'a.planet-card[href="#/explore/saturn"]', 0.5, 0.36)
                shot(page, "screenshot-explore.png")

                page.evaluate("location.hash = '#/talk'")
                page.wait_for_selector(".talk .tabs")
                page.locator(".tabs .btn", has_text="Keyboard").click()
                for ch in "CAN YOU OPEN THE WIN":
                    page.locator(".key[aria-label='Space']" if ch == " " else f'.key[aria-label="{ch}"]').click()
                page.wait_for_selector('.suggestion:has-text("window")')
                look_at(page, '.suggestion:has-text("window")', 0.8, 0.55)
                shot(page, "screenshot-talk.png")

                page.evaluate("location.hash = '#/settings'")
                page.wait_for_selector('[data-gesture="motion"]')
                look_at(page, '[data-gesture="motion"] .opt[data-value="glide"]', 0.72, 0.62)
                shot(page, "screenshot-settings.png")
                ctx.close()
                browser.close()
                if errors:
                    raise SystemExit(f"the page reported errors: {errors}")
        finally:
            server.should_exit = True
            thread.join(timeout=5)


FIGURES = {
    "screenshots": fig_screenshots,
    "calibration-labels": fig_calibration_labels,
    "calibration-accuracy": fig_calibration_accuracy,
    "cursor-motion": fig_cursor_motion,
    "personalisation": fig_personalisation,
    "faceprint": fig_faceprint,
    "hand-mode": fig_hands,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", action="append", choices=sorted(FIGURES), help="make only this figure (repeatable)")
    ap.add_argument("--list", action="store_true", help="list the figures and exit")
    args = ap.parse_args(argv)
    if args.list:
        print("\n".join(FIGURES))
        return 0
    style()
    started = time.perf_counter()
    for name in args.only or FIGURES:
        t0 = time.perf_counter()
        print(f"{name}…", flush=True)
        FIGURES[name]()
        print(f"  ({time.perf_counter() - t0:.0f} s)", flush=True)
    print(f"Done in {time.perf_counter() - started:.0f} s → {OUT.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
