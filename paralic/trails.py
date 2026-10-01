"""Procedurally generated pursuit trails.

Smooth paths for *smooth-pursuit* calibration and for the eye-tracking practice
games: the user follows a moving dot with their eyes while the browser streams
labelled frames (the dot position is the gaze target), which both trains and
measures the gaze model under motion.

All coordinates are normalised to ``[0, 1]`` screen fractions, kept inside a
``margin`` border so the dot never sits under the window edge. Paths are
deterministic given a ``seed`` so a session can be reproduced.

Kinds:
* ``lissajous`` - a classic Lissajous figure (smooth, self-crossing loops),
* ``spline``    - a random smooth path (Catmull-Rom through random waypoints),
* ``snake``     - a boustrophedon sweep that covers the screen row by row, the
                  path followed in the Snake tracking game.
"""

from __future__ import annotations

import math
from typing import Optional

Point = tuple[float, float]

KINDS = ("lissajous", "spline", "snake")


def _clamp01(v: float, margin: float) -> float:
    lo, hi = margin, 1.0 - margin
    return lo if v < lo else hi if v > hi else v


def _scale(v: float, margin: float) -> float:
    """Map ``v`` in [-1, 1] into [margin, 1-margin]."""
    return margin + (v + 1.0) * 0.5 * (1.0 - 2.0 * margin)


def lissajous(n: int = 300, a: int = 3, b: int = 2, delta: float = math.pi / 2,
              margin: float = 0.1, turns: float = 1.0) -> list[Point]:
    """A Lissajous curve: x = sin(a t + delta), y = sin(b t), t over ``turns`` loops."""
    n = max(2, int(n))
    pts: list[Point] = []
    for i in range(n):
        t = turns * 2.0 * math.pi * i / (n - 1)
        x = _scale(math.sin(a * t + delta), margin)
        y = _scale(math.sin(b * t), margin)
        pts.append((x, y))
    return pts


def _catmull_rom(p0: Point, p1: Point, p2: Point, p3: Point, steps: int) -> list[Point]:
    out: list[Point] = []
    for s in range(steps):
        t = s / steps
        t2, t3 = t * t, t * t * t
        out.append((
            0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
                   + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                   + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
            0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
                   + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                   + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3),
        ))
    return out


def spline(n: int = 300, waypoints: int = 7, margin: float = 0.12, seed: int = 0) -> list[Point]:
    """A smooth random path: Catmull-Rom spline through random waypoints."""
    import random
    rng = random.Random(seed)
    wp = [(_clamp01(rng.random(), margin), _clamp01(rng.random(), margin)) for _ in range(max(3, waypoints))]
    # Pad ends so the spline passes through the first and last waypoints.
    pad = [wp[0]] + wp + [wp[-1]]
    steps = max(2, n // (len(wp)))
    pts: list[Point] = []
    for i in range(1, len(pad) - 2):
        pts.extend(_catmull_rom(pad[i - 1], pad[i], pad[i + 1], pad[i + 2], steps))
    pts.append(wp[-1])
    return [(_clamp01(x, margin), _clamp01(y, margin)) for x, y in pts]


def snake(cols: int = 5, rows: int = 4, margin: float = 0.1, per_leg: int = 40) -> list[Point]:
    """A boustrophedon (snake) sweep: left-to-right, down a row, right-to-left, …

    Returns a densely sampled path so a dot can glide along it. Used by the Snake
    pursuit game and as a thorough, screen-covering pursuit trail.
    """
    cols = max(2, cols)
    rows = max(2, rows)
    xs = [margin + (1 - 2 * margin) * c / (cols - 1) for c in range(cols)]
    ys = [margin + (1 - 2 * margin) * r / (rows - 1) for r in range(rows)]
    waypoints: list[Point] = []
    for r, y in enumerate(ys):
        row_xs = xs if r % 2 == 0 else list(reversed(xs))
        for x in row_xs:
            waypoints.append((x, y))
    # Densify each leg with linear interpolation (snake moves in straight lines).
    pts: list[Point] = []
    for i in range(len(waypoints) - 1):
        (x0, y0), (x1, y1) = waypoints[i], waypoints[i + 1]
        for s in range(per_leg):
            t = s / per_leg
            pts.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    pts.append(waypoints[-1])
    return pts


def trail(kind: str = "lissajous", n: int = 300, seed: int = 0, margin: float = 0.1) -> list[Point]:
    """Dispatch to a named trail generator. Unknown kinds fall back to lissajous."""
    if kind == "spline":
        return spline(n=n, margin=max(margin, 0.1), seed=seed)
    if kind == "snake":
        return snake(margin=margin)
    # vary the Lissajous figure a little with the seed for replay variety
    a, b = (3, 2) if seed % 2 == 0 else (5, 4)
    return lissajous(n=n, a=a, b=b, margin=margin)


def path_length(pts: list[Point]) -> float:
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
              for i in range(len(pts) - 1))
