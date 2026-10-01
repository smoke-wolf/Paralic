"""Procedural pursuit trails (paralic/trails.py)."""

import math

import pytest

from paralic import trails


ALL = list(trails.KINDS)


@pytest.mark.parametrize("kind", ALL)
def test_within_bounds(kind):
    pts = trails.trail(kind, n=200, seed=1, margin=0.1)
    assert len(pts) >= 50
    for x, y in pts:
        assert -1e-9 <= x <= 1 + 1e-9 and -1e-9 <= y <= 1 + 1e-9
        # inside the margin border
        assert 0.1 - 1e-6 <= x <= 0.9 + 1e-6
        assert 0.1 - 1e-6 <= y <= 0.9 + 1e-6


@pytest.mark.parametrize("kind", ALL)
def test_deterministic(kind):
    a = trails.trail(kind, n=200, seed=7)
    b = trails.trail(kind, n=200, seed=7)
    assert a == b


def test_seed_varies_spline():
    a = trails.trail("spline", n=200, seed=1)
    b = trails.trail("spline", n=200, seed=2)
    assert a != b


@pytest.mark.parametrize("kind", ALL)
def test_smooth_no_big_jumps(kind):
    pts = trails.trail(kind, n=400, seed=3)
    steps = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]
    # No single step should teleport across the screen.
    assert max(steps) < 0.2


def test_snake_covers_rows():
    pts = trails.snake(cols=5, rows=4, margin=0.1)
    ys = sorted({round(y, 3) for _, y in pts})
    assert ys[0] == pytest.approx(0.1, abs=1e-6)
    assert ys[-1] == pytest.approx(0.9, abs=1e-6)
    # path spans the full horizontal range too
    xs = [x for x, _ in pts]
    assert min(xs) == pytest.approx(0.1, abs=1e-6)
    assert max(xs) == pytest.approx(0.9, abs=1e-6)


def test_lissajous_length_reasonable():
    pts = trails.lissajous(n=300)
    assert 1.0 < trails.path_length(pts) < 20.0
