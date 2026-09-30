"""A simple geometric simulation of a person looking at a screen.

Used by the tests to generate realistic-looking feature vectors (with noise,
head movement and eyelid effects) for known on-screen gaze targets, so the
calibration + GazeNet pipeline can be checked without a webcam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from paralic.features import NUM_FEATURES

SCREEN_W, SCREEN_H = 1920, 1080
PX_PER_CM = 55.6          # 15.6" 1080p laptop panel


@dataclass
class Head:
    x: float = 0.0        # cm, right of camera
    y: float = 9.0        # cm, below camera (eyes roughly level with upper screen)
    dist: float = 60.0    # cm from the screen plane
    yaw: float = 0.0      # radians, + = turned to the (camera's) right
    pitch: float = 0.0    # radians, + = tilted up
    roll: float = 0.0


class VirtualUser:
    def __init__(self, seed: int = 0, noise: float = 1.0):
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        # Person-specific constants.
        self.k_x = 0.42 + 0.05 * self.rng.standard_normal()
        self.k_y = 0.30 + 0.04 * self.rng.standard_normal()
        self.open0 = 0.30 + 0.03 * self.rng.standard_normal()
        self.offset = 0.02 * self.rng.standard_normal(2)

    def features(self, sx: float, sy: float, head: Head) -> np.ndarray:
        """Feature vector for a gaze at screen pixel (sx, sy)."""
        tx = (sx - SCREEN_W / 2) / PX_PER_CM
        ty = sy / PX_PER_CM
        theta = math.atan2(tx - head.x, head.dist)            # gaze yaw in world
        phi = math.atan2(ty - head.y, head.dist)              # gaze pitch in world (+ = down)
        ey = theta - head.yaw                                  # eye-in-head angles
        ep = phi + head.pitch
        n = self.noise
        r = self.rng.standard_normal
        dx = self.k_x * math.sin(ey) + self.offset[0]
        dy = self.k_y * math.sin(ep) + 0.08 * math.sin(ep) ** 2 + self.offset[1]
        aperture = self.open0 - 0.22 * max(0.0, math.sin(ep)) + 0.06 * max(0.0, -math.sin(ep))
        look_in_l = max(0.0, min(1.0, -ey / 0.5))
        look_out_l = max(0.0, min(1.0, ey / 0.5))
        look_in_r = look_out_l
        look_out_r = look_in_l
        look_up = max(0.0, min(1.0, -ep / 0.4))
        look_down = max(0.0, min(1.0, ep / 0.4))
        v = np.array([
            dx + 0.012 * n * r(), dy + 0.012 * n * r(),
            dx + 0.012 * n * r(), dy + 0.012 * n * r(),
            aperture + 0.01 * n * r(), aperture + 0.01 * n * r(),
            look_in_l + 0.03 * n * r(), look_out_l + 0.03 * n * r(), look_up + 0.03 * n * r(), look_down + 0.03 * n * r(),
            look_in_r + 0.03 * n * r(), look_out_r + 0.03 * n * r(), look_up + 0.03 * n * r(), look_down + 0.03 * n * r(),
            head.yaw + 0.01 * n * r(), head.pitch + 0.01 * n * r(), head.roll + 0.01 * n * r(),
            head.x + 0.2 * n * r(), -head.y + 0.2 * n * r(), -head.dist + 0.3 * n * r(),
        ])
        assert v.shape == (NUM_FEATURES,)
        return v


def calibration_points(n_cols: int = 4, n_rows: int = 3, margin: float = 0.07) -> list[tuple[float, float]]:
    xs = np.linspace(margin, 1 - margin, n_cols) * SCREEN_W
    ys = np.linspace(margin, 1 - margin, n_rows) * SCREEN_H
    pts = [(float(x), float(y)) for y in ys for x in xs]
    pts.append((SCREEN_W / 2, SCREEN_H / 2))
    return pts


def jitter_head(rng: np.random.Generator, base: Head, amount: float = 1.0) -> Head:
    return Head(
        x=base.x + amount * 1.0 * rng.standard_normal(),
        y=base.y + amount * 0.8 * rng.standard_normal(),
        dist=base.dist + amount * 1.5 * rng.standard_normal(),
        yaw=base.yaw + amount * 0.02 * rng.standard_normal(),
        pitch=base.pitch + amount * 0.02 * rng.standard_normal(),
        roll=base.roll + amount * 0.01 * rng.standard_normal(),
    )


def simulate_calibration(user: VirtualUser, frames_per_point: int = 28, head_frames: int = 150,
                         head_motion: bool = True, seed: int = 1):
    """Return (X, Y, groups) like the real calibration produces."""
    rng = np.random.default_rng(seed)
    base = Head()
    X, Y, G = [], [], []
    for i, (sx, sy) in enumerate(calibration_points()):
        for _ in range(frames_per_point):
            h = jitter_head(rng, base, 0.3)
            X.append(user.features(sx, sy, h))
            Y.append((sx, sy))
            G.append(f"cal:{i}")
    if head_motion:
        for k in range(head_frames):
            ph = 2 * math.pi * k / head_frames
            h = Head(x=base.x + 3.0 * math.sin(ph), y=base.y + 2.0 * math.sin(2 * ph),
                     dist=base.dist + 2.0 * math.cos(ph),
                     yaw=0.12 * math.sin(ph), pitch=0.09 * math.sin(2 * ph + 0.5), roll=0.03 * math.sin(ph))
            X.append(user.features(SCREEN_W / 2, SCREEN_H / 2, h))
            Y.append((SCREEN_W / 2, SCREEN_H / 2))
            G.append("head:0")
    return np.array(X), np.array(Y, float), np.array(G)
