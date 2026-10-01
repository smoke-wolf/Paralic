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
    def __init__(self, seed: int = 0, noise: float = 1.0, eye_noise: tuple[float, float] = (1.0, 1.0),
                 wander: tuple[float, float] = (0.0, 0.0)):
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        # Extra noise of the (right, left) eye's features.
        self.eye_noise = eye_noise
        # A squint that comes and goes: each fixation, the (right, left) eye
        # points off by a random amount with this standard deviation.
        self.wander = wander
        self.seed = seed
        # Person-specific constants.
        self.k_x = 0.42 + 0.05 * self.rng.standard_normal()
        self.k_y = 0.30 + 0.04 * self.rng.standard_normal()
        self.open0 = 0.30 + 0.03 * self.rng.standard_normal()
        self.offset = 0.02 * self.rng.standard_normal(2)

    def features(self, sx: float, sy: float, head: Head, closed: str | None = None) -> np.ndarray:
        """Feature vector for a gaze at screen pixel (sx, sy).

        ``closed`` ("left" / "right") closes that eye, as in a wink: its iris
        and lid features then say little about where the person looks.
        """
        tx = (sx - SCREEN_W / 2) / PX_PER_CM
        ty = sy / PX_PER_CM
        theta = math.atan2(tx - head.x, head.dist)            # gaze yaw in world
        phi = math.atan2(ty - head.y, head.dist)              # gaze pitch in world (+ = down)
        ey = theta - head.yaw                                  # eye-in-head angles
        ep = phi + head.pitch
        n = self.noise
        nr, nl = n * self.eye_noise[0], n * self.eye_noise[1]
        r = self.rng.standard_normal
        # Per-eye rotation offsets (right yaw, right pitch, left yaw, left pitch), radians.
        w = np.zeros(4)
        if any(self.wander):
            fixation = np.random.default_rng([self.seed, int(round(sx)), int(round(sy))])
            w = fixation.standard_normal(4) * np.repeat(self.wander, 2)

        def eye(yaw: float, pitch: float) -> tuple:
            dx = self.k_x * math.sin(yaw) + self.offset[0]
            dy = self.k_y * math.sin(pitch) + 0.08 * math.sin(pitch) ** 2 + self.offset[1]
            aperture = self.open0 - 0.22 * max(0.0, math.sin(pitch)) + 0.06 * max(0.0, -math.sin(pitch))
            neg = max(0.0, min(1.0, -yaw / 0.5))
            pos = max(0.0, min(1.0, yaw / 0.5))
            up = max(0.0, min(1.0, -pitch / 0.4))
            down = max(0.0, min(1.0, pitch / 0.4))
            # Richer mesh signals: iris height between the lids (down -> lower),
            # and squint / wide blendshapes that track looking down / up.
            vlid = min(0.98, max(0.02, 0.5 + 0.9 * math.sin(pitch)))
            squint = max(0.0, min(1.0, 0.45 * math.sin(pitch)))
            wide = max(0.0, min(1.0, -0.45 * math.sin(pitch)))
            return dx, dy, aperture, neg, pos, up, down, vlid, squint, wide

        R = eye(ey + w[0], ep + w[1])
        Lf = eye(ey + w[2], ep + w[3])
        v = np.array([
            R[0] + 0.012 * nr * r(), R[1] + 0.012 * nr * r(),
            Lf[0] + 0.012 * nl * r(), Lf[1] + 0.012 * nl * r(),
            R[2] + 0.01 * nr * r(), Lf[2] + 0.01 * nl * r(),
            # left eye: look in / out / up / down
            Lf[3] + 0.03 * nl * r(), Lf[4] + 0.03 * nl * r(), Lf[5] + 0.03 * nl * r(), Lf[6] + 0.03 * nl * r(),
            # right eye: looking "in" for the right eye is the opposite direction
            R[4] + 0.03 * nr * r(), R[3] + 0.03 * nr * r(), R[5] + 0.03 * nr * r(), R[6] + 0.03 * nr * r(),
            head.yaw + 0.01 * n * r(), head.pitch + 0.01 * n * r(), head.roll + 0.01 * n * r(),
            head.x + 0.2 * n * r(), -head.y + 0.2 * n * r(), -head.dist + 0.3 * n * r(),
            # columns 20..27: vlid (r, l), tilt (r, l ~ head roll), squint, wide
            R[7] + 0.02 * nr * r(), Lf[7] + 0.02 * nl * r(),
            head.roll + 0.02 * nr * r(), head.roll + 0.02 * nl * r(),
            R[8] + 0.03 * nr * r(), Lf[8] + 0.03 * nl * r(),
            R[9] + 0.03 * nr * r(), Lf[9] + 0.03 * nl * r(),
        ])
        if closed is not None:
            # A shut eye: the "iris" sits low and central, the lid is closed,
            # the eye-movement blendshapes fade, and the richer per-eye signals
            # (vertical iris, squint, wide) go to neutral.
            cols = {"right": (0, 1, 4, (10, 11, 12, 13), 20, (24, 26)),
                    "left": (2, 3, 5, (6, 7, 8, 9), 21, (25, 27))}[closed]
            v[cols[0]] = 0.3 * v[cols[0]] + 0.01 * r()
            v[cols[1]] = 0.12 + 0.01 * r()
            v[cols[2]] = 0.04 + 0.005 * r()
            v[list(cols[3])] *= 0.2
            v[cols[4]] = 0.5 + 0.01 * r()
            v[list(cols[5])] *= 0.2
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
