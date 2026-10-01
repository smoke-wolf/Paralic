"""Synthetic eye images for integration tests with the real MediaPipe networks.

Starting from a public-domain portrait (downloaded on first use), we paint the
eyes closed or move the irises, to check that blink and gaze features react,
and paint glasses and reflections on their lenses.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from paralic import landmarks as L

CACHE = Path(__file__).resolve().parent / ".cache"
PORTRAIT_URL = "https://storage.googleapis.com/mediapipe-assets/portrait.jpg"


def portrait(width: int = 640) -> np.ndarray | None:
    """BGR portrait image, or None if it cannot be downloaded."""
    import cv2

    path = CACHE / "portrait.jpg"
    if not path.exists():
        try:
            CACHE.mkdir(exist_ok=True)
            urllib.request.urlretrieve(PORTRAIT_URL, path)
        except Exception:
            return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


ASSETS_URL = "https://storage.googleapis.com/mediapipe-assets/"


def face_image(name: str) -> np.ndarray | None:
    """Another MediaPipe test image with a face (BGR, cached), or None offline."""
    import cv2

    path = CACHE / name
    if not path.exists():
        try:
            CACHE.mkdir(exist_ok=True)
            urllib.request.urlretrieve(ASSETS_URL + name, path)
        except Exception:
            return None
    return cv2.imread(str(path))


def close_eyes(img: np.ndarray, pts: np.ndarray,
               contours: tuple = (L.RIGHT_EYE_CONTOUR, L.LEFT_EYE_CONTOUR)) -> np.ndarray:
    """Paint the eyes in ``contours`` closed (both by default; pass one contour for a wink)."""
    import cv2

    out = img.copy()
    for contour in contours:
        poly = pts[list(contour)]
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        ew = x1 - x0
        band = img[int(y0 - 0.35 * ew):int(y0 - 0.12 * ew), int(x0 + 0.2 * ew):int(x1 - 0.2 * ew)].reshape(-1, 3)
        skin = np.median(band, axis=0)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 255)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        fill = np.empty_like(out)
        fill[:] = skin.astype(np.uint8)
        yy = np.arange(img.shape[0])[:, None].astype(float)
        shade = np.clip((yy - y0) / max(y1 - y0, 1), 0, 1)
        fill = (fill * (1 - 0.25 * shade[..., None])).astype(np.uint8)
        m = cv2.GaussianBlur(mask, (5, 5), 0).astype(float)[..., None] / 255.0
        out = (out * (1 - m) + fill * m).astype(np.uint8)
        lower = np.round(pts[list(contour[:9])] + [0, -0.02 * ew]).astype(np.int32)
        cv2.polylines(out, [lower], False, (35, 28, 28), 2, cv2.LINE_AA)
    return out


def shift_iris(img: np.ndarray, pts: np.ndarray, dx_frac: float, dy_frac: float = 0.0) -> np.ndarray:
    """Move both irises by a fraction of the eye width (a synthetic gaze change)."""
    import cv2

    out = img.copy()
    for contour, iris in ((L.RIGHT_EYE_CONTOUR, L.RIGHT_IRIS), (L.LEFT_EYE_CONTOUR, L.LEFT_IRIS)):
        poly = np.round(pts[list(contour)]).astype(np.int32)
        eye_mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(eye_mask, [poly], 255)
        ic = pts[iris[0]]
        r = np.mean(np.linalg.norm(pts[list(iris[1:])] - ic, axis=1))
        ew = np.ptp(pts[list(contour)][:, 0])
        inside = img[eye_mask > 0].reshape(-1, 3)
        bright = inside[inside.sum(axis=1) > np.percentile(inside.sum(axis=1), 80)]
        sclera = np.median(bright, axis=0) if len(bright) else np.array([200, 200, 200])
        iris_mask = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(iris_mask, tuple(int(v) for v in np.round(ic)), int(r * 1.15) + 1, 255, -1)
        iris_mask &= eye_mask
        base = out.copy()
        base[iris_mask > 0] = sclera.astype(np.uint8)
        sx, sy = dx_frac * ew, dy_frac * ew
        shifted = cv2.warpAffine(img, np.float32([[1, 0, sx], [0, 1, sy]]), (img.shape[1], img.shape[0]),
                                 borderMode=cv2.BORDER_REFLECT)
        disk = np.zeros(img.shape[:2], np.uint8)
        cv2.circle(disk, tuple(int(v) for v in np.round(ic + [sx, sy])), int(r * 1.05), 255, -1)
        disk &= eye_mask
        base[disk > 0] = shifted[disk > 0]
        out = base
    return out


def eye_axes(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """The point between the eye centres, unit vectors along the eyes (towards
    the person's left eye) and downwards, and the distance between the eyes."""
    P = np.asarray(pts, float)[:, :2]
    right = (P[L.RIGHT_EYE_OUTER] + P[L.RIGHT_EYE_INNER]) / 2
    left = (P[L.LEFT_EYE_INNER] + P[L.LEFT_EYE_OUTER]) / 2
    d = float(np.linalg.norm(left - right))
    ex = (left - right) / d
    return (left + right) / 2, ex, np.array([-ex[1], ex[0]]), d


def paint_glasses(img: np.ndarray, pts: np.ndarray, style: str = "full", color: tuple = (28, 28, 32),
                  thickness: float = 0.04, half_w: float = 0.40, half_h: float = 0.26) -> np.ndarray:
    """Paint a pair of glasses: a rim around each eye (``style`` "full"; only the
    upper half for "half"; none for "rimless"), a bridge over the nose and the
    arms towards the ears. Sizes are in eye distances, ``color`` is BGR."""
    import cv2

    mid, ex, ey, d = eye_axes(pts)
    out = img.copy()
    width = max(1, int(round(thickness * d)))

    def draw(xy, closed=False):
        P = np.array([mid + d * (x * ex + y * ey) for x, y in xy])
        cv2.polylines(out, [np.round(P * 4).astype(np.int32)], closed, color, width, cv2.LINE_AA, shift=2)

    t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    for side in (-1, 1):                       # the person's right eye (image left), then the left
        cx = side * 0.565                       # lenses sit a little outside the eyes
        xs = cx + half_w * np.sign(c) * np.abs(c) ** (2 / 3)
        ys = 0.03 + half_h * np.sign(s) * np.abs(s) ** (2 / 3)
        if style == "full":
            draw(zip(xs, ys), closed=True)
        elif style == "half":
            draw([(x, y) for x, y in zip(xs, ys) if y < 0.03])
        edge = side * (0.565 + half_w)
        draw([(edge, -0.10), (edge + side * 0.35, -0.08)])
    inner = 0.565 - half_w
    xs = np.linspace(-inner - 0.03, inner + 0.03, 12)
    draw(zip(xs, -0.08 - 0.04 * (1 - (xs / inner) ** 2)))
    return out


def paint_glare(img: np.ndarray, pts: np.ndarray, eye: str = "right", size: float = 0.18,
                at: tuple = (0.0, -0.05), level: float = 250.0) -> np.ndarray:
    """Paint a soft, almost white reflection on one lens. ``eye``: the person's
    eye; ``at``: the reflection's centre relative to that eye's centre (x
    towards the person's left eye, y down) and ``size`` its radius, in eye
    distances."""
    mid, ex, ey, d = eye_axes(pts)
    centre = mid + d * ((at[0] + (0.5 if eye == "left" else -0.5)) * ex + at[1] * ey)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
    u = ((xx - centre[0]) * ex[0] + (yy - centre[1]) * ex[1]) / (1.3 * size * d)
    v = ((xx - centre[0]) * ey[0] + (yy - centre[1]) * ey[1]) / (size * d)
    a = np.clip(1.6 * np.exp(-1.5 * (u * u + v * v) ** 2), 0.0, 1.0)[..., None]
    return np.clip(img.astype(np.float32) * (1 - a) + level * a, 0, 255).astype(np.uint8)
