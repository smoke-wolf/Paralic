"""Synthetic eye images for integration tests with the real MediaPipe networks.

Starting from a public-domain portrait (downloaded on first use), we paint the
eyes closed or move the irises, to check that blink and gaze features react.
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
