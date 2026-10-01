"""Evaluate glasses and glare detection on public test faces.

    python tools/glasses_eval.py            # writes docs/glasses.md

Glasses are painted onto three of MediaPipe's public test faces (the test
portrait, a business portrait and a stylizer test face, downloaded once into
tests/.cache): thick and thin full-rim frames, half-rim and rimless frames
(no lower rim), small lenses, and thin light-coloured metal frames; and, as
things that are not glasses, frown lines across the nose. Each picture is
then changed like a webcam frame - dimmer or brighter, tilted, smaller (30
to 63 pixels between the eye centres), noisy, JPEG-compressed - and run
through the real MediaPipe face mesh. Reflections are painted onto a lens
after the light changed (a lamp's reflection stays white).

These are painted glasses on still photos, not real ones on people (no
shadows of the frame, no room reflected in the lenses, no tinted lenses), so
the numbers say the method works as intended, not how often it is right for
real glasses.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paralic import glasses as G  # noqa: E402
from tests import face_images  # noqa: E402

GAINS = (0.45, 0.7, 1.0, 1.3)
TILTS = (-7.0, 7.0)
SIZES = (0.3, 0.45, 0.62)        # picture scale: about 30, 45 and 63 pixels between the eye centres

FRAMES = {
    "no glasses": None,
    "frown line across the nose": {"frown": 22},
    "deep frown line across the nose": {"frown": 40},
    "thick full-rim frames": {},
    "thin full-rim frames": {"thickness": 0.015, "color": (60, 60, 70)},
    "half-rim frames (no lower rim)": {"style": "half", "thickness": 0.03, "color": (40, 40, 45)},
    "rimless (bridge and arms only)": {"style": "rimless", "thickness": 0.015, "color": (70, 70, 80)},
    "small lenses": {"thickness": 0.025, "half_w": 0.33, "half_h": 0.20},
    "thin light metal frames": {"thickness": 0.015, "color": (190, 190, 195)},
}
NOT_GLASSES = ("no glasses", "frown line across the nose", "deep frown line across the nose")

REFLECTIONS = {
    "none": [],
    "on the right lens, over the eye": [("right", {})],
    "small, on the left eye": [("left", {"size": 0.12, "at": (0.0, -0.02)})],
    "on the right lens, beside the eye": [("right", {"size": 0.13, "at": (-0.2, 0.12)})],
    "on both lenses": [("right", {"size": 0.18}), ("left", {"size": 0.18})],
    "dimmer (215 of 255), on the left lens": [("left", {"size": 0.2, "at": (0.05, -0.05), "level": 215})],
}


def landmarker():
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks.python import BaseOptions, vision

    model = ROOT / "models" / "face_landmarker.task"
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model.read_bytes()), running_mode=vision.RunningMode.IMAGE))


def detect(lm, bgr):
    import cv2
    import mediapipe as mp

    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    h, w = rgb.shape[:2]
    if not res.face_landmarks:
        return rgb, None
    return rgb, np.array([[p.x * w, p.y * h, p.z * w] for p in res.face_landmarks[0]])


def faces() -> dict:
    import cv2

    imgs = {"the test portrait": face_images.portrait(1024),
            "face B": face_images.face_image("business-person.png"),
            "face C": face_images.face_image("face_stylizer_raw_face_demo.png")}
    if any(i is None for i in imgs.values()):
        raise SystemExit("could not download the test images")
    imgs["face C"] = cv2.resize(imgs["face C"], (768, 768), interpolation=cv2.INTER_CUBIC)
    return imgs


def webcam(img, gain, tilt, size, rng):
    """The picture as a webcam might deliver it: other light, tilted, smaller, noisy, compressed."""
    import cv2

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), tilt, 1.0)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    out = np.clip(out * gain + rng.normal(0, 2.0, out.shape), 0, 255).astype(np.uint8)
    out = cv2.resize(out, (int(w * size), int(h * size)), interpolation=cv2.INTER_AREA)
    return cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])[1], cv2.IMREAD_COLOR)


def frown(img, pts, depth):
    """A soft dark line across the root of the nose, as when frowning."""
    mid, ex, ey, d = face_images.eye_axes(pts)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float32)
    u = ((xx - mid[0]) * ex[0] + (yy - mid[1]) * ex[1]) / d
    v = ((xx - mid[0]) * ey[0] + (yy - mid[1]) * ey[1]) / d
    a = np.exp(-((v + 0.18) / 0.02) ** 2) * np.clip(1 - (np.abs(u) / 0.16) ** 4, 0, 1)
    return np.clip(img.astype(np.float32) - depth * a[..., None], 0, 255).astype(np.uint8)


def painted(img, pts, kw):
    if kw is None:
        return img
    if "frown" in kw:
        return frown(img, pts, kw["frown"])
    return face_images.paint_glasses(img, pts, **kw)


def glasses_results(lm, imgs, rng) -> dict:
    out = {}
    for kind, kw in FRAMES.items():
        rows = []
        for name, img in imgs.items():
            _, pts = detect(lm, img)
            picture = painted(img, pts, kw)
            for gain in GAINS:
                for tilt in TILTS:
                    for size in SIZES:
                        rgb, p = detect(lm, webcam(picture, gain, tilt, size, rng))
                        m = None if p is None else G.measure(rgb, p)
                        if m is not None:
                            rows.append((name, m.score))
        out[kind] = rows
    return out


def glare_results(lm, imgs, rng) -> dict:
    out = {}
    for kind, spots in REFLECTIONS.items():
        rows = []
        for name, img in imgs.items():
            _, pts = detect(lm, img)
            worn = face_images.paint_glasses(img, pts)
            for gain in GAINS[:3]:
                picture = webcam(worn, gain, 0.0, 0.62, rng)
                _, p = detect(lm, picture)
                for eye, kw in spots:
                    picture = face_images.paint_glare(picture, p, eye, **kw)
                rgb, p = detect(lm, picture)
                m = None if p is None else G.measure(rgb, p)
                if m is not None:
                    rows.append((name, m.glare, m.skin))
        out[kind] = rows
    return out


def report(glasses: dict, glare: dict, seconds: float) -> str:
    about = " ".join(p.replace("\n", " ") for p in __doc__.split("\n\n")[2:])
    n_variants = len(GAINS) * len(TILTS) * len(SIZES)
    L = ["# Glasses and glare: detection on painted glasses", "",
         "Produced by `python tools/glasses_eval.py`. " + about, "",
         "## Glasses", "",
         f"Each face in {n_variants} webcam-like versions ({len(GAINS)} brightnesses, {len(TILTS)} tilts, "
         f"{len(SIZES)} sizes). The score is the nose-bridge edge in units of the skin's own edges, with up to "
         "half again for rims below both eyes (see `paralic/glasses.py`); one frame is judged on its own here, "
         f"while Paralic smooths the score over about half a second: glasses above {G.GLASSES_ON:g}, none "
         f"below {G.GLASSES_OFF:g}.", "",
         "| Picture | Seen as glasses (portrait · B · C) | Score: min / median / max |",
         "| --- | --- | --- |"]
    for kind, rows in glasses.items():
        scores = np.array([s for _, s in rows])
        seen = " · ".join(f"{sum(s >= G.GLASSES_ON for n, s in rows if n == face)}/"
                          f"{sum(1 for n, _ in rows if n == face)}"
                          for face in ("the test portrait", "face B", "face C"))
        label = kind + (" *(not glasses)*" if kind in NOT_GLASSES else "")
        L.append(f"| {label} | {seen} | {scores.min():.2f} / {np.median(scores):.2f} / {scores.max():.2f} |")
    L += ["", "Thin frames the colour of light skin hardly stand out from it and are not seen (they would be "
          "calibrated as \"without glasses\", as before Paralic noticed glasses).", "",
          "## Glare", "",
          f"Painted glasses, {len(GAINS) - 1} brightnesses per face, a reflection painted on top. Largest bright, "
          "nearly colourless blob around each eye, in iris areas, for single frames; Paralic smooths it and "
          f"flags glare above {G.GLARE_ON:g} (below {G.GLARE_OFF:g} it goes again). Faces lit almost white "
          f"(skin brighter than {G.GLARE_MAX_SKIN:.0f} of 255) are not judged.", "",
          "| Reflection | Right eye flagged | Left eye flagged | Blob, right / left (median) |",
          "| --- | --- | --- | --- |"]
    for kind, rows in glare.items():
        judged = [r for r in rows if r[2] <= G.GLARE_MAX_SKIN]
        right = sum(g["right"] >= G.GLARE_ON for _, g, _ in judged)
        left = sum(g["left"] >= G.GLARE_ON for _, g, _ in judged)
        med = [np.median([g[e] for _, g, _ in judged]) if judged else float("nan") for e in ("right", "left")]
        skipped = len(rows) - len(judged)
        note = f" ({skipped} lit almost white, not judged)" if skipped else ""
        L.append(f"| {kind}{note} | {right}/{len(judged)} | {left}/{len(judged)} | {med[0]:.2f} / {med[1]:.2f} |")
    L += ["", f"_Run time {seconds:.0f} s._"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "docs" / "glasses.md"))
    args = ap.parse_args(argv)
    started = time.perf_counter()
    imgs = faces()
    rng = np.random.default_rng(1)
    with landmarker() as lm:
        glasses = glasses_results(lm, imgs, rng)
        glare = glare_results(lm, imgs, rng)
    text = report(glasses, glare, time.perf_counter() - started)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
