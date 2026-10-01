"""Evaluate the face print on public test faces.

    python tools/faceprint_eval.py          # writes docs/faceprint.md

Five different people from MediaPipe's public test images (downloaded once
into tests/.cache): the test portrait, a business portrait, a stylizer test
face, and the two people in a two-person photo. Each image is altered many
times - turned, scaled, shifted, a slight perspective (a little head turn),
brighter or darker, unevenly lit, blurred, noisy - and run through the real
MediaPipe face mesh, like webcam frames. Prints are made from 12 altered
frames of each enrolled person; other altered frames are then recognised.

These are still photos made to vary, not people over days (no expressions, no
real head turns, no new haircut), so the numbers say the method works as
intended, not how it does on real people over time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paralic import faceprint as FP  # noqa: E402
from tests import face_images  # noqa: E402


def landmarker(num_faces: int = 1):
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks.python import BaseOptions, vision

    model = ROOT / "models" / "face_landmarker.task"
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model.read_bytes()), running_mode=vision.RunningMode.IMAGE,
        num_faces=num_faces))


def detect(lm, bgr):
    import cv2
    import mediapipe as mp

    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    h, w = rgb.shape[:2]
    return rgb, [np.array([[p.x * w, p.y * h, p.z * w] for p in face]) for face in res.face_landmarks]


def people() -> dict:
    import cv2

    imgs = {
        "the test portrait": [face_images.portrait(1024), face_images.face_image("portrait_rotated.jpg")],
        "face B": [face_images.face_image("business-person.png")],
        "face C": [face_images.face_image("face_stylizer_raw_face_demo.png")],
    }
    two = face_images.face_image("man-woman-okay.jpg")
    if any(i is None for v in imgs.values() for i in v) or two is None:
        raise SystemExit("could not download the test images")
    imgs["face C"] = [cv2.resize(imgs["face C"][0], (768, 768), interpolation=cv2.INTER_CUBIC)]
    with landmarker(4) as lm:
        _, faces = detect(lm, two)
    for name, pts in zip(("face D", "face E"), sorted(faces, key=lambda p: p[:, 0].mean())):
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        s = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * 1.6
        h, w = two.shape[:2]
        crop = two[int(max(0, cy - s)):int(min(h, cy + s)), int(max(0, cx - s)):int(min(w, cx + s))]
        imgs[name] = [cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3), interpolation=cv2.INTER_CUBIC)]
    return imgs


def altered(img, rng):
    import cv2

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), rng.uniform(0.8, 1.15))
    M[:, 2] += rng.uniform(-0.05, 0.05, 2) * [w, h]
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32(rng.uniform(-0.06, 0.06, (4, 2)) * [w, h])
    out = cv2.warpPerspective(out, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REFLECT)
    out = out.astype(float) * rng.uniform(0.6, 1.3) + rng.uniform(-30, 30)
    if rng.random() < 0.4:
        out *= np.linspace(rng.uniform(0.7, 1.0), rng.uniform(1.0, 1.3), w)[None, :, None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (5, 5), rng.uniform(0.5, 1.5))
    out = np.clip(out.astype(float) + rng.normal(0, 4, out.shape), 0, 255).astype(np.uint8)
    return cv2.resize(out, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)


def frames(lm, imgs, n, rng):
    out, tries = [], 0
    while len(out) < n and tries < 5 * n:
        tries += 1
        rgb, faces = detect(lm, altered(imgs[tries % len(imgs)], rng))
        if faces:
            s = FP.make_sample(rgb, faces[0])
            if s is not None:
                out.append(s)
    return out


def evaluate(enrol, test, enrolled) -> list[dict]:
    rec = FP.FaceRecognizer({u: enrol[u] for u in enrolled})
    rows = []
    for who, ss in test.items():
        truth = who if who in enrolled else None
        singles = [rec.identify([s]) for s in ss]
        windows = [rec.identify(ss[i:i + 4]) for i in range(0, len(ss) - 3, 2)]
        own = [r["scores"].get(who) for r in singles] if who in enrolled else []
        others = [min(v for u, v in r["scores"].items() if u != who) for r in singles
                  if any(u != who for u in r["scores"])]
        rows.append({"who": who, "enrolled": who in enrolled,
                     "single_right": sum(r["user"] == truth for r in singles), "single_n": len(singles),
                     "single_wrong_person": sum(r["user"] not in (None, who) for r in singles),
                     "window_right": sum(r["user"] == truth for r in windows), "window_n": len(windows),
                     "own": float(np.median(own)) if own else None,
                     "nearest_other": float(np.median(others)) if others else None})
    return rows


def report(results: dict, seconds: float) -> str:
    about = " ".join(p.replace("\n", " ") for p in __doc__.split("\n\n")[2:])
    L = ["# Face print: recognition on public test faces", "",
         "Produced by `python tools/faceprint_eval.py`. " + about, "",
         "Scores are in units of a person's own typical difference between two views (≈ 1 for their own face); "
         f"a face is recognised below {FP.ACCEPT:g} and when {FP.MARGIN:g}× closer to that person than to anyone "
         "else, otherwise the page asks.", ""]
    for title, rows in results.items():
        L += [f"## {title}", "", "| Face | Single frames right | Wrong person | 4-frame windows right | "
              "Own score | Nearest other |", "| --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            label = r["who"] + ("" if r["enrolled"] else " (never enrolled)")
            own = "–" if r["own"] is None else f"{r['own']:.1f}"
            other = "–" if r["nearest_other"] is None else f"{r['nearest_other']:.1f}"
            L.append(f"| {label} | {r['single_right']}/{r['single_n']} | {r['single_wrong_person']} | "
                     f"{r['window_right']}/{r['window_n']} | {own} | {other} |")
        L.append("")
    L.append(f"_Run time {seconds:.0f} s._")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "docs" / "faceprint.md"))
    args = ap.parse_args(argv)
    started = time.perf_counter()
    imgs = people()
    rng = np.random.default_rng(1)
    with landmarker() as lm:
        enrol = {name: frames(lm, v, 12, rng) for name, v in imgs.items()}
        test = {name: frames(lm, v, 10, rng) for name, v in imgs.items()}
    names = list(imgs)
    results = {
        "Five people enrolled": evaluate(enrol, test, names),
        "Three enrolled, two strangers": evaluate(enrol, test, names[:3]),
        "One person enrolled": evaluate(enrol, test, names[:1]),
    }
    text = report(results, time.perf_counter() - started)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
