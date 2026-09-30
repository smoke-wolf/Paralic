"""Create a fake-webcam video for the slow browser test.

    python tests/make_fake_video.py /tmp/face.y4m
    PARALIC_FAKE_VIDEO=/tmp/face.y4m python -m pytest --runslow tests/test_browser.py

The 6-second loop shows a (public-domain) portrait with open eyes and one
double blink (the eyes are painted closed for a few frames).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(out: str) -> None:
    import cv2

    from paralic.tracker import FaceTracker
    from tests import face_images

    img = face_images.portrait(640)
    if img is None:
        raise SystemExit("could not download the test portrait")
    tracker = FaceTracker((ROOT / "models" / "face_landmarker.task").read_bytes())
    obs = tracker.process(np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), 0)
    tracker.close()
    pts = obs.points_px[:, :2]
    closed = face_images.close_eyes(img, pts)
    cy = int(pts[:, 1].mean())
    y0 = max(0, min(img.shape[0] - 480, cy - 240))
    frames = {"o": img[y0:y0 + 480], "c": closed[y0:y0 + 480]}
    seq = "o" * 120 + "c" * 4 + "o" * 5 + "c" * 4 + "o" * 47
    rng = np.random.default_rng(0)
    with open(out, "wb") as fh:
        fh.write(b"YUV4MPEG2 W640 H480 F30:1 Ip A1:1 C420jpeg\n")
        for key in seq:
            f = frames[key].astype(np.int16) + rng.normal(0, 1.5, frames[key].shape)
            fh.write(b"FRAME\n")
            fh.write(cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2YUV_I420).tobytes())
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "face.y4m")
