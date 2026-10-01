"""Several faces in view: follow the person Paralic is set up for.

FaceLandmarker reports every face it sees (up to a few). Only one person
controls the cursor; everyone else's blinks and head movements must change
nothing. :class:`FaceSelector` picks that person's face on each frame:

1. **Following.** While the face is being followed, the face that continues it
   (the nearest to where it was a moment ago, about the same size) is theirs,
   however many other faces are in view.
2. **Alone.** With nobody else around for a while, a single face is theirs,
   as before (a face print that fails in odd light must never lock anyone out).
3. **Others around.** When the followed face is lost while someone else is, or
   recently was, in view, the person gets their place back only:
   - when their face print says it is them (who it is beats where they sit:
     people swap seats);
   - near where they were (they looked away, or tracking dropped for a moment)
     for ``RETURN_S`` - unless the face print says that face is someone else;
   - or, once ``RETURN_S`` has passed (or at the start), the largest face -
     the person nearest the screen - when there is no face print to ask, or
     ``GIVE_UP_S`` later when it recognises nobody (a new look, say glasses).
   Until then nobody controls the cursor - a bystander does not take over.

Positions are face boxes in image fractions; distances are measured in face
widths, so it works at any distance from the camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

FOLLOW_GAP_S = 0.6        # lost for longer than this: find the face again
FOLLOW_JUMP = 1.0         # the followed face may move this many face widths between frames
RETURN_S = 10.0           # how long a lost face's place is kept for it while others are around
RETURN_NEAR = 1.5         # face widths from where it was that count as "its place"
CROWD_MEMORY_S = 30.0     # "others around": another face was seen this recently
MATCH = 4.5               # face print score (own typical difference = 1) that counts as "them"
NOT_THEM = 6.0            # ...and above which a face in their place is taken to be someone else
CHECK_GAP_S = 0.3         # face prints are compared at most this often
GIVE_UP_S = 3.0           # the face print recognised nobody for this long: go by size after all


@dataclass
class FaceBox:
    cx: float
    cy: float
    w: float
    h: float

    @property
    def size(self) -> float:
        return max(self.w, self.h, 1e-6)

    def bounds(self) -> list[float]:
        return [round(self.cx - self.w / 2, 4), round(self.cy - self.h / 2, 4),
                round(self.cx + self.w / 2, 4), round(self.cy + self.h / 2, 4)]


def face_box(points_px: np.ndarray, image_size: Sequence[int]) -> FaceBox:
    """The face's bounding box in image fractions (0..1)."""
    w, h = float(image_size[0]), float(image_size[1])
    P = np.asarray(points_px, float)[:, :2]
    x0, y0 = P.min(axis=0)
    x1, y1 = P.max(axis=0)
    return FaceBox(cx=(x0 + x1) / 2 / w, cy=(y0 + y1) / 2 / h, w=(x1 - x0) / w, h=(y1 - y0) / h)


def _distance(a: FaceBox, b: FaceBox) -> float:
    """Centre distance in face widths (of ``b``), plus a penalty for a different size."""
    d = float(np.hypot(a.cx - b.cx, a.cy - b.cy)) / b.size
    return d + abs(np.log(a.size / b.size))


class FaceSelector:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Forget whose face was followed (e.g. another person was chosen)."""
        self.box: Optional[FaceBox] = None
        self.seen_at = -np.inf
        self.crowd_at = -np.inf
        self.lost_since: Optional[float] = None
        self._next_check = -np.inf
        self._crowded = False

    @property
    def crowded(self) -> bool:
        return self._crowded

    def select(self, t: float, boxes: list[FaceBox],
               scores: Optional[Callable[[], Optional[list[float]]]] = None) -> Optional[int]:
        """Index of the person's face among ``boxes``, or None.

        ``scores``, when the person has a face print, returns each face's
        score against it (lower = more like them; called only when needed).
        """
        n = len(boxes)
        if n >= 2:
            self.crowd_at = t
        self._crowded = t - self.crowd_at <= CROWD_MEMORY_S
        if n == 0:
            return None
        pick = None
        if self.box is not None and t - self.seen_at <= FOLLOW_GAP_S:
            i = min(range(n), key=lambda k: _distance(boxes[k], self.box))
            if _distance(boxes[i], self.box) <= FOLLOW_JUMP:
                pick = i
        if pick is None:
            pick = self._find(t, boxes, scores)
        if pick is None:
            if self.lost_since is None:
                self.lost_since = t
            return None
        self.box, self.seen_at, self.lost_since = boxes[pick], t, None
        return pick

    def _find(self, t: float, boxes: list[FaceBox], scores) -> Optional[int]:
        largest = max(range(len(boxes)), key=lambda k: boxes[k].w * boxes[k].h)
        if not self._crowded:
            return largest
        # Others are around: the person's own place, or their face print.
        near = None
        if self.box is not None and t - self.seen_at <= RETURN_S:
            i = min(range(len(boxes)), key=lambda k: _distance(boxes[k], self.box))
            if _distance(boxes[i], self.box) <= RETURN_NEAR:
                near = i
        if scores is None:
            if near is not None:
                return near
        else:
            # Who it is beats where they sit (people swap seats); the next look is soon.
            if t < self._next_check:
                return None
            self._next_check = t + CHECK_GAP_S
            values = scores()
            if values:
                best = int(np.argmin(values))
                if values[best] <= MATCH:
                    return best
                if near is not None and values[near] <= NOT_THEM:
                    return near            # unsure, but in their place and not clearly someone else
        if self.box is None or t - self.seen_at > RETURN_S:
            # Nothing (more) to go by: the person nearest the screen - at once
            # without a face print, after a moment when it recognises nobody.
            waited = 0.0 if self.lost_since is None else t - self.lost_since
            if scores is None or waited >= GIVE_UP_S:
                return largest
        return None
