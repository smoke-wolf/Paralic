"""Several faces in view: only the person Paralic is set up for controls it."""

import numpy as np
import pytest

from paralic import face_select as FS
from paralic.face_select import FaceBox, FaceSelector

DT = 1 / 30
ME = FaceBox(0.35, 0.45, 0.20, 0.26)
OTHER = FaceBox(0.72, 0.50, 0.15, 0.20)


def moved(b, dx=0.0, dy=0.0, scale=1.0):
    return FaceBox(b.cx + dx, b.cy + dy, b.w * scale, b.h * scale)


def run(sel, frames, t0=0.0, scores=None):
    return [sel.select(t0 + k * DT, boxes, scores) for k, boxes in enumerate(frames)]


def test_alone_any_face_is_the_person():
    sel = FaceSelector()
    assert run(sel, [[ME]] * 5) == [0] * 5
    # Back after a while somewhere else: still them (nobody else has been around).
    assert sel.select(3.0, [moved(ME, dx=0.3)]) == 0


def test_the_followed_face_keeps_control_while_others_move():
    sel = FaceSelector()
    picks = run(sel, [[ME]] * 3)
    # Someone comes into view and moves about; the person drifts slowly.
    frames = [[moved(OTHER, dx=-0.01 * k, dy=0.01 * (k % 3)), moved(ME, dx=0.002 * k)] for k in range(60)]
    picks = run(sel, frames, t0=0.2)
    assert picks == [1] * 60
    # The list order changes: still the same face.
    assert sel.select(3.0, [moved(ME, dx=0.12), OTHER]) == 0


def test_a_bystander_does_not_take_over_when_the_person_looks_away():
    sel = FaceSelector()
    run(sel, [[ME, OTHER]] * 10)
    # The person turns away (their face is lost); the other face stays.
    picks = run(sel, [[OTHER]] * int(5 / DT), t0=1.0)
    assert all(p is None for p in picks)
    # They come back where they were: theirs again at once.
    assert sel.select(6.5, [OTHER, moved(ME, dx=0.05)]) == 1


def test_after_a_long_absence_the_nearest_person_takes_over_without_a_face_print():
    sel = FaceSelector()
    run(sel, [[ME, OTHER]] * 10)
    t = 1.0 + FS.RETURN_S + 0.5
    assert sel.select(1.0, [OTHER]) is None
    assert sel.select(t, [OTHER]) == 0


def test_the_face_print_finds_the_person_somewhere_else():
    sel = FaceSelector()
    run(sel, [[ME, OTHER]] * 10)
    # Back in a different place (swapped seats): the face print says which face is theirs.
    elsewhere = [moved(OTHER, dx=-0.4), moved(ME, dx=0.4)]
    asked = []

    def scores():
        asked.append(1)
        return [9.5, 1.2]

    picks = [sel.select(2.0 + k * DT, elsewhere, scores) for k in range(3)]
    assert picks[0] == 1 and picks[1:] == [1, 1] and len(asked) == 1     # then simply followed
    # Strangers only: nobody, until it gives up after a long absence.
    sel2 = FaceSelector()
    run(sel2, [[ME, OTHER]] * 10)
    t = 1.0
    picks = [sel2.select(t + k * 0.1, [OTHER], lambda: [8.0]) for k in range(20)]
    assert all(p is None for p in picks)


def test_at_the_start_with_two_faces():
    # No face print: the larger face (nearest the screen).
    assert FaceSelector().select(0.0, [OTHER, ME]) == 1
    # A face print: the face that matches; if none does, the larger one after a moment.
    sel = FaceSelector()
    assert sel.select(0.0, [ME, OTHER], lambda: [7.0, 2.0]) == 1
    sel = FaceSelector()
    picks = [sel.select(k * 0.1, [ME, OTHER], lambda: [7.0, 8.0]) for k in range(40)]
    assert picks[0] is None and picks[-1] == 0 and FS.GIVE_UP_S <= 0.1 * picks.index(0) + 0.11


def test_face_box_in_image_fractions():
    pts = np.array([[100, 50, 0], [300, 250, 0], [200, 150, 0]], float)
    b = FS.face_box(pts, (400, 400))
    assert b.bounds() == pytest.approx([0.25, 0.125, 0.75, 0.625])
