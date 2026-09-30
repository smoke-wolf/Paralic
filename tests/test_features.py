import math

import numpy as np
import pytest

from paralic import landmarks as L
from paralic.features import NUM_FEATURES, extract_features, head_pose_from_matrix, overlay_points


def synthetic_face(iris_shift=(0.0, 0.0), aperture=10.0):
    """478 landmarks with two horizontal eyes 30 px wide."""
    pts = np.zeros((478, 3))
    pts[:, 0] = 320
    pts[:, 1] = 240

    def eye(outer, inner, upper, lower, iris, cx):
        left_x, right_x = cx - 15, cx + 15
        a, b = (outer, inner) if outer == L.RIGHT_EYE_OUTER else (inner, outer)
        pts[a, :2] = (left_x, 200)
        pts[b, :2] = (right_x, 200)
        for i in upper:
            pts[i, :2] = (cx, 200 - aperture / 2)
        for i in lower:
            pts[i, :2] = (cx, 200 + aperture / 2)
        centre = np.array([cx + iris_shift[0] * 30, 200 + iris_shift[1] * 30])
        pts[iris[0], :2] = centre
        for k, (dx, dy) in enumerate([(5, 0), (0, -5), (-5, 0), (0, 5)]):
            pts[iris[1 + k], :2] = centre + (dx, dy)

    eye(L.RIGHT_EYE_OUTER, L.RIGHT_EYE_INNER, L.RIGHT_UPPER_LID, L.RIGHT_LOWER_LID, L.RIGHT_IRIS, 280)
    eye(L.LEFT_EYE_OUTER, L.LEFT_EYE_INNER, L.LEFT_UPPER_LID, L.LEFT_LOWER_LID, L.LEFT_IRIS, 360)
    return pts


def test_iris_offsets_follow_iris():
    centred = extract_features(synthetic_face(), (640, 480))
    right = extract_features(synthetic_face((0.1, 0.0)), (640, 480))
    down = extract_features(synthetic_face((0.0, 0.05)), (640, 480))
    assert centred.vector.shape == (NUM_FEATURES,)
    assert abs(centred.right_eye.dx) < 1e-9 and abs(centred.left_eye.dx) < 1e-9
    assert right.right_eye.dx == pytest.approx(0.1) and right.left_eye.dx == pytest.approx(0.1)
    assert down.right_eye.dy == pytest.approx(0.05)


def test_closure_rises_when_lids_close():
    open_eye = extract_features(synthetic_face(aperture=9.0), (640, 480))
    closed = extract_features(synthetic_face(aperture=1.0), (640, 480))
    assert open_eye.aperture == pytest.approx(0.3)
    assert closed.closure > open_eye.closure + 0.5


def test_blendshapes_are_used_for_closure():
    bs_open = {"eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05}
    bs_closed = {"eyeBlinkLeft": 0.95, "eyeBlinkRight": 0.95}
    a = extract_features(synthetic_face(), (640, 480), bs_open)
    b = extract_features(synthetic_face(), (640, 480), bs_closed)
    assert b.closure > a.closure + 0.4


def test_head_pose_from_rotation():
    assert head_pose_from_matrix(np.eye(4))[:3] == (0.0, 0.0, 0.0)
    yaw = math.radians(20)
    m = np.eye(4)
    m[:3, :3] = [[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]]
    m[:3, 3] = [1.0, 2.0, -60.0]
    y, p, r, t = head_pose_from_matrix(m)
    assert y == pytest.approx(yaw) and p == pytest.approx(0.0) and r == pytest.approx(0.0)
    assert t.tolist() == [1.0, 2.0, -60.0]


def test_overlay_points_are_normalised():
    pts = synthetic_face()
    feats = extract_features(pts, (640, 480))
    ov = overlay_points(pts, (640, 480), feats)
    assert len(ov["r"]) == len(L.RIGHT_EYE_CONTOUR)
    assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in ov["r"] + ov["l"])


def test_requires_iris_landmarks():
    with pytest.raises(ValueError):
        extract_features(np.zeros((468, 3)), (640, 480))
