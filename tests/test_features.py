import math

import numpy as np
import pytest

from paralic import landmarks as L
from paralic.features import (FEATURE_NAMES, FEATURE_VERSION, NUM_FEATURES, extract_features,
                             head_pose_from_matrix, mesh_overlay, overlay_points)


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


def test_feature_vector_is_the_full_mesh_set():
    # The enriched contract: 28 features, version 2, with the new mesh signals.
    assert NUM_FEATURES == 28 and FEATURE_VERSION == 2
    for name in ("r_vlid", "l_vlid", "r_tilt", "l_tilt", "bs_squint_r", "bs_wide_l"):
        assert name in FEATURE_NAMES
    v = extract_features(synthetic_face(), (640, 480)).vector
    assert v.shape == (NUM_FEATURES,)


def test_vertical_iris_and_eye_tilt():
    centred = extract_features(synthetic_face(), (640, 480))
    down = extract_features(synthetic_face((0.0, 0.08)), (640, 480))
    # A centred iris sits mid-way between the lids; looking down lowers it.
    assert centred.right_eye.vlid == pytest.approx(0.5, abs=0.05)
    assert down.right_eye.vlid > centred.right_eye.vlid + 0.1
    # The synthetic eyes are horizontal, so the tilt signal is ~0.
    assert abs(centred.right_eye.tilt) < 1e-6 and abs(centred.left_eye.tilt) < 1e-6


def test_extra_eye_blendshapes_enter_the_vector():
    bs = {"eyeSquintRight": 0.7, "eyeWideLeft": 0.6}
    v = extract_features(synthetic_face(), (640, 480), bs).vector
    assert v[FEATURE_NAMES.index("bs_squint_r")] == pytest.approx(0.7)
    assert v[FEATURE_NAMES.index("bs_wide_l")] == pytest.approx(0.6)


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


def test_mesh_overlay_is_the_full_mask():
    pts = synthetic_face()
    mesh = mesh_overlay(pts, (640, 480))
    # Every one of the 478 landmarks, normalised into the frame.
    assert len(mesh["pts"]) == L.NUM_LANDMARKS_WITH_IRIS
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in mesh["pts"])
    # Wireframe loops reference valid landmark indices.
    assert len(mesh["lines"]) >= 4
    for line in mesh["lines"]:
        assert len(line) >= 3
        assert all(0 <= i < L.NUM_LANDMARKS_WITH_IRIS for i in line)
    assert len(mesh["iris"]) == 2
