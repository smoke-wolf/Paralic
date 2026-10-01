"""MediaPipe Face Mesh landmark indices used by the eye tracker.

"Right" and "left" follow MediaPipe's convention: they refer to the *subject's*
eyes, so in an un-mirrored camera image the right eye appears on the left.
"""

# Eye corners.
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263

# A few points along the middle of each eyelid (averaged for stability).
RIGHT_UPPER_LID = (158, 159, 160)
RIGHT_LOWER_LID = (144, 145, 153)
LEFT_UPPER_LID = (385, 386, 387)
LEFT_LOWER_LID = (373, 374, 380)

# Iris landmarks (only present in the 478-point model): centre + 4 contour points.
RIGHT_IRIS = (468, 469, 470, 471, 472)
LEFT_IRIS = (473, 474, 475, 476, 477)

# Full eye outlines, used for the camera preview overlay.
RIGHT_EYE_CONTOUR = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
LEFT_EYE_CONTOUR = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)

# Outlines for the live face-mesh overlay shown during calibration (ordered
# loops of landmark indices from the MediaPipe canonical face mesh).
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
             378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
             162, 21, 54, 103, 67, 109)
LIPS_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
              37, 39, 40, 185)
LEFT_BROW = (276, 283, 282, 295, 285)
RIGHT_BROW = (46, 53, 52, 65, 55)
NOSE_BRIDGE = (168, 6, 197, 195, 5, 4)

# The connected outlines the overlay draws as a light wireframe over the dots.
MESH_OUTLINES = (FACE_OVAL, LIPS_OUTER, RIGHT_EYE_CONTOUR, LEFT_EYE_CONTOUR,
                 LEFT_BROW, RIGHT_BROW, NOSE_BRIDGE)

NUM_LANDMARKS_WITH_IRIS = 478
