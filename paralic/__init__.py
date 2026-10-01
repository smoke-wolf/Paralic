"""Paralic: web browsing driven by webcam eye tracking - or by a hand.

The package is split into small, independently testable pieces:

* ``features``  - turns MediaPipe face landmarks into eye / head features
* ``blink``     - blink and double-blink detection (``gestures``: winks)
* ``filters``   - One Euro smoothing and blink-aware cursor stabilisation
* ``gazenet``   - the personal gaze neural network (NumPy MLP)
* ``calibration`` - calibration data handling, training and profiles
* ``faceprint`` - recognising who is at the camera
* ``hand_gestures``, ``hand_control`` - hand mode (point, pinch, open hand)
* ``tracker``, ``hands`` - MediaPipe FaceLandmarker / HandLandmarker wrappers
* ``session``   - per-browser-connection processing pipeline (eyes or hand)
* ``server``    - FastAPI web server (website + WebSocket)
"""

__version__ = "1.0.0"
