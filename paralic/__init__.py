"""Paralic: hands-free web browsing driven by webcam eye tracking.

The package is split into small, independently testable pieces:

* ``features``  - turns MediaPipe face landmarks into eye / head features
* ``blink``     - blink and double-blink detection
* ``filters``   - One Euro smoothing and blink-aware cursor stabilisation
* ``gazenet``   - the personal gaze neural network (NumPy MLP)
* ``calibration`` - calibration data handling, training and profiles
* ``tracker``   - MediaPipe FaceLandmarker wrapper
* ``session``   - per-browser-connection processing pipeline
* ``server``    - FastAPI web server (website + WebSocket)
"""

__version__ = "1.0.0"
