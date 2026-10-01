"""Paralic Inspector: replay a session recording and look inside the models.

Run ``python -m paralic.inspector`` and open the page it prints. It lists the
recordings in ``data/recordings/`` (made with the ● Rec button or
``python -m paralic --record``; see ``docs/recording-format.md``) and replays
one frame by frame: the camera image with the tracked points, where the gaze
landed on the screen, every signal over time, and - for each frame - what the
gaze networks computed inside (``internals.py``), the blink and wink detectors'
state, the calibrations and every event.

* ``internals.py`` - each step of a gaze network for one frame, and which inputs drive it
* ``views.py``     - what the panels show, computed from a recording (signals, timeline...)
* ``server.py``    - the local web server (JSON API + the page in ``static/``)
"""
