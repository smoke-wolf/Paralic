# Glasses and glare: detection on painted glasses

Produced by `python tools/glasses_eval.py`. Glasses are painted onto three of MediaPipe's public test faces (the test portrait, a business portrait and a stylizer test face, downloaded once into tests/.cache): thick and thin full-rim frames, half-rim and rimless frames (no lower rim), small lenses, and thin light-coloured metal frames; and, as things that are not glasses, frown lines across the nose. Each picture is then changed like a webcam frame - dimmer or brighter, tilted, smaller (30 to 63 pixels between the eye centres), noisy, JPEG-compressed - and run through the real MediaPipe face mesh. Reflections are painted onto a lens after the light changed (a lamp's reflection stays white). These are painted glasses on still photos, not real ones on people (no shadows of the frame, no room reflected in the lenses, no tinted lenses), so the numbers say the method works as intended, not how often it is right for real glasses. 

## Glasses

Each face in 24 webcam-like versions (4 brightnesses, 2 tilts, 3 sizes). The score is the nose-bridge edge in units of the skin's own edges, with up to half again for rims below both eyes (see `paralic/glasses.py`); one frame is judged on its own here, while Paralic smooths the score over about half a second: glasses above 2.1, none below 1.4.

| Picture | Seen as glasses (portrait · B · C) | Score: min / median / max |
| --- | --- | --- |
| no glasses *(not glasses)* | 0/24 · 0/24 · 0/24 | 0.10 / 0.25 / 0.53 |
| frown line across the nose *(not glasses)* | 0/24 · 0/24 · 0/24 | 0.21 / 0.79 / 1.09 |
| deep frown line across the nose *(not glasses)* | 0/24 · 0/24 · 0/24 | 0.49 / 1.32 / 1.69 |
| thick full-rim frames | 24/24 · 24/24 · 24/24 | 6.90 / 9.52 / 11.87 |
| thin full-rim frames | 24/24 · 24/24 · 24/24 | 3.73 / 6.28 / 7.87 |
| half-rim frames (no lower rim) | 24/24 · 24/24 · 24/24 | 4.30 / 5.71 / 7.00 |
| rimless (bridge and arms only) | 24/24 · 24/24 · 24/24 | 2.51 / 3.86 / 4.88 |
| small lenses | 24/24 · 24/24 · 24/24 | 7.32 / 9.72 / 12.34 |
| thin light metal frames | 0/24 · 0/24 · 0/24 | 0.10 / 0.59 / 0.93 |

Thin frames the colour of light skin hardly stand out from it and are not seen (they would be calibrated as "without glasses", as before Paralic noticed glasses).

## Glare

Painted glasses, 3 brightnesses per face, a reflection painted on top. Largest bright, nearly colourless blob around each eye, in iris areas, for single frames; Paralic smooths it and flags glare above 0.5 (below 0.25 it goes again). Faces lit almost white (skin brighter than 220 of 255) are not judged.

| Reflection | Right eye flagged | Left eye flagged | Blob, right / left (median) |
| --- | --- | --- | --- |
| none (1 lit almost white, not judged) | 0/8 | 0/8 | 0.00 / 0.00 |
| on the right lens, over the eye (1 lit almost white, not judged) | 8/8 | 0/8 | 2.98 / 0.00 |
| small, on the left eye (1 lit almost white, not judged) | 0/8 | 8/8 | 0.00 / 1.23 |
| on the right lens, beside the eye (1 lit almost white, not judged) | 8/8 | 0/8 | 1.72 / 0.00 |
| on both lenses | 9/9 | 9/9 | 2.92 / 2.90 |
| dimmer (215 of 255), on the left lens (1 lit almost white, not judged) | 0/8 | 7/8 | 0.00 / 1.41 |

_Run time 77 s._
