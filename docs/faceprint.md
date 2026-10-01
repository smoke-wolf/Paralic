# Face print: recognition on public test faces

Produced by `python tools/faceprint_eval.py`. Five different people from MediaPipe's public test images (downloaded once into tests/.cache): the test portrait, a business portrait, a stylizer test face, and the two people in a two-person photo. Each image is altered many times - turned, scaled, shifted, a slight perspective (a little head turn), brighter or darker, unevenly lit, blurred, noisy - and run through the real MediaPipe face mesh, like webcam frames. Prints are made from 12 altered frames of each enrolled person; other altered frames are then recognised. These are still photos made to vary, not people over days (no expressions, no real head turns, no new haircut), so the numbers say the method works as intended, not how it does on real people over time. 

Scores are in units of a person's own typical difference between two views (≈ 1 for their own face); a face is recognised below 3 and when 1.6× closer to that person than to anyone else, otherwise the page asks.

## Five people enrolled

| Face | Single frames right | Wrong person | 4-frame windows right | Own score | Nearest other |
| --- | --- | --- | --- | --- | --- |
| the test portrait | 9/10 | 0 | 4/4 | 1.1 | 30.3 |
| face B | 10/10 | 0 | 4/4 | 1.3 | 22.6 |
| face C | 10/10 | 0 | 4/4 | 0.8 | 17.8 |
| face D | 10/10 | 0 | 4/4 | 0.9 | 26.8 |
| face E | 9/10 | 0 | 4/4 | 1.2 | 59.4 |

## Three enrolled, two strangers

| Face | Single frames right | Wrong person | 4-frame windows right | Own score | Nearest other |
| --- | --- | --- | --- | --- | --- |
| the test portrait | 9/10 | 0 | 4/4 | 0.9 | 42.0 |
| face B | 10/10 | 0 | 4/4 | 1.3 | 26.4 |
| face C | 10/10 | 0 | 4/4 | 0.8 | 19.9 |
| face D (never enrolled) | 10/10 | 0 | 4/4 | – | 17.2 |
| face E (never enrolled) | 10/10 | 0 | 4/4 | – | 36.4 |

## One person enrolled

| Face | Single frames right | Wrong person | 4-frame windows right | Own score | Nearest other |
| --- | --- | --- | --- | --- | --- |
| the test portrait | 9/10 | 0 | 4/4 | 1.5 | – |
| face B (never enrolled) | 10/10 | 0 | 4/4 | – | 11.7 |
| face C (never enrolled) | 10/10 | 0 | 4/4 | – | 9.3 |
| face D (never enrolled) | 10/10 | 0 | 4/4 | – | 9.3 |
| face E (never enrolled) | 10/10 | 0 | 4/4 | – | 18.8 |

_Run time 11 s._
