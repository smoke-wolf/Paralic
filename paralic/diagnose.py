"""What limits the accuracy in a session recording, in plain words.

Every accuracy check in a recording (the validation dots after a calibration,
a quick adjust or an improving round) is taken apart:

* the **shift** all dots share (the mean error vector) - a quick adjust, or
  sitting where the calibration was made, removes it;
* the **scatter** between dots once the shift is gone - errors that differ
  across the screen; more calibration dots or an improving round help;
* the **jitter** within each dot (frame-to-frame spread around its own mean) -
  noise from the camera and the face mesh; light, distance and a steady head
  help, more training does not;
* left-right against up-down, and the edges of the screen against the middle.

The session's conditions are summed up as well (frame rate, how often the face
was found, glare and glasses, other faces in view, how much the person moved,
what learning from clicks did), and the most useful findings are written as
sentences. ``python -m paralic.diagnose [recording]`` prints them; the
Inspector shows them on its Calibration panel.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

EDGE = 0.15          # a dot within this fraction of the screen from an edge is an edge dot


def _vec(v) -> Optional[np.ndarray]:
    try:
        a = np.asarray(v, float)
    except (TypeError, ValueError):
        return None
    return a if a.shape == (2,) and np.isfinite(a).all() else None


def _direction(v: np.ndarray) -> str:
    """'to the left and up' for a screen vector (x right, y down)."""
    parts = []
    if abs(v[0]) >= 0.35 * np.hypot(*v):
        parts.append("to the right" if v[0] > 0 else "to the left")
    if abs(v[1]) >= 0.35 * np.hypot(*v):
        parts.append("down" if v[1] > 0 else "up")
    return " and ".join(parts) or "nowhere in particular"


def validation_parts(points: list, screen: Optional[dict] = None) -> Optional[dict]:
    """Take one accuracy check apart (see the module docstring). ``points`` are
    the validation result's dots: {target, mean, error, spread, n}."""
    tgt, err, spread, frames = [], [], [], []
    for p in points or []:
        if not isinstance(p, dict):
            continue
        t, m = _vec(p.get("target")), _vec(p.get("mean"))
        if t is None or m is None:
            continue
        tgt.append(t)
        err.append(m - t)
        spread.append(float(p.get("spread") or 0.0))
        frames.append(max(1, int(p.get("n") or 1)))
    if len(err) < 3:
        return None
    T, E = np.array(tgt), np.array(err)
    shift = E.mean(axis=0)
    rest = E - shift
    mean_error = float(np.linalg.norm(E, axis=1).mean())
    msq = float((np.linalg.norm(E, axis=1) ** 2).mean()) or 1e-9
    w = float((screen or {}).get("w") or max(T[:, 0].max(), 1.0))
    hgt = float((screen or {}).get("h") or max(T[:, 1].max(), 1.0))
    edge = (T[:, 0] < EDGE * w) | (T[:, 0] > (1 - EDGE) * w) | (T[:, 1] < EDGE * hgt) | (T[:, 1] > (1 - EDGE) * hgt)
    dist = np.linalg.norm(E, axis=1)
    return {
        "dots": int(len(E)),
        "mean_error_px": round(mean_error, 1),
        "shift_px": [round(float(shift[0]), 1), round(float(shift[1]), 1)],
        "shift_share": round(float(np.dot(shift, shift)) / msq, 2),
        "scatter_px": round(float(np.linalg.norm(rest, axis=1).mean()), 1),
        "jitter_px": round(float(np.mean(spread)), 1),
        # What the jitter leaves in a dot's average (its frames averaged): the
        # error no calibration can go below with this camera and light.
        "noise_floor_px": round(float(np.mean(np.array(spread) / np.sqrt(frames))), 1),
        "horizontal_px": round(float(np.sqrt((E[:, 0] ** 2).mean())), 1),
        "vertical_px": round(float(np.sqrt((E[:, 1] ** 2).mean())), 1),
        "edge_px": round(float(dist[edge].mean()), 1) if edge.any() else None,
        "middle_px": round(float(dist[~edge].mean()), 1) if (~edge).any() else None,
        "screen_diagonal_px": round(float(np.hypot(w, hgt)), 0),
    }


def _findings_for(v: dict) -> list[str]:
    """The sentences for one accuracy check, the most important first."""
    out = []
    err = v["mean_error_px"]
    shift = np.array(v["shift_px"])
    if v["shift_share"] >= 0.5 and np.hypot(*shift) >= 25:
        out.append(f"Most of the error ({round(100 * v['shift_share'])}%) is one shift of {round(float(np.hypot(*shift)))} px "
                   f"{_direction(shift)} shared by every dot - a quick adjust (or sitting where you calibrated) "
                   "removes most of it.")
    if err <= 1.5 * v["noise_floor_px"] + 5:
        out.append(f"The error ({round(err)} px) is about what the frame-to-frame noise leaves after averaging "
                   f"(~{round(v['noise_floor_px'])} px): the calibration is as good as this camera and light allow. "
                   "More training will not lower it; more light on the face, sitting a little closer and a steady "
                   "head will.")
    if v["jitter_px"] >= 60:
        out.append(f"The raw estimate jitters about {round(v['jitter_px'])} px from frame to frame (the cursor's "
                   "smoothing hides most of it, at the cost of some lag): more light and a steady head make the "
                   "cursor steadier and quicker.")
    if v["vertical_px"] >= 1.5 * max(v["horizontal_px"], 1.0):
        out.append(f"Up-down is the weak direction ({round(v['vertical_px'])} px against {round(v['horizontal_px'])} px "
                   "left-right): the eyelids hide part of the iris when looking down. A camera at eye height, or the "
                   "screen a little lower, helps.")
    elif v["horizontal_px"] >= 1.5 * max(v["vertical_px"], 1.0):
        out.append(f"Left-right is the weak direction ({round(v['horizontal_px'])} px against {round(v['vertical_px'])} px "
                   "up-down) - often the head turned a little: face the screen squarely.")
    if v["edge_px"] is not None and v["middle_px"] is not None and v["edge_px"] >= 1.6 * max(v["middle_px"], 1.0):
        out.append(f"The edges of the screen are worse ({round(v['edge_px'])} px) than the middle "
                   f"({round(v['middle_px'])} px): an improving round adds dots where the error is largest.")
    if v["scatter_px"] >= 0.6 * err and v["scatter_px"] >= 2 * v["noise_floor_px"]:
        out.append(f"After the shared shift, the dots still disagree by about {round(v['scatter_px'])} px: errors that "
                   "differ across the screen - another full calibration or an improving round helps more than a "
                   "quick adjust.")
    return out


def frame_conditions(rec) -> dict:
    """The session's conditions from its frames (eye or hand mode)."""
    n = present = glare = glasses = crowded = waiting = 0
    fps, dist, yaw, pitch, light = [], [], [], [], []
    nets: Counter = Counter()
    glare_eyes: Counter = Counter()
    glasses_changes, last_glasses = 0, None
    for f in rec.iter_frames(light=True):
        m = f.get("msg") or {}
        n += 1
        if m.get("face"):
            present += 1
            if isinstance(m.get("dist"), (int, float)):
                dist.append(float(m["dist"]))
            head = m.get("head")
            if isinstance(head, list) and len(head) >= 2:
                yaw.append(float(head[0]))
                pitch.append(float(head[1]))
            if m.get("net"):
                nets[m["net"]] += 1
        if isinstance(m.get("fps"), (int, float)) and m["fps"] > 0:
            fps.append(float(m["fps"]))
        if m.get("glare"):
            glare += 1
            glare_eyes[m["glare"]] += 1
        g = m.get("glasses")
        if g is not None:
            glasses += int(bool(g))
            if last_glasses is not None and bool(g) != last_glasses:
                glasses_changes += 1
            last_glasses = bool(g)
        if (m.get("faces") or 0) > 1 or (m.get("hands") or 0) > 1:
            crowded += 1
        waiting += int(bool(m.get("waiting")))
        lt = m.get("light")
        if isinstance(lt, dict) and isinstance(lt.get("face"), (int, float)):
            light.append(float(lt["face"]))
    pct = (lambda k: round(100.0 * k / n, 1)) if n else (lambda k: 0.0)
    return {
        "frames": n,
        "found_pct": pct(present),
        "fps": round(float(np.median(fps)), 1) if fps else None,
        "glare_pct": pct(glare),
        "glare_eyes": dict(glare_eyes),
        "glasses_pct": pct(glasses),
        "glasses_changes": glasses_changes,
        "others_in_view_pct": pct(crowded),
        "waiting_pct": pct(waiting),
        "distance_cm": round(float(np.median(dist)), 1) if dist else None,
        "distance_moved_cm": round(float(np.percentile(dist, 90) - np.percentile(dist, 10)), 1) if len(dist) > 10 else None,
        "head_turn_deg": round(float(np.std(yaw) + np.std(pitch)), 1) if len(yaw) > 10 else None,
        "face_light": round(float(np.median(light)), 0) if light else None,
        "networks": dict(nets),
    }


def learning_summary(events: list) -> dict:
    """What learning from clicks and the background fine-tuning did."""
    stored, skipped = 0, Counter()
    tuned = accepted = 0
    for e in events:
        d = e.get("data") if isinstance(e.get("data"), dict) else {}
        if e.get("type") == "label_stored" and e.get("kind") == "reply":
            if d.get("stored"):
                stored += 1
            elif d.get("reason"):
                skipped[str(d["reason"])] += 1
        elif e.get("type") == "finetune_result" and d.get("ok") is not False and "accepted" in d:
            tuned += 1
            accepted += int(bool(d.get("accepted")))
    return {"clicks_learned": stored, "clicks_skipped": dict(skipped), "finetunes": tuned,
            "finetunes_accepted": accepted}


def _condition_findings(c: dict, learning: dict, mode: str) -> list[str]:
    out = []
    thing = "hand" if mode == "hand" else "face"
    if c["frames"] and c["found_pct"] < 90:
        out.append(f"The {thing} was found in only {c['found_pct']}% of the frames: keep it in view and well lit.")
    if c["fps"] is not None and c["fps"] < 20:
        out.append(f"The camera ran at about {c['fps']} frames a second (30 is ideal): close other apps that use the "
                   "camera or the processor, or use a smaller browser window.")
    if c["glare_pct"] >= 10:
        out.append(f"A reflection on glasses hid an eye in {c['glare_pct']}% of the frames "
                   f"({', '.join(f'{k}: {v}' for k, v in c['glare_eyes'].items())}): tilt the screen a little or move "
                   "the lamp; meanwhile the other eye's network leads.")
    if c["glasses_changes"] >= 3:
        out.append(f"Glasses seemed to go on and off {c['glasses_changes']} times: if they did not, the glasses "
                   "detector is unsure in this light (a calibration may have been saved for the wrong state).")
    if c["distance_moved_cm"] is not None and c["distance_moved_cm"] >= 10:
        out.append(f"The distance to the camera varied by about {c['distance_moved_cm']} cm during the session: the "
                   "gaze network is most accurate where it was calibrated - the seating check before a quick adjust "
                   "guides you back.")
    if c["others_in_view_pct"] >= 5:
        out.append(f"Someone else was in view in {c['others_in_view_pct']}% of the frames"
                   + (f"; Paralic waited for you in {c['waiting_pct']}%." if c["waiting_pct"] else "."))
    if c["face_light"] is not None and c["face_light"] < 70:
        out.append(f"The face was dim (brightness {int(c['face_light'])} of 255): more light from the front lowers "
                   "the noise.")
    skipped = sum(learning["clicks_skipped"].values())
    if learning["clicks_learned"] + skipped >= 5 and skipped > learning["clicks_learned"]:
        top = max(learning["clicks_skipped"].items(), key=lambda kv: kv[1])[0]
        out.append(f"Most clicks could not be learned from ({skipped} of {skipped + learning['clicks_learned']}; "
                   f"mostly: {top}).")
    if learning["finetunes"] and not learning["finetunes_accepted"]:
        out.append(f"Background fine-tuning ran {learning['finetunes']} time(s) and kept the current network each time: "
                   "a new one must beat it on your most recent clicks, so changes come slowly - by design.")
    return out


def diagnose(view) -> dict:
    """The diagnosis of a recording (``view``: an inspector RecordingView)."""
    rec = view.rec
    screen = rec.screen() if callable(getattr(rec, "screen", None)) else None
    checks = []
    for s in view.calibrations().get("sessions", []):
        val = s.get("validation") or {}
        parts = validation_parts(val.get("points") or [], screen)
        if parts is None:
            continue
        checks.append({"t": s.get("t0"), "mode": s.get("mode") or s.get("kind"), **parts,
                       "findings": _findings_for(parts)})
    conditions = frame_conditions(rec)
    learning = learning_summary(rec.events())
    findings = list(checks[-1]["findings"]) if checks else []
    if len(checks) >= 2:
        first, last = checks[0]["mean_error_px"], checks[-1]["mean_error_px"]
        trend = "improved" if last < 0.9 * first else "got worse" if last > 1.1 * first else "stayed about the same"
        findings.append(f"Across the {len(checks)} accuracy checks the error {trend}: {round(first)} → {round(last)} px.")
    findings += _condition_findings(conditions, learning, rec.mode() if callable(getattr(rec, "mode", None)) else "eyes")
    if not checks:
        findings.insert(0, "No accuracy check in this recording: a calibration, quick adjust or 'check my accuracy' "
                           "measures it on fresh dots.")
    return {"checks": checks, "conditions": conditions, "learning": learning, "findings": findings}


def report(diag: dict, name: str = "") -> str:
    lines = [f"Paralic diagnosis{f' of {name}' if name else ''}", ""]
    for i, c in enumerate(diag["checks"], 1):
        lines.append(f"Accuracy check {i} ({c['mode']}): {c['mean_error_px']} px over {c['dots']} dots - "
                     f"shift {c['shift_px']} ({round(100 * c['shift_share'])}%), scatter {c['scatter_px']} px, "
                     f"jitter {c['jitter_px']} px (noise floor {c['noise_floor_px']} px), "
                     f"left-right {c['horizontal_px']} / up-down {c['vertical_px']} px")
    c = diag["conditions"]
    lines += ["", f"Frames {c['frames']}, found {c['found_pct']}%, {c['fps']} fps, glare {c['glare_pct']}%, "
                  f"glasses {c['glasses_pct']}%, others in view {c['others_in_view_pct']}%, "
                  f"distance {c['distance_cm']} cm (varied {c['distance_moved_cm']} cm)", ""]
    lines += [f"- {f}" for f in diag["findings"]] or ["- Nothing stands out."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    from .inspector.views import RecordingView
    from .recording import Recording, list_recordings, recordings_root

    ap = argparse.ArgumentParser(prog="python -m paralic.diagnose",
                                 description="What limits the accuracy in a session recording.")
    ap.add_argument("recording", nargs="?", help="recording folder or id (default: the newest)")
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    args = ap.parse_args(argv)
    path = Path(args.recording) if args.recording else None
    if path is None or not path.is_dir():
        root = recordings_root(args.data_dir)
        found = list_recordings(root)
        if args.recording:
            found = [r for r in found if r.get("id") == args.recording]
        if not found:
            print(f"No recording found in {root}", file=sys.stderr)
            return 1
        newest = max(found, key=lambda r: (r.get("started") or "", r.get("id") or ""))
        path = Path(newest["path"]) if newest.get("path") else root / newest["id"]
    view = RecordingView(Recording(path))
    print(report(diagnose(view), path.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
