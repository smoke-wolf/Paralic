// Calibration: teach GazeNet where on the screen you are looking.
//
// Full calibration (about a minute and a half):
//   0. get comfortable: the camera view with one hint at a time until the face
//      is lit, at a good distance and in front of the camera (position.js),
//   1. look at 21 dots in turn, head still. Each dot stays until the server
//      reports that the eyes have rested on it (the ``settled`` count), so
//      slower eyes get more time and nothing counts while they are still on
//      the way; the dot shrinks as it measures,
//   2. keep looking at the centre dot while turning, nodding and tilting the
//      head (teaches the network to compensate for head movement),
//   3. the server trains your personal neural network,
//   4. five more dots measure the accuracy (and are then used for a final fit),
//   5. blink twice three times when the dot turns purple: learns how you
//      blink (thresholds, timing, and which eye signal to watch).
//
// Quick adjust (about 20 seconds, eyes only): back to where you sat while
// calibrating (the network is most accurate there), then nine dots fit a
// correction for today. The mouse-guided tune-up ('adjust-mouse') is for a
// helper or anyone who can use a mouse: a dot glides along a path while the
// mouse pointer is kept where the eyes look, labelling every frame.
//
// Blink test / wink test: the blink step on its own, and "close your left
// eye, now your right eye" to learn how (and whether) each eye winks.

import { h, sleep, toast } from './dom.js';
import { icon } from './icons.js';
import { clientToScreen, screenToClient } from './screen-space.js';
import { sounds } from './sound.js';
import { speak, canSpeak, stopSpeaking } from './speech.js';
import { drawMesh } from './camera-panel.js';
import { assessPosition } from './position.js';

class Cancelled extends Error {}

// A denser 5x4 grid (plus the centre) than the old 13 dots: more positions make
// the gaze network interpolate better across the whole screen. Snake-ordered so
// the dot only ever makes short hops. Every run of dots starts in the centre,
// where the countdown was: the eyes are already there, and each later dot can
// tell the eyes arriving from the eyes still resting on the dot before it.
function gridPoints() {
  const xs = [0.06, 0.28, 0.5, 0.72, 0.94];
  const ys = [0.08, 0.37, 0.63, 0.92];
  const pts = [[0.5, 0.5]];
  ys.forEach((y, row) => {
    const rowXs = row % 2 === 0 ? xs : [...xs].reverse(); // snake order: short hops
    rowXs.forEach((x) => pts.push([x, y]));
  });
  return pts;
}

const FULL_POINTS = gridPoints();
const VALIDATION_POINTS = [[0.5, 0.55], [0.27, 0.28], [0.73, 0.28], [0.73, 0.8], [0.27, 0.8]];
// Quick adjust: the centre, then once around the edges. Nine dots fit the
// correction better than five (benchmark: 0.21x vs 0.24x of the drift error).
const ADJUST_POINTS = [[0.5, 0.5], [0.1, 0.12], [0.5, 0.12], [0.9, 0.12], [0.9, 0.5],
  [0.9, 0.88], [0.5, 0.88], [0.1, 0.88], [0.1, 0.5]];
// Head poses held while the eyes stay on the centre dot: turn (yaw), nod
// (pitch) and tilt (roll). The straight-ahead pose is already covered by the
// dots. Kept short: each is spoken in full before its frames are gathered.
const HEAD_STEPS = [
  'Turn your head a little to the left.',
  'Now a little to the right.',
  'Back to the middle, and tip your head up a little.',
  'Now tip it down a little.',
  'Tilt your head towards your left shoulder.',
  'Now towards your right shoulder.',
];
// Frames of a settled gaze each dot needs (see collect()).
const DOT_FRAMES = { cal: 16, val: 14, adjust: 14 };

// Improving rounds (see improveAccuracy): at most this many, each measured on
// dots of its own that no network was trained on (they start in the centre).
const REFINE_ROUNDS = 3;
const ROUND_VALIDATION = [
  [[0.5, 0.45], [0.15, 0.5], [0.5, 0.15], [0.85, 0.5], [0.5, 0.85]],
  [[0.5, 0.5], [0.2, 0.15], [0.8, 0.15], [0.8, 0.85], [0.2, 0.85]],
  [[0.5, 0.6], [0.38, 0.33], [0.62, 0.33], [0.9, 0.7], [0.1, 0.7]],
];
// Places for extra dots: between the calibration grid's dots, and along the
// edges and in the corners (where the eyes turn furthest and tracking is
// hardest).
const BETWEEN = [0.17, 0.39, 0.61, 0.83].flatMap((x) => [0.22, 0.5, 0.78].map((y) => [x, y]));
const EDGES = [[0.03, 0.05], [0.5, 0.04], [0.97, 0.05], [0.97, 0.5], [0.97, 0.95], [0.5, 0.96],
  [0.03, 0.95], [0.03, 0.5], [0.28, 0.04], [0.72, 0.96]];

/**
 * Extra dots for an improving round, where the last measurement says the
 * tracking is least sure. `measured`: [{x, y, error}] in page fractions.
 * Each round is a different exercise: round 1 fills in between the grid,
 * round 2 goes to the edges and corners, round 3 mixes both, shifted a
 * little so no place repeats. Starts in the centre (where the countdown is)
 * and is ordered as a short path.
 */
export function refinePoints(measured, round, n = 9) {
  const errorAt = ([x, y]) => {
    let wsum = 0;
    let esum = 0;
    for (const m of measured) {
      const w = 1 / (0.02 + (m.x - x) ** 2 + (m.y - y) ** 2);
      wsum += w;
      esum += w * m.error;
    }
    return wsum ? esum / wsum : 1;
  };
  const shift = round >= 3 ? 0.04 : 0;
  const pool = (round === 1 ? BETWEEN : round === 2 ? EDGES : [...BETWEEN, ...EDGES])
    .map(([x, y], i) => [Math.min(0.97, Math.max(0.03, x + (i % 2 ? shift : -shift))), y]);
  const ranked = pool.map((p) => ({ p, e: errorAt(p) })).sort((a, b) => b.e - a.e);
  const picked = [];
  for (const { p } of ranked) {
    if (picked.length >= n - 1) break;
    if (picked.every((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) >= 0.16)) picked.push(p);
  }
  // A short path from the centre: always on to the nearest dot left.
  const path = [[0.5, 0.5]];
  while (picked.length) {
    const [cx, cy] = path[path.length - 1];
    let best = 0;
    picked.forEach((q, i) => {
      if (Math.hypot(q[0] - cx, q[1] - cy) < Math.hypot(picked[best][0] - cx, picked[best][1] - cy)) best = i;
    });
    path.push(picked.splice(best, 1)[0]);
  }
  return path;
}

export function rateAccuracy(errorPx) {
  const rel = errorPx / Math.hypot(window.innerWidth, window.innerHeight);
  if (rel < 0.035) return 'excellent';
  if (rel < 0.055) return 'good';
  if (rel < 0.085) return 'fair';
  return 'poor';
}

export class Calibrator {
  constructor(app) {
    this.app = app;
    this.running = false;
    this.cancelled = false;
    window.addEventListener('keydown', (e) => {
      if (this.running && e.key === 'Escape') this.cancelled = true;
    });
  }

  get tracker() {
    return this.app.tracker;
  }

  /** Run a calibration. Resolves to the result, or null if cancelled. */
  async run(mode = 'full') {
    if (this.running) return null;
    this.running = true;
    this.cancelled = false;
    this.app.gaze.setSuspended(true);
    const ov = this.app.openOverlay('calib solid');
    this.ui = this.buildUI(ov);
    this.startMeshPreview();
    try {
      if (mode === 'adjust') return await this.runAdjust();
      if (mode === 'adjust-mouse') return await this.runPursuitAdjust();
      if (mode === 'blink') return await this.runBlinkTest();
      if (mode === 'wink') return await this.runWinkTest();
      if (mode === 'head') return await this.runHeadTest();
      return await this.runFull();
    } catch (err) {
      if (err instanceof Cancelled) {
        toast('Calibration cancelled');
        return null;
      }
      throw err;
    } finally {
      stopSpeaking();
      this._lastSpoken = null;
      this.stopMeshPreview();
      this.tracker.setLabel(null);
      this.tracker.setGesturePhase(null);
      this.tracker.setup = false;
      this.running = false;
      this.app.closeOverlay(ov);
      this.app.gaze.setSuspended(false);
    }
  }

  buildUI(ov) {
    const dot = h('div', { class: 'calib-dot done' });
    const text = h('div', { class: 'calib-text' });
    const warning = h('div', { class: 'calib-warning', hidden: true });
    const progress = h('div', { class: 'calib-progress' });
    // Live, mirrored camera preview with the face mesh drawn on top: large
    // while getting into position, small in a corner during the instructions
    // and head movements, and hidden while the eyes should rest on a dot (a
    // moving face at the edge of vision draws the eyes) unless the face is lost.
    const video = h('video', { class: 'calib-cam-video', autoplay: true, muted: true, playsinline: true });
    const meshCanvas = h('canvas', { class: 'calib-cam-mesh' });
    const faceGuide = h('div', { class: 'face-guide' });
    const camLabel = h('div', { class: 'calib-cam-label' }, 'Your face');
    const cam = h('div', { class: 'calib-cam' }, video, meshCanvas, faceGuide, camLabel);
    // A ring that fills as good frames are gathered for the current step.
    const ring = h('div', { class: 'calib-ring', hidden: true },
      h('svg', { viewBox: '0 0 48 48', html:
        '<circle class="track" cx="24" cy="24" r="21"></circle>' +
        '<circle class="fill" cx="24" cy="24" r="21"></circle>' }));
    // Visible transcript of the spoken guidance (accessibility; also helps when
    // speech is off or unsupported). Lives at the bottom, out of the way.
    const transcript = h('div', { class: 'calib-transcript', 'aria-live': 'polite' });
    ov.append(dot, text, warning, progress, cam, ring, transcript);
    return { ov, dot, text, warning, progress, transcript, cam, video, meshCanvas, faceGuide, ring };
  }

  /** Show the camera + live face mesh during the calibration. */
  startMeshPreview() {
    const t = this.tracker;
    if (t.simulated || !this.ui) { if (this.ui?.cam) this.ui.cam.hidden = true; return; }
    this._prevOverlay = t.overlay;
    t.overlay = true;    // make the server compute overlay/landmarks
    t.mesh = true;       // ask it to stream the full 478-point mesh
    try {
      if (t.stream) { this.ui.video.srcObject = t.stream; this.ui.video.play?.().catch(() => {}); }
    } catch { /* preview is best-effort */ }
    this._meshOff = t.on('frame', (m) => {
      if (!this.ui) return;
      const w = this.ui.video.videoWidth || 320;
      const hgt = this.ui.video.videoHeight || 240;
      drawMesh(this.ui.meshCanvas, m.face ? m.mesh : null, w, hgt);
      this.ui.cam.classList.toggle('no-face', !m.face);
    });
  }

  stopMeshPreview() {
    if (this._meshOff) { this._meshOff(); this._meshOff = null; }
    const t = this.tracker;
    t.mesh = false;
    if (this._prevOverlay !== undefined) t.overlay = this._prevOverlay;
    try { if (this.ui?.video) this.ui.video.srcObject = null; } catch { /* ignore */ }
  }

  /** Camera preview layout: 'big' (getting into position), 'corner', or 'dots'. */
  setCam(mode) {
    const cam = this.ui?.cam;
    if (!cam) return;
    cam.classList.remove('big', 'dots', 'lost');
    if (mode !== 'corner') cam.classList.add(mode);
  }

  /** While dots show, keep the (normally hidden) preview in the corner
   *  farthest from the dot, so a lost-face warning never covers it. */
  placeCamAwayFrom(x, y) {
    const cam = this.ui?.cam;
    if (!cam) return;
    const right = x < window.innerWidth / 2;
    const bottom = y < window.innerHeight / 2;
    cam.classList.toggle('at-r', right);
    cam.classList.toggle('at-b', bottom);
  }

  checkCancel() {
    if (this.cancelled) throw new Cancelled();
  }

  async wait(ms) {
    const end = performance.now() + ms;
    while (performance.now() < end) {
      this.checkCancel();
      await sleep(Math.min(50, end - performance.now()));
    }
  }

  say(title, body = '', { top = false, voice = true } = {}) {
    const { text } = this.ui;
    text.classList.toggle('top', top);
    text.innerHTML = '';
    if (title) text.append(h('h2', {}, title));
    if (body) text.append(typeof body === 'string' ? h('p', {}, body) : body);
    text.style.opacity = title || body ? '1' : '0';
    if (!title && !body) {
      // Cleared: the next instruction is spoken even if it repeats the last
      // one (each "Blink twice now"), and the transcript does not linger
      // next to the dots.
      this._lastSpoken = null;
      this.transcribe('');
      return;
    }
    // Read the instruction aloud (accessibility), synced with the on-screen
    // text and the visible transcript. Only speak real instructions, never the
    // spinner/animation bodies, and never the same line twice in a row.
    if (voice) this.announce(title, typeof body === 'string' ? body : '');
  }

  announce(title, body = '') {
    const line = [title, body].filter(Boolean).join('. ').trim();
    if (!line || line === this._lastSpoken) return;
    this._lastSpoken = line;
    this.transcribe(line);
    this._speech = canSpeak() ? speak(line) : null;
  }

  /** Show (and speak) an instruction, then wait until it has been said - or
   *  read, when speech is unavailable - but at most `maxMs`. */
  async sayAndWait(title, body = '', maxMs = 4000) {
    this._speech = null;
    this.say(title, body, { top: true });
    const words = `${title} ${body}`.split(/\s+/).filter(Boolean).length;
    const reading = Math.min(maxMs, Math.max(1000, words * 220));
    let spoken = !this._speech;
    if (this._speech) this._speech.then(() => { spoken = true; });
    const t0 = performance.now();
    for (;;) {
      this.checkCancel();
      const elapsed = performance.now() - t0;
      if (elapsed >= maxMs || (spoken && elapsed >= reading)) break;
      await sleep(50);
    }
  }

  /** Keep a small visible transcript of what was spoken, for anyone who can't
   *  hear it or has speech turned off. */
  transcribe(line) {
    const t = this.ui.transcript;
    if (!t) return;
    t.textContent = line;
  }

  setProgress(total, done) {
    const { progress } = this.ui;
    progress.innerHTML = '';
    for (let i = 0; i < total; i++) progress.append(h('i', { class: i < done ? 'done' : '' }));
  }

  placeDot(fx, fy, { instant = false } = {}) {
    const { dot } = this.ui;
    const x = fx * window.innerWidth;
    const y = fy * window.innerHeight;
    if (instant) dot.style.transition = 'none';
    // Position with the `translate` property: unlike `transform`, it is not
    // affected by the `scale` animation used to shrink the dot.
    dot.style.translate = `${x}px ${y}px`;
    if (instant) {
      void dot.offsetWidth;
      dot.style.transition = '';
    }
    dot.className = 'calib-dot';
    dot.style.removeProperty('--p');
    return { x, y };
  }

  /** Instructions with a 3-2-1 countdown in the middle of the screen, where
   *  the first dot then appears. */
  async countdown(title, body) {
    this.say(title, body);
    const n = h('div', { class: 'calib-count' });
    for (const k of [3, 2, 1]) {
      n.textContent = String(k);
      this.ui.text.append(n);
      await this.wait(750);
    }
    n.remove();
  }

  /**
   * Label frames with the dot position until enough good frames arrived.
   *
   * For the dots ('cal', 'val', 'adjust') a good frame is one of a settled
   * gaze: the server counts how many of the latest frames show the eyes
   * resting on this dot (``settled``, 0 while they are still on the previous
   * one), so slower eyes get more time and a glance away starts the count
   * again. Head steps count every labelled frame. ``onProgress(n,
   * minFrames)`` reports the count as it grows. Returns the number of
   * labelled frames.
   */
  async collect(kind, point, x, y, { minFrames = 20, minMs = 750, maxMs = null, onProgress = null } = {}) {
    const fixation = kind !== 'head';
    const limit = maxMs ?? (fixation ? 4000 : 3000);
    let count = 0;
    let settled = null;     // null until the server reports it (older servers don't)
    let missing = 0;
    const good = () => (fixation && settled !== null ? settled : count);
    const off = this.tracker.on('frame', (m) => {
      if (m.labeled) {
        count++;
        if (typeof m.settled === 'number') settled = m.settled;
      }
      missing = m.face ? 0 : missing + 1;
      if (onProgress) onProgress(good(), minFrames);
    });
    const s = clientToScreen(x, y);
    this.tracker.setLabel({ x: s.x, y: s.y, kind, pt: point });
    const t0 = performance.now();
    try {
      for (;;) {
        await this.wait(40);
        const elapsed = performance.now() - t0;
        const lost = missing > 4;
        this.warn(lost ? 'I can’t see your face — look at the screen and check the lighting' : null);
        this.ui.cam?.classList.toggle('lost', lost);
        if ((elapsed >= minMs && good() >= minFrames) || elapsed >= limit) break;
      }
    } finally {
      this.tracker.setLabel(null);
      this.ui.cam?.classList.remove('lost');
      off();
    }
    return count;
  }

  /** Show/fill/hide the circular progress ring (0..1). */
  setRing(fraction) {
    const ring = this.ui.ring;
    if (!ring) return;
    if (fraction == null) { ring.hidden = true; return; }
    ring.hidden = false;
    const fill = ring.querySelector('.fill');
    const C = 2 * Math.PI * 21;    // circumference for r=21
    const f = Math.max(0, Math.min(1, fraction));
    fill.style.strokeDasharray = `${C}`;
    fill.style.strokeDashoffset = `${C * (1 - f)}`;
    ring.classList.toggle('full', f >= 1);
  }

  warn(message) {
    const { warning } = this.ui;
    warning.hidden = !message;
    if (message) warning.textContent = message;
  }

  /** Show the dots one by one; each shrinks while it measures. */
  async showPoints(points, kind, opts = {}) {
    const dot = this.ui.dot;
    const minFrames = opts.minFrames ?? DOT_FRAMES[kind] ?? 16;
    let progress = 0;
    const shrink = (n, min) => {
      progress = Math.min(1, n / min);
      dot.style.setProperty('--p', String(progress));
    };
    let timeouts = 0;
    this.setCam('dots');
    this.setProgress(points.length, 0);
    try {
      for (let i = 0; i < points.length; i++) {
        const [fx, fy] = points[i];
        const { x, y } = this.placeDot(fx, fy, { instant: i === 0 });
        this.placeCamAwayFrom(x, y);
        // The dot glides over (0.55 s); collection then waits for the eyes.
        await this.wait(i === 0 ? 700 : 600);
        dot.classList.add('collect');
        // If the eyes never seem to settle on several dots in a row, the
        // tracking can't tell the dots apart well (dim light, a far camera):
        // don't make the person wait the full time on every dot.
        const limit = timeouts >= 2 ? { maxMs: 1600 } : {};
        progress = 0;
        const pt = (opts.pointBase || 0) + i;
        let n = await this.collect(kind, pt, x, y, { ...opts, ...limit, minFrames, onProgress: shrink });
        if (n < 5) n = await this.collect(kind, pt, x, y, { ...opts, ...limit, minFrames, onProgress: shrink }); // one retry
        timeouts = progress >= 1 ? 0 : timeouts + 1;
        sounds.point();
        this.setProgress(points.length, i + 1);
      }
    } finally {
      dot.className = 'calib-dot done';
      dot.style.removeProperty('--p');
      this.setCam('corner');
    }
  }

  /**
   * "Get comfortable": the big camera view with one hint at a time until the
   * face is found, lit, at a good distance and in front of the camera - and,
   * before a quick adjust, back where it was during the full calibration.
   * Moves on by itself once everything has been fine for a moment. Never
   * blocks: a double blink, a click or Space carries on at once, and after
   * 20 seconds it carries on anyway (some people cannot move).
   */
  async positionCheck(mode = 'full') {
    const t = this.tracker;
    if (t.simulated || !this.ui) return;
    const ref = mode === 'adjust' ? (this.app.state.personal || {}).pose || null : null;
    const ui = this.ui;
    const items = {
      face: h('li', {}, h('i'), 'Face'),
      place: h('li', {}, h('i'), ref ? 'Same place as last time' : 'Position'),
      light: h('li', {}, h('i'), 'Light'),
    };
    const list = h('ul', { class: 'setup-checks' }, items.face, items.place, items.light);
    const hint = h('p', { class: 'setup-hint' });
    ui.ov.append(list);
    list.append(hint);
    this.setCam('big');
    this.setProgress(0, 0);
    t.setup = true;
    let latest = null;
    const off = t.on('frame', (m) => { latest = m; });
    let skipped = false;
    const skip = () => { skipped = true; };
    const offBlink = this.app.gaze.onDoubleBlinkFirst(() => { skipped = true; return true; });
    const onKey = (e) => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); skip(); } };
    window.addEventListener('keydown', onKey);
    ui.ov.addEventListener('click', skip);
    this.say(ref ? 'Sit where you sat when you calibrated' : 'Get comfortable',
      ref ? 'Follow the hints below — it makes the tracking more accurate. Can’t move? Blink twice to carry on.'
        : 'Sit the way you will use the computer and face the screen. Can’t move? Blink twice to carry on.',
      { top: true });
    const t0 = performance.now();
    let goodSince = null;
    let spokenHint = '';
    let hintSince = 0;
    let lastHintSpoken = 0;
    let shownHint = null;
    try {
      for (;;) {
        await this.wait(80);
        const now = performance.now();
        const r = assessPosition(latest, ref);
        const mark = (li, ok) => { li.className = ok ? 'ok' : 'warn'; };
        mark(items.face, r.face);
        mark(items.place, r.face && !r.place);
        mark(items.light, r.face && !r.light);
        goodSince = r.ok ? (goodSince ?? now) : null;
        ui.faceGuide.classList.toggle('ok', r.ok);
        const text = r.ok ? 'Perfect — stay just like that…' : r.hint;
        if (text !== shownHint) {
          shownHint = text;
          hintSince = now;
          hint.textContent = text;
        }
        // Say a hint once it has stayed for a moment (not every flicker).
        if (!r.ok && text && text !== spokenHint && now - hintSince > 1200 && now - lastHintSpoken > 3500) {
          spokenHint = text;
          lastHintSpoken = now;
          this.announce(text);
        }
        if (goodSince !== null && now - goodSince >= 1200) {
          sounds.point();
          break;
        }
        if (skipped) break;
        if (now - t0 > 20000) {
          toast('Carrying on — the hints will help another time.', '', 4000);
          break;
        }
      }
    } finally {
      off();
      offBlink();
      window.removeEventListener('keydown', onKey);
      ui.ov.removeEventListener('click', skip);
      t.setup = false;
      list.remove();
      ui.faceGuide.classList.remove('ok');
      this.setCam('corner');
      this.say('');
    }
  }

  // -- full calibration ---------------------------------------------------------------
  async runFull() {
    this.cancelled = false; // an Esc pressed on a previous screen must not cancel this run
    const started = await this.tracker.request({ type: 'calibration_start', mode: 'full' }, 'calibration_started');
    if (!started.ok) throw new Error(started.error || 'Could not start calibration');

    await this.positionCheck('full');
    await this.countdown('Look at each dot', 'Keep your head still. Each dot shrinks while you look at it.');
    this.say('');
    await this.showPoints(FULL_POINTS, 'cal');

    // Head movements: eyes stay on the centre dot while the head turns, nods
    // and tilts. Each pose is spoken in full, then held while a ring fills
    // with frames (gated on a real frame count, so it never races ahead).
    const c = this.placeDot(0.5, 0.5);
    const guide = h('div', { class: 'head-guide', style: { left: `${c.x}px`, top: `${c.y}px` } });
    this.ui.ov.append(guide);
    this.ui.dot.classList.add('settle');
    this.setProgress(HEAD_STEPS.length, 0);
    await this.sayAndWait('Now some slow head movements',
      'Keep your eyes on the dot the whole time. If moving is hard, just keep looking — that is fine.', 7000);
    const ring = (n, min) => this.setRing(n / min);
    try {
      for (let i = 0; i < HEAD_STEPS.length; i++) {
        const step = HEAD_STEPS[i];
        this.ui.dot.classList.remove('collect');
        await this.sayAndWait(step, '', 3500);
        await this.wait(500);          // a moment to get there
        this.ui.dot.classList.add('collect');
        this.setRing(0);
        let got = await this.collect('head', i, c.x, c.y, { minFrames: 20, minMs: 1000, maxMs: 4000, onProgress: ring });
        if (got < 8) {                 // the face was lost: one calmer try
          this.warn(null);
          await this.sayAndWait(step, 'Let’s try that one again — slowly.', 3500);
          this.setRing(0);
          got = await this.collect('head', i, c.x, c.y, { minFrames: 20, minMs: 1000, maxMs: 4000, onProgress: ring });
        }
        this.setRing(null);
        this.ui.dot.classList.remove('collect');
        if (got >= 8) {
          sounds.point();
          this.say(step, 'Good.', { top: true, voice: false });
        } else {
          this.say(step, 'Skipped — that’s fine.', { top: true, voice: false });
        }
        this.setProgress(HEAD_STEPS.length, i + 1);
        await this.wait(350);
      }
    } finally {
      this.tracker.setLabel(null);
      this.setRing(null);
      guide.remove();
    }

    // Train.
    this.ui.dot.className = 'calib-dot done';
    const fit = await this.train('full');
    if (!fit) return this.runFull();

    // Validate.
    await this.countdown('Almost done', 'Look at five more dots so we can measure the accuracy.');
    this.say('');
    await this.showPoints(VALIDATION_POINTS, 'val');
    this.say('Measuring accuracy…', h('div', { class: 'spinner' }));
    let result = await this.tracker.request({ type: 'validation_finish' }, 'validation_result', 90000);
    if (!result.ok) {
      toast(result.error || 'Validation failed', 'warn');
      return { mode: 'full', fit };
    }
    // Not good enough yet? More rounds of exercises until it is.
    result = await this.improveAccuracy(result);
    this.app.gaze.resetBias();
    if (result.saved === false) {
      toast('Could not save the calibration to disk — it works until you reload the page.', 'warn', 7000);
    }
    // The network is trained and saved by now: Esc here skips only the blink step.
    let blinks = null;
    try {
      blinks = await this.blinkStep();
    } catch (err) {
      if (!(err instanceof Cancelled)) throw err;
      this.cancelled = false;
      toast('Skipped the blink step — kept your blink settings');
    }
    if (blinks && !blinks.ok) toast(`Kept the standard blink settings: ${blinks.error}`, 'warn', 6000);
    for (;;) {
      const choice = await this.showResults(result);
      if (choice === 'again') return this.runFull();
      if (choice !== 'better') break;
      // One more round on request, even when the accuracy is already good.
      const more = await this.improveAccuracy(result, { maxRounds: 1, force: true });
      result = { ...more, rounds: [...(result.rounds || []), ...more.rounds],
        first_error_px: result.first_error_px ?? more.first_error_px };
      this.app.gaze.resetBias();
    }
    return { mode: 'full', fit, validation: result };
  }

  /**
   * While the measured accuracy is below "good", run up to REFINE_ROUNDS more
   * rounds: extra dots where the tracking was least sure (a different kind of
   * exercise each round), a network trained with them, and new dots to measure
   * it. The server measures the new and the previous network on the same new
   * dots and keeps the better one, so a round never makes things worse.
   * Resolves to the final measurement, with `rounds` and `first_error_px`.
   */
  async improveAccuracy(result, { maxRounds = REFINE_ROUNDS, force = false } = {}) {
    const rounds = [];
    let current = result;
    const labels = { excellent: 'Excellent', good: 'Good', fair: 'Fair', poor: 'Poor' };
    const diag = Math.hypot(window.innerWidth, window.innerHeight);
    let stalled = 0;
    for (let round = 1; round <= maxRounds; round++) {
      const rating = rateAccuracy(current.mean_error_px);
      // `force`: one round on request even though the accuracy is good already.
      if ((rating === 'good' || rating === 'excellent') && !(force && round === 1)) break;
      await this.sayAndWait(round === 1 ? 'Let’s make it more accurate' : 'One more round',
        `${labels[rating]} so far. A few more dots, where the tracking was least sure.`, 7000);
      // A shaky cursor usually means dim or uneven light: check the setup again.
      if (current.precision_px > 0.03 * diag) await this.positionCheck('full');
      const started = await this.tracker.request({ type: 'calibration_start', mode: 'refine' }, 'calibration_started');
      if (!started.ok) break;
      const measured = (current.points || []).map((p) => {
        const c = screenToClient(p.target[0], p.target[1]);
        return { x: c.x / window.innerWidth, y: c.y / window.innerHeight, error: p.error };
      });
      await this.countdown('Look at each dot', 'Keep your head still, like before.');
      this.say('');
      await this.showPoints(refinePoints(measured, round), 'cal', { pointBase: started.point_base });
      this.say('Training your gaze neural network…', h('div', { class: 'spinner' }));
      const fit = await this.tracker.request({ type: 'calibration_fit', mode: 'full' }, 'calibration_result', 90000);
      if (!fit.ok) break;
      await this.countdown('Checking the accuracy', 'Five new dots.');
      this.say('');
      await this.showPoints(ROUND_VALIDATION[(round - 1) % ROUND_VALIDATION.length], 'val');
      this.say('Measuring accuracy…', h('div', { class: 'spinner' }));
      const res = await this.tracker.request({ type: 'validation_finish' }, 'validation_result', 90000);
      if (!res.ok) break;
      rounds.push({ round, ...(res.refine || {}), error_px: res.mean_error_px });
      stalled = res.refine && res.refine.kept === 'previous' ? stalled + 1 : 0;
      current = res;
      if (stalled >= 2) break;     // two rounds in a row without a gain: more won't help now
    }
    this.say('');
    return { ...current, rounds, first_error_px: result.mean_error_px };
  }

  /** After a quick adjust: measure the accuracy on five dots, and offer more
   *  rounds when it is below "good". */
  async checkAccuracy() {
    await this.countdown('Checking the accuracy', 'Five more dots.');
    this.say('');
    await this.showPoints(VALIDATION_POINTS, 'val');
    this.say('Measuring accuracy…', h('div', { class: 'spinner' }));
    const res = await this.tracker.request({ type: 'validation_finish', refit: false }, 'validation_result', 60000);
    this.say('');
    if (!res.ok) return null;
    const rating = rateAccuracy(res.mean_error_px);
    if (rating === 'good' || rating === 'excellent') return res;
    const ov = this.ui.ov;
    ov.classList.remove('solid');
    this.app.gaze.setSuspended(false);
    const choice = await this.app.choose(ov, {
      title: `Accuracy: ${rating} (${Math.round(res.mean_error_px)} px)`,
      subtitle: 'A few rounds of extra dots can make it better (about half a minute each).',
      choices: [
        { id: 'improve', label: 'Improve it now', sub: 'Recommended', icon: 'sparkle', primary: true },
        { id: 'later', label: 'Not now', sub: 'Use it as it is', icon: 'arrowRight' },
      ],
    });
    this.app.gaze.setSuspended(true);
    ov.classList.add('solid');
    if (choice !== 'improve') return res;
    const improved = await this.improveAccuracy(res);
    this.app.gaze.resetBias();
    const show = (px) => `${Math.round(px)} px`;
    toast(`Accuracy ${show(res.mean_error_px)} → ${show(improved.mean_error_px)} (${rateAccuracy(improved.mean_error_px)})`,
      'ok', 7000);
    return improved;
  }

  async train(mode) {
    this.say('Training your gaze neural network…', h('div', { class: 'nn-anim', html: '<span><i></i><i></i><i></i><i></i></span><span><i></i><i></i><i></i><i></i><i></i></span><span><i></i><i></i><i></i></span><span><i></i><i></i></span>' }));
    this.ui.text.querySelectorAll('.nn-anim i').forEach((n, i) => { n.style.animationDelay = `${(i % 5) * 0.12}s`; });
    const res = await this.tracker.request({ type: 'calibration_fit', mode }, 'calibration_result', 90000);
    if (res.ok) return res;
    // Failed: offer to retry (double blink works without calibration).
    this.say('Calibration didn’t work', h('div', {},
      h('p', {}, res.error || 'Not enough data.'),
      h('p', {}, 'Blink twice (or click) to try again. Press Esc to cancel.')));
    await this.waitForDoubleBlinkOrClick();
    return null;
  }

  waitForDoubleBlinkOrClick() {
    return new Promise((resolve, reject) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        offBlink();
        this.ui.ov.removeEventListener('click', finish);
        clearInterval(timer);
        resolve();
      };
      const offBlink = this.app.gaze.onDoubleBlinkFirst(() => { finish(); return true; });
      this.ui.ov.addEventListener('click', finish);
      const timer = setInterval(() => {
        if (this.cancelled && !done) {
          done = true;
          offBlink();
          clearInterval(timer);
          reject(new Cancelled());
        }
      }, 100);
    });
  }

  /** Show the accuracy and ask what to do. Resolves true to recalibrate. */
  async showResults(result) {
    const rating = rateAccuracy(result.mean_error_px);
    const labels = { excellent: 'Excellent', good: 'Good', fair: 'Fair', poor: 'Poor' };
    const map = h('div', { class: 'results-map' });
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 100 100');
    svg.setAttribute('preserveAspectRatio', 'none');
    map.append(svg);
    for (const p of result.points || []) {
      const t = screenToClient(p.target[0], p.target[1]);
      const m = screenToClient(p.mean[0], p.mean[1]);
      const tx = (t.x / window.innerWidth) * 100;
      const ty = (t.y / window.innerHeight) * 100;
      const mx = Math.max(0, Math.min(100, (m.x / window.innerWidth) * 100));
      const my = Math.max(0, Math.min(100, (m.y / window.innerHeight) * 100));
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      Object.entries({ x1: tx, y1: ty, x2: mx, y2: my, stroke: '#a78bfa', 'stroke-width': 0.4, 'vector-effect': 'non-scaling-stroke' })
        .forEach(([k, v]) => line.setAttribute(k, v));
      svg.append(line);
      map.append(h('i', { class: 't', style: { left: `${tx}%`, top: `${ty}%` } }));
      map.append(h('i', { class: 'p', style: { left: `${mx}%`, top: `${my}%` } }));
    }
    const dropped = ((result.refit || {}).dropped_points || []).length;
    const rounds = result.rounds || [];
    const history = rounds.length
      ? h('p', {}, `${rounds.length} extra round${rounds.length > 1 ? 's' : ''}: `
        + `${Math.round(result.first_error_px)} px → ${Math.round(result.mean_error_px)} px`
        + (result.mean_error_px < result.first_error_px - 1 ? '.' : ' (the first network stayed the best).'))
      : null;
    const summary = h('div', {},
      h('div', { class: 'eyebrow' }, 'Calibration complete'),
      h('div', { class: `score ${rating}` }, labels[rating]),
      h('p', { class: 'muted' }, `Average error ${Math.round(result.mean_error_px)} px, jitter ${Math.round(result.precision_px)} px. `,
        rating === 'poor' || rating === 'fair'
          ? 'Tip: sit centred in front of the camera with your face evenly lit, keep glasses free of reflections, '
            + 'and look right at each dot until it is gone — then try again.'
          : 'Rings show the dots, purple points where the network thinks you looked.'),
      history,
      dropped ? h('p', { class: 'muted' }, `${dropped} dot${dropped > 1 ? 's were' : ' was'} left out: your eyes seemed to be elsewhere then.`) : null,
    );
    const ov = this.ui.ov;
    ov.classList.remove('solid');
    this.say('');
    this.setProgress(0, 0);
    const card = h('div', { class: 'overlay-card', style: { position: 'fixed', top: '4vh', left: '50%', translate: '-50% 0', width: 'min(1100px, 92vw)' } },
      h('div', { class: 'results' }, map, summary));
    ov.append(card);
    // The network is trained: let the gaze cursor drive this screen (on a
    // first calibration it has never been switched on before).
    this.app.gaze.setActive(true);
    this.app.gaze.setSuspended(false);
    const choices = [
      { id: 'again', label: 'Recalibrate', sub: 'Run the calibration again', icon: 'refresh' },
      { id: 'go', label: 'Start browsing', sub: 'Look here and blink twice', icon: 'check', primary: true },
    ];
    // Not excellent yet: one more round of exercises is always on offer.
    if (rating !== 'excellent') {
      choices.splice(1, 0, { id: 'better', label: 'Make it even better', sub: 'A round of extra dots', icon: 'sparkle' });
    }
    const choice = await this.app.choose(ov, { top: false, choices, rowStyle: { paddingTop: '48vh' } });
    card.remove();
    this.app.gaze.setSuspended(true);
    ov.classList.add('solid');
    if (choice === 'go') sounds.success();
    return choice;
  }

  // -- blinks --------------------------------------------------------------------------------
  /** Blink twice whenever the dot turns purple (3 times): personalises the blink detector. */
  async blinkStep() {
    const started = await this.tracker.request({ type: 'blink_calibration_start' }, 'blink_calibration_started')
      .catch(() => null);
    if (!started || !started.ok) return null;
    this.placeDot(0.5, 0.5);
    this.ui.dot.classList.add('settle');
    try {
      await this.sayAndWait('Last step: your blinks', 'Each time the dot turns purple, blink twice — like a relaxed “yes, yes”.', 6000);
      await this.wait(400);
      for (let i = 0; i < 3; i++) {
        this.setProgress(3, i);
        this.ui.dot.classList.add('blink-now');
        sounds.point();
        this.say('Blink twice now', '', { top: true });
        await this.wait(1700);
        this.ui.dot.classList.remove('blink-now');
        this.say('', '');
        await this.wait(1300);
      }
    } catch (err) {
      // Stop recording without analysing (the person's settings stay as they were).
      this.tracker.send({ type: 'blink_calibration_cancel' });
      throw err;
    }
    this.setProgress(3, 3);
    this.ui.dot.className = 'calib-dot done';
    this.say('Learning your blinks…', h('div', { class: 'spinner' }));
    const res = await this.tracker.request({ type: 'blink_calibration_finish' }, 'blink_calibration_result', 20000)
      .catch((err) => ({ ok: false, error: err.message }));
    this.say('');
    return res;
  }

  async runBlinkTest() {
    this.cancelled = false;
    for (;;) {
      const res = await this.blinkStep();
      if (!res) throw new Error('Could not start the blink test');
      if (res.ok) {
        sounds.success();
        const which = { both: 'both eyes', mean: 'both eyes (averaged)', left: 'your left eye', right: 'your right eye' };
        toast(`Blinks learned: ${res.n_pairs} double blinks, watching ${which[res.signal] || 'your eyes'}`, 'ok', 6000);
        return { mode: 'blink', fit: res };
      }
      this.say('I couldn’t see clear double blinks', h('div', {},
        h('p', {}, res.error || ''),
        h('p', {}, 'Blink twice (or click) to try again. Press Esc to keep the current settings.')));
      await this.waitForDoubleBlinkOrClick();
    }
  }

  // -- winks ---------------------------------------------------------------------------------
  /** Close one eye, then the other: learns whether and how each eye winks. */
  async runWinkTest() {
    this.cancelled = false;
    for (;;) {
      const started = await this.tracker.request({ type: 'wink_calibration_start' }, 'wink_calibration_started');
      if (!started.ok) throw new Error(started.error || 'Could not start the wink test');
      this.placeDot(0.5, 0.5, { instant: true });
      this.ui.dot.classList.add('settle');
      await this.sayAndWait('Let’s see how you wink', 'Look at the dot. You’ll close one eye at a time and keep it closed until the ring is full. If you can’t, that’s fine — there are other ways.', 9000);
      const ring = h('div', { class: 'wink-ring', style: { left: '50%', top: '50%' } });
      this.ui.ov.append(ring);
      let finished = false;
      const steps = [
        ['rest', 1800, 'Keep both eyes open'],
        ['left', 2800, 'Close your LEFT eye', 'Keep your right eye open'],
        ['rest', 1900, 'Open both eyes'],
        ['right', 2800, 'Now close your RIGHT eye', 'Keep your left eye open'],
        ['rest', 1600, 'Open both eyes'],
      ];
      try {
        for (const [phase, ms, title, body] of steps) {
          this.tracker.setGesturePhase(phase);
          this.say(title, body || '', { top: true });
          this.ui.dot.classList.toggle('wink-now', phase !== 'rest');
          if (phase !== 'rest') sounds.point();
          const t0 = performance.now();
          while (performance.now() - t0 < ms) {
            ring.style.setProperty('--p', phase === 'rest' ? '0' : String(Math.min(1, (performance.now() - t0) / ms)));
            await this.wait(40);
          }
        }
        finished = true;
      } finally {
        this.tracker.setGesturePhase(null);
        ring.remove();
        if (!finished) this.tracker.send({ type: 'wink_calibration_cancel' });
      }
      this.ui.dot.className = 'calib-dot done';
      this.say('Checking…', h('div', { class: 'spinner' }));
      const res = await this.tracker.request({ type: 'wink_calibration_finish' }, 'wink_calibration_result', 20000);
      this.say('');
      const choice = await this.showWinkResults(res);
      if (choice === 'again') continue;
      if (choice === 'long_close') {
        await this.tracker.request({ type: 'gestures_set', gestures: { long_close: 'menu' } }, 'personal', 8000)
          .catch(() => {});
        toast('Close both eyes for about a second to open the menu (pick up, click, read aloud).', 'ok', 7000);
      }
      return { mode: 'wink', fit: res };
    }
  }

  async showWinkResults(res) {
    const ov = this.ui.ov;
    const card = h('div', { class: 'overlay-card', style: { position: 'fixed', top: '5vh', left: '50%', translate: '-50% 0', width: 'min(1000px, 92vw)' } });
    const eyes = h('div', { class: 'eye-results' });
    let anyOk = false;
    if (res.ok) {
      for (const eye of ['left', 'right']) {
        const r = res[eye] || {};
        anyOk = anyOk || !!r.ok;
        eyes.append(h('div', { class: `eye-result ${r.ok ? 'ok' : 'no'}` },
          h('h3', { html: `${icon(r.ok ? 'check' : 'eye')}<span> ${eye === 'left' ? 'Left' : 'Right'} eye</span>` }),
          h('p', {}, r.ok ? 'Winks work — hold it closed to drag, keep still for the menu.'
            : `Not used for gestures: ${r.reason || 'no clear wink'}.`)));
      }
    }
    card.append(h('div', { class: 'eyebrow' }, 'Wink test'),
      h('h1', {}, res.ok ? (anyOk ? 'Your winks are set up' : 'Winking seems hard — no problem') : 'The wink test didn’t work'),
      res.ok ? eyes : h('p', { class: 'muted' }, res.error || ''),
      h('p', { class: 'muted' }, anyOk ? 'Try it on the Arrange page: look at a planet, close that eye, look at a slot, open it.'
        : 'You can close both eyes for about a second instead, or rest your eyes on a button (dwell click in Settings).'));
    ov.classList.remove('solid');
    ov.append(card);
    this.app.gaze.setSuspended(false);
    const choices = [
      { id: 'done', label: 'Done', icon: 'check', primary: anyOk },
      { id: 'again', label: 'Try again', icon: 'refresh' },
    ];
    if (res.ok && !anyOk) choices.unshift({ id: 'long_close', label: 'Use “close both eyes”', sub: 'For about a second', icon: 'blink', primary: true });
    const choice = await this.app.choose(ov, { choices, rowStyle: { paddingTop: '56vh' } });
    card.remove();
    this.app.gaze.setSuspended(true);
    ov.classList.add('solid');
    return choice;
  }

  // -- head directions (for the head nudge) ------------------------------------------------
  /** Turn right, then tip up: learns which way of the head pose means right and
   *  up for this person (and camera), and how far they comfortably tilt (which
   *  sets the dead zone), then switches the head nudge on. */
  async runHeadTest() {
    this.cancelled = false;
    let latest = null;
    const off = this.tracker.on('frame', (m) => {
      if (m.face && Array.isArray(m.head)) latest = { head: m.head, at: performance.now() };
    });
    const sample = async (ms) => {
      const rows = [];
      const end = performance.now() + ms;
      while (performance.now() < end) {
        await this.wait(40);
        if (latest && performance.now() - latest.at < 250) rows.push(latest.head);
      }
      return rows;
    };
    const median = (rows, i) => {
      const v = rows.map((r) => r[i]).sort((a, b) => a - b);
      return v.length ? v[v.length >> 1] : null;
    };
    try {
      this.placeDot(0.5, 0.5, { instant: true });
      this.ui.dot.classList.add('settle');
      await this.sayAndWait('Which way is which?', 'Look at the dot with your head straight.', 5000);
      const rest = await sample(1000);
      await this.sayAndWait('Turn your head a little to your right', 'and hold it there.', 4000);
      const right = await sample(1200);
      await this.sayAndWait('Back to the middle', '', 3000);
      const rest2 = await sample(900);
      await this.sayAndWait('Now tip your head up a little', 'and hold it there.', 4000);
      const up = await sample(1200);
      this.say('');
      const dYaw = median(right, 0) - median(rest, 0);
      const dPitch = median(up, 1) - median(rest2, 1);
      if (!Number.isFinite(dYaw) || !Number.isFinite(dPitch) || Math.abs(dYaw) < 3 || Math.abs(dPitch) < 3) {
        toast('I couldn’t see your head move enough for the head nudge — it stays as it was. You can try again any time.', 'warn', 8000);
        return { mode: 'head', fit: { ok: false } };
      }
      // A dead zone well inside the comfortable tilt, so small tilts are enough.
      const reach = Math.min(Math.abs(dYaw), Math.abs(dPitch));
      const patch = {
        head_nudge: true,
        nudge_right: dYaw < 0 ? -1 : 1,
        nudge_up: dPitch < 0 ? -1 : 1,
        nudge_deadzone: Math.round(Math.max(2, Math.min(8, 0.4 * reach))),
      };
      const reply = await this.tracker.request({ type: 'gestures_set', gestures: patch }, 'personal', 8000);
      if (reply.ok === false) throw new Error(reply.error || 'could not save');
      sounds.success();
      toast('Head nudge is on: tilt your head a little to move the cursor, hold it straight to stop.', 'ok', 8000);
      return { mode: 'head', fit: { ok: true, ...patch } };
    } finally {
      off();
    }
  }

  // -- quick adjust (eyes only) ----------------------------------------------------------
  //
  // Back to the calibrated seating position, then nine dots: each network gets
  // an affine correction for today (quick and stable, it cannot destabilise
  // the network). Works with the eyes alone.
  async runAdjust() {
    this.cancelled = false;
    const started = await this.tracker.request({ type: 'calibration_start', mode: 'adjust' }, 'calibration_started');
    if (!started.ok) throw new Error(started.error || 'Could not start adjustment');
    await this.positionCheck('adjust');
    await this.countdown('Quick adjust', 'Look at each dot until it shrinks away.');
    this.say('');
    await this.showPoints(ADJUST_POINTS, 'adjust');
    const adjusted = await this.finishAdjust();
    if (adjusted.fit && adjusted.fit.ok) {
      const measured = await this.checkAccuracy();
      if (measured) adjusted.validation = measured;
    }
    return adjusted;
  }

  // -- mouse-guided tune-up: continuous, mouse-labelled pursuit ----------------------------
  //
  // For a helper, or anyone who can use a mouse. A dot glides slowly along a
  // procedural path covering the whole screen while the user follows it with
  // their eyes AND keeps the mouse pointer where they are looking. The MOUSE
  // position is the ground-truth label (more honest than assuming perfect
  // dot-following): every frame, if the pointer is near the dot and moving
  // slowly (so the user is really tracking), we label the frame with the live
  // mouse position. The dense (eye-features -> mouse-position) pairs fit the
  // affine correction.
  async runPursuitAdjust() {
    this.cancelled = false;
    const started = await this.tracker.request({ type: 'calibration_start', mode: 'adjust' }, 'calibration_started');
    if (!started.ok) throw new Error(started.error || 'Could not start adjustment');

    await this.sayAndWait('Mouse-guided tune-up',
      'Follow the dot with your eyes — and move the mouse so the pointer stays where you are looking.', 7000);

    let path = null;
    try { path = await this.fetchTrail('lissajous'); } catch { /* fall back below */ }
    if (!path || path.length < 2) {
      // The trail API was unavailable: fall back to the dots.
      this.say('');
      await this.showPoints(ADJUST_POINTS, 'adjust');
      return this.finishAdjust();
    }

    // Show the live gaze estimate (so the user sees tracker vs. truth).
    const gazeMark = h('div', { class: 'calib-gaze-mark', hidden: true });
    this.ui.ov.append(gazeMark);
    const offFrame = this.tracker.on('frame', (m) => {
      const g = (m.face && m.gaze) ? screenToClient(m.gaze[0], m.gaze[1]) : null;
      gazeMark.hidden = !g;
      if (g) gazeMark.style.translate = `${g.x}px ${g.y}px`;
    });

    // Track the mouse (the ground-truth label) and its speed.
    let mx = window.innerWidth / 2, my = window.innerHeight / 2, mt = performance.now();
    let mvx = 0, mvy = 0;
    const onMove = (e) => {
      const now = performance.now();
      const dt = Math.max(1e-3, (now - mt) / 1000);
      mvx = (e.clientX - mx) / dt; mvy = (e.clientY - my) / dt;
      mx = e.clientX; my = e.clientY; mt = now;
    };
    window.addEventListener('mousemove', onMove);

    this.say('Follow the dot', 'Keep the mouse pointer where you are looking.', { top: true });
    const DURATION = 20000;
    const diag = Math.hypot(window.innerWidth, window.innerHeight);
    const nearR = 0.17 * diag;        // pointer must be near the dot (really tracking)
    const maxSpeed = 2.2 * diag;      // px/s: skip flung/!tracking frames
    let collected = 0;
    const start = performance.now();
    try {
      for (;;) {
        this.checkCancel();
        const now = performance.now();
        const u = (now - start) / DURATION;
        if (u >= 1) break;
        const [fx, fy] = this.sampleTrail(path, u);
        const dx = fx * window.innerWidth, dy = fy * window.innerHeight;
        this.ui.dot.className = 'calib-dot collect';
        this.ui.dot.style.translate = `${dx}px ${dy}px`;
        this.setRing(u);
        // Guards: pointer near the dot and moving slowly, else don't collect.
        const nearDot = Math.hypot(mx - dx, my - dy) < nearR;
        const slow = Math.hypot(mvx, mvy) < maxSpeed;
        if (nearDot && slow) {
          const s = clientToScreen(mx, my);
          this.tracker.setLabel({ x: s.x, y: s.y, kind: 'adjust', pt: Math.floor((now - start) / 300) });
          collected++;
          this.warn(null);
        } else {
          this.tracker.setLabel(null);
          this.warn(nearDot ? null : 'Keep the mouse pointer on the dot');
        }
        await this.raf();
      }
    } finally {
      this.tracker.setLabel(null);
      this.setRing(null);
      window.removeEventListener('mousemove', onMove);
      offFrame();
      gazeMark.remove();
    }

    if (collected < 20) {
      toast('I didn’t get enough tracking — try again and keep the pointer on the dot.', 'warn', 6000);
      return { mode: 'adjust', fit: { ok: false, error: 'not enough data' } };
    }
    return this.finishAdjust();
  }

  async finishAdjust() {
    this.say('Tuning…', h('div', { class: 'spinner' }));
    const res = await this.tracker.request({ type: 'calibration_fit', mode: 'adjust' }, 'calibration_result', 30000);
    this.app.gaze.resetBias();
    if (res.ok) {
      const show = (px) => `${Math.round(px)} px (≈${(px / 37.8).toFixed(1)} cm)`;
      toast(`Tuned — error ${show(res.error_before_px)} → ${show(res.error_after_px)}`, 'ok', 6000);
      sounds.success();
    } else {
      toast(res.error || 'Tune-up failed', 'warn');
    }
    return { mode: 'adjust', fit: res };
  }

  // Fetch a procedural pursuit path (normalised points) from the server.
  async fetchTrail(kind) {
    const res = await fetch(`/api/trail/${kind}?n=400&seed=${Date.now() % 997}`);
    if (!res.ok) throw new Error('trail unavailable');
    const data = await res.json();
    return (data.points || []).map((p) => [p.x, p.y]);
  }

  // Position along the polyline at u in [0, 1] (linear between waypoints).
  sampleTrail(points, u) {
    const f = Math.max(0, Math.min(1, u)) * (points.length - 1);
    const i = Math.floor(f);
    const t = f - i;
    const a = points[i];
    const b = points[Math.min(points.length - 1, i + 1)];
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  }

  raf() { return new Promise((r) => requestAnimationFrame(() => r())); }
}
