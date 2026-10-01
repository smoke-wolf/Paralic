// Calibration: teach GazeNet where on the screen you are looking.
//
// Full calibration (about 30 seconds):
//   1. follow a dot across 13 positions while keeping your head still,
//   2. keep looking at the centre dot while gently moving your head
//      (teaches the network to compensate for head movement),
//   3. the server trains your personal neural network,
//   4. five more dots measure the accuracy (and are then used for a final fit).
//
//   5. blink twice three times when the dot turns purple: learns how you
//      blink (thresholds, timing, and which eye signal to watch).
//
// Quick adjust (about 8 seconds): 5 dots that correct a saved calibration for
// today's seating position.
//
// Blink test / wink test: the blink step on its own, and "close your left
// eye, now your right eye" to learn how (and whether) each eye winks.

import { h, sleep, toast } from './dom.js';
import { icon } from './icons.js';
import { clientToScreen, screenToClient } from './screen-space.js';
import { sounds } from './sound.js';
import { speak, canSpeak } from './speech.js';
import { drawMesh } from './camera-panel.js';

class Cancelled extends Error {}

// A denser 5x4 grid (plus the centre) than the old 13 dots: more positions make
// the gaze network interpolate better across the whole screen. Snake-ordered so
// the dot only ever makes short hops.
function gridPoints() {
  const xs = [0.06, 0.28, 0.5, 0.72, 0.94];
  const ys = [0.08, 0.37, 0.63, 0.92];
  const pts = [];
  ys.forEach((y, row) => {
    const rowXs = row % 2 === 0 ? xs : [...xs].reverse(); // snake order: short hops
    rowXs.forEach((x) => pts.push([x, y]));
  });
  pts.push([0.5, 0.5]);
  return pts;
}

const FULL_POINTS = gridPoints();
const VALIDATION_POINTS = [[0.27, 0.28], [0.73, 0.28], [0.5, 0.55], [0.73, 0.8], [0.27, 0.8]];
const ADJUST_POINTS = [[0.5, 0.5], [0.12, 0.14], [0.88, 0.14], [0.88, 0.86], [0.12, 0.86]];
// Explicit face tilt / rotation sub-steps. Keeping the eyes on the centre dot
// while the head moves teaches the network to compensate for head pose: yaw
// (turn), pitch (nod) and roll (tilt). Each is held long enough to gather
// frames across the movement. [text, hold ms].
const HEAD_STEPS = [
  ['Keep your eyes on the dot, and keep them there the whole time.', 2000],
  ['Slowly turn your head to the left — eyes still on the dot.', 2600],
  ['And slowly turn your head to the right.', 2600],
  ['Back to centre. Now slowly tip your head up, like a small nod up.', 2600],
  ['And slowly tip your head down.', 2600],
  ['Back to centre. Now tilt your head towards your left shoulder.', 2600],
  ['And tilt towards your right shoulder.', 2600],
  ['Lovely — back to the middle. Keep looking at the dot.', 2000],
];

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
      if (mode === 'blink') return await this.runBlinkTest();
      if (mode === 'wink') return await this.runWinkTest();
      return await this.runFull();
    } catch (err) {
      if (err instanceof Cancelled) {
        toast('Calibration cancelled');
        return null;
      }
      throw err;
    } finally {
      this.stopMeshPreview();
      this.tracker.setLabel(null);
      this.tracker.setGesturePhase(null);
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
    // Live camera preview with the full face mesh drawn on top, shown for the
    // whole calibration so the user can see their face and that tracking works.
    const video = h('video', { class: 'calib-cam-video', autoplay: true, muted: true, playsinline: true });
    const meshCanvas = h('canvas', { class: 'calib-cam-mesh' });
    const camLabel = h('div', { class: 'calib-cam-label' }, 'Your face');
    const cam = h('div', { class: 'calib-cam' }, video, meshCanvas, camLabel);
    // A ring that fills as good frames are gathered for the current step.
    const ring = h('div', { class: 'calib-ring', hidden: true },
      h('svg', { viewBox: '0 0 48 48', html:
        '<circle class="track" cx="24" cy="24" r="21"></circle>' +
        '<circle class="fill" cx="24" cy="24" r="21"></circle>' }));
    // Visible transcript of the spoken guidance (accessibility; also helps when
    // speech is off or unsupported). Lives at the bottom, out of the way.
    const transcript = h('div', { class: 'calib-transcript', 'aria-live': 'polite' });
    ov.append(dot, text, warning, progress, cam, ring, transcript);
    return { ov, dot, text, warning, progress, transcript, cam, video, meshCanvas, ring };
  }

  /** Show the camera + live face mesh for the whole calibration. */
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
    if (canSpeak()) speak(line);
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
    return { x, y };
  }

  async countdown(title, body) {
    this.say(title, body, { top: true });
    const n = h('div', { class: 'calib-count' });
    for (const k of [3, 2, 1]) {
      n.textContent = String(k);
      this.ui.text.append(n);
      await this.wait(750);
    }
    n.remove();
  }

  /** Label frames with the dot position until enough good frames arrived.
   *  ``onProgress(count, minFrames)`` is called as frames are gathered (used to
   *  fill the progress ring during the head-pose steps). */
  async collect(kind, point, x, y, { minFrames = 20, minMs = 750, maxMs = 3000, onProgress = null } = {}) {
    let count = 0;
    let missing = 0;
    const off = this.tracker.on('frame', (m) => {
      if (m.labeled) count++;
      missing = m.face ? 0 : missing + 1;
      if (onProgress) onProgress(count, minFrames);
    });
    const s = clientToScreen(x, y);
    this.tracker.setLabel({ x: s.x, y: s.y, kind, pt: point });
    const t0 = performance.now();
    try {
      for (;;) {
        await this.wait(40);
        const elapsed = performance.now() - t0;
        this.warn(missing > 4 ? 'I can’t see your face — look at the screen and check the lighting' : null);
        if ((elapsed >= minMs && count >= minFrames) || elapsed >= maxMs) break;
      }
    } finally {
      this.tracker.setLabel(null);
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

  /** A slow, visible countdown ("Hold it… 3, 2, 1") with spoken sync. */
  async holdCountdown(seconds = 3) {
    const n = h('div', { class: 'calib-count' });
    this.ui.text.append(n);
    for (let k = seconds; k >= 1; k--) {
      n.textContent = String(k);
      await this.wait(1000);
    }
    n.remove();
  }

  warn(message) {
    const { warning } = this.ui;
    warning.hidden = !message;
    if (message) warning.textContent = message;
  }

  async showPoints(points, kind, opts = {}) {
    this.setProgress(points.length, 0);
    for (let i = 0; i < points.length; i++) {
      const [fx, fy] = points[i];
      const { x, y } = this.placeDot(fx, fy, { instant: i === 0 });
      await this.wait(i === 0 ? 650 : 820);
      this.ui.dot.classList.add('settle');
      await this.wait(opts.settleMs ?? 520);
      this.ui.dot.classList.add('collect');
      let n = await this.collect(kind, i, x, y, opts);
      if (n < 5) n = await this.collect(kind, i, x, y, { ...opts, maxMs: 3000 }); // one retry
      sounds.point();
      this.setProgress(points.length, i + 1);
    }
    this.ui.dot.className = 'calib-dot done';
  }

  // -- full calibration ---------------------------------------------------------------
  async runFull() {
    this.cancelled = false; // an Esc pressed on a previous screen must not cancel this run
    const started = await this.tracker.request({ type: 'calibration_start', mode: 'full' }, 'calibration_started');
    if (!started.ok) throw new Error(started.error || 'Could not start calibration');

    await this.countdown('Follow the dot with your eyes', 'Keep your head still. Each dot shrinks while it is measuring.');
    this.say('');
    await this.showPoints(FULL_POINTS, 'cal');

    // Head-movement phase — deliberately slow. Each pose is held until enough
    // good frames are actually gathered (a filling ring shows the progress), so
    // the step never races ahead before the data is in. Short countdowns and
    // pauses sit between steps, with the instructions spoken in sync.
    const c = this.placeDot(0.5, 0.5);
    const guide = h('div', { class: 'head-guide', style: { left: `${c.x}px`, top: `${c.y}px` } });
    this.ui.ov.append(guide);
    this.ui.dot.classList.add('settle');
    this.setProgress(HEAD_STEPS.length, 0);
    this.say('Now the head movements', 'Keep your eyes on the dot the whole time, and move slowly. If moving is hard, just keep looking — that is fine.', { top: true });
    await this.wait(3200);
    const s = clientToScreen(c.x, c.y);
    try {
      for (let i = 0; i < HEAD_STEPS.length; i++) {
        const [step] = HEAD_STEPS[i];
        this.ui.dot.classList.remove('collect');
        // Announce the move and give the person a moment to get there.
        this.say(step, '', { top: true });
        await this.wait(1500);
        this.say(step, 'Hold it there…', { top: true, voice: false });
        await this.holdCountdown(2);
        // Now gather frames for this pose, gated on a real frame count.
        this.ui.dot.classList.add('collect');
        this.setRing(0);
        const got = await this.collect('head', i, c.x, c.y,
          { minFrames: 26, minMs: 1500, maxMs: 5000, onProgress: (n, min) => this.setRing(n / min) });
        if (got < 8) {   // face was lost — give it one more, calmer try
          this.warn(null);
          this.say(step, 'Let’s try that one again — hold still.', { top: true });
          await this.wait(900);
          this.setRing(0);
          await this.collect('head', i, c.x, c.y,
            { minFrames: 20, minMs: 1200, maxMs: 5000, onProgress: (n, min) => this.setRing(n / min) });
        }
        this.setRing(1);
        sounds.point();
        this.setProgress(HEAD_STEPS.length, i + 1);
        this.setRing(null);
        // A brief rest before the next move.
        this.ui.dot.classList.remove('collect');
        this.say(step, 'Good.', { top: true, voice: false });
        await this.wait(700);
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
    this.say('Almost done', 'Look at five more dots so we can measure the accuracy.');
    await this.wait(1500);
    this.say('');
    await this.showPoints(VALIDATION_POINTS, 'val', { minFrames: 14 });
    this.say('Measuring accuracy…', h('div', { class: 'spinner' }));
    const result = await this.tracker.request({ type: 'validation_finish' }, 'validation_result', 90000);
    if (!result.ok) {
      toast(result.error || 'Validation failed', 'warn');
      return { mode: 'full', fit };
    }
    this.app.gaze.resetBias();
    if (result.saved === false) {
      toast('Could not save the calibration to disk — it works until you reload the page.', 'warn', 7000);
    }
    const blinks = await this.blinkStep();
    if (blinks && !blinks.ok) toast(`Kept the standard blink settings: ${blinks.error}`, 'warn', 6000);
    const again = await this.showResults(result);
    if (again) return this.runFull();
    return { mode: 'full', fit, validation: result };
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
    const summary = h('div', {},
      h('div', { class: 'eyebrow' }, 'Calibration complete'),
      h('div', { class: `score ${rating}` }, labels[rating]),
      h('p', { class: 'muted' }, `Average error ${Math.round(result.mean_error_px)} px, jitter ${Math.round(result.precision_px)} px. `,
        rating === 'poor' || rating === 'fair'
          ? 'Tip: sit centred in front of the camera with your face evenly lit, then recalibrate.'
          : 'Rings show the dots, purple points where the network thinks you looked.'),
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
    const choice = await this.app.choose(ov, {
      top: false,
      choices: [
        { id: 'again', label: 'Recalibrate', sub: 'Run the calibration again', icon: 'refresh' },
        { id: 'go', label: 'Start browsing', sub: 'Look here and blink twice', icon: 'check', primary: true },
      ],
      rowStyle: { paddingTop: '48vh' },
    });
    card.remove();
    this.app.gaze.setSuspended(true);
    ov.classList.add('solid');
    if (choice === 'go') sounds.success();
    return choice === 'again';
  }

  // -- blinks --------------------------------------------------------------------------------
  /** Blink twice whenever the dot turns purple (3 times): personalises the blink detector. */
  async blinkStep() {
    const started = await this.tracker.request({ type: 'blink_calibration_start' }, 'blink_calibration_started')
      .catch(() => null);
    if (!started || !started.ok) return null;
    this.placeDot(0.5, 0.5);
    this.ui.dot.classList.add('settle');
    this.say('Last step: your blinks', 'Each time the dot turns purple, blink twice — like a relaxed “yes, yes”.', { top: true });
    await this.wait(2600);
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
      this.say('Let’s see how you wink', 'Look at the dot. You’ll close one eye at a time and keep it closed until the ring is full. If you can’t, that’s fine — there are other ways.', { top: true });
      await this.wait(3600);
      const ring = h('div', { class: 'wink-ring', style: { left: '50%', top: '50%' } });
      this.ui.ov.append(ring);
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
      } finally {
        this.tracker.setGesturePhase(null);
        ring.remove();
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

  // -- quick adjust ------------------------------------------------------------------------
  async runAdjust() {
    this.cancelled = false;
    const started = await this.tracker.request({ type: 'calibration_start', mode: 'adjust' }, 'calibration_started');
    if (!started.ok) throw new Error(started.error || 'Could not start adjustment');
    this.say('Quick adjustment', 'Look at each dot as it appears.');
    await this.wait(1600);
    this.say('');
    await this.showPoints(ADJUST_POINTS, 'adjust', { minFrames: 14 });
    this.say('Adjusting…', h('div', { class: 'spinner' }));
    const res = await this.tracker.request({ type: 'calibration_fit', mode: 'adjust' }, 'calibration_result', 30000);
    this.app.gaze.resetBias();
    if (res.ok) {
      toast(`Adjusted — error ${Math.round(res.error_before_px)} → ${Math.round(res.error_after_px)} px`, 'ok');
      sounds.success();
    } else {
      toast(res.error || 'Adjustment failed', 'warn');
    }
    return { mode: 'adjust', fit: res };
  }
}
