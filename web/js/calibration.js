// Calibration: teach GazeNet where on the screen you are looking.
//
// Full calibration (about 30 seconds):
//   1. follow a dot across 13 positions while keeping your head still,
//   2. keep looking at the centre dot while gently moving your head
//      (teaches the network to compensate for head movement),
//   3. the server trains your personal neural network,
//   4. five more dots measure the accuracy (and are then used for a final fit).
//
// Quick adjust (about 8 seconds): 5 dots that correct a saved calibration for
// today's seating position.

import { h, sleep, toast } from './dom.js';
import { clientToScreen, screenToClient } from './screen-space.js';
import { sounds } from './sound.js';

class Cancelled extends Error {}

function gridPoints() {
  const xs = [0.06, 0.35, 0.65, 0.94];
  const ys = [0.08, 0.5, 0.92];
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
const HEAD_STEPS = [
  'Keep your eyes on the dot…',
  'Slowly turn your head a little to the left',
  '…and a little to the right',
  'Now tilt your head slightly up',
  '…and slightly down',
  'Back to the middle. Great!',
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
    try {
      return mode === 'adjust' ? await this.runAdjust() : await this.runFull();
    } catch (err) {
      if (err instanceof Cancelled) {
        toast('Calibration cancelled');
        return null;
      }
      throw err;
    } finally {
      this.tracker.setLabel(null);
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
    ov.append(dot, text, warning, progress);
    return { ov, dot, text, warning, progress };
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

  say(title, body = '', { top = false } = {}) {
    const { text } = this.ui;
    text.classList.toggle('top', top);
    text.innerHTML = '';
    if (title) text.append(h('h2', {}, title));
    if (body) text.append(typeof body === 'string' ? h('p', {}, body) : body);
    text.style.opacity = title || body ? '1' : '0';
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

  /** Label frames with the dot position until enough good frames arrived. */
  async collect(kind, point, x, y, { minFrames = 16, minMs = 550, maxMs = 2600 } = {}) {
    let count = 0;
    let missing = 0;
    const off = this.tracker.on('frame', (m) => {
      if (m.labeled) count++;
      missing = m.face ? 0 : missing + 1;
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
      await this.wait(i === 0 ? 400 : 620);
      this.ui.dot.classList.add('settle');
      await this.wait(opts.settleMs ?? 380);
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

    // Head-movement phase.
    const c = this.placeDot(0.5, 0.5);
    const guide = h('div', { class: 'head-guide', style: { left: `${c.x}px`, top: `${c.y}px` } });
    this.ui.ov.append(guide);
    this.ui.dot.classList.add('settle');
    this.setProgress(0, 0);
    this.say('Now keep looking at the dot', 'and gently move your head. If moving is hard for you, just keep looking.', { top: true });
    await this.wait(1600);
    this.ui.dot.classList.add('collect');
    const s = clientToScreen(c.x, c.y);
    this.tracker.setLabel({ x: s.x, y: s.y, kind: 'head', pt: 0 });
    try {
      for (const step of HEAD_STEPS) {
        this.say(step, '', { top: true });
        await this.wait(1050);
      }
    } finally {
      this.tracker.setLabel(null);
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
