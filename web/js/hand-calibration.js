// Hand-mode setup: a slow, spoken calibration so finger control is reliable.
//
// Steps:
//   1. Hand span   - hold an open hand up; measures the palm width used to
//                    normalise every gesture threshold.
//   2. Pointing    - point your index finger at a grid of dots; fits a map from
//                    the fingertip's comfortable range to the whole screen.
//   3. Pinch       - open your hand, then pinch a few times; sets personal
//                    make/break click thresholds (hysteresis).
//   4. Practice    - pinch to pop a target, so you learn it and we confirm it.
//
// It reuses the calibration overlay styling and the tracker's frame-label
// channel (the same one eye calibration uses). The maths and persistence live
// in the Python hand session; this file drives the pacing and collection.

import { h, sleep, toast } from './dom.js';
import { speak, canSpeak } from './speech.js';
import { sounds } from './sound.js';

class Cancelled extends Error {}

const POINT_GRID = (() => {
  const xs = [0.1, 0.4, 0.6, 0.9];
  const ys = [0.12, 0.5, 0.88];
  const pts = [];
  ys.forEach((y, row) => (row % 2 ? [...xs].reverse() : xs).forEach((x) => pts.push([x, y])));
  pts.push([0.5, 0.5]);
  return pts;
})();

export class HandCalibrator {
  constructor(app) {
    this.app = app;
    this.cancelled = false;
    window.addEventListener('keydown', (e) => { if (e.key === 'Escape') this.cancelled = true; });
  }

  get tracker() { return this.app.tracker; }

  async run({ mode = 'full' } = {}) {
    this.cancelled = false;
    this.app.gaze.setSuspended(true);
    const ov = this.app.openOverlay('calib solid');
    this.ui = this.buildUI(ov);
    try {
      await this.tracker.request({ type: 'hand_calibration_start', mode }, 'hand_calibration_started', 8000)
        .catch(() => {});
      if (mode === 'full') await this.measureSpan();
      await this.pointing();
      if (mode === 'full') await this.pinch();
      const result = await this.tracker.request({ type: 'hand_calibration_fit' }, 'hand_calibration_result', 30000);
      if (!result.ok) {
        this.say('Setup didn’t work', result.error || 'Please try again.');
        this.speak('Setup did not work. Let us try again.');
        await this.wait(2500);
        return this.run({ mode });
      }
      await this.showResult(result);
      if (mode === 'full') await this.practice();
      return { ok: true, result };
    } catch (err) {
      if (err instanceof Cancelled) { toast('Setup cancelled'); return { ok: false }; }
      throw err;
    } finally {
      this.tracker.setLabel(null);
      this.app.closeOverlay(ov);
      this.app.gaze.setSuspended(false);
    }
  }

  // -- UI helpers (shared styling with eye calibration) --------------------------
  buildUI(ov) {
    const dot = h('div', { class: 'calib-dot done' });
    const text = h('div', { class: 'calib-text' });
    const meter = h('div', { class: 'calib-meter', hidden: true }, h('i'));
    const progress = h('div', { class: 'calib-progress' });
    ov.append(dot, text, meter, progress);
    return { ov, dot, text, meter, progress };
  }

  say(title, body = '') {
    const { text } = this.ui;
    text.innerHTML = '';
    if (title) text.append(h('h2', {}, title));
    if (body) text.append(h('p', {}, body));
    text.style.opacity = title || body ? '1' : '0';
  }

  speak(t) { if (canSpeak()) speak(t); }

  setMeter(frac) {
    const { meter } = this.ui;
    meter.hidden = frac == null;
    if (frac != null) meter.firstChild.style.width = `${Math.round(Math.max(0, Math.min(1, frac)) * 100)}%`;
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
    dot.style.translate = `${x}px ${y}px`;
    if (instant) { void dot.offsetWidth; dot.style.transition = ''; }
    dot.className = 'calib-dot';
    return { x, y };
  }

  checkCancel() { if (this.cancelled) throw new Cancelled(); }

  async wait(ms) {
    const end = performance.now() + ms;
    while (performance.now() < end) { this.checkCancel(); await sleep(Math.min(50, end - performance.now())); }
  }

  /** Collect labelled frames until enough arrive or the time runs out. */
  async collect(label, { minFrames = 18, minMs = 900, maxMs = 2600, onFrame } = {}) {
    let count = 0;
    const off = this.tracker.on('frame', (m) => {
      if (m.labeled) count++;
      if (onFrame) onFrame(m);
    });
    this.tracker.setLabel(label);
    const t0 = performance.now();
    try {
      for (;;) {
        await this.wait(40);
        const elapsed = performance.now() - t0;
        if ((elapsed >= minMs && count >= minFrames) || elapsed >= maxMs) break;
      }
    } finally {
      this.tracker.setLabel(null);
      off();
    }
    return count;
  }

  // -- step 1: hand span ---------------------------------------------------------
  async measureSpan() {
    this.say('Hold up your open hand', 'Spread your fingers and keep your hand still.');
    this.speak('First, hold up your open hand with your fingers spread, and keep it still.');
    await this.wait(2600);
    this.setMeter(0);
    let seen = 0;
    await this.collect({ kind: 'hspan' }, {
      minMs: 1500, maxMs: 2600, minFrames: 20,
      onFrame: (m) => { if (m.hand && m.hand.open) { seen++; this.setMeter(Math.min(1, seen / 25)); } },
    });
    this.setMeter(null);
    sounds.point();
  }

  // -- step 2: pointing map ------------------------------------------------------
  async pointing() {
    this.say('Point at each dot', 'Point your index finger at the dot. Hold until it fills.');
    this.speak('Now point your index finger at each dot. Hold steady until the dot fills in.');
    await this.wait(3200);
    this.say('');
    for (let i = 0; i < POINT_GRID.length; i++) {
      this.checkCancel();
      const [fx, fy] = POINT_GRID[i];
      this.placeDot(fx, fy, { instant: i === 0 });
      await this.wait(i === 0 ? 500 : 650);
      this.ui.dot.classList.add('settle');
      await this.wait(450);
      this.ui.dot.classList.add('collect');
      let n = await this.collect({ kind: 'hpoint', i, fx, fy }, { minFrames: 16, minMs: 900, maxMs: 2600 });
      if (n < 6) n = await this.collect({ kind: 'hpoint', i, fx, fy }, { minFrames: 10, minMs: 700, maxMs: 2600 });
      sounds.point();
      this.setProgress(POINT_GRID.length, i + 1);
    }
    this.ui.dot.className = 'calib-dot done';
    this.setProgress(0, 0);
  }

  // -- step 3: pinch thresholds --------------------------------------------------
  async pinch() {
    this.say('Relax your open hand', 'Keep your hand open and still for a moment.');
    this.speak('Keep your hand open and relaxed for a moment.');
    await this.wait(2600);
    await this.collect({ kind: 'hpinch', state: 'open' }, { minMs: 1500, maxMs: 2400, minFrames: 20 });
    sounds.point();

    this.say('Now pinch a few times', 'Touch your thumb and index finger together, then open. Repeat slowly.');
    this.speak('Now pinch your thumb and index finger together and open again. Do it slowly, a few times.');
    await this.wait(3200);
    // Only label frames that actually look pinched, so "closed" stays clean.
    const off = this.tracker.on('frame', (m) => {
      this.tracker.setLabel(m.hand && typeof m.hand.pinch === 'number' && m.hand.pinch < 0.5
        ? { kind: 'hpinch', state: 'closed' } : null);
    });
    try { await this.wait(5000); } finally { this.tracker.setLabel(null); off(); }
    sounds.point();
  }

  // -- step 4: result + practice -------------------------------------------------
  async showResult(result) {
    const parts = [];
    if (result.has_pointing) parts.push('pointing map set');
    if (result.pinch_on != null) parts.push('pinch tuned');
    this.say('Hand setup complete', parts.join(' · ') || 'Saved.');
    this.speak('Great. Your hand setup is complete.');
    await this.wait(2600);
    this.say('');
  }

  async practice() {
    this.app.gaze.setActive(true);
    this.app.gaze.setSuspended(false);
    this.say('Try it: pinch the target', 'Point at the circle and pinch to pop it.');
    this.speak('Let us practice. Point at the circle, then pinch to pop it.');
    const target = h('button', { class: 'practice-target', 'aria-label': 'Practice target' });
    Object.assign(target.style, { position: 'fixed', left: '50%', top: '46%', translate: '-50% -50%',
      width: '120px', height: '120px', borderRadius: '50%' });
    this.ui.ov.append(target);
    let popped = false;
    await new Promise((resolve) => {
      const done = () => { if (!popped) { popped = true; resolve(); } };
      target.addEventListener('click', done);
      const timer = setInterval(() => { if (this.cancelled) { clearInterval(timer); resolve(); } }, 100);
    });
    target.remove();
    if (popped) { sounds.success(); this.speak('Nice. You are ready to browse with your hands.'); }
    this.app.gaze.setSuspended(true);
  }
}
