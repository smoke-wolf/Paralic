// Hand-mode setup: a slow, spoken setup so finger control is reliable.
//
// Steps:
//   1. Hand size  - hold up an open hand; measures the palm width every
//                   gesture distance is relative to.
//   2. Pointing   - point the index finger at a grid of dots; fits a map from
//                   the fingertip's comfortable range to the whole screen.
//   3. Pinch      - relax the hand, then pinch a few times; the server splits
//                   the distances into "open" and "closed" and sets this
//                   person's click thresholds.
//   4. Practice   - point at targets and pinch to pop them (with a time limit
//                   and a Skip button, so nobody gets stuck).
//
// A quick re-point ('point') redoes only step 2 and keeps the pinch. The dots
// are labelled in *screen* fractions, like eye calibration, so the setup is
// right whether or not the page is full screen. The maths and the saving live
// in the Python session (hand_control.py); this file paces the steps.

import { h, sleep, toast } from './dom.js';
import { clientToScreen } from './screen-space.js';
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
const PRACTICE = [[0.5, 0.45], [0.25, 0.3], [0.75, 0.65]];
const ATTEMPTS = 3;
const PINCHES = 4;

export class HandCalibrator {
  constructor(app) {
    this.app = app;
    this.running = false;
    this.cancelled = false;
    this.handSeenAt = 0;
    this.onKey = (e) => { if (e.key === 'Escape') this.cancelled = true; };
  }

  get tracker() { return this.app.tracker; }

  /** Run the setup ('full' or 'point'). Resolves to {ok, result}. */
  async run({ mode = 'full' } = {}) {
    if (this.running) return { ok: false };
    this.running = true;
    this.cancelled = false;
    window.addEventListener('keydown', this.onKey);
    const offFrame = this.tracker.on('frame', (m) => { if (m.face) this.handSeenAt = performance.now(); });
    this.app.gaze.setSuspended(true);
    const ov = this.app.openOverlay('calib solid');
    this.ui = this.buildUI(ov);
    let outcome = { ok: false };
    try {
      for (let attempt = 1; attempt <= ATTEMPTS; attempt++) {
        const result = await this.attempt(mode);
        if (result.ok) {
          outcome = { ok: true, result };
          break;
        }
        if (result.mode === 'point' || /re-point/.test(result.error || '')) mode = 'full';
        if (attempt === ATTEMPTS) break;
        this.say('Let’s try that again', result.error || 'The setup did not work.');
        this.speak(`That did not work. ${result.error || ''} Let us try again.`);
        await this.wait(4500);
      }
      if (outcome.ok) {
        await this.showResult(outcome.result);
        if (mode === 'full') await this.practice();
      } else {
        toast('The hand setup did not finish. Try again from Settings when you are ready.', 'warn', 8000);
      }
    } catch (err) {
      if (!(err instanceof Cancelled)) throw err;
      toast('Hand setup cancelled');
    } finally {
      this.tracker.setLabel(null);
      if (!outcome.ok) this.tracker.send({ type: 'hand_calibration_cancel' });
      offFrame();
      window.removeEventListener('keydown', this.onKey);
      this.app.closeOverlay(ov);
      this.app.gaze.setSuspended(false);
      this.running = false;
    }
    return outcome;
  }

  async attempt(mode) {
    const started = await this.tracker.request({ type: 'hand_calibration_start', mode }, 'hand_calibration_started', 8000);
    if (!started.ok) return { ...started, mode };
    if (mode === 'full') await this.measureSpan();
    await this.pointing();
    if (mode === 'full') await this.pinch();
    this.say('One moment…');
    return this.tracker.request({ type: 'hand_calibration_fit' }, 'hand_calibration_result', 30000);
  }

  // -- UI helpers (the eye calibration's styling) ---------------------------------
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
    // What the server learns: where the dot is on the *screen*.
    const s = clientToScreen(x, y);
    return { fx: s.x / window.screen.width, fy: s.y / window.screen.height };
  }

  checkCancel() { if (this.cancelled) throw new Cancelled(); }

  async wait(ms) {
    const end = performance.now() + ms;
    while (performance.now() < end) { this.checkCancel(); await sleep(Math.min(50, end - performance.now())); }
  }

  get handVisible() { return performance.now() - this.handSeenAt < 400; }

  /** Collect labelled frames until enough arrive. Time only runs while the
   *  hand is in view (up to a hard limit), and a hint asks for it when not. */
  async collect(label, { minFrames = 18, minMs = 900, maxMs = 2600, onFrame } = {}) {
    let count = 0;
    const off = this.tracker.on('frame', (m) => {
      if (m.labeled) count++;
      if (onFrame) onFrame(m);
    });
    const hint = h('p', { class: 'calib-hint', hidden: true }, 'I can’t see your hand — raise it into view.');
    this.ui.ov.append(hint);
    this.tracker.setLabel(label);
    let active = 0;
    const started = performance.now();
    let last = started;
    try {
      for (;;) {
        await this.wait(40);
        const now = performance.now();
        if (this.handVisible) active += now - last;
        last = now;
        hint.hidden = this.handVisible || now - started < 600;
        if ((active >= minMs && count >= minFrames) || active >= maxMs || now - started >= 3 * maxMs + 4000) break;
      }
    } finally {
      this.tracker.setLabel(null);
      hint.remove();
      off();
    }
    return count;
  }

  // -- step 1: hand size -------------------------------------------------------------
  async measureSpan() {
    this.say('Hold up your open hand', 'Palm towards the camera, fingers spread and thumb out. Keep it still.');
    this.speak('First, hold up your open hand, palm towards the camera, with your fingers spread. Keep it still.');
    await this.wait(3000);
    let n = 0;
    this.setMeter(0);
    await this.collect({ kind: 'hspan' }, {
      minMs: 1200, maxMs: 4000, minFrames: 20,
      onFrame: (m) => { if (m.labeled) this.setMeter(++n / 20); },
    });
    this.setMeter(null);
    sounds.point();
  }

  // -- step 2: pointing ----------------------------------------------------------------
  async pointing() {
    this.say('Point at each dot', 'Point your index finger at the dot and hold it there until the dot fills in.');
    this.speak('Now point your index finger at each dot. Hold it there until the dot fills in.');
    await this.wait(3400);
    this.say('');
    for (let i = 0; i < POINT_GRID.length; i++) {
      this.checkCancel();
      const [x, y] = POINT_GRID[i];
      const at = this.placeDot(x, y, { instant: i === 0 });
      await this.wait(i === 0 ? 500 : 700);
      this.ui.dot.classList.add('settle');
      await this.wait(400);
      this.ui.dot.classList.add('collect');
      const label = { kind: 'hpoint', i, fx: at.fx, fy: at.fy };
      let n = 0;
      await this.collect(label, {
        minFrames: 16, minMs: 900, maxMs: 2600,
        onFrame: (m) => { if (m.labeled) this.ui.dot.style.setProperty('--p', Math.min(1, ++n / 16).toFixed(2)); },
      });
      this.ui.dot.style.removeProperty('--p');
      sounds.point();
      this.setProgress(POINT_GRID.length, i + 1);
    }
    this.ui.dot.className = 'calib-dot done';
    this.setProgress(0, 0);
  }

  // -- step 3: pinch ----------------------------------------------------------------------
  async pinch() {
    this.say('Relax your open hand', 'Keep your hand open and still for a moment.');
    this.speak('Now keep your hand open and relaxed for a moment.');
    await this.wait(2600);
    await this.collect({ kind: 'hpinch', state: 'open' }, { minMs: 1200, maxMs: 2400, minFrames: 20 });
    sounds.point();

    this.say('Now pinch a few times', `Touch your thumb and index finger together, then open again — ${PINCHES} times, at your own pace.`);
    this.speak(`Now pinch your thumb and index finger together, and open again. Do it ${PINCHES} times, at your own pace.`);
    await this.wait(3400);
    // Count pinches with this person's own range (min..max so far), not a fixed distance.
    let lo = Infinity;
    let hi = -Infinity;
    let closed = false;
    let pinches = 0;
    this.setMeter(0);
    await this.collect({ kind: 'hpinch', state: 'cycle' }, {
      minFrames: 60, minMs: 4000, maxMs: 12000,
      onFrame: (m) => {
        const d = m.hand && m.hand.pinch;
        if (typeof d !== 'number') return;
        lo = Math.min(lo, d);
        hi = Math.max(hi, d);
        if (hi - lo < 0.15) return;
        if (!closed && d < lo + 0.35 * (hi - lo)) closed = true;
        else if (closed && d > lo + 0.65 * (hi - lo)) {
          closed = false;
          pinches++;
          sounds.point();
          this.setMeter(pinches / PINCHES);
          this.say('Now pinch a few times', `${pinches} of ${PINCHES}`);
        }
      },
    }).then(async () => {
      // Not finished yet? Give a little more time for the last pinches.
      if (pinches < PINCHES && !this.cancelled) {
        await this.collect({ kind: 'hpinch', state: 'cycle' }, { minFrames: 0, minMs: 3000, maxMs: 3000 });
      }
    });
    this.setMeter(null);
  }

  // -- step 4: result + practice -------------------------------------------------------
  async showResult(result) {
    const parts = [];
    if (result.pointing_error_px != null) parts.push(`pointing within about ${Math.round(result.pointing_error_px)} px`);
    if (result.pinch_tuned) parts.push('pinch tuned to your hand');
    this.say('Hand setup complete', parts.join(' · ') || 'Saved.');
    this.speak('Great. Your hand setup is complete.');
    this.app.state.hand = { ...result, calibrated: true };
    this.app.emit('hand', this.app.state.hand);
    await this.wait(2600);
    this.say('');
  }

  async practice() {
    this.app.gaze.setActive(true);
    this.app.gaze.setSuspended(false);
    this.speak('Let us practise. Point at the circle, then pinch to pop it.');
    const skip = h('button', { class: 'btn', type: 'button', style: { position: 'fixed', right: '4vw', bottom: '4vh' } }, 'Skip practice');
    let skipped = false;
    skip.addEventListener('click', () => { skipped = true; });
    this.ui.ov.append(skip);
    let popped = 0;
    try {
      for (let i = 0; i < PRACTICE.length && !skipped; i++) {
        this.say('Point at the circle and pinch', `Target ${i + 1} of ${PRACTICE.length}`);
        const [x, y] = PRACTICE[i];
        const target = h('button', { class: 'practice-target', type: 'button', 'aria-label': 'Practice target' });
        Object.assign(target.style, { position: 'fixed', left: `${x * 100}%`, top: `${y * 100}%`, translate: '-50% -50%',
          width: '120px', height: '120px', borderRadius: '50%' });
        this.ui.ov.append(target);
        const hit = await new Promise((resolve) => {
          const until = performance.now() + 20000;
          target.addEventListener('click', () => resolve(true));
          const timer = setInterval(() => {
            if (this.cancelled || skipped || performance.now() > until) { clearInterval(timer); resolve(false); }
          }, 100);
        });
        target.remove();
        if (this.cancelled) throw new Cancelled();
        if (hit) {
          popped++;
          sounds.success();
        }
      }
    } finally {
      skip.remove();
      this.app.gaze.setSuspended(true);
    }
    if (popped === PRACTICE.length) this.speak('Nice. You are ready to browse with your hand.');
    else if (!skipped) toast('Tip: pinch quickly and let go — a long pinch does not click.', 'ok', 7000);
  }
}
