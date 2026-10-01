// The gaze cursor: turns gaze predictions into a pointer that can hover,
// "click" with a double blink and scroll the page.
//
// * The Python server sends a smoothed gaze point (screen coordinates) ~30x/s.
// * Here we convert it to page coordinates, animate the cursor at 60 fps,
//   find the element you are looking at (snapping to the nearest button so you
//   do not need pixel-perfect accuracy) and highlight it.
// * A double blink clicks the element that was highlighted *just before the
//   first blink* (the server tells us which frame that was), so the cursor
//   drifting while your eyes close cannot make you miss.
// * Looking at the scroll rail on the right scrolls the page.
// * Optional drift correction: each click nudges a small offset so that the
//   cursor lines up better with the buttons you actually pick.
// * Optional dwell click (per person): resting the eyes on a button for a
//   moment clicks it; a ring on the cursor fills up first.
// * Graceful motion (per person, see motion.js): the cursor glides instead of
//   jumping and, with "hold still while you look", rests on the average of
//   each fixation. Optional head nudge: small head tilts move the cursor the
//   last few pixels.

import { $$, clamp, distToRect, nearestPointIn, isVisible, h } from './dom.js';
import { CursorMotion, HeadNudge } from './motion.js';
import { screenToClient } from './screen-space.js';
import { getSettings } from './settings.js';
import { sounds } from './sound.js';

export const TARGET_SELECTOR = 'a[href], button:not([disabled]), [data-gaze]';

// Snap radius is a fraction of the viewport diagonal.
const SNAP = {
  off: { radius: 0, pull: 0 },
  normal: { radius: 0.05, pull: 0.3 },
  strong: { radius: 0.085, pull: 0.55 },
};
const SCROLL_SPEED = { slow: 240, normal: 460, fast: 780 }; // px/s
const RAIL_DWELL_MS = 280;
const HISTORY_SIZE = 150;

export class GazeController extends EventTarget {
  constructor(tracker, { cursorEl, pageEl, railEl }) {
    super();
    this.tracker = tracker;
    this.cursorEl = cursorEl;
    this.pageEl = pageEl;
    this.railEl = railEl;
    this.active = false;       // tracking is running and calibrated
    this.suspended = false;    // e.g. while calibrating
    this.paused = false;
    this.target = null;        // latest gaze point, page coordinates (drift-corrected)
    this.pos = null;           // animated cursor position (with the head nudge)
    this.motion = new CursorMotion();
    this.nudge = new HeadNudge();
    this.head = null;          // latest head pose [yaw, pitch, roll] (degrees)
    this.headAt = 0;
    this._jumped = false;
    this.display = null;       // position after magnetism
    this.hover = null;
    this.hoverSince = 0;
    this.lastGazeAt = 0;
    this.frozen = false;
    this.armed = false;
    this.bias = { x: 0, y: 0 };
    this.history = new Map();
    this.doubleBlinkHandlers = [];
    this.personalMagnet = null;  // {radius_px, pull} learned for the current person
    this.magnetOverride = null;  // set temporarily by an A/B experiment round
    this.gestures = null;        // the person's gesture settings (dwell click etc.)
    this.dwellOverride = null;   // dwell time (ms) set temporarily by an A/B experiment round
    this.selector = TARGET_SELECTOR;
    this.progress = 0;           // 0..1 ring on the cursor (dwell click / long press)
    this.pressProgress = 0;
    this.lastFrameId = null;
    this.winking = null;
    this._dwell = { el: null, since: 0, done: false };
    this._lastT = performance.now();
    this._armTimer = null;

    tracker.on('frame', (m) => !this.dead && this.onFrame(m));
    tracker.on('blink', (m) => !this.dead && this.onBlink(m));
    tracker.on('blink_expired', () => !this.dead && this.disarm());
    tracker.on('double_blink', (m) => !this.dead && this.onDoubleBlink(m));

    this._onRailClick = (e) => {
      const zone = e.target.closest('[data-scroll]');
      if (zone && !this.dead) this.scrollByPage(zone.dataset.scroll === 'up' ? -1 : 1);
    };
    railEl.addEventListener('click', this._onRailClick);

    requestAnimationFrame((t) => this.loop(t));
  }

  destroy() {
    this.dead = true;
    this.railEl.removeEventListener('click', this._onRailClick);
    this.clearHover();
    this.cursorEl.hidden = true;
  }

  // -- public API -------------------------------------------------------------
  setActive(on) {
    this.active = on;
    if (!on) this.clearHover();
  }

  setSuspended(on) {
    this.suspended = on;
    if (on) {
      this.clearHover();
      this.disarm();
    } else {
      this.restartMotion();
    }
  }

  setPaused(on) {
    this.paused = on;
    document.body.classList.toggle('is-paused', on);
    if (on) {
      this.clearHover();
      this.disarm();
    } else {
      this.restartMotion();
    }
    this.dispatchEvent(new CustomEvent('pausechange', { detail: on }));
  }

  /** Start the cursor afresh at the latest gaze point (no glide across the screen). */
  restartMotion() {
    this.motion.reset(this.target);
    this.nudge.clear();
  }

  resetBias() {
    this.bias = { x: 0, y: 0 };
  }

  /** Apply the person's motion settings (gestures from the server; see GESTURE_DEFAULTS). */
  configureMotion() {
    const g = this.gestures || {};
    // Hold radius: a few times the person's cursor jitter (their smoothed precision).
    const jitter = Number(this.precisionPx) || 14;
    this.motion.configure({
      style: g.motion || 'balanced',
      hold: g.hold_still !== false,
      radius: clamp(3 * jitter, 25, 90),
    });
    this.nudge.configure({
      enabled: !!g.head_nudge,
      speed: Number(g.nudge_speed) || 60,
      deadzone: Number(g.nudge_deadzone) || 5,
      signs: { right: Number(g.nudge_right) || -1, up: Number(g.nudge_up) || 1 },
    });
  }

  /** What can be highlighted: buttons and links normally, drop zones while dragging. */
  setTargetSelector(selector) {
    this.selector = selector || TARGET_SELECTOR;
    this.clearHover();
  }

  /** Where the eyes were (and what was highlighted) at a given frame, or null. */
  entryFor(frameId) {
    return frameId !== null && frameId !== undefined ? this.history.get(frameId) || null : null;
  }

  get dwellMs() {
    if (this.dwellOverride) return this.dwellOverride;
    const g = this.gestures;
    return g && g.dwell ? Number(g.dwell_ms) || 1000 : 0;
  }

  /** Register a handler that gets double blinks first; return true to consume. */
  onDoubleBlinkFirst(fn) {
    this.doubleBlinkHandlers.push(fn);
    return () => {
      const i = this.doubleBlinkHandlers.indexOf(fn);
      if (i >= 0) this.doubleBlinkHandlers.splice(i, 1);
    };
  }

  get point() {
    return this.display ? { ...this.display } : null;
  }

  /** The cursor position before the magnet pulls it towards a button. */
  get rawPoint() {
    if (!this.pos) return null;
    return { x: clamp(this.pos.x, 0, window.innerWidth), y: clamp(this.pos.y, 0, window.innerHeight) };
  }

  get hasGaze() {
    return !!this.target && performance.now() - this.lastGazeAt < 1500;
  }

  // -- incoming data --------------------------------------------------------------
  onFrame(msg) {
    if (msg.gaze) {
      const c = screenToClient(msg.gaze[0], msg.gaze[1]);
      this.target = { x: c.x + this.bias.x, y: c.y + this.bias.y };
      // Without a face the server repeats the last position: let it go stale.
      if (msg.face) this.lastGazeAt = performance.now();
      if (!this.pos) this.pos = { ...this.target };
      // A new sample for the cursor's motion - not while the eyes are shut
      // (the server holds the point still then).
      if (msg.face && !msg.frozen && this.motion.addSample(this.target)) this._jumped = true;
    }
    if (msg.face && Array.isArray(msg.head)) {
      this.head = msg.head;
      this.headAt = performance.now();
    }
    this.frozen = !!msg.frozen;
    this.winking = msg.winking || null;
    if (msg.id !== undefined && msg.id !== null) {
      this.lastFrameId = msg.id;
      this.history.set(msg.id, {
        hover: this.hover,
        x: this.target ? this.target.x : null,
        y: this.target ? this.target.y : null,
      });
      if (this.history.size > HISTORY_SIZE) this.history.delete(this.history.keys().next().value);
    }
  }

  onBlink(msg) {
    if (msg.n !== 1 || this.suspended || !this.active) return;
    this.armed = true;
    if (!this.paused) sounds.firstBlink();
    if (this.hover) this.hover.classList.add('gaze-armed');
    clearTimeout(this._armTimer);
    this._armTimer = setTimeout(() => this.disarm(), 1600);
  }

  disarm() {
    this.armed = false;
    clearTimeout(this._armTimer);
    $$('.gaze-armed').forEach((el) => el.classList.remove('gaze-armed'));
  }

  onDoubleBlink(msg) {
    this.disarm();
    const entry = msg.pre_frame !== null && msg.pre_frame !== undefined ? this.history.get(msg.pre_frame) : null;
    for (let i = this.doubleBlinkHandlers.length - 1; i >= 0; i--) {
      if (this.doubleBlinkHandlers[i](msg, entry)) return;
    }
    if (this.paused) {
      this.setPaused(false);
      sounds.success();
      return;
    }
    if (!this.active || this.suspended) return;

    // Trust what was highlighted just before the first blink (even "nothing");
    // fall back to the current highlight only if that moment is unknown.
    let el = entry ? entry.hover : this.hover;
    if (el && (!el.isConnected || !isVisible(el))) el = null;
    const point = entry && entry.x !== null ? { x: entry.x, y: entry.y } : this.point;
    this.activate(el, point, entry, msg.pre_frame ?? null);
  }

  /** Perform a gaze click on `el` (or report a miss at `point`). */
  activate(el, point, entry = null, preFrame = null, via = 'blink') {
    const at = el ? centerOf(el.getBoundingClientRect()) : point;
    if (at) this.ripple(at);
    if (via !== 'dwell') this.resetDwell();
    this.dispatchEvent(new CustomEvent('activate', { detail: { element: el || null, point, preFrame, via } }));
    if (!el) {
      sounds.miss();
      return;
    }
    sounds.click();
    el.classList.remove('gaze-pressed');
    void el.offsetWidth; // restart the animation
    el.classList.add('gaze-pressed');
    setTimeout(() => el.classList.remove('gaze-pressed'), 400);
    if (entry && entry.x !== null) this.learnFromClick(el, entry);
    el.click();
  }

  learnFromClick(el, entry) {
    if (!getSettings().driftCorrection || el.closest('.scroll-rail, .no-drift')) return;
    const r = el.getBoundingClientRect();
    // Huge targets say little about where exactly you looked.
    if (r.width > window.innerWidth * 0.45 || r.height > window.innerHeight * 0.45) return;
    const np = nearestPointIn(entry.x, entry.y, r, 0.35);
    const maxBias = Math.hypot(window.innerWidth, window.innerHeight) * 0.12;
    this.bias.x = clamp(this.bias.x + 0.3 * (np.x - entry.x), -maxBias, maxBias);
    this.bias.y = clamp(this.bias.y + 0.3 * (np.y - entry.y), -maxBias, maxBias);
  }

  ripple(p) {
    const node = h('div', { class: 'click-ripple', style: { left: `${p.x}px`, top: `${p.y}px` } });
    document.body.append(node);
    setTimeout(() => node.remove(), 600);
  }

  // -- per-frame work ---------------------------------------------------------------
  loop(t) {
    if (this.dead) return;
    const dt = Math.min(0.1, Math.max(0.001, (t - this._lastT) / 1000));
    this._lastT = t;
    try {
      this.step(dt, t);
    } catch (err) {
      console.error(err);
    }
    requestAnimationFrame((tt) => this.loop(tt));
  }

  step(dt, now) {
    const s = getSettings();
    const cursor = this.cursorEl;
    const show = this.active && !this.suspended && !this.paused && this.hasGaze;
    this.updateRail();
    if (!show) {
      cursor.hidden = true;
      if (this.hover) this.clearHover();
      return;
    }

    // Glide towards the gaze point (motion.js), then add the head nudge. While
    // the head is tilted to nudge, the gaze point is held where it was.
    this.configureMotion();
    if (!this.motion.pos) this.motion.reset(this.target);
    const base = this.motion.step(dt);
    const head = now - this.headAt < 300 ? this.head : null;
    const off = this.nudge.update(head, dt, this._jumped);
    this._jumped = false;
    this.motion.lock(this.nudge.active);
    this.pos = { x: base.x + off.x, y: base.y + off.y };
    cursorClasses(this.cursorEl, this.nudge);
    const p = {
      x: clamp(this.pos.x, 0, window.innerWidth),
      y: clamp(this.pos.y, 0, window.innerHeight),
    };

    if (!this.frozen) this.updateHover(p, s, now);

    // Magnetism: pull the drawn cursor towards the hovered element.
    let d = p;
    const snap = this.snapSettings(s);
    if (this.hover && snap.pull > 0) {
      const c = centerOf(this.hover.getBoundingClientRect());
      d = { x: p.x + (c.x - p.x) * snap.pull, y: p.y + (c.y - p.y) * snap.pull };
    }
    this.display = d;

    cursor.hidden = false;
    cursor.style.transform = `translate3d(${d.x}px, ${d.y}px, 0)`;
    cursor.classList.toggle('on-target', !!this.hover);
    cursor.classList.toggle('frozen', this.frozen);
    cursor.classList.toggle('armed', this.armed);
    cursor.classList.toggle('size-small', s.cursorSize === 'small');
    cursor.classList.toggle('size-large', s.cursorSize === 'large');
    cursor.classList.toggle('stale', now - this.lastGazeAt > 500);
    cursor.classList.toggle('winking', !!this.winking);

    this.updateDwell(now);
    const ring = Math.max(this.progress, this.pressProgress);
    cursor.style.setProperty('--p', ring.toFixed(3));
    cursor.classList.toggle('progress', ring > 0.02);

    this.updateScroll(p, s, dt, now);
  }

  /** Dwell click: resting on a button for `dwellMs` clicks it (once per visit). */
  updateDwell(now) {
    const ms = this.dwellMs;
    const el = this.hover;
    const d = this._dwell;
    if (!ms || !el || this.frozen || this.winking || this.selector !== TARGET_SELECTOR
        || el.closest('[data-scroll], [data-no-dwell]')) {
      this.progress = 0;
      if (!el) d.el = null;
      return;
    }
    if (d.el !== el) {
      d.el = el;
      d.since = now;
      d.done = false;
    }
    if (d.done) {
      this.progress = 0;
      return;
    }
    this.progress = Math.min(1, (now - d.since) / ms);
    if (this.progress >= 1) {
      d.done = true;
      this.progress = 0;
      this.activate(el, this.point, null, this.lastFrameId, 'dwell');
    }
  }

  /** Restart the dwell timer (e.g. after a different kind of click). */
  resetDwell() {
    this._dwell = { el: this.hover, since: performance.now(), done: true };
    this.progress = 0;
  }

  updateHover(p, s, now) {
    const radius = this.snapSettings(s).radiusPx;
    let best = null;
    let bestDist = Infinity;
    let bestArea = Infinity;
    let currentDist = Infinity;
    for (const el of $$(this.selector)) {
      if (el.closest('[inert], .gaze-ignore')) continue;
      const r = el.getBoundingClientRect();
      if (r.width < 4 || r.height < 4 || r.bottom < 0 || r.top > window.innerHeight || r.right < 0 || r.left > window.innerWidth) continue;
      const dist = distToRect(p.x, p.y, r);
      if (el === this.hover) currentDist = dist;
      if (dist > radius && dist > 0) continue;
      const area = r.width * r.height;
      // Prefer containing elements (smallest first), then the nearest one.
      if (dist < bestDist - 0.5 || (dist === 0 && bestDist === 0 && area < bestArea)) {
        if (!isVisible(el)) continue;
        best = el;
        bestDist = dist;
        bestArea = area;
      }
    }
    // Hysteresis: keep the current target unless the new one is clearly better.
    if (this.hover && best !== this.hover && currentDist <= radius * 1.4 + 1 && this.hover.isConnected) {
      // Looking at something inside the highlighted element (a card in a list) also counts.
      const clearlyBetter = (bestDist === 0 && currentDist > 0) || bestDist + 30 < currentDist
        || (bestDist === 0 && !!best && this.hover.contains(best));
      if (!best || !clearlyBetter) best = this.hover;
    }
    if (best !== this.hover) {
      if (this.hover) this.hover.classList.remove('gaze-hover', 'gaze-armed');
      this.hover = best;
      this.hoverSince = now;
      if (best) {
        best.classList.add('gaze-hover');
        if (this.armed) best.classList.add('gaze-armed');
      }
    }
  }

  /** Magnet radius (px) and pull: experiment override > personal ("auto") > preset. */
  snapSettings(s) {
    const diag = Math.hypot(window.innerWidth, window.innerHeight);
    const m = this.magnetOverride || (s.snap === 'auto' ? this.personalMagnet : null);
    if (m) return { radiusPx: Number(m.radius_px) || 0, pull: Number(m.pull) || 0 };
    const preset = SNAP[s.snap] || SNAP.normal;
    return { radiusPx: preset.radius * diag, pull: preset.pull };
  }

  clearHover() {
    if (this.hover) this.hover.classList.remove('gaze-hover', 'gaze-armed');
    this.hover = null;
    $$('.rail-zone.is-scrolling').forEach((z) => z.classList.remove('is-scrolling'));
  }

  // -- scrolling ------------------------------------------------------------------
  scrollContainer() {
    return this.pageEl;
  }

  updateRail() {
    const sc = this.scrollContainer();
    const scrollable = sc.scrollHeight > sc.clientHeight + 4;
    this.railEl.classList.toggle('is-hidden', !scrollable);
    const [up, down] = this.railEl.querySelectorAll('[data-scroll]');
    up.classList.toggle('is-disabled', !scrollable || sc.scrollTop <= 1);
    down.classList.toggle('is-disabled', !scrollable || sc.scrollTop + sc.clientHeight >= sc.scrollHeight - 1);
  }

  updateScroll(p, s, dt, now) {
    const zone = this.hover && this.hover.closest('[data-scroll]');
    $$('.rail-zone.is-scrolling').forEach((z) => { if (z !== zone) z.classList.remove('is-scrolling'); });
    if (!zone || this.frozen || now - this.hoverSince < RAIL_DWELL_MS) return;
    const sc = this.scrollContainer();
    if (sc.scrollHeight <= sc.clientHeight + 4) return;
    const r = zone.getBoundingClientRect();
    const up = zone.dataset.scroll === 'up';
    // Faster the further you look towards the arrow's end of the zone.
    const depth = clamp(up ? (r.bottom - p.y) / r.height : (p.y - r.top) / r.height, 0, 1);
    const speed = (SCROLL_SPEED[s.scrollSpeed] || SCROLL_SPEED.normal) * (0.5 + depth);
    zone.classList.add('is-scrolling');
    sc.scrollTop += (up ? -1 : 1) * speed * dt;
  }

  scrollByPage(dir) {
    const sc = this.scrollContainer();
    sc.scrollBy({ top: dir * sc.clientHeight * 0.8, behavior: 'smooth' });
  }
}

function centerOf(r) {
  return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
}

function cursorClasses(cursor, nudge) {
  cursor.classList.toggle('nudging', nudge.active);
}
