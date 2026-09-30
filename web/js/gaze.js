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

import { $$, clamp, distToRect, nearestPointIn, isVisible, h } from './dom.js';
import { screenToClient } from './screen-space.js';
import { getSettings } from './settings.js';
import { sounds } from './sound.js';

const TARGET_SELECTOR = 'a[href], button:not([disabled]), [data-gaze]';

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
    this.pos = null;           // animated cursor position
    this.display = null;       // position after magnetism
    this.hover = null;
    this.hoverSince = 0;
    this.lastGazeAt = 0;
    this.frozen = false;
    this.armed = false;
    this.bias = { x: 0, y: 0 };
    this.history = new Map();
    this.doubleBlinkHandlers = [];
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
    }
  }

  setPaused(on) {
    this.paused = on;
    document.body.classList.toggle('is-paused', on);
    if (on) {
      this.clearHover();
      this.disarm();
    }
    this.dispatchEvent(new CustomEvent('pausechange', { detail: on }));
  }

  resetBias() {
    this.bias = { x: 0, y: 0 };
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
    }
    this.frozen = !!msg.frozen;
    if (msg.id !== undefined && msg.id !== null) {
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
    this.activate(el, point, entry);
  }

  /** Perform a gaze click on `el` (or report a miss at `point`). */
  activate(el, point, entry = null) {
    const at = el ? centerOf(el.getBoundingClientRect()) : point;
    if (at) this.ripple(at);
    this.dispatchEvent(new CustomEvent('activate', { detail: { element: el || null, point } }));
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

    // Animate towards the latest gaze point (~60 ms time constant).
    const k = 1 - Math.exp(-dt / 0.06);
    const dx = this.target.x - this.pos.x;
    const dy = this.target.y - this.pos.y;
    if (Math.hypot(dx, dy) > 1.5) {
      this.pos.x += dx * k;
      this.pos.y += dy * k;
    }
    const p = {
      x: clamp(this.pos.x, 0, window.innerWidth),
      y: clamp(this.pos.y, 0, window.innerHeight),
    };

    if (!this.frozen) this.updateHover(p, s, now);

    // Magnetism: pull the drawn cursor towards the hovered element.
    let d = p;
    const snap = SNAP[s.snap] || SNAP.normal;
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

    this.updateScroll(p, s, dt, now);
  }

  updateHover(p, s, now) {
    const snap = SNAP[s.snap] || SNAP.normal;
    const radius = snap.radius * Math.hypot(window.innerWidth, window.innerHeight);
    let best = null;
    let bestDist = Infinity;
    let bestArea = Infinity;
    let currentDist = Infinity;
    for (const el of $$(TARGET_SELECTOR)) {
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
      const clearlyBetter = (bestDist === 0 && currentDist > 0) || bestDist + 30 < currentDist;
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
