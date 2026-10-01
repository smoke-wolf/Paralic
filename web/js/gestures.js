// Eye gestures beyond the double blink: pressing, holding, dragging and the
// "right-click" menu, for people who need other ways to act than a click.
//
// * Hold one eye closed (a wink of about a third of a second or longer) =
//   hold the mouse button down at the spot you were looking at:
//     - look somewhere else while the eye stays closed to drag (draggable
//       things move; on other things it is a normal press-and-move, e.g. to
//       draw), then open the eye to drop / release;
//     - keep looking at the same spot for a second instead to open the menu
//       (a long press = right click);
//     - open the eye again quickly without moving = a click.
// * Close both eyes for about a second (optional): open the menu, pick up /
//   drop, or click - for people who cannot close one eye on its own.
// * A short wink (optional): click or menu.
// * Drag lock: "Pick up to move" in the menu (or a long close) picks a thing
//   up without holding anything; it follows your gaze until you blink twice,
//   rest your eyes on a place (dwell) or close your eyes again to drop it.
//
// What each gesture does is a per-person choice (Settings -> Eye gestures);
// the server detects the gestures and the page decides what they do here.
//
// Pages opt in with attributes:
//   data-draggable          an item that can be picked up
//   data-dropzone           a place it can be dropped (gets a "gazedrop" event)
//   data-no-longpress       holding still here does not open the menu (e.g. a canvas)
// and they may add menu items by listening for "gazemenu" (detail.items.push({...})).

import { $$, clamp, h, isVisible } from './dom.js';
import { icon } from './icons.js';
import { sounds } from './sound.js';
import { speak } from './speech.js';
import { TARGET_SELECTOR } from './gaze.js';

const POINTER_ID = 41;          // synthetic pointer events from the eyes
const MENU_TIMEOUT_MS = 15000;
const DEFAULTS = {
  left_hold: 'drag', right_hold: 'drag', left_quick: 'off', right_quick: 'off',
  long_close: 'off', long_close_ms: 1000, hold_ms: 350, long_press_ms: 1000, dwell: false, dwell_ms: 1000,
};

export class GestureController {
  constructor(app) {
    this.app = app;
    this.press = null;   // a held wink: {eye, el, item, point, start, moved, drag, pointerTarget, last}
    this.lock = null;    // drag lock: a DragSession that follows the gaze without holding
    this.menu = null;
    this.offs = [];
  }

  get gaze() {
    return this.app.gaze;
  }

  get settings() {
    return { ...DEFAULTS, ...((this.app.state.personal && this.app.state.personal.gestures) || {}) };
  }

  attach(tracker) {
    this.detach();
    this.offs = [
      tracker.on('wink_start', (m) => this.onWinkStart(m)),
      tracker.on('wink_end', (m) => this.onWinkEnd(m)),
      tracker.on('wink', (m) => this.onQuickWink(m)),
      tracker.on('long_close_ready', () => this.ready() && sounds.firstBlink()),
      tracker.on('long_close', (m) => this.onLongClose(m)),
      this.gaze.onDoubleBlinkFirst((m, entry) => this.onDoubleBlink(m, entry)),
    ];
    this._raf = requestAnimationFrame((t) => this.loop(t));
  }

  detach() {
    this.offs.forEach((off) => off());
    this.offs = [];
    cancelAnimationFrame(this._raf);
    this.cancelAll();
  }

  /** Gestures act only while the gaze cursor is live. */
  ready() {
    const g = this.gaze;
    return g && g.active && !g.suspended && !g.paused && !this.app.calibrator.running;
  }

  cancelAll() {
    if (this.press) this.endPress(true);
    if (this.lock) this.endLock(true);
    if (this.menu) this.menu.close();
  }

  // -- where the eyes were -------------------------------------------------------------
  /** The element and point the eyes were on at frame `preFrame` (before the eye closed). */
  spot(preFrame) {
    const entry = this.gaze.entryFor(preFrame);
    const point = entry && entry.x !== null ? { x: entry.x, y: entry.y } : this.gaze.point;
    let el = entry ? entry.hover : this.gaze.hover;
    if (el && (!el.isConnected || !isVisible(el))) el = null;
    const under = point ? elementAt(point) : null;
    return { point, el, under, entry };
  }

  // -- holding one eye closed ----------------------------------------------------------------
  onWinkStart(msg) {
    if (!this.ready()) return;
    const mapping = this.settings[`${msg.eye}_hold`];
    if (mapping === 'off') return;
    const { point, el, under } = this.spot(msg.pre_frame);
    if (!point) return;
    if (this.menu) {
      // In the menu a held wink presses the highlighted item (like a click).
      this.press = { eye: msg.eye, el, under, point, start: performance.now(), moved: false, inMenu: true };
      return;
    }
    if (this.lock) {
      // Already carrying something: closing an eye drops it here.
      this.dropLock(this.gaze.point);
      this.press = { eye: msg.eye, consumed: true };
      return;
    }
    if (mapping === 'menu') {
      this.press = { eye: msg.eye, consumed: true };
      this.openMenu(el || under, point);
      return;
    }
    this.startPress(msg.eye, point, el, under);
  }

  startPress(eye, point, el, under) {
    const target = el || under;
    const item = target ? target.closest('[data-draggable]') : null;
    const press = { eye, el, under, item, point, start: performance.now(), moved: false, drag: null, last: point };
    this.press = press;
    document.body.classList.add('gaze-holding');
    (item || el || under)?.classList.add('gaze-held');
    sounds.firstBlink();
    if (!item && under) {
      // A plain press: pages see ordinary pointer events (e.g. to draw).
      press.pointerTarget = under;
      firePointer(under, 'pointerdown', point);
    }
  }

  onWinkEnd(msg) {
    const p = this.press;
    if (!p || p.eye !== msg.eye) return;
    if (p.consumed) {
      this.press = null;
      return;
    }
    if (p.inMenu) {
      this.press = null;
      if (!msg.cancelled && p.el && p.el.closest('.gaze-menu')) this.gaze.activate(p.el, p.point);
      return;
    }
    this.endPress(!!msg.cancelled);
  }

  endPress(cancelled) {
    const p = this.press;
    if (!p) return;
    this.press = null;
    document.body.classList.remove('gaze-holding');
    $$('.gaze-held').forEach((n) => n.classList.remove('gaze-held'));
    this.gaze.pressProgress = 0;
    const point = (p.drag ? this.gaze.point : this.gaze.rawPoint) || p.last;
    if (p.drag) {
      if (cancelled) p.drag.cancel();
      else p.drag.drop(point);
      return;
    }
    if (p.pointerTarget) firePointer(p.pointerTarget, cancelled ? 'pointercancel' : 'pointerup', point);
    if (cancelled || p.moved) return;
    // Pressed and released without moving: a click on what was highlighted.
    const target = p.el || (p.item && p.item.matches(TARGET_SELECTOR) ? p.item : null);
    if (target || !p.pointerTarget) this.gaze.activate(target, p.point, null, null, 'wink');
  }

  /** Called every animation frame while something is held or carried. */
  loop() {
    this._raf = requestAnimationFrame((t) => this.loop(t));
    const p = this.press;
    const shown = this.gaze && this.gaze.point;       // pulled towards drop zones by the magnet
    const point = this.gaze && this.gaze.rawPoint;    // where the eyes are
    if (this.lock && !this.ready()) this.endLock(true); // paused or recalibrating: put it back
    if (this.lock && shown) this.lock.move(shown);
    if (!p || p.consumed || p.inMenu || !point) return;
    if (!this.ready()) {
      this.endPress(true);
      return;
    }
    p.last = point;
    const s = this.settings;
    const still = Math.hypot(point.x - p.point.x, point.y - p.point.y) <= stillRadius();
    if (!still && !p.moved) {
      p.moved = true;
      this.gaze.pressProgress = 0;
      if (p.item) {
        p.drag = new DragSession(this, p.item, p.point);
        sounds.point();
      }
    }
    if (p.drag) p.drag.move(shown);
    else if (p.pointerTarget) firePointer(p.pointerTarget, 'pointermove', point);
    if (!p.moved) {
      const target = p.item || p.el || p.under;
      const noMenu = target && target.closest('[data-no-longpress]');
      const progress = noMenu ? 0 : (performance.now() - p.start) / Number(s.long_press_ms || 1000);
      this.gaze.pressProgress = clamp(progress, 0, 1);
      if (progress >= 1) {
        // A long press: the menu (right click) instead of a drag.
        if (p.pointerTarget) firePointer(p.pointerTarget, 'pointercancel', point);
        this.press = { eye: p.eye, consumed: true };
        document.body.classList.remove('gaze-holding');
        $$('.gaze-held').forEach((n) => n.classList.remove('gaze-held'));
        this.gaze.pressProgress = 0;
        this.openMenu(target, p.point);
      }
    }
  }

  // -- short winks, long closes, double blinks ---------------------------------------------------
  onQuickWink(msg) {
    if (!this.ready()) return;
    const mapping = this.settings[`${msg.eye}_quick`];
    if (!mapping || mapping === 'off') return;
    const { point, el, under } = this.spot(msg.pre_frame);
    if (mapping === 'click') this.gaze.activate(el, point, null, msg.pre_frame, 'wink');
    else if (mapping === 'menu' && !this.menu) this.openMenu(el || under, point);
  }

  onLongClose(msg) {
    if (!this.ready()) return;
    const mapping = this.settings.long_close;
    if (!mapping || mapping === 'off') return;
    const { point, el, under } = this.spot(msg.pre_frame);
    if (!point) return;
    if (this.lock) {
      this.dropLock(point);
      return;
    }
    if (this.menu) return;
    if (mapping === 'click') {
      this.gaze.activate(el, point, null, msg.pre_frame, 'long_close');
    } else if (mapping === 'grab') {
      const item = (el || under)?.closest('[data-draggable]');
      if (item) this.startLock(item, point);
      else this.openMenu(el || under, point);
    } else {
      this.openMenu(el || under, point);
    }
  }

  onDoubleBlink(msg, entry) {
    if (!this.lock) return false;
    // Carrying something: a double blink drops it where you were looking.
    const point = entry && entry.x !== null ? { x: entry.x, y: entry.y } : this.gaze.point;
    this.dropLock(point);
    return true;
  }

  // -- drag lock ----------------------------------------------------------------------------------
  startLock(item, point) {
    if (this.lock) this.endLock(true);
    this.lock = new DragSession(this, item, point, { locked: true });
    sounds.point();
  }

  dropLock(point) {
    const lock = this.lock;
    if (!lock) return;
    this.lock = null;
    lock.drop(point);
  }

  endLock(cancelled) {
    const lock = this.lock;
    this.lock = null;
    if (lock && cancelled) lock.cancel();
  }

  // -- the menu ("right click") -------------------------------------------------------------------
  /** Open the gaze menu for `el` at `point`, after giving the page a real contextmenu event. */
  openMenu(el, point) {
    if (this.menu) this.menu.close();
    const target = el || (point ? elementAt(point) : null);
    if (target) {
      const ev = new MouseEvent('contextmenu', {
        bubbles: true, cancelable: true, clientX: point.x, clientY: point.y, button: 2, buttons: 2,
      });
      if (!target.dispatchEvent(ev)) return; // the page showed its own menu
    }
    const items = [];
    const clickable = target ? target.closest(TARGET_SELECTOR) : null;
    const item = target ? target.closest('[data-draggable]') : null;
    if (item) items.push({ id: 'move', label: 'Pick up to move', icon: 'grab', run: () => this.startLock(item, point) });
    if (clickable && clickable !== item) {
      items.push({ id: 'click', label: 'Click', icon: 'check', run: () => this.gaze.activate(clickable, point, null, null, 'menu') });
    }
    const text = readableText(target);
    if (text) items.push({ id: 'read', label: 'Read aloud', icon: 'speaker', run: () => speak(text) });
    const extra = [];
    if (target) target.dispatchEvent(new CustomEvent('gazemenu', { bubbles: true, detail: { items: extra, point } }));
    items.push(...extra);
    items.push({ id: 'cancel', label: 'Cancel', icon: 'back', run: () => {} });
    this.menu = new GazeMenu(this.app, point, items, label(target), () => { this.menu = null; });
    sounds.pop();
  }
}

// ---------------------------------------------------------------------------------------------------
// Dragging
// ---------------------------------------------------------------------------------------------------

class DragSession {
  constructor(gestures, item, point, { locked = false } = {}) {
    this.gestures = gestures;
    this.gaze = gestures.gaze;
    this.item = item;
    this.locked = locked;
    const r = item.getBoundingClientRect();
    this.offset = { x: clamp(point.x - r.left, 0, r.width), y: clamp(point.y - r.top, 0, r.height) };
    this.size = { w: r.width, h: r.height };
    this.ghost = item.cloneNode(true);
    this.ghost.removeAttribute('id');
    this.ghost.classList.remove('gaze-hover', 'gaze-held', 'gaze-armed');
    this.ghost.classList.add('drag-ghost');
    Object.assign(this.ghost.style, { width: `${r.width}px`, height: `${r.height}px` });
    this.ghost.setAttribute('aria-hidden', 'true');
    document.body.append(this.ghost);
    item.classList.add('is-dragging');
    document.body.classList.add('gaze-dragging');
    this.gaze.setTargetSelector('[data-dropzone]');
    this.zone = null;
    this.dwellZone = null;
    this.dwellSince = 0;
    this.place(point);
    item.dispatchEvent(new CustomEvent('gazedragstart', { bubbles: true, detail: { item, locked } }));
  }

  place(point) {
    this.ghost.style.translate = `${point.x - this.offset.x}px ${point.y - this.offset.y}px`;
  }

  move(point) {
    this.place(point);
    const zone = this.gaze.hover && this.gaze.hover.closest('[data-dropzone]');
    if (zone !== this.zone) {
      if (this.zone) this.zone.classList.remove('drop-hover');
      this.zone = zone || null;
      if (this.zone) this.zone.classList.add('drop-hover');
      this.dwellZone = this.zone;
      this.dwellSince = performance.now();
    }
    // Carrying (drag lock) + dwell click: resting on a place drops the item there.
    const ms = this.locked ? this.gaze.dwellMs : 0;
    if (ms && this.zone) {
      const progress = (performance.now() - this.dwellSince) / ms;
      this.gaze.pressProgress = clamp(progress, 0, 1);
      if (progress >= 1) {
        this.gaze.pressProgress = 0;
        this.gestures.dropLock(point);
      }
    } else if (this.locked) {
      this.gaze.pressProgress = 0;
    }
  }

  drop(point) {
    const zone = this.zone || (point ? elementAt(point)?.closest('[data-dropzone]') : null);
    if (!zone) {
      this.cancel();
      return;
    }
    const ok = zone.dispatchEvent(new CustomEvent('gazedrop', {
      bubbles: true, cancelable: true, detail: { item: this.item, zone, point },
    }));
    sounds.click();
    this.finish();
    if (ok) zone.classList.add('drop-done');
    setTimeout(() => zone.classList.remove('drop-done'), 450);
  }

  cancel() {
    // Fly back to where the item came from.
    const r = this.item.getBoundingClientRect();
    this.ghost.classList.add('returning');
    this.ghost.style.translate = `${r.left}px ${r.top}px`;
    sounds.miss();
    const ghost = this.ghost;
    this.ghost = null;
    setTimeout(() => ghost.remove(), 260);
    this.finish();
  }

  finish() {
    if (this.ghost) this.ghost.remove();
    if (this.zone) this.zone.classList.remove('drop-hover');
    this.item.classList.remove('is-dragging');
    document.body.classList.remove('gaze-dragging');
    this.gaze.setTargetSelector(null);
    this.gaze.pressProgress = 0;
    this.item.dispatchEvent(new CustomEvent('gazedragend', { bubbles: true, detail: { item: this.item } }));
  }
}

// ---------------------------------------------------------------------------------------------------
// The menu
// ---------------------------------------------------------------------------------------------------

class GazeMenu {
  constructor(app, point, items, title, onClose) {
    this.app = app;
    this.onClose = onClose;
    this.ov = app.openOverlay('menu-overlay');
    const list = h('div', { class: 'gaze-menu-items' });
    for (const it of items) {
      const b = h('button', { class: `gaze-menu-item ${it.id === 'cancel' ? 'cancel' : ''}`, type: 'button',
        'data-menu': it.id, html: `${icon(it.icon || 'check')}<span>${it.label}</span>` });
      b.addEventListener('click', () => {
        this.close();
        try {
          it.run && it.run();
        } catch (err) {
          console.error(err);
        }
      });
      list.append(b);
    }
    const card = h('div', { class: 'gaze-menu', role: 'menu' },
      title ? h('div', { class: 'gaze-menu-title' }, title) : null, list);
    this.ov.append(card);
    // Next to where you looked, but fully on screen.
    const r = card.getBoundingClientRect();
    const x = clamp(point.x + 24, 16, window.innerWidth - r.width - 16);
    const y = clamp(point.y - r.height / 2, 16, window.innerHeight - r.height - 16);
    Object.assign(card.style, { left: `${x}px`, top: `${y}px` });
    this.ov.addEventListener('click', (e) => { if (e.target === this.ov) this.close(); });
    this.timer = setTimeout(() => this.close(), MENU_TIMEOUT_MS);
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    clearTimeout(this.timer);
    this.app.closeOverlay(this.ov);
    this.onClose();
  }
}

// ---------------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------------

function elementAt(point) {
  const el = document.elementFromPoint(point.x, point.y);
  return el && !el.closest('#gaze-cursor, .drag-ghost') ? el : null;
}

/** Movement (px) that still counts as "holding still" for a long press. */
function stillRadius() {
  return clamp(0.045 * Math.hypot(window.innerWidth, window.innerHeight), 60, 140);
}

function firePointer(target, type, point) {
  const down = type === 'pointerdown' || type === 'pointermove';
  target.dispatchEvent(new PointerEvent(type, {
    bubbles: true, cancelable: true, composed: true, clientX: point.x, clientY: point.y,
    pointerId: POINTER_ID, pointerType: 'pen', isPrimary: true, button: type === 'pointermove' ? -1 : 0,
    buttons: down ? 1 : 0, pressure: down ? 0.5 : 0,
  }));
}

/** Text as shown on screen (line breaks between blocks become spaces). */
function shownText(el) {
  return String(el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim();
}

function label(el) {
  if (!el) return '';
  const named = el.closest('[aria-label], [data-label]');
  const target = el.closest(`${TARGET_SELECTOR}, [data-draggable]`) || el;
  const heading = target.querySelector('h1, h2, h3, h4, b, strong');
  const text = (named && (named.dataset.label || named.getAttribute('aria-label')))
    || (heading && shownText(heading)) || shownText(target);
  return text.length > 40 ? `${text.slice(0, 38)}…` : text;
}

function readableText(el) {
  if (!el) return '';
  const block = el.closest('p, li, h1, h2, h3, blockquote, figcaption, [data-read], button, a, [data-draggable]') || el;
  return ((block.dataset && block.dataset.read) || shownText(block)).slice(0, 600);
}
