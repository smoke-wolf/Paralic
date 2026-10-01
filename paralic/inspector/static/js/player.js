// Playback: the current frame and time, play / pause at 0.25-4x, frame
// stepping, and the transport bar with its timeline scrubber.

import { C, EYE, PHASE, PHASE_NAMES, S, bisect, clamp, clock, count, fitCanvas, h, hideTip, perFrame, showTip, ticks }
  from './util.js';

export const SPEEDS = [0.25, 0.5, 1, 2, 4];

export class Player {
  constructor(index) {
    this.t = Float64Array.from(index.t);
    this.ids = Float64Array.from(index.i);
    // For each frame, the frame index of the image that shows it (itself or the nearest earlier one).
    this.videoOf = new Int32Array(this.t.length);
    let last = -1;
    for (let n = 0; n < this.t.length; n++) {
      if (index.v && index.v[n]) last = n;
      this.videoOf[n] = last;
    }
    this.n = 0;
    this.time = this.t.length ? this.t[0] : 0;
    this.playing = false;
    this.speed = 1;
    this.listeners = { frame: new Set(), time: new Set(), state: new Set() };
    this._tick = this._tick.bind(this);
  }

  get length() { return this.t.length; }
  get start() { return this.t.length ? this.t[0] : 0; }
  get end() { return this.t.length ? this.t[this.t.length - 1] : 0; }
  get frameId() { return this.ids[this.n]; }

  on(event, fn) {
    this.listeners[event].add(fn);
    return () => this.listeners[event].delete(fn);
  }

  emit(event) {
    for (const fn of this.listeners[event]) fn(this);
  }

  indexAt(time) {
    return clamp(bisect(this.t, time), 0, Math.max(0, this.length - 1));
  }

  /** The browser frame id of the camera image for frame n (null: no image yet). */
  videoId(n = this.n) {
    const k = this.videoOf[n];
    return k >= 0 ? this.ids[k] : null;
  }

  seek(time) {
    this.time = clamp(time, this.start, this.end);
    this._setFrame(this.indexAt(this.time));
    this.emit('time');
  }

  go(n) {
    n = clamp(Math.round(n), 0, Math.max(0, this.length - 1));
    this.time = this.t[n];
    this._setFrame(n);
    this.emit('time');
  }

  step(k) {
    this.pause();
    this.go(this.n + k);
  }

  _setFrame(n) {
    if (n !== this.n) {
      this.n = n;
      this.emit('frame');
    }
  }

  play() {
    if (this.playing || !this.length) return;
    if (this.n >= this.length - 1) this.go(0);
    this.playing = true;
    this._last = performance.now();
    this.emit('state');
    requestAnimationFrame(this._tick);
  }

  pause() {
    if (!this.playing) return;
    this.playing = false;
    this.emit('state');
  }

  toggle() {
    if (this.playing) this.pause();
    else this.play();
  }

  setSpeed(s) {
    this.speed = s;
    this.emit('state');
  }

  _tick(now) {
    if (!this.playing) return;
    const dt = Math.min(0.25, (now - this._last) / 1000);
    this._last = now;
    let time = this.time + dt * this.speed;
    if (time >= this.end) {
      time = this.end;
      this.playing = false;
      this.emit('state');
    }
    this.time = time;
    this._setFrame(this.indexAt(time));
    this.emit('time');
    if (this.playing) requestAnimationFrame(this._tick);
  }
}

// -- the transport bar --------------------------------------------------------------

const ICONS = {
  first: 'M6 5h2v14H6zM9.5 12 18 5v14z',
  back: 'M7 5h2v14H7zM20 5v14l-9-7z',
  play: 'M8 5v14l11-7z',
  pause: 'M7 5h4v14H7zM13 5h4v14h-4z',
  fwd: 'M4 5v14l9-7zM15 5h2v14h-2z',
  last: 'M6 5v14l8.5-7zM16 5h2v14h-2z',
};

function icon(name) {
  const ns = 'http://www.w3.org/2000/svg';
  const s = document.createElementNS(ns, 'svg');
  s.setAttribute('viewBox', '0 0 24 24');
  s.setAttribute('aria-hidden', 'true');
  const p = document.createElementNS(ns, 'path');
  p.setAttribute('d', ICONS[name]);
  s.append(p);
  return s;
}

export class Transport {
  constructor(player, { mode, onSection }) {
    this.player = player;
    this.mode = mode;
    const btn = (name, title, fn) => h('button', { class: 'tbtn', type: 'button', title, 'aria-label': title, onclick: fn },
      icon(name));
    this.playBtn = btn('play', 'Play / pause (Space)', () => player.toggle());
    this.playBtn.classList.add('play');
    this.speedBtns = SPEEDS.map((s) => h('button', {
      class: 'chip', type: 'button', title: `Play at ${s}× speed`, onclick: () => player.setSpeed(s),
    }, `${s}×`));
    this.timeEl = h('span', { class: 'tclock' });
    this.frameEl = h('span', { class: 'tframe' });
    const sections = ['Camera', 'Signals', 'Under the hood', 'Calibration', 'Events', 'Config'];
    this.scrubber = new Scrubber(player, mode);
    this.el = h('div', { class: 'transport' },
      h('div', { class: 'tbar' },
        h('div', { class: 'tgroup' },
          btn('first', 'First frame (Home)', () => { player.pause(); player.go(0); }),
          btn('back', 'Previous frame (←, Shift+← 10 frames)', () => player.step(-1)),
          this.playBtn,
          btn('fwd', 'Next frame (→, Shift+→ 10 frames)', () => player.step(1)),
          btn('last', 'Last frame (End)', () => { player.pause(); player.go(player.length - 1); })),
        h('div', { class: 'tgroup speeds', role: 'group', 'aria-label': 'Speed' }, this.speedBtns),
        h('div', { class: 'tgroup readout' }, this.timeEl, this.frameEl),
        h('nav', { class: 'tgroup sections' }, sections.map((name) => h('a', {
          href: '#', onclick: (e) => { e.preventDefault(); onSection(name); },
        }, name)))),
      this.scrubber.el);
    player.on('time', perFrame(() => this.render()));
    player.on('state', () => this.render());
    this.render();
  }

  render() {
    const p = this.player;
    this.playBtn.replaceChildren(icon(p.playing ? 'pause' : 'play'));
    this.speedBtns.forEach((b, k) => b.classList.toggle('on', SPEEDS[k] === p.speed));
    this.timeEl.textContent = `${clock(p.time - p.start)} / ${clock(p.end - p.start)}`;
    this.frameEl.textContent = p.length
      ? `frame ${count(p.n + 1)} of ${count(p.length)} · id ${count(p.frameId)}` : 'no frames';
    this.scrubber.draw();
  }
}

// -- the scrubber: the whole recording at a glance ---------------------------------------------

const ROWS = [
  { key: 'calibration', label: 'calibration', h: 14 },
  { key: 'face', label: 'face', h: 10 },
  { key: 'glasses', label: 'glasses', h: 8 },
  { key: 'gestures', label: 'gestures', h: 16 },
  { key: 'changes', label: 'changes', h: 12 },
];
const GUTTER = 78;

export class Scrubber {
  constructor(player, mode) {
    this.player = player;
    this.mode = mode;
    this.timeline = null;
    this.view = null;            // the signals panel's visible window [t0, t1]
    this.canvas = h('canvas', { class: 'scrub-canvas', 'aria-label': 'Timeline: click or drag to move' });
    this.el = h('div', { class: 'scrubber' }, this.canvas, this._legend());
    this.dragging = false;
    this.canvas.addEventListener('pointerdown', (e) => {
      if (e.offsetX < GUTTER) return;
      this.dragging = true;
      this.canvas.setPointerCapture(e.pointerId);
      this.player.pause();
      this._seek(e);
    });
    this.canvas.addEventListener('pointermove', (e) => {
      if (this.dragging) this._seek(e);
      this._hover(e);
    });
    this.canvas.addEventListener('pointerup', () => { this.dragging = false; });
    this.canvas.addEventListener('pointerleave', () => hideTip());
    new ResizeObserver(() => this.draw()).observe(this.el);
  }

  _legend() {
    const items = this.mode === 'hand'
      ? [['hand size', PHASE.hspan], ['pointing dots', PHASE.hpoint], ['pinch', PHASE.hpinch], ['no hand', C.noface],
        ['more than one hand', S[6]], ['pinching', S[0]], ['click', S[4]], ['scroll', S[2]], ['open hand', S[5]]]
      : [['calibration dots', PHASE.cal], ['head movements', PHASE.head], ['validation', PHASE.val],
        ['quick adjust', PHASE.adjust], ['no face', C.noface], ['other faces', S[6]], ['only someone else', S[3]],
        ['glasses', S[5]], ['glare', S[7]], ['blink', C.muted], ['double blink', S[4]], ['left wink', EYE.left],
        ['right wink', EYE.right]];
    return h('div', { class: 'legend scrub-legend' },
      items.map(([label, color]) => h('span', { class: 'legend-item' },
        h('span', { class: 'legend-key box', style: { '--c': color } }), label)),
      h('span', { class: 'legend-item' }, h('span', { class: 'legend-key diamond', style: { '--c': C.text } }), 'model change'),
      h('span', { class: 'legend-item' }, h('span', { class: 'legend-key diamond', style: { '--c': C.muted } }),
        'setting or command'));
  }

  setTimeline(tl) {
    this.timeline = tl;
    this.draw();
  }

  setView(t0, t1) {
    this.view = [t0, t1];
    this.draw();
  }

  _x(t) {
    const p = this.player;
    const w = this.width - GUTTER - 8;
    return GUTTER + ((t - p.start) / Math.max(p.end - p.start, 1e-6)) * w;
  }

  _t(x) {
    const p = this.player;
    const w = this.width - GUTTER - 8;
    return p.start + ((x - GUTTER) / w) * (p.end - p.start);
  }

  _seek(e) {
    this.player.seek(this._t(e.offsetX));
  }

  _hover(e) {
    if (e.offsetX < GUTTER || !this.timeline) {
      hideTip();
      return;
    }
    const t = this._t(e.offsetX);
    const near = this.timeline.markers
      .map((m) => ({ m, d: Math.abs(this._x(m.t) - e.offsetX) }))
      .filter((x) => x.d <= 5)
      .sort((a, b) => a.d - b.d)
      .slice(0, 4)
      .map(({ m }) => ({ value: clock(m.t - this.player.start), label: `${m.kind} · ${m.type}${m.eye ? ` (${m.eye})` : ''}` }));
    const spans = (this.timeline.stretches || []).filter((s) => s.t0 <= t && t <= s.t1).slice(0, 4)
      .map((s) => ({ value: (s.t1 - s.t0).toFixed(2) + ' s', label: stretchName(s) }));
    showTip(e.clientX, e.clientY, [{ title: clock(t - this.player.start) }, ...near, ...spans]);
  }

  draw() {
    const width = this.el.clientWidth;
    if (!width) return;
    this.width = width;
    const height = ROWS.reduce((a, r) => a + r.h + 3, 0) + 18;
    const ctx = fitCanvas(this.canvas, width, height);
    ctx.clearRect(0, 0, width, height);
    ctx.font = '11px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    let y = 2;
    const rowY = {};
    for (const r of ROWS) {
      rowY[r.key] = [y, r.h];
      ctx.fillStyle = 'rgba(148, 163, 214, 0.06)';
      ctx.fillRect(GUTTER, y, width - GUTTER - 8, r.h);
      ctx.fillStyle = C.faint;
      ctx.fillText(r.label, 6, y + r.h / 2);
      y += r.h + 3;
    }
    const p = this.player;
    const span = p.end - p.start;
    const marks = ticks(0, span, Math.max(2, Math.floor((width - GUTTER) / 110)));
    ctx.fillStyle = 'rgba(148, 163, 214, 0.14)';
    for (const t of marks) ctx.fillRect(Math.round(this._x(p.start + t)), 0, 1, y - 1);
    const tl = this.timeline;
    if (tl) {
      for (const s of tl.stretches || []) this._stretch(ctx, s, rowY);
      for (const m of tl.markers || []) this._marker(ctx, m, rowY);
    }
    // The signals panel's window.
    if (this.view) {
      const [a, b] = this.view.map((t) => this._x(t));
      ctx.fillStyle = 'rgba(238, 242, 255, 0.07)';
      ctx.fillRect(a, 0, Math.max(b - a, 1), y - 2);
      ctx.strokeStyle = 'rgba(238, 242, 255, 0.35)';
      ctx.lineWidth = 1;
      ctx.strokeRect(a + 0.5, 0.5, Math.max(b - a - 1, 1), y - 3);
    }
    // Time axis labels.
    ctx.fillStyle = C.faint;
    ctx.textBaseline = 'top';
    ctx.textAlign = 'center';
    for (const t of marks) ctx.fillText(clock(t, span < 20), this._x(p.start + t), y + 1);
    ctx.textAlign = 'left';
    // Playhead.
    const x = this._x(p.time);
    ctx.fillStyle = C.accent;
    ctx.fillRect(Math.round(x) - 1, 0, 2, y);
    ctx.beginPath();
    ctx.moveTo(x - 5, 0);
    ctx.lineTo(x + 5, 0);
    ctx.lineTo(x, 6);
    ctx.fill();
  }

  _stretch(ctx, s, rowY) {
    let row = null;
    let color = null;
    let band = [0, 1];
    if (s.kind === 'label') {
      row = 'calibration';
      color = PHASE[s.value] || C.muted;
    } else if (s.kind === 'noface') {
      row = 'face';
      color = C.noface;
    } else if (s.kind === 'faces' || s.kind === 'hands') {
      row = 'face';
      color = S[6];
    } else if (s.kind === 'waiting') {
      row = 'face';
      color = S[3];
    } else if (s.kind === 'wink_test') {
      row = 'calibration';
      color = EYE[s.value] || C.faint;
      band = [0.25, 0.75];
    } else if (s.kind === 'glasses') {
      row = 'glasses';
      color = S[5];
      band = [0, 0.5];
    } else if (s.kind === 'glare') {
      row = 'glasses';
      color = S[7];
      band = [0.5, 1];
    } else if (s.kind === 'wink') {
      row = 'gestures';
      color = EYE[s.value] || S[6];
      band = [0.15, 0.85];
    } else if (s.kind === 'pinch') {
      row = 'gestures';
      color = S[0];
      band = [0.3, 0.7];
    } else if (s.kind === 'palm') {
      row = 'gestures';
      color = S[5];
      band = [0.1, 0.9];
    }
    if (!row) return;
    const [y, hgt] = rowY[row];
    const a = this._x(s.t0);
    const b = this._x(s.t1);
    ctx.fillStyle = color;
    ctx.fillRect(a, y + band[0] * hgt, Math.max(b - a, 1.5), (band[1] - band[0]) * hgt);
  }

  _marker(ctx, m, rowY) {
    const x = Math.round(this._x(m.t)) + 0.5;
    const [gy, gh] = rowY.gestures;
    const [cy, ch] = rowY.changes;
    const diamond = (color, size = 4) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(x, cy + ch / 2 - size);
      ctx.lineTo(x + size, cy + ch / 2);
      ctx.lineTo(x, cy + ch / 2 + size);
      ctx.lineTo(x - size, cy + ch / 2);
      ctx.fill();
    };
    switch (m.cat) {
      case 'blink':
      case 'blink2':
      case 'expired':
        if (m.hand) return;
        ctx.fillStyle = m.cat === 'expired' ? C.faint : C.muted;
        ctx.fillRect(x - 0.5, gy + 4, 1, gh - 8);
        break;
      case 'click':
        if (m.kind === 'command') return;      // the page's label_event repeats the click
        ctx.fillStyle = S[4];
        ctx.beginPath();
        ctx.arc(x, gy + gh / 2, 3.5, 0, Math.PI * 2);
        ctx.fill();
        break;
      case 'long_close':
        ctx.fillStyle = S[7];
        ctx.fillRect(x - 1, gy, 2, gh);
        break;
      case 'scroll':
        ctx.fillStyle = S[2];
        ctx.fillRect(x - 0.5, gy + 2, 1, gh - 4);
        break;
      case 'palm':
        ctx.fillStyle = S[5];
        ctx.fillRect(x - 1, gy, 2, gh);
        break;
      case 'model':
        diamond(C.text, 5);
        break;
      case 'error':
        diamond(C.critical, 5);
        break;
      case 'setting':
      case 'calibration':
      case 'person':
      case 'face':
      case 'glasses':
      case 'video':
      case 'push':
        diamond(C.muted, 3.5);
        break;
      default:
        break;
    }
  }
}

export function stretchName(s) {
  switch (s.kind) {
    case 'label': return PHASE_NAMES[s.value] || s.value;
    case 'noface': return 'no face / hand in view';
    case 'faces': return 'other faces in view';
    case 'waiting': return 'only someone else in view: waiting';
    case 'hands': return 'more than one hand in view';
    case 'wink_test': return s.value === 'rest' ? 'wink test: both eyes open' : `wink test: close the ${s.value} eye`;
    case 'glasses': return 'glasses on';
    case 'glare': return 'glare on a lens';
    case 'wink': return `${s.value} eye winking`;
    case 'closed': return 'eyes closed (blink detector)';
    case 'pinch': return 'pinching';
    case 'scroll': return 'pinch-drag scrolling';
    case 'palm': return 'open hand';
    default: return s.kind;
  }
}
