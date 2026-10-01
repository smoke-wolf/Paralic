// Signals: every frame's numbers over time, in tracks that share one time axis.
//
// Wheel zooms around the pointer, dragging pans, a click moves the playhead
// there, a double click goes back to following the playhead. The visible
// window shows on the scrubber. Long stretches arrive from the server as the
// minimum and maximum per pixel bucket, drawn as a band, so a blink of three
// frames still shows in an hour-long view.

import { Latest, api } from './api.js';
import {
  C, EYE, GAZE, HEAD, PHASE, PHASE_NAMES, S, alpha, bisect, clamp, clock, debounce, fitCanvas, h, hideTip, isNum, num,
  perFrame, showTip, ticks,
} from './util.js';

const GUTTER = 148;
const HEAD_H = 18;
const GAP = 10;
const AXIS_H = 22;
const LABEL_CODES = { cal: 1, head: 2, val: 3, adjust: 4, ft: 5, hspan: 6, hpinch: 7, hpoint: 8 };
const ZOOMS = [['5 s', 5], ['20 s', 20], ['1 min', 60], ['5 min', 300], ['All', Infinity]];

function phaseRow(kinds) {
  return {
    label: 'phase', col: 'label',
    values: Object.fromEntries(kinds.map((k) => [LABEL_CODES[k], [PHASE[k], PHASE_NAMES[k]]])),
  };
}

// ``cols``: the columns with any value in the recording; ``top``: each column's largest value.
function eyeTracks(info, cols, top) {
  const { w: W, h: H } = info.screen;
  const tracks = [
    {
      key: 'closure', title: 'Eye closure', unit: '0 open, 1 shut', h: 112, range: [0, 1],
      series: [{ col: 'cl', label: 'left eye', color: EYE.left }, { col: 'cr', label: 'right eye', color: EYE.right },
        { col: 'signal', label: 'blink signal', color: S[2] }],
      refs: [{ col: 'close_thr', label: 'closed above', color: C.text, dash: [6, 4] },
        { col: 'open_thr', label: 'reopened below', color: C.muted, dash: [2, 3] },
        ...(cols.has('baseline') ? [{ col: 'baseline', label: 'baseline', color: C.faint }] : [])],
      shades: [{ col: 'closed', color: 'rgba(238, 242, 255, 0.09)', label: 'closed' }],
      markers: true,
    },
    ...(cols.has('wclose_l') ? [{
      key: 'wink', title: 'Wink detector', unit: 'per eye', h: 96, range: [0, 1],
      series: [{ col: 'cl', label: 'left eye', color: EYE.left }, { col: 'cr', label: 'right eye', color: EYE.right }],
      refs: [{ col: 'wclose_l', label: 'left closed above', color: EYE.left, dash: [6, 4] },
        { col: 'wclose_r', label: 'right closed above', color: EYE.right, dash: [6, 4] }],
      shades: [{ col: 'wstate', value: 1, color: alpha(S[6], 0.14), label: 'one eye closing' },
        { col: 'wstate', value: 2, color: alpha(S[6], 0.3), label: 'held: a press' }],
    }] : []),
    {
      key: 'gx', title: 'Gaze x', unit: 'screen px', h: 84, range: [0, W],
      series: [{ col: 'gx', label: 'cursor', color: GAZE.cursor }, { col: 'rx', label: 'raw', color: GAZE.raw },
        { col: 'tx', label: 'target', color: GAZE.target }],
    },
    {
      key: 'gy', title: 'Gaze y', unit: 'screen px, down', h: 84, range: [0, H], invert: true,
      series: [{ col: 'gy', label: 'cursor', color: GAZE.cursor }, { col: 'ry', label: 'raw', color: GAZE.raw },
        { col: 'ty', label: 'target', color: GAZE.target }],
    },
    {
      key: 'model', title: 'Gaze network', unit: 'px', h: 62, auto: true, min0: true,
      series: [{ col: 'spread', label: 'ensemble spread', color: S[0] },
        { col: 'check', label: '|computed − recorded|', color: S[1] }],
    },
    {
      key: 'head', title: 'Head rotation', unit: 'degrees', h: 74, auto: true,
      series: [{ col: 'yaw', label: 'yaw', color: HEAD.yaw }, { col: 'pitch', label: 'pitch', color: HEAD.pitch },
        { col: 'roll', label: 'roll', color: HEAD.roll }],
    },
    { key: 'dist', title: 'Distance', unit: 'cm from camera', h: 48, auto: true, series: [{ col: 'dist', label: 'distance', color: S[0] }] },
  ];
  if (cols.has('glasses')) {
    tracks.push({
      key: 'glare', title: 'Glare on the lenses', unit: 'score', h: 50, range: [0, 1],
      series: [{ col: 'glare_l', label: 'left lens', color: EYE.left }, { col: 'glare_r', label: 'right lens', color: EYE.right }],
      shades: [{ col: 'glasses', color: alpha(S[5], 0.18), label: 'glasses on' }],
    });
  }
  tracks.push({
    key: 'state', title: 'State', kind: 'bands',
    rows: [
      { label: 'face', col: 'faces', values: { 0: [C.noface, 'no face'], 2: [S[6], '2 faces'], 3: [S[6], '3+ faces'] } },
      ...(top.waiting > 0 ? [{ label: 'waiting', col: 'waiting', values: { 1: [S[3], 'faces in view, not this person'] } }] : []),
      { label: 'network', col: 'net', values: { 0: ['#5b6690', 'both eyes'], 1: [EYE.left, 'left eye'], 2: [EYE.right, 'right eye'] } },
      { label: 'winking', col: 'winking', values: { 1: [EYE.left, 'left'], 2: [EYE.right, 'right'] } },
      { label: 'pressed', col: 'pressed', values: { 1: [EYE.left, 'left held'], 2: [EYE.right, 'right held'] } },
      ...(cols.has('gesture') ? [{ label: 'wink test', col: 'gesture',
        values: { 0: [C.muted, 'asks: eyes open'], 1: [EYE.left, 'asks: close left'], 2: [EYE.right, 'asks: close right'] } }] : []),
      ...(cols.has('glasses') ? [{ label: 'glasses', col: 'glasses', values: { 1: [S[5], 'on'] } },
        { label: 'glare', col: 'glare', values: { 1: [S[7], 'left'], 2: [S[7], 'right'], 3: [S[7], 'both'] } }] : []),
      phaseRow(['cal', 'head', 'val', 'adjust', 'ft']),
      { label: 'kept', col: 'stored', values: { 1: [C.muted, 'kept as sample'] } },
      { label: 'frozen', col: 'frozen', values: { 1: [C.muted, 'cursor frozen'] } },
    ],
  });
  tracks.push({ key: 'fps', title: 'Frame rate', unit: 'fps', h: 44, auto: true, min0: true, series: [{ col: 'fps', label: 'fps', color: S[0] }] });
  tracks.push({ key: 'ms', title: 'Processing', unit: 'ms per frame', h: 44, auto: true, min0: true, series: [{ col: 'ms', label: 'ms', color: S[0] }] });
  return tracks;
}

function handTracks(info, top) {
  const { w: W, h: H } = info.screen;
  return [
    {
      key: 'pinch', title: 'Pinch distance', unit: 'palm widths', h: 112, auto: true, min0: true,
      series: [{ col: 'pinch', label: 'distance', color: S[0] }],
      thresholds: info.pinch ? info.pinch.history : null,
      shades: [{ col: 'pinching', color: 'rgba(238, 242, 255, 0.09)', label: 'pinching' },
        { col: 'scrolling', color: alpha(S[2], 0.16), label: 'scrolling' }],
      markers: true,
    },
    {
      key: 'gx', title: 'Cursor x', unit: 'screen px', h: 84, range: [0, W],
      series: [{ col: 'gx', label: 'cursor', color: GAZE.cursor }, { col: 'rx', label: 'fingertip, mapped', color: GAZE.raw },
        { col: 'tx', label: 'target', color: GAZE.target }],
    },
    {
      key: 'gy', title: 'Cursor y', unit: 'screen px, down', h: 84, range: [0, H], invert: true,
      series: [{ col: 'gy', label: 'cursor', color: GAZE.cursor }, { col: 'ry', label: 'fingertip, mapped', color: GAZE.raw },
        { col: 'ty', label: 'target', color: GAZE.target }],
    },
    {
      key: 'tip', title: 'Fingertip', unit: 'image, mirrored', h: 64, range: [0, 1],
      series: [{ col: 'tipx', label: 'x', color: S[0] }, { col: 'tipy', label: 'y', color: S[1] }],
    },
    { key: 'span', title: 'Palm width', unit: 'image widths', h: 48, auto: true, series: [{ col: 'span', label: 'palm width', color: S[0] }] },
    { key: 'scroll', title: 'Scrolling', unit: 'px per frame', h: 50, kind: 'eventbars', event: 'hand_scroll', field: 'dy', color: S[2] },
    {
      key: 'state', title: 'State', kind: 'bands',
      rows: [
        { label: 'hand', col: 'face', values: { 0: [C.noface, 'no hand'] } },
        ...(top.hands >= 2 ? [{ label: 'hands', col: 'hands', values: { 2: [S[6], 'more than one hand'], 3: [S[6], 'more than one hand'] } }] : []),
        { label: 'pinching', col: 'pinching', values: { 1: [S[0], 'pinching'] } },
        { label: 'scrolling', col: 'scrolling', values: { 1: [S[2], 'scrolling'] } },
        { label: 'open hand', col: 'open', values: { 1: [S[5], 'open "stop" hand'] } },
        phaseRow(['hspan', 'hpoint', 'hpinch']),
        { label: 'kept', col: 'stored', values: { 1: [C.muted, 'kept as sample'] } },
      ],
    },
    { key: 'fps', title: 'Frame rate', unit: 'fps', h: 44, auto: true, min0: true, series: [{ col: 'fps', label: 'fps', color: S[0] }] },
    { key: 'ms', title: 'Processing', unit: 'ms per frame', h: 44, auto: true, min0: true, series: [{ col: 'ms', label: 'ms', color: S[0] }] },
  ];
}

const MARKS = {
  blink: { color: C.muted, shape: 'tick', label: 'blink' },
  blink2: { color: C.text, shape: 'tick', label: 'second blink' },
  click: { color: S[4], shape: 'dot', label: 'click (double blink / pinch)' },
  wink: { color: S[6], shape: 'tri', label: 'wink' },
  long_close: { color: S[7], shape: 'bar', label: 'long close' },
  expired: { color: C.faint, shape: 'x', label: 'first blink expired' },
  scroll: { color: S[2], shape: 'tick', label: 'scroll' },
  palm: { color: S[5], shape: 'diamond', label: 'open hand: pause' },
};

export class SignalsPanel {
  constructor(rv) {
    this.rv = rv;
    this.info = rv.info;
    this.mode = rv.info.mode;
    this.cols = new Set(rv.info.columns);
    this.tracks = [];
    this.overview = null;
    this.window = null;
    this.timeline = null;
    this.events = null;
    this.follow = true;
    this.hover = null;
    const p = rv.player;
    const total = p.end - p.start;
    this.span = Math.min(20, Math.max(total, 1));
    this.view = [p.start, p.start + this.span];
    this.canvas = h('canvas', { class: 'sig-canvas', 'aria-label': 'Signals over time' });
    this.zoomBtns = ZOOMS.map(([label, span]) => h('button', {
      class: 'chip', type: 'button', onclick: () => this.zoomTo(span),
    }, label));
    this.followBtn = h('button', { class: 'chip', type: 'button', onclick: () => this.setFollow(!this.follow) }, 'Follow playhead');
    this.status = h('span', { class: 'sig-status' });
    this.bandLegend = h('div', { class: 'legend band-legend' });
    this.el = h('section', { class: 'panel signals', id: 'sec-signals' },
      h('header', { class: 'panel-head' }, h('h2', {}, 'Signals'),
        h('div', { class: 'toggles' }, this.zoomBtns, this.followBtn, this.status),
        h('span', { class: 'hint' }, 'wheel: zoom · drag: pan · click: go there · double click: follow')),
      h('div', { class: 'sig-wrap' }, this.canvas),
      this.bandLegend);
    this.draw = perFrame(() => this._draw());
    this.windowLoader = new Latest(
      (key) => api.signals(rv.id, key[0], key[1], key[2]),
      (key, data) => {
        this.window = data;
        this.draw();
      });
    this.ensure = debounce(() => this._ensure(), 60);
    this._bind();
    new ResizeObserver(() => this.draw()).observe(this.el);
    p.on('time', () => {
      if (this.follow) this._followPlayhead();
      this.draw();
    });
    this._updateButtons();
    this._loadOverview();
  }

  /** The whole recording at overview resolution; while the server is still reading it, ask again. */
  _loadOverview() {
    const p = this.rv.player;
    api.signals(this.rv.id, p.start, p.end, 3000).then((data) => {
      if (this.rv.closed) return;
      this.overview = data;
      const top = {};
      for (const [k, v] of Object.entries(data.cols)) {
        top[k] = (Array.isArray(v) ? v : v.max).reduce((a, x) => (x != null && x > a ? x : a), -Infinity);
      }
      const present = new Set(Object.keys(top).filter((k) => top[k] > -Infinity));
      this.tracks = this.mode === 'hand' ? handTracks(this.info, top) : eyeTracks(this.info, present, top);
      this._bandLegend();
      if (data.partial) setTimeout(() => this._loadOverview(), 1200);
      this._ensure();
      this.draw();
    }).catch((err) => {
      this.status.textContent = `could not load signals: ${err.message}`;
    });
  }

  setTimeline(tl) {
    this.timeline = tl;
    this.draw();
  }

  /** What the state rows' colours mean, and the markers. */
  _bandLegend() {
    const bands = this.tracks.find((t) => t.kind === 'bands');
    const items = [];
    if (bands) {
      for (const row of bands.rows) {
        const seen = new Set();
        const parts = Object.values(row.values).filter(([color, label]) => {
          const key = color + label;
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });
        items.push(h('span', { class: 'legend-group' }, h('span', { class: 'legend-title' }, `${row.label}:`),
          parts.map(([color, label]) => h('span', { class: 'legend-item' },
            h('span', { class: 'legend-key box', style: { '--c': color } }), label))));
      }
    }
    const marks = this.mode === 'hand' ? ['click', 'scroll', 'palm', 'expired'] : ['blink', 'blink2', 'click', 'wink', 'long_close', 'expired'];
    items.push(h('span', { class: 'legend-group' }, h('span', { class: 'legend-title' }, 'markers:'),
      marks.map((m) => h('span', { class: 'legend-item' },
        h('span', { class: `legend-key ${MARKS[m].shape === 'dot' ? 'dot' : 'tick'}`, style: { '--c': MARKS[m].color } }),
        MARKS[m].label))));
    this.bandLegend.replaceChildren(...items);
  }

  setEvents(events) {
    this.events = events;
    this.draw();
  }

  setFollow(on) {
    this.follow = on;
    if (on) this._followPlayhead();
    this._updateButtons();
    this.draw();
  }

  zoomTo(span) {
    const p = this.rv.player;
    const total = Math.max(p.end - p.start, 1e-3);
    this.span = Math.min(span, total);
    if (span === Infinity || this.span >= total) {
      this.view = [p.start, p.end];
      this.follow = false;
    } else {
      const mid = p.time;
      this._setView(mid - this.span / 2, mid + this.span / 2);
    }
    this._updateButtons();
    this.ensure();
    this.draw();
  }

  _updateButtons() {
    this.followBtn.classList.toggle('on', this.follow);
    const span = this.view[1] - this.view[0];
    const p = this.rv.player;
    this.zoomBtns.forEach((b, k) => {
      const z = ZOOMS[k][1];
      b.classList.toggle('on', z === Infinity ? span >= p.end - p.start - 1e-6 : Math.abs(span - z) < 1e-6);
    });
  }

  _setView(t0, t1) {
    const p = this.rv.player;
    const span = Math.min(t1 - t0, Math.max(p.end - p.start, 1e-3));
    let a = t0;
    if (a < p.start) a = p.start;
    if (a + span > p.end) a = Math.max(p.start, p.end - span);
    this.view = [a, a + span];
    this.rv.scrubber.setView(this.view[0], this.view[1]);
  }

  _followPlayhead() {
    const p = this.rv.player;
    const span = this.view[1] - this.view[0];
    this._setView(p.time - span / 2, p.time + span / 2);
    this.ensure();
  }

  // -- data ------------------------------------------------------------------------------
  _plotWidth() {
    return Math.max(100, (this.el.clientWidth || 800) - GUTTER - 24);
  }

  _ensure() {
    const p = this.rv.player;
    const [v0, v1] = this.view;
    const span = v1 - v0;
    const total = p.end - p.start;
    if (span >= 0.6 * total) return;           // the overview is enough
    const need = this._plotWidth();
    const w = this.window;
    if (w && !this.stale && w.t0 <= v0 && w.t1 >= v1) {
      const res = w.decimated ? (w.t.length * span) / Math.max(w.t1 - w.t0, 1e-6) : Infinity;
      if (res >= 0.8 * need && (w.t1 - w.t0) <= 8 * span) {
        if (w.partial && !this.retry) {
          // Part of it was still being read: ask again in a moment (keeping what is drawn meanwhile).
          this.retry = setTimeout(() => {
            this.retry = null;
            this.stale = true;
            this._ensure();
          }, 1200);
        }
        return;
      }
    }
    this.stale = false;
    const a = Math.max(p.start, v0 - span);
    const b = Math.min(p.end, v1 + span);
    const points = clamp(Math.round((3 * need * (b - a)) / span), 200, 9000);
    this.windowLoader.want([a, b, points]);
  }

  _data() {
    const [v0, v1] = this.view;
    const w = this.window;
    if (w && w.t0 <= v0 + 1e-6 && w.t1 >= v1 - 1e-6) return w;
    return this.overview;
  }

  // -- interaction -------------------------------------------------------------------------
  _bind() {
    const c = this.canvas;
    let drag = null;
    c.addEventListener('wheel', (e) => {
      if (!this.tracks.length) return;
      e.preventDefault();
      const t = this._t(e.offsetX);
      const f = Math.exp((e.deltaY || e.deltaX) * 0.0015);
      const [v0, v1] = this.view;
      const p = this.rv.player;
      const span = clamp((v1 - v0) * f, 0.5, Math.max(p.end - p.start, 0.5));
      const k = (t - v0) / (v1 - v0);
      this.follow = false;
      this._setView(t - k * span, t - k * span + span);
      this._updateButtons();
      this.ensure();
      this.draw();
    }, { passive: false });
    c.addEventListener('pointerdown', (e) => {
      drag = { x: e.offsetX, view: [...this.view], moved: false };
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener('pointermove', (e) => {
      if (drag) {
        const dx = e.offsetX - drag.x;
        if (Math.abs(dx) > 3) drag.moved = true;
        if (drag.moved) {
          const span = drag.view[1] - drag.view[0];
          const dt = (-dx / this._plotWidth()) * span;
          this.follow = false;
          this._setView(drag.view[0] + dt, drag.view[1] + dt);
          this._updateButtons();
          this.ensure();
          this.draw();
        }
      }
      this.hover = { x: e.offsetX, y: e.offsetY, cx: e.clientX, cy: e.clientY };
      this.draw();
    });
    c.addEventListener('pointerup', (e) => {
      if (drag && !drag.moved && e.offsetX >= GUTTER) {
        this.rv.player.pause();
        this.rv.player.seek(this._t(e.offsetX));
      }
      drag = null;
    });
    c.addEventListener('pointerleave', () => {
      this.hover = null;
      hideTip();
      this.draw();
    });
    c.addEventListener('dblclick', () => this.setFollow(true));
  }

  _x(t) {
    const [v0, v1] = this.view;
    return GUTTER + ((t - v0) / Math.max(v1 - v0, 1e-9)) * this._plotWidth();
  }

  _t(x) {
    const [v0, v1] = this.view;
    return v0 + ((x - GUTTER) / this._plotWidth()) * (v1 - v0);
  }

  // -- drawing -----------------------------------------------------------------------------
  _layout() {
    let y = 0;
    return this.tracks.map((tr) => {
      const plotH = tr.kind === 'bands' ? tr.rows.length * 12 : tr.h;
      const box = { tr, top: y, y0: y + HEAD_H, h: plotH };
      y += HEAD_H + plotH + GAP;
      return box;
    });
  }

  _draw() {
    const width = this.el.clientWidth - 24;
    if (!width || !this.tracks.length) return;
    const boxes = this._layout();
    const last = boxes[boxes.length - 1];
    const height = last.y0 + last.h + AXIS_H;
    const ctx = fitCanvas(this.canvas, width, height);
    ctx.clearRect(0, 0, width, height);
    const data = this._data();
    if (!data) return;
    const p = this.rv.player;
    const [v0, v1] = this.view;
    const T = data.t;
    const i0 = Math.max(0, bisect(T, v0) - 1);
    const i1 = Math.min(T.length - 1, bisect(T, v1) + 1);
    const ctxInfo = { ctx, data, T, i0, i1, width };
    for (const box of boxes) this._track(ctxInfo, box);
    // Time axis.
    const pw = this._plotWidth();
    ctx.strokeStyle = C.axis;
    ctx.beginPath();
    ctx.moveTo(GUTTER, last.y0 + last.h + 0.5);
    ctx.lineTo(GUTTER + pw, last.y0 + last.h + 0.5);
    ctx.stroke();
    ctx.fillStyle = C.faint;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textBaseline = 'top';
    const span = v1 - v0;
    for (const t of ticks(v0 - p.start, v1 - p.start, Math.max(2, Math.floor(pw / 110)))) {
      const x = this._x(p.start + t);
      if (x < GUTTER - 1 || x > GUTTER + pw + 1) continue;
      ctx.fillRect(Math.round(x), last.y0 + last.h, 1, 4);
      const text = clock(t, span < 30);
      ctx.fillText(text, Math.min(x + 3, GUTTER + pw - ctx.measureText(text).width), last.y0 + last.h + 6);
    }
    // Playhead and the hover crosshair.
    const px = this._x(p.time);
    if (px >= GUTTER && px <= GUTTER + pw) {
      ctx.fillStyle = C.accent;
      ctx.fillRect(Math.round(px) - 1, 0, 2, last.y0 + last.h);
    }
    if (this.hover && this.hover.x >= GUTTER && this.hover.x <= GUTTER + pw) {
      ctx.fillStyle = 'rgba(238, 242, 255, 0.45)';
      ctx.fillRect(Math.round(this.hover.x), 0, 1, last.y0 + last.h);
      this._tooltip(boxes, data);
    }
    const shown = bisect(p.t, v1) - bisect(p.t, v0);         // frames in view
    this.status.textContent = `${shown.toLocaleString()} frames in view${data.decimated ? ', shown as min–max per pixel' : ''}` +
      (data.partial ? ` · reading the recording… ${Math.round((data.progress || 0) * 100)}%` : '');
  }

  _track({ ctx, data, T, i0, i1 }, box) {
    const { tr, top, y0, h: plotH } = box;
    const pw = this._plotWidth();
    // Title, unit and legend.
    ctx.font = '600 12px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = C.text;
    ctx.fillText(tr.title, 8, top + HEAD_H / 2 + 2);
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillStyle = C.faint;
    // The unit shares the gutter with the y-axis labels: keep it clear of them.
    ctx.fillText(ellipsize(ctx, tr.unit || '', GUTTER - (tr.kind === 'bands' ? 16 : 54)), 8, top + HEAD_H / 2 + 15);
    this._legend(ctx, tr, top);
    // Plot background, then the grid and its labels (in the gutter, outside the clip).
    ctx.fillStyle = C.plot;
    ctx.fillRect(GUTTER, y0, pw, plotH);
    if (tr.kind === 'eventbars') this._eventScale(box);
    else if (tr.kind !== 'bands') {
      box.scale = this._yScale(tr, data, i0, i1, y0, plotH);
      this._grid(ctx, box);
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(GUTTER, y0, pw, plotH);
    ctx.clip();
    if (tr.kind === 'bands') this._bands(ctx, data, T, i0, i1, box);
    else if (tr.kind === 'eventbars') this._eventBars(ctx, box);
    else this._lines(ctx, data, T, i0, i1, box);
    ctx.restore();
    if (tr.kind === 'bands') {
      ctx.font = '11px system-ui, sans-serif';
      ctx.fillStyle = C.muted;
      ctx.textBaseline = 'middle';
      tr.rows.forEach((row, k) => {
        ctx.textAlign = 'right';
        ctx.fillText(row.label, GUTTER - 8, y0 + k * 12 + 6);
        ctx.textAlign = 'left';
      });
    }
  }

  _legend(ctx, tr, top) {
    const items = [];
    for (const s of tr.series || []) items.push({ color: s.color, label: s.label, kind: 'line' });
    for (const r of tr.refs || []) items.push({ color: r.color, label: r.label, kind: r.dash ? 'dash' : 'line' });
    if (tr.thresholds) items.push({ color: C.text, label: 'closes below', kind: 'dash' }, { color: C.muted, label: 'opens above', kind: 'dash' });
    for (const s of tr.shades || []) items.push({ color: s.color, label: s.label, kind: 'box' });
    // One series needs no legend: the title names it.
    if ((tr.series || []).length < 2 && !(tr.refs || []).length && !(tr.shades || []).length && !tr.thresholds) return;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    let x = GUTTER;
    const y = top + HEAD_H / 2;
    const right = GUTTER + this._plotWidth();
    for (const it of items) {
      if (x + 18 + ctx.measureText(it.label).width > right) {
        ctx.fillStyle = C.faint;
        ctx.fillText('…', x, y);            // a narrow window: the rest of the legend does not fit
        break;
      }
      if (it.kind === 'box') {
        ctx.fillStyle = it.color;
        ctx.fillRect(x, y - 5, 12, 10);
        ctx.strokeStyle = C.axis;
        ctx.strokeRect(x + 0.5, y - 4.5, 11, 9);
      } else {
        ctx.strokeStyle = it.color;
        ctx.lineWidth = 2;
        ctx.setLineDash(it.kind === 'dash' ? [4, 3] : []);
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + 14, y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.fillStyle = C.muted;
      ctx.fillText(it.label, x + 18, y);
      x += 26 + ctx.measureText(it.label).width;
    }
  }

  _yScale(tr, data, i0, i1, y0, plotH) {
    let lo;
    let hi;
    if (tr.range) [lo, hi] = tr.range;
    else {
      lo = Infinity;
      hi = -Infinity;
      for (const s of [...(tr.series || []), ...(tr.refs || [])]) {
        const v = data.cols[s.col];
        if (!v) continue;
        const a = Array.isArray(v) ? v : v.min;
        const b = Array.isArray(v) ? v : v.max;
        for (let i = i0; i <= i1; i++) {
          if (a[i] != null && a[i] < lo) lo = a[i];
          if (b[i] != null && b[i] > hi) hi = b[i];
        }
      }
      if (tr.thresholds) for (const th of tr.thresholds) hi = Math.max(hi, th.off);
      if (!Number.isFinite(lo)) [lo, hi] = [0, 1];
      if (tr.min0) lo = Math.min(0, lo);
      if (hi - lo < 1e-6) {
        hi += 0.5;
        lo -= tr.min0 ? 0 : 0.5;
      }
      const pad = (hi - lo) * 0.08;
      hi += pad;
      if (!tr.min0) lo -= pad;
    }
    const pad = 3;
    const y = (v) => (tr.invert ? y0 + pad + ((v - lo) / (hi - lo)) * (plotH - 2 * pad)
      : y0 + plotH - pad - ((v - lo) / (hi - lo)) * (plotH - 2 * pad));
    return { lo, hi, y };
  }

  _grid(ctx, box) {
    const { y0, h: plotH, scale: sc } = box;
    const pw = this._plotWidth();
    ctx.font = '10px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    ctx.lineWidth = 1;
    for (const v of ticks(sc.lo, sc.hi, Math.max(3, Math.floor(plotH / 26)))) {
      const y = Math.round(sc.y(v)) + 0.5;
      if (y < y0 - 1 || y > y0 + plotH + 1) continue;
      ctx.strokeStyle = C.grid;
      ctx.beginPath();
      ctx.moveTo(GUTTER, y);
      ctx.lineTo(GUTTER + pw, y);
      ctx.stroke();
      ctx.fillStyle = C.faint;
      ctx.fillText(fmtTick(v, sc.hi - sc.lo), GUTTER - 6, y);
    }
    ctx.textAlign = 'left';
  }

  _lines(ctx, data, T, i0, i1, box) {
    const { tr, y0, h: plotH, scale: sc } = box;
    // Shades (boolean columns).
    for (const sh of tr.shades || []) this._shade(ctx, data, T, i0, i1, sh, y0, plotH);
    // Hand mode: the pinch thresholds in force over time.
    if (tr.thresholds) {
      const p = this.rv.player;
      tr.thresholds.forEach((th, k) => {
        const a = th.t == null ? p.start : th.t;
        const b = k + 1 < tr.thresholds.length ? tr.thresholds[k + 1].t : p.end;
        for (const [v, color, dash] of [[th.on, C.text, [6, 4]], [th.off, C.muted, [2, 3]]]) {
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.setLineDash(dash);
          ctx.beginPath();
          ctx.moveTo(this._x(a), sc.y(v));
          ctx.lineTo(this._x(b), sc.y(v));
          ctx.stroke();
          ctx.setLineDash([]);
        }
      });
    }
    for (const r of tr.refs || []) this._series(ctx, data, T, i0, i1, r.col, r.color, sc, 1.25, r.dash);
    for (const s of tr.series || []) this._series(ctx, data, T, i0, i1, s.col, s.color, sc, 1.6);
    if (tr.markers) this._markers(ctx, box);
  }

  _series(ctx, data, T, i0, i1, col, color, sc, width, dash = null) {
    const v = data.cols[col];
    if (!v) return;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.setLineDash(dash || []);
    if (Array.isArray(v)) {
      // One sample per frame: a line, broken where the value is missing.
      const gap = medianStep(T, i0, i1) * 4;
      ctx.beginPath();
      let pen = false;
      let lastT = null;
      for (let i = i0; i <= i1; i++) {
        const val = v[i];
        if (val == null || (lastT != null && T[i] - lastT > gap)) {
          pen = false;
          if (val == null) {
            lastT = null;
            continue;
          }
        }
        const x = this._x(T[i]);
        const y = sc.y(val);
        if (pen) ctx.lineTo(x, y);
        else ctx.moveTo(x, y);
        pen = true;
        lastT = T[i];
      }
      ctx.stroke();
    } else {
      // Buckets: the band between each bucket's minimum and maximum.
      const lo = data.t_lo;
      const hi = data.t_hi;
      let run = [];
      const flush = () => {
        if (!run.length) return;
        ctx.beginPath();
        run.forEach(([x0, x1, a], k) => {
          if (k) ctx.lineTo(x0, a);
          else ctx.moveTo(x0, a);
          ctx.lineTo(x1, a);
        });
        for (let k = run.length - 1; k >= 0; k--) {
          const [x0, x1, , b] = run[k];
          ctx.lineTo(x1, b);
          ctx.lineTo(x0, b);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        run = [];
      };
      ctx.lineWidth = 1;
      for (let i = i0; i <= i1; i++) {
        const a = v.max[i];
        const b = v.min[i];
        if (a == null || b == null) {
          flush();
          continue;
        }
        run.push([this._x(lo[i]), this._x(hi[i]), sc.y(a), sc.y(b)]);
      }
      flush();
    }
    ctx.setLineDash([]);
  }

  /** Shade where column ``col`` equals ``value`` (1: a true flag). */
  _shade(ctx, data, T, i0, i1, { col, color, value = 1 }, y0, plotH) {
    const v = data.cols[col];
    if (!v) return;
    const vals = Array.isArray(v) ? v : v.max;
    const ends = Array.isArray(v) ? null : data.t_hi;
    const starts = Array.isArray(v) ? T : data.t_lo;
    ctx.fillStyle = color;
    let a = null;
    for (let i = i0; i <= i1 + 1; i++) {
      const on = i <= i1 && vals[i] === value;
      if (on && a == null) a = starts[i];
      if (!on && a != null) {
        const b = ends ? ends[i - 1] : (i <= i1 ? T[i] : T[i - 1]);
        const x0 = this._x(a);
        ctx.fillRect(x0, y0, Math.max(this._x(b) - x0, 1), plotH);
        a = null;
      }
    }
  }

  _bands(ctx, data, T, i0, i1, box) {
    const { tr, y0 } = box;
    const starts = data.decimated ? data.t_lo : T;
    tr.rows.forEach((row, k) => {
      const v = data.cols[row.col];
      if (!v) return;
      const vals = Array.isArray(v) ? v : v.min;
      const y = y0 + k * 12 + 1;
      ctx.fillStyle = 'rgba(148, 163, 214, 0.05)';
      ctx.fillRect(GUTTER, y, this._plotWidth(), 10);
      let a = null;
      let cur = null;
      const close = (endT) => {
        if (a == null) return;
        const spec = row.values[cur] || (cur >= 3 && row.values[3]);
        if (spec && spec[0]) {
          ctx.fillStyle = spec[0];
          const x0 = this._x(a);
          ctx.fillRect(x0, y, Math.max(this._x(endT) - x0, 1), 10);
        }
        a = null;
      };
      for (let i = i0; i <= i1; i++) {
        const val = vals[i] == null ? null : Math.round(vals[i]);
        if (val !== cur) {
          close(starts[i]);
          cur = val;
          if (val != null) a = starts[i];
        }
      }
      close(data.decimated ? data.t_hi[i1] : T[i1]);
    });
  }

  _eventScale(box) {
    const { tr, y0, h: plotH } = box;
    const [v0, v1] = this.view;
    const evs = (this.events || []).filter((e) => e.type === tr.event && e.t >= v0 && e.t <= v1 && e.data);
    const vals = evs.map((e) => Number(e.data[tr.field]) || 0);
    box.bars = { evs, vals, m: Math.max(1, ...vals.map(Math.abs)) };
    const ctx = this.canvas.getContext('2d');
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = C.faint;
    ctx.fillText(`+${num(box.bars.m, 0)}`, GUTTER - 6, y0 + 4);
    ctx.fillText('0', GUTTER - 6, y0 + plotH / 2);
    ctx.fillText(`−${num(box.bars.m, 0)}`, GUTTER - 6, y0 + plotH - 4);
    ctx.textAlign = 'left';
  }

  _eventBars(ctx, box) {
    const { tr, y0, h: plotH } = box;
    if (!box.bars) return;
    const { evs, vals, m } = box.bars;
    const mid = y0 + plotH / 2;
    ctx.strokeStyle = C.axis;
    ctx.beginPath();
    ctx.moveTo(GUTTER, mid + 0.5);
    ctx.lineTo(GUTTER + this._plotWidth(), mid + 0.5);
    ctx.stroke();
    ctx.fillStyle = tr.color;
    evs.forEach((e, k) => {
      const x = this._x(e.t);
      const hgt = (vals[k] / m) * (plotH / 2 - 3);
      ctx.fillRect(x - 1.5, Math.min(mid, mid + hgt), 3, Math.abs(hgt));
    });
  }

  _markers(ctx, box) {
    if (!this.timeline) return;
    const [v0, v1] = this.view;
    const y = box.y0 + 7;
    for (const m of this.timeline.markers) {
      if (m.t < v0 || m.t > v1) continue;
      if (m.kind === 'command') continue;
      const spec = MARKS[m.cat];
      if (!spec) continue;
      if (this.mode === 'eyes' && m.hand) continue;
      const x = this._x(m.t);
      ctx.fillStyle = m.cat === 'wink' && m.eye ? EYE[m.eye] : spec.color;
      ctx.strokeStyle = ctx.fillStyle;
      ctx.lineWidth = 1.5;
      switch (spec.shape) {
        case 'tick':
          ctx.fillRect(x - 0.75, box.y0, 1.5, 12);
          break;
        case 'dot':
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = C.plot;
          ctx.lineWidth = 2;
          ctx.stroke();
          break;
        case 'tri': {
          const down = m.type === 'wink_start' || m.type === 'wink';
          ctx.beginPath();
          ctx.moveTo(x - 4.5, down ? y - 4 : y + 4);
          ctx.lineTo(x + 4.5, down ? y - 4 : y + 4);
          ctx.lineTo(x, down ? y + 4 : y - 4);
          ctx.fill();
          break;
        }
        case 'bar':
          ctx.fillRect(x - 1.5, box.y0, 3, box.h);
          break;
        case 'diamond':
          ctx.beginPath();
          ctx.moveTo(x, y - 5);
          ctx.lineTo(x + 5, y);
          ctx.lineTo(x, y + 5);
          ctx.lineTo(x - 5, y);
          ctx.fill();
          break;
        case 'x':
          ctx.beginPath();
          ctx.moveTo(x - 3, y - 3);
          ctx.lineTo(x + 3, y + 3);
          ctx.moveTo(x + 3, y - 3);
          ctx.lineTo(x - 3, y + 3);
          ctx.stroke();
          break;
        default:
          break;
      }
    }
  }

  _tooltip(boxes, data) {
    const { x, y, cx, cy } = this.hover;
    const box = boxes.find((b) => y >= b.top && y <= b.y0 + b.h + GAP / 2);
    const t = this._t(x);
    const T = data.t;
    let i = bisect(T, t);
    if (i < 0) i = 0;
    if (i + 1 < T.length && Math.abs(T[i + 1] - t) < Math.abs(T[i] - t)) i += 1;
    const rows = [{ title: `${clock(t - this.rv.player.start)}${data.decimated ? '  (min – max of the bucket)' : ''}` }];
    if (!box) {
      showTip(cx, cy, rows);
      return;
    }
    const tr = box.tr;
    const fmt = (v) => (v == null ? '—' : fmtValue(v));
    const valueOf = (col) => {
      const v = data.cols[col];
      if (!v) return null;
      if (Array.isArray(v)) return fmt(v[i]);
      return v.min[i] === v.max[i] ? fmt(v.min[i]) : `${fmt(v.min[i])} – ${fmt(v.max[i])}`;
    };
    for (const s of tr.series || []) rows.push({ color: s.color, value: valueOf(s.col), label: s.label });
    for (const r of tr.refs || []) rows.push({ color: r.color, value: valueOf(r.col), label: r.label });
    for (const sh of tr.shades || []) rows.push({ value: valueOf(sh.col) === '1' ? 'yes' : 'no', label: sh.label });
    if (tr.kind === 'bands') {
      for (const row of tr.rows) {
        const v = data.cols[row.col];
        if (!v) continue;
        const val = Array.isArray(v) ? v[i] : v.min[i];
        const spec = val == null ? null : row.values[Math.round(val)] || (val >= 3 && row.values[3]);
        rows.push({ color: spec ? spec[0] : null, value: spec ? spec[1] : (val == null ? '—' : val === 0 ? 'no' : String(val)), label: row.label });
      }
    }
    if (tr.markers && this.timeline) {
      const near = this.timeline.markers.filter((m) => Math.abs(this._x(m.t) - x) <= 5 && MARKS[m.cat] && m.kind !== 'command');
      for (const m of near.slice(0, 4)) rows.push({ color: MARKS[m.cat].color, value: m.type, label: m.eye ? `${m.eye} eye` : '' });
    }
    if (tr.kind === 'eventbars' && this.events) {
      const near = this.events.filter((e) => e.type === tr.event && Math.abs(this._x(e.t) - x) <= 4);
      for (const e of near.slice(0, 4)) rows.push({ color: tr.color, value: `${num(e.data[tr.field], 1)} px`, label: e.type });
    }
    showTip(cx, cy, rows);
  }
}

function ellipsize(ctx, text, width) {
  if (ctx.measureText(text).width <= width) return text;
  let s = text;
  while (s.length > 1 && ctx.measureText(`${s}…`).width > width) s = s.slice(0, -1);
  return `${s.trimEnd()}…`;
}

function medianStep(T, i0, i1) {
  const n = Math.min(i1 - i0, 50);
  if (n < 2) return 1 / 30;
  const steps = [];
  for (let k = 0; k < n; k++) steps.push(T[i0 + k + 1] - T[i0 + k]);
  steps.sort((a, b) => a - b);
  return Math.max(steps[Math.floor(n / 2)], 1e-3);
}

function fmtTick(v, span) {
  if (span >= 100) return Math.round(v).toString();
  if (span >= 10) return v.toFixed(0);
  if (span >= 1) return v.toFixed(1);
  if (span >= 0.1) return v.toFixed(2);
  return v.toFixed(3);
}

function fmtValue(v) {
  if (!isNum(v)) return String(v);
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}
