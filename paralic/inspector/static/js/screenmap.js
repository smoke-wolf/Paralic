// Screen map: where the gaze landed on the screen (CSS pixels of the monitor).
//
// The raw prediction of the network (or the fingertip, in hand mode), the
// smoothed cursor with its trail over the last second and a half, the
// calibration dot when the frame was labelled (and how far off the
// prediction was), recent clicks, and - from the frame's internals - where
// each gaze network (both eyes, left, right) would have put it.

import { C, GAZE, PHASE, PHASE_NAMES, alpha, fitCanvas, h, hideTip, isNum, num, showTip, xy } from './util.js';

const TRAIL_S = 1.5;

export class ScreenMap {
  constructor(rv) {
    this.rv = rv;
    this.screen = rv.info.screen;
    this.mode = rv.info.mode;
    this.canvas = h('canvas', { class: 'map-canvas', 'aria-label': 'Screen map' });
    this.readout = h('div', { class: 'map-readout' });
    const key = 'paralic.inspector.map';
    let saved = {};
    try {
      saved = JSON.parse(localStorage.getItem(key) || '{}');
    } catch (_) { /* private mode */ }
    this.opts = { trail: saved.trail ?? true, networks: saved.networks ?? (this.mode === 'eyes'), clicks: saved.clicks ?? true };
    const toggle = (k, text) => {
      const input = h('input', { type: 'checkbox', checked: this.opts[k] });
      input.addEventListener('change', () => {
        this.opts[k] = input.checked;
        try {
          localStorage.setItem(key, JSON.stringify(this.opts));
        } catch (_) { /* private mode */ }
        this.draw();
      });
      return h('label', { class: 'toggle' }, input, text);
    };
    const keys = [
      { label: 'cursor (smoothed)', color: GAZE.cursor, kind: 'ring' },
      { label: this.mode === 'hand' ? 'fingertip (mapped)' : 'raw prediction', color: GAZE.raw, kind: 'dot' },
      { label: 'calibration target', color: GAZE.target, kind: 'dot' },
      { label: 'click', color: GAZE.click, kind: 'ring' },
    ];
    this.el = h('section', { class: 'panel screen', id: 'sec-screen' },
      h('header', { class: 'panel-head' }, h('h2', {}, 'Screen'),
        h('div', { class: 'toggles' }, toggle('trail', 'Trail'), toggle('clicks', 'Clicks'),
          this.mode === 'eyes' ? toggle('networks', 'Each network') : null)),
      h('div', { class: 'map-wrap' }, this.canvas),
      h('div', { class: 'legend' }, keys.map((k) => h('span', { class: 'legend-item' },
        h('span', { class: `legend-key ${k.kind}`, style: { '--c': k.color } }), k.label)),
      this.mode === 'eyes' ? h('span', { class: 'legend-item' }, h('span', { class: 'legend-key square', style: { '--c': C.text } }),
        'network outputs (B, L, R)') : null),
      this.readout);
    this.detail = null;
    this.canvas.addEventListener('pointermove', (e) => {
      if (!this.scale) return;
      const sx = e.offsetX / this.scale;
      const sy = e.offsetY / this.scale;
      showTip(e.clientX, e.clientY, [{ value: `${Math.round(sx)}, ${Math.round(sy)}`, label: 'screen px' }]);
    });
    this.canvas.addEventListener('pointerleave', () => hideTip());
    new ResizeObserver(() => this.draw()).observe(this.el);
  }

  onDetail(d) {
    this.detail = d;
    this.draw();
  }

  draw() {
    const wrap = this.canvas.parentElement;
    const width = wrap.clientWidth;
    if (!width) return;
    const W = this.screen.w;
    const H = this.screen.h;
    const height = Math.min(width * (H / W), 460);
    const cw = height * (W / H);
    const ctx = fitCanvas(this.canvas, cw, height);
    const s = cw / W;
    this.scale = s;
    ctx.fillStyle = C.plot;
    ctx.fillRect(0, 0, cw, height);
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    for (let k = 1; k < 10; k++) {
      ctx.beginPath();
      ctx.moveTo(Math.round((k * cw) / 10) + 0.5, 0);
      ctx.lineTo(Math.round((k * cw) / 10) + 0.5, height);
      ctx.moveTo(0, Math.round((k * height) / 10) + 0.5);
      ctx.lineTo(cw, Math.round((k * height) / 10) + 0.5);
      ctx.stroke();
    }
    ctx.strokeStyle = C.axis;
    ctx.strokeRect(0.5, 0.5, cw - 1, height - 1);
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillStyle = C.faint;
    ctx.fillText(`${W} × ${H} screen px`, 6, height - 6);

    const p = this.rv.player;
    const n = p.n;
    const frames = this.rv.frames;
    const lo = p.indexAt(p.t[n] - TRAIL_S);
    const trail = frames.range(lo, n + 1);
    const cur = trail.length ? trail[trail.length - 1] : null;
    const P = (v) => [v[0] * s, v[1] * s];

    if (this.opts.trail && trail.length > 1) {
      // Smoothed cursor path, fading into the past; raw predictions as dots.
      for (let k = 1; k < trail.length; k++) {
        const a = trail[k - 1];
        const b = trail[k];
        if (!a || !b) continue;
        const age = (p.t[n] - b.t) / TRAIL_S;
        const ga = a.msg && a.msg.gaze;
        const gb = b.msg && b.msg.gaze;
        if (Array.isArray(ga) && Array.isArray(gb)) {
          ctx.strokeStyle = alpha(GAZE.cursor, 0.75 * (1 - age));
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(...P(ga));
          ctx.lineTo(...P(gb));
          ctx.stroke();
        }
        const rb = b.msg && b.msg.raw;
        if (Array.isArray(rb)) {
          ctx.fillStyle = alpha(GAZE.raw, 0.55 * (1 - age));
          ctx.beginPath();
          ctx.arc(...P(rb), 2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    const msg = (cur && cur.msg) || {};
    const label = cur && cur.label;
    let target = null;
    if (label && isNum(label.x)) target = [label.x, label.y];
    else if (label && isNum(label.fx)) target = [label.fx * W, label.fy * H];
    if (target) {
      const color = PHASE[label.kind] || GAZE.target;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(...P(target), 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = C.plot;
      ctx.lineWidth = 2;
      ctx.stroke();
      tag(ctx, P(target)[0] + 10, P(target)[1] - 8,
        `${PHASE_NAMES[label.kind] || label.kind} ${label.pt ?? label.i ?? ''}${cur.stored ? ' · kept' : ' · not kept'}`);
      if (Array.isArray(msg.raw)) {
        ctx.strokeStyle = 'rgba(238, 242, 255, 0.6)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(...P(target));
        ctx.lineTo(...P(msg.raw));
        ctx.stroke();
      }
    }

    // Recent clicks (eye: double blinks; hand: pinches), fading.
    if (this.opts.clicks && this.detail && this.detail.recent) {
      for (const ev of this.detail.recent) {
        if (ev.type !== 'double_blink' || !Array.isArray(ev.data.at)) continue;
        const age = (p.t[n] - ev.t) / 1.5;
        if (age < 0 || age > 1) continue;
        const [x, y] = P(ev.data.at);
        ctx.strokeStyle = alpha(GAZE.click, 1 - age);
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(x, y, 10 + 18 * age, 0, Math.PI * 2);
        ctx.stroke();
        tag(ctx, x + 14, y + 22, ev.data.hand ? 'pinch click' : 'double blink: click');
      }
    }

    // Each network's output for this frame (eye mode).
    const ins = this.detail && this.detail.n === n && this.detail.internals;
    if (this.opts.networks && ins) {
      for (const [eye, net] of Object.entries(ins.networks)) {
        if (!Array.isArray(net.final)) continue;
        const [x, y] = P(net.final);
        ctx.strokeStyle = C.text;
        ctx.lineWidth = eye === ins.led ? 2 : 1;
        ctx.strokeRect(x - 4, y - 4, 8, 8);
        ctx.fillStyle = C.text;
        ctx.font = '10px system-ui, sans-serif';
        ctx.fillText({ both: 'B', left: 'L', right: 'R' }[eye], x + 6, y + 4);
      }
    }

    if (Array.isArray(msg.raw)) {
      const [x, y] = P(msg.raw);
      ctx.fillStyle = GAZE.raw;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = C.plot;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    if (Array.isArray(msg.gaze)) {
      const [x, y] = P(msg.gaze);
      ctx.strokeStyle = GAZE.cursor;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(x, y, 13, 0, Math.PI * 2);
      ctx.stroke();
      if (msg.frozen) {
        ctx.fillStyle = alpha(GAZE.cursor, 0.35);
        ctx.fill();
        tag(ctx, x + 16, y + 4, msg.pinching ? 'held while pinching' : 'frozen (eyes closing)');
      }
    }
    if (cur && !msg.face) tag(ctx, 8, 18, this.mode === 'hand' ? 'no hand in view' : 'no face in view');

    // Readout.
    const err = target && Array.isArray(msg.raw) ? Math.hypot(msg.raw[0] - target[0], msg.raw[1] - target[1]) : null;
    const parts = [
      [this.mode === 'hand' ? 'fingertip' : 'raw', xy(msg.raw)],
      ['cursor', xy(msg.gaze) + (msg.frozen ? ' (frozen)' : '')],
    ];
    if (target) parts.push(['target', xy(target)], ['off by', isNum(err) ? `${num(err, 0)} px` : '—']);
    if (this.mode === 'eyes') parts.push(['network', msg.net || '—']);
    this.readout.replaceChildren(...parts.map(([k, v]) => h('span', { class: 'kv' }, h('span', { class: 'k' }, k), h('span', { class: 'v' }, v))));
  }
}

function tag(ctx, x, y, text) {
  ctx.font = '11px system-ui, sans-serif';
  const w = ctx.measureText(text).width;
  ctx.fillStyle = 'rgba(10, 15, 31, 0.78)';
  ctx.fillRect(x - 3, y - 11, w + 6, 15);
  ctx.fillStyle = C.text;
  ctx.fillText(text, x, y);
}
